/* halo_gemm_fp8a8.hip - the block-scaled FP8 GEMMs (gemm_fp8a8_nt_m16, gemm_fp8a8_tiled and
 * the bias narrow row), gfx1151.
 *
 * C[M,N] = A[M,K] @ W[N,K]^T with both operands E4M3: A carries an f32 scale per (row, 128 K),
 * W a bf16 scale per 128x128 tile, read as the checkpoint ships them. gfx1151 has no fp8 matrix
 * instruction, so both operands are dequantised to bf16 in registers (halo_wmma.h's deq16_e4m3,
 * exact: every e4m3 value is a bf16) and multiplied on the bf16 WMMA, one 128-wide scale group
 * at a time, each group's product folded into the f32 accumulator with the product of the two
 * scales. The kernel design -- a wave a column, the weight fragment issued ahead of the
 * activation fragments, split-K folded through LDS in split order -- is the strix-halo-vllm
 * libhalo's production FP8 kernel, reworked for this family's two-sided scales: on that kernel
 * the activation is bf16 and only the weight carries a scale, here both do, and the activation
 * scale changes per row tile, so the fold is per tile rather than per column.
 *
 * Split-K is in-block (SK waves a column tile fold through LDS), so the caller's scratch the
 * narrow row declares is not used and its size query answers 0. libr4d splits over the caller's
 * scratch instead; the summation order is fixed either way, and a run repeats exactly.
 */
#include "halo.h"
#include "halo_launch.h"   // HALO_DYN_LDS: tests/halo_emu runs this unit on the host
#include "halo_fp8_args.h"
#include "halo_fp8_wmma.h"   // halo_fp8::e4m3_pk: the exact e4m3 decoder
#include "halo_wmma.h"
#include "halo_gdn_wmma.h"   // fragP / accRow for the LDS-tiled prefill kernel
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

typedef int int4v __attribute__((ext_vector_type(4)));
using halo::deq16_e4m3;

/* 2^120: what halo_wmma.h's deq16_e4m3 leaves out of every value it produces (its exponent is
 * unbiassed), folded into the weight scale below. Exact as a float. Only the weight takes that
 * path: with both operands on it the products would be 2^-240 down and underflow the f32
 * accumulator, so the activation is dequantised exactly instead. */
#define HALO_FP8_FIX 1.329227995784916e36f

/* Sixteen e4m3 bytes to the exact bf16 fragment, through halo_fp8_wmma.h's decoder (the same
 * values the reference reads). Slower than the exponent trick above; the activation fragment
 * is the one that cannot take it. */
__device__ __forceinline__ v16bf deq16_exact(const uint32_t* __restrict__ w4) {
    union { uint32_t u[8]; v16bf f; } q;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
        const v2f p0 = halo_fp8::e4m3_pk<false>(w4[d]);
        const v2f p1 = halo_fp8::e4m3_pk<true>(w4[d]);
        q.u[d * 2]     = halo::pk_bf16(p0.x, p0.y);
        q.u[d * 2 + 1] = halo::pk_bf16(p1.x, p1.y);
    }
    return q.f;
}

#define WAVE 32
#define HALO_GEMM_FP8_MAX_M 64

/* One wave owns NT output columns and up to MT 16-row tiles; a lane's eight accumulator
 * slots are eight rows of that column (halo_wmma.h's D mapping). A block is BW waves, SK of
 * them splitting K for each column group (see fp8_launch for BW). NT > 1 amortises the
 * activation dequant (deq16_exact, ~180 ops against the
 * weight fragment's ~10) and its loads over NT weight fragments; per-output accumulation
 * order is unchanged, so the bytes are the NT == 1 kernel's. */
template <int MT, int TEMPORAL, int NT = 1, int BW = 16>
__global__ __launch_bounds__(BW * 32) void halo_gemm_fp8a8_kernel(
        const uint8_t* __restrict__ A, const float* __restrict__ Ascale,
        const uint8_t* __restrict__ W, const uint16_t* __restrict__ Ws,
        const uint16_t* __restrict__ Bias, __hip_bfloat16* __restrict__ C,
        int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld, int c_ld, int SK,
        const uint16_t* __restrict__ Adq) {
    HALO_DYN_LDS(float, red);                     // [BW waves][32 lanes][8], one tile folded at a time
    const int lane = threadIdx.x & (WAVE - 1), wave = threadIdx.x / WAVE;
    const int wpb = BW / SK;                           // wave column groups a block
    const int ctile = wave / SK, ks = wave % SK;
    const int slot = lane & 15;
    const int tile0 = (blockIdx.x * wpb + ctile) * NT;
    int n_j[NT], ns_j[NT];
    bool nlive_j[NT];
    const uint16_t* Sp_j[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        n_j[j] = (tile0 + j) * 16 + slot;
        nlive_j[j] = n_j[j] < N;
        ns_j[j] = nlive_j[j] ? n_j[j] : 0;
        Sp_j[j] = Ws + (size_t)(ns_j[j] / 128) * sw_ld;
    }
    const int Kb = K / SK, kbase = ks * Kb;
    const int ngroups = Kb / 128;
    /* The weight is in halo_fp8_frag.h's fragment order, not row-major: (16 row, 64 k) tiles of
     * 32 lanes x 32 bytes, so this column's bytes for a k step are two 8-byte chunks, this
     * lane's and its cross-half partner's, inside one tile. w_ld is unused: the stored plane is
     * exactly N*K bytes and its geometry is K itself. */
    (void)w_ld;
#pragma unroll
    for (int j = 0; j < NT; ++j) Sp_j[j] += kbase / 128;
    const uint8_t* Ap[MT];
    int mrow[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        const int m = blockIdx.y * (MT * 16) + t * 16 + slot;
        mrow[t] = m < M ? m : M - 1;
        Ap[t] = A + (size_t)mrow[t] * a_ld + kbase;
    }

    v8f acc[MT][NT];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[t][j] = (v8f)0.f;

    for (int g = 0; g < ngroups; ++g) {
        const int k0 = g * 128;
        /* The weight scale is one scalar a column (all eight accumulator slots share it); a dead
         * column folds by zero instead of branching the wave. */
        float wsc_j[NT];
#pragma unroll
        for (int j = 0; j < NT; ++j)
            wsc_j[j] = nlive_j[j] ? __uint_as_float((uint32_t)Sp_j[j][g] << 16) : 0.f;
        v8f tmp[MT][NT];
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int j = 0; j < NT; ++j) tmp[t][j] = (v8f)0.f;
        /* Fully unrolled, the eight steps' activation loads are hoisted ahead of the first WMMA:
         * eight MT-tile fragment sets, which spills from MT 2 on (MT 4: 203 VGPRs to scratch). */
        constexpr int KU = MT == 1 ? 8 : 2;
#pragma unroll KU
        for (int u = 0; u < 8; ++u) {
            const int kabs = kbase + k0 + u * 16;
            v16bf af[MT];
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                if (Adq) {
                    /* The launch's activation plane: deq16_exact's outputs, read back in
                     * fragment order (two uint4 = the fragment's eight dwords). */
                    const uint16_t* src = Adq + (size_t)mrow[t] * K + kbase + k0 + u * 16;
                    uint32_t* dst = (uint32_t*)&af[t];
                    *(uint4*)dst = *(const uint4*)src;
                    *(uint4*)(dst + 4) = *(const uint4*)(src + 8);
                } else {
                    const int4v av = *(const int4v*)(Ap[t] + k0 + u * 16);
                    af[t] = deq16_exact((const uint32_t*)&av);
                }
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint8_t* ch = W + ((size_t)(ns_j[j] >> 4) * (K >> 6) + (kabs >> 6)) * 1024
                                        + slot * 32 + ((kabs >> 4) & 3) * 8;
                uint32_t wd[4];
                *(uint2*)&wd[0] = *(const uint2*)ch;
                *(uint2*)&wd[2] = *(const uint2*)(ch + 16 * 32);
                const v16bf bf = deq16_e4m3(wd);
#pragma unroll
                for (int t = 0; t < MT; ++t) tmp[t][j] = halo::mma(af[t], bf, tmp[t][j]);
            }
        }
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                /* Accumulator slot e is output row 2e + (lane >> 4) of the tile, not this lane's
                 * fragment row: each slot folds with its own row's activation scale. (libr4d's
                 * transposed formulation makes the slots share one; this one pays eight loads.) */
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    int mr = blockIdx.y * (MT * 16) + t * 16 + 2 * e + (lane >> 4);
                    mr = mr < M ? mr : M - 1;
                    const float sc = Ascale[(size_t)mr * sa_ld + kbase / 128 + g] * wsc_j[j] * HALO_FP8_FIX;
                    acc[t][j][e] = fmaf(tmp[t][j][e], sc, acc[t][j][e]);
                }
            }
    }

    if (SK > 1) {
        /* Fold the splits in split order, one (row tile, column) at a time to keep the
         * buffer at 16 waves x 32 lanes x 8 floats. */
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                float* mine = red + ((size_t)wave * WAVE + lane) * 8;
#pragma unroll
                for (int e = 0; e < 8; ++e) mine[e] = acc[t][j][e];
                __syncthreads();
                if (ks == 0) {
                    for (int s2 = 1; s2 < SK; ++s2) {
                        const float* o = red + ((size_t)(wave + s2) * WAVE + lane) * 8;
#pragma unroll
                        for (int e = 0; e < 8; ++e) acc[t][j][e] += o[e];
                    }
                }
                __syncthreads();
            }
    }

    if (ks != 0) return;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        if (!nlive_j[j]) continue;
        const int n = n_j[j];
        const float bv = Bias ? (float)*(const __hip_bfloat16*)(Bias + n) : 0.f;
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = blockIdx.y * (MT * 16) + t * 16 + 2 * e + (lane >> 4);
                if (m < M) C[(size_t)m * c_ld + n] = (__hip_bfloat16)(acc[t][j][e] + bv);
            }
    }
}

/* SK law: libr4d's narrow_ksplit (r4d_gemm_fp8a8.hip), because two byte-level contracts read
 * it -- a row slice issued with split_n must accumulate as the whole GEMM does, and the gated
 * row must refuse exactly the grids this law would split. Split only when K >= 512 and the
 * unsplit grid (gx blocks of mw waves over the weight rows) is under 128 waves; the count
 * follows libr4d's slice-width rounding, then caps to this kernel's legal splits (SK | 16 and
 * SK | ngroups, whole 128-groups a split). */
static int fp8_sk(int M, int K, int N) {
    if (K < 512) return 1;
    int mw = 4;
    if (M > 16 && (N % 128) == 0) mw = 8;
    if (M > 32 && N >= 2048 && (N % 256) == 0) mw = 16;
    const int waves0 = ((N + mw * 16 - 1) / (mw * 16)) * mw;
    if (waves0 >= 128) return 1;
    const int want = (128 + waves0 - 1) / waves0;
    int ckw = ((K / want) + 127) / 128 * 128;
    if (ckw < 128) ckw = 128;
    int k2 = (K + ckw - 1) / ckw;
    if (k2 > 64 && (long long)N * M < 8192) k2 = 64;
    if (k2 < 2) return 1;
    const int fk = (K % 256 == 0) ? 256 : 128;
    const int ck = ((K + k2 - 1) / k2 + fk - 1) / fk * fk;
    k2 = (K + ck - 1) / ck;
    if (k2 < 2) return 1;
    const int ngroups = K / 128;
    for (int s : { 16, 8, 4, 2 })
        if (s <= k2 && (ngroups % s) == 0) return s;
    return 1;
}

/* The activation plane: deq16_exact over the whole [M, K) activation, once per launch.
 * In the GEMM kernel every wave re-dequantises the same codes for its column (over a
 * thousand wave-columns at wide N); the plane pays it once. Values are deq16_exact's
 * own, so products through the plane are bit-identical. */
__global__ __launch_bounds__(256) void fp8a8_adq_kernel(const uint8_t* __restrict__ A,
                                                        uint16_t* __restrict__ Adq,
                                                        int M, int K, int a_ld) {
    const int idx = blockIdx.x * (int)blockDim.x + threadIdx.x;
    if (idx >= M * (K / 16)) return;
    const int r = idx / (K / 16), c = (idx % (K / 16)) * 16;
    const int4v av = *(const int4v*)(A + (size_t)r * a_ld + c);
    const v16bf f = deq16_exact((const uint32_t*)&av);
    uint4* dst = (uint4*)(Adq + (size_t)r * K + c);
    const uint4* src = (const uint4*)&f;
    dst[0] = src[0];
    dst[1] = src[1];
}

/* ---- the prefill tile (gemm_fp8a8_tiled past one 64-row band) ------------------------------
 *
 * The narrow kernel above at M = 2048 re-reads the activation once per 16-column tile and keeps
 * no operand on chip; this is halo_gemm_bf16_nt_tiled's block shape over the same arithmetic.
 * Eight waves stage a BM x 64 activation tile and a BN x 64 weight tile through LDS as bf16 --
 * every operand decoded once per block, not once per wave-column -- and each wave computes a
 * (BM/2) x (BN/4) patch, the next K tile loading into registers while the current one is
 * multiplied.
 *
 * Contract: byte-identical to the narrow kernel. Every product sees the same bf16 values
 * (deq16_exact's for the activation, from the plane or decoded while staging; deq16_e4m3's for
 * the weight, decoded per dword exactly as there), each 128-group runs its eight 16-K WMMA
 * steps from zero in K order into `tmp`, and folds into `acc` with fmaf(tmp, (as * ws) * FIX)
 * in group order; the store adds the same +0.f. Only where operands sit changes.
 *
 * The weight is staged from halo_fp8_frag.h's order: a (16 row, 64 k) tile is 1 KiB with byte
 * h*512 + row*32 + t*8 + j holding k = t*16 + h*8 + j, so a 16-byte chunk is two 8-wide k runs
 * of one row. The tiled row constrains N % 128 == 0, so a block's columns share one weight
 * scale per group (n0 is a multiple of BN, which divides 128). */
#define FT_BK 64
#define FT_PITCH (FT_BK + 8)    // halo_gemm_bf16_nt_tiled's pitch: conflict-free fragment reads

/* The kernel's body. OWN: it declares its LDS itself (the plain kernel); otherwise it takes the
 * caller's (the two-body kernel, one set for both bodies), as halo_gemm_mxfp4a8_tiled.hip splits
 * it, where a body taking its LDS by pointer measured slower on full blocks. PART: a block the M
 * edge cuts, which skips the WMMAs (and folds) of its padding fragments past M; their rows are
 * dropped at the store anyway. */
template <int TM, int TN, bool APLANE, bool PART, bool OWN>
__device__ __forceinline__ void fp8_tiled_block(
        const uint8_t* __restrict__ A, const uint16_t* __restrict__ Adq,
        const float* __restrict__ Ascale, const uint8_t* __restrict__ W,
        const uint16_t* __restrict__ Ws, __hip_bfloat16* __restrict__ C,
        int M, int K, int N, int a_ld, int sa_ld, int sw_ld, int c_ld, int m0, int n0,
        unsigned short* __restrict__ sA_in, unsigned short* __restrict__ sW_in,
        float* __restrict__ sS_in) {
    constexpr int BM = 2 * TM * 16, BN = 4 * TN * 16;
    __shared__ __attribute__((aligned(16))) unsigned short sA_own[OWN ? BM * FT_PITCH : 8];
    __shared__ __attribute__((aligned(16))) unsigned short sW_own[OWN ? BN * FT_PITCH : 8];
    __shared__ float sS_own[OWN ? BM : 1];              // the current group's activation scales
    unsigned short* __restrict__ sA = OWN ? sA_own : sA_in;
    unsigned short* __restrict__ sW = OWN ? sW_own : sW_in;
    float* __restrict__ sS = OWN ? sS_own : sS_in;
    /* Activation chunks a thread stages per K tile: 16 bytes of the bf16 plane (8 values), or 16
     * e4m3 codes (16 values, two LDS stores) when there is no plane. */
    constexpr int ACH = APLANE ? 8 : 16;
    constexpr int AL = BM * (FT_BK / ACH) / 256;
    constexpr int WL = BN * FT_BK / 16 / 256;           // 16-byte weight chunks a thread
    static_assert(AL >= 1 && WL >= 1, "a tile is whole chunks for every thread");

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave / 4, wn = wave % 4;
    /* PART: this wave's M fragments holding a row below M, 0..TM and wave-uniform. */
    const int live = PART ? __builtin_amdgcn_readfirstlane(
                                min(TM, max(0, (M + 15) / 16 - (m0 / 16 + wm * TM))))
                          : TM;

    const uint8_t* asrc[AL];
    const uint16_t* psrc[AL];
    unsigned short* adst[AL];
#pragma unroll
    for (int i = 0; i < AL; ++i) {
        constexpr int CPR = FT_BK / ACH;
        const int c = tid + 256 * i, r = c / CPR, e = (c % CPR) * ACH;
        const int gr = m0 + r < M ? m0 + r : M - 1;
        asrc[i] = A + (size_t)gr * a_ld + e;
        psrc[i] = Adq + (size_t)gr * K + e;
        adst[i] = sA + r * FT_PITCH + e;
    }
    const uint8_t* wsrc[WL];
    unsigned short* wdst[WL];
#pragma unroll
    for (int i = 0; i < WL; ++i) {
        const int c = tid + 256 * i, t = c >> 6, o = (c & 63) * 16;
        const int h = o >> 9, row = (o >> 5) & 15, f = (o & 31) >> 3;
        wsrc[i] = W + (size_t)((n0 >> 4) + t) * (size_t)(K >> 6) * 1024 + o;
        wdst[i] = sW + (t * 16 + row) * FT_PITCH + f * 16 + h * 8;
    }
    const int srow = tid < BM ? (m0 + tid < M ? m0 + tid : M - 1) : 0;
    const float* sp = Ascale + (size_t)srow * sa_ld;
    const uint16_t* wsp = Ws + (size_t)(n0 >> 7) * sw_ld;

    const unsigned short* fa = sA + (wm * TM * 16 + (lane & 15)) * FT_PITCH;
    const unsigned short* fw = sW + (wn * TN * 16 + (lane & 15)) * FT_PITCH;

    v8f acc[TM][TN], tmp[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) { acc[i][j] = (v8f)0.f; tmp[i][j] = (v8f)0.f; }

    halo_u32x4 ra[AL], rw[WL];
    float rs = 0.f;
    auto fetch = [&](int kt) {
        const int k = kt * FT_BK;
#pragma unroll
        for (int i = 0; i < AL; ++i)
            ra[i] = APLANE ? *(const halo_u32x4*)(psrc[i] + k) : *(const halo_u32x4*)(asrc[i] + k);
#pragma unroll
        for (int i = 0; i < WL; ++i) rw[i] = *(const halo_u32x4*)(wsrc[i] + (size_t)(kt) * 1024);
        if (!(kt & 1) && tid < BM) rs = sp[kt >> 1];
    };
    fetch(0);

    const int nk = K / FT_BK;
    for (int kt = 0; kt < nk; ++kt) {
        __syncthreads();                                // the previous tile's fragments are read
#pragma unroll
        for (int i = 0; i < AL; ++i) {
            if constexpr (APLANE) {
                *(halo_u32x4*)adst[i] = ra[i];
            } else {
                const v16bf f = deq16_exact((const uint32_t*)&ra[i]);
                const halo_u32x4* q = (const halo_u32x4*)&f;
                *(halo_u32x4*)adst[i] = q[0];
                *(halo_u32x4*)(adst[i] + 8) = q[1];
            }
        }
#pragma unroll
        for (int i = 0; i < WL; ++i) {
            const v16bf f = halo::deq16_e4m3((const uint32_t*)&rw[i]);
            const halo_u32x4* q = (const halo_u32x4*)&f;
            *(halo_u32x4*)wdst[i] = q[0];                // k run t*16 + h*8 .. +8
            *(halo_u32x4*)(wdst[i] + 16) = q[1];         // the same half of fragment t + 1
        }
        if (!(kt & 1) && tid < BM) sS[tid] = rs;
        __syncthreads();
        if (kt + 1 < nk) fetch(kt + 1);
#pragma unroll
        for (int s = 0; s < FT_BK / 16; ++s) {
            v16bf af[TM], bf[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) af[i] = fragP(fa + i * 16 * FT_PITCH + s * 16);
#pragma unroll
            for (int j = 0; j < TN; ++j) bf[j] = fragP(fw + j * 16 * FT_PITCH + s * 16);
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                if (PART && i >= live) break;           // wave-uniform: padding fragments past M
#pragma unroll
                for (int j = 0; j < TN; ++j) tmp[i][j] = halo::mma(af[i], bf[j], tmp[i][j]);
            }
            if constexpr (!PART) {
                __builtin_amdgcn_sched_group_barrier(0x100, TM + TN, 0);    /* DS reads */
                __builtin_amdgcn_sched_group_barrier(0x008, TM * TN, 0);    /* WMMA */
            }
        }
        if (kt & 1) {
            /* The 128-group closed: fold it with the narrow kernel's scale product. */
            const float wsc = __uint_as_float((uint32_t)wsp[kt >> 1] << 16);
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                if (PART && i >= live) break;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float sc = sS[(wm * TM + i) * 16 + accRow(lane, e)] * wsc * HALO_FP8_FIX;
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        acc[i][j][e] = fmaf(tmp[i][j][e], sc, acc[i][j][e]);
                        tmp[i][j][e] = 0.f;
                    }
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n = n0 + (wn * TN + j) * 16 + (lane & 15);
            const int mb = m0 + (wm * TM + i) * 16;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = mb + accRow(lane, e);
                if (m < M) C[(size_t)m * c_ld + n] = (__hip_bfloat16)(acc[i][j][e] + 0.f);
            }
        }
}

/* halo_gemm_bf16_nt_tiled's block order: groups of 8 activation bands, M fastest, so the bands
 * of one column group run together and share its weight tile through the cache. */
__device__ __forceinline__ void fp8_tiled_origin(int BM, int BN, int* m0, int* n0) {
    constexpr unsigned kGroupM = 8;
    const unsigned per = kGroupM * gridDim.x, lin = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned first = lin / per * kGroupM, rr = lin % per;
    const unsigned gm = min(gridDim.y - first, kGroupM);
    *m0 = (int)(first + rr % gm) * BM;
    *n0 = (int)(rr / gm) * BN;
}

/* The plain kernel: every block the unguarded body, over the body's own LDS. */
template <int TM, int TN, bool APLANE>
__global__ __launch_bounds__(256) void halo_gemm_fp8a8_tiled_kernel(
        const uint8_t* __restrict__ A, const uint16_t* __restrict__ Adq,
        const float* __restrict__ Ascale, const uint8_t* __restrict__ W,
        const uint16_t* __restrict__ Ws, __hip_bfloat16* __restrict__ C,
        int M, int K, int N, int a_ld, int sa_ld, int sw_ld, int c_ld) {
    constexpr int BM = 2 * TM * 16, BN = 4 * TN * 16;
    int m0, n0;
    fp8_tiled_origin(BM, BN, &m0, &n0);
    fp8_tiled_block<TM, TN, APLANE, false, true>(A, Adq, Ascale, W, Ws, C, M, K, N, a_ld, sa_ld,
                                                 sw_ld, c_ld, m0, n0, nullptr, nullptr, nullptr);
}

/* The two-body kernel, for a thin last row of blocks: a block of that row takes the guarded body,
 * every other block the unguarded one, uniform over the block, over one set of LDS arrays. A
 * guard in the plain kernel would slow every full block. */
template <int TM, int TN, bool APLANE>
__global__ __launch_bounds__(256) void halo_gemm_fp8a8_tiled_part_kernel(
        const uint8_t* __restrict__ A, const uint16_t* __restrict__ Adq,
        const float* __restrict__ Ascale, const uint8_t* __restrict__ W,
        const uint16_t* __restrict__ Ws, __hip_bfloat16* __restrict__ C,
        int M, int K, int N, int a_ld, int sa_ld, int sw_ld, int c_ld) {
    constexpr int BM = 2 * TM * 16, BN = 4 * TN * 16;
    __shared__ __attribute__((aligned(16))) unsigned short sA[BM * FT_PITCH];
    __shared__ __attribute__((aligned(16))) unsigned short sW[BN * FT_PITCH];
    __shared__ float sS[BM];
    int m0, n0;
    fp8_tiled_origin(BM, BN, &m0, &n0);
    if (m0 + BM > M)
        fp8_tiled_block<TM, TN, APLANE, true, false>(A, Adq, Ascale, W, Ws, C, M, K, N, a_ld,
                                                     sa_ld, sw_ld, c_ld, m0, n0, sA, sW, sS);
    else
        fp8_tiled_block<TM, TN, APLANE, false, false>(A, Adq, Ascale, W, Ws, C, M, K, N, a_ld,
                                                      sa_ld, sw_ld, c_ld, m0, n0, sA, sW, sS);
}

/* Tile choice, halo_gemm_bf16_nt_tiled's rule: the 128 x 128 tile once its grid reaches 32
 * blocks, the 64 x 64 tile below. A partial last band of rows (M % BM) would otherwise run every
 * WMMA of its padding fragments: at M 129 the second band is 127 rows of padding, and the GEMM
 * is matrix-bound there, so it cost as much as a full band. */
static int fp8_tiled_launch(const uint8_t* A, const uint16_t* Adq, const float* As,
                            const uint8_t* W, const uint16_t* Ws, __hip_bfloat16* C, int M,
                            int K, int N, int a_ld, int sa_ld, int sw_ld, int c_ld,
                            hipStream_t stream) {
    const long long g128 = (long long)(N / 128) * ((M + 127) / 128);
    const bool big = g128 >= 32;
    const int bm = big ? 128 : 64, bn = big ? 128 : 64;
    const dim3 grid((unsigned)(N / bn), (unsigned)((M + bm - 1) / bm));
    /* A row of blocks the M edge cuts takes the two-body kernel. Measured on gfx1151 at 34816 x
     * 5120: M 129 2219 -> 1527 us, 258 3263 -> 2519, 65 1189 -> 1007; whole bands unchanged. */
    const bool part = M % bm != 0;
#define FT_LAUNCH(TM, TN, P) do { if (part) \
        hipLaunchKernelGGL((halo_gemm_fp8a8_tiled_part_kernel<TM, TN, P>), grid, \
        dim3(256), 0, stream, A, Adq, As, W, Ws, C, M, K, N, a_ld, sa_ld, sw_ld, c_ld); else \
        hipLaunchKernelGGL((halo_gemm_fp8a8_tiled_kernel<TM, TN, P>), grid, \
        dim3(256), 0, stream, A, Adq, As, W, Ws, C, M, K, N, a_ld, sa_ld, sw_ld, c_ld); } while (0)
    if (big) { if (Adq) FT_LAUNCH(4, 2, true); else FT_LAUNCH(4, 2, false); }
    else     { if (Adq) FT_LAUNCH(2, 1, true); else FT_LAUNCH(2, 1, false); }
#undef FT_LAUNCH
    return hipGetLastError() == hipSuccess ? 0 : -5;
}

template <int MT, int TEMPORAL, int NT = 1, int BW = 16>
static int fp8_launch_bw(const uint8_t* A, const float* As, const uint8_t* W, const uint16_t* Ws,
                      const uint16_t* Bias, __hip_bfloat16* C, int M, int K, int N,
                      int a_ld, int sa_ld, int w_ld, int sw_ld, int c_ld, int SK,
                      hipStream_t stream, const uint16_t* Adq = nullptr) {
    if (SK < 1 || (16 % SK) != 0) SK = 1;
    const int wpb = BW / SK;
    const int mtiles = (M + MT * 16 - 1) / (MT * 16);
    const int nblocks = (N / 16 + wpb * NT - 1) / (wpb * NT);
    const size_t smem = SK > 1 ? (size_t)BW * WAVE * 8 * sizeof(float) : 0;
    hipLaunchKernelGGL((halo_gemm_fp8a8_kernel<MT, TEMPORAL, NT, BW>), dim3(nblocks, mtiles), dim3(BW * 32),
                       smem, stream, A, As, W, Ws, Bias, C, M, K, N, a_ld, sa_ld, w_ld, sw_ld,
                       c_ld, SK, Adq);
    return hipGetLastError() == hipSuccess ? 0 : -5;
}

/* The block's wave count. Sixteen waves a block leave a small weight a handful of blocks (four at
 * N 2048, K 4096, SK 1: 64 waves on 40 CUs), so at one row tile (MT 1) the block narrows until
 * the grid has kFp8MinBlocks of them; the split, hence every output's accumulation order, does not
 * change. Measured on gfx1151 at M 1: 2048 x 4096 60.3 -> 55.3 us, 1024 x 5120 41.7 -> 38.3,
 * 1024 x 2048 22.0 -> 20.0, 2048 x 512 14.7 -> 12.9; shapes already at 16 blocks keep 16 waves
 * (8192 x 2048 is 99 us at 16 waves, 113 at 8). Taller tiles keep 16: the narrower block spilled
 * more there and lost up to 25% (1024 x 5120 at M 64). */
enum { kFp8MinBlocks = 16 };
template <int MT, int TEMPORAL, int NT = 1>
static int fp8_launch(const uint8_t* A, const float* As, const uint8_t* W, const uint16_t* Ws,
                      const uint16_t* Bias, __hip_bfloat16* C, int M, int K, int N,
                      int a_ld, int sa_ld, int w_ld, int sw_ld, int c_ld, int SK,
                      hipStream_t stream, const uint16_t* Adq = nullptr) {
    if (SK < 1 || (16 % SK) != 0) SK = 1;
    const long long mtiles = (M + MT * 16 - 1) / (MT * 16);
    int bw = 16;
    while (MT == 1 && bw > 4 && bw / 2 >= SK &&
           ((N / 16 + (bw / SK) * NT - 1) / ((bw / SK) * NT)) * mtiles < kFp8MinBlocks)
        bw /= 2;
#define FBW(V) return fp8_launch_bw<MT, TEMPORAL, NT, V>(A, As, W, Ws, Bias, C, M, K, N, a_ld, \
                                                          sa_ld, w_ld, sw_ld, c_ld, SK, stream, Adq)
    switch (bw) { case 4: FBW(4); case 8: FBW(8); default: FBW(16); }
#undef FBW
}

static int fp8_gemm(long a, long ascale, long wq, long ws, long bias, long c, int M, int K, int N,
                    int a_ld, int sa_ld, int w_ld, int sw_ld, int c_ld, int sk_want,
                    hipStream_t stream, long adq = 0) {
    if (M < 1 || K < 128 || (K % 128) || N < 16 || (N % 16)) return -1;
    auto A = (const uint8_t*)a; auto As = (const float*)ascale;
    auto W = (const uint8_t*)wq; auto Ws = (const uint16_t*)ws;
    auto B = (const uint16_t*)bias; auto C = (__hip_bfloat16*)c;
    if (c_ld <= 0) c_ld = N;
    if (a_ld <= 0) a_ld = K;
    if (sa_ld <= 0) sa_ld = K / 128;
    if (w_ld <= 0) w_ld = K;
    if (sw_ld <= 0) sw_ld = K / 128;
    const uint16_t* Adq = nullptr;
    if (adq) {
        Adq = (const uint16_t*)(uintptr_t)adq;
        const int total = M * (K / 16);
        hipLaunchKernelGGL((fp8a8_adq_kernel), dim3((total + 255) / 256), 256, 0, stream, 
            A, (uint16_t*)Adq, M, K, a_ld);
        if (hipGetLastError() != hipSuccess) return -1;
    }
    const int MT = M <= 16 ? 1 : M <= 32 ? 2 : M <= 48 ? 3 : 4;
    const int sk = sk_want > 0 ? sk_want
                               : fp8_sk(M, K, N);
    /* Two columns a wave where the accumulator count allows it (MT <= 2): the activation
     * dequant and its loads are paid once for two weight fragments. Past MT 2 the
     * MT*NT accumulators spill, so the tall bands keep one column. */
    switch (MT) {
      case 1: return fp8_launch<1, 1, 2>(A, As, W, Ws, B, C, M, K, N, a_ld, sa_ld, w_ld, sw_ld, c_ld, sk, stream, Adq);
      case 2: return fp8_launch<2, 1, 2>(A, As, W, Ws, B, C, M, K, N, a_ld, sa_ld, w_ld, sw_ld, c_ld, sk, stream, Adq);
      case 3: return fp8_launch<3, 1>(A, As, W, Ws, B, C, M, K, N, a_ld, sa_ld, w_ld, sw_ld, c_ld, sk, stream, Adq);
      default: return fp8_launch<4, 1>(A, As, W, Ws, B, C, M, K, N, a_ld, sa_ld, w_ld, sw_ld, c_ld, sk, stream, Adq);
    }
}

/* The LDS-tiled kernel over [M, N) with an optional activation plane `adq` (decoded here), the
 * output at pitch c_ld. Byte-identical to the narrow kernel at SK 1 (the tiled kernel's contract),
 * so a caller may take it wherever the narrow kernel would run unsplit. */
static int fp8_tiled_body(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                          int a_ld, int sa_ld, int sw_ld, int c_ld, hipStream_t stream, long adq) {
    if (K < 128 || (K % 128) || (N % 128)) return -1;
    if (a_ld <= 0) a_ld = K;
    if (sa_ld <= 0) sa_ld = K / 128;
    if (sw_ld <= 0) sw_ld = K / 128;
    const uint16_t* Adq = nullptr;
    if (adq) {
        Adq = (const uint16_t*)(uintptr_t)adq;
        const long long total = (long long)M * (K / 16);
        hipLaunchKernelGGL((fp8a8_adq_kernel), dim3((unsigned)((total + 255) / 256)), 256, 0, stream,
            (const uint8_t*)a, (uint16_t*)Adq, M, K, a_ld);
        if (hipGetLastError() != hipSuccess) return -1;
    }
    return fp8_tiled_launch((const uint8_t*)a, Adq, (const float*)ascale, (const uint8_t*)wq,
                            (const uint16_t*)ws, (__hip_bfloat16*)c, M, K, N, a_ld, sa_ld, sw_ld,
                            c_ld, stream);
}

/* Past HALO_FP8_TILED_MIN_M rows the narrow rows run the tiled kernel: the narrow kernel gives a
 * wave one column and re-reads and re-decodes the weight for every MT*16 rows, so its time grows
 * with M (4.9 ms at M 64, N 34816, K 5120, against 0.9 ms at M 1), while the tile decodes each
 * weight fragment once for its 64 or 128 rows. Only where the split law gives SK 1 (the bytes
 * agree there) and the weight is the plain frag64 plane. Measured crossover: below 17 rows the
 * narrow kernel is faster (0.91 against 1.12 ms at N 34816), from 17 the tile (1.12 against 1.72). */
#define HALO_FP8_TILED_MIN_M 16
static int fp8_tiled_min_m(void) { return HALO_FP8_TILED_MIN_M; }
static bool fp8_takes_tiled(int M, int K, int N, int w_ld, int sk_want) {
    if (M <= fp8_tiled_min_m() || (N % 128) || (K % 128) || (w_ld > 0 && w_ld != K)) return false;
    return (sk_want > 0 ? sk_want : fp8_sk(M, K, N)) == 1;
}

/* The narrow row's launch with the split it must take (law_sk: a `split_n` slice folds as the
 * whole GEMM does); halo_gemm_fp8a8_nt_m16 below is libr4d's raw signature over it. */
static int fp8_nt_m16(long a, long ascale, long wq, long ws, long c,
                      int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                      long scratch, long scratch_bytes, long stream, int sk_want) {
    if (M > HALO_GEMM_FP8_MAX_M) return -2;
    /* The scratch carries the activation plane when the caller sized it (the row's
     * scratch hook asks for exactly that); split-K itself is in-block. */
    const long adq = (scratch && scratch_bytes >= (long)M * K * 2) ? scratch : 0;
    if (fp8_takes_tiled(M, K, N, w_ld, sk_want))
        return fp8_tiled_body(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, sw_ld, N,
                              (hipStream_t)stream, adq);
    return fp8_gemm(a, ascale, wq, ws, 0, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld, N, sk_want,
                    (hipStream_t)stream, adq);
}

int halo_gemm_fp8a8_nt_m16(long a, long ascale, long wq, long ws, long c,
                           int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                           long scratch, long scratch_bytes, long stream) {
    return fp8_nt_m16(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld, scratch,
                      scratch_bytes, stream, 0);
}

/* The split a call with `split_n` must take: the law run on the whole width, so a row slice
 * of a linear accumulates its rows in the whole GEMM's order and the bytes agree. */
static int law_sk(const RadArgs* a, long long M, long long K, long long N) {
    const long long law_n = halo_geti_or(a, "split_n", N);
    if (law_n <= N) return 0;
    return fp8_sk((int)M, (int)K, (int)law_n);
}

long halo_gemm_fp8a8_nt_m16_scratch(int M, int K, int N) {
    (void)M; (void)K; (void)N;
    return 0;
}

int halo_gemm_fp8a8_nt_m16_max_m(void) { return HALO_GEMM_FP8_MAX_M; }

static int g_fp8_pin_mw = 0;
void halo_gemm_fp8a8_pin_mw(int mw) { g_fp8_pin_mw = mw; }

/* The token-tile pin (halo.h): only instantiated counts are accepted, anything else
 * clears the override. The gated kernel's MT is its token-tile count, so the pin
 * selects it where it covers M; the ungated narrow kernel has no such parameter and
 * its bytes cannot depend on it. */
static int g_fp8_pin_nt = 0;
extern "C" void halo_gemm_fp8a8_pin_nt(int nt) {
    g_fp8_pin_nt = (nt == 1 || nt == 2 || nt == 3 || nt == 4) ? nt : 0;
}

int halo_gemm_fp8a8_bias_nt_m16(long a, long ascale, long wq, long ws, long bias, long c,
                                int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                int c_ld, long stream) {
    if (M > HALO_GEMM_FP8_MAX_M) return -2;
    return fp8_gemm(a, ascale, wq, ws, bias, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld, c_ld, 1,
                    (hipStream_t)stream);
}

/* The tiled row's launch with the op's scratch (the activation plane when it fits);
 * halo_gemm_fp8a8_tiled below is libr4d's raw signature over it, with none. */
static int fp8_tiled(long a, long ascale, long wq, long ws, long c,
                     int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                     long stream, long scratch, long scratch_bytes) {
    /* The prefill band: the same wave-per-column kernel with four row tiles a wave and the
     * M-tile grid covering any M. Its tile differs from libr4d's (a 128x128 LDS tile); what the
     * row promises -- the math and the store -- is the narrow kernel's. */
    const long adq = (scratch && scratch_bytes >= (long)M * K * 2) ? scratch : 0;
    if ((M <= HALO_GEMM_FP8_MAX_M && !fp8_takes_tiled(M, K, N, w_ld, 1)) || (N % 128))
        return fp8_gemm(a, ascale, wq, ws, 0, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld, N, 1,
                        (hipStream_t)stream, adq);
    /* Past one 64-row band, the LDS-tiled kernel: the same bytes (see its contract). */
    (void)w_ld;
    return fp8_tiled_body(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, sw_ld, N,
                          (hipStream_t)stream, adq);
}

int halo_gemm_fp8a8_tiled(long a, long ascale, long wq, long ws, long c,
                          int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                          long stream) {
    return fp8_tiled(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld, stream, 0, 0);
}

int halo_gemm_fp8a8_block(void) { return 128; }

/* ---- the int8 twins (gemm_i8a8_nt_m16, gemm_i8a8_tiled) --------------------------------------
 *
 * The same wave-per-column kernel at i8 codes: the activation plain row-major (quant_act_i8g's
 * output), the weight in the fp8 family's fragment order, the iu8 WMMA accumulating in int32
 * per 128-wide group. The
 * weight scale is tile-major ([N/16][K/128][16], halo_fp8_layout.cpp's i8 hooks), so a lane's
 * scale is one bf16 at its column; the activation scale is per (row, group) and folds per
 * accumulator slot, exactly as in the fp8 kernel above. */
typedef int v4i_ __attribute__((ext_vector_type(4)));
typedef int v8i_ __attribute__((ext_vector_type(8)));

template <int MT, int BW = 16>
__global__ __launch_bounds__(BW * 32) void halo_gemm_i8a8_kernel(
        const uint8_t* __restrict__ A, const float* __restrict__ Ascale,
        const uint8_t* __restrict__ W, const uint16_t* __restrict__ Ws,
        __hip_bfloat16* __restrict__ C,
        int M, int K, int N, int a_ld, int sa_ld, int c_ld, int SK) {
    HALO_DYN_LDS(float, red);
    const int lane = threadIdx.x & (WAVE - 1), wave = threadIdx.x / WAVE;
    const int wpb = BW / SK;
    const int ctile = wave / SK, ks = wave % SK;
    const int slot = lane & 15;
    const int n = (blockIdx.x * wpb + ctile) * 16 + slot;
    const bool nlive = n < N;
    const int ns = nlive ? n : 0;

    const int Kb = K / SK, kbase = ks * Kb;
    const int ngroups = Kb / 128, nkb = K / 128;
    const float wsc0 = nlive ? 1.0f : 0.0f;

    const uint8_t* Ap[MT];
    int mrow[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        const int m = blockIdx.y * (MT * 16) + t * 16 + slot;
        mrow[t] = m < M ? m : M - 1;
        Ap[t] = A + (size_t)mrow[t] * a_ld + kbase;
    }

    v8f acc[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[t] = (v8f)0.f;

    if constexpr (MT == 1) {
    /* At one row tile (decode) a group's loads -- its weight chunk, the activation's 128 codes and
     * the scales its fold reads -- are issued a group ahead, into the registers the group after it
     * frees. A wave walks K one group at a time, and at narrow N (1280-2560 columns, a few hundred
     * waves on the card) each group's DRAM round trip would otherwise sit between two groups'
     * WMMAs. Same products, same order. Wider tiles keep the in-place loads below: the ring's
     * registers would cost them occupancy. */
    struct Grp { uint4 lo2[4], hi2[4], av[8]; float wsc, asc[8]; };
    auto load = [&](Grp& q, int g) {
        const int k0 = g * 128, gabs = kbase / 128 + g;
        q.wsc = wsc0 * __uint_as_float(
            (uint32_t)Ws[((size_t)(ns >> 4) * nkb + gabs) * 16 + slot] << 16);
#pragma unroll
        for (int c2 = 0; c2 < 2; ++c2) {
            const int kabs = kbase + k0 + c2 * 64;
            const uint8_t* ch = W + ((size_t)(ns >> 4) * (K >> 6) + (kabs >> 6)) * 1024 + slot * 32;
            q.lo2[2 * c2] = *(const uint4*)ch;
            q.lo2[2 * c2 + 1] = *(const uint4*)(ch + 16);
            q.hi2[2 * c2] = *(const uint4*)(ch + 16 * 32);
            q.hi2[2 * c2 + 1] = *(const uint4*)(ch + 16 * 32 + 16);
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) q.av[u] = *(const uint4*)(Ap[0] + k0 + u * 16);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            int mr = blockIdx.y * 16 + 2 * e + (lane >> 4);
            mr = mr < M ? mr : M - 1;
            q.asc[e] = Ascale[(size_t)mr * sa_ld + gabs];
        }
    };
    Grp cur, nxt;
    load(cur, 0);
    for (int g = 0; g < ngroups; ++g) {
        /* Unconditional (the last group reloads itself), so the waitcnt pass can count them. */
        load(nxt, g + 1 < ngroups ? g + 1 : g);
        v8i_ tmp = (v8i_)0;
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const uint4 l = cur.lo2[u >> 1], h = cur.hi2[u >> 1];
            v4i_ bf, af;
            bf[0] = (int)((u & 1) ? l.z : l.x); bf[1] = (int)((u & 1) ? l.w : l.y);
            bf[2] = (int)((u & 1) ? h.z : h.x); bf[3] = (int)((u & 1) ? h.w : h.y);
            af[0] = (int)cur.av[u].x; af[1] = (int)cur.av[u].y;
            af[2] = (int)cur.av[u].z; af[3] = (int)cur.av[u].w;
            tmp = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af, true, bf, tmp, false);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float sc = cur.asc[e] * cur.wsc;
            acc[0][e] = fmaf((float)tmp[e], sc, acc[0][e]);
        }
        cur = nxt;
    }
    } else {
    for (int g = 0; g < ngroups; ++g) {
        const int k0 = g * 128;
        const int gabs = kbase / 128 + g;
        const float wsc = wsc0 * __uint_as_float(
            (uint32_t)Ws[((size_t)(ns >> 4) * nkb + gabs) * 16 + slot] << 16);
        v8i_ tmp[MT];
#pragma unroll
        for (int t = 0; t < MT; ++t) tmp[t] = (v8i_)0;
        /* A lane's fragments of one 64-wide chunk are two contiguous 32-byte runs (low and high
         * halves, 512 bytes apart): four 16-byte loads serve the chunk's four k steps. */
        uint4 lo2[4], hi2[4];
#pragma unroll
        for (int c2 = 0; c2 < 2; ++c2) {
            const int kabs = kbase + k0 + c2 * 64;
            const uint8_t* ch = W + ((size_t)(ns >> 4) * (K >> 6) + (kabs >> 6)) * 1024 + slot * 32;
            lo2[2 * c2] = *(const uint4*)ch;
            lo2[2 * c2 + 1] = *(const uint4*)(ch + 16);
            hi2[2 * c2] = *(const uint4*)(ch + 16 * 32);
            hi2[2 * c2 + 1] = *(const uint4*)(ch + 16 * 32 + 16);
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            v4i_ bf;
            {
                const uint4 l = lo2[u >> 1], h = hi2[u >> 1];
                bf[0] = (int)((u & 1) ? l.z : l.x); bf[1] = (int)((u & 1) ? l.w : l.y);
                bf[2] = (int)((u & 1) ? h.z : h.x); bf[3] = (int)((u & 1) ? h.w : h.y);
            }
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                /* The activation is quant_act_i8g's: plain row-major codes, so the fragment
                 * is the row's sixteen consecutive bytes. (The shuffled plane belongs to
                 * quant_act_i8 and the skinny int8 GEMMs, not to this family.) */
                const uint4 av4 = *(const uint4*)(Ap[t] + k0 + u * 16);
                v4i_ af;
                af[0] = (int)av4.x; af[1] = (int)av4.y; af[2] = (int)av4.z; af[3] = (int)av4.w;
                tmp[t] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af, true, bf, tmp[t], false);
            }
        }
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                int mr = blockIdx.y * (MT * 16) + t * 16 + 2 * e + (lane >> 4);
                mr = mr < M ? mr : M - 1;
                const float sc = Ascale[(size_t)mr * sa_ld + gabs] * wsc;
                acc[t][e] = fmaf((float)tmp[t][e], sc, acc[t][e]);
            }
    }

    }

    if (SK > 1) {
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            float* mine = red + ((size_t)wave * WAVE + lane) * 8;
#pragma unroll
            for (int e = 0; e < 8; ++e) mine[e] = acc[t][e];
            __syncthreads();
            if (ks == 0) {
                for (int s2 = 1; s2 < SK; ++s2) {
                    const float* o = red + ((size_t)(wave + s2) * WAVE + lane) * 8;
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[t][e] += o[e];
                }
            }
            __syncthreads();
        }
    }

    if (!nlive || ks != 0) return;
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int m = blockIdx.y * (MT * 16) + t * 16 + 2 * e + (lane >> 4);
            if (m < M) C[(size_t)m * c_ld + n] = (__hip_bfloat16)acc[t][e];
        }
}

template <int MT>
static int i8_launch(const uint8_t* A, const float* As, const uint8_t* W, const uint16_t* Ws,
                     __hip_bfloat16* C, int M, int K, int N, int a_ld, int sa_ld, int c_ld,
                     int SK, hipStream_t stream) {
    if (SK < 1 || (16 % SK) != 0) SK = 1;
    const int mtiles = (M + MT * 16 - 1) / (MT * 16);
    /* The fp8 kernel's block rule (fp8_launch): narrow the block on a small grid. */
    int bw = 16;
    while (MT == 1 && bw > 4 && bw / 2 >= SK &&
           ((N / 16 + bw / SK - 1) / (bw / SK)) * mtiles < kFp8MinBlocks)
        bw /= 2;
    const int wpb = bw / SK;
    const int nblocks = (N / 16 + wpb - 1) / wpb;
    const size_t smem = SK > 1 ? (size_t)bw * WAVE * 8 * sizeof(float) : 0;
#define I8BW(V) hipLaunchKernelGGL((halo_gemm_i8a8_kernel<MT, V>), dim3(nblocks, mtiles), dim3(V * 32), \
                                   smem, stream, A, As, W, Ws, C, M, K, N, a_ld, sa_ld, c_ld, SK)
    switch (bw) { case 4: I8BW(4); break; case 8: I8BW(8); break; default: I8BW(16); break; }
#undef I8BW
    return hipGetLastError() == hipSuccess ? 0 : -5;
}

static int i8_gemm(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                   int a_ld, int sa_ld, int c_ld, int sk_want, hipStream_t stream) {
    if (M < 1 || K < 128 || (K % 128) || N < 16 || (N % 16)) return -1;
    auto A = (const uint8_t*)a; auto As = (const float*)ascale;
    auto W = (const uint8_t*)wq; auto Ws = (const uint16_t*)ws;
    auto C = (__hip_bfloat16*)c;
    if (c_ld <= 0) c_ld = N;
    if (a_ld <= 0) a_ld = K;
    if (sa_ld <= 0) sa_ld = K / 128;
    const int MT = M <= 16 ? 1 : M <= 32 ? 2 : M <= 48 ? 3 : 4;
    int sk = sk_want > 0 ? sk_want
                         : fp8_sk(M, K, N);
    /* One row tile at K up to 4096 takes four waves a column tile, each a quarter of K, whenever
     * the groups divide: with the kernel's one-group prefetch a wave's K walk is latency-bound,
     * and shorter walks over more waves stream faster. Measured on gfx1151 with the weight
     * streamed from DRAM (K 2560, four rows): N 16384 291 -> 214 us, 12288 191 -> 160, 1280
     * 27.6 -> 21.4 (the occupancy rule's two), 512 at its rule's four. Two ways measured slower
     * at wide N; K 6144 (48 groups, N 2560) is fastest unsplit. */
    if (sk_want <= 0 && MT == 1 && sk < 4 && K <= 4096 && (K / 128) % 4 == 0) sk = 4;
    switch (MT) {
      case 1: return i8_launch<1>(A, As, W, Ws, C, M, K, N, a_ld, sa_ld, c_ld, sk, stream);
      case 2: return i8_launch<2>(A, As, W, Ws, C, M, K, N, a_ld, sa_ld, c_ld, sk, stream);
      case 3: return i8_launch<3>(A, As, W, Ws, C, M, K, N, a_ld, sa_ld, c_ld, sk, stream);
      default: return i8_launch<4>(A, As, W, Ws, C, M, K, N, a_ld, sa_ld, c_ld, sk, stream);
    }
}

/* The narrow int8 launch with the split it must take; halo_gemm_i8a8_nt_m16 below is libr4d's
 * raw signature over it (its weight pitches and scratch are not read: the plane's geometry is K,
 * the split in-block). */
static int i8_nt_m16(long a, long ascale, long wq, long ws, long c,
                     int M, int K, int N, int a_ld, int sa_ld, long stream, int sk_want) {
    if (M > HALO_GEMM_FP8_MAX_M) return -2;
    return i8_gemm(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, N, sk_want, (hipStream_t)stream);
}

int halo_gemm_i8a8_nt_m16(long a, long ascale, long wq, long ws, long c,
                          int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                          long scratch, long scratch_bytes, long stream) {
    (void)w_ld; (void)sw_ld; (void)scratch; (void)scratch_bytes;
    return i8_nt_m16(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, stream, 0);
}

/* ---- the int8 prefill tile (gemm_i8a8_tiled past one 64-row band) ---------------------------
 *
 * The fp8 prefill tile's block shape at i8 codes: both operands stage through LDS as bytes (no
 * decode), each wave a (BM/2) x (BN/4) patch of iu8 WMMAs. A 128-group's product is an exact
 * int32 sum whatever the order of its steps, and the groups fold in ascending order with the
 * narrow kernel's fmaf((float)tmp, as * ws), so the bytes are the narrow kernel's. The weight
 * scale is per column ([N/16][K/128][16], one bf16 a lane), the activation scale per row from
 * LDS. */
#define IT_PITCH 80             // bytes a staged row: 64 + 16, so every row starts 16-aligned

/* The body; OWN and PART as in fp8_tiled_block (padding fragments' WMMAs and folds skipped). */
template <int TM, int TN, bool PART, bool OWN>
__device__ __forceinline__ void i8_tiled_block(
        const uint8_t* __restrict__ A, const float* __restrict__ Ascale,
        const uint8_t* __restrict__ W, const uint16_t* __restrict__ Ws,
        __hip_bfloat16* __restrict__ C, int M, int K, int N, int a_ld, int sa_ld, int c_ld,
        int m0, int n0, uint8_t* __restrict__ sA_in, uint8_t* __restrict__ sW_in,
        float* __restrict__ sS_in) {
    constexpr int BM = 2 * TM * 16, BN = 4 * TN * 16;
    __shared__ __attribute__((aligned(16))) uint8_t sA_own[OWN ? BM * IT_PITCH : 16];
    __shared__ __attribute__((aligned(16))) uint8_t sW_own[OWN ? BN * IT_PITCH : 16];
    __shared__ float sS_own[OWN ? BM : 1];
    uint8_t* __restrict__ sA = OWN ? sA_own : sA_in;
    uint8_t* __restrict__ sW = OWN ? sW_own : sW_in;
    float* __restrict__ sS = OWN ? sS_own : sS_in;
    constexpr int AL = BM * (FT_BK / 16) / 256, WL = BN * FT_BK / 16 / 256;
    static_assert(AL >= 1 && WL >= 1, "a tile is whole chunks for every thread");

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave / 4, wn = wave % 4;
    const int live = PART ? __builtin_amdgcn_readfirstlane(
                                min(TM, max(0, (M + 15) / 16 - (m0 / 16 + wm * TM))))
                          : TM;

    const uint8_t* asrc[AL];
    uint8_t* adst[AL];
#pragma unroll
    for (int i = 0; i < AL; ++i) {
        const int c = tid + 256 * i, r = c >> 2, e = (c & 3) * 16;
        const int gr = m0 + r < M ? m0 + r : M - 1;
        asrc[i] = A + (size_t)gr * a_ld + e;
        adst[i] = sA + r * IT_PITCH + e;
    }
    const uint8_t* wsrc[WL];
    uint8_t* wdst[WL];
#pragma unroll
    for (int i = 0; i < WL; ++i) {
        const int c = tid + 256 * i, t = c >> 6, o = (c & 63) * 16;
        const int h = o >> 9, row = (o >> 5) & 15, f = (o & 31) >> 3;
        wsrc[i] = W + (size_t)((n0 >> 4) + t) * (size_t)(K >> 6) * 1024 + o;
        wdst[i] = sW + (t * 16 + row) * IT_PITCH + f * 16 + h * 8;
    }
    const int srow = tid < BM ? (m0 + tid < M ? m0 + tid : M - 1) : 0;
    const float* sp = Ascale + (size_t)srow * sa_ld;
    const int nkb = K / 128;
    const uint16_t* wsp[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int n = n0 + (wn * TN + j) * 16 + (lane & 15);
        wsp[j] = Ws + (size_t)(n >> 4) * nkb * 16 + (n & 15);
    }

    const uint8_t* fa = sA + (wm * TM * 16 + (lane & 15)) * IT_PITCH;
    const uint8_t* fw = sW + (wn * TN * 16 + (lane & 15)) * IT_PITCH;

    v8f acc[TM][TN];
    v8i_ tmp[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) { acc[i][j] = (v8f)0.f; tmp[i][j] = (v8i_)0; }

    halo_u32x4 ra[AL], rw[WL];
    float rs = 0.f;
    auto fetch = [&](int kt) {
#pragma unroll
        for (int i = 0; i < AL; ++i) ra[i] = *(const halo_u32x4*)(asrc[i] + kt * FT_BK);
#pragma unroll
        for (int i = 0; i < WL; ++i) rw[i] = *(const halo_u32x4*)(wsrc[i] + (size_t)kt * 1024);
        if (!(kt & 1) && tid < BM) rs = sp[kt >> 1];
    };
    fetch(0);

    const int nk = K / FT_BK;
    for (int kt = 0; kt < nk; ++kt) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < AL; ++i) *(halo_u32x4*)adst[i] = ra[i];
#pragma unroll
        for (int i = 0; i < WL; ++i) {
            *(uint2*)wdst[i] = make_uint2(rw[i][0], rw[i][1]);          // k run t*16 + h*8
            *(uint2*)(wdst[i] + 16) = make_uint2(rw[i][2], rw[i][3]);   // fragment t + 1's
        }
        if (!(kt & 1) && tid < BM) sS[tid] = rs;
        __syncthreads();
        if (kt + 1 < nk) fetch(kt + 1);
#pragma unroll
        for (int s = 0; s < FT_BK / 16; ++s) {
            v4i_ af[TM], bf[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) af[i] = *(const v4i_*)(fa + i * 16 * IT_PITCH + s * 16);
#pragma unroll
            for (int j = 0; j < TN; ++j) bf[j] = *(const v4i_*)(fw + j * 16 * IT_PITCH + s * 16);
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                if (PART && i >= live) break;           // wave-uniform: padding fragments past M
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    tmp[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[i], true, bf[j],
                                                                           tmp[i][j], false);
            }
            if constexpr (!PART) {
                __builtin_amdgcn_sched_group_barrier(0x100, TM + TN, 0);    /* DS reads */
                __builtin_amdgcn_sched_group_barrier(0x008, TM * TN, 0);    /* WMMA */
            }
        }
        if (kt & 1) {
            const int g = kt >> 1;
            float wsc[TN];
#pragma unroll
            for (int j = 0; j < TN; ++j) wsc[j] = __uint_as_float((uint32_t)wsp[j][g * 16] << 16);
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                if (PART && i >= live) break;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float as = sS[(wm * TM + i) * 16 + accRow(lane, e)];
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        acc[i][j][e] = fmaf((float)tmp[i][j][e], as * wsc[j], acc[i][j][e]);
                        tmp[i][j][e] = 0;
                    }
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n = n0 + (wn * TN + j) * 16 + (lane & 15);
            const int mb = m0 + (wm * TM + i) * 16;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = mb + accRow(lane, e);
                if (m < M) C[(size_t)m * c_ld + n] = (__hip_bfloat16)acc[i][j][e];
            }
        }
}

/* PART: the two-body form for a partial last band of rows, as halo_gemm_fp8a8_tiled_part_kernel;
 * otherwise the plain kernel over the body's own LDS. */
template <int TM, int TN, bool PART>
__global__ __launch_bounds__(256) void halo_gemm_i8a8_tiled_kernel(
        const uint8_t* __restrict__ A, const float* __restrict__ Ascale,
        const uint8_t* __restrict__ W, const uint16_t* __restrict__ Ws,
        __hip_bfloat16* __restrict__ C, int M, int K, int N, int a_ld, int sa_ld, int c_ld) {
    constexpr int BM = 2 * TM * 16, BN = 4 * TN * 16;
    int m0, n0;
    fp8_tiled_origin(BM, BN, &m0, &n0);
    if constexpr (!PART) {
        i8_tiled_block<TM, TN, false, true>(A, Ascale, W, Ws, C, M, K, N, a_ld, sa_ld, c_ld, m0,
                                            n0, nullptr, nullptr, nullptr);
    } else {
        __shared__ __attribute__((aligned(16))) uint8_t sA[BM * IT_PITCH];
        __shared__ __attribute__((aligned(16))) uint8_t sW[BN * IT_PITCH];
        __shared__ float sS[BM];
        if (m0 + BM > M)
            i8_tiled_block<TM, TN, true, false>(A, Ascale, W, Ws, C, M, K, N, a_ld, sa_ld, c_ld,
                                                m0, n0, sA, sW, sS);
        else
            i8_tiled_block<TM, TN, false, false>(A, Ascale, W, Ws, C, M, K, N, a_ld, sa_ld, c_ld,
                                                 m0, n0, sA, sW, sS);
    }
}

int halo_gemm_i8a8_tiled(long a, long ascale, long wq, long ws, long c,
                         int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                         long stream) {
    (void)w_ld; (void)sw_ld;
    if (M <= HALO_GEMM_FP8_MAX_M || (N % 128) || K < 128 || (K % 128))
        return i8_gemm(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, N, 1, (hipStream_t)stream);
    /* Past one 64-row band, the LDS-tiled kernel: the same bytes (see its contract). */
    if (a_ld <= 0) a_ld = K;
    if (sa_ld <= 0) sa_ld = K / 128;
    const long long g128 = (long long)(N / 128) * ((M + 127) / 128);
    const bool big = g128 >= 32;
    const int bt = big ? 128 : 64;
    const dim3 grid((unsigned)(N / bt), (unsigned)((M + bt - 1) / bt));
    auto A = (const uint8_t*)a; auto As = (const float*)ascale;
    auto W = (const uint8_t*)wq; auto Ws = (const uint16_t*)ws;
    auto C = (__hip_bfloat16*)c;
    hipStream_t st = (hipStream_t)stream;
    const bool part = M % bt != 0;
#define IT_GO(TM, TN, P) hipLaunchKernelGGL((halo_gemm_i8a8_tiled_kernel<TM, TN, P>), grid, \
        dim3(256), 0, st, A, As, W, Ws, C, M, K, N, a_ld, sa_ld, N)
    if (big) { if (part) IT_GO(4, 2, true); else IT_GO(4, 2, false); }
    else     { if (part) IT_GO(2, 1, true); else IT_GO(2, 1, false); }
#undef IT_GO
    return hipGetLastError() == hipSuccess ? 0 : -5;
}

/* ---- the gated narrow row (gemm_fp8a8_gated_nt_m16) ----------------------------------------
 *
 * The narrow GEMM with gated_quant_fp8's epilogue folded in, byte-identical to the pair: the
 * accumulators round to bf16 first (as the GEMM's store does), silu(gate) * up is taken over
 * those rounded values and rounded again (as the quantiser's pass does), and the group's amax
 * closes over the block's eight column tiles in LDS. One K loop serves both halves -- the
 * activation fragment is shared, the weight fragment read twice -- so the accumulation order
 * per element is the ungated kernel's own. No split-K: the shim refuses the grids the split
 * law would split, as libr4d's entry point does. */
template <int MT>
__global__ __launch_bounds__(512) void halo_gemm_fp8a8_gated_kernel(
        const uint8_t* __restrict__ A, const float* __restrict__ Ascale,
        const uint8_t* __restrict__ W, const uint16_t* __restrict__ Ws,
        uint8_t* __restrict__ Q, float* __restrict__ Qs, uint16_t* __restrict__ Y,
        int M, int K, int N, int a_ld, int sa_ld, int sw_ld, int q_ld, int s_ld, int y_ld) {
    HALO_DYN_LDS(float, red);                     // [16 waves][32 lanes][8] products, a tile at a time
    const int lane = threadIdx.x & (WAVE - 1), wave = threadIdx.x / WAVE;
    const int slot = lane & 15;
    const int nout = N / 2;
    const int n = (int)blockIdx.x * 256 + wave * 16 + slot;
    const bool nlive = n < nout;
    const int ns = nlive ? n : 0;

    const int ngroups = K / 128;
    const uint16_t* Sg = Ws + (size_t)(ns / 128) * sw_ld;
    const uint16_t* Su = Ws + (size_t)((ns + nout) / 128) * sw_ld;
    const uint8_t* Ap[MT];
    int mrow[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        const int m = blockIdx.y * (MT * 16) + t * 16 + slot;
        mrow[t] = m < M ? m : M - 1;
        Ap[t] = A + (size_t)mrow[t] * a_ld;
    }

    v8f accg[MT], accu[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) accg[t] = accu[t] = (v8f)0.f;

    for (int g = 0; g < ngroups; ++g) {
        const int k0 = g * 128;
        const float wsg = nlive ? __uint_as_float((uint32_t)Sg[g] << 16) : 0.f;
        const float wsu = nlive ? __uint_as_float((uint32_t)Su[g] << 16) : 0.f;
        v8f tmpg[MT], tmpu[MT];
#pragma unroll
        for (int t = 0; t < MT; ++t) tmpg[t] = tmpu[t] = (v8f)0.f;
        /* As in the ungated kernel: a full unroll hoists every step's loads and spills from MT 3. */
        constexpr int KU = MT <= 2 ? 8 : 2;
#pragma unroll KU
        for (int u = 0; u < 8; ++u) {
            const int kabs = k0 + u * 16;
            const size_t wbase = ((size_t)(kabs >> 6)) * 1024 + slot * 32 + ((kabs >> 4) & 3) * 8;
            uint32_t wd[4];
            const uint8_t* chg = W + ((size_t)(ns >> 4) * (K >> 6)) * 1024 + wbase;
            *(uint2*)&wd[0] = *(const uint2*)chg;
            *(uint2*)&wd[2] = *(const uint2*)(chg + 16 * 32);
            const v16bf bfg = deq16_e4m3(wd);
            const uint8_t* chu = W + ((size_t)((ns + nout) >> 4) * (K >> 6)) * 1024 + wbase;
            *(uint2*)&wd[0] = *(const uint2*)chu;
            *(uint2*)&wd[2] = *(const uint2*)(chu + 16 * 32);
            const v16bf bfu = deq16_e4m3(wd);
            v16bf af[MT];
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                const int4v av = *(const int4v*)(Ap[t] + k0 + u * 16);
                af[t] = deq16_exact((const uint32_t*)&av);
            }
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                tmpg[t] = halo::mma(af[t], bfg, tmpg[t]);
                tmpu[t] = halo::mma(af[t], bfu, tmpu[t]);
            }
        }
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                int mr = blockIdx.y * (MT * 16) + t * 16 + 2 * e + (lane >> 4);
                mr = mr < M ? mr : M - 1;
                const float as = Ascale[(size_t)mr * sa_ld + g];
                accg[t][e] = fmaf(tmpg[t][e], as * wsg * HALO_FP8_FIX, accg[t][e]);
                accu[t][e] = fmaf(tmpu[t][e], as * wsu * HALO_FP8_FIX, accu[t][e]);
            }
    }

    /* No early return here: the epilogue's __syncthreads needs every wave, live or not. */
    const int gkb = s_ld;
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        /* The quantiser over the bf16-rounded product, per (row, 128 columns). */
        float v[8];
        uint16_t h[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float gv = (float)(__hip_bfloat16)accg[t][e];
            const float uv = (float)(__hip_bfloat16)accu[t][e];
            const uint16_t hb = f32_to_bf16(gv / (1.0f + expf(-gv)) * uv);
            h[e] = hb;
            v[e] = bf16_to_f32(hb);
        }
        float* mine = red + ((size_t)wave * WAVE + lane) * 8;
#pragma unroll
        for (int e = 0; e < 8; ++e) mine[e] = v[e];
        __syncthreads();
        const int gw = (wave / 8) * 8;                 // the eight column tiles of this group
        const int half = lane >> 4;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float amax = 0.f;
#pragma unroll
            for (int w2 = 0; w2 < 8; ++w2)
#pragma unroll
                for (int s2 = 0; s2 < 16; ++s2)
                    amax = fmaxf(amax, fabsf(red[((size_t)(gw + w2) * WAVE + s2 + 16 * half) * 8 + e]));
            const float sc = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
            const int m = blockIdx.y * (MT * 16) + t * 16 + 2 * e + half;
            if (nlive && m < M) {
                if (slot == 0)
                    Qs[(size_t)m * gkb + (int)blockIdx.x * 2 + wave / 8] = sc;
                Q[(size_t)m * q_ld + n] = (uint8_t)halo_fp8::f32_to_e4m3(
                    fminf(fmaxf(v[e] * (1.0f / sc), -448.0f), 448.0f));
                if (Y) Y[(size_t)m * y_ld + n] = h[e];
            }
        }
        __syncthreads();
    }
}

// ============================================================== the plugin ABI entry points

extern "C" int halo_rad_gemm_fp8a8_m16(const RadArgs* a, RadStream s) {
    Fp8Ops g;
    int rc = fp8_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.M > HALO_GEMM_FP8_MAX_M) return HALO_NO(RAD_E_SHAPE);
    if ((g.K % 128) || (g.N % 16)) return HALO_NO(RAD_E_SHAPE);
    rc = fp8_nt_m16(halo_addr(g.a), halo_addr(g.ascale), halo_addr(g.b), halo_addr(g.bscale),
                    halo_addr(g.y), (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld,
                    (int)g.w_ld, (int)g.sw_ld, (long)(uintptr_t)a->scratch, a->scratch_bytes,
                    (long)(uintptr_t)s, law_sk(a, g.M, g.K, g.N));
    return rc ? HALO_NO(RAD_E_DEVICE) : RAD_OK;
}

extern "C" int64_t halo_rad_gemm_fp8a8_m16_scratch(const RadArgs* a) {
    /* Declare-time sizing runs with no tensors (core/build/rad_builder.cpp), so the
     * geometry comes from the parameters, as libr4d's hook does. Split-K is in-block
     * (the helper's 0); the bytes asked for are the activation plane's, M*K bf16. */
    const long long M = halo_geti_or(a, "M", 0), N = halo_geti_or(a, "N", 0),
                    K = halo_geti_or(a, "K", 0);
    if (M < 1 || N < 1 || K < 1) return 0;
    if (M > HALO_GEMM_FP8_MAX_M) return 0;
    return halo_gemm_fp8a8_nt_m16_scratch((int)M, (int)K, (int)N) + M * K * 2;
}

/* The tiled row's scratch: the activation plane, while it fits 256 MiB. Past that (a long
 * prefill band at a wide K) the prefill kernel decodes while staging and the program does not
 * reserve the plane. */
extern "C" int64_t halo_rad_gemm_fp8a8_tiled_scratch(const RadArgs* a) {
    const long long M = halo_geti_or(a, "M", 0), K = halo_geti_or(a, "K", 0);
    if (M < 1 || K < 1) return 0;
    const long long b = M * K * 2;   /* the activation plane */
    return b <= (256ll << 20) ? b : 0;
}

extern "C" int halo_rad_gemm_fp8a8_tiled(const RadArgs* a, RadStream s) {
    Fp8Ops g;
    int rc = fp8_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if ((g.K % 128) || (g.N % 16)) return HALO_NO(RAD_E_SHAPE);
    rc = fp8_tiled(halo_addr(g.a), halo_addr(g.ascale), halo_addr(g.b), halo_addr(g.bscale),
                   halo_addr(g.y), (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld,
                   (int)g.w_ld, (int)g.sw_ld, (long)(uintptr_t)s, (long)(uintptr_t)a->scratch,
                   a->scratch_bytes);
    return rc ? HALO_NO(RAD_E_DEVICE) : RAD_OK;
}

extern "C" int halo_rad_gemm_fp8a8_bias_m16(const RadArgs* a, RadStream s) {
    /* gemm_nt_q_bias's operands: fp8_ops's five with the bias at 4 and C at 5 (oGemmNtQBias),
     * and C at its own pitch. Read directly, as libr4d's shim does. */
    const RadTensor *ta = halo_opt(a, 0), *ts = halo_opt(a, 1), *tw = halo_opt(a, 2),
                    *tws = halo_opt(a, 3), *tb = halo_opt(a, 4), *ty = halo_opt(a, 5);
    if (!ta || !ts || !tw || !tws || !tb || !ty) return HALO_NO(RAD_E_INVAL);
    if (ty->rank < 2) return HALO_NO(RAD_E_SHAPE);
    const long long M = ty->shape[0];
    if (M < 1 || M > HALO_GEMM_FP8_MAX_M || halo_numel(ty) % M) return HALO_NO(RAD_E_SHAPE);
    const long long N = halo_numel(ty) / M, K = tw->shape[1];
    if (tw->rank != 2 || tw->shape[0] != N) return HALO_NO(RAD_E_SHAPE);
    if ((K % 128) || (N % 16)) return HALO_NO(RAD_E_SHAPE);
    const long long yp = halo_row_pitch(ty);
    if (yp < N) return HALO_NO(RAD_E_STRIDE);
    const int rc = halo_gemm_fp8a8_bias_nt_m16(
        halo_addr(ta), halo_addr(ts), halo_addr(tw), halo_addr(tws), halo_addr(tb), halo_addr(ty),
        (int)M, (int)K, (int)N, (int)halo_row_pitch(ta), (int)halo_row_pitch(ts),
        (int)halo_row_pitch(tw), (int)halo_row_pitch(tws), (int)yp, (long)(uintptr_t)s);
    return rc ? HALO_NO(RAD_E_DEVICE) : RAD_OK;
}

extern "C" int halo_rad_gemm_i8a8_m16(const RadArgs* a, RadStream s) {
    Fp8Ops g;
    int rc = fp8_ops(a, &g, 1, FP8_OPS_ROW_SCALE);
    if (rc != RAD_OK) return rc;
    if (g.M > HALO_GEMM_FP8_MAX_M) return HALO_NO(RAD_E_SHAPE);
    if ((g.K % 128) || (g.N % 16)) return HALO_NO(RAD_E_SHAPE);
    rc = i8_nt_m16(halo_addr(g.a), halo_addr(g.ascale), halo_addr(g.b), halo_addr(g.bscale),
                   halo_addr(g.y), (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld,
                   (long)(uintptr_t)s, law_sk(a, g.M, g.K, g.N));
    return rc ? HALO_NO(RAD_E_DEVICE) : RAD_OK;
}

extern "C" int64_t halo_rad_gemm_i8a8_m16_scratch(const RadArgs* a) {
    (void)a;
    return 0;   /* split-K is in-block, as in the fp8 kernel; declare-time sizing has
                 * no tensors to read, and libr4d's hook just forwards to its m16 one */
}

extern "C" int halo_rad_gemm_i8a8_tiled(const RadArgs* a, RadStream s) {
    Fp8Ops g;
    int rc = fp8_ops(a, &g, 1, FP8_OPS_ROW_SCALE);
    if (rc != RAD_OK) return rc;
    if ((g.K % 128) || (g.N % 16)) return HALO_NO(RAD_E_SHAPE);
    rc = halo_gemm_i8a8_tiled(halo_addr(g.a), halo_addr(g.ascale), halo_addr(g.b),
                              halo_addr(g.bscale), halo_addr(g.y), (int)g.M, (int)g.K,
                              (int)g.N, (int)g.a_ld, (int)g.sa_ld, (int)g.w_ld, (int)g.sw_ld,
                              (long)(uintptr_t)s);
    return rc ? HALO_NO(RAD_E_DEVICE) : RAD_OK;
}

extern "C" int halo_rad_gemm_fp8a8_gated_m16(const RadArgs* a, RadStream s) {
    /* gemm_nt_q_gated's operands: the narrow GEMM's five, then the quantiser's codes, scales
     * and optional bf16 product (oGemmNtQGated). Read directly, as the bias shim does. */
    const RadTensor *ta = halo_opt(a, 0), *ts = halo_opt(a, 1), *tw = halo_opt(a, 2),
                    *tws = halo_opt(a, 3), *tq = halo_opt(a, 4), *tqs = halo_opt(a, 5),
                    *ty = halo_opt(a, 6);
    if (!ta || !ts || !tw || !tws || !tq || !tqs) return HALO_NO(RAD_E_INVAL);
    if (ta->dtype != RAD_F8E4M3 || tw->dtype != RAD_F8E4M3 || tq->dtype != RAD_F8E4M3 ||
        ts->dtype != RAD_F32 || tws->dtype != RAD_BF16 || tqs->dtype != RAD_F32 ||
        (ty && ty->dtype != RAD_BF16))
        return HALO_NO(RAD_E_DTYPE);
    const char* dt = halo_gets(a, "dtype");
    if (!dt || std::strcmp(dt, "fp8a8")) return HALO_NO(RAD_E_DTYPE);
    const char* act = halo_gets(a, "act");
    if (act && *act && std::strcmp(act, "silu")) return HALO_NO(RAD_E_UNSUPPORTED);
    if (halo_geti_or(a, "group", 128) != 128) return HALO_NO(RAD_E_SHAPE);

    const long long N = halo_geti_or(a, "N", halo_dim(tw, 0));
    const long long K = halo_geti_or(a, "K", halo_dim(tw, 1));
    if (K < 1) return HALO_NO(RAD_E_SHAPE);
    /* The rows are the activation's; `M` is the band a step may undershoot. */
    const long long M = halo_numel(ta) / K;
    if (M < 1 || M > HALO_GEMM_FP8_MAX_M || M > halo_geti_or(a, "M", M)) return HALO_NO(RAD_E_SHAPE);
    if ((K % 256) || (N % 256)) return HALO_NO(RAD_E_SHAPE);
    /* The fold has no partial plane: a grid the split law would split is refused, as at
     * libr4d's entry point. */
    if (fp8_sk((int)M, (int)K, (int)N) != 1) return HALO_NO(RAD_E_SHAPE);
    const int nout = (int)(N / 2);
    if (halo_numel(tq) < M * nout || halo_numel(tqs) < M * (nout / 128) ||
        (ty && halo_numel(ty) < M * nout))
        return HALO_NO(RAD_E_SHAPE);

    /* Past the crossover, the pair this row is byte-identical to, run as such: the tiled GEMM
     * (unsplit, as fp8_sk said above, so its bytes are the narrow kernel's) into the scratch,
     * then gated_quant_fp8 over it. The fold launches nout/256 blocks whatever M is (68 at
     * gate_up's 34816 rows) and spills at MT 3-4; the pair runs at the tile's rate. */
    const long long gu_bytes = M * N * 2;
    if (M > fp8_tiled_min_m() && a->scratch && a->scratch_bytes >= gu_bytes && g_fp8_pin_nt == 0) {
        char* gu = (char*)a->scratch;
        const long adq = a->scratch_bytes >= gu_bytes + M * K * 2 ? (long)(uintptr_t)(gu + gu_bytes) : 0;
        if (fp8_tiled_body(halo_addr(ta), halo_addr(ts), halo_addr(tw), halo_addr(tws),
                           (long)(uintptr_t)gu, (int)M, (int)K, (int)N, (int)halo_row_pitch(ta),
                           (int)halo_row_pitch(ts), (int)halo_row_pitch(tws), (int)N,
                           (hipStream_t)s, adq))
            return HALO_NO(RAD_E_DEVICE);
        const int e = halo_gated_quant_fp8((long)(uintptr_t)gu, ty ? halo_addr(ty) : 0,
                                           halo_addr(tq), halo_addr(tqs), (int)M, nout, 0, (int)N,
                                           nout, nout, nout / 128, (long)(uintptr_t)s);
        return e ? HALO_NO(RAD_E_DEVICE) : RAD_OK;
    }

    int MT = M <= 16 ? 1 : M <= 32 ? 2 : M <= 48 ? 3 : 4;
    if (g_fp8_pin_nt && M <= g_fp8_pin_nt * 16) MT = g_fp8_pin_nt;
    const dim3 grid((nout + 255) / 256, (unsigned)((M + MT * 16 - 1) / (MT * 16)));
    const size_t smem = (size_t)16 * WAVE * 8 * sizeof(float);
    auto launch = [&](auto kern) {
        hipLaunchKernelGGL((kern), grid, 512, smem, (hipStream_t)s, 
            (const uint8_t*)halo_p(ta), (const float*)halo_p(ts), (const uint8_t*)halo_p(tw),
            (const uint16_t*)halo_p(tws), (uint8_t*)halo_p(tq), (float*)halo_p(tqs),
            ty ? (uint16_t*)halo_p(ty) : nullptr, (int)M, (int)K, (int)N,
            (int)halo_row_pitch(ta), (int)halo_row_pitch(ts), (int)halo_row_pitch(tws),
            nout, nout / 128, nout);
    };
    switch (MT) {
        case 1: launch(halo_gemm_fp8a8_gated_kernel<1>); break;
        case 2: launch(halo_gemm_fp8a8_gated_kernel<2>); break;
        case 3: launch(halo_gemm_fp8a8_gated_kernel<3>); break;
        default: launch(halo_gemm_fp8a8_gated_kernel<4>); break;
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* The gated row's scratch: the gate_up product (bf16 [M, N]) and the activation plane for the
 * tiled pair above, past the crossover. Declare-time sizing has only the parameters. */
extern "C" int64_t halo_rad_gemm_fp8a8_gated_m16_scratch(const RadArgs* a) {
    const long long M = halo_geti_or(a, "M", 0), N = halo_geti_or(a, "N", 0),
                    K = halo_geti_or(a, "K", 0);
    if (M <= fp8_tiled_min_m() || M > HALO_GEMM_FP8_MAX_M || N < 1 || K < 1) return 0;
    return M * N * 2 + M * K * 2;
}

/* The raw-pointer gated entry libr4d exports beside the row (halo.h): same kernel, the
 * pitches spelled out. `wq` is this library's fragment-ordered gate_up plane, as the
 * row's, so `w_ld` (a row-major pitch in libr4d's spelling) has no meaning here and is
 * accepted and ignored. Negative codes, as libr4d's: -1 a null pointer, -2 a
 * non-positive extent, -3 M past the band, -4 K not a multiple of 256, -5 `nout` not a
 * multiple of 128, -8 an activation other than silu, -9 a grid the split rule would
 * widen, -6 the launch failed. */
extern "C" int halo_gemm_fp8a8_gated_nt_m16_max_m(void) { return 64; }

extern "C" int halo_gemm_fp8a8_gated_nt_m16(long a, long ascale, long wq, long ws,
                                            long q, long s, long y, int act,
                                            int M, int K, int nout,
                                            int a_ld, int sa_ld, int w_ld, int sw_ld,
                                            int q_ld, int s_ld, int y_ld, long stream) {
    (void)w_ld;
    if (!a || !ascale || !wq || !ws || !q || !s) return -1;
    if (M <= 0 || K <= 0 || nout <= 0) return -2;
    if (M > halo_gemm_fp8a8_gated_nt_m16_max_m()) return -3;
    if (K % 256) return -4;
    if (nout % 128) return -5;
    if (act != 0) return -8;
    const int N = 2 * nout;
    if (fp8_sk(M, K, N) != 1) return -9;
    if (!a_ld) a_ld = K;
    if (!sa_ld) sa_ld = K / 128;
    if (!sw_ld) sw_ld = K / 128;
    if (!q_ld) q_ld = nout;
    if (!s_ld) s_ld = nout / 128;
    if (!y_ld) y_ld = nout;
    int MT = M <= 16 ? 1 : M <= 32 ? 2 : M <= 48 ? 3 : 4;
    if (g_fp8_pin_nt && M <= g_fp8_pin_nt * 16) MT = g_fp8_pin_nt;
    const dim3 grid((nout + 255) / 256, (unsigned)((M + MT * 16 - 1) / (MT * 16)));
    const size_t smem = (size_t)16 * WAVE * 8 * sizeof(float);
    hipStream_t st = (hipStream_t)stream;
    auto launch = [&](auto kern) {
        hipLaunchKernelGGL((kern), grid, 512, smem, st, 
            (const uint8_t*)(uintptr_t)a, (const float*)(uintptr_t)ascale,
            (const uint8_t*)(uintptr_t)wq, (const uint16_t*)(uintptr_t)ws,
            (uint8_t*)(uintptr_t)q, (float*)(uintptr_t)s,
            y ? (uint16_t*)(uintptr_t)y : nullptr, M, K, N, a_ld, sa_ld, sw_ld,
            q_ld, s_ld, y_ld);
    };
    switch (MT) {
        case 1: launch(halo_gemm_fp8a8_gated_kernel<1>); break;
        case 2: launch(halo_gemm_fp8a8_gated_kernel<2>); break;
        case 3: launch(halo_gemm_fp8a8_gated_kernel<3>); break;
        default: launch(halo_gemm_fp8a8_gated_kernel<4>); break;
    }
    return hipGetLastError() == hipSuccess ? 0 : -6;
}
