/* r3d_gemm_nt_q_w2a8.hip -- gemm_nt_q at dtype=w2a8: the vocabulary projection, decode-sized.
 *
 *   y[M,N] = a[M,K] @ b[N,K]^T over a 2-bit unsigned weight code with an f16 scale and an unsigned
 *   integer zero per (row, group of K), an int8 activation with one f32 scale per row, and the int8
 *   code sums `a_sum` that remove the zero. f32 accumulate, bf16 store.
 *
 * MiniCPM5-2B's `output.weight` is N=130560, K=2048, group=128, dtype=w2a8, stored as the
 * container's joined `rad.planes` operand: a scale plane and a zero plane behind one operand
 * (b_scale). libref's `gemm_q_shape` declines w2a8, so kbench has no case for it; the row is
 * verified by a driver that builds the joined operand with `rad_planes_relayout` and compares every
 * element against `ref_gemm_nt_q`. libref's host row (`gemm_zp`) costs 135 GFLOP of CPU work per
 * token, and the engine refuses a placement no loaded library can read, so this row needs
 * `r3d_layout_gemm_nt_q_w2a8` below: it describes the container's bytes (RadLayoutFn in
 * rad_abi.h) and does no repacking.
 *
 * ---- the law, and the order inside it ---------------------------------------------------------
 *
 * gemm_zp's ZP_I8_ROW arm is the spec, and this kernel runs its order:
 *
 *   q[k]   = rad_load_bits(row n of b, 2, k)                          bits [2k,2k+2) of byte k/4
 *   is(g)  = SUM_{k in group g} (int)a[m][i8_step_pos(k)] * (int)q[k]  integers: exact
 *   sub(g) = (float)is(g) - (float)z[n][g] * a_sum[g*mpad + m]         mpad = ceil(M/16)*16 = 16
 *   acc    = acc + (sub(g) * sa[m]) * sb[n][g]                         g ascending
 *   y[m,n] = bf16(acc)
 *
 * `a_sum` is per row and per group, group-major. `sa` is one f32 per row (ZP_I8_ROW: quant_act_i8
 * scales a whole row, not a group). The parentheses in `(sub * sa) * sb` are the reference's order.
 *
 * The activation is in quant_act_i8's step order: column c sits at byte
 * i8_step_pos(c) = (c & ~12) | ((c & 4) << 1) | ((c & 8) >> 1), bits 2 and 3 of c swapped, a
 * self-inverse map. On the dword index that is a swap of bits 0 and 1, so a dword's four columns
 * stay contiguous: staging reads a's dwords in permuted order, one 4-byte load per four codes.
 *
 * Exactness. In exact f32 arithmetic `is` (integer partials under 2^24) would be exact and the
 * float steps follow the reference's order. The dot instruction is not that exact (below), so
 * 0.1-0.2% of outputs land one bf16 ulp off the reference: rel_l2 5.0e-3 at the band, no
 * non-finite values. The row is not bit-exact, and its row-table text says so.
 *
 * ---- U2 to f16 through a bias, and the bias removed with a_sum ---------------------------------
 *
 * This ISA has no int8 dot: `__builtin_amdgcn_s32sd4`, `__builtin_amdgcn_s32ud4`, `__dp4a` and the
 * matrix ops `__builtin_amdgcn_madocx_i32_i8` / `_u32_u8` are undeclared in AMD clang 24.0.0git
 * for gfx1102. The codes therefore widen to f16 and the dot runs on dot2_f16x2. v_fdot2_f32_f16
 * carries about 22 bits of mantissa on integer operands, not 24, so integer partials pushed
 * through it are inexact once they reach about 2^20. The bias choice is explained at u2x4_biased.
 *
 * The widening uses the fact that the f16 bits of (128 + x) are 0x5800 + 8x for 0 <= x < 8, so a
 * 2-bit code becomes an f16 with a mask, two shifts and an add. The dot then returns
 * is' = SUM a[k]*(128 + q[k]) = is + 128*SUM a, and SUM a over the group is a_sum[g*mpad + m],
 * which the op already carries for the zero. 128 * a_sum is exact (a power-of-two scale of an
 * integer-valued f32), so is = is' - 128*a_sum is exact and the reference's
 * sub = (float)is - (float)z*a_sum follows unchanged, at one f32 subtraction per (m, group).
 *
 * The row is not memory-bound: its 66.8 MB of codes have a streaming floor near 0.41 ms, and the
 * cost is in the loop (the group-term plane and the two barriers per chunk).
 *
 * ---- the lane map ------------------------------------------------------------------------------
 *
 * One warp per weight row n; a block of 8 waves covers 8 rows. MT is the power of two covering M,
 * and KL = 32/MT is the number of k-lanes per activation row. A chunk is CHUNK = 2048/MT codes, so
 * the staged tile is always MT*CHUNK/2 = 1024 dwords = 4 KiB, which keeps the block resident at
 * M=16. A chunk holds CHUNK/128 = 16/MT groups. Each lane takes 64 codes per chunk: one 16-byte
 * load of codes (U2 is four codes per byte), 32 dwords of activation from LDS and 32 dot2 in 16
 * dot4_f16x4. Lane l owns row m = l % MT, k-lane kl = l / MT and group ch*(16/MT) + kl/2, whose two
 * halves are the pair (l, l^MT), folded with one shfl_xor.
 *
 * Group indices ascend with the chunk index, and within a chunk the row's owner lane adds the
 * chunk's group terms in ascending g, so the accumulation is the reference's ascending g loop.
 * Reassociating the g sum would change the result; no lane does.
 *
 * LDS is accessed one quad (two dwords, 8 bytes) at a time, swizzled by `p_ix` below; unswizzled,
 * the wavefront's 32 quads for one u would all sit in the same bank pair.
 *
 * ---- refusals (codes; RAD_R3D_TRACE=1 prints the line) -----------------------------------------
 *
 * The shim mirrors gemm_ntq_impl's and gemm_zp's checks and declines two things the reference
 * accepts, each named at its site: M > 16 (the tile is 16 rows and mpad is the reference's 16-row
 * pad; the prefill vocab projection is `logits_gemm`, a tiled bf16 row), and a zero plane stored
 * sub-byte (U2/U4/U5), because the kernel reads a zero one byte at a time; that would need the
 * bit-extract path (`rad_load_bits(zb, bits, g)`). w2a8's practical form, and this container's,
 * is U8.
 */

#include <hip/hip_runtime.h>

#include "r3d_common.h"
#include "r3d_codecs.h"
#include "r3d_launch.h"
#include "r3d_args.h"

#include "rad_planes.h"

namespace {

constexpr int kBlockThreads = 256;
constexpr int kWaves        = kBlockThreads / kWave;   /* 8 weight rows per block */
constexpr int kMTMax        = 16;                      /* the band's M */
constexpr int kTileQuads    = 512;        /* MT * CHUNK/4, constant over MT: 512 quads, 4 KiB */
constexpr int kGroup        = 128;                     /* the block size w2a8 is stored at */
constexpr int kMpad         = 16;                      /* the reference's (M+15)/16*16, M <= 16 */

/* Column c sits at byte step_pos(c): bits 2 and 3 of c swapped, a self-inverse map. On the dword
 * index it is a swap of bits 0 and 1 (the middle two dwords of a 16-byte step exchanged), so
 * dword d starts at byte dword_pos(d). Not 4 * step_pos(d), which swaps the wrong pair of bits. */
[[maybe_unused]] __device__ __forceinline__ int64_t step_pos(int64_t c)
{
    return (c & ~(int64_t)12) | ((c & 4) << 1) | ((c & 8) >> 1);
}
__device__ __forceinline__ int64_t dword_pos(int64_t d)
{
    return 4 * ((d & ~(int64_t)3) | ((d & 1) << 1) | ((d & 2) >> 1));
}

/* The LDS swizzle, at the tile's access granularity: a quad of four codes (two dwords) moved as one
 * 8-byte word. Quads are numbered q = r*(CHUNK/4) + c; p_ix XORs q's low four bits with bits 4..7,
 * a bijection on 16-quad groups and so on any tile whose quad count is a multiple of 16. A
 * wavefront reads q = m*(CHUNK/4) + kl*16 + u for fixed u; CHUNK/4 is a multiple of 16 at every MT,
 * so q ^ (q >> 4) mod 16 is u ^ (4m + kl) mod 16 over the 32 lanes: 16 distinct values, each an
 * 8-byte access spanning two banks, 32 banks with no conflict. The swizzle works on quad indices
 * so the two dwords of a quad stay adjacent; XORing bits 5..9 of the dword index would not. */
__device__ __forceinline__ int p_ix(int q) { return q ^ ((q >> 4) & 15); }

/* Four int8 codes -> two f16x2 dwords. Exact: an integer of |a| <= 127 needs no rounding. */
__device__ __forceinline__ void i8x4_to_f16x2(uint32_t w, uint32_t& lo, uint32_t& hi)
{
    lo = f16x2_pack((float)(int32_t)(int8_t)(w & 0xffu), (float)(int32_t)(int8_t)((w >> 8) & 0xffu));
    hi = f16x2_pack((float)(int32_t)(int8_t)((w >> 16) & 0xffu),
                    (float)(int32_t)(int8_t)((w >> 24) & 0xffu));
}

/* Four U2 codes (one byte) -> two f16x2 dwords, biased by 128: each half is the f16 bits of
 * (128 + code). An f16 at exponent 2^7 steps its mantissa by 1/8, so 128 + code is 0x5800 + 8*code.
 * The dot returns is + 128 * (the activation's group sum), which the epilogue removes with a_sum.
 *
 * The bias sets the magnitude of the dot's partial sums: a * (B + q) is up to 128 * B, and a lane's
 * 64 of them reach 64 * 128 * (B + 3), about 2^23 at B = 1024 and 2^20 at B = 128. The dot unit
 * keeps about 22 bits of mantissa on integers, so its error grows with the magnitude: over 262144
 * sampled dots (probe below), the worst error is 1/8 at B = 1024 and 1/64 at B = 128. 128 is the
 * largest power of two that keeps a lane's partial sum under 2^21. */
__device__ __forceinline__ void u2x4_biased(uint32_t w, uint32_t& lo, uint32_t& hi)
{
    lo = 0x58005800u + ((w & 3u) << 3) + (((w >> 2) & 3u) << 19);
    hi = 0x58005800u + (((w >> 4) & 3u) << 3) + (((w >> 6) & 3u) << 19);
}

template <int MT>
__global__ void __launch_bounds__(kBlockThreads) R3D_WAVES(2)
gemm_w2a8_kernel(const int8_t* __restrict__ a, const float* __restrict__ sa, int sa_stride,
                 const uint8_t* __restrict__ b, const uint16_t* __restrict__ sb, int sb_rb,
                 const uint8_t* __restrict__ zq, int z_rb, const float* __restrict__ asum,
                 int asum_stride, uint16_t* __restrict__ y, int M, int N, int K, int chunk,
                 int a_pitch, int y_pitch)
{
    constexpr int KL = kWave / MT;                    /* k-lanes per activation row */
    constexpr int CG = (2048 / MT) / kGroup;          /* groups per chunk */
    constexpr int CL = (2048 / MT) / KL;              /* codes per lane per chunk: 64 */

    /* The tile: one quad of four codes (two f16x2 dwords) per slot, typed as the quad the staging
     * writes and the dot reads. It is a `uint2[]`, not a `uint32_t[]` written through a vector
     * pointer: an f32 tile written through `*(uint4*)&sA[]` at -O2 read back content that was never
     * written. */
    __shared__ __align__(16) uint2 sA[kTileQuads];       /* MT rows x CHUNK/2 f16, swizzled */
    __shared__ float sT[kWaves * 16];        /* each warp's chunk terms: warp*16 + m*CG + j */
    __shared__ float sSa[MT];

    const int lane = threadIdx.x & (kWave - 1);
    const int warp = (int)(threadIdx.x / kWave);
    const int n = blockIdx.x * kWaves + warp;
    /* A wavefront past the last row skips the dot but still stages its share of the block-wide
     * tile. Only a block whose eight rows are all past N returns early: that test is block-uniform
     * and cannot strand a barrier, whereas a per-row return would leave tile rows unstaged. The
     * store is guarded by `live`. */
    const bool live = n < N;
    if ((int)blockIdx.x * kWaves >= N) return;

    const int m = lane & (MT - 1);
    const int kl = lane / MT;
    const int row_bytes = K >> 2;                     /* rad_enc_row_bytes(RAD_U2, K), K % 16 == 0 */
    const int cq = chunk >> 2;                        /* the chunk's code bytes */
    const int nchunk = K / chunk;

    if (lane < MT) sSa[lane] = lane < M ? sa[(long long)lane * sa_stride] : 0.0f;
    __syncthreads();

    float acc = 0.0f;
    for (int ch = 0; ch < nchunk; ++ch) {
        /* Stage the chunk's activation rows, widened and step-unpermuted. Cells are four codes:
         * MT * CHUNK/4 = 512 at every MT, two per thread. */
        for (int i = threadIdx.x; i < MT * (chunk >> 2); i += kBlockThreads) {
            const int r = i / (chunk >> 2), c = i % (chunk >> 2);
            uint32_t w = 0;
            if (r < M)
                w = *(const uint32_t*)(a + (long long)r * a_pitch +
                                       dword_pos((int64_t)ch * (chunk >> 2) + c));
            const int q = r * (chunk >> 2) + c;
            uint32_t lo, hi;
            i8x4_to_f16x2(w, lo, hi);
            sA[p_ix(q)] = make_uint2(lo, hi);
        }
        __syncthreads();

        uint4 qw = make_uint4(0, 0, 0, 0);
        if (live)
            qw = *(const uint4*)(b + (long long)n * row_bytes + (long long)ch * cq + kl * (CL >> 2));
        const uint32_t w4[4] = { qw.x, qw.y, qw.z, qw.w };

        if (live) {
            const int qb = m * (chunk >> 2) + kl * (CL >> 2);
            float p = 0.0f;                            /* the lane's half-group, an exact integer */
#pragma unroll
            for (int v = 0; v < 4; ++v) {              /* four bytes, four quads, sixteen codes */
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    uint32_t ql, qh;
                    u2x4_biased((w4[v] >> (8 * b)) & 0xffu, ql, qh);
                    const uint2 av = sA[p_ix(qb + 4 * v + b)];
                    p = dot4_f16x4(av.x, av.y, ql, qh, p);
                }
            }
            /* Fold the pair: lanes l and l^MT are the same row, the two halves of one group. */
            const float isg = p + __shfl_xor(p, MT);

            const int g = ch * CG + (kl >> 1);
            const float as = asum[(long long)g * asum_stride + m];
            const float is = isg - 128.0f * as;                    /* exact: the de-bias */
            const float iz = (float)zq[(long long)n * z_rb + g];
            const float sc = f16_to_f32(sb[(long long)n * (sb_rb >> 1) + g]);
            const float term = ((is - iz * as) * sSa[m]) * sc;     /* gemm_zp's parentheses */
            /* One slot per (row, group-in-chunk); MT*CG is 16 at every MT (64 bytes). Both
             * half-lanes of a group write the same folded value. */
            sT[warp * 16 + m * CG + (kl >> 1)] = term;
        }
        __syncthreads();
        /* Groups ch*CG + j, j ascending, into the row's owner lane: f32 adds are not associative
         * and the reference walks g ascending. This gather is the only cross-k-lane step; a
         * barrier and a few 64-byte LDS reads are cheaper than a shuffle round-trip. */
        if (live && kl == 0) {

            /* Group ch*CG + j comes from k-lanes 2j and 2j+1, already folded. */
#pragma unroll
            for (int j = 0; j < CG; ++j)
                if (m < M) acc += sT[warp * 16 + m * CG + j];
        }
        __syncthreads();
    }

    if (live && kl == 0 && m < M) y[(long long)m * y_pitch + n] = f32_to_bf16(acc);
}

int mt_for(int M) { return M <= 1 ? 1 : M <= 2 ? 2 : M <= 4 ? 4 : M <= 8 ? 8 : kMTMax; }

}   // namespace

/* ================================================================== the launcher
 *
 * Codes: 0, -1 a required pointer is null, -2 an unsupported shape, -3 an unreadable pitch, -4 HIP.
 * `nt` is accepted for signature uniformity and ignored: the work is one warp per weight row.
 *
 * `b_scale` is the joined operand's base. The launcher finds the two planes through its header
 * (rad_planes.h's rad_planes_find) and passes each plane's base and row pitch in bytes, so the
 * kernel reads the container's bytes in place. */

/* A probe of the dot's exactness: one 64-code sample per lane, the f32 the dot produces beside the
 * exact integer. The test driver runs it as `dot [n]`; it is what sizes the bias. Two extern "C"
 * symbols, no row, nothing at serve time. */
extern "C" __global__ void __launch_bounds__(256) probe_dot_kernel(const int8_t* a,
                                                                   const uint8_t* codes,
                                                                   float* out, long long* exact)
{
    const int g = blockIdx.x * 32 + (int)(threadIdx.x / kWave) * 4 + (int)(threadIdx.x & 3);
    const uint32_t* w4 = (const uint32_t*)(codes + (long long)g * 16);
    float p = 0.0f;
    long long ex = 0;
#pragma unroll
    for (int v = 0; v < 4; ++v) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t aw = *(const uint32_t*)(a + (long long)g * 64 + 4 * (4 * v + b));
            uint32_t ql, qh;
            u2x4_biased((w4[v] >> (8 * b)) & 0xffu, ql, qh);
            uint32_t lo, hi;
            i8x4_to_f16x2(aw, lo, hi);
            p = dot4_f16x4(lo, hi, ql, qh, p);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                ex += (long long)(int)(int8_t)((aw >> (8 * j)) & 0xffu) *
                      (long long)(128 + ((w4[v] >> (8 * b + 2 * j)) & 3u));
        }
    }
    out[g] = p;
    exact[g] = ex;
}

extern "C" int r3d_w2a8_probe_launch(const int8_t* a, const uint8_t* codes, float* out,
                                     long long* exact, int n)
{
    probe_dot_kernel<<<(n + 31) / 32, 256, 0, 0>>>(a, codes, out, exact);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

extern "C" int r3d_gemm_nt_q_w2a8(long a, long a_scale, long b, long b_scale, long a_sum, long y,
                                  int M, int N, int K, int group, int a_ld, int y_ld, int nt,
                                  long stream)
{
    (void)nt;
    if (!a || !a_scale || !b || !b_scale || !a_sum || !y) return -1;
    if (M < 1 || N < 1 || K < 1) return -2;
    if (M > kMTMax) return -2;                        /* as the row's C_LE("M", 16) */
    if (group != kGroup) return -2;                   /* chunk = 128*16/MT */
    if (K % kGroup) return -2;
    if (K % 16) return -2;                            /* the step order is a 16-byte step */
    const int mt = mt_for(M);
    const int chunk = 2048 / mt;
    if (K % chunk) return -2;                         /* ragged chunk: the tile would read past K */
    const int G = K / group;
    if (G < 1) return -2;
    if (!a_ld) a_ld = K;
    if (!y_ld) y_ld = N;
    if (a_ld % 4) return -3;                          /* four codes per dword, staged whole */
    if (((uintptr_t)b & 15u) || ((uintptr_t)a & 3u)) return -3;

    const RadPlanesHeader* h = rad_planes_header((const void*)(uintptr_t)b_scale);
    if (!h) return -2;                                /* not the joined form: rad.planes is expected */
    const RadPlanesEntry* es = nullptr;
    const RadPlanesEntry* ez = nullptr;
    const void* sp = rad_planes_find((const void*)(uintptr_t)b_scale, "scale", &es);
    const void* zp = rad_planes_find((const void*)(uintptr_t)b_scale, "zero", &ez);
    if (!sp || !zp || !es || !ez) return -2;
    if (es->dtype != RAD_F16) return -2;              /* this kernel reads an f16 scale */
    if (ez->dtype != RAD_U8) return -2;               /* ...and a byte-wide zero; see the header */
    if (es->rows != N || es->cols != G || ez->rows != N || ez->cols != G) return -2;
    if ((es->row_bytes & 1) || es->row_bytes < 2 * G || ez->row_bytes < G) return -3;

#define R3D_W2A8_LAUNCH(MT)                                                                          \
    do {                                                                                             \
        const dim3 grid((N + kWaves - 1) / kWaves), block(kBlockThreads);                             \
        R3D_KLAUNCH(R3D_KNAME((gemm_w2a8_kernel<MT>)), grid, block, (hipStream_t)stream,             \
                    (const int8_t*)a, (const float*)a_scale, 1, (const uint8_t*)b,                   \
                    (const uint16_t*)sp, (int)es->row_bytes, (const uint8_t*)zp, (int)ez->row_bytes, \
                    (const float*)a_sum, kMpad, (uint16_t*)y, M, N, K, chunk, a_ld, y_ld);           \
    } while (0)

    switch (mt) {
        case 1:  R3D_W2A8_LAUNCH(1);  break;
        case 2:  R3D_W2A8_LAUNCH(2);  break;
        case 4:  R3D_W2A8_LAUNCH(4);  break;
        case 8:  R3D_W2A8_LAUNCH(8);  break;
        default: R3D_W2A8_LAUNCH(16); break;
    }
#undef R3D_W2A8_LAUNCH
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* ================================================================== the shim
 *
 * gemm_nt_q's slots, as gemm_ntq_impl takes them: 0 a (i8, step order), 1 a_scale (f32, one per
 * row), 2 b (the u2 codes), 3 b_scale (the joined scale+zero), 4 y (bf16), 5 a_sum (f32,
 * group-major, mpad = 16 columns). */
extern "C" int r3d_rad_gemm_nt_q_w2a8(const RadArgs* a, RadStream s)
{
    if (!a) return R3D_NO(RAD_E_INVAL);
    const char* dt = r3d_gets(a, "dtype");
    if (dt && std::strcmp(dt, "w2a8") != 0) return R3D_NO(RAD_E_DTYPE);   /* the row is w2a8 only */

    const RadTensor* A = r3d_opt(a, 0);
    const RadTensor* AS = r3d_opt(a, 1);
    const RadTensor* B = r3d_opt(a, 2);
    const RadTensor* BS = r3d_opt(a, 3);
    const RadTensor* Y = r3d_opt(a, 4);
    const RadTensor* ASUM = r3d_opt(a, 5);
    if (!r3d_p(A) || !r3d_p(AS) || !r3d_p(B) || !r3d_p(BS) || !r3d_p(Y) || !r3d_p(ASUM))
        return R3D_NO(RAD_E_INVAL);

    if (A->dtype != RAD_I8) return R3D_NO(RAD_E_DTYPE);       /* quant_act_i8's codes */
    if (B->dtype != RAD_U2) return R3D_NO(RAD_E_DTYPE);       /* the grid's codes */
    if (Y->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f32(AS) || !r3d_is_f32(ASUM)) return R3D_NO(RAD_E_DTYPE);

    const long long N = r3d_geti_or(a, "N", 0), K = r3d_geti_or(a, "K", 0);
    const long long group = r3d_geti_or(a, "group", 0);
    if (N < 1 || K < 1) return R3D_NO(RAD_E_SHAPE);
    if (group != kGroup) return R3D_NO(RAD_E_SHAPE);          /* no defaulting: gemm_zp refuses 0 */
    if (K % group || K % 16) return R3D_NO(RAD_E_SHAPE);
    const long long G = K / group;

    const long long ynel = r3d_numel(Y);
    if (N < 1 || ynel % N) return R3D_NO(RAD_E_SHAPE);
    const long long M = ynel / N;
    if (M < 1 || M > kMTMax) return R3D_NO(RAD_E_SHAPE);      /* declined; see the header */
    if (r3d_numel(A) < M * K || r3d_numel(B) < N * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(AS) < M) return R3D_NO(RAD_E_INVAL);        /* gemm_zp's code for this one */
    if (r3d_numel(ASUM) < G * kMpad) return R3D_NO(RAD_E_INVAL);

    const long long ap = r3d_row_pitch(A);
    if (ap < K || ap % 4) return R3D_NO(RAD_E_STRIDE);        /* step order: four codes a dword */
    if (!r3d_contig(B)) return R3D_NO(RAD_E_STRIDE);          /* gemm_zp: contiguous codes only */
    long long yrows = 0;
    const long long yp = r3d_rows_pitch(Y, N, &yrows);
    if (yp < N) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_contig(ASUM)) return R3D_NO(RAD_E_STRIDE);

    const RadPlanesHeader* h = rad_planes_header(r3d_p(BS));
    if (!h) return R3D_NO(RAD_E_DTYPE);                       /* the joined form is the operand */
    const RadPlanesEntry* es = nullptr;
    const RadPlanesEntry* ez = nullptr;
    const void* sp = rad_planes_find(r3d_p(BS), "scale", &es);
    const void* zp = rad_planes_find(r3d_p(BS), "zero", &ez);
    if (!sp || !zp || !es || !ez) return R3D_NO(RAD_E_DTYPE);
    if (es->dtype != RAD_F16) return R3D_NO(RAD_E_DTYPE);
    if (ez->dtype != RAD_U8)
        return R3D_NO(RAD_E_DTYPE);   /* declined: a sub-byte zero plane needs the bit-extract
                                       * path (rad_load_bits). U8 is w2a8's form. */
    if (es->rows != N || es->cols != G || ez->rows != N || ez->cols != G) return R3D_NO(RAD_E_SHAPE);
    if ((es->row_bytes & 1) || es->row_bytes < 2 * G || ez->row_bytes < G) return R3D_NO(RAD_E_STRIDE);

    if (!r3d_aligned(r3d_p(A), 4) || !r3d_aligned(r3d_p(B), 16)) return R3D_NO(RAD_E_ALIGN);

    const int rc = r3d_gemm_nt_q_w2a8(
        r3d_addr(A), r3d_addr(AS), r3d_addr(B), r3d_addr(BS), r3d_addr(ASUM), r3d_addr(Y), (int)M,
        (int)N, (int)K, (int)group, (int)ap, (int)yp, 0, (long)s);
    /* Launcher codes (null, shape, pitch, HIP) to RAD codes, as in the gated row. */
    if (rc == 0) return RAD_OK;
    return R3D_NO(rc == -1 ? RAD_E_INVAL : rc == -2 ? RAD_E_SHAPE
                            : rc == -3 ? RAD_E_STRIDE : RAD_E_DEVICE);
}

/* ================================================================== the layout hook
 *
 * One of libr3d's few layout hooks, transcribed from libref's `ref_layout_gemm_q` with the checks
 * and their order unchanged:
 *
 *     if (n < 2 || operand != 3) return RAD_E_UNSUPPORTED;
 *     const ZpGrid* zg = zp_grid(dtype);  group = param "group";
 *     if (!zg || group <= 0 || !rad_enc_is(enc, "affine")) return RAD_E_DTYPE;
 *     transform empty, or beginning "givens";
 *     sel_roles(enc, sel, n, "scale", "zero") && enc->n_planes >= 3;
 *     plane[0]: role "codes", dtype == zg->codes, block {1,1};
 *     scale:    RAD_PLANE_TILED, RAD_F16, block {1, group};
 *     zero:     RAD_PLANE_TILED, zero_dt(...), block {1, group};
 *     return rad_planes_layout(enc, sel, planes, n, out);
 *
 * zp_grid, zero_dt and sel_roles are libref-internal statics, inlined below for w2a8: the grid is
 * { "w2a8", RAD_U2, ZP_I8_ROW }, zero_dt is U8|U2|U4|U5, and the practical form is U8 (the kernel
 * declines a sub-byte zero at its site). The answer is rad_planes.h's: the tag "rad.planes", the
 * dtype the first plane's, the shape the byte count. The hook moves no bytes.
 */
/* The writer for the joined operand this hook names: codes, scale and zero as one `rad.planes`
 * weight, assembled by rad_planes_relayout (header, entries, each plane's rows, copied). A layout
 * query answering RAD_OK promises "out describes the stored form, and relayout will produce it"
 * (rad_abi.h). Only a weight with no hook can be read in place, so a joined weight is placed in
 * VRAM while a plain one can stay on the mapped tier. No code is transformed. */
extern "C" int r3d_relayout_gemm_nt_q_w2a8(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const RadTensor* planes, int n_planes, void* dst,
                                           int64_t dst_bytes)
{
    (void)p; (void)n_p;
    if (operand != 3) return RAD_E_UNSUPPORTED;
    return rad_planes_relayout(enc, sel, planes, n_planes, dst, dst_bytes);
}

/* Under RAD_R3D_TRACE, prints each layout query (operand, band dtype, planes offered) and its
 * return code. Otherwise a model that fails to load reports only that two rows want one weight in
 * two forms. */
static int r3d_layout_trace(const char* who, int operand, const char* dt, int n_planes, int rc) {
    if (r3d_trace_on())
        std::fprintf(stderr, "[r3d] %s: operand %d dtype %s planes %d -> %d\n", who, operand,
                     dt ? dt : "(none)", n_planes, rc);
    return rc;
}

extern "C" int r3d_layout_gemm_nt_q_w2a8(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel,
                                         const RadTensor* planes, int n, RadLayout* out)
{
    if (n < 2 || operand != 3) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_UNSUPPORTED);
    const char* dt = r3d_pgets(p, n_p, "dtype");
    /* The (scale, zero) pair joined on operand 3 describes the container, not the code width, and
     * every row that reads one weight must answer with one form. So both packed grids pass here;
     * the code width is checked by each shim (gemm_nt_q_w2a8_m16 reads U2 only,
     * gemm_nt_q_w2a8_tiled reads U2 and U4). */
    if (!dt || (std::strcmp(dt, "w2a8") != 0 && std::strcmp(dt, "w4a8_asym") != 0))
        return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);                                            /* the packed grids */
    const long long group = r3d_pgeti_or(p, n_p, "group", 0);
    if (group <= 0) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
    if (!enc || !rad_enc_is(enc, "affine")) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
    if (enc->transform[0] && std::strncmp(enc->transform, "givens", RAD_ENC_STR) != 0)
        return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
    if (!enc || enc->n_planes < 3 || !sel) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);

    /* sel_roles(enc, sel, n, "scale", "zero"): one role per selected plane, in order. */
    if (n != 2) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
    for (int i = 0; i < 2; ++i) {
        const char* want = i == 0 ? "scale" : "zero";
        if (sel[i] < 0 || sel[i] >= enc->n_planes) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
        if (std::strncmp(enc->plane[sel[i]].role, want, RAD_ENC_STR) != 0) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
    }
    /* The codes plane is not part of this operand: role "codes", the grid's dtype, block {1,1}.
     * libref finds it by role over enc->plane, not by sel. */
    int codes = -1;
    for (int i = 0; i < enc->n_planes; ++i)
        if (std::strncmp(enc->plane[i].role, "codes", RAD_ENC_STR) == 0) { codes = i; break; }
    if (codes < 0) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
    if (enc->plane[codes].block[0] != 1 || enc->plane[codes].block[1] != 1) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);

    for (int i = 0; i < 2; ++i) {
        const RadEncPlane& q = enc->plane[sel[i]];
        const RadTensor& t = planes[i];
        if (q.kind != RAD_PLANE_TILED) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
        if (q.block[0] != 1 || q.block[1] != group) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
        if (i == 0) {
            if (t.dtype != RAD_F16) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);               /* the scale is f16 */
        } else {
            const uint32_t zd = t.dtype;
            const bool ok = zd == RAD_U8 || zd == RAD_U2 || zd == RAD_U4 || zd == RAD_U5;
            if (!ok) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);                              /* zero_dt: never signed */
        }
    }
    return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, rad_planes_layout(enc, sel, planes, n, out));
}
