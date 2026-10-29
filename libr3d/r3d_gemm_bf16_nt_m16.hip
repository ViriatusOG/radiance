/* r3d_gemm_bf16_nt_m16.hip -- the skinny bf16 GEMM, M <= 16.
 *
 *   y[M,N] = a[M,K] @ b[N,K]^T, f32 accumulate, narrowed once by the store.
 *
 * The band is decode and speculative verify, where the weight stream is the cost. The kernel
 * streams the whole weight plane once, and every output's sum is the sum libref forms.
 *
 * ---- the lane map --------------------------------------------------------------------------
 *
 * One output element per lane, walking k ascending. A wave covers MT activation rows by 32/MT
 * weight rows: lane l owns (m = l % MT, n = n0 + l / MT), where MT (the template argument) is M
 * rounded up to 1, 2, 4, 8 or 16. Lanes sharing an n read the same weight words; lanes sharing an m
 * read the same activation words from LDS. Each lane owns one accumulator from 0 to K and never
 * exchanges a value with another lane:
 *
 *     acc = 0; for k ascending: acc = dot2(pair k+1, dot2(pair k, acc))
 *
 * which is `s += a[k]*b[k]` in ascending k, the loop r4d_gemm_bf16_nt_tiled's contract quotes and
 * ref_gemm.h's quant_tile runs. No wave reduction, so nothing reassociates: at sk = 1 the output
 * is bit-identical to libref's.
 *
 * Rows past M and columns past N are clamped, not skipped: a tail lane re-reads the last real row
 * and its result is dropped at the store, so the k loop stays uniform and the loads straight-line.
 *
 * ---- where the bytes come from --------------------------------------------------------------
 *
 * The activation is staged in LDS: a tile is MT * 128 bf16 (4.25 KiB at MT = 16), one 16-byte load
 * per thread. Staging widens bf16 -> f16 once per element with the exact bf16x2_to_f16x2, so the
 * dot2 reads f16 pairs straight from LDS and the conversion is not repeated for every weight row.
 * The pitch is 68 dwords: 68 mod 32 = 4 puts the eight lanes of a ds_read_b128 phase on banks 4
 * apart (conflict-free), and 68 * 4 = 272 bytes keeps a 16-byte LDS read aligned.
 *
 * The weight is not staged: with M this small a staged tile would be read by 32/MT lanes and
 * discarded, so LDS buys no reuse. Weight bytes go from global memory into registers, are widened
 * there by four v_cvt_pkrtz per eight values (one convert per two dot2 products), and live for the
 * four products they feed. Each wave keeps NT = 32/MT rows resident in L1 (NT lines of 128 B,
 * 4 KiB at MT = 1, the worst case), so 16-byte-per-lane strided reads across NT rows cost only what
 * they fetch.
 *
 * ---- the register budget --------------------------------------------------------------------
 *
 * A gfx11 SIMD has 1024 VGPRs for a wave32. The loop holds one accumulator, four activation pairs,
 * four weight dwords, one 16-byte staging quad and some pointers and counters: about 24 live
 * VGPRs, far under the 128 that R3D_WAVES(8) (eight waves per SIMD, 1024/8) allows. The target is
 * stated for the reason r3d_common.h gives: otherwise the scheduler picks the occupancy, and this
 * kernel's throughput is waves in flight, each waiting hundreds of cycles on a memory load. Eight
 * waves per 256-thread block is the default; `wv` is the tuner's axis.
 *
 * ---- split-K --------------------------------------------------------------------------------
 *
 * At M = 1 the only parallelism is N, and N = 48 (a GDN a|b projection) gives three waves on a
 * 32-CU card. `sk` splits K into that many contiguous slabs, one per blockIdx.y, each writing an
 * f32 partial to the caller's `scratch` arena; a second kernel on the same stream adds the partials
 * per output in ascending split order and narrows once.
 *
 * Summing sk per-slab partials is not libref's single ascending chain: the slab sums bracket the
 * products differently. sk = 1 is bit-identical; sk > 1 agrees to the last bits, the same contract
 * libr4d's tiled row states for its slices. The row is verified at every sk kbench measures.
 *
 * `sk == 0` picks the largest legal split from the axis's ladder {16, 8, 4, 2, 1}. A split is legal
 * when K % (sk * 8) == 0, so a slice boundary never falls inside a lane's step of eight k (four f16
 * pairs). The staged tile is 128 k wide with a ragged last tile, which allows slices as short as
 * 8 k. The cap of 16 is the split the row's scratch hook sizes the arena for.
 *
 * ---- `nt` -----------------------------------------------------------------------------------
 *
 * RDNA3 has no global_load_tr_b128 (an RDNA4 instruction) and no cache-swizzle field on a global
 * load. nt = 1 issues the weight reads through an address-space-1 pointer, which lowers to
 * global_load_b128 with no cache modifier and no swizzle; nt = 0 issues the same 16-byte reads
 * through the generic pointer, which lowers to flat_load_dwordx4. The axis lets kbench price the
 * two lowerings, which on this card stream the same.
 */

/* hip_runtime.h first: it defines __forceinline__, which r3d_codecs.h's R3D_HD uses, and
 * r3d_common.h includes the codecs before the runtime. */
#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <algorithm>

namespace {

constexpr int kBlockThreads = 256;           // eight waves, the default
constexpr int kWaves        = kBlockThreads / kWave;
constexpr int kKTile        = 128;           // k covered by one staged tile of a
constexpr int kAPitch       = kKTile / 2 + 4;   // dwords per staged row of a
constexpr int kMTMax        = 16;            // the band's upper bound

/* The split rule, shared by the launcher, the shim and the row's gemm_tune_valid. `kUnit` is the
 * k one lane step consumes (eight elements, four f16 pairs); a slice must be whole steps, so a
 * split is legal when K divides by sk * kUnit. The ladder is the row's `sk` axis descending; its
 * top is the split the scratch hook sizes the arena for. */
constexpr int kUnit = 8;
inline int pick_sk(long long K, int maxsk)
{
    static const int kLadder[] = { 16, 8, 4, 2, 1 };
    for (int s : kLadder)
        if (s <= maxsk && K % ((long long)s * kUnit) == 0) return s;
    return 1;
}

/* A 16-byte weight read. NT true is the address-space-1 form (global_load_b128, no cache modifier,
 * no swizzle); NT false goes through the generic pointer (flat_load_dwordx4). See the header. */
template <bool NT>
__device__ __forceinline__ r3d_u32x4 ldw(const uint16_t* p)
{
    if constexpr (NT) return *(const __attribute__((address_space(1))) r3d_u32x4*)p;
    else return __builtin_bit_cast(r3d_u32x4, *(const uint4*)p);
}

__device__ __forceinline__ r3d_u32x4 lds_a(const uint32_t* p)
{
    return *(const r3d_u32x4*)p;
}

/* y[M,N] = a[M,K] @ b[N,K]^T, one output element per lane.
 *
 * `kslab` is the k range one blockIdx.y owns. With sk = 1 it is all of K and the store is bf16.
 * With sk > 1 each split writes an f32 partial into a dense [sk][M][N] plane, and
 * gemm_split_reduce adds them. */
template <int MT, bool NT, bool SPLIT>
__global__ __launch_bounds__(kBlockThreads) R3D_WAVES(8) void gemm_m16_kernel(
    const uint16_t* __restrict__ A, const uint16_t* __restrict__ B,
    uint16_t* __restrict__ Y, float* __restrict__ Part,
    int M, int N, int K, int a_ld, int y_ld, int w_ld, int kslab)
{
    /* Staged activation, widened to f16 pairs. Sized for the widest MT (4.25 KiB) in every
     * instantiation. */
    __shared__ uint32_t s_a[kMTMax * kAPitch];

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wave = tid / kWave;
    const uint32_t waves = blockDim.x / kWave;
    constexpr uint32_t kNT = kWave / MT;              // weight rows a wave covers

    const uint32_t n0  = ((uint32_t)blockIdx.x * waves + wave) * kNT + lane / MT;
    const uint32_t m   = lane & (MT - 1);
    const int n_real = (int)(n0 < (uint32_t)N ? n0 : (uint32_t)(N - 1));
    const int m_real = (int)(m < (uint32_t)M ? m : (uint32_t)(M - 1));

    const int kbeg = (int)blockIdx.y * kslab;
    const int kend = kbeg + kslab < K ? kbeg + kslab : K;

    const uint16_t* w = B + (size_t)n_real * w_ld;
    float acc = 0.0f;

    for (int kt = kbeg; kt < kend; kt += kKTile) {
        const int kl = std::min(kKTile, kend - kt);

        /* Stage: eight elements per unit, MT * (kl/8) units. A row past M re-reads the last real
         * row, whose products the store drops. */
        const int nu = (kl + 7) / 8;
        /* Stride by blockDim.x, not 256: the wv axis launches one, two, four or eight waves, and a
         * fixed stride would leave part of the tile unstaged (wv = 1 would stage 32 of MT*16
         * units). */
        for (int i = (int)tid; i < MT * nu; i += (int)blockDim.x) {
            const int r = i / nu, c = (i % nu) * 8;
            const int rr = r < M ? r : M - 1;
            const uint16_t* src = A + (size_t)rr * a_ld + kt + c;
            uint32_t* dst = s_a + (size_t)r * kAPitch + c / 2;
            if (c + 8 <= kl) {
                const r3d_u32x4 q = *(const __attribute__((address_space(1))) r3d_u32x4*)src;
                *(r3d_u32x4*)dst = r3d_u32x4{
                    bf16x2_to_f16x2(q.x), bf16x2_to_f16x2(q.y),
                    bf16x2_to_f16x2(q.z), bf16x2_to_f16x2(q.w) };
            } else {
                uint32_t p[4] = { 0, 0, 0, 0 };
                for (int e = 0; e < kl - c; e += 2) {
                    const uint32_t lo = ldg_u16(src + e);
                    const uint32_t hi = (e + 1 < kl) ? ldg_u16(src + e + 1) : 0u;
                    p[e / 2] = bf16x2_to_f16x2(lo | (hi << 16));
                }
                for (int e = 0; e < (kl - c + 1) / 2; ++e) dst[e] = p[e];
            }
        }
        group_barrier();

        const uint32_t* ap = s_a + (size_t)m * kAPitch;
        const int kf = kl & ~7;
        for (int k = 0; k < kf; k += 8) {
            const r3d_u32x4 aw = lds_a(ap + k / 2);
            const r3d_u32x4 b16 = ldw<NT>(w + kt + k);
            /* The weight must be widened: dot2's operands are f16, and a bf16 bit pattern read as
             * f16 is a different number (bf16 1.0, 0x3F80, reads as 1.875; magnitudes >= 2^16
             * become infinities). Four v_cvt_pk_rtz_f16_f32_e32 per eight values, each with two
             * VALU of unpack, so the widening costs one ALU slot per dot2 product; RDNA3 pairs
             * some of the dots as v_dual_dot2acc_f32_f16 alongside the converts. */
            const r3d_u32x4 bw = r3d_u32x4{ bf16x2_to_f16x2(b16.x), bf16x2_to_f16x2(b16.y),
                                            bf16x2_to_f16x2(b16.z), bf16x2_to_f16x2(b16.w) };
            acc = dot4_f16x4(aw.x, aw.y, bw.x, bw.y, acc);
            acc = dot4_f16x4(aw.z, aw.w, bw.z, bw.w, acc);
        }
        /* The elements past the last whole step of eight, ascending. The launcher's
         * K % (sk * 8) == 0 rule leaves none, so this loop does not run. */
        for (int k = kf; k < kl; ++k)
            acc = acc + bf16_to_f32((uint16_t)ldg_u16(A + (size_t)m_real * a_ld + kt + k))
                      * bf16_to_f32((uint16_t)ldg_u16(w + kt + k));
        group_barrier();                      // the next tile's stage must not race these reads
    }

    const bool keep = (n0 < (uint32_t)N) && (m < (uint32_t)M);
    if (!keep) return;
    if constexpr (SPLIT) {
        /* Partial plane [sk][M][N], split-major: the reducer reads sk values M*N apart. */
        Part[((size_t)blockIdx.y * M + m) * N + n0] = acc;
    } else {
        Y[(size_t)m * y_ld + n0] = f32_to_bf16(acc);   // narrowed once, here
    }
}

/* The split reduction: v = ((p0 + p1) + p2) + ... in ascending split order, then one narrowing.
 * At sk > 1 this differs from libref's single chain in the last bits (see the header). */
__global__ __launch_bounds__(kBlockThreads) R3D_WAVES(8) void gemm_split_reduce(
    const float* __restrict__ Part, uint16_t* __restrict__ Y, int M, int N, int y_ld, int sk)
{
    const int mn = M * N;
    const int stride = gridDim.x * kBlockThreads;
    for (int i = blockIdx.x * kBlockThreads + threadIdx.x; i < mn; i += stride) {
        const int m = i / N, n = i - m * N;
        const float* p = Part + (size_t)m * N + n;
        float v = p[0];
        for (int s = 1; s < sk; ++s) v = v + p[(size_t)s * M * N];
        Y[(size_t)m * y_ld + n] = f32_to_bf16(v);
    }
}

}  // namespace

/* y[M,N] = a[M,K] @ b[N,K]^T, bf16 in and out, f32 accumulate, M <= 16.
 *
 *   a  [M][K] bf16, `a_ld` elements per row (0 = K)
 *   b  [N][K] bf16, row-major, k contiguous
 *   y  [M][N] bf16, `y_ld` elements per row (0 = N)
 *
 * `sk` splits K (0 = the largest legal split), `wv` is waves per block (0 = 8), `nt` picks the
 * weight load's lowering (see the file header), and `scratch` is the f32 partial arena,
 * sk * M * N * 4 bytes, required when sk > 1. Returns 0, -1 a null pointer, -2 an unsupported
 * shape, -3 an unreadable pitch, -4 the HIP launch failed. */
extern "C" int r3d_gemm_nt_bf16_m16(long a, long b, long y, int M, int N, int K, int a_ld,
                                    int y_ld, int sk, int wv, int nt, long scratch, long stream)
{
    if (!a || !b || !y) return -1;
    if (M < 1 || N < 1 || K < 1) return -2;
    if (M > kMTMax) return -2;
    if (!a_ld) a_ld = K;
    if (!y_ld) y_ld = N;
    if (sk == 0) sk = pick_sk(K, 16);
    if (sk < 1) return -2;
    if (K % ((long long)sk * kUnit)) return -2;
    if (sk > 1 && !scratch) return -1;
    if (wv <= 0) wv = kWaves;
    if (wv != 1 && wv != 2 && wv != 4 && wv != kWaves) return -2;

    const int mt = M <= 1 ? 1 : M <= 2 ? 2 : M <= 4 ? 4 : M <= 8 ? 8 : kMTMax;
    const int kslab = K / sk;

#define R3D_M16_NT(MTV, NTV)                                                                    \
    do {                                                                                        \
        const int nt_ = (N + (kWave / MTV) - 1) / (kWave / MTV);                                \
        const dim3 grid((nt_ + wv - 1) / wv, (unsigned)sk), block((unsigned)(wv * kWave));      \
        if (sk > 1) {                                                                           \
            R3D_KLAUNCH(R3D_KNAME((gemm_m16_kernel<MTV, NTV, true>)), grid, block,              \
                        (hipStream_t)stream, (const uint16_t*)a, (const uint16_t*)b,           \
                        (uint16_t*)y, (float*)scratch, M, N, K, a_ld, y_ld, K, kslab);          \
            const int mn = M * N;                                                               \
            const int nb = std::min((mn + kBlockThreads - 1) / kBlockThreads, 4096);            \
            R3D_KLAUNCH(gemm_split_reduce, dim3(nb), dim3(kBlockThreads), (hipStream_t)stream,  \
                        (const float*)scratch, (uint16_t*)y, M, N, y_ld, sk);                   \
        } else {                                                                                \
            R3D_KLAUNCH(R3D_KNAME((gemm_m16_kernel<MTV, NTV, false>)), grid, block,             \
                        (hipStream_t)stream, (const uint16_t*)a, (const uint16_t*)b,           \
                        (uint16_t*)y, (float*)nullptr, M, N, K, a_ld, y_ld, K, kslab);          \
        }                                                                                       \
    } while (0)
#define R3D_M16_MT(MTV) do { if (nt) R3D_M16_NT(MTV, true); else R3D_M16_NT(MTV, false); } while (0)
    switch (mt) {
        case 1:  R3D_M16_MT(1);  break;
        case 2:  R3D_M16_MT(2);  break;
        case 4:  R3D_M16_MT(4);  break;
        case 8:  R3D_M16_MT(8);  break;
        default: R3D_M16_MT(16); break;
    }
#undef R3D_M16_MT
#undef R3D_M16_NT
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* ============================================================== the plugin ABI entry point
 *
 * gemm_nt's operand list is [a, b, y] and its parameters M, N, K, dtype (libref's pGemm/oGemm). */
enum { G_A = 0, G_B, G_Y };

#define R3D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R3D_NO(_s); } while (0)
static int dt_bf16(const RadTensor* t) { return r3d_is_bf16(t) ? RAD_OK : RAD_E_DTYPE; }

extern "C" int r3d_rad_gemm_nt_bf16_m16(const RadArgs* a, RadStream s)
{
    const RadTensor* A = r3d_opt(a, G_A);
    const RadTensor* B = r3d_opt(a, G_B);
    const RadTensor* Y = r3d_opt(a, G_Y);
    if (!A || !B || !Y) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(dt_bf16(A));
    R3D_NEED(dt_bf16(B));
    R3D_NEED(dt_bf16(Y));

    const long long K = r3d_geti_or(a, "K", r3d_dim(B, (int)B->rank - 1));
    const long long N = r3d_geti_or(a, "N", r3d_dim(B, (int)B->rank - 2));
    if (K < 1 || N < 1) return R3D_NO(RAD_E_SHAPE);
    /* The row count comes from the output, never from `M`, which is the band's upper bound. */
    const long long nel = r3d_numel(Y);
    if (nel < N || nel % N) return R3D_NO(RAD_E_SHAPE);
    const long long M = nel / N;
    if (M < 1 || M > kMTMax) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(A) < M * K || r3d_numel(B) < N * K) return R3D_NO(RAD_E_SHAPE);

    long long arows = 0, yrows = 0;
    const long long a_ld = r3d_rows_pitch(A, K, &arows);
    const long long y_ld = r3d_rows_pitch(Y, N, &yrows);
    if (a_ld < 0 || y_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (arows < M || yrows < M) return R3D_NO(RAD_E_SHAPE);
    /* The weight must be dense: every wave reads b[n][k] at n * K + k, and the launcher takes no
     * weight pitch. */
    if (!r3d_contig(B)) return R3D_NO(RAD_E_STRIDE);
    /* 16-byte loads at a row base and at a tile base, both multiples of 8 elements. */
    if (a_ld % 8) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_aligned(A->data, 16) || !r3d_aligned(B->data, 16)) return R3D_NO(RAD_E_ALIGN);

    R3dGemmCfg c;
    r3d_cfg_gemm(a, &c);
    int wv = c.wv > 0 ? c.wv : kWaves;
    if (wv > kWaves) wv = kWaves;
    if (wv != 1 && wv != 2 && wv != 4 && wv != kWaves) return R3D_NO(RAD_E_SHAPE);
    int sk = c.sk > 0 ? c.sk : pick_sk(K, 16);
    if (sk > 16) return R3D_NO(RAD_E_SHAPE);            // the scratch hook sizes for 16
    if (K % ((long long)sk * kUnit)) return R3D_NO(RAD_E_SHAPE);
    if (sk > 1) {
        const long long need = (long long)sk * M * N * 4;
        if (!a->scratch || a->scratch_bytes < need) return R3D_NO(RAD_E_SCRATCH);
    }

    const int e = r3d_gemm_nt_bf16_m16(r3d_addr(A), r3d_addr(B), r3d_addr(Y), (int)M, (int)N,
                                       (int)K, (int)a_ld, (int)y_ld, sk, wv, c.nt,
                                       (long)(uintptr_t)(sk > 1 ? a->scratch : nullptr),
                                       (long)(uintptr_t)s);
    if (e == 0) return RAD_OK;
    return R3D_NO(e == -1 ? RAD_E_INVAL : e == -2 ? RAD_E_SHAPE
                        : e == -3 ? RAD_E_STRIDE : RAD_E_DEVICE);
}
