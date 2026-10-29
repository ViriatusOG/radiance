/* r3d_gemm_bf16_nt_m16.hip -- the skinny bf16 GEMM, M <= 16: the linear a token is made of.
 *
 *   y[M,N] = a[M,K] @ b[N,K]^T, f32 accumulate, narrowed once by the store.
 *
 * The band is decode and speculative verify, where the weight is the cost and the activation is
 * free: the kernel is built so the whole weight plane streams once, and so every output's sum is
 * the sum libref forms.
 *
 * ---- the lane map --------------------------------------------------------------------------
 *
 * One output element a lane, walking k ascending. A wave covers MT rows of the activation by
 * 32/MT rows of the weight: lane l owns (m = l % MT, n = n0 + l / MT), MT being the template
 * argument -- the request's M rounded up to 1, 2, 4, 8 or 16. Lanes sharing an n read the same
 * weight words, so the wave issues 32/MT loads per step and broadcasts them to MT lanes each;
 * lanes sharing an m read the same activation words out of LDS. A lane therefore owns its own
 * accumulator from 0 to K and never exchanges a value with another lane: its chain is
 *
 *     acc = 0; for k ascending: acc = dot2(pair k+1, dot2(pair k, acc))
 *
 * with pairs taken two k at a time, which is `s += a[k]*b[k]` in ascending k, summand for summand,
 * the loop r4d_gemm_bf16_nt_tiled's contract quotes and ref_gemm.h's quant_tile runs. There is no
 * wave reduction anywhere in this kernel, so nothing reassociates: at sk = 1 the output is
 * bit-identical to libref's, not merely within a tolerance.
 *
 * Rows past M and columns past N are clamped, not skipped -- a tail lane re-reads the last real row
 * and its result is dropped at the store -- so the k loop stays uniform and the loads stay
 * straight-line.
 *
 * ---- where the bytes come from --------------------------------------------------------------
 *
 * The activation is staged in LDS and widened on the way in: one row is M <= 16 elements wide per
 * k, so a tile is MT * 128 bf16 (4.25 KiB at MT = 16) and one 16-byte load a thread. Staging
 * converts bf16 -> f16 once per element with bf16x2_to_f16x2, which is exact, so the dot2 reads
 * f16 pairs straight out of LDS and the widening is paid N/MT times less often than it would be
 * read. The pitch is 68 dwords: 68 mod 32 = 4, so the eight lanes of a ds_read_b128 phase land on
 * banks 4 apart -- conflict-free -- and 68 * 4 = 272 keeps a 16-byte LDS read aligned.
 *
 * The weight is NOT staged. M is tiny, so a staged tile would be read by 32/MT lanes and then
 * thrown away; LDS staging buys reuse a skinny shape does not have. Weight bytes go straight from
 * global memory into registers, widened there by four v_cvt_pkrtz per eight values (one per dword,
 * so one convert per two dot2 products -- the price of the dot2 path being the widening the staged
 * row pays once per element), and live for the four products they feed.
 * Each wave keeps NT = 32/MT rows resident in L1 (NT lines of 128 B, 4 KiB at MT = 1, the worst
 * case), which is what makes 16-byte-per-lane strided reads across NT rows cost what they fetch.
 *
 * ---- the register budget --------------------------------------------------------------------
 *
 * A gfx11 SIMD has 1024 VGPRs for a wave32. The loop is one accumulator, four activation pairs,
 * four weight dwords, one 16-byte staging quad and a handful of pointers and counters: about 24
 * live VGPRs, so the kernel is nowhere near the 128 that R3D_WAVES(8) -- eight waves a SIMD,
 * 1024/8 -- asks for. The target is stated for the reason r3d_common.h gives: unstated, the
 * scheduler picks the occupancy itself, and this kernel's throughput is waves in flight, since
 * every wave is waiting on a memory load that takes hundreds of cycles. Eight waves a block of
 * 256 threads is the default, and `wv` is the axis a tuner walks.
 *
 * ---- split-K --------------------------------------------------------------------------------
 *
 * At M = 1 a wide N is the only parallelism there is, and N = 48 (a GDN a|b projection) gives three
 * waves on a 32-CU card. `sk` splits K into that many contiguous slabs, each owned by one
 * blockIdx.y, each writing an f32 partial to the caller's `scratch` arena; a second kernel on the
 * same stream adds the partials per output in ascending split order and narrows once.
 *
 * The order matters and is stated: summing sk per-slab partials is NOT the ascending sum libref
 * forms. Each partial is an ascending sum over its own slab, and the slab sums are then added in
 * ascending slab order, which brackets the products differently than a single running chain does.
 * sk = 1 is the bit-identical case; sk > 1 agrees to the last bits, which is the same contract
 * libr4d's tiled row states about its slices. The row is verified at every sk kbench measures.
 *
 * `sk == 0` means "the largest legal split for this K" off the axis's ladder {16, 8, 4, 2, 1},
 * legality being K % (sk * 8) == 0: a slice must end where the inner loop stops, on a group of the
 * eight k (four f16 pairs) a lane consumes per step, so a slice boundary never falls inside a
 * step. The staged tile is 128 k wide and its last tile is ragged, which is what lets a slice be
 * 8 k long, and the cap of 16 is the split the row's scratch hook sizes the arena for.
 *
 * ---- `nt` -----------------------------------------------------------------------------------
 *
 * RDNA3 has no global_load_tr_b128 (that is RDNA4) and no cache-swizzle field on a global load. So
 * `nt` is an axis that measures a knob the ISA does not special-case: nt = 1 issues the weight
 * reads through an address-space-1 pointer, which lowers to global_load_b128 with no cache modifier
 * and no swizzle -- a plain stream -- and nt = 0 issues the same 16-byte reads through the generic
 * pointer, which lowers to flat_load_dwordx4. Nothing is invented here; the axis exists so kbench
 * can price the two lowerings and record that on this card they are the same stream.
 */

/* hip_runtime.h first: it defines __forceinline__, which r3d_codecs.h's R3D_HD spells, and
 * r3d_common.h reaches the codecs before it reaches the runtime. */
#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <algorithm>

namespace {

constexpr int kBlockThreads = 256;           // eight waves, the default
constexpr int kWaves        = kBlockThreads / kWave;
constexpr int kKTile        = 128;           // k a staged tile of a covers
constexpr int kAPitch       = kKTile / 2 + 4;   // dwords a staged a row
constexpr int kMTMax        = 16;            // the band's upper bound

/* The split rule, in one place so the launch, the shim and the row's gemm_tune_valid agree.
 * `kUnit` is the k a lane's step consumes -- eight elements, four f16 pairs -- and a slice must be
 * whole steps, so a split is legal when K divides by sk * kUnit. The ladder is the row's `sk` axis
 * descending, and its top is the split the scratch hook sizes for: a wider split the arena cannot
 * hold is not a legal default. */
constexpr int kUnit = 8;
inline int pick_sk(long long K, int maxsk)
{
    static const int kLadder[] = { 16, 8, 4, 2, 1 };
    for (int s : kLadder)
        if (s <= maxsk && K % ((long long)s * kUnit) == 0) return s;
    return 1;
}

/* A 16-byte weight read. NT true is the address-space-1 form: global_load_b128, no cache modifier,
 * no swizzle, the plain stream the row's `nt` axis names. NT false goes through the generic
 * pointer, which the compiler lowers to flat_load_dwordx4. */
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

/* y[M,N] = a[M,K] @ b[N,K]^T, one output element a lane.
 *
 * `kslab` is the k range a blockIdx.y owns; with sk = 1 it is all of K and the store is the
 * narrowed bf16 the caller asked for. With sk > 1 every split writes its own f32 partial, dense
 * [sk][M][N], and gemm_split_reduce adds them. */
template <int MT, bool NT, bool SPLIT>
__global__ __launch_bounds__(kBlockThreads) R3D_WAVES(8) void gemm_m16_kernel(
    const uint16_t* __restrict__ A, const uint16_t* __restrict__ B,
    uint16_t* __restrict__ Y, float* __restrict__ Part,
    int M, int N, int K, int a_ld, int y_ld, int w_ld, int kslab)
{
    /* Staged activation, widened to f16 pairs on the way in. One allocation for the widest MT:
     * this kernel is instantiated for MT = 1, 2, 4, 8 and 16 and 4.25 KiB is not worth a template. */
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

        /* Stage: eight elements a unit, MT * (kl/8) units, 256 threads. A row past M re-reads the
         * last real row, whose products the store drops. */
        const int nu = (kl + 7) / 8;
        /* blockDim.x, not the 256-thread default: the wv axis launches one, two, four or eight
         * waves, and a stage loop that strides the block constant leaves units staged only for
         * the waves that exist -- wv = 1 stages 32 of MT*16 units and the k loop reads the rest of
         * the tile out of LDS that was never written. */
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
            /* The weight is widened here and it is not optional: dot2's operands are f16, and a
             * bf16 bit pattern read as f16 is a different number -- bf16's 1.0 (0x3F80) is 1.875 --
             * and any magnitude at or above 2^16 is an infinity. Feeding the bytes straight in is
             * the trap this ISA sets for a bf16 GEMM that reaches for v_dot2acc_f32_f16, and it
             * arrives as an output uncorrelated with the reference at rel ~ 1, with infinities in
             * it, which reads as a numerics problem and is a reading problem. Four
             * v_cvt_pk_rtz_f16_f32_e32 per eight values, each with the pair's two VALU of unpack,
             * so the widening costs one ALU slot per product the dot2 makes; RDNA3 pairs some of
             * the dots as v_dual_dot2acc_f32_f16 with the converts riding alongside. */
            const r3d_u32x4 bw = r3d_u32x4{ bf16x2_to_f16x2(b16.x), bf16x2_to_f16x2(b16.y),
                                            bf16x2_to_f16x2(b16.z), bf16x2_to_f16x2(b16.w) };
            acc = dot4_f16x4(aw.x, aw.y, bw.x, bw.y, acc);
            acc = dot4_f16x4(aw.z, aw.w, bw.z, bw.w, acc);
        }
        /* The ragged end of the last slab, ascending. A split is legal only when its slab is whole
         * tiles, so this runs at sk = 1 with a K that is not a multiple of 128. */
        for (int k = kf; k < kl; ++k)
            acc = acc + bf16_to_f32((uint16_t)ldg_u16(A + (size_t)m_real * a_ld + kt + k))
                      * bf16_to_f32((uint16_t)ldg_u16(w + kt + k));
        group_barrier();                      // the next tile's stage must not race this one's reads
    }

    const bool keep = (n0 < (uint32_t)N) && (m < (uint32_t)M);
    if (!keep) return;
    if constexpr (SPLIT) {
        /* Partial plane [sk][M][N]: split-major, so the reducer reads sk values M*N apart and this
         * store keeps the same [M][N] shape at every split. */
        Part[((size_t)blockIdx.y * M + m) * N + n0] = acc;
    } else {
        Y[(size_t)m * y_ld + n0] = f32_to_bf16(acc);   // narrowed once, here
    }
}

/* The split reduction: v = ((p0 + p1) + p2) + ... in ascending split order, then one narrowing.
 *
 * This is the order libref's ascending sum produces at sk = 1 and not at sk > 1: each partial is
 * ascending over its own slab, and the slab sums bracket the products differently than one chain
 * does. The difference is in the last bits and is the split's, not a bug in it. */
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
 *   a  [M][K] bf16, `a_ld` elements a row (0 = K)
 *   b  [N][K] bf16, row-major, k contiguous
 *   y  [M][N] bf16, `y_ld` elements a row (0 = N)
 *
 * `sk` splits K (0 = the largest legal split), `wv` is waves per block (0 = 8), `nt` picks the
 * weight load's lowering (see the file header), and `scratch` is the f32 partial arena,
 * sk * M * N * 4 bytes, required when the split is on. Returns 0, -1 a null pointer, -2 a shape it
 * does not serve, -3 a pitch it cannot read, -4 the HIP launch failed. */
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
    /* The weight must be dense: every wave reads b[n][k] at n * K + k, and a pitch the launcher
     * does not carry cannot be described to it. */
    if (!r3d_contig(B)) return R3D_NO(RAD_E_STRIDE);
    /* 16-byte weight loads, at a row base and at a tile base, both multiples of 8 elements. */
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
