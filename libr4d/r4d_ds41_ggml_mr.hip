/* r4d_ds41_ggml_mr.hip -- gemm_nt_ggml / logits_gemm_ggml's decode band (M <= 48 activation rows)
 * in a multi-row form: every weight block read and dequantised ONCE for all the call's activation rows, with
 * each output's floating-point operations exactly those of the stock kernel (gg_gemv_kernel), so the bytes are
 * the same. On at the measured table's shapes (r4d_ds41_ggml_mr.h); gg_gemm calls it ahead of the stock launch.
 *
 * ---- why the stock decode band's cost grows with M (r4d_ds41_ggml.hip, gg_gemm's M <= 48 band)
 *
 *  (a) A pass takes at most kGgXM = 8 activation rows. Each block walks its weight row group once a pass, so
 *      above 8 rows the group is re-read (from L2) and re-dequantised ceil(M / 8) times: at M = 24 three times,
 *      at 48 six.
 *  (b) The activation rows are staged as f32 in a 36 KiB buffer (gg_xbuf, 9216 floats with a pad of 4 every
 *      32). A whole row of K = 5120 takes 5760 floats: one row fits (K 4096: 2, 2048: 4, 1280: 6, 768: 10). A
 *      pass with more rows than fit runs "chunked": K in chunks of 1024 values (2048 at 3-4 rows), restaged
 *      for EVERY row group (8 or 16 weight rows), each chunk between two block barriers around ONE unit a lane
 *      (a chunk is 32 units, lane = unit mod 32). The chunk's weight loads are issued only after its barrier,
 *      so every chunk of every row group exposes a DRAM round trip, and mc x KC activations are re-read from
 *      L2 and widened a row group. That is the jump from M = 1 to M = 2-4 at K >= 4096 and the slope to 8.
 *  (c) Inside a pass the decode IS shared by the pass's rows (gg_unit dequantises a unit's group once, then
 *      loops the rows), but each row costs a mul + an add a weight value (no FMA: -ffp-contract=off), two LDS
 *      float4 reads per 8 values, and a 5-level butterfly an output. At K = 1280 (attn_q_b, 40 units a row)
 *      lane = unit mod 32 gives lanes 0-7 two units and the other 24 one (62.5 % of the lanes busy), at K = 768
 *      (shexp_down, 24 units) 75 %, at 512 50 %.
 *  So the stock band neither keeps one weight read for all rows (a, b) nor keeps its lanes busy at short K (c).
 *
 * ---- the form (gg_mr_kernel<T, H, R, MT>)
 *
 *  - A wave holds H weight rows side by side, 32 / H lanes (LPR) a row, and each lane R rows of its own (R
 *    consecutive rows dequantised from the same staged activations). A "stage" is LPR units of every row: lane
 *    i of a row slot takes unit U = s * LPR + i. With H = 4 a stage is one 256-value block of a K-quant row, and
 *    K = 1280 / 768 / 512 split evenly (5 / 3 / 2 units a lane).
 *  - The block (WN waves, WN * H * R weight rows) stages MT activation rows (its slice of M; M > MT splits M
 *    over grid.y) for SS stages at a time as f32 in LDS, in the lanes' read order ([stage][row][group][half]
 *    [lane][4]: a lane's eight values of a group are two float4 at consecutive lanes' addresses, no bank
 *    conflicts, and the H row slots read the same addresses). Two barriers a slab, not a chunk a row group.
 *  - A lane dequantises its unit of each of its R rows ONCE a stage (all four groups into registers), issues
 *    the loads of the stage PD ahead into the raw words it just freed (PD buffers, the stage loop unrolled PD
 *    times: 1 to 4 stages in flight, more when MT x R makes a stage's arithmetic short), then applies the 32
 *    values to all MT rows (2 LDS float4 a group a row, 8 R mul + 8 R add). Weight loads stay in flight across
 *    the slabs' barriers.
 *  - r4d_ds41_ggml_mr64.hip compiles the same kernels wave64 (bench pin only): the per-lane code is the same, the
 *    question is whether the dual ALUs run a wave64 mul / add in one pass (wave32 gets about a quarter of its
 *    VALU slots paired as VOPD).
 *  - The weight words are the bytes Gg<T>::load reads, in fewer, wider loads (mr_load: 8-byte loads at 2-byte
 *    alignment, as the stock IQ load_w forms do), so the dequantised values are the same.
 *
 * ---- exactness: each output (row n, activation row m) is the same sequence of operations as gg_gemv_kernel's
 *
 *  Stock: lane p (0..31) of the wave owning row n sums, from acc = 0.0f, its units U = p, p + 32, p + 64, ...
 *  (gg_row: U = U0 + lane, step 32, the UF units in flight accumulated in ascending V; K chunks in ascending
 *  order; the bits do not depend on M, XM, RPW, the grid or whether K was staged whole -- r4d_ds41_ggml.hip
 *  says so and the selftest checks it), each unit as four groups G = 0..3 of eight values t = 0..7:
 *  acc += x[k] * w[k], x the bf16 widened to f32 (exact), w = Gg<T>::dec<G> of the unit's words, the product
 *  rounded on its own (no contraction in either unit: CMakeLists compiles every unit -ffp-contract=off). Then
 *  r4d_wave_sum: v += xor16, then xor 8, 4, 2, 1 (DPP), and lane 0 rounds once (gg_put).
 *  Here: partial p of row n lives in the lane of n's row slot with i = p mod LPR, as its partial j = p / LPR.
 *  Stage s feeds unit U = s LPR + i, so p = U mod 32 = (s mod H) LPR + i: partial j = s mod H gets units
 *  U = p, p + 32, ... in ascending order (stages ascend), each unit by the same dec<G> and the same G, t order,
 *  from 0.0f, with the same x values. acc[r][m][0] is always the current stage's partial: after each stage the
 *  H partials rotate one slot (register moves, no arithmetic), and the stage count is padded to a multiple of H
 *  with rotations only, so slot j ends holding partial j. The butterfly: its first log2(H) levels (xor 16 down
 *  to xor LPR) pair partials that sit in one lane (p and p + X are partials j and j + X / LPR of lane i), the
 *  rest (xor LPR / 2 .. 1) are r4d_dpp_xor_add inside the row slot's LPR lanes (DPP row_xmask stays inside a
 *  row of 16, so slots never mix). Each level adds the same two values as the stock level does, the lower
 *  partial's value first, so the sum is the same IEEE operation (an IEEE add is commutative, every bit, for
 *  any non-NaN operands; only two NaN payloads could depend on the order). H = 1 calls r4d_wave_sum itself.
 *  Lane i = 0 of the slot rounds once with gg_put. Activation rows past M are staged as zeros and never
 *  stored; weight rows past N read row N - 1 and are never stored; units past the row (a partial last stage)
 *  are skipped by their lanes, as gg_row's V < U1 test skips them.
 *
 *  r4d_selftest --mrcheck byte-compares this form against the stock path at every DeepSeek-V4.1 dense TP4
 *  shape, q5_K / q6_K / q8_0, M = 1..48; tests/r4d_emu's ds41mr runs both kernels on the host emulator.
 */
#define R4D_GG_BLOCKS_ONLY 1
#include "r4d_ds41_ggml.hip"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kMrThreads = 256;          /* WN <= 8 waves a block */
constexpr int kMrLdsMax = 49152;         /* one stage of MT rows at most */
constexpr int kMrMaxM = 48;              /* = kGgDecodeRows */

struct GgMrArgs {
    const uint16_t* x;
    long long x_ld;                      /* elements */
    const uint8_t* w;
    long long w_ld;                      /* bytes a weight row */
    void* y;
    long long y_ld;                      /* elements */
    int M, N, K;
    int UK;                              /* units a row */
    int NS;                              /* stages a row (the last may be partial) */
    int SS;                              /* stages a slab */
    int x_al;
};

/* ------------------------------------------------------------------ weight words
 * Gg<T>::load's dwords, the same bytes in the same order, by 8-byte loads at the blocks' 2-byte (q6_K, q8_0) or
 * 4-byte (q5_K) alignment: the device runs with unaligned access enabled, as Gg<IQ2_XXS>::load_w relies on. */
__device__ __forceinline__ void mr_b64(const uint8_t* p, uint32_t* r) {
    const gg_u32x2 v = *(gg_gu64a2)(const void*)p;
    r[0] = v.x;
    r[1] = v.y;
}
template <int T>
__device__ __forceinline__ void mr_load(const uint8_t* b, int u, uint32_t* r) {
    if constexpr (T == GG_Q8_0) {
        r[0] = gg_u16(b);
        mr_b64(b + 2, r + 1);
        mr_b64(b + 10, r + 3);
        mr_b64(b + 18, r + 5);
        mr_b64(b + 26, r + 7);
    } else if constexpr (T == GG_Q6_K) {
        const int h = u >> 2, l0 = 8 * (u & 3);
        mr_b64(b + 64 * h + l0, r);
        mr_b64(b + 64 * h + 32 + l0, r + 2);
        mr_b64(b + 128 + 32 * h + l0, r + 4);
        mr_b64(b + 192 + 8 * h, r + 6);
        r[8] = gg_u16(b + 208);
    } else if constexpr (T == GG_Q5_K) {
        const int jp = u >> 2, l0 = 8 * (u & 3);
        mr_b64(b, r);
        mr_b64(b + 8, r + 2);
        mr_b64(b + 16 + l0, r + 4);
        mr_b64(b + 48 + 64 * jp + l0, r + 6);
        mr_b64(b + 80 + 64 * jp + l0, r + 8);
    } else {
        Gg<T>::load(b, u, r);
    }
}

/* The unit U and group G whose eight values start at K position k (a multiple of 8): Gg<T>::goff inverted. */
template <int T>
__device__ __forceinline__ void mr_piece(int k, int& U, int& G) {
    if constexpr (Gg<T>::QK == 32) {                  /* q8_0, iq4_nl, the float form: goff = 8 g */
        U = k >> 5;
        G = (k >> 3) & 3;
    } else if constexpr (T == GG_Q5_K || T == GG_Q6_K) {
        /* goff = 128 (u >> 2) + 32 g + 8 (u & 3) (Q5_K's 64 (g >> 1) + 32 (g & 1) is 32 g) */
        const int q = (k >> 3) & 31;
        U = 8 * (k >> 8) + 4 * (q >> 4) + (q & 3);
        G = (q >> 2) & 3;
    } else {                                          /* the I-quants: goff = 32 u + 8 g */
        const int q = (k >> 3) & 31;
        U = 8 * (k >> 8) + (q >> 2);
        G = q & 3;
    }
}

/* Stages [s0, s1) of activation rows m0 .. m0 + MT - 1 (rows from mc on, and values past K, as zeros) into the
 * slab: [stage][row][group][half][lane][4] f32, VPS floats a stage a row. */
template <int T, int H, int MT>
__device__ __forceinline__ void mr_stage(const GgMrArgs& A, float* xs, int m0, int mc, int s0, int s1) {
    constexpr int LPR = 32 / H, VPS = 32 * LPR, PPS = VPS / 8;
    const int np = (s1 - s0) * MT * PPS;
    for (int idx = (int)threadIdx.x; idx < np; idx += (int)blockDim.x) {
        const int q = idx % PPS, rest = idx / PPS;
        const int m = rest % MT, ss = rest / MT;
        const int s = s0 + ss, k = s * VPS + 8 * q;
        float v[8];
        if (m < mc && k < A.K) {
            gg_x8(A.x + (long long)(m0 + m) * A.x_ld + k, A.x_al != 0, v);
        } else {
#pragma unroll
            for (int t = 0; t < 8; ++t) v[t] = 0.0f;
        }
        int U = 0, G = 0;
        mr_piece<T>(k, U, G);
        const int i = U - s * LPR;
        float* d = xs + (ss * MT + m) * VPS;
        *(float4*)(d + ((2 * G) * LPR + i) * 4) = make_float4(v[0], v[1], v[2], v[3]);
        *(float4*)(d + ((2 * G + 1) * LPR + i) * 4) = make_float4(v[4], v[5], v[6], v[7]);
    }
}

/* The butterfly of one output over its H partials in this lane (slot j = partial j, p = j LPR + i) and the row
 * slot's LPR lanes: r4d_wave_sum's levels xor 16, 8, 4, 2, 1 with the same operands. */
template <int H>
__device__ __forceinline__ float mr_reduce(const float (&a)[H]) {
    if constexpr (H == 1) {
        return r4d_wave_sum(a[0]);
    } else {
        constexpr int LPR = 32 / H;
        float v[H];
#pragma unroll
        for (int j = 0; j < H; ++j) v[j] = a[j];
        /* in-lane levels: xor X pairs partial j with j + X / LPR (the lower p first, as the stock lane p keeps
         * its own value first); after them v[0] is the stock's level-(xor LPR) value of lane p = i */
#pragma unroll
        for (int X = 16; X >= LPR; X >>= 1) {
            const int D = X / LPR;
#pragma unroll
            for (int j = 0; j < D; ++j) v[j] = v[j] + v[j + D];
        }
        float r = v[0];
        if constexpr (LPR >= 16) r = r4d_dpp_xor_add<0x168>(r);
        if constexpr (LPR >= 8) r = r4d_dpp_xor_add<0x164>(r);
        if constexpr (LPR >= 4) r = r4d_dpp_xor_add<0x162>(r);
        if constexpr (LPR >= 2) r = r4d_dpp_xor_add<0x161>(r);
        return r;
    }
}

/* Stages of weight words in flight ahead of the one computed (the stage loop is unrolled this many times, a
 * buffer a stage, so no word moves between registers while its load is pending): enough stages of arithmetic to
 * cover a DRAM round trip (~2 us) at a few waves a SIMD, as MT x R shrinks the arithmetic a stage. */
template <int MT, int R>
constexpr int mr_pd() { return MT * R >= 24 ? 1 : (MT * R >= 8 ? 2 : 4); }

template <int T, int H, int R, int MT, typename OT>
__global__ __launch_bounds__(kMrThreads) void gg_mr_kernel(GgMrArgs A) {
    using Q = Gg<T>;
    static_assert(H == 1 || H == 2 || H == 4 || H == 8, "row slots a wave");
    constexpr int LPR = 32 / H, VPS = 32 * LPR, PD = mr_pd<MT, R>();
#ifdef R4D_EMU_HOST   /* tests/r4d_emu: the launch's dynamic LDS (no extern __shared__ on the host) */
    float* const xs = (float*)emu_dyn_lds();
#else
    extern __shared__ __attribute__((aligned(16))) float mr_xs[];
    float* const xs = mr_xs;
#endif
    const int lane = (int)threadIdx.x & 31, wave = (int)threadIdx.x >> 5, nw = (int)blockDim.x >> 5;
    const int g = lane / LPR, i = lane - g * LPR;
    const int nl = ((int)blockIdx.x * nw + wave) * (H * R) + g * R;   /* this lane's first weight row */
    const int m0 = (int)blockIdx.y * MT, mc = min(MT, A.M - m0);
    const uint8_t* wr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wr[r] = A.w + (long long)min(nl + r, A.N - 1) * A.w_ld;

    float acc[R][MT][H];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int j = 0; j < H; ++j) acc[r][m][j] = 0.0f;

    /* stage s's words live in raw[s % PD], loaded PD stages ahead */
    uint32_t raw[PD][R][Q::NR];
    auto load = [&](int s, uint32_t (&rw)[R][Q::NR]) {
        const int U = s * LPR + i;
        if (s < A.NS && U < A.UK) {
            const int b = U / Q::UPB;
#pragma unroll
            for (int r = 0; r < R; ++r) mr_load<T>(wr[r] + (long long)b * Q::BB, U - b * Q::UPB, rw[r]);
        }
    };
#pragma unroll
    for (int k = 0; k < PD; ++k) load(k, raw[k]);

    auto rotate = [&]() {
        if constexpr (H > 1) {
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    const float t0 = acc[r][m][0];
#pragma unroll
                    for (int j = 0; j + 1 < H; ++j) acc[r][m][j] = acc[r][m][j + 1];
                    acc[r][m][H - 1] = t0;
                }
        }
    };
    /* one stage: dequantise this lane's unit of its R rows once (all four groups), put the stage PD ahead in
     * flight in the words just freed, apply the 32 values to the MT staged rows, rotate the partials */
    auto stage = [&](int s, const float* xb, uint32_t (&rw)[R][Q::NR]) {
        const int U = s * LPR + i;
        const bool act = U < A.UK;
        const int u = U - (U / Q::UPB) * Q::UPB;
        float w[R][32];
        if (act) {
#pragma unroll
            for (int r = 0; r < R; ++r)
                gg_for<4>([&](auto gi) {
                    constexpr int G = decltype(gi)::value;
                    Q::template dec<G>(rw[r], u, nullptr, &w[r][8 * G]);
                });
        }
        load(s + PD, rw);
        if (act) {
#pragma unroll
            for (int G = 0; G < 4; ++G) {
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    const float* xp = xb + m * VPS;
                    const float4 a = *(const float4*)(xp + ((2 * G) * LPR + i) * 4);
                    const float4 c = *(const float4*)(xp + ((2 * G + 1) * LPR + i) * 4);
                    const float xv[8] = { a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w };
#pragma unroll
                    for (int r = 0; r < R; ++r)
#pragma unroll
                        for (int t = 0; t < 8; ++t) acc[r][m][0] += xv[t] * w[r][8 * G + t];
                }
            }
        }
        rotate();
    };

    for (int s0 = 0; s0 < A.NS; s0 += A.SS) {
        const int s1 = min(s0 + A.SS, A.NS);
        lds_barrier();
        mr_stage<T, H, MT>(A, xs, m0, mc, s0, s1);
        lds_barrier();
        /* groups of PD stages from a multiple of PD, so stage s is raw[s % PD] whatever the slab's start */
#pragma unroll 1
        for (int sb = s0 - s0 % PD; sb < s1; sb += PD) {
#pragma unroll
            for (int k = 0; k < PD; ++k)
                if (sb + k >= s0 && sb + k < s1) stage(sb + k, xs + (sb + k - s0) * MT * VPS, raw[k]);
        }
    }
    if constexpr (H > 1) {
        /* the stage count padded to a multiple of H: rotations only, so slot j holds partial j */
        const int pad = (H - A.NS % H) % H;
#pragma unroll 1
        for (int e = 0; e < pad; ++e) rotate();
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int n = nl + r;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const float v = mr_reduce<H>(acc[r][m]);
            if (i == 0 && m < mc && n < A.N) gg_put((OT*)A.y + (long long)(m0 + m) * A.y_ld + n, v);
        }
    }
}

/* ------------------------------------------------------------------ forms
 * (H, R, MT): H weight rows a wave side by side, R a lane, MT activation rows a block. MT is the accumulator
 * count's third factor (R x MT x H floats a lane, <= 96) and one stage of MT rows must fit kMrLdsMax. */
#define MR_FORMS(X)                                                                                        \
    X(1, 1, 1) X(1, 1, 2) X(1, 1, 3) X(1, 1, 4) X(1, 1, 6) X(1, 1, 8) X(1, 1, 12)                           \
    X(2, 1, 1) X(2, 1, 2) X(2, 1, 3) X(2, 1, 4) X(2, 1, 6) X(2, 1, 8) X(2, 1, 12) X(2, 1, 16) X(2, 1, 24)   \
    X(2, 2, 1) X(2, 2, 2) X(2, 2, 3) X(2, 2, 4) X(2, 2, 6) X(2, 2, 8) X(2, 2, 12) X(2, 2, 16) X(2, 2, 24)   \
    X(4, 1, 1) X(4, 1, 2) X(4, 1, 3) X(4, 1, 4) X(4, 1, 6) X(4, 1, 8) X(4, 1, 12) X(4, 1, 16) X(4, 1, 24)   \
    X(4, 2, 1) X(4, 2, 2) X(4, 2, 3) X(4, 2, 4) X(4, 2, 6) X(4, 2, 8) X(4, 2, 12)

/* The MT values a (H, R) pair has, ascending; their count. */
[[maybe_unused]] int mr_mts(int H, int R, int* out) {
    int n = 0;
#define MR_MT(h, r, mt) if (H == h && R == r) out[n++] = mt;
    MR_FORMS(MR_MT)
#undef MR_MT
    return n;
}

template <int T, typename OT>
int mr_launch_t(RadStream s, int H, int R, int MT, const GgMrArgs& A, dim3 grid, unsigned thr, unsigned lds) {
#define MR_CASE(h, r, mt)                                                                                  \
    if (H == h && R == r && MT == mt) return R4D_EMIT(s, (gg_mr_kernel<T, h, r, mt, OT>), grid, dim3(thr), lds, A);
    MR_FORMS(MR_CASE)
#undef MR_CASE
    return R4D_GG_MR_DECLINE;
}

int mr_launch(RadStream s, int ty, bool f32out, int H, int R, int MT, const GgMrArgs& A, dim3 grid, unsigned thr,
              unsigned lds) {
#ifdef R4D_EMU_HOST   /* tests/r4d_emu: f32 outputs at every type (a sharper order check than bf16's) */
    if (f32out) {
        switch (ty) {
        case GG_Q5_K: return mr_launch_t<GG_Q5_K, float>(s, H, R, MT, A, grid, thr, lds);
        case GG_Q6_K: return mr_launch_t<GG_Q6_K, float>(s, H, R, MT, A, grid, thr, lds);
        case GG_Q8_0: return mr_launch_t<GG_Q8_0, float>(s, H, R, MT, A, grid, thr, lds);
        default: return R4D_GG_MR_DECLINE;
        }
    }
#endif
    if (f32out) return ty == GG_Q6_K ? mr_launch_t<GG_Q6_K, float>(s, H, R, MT, A, grid, thr, lds) : R4D_GG_MR_DECLINE;
    switch (ty) {
    case GG_Q5_K: return mr_launch_t<GG_Q5_K, uint16_t>(s, H, R, MT, A, grid, thr, lds);
    case GG_Q6_K: return mr_launch_t<GG_Q6_K, uint16_t>(s, H, R, MT, A, grid, thr, lds);
    case GG_Q8_0: return mr_launch_t<GG_Q8_0, uint16_t>(s, H, R, MT, A, grid, thr, lds);
    default: return R4D_GG_MR_DECLINE;
    }
}

[[maybe_unused]] int mr_cu_count() {
#ifdef R4D_EMU_HOST
    return 64;
#else
    static int cache[64];
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) return 64;
    if (cache[dev] <= 0) {
        int v = 0;
        if (hipDeviceGetAttribute(&v, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess || v <= 0) v = 64;
        cache[dev] = v;
    }
    return cache[dev];
#endif
}

#ifndef R4D_GG_MR_W64
/* ------------------------------------------------------------------ the switch, the pin, the rule */
/* The multi-row form runs at the measured table's shapes (r4d_ds41_gg_mr_forms.h) and declines to the stock
 * kernel everywhere else. Every form has the stock kernel's per-output arithmetic, so the bytes are the same
 * whichever runs. r4d_gg_mr_set(0) (r4d_selftest's stock reference) runs the stock kernel throughout. */
int g_mr_on = 1;                          /* the table's forms; r4d_gg_mr_set(0) the stock kernel */
int g_mr_pin_h = 0, g_mr_pin_r = 0, g_mr_pin_mt = 0, g_mr_pin_wn = 0;   /* 0: no pin; h -1: the rule, forced */
int g_mr_pin_w64 = 0;                     /* the wave64 unit's kernels (r4d_ds41_ggml_mr64.hip) */
char g_mr_last[160];                      /* the last launch's form, for the bench's lines */

struct MrForm {
    int H, R, MT, WN, SS, gx, gy, w64;
    unsigned lds;
};

/* The form for a call: the pin's (H, R, MT cap, WN, wave64), else the rule. MT is the smallest of the pair's
 * values that holds ceil(M / tiles) rows, tiles = ceil(M / MT cap); WN halves from 8 while the grid has under two
 * blocks a CU (wave64: at least 2, a whole wave); SS fills the slab budget (16 waves a CU's 64 KiB), whole groups
 * of the kernel's prefetch depth where more than one fits, at least one stage. false: not taken. */
/* The measured forms (r4d_ds41_gg_mr_forms.h, generated by tools/gg_mr_forms.py from r4d_selftest --perfmr's
 * MRPERF lines): an entry names (type, output f32 or bf16, N, K, an M band) and the wave32 form (H, R, MT cap, WN;
 * 0 = the rule's) that beat the STOCK kernel there with the stock kernel's bytes. With the form switched on
 * only a call with an entry takes the multi-row form; every other call declines to the stock kernel, and the
 * form's own rule, which would run at every shape, is reached only through a pin. Every form has the same per-output arithmetic (the exactness argument above), so an entry moves speed only. */
struct MrTab {
    int ty, f32, N, K, m_lo, m_hi;
    int H, R, MT, WN;
};
#include "r4d_ds41_gg_mr_forms.h"

bool mr_tab(const R4dGgMrCall& c, int* H, int* R, int* cap, int* WN) {
    for (const MrTab& t : kMrForms)
        if (t.H > 0 && t.ty == c.ty && t.f32 == (c.f32out ? 1 : 0) && t.N == c.N && t.K == c.K && c.M >= t.m_lo &&
            c.M <= t.m_hi) {
            *H = t.H; *R = t.R > 0 ? t.R : 1; *cap = t.MT; *WN = t.WN;
            return true;
        }
    return false;
}

bool mr_pick(const R4dGgMrCall& c, MrForm* f) {
    int H = 4, R = c.N >= 4096 ? 2 : 1, cap = 0, WN = 0, w64 = 0;
    /* The switch (no pin): the table's form, or decline to the stock kernel. A pin, and the bench's forced rule
     * (h -1), keep their own form. */
    if (g_mr_pin_h == 0 && !mr_tab(c, &H, &R, &cap, &WN)) return false;
    if (g_mr_pin_h > 0) {
        H = g_mr_pin_h;
        R = g_mr_pin_r > 0 ? g_mr_pin_r : 1;
        cap = g_mr_pin_mt;
        WN = g_mr_pin_wn;
        w64 = g_mr_pin_w64;
    }
    int mts[16];
    const int nmt = mr_mts(H, R, mts);
    if (nmt == 0) return false;
    if (cap <= 0 || cap > mts[nmt - 1]) cap = mts[nmt - 1];
    const int tiles = (c.M + cap - 1) / cap, need = (c.M + tiles - 1) / tiles;
    int MT = 0;
    for (int k = 0; k < nmt; ++k)
        if (mts[k] >= need && mts[k] <= cap) { MT = mts[k]; break; }
    if (MT == 0) return false;
    const int LPR = 32 / H, VPS = 32 * LPR, PD = MT * R >= 24 ? 1 : (MT * R >= 8 ? 2 : 4);   /* = mr_pd<MT, R>() */
    const long long sb = (long long)MT * VPS * 4;
    if (sb > kMrLdsMax) return false;
    const int gy = (c.M + MT - 1) / MT;
    const long long rw = (long long)H * R, nwv = (c.N + rw - 1) / rw;
    const int wmin = w64 ? 2 : 1;
    if (WN <= 0) {
        const long long want = 2LL * mr_cu_count();
        WN = 8;
        while (WN > wmin && ((nwv + WN - 1) / WN) * gy < want) WN >>= 1;
    }
    if (WN != 1 && WN != 2 && WN != 4 && WN != 8) return false;
    if (WN < wmin) return false;
    const long long gx = (nwv + WN - 1) / WN;
    if (gx > 0x7fffffffLL || gy > 65535) return false;
    const int Q = c.ty == GG_Q8_0 ? 32 : 256, UPB = c.ty == GG_Q8_0 ? 1 : 8;
    const int UK = c.K / Q * UPB, NS = (UK + LPR - 1) / LPR;
    const long long budget = 4096LL * WN;     /* 64 KiB a CU over 16 waves */
    long long SS = budget / sb;
    if (SS > PD) SS = SS / PD * PD;           /* whole groups of the kernel's unrolled stages */
    if (SS < 1) SS = 1;
    if (SS >= NS) SS = NS;                    /* one slab */
    if (SS * sb > 65536) return false;
    f->H = H; f->R = R; f->MT = MT; f->WN = WN; f->SS = (int)SS; f->gx = (int)gx; f->gy = gy; f->w64 = w64;
    f->lds = (unsigned)(SS * sb);
    return true;
}

}  // namespace

#ifndef R4D_EMU_HOST
/* r4d_ds41_ggml_mr64.hip: the same kernels compiled wave64 (two of this unit's 32-lane waves a hardware wave) */
int r4d_gg_mr_launch_w64(RadStream s, int ty, bool f32out, int H, int R, int MT, const void* args, unsigned gx,
                         unsigned gy, unsigned thr, unsigned lds);
#endif

int r4d_gg_mr_on() { return g_mr_pin_h != 0 ? 1 : g_mr_on; }

extern "C" void r4d_gg_mr_set(int on) { g_mr_on = on ? 1 : 0; }

int r4d_gg_mr_gemm(RadStream s, const R4dGgMrCall& c) {
    if (c.ty != GG_Q5_K && c.ty != GG_Q6_K && c.ty != GG_Q8_0) return R4D_GG_MR_DECLINE;
    if (c.M < 1 || c.M > kMrMaxM || c.N < 1 || c.K < 1) return R4D_GG_MR_DECLINE;
    if (c.ty == GG_Q5_K && ((uintptr_t)c.w & 3u)) return R4D_GG_MR_DECLINE;
    MrForm f{};
    if (!mr_pick(c, &f)) return R4D_GG_MR_DECLINE;
    GgMrArgs A{};
    A.x = c.x; A.x_ld = c.x_ld; A.w = c.w; A.w_ld = c.w_ld; A.y = c.y; A.y_ld = c.y_ld;
    A.M = c.M; A.N = c.N; A.K = c.K;
    const int Q = c.ty == GG_Q8_0 ? 32 : 256, UPB = c.ty == GG_Q8_0 ? 1 : 8;
    A.UK = c.K / Q * UPB;
    A.NS = (A.UK + 32 / f.H - 1) / (32 / f.H);
    A.SS = f.SS;
    A.x_al = c.x_al;
    std::snprintf(g_mr_last, sizeof g_mr_last, "H%d R%d MT%d WN%d SS%d grid %dx%d lds %u%s", f.H, f.R, f.MT, f.WN,
                  f.SS, f.gx, f.gy, f.lds, f.w64 ? " w64" : "");
#ifndef R4D_EMU_HOST
    if (f.w64)
        return r4d_gg_mr_launch_w64(s, c.ty, c.f32out, f.H, f.R, f.MT, &A, (unsigned)f.gx, (unsigned)f.gy, 32u * f.WN,
                                    f.lds);
#endif
    return mr_launch(s, c.ty, c.f32out, f.H, f.R, f.MT, A, dim3((unsigned)f.gx, (unsigned)f.gy), 32u * f.WN, f.lds);
}

/* r4d_gg_mr_pin (bench / test hook, r4d_selftest --mrcheck / --perfmr): h 0 clears the pin (r4d_gg_mr_set
 * decides: the table's shapes unless it was set to 0), -1 forces the rule, 1 / 2 / 4 pins H with R rows a lane (1 or 2), MT cap `mt` (0: the
 * pair's largest) and WN waves a block (0: the rule's). r4d_gg_mr_pin_w64(1): a pinned form runs the wave64
 * unit's kernels (the same source, so the same bits; on the emulator the wave32 ones). */
extern "C" void r4d_gg_mr_pin(int h, int r, int mt, int wn) {
    g_mr_pin_h = (h == -1 || h == 1 || h == 2 || h == 4 || h == 8) ? h : 0;
    g_mr_pin_r = r;
    g_mr_pin_mt = mt;
    g_mr_pin_wn = wn;
}
extern "C" void r4d_gg_mr_pin_w64(int on) { g_mr_pin_w64 = on ? 1 : 0; }

/* The last launch's form ("H4 R2 MT8 WN8 SS4 grid 256x1 lds 32768"), for the bench's lines. */
extern "C" const char* r4d_gg_mr_last() { return g_mr_last; }

#else  /* R4D_GG_MR_W64: r4d_ds41_ggml_mr64.hip, compiled -mwavefrontsize64 */

}  // namespace

int r4d_gg_mr_launch_w64(RadStream s, int ty, bool f32out, int H, int R, int MT, const void* args, unsigned gx,
                         unsigned gy, unsigned thr, unsigned lds) {
    return mr_launch(s, ty, f32out, H, R, MT, *(const GgMrArgs*)args, dim3(gx, gy), thr, lds);
}

#endif  /* R4D_GG_MR_W64 */

#if defined(R4D_EMU_HOST) && !defined(R4D_GG_MR_W64)
/* tests/r4d_emu (ds41mr): the stock decode band's launch as gg_gemm issues it (gg_xbuf's staging, rows a wave
 * `rpw`), from this unit's copy of gg_gemv_kernel -- the same source as the plugin's. The grid is capped at 256
 * where gg_gemm launches a block per row group; the rows are independent, so the bits are the same. */
long long mr_emu_xbuf(int xm, long long K, int* xlm, int* xr, int* kc) {
    *xr = (int)(K + K / 8);
    const long long f = 9216 / (K + K / 8);
    *xlm = (int)(f < xm ? f : xm);
    long long cc = 9216 / ((long long)xm * 1152);
    if (cc < 1) cc = 1;
    const long long kr = (K + 1023) / 1024 * 1024;
    *kc = (int)(1024LL * cc < kr ? 1024LL * cc : kr);
    const long long whole = (long long)*xlm * *xr, chunk = (long long)xm * (*kc + *kc / 8);
    return *xlm >= xm ? whole : (whole > chunk ? whole : chunk);
}
template <int T, int RPW, typename OT>
int mr_emu_stock_t(const R4dGgMrCall& c, unsigned grid, unsigned lds, const GgGemvArgs& A) {
    return R4D_EMIT(nullptr, (gg_gemv_kernel<T, RPW, false, OT>), dim3(grid), dim3(kGgThreads), lds, A);
}
int r4d_gg_mr_emu_stock(const R4dGgMrCall& c, int rpw) {
    GgGemvArgs A{};
    A.x = c.x; A.x_ld = c.x_ld;
    A.w = c.w; A.wtab = nullptr; A.w_ld = c.w_ld; A.w_es = 0;
    A.y = c.y; A.y_ld = c.y_ld;
    A.M = c.M; A.N = c.N; A.K = c.K;
    A.XM = c.M < kGgXM ? c.M : kGgXM;
    const long long xbuf = mr_emu_xbuf(A.XM, c.K, &A.XLM, &A.XR, &A.KC);
    const long long nrg = (c.N + 8 * rpw - 1) / (8 * rpw);
    A.rgpi = 1; A.ipe = (int)nrg;
    A.x_al = c.x_al;
    A.live_off = 0; A.x_off = 0;
    const unsigned grid = (unsigned)(nrg < 256 ? nrg : 256), lds = (unsigned)(xbuf * 4);
#define MR_EMU_ST(T) \
    (rpw == 2 ? (c.f32out ? mr_emu_stock_t<T, 2, float>(c, grid, lds, A) : mr_emu_stock_t<T, 2, uint16_t>(c, grid, lds, A)) \
              : (c.f32out ? mr_emu_stock_t<T, 1, float>(c, grid, lds, A) : mr_emu_stock_t<T, 1, uint16_t>(c, grid, lds, A)))
    switch (c.ty) {
    case GG_Q5_K: return MR_EMU_ST(GG_Q5_K);
    case GG_Q6_K: return MR_EMU_ST(GG_Q6_K);
    case GG_Q8_0: return MR_EMU_ST(GG_Q8_0);
    default: return -1;
    }
#undef MR_EMU_ST
}
#endif
