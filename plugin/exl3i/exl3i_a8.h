/* exl3i_a8.h -- exl3i's A8 prefill (the two-pass int8 form, EXL3-V3 design option C): what its host side, its device
 * side and its host tests share. Plain C; the index maps and the quantiser's arithmetic are EXL3I_HD (host and device
 * compile the same lines, so the host tests check what the kernels compute).
 *
 * THE PATH. A weight in codebook mad whose activation mode (op parameter act, include/exl3_act.h) is a8t or
 * a8g<G> takes it at a launch of M >= EXL3I_A8_MIN_ROWS real rows; below, the f16 path (nt_pass) as for act f16.
 * One prefill step, per weight:
 *   1. actq (exl3i_a8_prep.hip): each row's x scaled by suh (suh.1 for a stacked gate_up's up half, a second plane)
 *      and rotated (rotate128_math: the f16 path's rotated values, bit for bit), then int8 with one f32 scale a
 *      (row, G inputs): aq [mats][M][K] int8, sa [mats][M][K/128] f32 (a scale repeated over the G/128 blocks of its
 *      group, so the GEMM folds one scale a 128-K block whatever G is);
 *   2. per column chunk of the weight (EXL3I_A8_CHUNK_BYTES of plane at most; one chunk at every trunk shape):
 *      wdec8 (exl3i_a8_prep.hip): the trellis decoded once into the int8 plane radiance's i8a8 prefill tile reads
 *      (libr4d/r4d_fp8_frag.h's fragment order, exl3i_i8plane_off below), each value q = the mad value / WS;
 *   3. gemm (exl3i_a8_gemm.hip): radiance's i8a8 prefill tile (a copy, exl3i_8a8.hip.h) over aq and the plane:
 *      per 128-K block one FMA a slot with the row's activation scale (the biased-int fold), no weight scale in the
 *      loop; at the store the block's 128 output rows rotated back (the output Hadamard of EXL3's
 *      W = diag(suh) Hb W_hat Hb diag(svh)), times WS / sqrt(128), times svh.
 *
 * WS, the int8 grid's step: mad's value is fp16(q * 4 K_INV), K_INV = fp16 0x1EEE (common/exl3i_decode.h), so
 * q = rint(value / WS) exactly (|q| <= 127; checked for all 65536 states by tests/exl3i_twopass_host_test.c). */
#ifndef EXL3I_A8_H
#define EXL3I_A8_H

#include <math.h>
#include <stdint.h>

#ifdef __HIPCC__
#define EXL3I_A8_HD __host__ __device__ __forceinline__
#else
#define EXL3I_A8_HD static inline
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define EXL3I_A8_MIN_ROWS 512                 /* the A8 path's threshold: real rows, not the band's */
#define EXL3I_A8_CHUNK_BYTES (192LL << 20)    /* the int8 plane of one column chunk, at most (T2's chunk) */
#define EXL3I_A8_WS 0.027069091796875f        /* 4 K_INV = 4 x 0x1EEE = 1.732421875 x 2^-6, exact in f32 */
#define EXL3I_A8_ALIGN 256

/* Test pins move both (exl3i_pin.h: EXL3I_PIN_A8_MIN_ROWS, EXL3I_PIN_A8_CHUNK_MB, for a test's multi-chunk launch at
 * small shapes); nt.c reads them at every scratch-hook and launch call, so the two agree. */

/* The A8 arena: the op's scratch at a declared row count M (the scratch hook) and at a launch's real rows (the
 * launch lays it out again at its rows, which needs no more). One path runs a launch, so the f16 path's region
 * (below EXL3I_A8_MIN_ROWS rows) and the A8 pieces share the arena's start. */
typedef struct Exl3iA8Plan {
    int64_t M, N, K;
    int32_t mats;      /* activation planes: 2 for a stacked gate_up (suh, suh.1), else 1 */
    int32_t group;     /* inputs an activation scale: 128 .. K, dividing K */
    int64_t nkb;       /* K / 128 */
    int64_t chunk;     /* weight columns a chunk: a multiple of 128 (of 256 is not needed: the seam is per block) */
    int64_t aq_off, aq_bytes;   /* int8 [mats][M][K] */
    int64_t sa_off, sa_bytes;   /* f32 [mats][M][nkb] */
    int64_t w_off, w_bytes;     /* int8 plane of one chunk: [chunk / 16][K / 64][1024] */
    int64_t a8_bytes;           /* the A8 pieces' end (0 below the A8 threshold) */
    int64_t f16_bytes;          /* the f16 path's region at min(M, threshold - 1) rows: [rows][2K] f16 */
    int64_t total;              /* max of the two, what the scratch hook answers */
} Exl3iA8Plan;

static inline int64_t exl3i_a8_align(int64_t b) { return (b + EXL3I_A8_ALIGN - 1) / EXL3I_A8_ALIGN * EXL3I_A8_ALIGN; }

/* The inputs a scale of an activation mode (exl3_act.h kinds: 1 a8t, 2 a8g): a8t K, a8g its group. 0 for f16 or a
 * group that does not divide K (refused: a scale never straddles a row's end). */
static inline int64_t exl3i_a8_group(int kind, int group, int64_t K) {
    if (kind == 1) return K;
    if (kind == 2 && group >= 128 && group % 128 == 0 && K % group == 0) return group;
    return 0;
}

/* Lay the arena out for the A8 path from min_rows rows (EXL3I_A8_MIN_ROWS unless a test pins another). 0, or -1: bad geometry (K, N not positive multiples of 128, N % 256 with stacked, M < 1, a group
 * not 128 .. K dividing K, a chunk budget under one 128-column block). */
static inline int exl3i_a8_plan(int64_t M, int64_t N, int64_t K, int stacked, int64_t group, int64_t chunk_bytes,
                                int64_t min_rows, Exl3iA8Plan* p) {
    int64_t f16_rows, ch;
    if (!p || M < 1 || K < 128 || K % 128 || N < 128 || N % 128 || (stacked && N % 256) || min_rows < 1) return -1;
    if (group < 128 || group % 128 || K % group) return -1;
    if (chunk_bytes < 128 * K) return -1;
    p->M = M; p->N = N; p->K = K;
    p->mats = stacked ? 2 : 1;
    p->group = (int32_t)group;
    p->nkb = K / 128;
    ch = chunk_bytes / K / 128 * 128;
    p->chunk = ch < N ? ch : N;
    f16_rows = M < min_rows ? M : min_rows - 1;
    p->f16_bytes = exl3i_a8_align(f16_rows * 2 * K * 2);   /* nt.c's region: [T][2K] f16 (both halves' rows) */
    p->aq_off = p->sa_off = p->w_off = p->aq_bytes = p->sa_bytes = p->w_bytes = p->a8_bytes = 0;
    if (M >= min_rows) {
        p->aq_off = 0;
        p->aq_bytes = exl3i_a8_align(p->mats * M * K);
        p->sa_off = p->aq_off + p->aq_bytes;
        p->sa_bytes = exl3i_a8_align(p->mats * M * p->nkb * 4);
        p->w_off = p->sa_off + p->sa_bytes;
        p->w_bytes = exl3i_a8_align(p->chunk * K);
        p->a8_bytes = p->w_off + p->w_bytes;
    }
    p->total = p->a8_bytes > p->f16_bytes ? p->a8_bytes : p->f16_bytes;
    return 0;
}

/* ---- the int8 plane: radiance's i8a8 fragment order (libr4d/r4d_fp8_frag.h, frag_src) ----
 * [N/16 row tiles][K/64 supertiles][32 lanes][32 bytes]: lane l's bytes 8 t + j are W[16 rt + (l & 15)]
 * [64 ks + 16 t + 8 (l >> 4) + j]. The byte offset of W[n][k], n chunk-local. */
EXL3I_A8_HD int64_t exl3i_i8plane_off(int64_t n, int64_t k, int64_t K) {
    const int64_t rt = n >> 4, ks = k >> 6;
    const int kk = (int)(k & 63), t = kk >> 4, hh = (kk >> 3) & 1, j = kk & 7, lane = (int)(n & 15) + 16 * hh;
    return (rt * (K >> 6) + ks) * 1024 + lane * 32 + 8 * t + j;
}

/* ---- the f16 decode fragment (exl3i_rate.hip.h's RateInt / RateFrac decode, the T2 plane's unit) ----
 * Lane l's element e of a 16 x 16 tile is W_hat[k = 8 (e >> 2) + 4 (l >> 4) + (e & 3)][n = l & 15]. */
EXL3I_A8_HD int exl3i_frag16_k(int lane, int e) { return 8 * (e >> 2) + 4 * (lane >> 4) + (e & 3); }

/* T2's f16 plane (exl3i_nt.hip.h exl3i_wdec_kernel): [K/16][ntc chunk tiles][32 lanes] v8h. The half offset of
 * W[n][k], n chunk-local. */
EXL3I_A8_HD int64_t exl3i_f16plane_off(int64_t n, int64_t k, int64_t ntc) {
    const int64_t ks = k >> 4, nt = n >> 4;
    const int kk = (int)(k & 15), hh = (kk >> 2) & 1, e = 4 * (kk >> 3) + (kk & 3), lane = (int)(n & 15) + 16 * hh;
    return ((ks * ntc + nt) * 32 + lane) * 8 + e;
}

/* wdec8's lane exchange: a decode lane holds k {4h .. 4h + 3} (dword d0, its bytes e 0..3) and {8 + 4h .. 8 + 4h + 3}
 * (d1); plane lane l (same n) wants k {8h .. 8h + 7} as one 8-byte fragment. Each lane sends one dword to lane ^ 16:
 * h 0 sends d1, h 1 sends d0; the fragment is (d0, recv) at h 0 and (recv, d1) at h 1. */
EXL3I_A8_HD uint32_t exl3i_wdec8_send(int lane, uint32_t d0, uint32_t d1) { return (lane & 16) ? d0 : d1; }
EXL3I_A8_HD void exl3i_wdec8_frag(int lane, uint32_t d0, uint32_t d1, uint32_t recv, uint32_t* lo, uint32_t* hi) {
    *lo = (lane & 16) ? recv : d0;
    *hi = (lane & 16) ? d1 : recv;
}

/* ---- the activation quantiser: one (row, group), from the rotated f16 values as f32 ----
 * scale = amax / 127, q = rint(clamp(v x (127 / amax), +-127)); an all-zero group has scale 0 and q 0. IEEE f32,
 * no contraction (both sides: -ffp-contract=off on the host; on the device a multiply then a clamp). */
EXL3I_A8_HD float exl3i_a8_scale(float amax) { return amax / 127.0f; }
EXL3I_A8_HD float exl3i_a8_inv(float amax) { return amax > 0.0f ? 127.0f / amax : 0.0f; }
EXL3I_A8_HD int exl3i_a8_q(float v, float inv) {
    float q = v * inv;
    q = q < -127.0f ? -127.0f : q > 127.0f ? 127.0f : q;
    return (int)rintf(q);
}

/* A decoded int8-grid value (as f32) -> its int8 grid index (the test references and the plane's writers). */
EXL3I_A8_HD int exl3i_a8_wq(float v) { return (int)rintf(v * (1.0f / EXL3I_A8_WS)); }

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* EXL3I_A8_H */
