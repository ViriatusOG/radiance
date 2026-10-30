/* Derived from libr4d/r4d_fp8_layout.cpp by the libhalo rename transform (2026-10-26);
 * see libhalo/README.md. Op schemas, row names and stored weight layouts are
 * shared with libr4d by design. Everything else in this file is libhalo's own
 * and diverges as it is measured on gfx1151. */
/* halo_fp8_layout.cpp -- the stored form of libhalo's FP8 and int8 weights, and the relayouts that
 * produce it from the weight's canonical planes (spec.md §4.3). Pure host C++: nothing here
 * includes halo.h or <hip/hip_runtime.h>, so packing can be tested without ROCm.
 *
 * ================================================================= what they read
 *
 * The block-scaled linears read the encoding a block-FP8 checkpoint ships, `fp8_e4m3*bf16[128x128]`:
 *
 *   codes   F8_E4M3, [N][K]                       -- as safetensors stores `X.weight`
 *   scale   BF16,    [ceil(N/128)][ceil(K/128)]   -- as `X.weight_scale_inv`
 *
 * E.g. Qwen3.8-27B-FP8 `mlp.gate_proj.weight` is F8_E4M3 [17408, 5120] with a BF16 [136, 40]
 * scale. The kernels index `S[(row / 128) * sc_cols + (c / 128)]`, so the scale plane is stored
 * unchanged and only the codes move: into WMMA fragment order for the fp8a8 GEMMs (a lane reads
 * its own bytes from one coalesced load), and not at all for the fp8a16 matvec, which reads
 * sixteen consecutive k of one row. The two forms differ, so a weight is declared for one or the
 * other and the builder refuses the pair.
 *
 * The lm_head is a single row-sharded operand, so each row carries its block scales in its own
 * tail: [K codes][K/128 bf16 scales], padded to 16 bytes so a uint4 load of the next row is
 * aligned (halo_fp8_lm.h). hc_read's mixing matrices and mtp_enter's fc use the same form with an
 * f32 scale per group of the row (halo_hc_fp8.h).
 *
 * ================================================================= BF16 scales, not UE8M0
 *
 * Some block-FP8 checkpoints carry a UE8M0 exponent byte per tile instead of a bf16. Converting to
 * UE8M0 rounds each scale to a power of two, which changes the tile's gain and would mean
 * requantising every E4M3 byte. Decoding either costs one shift and one bitcast, and the bf16
 * plane is 0.02% of its weight.
 *
 * This file chooses no numbers: a checkpoint's planes are served as they are, and anything
 * quantised (e.g. a bf16 lm_head made fp8) was quantised before the container was written, by
 * libquant's `rtn` over `fp8_e4m3` with a 128x128 bf16 block. Its rule matches these kernels'
 * activation quantiser: amax/448 rounded to bf16, a zero block scaled by one.
 */

#include "halo_plugin.h"
#include "halo_fp8_frag.h"
#include "halo_fp8_lm.h"
#include "halo_hc_fp8.h"
#include "halo_enc.h"

#include <cmath>
#include <cstdint>
#include <cstring>

/* Operand indices of `gemm_nt_q`. The schema is positional, so these must follow its order in
 * halo_rows.cpp. */
enum { FL_A = 0, FL_ASCALE = 1, FL_B = 2, FL_BSCALE = 3, FL_Y = 4 };

/* K span of one fragment-order tile, compiled into libhalo. */
enum { HALO_FP8_SUPER_K = halo_fp8::kSuperK };

namespace {

/* Walks the fragment-order plane front to back, eight bytes at a time, passing each word's
 * row-major offset in the [N][K] weight; used in both directions. A fragment is eight consecutive
 * k of one row (halo_fp8_frag.h frag_src over j = 0..7), so it moves as one word. Placing each byte
 * separately would dominate the CPU cost of loading a block-FP8 model. */
template <typename Move>
void frag_walk(long long N, long long K, Move move) {
    constexpr int kLanes = halo_fp8::kTileBytes / halo_fp8::kLaneBytes;
    constexpr int kFrags = halo_fp8::kLaneBytes / 8;
    const long long ksuper = K / halo_fp8::kSuperK;
    int64_t at = 0;
    for (long long rt = 0; rt < N / halo_fp8::kFragRows; ++rt)
        for (long long ks = 0; ks < ksuper; ++ks)
            for (int lane = 0; lane < kLanes; ++lane)
                for (int t = 0; t < kFrags; ++t, at += 8) {
                    int64_t row, k;
                    halo_fp8::frag_src(rt, ks, lane, t, 0, &row, &k);
                    move(at, row * K + k);
                }
}

struct FGeom { long long N, K, sn, sk; };

int fgeom(const RadParam* p, int n_p, FGeom* g) {
    g->N = halo_pgeti_or(p, n_p, "N", 0);
    g->K = halo_pgeti_or(p, n_p, "K", 0);
    if (g->N <= 0 || g->K <= 0) return RAD_E_SHAPE;
    /* The block is compiled into the kernels as `>> 7`. The constraint table already refuses
     * other groups; this catches a caller that invokes the hook without selection. */
    if (halo_pgeti_or(p, n_p, "group", HALO_FP8_BLOCK) != HALO_FP8_BLOCK) return RAD_E_SHAPE;
    g->sn = (g->N + HALO_FP8_BLOCK - 1) / HALO_FP8_BLOCK;
    g->sk = (g->K + HALO_FP8_BLOCK - 1) / HALO_FP8_BLOCK;
    return RAD_OK;
}

void ffill(RadLayout* out, const char* tag, uint32_t dt, long long d0, long long d1,
           long long bytes) {
    std::memset(out, 0, sizeof(*out));
    out->tag = tag;
    out->dtype = dt;
    out->rank = 2;
    out->shape[0] = d0;
    out->shape[1] = d1;
    out->align = 16;   /* 16-byte loads; see the padding note in halo_fp8_lm.h */
    out->bytes = bytes;
}


/* `logits_gemm` names its geometry `n_vocab` and `n_embd` where the GEMMs say N and K. */
int lgeom(const RadParam* p, int n_p, FGeom* g) {
    g->N = halo_pgeti_or(p, n_p, "n_vocab", 0);
    g->K = halo_pgeti_or(p, n_p, "n_embd", 0);
    if (g->N <= 0 || g->K <= 0) return RAD_E_SHAPE;
    if (g->K % HALO_FP8_BLOCK) return RAD_E_SHAPE;
    g->sn = (g->N + HALO_FP8_BLOCK - 1) / HALO_FP8_BLOCK;
    g->sk = g->K / HALO_FP8_BLOCK;
    return RAD_OK;
}

/* Operand 1 is the weight; see oLogits in halo_rows.cpp. */
enum { LGL_X = 0, LGL_W = 1, LGL_Y = 2 };

/* Operand indices of hc_read -- see oHcRead in halo_rows.cpp. */
enum { HCL_DOWN = 2, HCL_UP = 3 };

struct HGeom { long long rows, K, G; };

int hgeom(const RadParam* p, int n_p, int operand, HGeom* g) {
    const long long n  = halo_pgeti_or(p, n_p, "n", 0);
    const long long hc = halo_pgeti_or(p, n_p, "hc", 0);
    const long long lr = halo_pgeti_or(p, n_p, "lowrank", 0);
    if (n <= 0 || hc <= 0 || lr <= 0) return RAD_E_SHAPE;
    if (operand == HCL_DOWN)    { g->rows = lr;     g->K = hc * n; g->G = halo_hc8::down_group(g->K); }
    else if (operand == HCL_UP) { g->rows = hc * n; g->K = lr;     g->G = halo_hc8::up_group(g->K); }
    else return RAD_E_UNSUPPORTED;
    if (!halo_hc8::group_ok(g->K, g->G)) return RAD_E_SHAPE;
    return RAD_OK;
}

enum { MTL_FCH = 4, MTL_FCE = 5 };   /* oMtpEnter in halo_rows.cpp */

int mgeom(const RadParam* p, int n_p, int operand, HGeom* g) {
    if (operand != MTL_FCH && operand != MTL_FCE) return RAD_E_UNSUPPORTED;
    const long long n = halo_pgeti_or(p, n_p, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    g->rows = n; g->K = n; g->G = halo_hc8::kDownGroup;
    if (!halo_hc8::group_ok(g->K, g->G)) return RAD_E_SHAPE;
    return RAD_OK;
}

}  /* namespace */

/* ==================================================================== the hooks
 *
 * Each checks the encoding its kernel reads and refuses any other with RAD_E_DTYPE (the
 * declare-time error for a mis-converted container). They move bits and choose none. */

namespace {

/* gemm_nt_q's block-fp8 weight: e4m3 codes with a bf16 scale per 128x128 block. */
bool fp8_block_enc(const RadEncoding* e) {
    return halo_enc::affine(e, RAD_F8E4M3, RAD_BF16, HALO_FP8_BLOCK, HALO_FP8_BLOCK);
}

int fp8_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                   const int* sel, const RadTensor* pl, int n, RadLayout* out, bool frag) {
    if (!out || !pl) return RAD_E_INVAL;
    FGeom g;
    const int rc = fgeom(p, n_p, &g);
    if (rc != RAD_OK) return rc;
    if (operand != FL_B && operand != FL_BSCALE) return RAD_E_UNSUPPORTED;
    if (!fp8_block_enc(enc)) return RAD_E_DTYPE;
    if (operand == FL_BSCALE) {
        if (!halo_enc::selects(enc, sel, n, "scale") || !halo_enc::dims(pl[0], g.sn, g.sk))
            return RAD_E_SHAPE;
        return RAD_E_UNSUPPORTED;                      /* the checkpoint's own bf16 plane */
    }
    if (!halo_enc::selects(enc, sel, n, "codes") || !halo_enc::dims(pl[0], g.N, g.K))
        return RAD_E_SHAPE;
    if (!frag) return RAD_E_UNSUPPORTED;               /* row-major, as the matvec reads it */
    if (g.K % HALO_FP8_SUPER_K || g.N % halo_fp8::kFragRows) return RAD_E_SHAPE;
    ffill(out, "r4d.fp8.w.frag64", RAD_F8E4M3, g.N, g.K, halo_fp8::frag_bytes(g.N, g.K));
    return RAD_OK;
}

}  /* namespace */

extern "C" int halo_rad_layout_fp8_block(const RadParam* p, int n_p, int operand,
                                            const RadEncoding* enc, const int* sel,
                                            const RadTensor* pl, int n, RadLayout* out) {
    return fp8_enc_layout(p, n_p, operand, enc, sel, pl, n, out, true);
}

extern "C" int halo_rad_layout_fp8_rowmajor(const RadParam* p, int n_p, int operand,
                                               const RadEncoding* enc, const int* sel,
                                               const RadTensor* pl, int n, RadLayout* out) {
    return fp8_enc_layout(p, n_p, operand, enc, sel, pl, n, out, false);
}

extern "C" int halo_rad_relayout_fp8_block(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const RadTensor* pl, int n, void* dst,
                                          int64_t dst_bytes) {
    RadLayout L;
    const int rc = fp8_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, true);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    const uint8_t* s = (const uint8_t*)pl[0].data;
    uint8_t* d = (uint8_t*)dst;
    frag_walk(pl[0].shape[0], pl[0].shape[1],
              [&](int64_t at, int64_t from) { std::memcpy(d + at, s + from, 8); });
    return RAD_OK;
}

extern "C" int halo_rad_unrelayout_fp8_block(const RadParam* p, int n_p, int operand,
                                            const RadEncoding* enc, const int* sel,
                                            const void* src, int64_t src_bytes,
                                            const RadTensor* pl, int n) {
    RadLayout L;
    const int rc = fp8_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, true);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* d = (uint8_t*)pl[0].data;
    frag_walk(pl[0].shape[0], pl[0].shape[1],
              [&](int64_t at, int64_t from) { std::memcpy(d + from, s + at, 8); });
    return RAD_OK;
}

/* ---- the int8 twin: i8 codes in the same fragment order as E4M3, with a bf16 scale per row per
 * 128 K.
 *
 * The scale plane is tile-major, [N/16][K/128][16]: the sixteen rows of a fragment tile side by
 * side for one 128-K block. On the narrow path a lane's eight accumulator rows are eight
 * consecutive rows of one tile, so their scales are one 16-byte load. A row-major plane needs
 * eight scattered loads, which if-converted into every k step cost the decode kernel 70% at
 * M = 4. The prefill tile reads one row's scale per lane and works with either form.
 *
 * `i8a8h` is the same planes holding a weight rotated by the unnormalised 128-wide Hadamard along
 * K, the 1/128 in its scale (libquant's `transform=fwht128`), to be met by had_quant_act_i8g's
 * activation. The bytes cannot tell the two apart, so each dtype accepts only its own transform
 * and the rotated codes carry a tag of their own. */
namespace {
bool i8_rotated(const RadParam* p, int n_p) {
    const char* d = halo_pgets(p, n_p, "dtype");
    return d && std::strcmp(d, "i8a8h") == 0;
}

int i8_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* pl, int n, RadLayout* out) {
    if (!out || !pl) return RAD_E_INVAL;
    FGeom g;
    const int rc = fgeom(p, n_p, &g);
    if (rc != RAD_OK) return rc;
    if (operand != FL_B && operand != FL_BSCALE) return RAD_E_UNSUPPORTED;
    const bool rot = i8_rotated(p, n_p);
    if (!halo_enc::affine(enc, RAD_I8, RAD_BF16, 1, HALO_FP8_BLOCK, RAD_DT_INVALID,
                         rot ? "fwht128" : ""))
        return RAD_E_DTYPE;
    if (operand == FL_BSCALE) {
        if (!halo_enc::selects(enc, sel, n, "scale") || !halo_enc::dims(pl[0], g.N, g.sk))
            return RAD_E_SHAPE;
        if (g.N % halo_fp8::kFragRows) return RAD_E_SHAPE;
        ffill(out, "r4d.i8.s.t16", RAD_BF16, g.N, g.sk, g.N * g.sk * 2);
        return RAD_OK;
    }
    if (!halo_enc::selects(enc, sel, n, "codes") || !halo_enc::dims(pl[0], g.N, g.K))
        return RAD_E_SHAPE;
    if (g.K % HALO_FP8_SUPER_K || g.N % halo_fp8::kFragRows) return RAD_E_SHAPE;
    ffill(out, rot ? "r4d.i8h.w.frag64" : "r4d.i8.w.frag64", RAD_I8, g.N, g.K,
          halo_fp8::frag_bytes(g.N, g.K));
    return RAD_OK;
}
}  /* namespace */

extern "C" int halo_rad_layout_i8_block(const RadParam* p, int n_p, int operand,
                                       const RadEncoding* enc, const int* sel,
                                       const RadTensor* pl, int n, RadLayout* out) {
    return i8_enc_layout(p, n_p, operand, enc, sel, pl, n, out);
}

extern "C" int halo_rad_relayout_i8_block(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel,
                                         const RadTensor* pl, int n, void* dst,
                                         int64_t dst_bytes) {
    RadLayout L;
    const int rc = i8_enc_layout(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    if (operand == FL_BSCALE) {
        const long long N = pl[0].shape[0], nkb = pl[0].shape[1];
        const uint16_t* s = (const uint16_t*)pl[0].data;
        uint16_t* d = (uint16_t*)dst;
        for (long long r = 0; r < N; ++r)
            for (long long kb = 0; kb < nkb; ++kb)
                d[((r / 16) * nkb + kb) * 16 + r % 16] = s[r * nkb + kb];
        return RAD_OK;
    }
    const uint8_t* s = (const uint8_t*)pl[0].data;
    uint8_t* d = (uint8_t*)dst;
    frag_walk(pl[0].shape[0], pl[0].shape[1],
              [&](int64_t at, int64_t from) { std::memcpy(d + at, s + from, 8); });
    return RAD_OK;
}

extern "C" int halo_rad_unrelayout_i8_block(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const void* src, int64_t src_bytes,
                                           const RadTensor* pl, int n) {
    RadLayout L;
    const int rc = i8_enc_layout(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    if (operand == FL_BSCALE) {
        const long long N = pl[0].shape[0], nkb = pl[0].shape[1];
        const uint16_t* s = (const uint16_t*)src;
        uint16_t* d = (uint16_t*)pl[0].data;
        for (long long r = 0; r < N; ++r)
            for (long long kb = 0; kb < nkb; ++kb)
                d[r * nkb + kb] = s[((r / 16) * nkb + kb) * 16 + r % 16];
        return RAD_OK;
    }
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* d = (uint8_t*)pl[0].data;
    frag_walk(pl[0].shape[0], pl[0].shape[1],
              [&](int64_t at, int64_t from) { std::memcpy(d + from, s + at, 8); });
    return RAD_OK;
}

/* ---- the lm_head: codes and scales in one row, the scales of the row's 128-row block repeated
 * into its tail. */
namespace {
/* `i8` selects the int8 head (logits_gemm_i8): i8 codes with a bf16 scale per row per 128
 * columns. Same stored row, but each row's tail holds its own scales instead of its block's. */
int lm_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* pl, int n, RadLayout* out, FGeom* gout, bool i8 = false) {
    if (!out || !pl) return RAD_E_INVAL;
    if (operand != LGL_W) return RAD_E_UNSUPPORTED;
    FGeom g;
    const int rc = lgeom(p, n_p, &g);
    if (rc != RAD_OK) return rc;
    if (i8 ? !halo_enc::affine(enc, RAD_I8, RAD_BF16, 1, HALO_FP8_BLOCK) : !fp8_block_enc(enc))
        return RAD_E_DTYPE;
    if (!halo_enc::selects(enc, sel, n, "codes", "scale")) return RAD_E_SHAPE;
    if (!halo_enc::dims(pl[0], g.N, g.K) || !halo_enc::dims(pl[1], i8 ? g.N : g.sn, g.sk))
        return RAD_E_SHAPE;
    const long long stride = halo_fp8::lm_row_bytes(g.K);
    ffill(out, i8 ? "r4d.i8lm.r128" : "r4d.fp8lm.b128", i8 ? RAD_I8 : RAD_F8E4M3, g.N, stride,
          g.N * stride);
    if (gout) *gout = g;
    return RAD_OK;
}
}  /* namespace */

extern "C" int halo_rad_layout_fp8_logits(const RadParam* p, int n_p, int operand,
                                             const RadEncoding* enc, const int* sel,
                                             const RadTensor* pl, int n, RadLayout* out) {
    return lm_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr);
}

extern "C" int halo_rad_relayout_fp8_logits(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const RadTensor* pl, int n, void* dst,
                                           int64_t dst_bytes) {
    RadLayout L;
    FGeom g;
    const int rc = lm_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const long long stride = halo_fp8::lm_row_bytes(g.K);
    const uint8_t* codes = (const uint8_t*)pl[0].data;
    const uint16_t* sc = (const uint16_t*)pl[1].data;
    uint8_t* d = (uint8_t*)dst;
    std::memset(d, 0, (size_t)dst_bytes);
    for (long long r = 0; r < g.N; ++r) {
        uint8_t* row = d + r * stride;
        std::memcpy(row, codes + r * g.K, (size_t)g.K);
        std::memcpy(row + g.K, sc + (r / HALO_FP8_BLOCK) * g.sk, (size_t)(2 * g.sk));
    }
    return RAD_OK;
}

extern "C" int halo_rad_unrelayout_fp8_logits(const RadParam* p, int n_p, int operand,
                                             const RadEncoding* enc, const int* sel,
                                             const void* src, int64_t src_bytes,
                                             const RadTensor* pl, int n) {
    RadLayout L;
    FGeom g;
    const int rc = lm_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const long long stride = halo_fp8::lm_row_bytes(g.K);
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* codes = (uint8_t*)pl[0].data;
    uint16_t* sc = (uint16_t*)pl[1].data;
    for (long long r = 0; r < g.N; ++r) {
        std::memcpy(codes + r * g.K, s + r * stride, (size_t)g.K);
        if (r % HALO_FP8_BLOCK == 0)
            std::memcpy(sc + (r / HALO_FP8_BLOCK) * g.sk, s + r * stride + g.K, (size_t)(2 * g.sk));
    }
    return RAD_OK;
}

extern "C" int halo_rad_layout_i8_logits(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel,
                                        const RadTensor* pl, int n, RadLayout* out) {
    return lm_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr, true);
}

extern "C" int halo_rad_relayout_i8_logits(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const RadTensor* pl, int n, void* dst,
                                          int64_t dst_bytes) {
    RadLayout L;
    FGeom g;
    const int rc = lm_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g, true);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const long long stride = halo_fp8::lm_row_bytes(g.K);
    const uint8_t* codes = (const uint8_t*)pl[0].data;
    const uint16_t* sc = (const uint16_t*)pl[1].data;
    uint8_t* d = (uint8_t*)dst;
    std::memset(d, 0, (size_t)dst_bytes);
    for (long long r = 0; r < g.N; ++r) {
        uint8_t* row = d + r * stride;
        std::memcpy(row, codes + r * g.K, (size_t)g.K);
        std::memcpy(row + g.K, sc + r * g.sk, (size_t)(2 * g.sk));
    }
    return RAD_OK;
}

extern "C" int halo_rad_unrelayout_i8_logits(const RadParam* p, int n_p, int operand,
                                            const RadEncoding* enc, const int* sel,
                                            const void* src, int64_t src_bytes,
                                            const RadTensor* pl, int n) {
    RadLayout L;
    FGeom g;
    const int rc = lm_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g, true);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const long long stride = halo_fp8::lm_row_bytes(g.K);
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* codes = (uint8_t*)pl[0].data;
    uint16_t* sc = (uint16_t*)pl[1].data;
    for (long long r = 0; r < g.N; ++r) {
        std::memcpy(codes + r * g.K, s + r * stride, (size_t)g.K);
        std::memcpy(sc + r * g.sk, s + r * stride + g.K, (size_t)(2 * g.sk));
    }
    return RAD_OK;
}

/* ---- hc_read's mixing matrices and mtp_enter's fc: e4m3 codes with an f32 scale per group of
 * the row, the scales in the row's tail and the row padded to 16 bytes. */
namespace {
int hcrows_enc_layout(const HGeom& g, const char* tag, const RadEncoding* enc, const int* sel,
                      const RadTensor* pl, int n, RadLayout* out) {
    if (!out || !pl) return RAD_E_INVAL;
    if (!halo_enc::affine(enc, RAD_F8E4M3, RAD_F32, 1, g.G)) return RAD_E_DTYPE;
    if (!halo_enc::selects(enc, sel, n, "codes", "scale")) return RAD_E_SHAPE;
    if (!halo_enc::dims(pl[0], g.rows, g.K) || !halo_enc::dims(pl[1], g.rows, g.K / g.G))
        return RAD_E_SHAPE;
    const long long stride = halo_hc8::row_bytes(g.K, g.G);
    ffill(out, tag, RAD_F8E4M3, g.rows, stride, g.rows * stride);
    return RAD_OK;
}

int hcrows_relayout(const HGeom& g, const RadTensor* pl, void* dst, int64_t dst_bytes) {
    const long long stride = halo_hc8::row_bytes(g.K, g.G), ng = g.K / g.G;
    if (dst_bytes != g.rows * stride || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const uint8_t* codes = (const uint8_t*)pl[0].data;
    const float* sc = (const float*)pl[1].data;
    uint8_t* d = (uint8_t*)dst;
    std::memset(d, 0, (size_t)dst_bytes);
    for (long long r = 0; r < g.rows; ++r) {
        std::memcpy(d + r * stride, codes + r * g.K, (size_t)g.K);
        std::memcpy(d + r * stride + g.K, sc + r * ng, (size_t)(4 * ng));
    }
    return RAD_OK;
}

int hcrows_unrelayout(const HGeom& g, const void* src, int64_t src_bytes, const RadTensor* pl) {
    const long long stride = halo_hc8::row_bytes(g.K, g.G), ng = g.K / g.G;
    if (src_bytes != g.rows * stride || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const uint8_t* s = (const uint8_t*)src;
    for (long long r = 0; r < g.rows; ++r) {
        std::memcpy((uint8_t*)pl[0].data + r * g.K, s + r * stride, (size_t)g.K);
        std::memcpy((float*)pl[1].data + r * ng, s + r * stride + g.K, (size_t)(4 * ng));
    }
    return RAD_OK;
}

const char* hc_tag(int operand) {
    return operand == HCL_DOWN ? "halo.hc.e4m3.g128" : "halo.hc.e4m3.q4";
}
}  /* namespace */

extern "C" int halo_rad_layout_hc_e4m3(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const RadTensor* pl, int n, RadLayout* out) {
    HGeom g;
    const int rc = hgeom(p, n_p, operand, &g);
    if (rc != RAD_OK) return rc;
    return hcrows_enc_layout(g, hc_tag(operand), enc, sel, pl, n, out);
}

extern "C" int halo_rad_relayout_hc_e4m3(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel,
                                        const RadTensor* pl, int n, void* dst, int64_t dst_bytes) {
    HGeom g;
    RadLayout L;
    int rc = hgeom(p, n_p, operand, &g);
    if (rc == RAD_OK) rc = hcrows_enc_layout(g, hc_tag(operand), enc, sel, pl, n, &L);
    return rc != RAD_OK ? rc : hcrows_relayout(g, pl, dst, dst_bytes);
}

extern "C" int halo_rad_unrelayout_hc_e4m3(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel, const void* src,
                                          int64_t src_bytes, const RadTensor* pl, int n) {
    HGeom g;
    RadLayout L;
    int rc = hgeom(p, n_p, operand, &g);
    if (rc == RAD_OK) rc = hcrows_enc_layout(g, hc_tag(operand), enc, sel, pl, n, &L);
    return rc != RAD_OK ? rc : hcrows_unrelayout(g, src, src_bytes, pl);
}

extern "C" int halo_rad_layout_mtp_e4m3(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const RadTensor* pl, int n, RadLayout* out) {
    HGeom g;
    const int rc = mgeom(p, n_p, operand, &g);
    if (rc != RAD_OK) return rc;
    return hcrows_enc_layout(g, "r4d.mtp.e4m3.g128", enc, sel, pl, n, out);
}

extern "C" int halo_rad_relayout_mtp_e4m3(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel,
                                         const RadTensor* pl, int n, void* dst, int64_t dst_bytes) {
    HGeom g;
    RadLayout L;
    int rc = mgeom(p, n_p, operand, &g);
    if (rc == RAD_OK) rc = hcrows_enc_layout(g, "r4d.mtp.e4m3.g128", enc, sel, pl, n, &L);
    return rc != RAD_OK ? rc : hcrows_relayout(g, pl, dst, dst_bytes);
}

extern "C" int halo_rad_unrelayout_mtp_e4m3(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const void* src, int64_t src_bytes,
                                           const RadTensor* pl, int n) {
    HGeom g;
    RadLayout L;
    int rc = mgeom(p, n_p, operand, &g);
    if (rc == RAD_OK) rc = hcrows_enc_layout(g, "r4d.mtp.e4m3.g128", enc, sel, pl, n, &L);
    return rc != RAD_OK ? rc : hcrows_unrelayout(g, src, src_bytes, pl);
}

/* ================================================================= embed_lookup_q's table
 *
 * The gather reads the table as stored, so this hook only checks that it can: E4M3 codes with one
 * bf16 scale for the whole table (`fp8_e4m3*bf16[*x*]`, the n-gram table's recipe), or plain bf16
 * as the checkpoint ships it. libavx's host gather (avx_layout_ngram) accepts the same two forms
 * with the same checks, so neither refuses at declare a table the other reads. */
extern "C" int halo_rad_layout_embed_q(const RadParam* p, int n_p, int operand,
                                      const RadEncoding* enc, const int* sel,
                                      const RadTensor* pl, int n, RadLayout* out) {
    (void)out;
    enum { EQ_WTE = 1, EQ_SCALE = 2 };
    if (operand != EQ_WTE && operand != EQ_SCALE) return RAD_E_UNSUPPORTED;
    const long long rows = rad_param_geti(p, n_p, "n_vocab", 0);
    const long long dim  = rad_param_geti(p, n_p, "n_embd", 0);
    if (rows <= 0 || dim <= 0) return RAD_E_SHAPE;
    if (rad_enc_is(enc, "plain") && enc->n_planes == 1 && enc->plane[0].dtype == RAD_BF16) {
        if (operand != EQ_WTE) return RAD_E_DTYPE;
        if (n != 1 || !pl || !halo_enc::dims(pl[0], rows, dim)) return RAD_E_SHAPE;
        return RAD_E_UNSUPPORTED;
    }
    const RadEncPlane* sc = rad_enc_plane(enc, "scale");
    if (!rad_enc_is(enc, "affine") || enc->n_planes != 2 || enc->transform[0] ||
        enc->plane[0].dtype != RAD_F8E4M3 || !sc || sc->dtype != RAD_BF16 || sc->block[0] != 0 ||
        sc->block[1] != 0)
        return RAD_E_DTYPE;
    if (n != 1 || !pl || !sel) return RAD_E_SHAPE;
    if (!halo_enc::selects(enc, sel, n, operand == EQ_WTE ? "codes" : "scale")) return RAD_E_SHAPE;
    if (!halo_enc::dims(pl[0], operand == EQ_WTE ? rows : 1, operand == EQ_WTE ? dim : 1))
        return RAD_E_SHAPE;
    return RAD_E_UNSUPPORTED;
}
