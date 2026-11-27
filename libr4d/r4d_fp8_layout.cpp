/* r4d_fp8_layout.cpp -- the stored form of libr4d's 8-bit-code weights and the relayouts that
 * produce it from the weight's canonical planes (spec §4.3). Pure host C++: nothing here includes
 * r4d.h or <hip/hip_runtime.h>, so packing can be tested without ROCm.
 *
 * Int8 linears (gemm_nt_q): codes `i8 [N][K]` with a bf16 scale per row per 128 K (256 at IQ8R).
 * The codes move into WMMA fragment order (r4d_fp8_frag.h), so a lane reads its own bytes from one
 * coalesced load; the scale plane is stored tile-major, [N/16][K/group][16], so a lane's eight
 * accumulator rows read their scales as one 16-byte load.
 *
 * The lm_head is a single row-sharded operand, so each row carries its scales in its own tail:
 * [K codes][K/128 bf16 scales], padded to 16 bytes so a uint4 load of the next row is aligned
 * (r4d_lm_row.h). hc_read's mixing matrices and mtp_enter's fc are E4M3 rows with an f32 scale per
 * group of the row in the same form (r4d_hc_fp8.h).
 *
 * This file chooses no numbers: a weight's planes are served as they are, quantised before the
 * container was written.
 */

#include "r4d_plugin.h"
#include "r4d_fp8_frag.h"
#include "r4d_lm_row.h"
#include "r4d_hc_fp8.h"
#include "r4d_enc.h"
#include "r4d_hsimd.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

/* Operand indices of `gemm_nt_q`. The schema is positional, so these must follow its order in
 * r4d_rows.cpp. */
enum { FL_A = 0, FL_ASCALE = 1, FL_B = 2, FL_BSCALE = 3, FL_Y = 4 };

/* K span of one fragment-order tile, compiled into libr4d. */
enum { R4D_SUPER_K = r4d_fp8::kSuperK };

namespace {

/* Walks the fragment-order plane front to back, eight bytes at a time, passing each word's
 * row-major offset in the [N][K] weight; used in both directions. A fragment is eight consecutive
 * k of one row (r4d_fp8_frag.h frag_src over j = 0..7), so it moves as one word. Placing each byte
 * separately would dominate the CPU cost of loading a model. */
template <typename Move>
void frag_walk(long long N, long long K, Move move) {
    constexpr int kLanes = r4d_fp8::kTileBytes / r4d_fp8::kLaneBytes;
    constexpr int kFrags = r4d_fp8::kLaneBytes / 8;
    const long long ksuper = K / r4d_fp8::kSuperK;
    int64_t at = 0;
    for (long long rt = 0; rt < N / r4d_fp8::kFragRows; ++rt)
        for (long long ks = 0; ks < ksuper; ++ks)
            for (int lane = 0; lane < kLanes; ++lane)
                for (int t = 0; t < kFrags; ++t, at += 8) {
                    int64_t row, k;
                    r4d_fp8::frag_src(rt, ks, lane, t, 0, &row, &k);
                    move(at, row * K + k);
                }
}

/* frag_walk's forward direction a 16-row tile at a time (r4d_hsimd, by CPU level): the load
 * path of every int8 GEMM weight. */
void frag_tiles(const uint8_t* s, long long N, long long K, uint8_t* d) {
    const long long ksuper = K / r4d_fp8::kSuperK;
    for (long long rt = 0; rt < N / r4d_fp8::kFragRows; ++rt)
        r4d_hsimd::fp8_frag_tile(s + rt * r4d_fp8::kFragRows * K, K, ksuper,
                                 d + rt * r4d_fp8::kFragRows * K);
}

struct FGeom { long long N, K, sk; };

int fgeom(const RadParam* p, int n_p, FGeom* g) {
    g->N = r4d_pgeti_or(p, n_p, "N", 0);
    g->K = r4d_pgeti_or(p, n_p, "K", 0);
    if (g->N <= 0 || g->K <= 0) return RAD_E_SHAPE;
    /* The block is compiled into the kernels as `>> 7`. The constraint table already refuses
     * other groups; this catches a caller that invokes the hook without selection. */
    if (r4d_pgeti_or(p, n_p, "group", R4D_I8_GROUP) != R4D_I8_GROUP) return RAD_E_SHAPE;
    g->sk = (g->K + R4D_I8_GROUP - 1) / R4D_I8_GROUP;
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
    out->align = 16;   /* 16-byte loads; see the padding note in r4d_lm_row.h */
    out->bytes = bytes;
}


/* `logits_gemm` names its geometry `n_vocab` and `n_embd` where the GEMMs say N and K. */
int lgeom(const RadParam* p, int n_p, FGeom* g) {
    g->N = r4d_pgeti_or(p, n_p, "n_vocab", 0);
    g->K = r4d_pgeti_or(p, n_p, "n_embd", 0);
    if (g->N <= 0 || g->K <= 0) return RAD_E_SHAPE;
    if (g->K % R4D_I8_GROUP) return RAD_E_SHAPE;
    g->sk = g->K / R4D_I8_GROUP;
    return RAD_OK;
}

/* Operand 1 is the weight; see oLogits in r4d_rows.cpp. */
enum { LGL_X = 0, LGL_W = 1, LGL_Y = 2 };

/* Operand indices of hc_read -- see oHcRead in r4d_rows.cpp. */
enum { HCL_DOWN = 2, HCL_UP = 3 };

struct HGeom { long long rows, K, G; };

int hgeom(const RadParam* p, int n_p, int operand, HGeom* g) {
    const long long n  = r4d_pgeti_or(p, n_p, "n", 0);
    const long long hc = r4d_pgeti_or(p, n_p, "hc", 0);
    const long long lr = r4d_pgeti_or(p, n_p, "lowrank", 0);
    if (n <= 0 || hc <= 0 || lr <= 0) return RAD_E_SHAPE;
    if (operand == HCL_DOWN)    { g->rows = lr;     g->K = hc * n; g->G = r4d_hc8::down_group(g->K); }
    else if (operand == HCL_UP) { g->rows = hc * n; g->K = lr;     g->G = r4d_hc8::up_group(g->K); }
    else return RAD_E_UNSUPPORTED;
    if (!r4d_hc8::group_ok(g->K, g->G)) return RAD_E_SHAPE;
    return RAD_OK;
}

enum { MTL_FCH = 4, MTL_FCE = 5 };   /* oMtpEnter in r4d_rows.cpp */

int mgeom(const RadParam* p, int n_p, int operand, HGeom* g) {
    if (operand != MTL_FCH && operand != MTL_FCE) return RAD_E_UNSUPPORTED;
    const long long n = r4d_pgeti_or(p, n_p, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    g->rows = n; g->K = n; g->G = r4d_hc8::kDownGroup;
    if (!r4d_hc8::group_ok(g->K, g->G)) return RAD_E_SHAPE;
    return RAD_OK;
}

}  /* namespace */

/* ==================================================================== the hooks
 *
 * Each checks the encoding its kernel reads and refuses any other with RAD_E_DTYPE (the
 * declare-time error for a mis-converted container). They move bits and choose none. */

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
 * and the rotated codes carry a tag of their own.
 *
 * Int8 codes only. A narrower weight on the same grid (the IQR forms) is kept at its codes' width
 * and read by the gemm_iqr rows (r4d_iqr_layout.cpp). */
namespace {
bool i8_rotated(const RadParam* p, int n_p) {
    const char* d = r4d_pgets(p, n_p, "dtype");
    return d && std::strcmp(d, "i8a8h") == 0;
}

/* int8 codes with a bf16 scale per (row, 128), at transform `tr`. */
bool i8_codes(const RadEncoding* enc, const char* tr) {
    return r4d_enc::affine(enc, RAD_I8, RAD_BF16, 1, R4D_I8_GROUP, RAD_DT_INVALID, tr);
}

/* IQ8R: int8 codes rotated 128 wide, a u8 scale per (row, 256) under one bf16 scale a row
 * (libquant's `codes=i8 group=256 scale=u8 scale2=bf16 block2=1x* transform=fwht128`). A weight
 * element is code * scale * scale.1. */
bool iq8r_codes(const RadEncoding* enc) {
    if (!enc || !rad_enc_is(enc, "affine") || enc->n_planes != 3 ||
        std::strncmp(enc->transform, "fwht128", RAD_ENC_STR) != 0)
        return false;
    if (enc->plane[0].dtype != RAD_I8 || enc->plane[0].block[1] != 1) return false;
    const int s = rad_enc_find(enc, "scale"), s2 = rad_enc_find(enc, "scale.1");
    return s >= 0 && enc->plane[s].dtype == RAD_U8 && enc->plane[s].kind == RAD_PLANE_TILED &&
           enc->plane[s].block[0] == 1 && enc->plane[s].block[1] == R4D_I8_GROUP_WIDE &&
           s2 >= 0 && enc->plane[s2].dtype == RAD_BF16 && enc->plane[s2].kind == RAD_PLANE_TILED &&
           enc->plane[s2].block[0] == 1 && enc->plane[s2].block[1] == 0;
}

/* The scale group is 128, or 256 for IQ8R (the rotation stays 128 wide). The kernels take it as a
 * template argument, so no other width has a layout. Either way the kernels read one bf16 scale
 * per (row, group), tile-major: IQ8R's is the product of its two stored levels, rounded once. */
int i8_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* pl, int n, RadLayout* out) {
    if (!out || !pl) return RAD_E_INVAL;
    FGeom g;
    g.N = r4d_pgeti_or(p, n_p, "N", 0);
    g.K = r4d_pgeti_or(p, n_p, "K", 0);
    const long long group = r4d_pgeti_or(p, n_p, "group", R4D_I8_GROUP);
    if (g.N <= 0 || g.K <= 0 || (group != R4D_I8_GROUP && group != R4D_I8_GROUP_WIDE))
        return RAD_E_SHAPE;
    g.sk = (g.K + group - 1) / group;
    if (operand != FL_B && operand != FL_BSCALE) return RAD_E_UNSUPPORTED;
    const bool rot = i8_rotated(p, n_p);
    const bool wide = group == R4D_I8_GROUP_WIDE;
    if (wide ? !(rot && iq8r_codes(enc)) : !i8_codes(enc, rot ? "fwht128" : "")) return RAD_E_DTYPE;
    if (operand == FL_BSCALE) {
        if (wide) {
            if (!r4d_enc::selects(enc, sel, n, "scale", "scale.1") ||
                !r4d_enc::dims(pl[0], g.N, g.sk) || !r4d_enc::dims(pl[1], g.N, 1))
                return RAD_E_SHAPE;
        } else if (!r4d_enc::selects(enc, sel, n, "scale") || !r4d_enc::dims(pl[0], g.N, g.sk)) {
            return RAD_E_SHAPE;
        }
        if (g.N % r4d_fp8::kFragRows) return RAD_E_SHAPE;
        ffill(out, wide ? "r4d.iq8r.s.t16" : "r4d.i8.s.t16", RAD_BF16, g.N, g.sk, g.N * g.sk * 2);
        return RAD_OK;
    }
    if (!r4d_enc::selects(enc, sel, n, "codes") || !r4d_enc::dims(pl[0], g.N, g.K))
        return RAD_E_SHAPE;
    if (g.K % R4D_SUPER_K || g.N % r4d_fp8::kFragRows) return RAD_E_SHAPE;
    ffill(out, rot ? "r4d.i8h.w.frag64" : "r4d.i8.w.frag64", RAD_I8, g.N, g.K,
          r4d_fp8::frag_bytes(g.N, g.K));
    return RAD_OK;
}
}  /* namespace */

extern "C" int r4d_rad_layout_i8_block(const RadParam* p, int n_p, int operand,
                                       const RadEncoding* enc, const int* sel,
                                       const RadTensor* pl, int n, RadLayout* out) {
    return i8_enc_layout(p, n_p, operand, enc, sel, pl, n, out);
}

/* gemm_nt_q_multi's weights: operand 6 + 2 i is slice i's codes and 7 + 2 i its scale
 * (oGemmNtQMulti in r4d_rows.cpp), each in gemm_nt_q's stored form at that slice's N, so a
 * weight the single linear also reads is stored once. */
namespace {
int i8_multi_args(const RadParam* p, int n_p, int operand, std::vector<RadParam>* pp, int* opd) {
    static const char* const kN[5] = { "N0", "N1", "N2", "N3", "N4" };
    const int i = (operand - 6) / 2;
    if (operand < 6 || i >= 5) return RAD_E_UNSUPPORTED;
    const long long Ni = r4d_pgeti_or(p, n_p, kN[i], 0);
    if (Ni <= 0) return RAD_E_SHAPE;
    pp->assign(p, p + n_p);
    bool set = false;
    for (RadParam& q : *pp)
        if (q.key && std::strcmp(q.key, "N") == 0) { q.kind = RAD_P_INT; q.ival = Ni; set = true; }
    if (!set) {
        RadParam q{};
        q.key = "N";
        q.kind = RAD_P_INT;
        q.ival = Ni;
        pp->push_back(q);
    }
    *opd = (operand - 6) % 2 ? FL_BSCALE : FL_B;
    return RAD_OK;
}
}  /* namespace */

extern "C" int r4d_rad_layout_i8_multi(const RadParam* p, int n_p, int operand,
                                       const RadEncoding* enc, const int* sel,
                                       const RadTensor* pl, int n, RadLayout* out) {
    std::vector<RadParam> pp;
    int opd = 0;
    const int rc = i8_multi_args(p, n_p, operand, &pp, &opd);
    if (rc != RAD_OK) return rc;
    return r4d_rad_layout_i8_block(pp.data(), (int)pp.size(), opd, enc, sel, pl, n, out);
}

extern "C" int r4d_rad_relayout_i8_multi(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel,
                                         const RadTensor* pl, int n, void* dst,
                                         int64_t dst_bytes) {
    std::vector<RadParam> pp;
    int opd = 0;
    const int rc = i8_multi_args(p, n_p, operand, &pp, &opd);
    if (rc != RAD_OK) return rc;
    return r4d_rad_relayout_i8_block(pp.data(), (int)pp.size(), opd, enc, sel, pl, n, dst,
                                     dst_bytes);
}

extern "C" int r4d_rad_unrelayout_i8_multi(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const void* src, int64_t src_bytes,
                                           const RadTensor* pl, int n) {
    std::vector<RadParam> pp;
    int opd = 0;
    const int rc = i8_multi_args(p, n_p, operand, &pp, &opd);
    if (rc != RAD_OK) return rc;
    return r4d_rad_unrelayout_i8_block(pp.data(), (int)pp.size(), opd, enc, sel, src, src_bytes,
                                       pl, n);
}

/* The same int8 weight at one scale per (row, 128 of K), both planes row-major as stored: what
 * the grouped expert GEMM reads, which stages a run's rows through LDS and gains nothing from a
 * permutation. An expert then goes from the container or a host pool to the kernel unchanged. */
extern "C" int r4d_rad_layout_i8_rowmajor(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const RadTensor* pl, int n, RadLayout*) {
    if (!pl) return RAD_E_INVAL;
    FGeom g;
    const int rc = fgeom(p, n_p, &g);
    if (rc != RAD_OK) return rc;
    if (operand != FL_B && operand != FL_BSCALE) return RAD_E_UNSUPPORTED;
    if (!i8_codes(enc, i8_rotated(p, n_p) ? "fwht128" : "")) return RAD_E_DTYPE;
    if (operand == FL_BSCALE
            ? !r4d_enc::selects(enc, sel, n, "scale") || !r4d_enc::dims(pl[0], g.N, g.sk)
            : !r4d_enc::selects(enc, sel, n, "codes") || !r4d_enc::dims(pl[0], g.N, g.K))
        return RAD_E_SHAPE;
    return RAD_E_UNSUPPORTED;                          /* the stored plane, as it is */
}

extern "C" int r4d_rad_relayout_i8_block(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel,
                                         const RadTensor* pl, int n, void* dst,
                                         int64_t dst_bytes) {
    RadLayout L;
    const int rc = i8_enc_layout(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    if (operand == FL_BSCALE) {
        const long long N = pl[0].shape[0], nkb = pl[0].shape[1];
        uint16_t* d = (uint16_t*)dst;
        if (n == 2) {
            /* IQ8R: the two levels' product, the one bf16 the fold multiplies by. */
            if (!pl[1].data) return RAD_E_SHAPE;
            for (long long r = 0; r < N; ++r) {
                const uint8_t* s1 = r4d_enc::row(pl[0], r);
                const float s2 = rad_load_f32(pl[1].data, RAD_BF16, r);
                for (long long kb = 0; kb < nkb; ++kb)
                    rad_store_f32(d, RAD_BF16, ((r / 16) * nkb + kb) * 16 + r % 16,
                                  (float)s1[kb] * s2);
            }
            return RAD_OK;
        }
        const uint16_t* s = (const uint16_t*)pl[0].data;
        for (long long r = 0; r < N; ++r)
            for (long long kb = 0; kb < nkb; ++kb)
                d[((r / 16) * nkb + kb) * 16 + r % 16] = s[r * nkb + kb];
        return RAD_OK;
    }
    frag_tiles((const uint8_t*)pl[0].data, pl[0].shape[0], pl[0].shape[1], (uint8_t*)dst);
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_i8_block(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const void* src, int64_t src_bytes,
                                           const RadTensor* pl, int n) {
    RadLayout L;
    const int rc = i8_enc_layout(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    if (operand == FL_BSCALE) {
        /* IQ8R's product does not come apart into its two stored levels. */
        if (n == 2) return RAD_E_UNSUPPORTED;
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

/* ---- the lm_head: each row's int8 codes, then its bf16 scales, then the alignment pad. */
namespace {
/* The int8 head (logits_gemm_i8): i8 codes with a bf16 scale per row per 128 columns, each row's
 * tail holding its own scales. */
int lm_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* pl, int n, RadLayout* out, FGeom* gout) {
    if (!out || !pl) return RAD_E_INVAL;
    if (operand != LGL_W) return RAD_E_UNSUPPORTED;
    FGeom g;
    const int rc = lgeom(p, n_p, &g);
    if (rc != RAD_OK) return rc;
    if (!i8_codes(enc, "")) return RAD_E_DTYPE;
    if (!r4d_enc::selects(enc, sel, n, "codes", "scale")) return RAD_E_SHAPE;
    if (!r4d_enc::dims(pl[0], g.N, g.K) || !r4d_enc::dims(pl[1], g.N, g.sk)) return RAD_E_SHAPE;
    const long long stride = r4d_lm::lm_row_bytes(g.K);
    ffill(out, "r4d.i8lm.r128", RAD_I8, g.N, stride, g.N * stride);
    if (gout) *gout = g;
    return RAD_OK;
}
}  /* namespace */

extern "C" int r4d_rad_layout_i8_logits(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel,
                                        const RadTensor* pl, int n, RadLayout* out) {
    return lm_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr);
}

extern "C" int r4d_rad_relayout_i8_logits(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const RadTensor* pl, int n, void* dst,
                                          int64_t dst_bytes) {
    RadLayout L;
    FGeom g;
    const int rc = lm_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const long long stride = r4d_lm::lm_row_bytes(g.K);
    const uint8_t* codes = (const uint8_t*)pl[0].data;
    const uint16_t* sc = (const uint16_t*)pl[1].data;
    uint8_t* d = (uint8_t*)dst;
    /* Each row written once: its codes, its scales, then the padding cleared. */
    const long long used = g.K + 2 * g.sk;
    for (long long r = 0; r < g.N; ++r) {
        uint8_t* row = d + r * stride;
        std::memcpy(row, codes + r * g.K, (size_t)g.K);
        std::memcpy(row + g.K, sc + r * g.sk, (size_t)(2 * g.sk));
        if (stride > used) std::memset(row + used, 0, (size_t)(stride - used));
    }
    if (dst_bytes > g.N * stride) std::memset(d + g.N * stride, 0, (size_t)(dst_bytes - g.N * stride));
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_i8_logits(const RadParam* p, int n_p, int operand,
                                            const RadEncoding* enc, const int* sel,
                                            const void* src, int64_t src_bytes,
                                            const RadTensor* pl, int n) {
    RadLayout L;
    FGeom g;
    const int rc = lm_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const long long stride = r4d_lm::lm_row_bytes(g.K);
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
    if (!r4d_enc::affine(enc, RAD_F8E4M3, RAD_F32, 1, g.G)) return RAD_E_DTYPE;
    if (!r4d_enc::selects(enc, sel, n, "codes", "scale")) return RAD_E_SHAPE;
    if (!r4d_enc::dims(pl[0], g.rows, g.K) || !r4d_enc::dims(pl[1], g.rows, g.K / g.G))
        return RAD_E_SHAPE;
    const long long stride = r4d_hc8::row_bytes(g.K, g.G);
    ffill(out, tag, RAD_F8E4M3, g.rows, stride, g.rows * stride);
    return RAD_OK;
}

int hcrows_relayout(const HGeom& g, const RadTensor* pl, void* dst, int64_t dst_bytes) {
    const long long stride = r4d_hc8::row_bytes(g.K, g.G), ng = g.K / g.G;
    if (dst_bytes != g.rows * stride || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const uint8_t* codes = (const uint8_t*)pl[0].data;
    const float* sc = (const float*)pl[1].data;
    uint8_t* d = (uint8_t*)dst;
    /* Each row written once: its codes, its scales, then the padding cleared. */
    const long long used = g.K + 4 * ng;
    for (long long r = 0; r < g.rows; ++r) {
        uint8_t* row = d + r * stride;
        std::memcpy(row, codes + r * g.K, (size_t)g.K);
        std::memcpy(row + g.K, sc + r * ng, (size_t)(4 * ng));
        if (stride > used) std::memset(row + used, 0, (size_t)(stride - used));
    }
    return RAD_OK;
}

int hcrows_unrelayout(const HGeom& g, const void* src, int64_t src_bytes, const RadTensor* pl) {
    const long long stride = r4d_hc8::row_bytes(g.K, g.G), ng = g.K / g.G;
    if (src_bytes != g.rows * stride || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const uint8_t* s = (const uint8_t*)src;
    for (long long r = 0; r < g.rows; ++r) {
        std::memcpy((uint8_t*)pl[0].data + r * g.K, s + r * stride, (size_t)g.K);
        std::memcpy((float*)pl[1].data + r * ng, s + r * stride + g.K, (size_t)(4 * ng));
    }
    return RAD_OK;
}

const char* hc_tag(int operand) {
    return operand == HCL_DOWN ? "r4d.hc.e4m3.g128" : "r4d.hc.e4m3.q4";
}
}  /* namespace */

extern "C" int r4d_rad_layout_hc_e4m3(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const RadTensor* pl, int n, RadLayout* out) {
    HGeom g;
    const int rc = hgeom(p, n_p, operand, &g);
    if (rc != RAD_OK) return rc;
    return hcrows_enc_layout(g, hc_tag(operand), enc, sel, pl, n, out);
}

extern "C" int r4d_rad_relayout_hc_e4m3(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel,
                                        const RadTensor* pl, int n, void* dst, int64_t dst_bytes) {
    HGeom g;
    RadLayout L;
    int rc = hgeom(p, n_p, operand, &g);
    if (rc == RAD_OK) rc = hcrows_enc_layout(g, hc_tag(operand), enc, sel, pl, n, &L);
    return rc != RAD_OK ? rc : hcrows_relayout(g, pl, dst, dst_bytes);
}

extern "C" int r4d_rad_unrelayout_hc_e4m3(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel, const void* src,
                                          int64_t src_bytes, const RadTensor* pl, int n) {
    HGeom g;
    RadLayout L;
    int rc = hgeom(p, n_p, operand, &g);
    if (rc == RAD_OK) rc = hcrows_enc_layout(g, hc_tag(operand), enc, sel, pl, n, &L);
    return rc != RAD_OK ? rc : hcrows_unrelayout(g, src, src_bytes, pl);
}

extern "C" int r4d_rad_layout_mtp_e4m3(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const RadTensor* pl, int n, RadLayout* out) {
    HGeom g;
    const int rc = mgeom(p, n_p, operand, &g);
    if (rc != RAD_OK) return rc;
    return hcrows_enc_layout(g, "r4d.mtp.e4m3.g128", enc, sel, pl, n, out);
}

extern "C" int r4d_rad_relayout_mtp_e4m3(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel,
                                         const RadTensor* pl, int n, void* dst, int64_t dst_bytes) {
    HGeom g;
    RadLayout L;
    int rc = mgeom(p, n_p, operand, &g);
    if (rc == RAD_OK) rc = hcrows_enc_layout(g, "r4d.mtp.e4m3.g128", enc, sel, pl, n, &L);
    return rc != RAD_OK ? rc : hcrows_relayout(g, pl, dst, dst_bytes);
}

extern "C" int r4d_rad_unrelayout_mtp_e4m3(const RadParam* p, int n_p, int operand,
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
extern "C" int r4d_rad_layout_embed_q(const RadParam* p, int n_p, int operand,
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
        if (n != 1 || !pl || !r4d_enc::dims(pl[0], rows, dim)) return RAD_E_SHAPE;
        return RAD_E_UNSUPPORTED;
    }
    const RadEncPlane* sc = rad_enc_plane(enc, "scale");
    if (!rad_enc_is(enc, "affine") || enc->n_planes != 2 || enc->transform[0] ||
        enc->plane[0].dtype != RAD_F8E4M3 || !sc || sc->dtype != RAD_BF16 || sc->block[0] != 0 ||
        sc->block[1] != 0)
        return RAD_E_DTYPE;
    if (n != 1 || !pl || !sel) return RAD_E_SHAPE;
    if (!r4d_enc::selects(enc, sel, n, operand == EQ_WTE ? "codes" : "scale")) return RAD_E_SHAPE;
    if (!r4d_enc::dims(pl[0], operand == EQ_WTE ? rows : 1, operand == EQ_WTE ? dim : 1))
        return RAD_E_SHAPE;
    return RAD_E_UNSUPPORTED;
}

/* ================================================================= embed_lookup's table
 *
 * The bf16 gather reads a plain bf16 table as stored; a table on the int8 grid is the IQR gather's
 * (embed_lookup_iqr, r4d_iqr_layout.cpp), kept at its codes' width. */
extern "C" int r4d_rad_layout_embed_bf16(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel,
                                         const RadTensor* pl, int n, RadLayout* out) {
    (void)p; (void)n_p; (void)sel; (void)pl; (void)n; (void)out;
    enum { E_WTE = 1 };
    if (operand != E_WTE) return RAD_E_UNSUPPORTED;
    return rad_enc_is(enc, "plain") ? RAD_E_UNSUPPORTED : RAD_E_DTYPE;
}
