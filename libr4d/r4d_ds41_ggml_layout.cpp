/* r4d_ds41_ggml_layout.cpp -- the native-block GEMMs' weight layout: affine planes (a GGUF import)
 * relayouted into llama.cpp's blocks at load, through libquant's definition of them. Pure host.
 *
 * libref's hooks (libref/ref_ggml.cpp: ref_ggml_layout, ref_ggml_relayout, ref_ggml_unrelayout),
 * restated here because a plugin links nothing of another: the operand must carry exactly the
 * type's encoding with every plane selected in order; the stored form is u8 [rows, K / block *
 * block_bytes], the type's native blocks rows back to back (pack_of), tagged "ggml_<type>" like
 * libref's so a weight both libraries read is stored once. The unrelayout unpacks the blocks
 * back into the planes and writes the shared tables (an IQ grid). The row length is `K` for the
 * GEMMs and `n_embd` for the vocabulary ops. Only the types the kernels decode are served (pack
 * and dequant both written in libquant): q8_0, q5_K, q6_K, iq4_nl, iq2_xxs, iq1_s, iq3_xxs
 * (r4d_ds41_ggml). */
#include "r4d_plugin.h"

#include "lq_ggml_dequant.h"
#include "lq_ggml_planes.h"

#include <cstring>
#include <vector>

namespace {

const lq::ggml::PlaneKind* gg_kind_named(const char* s) {
    if (!s) return nullptr;
    for (uint32_t t = 0; t < 64; ++t) {
        const lq::ggml::PlaneKind* k = lq::ggml::plane_kind(t);
        if (k && !std::strcmp(k->name, s) && lq::ggml::pack_of(t) && lq::ggml::dequant_of(t)) return k;
    }
    return nullptr;
}

const lq::ggml::PlaneKind* gg_kind_of(const RadParam* p, int n_p) {
    const RadParam* t = rad_param_find(p, n_p, "type");
    return gg_kind_named(t && t->kind == RAD_P_STR ? t->sval : nullptr);
}

/* An integer parameter, a range read as its ceiling (libref's pint). */
int64_t gg_pint(const RadParam* p, int n_p, const char* key) {
    const RadParam* q = rad_param_find(p, n_p, key);
    if (!q) return 0;
    if (q->kind == RAD_P_RANGE) return (int64_t)q->ihi;
    return q->kind == RAD_P_INT ? (int64_t)q->ival : 0;
}

/* The selection must be exactly the type's encoding, every plane in order. */
int gg_check(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
             int n_planes, const lq::ggml::PlaneKind** out) {
    const lq::ggml::PlaneKind* k = gg_kind_of(p, n_p);
    if (operand != 1 || !k || !enc || !sel) return RAD_E_DTYPE;
    const RadEncoding want = lq::ggml::encoding_of(k->type);
    if (!rad_enc_equal(enc, &want) || n_planes != want.n_planes) return RAD_E_DTYPE;
    for (int i = 0; i < n_planes; ++i)
        if (sel[i] != i) return RAD_E_SHAPE;
    *out = k;
    return RAD_OK;
}

/* Plane i's row r, or null for a TABLE or per-tensor plane (pack and unpack do not touch those). */
uint8_t* gg_plane_row(const RadEncoding* enc, const RadTensor* planes, int i, int64_t r) {
    const RadEncPlane& pl = enc->plane[i];
    if (pl.kind == RAD_PLANE_TABLE || (pl.block[0] == 0 && pl.block[1] == 0)) return nullptr;
    const int64_t cols = planes[i].rank >= 2 ? planes[i].shape[1] : 0;
    return (uint8_t*)planes[i].data + r * rad_enc_row_bytes(pl.dtype, cols);
}

int64_t gg_rows(const RadTensor* planes) { return planes[0].rank >= 1 ? planes[0].shape[0] : 0; }

int64_t gg_row_len(const RadParam* p, int n_p) {
    const int64_t K = gg_pint(p, n_p, "K");
    return K > 0 ? K : gg_pint(p, n_p, "n_embd");
}

/* A stable tag per type (the RadLayout keeps the pointer). */
const char* gg_tag(const lq::ggml::PlaneKind* k) {
    static const char* const kTags[][2] = {
        { "q8_0", "ggml_q8_0" },     { "q5_K", "ggml_q5_K" },       { "q6_K", "ggml_q6_K" },
        { "iq4_nl", "ggml_iq4_nl" }, { "iq2_xxs", "ggml_iq2_xxs" }, { "iq1_s", "ggml_iq1_s" },
        { "iq3_xxs", "ggml_iq3_xxs" },
    };
    for (const auto& t : kTags)
        if (!std::strcmp(t[0], k->name)) return t[1];
    static thread_local char tag[32];
    snprintf(tag, sizeof tag, "ggml_%s", k->name);
    return tag;
}

}  // namespace

extern "C" int r4d_ds41_ggml_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                    const int* sel, const RadTensor* planes, int n_planes,
                                    RadLayout* out) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const lq::ggml::PlaneKind* k = nullptr;
    const int rc = gg_check(p, n_p, operand, enc, sel, n_planes, &k);
    if (rc < 0) return rc;
    const int64_t K = gg_row_len(p, n_p), rows = gg_rows(planes);
    if (K <= 0 || K % k->block || rows <= 0) return RAD_E_SHAPE;
    std::memset(out, 0, sizeof *out);
    out->tag = gg_tag(k);
    out->dtype = RAD_U8;
    out->rank = 2;
    out->shape[0] = rows;
    out->shape[1] = K / k->block * (int64_t)k->block_bytes;
    out->align = 64;
    out->bytes = out->shape[0] * out->shape[1];
    return RAD_OK;
}

extern "C" int r4d_ds41_ggml_relayout(const RadParam* p, int n_p, int operand,
                                      const RadEncoding* enc, const int* sel,
                                      const RadTensor* planes, int n_planes, void* dst,
                                      int64_t dst_bytes) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const lq::ggml::PlaneKind* k = nullptr;
    const int rc = gg_check(p, n_p, operand, enc, sel, n_planes, &k);
    if (rc < 0) return rc;
    const int64_t K = gg_row_len(p, n_p), rows = gg_rows(planes);
    const int64_t nb = K / k->block, rb = nb * (int64_t)k->block_bytes;
    if (K <= 0 || K % k->block || rows <= 0 || dst_bytes < rows * rb) return RAD_E_SHAPE;
    const lq::ggml::PackFn pack = lq::ggml::pack_of(k->type);
    std::vector<const uint8_t*> row((size_t)n_planes);
    for (int64_t r = 0; r < rows; ++r) {
        for (int i = 0; i < n_planes; ++i) row[(size_t)i] = gg_plane_row(enc, planes, i, r);
        pack(row.data(), nb, (uint8_t*)dst + r * rb);
    }
    return RAD_OK;
}

extern "C" int r4d_ds41_ggml_unrelayout(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel, const void* src,
                                        int64_t src_bytes, const RadTensor* planes, int n_planes) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const lq::ggml::PlaneKind* k = nullptr;
    const int rc = gg_check(p, n_p, operand, enc, sel, n_planes, &k);
    if (rc < 0) return rc;
    const int64_t K = gg_row_len(p, n_p), rows = gg_rows(planes);
    const int64_t nb = K / k->block, rb = nb * (int64_t)k->block_bytes;
    if (K <= 0 || K % k->block || rows <= 0 || src_bytes < rows * rb) return RAD_E_SHAPE;
    std::vector<uint8_t*> row((size_t)n_planes);
    std::vector<void*> all((size_t)n_planes);
    for (int i = 0; i < n_planes; ++i) all[(size_t)i] = planes[i].data;
    for (int64_t r = 0; r < rows; ++r) {
        for (int i = 0; i < n_planes; ++i) row[(size_t)i] = gg_plane_row(enc, planes, i, r);
        k->unpack((const uint8_t*)src + r * rb, nb, row.data());
    }
    lq::ggml::write_shared(k->type, *enc, all.data());
    return RAD_OK;
}
