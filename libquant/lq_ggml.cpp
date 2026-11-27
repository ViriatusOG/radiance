/* lq_ggml.cpp -- the `ggml` quantiser: llama.cpp's formats, llama.cpp's codes, radiance's planes.
 *
 * A row is quantised by llama.cpp's own quantize_<type> (lq_ggml_port.h), with the imatrix's
 * column importances as quant_weights, and the resulting GGUF blocks are unpacked into the affine
 * planes of rad_encoding.h. A llama.cpp model and a radiance container quantised from the same
 * weights and imatrix therefore hold the same numbers, but the container stores a generic encoding
 * instead of a block layout tied to one kernel family.
 *
 *
 * The planes each type writes, and the unpacking of a row's blocks into them, live in
 * lq_ggml_planes.cpp, shared with the core's GGUF import.
 */
#include "lq_ggml.h"
#include "lq_ggml_planes.h"
#include "lq_ggml_port.h"
#include "lq_ggml_gpu.h"
#include "lq_quant.h"
#include "rad_plugin.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace lq {

namespace {

using namespace lq::ggml;

/* ================================================================== the types */

typedef size_t (*QuantizeFn)(const float*, void*, int64_t, int64_t, const float*);

struct Kind {
    const char* name;      /* as llama.cpp spells it */
    ggml_type   type;
    int64_t     block;     /* values one GGUF block covers */
    QuantizeFn  quantize;
};

const Kind kKinds[] = {
    { "q4_0",    GGML_TYPE_Q4_0,    QK4_0,  quantize_q4_0 },
    { "q4_1",    GGML_TYPE_Q4_1,    QK4_1,  quantize_q4_1 },
    { "q5_0",    GGML_TYPE_Q5_0,    QK5_0,  quantize_q5_0 },
    { "q5_1",    GGML_TYPE_Q5_1,    QK5_1,  quantize_q5_1 },
    { "q8_0",    GGML_TYPE_Q8_0,    QK8_0,  quantize_q8_0 },
    { "q2_K",    GGML_TYPE_Q2_K,    QK_K,   quantize_q2_K },
    { "q3_K",    GGML_TYPE_Q3_K,    QK_K,   quantize_q3_K },
    { "q4_K",    GGML_TYPE_Q4_K,    QK_K,   quantize_q4_K },
    { "q5_K",    GGML_TYPE_Q5_K,    QK_K,   quantize_q5_K },
    { "q6_K",    GGML_TYPE_Q6_K,    QK_K,   quantize_q6_K },
    { "iq4_nl",  GGML_TYPE_IQ4_NL,  QK4_NL, quantize_iq4_nl },
    { "iq4_xs",  GGML_TYPE_IQ4_XS,  QK_K,   quantize_iq4_xs },
    { "iq2_xxs", GGML_TYPE_IQ2_XXS, QK_K,   quantize_iq2_xxs },
    { "iq2_xs",  GGML_TYPE_IQ2_XS,  QK_K,   quantize_iq2_xs },
    { "iq2_s",   GGML_TYPE_IQ2_S,   QK_K,   quantize_iq2_s },
    { "iq3_xxs", GGML_TYPE_IQ3_XXS, QK_K,   quantize_iq3_xxs },
    { "iq3_s",   GGML_TYPE_IQ3_S,   QK_K,   quantize_iq3_s },
    { "iq1_s",   GGML_TYPE_IQ1_S,   QK_K,   quantize_iq1_s },
    { "iq1_m",   GGML_TYPE_IQ1_M,   QK_K,   quantize_iq1_m },
};

/* ================================================================== the quantiser */

bool same_name(const char* a, const char* b) {
    for (; *a && *b; ++a, ++b)
        if (std::tolower((unsigned char)*a) != std::tolower((unsigned char)*b)) return false;
    return *a == *b;
}

/* The type the options name, or RAD_E_INVAL with a message naming the weight. */
int kind_of(const RadParam* o, int n_o, const RadQuantWeight* w, const Kind** out) {
    const char* t = rad_param_gets(o, n_o, "type", nullptr);
    if (!t)
        return refuse(w, "type= is required: the llama.cpp type to write, e.g. q4_K, q8_0, iq4_xs");
    for (const Kind& k : kKinds)
        if (same_name(t, k.name)) {
            *out = &k;
            return RAD_OK;
        }
    std::string all;
    for (const Kind& k : kKinds) all += std::string(all.empty() ? "" : " ") + k.name;
    return refuse(w, ("type=" + std::string(t) + " is not a llama.cpp type this quantiser writes (" +
                      all + ")").c_str());
}

/* A rank-1 weight, or one whose rows do not divide into the type's blocks, is declined so the next
 * rule takes it (a norm, or a 2880-wide projection a q4_K pattern happened to match). */
int ggml_encoding(const RadParam* o, int n_o, const RadQuantWeight* w, RadEncoding* out) {
    if (declines(w)) return RAD_E_UNSUPPORTED;
    const Kind* k = nullptr;
    const int st = kind_of(o, n_o, w, &k);
    if (st != RAD_OK) return st;
    if (w->cols <= 0 || w->rows <= 0) return RAD_E_SHAPE;
    if (w->cols % k->block) return RAD_E_UNSUPPORTED;
    *out = encoding_of(k->type);
    return RAD_OK;
}

/* Rows are independent in every type: a block never spans rows, and the imatrix is per column. */
int64_t ggml_row_block(const RadParam*, int, const RadQuantWeight*) { return 1; }

/* Unpack a band of GGUF blocks the card wrote into the planes the host loop writes. */
static void unpack_gpu_blocks(const Kind* k, const RadEncoding& e, const int64_t* row_bytes,
                              const bool* per_row, const uint8_t* blocks, int64_t rsz,
                              int64_t rows, int64_t C, void* const* planes) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int64_t r = 0; r < rows; ++r) {
        uint8_t* row[RAD_ENC_MAX_PLANES] = {};
        for (int i = 0; i < e.n_planes; ++i)
            if (per_row[i]) row[i] = (uint8_t*)planes[i] + r * row_bytes[i];
        plane_kind((uint32_t)k->type)->unpack(blocks + r * rsz, C / k->block, row);
    }
}

int ggml_quantize(const RadParam* o, int n_o, const RadQuantWeight* w, float* src, int64_t row0,
                  int64_t rows, void* const* planes) {
    const Kind* k = nullptr;
    const int st = kind_of(o, n_o, w, &k);
    if (st != RAD_OK) return st;
    if (!w || w->rank < 2 || w->cols <= 0 || w->rows <= 0 || w->cols % k->block || row0 < 0 ||
        rows < 0 || row0 + rows > w->rows)
        return RAD_E_SHAPE;
    /* llama.cpp's iq2_xxs, iq2_xs and iq1_s require the imatrix and have no other path;
     * substituting one would write, under llama.cpp's type name, data llama.cpp could not have
     * produced. */
    if (!w->importance && ggml_quantize_requires_imatrix(k->type))
        return refuse(w, ("type=" + std::string(k->name) + " needs an imatrix, and the converter "
                          "has none for this weight (rad-convert --imatrix FILE)").c_str());
    ggml_quantize_init(k->type);

    const RadEncoding e = encoding_of(k->type);
    const int64_t C = w->cols, nb = C / k->block;
    int64_t row_bytes[RAD_ENC_MAX_PLANES] = {};
    bool per_row[RAD_ENC_MAX_PLANES] = {};
    for (int i = 0; i < e.n_planes; ++i) {
        const RadEncPlane& p = e.plane[i];
        int64_t pr = 0, pc = 0;
        rad_enc_plane_dims(&p, w->rows, C, &pr, &pc);
        row_bytes[i] = rad_enc_row_bytes(p.dtype, pc);
        per_row[i] = p.kind == RAD_PLANE_TILED && p.block[0] == 1;
    }
    write_shared(k->type, e, planes);

    const size_t rsz = ggml_row_size(k->type, C);
    const float* imp = w->importance;
#if defined(LQ_CARD)
    /* The card path is for the two GGML I types whose search is expensive and whose layouts this
     * build has a tested kernel for. Small weights stay on the host: rad-convert already gives
     * large weights a worker per card, and this keeps the many small-weight threads off the quiet
     * cards. A card failure other than an unsupported shape is loud, so a broken kernel cannot
     * silently become a different conversion. */
    const bool gpu_rule = rad_param_geti(o, n_o, "gpu", 1) != 0;
    const int64_t gpu_min = rad_param_geti(o, n_o, "gpu_min_bytes", (int64_t)32 << 20);
    const bool gpu_big = w->rows * C * 4 >= gpu_min;
    if (gpu_rule && gpu_big && (k->type == GGML_TYPE_IQ2_XXS || k->type == GGML_TYPE_IQ3_XXS) &&
        ggml::gpu_available()) {
        std::vector<uint8_t> gblocks((size_t)rows * rsz);
        std::string why;
        const int gs = ggml::quantize_gpu(k->type, src, gblocks.data(), rows, C, imp, &why);
        if (gs == RAD_OK) {
            unpack_gpu_blocks(k, e, row_bytes, per_row, gblocks.data(), (int64_t)rsz, rows, C,
                              planes);
            return RAD_OK;
        }
        if (gs != RAD_E_UNSUPPORTED)
            return refuse(w, why.empty() ? "the GGML card path failed" : why.c_str());
        /* The card has no path for this shape or has no room: fall back to the host oracle. */
    }
#endif
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<uint8_t> blocks(rsz);
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (int64_t r = 0; r < rows; ++r) {
            k->quantize(src + r * C, blocks.data(), 1, C, imp);
            uint8_t* row[RAD_ENC_MAX_PLANES] = {};
            for (int i = 0; i < e.n_planes; ++i)
                if (per_row[i]) row[i] = (uint8_t*)planes[i] + r * row_bytes[i];
            plane_kind((uint32_t)k->type)->unpack(blocks.data(), nb, row);
        }
    }
    return RAD_OK;
}

const RadQuantOption kOptions[] = {
    { "type", RAD_P_STR, nullptr,
      "the llama.cpp type, any case: q4_0 q4_1 q5_0 q5_1 q8_0 q2_K q3_K q4_K q5_K q6_K iq4_nl "
      "iq4_xs iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq1_s iq1_m. iq2_xxs, iq2_xs and iq1_s need an "
      "imatrix" },
    { "gpu", RAD_P_INT, "1",
      "0: keep this rule on the host even when --gpus names cards; 1: let large IQ2_XXS/IQ3_XXS "
      "weights use one card. The host path remains the fallback for small or unsupported weights" },
    { "gpu_min_bytes", RAD_P_INT, "33554432",
      "with gpu 1, the whole weight's f32 bytes from which it uses a card: the many small weights "
      "stay on the host threads, the cards take the large ones. The same bytes either way" },
};

}  /* namespace */

const RadQuantizerInfo ggml_quantizer = {
    "ggml",
    "llama.cpp's legacy, K- and I-quants, the codes and scales chosen by llama.cpp's own search "
    "(with the imatrix when the converter has one) and stored as affine planes",
    kOptions, (int)(sizeof kOptions / sizeof kOptions[0]),
    ggml_encoding, ggml_row_block, ggml_quantize, nullptr,
};

}  /* namespace lq */
