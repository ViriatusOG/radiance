/* lq_registry.cpp -- libquant's quantiser table and plugin exports. spec.md §4.5.
 *
 * Most quantisers are an algorithm over a grid (lq_grid.h): `rtn` rounds each value to its nearest
 * grid point, `gptq` feeds each column's rounding error into later columns, `awq` scales the input
 * channels the activations say matter, `autoround` descends on the rounding itself and `paroquant`
 * first learns a rotation of the input channels. They share the grid options, so anything added to
 * the grid is available to all. Two are not over a grid: `ggml` writes llama.cpp's formats by
 * llama.cpp's own searches (lq_ggml.h), and `cast` converts a weight to another float dtype.
 */
#include "lq_ggml.h"
#include "lq_gptq.h"
#include "lq_gpu.h"
#include "lq_quant.h"
#include "lq_common.h"
#include "rad_plugin.h"
#include "rad_abi_layout.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace lq {

int grid_of(const RadParam* o, int n_o, const RadQuantWeight* w, Grid* g, bool vq_ok) {
    std::string why;
    if (grid_parse(o, n_o, g, &why) != RAD_OK) return refuse(w, why.c_str());
    /* Only GPTQ's loop chooses vector codewords. */
    if (!g->vq.empty() && !vq_ok) return refuse(w, "vq= is quantised by gptq");
    if (g->tcq_L && !vq_ok) return refuse(w, "tcq= is quantised by gptq");
    if (!w || w->cols <= 0 || w->rows <= 0) return RAD_E_SHAPE;
    if (g->fwht && w->cols % g->fwht) return RAD_E_SHAPE;
    return RAD_OK;
}

int refuse(const RadQuantWeight* w, const char* why) {
    std::fprintf(stderr, "libquant: %s: %s\n", w && w->name ? w->name : "?", why);
    return RAD_E_INVAL;
}

uint64_t name_seed(const char* s) {
    uint64_t h = 0xcbf29ce484222325ull;                   /* FNV-1a */
    for (; s && *s; ++s) h = (h ^ (uint8_t)*s) * 0x100000001b3ull;
    return h;
}

}  /* namespace lq */

namespace {

using namespace lq;

const RadQuantOption kRtnOptions[] = { LQ_GRID_OPTIONS };
const RadQuantOption kGptqOptions[] = {
    LQ_GRID_OPTIONS,
    LQ_CALIB_OPTION("every value is rounded to nearest"),
    { "damp", RAD_P_F64, "0.01", "the Hessian's damping, a fraction of its mean diagonal" },
    { "act_order", RAD_P_INT, "0",
      "1: quantise the columns by descending input energy, stored in that order with the "
      "permutation as the encoding's transform (a kernel permutes its activation to match)" },
    { "cd", RAD_P_INT, "0",
      "sweeps of coordinate descent after the loop: every code re-chosen at its conditional "
      "optimum against the same damped Hessian, the other columns held (lq_gptq.h)" },
    { "dyn", RAD_P_INT, "0",
      "1: each 128-column group's scale chosen when the loop reaches it, from the values earlier "
      "columns' error has moved it to (one scale level, groups of 128)" },
};
const RadQuantOption kCastOptions[] = {
    { "dtype", RAD_P_STR, nullptr,
      "the dtype every value is converted to, rounding to nearest even: f32, bf16, f16, "
      "fp8_e4m3 (saturating at 448), fp8_e5m2" },
};

/* ------------------------------------------------------------------ rtn */
int rtn_encoding(const RadParam* o, int n_o, const RadQuantWeight* w, RadEncoding* out) {
    if (declines(w)) return RAD_E_UNSUPPORTED;
    Grid g;
    const int st = grid_of(o, n_o, w, &g);
    if (st != RAD_OK) return st;
    *out = grid_encoding(g);
    return RAD_OK;
}

int64_t rtn_row_block(const RadParam* o, int n_o, const RadQuantWeight* w) {
    Grid g;
    if (grid_of(o, n_o, w, &g) != RAD_OK) return 0;
    return grid_row_block(g);
}

int rtn_quantize(const RadParam* o, int n_o, const RadQuantWeight* w, float* src, int64_t row0,
                 int64_t rows, void* const* planes) {
    Grid g;
    const int st = grid_of(o, n_o, w, &g);
    if (st != RAD_OK) return st;
    grid_transform_rows(g, src, rows, w->cols);
    return grid_rtn(g, src, row0, rows, w->cols, grid_planes(g, w->cols, planes), w->importance);
}

/* ------------------------------------------------------------------ gptq */
/* Act order is a transform of the encoding: codes are stored in quantisation order, so scale
 * groups are runs of that order, and the reader gets the permutation ("perm", a t.perm table of one
 * i32 per column) to undo it. It cannot combine with the grid's rotation, since an encoding carries
 * one transform. */
int gptq_encoding(const RadParam* o, int n_o, const RadQuantWeight* w, RadEncoding* out) {
    if (declines(w)) return RAD_E_UNSUPPORTED;
    Grid g;
    const int st = grid_of(o, n_o, w, &g, true);
    if (st != RAD_OK) return st;
    *out = grid_encoding(g);
    if (rad_param_geti(o, n_o, "act_order", 0)) {
        if (g.fwht) return refuse(w, "act_order=1 and transform= are two transforms, and an "
                                     "encoding carries one");
        if (rad_param_geti(o, n_o, "cd", 0))
            return refuse(w, "cd= sweeps the columns in their stored order, and act_order=1 "
                             "stores them permuted; one or the other");
        if (!g.vq.empty()) return refuse(w, "act_order=1 and vq= : a codeword spans adjacent columns");
        if (g.tcq_L) return refuse(w, "act_order=1 and tcq= : the columns are stored in order");
        rad_enc_copy_str(out->transform, "perm");
        if (rad_enc_add_table(out, "t.perm", RAD_I32, 1, w->cols) < 0)
            return refuse(w, "the encoding has no room for the permutation table");
    }
    return RAD_OK;
}

int64_t whole_weight(const RadParam*, int, const RadQuantWeight*) { return 0; }

/* gptq's one scheme the core does not decode: a trellis grid's. */
int gptq_decode(const RadEncoding* e, const RadQuantWeight* w, const void* const* planes,
                int64_t row0, int64_t rows, float* out) {
    if (!rad_enc_is(e, "trellis")) return RAD_E_UNSUPPORTED;
    if (!w || w->rows <= 0 || w->cols <= 0) return RAD_E_SHAPE;
    return grid_tcq_decode(e, w->rows, w->cols, planes, row0, rows, out);
}

int gptq_quantize(const RadParam* o, int n_o, const RadQuantWeight* w, float* src, int64_t row0,
                  int64_t rows, void* const* planes) {
    Grid g;
    const int st = grid_of(o, n_o, w, &g, true);
    if (st != RAD_OK) return st;
    if (row0 != 0 || rows != w->rows) return RAD_E_SHAPE;
    const int64_t N = w->rows, K = w->cols;
    grid_transform_rows(g, src, N, K);

    const char* dir = rad_param_gets(o, n_o, "calib", nullptr);
    const double damp = rad_param_getf(o, n_o, "damp", 0.01);
    const bool act = rad_param_geti(o, n_o, "act_order", 0) != 0;
    std::vector<int32_t> perm;
    std::vector<float> imp_p;
    const float* imp = w->importance;
    if (act) {
        perm = gptq_act_order(dir, w->name, K, g.fwht);
        std::vector<float> row((size_t)K);
        for (int64_t r = 0; r < N; ++r) {
            float* wr = src + r * K;
            for (int64_t j = 0; j < K; ++j) row[(size_t)j] = wr[perm[(size_t)j]];
            std::memcpy(wr, row.data(), (size_t)K * sizeof(float));
        }
        if (imp) {
            imp_p.resize((size_t)K);
            for (int64_t j = 0; j < K; ++j) imp_p[(size_t)j] = imp[perm[(size_t)j]];
            imp = imp_p.data();
        }
    }

    /* Scales come first, from the weight as it stands, and stay fixed through the loop -- or with
     * dyn=1 each group's is chosen again as the loop reaches it (gptq_quantise). */
    Scales sc = grid_scales(g, src, N, K, imp);
    const Planes p = grid_planes(g, K, planes);
    const bool dyn = rad_param_geti(o, n_o, "dyn", 0) != 0;
    if (dyn && !gptq_dynamic_ok(g, sc))
        return refuse(w, "dyn=1 chooses a 128-column group's scale as GPTQ reaches it, which needs "
                         "one scale level per row and 128 columns");

    const Gram uf = gptq_factor(dir, w->name, K, g.fwht, damp, act ? perm.data() : nullptr);
    const float* u = uf ? uf->data() : nullptr;

    std::vector<int32_t> codes((size_t)(N * K));
    const int cd = (int)rad_param_geti(o, n_o, "cd", 0);
    std::vector<float> x0;
    if (cd > 0 && u) x0.assign(src, src + (size_t)(N * K));
    {
        std::string why;
        const int st = gptq_quantise(g, src, codes.data(), sc, N, K, u, dyn && u, &why);
        if (st != RAD_OK) {
            std::fprintf(stderr, "libquant: %s: %s\n", w->name ? w->name : "?", why.c_str());
            return st;
        }
    }
    grid_store_scales(g, sc, p);
    if (!x0.empty()) {
        const Gram h = calib_gram(dir, w->name, K, g.fwht, "gptq cd");
        if (h) gptq_refine(g, x0.data(), src, codes.data(), sc, N, K, h->data(), damp, cd);
    }
    const int64_t V = g.vq.empty() ? 1 : g.vdim;            /* one code a vector */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (N * K > (1 << 20))
#endif
    for (int64_t r = 0; r < N; ++r) {
        uint8_t* cr = p.codes + r * p.codes_row;
        for (int64_t c = 0; c < K; c += V) grid_store_code(g, cr, c / V, codes[(size_t)(r * K + c)]);
    }
    if (act) {
        const int tp = grid_encoding(g).n_planes;      /* t.perm follows the grid's planes */
        std::memcpy(planes[tp], perm.data(), (size_t)K * sizeof(int32_t));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ cast */
/* A float weight in another float dtype, round to nearest even: an f32 checkpoint served as bf16,
 * bf16 as f16, or plain unscaled fp8. Every rank is accepted, norms included: a cast is requested by
 * name, and a kernel that cannot read the result refuses it at declare like any other encoding. */
int cast_dtype(const RadParam* o, int n_o, const RadQuantWeight* w, uint32_t* dt) {
    const char* s = rad_param_gets(o, n_o, "dtype", nullptr);
    *dt = s ? rad_dtype_parse(s) : (uint32_t)RAD_DT_INVALID;
    if (*dt != RAD_F32 && *dt != RAD_BF16 && *dt != RAD_F16 && *dt != RAD_F8E4M3 &&
        *dt != RAD_F8E5M2)
        return refuse(w, "cast needs dtype= f32, bf16, f16, fp8_e4m3 or fp8_e5m2");
    return RAD_OK;
}

int cast_encoding(const RadParam* o, int n_o, const RadQuantWeight* w, RadEncoding* out) {
    uint32_t dt = 0;
    const int st = cast_dtype(o, n_o, w, &dt);
    if (st != RAD_OK) return st;
    *out = rad_enc_plain(dt);
    return RAD_OK;
}

int64_t cast_row_block(const RadParam*, int, const RadQuantWeight*) { return 1; }

int cast_quantize(const RadParam* o, int n_o, const RadQuantWeight* w, float* src, int64_t,
                  int64_t rows, void* const* planes) {
    uint32_t dt = 0;
    const int st = cast_dtype(o, n_o, w, &dt);
    if (st != RAD_OK) return st;
    const int64_t n = rows * w->cols;
    uint8_t* out = (uint8_t*)planes[0];
    auto each = [n](auto store) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t i = 0; i < n; ++i) store(i);
    };
    switch (dt) {
        case RAD_F32:    std::memcpy(out, src, (size_t)n * sizeof(float)); break;
        case RAD_BF16:   each([&](int64_t i) { ((uint16_t*)out)[i] = f32_to_bf16(src[i]); }); break;
        case RAD_F16:    each([&](int64_t i) { ((uint16_t*)out)[i] = f32_to_f16(src[i]); }); break;
        case RAD_F8E4M3: each([&](int64_t i) { out[i] = rad_f32_to_fp8e4m3(src[i]); }); break;
        default:         each([&](int64_t i) { out[i] = rad_f32_to_fp8e5m2(src[i]); }); break;
    }
    return RAD_OK;
}

const RadQuantizerInfo kQuantizers[] = {
    { "rtn",
      "round to nearest: every value to its nearest point on the grid, the scale of each block "
      "chosen by the grid's rule",
      kRtnOptions, (int)(sizeof kRtnOptions / sizeof kRtnOptions[0]),
      rtn_encoding, rtn_row_block, rtn_quantize, nullptr },
    { "gptq",
      "error feedback against the layer's input Hessian: each column's rounding error is spread "
      "over the columns after it, on any grid, the scales fixed from the weight first",
      kGptqOptions, (int)(sizeof kGptqOptions / sizeof kGptqOptions[0]),
      gptq_encoding, whole_weight, gptq_quantize, gptq_decode },
    { "cast",
      "a float weight converted to another float dtype, to nearest even, with no scale",
      kCastOptions, (int)(sizeof kCastOptions / sizeof kCastOptions[0]),
      cast_encoding, cast_row_block, cast_quantize, nullptr },
};

/* Every quantiser the plugin offers, in the order --list-quantizers prints them. */
const std::vector<const RadQuantizerInfo*>& table() {
    static const std::vector<const RadQuantizerInfo*> t = [] {
        std::vector<const RadQuantizerInfo*> v;
        for (const RadQuantizerInfo& q : kQuantizers) v.push_back(&q);
        v.push_back(&awq_info());
        v.push_back(&autoround_info());
        v.push_back(&paroquant_info());
        v.push_back(&ggml_quantizer);
        v.push_back(&m2xfp_info());
        return v;
    }();
    return t;
}

const RadPluginInfo kInfo = {
    RAD_PLUGIN_QUANT,
    "libquant",
    "0.2.0",
    "the built-in quantisers: round-to-nearest, GPTQ, AWQ, AutoRound and ParoQuant over any "
    "integer, float or codebook grid, with grouped, blocked, two-level and shared-exponent scales "
    "and a Hadamard, permutation or Givens transform; llama.cpp's legacy, K- and I-quants by its "
    "own searches; and a plain float cast (spec.md §4.5)",
    "host"
};

}  /* namespace */

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
extern "C" int rad_quant_count(void) { return (int)table().size(); }
extern "C" const RadQuantizerInfo* rad_quant_at(int i) {
    return (i >= 0 && i < rad_quant_count()) ? table()[(size_t)i] : nullptr;
}
extern "C" int rad_quant_devices(const int* ids, int n) { return lq::gpu_set_devices(ids, n); }
