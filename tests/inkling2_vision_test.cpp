/* inkling2_vision_test.cpp -- Inkling 2 Mini's vision tower block (arch/common/
 * rad_block_inkling2_vision.h) against the float32 golden of tests/mkinkling2vision.py, run through
 * the libref kernels the engine would resolve. The tiny checkpoint is the one the golden was made
 * from: its tensors are read through core's safetensors reader, converted from bf16 as the released
 * container stores them, and packed by the block exactly as the engine would at declare.
 *
 * The block issues ops by name through a launch callback; this file binds that callback to libref's
 * kernel table (reached by dlopen from $RADIANCE_HOME, as ref_test reaches it), so every op the tower
 * uses is checked by its oracle and nothing in the chain is re-implemented here. */
#include "rad_test.h"
#include "rad_abi.h"
#include "rad_types.h"
#include "../arch/common/rad_block_inkling2_vision.h"
#include "../core/format/safetensors.h"
#include "inkling2_vision_golden.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dlfcn.h>

using namespace rad::arch::inkling2v;

namespace {

/* ================================================================== the op table */

struct RefTable {
    void*                handle = nullptr;
    int                  (*count)(void) = nullptr;
    const RadKernelInfo* (*at)(int) = nullptr;
};

const RefTable& ref_table() {
    static RefTable t = [] {
        RefTable r;
        const std::string file = "libref.so";
        std::vector<std::string> paths;
        if (const char* home = std::getenv("RADIANCE_HOME")) paths.push_back(std::string(home) + "/kernels/" + file);
        paths.push_back("build/radiance_home/kernels/" + file);
        paths.push_back("radiance_home/kernels/" + file);
        for (auto& p : paths) {
            void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!h) continue;
            r.handle = h;
            r.count = (int (*)(void))dlsym(h, "rad_kernel_count");
            r.at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
            if (r.count && r.at) return r;
        }
        std::fprintf(stderr, "inkling2_vision_test: %s not found; set RADIANCE_HOME\n", file.c_str());
        std::exit(2);
    }();
    return t;
}

const RadKernelInfo* kern(const char* op) {
    const RefTable& t = ref_table();
    for (int i = 0; i < t.count(); ++i) {
        const RadKernelInfo* k = t.at(i);
        if (k && std::strcmp(k->op, op) == 0) return k;
    }
    return nullptr;
}

/* The engine's issue, for the host: one launch of the named kernel on the given operands. */
int launch_ref(void*, const char* op, const RadTensor* t, int nt, const RadParam* p, int np) {
    const RadKernelInfo* k = kern(op);
    if (!k || !k->launch) return RAD_E_NOKERNEL;
    RadArgs a{};
    a.t = t;
    a.n_t = nt;
    a.p = p;
    a.n_p = np;
    return k->launch(&a, nullptr);
}

/* ================================================================== the checkpoint */

/* The tensors as float32, converted once and kept for the case. BF16 is the high half of an f32. */
struct Weights {
    rad::SafeTensorsFile                 file;
    std::vector<std::vector<float>> owned;

    bool open(const std::string& path) { return file.open(path) == 0; }

    /* The named tensor as f32 with `numel` elements, or null when it is missing or the wrong size. */
    const float* get(const std::string& name, int64_t numel) {
        const rad::StTensor* st = file.find(name);
        if (!st || st->numel() != numel) return nullptr;
        std::vector<float> v((size_t)numel);
        if (st->dtype == RAD_BF16) {
            const uint16_t* h = (const uint16_t*)st->data;
            for (int64_t i = 0; i < numel; ++i) {
                uint32_t bits = (uint32_t)h[i] << 16;
                std::memcpy(&v[(size_t)i], &bits, 4);
            }
        } else if (st->dtype == RAD_F32) {
            std::memcpy(v.data(), st->data, (size_t)numel * 4);
        } else {
            return nullptr;
        }
        owned.push_back(std::move(v));
        return owned.back().data();
    }
};

const char* kBackbone = "model.visual.backbone.";

/* Reads the tower's tensors under the released names into the block's input, and sets the geometry
 * from the golden's own constants. Returns false when a tensor is missing or mis-shaped. */
bool read_tower(Weights& w, Geometry* g, TowerIn* in) {
    g->depth = kGoldInkVisDepth;
    g->width = kGoldInkVisWidth;
    g->heads = kGoldInkVisHeads;
    g->patch = kGoldInkVisPatch;
    g->side = kGoldInkVisSide;
    g->merge = kGoldInkVisMerge;
    g->text = kGoldInkVisText;
    g->eps = 1e-5f;
    g->theta = 1e4f;
    const int64_t W = g->width, P = g->patch, side = g->side;

    const rad::StTensor* pin = w.file.find("model.visual.proj_in.weight");
    if (!pin || pin->shape.size() != 2 || pin->shape[1] != 4 * W) return false;
    g->proj = pin->shape[0];
    const rad::StTensor* pout = w.file.find("model.visual.proj_out.weight");
    if (!pout || pout->shape.size() != 2 || pout->shape[1] != g->proj || pout->shape[0] != g->text) return false;

    in->cls = w.get(kBackbone + std::string("embeddings.class_embedding"), W);
    in->cls_pos = w.get(kBackbone + std::string("embeddings.cls_position_embedding"), W);
    in->patch_w = w.get(kBackbone + std::string("embeddings.patch_embedding.weight"), W * 3 * P * P);
    in->pos = w.get(kBackbone + std::string("embeddings.position_embedding"), side * side * W);
    in->pre_w = w.get(kBackbone + std::string("ln_pre.weight"), W);
    in->pre_b = w.get(kBackbone + std::string("ln_pre.bias"), W);
    in->in_w = w.get("model.visual.proj_in.weight", g->proj * 4 * W);
    in->out_w = w.get("model.visual.proj_out.weight", g->text * g->proj);
    if (!in->cls || !in->cls_pos || !in->patch_w || !in->pos || !in->pre_w || !in->pre_b ||
        !in->in_w || !in->out_w)
        return false;

    for (int64_t l = 0; l < g->depth; ++l) {
        const std::string L = kBackbone + std::string("layers.") + std::to_string(l) + ".";
        LayerIn d{};
        d.ln1_w = w.get(L + "ln_1.weight", W);
        d.ln1_b = w.get(L + "ln_1.bias", W);
        d.ln2_w = w.get(L + "ln_2.weight", W);
        d.ln2_b = w.get(L + "ln_2.bias", W);
        d.ls1 = w.get(L + "ls_1.gamma", W);
        d.ls2 = w.get(L + "ls_2.gamma", W);
        d.q_w = w.get(L + "self_attn.q_proj.weight", W * W);
        d.q_b = w.get(L + "self_attn.q_proj.bias", W);
        d.k_w = w.get(L + "self_attn.k_proj.weight", W * W);
        d.k_b = w.get(L + "self_attn.k_proj.bias", W);
        d.v_w = w.get(L + "self_attn.v_proj.weight", W * W);
        d.v_b = w.get(L + "self_attn.v_proj.bias", W);
        d.o_w = w.get(L + "self_attn.out_proj.weight", W * W);
        d.o_b = w.get(L + "self_attn.out_proj.bias", W);
        d.fc_w = w.get(L + "mlp.c_fc.weight", 4 * W * W);
        d.fc_b = w.get(L + "mlp.c_fc.bias", 4 * W);
        d.pr_w = w.get(L + "mlp.c_proj.weight", W * 4 * W);
        d.pr_b = w.get(L + "mlp.c_proj.bias", W);
        if (!d.ln1_w || !d.ln1_b || !d.ln2_w || !d.ln2_b || !d.ls1 || !d.ls2 || !d.q_w || !d.q_b ||
            !d.k_w || !d.k_b || !d.v_w || !d.v_b || !d.o_w || !d.o_b || !d.fc_w || !d.fc_b ||
            !d.pr_w || !d.pr_b)
            return false;
        in->layers.push_back(d);
    }
    return true;
}

std::string ckpt_path() { return std::string(RAD_GOLDEN_FIXTURES) + "/inkling2_vision_tiny.safetensors"; }

/* The golden's image, as the block takes it: [side * patch, side * patch * 3]. */
const float* golden_image() { return kGoldInkVisImage; }

/* Largest absolute difference between two float arrays. */
double max_abs_diff(const float* a, const float* b, int64_t n) {
    double m = 0.0;
    for (int64_t i = 0; i < n; ++i) m = std::max(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}

constexpr double kTol = 2e-4;  /* f32 end to end; the golden is f32 torch, the ops are libref f32 */

}  // namespace

/* ================================================================== cases */

TEST(vision_matches_golden) {
    Weights w;
    REQUIRE(w.open(ckpt_path()));
    Geometry g;
    TowerIn in;
    REQUIRE(read_tower(w, &g, &in));
    Packed P;
    REQUIRE_EQ(pack(g, in, &P), RAD_OK);
    Workspace ws;
    REQUIRE_EQ(ws.init(g, kGoldInkVisSide, kGoldInkVisSide), RAD_OK);

    const int64_t cells = kGoldInkVisTokens;
    const int64_t n = (int64_t)kGoldInkVisSide * kGoldInkVisSide;
    std::vector<float> tokens((size_t)(n * g.width));
    std::vector<float> out((size_t)(cells * g.text));
    REQUIRE_EQ(forward(P, ws, launch_ref, nullptr, golden_image(), kGoldInkVisImageH, kGoldInkVisImageW,
                       tokens.data(), out.data()),
               RAD_OK);

    /* The tower's tokens first, so a failure names the stage that broke: the ViT or the projector. */
    CHECK_NEAR(max_abs_diff(tokens.data(), kGoldInkVisTokensOut, n * g.width), 0.0, kTol);
    CHECK_NEAR(max_abs_diff(out.data(), kGoldInkVisOut, cells * g.text), 0.0, kTol);
}

TEST(vision_tokens_are_the_golden_grid) {
    /* The golden's own shapes: side^2 tokens of width, side^2 / merge^2 cells of 4 width. */
    CHECK_EQ(kGoldInkVisSide * kGoldInkVisSide / (kGoldInkVisMerge * kGoldInkVisMerge), kGoldInkVisTokens);
    CHECK_EQ(kGoldInkVisImageH, kGoldInkVisSide * kGoldInkVisPatch);
    CHECK_EQ(kGoldInkVisImageW, kGoldInkVisSide * kGoldInkVisPatch);
}

/* The table resampled to a grid other than its own, against the float32 golden of the same
 * reference (posemb_resample, then the tower on the resampled table). The tokens check localises a
 * failure to the tower, the output to the projector. */
TEST(vision_resamples_the_table_to_another_grid) {
    Weights w;
    REQUIRE(w.open(ckpt_path()));
    Geometry g;
    TowerIn in;
    REQUIRE(read_tower(w, &g, &in));
    Packed P;
    REQUIRE_EQ(pack(g, in, &P), RAD_OK);
    Workspace ws;
    REQUIRE_EQ(ws.init(g, kGoldInkVisRGridH, kGoldInkVisRGridW), RAD_OK);

    const int64_t n = (int64_t)kGoldInkVisRGridH * kGoldInkVisRGridW;
    const int64_t cells = n / (g.merge * g.merge);
    std::vector<float> tokens((size_t)(n * g.width));
    std::vector<float> out((size_t)(cells * g.text));
    REQUIRE_EQ(forward(P, ws, launch_ref, nullptr, kGoldInkVisRImage, kGoldInkVisRImageH, kGoldInkVisRImageW,
                       tokens.data(), out.data()),
               RAD_OK);
    CHECK_NEAR(max_abs_diff(tokens.data(), kGoldInkVisRTokensOut, n * g.width), 0.0, kTol);
    CHECK_NEAR(max_abs_diff(out.data(), kGoldInkVisROut, cells * g.text), 0.0, kTol);
}

/* A workspace is sized for one grid: an image on any other is refused, not run on a wrong buffer. */
TEST(vision_refuses_an_image_off_the_workspace_grid) {
    Weights w;
    REQUIRE(w.open(ckpt_path()));
    Geometry g;
    TowerIn in;
    REQUIRE(read_tower(w, &g, &in));
    Packed P;
    REQUIRE_EQ(pack(g, in, &P), RAD_OK);
    Workspace ws;
    REQUIRE_EQ(ws.init(g, kGoldInkVisSide, kGoldInkVisSide), RAD_OK);
    std::vector<float> out((size_t)(kGoldInkVisTokens * g.text));
    CHECK_EQ(forward(P, ws, launch_ref, nullptr, golden_image(), kGoldInkVisRImageH, kGoldInkVisRImageW,
                     nullptr, out.data()),
             RAD_E_SHAPE);
    /* One patch row short of the workspace's grid. */
    CHECK_EQ(forward(P, ws, launch_ref, nullptr, golden_image(), kGoldInkVisImageH - g.patch, kGoldInkVisImageW,
                     nullptr, out.data()),
             RAD_E_SHAPE);
}

TEST(vision_workspace_refuses_an_odd_grid) {
    Geometry g;
    g.depth = 2; g.width = 64; g.heads = 4; g.patch = 4; g.side = 6; g.merge = 2; g.proj = 256; g.text = 48;
    Workspace ws;
    CHECK_EQ(ws.init(g, 7, 6), RAD_E_SHAPE);  /* an odd row count cannot merge 2 x 2 */
    CHECK_EQ(ws.init(g, 8, 10), RAD_OK);
}

TEST(vision_pack_refuses_a_bad_geometry) {
    Geometry g;
    g.depth = 2; g.width = 64; g.heads = 3; g.patch = 4; g.side = 6; g.merge = 2; g.proj = 256; g.text = 48;
    TowerIn in;
    Packed P;
    CHECK_EQ(pack(g, in, &P), RAD_E_SHAPE);  /* 64 is not a multiple of 3 heads */
    g.heads = 4;
    g.merge = 4;
    CHECK_EQ(pack(g, in, &P), RAD_E_SHAPE);  /* the merge is 2 in this tower */
}

RAD_TEST_MAIN()
