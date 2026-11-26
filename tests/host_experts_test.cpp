/* host_experts_test.cpp -- Qwen3.8-Flash-Next's routed feed-forward, declared for a CPU-only engine
 * against libavx alone (`--kernels libavx` on the host backend).
 *
 * The block is arch/common/rad_block_moe_q.h with the connection around it
 * (arch/common/rad_block_hc.h), declared as qwen4exp_fp8 declares them, at a small geometry with
 * the served container's structure and encodings (data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe):
 * the connection's read writing the rotated E4M3 codes and the int8 plain codes, the router, its
 * top-k and the sort with two protected experts, the codebook experts' grouped GEMMs, the rotated
 * gated quantiser between them, the protected experts' bf16 pair, the int8 shared expert and its
 * gate, the gather with the shared arm folded in, and the connection's write. Every op must resolve
 * to a libavx host row, and every codebook view of several planes must be stored in the joined
 * form both host libraries read (abi/rad_planes.h). The mixing matrices are plain bf16 here; their
 * E4M3 form is a separate weight's concern.
 */
#include "rad_test.h"
#include "rad_planes.h"

#include "build/rad_build.h"
#include "plugin/registry.h"
#include "rad_internal.h"

#include <arch/rad_block_hc.h>
#include <arch/rad_block_moe_q.h>

#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>

using namespace rad;
using namespace rad::arch;

namespace {

/* One layer at Flash-Next's structure: n_embd and the expert width whole 128-wide groups, eight
 * experts at top-k 2 with a shared expert as wide, four streams. */
constexpr int64_t kEmbd = 256, kFF = 128, kExperts = 8, kTopK = 2, kHC = 4, kLowRank = 32,
                  kTok = 64;
/* The protected experts, kept bf16 by the recipe: a pair, so the rest split by parity. */
bool protected_expert(long long e) { return e == 3 || e == 5; }

RadEncoding codebook(const char* dtype) {
    const bool w5 = !std::strcmp(dtype, "w5nl64a8h");
    const int64_t group = !std::strcmp(dtype, "w4nl32a8h") ? 32 : 64;
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_copy_str(e.transform, "fwht128");
    rad_enc_add_plane(&e, "codes", w5 ? RAD_U5 : RAD_U4, 1, 1);
    rad_enc_add_plane(&e, "scale", RAD_F8E4M3, 1, group);
    rad_enc_add_plane(&e, "scale.1", RAD_F32, 0, 0);
    rad_enc_add_table(&e, "table", RAD_F32, 1, w5 ? 32 : 16);
    return e;
}

/* The container's encodings by declared name; anything else is read as plain in its declared
 * dtype. */
int sources(const char* dtype, const std::string& name, WeightSource* out) {
    auto expert = [&](const char* prefix, bool gate_up) {
        const size_t n = std::strlen(prefix);
        if (name.compare(0, n, prefix) != 0) return false;
        const long long e = std::atoll(name.c_str() + n);
        out->rank = 2;
        out->shape[0] = gate_up ? 2 * kFF : kEmbd;
        out->shape[1] = gate_up ? kEmbd : kFF;
        out->enc = protected_expert(e) ? rad_enc_plain(RAD_BF16) : codebook(dtype);
        return true;
    };
    if (expert("blk.0.ffn_gate_up_exps.", true) || expert("blk.0.ffn_down_exps.", false))
        return RAD_OK;
    const bool gu = name == "blk.0.ffn_gate_up_shexp.weight";
    if (gu || name == "blk.0.ffn_down_shexp.weight") {
        out->rank = 2;
        out->shape[0] = gu ? 2 * kFF : kEmbd;
        out->shape[1] = gu ? kEmbd : kFF;
        out->enc = rad_enc_affine(RAD_I8, RAD_BF16, 1, RAD_SCALE_GROUP);
        return RAD_OK;
    }
    return RAD_E_NOTFOUND;
}

RadModelMeta meta() {
    RadModelMeta m{};
    m.arch_id  = "qwen4exp";
    m.name     = "host-experts";
    m.quant    = "";
    m.n_layers = 1;
    m.n_embd   = kEmbd;
    return m;
}

RadBuildCtx ctx() {
    RadBuildCtx c{};
    c.rank = 0;
    c.world_size = 1;
    c.max_tok  = kTok;
    c.max_seqs = 8;
    c.max_ctx  = 4096;
    c.scope = "";
    return c;
}

/* The connection and the block on `b`, as qwen4exp_fp8 declares one layer's feed-forward. */
int declare_layer(RadBuilder* b, MoeQ* mlp, HyperConn* hc) {
    Names nm;
    Geom g;
    g.n_layer = 1;
    g.n_embd = kEmbd;
    g.n_ff = g.n_ff_all = kFF;
    g.max_tok = kTok;
    g.max_seqs = 8;
    g.max_ctx = 4096;
    g.max_logit_rows = 8;
    g.dtype = "bf16";
    g.act_dtype = RAD_BF16;

    MoeQ::Config mc;
    mc.n_expert = mc.n_expert_all = kExperts;
    mc.top_k = kTopK;
    mc.n_ff_exp = kFF;
    mc.n_ff_shared = kFF;
    mc.rows = kTok;
    MoeQ::Src ms{};
    RAD_ARCH_TRY(moe_expert_sources(b, nm, "format", "model.layers.0.mlp.experts", &ms));
    RAD_ARCH_TRY(MoeQ::probe(b, nm, 0, ms, &mc));
    if (!mc.expert_rot || !mc.expert_nl) return RAD_E_DTYPE;

    const int64_t T = kTok, ER = mc.rows * mc.top_k;
    ActQ x, hr, eff, sff;
    x.x = decl_b(b, nm.f("x"), RAD_BF16, { T, kEmbd });
    RAD_ARCH_TRY(x.declare_qs(b, nm, g, "x", kEmbd, 0, true));
    x.q8_fed = true;
    RAD_ARCH_TRY(hr.declare_qs(b, nm, g, "moe_hr", kEmbd, 0, true, false, RAD_SCALE_GROUP));
    hr.q8_fed = hr.q8_rot = true;
    eff.x = decl_b(b, nm.f("moe_eff"), RAD_BF16, { ER, kFF });
    RAD_ARCH_TRY(eff.declare_qs(b, nm, g, "moe_eff", kFF, ER, true, false, RAD_SCALE_GROUP));
    eff.q8_fed = eff.q8_rot = true;
    sff.x = decl_b(b, nm.f("moe_sff"), RAD_BF16, { T, kFF });
    RAD_ARCH_TRY(sff.declare_qs(b, nm, g, "moe_sff", kFF, 0, true));
    sff.q8_fed = true;

    HyperConn::Wire hw{};
    hw.h = decl_b(b, nm.f("h_wide"), RAD_BF16, { T, kHC * kEmbd });
    hw.x = x;
    hw.inj = decl_b(b, nm.f("hc_inj"), RAD_BF16, { T, kHC });
    hw.rq = hr.cq();
    hw.rs = hr.cs();
    HyperConn::Config hcfg;
    hcfg.hc = kHC;
    hcfg.lowrank = kLowRank;
    hcfg.rotate = 1;
    hcfg.codes_i8 = 1;
    HyperConn::Src hs{};
    hs.norm     = nm.ckpt("model.layers.0.mlp_hyper_connection.hc_norm.weight");
    hs.mix_down = nm.ckpt("model.layers.0.mlp_hyper_connection.input_mix_weight_down.weight");
    hs.mix_up   = nm.ckpt("model.layers.0.mlp_hyper_connection.input_mix_weight_up.weight");
    hs.inject_w = nm.ckpt("model.layers.0.mlp_hyper_connection.block_inject_weight.weight");
    RAD_ARCH_TRY(hc->declare(b, nm, g, "blk.0.ffn_hc", hcfg, hw, hs));

    MoeQ::Wire mw{};
    mw.x = hw.h;
    mw.h = x;
    mw.hr = hr;
    mw.logits = decl_b(b, nm.f("moe_logits"), RAD_BF16, { T, kExperts });
    mw.ids    = decl_b(b, nm.f("moe_ids"), RAD_I32, { T, kTopK });
    mw.ew     = decl_b(b, nm.f("moe_ew"), RAD_BF16, { T, kTopK });
    mw.eoff   = decl_b(b, nm.f("moe_eoff"), RAD_I32, { kExperts + 1 });
    mw.ecnt   = decl_b(b, nm.f("moe_ecnt"), RAD_I32, { kExperts }, RAD_BUF_PERSIST);
    mw.sorted = decl_b(b, nm.f("moe_sorted"), RAD_I32, { ER });
    mw.egu    = decl_b(b, nm.f("moe_egu"), RAD_BF16, { ER, 2 * kFF });
    mw.eff    = eff;
    mw.edn    = decl_b(b, nm.f("moe_edn"), RAD_BF16, { ER, kEmbd });
    mw.moe    = decl_b(b, nm.f("moe_out"), RAD_BF16, { T, kEmbd });
    mw.sgu    = decl_b(b, nm.f("moe_sgu"), RAD_BF16, { T, 2 * kFF });
    mw.sff    = sff;
    mw.sout   = decl_b(b, nm.f("moe_sout"), RAD_BF16, { T, kEmbd });
    mw.sgate  = decl_b(b, nm.f("moe_sgate"), RAD_BF16, { T, 1 });
    ms.cfg = mc;
    ms.router = nm.ckpt("model.layers.0.mlp.gate.weight");
    ms.shared = nm.ckpt("model.layers.0.mlp.shared_expert");
    ms.shared_gate = nm.ckpt("model.layers.0.mlp.shared_expert_gate.weight");
    mlp->hr_in = true;
    return mlp->declare(b, nm, g, 0, mw, ms, false, false, kArNone, false);
}

}  /* namespace */

/* Every op of the layer resolves to libavx with no other kernel library loaded, at each codebook
 * dtype, and each codebook view of several planes is stored joined. */
TEST(flash_next_experts_declare_on_libavx_alone) {
    Registry reg;
    const int st = reg.load_dir(home() + "/kernels", { "libavx" });
    if (st == RAD_E_IO || !reg.plugin("libavx")) ARTIFACT_MISSING(home() + "/kernels/libavx.so");
    /* The routed arm at int8 activations writes its gated product with gated_had_quant_i8g, which
     * libavx has no row for. */
    if (!reg.schema("gated_had_quant_i8g"))
        SKIP("libavx declares no gated_had_quant_i8g, the int8 routed arm's gated quantiser");
    for (const char* dtype : { "w4nl64a8h", "w4nl32a8h" }) {
        Builder b(reg, meta(), ctx());
        b.set_sources([dtype](const std::string& name, const Program&, WeightSource* out) {
            return sources(dtype, name, out);
        });
        MoeQ mlp;
        HyperConn hc;
        std::string log;
        int rc = RAD_OK;
        log = stderr_of([&] {
            rc = declare_layer(&b, &mlp, &hc);
            if (rc == RAD_OK) rc = b.finish();
        });
        if (rc != RAD_OK || !b.errors().empty()) {
            fprintf(stderr, "    %s: declare %d\n%s", dtype, rc, log.c_str());
            for (const auto& e : b.errors()) fprintf(stderr, "    %s\n", e.c_str());
        }
        REQUIRE_EQ(rc, RAD_OK);
        CHECK(b.errors().empty());

        /* Every band of every op on a libavx host row. */
        std::set<std::string> ops;
        const Program& P = b.program();
        for (size_t i = 1; i < P.ops.size(); ++i) {
            const OpInfo& o = P.ops[i];
            ops.insert(o.op);
            for (const Band& band : o.bands) {
                const Resolved& r = band.dom[RAD_DOMAIN_HOST];
                if (!r.row || r.row->plugin != "libavx") {
                    fprintf(stderr, "    %s: op '%s' has no libavx host row\n", dtype,
                            o.op.c_str());
                    CHECK(false);
                }
            }
        }
        for (const char* want : { "hc_read", "hc_write", "gemm_nt", "router_topk", "moe_scatter",
                                  "moe_gemm_q", "silu_mul", "moe_gemm",
                                  "gemm_nt_q", "moe_gather" })
            if (!ops.count(want)) {
                fprintf(stderr, "    %s: the layer declares no '%s'\n", dtype, want);
                CHECK(false);
            }

        /* Each codebook view of several planes joined; the protected experts' and a single
         * plane's stored as they are. */
        int joined = 0;
        for (size_t h = 1; h < P.weights.size(); ++h) {
            const WeightInfo& w = P.weights[h];
            if (w.name.find("_exps.") == std::string::npos) continue;
            if (w.n_sel > 1) {
                CHECK_EQ(w.layout_tag, std::string(RAD_PLANES_TAG));
                CHECK_EQ(w.lay_dom, (int)RAD_DOMAIN_HOST);
                ++joined;
            } else {
                CHECK(w.layout_tag.empty());
            }
        }
        /* Six codebook experts, codes view and scale view, both projections. */
        CHECK_EQ(joined, 4 * 6);
    }
}

RAD_TEST_MAIN()
