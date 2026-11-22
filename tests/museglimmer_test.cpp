/* museglimmer_test.cpp -- the declared graph and the steps of Meta's Muse Glimmer plugin: the
 * trunk (gated attention alternating sliding-window rotary and NoPE full layers under Gemma-2
 * style double norms), its refusals, the InternVit-shaped vision tower and its adapter, and the
 * scatter that puts the tower's rows into the trunk's embeddings. Asserted statically against the
 * recording builder of arch_fakes.h, per op and per operand. */
#define RAD_ARCH_NO_EXPORTS 1

#include "arch_fakes.h"

#include "../arch/museglimmer/museglimmer.cpp"
#include "../arch/museglimmer_ct/museglimmer_ct.cpp"

#include <vector>

/* ==================================================================== fixtures */

namespace {

/* The declared handle of a weight or buffer recorded by the fake builder. */
template <class R>
int64_t handle_of(const std::vector<R>& v, const char* name) {
    for (size_t i = 0; i < v.size(); ++i)
        if (v[i].name == name) return (int64_t)(i + 1);
    return 0;
}
int64_t w_h(const RadBuilder& b, const char* name) { return handle_of(b.weights, name); }
int64_t b_h(const RadBuilder& b, const char* name) { return handle_of(b.bufs, name); }

/* Doubles the plugin read through float config values: compared with a tolerance, not ==. */
bool near(double a, double b, double tol) { return std::abs(a - b) < tol; }

/* The trunk's config, at a fifth of the 30B's width and eight layers, with the family's free-form
 * keys under both spellings rad-convert flattens to (text_config.X and the bare alias). The
 * layer schedule is the 30B's pattern at eight layers: full attention on layers 3 and 7, the
 * rest sliding with rotary. */
MetaKV muse_meta() {
    RadModelMeta m{};
    m.n_layers = 8;
    m.n_embd = 128;
    m.n_head = 4;
    m.n_head_kv = 2;
    m.head_dim = 8;
    m.n_ff = 256;
    m.n_vocab = 512;
    m.rms_eps = 1e-5f;
    m.rope_theta = 500000.0;
    MetaKV k(m);
    for (const char* pre : { "", "text_config." }) {
        k.set((std::string(pre) + "sliding_window").c_str(), "4")
            .set((std::string(pre) + "qk_scale_factor").c_str(), "3.87")
            .set((std::string(pre) + "post_norm_eps").c_str(), "1e-8")
            .set((std::string(pre) + "output_multiplier").c_str(), "0.19611613513818404")
            .set((std::string(pre) + "final_logit_softcapping").c_str(), "20")
            .set((std::string(pre) + "rope_parameters.rope_theta").c_str(), "500000")
            .set((std::string(pre) + "rope_parameters.rope_type").c_str(), "default")
            .set((std::string(pre) + "layer_types").c_str(),
                 "sliding_attention sliding_attention sliding_attention full_attention "
                 "sliding_attention sliding_attention sliding_attention full_attention")
            .set((std::string(pre) + "layer_rope_theta").c_str(),
                 "500000 500000 500000 0 500000 500000 500000 0");
    }
    return k;
}

/* The tower, at a quarter of the release's width in four blocks: head_dim 12, so the axial rotary
 * runs sections 3 and 3; the position grid 4 x 4, so the windows are 4 x 4 patches. Everything the
 * tower reads is a config key, so the tiny geometry is served by the same code paths. The family
 * rule puts a tower's full-attention layer on every fourth from the first, plus the last: layer 3
 * here. */
MetaKV with_vision(MetaKV k) {
    k.set("vision_config.model_type", "muse_glimmer_vision")
        .set("vision_config.num_hidden_layers", "4")
        .set("vision_config.hidden_size", "96")
        .set("vision_config.num_attention_heads", "8")
        .set("vision_config.intermediate_size", "256")
        .set("vision_config.patch_size", "14")
        .set("vision_config.patch_temporal", "2")
        .set("vision_config.merge_size", "2")
        .set("vision_config.pos_emb_height", "4")
        .set("vision_config.pos_emb_width", "4")
        .set("vision_config.layer_norm_eps", "1e-05")
        .set("vision_config.hidden_act", "gelu")
        .set("vision_config.rope_parameters.rope_type", "axial")
        .set("vision_config.rope_parameters.rope_theta", "10000")
        .set("vision_config.layer_types",
             "window_attention window_attention window_attention full_attention")
        .set("projector_hidden_size", "128")
        .set("projector_hidden_act", "gelu")
        .set("out_hidden_size", "384");   /* merge 2 squared x width 96 */
    return k;
}

RadBuildCtx muse_ctx(int rank = 0, int world = 1, int64_t patches = 0) {
    RadBuildCtx c = build_ctx(rank, world);
    c.max_enc_patches = patches;
    return c;
}

/* One trunk layer's declared ops, in order: the double-norm block with its four norms, the gated
 * attention, and the rotary only where the layer carries one. */
std::vector<std::string> layer_ops(bool rope, bool ar) {
    std::vector<std::string> v = { "rmsnorm", "gemm_nt", "gemm_nt", "gemm_nt", "gemm_nt",
                                   "rmsnorm", "rmsnorm" };
    if (rope) { v.push_back("rope"); v.push_back("rope"); }
    for (const char* s : { "attn_paged", "kv_store", "sigmoid", "mul", "gemm_nt" })
        v.push_back(s);
    if (ar) v.push_back("all_reduce");
    for (const char* s : { "rmsnorm", "add", "rmsnorm", "gemm_nt", "silu_mul", "gemm_nt" })
        v.push_back(s);
    if (ar) v.push_back("all_reduce");
    for (const char* s : { "rmsnorm", "add" }) v.push_back(s);
    return v;
}

/* The trunk's ops as DECLARED, from the embedding to the capped logits (attn_paged before
 * kv_store, the family's declare convention). */
std::vector<std::string> trunk_decl(int n_layer, int world) {
    std::vector<std::string> want = { "embed_lookup" };
    if (world > 1) want.push_back("all_reduce");
    want.push_back("rmsnorm");                       /* the weightless embedding norm */
    for (int l = 0; l < n_layer; ++l) {
        const bool glob = l % 4 == 3;                /* is_full_attention(l, 4) at 8 layers */
        for (const std::string& s : layer_ops(!glob, world > 1)) want.push_back(s);
    }
    want.push_back("rmsnorm");                       /* the final norm */
    want.push_back("gather_rows");
    want.push_back("logits_gemm");
    want.push_back("logit_softcap");
    return want;
}

/* The trunk's ops as ISSUED: kv_store runs before its layer's attention. */
std::vector<std::string> trunk_step(int n_layer, int world) {
    std::vector<std::string> want = trunk_decl(n_layer, world);
    for (size_t i = 0; i + 1 < want.size(); ++i)
        if (want[i] == "attn_paged" && want[i + 1] == "kv_store")
            std::swap(want[i], want[i + 1]);
    return want;
}

}  /* namespace */

/* ==================================================================== 1. the text graph */

/* Per layer the exact op row the family runs, the two KV groups with every layer bound to exactly
 * one, the weightless norms, the logits' pre-scale and cap, and the gate/up concat in the
 * checkpoint's row order. */
TEST(museglimmer_declares_the_trunk_graph) {
    MetaKV mk = muse_meta();
    for (int world : { 1, 4 }) {
        const int rank = world - 1;
        RadBuilder b;
        RadBuildCtx ctx = muse_ctx(rank, world);
        REQUIRE_EQ(museglimmer::declare(&b, mk.meta(), &ctx), RAD_OK);

        /* Two KV groups: the sliding layers' window pool and the full layers' pool. */
        REQUIRE_EQ(b.kv_groups.size(), (size_t)2);
        CHECK_EQ(b.kv_groups[0].name, std::string("kv_sliding"));
        CHECK_EQ(b.kv_groups[0].d.kind, RAD_KV_WINDOW);
        CHECK_EQ(b.kv_groups[0].d.window, 4);
        CHECK_EQ(b.kv_groups[0].d.n_head_kv, world == 1 ? 2 : 1);
        CHECK_EQ(b.kv_groups[0].d.head_dim, 8);
        CHECK_EQ(b.kv_groups[1].name, std::string("kv_full"));
        CHECK_EQ(b.kv_groups[1].d.kind, RAD_KV_FULL);
        CHECK_EQ(b.kv_groups[1].d.n_head_kv, world == 1 ? 2 : 1);
        /* Every layer bound exactly once, to its group. */
        REQUIRE_EQ(b.binds.size(), (size_t)8);
        const rad_kvgroup win = 1, full = 2;
        for (int l = 0; l < 8; ++l) {
            CHECK_EQ(b.binds[(size_t)l].first, l);
            CHECK_EQ(b.binds[(size_t)l].second, (rad_kvgroup)((l % 4 == 3) ? full : win));
        }

        /* The whole graph, in declare order. */
        CHECK_EQ(join(op_names(b)), join(trunk_decl(8, world)));

        /* The embedding norm is weightless: no gain operand, no centered 1+. */
        const RecOp* en = find_op(b, "rmsnorm");
        REQUIRE(en != nullptr);
        CHECK(en->w.empty());
        CHECK(param(en, "wadd") == nullptr);
        CHECK(near(pflt(en, "eps"), 1e-5, 1e-12));
        CHECK_EQ(pint(en, "n"), 128);

        /* Layer 0 (sliding) end to end. The block's input norm carries the centered 1+. */
        const size_t at = (size_t)(std::find_if(b.ops.begin(), b.ops.end(),
                                                [](const RecOp& o) {
                                                    return o.op == "rmsnorm" && o.w.size() == 1;
                                                }) - b.ops.begin());
        REQUIRE(at + 9 < b.ops.size());
        const RecOp* norm = &b.ops[at];
        CHECK_EQ(norm->w.size(), (size_t)1);
        CHECK_EQ(norm->w[0], w_h(b, "blk.0.attn_norm.weight"));
        CHECK(near(pflt(norm, "wadd"), 1.0, 1e-12));
        CHECK(near(pflt(norm, "eps"), 1e-5, 1e-12));
        const RecOp* q = &b.ops[at + 1];
        CHECK_EQ(q->op, std::string("gemm_nt"));
        CHECK_EQ(pint(q, "N"), world == 1 ? 32 : 8);       /* q_dim, the rank's heads */
        CHECK_EQ(pint(q, "K"), 128);
        CHECK(q->w == std::vector<rad_weight>({ w_h(b, "blk.0.attn_q.weight") }));
        /* The q/k norms are weightless; the query carries qk_scale_factor. */
        CHECK_EQ(pint(&b.ops[at + 5], "n"), 8);
        CHECK(near(pflt(&b.ops[at + 5], "mult"), 3.87, 1e-6));
        CHECK(b.ops[at + 5].w.empty());
        CHECK_EQ(pint(&b.ops[at + 6], "n"), 8);
        CHECK(param(&b.ops[at + 6], "mult") == nullptr);
        CHECK(b.ops[at + 6].w.empty());
        /* The rotary, on the sliding layers only, at the sliding theta. */
        const RecOp* rq = find_op(b, "rope");
        REQUIRE(rq != nullptr);
        CHECK(near(pflt(rq, "theta"), 500000.0, 1e-9));
        CHECK_EQ(pstr(rq, "mode"), std::string("neox"));
        CHECK_EQ(pint(rq, "rotary_dim"), 8);
        CHECK_EQ(pint(rq, "head_dim"), 8);
        /* The paged attention: a window over the sliding pool, whole-context over the full. */
        const RecOp* sl_attn = find_op(b, "attn_paged");
        REQUIRE(sl_attn != nullptr);
        CHECK_EQ(pint(sl_attn, "window"), 4);
        CHECK_EQ(pint(sl_attn, "gqa"), world == 1 ? 2 : 1);
        CHECK_EQ(pint(sl_attn, "causal"), 1);
        int seen = 0;
        for (const auto& o : b.ops) {
            if (o.op != "attn_paged") continue;
            if (seen++ == 3) CHECK_EQ(pint(&o, "window"), 0);
        }
        CHECK_EQ(seen, 8);
        /* Four norms a layer: the two inputs at rms_norm_eps with the centered 1+, the two posts
         * at post_norm_eps with it. */
        int post8 = 0;
        for (const auto& o : b.ops)
            if (o.op == "rmsnorm" && near(pflt(&o, "eps"), 1e-8, 1e-12)) {
                CHECK(near(pflt(&o, "wadd"), 1.0, 1e-12));
                ++post8;
            }
        CHECK_EQ(post8, 16);
        /* The concat map rows gate then up, and the weight's two row parts say so. */
        const RecMap* gu = find_map(b, "blk.0.ffn_gate_up.weight");
        REQUIRE(gu != nullptr);
        CHECK_EQ(gu->mode, (int)RAD_MAP_CONCAT);
        REQUIRE_EQ(gu->src.size(), (size_t)2);
        CHECK_EQ(gu->src[0], std::string("model.language_model.layers.0.mlp.gate_proj.weight"));
        CHECK_EQ(gu->src[1], std::string("model.language_model.layers.0.mlp.up_proj.weight"));
        const RecWeight* guw = find_w(b, "blk.0.ffn_gate_up.weight");
        REQUIRE(guw != nullptr);
        REQUIRE_EQ(guw->d.n_row_parts, 2);
        CHECK_EQ(guw->d.row_parts[0], 256 / world);   /* the rank's feed-forward width */
        CHECK_EQ(guw->d.row_parts[1], 256 / world);
        CHECK_EQ(count_op(b, "silu_mul"), 8);
        /* The final norm: a plain gain (wadd 0) carrying the logits' pre-scale. The embedding
         * norm is the first rmsnorm, six a layer follow, the last is this one. */
        const RecOp* fn = find_op(b, "rmsnorm", 49);
        REQUIRE(fn != nullptr);
        CHECK(near(pflt(fn, "wadd"), 0.0, 1e-12));
        CHECK(near(pflt(fn, "mult"), 0.19611613513818404, 1e-6));
        /* The head and its cap. */
        const RecOp* cap = find_op(b, "logit_softcap");
        REQUIRE(cap != nullptr);
        CHECK(near(pflt(cap, "cap"), 20.0, 1e-12));
        CHECK_EQ(pstr(cap, "dtype"), std::string("f32"));
    }
}

/* ==================================================================== 2. the text steps */

/* A decode batch and a prefill chunk: the issue order, and the head over the sampled rows only. */
TEST(museglimmer_issues_decode_and_prefill_in_order) {
    MetaKV mk = muse_meta();
    for (int world : { 1, 4 }) {
        RadBuilder b;
        RadBuildCtx ctx = muse_ctx(0, world);
        REQUIRE_EQ(museglimmer::declare(&b, mk.meta(), &ctx), RAD_OK);

        auto dec = kit(b, { 1, 1, 1, 1 }, {});
        auto v = run_passes(b, museglimmer::step, { dec.get() }, 0, world);
        CHECK_PASSES(b, v);
        CHECK_EQ(join(issued_ops(b, *v[0])), join(trunk_step(8, world)));
        /* The head ran over the four sampled rows, not the step's rows. */
        for (const char* name : { "gather_rows", "logits_gemm", "logit_softcap" }) {
            const RecIssue* is = issue_of(b, *v[0], name);
            REQUIRE(is != nullptr);
            CHECK_EQ(is->n, 4);
        }
        CHECK_EQ(count_issued(issued_ops(b, *v[0]), "attn_paged"), 8);
        CHECK_EQ(count_issued(issued_ops(b, *v[0]), "kv_store"), 8);

        /* A prefill chunk of two 8-token sequences: two sampled rows. */
        auto pf = kit(b, {}, { 8, 8 });
        auto p = run_passes(b, museglimmer::step, { pf.get() }, 0, world);
        CHECK_PASSES(b, p);
        const RecIssue* pg = issue_of(b, *p[0], "gather_rows");
        REQUIRE(pg != nullptr);
        CHECK_EQ(pg->n, 2);
        CHECK_EQ(count_issued(issued_ops(b, *p[0]), "logits_gemm"), 1);
    }
}

/* ==================================================================== 3. refusals */

/* A layer the schedule rule disagrees with, a rotary table of the wrong length, and a
 * non-positive query scale are refused by name, not built into fluent wrong text. A missing
 * qk_scale_factor takes the reference config's own default (3.87), which is the family. */
TEST(museglimmer_refuses_a_trunk_it_does_not_serve) {
    const struct { const char* key; const char* val; int rc; } kCases[] = {
        /* layer 0 full, where the interval rule puts a window layer */
        { "text_config.layer_types",
          "full_attention sliding_attention sliding_attention full_attention sliding_attention "
          "sliding_attention sliding_attention full_attention", RAD_E_UNSUPPORTED },
        /* seven rotary bases for eight layers */
        { "text_config.layer_rope_theta", "500000 500000 500000 0 500000 500000 500000",
          RAD_E_INVAL },
        { "text_config.qk_scale_factor", "0", RAD_E_INVAL },
        /* a vocabulary the trunk does not know */
        { "text_config.layer_types",
          "window_attention sliding_attention sliding_attention full_attention sliding_attention "
          "sliding_attention sliding_attention full_attention", RAD_E_UNSUPPORTED },
    };
    for (const auto& kc : kCases) {
        MetaKV mk = muse_meta();
        mk.set(kc.key, kc.val);
        RadBuilder b;
        RadBuildCtx ctx = muse_ctx(0, 1);
        const int rc = stderr_of_rc([&] { return museglimmer::declare(&b, mk.meta(), &ctx); });
        if (rc != kc.rc) fprintf(stderr, "    %s = %s: rc %d\n", kc.key, kc.val, rc);
        CHECK_EQ(rc, kc.rc);
    }
    /* A routed model is not this family. */
    MetaKV mk = muse_meta();
    RadModelMeta routed = *mk.meta();
    routed.n_expert = 8;
    RadBuilder b;
    RadBuildCtx ctx = muse_ctx(0, 1);
    const int rc = stderr_of_rc([&] { return museglimmer::declare(&b, &routed, &ctx); });
    CHECK_EQ(rc, RAD_E_UNSUPPORTED);
}

/* ==================================================================== 4. the media declare */

/* The tower is declared with the deployment's encoder budget, last, its weights replicated with
 * the blocks in their own groups; the full-attention layer bands on the whole pass and the window
 * layers on one window; the adapter folds the merge with a view and two activations. */
TEST(museglimmer_declares_its_vision_tower_when_the_engine_asks_for_it) {
    MetaKV mk = with_vision(muse_meta());
    /* Without a budget, nothing: the trunk's graph alone. The probe still names the merge, so
     * --mm-max-patches auto rounds to whole merged rows. */
    {
        RadBuilder b;
        RadBuildCtx ctx = muse_ctx(0, 1, 0);
        REQUIRE_EQ(museglimmer::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK(b.encoder_name.empty());
        CHECK(find_w(b, "v.patch.weight") == nullptr);
        CHECK_EQ(count_op(b, "scatter_rows"), 0);
        CHECK_EQ(count_op(b, "window_order"), 0);
        RadArchProbe p{};
        CHECK_OK(museglimmer::probe(mk.meta(), &p));
        CHECK_EQ(p.enc_merge, 4);
        RadArchProbe none{};
        CHECK_OK(museglimmer::probe(muse_meta().meta(), &none));
        CHECK_EQ(none.enc_merge, 0);
    }
    RadBuilder b;
    RadBuildCtx ctx = muse_ctx(0, 1, 64);
    REQUIRE_EQ(museglimmer::declare(&b, mk.meta(), &ctx), RAD_OK);

    CHECK_EQ(b.encoder_name, std::string("vision"));
    CHECK_EQ(b.encoder.modalities, (uint32_t)(RAD_MM_IMAGE | RAD_MM_VIDEO));
    CHECK_EQ(b.encoder.patch_dim, 1176);      /* 3 x patch_temporal 2 x patch 14 squared */
    CHECK_EQ(b.encoder.merge, 4);
    CHECK_EQ(b.encoder.max_patches, 64);
    CHECK_EQ(b.encoder.n_embd, 128);
    const RecBuf* out = find_b(b, "v.out");
    REQUIRE(out != nullptr);
    CHECK_EQ(out->d.shape[0], 16);            /* 64 patches / merge 2 squared */
    CHECK_EQ(out->d.shape[1], 128);
    CHECK_EQ(count_op(b, "scatter_rows"), 1);

    /* The tower's weights, replicated: every one mapped from its checkpoint name, blocks in their
     * own groups past the trunk's. */
    for (const auto& w : b.weights)
        if (w.name.rfind("v.", 0) == 0) {
            const RecMap* mp = find_map(b, w.name);
            REQUIRE(mp != nullptr);
            for (const auto& s : mp->src) {
                const bool tower = s.rfind("model.vision_tower.", 0) == 0;
                const bool adapter = s.rfind("model.vision_adapter.", 0) == 0 ||
                                     s == "model.vision_projection.weight";
                CHECK(tower || adapter);
            }
            CHECK_EQ(w.d.shard, (int)RAD_SHARD_NONE);
            CHECK(w.d.group.layer == -1 || w.d.group.layer >= 8);
        }
    const RecWeight* patch = find_w(b, "v.patch.weight");
    REQUIRE(patch != nullptr);
    CHECK_EQ(patch->d.shape[0], 96);
    CHECK_EQ(patch->d.shape[1], 1176);
    const RecWeight* pos = find_w(b, "v.pos.weight");
    REQUIRE(pos != nullptr);
    CHECK_EQ(pos->d.shape[0], 16);            /* the 4 x 4 position table */
    CHECK_EQ(pos->d.shape[1], 96);
    /* The reference's three biased projections, joined q | k | v. */
    const RecMap* qkv = find_map(b, "v.blk.0.qkv.weight");
    REQUIRE(qkv != nullptr);
    REQUIRE_EQ(qkv->src.size(), (size_t)3);
    CHECK_EQ(qkv->src[0], std::string("model.vision_tower.layers.0.attn.q_proj.weight"));
    CHECK_EQ(qkv->src[1], std::string("model.vision_tower.layers.0.attn.k_proj.weight"));
    CHECK_EQ(qkv->src[2], std::string("model.vision_tower.layers.0.attn.v_proj.weight"));
    const RecWeight* qkvw = find_w(b, "v.blk.0.qkv.weight");
    REQUIRE(qkvw != nullptr);
    REQUIRE_EQ(qkvw->d.n_row_parts, 3);
    CHECK_EQ(qkvw->d.row_parts[0], 96);
    CHECK_EQ(qkvw->d.row_parts[1], 96);
    CHECK_EQ(qkvw->d.row_parts[2], 96);
    const RecWeight* a1 = find_w(b, "v.mlp.fc1.weight");
    REQUIRE(a1 != nullptr);
    CHECK_EQ(a1->d.shape[0], 128);
    CHECK_EQ(a1->d.shape[1], 384);            /* the merged width, the view's row */
    const RecWeight* a2 = find_w(b, "v.mlp.fc2.weight");
    REQUIRE(a2 != nullptr);
    CHECK_EQ(a2->d.shape[0], 128);
    CHECK_EQ(a2->d.shape[1], 128);
    const RecWeight* vp = find_w(b, "v.proj.weight");
    REQUIRE(vp != nullptr);
    CHECK_EQ(vp->d.shape[0], 128);            /* the model's width */
    CHECK_EQ(vp->d.shape[1], 128);
    const RecWeight* blk1 = find_w(b, "v.blk.1.norm1.weight");
    REQUIRE(blk1 != nullptr);
    CHECK_EQ(blk1->d.group.layer, 9);         /* the trunk's 8 groups, then the tower's */

    /* The graph after the trunk: the stem, the order op, four blocks, the un-permute, ln_post and
     * the adapter with its two activations. */
    std::vector<std::string> want = { "gemm_nt", "grid_embed_pad", "layernorm", "window_order",
                                      "gather_rows" };
    for (int l = 0; l < 4; ++l)
        for (const char* s : { "layernorm", "gemm_nt_bias", "rope", "attn_dense",
                               "gemm_nt_bias", "layernorm", "gemm_nt_bias", "gemm_nt_bias" })
            want.push_back(s);
    for (const char* s : { "gather_rows", "layernorm", "pixtral_merge", "gemm_nt", "gelu",
                           "gemm_nt", "gelu", "gemm_nt", "rmsnorm" })
        want.push_back(s);
    std::vector<const RecOp*> tail;
    bool trunk_done = false;
    for (const auto& o : b.ops) {
        if (!trunk_done && o.op == "logit_softcap") { trunk_done = true; continue; }
        if (trunk_done) tail.push_back(&o);
    }
    REQUIRE_EQ(tail.size(), want.size());
    for (size_t i = 0; i < want.size(); ++i)
        CHECK_EQ(tail[i]->op, want[i]);

    /* The stem: a bias-less patch GEMM and the padded position resample at ln_pre's epsilon. */
    CHECK_EQ(tail[0]->op, std::string("gemm_nt"));
    CHECK_EQ(pint(tail[0], "N"), 96);
    CHECK_EQ(pint(tail[0], "K"), 1176);
    CHECK(tail[0]->w == std::vector<rad_weight>({ w_h(b, "v.patch.weight") }));
    const RecOp* gep = tail[1];
    CHECK_EQ(gep->op, std::string("grid_embed_pad"));
    CHECK_EQ(pint(gep, "n"), 96);
    CHECK_EQ(pint(gep, "side"), 4);
    const RecOp* lpre = tail[2];
    CHECK_EQ(pint(lpre, "n"), 96);
    CHECK_EQ(pflt(lpre, "eps"), 1e-05);
    /* The order op: windows of 4 patches; it reads the batch's arrays and writes its four
     * outputs. */
    const RecOp* wo = tail[3];
    CHECK_EQ(pint(wo, "window"), 4);
    CHECK(wo->reads.empty());
    CHECK(wo->writes == std::vector<rad_buf>({ b_h(b, "v.idx"), b_h(b, "v.inv"), b_h(b, "v.pos"),
                                               b_h(b, "v.cuw") }));

    /* The rotary: axial over the whole head, sections 3 and 3, at the tower's theta. */
    int ropes = 0;
    for (const RecOp* o : tail)
        if (o->op == "rope") {
            ++ropes;
            CHECK_EQ(pstr(o, "mode"), std::string("axial"));
            CHECK_EQ(pstr(o, "sections"), std::string("3 3"));
            CHECK_EQ(pint(o, "head_dim"), 12);
            CHECK_EQ(pint(o, "rotary_dim"), 12);
            CHECK_EQ(pint(o, "n_head"), 8);
            CHECK_EQ(pint(o, "n_head_kv"), 8);
            CHECK_EQ(pflt(o, "theta"), 10000.0);
        }
    CHECK_EQ(ropes, 4);

    /* The attention: non-causal; the three window layers band on one window (4 x 4), the full
     * layer on the whole pass. */
    int win_attn = 0, full_attn = 0;
    for (const RecOp* o : tail)
        if (o->op == "attn_dense") {
            CHECK_EQ(pint(o, "causal"), 0);
            CHECK_EQ(pint(o, "gqa"), 1);
            if (pint(o, "max_seqlen") == 16) ++win_attn;
            if (pint(o, "max_seqlen") == 64) ++full_attn;
        }
    CHECK_EQ(win_attn, 3);
    CHECK_EQ(full_attn, 1);

    /* The blocks' MLP: exact-erf gelu in the first linear's epilogue; the adapter's two
     * activations are gelu ops of their own. */
    int block_gelu = 0, plain = 0;
    for (const RecOp* o : tail) {
        if (o->op == "gemm_nt_bias" && pstr(o, "act") == "gelu") ++block_gelu;
        if (o->op == "gemm_nt") ++plain;
    }
    CHECK_EQ(block_gelu, 4);
    CHECK_EQ(plain, 4);                       /* patch, adapter fc1 and fc2, projection */
    CHECK_EQ(count_op(b, "gelu"), 2);
    /* The perception norm: weightless, no centered 1+, at the text model's rms_norm_eps. */
    const RecOp* perc = tail.back();
    CHECK_EQ(perc->op, std::string("rmsnorm"));
    CHECK(perc->w.empty());
    CHECK(param(perc, "wadd") == nullptr);
    CHECK(near(pflt(perc, "eps"), 1e-5, 1e-12));
    CHECK_EQ(pint(perc, "n"), 128);
}

/* A tower whose config leaves the family is refused by name. */
TEST(museglimmer_refuses_a_vision_tower_it_does_not_serve) {
    const struct { const char* key; const char* val; int rc; } kCases[] = {
        { "vision_config.hidden_act", "gelu_pytorch_tanh", RAD_E_UNSUPPORTED },
        { "projector_hidden_act", "gelu_pytorch_tanh", RAD_E_UNSUPPORTED },
        { "vision_config.rope_parameters.rope_type", "linear", RAD_E_UNSUPPORTED },
        /* layer 0 full, where the family rule puts a window layer */
        { "vision_config.layer_types", "full_attention window_attention window_attention "
                                       "window_attention", RAD_E_UNSUPPORTED },
        /* three entries for four blocks */
        { "vision_config.layer_types", "window_attention window_attention window_attention",
          RAD_E_INVAL },
        /* the last layer window, where the rule puts the tower's one full layer past the thirds */
        { "vision_config.layer_types", "window_attention window_attention full_attention "
                                       "window_attention", RAD_E_UNSUPPORTED },
        /* the merge op folds a 2x2 block */
        { "vision_config.merge_size", "4", RAD_E_UNSUPPORTED },
        { "out_hidden_size", "256", RAD_E_INVAL },
        { "vision_config.pos_emb_width", "8", RAD_E_INVAL },
        /* the counts the declare walks and hosts-allocates by: a container naming a layer count
         * or grid past what a tower is must refuse, not throw a vector length out of bringup.
         * 129 is the first depth the bound refuses; the published maximum is 48. */
        { "vision_config.num_hidden_layers", "129", RAD_E_INVAL },
        { "vision_config.num_hidden_layers", "4611686018427387904", RAD_E_INVAL },
        { "vision_config.pos_emb_height", "4611686018427387904", RAD_E_INVAL },
    };
    for (const auto& kc : kCases) {
        MetaKV mk = with_vision(muse_meta());
        mk.set(kc.key, kc.val);
        RadBuilder b;
        RadBuildCtx ctx = muse_ctx(0, 1, 64);
        const int rc = stderr_of_rc([&] { return museglimmer::declare(&b, mk.meta(), &ctx); });
        if (rc != kc.rc) fprintf(stderr, "    %s = %s: rc %d\n", kc.key, kc.val, rc);
        CHECK_EQ(rc, kc.rc);
    }
}

TEST(museglimmer_refuses_a_position_grid_past_its_bounds) {
    /* One key at a time cannot pin the grid bound: pos_emb_height alone the height==width
     * rule refuses before the cap is consulted. Both sides together reach it -- 513 is the
     * first side the bound refuses, against a published maximum of 92. */
    MetaKV mk = with_vision(muse_meta());
    mk.set("vision_config.pos_emb_height", "513");
    mk.set("vision_config.pos_emb_width", "513");
    RadBuilder b;
    RadBuildCtx ctx = muse_ctx(0, 1, 64);
    CHECK_EQ(stderr_of_rc([&] { return museglimmer::declare(&b, mk.meta(), &ctx); }), RAD_E_INVAL);
}

/* ==================================================================== 5. the encoder step */

/* An encoder pass issues the tower and nothing else, over two 8 x 4 frames (two windows each);
 * the rotary reads the order op's positions and each attention its own segment bounds. */
TEST(a_museglimmer_encoder_pass_issues_the_tower_and_nothing_else) {
    MetaKV mk = with_vision(muse_meta());
    RadBuilder b;
    RadBuildCtx ctx = muse_ctx(0, 1, 64);
    REQUIRE_EQ(museglimmer::declare(&b, mk.meta(), &ctx), RAD_OK);
    static uint16_t pix[64 * 1176];
    static int32_t coord[4 * 64];
    for (int64_t i = 0; i < 64; ++i) {
        const int64_t in = i % 32;
        coord[i * 4 + 0] = (int32_t)(in / 4);        /* row */
        coord[i * 4 + 1] = (int32_t)(in % 4);        /* column */
        coord[i * 4 + 2] = 8;                        /* segment height in patches */
        coord[i * 4 + 3] = 4;                        /* segment width */
    }
    static const int32_t cu[] = { 0, 32, 64 };
    static const int32_t sh[] = { 8, 8 }, sw[] = { 4, 4 };
    RadBatch batch{};
    batch.enc         = 1;
    batch.enc_n_patch = 64;
    batch.enc_pixels  = pix;
    batch.enc_coord   = coord;
    batch.enc_n_seg   = 2;
    batch.enc_cu      = cu;
    batch.enc_seg_h   = sh;
    batch.enc_seg_w   = sw;
    batch.enc_max_seg = 32;
    RadCtx c{ .b = &b, .batch = &batch };
    c.rank = 0;
    c.world = 1;
    museglimmer::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());

    /* The issue order is the declare order: the stem, the order op, four blocks, the un-permute,
     * ln_post and the adapter. */
    std::vector<std::string> want = { "gemm_nt", "grid_embed_pad", "layernorm", "window_order",
                                      "gather_rows" };
    for (int l = 0; l < 4; ++l)
        for (const char* s : { "layernorm", "gemm_nt_bias", "rope", "attn_dense", "gemm_nt_bias",
                               "layernorm", "gemm_nt_bias", "gemm_nt_bias" })
            want.push_back(s);
    for (const char* s : { "gather_rows", "layernorm", "pixtral_merge", "gemm_nt", "gelu",
                           "gemm_nt", "gelu", "gemm_nt", "rmsnorm" })
        want.push_back(s);
    CHECK_EQ(join(issued_ops(b, c)), join(want));
    /* ln_post and the merge run over all 64 patches; the adapter, projection and perception norm
     * over the 16 merged rows. */
    CHECK_EQ(c.issues[c.issues.size() - 8].n, 64);
    CHECK_EQ(c.issues[c.issues.size() - 7].n, 64);
    CHECK_EQ(c.issues[c.issues.size() - 7].opd[2].rows, 16);   /* the merge's output */
    CHECK_EQ(c.issues[c.issues.size() - 2].n, 16);
    CHECK_EQ(c.issues.back().n, 16);

    /* The order op reads the [4, P] coordinates and the frame bounds, and writes the four
     * outputs; the rotary reads its positions. */
    const RecIssue* wo = issue_of(b, c, "window_order");
    REQUIRE(wo != nullptr);
    CHECK_EQ(wo->n, 64);
    CHECK(wo->opd[0].raw == coord);
    CHECK_EQ(wo->opd[0].rows, 4);
    CHECK_EQ(wo->opd[0].cols, 64);
    CHECK(wo->opd[1].raw == cu);
    CHECK_EQ(wo->opd[1].rows, 3);
    CHECK_EQ(wo->opd[2].rows, 64);
    CHECK_EQ(wo->opd[2].handle, b_h(b, "v.idx"));
    CHECK_EQ(wo->opd[3].handle, b_h(b, "v.inv"));
    CHECK_EQ(wo->opd[4].handle, b_h(b, "v.pos"));
    CHECK_EQ(wo->opd[5].handle, b_h(b, "v.cuw"));
    CHECK_EQ(wo->opd[5].rows, 5);                      /* four windows, plus the leading 0 */
    const RecIssue* rp = issue_of(b, c, "rope");
    REQUIRE(rp != nullptr);
    CHECK_EQ(rp->opd[1].rows, 2);
    CHECK_EQ(rp->opd[1].cols, 64);
    CHECK_EQ(rp->opd[1].handle, wo->opd[4].handle);    /* the order op's positions output */
    /* The window layers read the per-window bounds; the full layer the frame bounds. */
    int at = 0;
    for (const auto& is : c.issues) {
        const RecOp* o = b.op_of(is.op);
        if (!o || o->op != "attn_dense") continue;
        if (at < 3) {
            CHECK_EQ(is.opd[3].handle, wo->opd[5].handle);
            CHECK_EQ(is.opd[3].rows, 5);
        } else {
            CHECK(is.opd[3].raw == cu);
            CHECK_EQ(is.opd[3].rows, 3);
        }
        CHECK_EQ(is.opd[0].cols, 96);
        CHECK_EQ(is.opd[1].offset, 96);
        CHECK_EQ(is.opd[2].offset, 192);
        ++at;
    }
    CHECK_EQ(at, 4);
    /* The un-permute reads the order op's inverse. */
    const RecIssue* gi = issue_of(b, c, "gather_rows", 1);
    REQUIRE(gi != nullptr);
    CHECK_EQ(gi->opd[1].handle, wo->opd[3].handle);
    /* ln_post writes all 64 patch rows; the merge folds them into 16 rows for the adapter. */
    const RecIssue* lp = issue_of(b, c, "layernorm", 9);   /* ln_pre, then two a block, then it */
    REQUIRE(lp != nullptr);
    CHECK_EQ(lp->n, 64);
    CHECK_EQ(lp->opd[3].rows, 64);
    const RecIssue* mg = issue_of(b, c, "pixtral_merge");
    REQUIRE(mg != nullptr);
    CHECK_EQ(mg->n, 64);
    CHECK_EQ(mg->opd[1].raw, coord);                   /* the batch's coordinate planes */
    CHECK_EQ(mg->opd[2].rows, 16);
    const RecIssue* m1 = issue_of(b, c, "gemm_nt", 1);     /* the patch GEMM, then the adapter */
    REQUIRE(m1 != nullptr);
    CHECK_EQ(m1->opd[0].rows, 16);

    /* A pass larger than the declaration is refused, not truncated. */
    RadBatch big = batch;
    big.enc_n_patch = 2048;
    RadCtx c2{ .b = &b, .batch = &big };
    museglimmer::step(&c2, &big);
    CHECK(!c2.step_fail.empty());
    CHECK(c2.issues.empty());
}

/* A prefill chunk whose tokens 1..4 are a picture's rows: the trunk scatters them over the lookup
 * before the embedding's reduction, so the reduction counts each row once. */
TEST(a_museglimmer_step_with_a_picture_scatters_its_rows_before_the_reduction) {
    MetaKV mk = with_vision(muse_meta());
    for (int world : { 1, 4 }) {
        RadBuilder b;
        RadBuildCtx ctx = muse_ctx(0, world, 64);
        REQUIRE_EQ(museglimmer::declare(&b, mk.meta(), &ctx), RAD_OK);
        static const int32_t rows[] = { 1, 2, 3, 4 };
        static uint16_t embd[4 * 128];
        auto pf = kit(b, {}, { 6 });
        pf->b.n_mm_rows = 4;
        pf->b.mm_rows = rows;
        pf->b.mm_embd = embd;
        auto v = run_passes(b, museglimmer::step, { pf.get() }, 0, world);
        CHECK_PASSES(b, v);
        const std::vector<std::string> ops = issued_ops(b, *v[0]);
        REQUIRE(ops.size() > 3);
        CHECK_EQ(ops[0], std::string("embed_lookup"));
        CHECK_EQ(ops[1], std::string("scatter_rows"));
        if (world > 1) CHECK_EQ(ops[2], std::string("all_reduce"));
        const RecIssue* sc = issue_of(b, *v[0], "scatter_rows");
        REQUIRE(sc != nullptr);
        CHECK_EQ(sc->n, 4);
        CHECK(sc->opd[0].raw == embd);
        CHECK_EQ(sc->opd[0].cols, 128);
        CHECK(sc->opd[1].raw == rows);
        CHECK_EQ(sc->opd[2].rows, 6);
        /* A text-only step issues no scatter. */
        auto dc = kit(b, { 1, 1 }, {});
        auto t = run_passes(b, museglimmer::step, { dc.get() }, 0, world);
        CHECK_PASSES(b, t);
        CHECK(issue_of(b, *t[0], "scatter_rows") == nullptr);
    }
}
/* ==================================================================== 6. compressed-tensors */

namespace {

/* The quant tests need a wider attention: the asymmetric and int8 grids take K in whole 128s, and
 * o_proj's K is the query width (4 heads x 8 = 32 at muse_meta's shape). 32 heads over the same
 * 2 KV heads keeps every other line of the family's config. */
MetaKV muse_meta_quant() {
    MetaKV k = muse_meta();
    RadModelMeta m = *k.meta();
    m.n_head = 32;
    /* Four KV heads: k and v's rows (heads x head_dim) must come in 16s on the quantised grids,
     * and two heads give 8 a rank at world 2. */
    m.n_head_kv = 4;
    k = MetaKV(m);
    for (const char* pre : { "", "text_config." })
        k.set((std::string(pre) + "layer_types").c_str(),
              "sliding_attention sliding_attention sliding_attention full_attention "
              "sliding_attention sliding_attention sliding_attention full_attention")
            .set((std::string(pre) + "layer_rope_theta").c_str(),
                 "500000 500000 500000 0 500000 500000 500000 0");
    return k;
}

/* The asymmetric int4/int5 affine grid the compressed-tensors recipes write: codes u4 or u5, a
 * u8 zero point and an f16 scale per [1 x 128] (libquant `codes=u4 zero=u8 group=128 scale=f16`). */
RadEncoding muse_a4_enc(uint32_t codes) {
    RadEncoding e = rad_enc_affine(codes, RAD_F16, 1, rad::arch::RAD_FP8_BLOCK);
    rad_enc_add_plane(&e, "zero", RAD_U8, 1, rad::arch::RAD_FP8_BLOCK);
    return e;
}

/* The int8-a-128 grid: i8 codes, a bf16 scale per [1 x 128] (linear_is_w8). */
RadEncoding muse_i8g_enc() { return rad_enc_affine(RAD_I8, RAD_BF16, 1, rad::arch::RAD_FP8_BLOCK); }

/* The symmetric grid data/recipes/museglimmer-30b-w4a8.recipe writes: `codes=i4 group=128
 * scale=f16` (the int8 variant `codes=i8 group=128 scale=f16` for the rows int4 hurts). */
RadEncoding muse_sym_enc(uint32_t codes) { return rad_enc_affine(codes, RAD_F16, 1, 128); }

/* Every trunk linear's declared-name key, for b.encs. */
std::vector<std::pair<std::string, RadEncoding>> muse_linears(const RadEncoding& e) {
    return { { "attn_q.", e },        { "attn_gate.", e },   { "attn_k.", e },
             { "attn_v.", e },        { "attn_output.", e }, { "ffn_gate_up.", e },
             { "ffn_down.", e } };
}

/* The recipe's rules verbatim: every trunk linear at `rtn codes=i4 group=128 scale=f16
 * rule=search`, attention value at `codes=i8 group=128 scale=f16`. */
std::vector<std::pair<std::string, RadEncoding>> muse_recipe_encs() {
    std::vector<std::pair<std::string, RadEncoding>> v = muse_linears(muse_sym_enc(RAD_I4));
    for (auto& [key, e] : v)
        if (key == "attn_v.") e = muse_sym_enc(RAD_I8);
    return v;
}

/* The prefill GEMM of one linear: the gemm_nt_q whose codes weight is this weight's. */
const RecOp* q_gemm_of(const RadBuilder& b, const char* base) {
    const RecWeight* w = find_w(b, (std::string(base) + ".weight").c_str());
    if (!w) return nullptr;
    for (const RecOp& o : b.ops)
        if (o.op == "gemm_nt_q" && o.w.size() == 2 && o.w[0] == w_h(b, w->name.c_str()))
            return &o;
    return nullptr;
}

}  /* namespace */

/* Per layer seven linears, each on the asymmetric u4 grid: quant_act_i8g over the norm output,
 * gemm_nt_q at w4a8i8_asym for prefill and the a16 split (an f16 cast, gemm_nt_q at
 * w4a16_asym under 64 rows) for decode; the views' planes and shapes; gate, k and v keeping
 * their own quantisers (an a4 q writes no pair below its decode band); the fused gate/up concat
 * quantised in the checkpoint's row order; a decode step issuing only the f16 band and a 200-row
 * prefill only the int8 one. */
TEST(museglimmer_ct_declares_the_asymmetric_int4_form) {
    const RadEncoding e = muse_a4_enc(RAD_U4);
    for (int world : { 1, 2 }) {
        RadBuilder b;
        b.encs = muse_linears(e);
        RadBuildCtx ctx = muse_ctx(world - 1, world);
        REQUIRE_EQ(museglimmer_ct::declare(&b, muse_meta_quant().meta(), &ctx), RAD_OK);

        /* Per linear: the codes view, the joined scale+zero view, the quantiser, the int8 GEMM,
         * the f16 cast and the decode GEMM. */
        CHECK_EQ(count_op(b, "quant_act_i8g"), 56);        /* 7 linears x 8 layers */
        CHECK_EQ(count_op(b, "cast"), 56);
        int a8 = 0, a16 = 0;
        for (const RecOp& o : b.ops) {
            if (o.op != "gemm_nt_q") continue;
            const std::string dt = pstr(&o, "dtype");
            if (dt == "w4a8i8_asym") {
                ++a8;
                CHECK_EQ(pint(&o, "group"), (int)rad::arch::RAD_FP8_BLOCK);
            } else if (dt == "w4a16_asym") {
                ++a16;
                const RecParam* m = param(&o, "M");
                REQUIRE(m != nullptr);
                CHECK_EQ(m->ihi, 64LL);
            }
            CHECK(param(&o, "pb1") == nullptr);
        }
        CHECK_EQ(a8, 56);
        CHECK_EQ(a16, 56);

        /* The views: codes u4, the joined scale+zero f16 over the same row. Row-sharded weights
         * cut N; o_proj and down are cut along K in whole scale groups. */
        const struct { const char* base; int64_t n, k; bool col; } linears[] = {
            { "blk.0.attn_q", 256, 128, false },     { "blk.0.attn_gate", 256, 128, false },
            { "blk.0.attn_k", 32, 128, false },      { "blk.0.attn_v", 32, 128, false },
            { "blk.0.attn_output", 128, 256, true }, { "blk.0.ffn_gate_up", 512, 128, false },
            { "blk.0.ffn_down", 128, 256, true },
        };
        for (const auto& ln : linears) {
            const int64_t rows = ln.col ? ln.n : ln.n / world;
            const int64_t cols = ln.col ? ln.k / world : ln.k;
            const RecWeight* w = find_w(b, (std::string(ln.base) + ".weight").c_str());
            REQUIRE(w != nullptr);
            CHECK_EQ(std::string(w->d.planes), std::string("codes"));
            CHECK_EQ(w->d.dtype, (uint32_t)RAD_U4);
            CHECK_EQ(w->d.shape[0], rows);
            CHECK_EQ(w->d.shape[1], cols);
            const RecWeight* ws = find_w(b, (std::string(ln.base) + ".scale").c_str());
            REQUIRE(ws != nullptr);
            CHECK_EQ(std::string(ws->d.planes), std::string("scale,zero"));
            CHECK_EQ(ws->d.dtype, (uint32_t)RAD_F16);
            CHECK_EQ(ws->d.shape[0], rows);
            CHECK_EQ(ws->d.shape[1], cols);
        }
        /* The concat map keeps the checkpoint's row order under quant, gate then up, and the
         * weight's two row parts say so. */
        const RecMap* gu = find_map(b, "blk.0.ffn_gate_up.weight");
        REQUIRE(gu != nullptr);
        REQUIRE_EQ(gu->src.size(), (size_t)2);
        CHECK_EQ(gu->src[0], std::string("model.language_model.layers.0.mlp.gate_proj.weight"));
        CHECK_EQ(gu->src[1], std::string("model.language_model.layers.0.mlp.up_proj.weight"));
        const RecWeight* guw = find_w(b, "blk.0.ffn_gate_up.weight");
        REQUIRE(guw != nullptr);
        REQUIRE_EQ(guw->d.n_row_parts, 2);
        CHECK_EQ(guw->d.row_parts[0], 256 / world);
        CHECK_EQ(guw->d.row_parts[1], 256 / world);

        /* Nothing else quantised: the norms stay f32 gains, the head reads at its codes' dtype
         * (bf16 here), the tower is untouched (absent at no budget). */
        CHECK(find_w(b, "blk.0.attn_norm.weight") != nullptr);
        CHECK_EQ(find_w(b, "blk.0.attn_norm.weight")->d.dtype, (uint32_t)RAD_F32);
        CHECK(find_w(b, "v.patch.weight") == nullptr);

        /* The steps: a decode batch inside the a16 band issues only cast + w4a16_asym per linear;
         * a 200-row prefill issues only quant_act_i8g + w4a8i8_asym. */
        auto dec = kit(b, { 1, 1, 1, 1 }, {});
        auto v = run_passes(b, museglimmer_ct::step, { dec.get() }, world - 1, world);
        CHECK_PASSES(b, v);
        CHECK_LINT(b, v[0].get());
        CHECK(v[0]->step_fail.empty());
        int casts = 0, gemms = 0, quants = 0;
        for (const auto& i : v[0]->issues) {
            const RecOp* o = b.op_of(i.op);
            if (!o) continue;
            if (o->op == "cast" && pstr(o, "to") == "f16") ++casts;
            if (o->op == "quant_act_i8g") ++quants;
            if (o->op == "gemm_nt_q") {
                ++gemms;
                CHECK_EQ(pstr(o, "dtype"), std::string("w4a16_asym"));
            }
        }
        CHECK_EQ(casts, 56);
        CHECK_EQ(gemms, 56);
        CHECK_EQ(quants, 0);

        auto pf = kit(b, {}, { 200 });
        auto p = run_passes(b, museglimmer_ct::step, { pf.get() }, world - 1, world);
        CHECK_PASSES(b, p);
        CHECK_LINT(b, p[0].get());
        CHECK(p[0]->step_fail.empty());
        casts = gemms = quants = 0;
        for (const auto& i : p[0]->issues) {
            const RecOp* o = b.op_of(i.op);
            if (!o) continue;
            if (o->op == "cast" && pstr(o, "to") == "f16") ++casts;
            if (o->op == "quant_act_i8g") ++quants;
            if (o->op == "gemm_nt_q") {
                ++gemms;
                CHECK_EQ(pstr(o, "dtype"), std::string("w4a8i8_asym"));
            }
        }
        CHECK_EQ(casts, 0);
        CHECK_EQ(gemms, 56);
        CHECK_EQ(quants, 56);
    }
}

/* gate, k and v quantise q's same norm output on the same per-row grid: on the int8-a-128 grid
 * (where the shared pair is written at every height) the later three read q's int8 pair and the
 * layer declares one quantiser for the four, k after q. */
TEST(museglimmer_ct_shares_one_activation_pair_between_q_gate_k_and_v_on_i8g) {
    const RadEncoding e = muse_i8g_enc();
    for (int world : { 1, 2 }) {
        RadBuilder b;
        b.encs = muse_linears(e);
        RadBuildCtx ctx = muse_ctx(world - 1, world);
        REQUIRE_EQ(museglimmer_ct::declare(&b, muse_meta_quant().meta(), &ctx), RAD_OK);

        /* q, o, gate_up and down keep quantisers; gate, k and v read q's pair. */
        CHECK_EQ(count_op(b, "quant_act_i8g"), 32);   /* 4 quantisers x 8 layers */
        CHECK_EQ(count_op(b, "cast"), 0);
        int i8 = 0;
        for (const RecOp& o : b.ops)
            if (o.op == "gemm_nt_q") {
                ++i8;
                CHECK_EQ(pstr(&o, "dtype"), std::string("i8a8"));
                CHECK_EQ(pint(&o, "group"), (int)rad::arch::RAD_FP8_BLOCK);
            }
        CHECK_EQ(i8, 56);

        /* The views: i8 codes, a bf16 scale per [1 x 128]. */
        const RecWeight* q = find_w(b, "blk.0.attn_q.weight");
        REQUIRE(q != nullptr);
        CHECK_EQ(std::string(q->d.planes), std::string("codes"));
        CHECK_EQ(q->d.dtype, (uint32_t)RAD_I8);
        const RecWeight* qs = find_w(b, "blk.0.attn_q.scale");
        REQUIRE(qs != nullptr);
        CHECK_EQ(std::string(qs->d.planes), std::string("scale"));
        CHECK_EQ(qs->d.dtype, (uint32_t)RAD_BF16);
        CHECK_EQ(qs->d.shape[1], 128 / (int64_t)rad::arch::RAD_FP8_BLOCK);

        /* k after q: k's GEMM reads the pair q's quantiser writes. */
        const RecOp* qdec = q_gemm_of(b, "blk.0.attn_q");
        REQUIRE(qdec != nullptr);
        const RecOp* qg = [&] {
            for (const RecOp& o : b.ops)
                if (o.op == "quant_act_i8g" && o.writes.size() == 2 &&
                    o.writes[0] == qdec->reads[0])
                    return &o;
            return (const RecOp*)nullptr;
        }();
        REQUIRE(qg != nullptr);
        CHECK_EQ(qg->writes[0], b_h(b, "blk.0.attn_q.q8"));
        CHECK_EQ(qg->writes[1], b_h(b, "blk.0.attn_q.s8"));
        for (const char* base : { "blk.0.attn_gate", "blk.0.attn_k", "blk.0.attn_v" }) {
            const RecOp* g = q_gemm_of(b, base);
            REQUIRE(g != nullptr);
            CHECK_EQ(g->reads[0], qdec->reads[0]);
            CHECK_EQ(g->reads[1], qdec->reads[1]);
        }
        /* o_proj, whose input is not q's norm output, keeps its own pair. */
        const RecOp* og = q_gemm_of(b, "blk.0.attn_output");
        REQUIRE(og != nullptr);
        CHECK(og->reads[0] != qdec->reads[0]);

        /* The pair is issued once per layer step and all four GEMMs read it. */
        auto dec = kit(b, { 1, 1, 1, 1 }, {});
        auto v = run_passes(b, museglimmer_ct::step, { dec.get() }, world - 1, world);
        CHECK_PASSES(b, v);
        CHECK_LINT(b, v[0].get());
        CHECK(v[0]->step_fail.empty());
        CHECK_EQ(count_issued(issued_ops(b, *v[0]), "quant_act_i8g"), 32);
        CHECK_EQ(count_issued(issued_ops(b, *v[0]), "gemm_nt_q"), 56);
    }
}

/* A container that holds bf16 through the ct plugin: the graph is the bf16 plugin's, op for op,
 * every linear a plain gemm_nt -- the descriptor picks the plugin, the encodings pick the ops. */
TEST(museglimmer_ct_serves_a_bf16_container_as_the_bf16_plugin_does) {
    const RadEncoding e = rad_enc_plain(RAD_BF16);
    for (int world : { 1, 4 }) {
        RadBuilder b;
        b.encs = muse_linears(e);
        RadBuildCtx ctx = muse_ctx(world - 1, world);
        REQUIRE_EQ(museglimmer_ct::declare(&b, muse_meta_quant().meta(), &ctx), RAD_OK);
        CHECK_EQ(count_op(b, "gemm_nt"), 56);           /* seven linears x eight layers */
        CHECK_EQ(count_op(b, "gemm_nt_q"), 0);
        CHECK_EQ(count_op(b, "quant_act_i8g"), 0);
        CHECK_EQ(join(op_names(b)), join(trunk_decl(8, world)));
        auto dec = kit(b, { 1, 1, 1, 1 }, {});
        auto v = run_passes(b, museglimmer_ct::step, { dec.get() }, world - 1, world);
        CHECK_PASSES(b, v);
        CHECK_LINT(b, v[0].get());
        CHECK_EQ(join(issued_ops(b, *v[0])), join(trunk_step(8, world)));
    }
}

/* The same grid at five bits: u5 codes, gemm_nt_q at w5a8i8 and the a16 decode band at w5a16. */
TEST(museglimmer_ct_declares_the_asymmetric_int5_form) {
    RadBuilder b;
    b.encs = muse_linears(muse_a4_enc(RAD_U5));
    RadBuildCtx ctx = muse_ctx(0, 1);
    REQUIRE_EQ(museglimmer_ct::declare(&b, muse_meta_quant().meta(), &ctx), RAD_OK);
    int a8 = 0, a16 = 0;
    for (const RecOp& o : b.ops) {
        if (o.op != "gemm_nt_q") continue;
        const std::string dt = pstr(&o, "dtype");
        if (dt == "w5a8i8") ++a8;
        if (dt == "w5a16") ++a16;
    }
    CHECK_EQ(a8, 56);
    CHECK_EQ(a16, 56);
    const RecWeight* w = find_w(b, "blk.0.attn_q.weight");
    REQUIRE(w != nullptr);
    CHECK_EQ(w->d.dtype, (uint32_t)RAD_U5);
    auto dec = kit(b, { 1 }, {});
    auto v = run_passes(b, museglimmer_ct::step, { dec.get() }, 0, 1);
    CHECK_PASSES(b, v);
    for (const auto& i : v[0]->issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == "gemm_nt_q") CHECK_EQ(pstr(o, "dtype"), std::string("w5a16"));
    }
}

/* Block fp8 (at a bf16 or an f32 scale) and MXFP4 are forms no plugin of this build serves: a
 * linear stored in one is refused by name at declare, as LinearFP8 refuses it elsewhere. */
TEST(museglimmer_ct_refuses_block_fp8_and_mxfp4_by_name) {
    const RadEncoding retired[] = {
        rad_enc_affine(RAD_F8E4M3, RAD_BF16, rad::arch::RAD_FP8_BLOCK, rad::arch::RAD_FP8_BLOCK),
        rad_enc_affine(RAD_F8E4M3, RAD_F32, rad::arch::RAD_FP8_BLOCK, rad::arch::RAD_FP8_BLOCK),
        rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32),
    };
    for (const RadEncoding& e : retired) {
        RadBuilder b;
        b.encs = { { "attn_q.", e } };
        RadBuildCtx ctx = muse_ctx(0, 1);
        CHECK_EQ(museglimmer_ct::declare(&b, muse_meta_quant().meta(), &ctx), RAD_E_UNSUPPORTED);
        CHECK(find_op(b, "gemm_nt_q", 0) == nullptr);
    }
}

/* What the encodings cannot serve is refused naming the weight: a trunk grid outside the forms
 * the plugin takes, or a vision linear that is neither BF16 nor int8 per (row, 128). */
TEST(museglimmer_ct_refuses_encodings_it_does_not_serve_by_name) {
    {
        RadBuilder b;
        RadEncoding bad = rad_enc_affine(RAD_U3, RAD_F16, 1, 128);   /* a 3-bit grid */
        b.encs = { { "attn_q.", bad } };
        RadBuildCtx ctx = muse_ctx(0, 1);
        const int rc = stderr_of_rc(
            [&] { return museglimmer_ct::declare(&b, muse_meta_quant().meta(), &ctx); });
        CHECK_EQ(rc, RAD_E_DTYPE);
    }
    {
        RadBuilder b;
        b.encs = { { "vision_tower", muse_a4_enc(RAD_U4) } };
        RadBuildCtx ctx = muse_ctx(0, 1, 64);
        const int rc = stderr_of_rc([&] {
            return museglimmer_ct::declare(&b, with_vision(muse_meta_quant()).meta(), &ctx);
        });
        CHECK_EQ(rc, RAD_E_DTYPE);
    }
}

/* Int8 vision: codes with a bf16 scale per (row, 128), gemm_nt_q at i8a8 with w_rb 1 so the
 * container-form kernel is selected. Both plugins share the tower. */
TEST(museglimmer_vision_serves_int8_linears_and_issues_the_encoder) {
    const RadEncoding i8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    for (bool ct : { false, true }) {
        MetaKV mk = with_vision(ct ? muse_meta_quant() : muse_meta());
        mk.set("vision_config.hidden_size", "128").set("out_hidden_size", "512")
          .set("vision_config.num_attention_heads", "8");
        RadBuilder b;
        for (int l = 0; l < 4; ++l)
            for (const char* name : { "qkv", "proj", "fc1", "fc2" })
                b.encs.push_back({ "v.blk." + std::to_string(l) + "." + name + ".weight", i8 });
        for (const char* name : { "v.mlp.fc1.weight", "v.mlp.fc2.weight", "v.proj.weight" })
            b.encs.push_back({ name, i8 });
        RadBuildCtx ctx = muse_ctx(0, 1, 128);
        REQUIRE_EQ(ct ? museglimmer_ct::declare(&b, mk.meta(), &ctx)
                      : museglimmer::declare(&b, mk.meta(), &ctx), RAD_OK);
        const RecWeight* qkv = find_w(b, "v.blk.0.qkv.weight");
        const RecWeight* scale = find_w(b, "v.blk.0.qkv.scale");
        REQUIRE(qkv && scale);
        CHECK_EQ(qkv->d.dtype, (uint32_t)RAD_I8);
        CHECK_EQ(qkv->d.shape[0], 384);
        CHECK_EQ(qkv->d.shape[1], 128);
        CHECK_EQ(scale->d.dtype, (uint32_t)RAD_BF16);
        CHECK_EQ(scale->d.shape[0], 384);
        CHECK_EQ(scale->d.shape[1], 1);
        REQUIRE_EQ(scale->d.n_row_parts, 3);
        CHECK_EQ(scale->d.row_parts[0], 128);
        CHECK_EQ(scale->d.row_parts[1], 128);
        CHECK_EQ(scale->d.row_parts[2], 128);
        CHECK(find_b(b, "v.i8_width.q") != nullptr);
        CHECK(find_b(b, "v.i8_ff.q") != nullptr);
        CHECK(find_b(b, "v.i8_merged.q") != nullptr);
        CHECK(find_b(b, "v.i8_projector.q") != nullptr);
        CHECK_EQ(count_op(b, "quant_act_i8g"), 19);
        CHECK_EQ(count_op(b, "table_add"), 16);
        int wrb = 0;
        for (const RecOp& o : b.ops) {
            if (o.op != "gemm_nt_q") continue;
            CHECK_EQ(pstr(&o, "dtype"), std::string("i8a8"));
            for (const RecParam& p : o.p)
                if (p.key == "w_rb") { CHECK_EQ(p.ival, 1); ++wrb; }
        }
        CHECK_EQ(wrb, 19);

        static uint16_t pix[128 * 1176];
        static int32_t coord[4 * 128];
        static const int32_t cu[] = { 0, 128 }, sh[] = { 16 }, sw[] = { 8 };
        RadBatch batch{};
        batch.enc = 1; batch.enc_n_patch = 128; batch.enc_pixels = pix; batch.enc_coord = coord;
        batch.enc_n_seg = 1; batch.enc_cu = cu; batch.enc_seg_h = sh; batch.enc_seg_w = sw;
        batch.enc_max_seg = 128;
        RadCtx c{ .b = &b, .batch = &batch };
        c.rank = 0; c.world = 1;
        if (ct) museglimmer_ct::step(&c, &batch); else museglimmer::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        const RecIssue* first = issue_of(b, c, "gemm_nt_q");
        REQUIRE(first != nullptr);
        CHECK_EQ(first->n, 128);
        CHECK_EQ(first->opd[0].handle, b_h(b, "v.i8_width.q"));
        CHECK_EQ(first->opd[1].handle, b_h(b, "v.i8_width.s"));
    }
}

TEST(museglimmer_vision_refuses_retired_and_ragged_int8_grids) {
    MetaKV mk = with_vision(muse_meta());
    mk.set("vision_config.hidden_size", "128").set("out_hidden_size", "512");
    RadBuildCtx ctx = muse_ctx(0, 1, 128);
    {
        RadBuilder b;
        b.encs = { { "v.blk.0.qkv.weight", rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128) } };
        CHECK_EQ(stderr_of_rc([&] { return museglimmer::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
    {
        /* hidden 96 is not a whole 128, so an int8 qkv is refused rather than relaid out. */
        MetaKV narrow = with_vision(muse_meta());
        RadBuilder b;
        b.encs = { { "v.blk.0.qkv.weight", rad_enc_affine(RAD_I8, RAD_BF16, 1, 128) } };
        CHECK_EQ(stderr_of_rc([&] {
                     return museglimmer::declare(&b, narrow.meta(), &ctx);
                 }),
                 RAD_E_SHAPE);
    }
}

/* The selection key: this plugin claims (arch museglimmer, quant compressed-tensors). The test TU
 * builds with RAD_ARCH_NO_EXPORTS, so the descriptor is asserted where the exports are linked --
 * tests/shipped_plugins_test.cpp wants { "museglimmer", "compressed-tensors" } beside the bf16
 * plugin's { "museglimmer", "" }, and the plugin macro here carries that exact pair. */
TEST(museglimmer_ct_declares_the_same_family_graph_and_probe_as_the_bf16_plugin) {
    RadBuilder b;
    RadBuildCtx ctx = muse_ctx(0, 1);
    REQUIRE_EQ(museglimmer_ct::declare(&b, muse_meta_quant().meta(), &ctx), RAD_OK);
    /* The tower's probe, the family's merge of 4: identical to the bf16 plugin's. */
    RadArchProbe p{};
    CHECK_OK(museglimmer_ct::probe(muse_meta_quant().meta(), &p));
    CHECK_EQ(p.enc_merge, 0);
    RadArchProbe pv{};
    CHECK_OK(museglimmer_ct::probe(with_vision(muse_meta_quant()).meta(), &pv));
    CHECK_EQ(pv.enc_merge, 4);
    /* The weightless norms, the centered doubles, the soft cap: unchanged under quantisation.
     * The first normed rmsnorm is a layer's input norm; its fifth op after is the q norm. */
    const size_t at = (size_t)(std::find_if(b.ops.begin(), b.ops.end(),
                                            [](const RecOp& o) {
                                                return o.op == "rmsnorm" && o.w.size() == 1;
                                            }) - b.ops.begin());
    REQUIRE(at + 5 < b.ops.size());
    CHECK_EQ(b.ops[at + 5].op, std::string("rmsnorm"));
    CHECK(b.ops[at + 5].w.empty());
    CHECK(near(pflt(&b.ops[at + 5], "mult"), 3.87, 1e-6));
    const RecOp* fn = find_op(b, "rmsnorm", 49);
    REQUIRE(fn != nullptr);
    CHECK(near(pflt(fn, "wadd"), 0.0, 1e-12));
    CHECK(near(pflt(fn, "mult"), 0.19611613513818404, 1e-6));
    const RecOp* cap = find_op(b, "logit_softcap");
    REQUIRE(cap != nullptr);
    CHECK(near(pflt(cap, "cap"), 20.0, 1e-12));
    /* Two KV groups, every layer bound exactly once, to its group. */
    REQUIRE_EQ(b.binds.size(), (size_t)8);
    const rad_kvgroup win = 1, full = 2;
    for (int l = 0; l < 8; ++l) {
        CHECK_EQ(b.binds[(size_t)l].first, l);
        CHECK_EQ(b.binds[(size_t)l].second, (rad_kvgroup)((l % 4 == 3) ? full : win));
    }
}

/* The recipe's grid, rule for rule (museglimmer-30b-w4a8.recipe): every trunk linear at
 * `codes=i4 group=128 scale=f16`, attention value at `codes=i8 group=128 scale=f16`. Each linear
 * is gemm_nt_q at w4a8 (w8a8 for v) over quant_act_i8's per-row int8 pair; gate, k and v read
 * q's pair (one quantiser for the four, k after q); the views are codes at [out, in] and an f16
 * scale per 128 columns; a step issues the quantiser at every height (no decode band here). */
TEST(museglimmer_ct_serves_the_w4a8_recipe_grid_symmetric_i4_and_i8) {
    for (int world : { 1, 2 }) {
        RadBuilder b;
        b.encs = muse_recipe_encs();
        RadBuildCtx ctx = muse_ctx(world - 1, world);
        REQUIRE_EQ(museglimmer_ct::declare(&b, muse_meta_quant().meta(), &ctx), RAD_OK);

        /* q, o, gate_up and down keep quantisers; gate, k and v read q's pair. */
        CHECK_EQ(count_op(b, "quant_act_i8"), 32);
        CHECK_EQ(count_op(b, "quant_act_i8g"), 0);
        CHECK_EQ(count_op(b, "cast"), 0);
        int w4 = 0, w8 = 0;
        for (const RecOp& o : b.ops) {
            if (o.op != "gemm_nt_q") continue;
            const std::string dt = pstr(&o, "dtype");
            if (dt == "w4a8") ++w4;
            if (dt == "w8a8") ++w8;
            CHECK_EQ(pint(&o, "group"), 128);
        }
        CHECK_EQ(w4, 48);   /* six i4 linears x eight layers */
        CHECK_EQ(w8, 8);    /* attention value, the recipe's int8 row */

        /* The views: i4 codes for the i4 linears, i8 for v; an f16 scale per 128 columns. */
        const struct { const char* base; int64_t n, k; bool col; uint32_t dt; } linears[] = {
            { "blk.0.attn_q", 256, 128, false, RAD_I4 },
            { "blk.0.attn_gate", 256, 128, false, RAD_I4 },
            { "blk.0.attn_k", 32, 128, false, RAD_I4 },
            { "blk.0.attn_v", 32, 128, false, RAD_I8 },
            { "blk.0.attn_output", 128, 256, true, RAD_I4 },
            { "blk.0.ffn_gate_up", 512, 128, false, RAD_I4 },
            { "blk.0.ffn_down", 128, 256, true, RAD_I4 },
        };
        for (const auto& ln : linears) {
            const int64_t rows = ln.col ? ln.n : ln.n / world;
            const int64_t cols = ln.col ? ln.k / world : ln.k;
            const RecWeight* w = find_w(b, (std::string(ln.base) + ".weight").c_str());
            REQUIRE(w != nullptr);
            CHECK_EQ(std::string(w->d.planes), std::string("codes"));
            CHECK_EQ(w->d.dtype, (uint32_t)ln.dt);
            CHECK_EQ(w->d.shape[0], rows);
            CHECK_EQ(w->d.shape[1], cols);
            const RecWeight* ws = find_w(b, (std::string(ln.base) + ".scale").c_str());
            REQUIRE(ws != nullptr);
            CHECK_EQ(std::string(ws->d.planes), std::string("scale"));
            CHECK_EQ(ws->d.dtype, (uint32_t)RAD_F16);
            CHECK_EQ(ws->d.shape[0], rows);
            CHECK_EQ(ws->d.shape[1], cols / 128);
        }

        /* k after q: k's GEMM reads the pair q's quantiser writes. */
        const RecOp* qdec = q_gemm_of(b, "blk.0.attn_q");
        REQUIRE(qdec != nullptr);
        for (const char* base : { "blk.0.attn_gate", "blk.0.attn_k", "blk.0.attn_v" }) {
            const RecOp* g = q_gemm_of(b, base);
            REQUIRE(g != nullptr);
            CHECK_EQ(g->reads[0], qdec->reads[0]);
            CHECK_EQ(g->reads[1], qdec->reads[1]);
        }

        /* A step quantises at every height: the symmetric grid has no decode band. */
        auto dec = kit(b, { 1, 1, 1, 1 }, {});
        auto v = run_passes(b, museglimmer_ct::step, { dec.get() }, world - 1, world);
        CHECK_PASSES(b, v);
        CHECK_LINT(b, v[0].get());
        CHECK(v[0]->step_fail.empty());
        CHECK_EQ(count_issued(issued_ops(b, *v[0]), "quant_act_i8"), 32);
        int gemms = 0;
        for (const auto& i : v[0]->issues) {
            const RecOp* o = b.op_of(i.op);
            if (o && o->op == "gemm_nt_q") ++gemms;
        }
        CHECK_EQ(gemms, 56);
    }
}

/* The recipe's documented int8 variant (`codes=i8 group=128 scale=f16` for a whole trunk): every
 * linear at w8a8 on the same views. */
TEST(museglimmer_ct_serves_the_symmetric_int8_variant) {
    RadBuilder b;
    b.encs = muse_linears(muse_sym_enc(RAD_I8));
    RadBuildCtx ctx = muse_ctx(0, 1);
    REQUIRE_EQ(museglimmer_ct::declare(&b, muse_meta_quant().meta(), &ctx), RAD_OK);
    int w8 = 0;
    for (const RecOp& o : b.ops) {
        if (o.op != "gemm_nt_q") continue;
        CHECK_EQ(pstr(&o, "dtype"), std::string("w8a8"));
        ++w8;
    }
    CHECK_EQ(w8, 56);
    const RecWeight* w = find_w(b, "blk.0.attn_q.weight");
    REQUIRE(w != nullptr);
    CHECK_EQ(w->d.dtype, (uint32_t)RAD_I8);
    auto dec = kit(b, { 1 }, {});
    auto v = run_passes(b, museglimmer_ct::step, { dec.get() }, 0, 1);
    CHECK_PASSES(b, v);
    for (const auto& i : v[0]->issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == "gemm_nt_q") CHECK_EQ(pstr(o, "dtype"), std::string("w8a8"));
    }
}

/* ==================================================================== 6b. an IQ8R trunk */

namespace {

/* IQ8R as libquant states it (`codes=i8 group=256 scale=u8 scale2=bf16 block2=1x*
 * transform=fwht128`): int8 codes rotated 128 wide, a u8 scale per [1 x 256] of K under one bf16
 * scale a row (linear_is_iq8r). */
RadEncoding muse_iq8r_enc() {
    RadEncoding e = rad_enc_affine(RAD_I8, RAD_U8, 1, rad::arch::RAD_I8_GROUP_WIDE);
    rad_enc_add_plane(&e, "scale.1", RAD_BF16, 1, 0);
    rad_enc_copy_str(e.transform, "fwht128");
    return e;
}

/* data/recipes/museglimmer-30b-iq8r.recipe's encodings by declared name: the seven trunk linears
 * IQ8R, the head plain int8 at a bf16 scale per [1 x 128]. The head comes last because
 * `output.weight` is in `attn_output.weight`. */
std::vector<std::pair<std::string, RadEncoding>> muse_iq8r_encs() {
    std::vector<std::pair<std::string, RadEncoding>> v = muse_linears(muse_iq8r_enc());
    v.push_back({ "output.weight", rad_enc_affine(RAD_I8, RAD_BF16, 1, rad::arch::RAD_FP8_BLOCK) });
    return v;
}

/* The trunk at widths IQ8R's 256-column group divides at one rank and at two: hidden 256,
 * attention 64 heads x 8 = 512, feed-forward 512; four KV heads keep k and v in 16-row tiles a
 * rank. Every other line of the family's config is muse_meta_quant's. */
MetaKV muse_meta_iq8r() {
    MetaKV k = muse_meta();
    RadModelMeta m = *k.meta();
    m.n_embd = 256;
    m.n_head = 64;
    m.n_head_kv = 4;
    m.n_ff = 512;
    k = MetaKV(m);
    for (const char* pre : { "", "text_config." })
        k.set((std::string(pre) + "layer_types").c_str(),
              "sliding_attention sliding_attention sliding_attention full_attention "
              "sliding_attention sliding_attention sliding_attention full_attention")
            .set((std::string(pre) + "layer_rope_theta").c_str(),
                 "500000 500000 500000 0 500000 500000 500000 0");
    return k;
}

/* Muse IQ4R-V4's computed trellis: 128-wide FWHT, unsigned codes, the MAD plane, and either
 * one f32 scale a row plus a shared f32 column factor or a bf16 block scale (the 5-bit o_proj). */
RadEncoding muse_iqr_v4_enc(uint32_t codes, int64_t tcq_rows, bool row_scale = true) {
    RadEncoding e{};
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "trellis");
    rad_enc_add_plane(&e, "codes", codes, 1, 1);
    if (row_scale) {
        rad_enc_add_plane(&e, "scale", RAD_F32, 1, 0);
        rad_enc_add_plane(&e, "scale.col", RAD_F32, 0, 128);
    } else {
        rad_enc_add_plane(&e, "scale", RAD_BF16, 1, 128);
    }
    const int t = rad_enc_add_table(&e, "trellis", RAD_I8, (int64_t)1 << 16, 1);
    e.plane[t].block[0] = tcq_rows;
    e.plane[t].block[1] = 1;
    rad_enc_add_table(&e, "trellis.mad", RAD_U32, 2, 1);
    rad_enc_copy_str(e.transform, "fwht128");
    return e;
}

std::vector<std::pair<std::string, RadEncoding>> muse_iqr_v4_encs() {
    return { { "attn_q.", muse_iqr_v4_enc(RAD_U4, 1024) },
             { "attn_gate.", muse_iqr_v4_enc(RAD_U4, 1024) },
             { "attn_k.", muse_iqr_v4_enc(RAD_U4, 128) },
             { "attn_v.", muse_iqr_v4_enc(RAD_U4, 128) },
             { "attn_output.", muse_iqr_v4_enc(RAD_U5, 256, false) },
             { "ffn_gate_up.", muse_iqr_v4_enc(RAD_U4, 512) },
             { "ffn_down.", muse_iqr_v4_enc(RAD_U4, 1024) } };
}

MetaKV muse_meta_iqr_v4() {
    MetaKV k = muse_meta();
    RadModelMeta m = *k.meta();
    m.n_embd = 256;
    m.n_head = 2;
    m.n_head_kv = 2;
    m.head_dim = 128;
    m.n_ff = 512;
    k = MetaKV(m);
    return k;
}

MetaKV muse_dflash2_meta() {
    MetaKV k = muse_meta_iqr_v4();
    k.set("draft.architectures", "DFlash2DraftModel")
        .set("draft.num_hidden_layers", "2")
        .set("draft.head_dim", "128")
        .set("draft.num_attention_heads", "2")
        .set("draft.num_key_value_heads", "2")
        .set("draft.hidden_size", "256")
        .set("draft.intermediate_size", "512")
        .set("draft.rms_norm_eps", "1e-5")
        .set("draft.rope_theta", "500000")
        .set("draft.vocab_size", "512")
        .set("draft.use_sliding_window", "1")
        .set("draft.sliding_window", "4")
        .set("draft.dflash_config.input_embedding_scale", "1")
        .set("draft.dflash_config.output_multiplier", "0.19611613513818404")
        .set("draft.dflash_config.final_logit_softcapping", "20")
        .set("draft.dflash_config.target_layer_ids", "1 3")
        .set("draft.dflash_config.block_size", "4")
        .set("draft.dflash_config.mask_token_id", "511")
        .set("draft.dflash_config.selector_rank", "64")
        .set("draft.dflash_config.selector_top_k", "8")
        .set("draft.dflash_config.conv_kernel_size", "2")
        .set("draft.dflash_config.conv_group_size", "16");
    return k;
}

std::vector<std::pair<std::string, RadEncoding>> muse_dflash2_iqr_encs(bool include_trunk) {
    std::vector<std::pair<std::string, RadEncoding>> encs;
    auto add = [&](const std::string& name, uint32_t codes, int64_t rows) {
        encs.push_back({name, muse_iqr_v4_enc(codes, rows)});
    };
    add("dflash.fc.", RAD_U4, 256);
    encs.push_back({ "dflash.draft_head.", rad_enc_plain(RAD_BF16) });
    for (int l = 0; l < 2; ++l) {
        const std::string p = "dflash.blk." + std::to_string(l) + ".";
        for (const char* n : { "attn_q.", "attn_k.", "attn_v.", "attn_output." })
            add(p + n, RAD_U4, 256);
        add(p + "ffn_gate_up.", RAD_U3, 256);
        add(p + "ffn_down.", RAD_U3, 256);
    }
    if (include_trunk) {
        const auto trunk = muse_iqr_v4_encs();
        encs.insert(encs.end(), trunk.begin(), trunk.end());
    }
    return encs;
}

/* The op that writes buffer `buf` first among the ops named `name`. */
const RecOp* writer_of(const RadBuilder& b, const char* name, rad_buf buf) {
    for (const RecOp& o : b.ops)
        if (o.op == name && in(o.writes, buf)) return &o;
    return nullptr;
}

}  /* namespace */

/* Computed-MAD IQ4R-V4: the full trunk is on the 128-wide rotated activation grid, including
 * centered norm and gated-product producers; q/k/v/gate share a pair, output and down have their
 * own. Check decode and prefill graph lint at TP1 and TP2, including the skinny KV shard. */
TEST(museglimmer_ct_serves_an_iq4r_v4_trunk) {
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            b.encs = muse_iqr_v4_encs();
            RadBuildCtx ctx = muse_ctx(rank, world);
            REQUIRE_EQ(museglimmer_ct::declare(&b, muse_meta_iqr_v4().meta(), &ctx), RAD_OK);
            const museglimmer_ct::Model& m = museglimmer_ct::g_model[rad_build_slot(&ctx)];
            CHECK(m.iqr_v4);
            CHECK(!m.iq8r);
            CHECK(m.rotated_i8);
            CHECK_EQ(m.rot_group, 128);
            CHECK(m.a_h.q8_fed);
            CHECK(m.a_ffn.q8_fed);
            CHECK(!m.a_attn.q8_fed);
            CHECK_EQ(count_op(b, "gemm_nt_q"), 56);
            CHECK_EQ(count_op(b, "gemm_nt"), 0);
            for (const RecOp& o : b.ops)
                if (o.op == "gemm_nt_q") {
                    CHECK_EQ(pstr(&o, "dtype"), std::string("i8a8h"));
                    CHECK_EQ(pint(&o, "group"), 128);
                }
            for (int pass = 0; pass < 2; ++pass) {
                auto k = pass == 0 ? kit(b, { 1, 1, 1, 1 }, {}) : kit(b, {}, { 200 });
                auto v = run_passes(b, museglimmer_ct::step, { k.get() }, rank, world);
                CHECK_PASSES(b, v);
                CHECK_LINT(b, v[0].get());
                CHECK(v[0]->step_fail.empty());
                const auto is = issued_ops(b, *v[0]);
                CHECK_EQ(count_issued(is, "gemm_nt_q"), 56);
                CHECK_EQ(count_issued(is, "rmsnorm_had_quant_i8g"), 16);
                CHECK_EQ(count_issued(is, "gated_had_quant_i8g"), 8);
                CHECK_EQ(count_issued(is, "had_quant_act_i8g"), 8);
            }
        }
}

/* The IQ8R container at one rank and at two: every trunk linear gemm_nt_q at i8a8h, group 256, over
 * the codes view (i8 [N, K]) and one "scale,scale.1" u8 view at the logical [N, K]; both pre-norms
 * rmsnorm_had_quant_i8g (centered, at rms_norm_eps, no residual, no all-reduce form at two ranks:
 * the collectives feed the post norms) writing the pair q, gate, k, v and gate/up read; the gated
 * product gated_had_quant_i8g writing down_proj's pair and no bf16 product; o_proj's own
 * had_quant_act_i8g over the gated attention output; the head plain int8. No other quantiser, no
 * E4M3 plane, no bf16 GEMM. Decode and prefill issue that graph lint-clean. */
TEST(museglimmer_ct_serves_an_iq8r_trunk) {
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            b.encs = muse_iq8r_encs();
            RadBuildCtx ctx = muse_ctx(rank, world);
            REQUIRE_EQ(museglimmer_ct::declare(&b, muse_meta_iq8r().meta(), &ctx), RAD_OK);
            const museglimmer_ct::Model& m = museglimmer_ct::g_model[rad_build_slot(&ctx)];
            CHECK(m.iq8r);
            CHECK(m.a_h.q8_fed);
            CHECK(m.a_ffn.q8_fed);
            CHECK(!m.a_attn.q8_fed);

            const int64_t E = 256, Q = 512 / world, KV = 32 / world, F = 512 / world;
            const rad_buf hq = (rad_buf)b_h(b, "h.q8"), hs = (rad_buf)b_h(b, "h.s8");
            const rad_buf aq = (rad_buf)b_h(b, "attn_out.q8"), as = (rad_buf)b_h(b, "attn_out.s8");
            const rad_buf fq = (rad_buf)b_h(b, "ffn.q8"), fs = (rad_buf)b_h(b, "ffn.s8");
            REQUIRE(hq && hs && aq && as && fq && fs);
            /* One f32 scale a (row, 256). */
            CHECK_EQ(find_b(b, "h.s8")->d.shape[1], E / 256);
            CHECK_EQ(find_b(b, "attn_out.s8")->d.shape[1], Q / 256);
            CHECK_EQ(find_b(b, "ffn.s8")->d.shape[1], F / 256);
            for (const char* none : { "h.q", "h.s", "attn_out.q", "ffn.q" })
                CHECK(find_b(b, none) == nullptr);

            /* Seven GEMMs a layer, all i8a8h at group 256; nothing bf16 in the trunk. */
            CHECK_EQ(count_op(b, "gemm_nt_q"), 56);
            CHECK_EQ(count_op(b, "gemm_nt"), 0);
            for (const RecOp& o : b.ops)
                if (o.op == "gemm_nt_q") {
                    CHECK_EQ(pstr(&o, "dtype"), std::string("i8a8h"));
                    CHECK_EQ(pint(&o, "group"), 256);
                }

            /* The producers: two rotating norms, one gated quantiser and o_proj's quantiser a
             * layer, each at group 256 over its row's width. */
            CHECK_EQ(count_op(b, "rmsnorm_had_quant_i8g"), 16);
            CHECK_EQ(count_op(b, "ar_rmsnorm_had_quant_i8g"), 0);
            CHECK_EQ(count_op(b, "gated_had_quant_i8g"), 8);
            CHECK_EQ(count_op(b, "had_quant_act_i8g"), 8);
            for (const char* op : { "quant_act_i8g", "quant_act_i8", "quant_act_fp8",
                                    "rmsnorm_quant_fp8", "gated_quant_fp8", "silu_mul", "cast",
                                    "gemm_nt_q_gated" })
                CHECK_EQ(count_op(b, op), 0);
            const int64_t w_attn_norm = w_h(b, "blk.0.attn_norm.weight");
            const int64_t w_ffn_norm = w_h(b, "blk.0.ffn_norm.weight");
            int norms = 0;
            for (const RecOp& o : b.ops) {
                if (o.op == "rmsnorm_had_quant_i8g") {
                    CHECK_EQ(pint(&o, "n"), E);
                    CHECK_EQ(pint(&o, "group"), 256);
                    CHECK(near(pflt(&o, "wadd"), 1.0, 1e-12));
                    CHECK(near(pflt(&o, "eps"), 1e-5, 1e-9));
                    REQUIRE_EQ(o.reads.size(), (size_t)1);          /* the stream, no residual */
                    CHECK_EQ(o.reads[0], (rad_buf)b_h(b, "x"));
                    REQUIRE_EQ(o.writes.size(), (size_t)3);
                    CHECK_EQ(o.writes[0], hq);
                    CHECK_EQ(o.writes[1], hs);
                    CHECK_EQ(o.writes[2], (rad_buf)b_h(b, "h"));
                    if (norms == 0) CHECK_EQ((int64_t)o.w[0], w_attn_norm);
                    if (norms == 1) CHECK_EQ((int64_t)o.w[0], w_ffn_norm);
                    ++norms;
                }
                if (o.op == "gated_had_quant_i8g") {
                    CHECK_EQ(pint(&o, "n"), F);
                    CHECK_EQ(pint(&o, "group"), 256);
                    CHECK_EQ(pstr(&o, "act"), std::string("silu"));
                    CHECK(in(o.reads, (rad_buf)b_h(b, "gate_up")));
                    REQUIRE_EQ(o.writes.size(), (size_t)2);         /* codes only */
                    CHECK_EQ(o.writes[0], fq);
                    CHECK_EQ(o.writes[1], fs);
                }
                if (o.op == "had_quant_act_i8g") {
                    CHECK_EQ(pint(&o, "n"), Q);
                    CHECK_EQ(pint(&o, "group"), 256);
                    CHECK(in(o.reads, (rad_buf)b_h(b, "attn_out")));
                    CHECK(in(o.writes, aq));
                    CHECK(in(o.writes, as));
                }
            }
            CHECK_EQ(norms, 16);
            /* The post norms and the final norm stay plain rmsnorm (with the q/k norms). */
            for (const RecOp& o : b.ops)
                if (o.op == "rmsnorm" && o.w.size() == 1)
                    CHECK((int64_t)o.w[0] != w_attn_norm && (int64_t)o.w[0] != w_ffn_norm);

            /* The views and the pair each GEMM reads. */
            const struct { const char* base; int64_t n, k; bool col; rad_buf q, s; } linears[] = {
                { "blk.0.attn_q", 512, 256, false, hq, hs },
                { "blk.0.attn_gate", 512, 256, false, hq, hs },
                { "blk.0.attn_k", 32, 256, false, hq, hs },
                { "blk.0.attn_v", 32, 256, false, hq, hs },
                { "blk.0.attn_output", 256, 512, true, aq, as },
                { "blk.0.ffn_gate_up", 1024, 256, false, hq, hs },
                { "blk.0.ffn_down", 256, 512, true, fq, fs },
            };
            for (const auto& ln : linears) {
                const int64_t rows = ln.col ? ln.n : ln.n / world;
                const int64_t cols = ln.col ? ln.k / world : ln.k;
                const RecWeight* w = find_w(b, (std::string(ln.base) + ".weight").c_str());
                REQUIRE(w != nullptr);
                CHECK_EQ(std::string(w->d.planes), std::string("codes"));
                CHECK_EQ(w->d.dtype, (uint32_t)RAD_I8);
                CHECK_EQ(w->d.shape[0], rows);
                CHECK_EQ(w->d.shape[1], cols);
                const RecWeight* ws = find_w(b, (std::string(ln.base) + ".scale").c_str());
                REQUIRE(ws != nullptr);
                CHECK_EQ(std::string(ws->d.planes), std::string("scale,scale.1"));
                CHECK_EQ(ws->d.dtype, (uint32_t)RAD_U8);
                CHECK_EQ(ws->d.shape[0], rows);
                CHECK_EQ(ws->d.shape[1], cols);
                const RecOp* gm = q_gemm_of(b, ln.base);
                REQUIRE(gm != nullptr);
                CHECK_EQ(pint(gm, "N"), rows);
                CHECK_EQ(pint(gm, "K"), cols);
                REQUIRE_EQ(gm->reads.size(), (size_t)2);
                CHECK_EQ(gm->reads[0], ln.q);
                CHECK_EQ(gm->reads[1], ln.s);
            }
            CHECK_EQ(Q, pint(q_gemm_of(b, "blk.0.attn_q"), "N"));
            CHECK_EQ(KV, pint(q_gemm_of(b, "blk.0.attn_k"), "N"));
            /* gate|up in the checkpoint's row order, two row parts a rank. */
            const RecMap* gu = find_map(b, "blk.0.ffn_gate_up.weight");
            REQUIRE(gu != nullptr);
            REQUIRE_EQ(gu->src.size(), (size_t)2);
            CHECK_EQ(gu->src[0], std::string("model.language_model.layers.0.mlp.gate_proj.weight"));
            CHECK_EQ(gu->src[1], std::string("model.language_model.layers.0.mlp.up_proj.weight"));
            const RecWeight* guw = find_w(b, "blk.0.ffn_gate_up.weight");
            REQUIRE(guw != nullptr);
            REQUIRE_EQ(guw->d.n_row_parts, 2);
            CHECK_EQ(guw->d.row_parts[0], F);
            CHECK_EQ(guw->d.row_parts[1], F);
            /* Layer 0's q is mapped once, by the trunk probe. */
            int qmaps = 0;
            for (const RecMap& mp : b.maps) qmaps += mp.declared == "blk.0.attn_q.weight";
            CHECK_EQ(qmaps, 1);

            /* The head: plain int8, read by logits_gemm; the norms stay f32 gains. */
            const RecWeight* head = find_w(b, "output.weight");
            REQUIRE(head != nullptr);
            CHECK_EQ(head->d.dtype, (uint32_t)RAD_I8);
            const RecOp* lg = find_op(b, "logits_gemm");
            REQUIRE(lg != nullptr);
            REQUIRE_EQ(lg->w.size(), (size_t)1);
            CHECK_EQ((int64_t)lg->w[0], w_h(b, "output.weight"));
            CHECK_EQ(find_w(b, "blk.0.attn_norm.weight")->d.dtype, (uint32_t)RAD_F32);

            /* Decode and a 200-row prefill issue every producer and GEMM once a layer. */
            for (int pass = 0; pass < 2; ++pass) {
                auto k = pass == 0 ? kit(b, { 1, 1, 1, 1 }, {}) : kit(b, {}, { 200 });
                auto v = run_passes(b, museglimmer_ct::step, { k.get() }, rank, world);
                CHECK_PASSES(b, v);
                CHECK_LINT(b, v[0].get());
                CHECK(v[0]->step_fail.empty());
                const std::vector<std::string> is = issued_ops(b, *v[0]);
                CHECK_EQ(count_issued(is, "rmsnorm_had_quant_i8g"), 16);
                CHECK_EQ(count_issued(is, "gated_had_quant_i8g"), 8);
                CHECK_EQ(count_issued(is, "had_quant_act_i8g"), 8);
                CHECK_EQ(count_issued(is, "gemm_nt_q"), 56);
                CHECK_EQ(count_issued(is, "all_reduce"), world > 1 ? 1 + 16 : 0);
            }
        }
}

/* Where no loaded library declares the rotating norm or the rotating gated product (the CPU
 * kernels), the plain norm and silu_mul run and the first linear reading each input quantises it
 * with had_quant_act_i8g at group 256: q (gate, k and v read its pair), gate/up, o_proj and
 * down_proj, four a layer. */
TEST(museglimmer_ct_iq8r_quantises_its_inputs_without_the_rotating_producers) {
    for (int world : { 1, 2 }) {
        RadBuilder b;
        b.undeclared = { "rmsnorm_had_quant_i8g", "gated_had_quant_i8g" };
        b.encs = muse_iq8r_encs();
        RadBuildCtx ctx = muse_ctx(world - 1, world);
        REQUIRE_EQ(museglimmer_ct::declare(&b, muse_meta_iq8r().meta(), &ctx), RAD_OK);
        const museglimmer_ct::Model& m = museglimmer_ct::g_model[rad_build_slot(&ctx)];
        CHECK(m.iq8r);
        CHECK(!m.a_h.q8_fed);
        CHECK(!m.a_ffn.q8_fed);
        CHECK_EQ(count_op(b, "rmsnorm_had_quant_i8g"), 0);
        CHECK_EQ(count_op(b, "gated_had_quant_i8g"), 0);
        CHECK_EQ(count_op(b, "silu_mul"), 8);
        CHECK_EQ(count_op(b, "had_quant_act_i8g"), 32);
        for (const RecOp& o : b.ops)
            if (o.op == "had_quant_act_i8g") CHECK_EQ(pint(&o, "group"), 256);
        CHECK_EQ(count_op(b, "gemm_nt_q"), 56);

        /* q's quantiser writes the pair gate, k and v read. */
        const rad_buf hq = (rad_buf)b_h(b, "h.q8");
        const RecOp* qq = writer_of(b, "had_quant_act_i8g", hq);
        REQUIRE(qq != nullptr);
        CHECK(in(qq->reads, (rad_buf)b_h(b, "h")));
        for (const char* base : { "blk.0.attn_q", "blk.0.attn_gate", "blk.0.attn_k",
                                  "blk.0.attn_v", "blk.0.ffn_gate_up" }) {
            const RecOp* gm = q_gemm_of(b, base);
            REQUIRE(gm != nullptr);
            CHECK_EQ(gm->reads[0], hq);
        }
        const RecOp* dq = writer_of(b, "had_quant_act_i8g", (rad_buf)b_h(b, "ffn.q8"));
        REQUIRE(dq != nullptr);
        CHECK(in(dq->reads, (rad_buf)b_h(b, "ffn")));

        auto k = kit(b, { 1, 1, 1, 1 }, {});
        auto v = run_passes(b, museglimmer_ct::step, { k.get() }, world - 1, world);
        CHECK_PASSES(b, v);
        CHECK_LINT(b, v[0].get());
        CHECK_EQ(count_issued(issued_ops(b, *v[0]), "had_quant_act_i8g"), 32);
        CHECK_EQ(count_issued(issued_ops(b, *v[0]), "gemm_nt_q"), 56);
    }
}

/* An IQ8R trunk is IQ8R in every trunk linear, and its widths take whole 256-column groups. Refused
 * by name: an IQ8R trunk with one linear on the 128-group int8 grid or left bf16, an IQ8R linear
 * under a trunk whose layer 0 q is not IQ8R, an IQ8R-shaped weight at the wrong group (not IQ8R,
 * so no form matches), and a model whose hidden width 256 does not divide. */
TEST(museglimmer_ct_refuses_a_mixed_or_malformed_iq8r_trunk) {
    MetaKV mk = muse_meta_iq8r();
    auto declare_rc = [&](const std::vector<std::pair<std::string, RadEncoding>>& encs,
                          const RadModelMeta* meta) {
        RadBuilder b;
        b.encs = encs;
        RadBuildCtx ctx = muse_ctx(0, 1);
        return stderr_of_rc([&] { return museglimmer_ct::declare(&b, meta, &ctx); });
    };
    {
        auto e = muse_iq8r_encs();
        for (auto& [key, enc] : e)
            if (key == "attn_v.") enc = muse_i8g_enc();
        CHECK_EQ(declare_rc(e, mk.meta()), RAD_E_DTYPE);
    }
    {
        auto e = muse_iq8r_encs();
        for (auto& [key, enc] : e)
            if (key == "ffn_down.") enc = rad_enc_plain(RAD_BF16);
        CHECK_EQ(declare_rc(e, mk.meta()), RAD_E_DTYPE);
    }
    {
        auto e = muse_linears(rad_enc_plain(RAD_BF16));
        for (auto& [key, enc] : e)
            if (key == "attn_output.") enc = muse_iq8r_enc();
        CHECK_EQ(declare_rc(e, mk.meta()), RAD_E_DTYPE);
    }
    {
        RadEncoding bad = rad_enc_affine(RAD_I8, RAD_U8, 1, rad::arch::RAD_FP8_BLOCK);
        rad_enc_add_plane(&bad, "scale.1", RAD_BF16, 1, 0);
        rad_enc_copy_str(bad.transform, "fwht128");
        CHECK_EQ(declare_rc(muse_linears(bad), mk.meta()), RAD_E_DTYPE);
    }
    CHECK_EQ(declare_rc(muse_iq8r_encs(), muse_meta_quant().meta()), RAD_E_UNSUPPORTED);
}

/* ==================================================================== 6. the DFlash drafter */

/* The container's `draft.` config for the assistant (`muse-glimmer-assistant`): two plain
 * attention layers that denoise a block of four (the anchor plus three masks, three drafts), the
 * trunk's residual tapped after layers 1, 3 and 5. Heads run wider than the trunk fixture's so
 * the recipe's quantised grids fit their scale blocks a rank at a time (the trunk fixture's two
 * KV heads leave k and v eight rows a rank at tp 2, which no quantised form serves). */
MetaKV muse_draft_meta() {
    MetaKV k = muse_meta();
    k.set("draft.num_hidden_layers", "2")
        .set("draft.head_dim", "32")
        .set("draft.num_attention_heads", "32")
        .set("draft.num_key_value_heads", "8")
        .set("draft.intermediate_size", "512")
        .set("draft.rms_norm_eps", "1e-5")
        .set("draft.rope_theta", "500000")
        .set("draft.block_size", "4")
        .set("draft.target_layer_ids", "1 3 5")
        .set("draft.mask_token_id", "511");
    return k;
}

/* The recipe's drafter grid (the dspark rules of museglimmer-30b-modelopt-nvfp4.recipe, what the
 * shipped `*-dflash-fp4.rad` carries): attention at block fp8 e4m3*bf16[128x128], fc and the
 * MLPs at MXFP4 e2m1*e8m0[1x32], the head the trunk's own. */
std::vector<std::pair<std::string, RadEncoding>> muse_draft_encs() {
    const RadEncoding fp8 = rad_enc_affine(RAD_F8E4M3, RAD_BF16, rad::arch::RAD_FP8_BLOCK,
                                           rad::arch::RAD_FP8_BLOCK);
    const RadEncoding mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, rad::arch::RAD_MX_GROUP);
    return { { "dspark.fc.", mx },    { "attn_q.", fp8 },      { "attn_k.", fp8 },
             { "attn_v.", fp8 },      { "attn_output.", fp8 }, { "ffn_gate_up.", mx },
             { "ffn_down.", mx } };
}

/* The block embeds from the trunk's vocab-sharded table, so its lookup is partial at tp > 1:
 * every row whose token sits in another rank's slice reads zero -- and the block's rows are one
 * sampled token plus masks whose one id lives on a single rank. So the query pass reduces the
 * lookup exactly, right after `embed_lookup` (the trunk's own embedding reduce, and dflash2's and
 * dspark's), and the context pass -- k and v from row-split weights over the replicated taps --
 * runs no collective at all. tp 1 declares none of it. */
TEST(muse_dflash_reduces_the_sharded_embedding_before_the_block) {
    MetaKV mk = muse_draft_meta();
    for (int world : { 1, 2 }) {
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(world - 1, world, 3);
        REQUIRE_EQ(museglimmer::declare(&b, mk.meta(), &ctx), RAD_OK);

        /* One engine step, as the scheduler drives it: the trunk's step, the drafter's context
         * pass over what the trunk committed, the query pass over the staged block. */
        auto trunk = kit(b, { 1, 1 }, {});
        auto hp = context_pass(b, *trunk);
        auto qp = query_pass(b, 1, 4, 3);
        auto v = run_passes(b, museglimmer::step, { trunk.get(), hp.get(), qp.get() },
                            world - 1, world);
        CHECK_PASSES(b, v);

        /* The context pass projects and stores k and v; nothing it runs is a partial sum. */
        const auto ci = issued_ops(b, *v[1]);
        CHECK_EQ(count_issued(ci, "all_reduce"), 0);
        CHECK_EQ(count_issued(ci, "embed_lookup"), 0);

        /* The query pass: the reduce directly after the lookup, before the block's layers; the
         * per-layer reduces behind o and down; the ranks' logits gathered for the draft's argmax.
         * At tp 1 none of it. */
        const auto qi = issued_ops(b, *v[2]);
        REQUIRE(qi.size() >= 2);
        CHECK_EQ(qi[0], std::string("embed_lookup"));
        CHECK_EQ(qi[1], std::string(world > 1 ? "all_reduce" : "cast"));
        CHECK_EQ(count_issued(qi, "all_reduce"), world > 1 ? 5 : 0);   /* embed + 2 a layer */
        CHECK_EQ(count_issued(qi, "all_gather"), world > 1 ? 1 : 0);
        if (world > 1) {
            /* Exact, over the block's embedded rows, in place. */
            const RecIssue* ar = issue_of(b, *v[2], "all_reduce", 0);
            REQUIRE(ar != nullptr);
            const RecOp* o = b.op_of(ar->op);
            REQUIRE(o != nullptr);
            CHECK_EQ(pint(o, "exact"), 1);
            CHECK_EQ(pint(o, "world_size"), world);
            CHECK_EQ(pstr(o, "dtype"), std::string("bf16"));
            REQUIRE(!o->reads.empty());
            CHECK_EQ(o->reads[0], b_h(b, "ds_h"));
            CHECK_EQ(o->writes[0], b_h(b, "ds_h"));
        }
        /* The walk's winner lands in the proposal buffer the core reads. */
        const RecIssue* tk = issue_of(b, *v[2], "row_topk", 0);
        REQUIRE(tk != nullptr);
        CHECK_EQ(tk->opd[1].kind, (uint8_t)RAD_OPK_BUF);
        CHECK_EQ(tk->opd[1].handle, b.drafter.proposal);
        /* The head scores the rows `out_ids` names, gathered into `hg`: with several sequences
         * staged, "the last S rows" of the block would drop each sequence's first mask row and
         * score its neighbour's anchor, whose logits pose as that sequence's first draft. */
        CHECK_EQ(count_issued(qi, "gather_rows"), 1);
        const RecIssue* lg = issue_of(b, *v[2], "logits_gemm", 0);
        REQUIRE(lg != nullptr);
        const RecOp* lgop = b.op_of(lg->op);
        REQUIRE(lgop != nullptr);
        REQUIRE(!lgop->reads.empty());
        CHECK_EQ(lgop->reads[0], b_h(b, "ds_hg"));
    }
    /* Two drafting sequences: the gather still takes the rows `out_ids` names -- the fake names
     * rows 1..3 and 5..7 of the eight staged -- and the pass lints clean. */
    {
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(1, 2, 3);
        REQUIRE_EQ(museglimmer::declare(&b, mk.meta(), &ctx), RAD_OK);
        auto trunk = kit(b, { 1, 1 }, {});
        auto hp = context_pass(b, *trunk);
        auto qp = query_pass(b, 2, 4, 3);
        auto v = run_passes(b, museglimmer::step, { trunk.get(), hp.get(), qp.get() }, 1, 2);
        CHECK_PASSES(b, v);
        const auto qi = issued_ops(b, *v[2]);
        CHECK_EQ(count_issued(qi, "gather_rows"), 1);
        CHECK_EQ(count_issued(qi, "logits_gemm"), 1);
    }
}

/* DFlash2 recognizes its own config fields, probes the IQR fc before declaring activation pairs,
 * and serves its U3/U4 linears on the same 128-wide rotated grid. Its 202048-row production head
 * is unsupported by tiled 2-bit/int8 head kernels, so retain a BF16 head and apply Muse's own
 * multiplier and softcap in the draft graph. */
TEST(muse_dflash2_serves_a_rotated_iqr_drafter) {
    MetaKV mk = muse_dflash2_meta();
    RadBuilder b;
    b.encs = muse_dflash2_iqr_encs(false);
    RadBuildCtx ctx = spec_ctx(0, 1, 3);
    REQUIRE_EQ(museglimmer::declare(&b, mk.meta(), &ctx), RAD_OK);
    const museglimmer::Model& m = museglimmer::g_model[rad_build_slot(&ctx)];
    CHECK(m.have_df2);
    CHECK(!m.have_df);
    CHECK(m.df2.cfg.act_i8);
    CHECK_EQ(m.df2.cfg.act_g8, 128);
    CHECK(near(m.df2.cfg.omul, 0.19611613513818404, 1e-12));
    CHECK_EQ(m.df2.cfg.scap, 20.0);
    CHECK_EQ(find_b(b, "df_srcq")->d.dtype, (uint32_t)RAD_I8);
    CHECK_EQ(find_b(b, "df_h2q")->d.dtype, (uint32_t)RAD_I8);
    CHECK_EQ(find_b(b, "df_ats")->d.shape[1], 2);
    const RecWeight* fc = find_w(b, "dflash.fc.weight");
    REQUIRE(fc != nullptr);
    CHECK_EQ(fc->d.dtype, (uint32_t)RAD_U4);
    CHECK_EQ(std::string(fc->d.planes), std::string("codes,trellis,trellis.mad"));
    const RecWeight* gu = find_w(b, "dflash.blk.0.ffn_gate_up.weight");
    REQUIRE(gu != nullptr);
    CHECK_EQ(gu->d.dtype, (uint32_t)RAD_U3);
    const RecWeight* head = find_w(b, "dflash.draft_head.weight");
    REQUIRE(head != nullptr);
    CHECK_EQ(head->d.dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(count_op(b, "logit_softcap"), 2);  /* target head and DFlash2 head */
    bool bf16_draft_head = false;
    for (const RecOp& o : b.ops)
        if (o.op == "gemm_nt" && pint(&o, "N") == 512 &&
            in(o.w, (rad_weight)w_h(b, "dflash.draft_head.weight")))
            bf16_draft_head = true;
    CHECK(bf16_draft_head);

    auto trunk = kit(b, { 1, 1 }, {});
    auto hp = context_pass(b, *trunk);
    auto qp = query_pass(b, 1, 4, 3);
    auto v = run_passes(b, museglimmer::step, { trunk.get(), hp.get(), qp.get() }, 0, 1);
    CHECK_PASSES(b, v);
    for (const auto& pass : v) CHECK(pass->step_fail.empty());
    CHECK_EQ(count_issued(issued_ops(b, *v[1]), "had_quant_act_i8g"), 2);
    CHECK_EQ(count_issued(issued_ops(b, *v[2]), "had_quant_act_i8g"), 8);
    CHECK_EQ(count_issued(issued_ops(b, *v[2]), "gemm_nt_q"), 12);

    /* Exercise the same rotated drafter next to the actual compressed-tensors trunk path. */
    RadBuilder ct;
    ct.encs = muse_dflash2_iqr_encs(true);
    RadBuildCtx cctx = spec_ctx(0, 1, 3);
    REQUIRE_EQ(museglimmer_ct::declare(&ct, mk.meta(), &cctx), RAD_OK);
    const museglimmer_ct::Model& cm = museglimmer_ct::g_model[rad_build_slot(&cctx)];
    CHECK(cm.iqr_v4);
    CHECK(cm.have_df2);
    CHECK(cm.df2.cfg.act_i8);
    CHECK_EQ(cm.df2.cfg.act_g8, 128);
    auto ctrunk = kit(ct, { 1, 1 }, {});
    auto chp = context_pass(ct, *ctrunk);
    auto cqp = query_pass(ct, 1, 4, 3);
    auto cv = run_passes(ct, museglimmer_ct::step, { ctrunk.get(), chp.get(), cqp.get() }, 0, 1);
    CHECK_PASSES(ct, cv);
    for (const auto& pass : cv) CHECK(pass->step_fail.empty());
    CHECK_EQ(count_issued(issued_ops(ct, *cv[0]), "gemm_nt_q"), 56);
    CHECK_EQ(count_issued(issued_ops(ct, *cv[2]), "gemm_nt_q"), 12);
}

/* A drafter converted to block fp8 attention and MXFP4 MLPs (the grid of the NVFP4 release's
 * conversion) is refused with the trunk at declare: neither form is served. */
TEST(muse_dflash_refuses_a_block_fp8_and_mxfp4_drafter) {
    MetaKV mk = muse_draft_meta();
    for (int world : { 1, 2 }) {
        RadBuilder b;
        b.encs = muse_draft_encs();
        RadBuildCtx ctx = spec_ctx(world - 1, world, 3);
        CHECK_EQ(museglimmer::declare(&b, mk.meta(), &ctx), RAD_E_UNSUPPORTED);
    }
}

/* Context extension. The family is unusually suited to it: only the 2048-wide sliding layers
 * carry rotary and the full-attention layers are NoPE, so serving past the trained 131072
 * positions is `--max-model-len` alone (the engine warns; positions beyond training are
 * extrapolated), and `--rope-scaling yarn:FACTOR` writes text_config.rope_parameters for a
 * stronger stretch. The blend belongs to the sliding layers' rope ops only; the drafter reads its
 * own draft.rope_parameters and does not inherit the target's. */
TEST(museglimmer_carries_yarn_on_the_sliding_layers_rope) {
    MetaKV mk = muse_meta();
    mk.set("text_config.rope_parameters.rope_type", "yarn")
        .set("text_config.rope_parameters.factor", "8")
        .set("text_config.rope_parameters.original_max_position_embeddings", "131072");
    RadBuilder b;
    RadBuildCtx ctx = muse_ctx(0, 1);
    REQUIRE_EQ(museglimmer::declare(&b, mk.meta(), &ctx), RAD_OK);

    /* Six sliding layers x (q, k): every rope op carries the blend, computed at the sliding
     * layers' own theta 500000 over the 128 rotary dimensions. */
    const int ropes = count_op(b, "rope");
    CHECK_EQ(ropes, 12);
    const RecOp* first = find_op(b, "rope", 0);
    const RecOp* last = find_op(b, "rope", ropes - 1);
    REQUIRE(first != nullptr);
    REQUIRE(last != nullptr);
    for (const RecOp* rp : { first, last }) {
        CHECK(near(pflt(rp, "yarn_factor"), 8.0, 1e-9));
        CHECK(near(pflt(rp, "yarn_magnitude"), 1.0 + 0.1 * std::log(8.0), 1e-9));
        /* The fixture's rotary is 8 wide: the correction range is 1.98..3.03 at beta 32/1. */
        CHECK(near(pflt(rp, "yarn_low"), 1.0, 1e-9));    /* floor at beta_fast 32 */
        CHECK(near(pflt(rp, "yarn_high"), 4.0, 1e-9));   /* ceil at beta_slow 1 */
    }
}

TEST(the_muse_dflash_drafter_reads_its_own_yarn_keys) {
    MetaKV mk = muse_draft_meta();
    mk.set("draft.rope_parameters.rope_type", "yarn")
        .set("draft.rope_parameters.factor", "4")
        .set("draft.rope_parameters.original_max_position_embeddings", "131072");
    for (int world : { 1, 2 }) {
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(world - 1, world, 3);
        REQUIRE_EQ(museglimmer::declare(&b, mk.meta(), &ctx), RAD_OK);
        /* The drafter's three rope ops (context, query q and k) carry the blend at the drafter's
         * head_dim 32 and theta 500000; the trunk's twelve stay unscaled. */
        const RecOp* dr = find_op(b, "rope", 12);
        REQUIRE(dr != nullptr);
        CHECK(near(pflt(dr, "yarn_factor"), 4.0, 1e-9));
        CHECK(near(pflt(dr, "yarn_magnitude"), 1.0 + 0.1 * std::log(4.0), 1e-9));
        CHECK(near(pflt(dr, "yarn_low"), 7.0, 1e-9));
        CHECK(near(pflt(dr, "yarn_high"), 13.0, 1e-9));
        const RecOp* tr = find_op(b, "rope", 0);
        REQUIRE(tr != nullptr);
        CHECK_EQ(pflt(tr, "yarn_factor"), -12345.0);   /* absent: the trunk is unscaled */
    }
}

RAD_TEST_MAIN()
