/* deepseek41_test.cpp -- DeepSeek-V4.1's declared caches under --kv-cache-dtype, asserted statically
 * against the recording builder of arch_fakes.h. */
#define RAD_ARCH_NO_EXPORTS 1

#include "arch_fakes.h"

#include <map>
#include <string>

/* ==================================================================== the plugin under test */

/* The step and the chat format are not exercised here. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "../arch/deepseek41/deepseek41.cpp"
#pragma GCC diagnostic pop

namespace {

/* A small DeepSeek-V4.1 in llama.cpp's GGUF keys: three layers at compress ratios 0, r, r (the kv
 * and index source at layer 1), the 512-wide latent and 128-wide index keys the kernels serve,
 * everything else narrow (an expert whole 128 x 128 scale blocks on each of three ranks). Two
 * output groups, so a third rank is attention-zero. One ratio a model: the recording builder has
 * one block size for every group, and a compressed group's must be its ratio. */
struct Ds41Meta {
    std::vector<std::string> k, v;
    std::vector<const char*> kp, vp;
    RadModelMeta meta{};
    explicit Ds41Meta(int r) {
        const std::string ratios = "0," + std::to_string(r) + "," + std::to_string(r);
        const std::pair<const char*, const char*> kv[] = {
            { "deepseek41.block_count", "3" },
            { "deepseek41.embedding_length", "256" },
            { "deepseek41.attention.head_count", "4" },
            { "deepseek41.attention.key_length", "512" },
            { "deepseek41.rope.dimension_count", "64" },
            { "deepseek41.attention.q_lora_rank", "64" },
            { "deepseek41.attention.output_group_count", "2" },
            { "deepseek41.attention.output_lora_rank", "64" },
            { "deepseek41.attention.sliding_window", "128" },
            { "deepseek41.expert_count", "12" },
            { "deepseek41.expert_used_count", "2" },
            { "deepseek41.expert_feed_forward_length", "384" },
            { "deepseek41.hyper_connection.count", "4" },
            { "deepseek41.hyper_connection.sinkhorn_iterations", "3" },
            { "deepseek41.hyper_connection.epsilon", "1e-6" },
            { "deepseek41.attention.indexer.head_count", "4" },
            { "deepseek41.attention.indexer.key_length", "128" },
            { "deepseek41.attention.indexer.top_k", "64" },
            { "deepseek41.attention.compress_rope_freq_base", "160000" },
            { "deepseek41.attention.compress_ratios", ratios.c_str() },
            { "deepseek41.attention.kv_source_layers", "1" },
            { "deepseek41.attention.indexer.source_layers", "1" },
        };
        for (const auto& e : kv) { k.push_back(e.first); v.push_back(e.second); }
        for (size_t i = 0; i < k.size(); ++i) { kp.push_back(k[i].c_str()); vp.push_back(v[i].c_str()); }
        meta.arch_id = "deepseek41";
        meta.n_layers = 3;
        meta.n_embd = 256;
        meta.n_head = 4;
        meta.n_head_kv = 1;
        meta.head_dim = 512;
        meta.n_vocab = 1200;
        meta.n_expert = 12;
        meta.n_expert_used = 2;
        meta.n_expert_shared = 1;
        meta.rms_eps = 1e-6f;
        meta.rope_theta = 10000.0f;
        meta.n_kv = (int)kp.size();
        meta.kv_key = kp.data();
        meta.kv_val = vp.data();
    }
};

/* The routed experts as plain bf16 planes: the bf16 grouped GEMMs on the bf16 activation. */
void ds41_encs(RadBuilder& b) {
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    b.encs = { { "ffn_gate_up_exps", plain }, { "ffn_down_exps", plain } };
}

RadBuildCtx ds41_ctx(int kv_dtype, int rank, int world) {
    RadBuildCtx c{};
    c.rank = rank;
    c.world_size = world;
    c.max_tok = 64;
    c.max_seqs = 2;
    c.max_ctx = 4096;
    c.max_spec = 0;
    c.kv_dtype = kv_dtype;
    return c;
}

}  // namespace

/* --kv-cache-dtype fp8 narrows the compressed latent and index-key groups alone to block-scaled
 * E4M3 entries: each group is E4M3 at the entry's bytes (544 latent, 144 index keys, over 2r halves
 * of the r slots of a block), every entry_store packs in its group's format and every reader
 * (attn_sparse_sink at ratio > 0, index_score) names it. The window and the compressor's tail keep
 * bf16 and f32, and bf16 (or no dtype) declares the bf16 groups with no format anywhere. The
 * attention-zero rank (rank 2 of 3: two output groups) declares the same groups and stores. */
TEST(ds41_fp8_kv_cache_narrows_the_latent_and_index_groups_only) {
    for (int r : { 1, 2 })
    for (int world : { 1, 3 })
        for (int kv_dtype : { (int)RAD_DT_INVALID, (int)RAD_BF16, (int)RAD_F8E4M3 }) {
            Ds41Meta dm(r);
            const bool narrow = kv_dtype == RAD_F8E4M3;
            const int rank = world - 1;
            RadBuilder b;
            b.kv_block = r;
            ds41_encs(b);
            const RadBuildCtx build = ds41_ctx(kv_dtype, rank, world);
            int rc = RAD_OK;
            const std::string err = stderr_of([&] { rc = deepseek41::declare(&b, &dm.meta, &build); });
            if (rc != RAD_OK) std::fprintf(stderr, "%s", err.c_str());
            REQUIRE_EQ(rc, RAD_OK);
            std::map<std::string, const RecKV*> groups;
            for (const auto& g : b.kv_groups) groups[g.name] = &g;
            {
                const std::string cn = "kv_compressed_r" + std::to_string(r);
                const std::string in = "kv_index_r" + std::to_string(r);
                REQUIRE(groups.count(cn) && groups.count(in));
                const RecKV* cg = groups[cn];
                const RecKV* ig = groups[in];
                CHECK_EQ(cg->d.dtype, narrow ? (uint32_t)RAD_F8E4M3 : (uint32_t)RAD_BF16);
                CHECK_EQ(cg->d.head_dim, (narrow ? 544 : 512) / r / 2);
                CHECK_EQ(ig->d.dtype, narrow ? (uint32_t)RAD_F8E4M3 : (uint32_t)RAD_BF16);
                CHECK_EQ(ig->d.head_dim, (narrow ? 144 : 128) / r / 2);
            }
            REQUIRE(groups.count("kv_window"));
            CHECK_EQ(groups["kv_window"]->d.dtype, (uint32_t)RAD_BF16);
            if (r > 1) {           /* the compressor's ring of a ratio above 1 */
                REQUIRE(groups.count("kv_comp_tail_r2"));
                CHECK_EQ(groups["kv_comp_tail_r2"]->d.dtype, (uint32_t)RAD_F32);
            }
            int lat_st = 0, idx_st = 0, attn_c = 0, attn_w = 0, scores = 0, bad = 0;
            for (const auto& o : b.ops) {
                if (o.op == "entry_store") {
                    const long long n = pint(&o, "n");
                    const std::string want = !narrow ? "<absent>" : (n == 512 ? "e4m3_e4m3_16" : "e4m3_ue8m0_32");
                    lat_st += n == 512;
                    idx_st += n == 128;
                    if (pstr(&o, "format") != want) ++bad;
                } else if (o.op == "attn_sparse_sink") {
                    const bool comp = pint(&o, "ratio") > 0;
                    (comp ? attn_c : attn_w) += 1;
                    if (pstr(&o, "latent_format") != (narrow && comp ? "e4m3_e4m3_16" : "<absent>")) ++bad;
                } else if (o.op == "index_score") {
                    ++scores;
                    if (pstr(&o, "key_format") != (narrow ? "e4m3_ue8m0_32" : "<absent>")) ++bad;
                }
            }
            CHECK_EQ(bad, 0);
            CHECK_EQ(lat_st, 1);   /* one store a kv source (layer 1) */
            CHECK_EQ(idx_st, 1);
            if (rank == 0) {       /* an attention rank reads them: 2 compressing layers, 1 window-only,
                                    * the index source's runs of rows scored */
                CHECK_EQ(attn_c, 2);
                CHECK_EQ(attn_w, 1);
                CHECK(scores >= 1);
            }
            bool noted = false;
            for (const auto& n : b.notes) noted |= n.find("KV cache fp8") != std::string::npos;
            CHECK_EQ(noted, narrow);
        }
}

/* Only bf16 and fp8 caches are served; the probe says fp8 is one. */
TEST(ds41_kv_cache_dtypes_other_than_bf16_and_fp8_are_refused) {
    Ds41Meta dm(2);
    for (int kv_dtype : { (int)RAD_I8, (int)RAD_F16 }) {
        RadBuilder b;
        b.kv_block = 2;
        ds41_encs(b);
        const RadBuildCtx build = ds41_ctx(kv_dtype, 0, 1);
        int rc = RAD_OK;
        const std::string err = stderr_of([&] { rc = deepseek41::declare(&b, &dm.meta, &build); });
        CHECK_EQ(rc, RAD_E_UNSUPPORTED);
        CHECK(err.find("--kv-cache-dtype") != std::string::npos);
    }
    RadArchProbe pb{};
    REQUIRE_EQ(deepseek41::probe(&dm.meta, &pb), RAD_OK);
    CHECK_EQ(pb.kv_fp8_ok, 1);
}

/* EXL3 routed experts (radiance-exl3's encoding: gate_up stacked with suh.1, down plain) are served
 * whole on their rank by moe_exl3_gate_up / moe_exl3_down, two parity tables a plane, K and the
 * codebook each layer's own (layer 0 at K 2.5, the pair-symbol codes plane at half the columns; the
 * rest at K 2). At one rank and expert parallel over three (12 / 3 = 4 a rank), no other expert GEMM
 * is declared beside them, and a step issues each op over this rank's parity tables of the layer's
 * experts in the operand positions exl3_abi.h gives. */
TEST(ds41_exl3_experts_are_served_by_the_exl3_ops) {
    RadEncoding gu2{}, dn2{}, gu25{}, dn25{};
    REQUIRE_EQ(exl3_encoding_make(2, EXL3_CB_MUL1, 1, &gu2), RAD_OK);
    REQUIRE_EQ(exl3_encoding_make(2, EXL3_CB_MUL1, 0, &dn2), RAD_OK);
    REQUIRE_EQ(exl3_encoding_make(EXL3_HALF(2), EXL3_CB_MUL1, 1, &gu25), RAD_OK);
    REQUIRE_EQ(exl3_encoding_make(EXL3_HALF(2), EXL3_CB_MUL1, 0, &dn25), RAD_OK);
    for (int world : { 1, 3 }) {
        Ds41Meta dm(2);
        const int rank = world - 1;
        const int64_t ne = 12 / world, base = rank * ne;
        RadBuilder b;
        b.kv_block = 2;
        b.encs = { { "blk.0.ffn_gate_up_exps", gu25 }, { "blk.0.ffn_down_exps", dn25 },
                   { "ffn_gate_up_exps", gu2 }, { "ffn_down_exps", dn2 } };
        const RadBuildCtx build = ds41_ctx(RAD_BF16, rank, world);
        int rc = RAD_OK;
        const std::string err = stderr_of([&] { rc = deepseek41::declare(&b, &dm.meta, &build); });
        if (rc != RAD_OK) std::fprintf(stderr, "%s", err.c_str());
        REQUIRE_EQ(rc, RAD_OK);
        int gu = 0, dn = 0, other = 0;
        std::vector<rad_op> h_gu, h_dn;
        for (size_t i = 0; i < b.ops.size(); ++i) {
            const RecOp& o = b.ops[i];
            if (o.op == "moe_gemm" || o.op == "moe_gemm_ggml" || o.op.rfind("moe_gemm_q", 0) == 0) ++other;
            const bool g = o.op == "moe_exl3_gate_up", d = o.op == "moe_exl3_down";
            if (!g && !d) continue;
            const int l = g ? gu++ : dn++;
            (g ? h_gu : h_dn).push_back((rad_op)(i + 1));
            CHECK_EQ(pstr(&o, "dtype"), std::string(l == 0 ? "exl3k2h" : "exl3k2"));
            CHECK_EQ(pstr(&o, "codebook"), std::string("mul1"));
            CHECK_EQ(pint(&o, "n_expert"), (long long)ne);
            CHECK_EQ(pint(&o, "h_rot"), 1);
            CHECK_EQ(pint(&o, "N"), g ? 768 : 256);
            CHECK_EQ(pint(&o, "K"), g ? 256 : 384);
            /* gate_up: codes, suh, suh.1, svh and down's suh, each class's half; down: codes, suh, svh */
            REQUIRE_EQ(o.w.size(), (size_t)((g ? 5 : 3) * ne));
            const char* stem = g ? "ffn_gate_up_exps" : "ffn_down_exps";
            const auto name = [&](int64_t e, const char* tail) {
                return "blk." + std::to_string(l) + "." + stem + "." + std::to_string(base + e) + "." + tail;
            };
            /* class 0's codes table opens on the first even expert, class 1's on the first odd */
            const RecWeight& c0 = b.weights[(size_t)o.w[0] - 1];
            const RecWeight& c1 = b.weights[(size_t)o.w[(size_t)((g ? 5 : 3) * ne / 2)] - 1];
            CHECK_EQ(c0.name, name(0, "weight"));
            CHECK_EQ(c1.name, name(1, "weight"));
            CHECK_EQ(c0.d.shape[1], (g ? 256 : 384) / (l == 0 ? 2 : 1));
            CHECK_EQ(c0.d.dtype, exl3_codes_dtype(l == 0 ? EXL3_HALF(2) : 2));
            CHECK_EQ(b.weights[(size_t)o.w[(size_t)(ne / 2)] - 1].name, name(0, "suh"));
        }
        CHECK_EQ(gu, 3);
        CHECK_EQ(dn, 3);
        CHECK_EQ(other, 0);

        /* A step of two decode rows beside a 30-row prefill: each op once a layer, every operand
         * where exl3_positions() puts it, the tables this rank's halves. */
        auto k = kit(b, { 1, 1 }, { 30 });
        auto passes = run_passes(b, deepseek41::step, { k.get() }, rank, world);
        CHECK(steps_ok(passes));
        /* The lint over the routed experts' issues (the attention's findings under these fakes are
         * the trunk's own, not this test's). */
        size_t moe_lint = 0;
        for (const std::string& f : lint_step(b, view(passes)))
            if (f.find("exl3") != std::string::npos || f.find("moe_") != std::string::npos) {
                std::fprintf(stderr, "    lint: %s\n", f.c_str());
                ++moe_lint;
            }
        CHECK_EQ(moe_lint, (size_t)0);
        int n_gu = 0, n_dn = 0;
        for (const auto& is : passes[0]->issues) {
            const bool g = std::find(h_gu.begin(), h_gu.end(), is.op) != h_gu.end();
            const bool d = std::find(h_dn.begin(), h_dn.end(), is.op) != h_dn.end();
            if (!g && !d) continue;
            (g ? n_gu : n_dn) += 1;
            const Exl3Positions& at = *exl3_positions(g ? EXL3_GATE_UP : EXL3_DOWN);
            REQUIRE_EQ(is.opd.size(), (size_t)at.n_opd);
            for (int q = 0; q < 2; ++q) {
                const RadOperand& c = is.opd[(size_t)at.plane[q][EXL3_P_CODES]];
                CHECK_EQ((int)c.kind, (int)RAD_OPK_WTAB);
                CHECK_EQ(c.rows, ne / 2);
            }
            CHECK_EQ((int)is.opd[(size_t)at.x].kind, (int)RAD_OPK_BUF);
            CHECK_EQ((int)is.opd[(size_t)at.y].kind, (int)RAD_OPK_BUF);
        }
        CHECK_EQ(n_gu, 3);
        CHECK_EQ(n_dn, 3);
    }
}

RAD_TEST_MAIN()
