/* gemma4_exl3_test.cpp -- the declared graph of the gemma4_exl3 architecture plugin, with the fakes of
 * tests/arch_fakes.h (no GPU, no model file): an EXL3-V3 container's trunk linears become exl3i_dense ops,
 * the residual junctions write the bf16 norm row they read, at tensor parallelism 2 and 4 each rank declares its
 * share (a KV head shared by some ranks is refused), a step
 * issues them, and two engines in one process keep
 * their own g_model slots. Geometries: Gemma 4 31B and 12B cut to twelve layers, and E4B whole (KV-shared layers,
 * per-layer embeddings).
 */

#define RAD_ARCH_NO_EXPORTS 1
#undef NDEBUG   /* the checks below are assert()s: keep them in a Release build */

#include "arch_fakes.h"
#include "exl3i_encoding.h"

/* The container: these weights answer exl3i encodings, every other weight the fakes' (plain bf16). */
struct V3Weight { std::string name; int m, cb, stacked; int64_t rows, cols; };
static std::vector<V3Weight> g_v3;

/* A vocabulary table's form, answered for token_embd.weight (g_tok) and for the E-series' ple_token_embd.weight
 * (g_ple): 0 the fakes' bf16; B > 0 tq, B-bit codes with a bf16 scale per 128 columns and a 2^B table under fwht128
 * (a B past 8 is a u9 plane, which embed_tq does not take); -m an exl3i trellis at rate m over [rows, cols], stacked
 * when asked. */
static int     g_tok = 0, g_tok_stacked = 0, g_ple = 0;
static int64_t g_tok_rows = 262144, g_tok_cols = 5376, g_ple_rows = 262144, g_ple_cols = 42 * 256;
static int table_encoding(int form, int stacked, int64_t rows, int64_t cols, RadEncoding* out) {
    if (form < 0) return exl3i_encoding_make(-form, EXL3I_CB_MAD, stacked, rows, cols, out);
    char dt[8];
    std::snprintf(dt, sizeof dt, "u%d", form);
    rad_enc_clear(out);
    rad_enc_copy_str(out->scheme, "affine");
    rad_enc_copy_str(out->transform, "fwht128");
    rad_enc_add_plane(out, "codes", rad_dtype_parse(dt), 1, 1);
    rad_enc_add_plane(out, "scale", RAD_BF16, 1, 128);
    rad_enc_add_table(out, "table", RAD_F32, 1, (int64_t)1 << (form <= 8 ? form : 8));
    return RAD_OK;
}
static int tok_encoding(RadEncoding* out) { return table_encoding(g_tok, g_tok_stacked, g_tok_rows, g_tok_cols, out); }

static int test_weight_encoding(RadBuilder* b, const char* source, RadEncoding* out, int64_t* shape, uint32_t* rank) {
    if (source && g_tok && !std::strcmp(source, "token_embd.weight")) return tok_encoding(out);
    if (source && g_ple && !std::strcmp(source, "ple_token_embd.weight"))
        return table_encoding(g_ple, 0, g_ple_rows, g_ple_cols, out);
    for (const auto& w : g_v3)
        if (source && w.name == source)
            return exl3i_encoding_make(w.m, w.cb, w.stacked, w.rows, w.cols, out);
    return rad_weight_encoding(b, source, out, shape, rank);
}
#define rad_weight_encoding test_weight_encoding
#include "../gemma4_exl3.cpp"
#undef rad_weight_encoding

#include <cassert>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

/* Gemma 4 31B's text geometry cut to twelve layers: global layers at (l + 1) % 6 == 0 (5 and 11), ten sliding. */
const char* k_keys[] = {
    "sliding_window", "global_head_dim", "num_global_key_value_heads",
    "rope_parameters.sliding_attention.rope_theta", "rope_parameters.full_attention.rope_theta",
    "rope_parameters.full_attention.partial_rotary_factor", "num_kv_shared_layers",
    "final_logit_softcapping", "full_attention_interval", "attention_k_eq_v",
};
const char* k_vals[] = { "1024", "512", "4", "10000", "1000000", "0.25", "0", "30", "6", "1" };

constexpr int64_t kEmbd = 5376, kFf = 21504, kHeads = 32, kKvS = 16, kKvG = 4, kHdS = 256, kHdG = 512;

/* A model's text geometry: Gemma 4 31B (the constants above) and Gemma 4 12B, whose global layers have one KV head. */
struct Geo { int64_t embd, ff, heads, kv_s, hd_s, hd_g; };
constexpr Geo k31{ kEmbd, kFf, kHeads, kKvS, kHdS, kHdG };
constexpr Geo k12{ 3840, 15360, 16, 8, 256, 512 };

RadModelMeta make_meta(const std::vector<const char*>& keys, const std::vector<const char*>& vals,
                       const Geo& geo = k31) {
    RadModelMeta m{};
    m.arch_id     = "gemma4";
    m.name        = "test-gemma4-exl3";
    m.quant       = "exl3";
    m.n_layers    = 12;
    m.n_embd      = geo.embd;
    m.n_head      = geo.heads;
    m.n_head_kv   = geo.kv_s;
    m.head_dim    = geo.hd_s;
    m.n_ff        = geo.ff;
    m.n_vocab     = 262144;
    m.n_ctx_train = 262144;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 1e6f;
    m.rope_scale  = 1.0f;
    m.n_kv        = (int)keys.size();
    m.kv_key      = keys.data();
    m.kv_val      = vals.data();
    return m;
}

RadModelMeta base_meta() {
    static std::vector<const char*> keys(std::begin(k_keys), std::end(k_keys));
    static std::vector<const char*> vals(std::begin(k_vals), std::end(k_vals));
    return make_meta(keys, vals);
}

/* The trunk's container weights for layers [l0, l1): attn_q / attn_k / attn_v (none on a global layer) / attn_o,
 * the stacked ffn_gate_up and ffn_down, each at the rate m (16 K, exl3i_encoding.h) and codebook mad. */
void fill_container(int l0, int l1, int m, int64_t kv_g = kKvG, const Geo& geo = k31) {
    g_v3.clear();
    for (int l = l0; l < l1; ++l) {
        const bool global = (l + 1) % 6 == 0;
        const int64_t qd = geo.heads * (global ? geo.hd_g : geo.hd_s);
        const int64_t kvd = global ? kv_g * geo.hd_g : geo.kv_s * geo.hd_s;
        auto add = [&](const char* n, int stacked, int64_t rows, int64_t cols) {
            g_v3.push_back({ "blk." + std::to_string(l) + "." + n + ".weight", m, EXL3I_CB_MAD, stacked, rows, cols });
        };
        add("attn_q", 0, qd, geo.embd);
        add("attn_k", 0, kvd, geo.embd);
        if (!global) add("attn_v", 0, kvd, geo.embd);
        add("attn_o", 0, geo.embd, qd);
        add("ffn_gate_up", 1, 2 * geo.ff, geo.embd);
        add("ffn_down", 0, geo.embd, geo.ff);
    }
}

int count_prefix(const RadBuilder& b, const char* prefix) {
    int n = 0;
    for (const auto& op : b.ops)
        if (op.op.rfind(prefix, 0) == 0) ++n;
    return n;
}

std::string param_s(const RecOp& op, const char* key) {
    for (const auto& p : op.p)
        if (p.key == key) return p.sval;
    return "";
}

void test_container_declares_exl3i_ops() {
    printf("[TEST] every trunk linear of an EXL3-V3 container is an exl3i_dense op, fed by the junctions...\n");
    fill_container(0, 12, 40);   /* K2.5 */
    RadBuilder b;
    RadModelMeta meta = base_meta();
    RadBuildCtx ctx = build_ctx();
    assert(gemma4_exl3::declare(&b, &meta, &ctx) == RAD_OK);

    /* 10 sliding x 6 (q, k, v, o, gate_up, down) + 2 global x 5 (no v) */
    assert(count_op(b, "exl3i_dense") == 70);
    assert(count_op(b, "gemm_nt") == 0);
    assert(count_op(b, "exl3_dense") == 0);
    /* nothing quantises an activation for an exl3 linear: they read bf16. The residual junctions fuse
     * the norms: the entry, then attention -> feed-forward in 12 layers and feed-forward -> attention in 11,
     * each writing the bf16 norm row (out_bf16) the next block's linears read. */
    assert(count_prefix(b, "quant_act") == 0);
    assert(count_op(b, "rmsnorm_quant_i8g") == 24);
    assert(count_op(b, "rmsnorm_quant_i8") == 0);
    assert(count_prefix(b, "ar_rmsnorm_quant") == 0);
    assert(count_prefix(b, "gelu_mul_quant") == 0);
    assert(count_op(b, "gelu_mul") == 12);
    assert(count_op(b, "logits_gemm") == 1);   /* the head stays stock */

    std::set<std::string> seen;
    for (const auto& op : b.ops) {
        if (op.op != "exl3i_dense") continue;
        assert(param_s(op, "act") == "f16");   /* no exl3.act.* key: f16 */
        for (const auto& p : op.p) assert(p.key != "aq_planes" || p.ival == 0);
        const auto& rec = b.weights[op.w[0] - 1];
        seen.insert(rec.name);
        const V3Weight* w = nullptr;
        for (const auto& v : g_v3) if (rec.name == v.name + ".record") w = &v;
        assert(w);
        long long N = 0, K = 0;
        for (const auto& p : op.p) { if (p.key == "N") N = p.ival; if (p.key == "K") K = p.ival; }
        assert(N == w->rows && K == w->cols && N % 128 == 0 && K % 128 == 0);
        assert(op.w.size() == (size_t)(w->stacked ? 4 : 3));
        char want[16];
        assert(param_s(op, "rate") == exl3i_rate_str(w->m, want));
        assert(param_s(op, "codebook") == "mad");
    }
    assert(seen.size() == 70);
    printf("  -> OK (70 exl3i_dense ops, record planes of the 70 container weights)\n");
}

/* Tensor parallelism 2 and 4 over an EXL3-V3 container: every trunk linear is still one exl3i_dense op a rank,
 * declared at this rank's extents (exl3_dense_op.h: the loader cuts each plane). q, k, v and the stacked ffn_gate_up
 * split their rows, attn_o and ffn_down their columns. A row split cuts the record by 16-row groups and svh with it
 * (gate_up's record and svh with their two parts) and reads suh whole; a column split cuts the record's K and suh and
 * reads svh whole. */
void test_tp_two_and_four_declare_each_rank_s_share() {
    printf("[TEST] at tensor parallelism 2 and 4 each rank declares its share of every exl3i weight...\n");
    fill_container(0, 12, 40);
    for (int world : { 2, 4 }) {
        for (int rank : { 0, world - 1 }) {
            RadBuilder b;
            RadModelMeta meta = base_meta();
            RadBuildCtx ctx = build_ctx(rank, world);
            assert(gemma4_exl3::declare(&b, &meta, &ctx) == RAD_OK);
            assert(count_op(b, "exl3i_dense") == 70);
            int n_row = 0, n_col = 0;
            for (const auto& op : b.ops) {
                if (op.op != "exl3i_dense") continue;
                const auto& rec = b.weights[op.w[0] - 1];
                const V3Weight* w = nullptr;
                for (const auto& v : g_v3) if (rec.name == v.name + ".record") w = &v;
                assert(w);
                long long N = 0, K = 0;
                for (const auto& p : op.p) {
                    if (p.key == "N") N = p.ival;
                    if (p.key == "K") K = p.ival;
                }
                const bool by_cols = rec.d.shard == RAD_SHARD_COL;
                assert(by_cols || rec.d.shard == RAD_SHARD_ROW);
                const bool o_or_down = w->name.find(".attn_o.") != std::string::npos ||
                                       w->name.find(".ffn_down.") != std::string::npos;
                assert(by_cols == o_or_down);
                assert(by_cols ? (N == w->rows && K * world == w->cols) : (N * world == w->rows && K == w->cols));
                assert(rec.d.shape[0] == N / 16 && rec.d.shape[1] == exl3i_row_words(K, w->m));
                assert(rec.d.n_row_parts == (w->stacked ? 2 : 0));
                if (w->stacked) assert(rec.d.row_parts[0] == kFf / world / 16);
                const auto& suh = b.weights[op.w[1] - 1];
                const auto& svh = b.weights[op.w[2] - 1];
                assert(svh.d.shard == (by_cols ? RAD_SHARD_NONE : RAD_SHARD_ROW));
                assert(suh.d.shard == (by_cols ? RAD_SHARD_COL : RAD_SHARD_NONE));
                if (w->stacked) assert(svh.d.n_row_parts == 2 && svh.d.row_parts[0] == kFf / world);
                if (by_cols) ++n_col; else ++n_row;
            }
            /* 12 layers: o and down by columns (24); q, k, v (no v on the 2 global layers) and gate_up by rows (46) */
            assert(n_col == 24 && n_row == 46);
        }
    }
    printf("  -> OK (TP2 and TP4, first and last rank: 70 exl3i_dense ops, 46 row shares, 24 column shares)\n");
}

/* A KV head shared by some ranks but not all (two global KV heads at four ranks) gives a rank rows that are neither
 * the whole k weight nor an even share of it, which an exl3i record cannot be cut to: refused at declare. The same
 * container at two ranks (one head a rank, an even share) declares. */
void test_a_kv_head_shared_by_some_ranks_is_refused() {
    printf("[TEST] a KV head shared by some ranks over exl3i weights is refused, an even share is not...\n");
    fill_container(0, 12, 40, 2);
    std::vector<const char*> keys(std::begin(k_keys), std::end(k_keys));
    std::vector<const char*> vals(std::begin(k_vals), std::end(k_vals));
    vals[2] = "2";   /* num_global_key_value_heads */
    for (int world : { 2, 4 }) {
        RadBuilder b;
        RadModelMeta meta = make_meta(keys, vals);
        RadBuildCtx ctx = build_ctx(0, world);
        const int rc = gemma4_exl3::declare(&b, &meta, &ctx);
        assert(world == 2 ? rc == RAD_OK : rc != RAD_OK);
    }
    printf("  -> OK (two ranks declare, four refused)\n");
}

/* The prefill overlap (two ranks, a step past two slices) issues the junctions as row slices. An EXL3 linear reads
 * the bf16 norm row on every row, so each slice writes that row (the junction's out_bf16): without it the EXL3
 * linears of a tensor-parallel prefill read a stale row. The test fails if no slice is issued at all. */
void test_overlap_slices_write_the_bf16_row() {
    printf("[TEST] the tensor-parallel prefill overlap's junction slices write the bf16 row the exl3i linears read...\n");
    fill_container(0, 12, 40);
    RadBuilder b;
    RadModelMeta meta = base_meta();
    RadBuildCtx ctx = build_ctx(0, 2);
    ctx.max_tok = 4096;   /* past two slices of the overlap */
    assert(gemma4_exl3::declare(&b, &meta, &ctx) == RAD_OK);
    const int64_t T = 2048;
    const int32_t out[1] = { (int32_t)(T - 1) };
    RadBatch batch{};
    batch.phase = RAD_PHASE_PREFILL; batch.n_tok = T; batch.n_seq = 1; batch.max_q_len = T;
    batch.max_ctx_len = T; batch.n_out = 1; batch.out_ids = out;
    RadCtx c; c.b = &b; c.rank = 0; c.world = 2; c.batch = &batch;
    gemma4_exl3::step(&c, &batch);
    int slices = 0, without = 0;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o || o->op != "rmsnorm_quant_i8g" || i.n < 0 || i.n >= T) continue;
        ++slices;
        if (i.opd.size() < 6 || i.opd[5].kind != RAD_OPK_BUF || i.opd[5].handle == 0) ++without;
    }
    printf("  junction slices %d, without the bf16 row %d\n", slices, without);
    assert(slices > 0);
    assert(without == 0);
    printf("  -> OK\n");
}

/* Gemma 4 12B's geometry (hidden 3840, ffn 15360, 16 q heads, 8 sliding KV heads of 256, one global KV head of 512).
 * At one rank every trunk linear is an exl3i_dense op at its container's extents. At two and four ranks each rank
 * holds the one global KV head whole: its k weight serves whole on every rank (no row share), and the other linears
 * split as at 31B. */
void test_the_12b_s_one_global_kv_head_serves_whole_on_every_rank() {
    printf("[TEST] Gemma 4 12B's geometry: exl3i_dense ops at tp 1, the one global KV head whole on every rank...\n");
    fill_container(0, 12, 40, 1, k12);
    std::vector<const char*> keys(std::begin(k_keys), std::end(k_keys));
    std::vector<const char*> vals(std::begin(k_vals), std::end(k_vals));
    vals[2] = "1";   /* num_global_key_value_heads */
    for (int world : { 1, 2, 4 }) {
        for (int rank : { 0, world - 1 }) {
            RadBuilder b;
            RadModelMeta meta = make_meta(keys, vals, k12);
            RadBuildCtx ctx = build_ctx(rank, world);
            const int rc = gemma4_exl3::declare(&b, &meta, &ctx);
            printf("  tp %d rank %d: declare rc %d\n", world, rank, rc);
            assert(rc == RAD_OK);
            assert(count_op(b, "exl3i_dense") == 70);
            int whole = 0;
            for (const auto& op : b.ops) {
                if (op.op != "exl3i_dense") continue;
                const auto& rec = b.weights[op.w[0] - 1];
                const V3Weight* w = nullptr;
                for (const auto& v : g_v3) if (rec.name == v.name + ".record") w = &v;
                assert(w);
                long long N = 0, K = 0;
                for (const auto& p : op.p) {
                    if (p.key == "N") N = p.ival;
                    if (p.key == "K") K = p.ival;
                }
                assert(N % 128 == 0 && K % 128 == 0);
                if (rec.d.shard == RAD_SHARD_NONE) {
                    assert(N == w->rows && K == w->cols);
                    if (world > 1) {
                        /* only a global layer's k: 2 of the 12 layers */
                        assert(w->name.find(".attn_k.") != std::string::npos && w->rows == k12.hd_g);
                        ++whole;
                    }
                } else if (rec.d.shard == RAD_SHARD_COL) {
                    assert(N == w->rows && K * world == w->cols);
                } else {
                    assert(rec.d.shard == RAD_SHARD_ROW && N * world == w->rows && K == w->cols);
                }
            }
            assert(whole == (world > 1 ? 2 : 0));
        }
    }
    printf("  -> OK (70 exl3i_dense ops at tp 1, 2 and 4; at tp 2 and 4 the two global k weights whole)\n");
}

void test_act_modes_reach_the_op() {
    printf("[TEST] exl3.act.* keys reach the op as act...\n");
    fill_container(0, 12, 40);
    std::vector<const char*> keys(std::begin(k_keys), std::end(k_keys));
    std::vector<const char*> vals(std::begin(k_vals), std::end(k_vals));
    keys.push_back("exl3.act.blk.0.attn_q.weight");   vals.push_back("a8g128");
    keys.push_back("exl3.act.blk.3.ffn_down.weight"); vals.push_back("a8t");
    RadBuilder b;
    RadModelMeta meta = make_meta(keys, vals);
    RadBuildCtx ctx = build_ctx();
    assert(gemma4_exl3::declare(&b, &meta, &ctx) == RAD_OK);
    int a8g = 0, a8t = 0, f16 = 0;
    for (const auto& op : b.ops) {
        if (op.op != "exl3i_dense") continue;
        const std::string act = param_s(op, "act");
        const std::string rec = b.weights[op.w[0] - 1].name;
        if (act == "a8g128") { ++a8g; assert(rec == "blk.0.attn_q.weight.record"); }
        else if (act == "a8t") { ++a8t; assert(rec == "blk.3.ffn_down.weight.record"); }
        else ++f16;
    }
    assert(a8g == 1 && a8t == 1 && f16 == 68);

    /* a value that is not a mode refuses the declaration */
    keys.push_back("exl3.act.blk.1.attn_o.weight"); vals.push_back("a7x");
    RadBuilder b2;
    RadModelMeta meta2 = make_meta(keys, vals);
    assert(gemma4_exl3::declare(&b2, &meta2, &ctx) != RAD_OK);
    printf("  -> OK\n");
}

/* One step over every shape: the declared ops are issued, the lint (operand order, buffer liveness) is clean. */
void run_step(const RadBuilder& b, RadBatch batch, int want_exl3i) {
    RadCtx c; c.b = const_cast<RadBuilder*>(&b); c.rank = 0; c.world = 1; c.batch = &batch;
    gemma4_exl3::step(&c, &batch);
    assert(c.step_fail.empty());
    assert(report_lint(lint_step(b, { &c })) == 0);
    int n = 0;
    for (const std::string& o : issued_ops(b, c)) if (o == "exl3i_dense") ++n;
    assert(n == want_exl3i);
}

void test_step_issues_the_exl3_ops() {
    printf("[TEST] a step issues the exl3i_dense ops in the layers' order...\n");
    fill_container(0, 12, 40);
    RadBuilder b;
    RadModelMeta meta = base_meta();
    RadBuildCtx ctx = build_ctx();
    assert(gemma4_exl3::declare(&b, &meta, &ctx) == RAD_OK);
    run_step(b, prefill_batch(), 70);
    run_step(b, decode_batch(), 70);
    /* the junctions are issued, and nothing of the chains they replace: no add into the residual
     * stream and no layer scalar between blocks */
    for (RadBatch batch : { prefill_batch(), decode_batch() }) {
        RadCtx c; c.b = &b; c.rank = 0; c.world = 1; c.batch = &batch;
        gemma4_exl3::step(&c, &batch);
        int junc = 0, add = 0, scale = 0;
        for (const std::string& o : issued_ops(b, c)) {
            junc  += o == "rmsnorm_quant_i8g";
            add   += o == "add";
            scale += o == "scale";
        }
        assert(junc == 24);
        assert(add == 1 && scale == 1);   /* the last layer closes on its own: one add, one scalar */
    }
    printf("  -> OK (prefill and decode steps, 24 junctions)\n");
}

void test_mixed_container() {
    printf("[TEST] a container with exl3i in layer 0 only: the other layers stay plain bf16...\n");
    fill_container(0, 1, 40);
    RadBuilder b;
    RadModelMeta meta = base_meta();
    RadBuildCtx ctx = build_ctx();
    assert(gemma4_exl3::declare(&b, &meta, &ctx) == RAD_OK);
    assert(count_op(b, "exl3i_dense") == 6);
    assert(count_op(b, "gemm_nt") == 64);
    /* a bf16 linear reads neither a pair nor a junction's row: no junction anywhere, every block
     * keeps its own norm, post-norm and add */
    assert(count_prefix(b, "rmsnorm_quant") == 0);
    run_step(b, prefill_batch(), 6);
    printf("  -> OK (6 exl3i_dense, 64 gemm_nt, no junction)\n");
}

/* Two engines in one process (rad_arch_multi_instance): each declare writes its own slot of g_model
 * (RadBuildCtx::slot_base), and each step reads its own, so a second engine neither clobbers the first's
 * state nor serves it. Engine A holds a full EXL3-V3 container (70 exl3i ops), engine B a layer-0-only one
 * (6): one shared slot would fail the counts. The fakes' rad_slot honours slot_base (arch_fakes.h); at
 * slot_base zero, every other test, the slot stays the rank. */
void test_two_engines_in_one_process() {
    printf("[TEST] two engines in one process keep their own g_model slots...\n");
    fill_container(0, 12, 40);
    RadBuilder b0;
    RadModelMeta meta = base_meta();
    RadBuildCtx ctx0 = build_ctx();                     /* slot 0 */
    assert(gemma4_exl3::declare(&b0, &meta, &ctx0) == RAD_OK);
    assert(count_op(b0, "exl3i_dense") == 70);

    fill_container(0, 1, 40);
    RadBuilder b1;
    RadBuildCtx ctx1 = build_ctx();
    ctx1.slot_base = 1;                                 /* slot 1: the second engine's */
    assert(gemma4_exl3::declare(&b1, &meta, &ctx1) == RAD_OK);
    assert(count_op(b1, "exl3i_dense") == 6);

    for (int round = 0; round < 2; ++round) {           /* twice, in either order */
        RadBatch p0 = prefill_batch(), p1 = prefill_batch();
        RadCtx c0; c0.b = &b0; c0.rank = 0; c0.world = 1; c0.slot_base = 0; c0.batch = &p0;
        RadCtx c1; c1.b = &b1; c1.rank = 0; c1.world = 1; c1.slot_base = 1; c1.batch = &p1;
        gemma4_exl3::step(&c0, &p0);
        gemma4_exl3::step(&c1, &p1);
        assert(c0.step_fail.empty() && c1.step_fail.empty());
        assert(report_lint(lint_step(b0, { &c0 })) == 0);
        assert(report_lint(lint_step(b1, { &c1 })) == 0);
        int n0 = 0, n1 = 0;
        for (const std::string& o : issued_ops(b0, c0)) n0 += o == "exl3i_dense";
        for (const std::string& o : issued_ops(b1, c1)) n1 += o == "exl3i_dense";
        assert(n0 == 70 && n1 == 6);
    }
    printf("  -> OK (70 and 6 exl3i ops a step, from each engine's own slot, twice)\n");
}


/* ---- the token table (G4TokTable) */

/* One parameter's value, integer or float. */
long long param_i(const RecOp& op, const char* key) {
    for (const auto& p : op.p) if (p.key == key) return p.ival;
    return LLONG_MIN;
}
double param_d(const RecOp& op, const char* key) {
    for (const auto& p : op.p) if (p.key == key) return p.dval;
    return -1.0;
}

/* The ops of the trunk's embedding: the gather (embed_tq or embed_exl3i), the scale's fill and mul, a zero fill on a
 * rank past 0, the all-reduce. Found by what they read and write, from the gather's or the zero fill's output. */
struct EmbedOps { const RecOp *gather = nullptr, *sfill = nullptr, *mul = nullptr, *zero = nullptr, *ar = nullptr;
                  rad_op h_gather = 0, h_sfill = 0, h_mul = 0, h_zero = 0, h_ar = 0; int n_gather = 0; };
EmbedOps embed_ops(const RadBuilder& b, int64_t embd, double scale) {
    EmbedOps e;
    rad_buf x = 0;
    for (size_t i = 0; i < b.ops.size(); ++i) {
        const RecOp& o = b.ops[i];
        if (o.op == "embed_tq" || o.op == "embed_exl3i") {
            ++e.n_gather;
            if (!e.gather) { e.gather = &o; e.h_gather = (rad_op)(i + 1); x = o.writes.at(0); }
        }
        if (o.op == "fill" && param_i(o, "n") == embd && param_d(o, "value") == 0.0 && !e.zero &&
            (!x || o.writes.at(0) == x)) {
            e.zero = &o; e.h_zero = (rad_op)(i + 1); x = o.writes.at(0);
        }
    }
    for (size_t i = 0; i < b.ops.size(); ++i) {
        const RecOp& o = b.ops[i];
        if (o.op == "fill" && param_i(o, "n") == embd && param_d(o, "value") == scale && param_i(o, "M") == 1 &&
            !e.sfill) { e.sfill = &o; e.h_sfill = (rad_op)(i + 1); }
        if (o.op == "mul" && e.sfill && o.reads.size() == 2 && o.reads[0] == x && o.reads[1] == e.sfill->writes.at(0) &&
            o.writes.at(0) == x) { e.mul = &o; e.h_mul = (rad_op)(i + 1); }
        if (o.op == "all_reduce" && !o.reads.empty() && o.reads[0] == x && !e.ar) {
            e.ar = &o; e.h_ar = (rad_op)(i + 1);
        }
    }
    return e;
}

/* The position of an op's first issue in a step, or -1. */
int issued_at(const RadCtx& c, rad_op h) {
    for (size_t i = 0; i < c.issues.size(); ++i) if (c.issues[i].op == h) return (int)i;
    return -1;
}

/* A quantised token table (tq at 4 and 5 bits, an exl3i trellis at rate 4) on Gemma 4 12B's geometry. Every rank
 * gathers it whole (n_vocab 262144, every plane unsharded), and each plane is read by that gather alone: the placement
 * finds a table by its readers in each rank's program and stops the start where a rank reads none (tp 2 with no draft
 * head). Rank 0 scales the rows by sqrt(n_embd) rounded to bf16, 62.0 at 3840: a fill of one row and a
 * mul, since neither gather scales. A rank past 0 writes zeros over its rows instead, and the all-reduce stays at two
 * ranks, so the media rows rank 0 alone holds still sum in exactly. No embed_lookup reads the table. A step issues
 * the gather, then the scale or the zeros, before the all-reduce and before the first block. */
void test_a_quantised_token_table_is_gathered_on_every_rank_and_scaled_on_rank_0() {
    printf("[TEST] a quantised token table: every rank gathers, rank 0 scales by bf16(sqrt(n_embd)), others zero...\n");
    assert(gemma4::embed_scale(3840) == 62.0 && gemma4::embed_scale(5376) == 73.5);
    fill_container(0, 12, 64, 1, k12);
    std::vector<const char*> keys(std::begin(k_keys), std::end(k_keys));
    std::vector<const char*> vals(std::begin(k_vals), std::end(k_vals));
    vals[2] = "1";   /* num_global_key_value_heads */
    g_tok_rows = 262144; g_tok_cols = k12.embd; g_tok_stacked = 0;
    for (int form : { 4, 5, -64 }) {
        g_tok = form;
        for (int world : { 1, 2 }) {
            for (int rank = 0; rank < world; ++rank) {
                RadBuilder b;
                RadModelMeta meta = make_meta(keys, vals, k12);
                RadBuildCtx ctx = build_ctx(rank, world);
                assert(gemma4_exl3::declare(&b, &meta, &ctx) == RAD_OK);
                assert(count_op(b, "embed_lookup") == 0);
                const EmbedOps e = embed_ops(b, k12.embd, 62.0);
                assert((world > 1) == (e.ar != nullptr));
                assert(e.n_gather == 1 && e.gather);
                {
                    assert(e.gather->op == (form > 0 ? "embed_tq" : "embed_exl3i"));
                    assert(param_i(*e.gather, "n_vocab") == 262144 && param_i(*e.gather, "n_embd") == k12.embd);
                    assert(e.gather->w.size() == 3);
                    for (rad_weight pw : e.gather->w) {   /* read by the gather alone, on this rank too */
                        int readers = 0;
                        for (const RecOp& o : b.ops) for (rad_weight x : o.w) readers += x == pw;
                        assert(readers == 1);
                    }
                    const auto& p0 = b.weights[e.gather->w[0] - 1].d;
                    const auto& p1 = b.weights[e.gather->w[1] - 1].d;
                    const auto& p2 = b.weights[e.gather->w[2] - 1].d;
                    for (const auto* d : { &p0, &p1, &p2 }) assert(d->shard == RAD_SHARD_NONE);
                    if (form > 0) {
                        assert(param_i(*e.gather, "bits") == form);
                        assert(p0.shape[0] == 262144 && p0.shape[1] == k12.embd && rad_dtype_bits(p0.dtype) == form);
                        assert(p1.shape[0] == 262144 && p1.shape[1] == k12.embd / 128 && p1.dtype == RAD_BF16);
                        assert(p2.shape[0] == 1 && p2.shape[1] == (1 << form) && p2.dtype == RAD_F32);
                    } else {
                        assert(param_i(*e.gather, "m") == 64 && param_s(*e.gather, "codebook") == "mad");
                        assert(p0.shape[0] == 262144 / 16 && p0.shape[1] == exl3i_row_words(k12.embd, 64));
                        assert(p1.shape[0] == 1 && p1.shape[1] == k12.embd && p1.dtype == RAD_F16);
                        assert(p2.shape[0] == 262144 && p2.shape[1] == 1 && p2.dtype == RAD_F16);
                    }
                }
                if (rank == 0) {
                    assert(e.sfill && e.mul && !e.zero);
                    assert(param_d(*e.sfill, "value") == gemma4::embed_scale(k12.embd));
                } else {
                    assert(e.zero && !e.sfill && !e.mul);
                }
                for (RadBatch batch : { prefill_batch(), decode_batch() }) {
                    RadCtx c; c.b = &b; c.rank = rank; c.world = world; c.batch = &batch;
                    gemma4_exl3::step(&c, &batch);
                    assert(c.step_fail.empty());
                    assert(report_lint(lint_step(b, { &c })) == 0);
                    int first_block = -1;
                    for (size_t i = 0; i < c.issues.size() && first_block < 0; ++i) {
                        const RecOp* o = b.op_of(c.issues[i].op);
                        if (o && o->op == "exl3i_dense") first_block = (int)i;
                    }
                    const int last = rank == 0 ? issued_at(c, e.h_mul) : issued_at(c, e.h_zero);
                    const int g = issued_at(c, e.h_gather);
                    assert(g >= 0 && last > g);
                    assert(c.issues[(size_t)g].n == batch.n_tok && c.issues[(size_t)last].n == batch.n_tok);
                    if (rank == 0) assert(issued_at(c, e.h_sfill) > g && last > issued_at(c, e.h_sfill));
                    assert(last >= 0 && first_block > last);
                    if (world > 1) assert(issued_at(c, e.h_ar) > last && issued_at(c, e.h_ar) < first_block);
                }
            }
        }
    }
    g_tok = 0;
    printf("  -> OK (tq 4 and 5 bits and exl3i K4, at tp 1 and 2, every rank, prefill and decode)\n");
}

/* The assistant head embeds through the trunk's table, scaled as the trunk scales it. With a quantised table every
 * rank gathers whole rows into the head's entry, so there is no all-reduce there at two ranks, and the fill and the
 * mul follow the gather. The head's own lm_head is untouched. */
void test_the_assistant_head_embeds_through_a_quantised_table() {
    printf("[TEST] the assistant head gathers and scales a quantised trunk table on every rank, no all-reduce...\n");
    fill_container(0, 12, 64);
    std::vector<const char*> keys(std::begin(k_keys), std::end(k_keys));
    std::vector<const char*> vals(std::begin(k_vals), std::end(k_vals));
    const char* dk[] = { "draft.architectures", "draft.hidden_size", "draft.backbone_hidden_size",
                         "draft.num_hidden_layers", "draft.num_attention_heads", "draft.num_key_value_heads",
                         "draft.num_global_key_value_heads", "draft.head_dim", "draft.global_head_dim",
                         "draft.intermediate_size", "draft.sliding_window", "draft.vocab_size", "draft.rms_norm_eps",
                         "draft.rope_parameters.sliding_attention.rope_theta",
                         "draft.rope_parameters.full_attention.rope_theta",
                         "draft.rope_parameters.full_attention.partial_rotary_factor", "draft.layer_types" };
    const char* dv[] = { "Gemma4AssistantForCausalLM", "1024", "5376", "4", "32", "16", "4", "256", "512", "4096",
                         "1024", "262144", "1e-06", "10000", "1000000", "0.25",
                         "sliding_attention sliding_attention sliding_attention full_attention" };
    for (size_t i = 0; i < sizeof dk / sizeof dk[0]; ++i) { keys.push_back(dk[i]); vals.push_back(dv[i]); }
    g_tok_rows = 262144; g_tok_cols = kEmbd; g_tok_stacked = 0;
    for (int form : { 0, 4, -64 }) {
        g_tok = form;
        for (int world : { 1, 2 }) {
            RadBuilder b;
            RadModelMeta meta = make_meta(keys, vals);
            RadBuildCtx ctx = spec_ctx(0, world, 3);
            const int rc = gemma4_exl3::declare(&b, &meta, &ctx);
            printf("  form %d tp %d: declare rc %d\n", form, world, rc);
            assert(rc == RAD_OK);
            const int lookups = count_op(b, "embed_lookup"), tq = count_op(b, "embed_tq"),
                      x3 = count_op(b, "embed_exl3i");
            /* the trunk's and the head's gather: two of one kind */
            assert(form == 0 ? (lookups == 2 && tq == 0 && x3 == 0)
                             : (lookups == 0 && (form > 0 ? tq : x3) == 2 && (form > 0 ? x3 : tq) == 0));
            /* the head's gather is the one at its rows (max_seqs) */
            const RecOp* hg = nullptr;
            for (const auto& o : b.ops)
                if ((o.op == "embed_lookup" || o.op == "embed_tq" || o.op == "embed_exl3i") && o.p.size() &&
                    o.p[0].key == "M" && o.p[0].ihi == ctx.max_seqs) hg = &o;
            assert(hg);
            const rad_buf e = hg->writes.at(0);
            int ar_on_e = 0, mul_on_e = 0;
            for (const auto& o : b.ops) {
                ar_on_e  += o.op == "all_reduce" && !o.reads.empty() && o.reads[0] == e;
                mul_on_e += o.op == "mul" && !o.reads.empty() && o.reads[0] == e && o.writes.at(0) == e;
            }
            assert(ar_on_e == (form == 0 && world > 1 ? 1 : 0));
            assert(mul_on_e == (form == 0 ? 0 : 1));
            if (form == 0) assert(param_d(*hg, "mult") == 73.5);

            /* a decode step, the head's history pass over its rows, then a drafting round: each pass embeds */
            RadBatch tb = decode_batch(), hb = decode_batch(), rb = decode_batch();
            hb.draft_pass = -1; hb.n_draft_out = hb.n_tok;
            rb.draft_pass = 1;
            RadCtx ct; ct.b = &b; ct.rank = 0; ct.world = world; ct.batch = &tb;
            RadCtx ch; ch.b = &b; ch.rank = 0; ch.world = world; ch.batch = &hb;
            RadCtx cr; cr.b = &b; cr.rank = 0; cr.world = world; cr.batch = &rb;
            gemma4_exl3::step(&ct, &tb);
            gemma4_exl3::step(&ch, &hb);
            gemma4_exl3::step(&cr, &rb);
            assert(ct.step_fail.empty() && ch.step_fail.empty() && cr.step_fail.empty());
            assert(report_lint(lint_step(b, { &ct, &ch, &cr })) == 0);
            for (const RadCtx* c : { &ch, &cr }) {
                int g = -1, m = -1;
                for (size_t i = 0; i < c->issues.size(); ++i) {
                    const RecOp* o = b.op_of(c->issues[i].op);
                    if (o == hg) g = (int)i;
                    if (o && o->op == "mul" && o->writes.at(0) == e) m = (int)i;
                }
                assert(g >= 0 && (form == 0 ? m < 0 : m > g));
            }
        }
    }
    g_tok = 0;
    printf("  -> OK (bf16, tq and exl3i at tp 1 and 2: the head's gather at its rows, scaled, no all-reduce)\n");
}

/* Token tables the gathers do not serve are refused at declare, each by name: an exl3i table that is stacked or not
 * [n_vocab, n_embd], tq codes wider than 8 bits, and a quantised table under a DFlash drafter, which embeds through
 * the bf16 table. The same container with a bf16 table, or without the drafter, declares. */
void test_token_tables_the_gathers_do_not_serve_are_refused() {
    printf("[TEST] token tables the gathers do not serve, and a quantised one under DFlash, are refused...\n");
    fill_container(0, 12, 64);
    auto declare_rc = [](int form, int stacked, int64_t rows, bool dflash) {
        std::vector<const char*> keys(std::begin(k_keys), std::end(k_keys));
        std::vector<const char*> vals(std::begin(k_vals), std::end(k_vals));
        if (dflash) {
            keys.push_back("draft.architectures"); vals.push_back("DFlashDraftModel");
            keys.push_back("draft.target_layer_ids"); vals.push_back("1 5");
        }
        g_tok = form; g_tok_stacked = stacked; g_tok_rows = rows; g_tok_cols = kEmbd;
        RadBuilder b;
        RadModelMeta meta = make_meta(keys, vals);
        RadBuildCtx ctx = spec_ctx(0, 1, dflash ? 4 : 0);
        const int rc = gemma4_exl3::declare(&b, &meta, &ctx);
        g_tok = 0; g_tok_stacked = 0; g_tok_rows = 262144;
        /* the drafter's taps were read (layers 1 and 5): the refusal is the table's, not the tap list's */
        if (dflash) assert(gemma4_exl3::g_model[0].n_taps == 2);
        return rc;
    };
    assert(declare_rc(-64, 1, 262144, false) == RAD_E_FORMAT);   /* stacked */
    assert(declare_rc(-64, 0, 131072, false) == RAD_E_FORMAT);   /* not n_vocab rows */
    assert(declare_rc(9, 0, 262144, false) == RAD_E_DTYPE);      /* u9 codes */
    assert(declare_rc(4, 0, 262144, true) == RAD_E_UNSUPPORTED);
    assert(declare_rc(-64, 0, 262144, true) == RAD_E_UNSUPPORTED);
    assert(declare_rc(4, 0, 262144, false) == RAD_OK);
    printf("  -> OK (four refusals by name; the served form declares)\n");
}

/* Gemma 4 E4B's text geometry, all 42 layers: global layers at (l + 1) % 6 == 0, two KV heads in both classes
 * (no num_global_key_value_heads, so the global layers take num_key_value_heads), a global layer with its own v
 * (attention_k_eq_v 0), the last 18 layers KV-shared, per-layer embeddings 256 wide. */
const char* k_e4b_keys[] = {
    "sliding_window", "global_head_dim", "num_key_value_heads",
    "rope_parameters.sliding_attention.rope_theta", "rope_parameters.full_attention.rope_theta",
    "rope_parameters.full_attention.partial_rotary_factor", "num_kv_shared_layers",
    "final_logit_softcapping", "full_attention_interval", "attention_k_eq_v", "hidden_size_per_layer_input",
};
const char* k_e4b_vals[] = { "512", "512", "2", "10000", "1000000", "0.25", "18", "30", "6", "0", "256" };
constexpr Geo kE4b{ 2560, 10240, 8, 2, 256, 512 };
constexpr int kE4bLayers = 42, kE4bOwn = 24;   /* layers 24..41 share KV */

RadModelMeta e4b_meta() {
    static std::vector<const char*> keys(std::begin(k_e4b_keys), std::end(k_e4b_keys));
    static std::vector<const char*> vals(std::begin(k_e4b_vals), std::end(k_e4b_vals));
    RadModelMeta m = make_meta(keys, vals, kE4b);
    m.n_layers = kE4bLayers;
    return m;
}

/* E4B's container: a layer with a cache of its own holds q, k, v (a global layer too), o, gate_up and down; a
 * KV-shared layer holds q, o, gate_up and down (the checkpoint's k and v of these layers are unused). The
 * per-layer branch's linears are not in it: they stay bf16. */
void fill_e4b(int m) {
    g_v3.clear();
    for (int l = 0; l < kE4bLayers; ++l) {
        const bool global = (l + 1) % 6 == 0;
        const int64_t qd = kE4b.heads * (global ? kE4b.hd_g : kE4b.hd_s);
        const int64_t kvd = kE4b.kv_s * (global ? kE4b.hd_g : kE4b.hd_s);
        auto add = [&](const char* n, int stacked, int64_t rows, int64_t cols) {
            g_v3.push_back({ "blk." + std::to_string(l) + "." + n + ".weight", m, EXL3I_CB_MAD, stacked, rows, cols });
        };
        add("attn_q", 0, qd, kE4b.embd);
        if (l < kE4bOwn) {
            add("attn_k", 0, kvd, kE4b.embd);
            add("attn_v", 0, kvd, kE4b.embd);
        }
        add("attn_o", 0, kE4b.embd, qd);
        add("ffn_gate_up", 1, 2 * kE4b.ff, kE4b.embd);
        add("ffn_down", 0, kE4b.embd, kE4b.ff);
    }
}

/* Gemma 4 E4B at one rank: every container weight is an exl3i_dense op, a KV-shared layer declares no k or v
 * and binds no cache, and its attention reads the last earlier layer of its class that keeps one (sliding 22,
 * global 23). The per-layer branch (each layer's gate and projection, the model projection) stays bf16 gemm_nt.
 * A prefill and a decode step issue every exl3i op and lint clean. */
void test_the_e4b_s_shared_layers_hold_no_k_or_v_and_its_per_layer_branch_stays_bf16() {
    printf("[TEST] Gemma 4 E4B at tp 1: exl3i_dense ops, no k or v on the 18 KV-shared layers, the per-layer branch bf16...\n");
    fill_e4b(40);
    /* 24 layers x 6 (q, k, v, o, gate_up, down) + 18 KV-shared x 4 (q, o, gate_up, down) */
    const int want = kE4bOwn * 6 + (kE4bLayers - kE4bOwn) * 4;
    assert((int)g_v3.size() == want);
    RadBuilder b;
    RadModelMeta meta = e4b_meta();
    RadBuildCtx ctx = build_ctx();
    assert(gemma4_exl3::declare(&b, &meta, &ctx) == RAD_OK);
    assert(count_op(b, "exl3i_dense") == want);
    /* the per-layer branch: a gate and a projection a layer, and the model projection of the embedding */
    assert(count_op(b, "gemm_nt") == 2 * kE4bLayers + 1);
    assert(count_op(b, "act_mul") == kE4bLayers);
    assert(count_op(b, "embed_lookup") == 2);   /* the token table and the per-layer table */
    assert((int)b.binds.size() == kE4bOwn);
    for (const auto& bd : b.binds) assert((int)bd.first < kE4bOwn);
    for (const auto& w : b.weights)
        for (int l = kE4bOwn; l < kE4bLayers; ++l)
            for (const char* kv : { ".attn_k.", ".attn_v.", ".attn_k_norm." })
                assert(w.name.rfind("blk." + std::to_string(l) + kv, 0) != 0);
    for (const auto& op : b.ops) {
        if (op.op != "exl3i_dense") continue;
        const auto& rec = b.weights[op.w[0] - 1];
        const V3Weight* w = nullptr;
        for (const auto& v : g_v3) if (rec.name == v.name + ".record") w = &v;
        assert(w);
        long long N = 0, K = 0;
        for (const auto& p : op.p) { if (p.key == "N") N = p.ival; if (p.key == "K") K = p.ival; }
        assert(N == w->rows && K == w->cols && N % 128 == 0 && K % 128 == 0);
    }
    run_step(b, prefill_batch(), want);
    run_step(b, decode_batch(), want);

    RadBatch batch = decode_batch();
    RadCtx c; c.b = &b; c.rank = 0; c.world = 1; c.batch = &batch;
    gemma4_exl3::step(&c, &batch);
    std::vector<int64_t> src;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o || o->op != "attn_paged") continue;
        assert(i.opd.size() > 1 && i.opd[1].kind == RAD_OPK_KV);
        src.push_back(i.opd[1].offset);
    }
    assert((int)src.size() == kE4bLayers);
    for (int l = 0; l < kE4bLayers; ++l)
        assert(src[(size_t)l] == (l < kE4bOwn ? l : (l + 1) % 6 == 0 ? 23 : 22));
    printf("  -> OK (%d exl3i_dense ops, %d gemm_nt in the per-layer branch, %d caches, shared layers read 22 / 23)\n",
           want, 2 * kE4bLayers + 1, kE4bOwn);
}

/* Gemma 4 E4B above one rank: at two ranks each rank holds one KV head of each class, an even share, and every
 * exl3i weight declares this rank's share. At four ranks each KV head is held by two of the four ranks: a k or v
 * weight's rows are then neither the whole weight nor an even share, which an exl3i record cannot be cut to, so
 * the declaration is refused (as 31B's two global KV heads at four ranks). */
void test_the_e4b_declares_at_two_ranks_and_is_refused_at_four() {
    printf("[TEST] Gemma 4 E4B: tp 2 declares each rank's share, tp 4 (KV heads held by some ranks) is refused...\n");
    fill_e4b(40);
    const int want = kE4bOwn * 6 + (kE4bLayers - kE4bOwn) * 4;
    for (int rank : { 0, 1 }) {
        RadBuilder b;
        RadModelMeta meta = e4b_meta();
        RadBuildCtx ctx = build_ctx(rank, 2);
        assert(gemma4_exl3::declare(&b, &meta, &ctx) == RAD_OK);
        assert(count_op(b, "exl3i_dense") == want);
        int n_row = 0, n_col = 0;
        for (const auto& op : b.ops) {
            if (op.op != "exl3i_dense") continue;
            const auto& rec = b.weights[op.w[0] - 1];
            const V3Weight* w = nullptr;
            for (const auto& v : g_v3) if (rec.name == v.name + ".record") w = &v;
            assert(w);
            long long N = 0, K = 0;
            for (const auto& p : op.p) { if (p.key == "N") N = p.ival; if (p.key == "K") K = p.ival; }
            if (rec.d.shard == RAD_SHARD_COL) { assert(N == w->rows && K * 2 == w->cols); ++n_col; }
            else { assert(rec.d.shard == RAD_SHARD_ROW && N * 2 == w->rows && K == w->cols); ++n_row; }
        }
        /* o and down by columns (2 a layer); q, gate_up and, on the 24 layers with a cache, k and v by rows */
        assert(n_col == 2 * kE4bLayers && n_row == want - 2 * kE4bLayers);
    }
    RadBuilder b;
    RadModelMeta meta = e4b_meta();
    RadBuildCtx ctx = build_ctx(0, 4);
    const int rc = gemma4_exl3::declare(&b, &meta, &ctx);
    printf("  tp 4: declare rc %d\n", rc);
    assert(rc != RAD_OK);
    printf("  -> OK (tp 2 ranks 0 and 1: %d exl3i_dense ops, each a share; tp 4 refused)\n", want);
}

/* E4B's per-layer table (262144 x 42 * 256 = 10752 wide), quantised as the token table may be (tq at 4, 5 and 8 bits,
 * an exl3i trellis at rate 4). Every rank gathers whole rows (every plane unsharded and read by the gather alone, so
 * the placement finds the table on each rank), the rows are scaled by sqrt(256) = 16 with a fill and a mul, and no
 * all-reduce reads them: each rank's rows are whole already. The bf16 token table keeps its embed_lookup. A prefill
 * and a decode step issue gather, fill and mul before the first per-layer read, at tp 1 and 2, and lint clean. */
void test_a_quantised_per_layer_table_is_gathered_whole_on_every_rank_and_scaled_by_sqrt_n_ple() {
    printf("[TEST] E4B's per-layer table quantised: every rank gathers whole rows, scaled by 16, no all-reduce...\n");
    fill_e4b(40);
    const int64_t P = (int64_t)kE4bLayers * 256;
    g_ple_rows = 262144; g_ple_cols = P;
    for (int form : { 4, 5, 8, -64 }) {
        g_ple = form;
        for (int world : { 1, 2 }) {
            for (int rank = 0; rank < world; ++rank) {
                RadBuilder b;
                RadModelMeta meta = e4b_meta();
                RadBuildCtx ctx = build_ctx(rank, world);
                assert(gemma4_exl3::declare(&b, &meta, &ctx) == RAD_OK);
                assert(count_op(b, "embed_lookup") == 1);   /* the bf16 token table only */
                const RecOp* gather = nullptr;
                rad_op hg = 0, hf = 0, hm = 0;
                for (size_t i = 0; i < b.ops.size(); ++i) {
                    const RecOp& o = b.ops[i];
                    if ((o.op == "embed_tq" || o.op == "embed_exl3i") && param_i(o, "n_embd") == P) {
                        assert(!gather); gather = &o; hg = (rad_op)(i + 1);
                    }
                }
                assert(gather && gather->op == (form > 0 ? "embed_tq" : "embed_exl3i"));
                assert(param_i(*gather, "n_vocab") == 262144);
                if (form > 0) assert(param_i(*gather, "bits") == form);
                const rad_buf x = gather->writes.at(0);
                for (rad_weight pw : gather->w) {
                    assert(b.weights[pw - 1].d.shard == RAD_SHARD_NONE);
                    int readers = 0;
                    for (const RecOp& o : b.ops) for (rad_weight y : o.w) readers += y == pw;
                    assert(readers == 1);
                }
                if (form > 0) {
                    const auto& c0 = b.weights[gather->w[0] - 1].d;
                    assert(c0.shape[0] == 262144 && c0.shape[1] == P && rad_dtype_bits(c0.dtype) == form);
                }
                for (size_t i = 0; i < b.ops.size(); ++i) {
                    const RecOp& o = b.ops[i];
                    if (o.op == "fill" && param_i(o, "n") == P && param_d(o, "value") == 16.0) hf = (rad_op)(i + 1);
                    if (o.op == "mul" && o.reads.size() == 2 && o.reads[0] == x && o.writes.at(0) == x) hm = (rad_op)(i + 1);
                    assert(!(o.op == "all_reduce" && !o.reads.empty() && o.reads[0] == x));
                }
                assert(hf && hm);
                for (RadBatch batch : { prefill_batch(), decode_batch() }) {
                    RadCtx c; c.b = &b; c.rank = rank; c.world = world; c.batch = &batch;
                    gemma4_exl3::step(&c, &batch);
                    assert(c.step_fail.empty());
                    assert(report_lint(lint_step(b, { &c })) == 0);
                    const int g = issued_at(c, hg), f = issued_at(c, hf), mu = issued_at(c, hm);
                    assert(g >= 0 && f > g && mu > f);
                    assert(c.issues[(size_t)g].n == batch.n_tok && c.issues[(size_t)mu].n == batch.n_tok);
                    int first_read = -1;   /* the first op after the mul that reads the rows */
                    for (size_t i = (size_t)mu + 1; i < c.issues.size() && first_read < 0; ++i) {
                        const RecOp* o = b.op_of(c.issues[i].op);
                        if (o) for (rad_buf r : o->reads) if (r == x) first_read = (int)i;
                    }
                    assert(first_read > mu);
                }
            }
        }
    }
    g_ple = 0;
    printf("  -> OK (tq 4, 5 and 8 bits and exl3i K4, at tp 1 and 2, every rank, prefill and decode)\n");
}

/* The per-layer table's refusals, by name: an exl3i one that is not [n_vocab, n_layer * n_ple], and tq codes wider
 * than 8 bits. */
void test_per_layer_tables_the_gathers_do_not_serve_are_refused() {
    printf("[TEST] per-layer tables the gathers do not serve are refused...\n");
    fill_e4b(40);
    auto rc = [](int form, int64_t rows, int64_t cols) {
        g_ple = form; g_ple_rows = rows; g_ple_cols = cols;
        RadBuilder b;
        RadModelMeta meta = e4b_meta();
        RadBuildCtx ctx = build_ctx();
        const int r = gemma4_exl3::declare(&b, &meta, &ctx);
        g_ple = 0; g_ple_rows = 262144; g_ple_cols = (int64_t)kE4bLayers * 256;
        return r;
    };
    const int64_t P = (int64_t)kE4bLayers * 256;
    assert(rc(-64, 131072, P) == RAD_E_FORMAT);       /* not n_vocab rows */
    assert(rc(-64, 262144, P - 256) == RAD_E_FORMAT); /* not n_layer x n_ple wide */
    assert(rc(9, 262144, P) == RAD_E_DTYPE);          /* u9 codes */
    assert(rc(4, 262144, P) == RAD_OK);
    printf("  -> OK (three refusals by name; the served form declares)\n");
}

/* ---- the DFlash drafters beside EXL3-V3 trunk linears */

/* The drafter's linears and the table's head copy on the int8 grid (w8a8), as a recipe makes them for radiance 2. */
void dflash_encs(RadBuilder& b) {
    const RadEncoding q8 = rad_enc_affine(RAD_I8, RAD_F16, 1, 128);
    b.encs = { { "dflash.fc.", q8 }, { "dflash.blk.", q8 }, { "dflash.draft_head.", q8 } };
}

int buf_named(const RadBuilder& b, const char* name) {
    for (size_t i = 0; i < b.bufs.size(); ++i) if (b.bufs[i].name == name) return (int)i + 1;
    return 0;
}

int writes_to(const RadBuilder& b, const RadCtx& c, int buf) {
    int n = 0;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o) for (rad_buf w : o->writes) n += (int)w == buf;
    }
    return n;
}

/* One decode step, then the drafter's context pass and its query pass, linted together (the context pass reads the
 * taps the step wrote). Neither pass issues a trunk linear. Returns the step's writes to the tap buffer. */
int dflash_step_and_passes(RadBuilder& b, int rank, int world) {
    const int src = buf_named(b, "dflash_src");
    assert(src > 0);
    RadBatch bd = decode_batch(), bc = decode_batch(), bq = decode_batch();
    bc.draft_pass = -1; bq.draft_pass = 1;
    RadCtx cd, cc, cq;
    for (auto [c, bt] : { std::pair<RadCtx*, RadBatch*>{ &cd, &bd }, { &cc, &bc }, { &cq, &bq } }) {
        c->b = &b; c->rank = rank; c->world = world; c->batch = bt;
        gemma4_exl3::step(c, bt);
        assert(c->step_fail.empty());
    }
    assert(report_lint(lint_step(b, { &cd, &cc, &cq })) == 0);
    for (const RadCtx* c : { &cc, &cq }) {
        const std::vector<std::string> ops = issued_ops(b, *c);
        assert(!ops.empty() && count_issued(ops, "exl3i_dense") == 0);
    }
    assert(issued_ops(b, cq).front() == "embed_lookup");
    return writes_to(b, cd, src);
}

/* A DFlash drafter (z-lab/gemma-4-31B-it-DFlash's shape: 5376 wide, 64 heads / 8 kv x 128, ffn 10752, five layers at
 * block 16) merged under `draft.` on the twelve-layer 31B container, reading the trunk at layers 1, 5 and 9. It
 * declares as gemma4_bf16 declares it, a block drafter past the trunk with its own cache, and a decode step writes each
 * tap once: the junctions leave the tapped rows whole. The context pass and the query pass run the drafter alone, and
 * the decode step and both passes lint clean. */
void test_a_dflash_drafter_declares_beside_exl3_trunk_linears_and_reads_whole_rows() {
    printf("[TEST] a DFlash drafter beside EXL3-V3 trunk linears: declared, three taps written, its passes clean...\n");
    fill_container(0, 12, 64);
    std::vector<const char*> keys(std::begin(k_keys), std::end(k_keys));
    std::vector<const char*> vals(std::begin(k_vals), std::end(k_vals));
    const char* dk[] = { "draft.architectures", "draft.num_hidden_layers", "draft.head_dim", "draft.num_attention_heads",
                         "draft.num_key_value_heads", "draft.intermediate_size", "draft.rms_norm_eps", "draft.rope_theta",
                         "draft.hidden_size", "draft.vocab_size", "draft.block_size", "draft.dflash_config.target_layer_ids",
                         "draft.dflash_config.mask_token_id", "draft.layer_types", "draft.sliding_window" };
    const char* dv[] = { "[\"DFlashDraftModel\"]", "5", "128", "64", "8", "10752", "1e-06", "1000000", "5376", "262144", "16",
                         "[1, 5, 9]", "4",
                         "[\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", \"sliding_attention\", "
                         "\"full_attention\"]", "2048" };
    for (size_t i = 0; i < sizeof dk / sizeof dk[0]; ++i) { keys.push_back(dk[i]); vals.push_back(dv[i]); }
    RadBuilder b;
    dflash_encs(b);
    RadModelMeta meta = make_meta(keys, vals);
    RadBuildCtx ctx = spec_ctx(0, 1, 15);
    assert(gemma4_exl3::declare(&b, &meta, &ctx) == RAD_OK);
    assert(b.drafter_name == "dflash" && (int)b.drafter.kind == (int)RAD_DRAFT_BLOCK);
    assert(b.drafter.depth == 15 && (int)b.drafter.block == 16 && b.drafter.mask_token == 4);
    assert(gemma4_exl3::g_model[0].n_taps == 3);
    assert(count_op(b, "exl3i_dense") == (int)g_v3.size());
    assert(dflash_step_and_passes(b, 0, 1) == 3);
    printf("  -> OK (block 16, depth 15, three taps written once a decode step, both passes clean)\n");
}

/* Nemotron-3.5-Bolt-E4B's DFlash drafter on Gemma 4 E4B's geometry: 2560 wide, 32 heads / 8 kv x 128, ffn 5120, five
 * sliding layers over a 1024 window at block 8, reading the trunk at layers 5, 11, 17, 23 and 35 (the last a KV-shared
 * layer, whose row is written all the same). It declares at tp 1 and at both ranks of tp 2 beside the per-layer branch,
 * every rank writes the five taps a decode step, and the step and both passes lint clean. */
void test_bolt_e4b_s_dflash_drafter_declares_at_one_and_two_ranks_beside_the_per_layer_branch() {
    printf("[TEST] Bolt-E4B's DFlash drafter on E4B: declared at tp 1 and 2, five taps a step, its passes clean...\n");
    fill_e4b(40);
    static std::vector<const char*> keys(std::begin(k_e4b_keys), std::end(k_e4b_keys));
    static std::vector<const char*> vals(std::begin(k_e4b_vals), std::end(k_e4b_vals));
    if (keys.size() == sizeof k_e4b_keys / sizeof k_e4b_keys[0]) {
        const char* dk[] = { "draft.architectures", "draft.num_hidden_layers", "draft.head_dim", "draft.num_attention_heads",
                             "draft.num_key_value_heads", "draft.intermediate_size", "draft.rms_norm_eps",
                             "draft.rope_theta", "draft.hidden_size", "draft.vocab_size", "draft.block_size",
                             "draft.dflash_config.target_layer_ids", "draft.dflash_config.mask_token_id",
                             "draft.dflash_config.use_swa", "draft.dflash_config.swa_window_size",
                             "draft.dflash_config.causal", "draft.layer_types", "draft.sliding_window" };
        const char* dv[] = { "[\"DFlashDraftModel\"]", "5", "128", "32", "8", "5120", "1e-06", "1000000.0", "2560", "262144",
                             "8", "[5, 11, 17, 23, 35]", "4", "1", "1024", "0",
                             "[\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", \"sliding_attention\", "
                             "\"sliding_attention\"]", "1024" };
        for (size_t i = 0; i < sizeof dk / sizeof dk[0]; ++i) { keys.push_back(dk[i]); vals.push_back(dv[i]); }
    }
    for (int world : { 1, 2 }) {
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            dflash_encs(b);
            RadModelMeta meta = make_meta(keys, vals, kE4b);
            meta.n_layers = kE4bLayers;
            RadBuildCtx ctx = spec_ctx(rank, world, 7);
            assert(gemma4_exl3::declare(&b, &meta, &ctx) == RAD_OK);
            assert(b.drafter_name == "dflash" && (int)b.drafter.block == 8 && b.drafter.depth == 7);
            assert(gemma4_exl3::g_model[0].n_taps == 5);
            assert(dflash_step_and_passes(b, rank, world) == 5);
        }
    }
    printf("  -> OK (block 8, five taps incl. the KV-shared layer 35, tp 1 and both ranks of tp 2)\n");
}

/* ---- Gemma 4 E2B */

/* Gemma 4 E2B's text geometry as google/gemma-4-E2B-it's config.json states it (revision 4adf1574): 35 layers, hidden
 * 1536, 8 q heads over 1 KV head in both classes (sliding 256, global 512, a global layer with its own v), a global
 * layer every fifth, the last 20 layers KV-shared, per-layer embeddings 256 wide, and use_double_wide_mlp: ffn 6144,
 * 12288 on the KV-shared layers. */
const char* k_e2b_keys[] = {
    "sliding_window", "global_head_dim", "num_key_value_heads",
    "rope_parameters.sliding_attention.rope_theta", "rope_parameters.full_attention.rope_theta",
    "rope_parameters.full_attention.partial_rotary_factor", "num_kv_shared_layers",
    "final_logit_softcapping", "full_attention_interval", "attention_k_eq_v", "hidden_size_per_layer_input",
    "use_double_wide_mlp",
};
const char* k_e2b_vals[] = { "512", "512", "1", "10000", "1000000", "0.25", "20", "30", "5", "0", "256", "true" };
constexpr Geo kE2b{ 1536, 6144, 8, 1, 256, 512 };
constexpr int kE2bLayers = 35, kE2bOwn = 15;   /* layers 15..34 share KV and have the double-wide MLP */

RadModelMeta e2b_meta() {
    static std::vector<const char*> keys(std::begin(k_e2b_keys), std::end(k_e2b_keys));
    static std::vector<const char*> vals(std::begin(k_e2b_vals), std::end(k_e2b_vals));
    RadModelMeta m = make_meta(keys, vals, kE2b);
    m.n_layers = kE2bLayers;
    return m;
}

/* E2B's container: a layer with a cache of its own holds q, k, v (a global layer too), o, gate_up and down; a KV-shared
 * layer holds q, o, gate_up and down, the last two twice as wide. */
void fill_e2b(int m) {
    g_v3.clear();
    for (int l = 0; l < kE2bLayers; ++l) {
        const bool global = (l + 1) % 5 == 0;
        const int64_t qd = kE2b.heads * (global ? kE2b.hd_g : kE2b.hd_s);
        const int64_t kvd = kE2b.kv_s * (global ? kE2b.hd_g : kE2b.hd_s);
        const int64_t ff = l < kE2bOwn ? kE2b.ff : 2 * kE2b.ff;
        auto add = [&](const char* n, int stacked, int64_t rows, int64_t cols) {
            g_v3.push_back({ "blk." + std::to_string(l) + "." + n + ".weight", m, EXL3I_CB_MAD, stacked, rows, cols });
        };
        add("attn_q", 0, qd, kE2b.embd);
        if (l < kE2bOwn) {
            add("attn_k", 0, kvd, kE2b.embd);
            add("attn_v", 0, kvd, kE2b.embd);
        }
        add("attn_o", 0, kE2b.embd, qd);
        add("ffn_gate_up", 1, 2 * ff, kE2b.embd);
        add("ffn_down", 0, kE2b.embd, ff);
    }
}

/* Gemma 4 E2B at tp 1, 2, 4 and 8: every container weight is an exl3i_dense op whose N and K are multiples of 128 on
 * every rank (each width of E2B is: 1536, 2048, 4096, 256, 512, 6144, 12288, and their shares). The one KV head is held
 * whole on every rank (unsharded); q, gate_up by rows and o, down by columns. A KV-shared layer declares no k or v and
 * the 15 others bind a cache each; the KV-shared layers' MLP is double-wide. A prefill and a decode step at tp 1, and a
 * decode step on each rank at tp 2, lint clean. tp 3 is refused: 8 heads do not split into three. */
void test_e2b_declares_at_one_two_four_and_eight_ranks_and_is_refused_at_three() {
    printf("[TEST] Gemma 4 E2B: tp 1, 2, 4 and 8 declare each rank's share, 128-multiples, tp 3 refused...\n");
    fill_e2b(40);
    const int want = kE2bOwn * 6 + (kE2bLayers - kE2bOwn) * 4;
    assert((int)g_v3.size() == want);
    for (int world : { 1, 2, 4, 8 }) {
        for (int rank : { 0, world - 1 }) {
            RadBuilder b;
            RadModelMeta meta = e2b_meta();
            RadBuildCtx ctx = build_ctx(rank, world);
            assert(gemma4_exl3::declare(&b, &meta, &ctx) == RAD_OK);
            assert(count_op(b, "exl3i_dense") == want);
            assert((int)b.binds.size() == kE2bOwn);
            int n_none = 0, n_wide = 0;
            for (const auto& op : b.ops) {
                if (op.op != "exl3i_dense") continue;
                const auto& rec = b.weights[op.w[0] - 1];
                const V3Weight* w = nullptr;
                for (const auto& v : g_v3) if (rec.name == v.name + ".record") w = &v;
                assert(w);
                long long N = 0, K = 0;
                for (const auto& p : op.p) { if (p.key == "N") N = p.ival; if (p.key == "K") K = p.ival; }
                assert(N % 128 == 0 && K % 128 == 0);
                if (w->name.find(".ffn_gate_up.") != std::string::npos && w->rows == 4 * kE2b.ff) ++n_wide;
                if (world == 1 || rec.d.shard == RAD_SHARD_NONE) {
                    assert(N == w->rows && K == w->cols);
                    if (world > 1) ++n_none;
                } else if (rec.d.shard == RAD_SHARD_COL) {
                    assert(N == w->rows && K * world == w->cols);
                } else {
                    assert(rec.d.shard == RAD_SHARD_ROW && N * world == w->rows && K == w->cols);
                }
            }
            assert(n_wide == kE2bLayers - kE2bOwn);
            if (world > 1) assert(n_none == 2 * kE2bOwn);   /* k and v, whole on every rank */
            for (const auto& wt : b.weights)
                for (int l = kE2bOwn; l < kE2bLayers; ++l)
                    for (const char* kv : { ".attn_k.", ".attn_v.", ".attn_k_norm." })
                        assert(wt.name.rfind("blk." + std::to_string(l) + kv, 0) != 0);
            if (world == 1) {
                run_step(b, prefill_batch(), want);
                run_step(b, decode_batch(), want);
            } else if (world == 2) {
                RadBatch batch = decode_batch();
                RadCtx c; c.b = &b; c.rank = rank; c.world = world; c.batch = &batch;
                gemma4_exl3::step(&c, &batch);
                assert(c.step_fail.empty());
                assert(report_lint(lint_step(b, { &c })) == 0);
            }
        }
    }
    RadBuilder b;
    RadModelMeta meta = e2b_meta();
    RadBuildCtx ctx = build_ctx(0, 3);
    const int rc = gemma4_exl3::declare(&b, &meta, &ctx);
    printf("  tp 3: declare rc %d\n", rc);
    assert(rc == RAD_E_INVAL);   /* "n_head = 8 is not divisible by world_size = 3" */
    printf("  -> OK (%d exl3i_dense ops at tp 1, 2, 4 and 8, first and last rank; k and v whole; tp 3 refused)\n", want);
}

} /* namespace */

int main() {
    test_container_declares_exl3i_ops();
    test_tp_two_and_four_declare_each_rank_s_share();
    test_a_kv_head_shared_by_some_ranks_is_refused();
    test_the_12b_s_one_global_kv_head_serves_whole_on_every_rank();
    test_the_e4b_s_shared_layers_hold_no_k_or_v_and_its_per_layer_branch_stays_bf16();
    test_the_e4b_declares_at_two_ranks_and_is_refused_at_four();
    test_overlap_slices_write_the_bf16_row();
    test_act_modes_reach_the_op();
    test_step_issues_the_exl3_ops();
    test_mixed_container();
    test_two_engines_in_one_process();
    test_a_quantised_token_table_is_gathered_on_every_rank_and_scaled_on_rank_0();
    test_the_assistant_head_embeds_through_a_quantised_table();
    test_token_tables_the_gathers_do_not_serve_are_refused();
    test_a_quantised_per_layer_table_is_gathered_whole_on_every_rank_and_scaled_by_sqrt_n_ple();
    test_per_layer_tables_the_gathers_do_not_serve_are_refused();
    test_a_dflash_drafter_declares_beside_exl3_trunk_linears_and_reads_whole_rows();
    test_bolt_e4b_s_dflash_drafter_declares_at_one_and_two_ranks_beside_the_per_layer_branch();
    test_e2b_declares_at_one_two_four_and_eight_ranks_and_is_refused_at_three();
    printf("ALL TESTS PASSED.\n");
    return 0;
}
