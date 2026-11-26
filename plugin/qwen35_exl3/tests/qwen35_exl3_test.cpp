/* qwen35_exl3_test.cpp -- Integration test proving the exl3i linears, Radiance block code structure (GDN,
 * AttnGated, MLP), and step execution: the main suite at world 1, plus test_exl3i's TP2 arm (the exl3i planes at
 * the rank's extents) and test_tp -- the whole container at TP2 / TP4 (the trunk sharded, ssm_ab / the R16 head /
 * the trellis table whole).
 */

#define RAD_ARCH_NO_EXPORTS 1
#undef NDEBUG   /* the checks below are assert()s: keep them in a Release build (CMAKE_BUILD_TYPE=Release sets NDEBUG) */

#include "arch_fakes.h"
#include "exl3i_encoding.h"

/* The fakes' rad_weight_encoding answers an encoding by name but no checkpoint shape, and declare reads the GDN's
 * plain a/b extents from it (48 rows of n_embd at 27B): the plugin's calls go through this wrapper, which answers
 * V3_WEIGHTS' exact names as exl3i, a/b as plain bf16 [48, 5120] (a container of plain a / b, as every container
 * before the ssm_ab form) and everything else as the fakes do. The GDN's fused a | b (ssm_ab) is absent unless
 * listed. r16: the rows are stored in R16 order (exl3i_encoding_make_r16; the head). */
struct V3Weight { const char* name; int m, cb, stacked; int64_t rows, cols; int r16 = 0; };
static const V3Weight* V3_WEIGHTS = nullptr;
static int N_V3 = 0;
static int test_weight_encoding(RadBuilder* b, const char* source, RadEncoding* out, int64_t* shape, uint32_t* rank) {
    for (int i = 0; source && i < N_V3; ++i)
        if (!std::strcmp(source, V3_WEIGHTS[i].name)) {
            if (shape) { shape[0] = V3_WEIGHTS[i].rows; shape[1] = V3_WEIGHTS[i].cols; }
            if (rank) *rank = 2;
            return V3_WEIGHTS[i].r16 ? exl3i_encoding_make_r16(V3_WEIGHTS[i].m, V3_WEIGHTS[i].cb, V3_WEIGHTS[i].stacked,
                                                               V3_WEIGHTS[i].rows, V3_WEIGHTS[i].cols, out)
                                     : exl3i_encoding_make(V3_WEIGHTS[i].m, V3_WEIGHTS[i].cb, V3_WEIGHTS[i].stacked,
                                                            V3_WEIGHTS[i].rows, V3_WEIGHTS[i].cols, out);
        }
    if (source && std::strstr(source, ".ssm_ab.")) return RAD_E_NOTFOUND;
    if (source && (std::strstr(source, ".ssm_a.") || std::strstr(source, ".ssm_b."))) {
        *out = rad_enc_plain(RAD_BF16);
        if (shape) { shape[0] = 48; shape[1] = 5120; }
        if (rank) *rank = 2;
        return RAD_OK;
    }
    return rad_weight_encoding(b, source, out, shape, rank);
}
#define rad_weight_encoding test_weight_encoding
#include "../qwen35_exl3.cpp"
#undef rad_weight_encoding

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

namespace {

RadModelMeta make_qwen35_meta(int64_t n_layers = 64) {
    RadModelMeta m{};
    m.arch_id = "qwen35";
    m.name = "test-qwen35-27b";
    m.quant = "exl3";
    m.n_layers = n_layers;
    m.n_embd = 5120;
    m.n_ff = 17408;
    m.n_head = 24;
    m.n_head_kv = 4;
    m.head_dim = 256;
    m.n_vocab = 248320;
    m.rms_eps = 1e-6f;
    m.rope_theta = 1e7f;
    m.rope_scale = 1.0f;
    return m;
}

RadBuildCtx make_ctx(int rank = 0, int world = 1, int64_t max_tok = 128, int64_t max_ctx = 262144) {
    RadBuildCtx c{};
    c.rank = rank;
    c.world_size = world;
    c.max_tok = max_tok;
    c.max_seqs = 1;
    c.max_ctx = max_ctx;
    return c;
}

/* Every projection weight of the model as an exl3i trellis (K4 = m 64, mad), the whole weight's extents (the record
 * plane goes whole to every slice: a tensor-parallel split of an exl3i weight is refused, test_exl3i). The GDN's 16
 * key / 48 value heads of 128 dims and the attention interval 4 are the meta's absent-key defaults. V3_WEIGHTS points
 * at the result for the wrapper; the names outlive the table. */
static std::vector<std::string> V3_ALL_NAMES;
static std::vector<V3Weight> V3_ALL;
static void reg_all_exl3i(const RadModelMeta& m, bool with_head = true) {
    V3_ALL.clear(); V3_ALL_NAMES.clear();
    V3_ALL_NAMES.reserve((size_t)m.n_layers * 7 + 2);
    const int64_t ne = m.n_embd, nf = m.n_ff;
    const int64_t qd = m.n_head * m.head_dim, kvd = m.n_head_kv * m.head_dim;
    const int64_t kh = 16 * 128, vh = 48 * 128;   /* the GDN's k / v dims */
    auto add = [&](const char* fmt, int l, int64_t rows, int64_t cols, int stacked) {
        char nm[96];
        std::snprintf(nm, sizeof nm, fmt, l);
        V3_ALL_NAMES.push_back(nm);
        V3_ALL.push_back({ V3_ALL_NAMES.back().c_str(), 64, EXL3I_CB_MAD, stacked, rows, cols });
    };
    for (int l = 0; l < (int)m.n_layers; ++l) {
        if (rad::arch::is_full_attention(l, 4)) {
            add("blk.%d.attn_qg.weight", l, 2 * qd, ne, 0);
            add("blk.%d.attn_k.weight", l, kvd, ne, 0);
            add("blk.%d.attn_v.weight", l, kvd, ne, 0);
            add("blk.%d.attn_output.weight", l, ne, qd, 0);
        } else {
            add("blk.%d.ssm_qkv.weight", l, 2 * kh + vh, ne, 0);
            add("blk.%d.ssm_z.weight", l, vh, ne, 0);
            add("blk.%d.ssm_out.weight", l, ne, vh, 0);
        }
        add("blk.%d.ffn_gate_up.weight", l, 2 * nf, ne, 1);
        add("blk.%d.ffn_down.weight", l, ne, nf, 0);
    }
    if (with_head) {
        V3_ALL_NAMES.push_back("output.weight");
        V3_ALL.push_back({ V3_ALL_NAMES.back().c_str(), 64, EXL3I_CB_MAD, 0, m.n_vocab, ne });
    }
    V3_WEIGHTS = V3_ALL.data(); N_V3 = (int)V3_ALL.size();
}

void test_64_layers_declare_and_blocks() {
    printf("[TEST] Checking full 64-layer declaration with derived Radiance blocks and EXL3 op...\n");
    RadBuilder b{};
    RadModelMeta m = make_qwen35_meta(64);
    RadBuildCtx ctx = make_ctx(0, 1, 128, 262144);

    /* Every projection weight an exl3i trellis (whole-weight extents) */
    reg_all_exl3i(m);

    int res = qwen35_exl3::declare(&b, &m, &ctx);
    assert(res == RAD_OK);

    /* Verify Radiance blocks declared their intrinsic ops:
     * - GDN: gdn_conv_prep, gdn_kkt_solve, gdn_chunk_scan, gdn_gated_rmsnorm, gdn_conv_update, gdn_recurrent_update
     * - AttnGated: attn_paged, kv_store, sigmoid, mul, rmsnorm
     * - Projections: exl3i_dense
     */
    int exl3_op_count = 0;
    int gdn_prep_count = 0;
    int gdn_scan_count = 0;
    int gdn_recur_count = 0;
    int attn_paged_count = 0;
    int sigmoid_count = 0;

    for (const auto& op : b.ops) {
        if (op.op == qwen35_exl3::kExl3iDenseOp) {
            exl3_op_count++;
            long long N = 0, K = 0;
            for (const auto& p : op.p) {
                if (p.key == "N") N = p.ival;
                if (p.key == "K") K = p.ival;
            }
            /* 128-alignment assertion */
            assert(N % 128 == 0);
            assert(K % 128 == 0);
        } else if (op.op == "gdn_conv_prep") {
            gdn_prep_count++;
        } else if (op.op == "gdn_chunk_scan") {
            gdn_scan_count++;
        } else if (op.op == "gdn_recurrent_update") {
            gdn_recur_count++;
        } else if (op.op == "attn_paged") {
            attn_paged_count++;
        } else if (op.op == "sigmoid") {
            sigmoid_count++;
        }
    }

    /* 48 GDN layers and 16 Full Attn layers */
    assert(gdn_prep_count == 48);
    assert(gdn_scan_count == 48);
    assert(gdn_recur_count == 48);
    assert(attn_paged_count == 16);
    assert(sigmoid_count == 16);

    /*
     * Projections declared via exl3i_dense:
     * - 16 Attn layers * 4 linears (qg, k, v, o) = 64
     * - 48 GDN layers * 3 linears (in, z, out) = 144
     * - 64 MLP layers * 2 linears (gate_up stacked, down) = 128
     * - 1 LM Head linear = 1
     * Total = 64 + 144 + 128 + 1 = 337 exl3i_dense ops (the model's 337 linears).
     */
    if (exl3_op_count != 337) fprintf(stderr, "exl3i_dense ops: %d\n", exl3_op_count);
    assert(exl3_op_count == 337);
    printf("  -> OK (Verified 48 GDN blocks with conv/recurrence, 16 AttnGated blocks with RoPE/sigmoid, 337 %s ops)\n",
           qwen35_exl3::kExl3iDenseOp);
}

void test_step_execution_and_ordering() {
    printf("[TEST] Checking step execution across GDN and AttnGated blocks...\n");
    RadBuilder b{};
    RadModelMeta m = make_qwen35_meta(4);
    RadBuildCtx ctx = make_ctx(0, 1, 128, 262144);

    reg_all_exl3i(m);

    int res = qwen35_exl3::declare(&b, &m, &ctx);
    assert(res == RAD_OK);

    RadCtx c{};
    c.rank = 0;
    int32_t fake_toks[16] = {0};
    int32_t fake_out[1] = {0};
    int32_t fake_cu[3] = {0, 16, 16};
    RadBatch batch{};
    batch.n_tok = 16;
    batch.n_seq = 1;
    batch.n_out = 1;
    batch.token_ids = fake_toks;
    batch.out_ids = fake_out;
    batch.cu_seqlens = fake_cu;

    qwen35_exl3::step(&c, &batch);
    assert(c.issues.size() > 0);

    /* Verify op issues: GDN chunk scan & recurrent update, attn_paged, and exl3i_dense */
    bool saw_gdn = false;
    bool saw_attn = false;
    bool saw_exl3 = false;

    for (const auto& issue : c.issues) {
        const auto* op_desc = b.op_of(issue.op);
        if (!op_desc) continue;
        if (op_desc->op == "gdn_chunk_scan" || op_desc->op == "gdn_conv_prep") saw_gdn = true;
        if (op_desc->op == "attn_paged") saw_attn = true;
        if (op_desc->op == qwen35_exl3::kExl3iDenseOp) saw_exl3 = true;
    }
    assert(saw_gdn);
    assert(saw_attn);
    assert(saw_exl3);
    printf("  -> OK (Verified execution of block ops and %s projections in step)\n", qwen35_exl3::kExl3iDenseOp);
}

} /* namespace */

/* The weights' activation modes: exl3.act.<weight> metadata keys reach the op as "act" (f16 where absent), and a
 * value that is not a mode refuses the declaration, as does a shared-input group (ssm qkv / z here) given two modes. */
void test_act_modes() {
    printf("[TEST] Checking per-weight activation modes (exl3.act.* metadata -> op parameter act)...\n");
    RadModelMeta m = make_qwen35_meta(4);
    const char* keys[] = { "exl3.act.blk.3.ffn_down.weight", "exl3.act.blk.0.ssm_qkv.weight", "exl3.act.blk.0.ssm_z.weight" };
    const char* vals[] = { "a8g128", "a8t", "a8t" };
    m.n_kv = 3; m.kv_key = keys; m.kv_val = vals;
    RadBuildCtx ctx = make_ctx(0, 1, 128, 262144);
    reg_all_exl3i(m);
    {
        RadBuilder b{};
        assert(qwen35_exl3::declare(&b, &m, &ctx) == RAD_OK);
        int n = 0, n_a8g = 0, n_a8t = 0, n_f16 = 0;
        for (const auto& op : b.ops) {
            if (op.op != qwen35_exl3::kExl3iDenseOp) continue;
            ++n;
            std::string act;
            for (const auto& p : op.p) if (p.key == "act") act = p.sval;
            const std::string w = b.weights[op.w[0] - 1].name;   /* the record plane: "<weight>.record" */
            if (w == "blk.3.ffn_down.weight.record") { assert(act == "a8g128"); ++n_a8g; }
            else if (w == "blk.0.ssm_qkv.weight.record" || w == "blk.0.ssm_z.weight.record") { assert(act == "a8t"); ++n_a8t; }
            else { assert(act == "f16"); ++n_f16; }
        }
        assert(n_a8g == 1 && n_a8t == 2 && n_f16 == n - 3);
        printf("  -> OK (%d exl3i_dense ops: 1 a8g128, 2 a8t, %d f16)\n", n, n_f16);
    }
    {
        const char* mixed[] = { "a8g128", "a8t", "a8g256" };   /* ssm qkv a8t, ssm z a8g256: one group, three modes */
        m.kv_val = mixed;
        RadBuilder b{};
        assert(qwen35_exl3::declare(&b, &m, &ctx) != RAD_OK);
        printf("  -> OK (a shared-input group with two modes refused)\n");
    }
    {
        const char* bad[] = { "a8g100", "a8t", "a8t" };
        m.kv_val = bad;
        RadBuilder b{};
        assert(qwen35_exl3::declare(&b, &m, &ctx) != RAD_OK);
        printf("  -> OK (a8g100 refused)\n");
    }
}

/* The trunk (and the head) as exl3i weights: each declares its record plane (u32 [rows/16, (cols/16) m/2], a stacked
 * weight's row parts in 16-row groups) and an exl3i_dense op with rate / codebook / act and no dtype; at TP2 the
 * planes are the rank's extents. */
void test_exl3i() {
    printf("[TEST] Checking exl3i (EXL3-V3) weights: the rates in 1/8 steps, their record planes, TP2 at the rank's extents...\n");
    /* four rates beside the K4 default, on an otherwise all-K4 trunk */
    static const V3Weight v3[] = {
        { "blk.0.ffn_down.weight", 40, EXL3I_CB_MAD, 0, 5120, 17408 },            /* K2.5 */
        { "blk.0.ffn_gate_up.weight", 36, EXL3I_CB_MAD, 1, 34816, 5120 },        /* K2.25, stacked */
        { "blk.3.attn_qg.weight", 8, EXL3I_CB_MAD, 0, 12288, 5120 },            /* K0.5 */
        { "blk.1.ssm_out.weight", 100, EXL3I_CB_MAD, 0, 5120, 6144 },           /* K6.25 */
    };
    RadModelMeta m = make_qwen35_meta(4);
    reg_all_exl3i(m);
    for (int i = 0; i < 4; ++i)
        for (auto& w : V3_ALL)
            if (std::strcmp(w.name, v3[i].name) == 0) { w.m = v3[i].m; w.cb = v3[i].cb; }
    {
        RadBuildCtx ctx = make_ctx(0, 1, 128, 262144);
        RadBuilder b{};
        assert(qwen35_exl3::declare(&b, &m, &ctx) == RAD_OK);
        int ni = 0, seen = 0;
        for (const auto& op : b.ops) {
            if (op.op != qwen35_exl3::kExl3iDenseOp) continue;
            ++ni;
            std::string rate, cb, act;
            bool dtype = false;
            for (const auto& p : op.p) {
                if (p.key == "rate") rate = p.sval;
                if (p.key == "codebook") cb = p.sval;
                if (p.key == "act") act = p.sval;
                if (p.key == "dtype") dtype = true;
            }
            assert(!dtype && act == "f16");
            const auto& rec = b.weights[op.w[0] - 1];
            for (int i = 0; i < 4; ++i) {
                if (rec.name != std::string(v3[i].name) + ".record") continue;
                ++seen;
                char want[16];
                assert(rate == exl3i_rate_str(v3[i].m, want) && cb == exl3i_codebook_name(v3[i].cb));
                assert(rec.d.dtype == RAD_U32 && rec.d.rank == 2 && rec.d.shape[0] == v3[i].rows / 16 &&
                       rec.d.shape[1] == exl3i_row_words(v3[i].cols, v3[i].m));
                assert(rec.d.planes && !std::strcmp(rec.d.planes, "record"));
                for (int j = 0; j < (int)rec.d.n_row_parts; ++j) assert(rec.d.row_parts[j] * 16 == m.n_ff);
                assert(op.w.size() == (size_t)(v3[i].stacked ? 4 : 3));
            }
        }
        assert(ni == 22 && seen == 4);   /* the 21 trunk weights + the head, all exl3i; the four rates kept */
        printf("  -> OK (22 exl3i_dense ops: the four rates 0.5 / 2.25 / 2.5 / 6.25 beside K4, every record plane)\n");
    }
    {
        /* TP2: the same container serves sharded -- the loader cuts each plane in its own elements (the record by
         * 16-row groups, the scales along their axis), so the declarations are the rank's extents: a row shard's
         * record [rows / 2 / 16, words(K)] with the suh whole and the svh halved, a column shard's record
         * [rows / 16, words(K / 2)] with the suh halved and the svh whole. */
        RadBuildCtx ctx = make_ctx(0, 2, 128, 262144);
        RadBuilder b{};
        assert(qwen35_exl3::declare(&b, &m, &ctx) == RAD_OK);
        auto recw = [&](const char* n) -> const RecWeight* {
            for (const auto& w : b.weights) if (w.name == n) return &w;
            return nullptr;
        };
        {   /* ffn_down: column shard, [5120, 17408] -> the rank's K 8704 */
            const RecWeight* c = recw("blk.0.ffn_down.weight.record");
            assert(c && c->d.shard == RAD_SHARD_COL && c->d.shape[0] == 5120 / 16 &&
                   c->d.shape[1] == exl3i_row_words(8704, 40));
            const RecWeight* s = recw("blk.0.ffn_down.weight.suh");
            const RecWeight* v = recw("blk.0.ffn_down.weight.svh");
            assert(s && s->d.shard == RAD_SHARD_COL && s->d.shape[1] == 8704 &&
                   v && v->d.shard == RAD_SHARD_NONE && v->d.shape[0] == 5120);
        }
        {   /* ffn_gate_up: row shard in two parts, [34816, 5120] -> the rank's 17408 rows (8704 a part) */
            const RecWeight* c = recw("blk.0.ffn_gate_up.weight.record");
            assert(c && c->d.shard == RAD_SHARD_ROW && c->d.n_row_parts == 2 && c->d.row_parts[0] == 8704 / 16 &&
                   c->d.shape[0] == 17408 / 16 && c->d.shape[1] == exl3i_row_words(5120, 36));
            const RecWeight* s = recw("blk.0.ffn_gate_up.weight.suh");
            const RecWeight* v = recw("blk.0.ffn_gate_up.weight.svh");
            assert(s && s->d.shard == RAD_SHARD_NONE && s->d.shape[1] == 5120 &&
                   v && v->d.shard == RAD_SHARD_ROW && v->d.n_row_parts == 2 && v->d.row_parts[0] == 8704 &&
                   v->d.shape[0] == 17408);
        }
        {   /* attn_qg: row shard, [12288, 5120] -> 6144 rows */
            const RecWeight* c = recw("blk.3.attn_qg.weight.record");
            assert(c && c->d.shard == RAD_SHARD_ROW && c->d.shape[0] == 6144 / 16 &&
                   c->d.shape[1] == exl3i_row_words(5120, 8));
        }
        {   /* ssm_out: column shard, [5120, 6144] -> the rank's K 3072 */
            const RecWeight* c = recw("blk.1.ssm_out.weight.record");
            assert(c && c->d.shard == RAD_SHARD_COL && c->d.shape[0] == 5120 / 16 &&
                   c->d.shape[1] == exl3i_row_words(3072, 100));
        }
        {   /* ssm_qkv: row shard in three parts, [10240, 5120] -> the rank's 5120 rows, a slice of each of q / k / v */
            const RecWeight* c = recw("blk.1.ssm_qkv.weight.record");
            assert(c && c->d.shard == RAD_SHARD_ROW && c->d.n_row_parts == 3 && c->d.row_parts[0] == 1024 / 16 &&
                   c->d.row_parts[1] == 1024 / 16 && c->d.row_parts[2] == 3072 / 16 && c->d.shape[0] == 5120 / 16 &&
                   c->d.shape[1] == exl3i_row_words(5120, 64));
            const RecWeight* v = recw("blk.1.ssm_qkv.weight.svh");
            assert(v && v->d.shard == RAD_SHARD_ROW && v->d.n_row_parts == 3 && v->d.row_parts[0] == 1024 &&
                   v->d.row_parts[1] == 1024 && v->d.row_parts[2] == 3072 && v->d.shape[0] == 5120);
        }
        int nsh = 0;
        for (const auto& o : b.ops) if (o.op == "all_reduce" && o.resolved) ++nsh;
        assert(nsh == 9);   /* 3 GDN out + 1 attention o + 4 MLP down, plus the row-sharded bf16 embed (this test's table) */
        printf("  -> OK (TP2: the exl3i planes declared at the rank's extents, record cut by 16-row groups, 9 all_reduces)\n");
    }
    V3_WEIGHTS = nullptr; N_V3 = 0;
}

/* AQ (exl3i_abi.h version 3): every trunk weight exl3i a8g128 mad (the one codebook).
 * Declared: one exl3i_actq a producer site (GDN norm -> qkv / z with h; attention norm -> qg / k / v; gate mul -> o; MLP
 * norm -> gate_up; silu_mul -> down), each consumer's aq_plane / aq_planes, the shared buffer once at the widest row.
 * A step of 16 rows issues as before; one of 1024 rows issues the sites in place of their producers and the consumers
 * with the aq operand. The step lint is clean over both. The aq_off pin declares none. The step's fusions are off here
 * (the fuse_off pin at every fusion: the counts are the unfused chain's); test_fuse has them with AQ. */
void test_aq() {
    printf("[TEST] Checking AQ: producer sites, consumers' planes, steps at 16 and 1024 rows...\n");
    qwen35_exl3::exl3_test_pins().fuse_off = 0xF;
    static std::vector<V3Weight> v3;
    static std::vector<std::string> names, keys_s;
    static std::vector<const char*> keys, vals;
    v3.clear(); names.clear(); keys_s.clear(); keys.clear(); vals.clear();
    names.reserve(64);
    auto add = [&](const char* fmt, int l) {
        char nm[96];
        std::snprintf(nm, sizeof nm, fmt, l);
        names.push_back(nm);
        keys_s.push_back(std::string("exl3.act.") + nm);
    };
    struct Sh { const char* fmt; int64_t rows, cols; int stacked; };
    const Sh gdn[] = { { "blk.%d.ssm_qkv.weight", 10240, 5120, 0 }, { "blk.%d.ssm_z.weight", 6144, 5120, 0 },
                       { "blk.%d.ssm_out.weight", 5120, 6144, 0 } };
    const Sh att[] = { { "blk.%d.attn_qg.weight", 12288, 5120, 0 }, { "blk.%d.attn_k.weight", 1024, 5120, 0 },
                       { "blk.%d.attn_v.weight", 1024, 5120, 0 }, { "blk.%d.attn_output.weight", 5120, 6144, 0 } };
    const Sh mlp[] = { { "blk.%d.ffn_gate_up.weight", 34816, 5120, 1 }, { "blk.%d.ffn_down.weight", 5120, 17408, 0 } };
    std::vector<std::pair<const Sh*, int>> all;
    for (int l = 0; l < 4; ++l) {
        if (l == 3) for (const Sh& x : att) all.push_back({ &x, l });
        else for (const Sh& x : gdn) all.push_back({ &x, l });
        for (const Sh& x : mlp) all.push_back({ &x, l });
    }
    for (auto& e : all) add(e.first->fmt, e.second);
    for (size_t i = 0; i < all.size(); ++i) {
        const Sh& x = *all[i].first;
        v3.push_back({ names[i].c_str(), 64, EXL3I_CB_MAD, x.stacked, x.rows, x.cols });
        keys.push_back(keys_s[i].c_str());
        vals.push_back("a8g128");
    }
    V3_WEIGHTS = v3.data(); N_V3 = (int)v3.size();
    RadModelMeta m = make_qwen35_meta(4);
    m.n_kv = (int)keys.size(); m.kv_key = keys.data(); m.kv_val = vals.data();
    RadBuildCtx ctx = make_ctx(0, 1, 4096, 262144);
    RadBuilder b{};
    assert(qwen35_exl3::declare(&b, &m, &ctx) == RAD_OK);

    auto pint = [](const RecOp& op, const char* k, long long dflt) {
        for (const auto& p : op.p) if (p.key == k) return (long long)p.ival;
        return dflt;
    };
    auto pstr = [](const RecOp& op, const char* k) {
        for (const auto& p : op.p) if (p.key == k) return p.sval;
        return std::string();
    };
    /* the sites (every trunk weight mad, so every layer's gate_up is served by AQ): 4 layers' gate_up norms, 3 GDN and 1
     * attention norms, layer 3's gate mul, 4 silu_mul (down) */
    int n_site = 0, n_rms = 0, n_silu = 0, n_mul = 0;
    for (const auto& op : b.ops) {
        if (op.op != EXL3I_OP_ACTQ) continue;
        ++n_site;
        const std::string pre = pstr(op, "pre");
        const long long pl = pint(op, "planes", 0);
        assert(op.w.size() == (size_t)(pre == "rmsnorm" ? 5 : 4));
        if (pre == "rmsnorm") { ++n_rms; assert(pl == 2 || pl == 3); assert(pint(op, "K", 0) == 5120); }
        else if (pre == "silu_mul") { ++n_silu; assert(pl == 1 && pint(op, "K", 0) == 17408); }
        else if (pre == "mul") { ++n_mul; assert(pl == 1 && pint(op, "K", 0) == 6144); }
        else assert(0);
        for (size_t i = (size_t)pl; i < 4; ++i) assert(op.w[i] == op.w[0]);   /* s0 stands in for the absent planes */
    }
    assert(n_rms == 3 + 1 + 4 && n_silu == 4 && n_mul == 1 && n_site == 13);
    /* the consumers' planes */
    int n_aq = 0, n_plain = 0;
    for (const auto& op : b.ops) {
        if (op.op != qwen35_exl3::kExl3iDenseOp) continue;
        const std::string w = b.weights[op.w[0] - 1].name;
        const long long p0 = pint(op, "aq_plane", -1), np = pint(op, "aq_planes", 0);
        if (np == 0) { ++n_plain; continue; }
        ++n_aq;
        if (w.find("attn_qg") != std::string::npos) assert(p0 == 0 && np == 3);
        else if (w.find("attn_k.") != std::string::npos) assert(p0 == 1 && np == 3);
        else if (w.find("attn_v.") != std::string::npos) assert(p0 == 2 && np == 3);
        else if (w.find("ssm_qkv") != std::string::npos) assert(p0 == 0 && np == 2);
        else if (w.find("ssm_z") != std::string::npos) assert(p0 == 1 && np == 2);
        else if (w.find("gate_up") != std::string::npos) assert(p0 == 0 && np == 2);
        else assert(p0 == 0 && np == 1);   /* down, o */
    }
    assert(n_aq == 3 * 2 + 4 + 4 + 4 && n_plain == 3);   /* ssm_out x 3 stay plain (the head is bf16: not an exl3i op) */
    int n_aqbuf = 0;
    for (const auto& bf : b.bufs)
        if (bf.name == "aq") {
            ++n_aqbuf;
            assert(bf.d.dtype == RAD_U8 && bf.d.shape[0] == 4096 && bf.d.shape[1] == exl3i_aq_row_bytes(17408, 1));
        }
    assert(n_aqbuf == 1);

    auto run = [&](int64_t T, RadCtx& c) {
        static std::vector<int32_t> toks;
        toks.assign((size_t)T, 0);
        static int32_t out1[1] = { 0 };
        static int32_t cu[3];
        cu[0] = 0; cu[1] = (int32_t)T; cu[2] = (int32_t)T;
        c.rank = 0;
        RadBatch batch{};
        batch.n_tok = T; batch.n_seq = 1; batch.n_out = 1;
        batch.token_ids = toks.data(); batch.out_ids = out1; batch.cu_seqlens = cu;
        qwen35_exl3::step(&c, &batch);
    };
    auto count = [&](const RadCtx& c, const char* opn, int* with_aq) {
        int n = 0;
        if (with_aq) *with_aq = 0;
        for (const auto& is : c.issues) {
            const auto* d = b.op_of(is.op);
            if (!d || d->op != opn) continue;
            ++n;
            if (with_aq && is.opd.size() > 6 && is.opd[6].kind == RAD_OPK_BUF) ++*with_aq;
        }
        return n;
    };
    {
        RadCtx c{};
        run(16, c);
        int wa = 0;
        assert(count(c, EXL3I_OP_ACTQ, nullptr) == 0);
        assert(count(c, qwen35_exl3::kExl3iDenseOp, &wa) == 21 && wa == 0);   /* the 21 trunk weights; the head is bf16 */
        assert(count(c, "silu_mul", nullptr) == 4);
        const auto lint = lint_step(b, { &c });
        assert(report_lint(lint) == 0);
        printf("  -> OK (16 rows: no exl3i_actq, 21 exl3i_dense without aq, lint clean)\n");
    }
    {
        RadCtx c{};
        run(1024, c);
        int wa = 0;
        assert(count(c, EXL3I_OP_ACTQ, nullptr) == 13);
        assert(count(c, qwen35_exl3::kExl3iDenseOp, &wa) == 21 && wa == 18);   /* the 21 trunk weights; the head is bf16 */
        assert(count(c, "silu_mul", nullptr) == 0 && count(c, "mul", nullptr) == 0);
        const int n_norm = count(c, "rmsnorm", nullptr);
        RadCtx c16{};
        run(16, c16);
        assert(n_norm == count(c16, "rmsnorm", nullptr) - 8);   /* the 8 norm sites are AQ's producers */
        const auto lint = lint_step(b, { &c });
        assert(report_lint(lint) == 0);
        printf("  -> OK (1024 rows: 12 exl3i_actq in place of 7 norms, 4 silu_mul, 1 mul; 18 of 21 exl3i_dense on aq; lint "
               "clean)\n");
    }
    {
        qwen35_exl3::exl3_test_pins().aq_off = true;
        RadBuilder b0{};
        assert(qwen35_exl3::declare(&b0, &m, &ctx) == RAD_OK);
        qwen35_exl3::exl3_test_pins().aq_off = false;
        for (const auto& op : b0.ops) assert(op.op != EXL3I_OP_ACTQ && pint(op, "aq_planes", 0) == 0);
        for (const auto& bf : b0.bufs) assert(bf.name != "aq");
        printf("  -> OK (aq_off pin: no site, no aq buffer)\n");
    }
    {
        RadBuilder b1{};
        b1.undeclared.insert(EXL3I_OP_ACTQ);   /* an exl3i from before version 3 */
        assert(qwen35_exl3::declare(&b1, &m, &ctx) == RAD_OK);
        for (const auto& op : b1.ops) assert(op.op != EXL3I_OP_ACTQ && pint(op, "aq_planes", 0) == 0);
        printf("  -> OK (no exl3i_actq in the vocabulary: the producers as before)\n");
    }
    qwen35_exl3::exl3_test_pins().fuse_off = 0;
    /* an exl3i_actq issue with the residual add (exl3i version 5: param res 1) */
    auto is_res = [&](const RadBuilder& bb, const RadCtx& c, const char* form) {
        int n = 0;
        for (const auto& is : c.issues) {
            const auto* d = bb.op_of(is.op);
            n += d && d->op == EXL3I_OP_ACTQ && pint(*d, "res", 0) == 1 && pstr(*d, "form") == form;
        }
        return n;
    };
    {   /* with the step's fusions (test_fuse): the add inside each AQ site but the first (actq's res), inside add_rmsnorm
         * where the norm is a plain one (the final norm, every norm at 16 rows) */
        RadBuilder b4{};
        assert(qwen35_exl3::declare(&b4, &m, &ctx) == RAD_OK);
        auto n4 = [&](const RadCtx& c, const char* opn) {
            int n = 0;
            for (const auto& is : c.issues) { const auto* d = b4.op_of(is.op); n += d && d->op == opn; }
            return n;
        };
        for (int64_t T : { (int64_t)1024, (int64_t)16 }) {
            static std::vector<int32_t> toks;
            toks.assign((size_t)T, 0);
            static int32_t out1[1] = { 0 };
            static int32_t cu[3];
            cu[0] = 0; cu[1] = (int32_t)T; cu[2] = (int32_t)T;
            RadCtx c{};
            c.rank = 0;
            RadBatch batch{};
            batch.n_tok = T; batch.n_seq = 1; batch.n_out = 1;
            batch.token_ids = toks.data(); batch.out_ids = out1; batch.cu_seqlens = cu;
            qwen35_exl3::step(&c, &batch);
            if (T == 1024) assert(n4(c, "add") == 0 && n4(c, "add_rmsnorm") == 1 && n4(c, EXL3I_OP_ACTQ) == 13 &&
                                  is_res(b4, c, "") == 7);
            else           assert(n4(c, "add") == 0 && n4(c, "add_rmsnorm") == 8 && n4(c, EXL3I_OP_ACTQ) == 0);
            assert(report_lint(lint_step(b4, { &c })) == 0);
        }
        printf("  -> OK (with the fusions: 1024 rows no add alone, 7 sites with the add + 1 add_rmsnorm, 16 rows 8 "
               "add_rmsnorm)\n");
    }
    {   /* mad consumers (version 4): at 1 row the sites issue their xb op, so the fusions leave them the norm (with the
         * add: version 5's xb form with res) and the gate's product as at 1024 rows (sigmoid then the site); 2 rows are
         * past the band: no site, 8 add_rmsnorm, act_mul */
        std::vector<V3Weight> v3m(v3);
        V3_WEIGHTS = v3m.data();
        RadBuilder b5{};
        qwen35_exl3::exl3_test_pins().aq_xb_off = false;   /* the decode band's xb sites on (main turns them off for the other tests) */
        assert(qwen35_exl3::declare(&b5, &m, &ctx) == RAD_OK);
        qwen35_exl3::exl3_test_pins().aq_xb_off = true;
        int n_xb = 0, n_xb_res = 0, n_res = 0;   /* the 7 rmsnorm sites have both forms with the add (version 5) too */
        for (const auto& op : b5.ops) {
            if (op.op != EXL3I_OP_ACTQ) continue;
            const bool xb = pstr(op, "form") == "xb", res = pint(op, "res", 0) == 1;
            n_xb += xb && !res; n_xb_res += xb && res; n_res += !xb && res;
        }
        /* layer 1's gate_up is an AQ site now (mad, after a residual add): one more site, so one more op in each class (the
         * xb form without res from declare, the xb and plain forms with res from declare_res) */
        assert(n_xb == 13 && n_xb_res == 8 && n_res == 8);
        auto n5 = [&](const RadCtx& c, const char* opn) {
            int n = 0;
            for (const auto& is : c.issues) { const auto* d = b5.op_of(is.op); n += d && d->op == opn; }
            return n;
        };
        for (int64_t T : { (int64_t)1, (int64_t)2 }) {
            static std::vector<int32_t> toks;
            toks.assign((size_t)T, 0);
            static int32_t out1[1] = { 0 };
            static int32_t cu[3];
            cu[0] = 0; cu[1] = (int32_t)T; cu[2] = (int32_t)T;
            RadCtx c{};
            c.rank = 0;
            RadBatch batch{};
            batch.n_tok = T; batch.n_seq = 1; batch.n_out = 1;
            batch.token_ids = toks.data(); batch.out_ids = out1; batch.cu_seqlens = cu;
            qwen35_exl3::step(&c, &batch);
            if (T == 1) assert(n5(c, "add") == 0 && n5(c, "add_rmsnorm") == 1 && n5(c, EXL3I_OP_ACTQ) == 13 &&
                               is_res(b5, c, "xb") == 7 && n5(c, "sigmoid") == 1 && n5(c, "act_mul") == 0 &&
                               n5(c, "mul") == 0);
            else        assert(n5(c, "add") == 0 && n5(c, "add_rmsnorm") == 8 && n5(c, EXL3I_OP_ACTQ) == 0 &&
                               n5(c, "sigmoid") == 0 && n5(c, "act_mul") == 1);
            assert(report_lint(lint_step(b5, { &c })) == 0);
        }
        printf("  -> OK (mad, with the fusions: 1 row 7 xb sites with the add + 1 add_rmsnorm, no add alone, the gate's "
               "sigmoid then o's site; 2 rows 8 add_rmsnorm, 1 act_mul)\n");
    }
    V3_WEIGHTS = nullptr; N_V3 = 0;
}

/* The step's fusions (exl3_fuse.h): a 4-layer trunk (GDN 0-2, attention 3), TP1. Declared: an add_rmsnorm a block
 * plus the final norm's, gdn_conv_recurrent_update in each GDN layer, the rope table with its fill and
 * qk_norm_rope_gate_kv_store in the attention layer. A decode step issues the residual adds only inside add_rmsnorm (one
 * plain rmsnorm: layer 0's input norm), the GDN fold in place of conv + recurrent update, the prologue in place of the q /
 * k norms, ropes and kv_store; a 512-row prefill chunk (past the prologue's band) and a mixed step of 102 rows the chain
 * there (the mixed step's decode rows the fold); a verify of 9 tokens a sequence the GDN pair (past the fold's window)
 * and the prologue. Every pass lints clean. The fuse_off pin at every fusion, a subset, a library that resolves none of the fused ops:
 * the chain where off. With AQ (every trunk weight exl3i a8g128 mad): at 1024 rows the add is issued alone before each
 * site after the first, at 16 rows inside add_rmsnorm. */
void test_fuse() {
    printf("[TEST] Checking the step's fusions (addnorm, gdn conv + recurrent, qk prologue)...\n");
    RadModelMeta m = make_qwen35_meta(4);
    reg_all_exl3i(m);
    auto build = [&](RadBuilder& b) {
        RadBuildCtx ctx = make_ctx(0, 1, 2048, 262144);
        ctx.max_seqs = 4;
        ctx.max_spec = 8;   /* a verify's rows sampled */
        return qwen35_exl3::declare(&b, &m, &ctx);
    };
    auto nops = [](const RadBuilder& b, const char* op) {
        int n = 0;
        for (const auto& o : b.ops) n += o.op == op;
        return n;
    };
    auto nis = [](const RadBuilder& b, const RadCtx& c, const char* op) {
        int n = 0;
        for (const auto& is : c.issues) { const auto* d = b.op_of(is.op); n += d && d->op == op; }
        return n;
    };
    RadBuilder b;
    assert(build(b) == RAD_OK);
    assert(nops(b, "add_rmsnorm") == 8 + 1 && nops(b, "gdn_conv_recurrent_update") == 3);
    assert(nops(b, "qk_norm_rope_gate_kv_store") == 1 && nops(b, "rope_table") == 1);
    bool cs = false;
    for (const auto& bf : b.bufs)
        if (bf.name == "rope_cs") { cs = true; assert(bf.d.kind == RAD_BUF_PERSIST && bf.d.shape[1] == 64); }
    assert(cs);
    const auto dec = kit(b, { 1, 1, 1, 1 }, {});
    const auto pre = kit(b, {}, { 512 });
    const auto mix = kit(b, { 1, 1 }, { 100 });
    const auto ver = kit(b, std::vector<int>(4, 9), {});
    {
        const auto v = run_passes(b, qwen35_exl3::step, { dec.get() });
        assert(steps_ok(v) && report_lint(lint_step(b, view(v))) == 0);
        const RadCtx& c = *v[0];
        assert(nis(b, c, "add") == 0 && nis(b, c, "add_rmsnorm") == 8 && nis(b, c, "rmsnorm") == 1);
        assert(nis(b, c, "gdn_conv_recurrent_update") == 3 && nis(b, c, "gdn_conv_update") == 0 &&
               nis(b, c, "gdn_recurrent_update") == 0);
        assert(nis(b, c, "rope_table") == 1 && nis(b, c, "qk_norm_rope_gate_kv_store") == 1);
        assert(nis(b, c, "rope") == 0 && nis(b, c, "kv_store") == 0);
        assert(nis(b, c, "act_mul") == 1 && nis(b, c, "sigmoid") == 0 && nis(b, c, "mul") == 0);
        printf("  -> OK (decode: 8 add_rmsnorm, no add, 3 GDN folds, rope_table + 1 prologue, 1 act_mul; lint clean)\n");
    }
    {
        const auto v = run_passes(b, qwen35_exl3::step, { pre.get() });
        assert(steps_ok(v) && report_lint(lint_step(b, view(v))) == 0);
        const RadCtx& c = *v[0];
        assert(nis(b, c, "add") == 0 && nis(b, c, "add_rmsnorm") == 8 && nis(b, c, "rmsnorm") == 1 + 2);
        assert(nis(b, c, "rope_table") == 0 && nis(b, c, "qk_norm_rope_gate_kv_store") == 0);
        assert(nis(b, c, "rope") == 2 && nis(b, c, "kv_store") == 1 && nis(b, c, "gdn_conv_recurrent_update") == 0);
        const auto vm = run_passes(b, qwen35_exl3::step, { mix.get() });
        assert(steps_ok(vm) && report_lint(lint_step(b, view(vm))) == 0);
        assert(nis(b, *vm[0], "gdn_conv_recurrent_update") == 3 && nis(b, *vm[0], "gdn_conv_prep") == 3);
        assert(nis(b, *vm[0], "qk_norm_rope_gate_kv_store") == 0 && nis(b, *vm[0], "rope") == 2);
        const auto vv = run_passes(b, qwen35_exl3::step, { ver.get() });
        assert(steps_ok(vv) && report_lint(lint_step(b, view(vv))) == 0);
        assert(nis(b, *vv[0], "gdn_conv_recurrent_update") == 0 && nis(b, *vv[0], "gdn_recurrent_update") == 3);
        assert(nis(b, *vv[0], "qk_norm_rope_gate_kv_store") == 1);
        printf("  -> OK (512-row prefill / 102-row mixed: the prologue's chain, mixed decode rows folded; 9-token verify: "
               "the GDN pair; lint clean)\n");
    }
    auto chain = [&](RadBuilder& b0, bool add, bool gdn, bool qk) {
        assert((nops(b0, "add_rmsnorm") > 0) == add && (nops(b0, "gdn_conv_recurrent_update") > 0) == gdn);
        assert((nops(b0, "qk_norm_rope_gate_kv_store") > 0) == qk && (nops(b0, "rope_table") > 0) == qk);
        const auto d0 = kit(b0, { 1, 1, 1, 1 }, {});
        const auto v = run_passes(b0, qwen35_exl3::step, { d0.get() });
        assert(steps_ok(v) && report_lint(lint_step(b0, view(v))) == 0);
        const RadCtx& c = *v[0];
        assert(nis(b0, c, "add") == (add ? 0 : 8) && nis(b0, c, "rmsnorm") == (add ? 1 : 9) + (qk ? 0 : 2));
        assert(nis(b0, c, "gdn_conv_update") == (gdn ? 0 : 3) && nis(b0, c, "kv_store") == (qk ? 0 : 1));
        assert(nis(b0, c, "act_mul") == (qk ? 1 : 0) && nis(b0, c, "sigmoid") == (qk ? 0 : 1));   /* gate rides with qk */
    };
    {
        qwen35_exl3::exl3_test_pins().fuse_off = 0xF;
        RadBuilder b0;
        assert(build(b0) == RAD_OK);
        chain(b0, false, false, false);
        qwen35_exl3::exl3_test_pins().fuse_off = qwen35_exl3::EXL3_FUSE_ADDNORM;   /* gdn, qk and gate on */
        RadBuilder b1;
        assert(build(b1) == RAD_OK);
        chain(b1, false, true, true);
        qwen35_exl3::exl3_test_pins().fuse_off = 0xF & ~qwen35_exl3::EXL3_FUSE_ADDNORM;   /* addnorm alone on */
        RadBuilder b2;
        assert(build(b2) == RAD_OK);
        chain(b2, true, false, false);
        qwen35_exl3::exl3_test_pins().fuse_off = 0;
        printf("  -> OK (fusions off / gdn,qk,gate / addnorm: the chain where off; lint clean)\n");
    }
    {
        RadBuilder b3;
        b3.refuse = { "add_rmsnorm", "gdn_conv_recurrent_update", "qk_norm_rope_gate_kv_store", "act_mul" };
        assert(build(b3) == RAD_OK);
        for (const auto& o : b3.ops) assert(o.resolved || b3.refuse.count(o.op));
        const auto d3 = kit(b3, { 1, 1, 1, 1 }, {});
        const auto v = run_passes(b3, qwen35_exl3::step, { d3.get() });
        assert(steps_ok(v) && report_lint(lint_step(b3, view(v))) == 0);
        const RadCtx& c = *v[0];
        /* the adds left to the next norm are issued alone: 7 by the next blocks, the last before the final norm */
        assert(nis(b3, c, "add") == 8 && nis(b3, c, "rmsnorm") == 9 + 2 && nis(b3, c, "gdn_conv_update") == 3);
        assert(nis(b3, c, "kv_store") == 1 && nis(b3, c, "rope_table") == 0 && nis(b3, c, "sigmoid") == 1);
        printf("  -> OK (no library resolves the fused ops: adds alone before the norms, the GDN pair, the chain)\n");
    }
}

/* The GDN's a | b as one trellis weight: ssm_ab exl3i K4 mad [128, 5120] in layers 0 and 2 (layer 1 keeps its
 * plain a / b), every trunk weight a8g128. A fused layer declares one exl3i_dense of N 128 and no a / b gemm_nt, the
 * GDN norm's site serves three planes (qkv 0, z 1, ab 2) and writes no h, and the GDN ops read a and b as 48-column
 * views of gdn_ab at prefill (16 and 1024 rows) and decode; layer 1 is as before. Refused: ssm_ab at another act mode
 * than qkv / z, a wrong shape, tensor parallelism 2. */
void test_ab() {
    printf("[TEST] Checking the fused GDN a | b (ssm_ab)...\n");
    static std::vector<V3Weight> v3;
    static std::vector<std::string> names, keys_s;
    static std::vector<const char*> keys, vals;
    v3.clear(); names.clear(); keys_s.clear(); keys.clear(); vals.clear();
    names.reserve(64);
    struct Sh { const char* fmt; int64_t rows, cols; int stacked; };
    const Sh gdn[] = { { "blk.%d.ssm_qkv.weight", 10240, 5120, 0 }, { "blk.%d.ssm_z.weight", 6144, 5120, 0 },
                       { "blk.%d.ssm_out.weight", 5120, 6144, 0 } };
    const Sh ab = { "blk.%d.ssm_ab.weight", 128, 5120, 0 };
    const Sh att[] = { { "blk.%d.attn_qg.weight", 12288, 5120, 0 }, { "blk.%d.attn_k.weight", 1024, 5120, 0 },
                       { "blk.%d.attn_v.weight", 1024, 5120, 0 }, { "blk.%d.attn_output.weight", 5120, 6144, 0 } };
    const Sh mlp[] = { { "blk.%d.ffn_gate_up.weight", 34816, 5120, 1 }, { "blk.%d.ffn_down.weight", 5120, 17408, 0 } };
    std::vector<std::pair<const Sh*, int>> all;
    for (int l = 0; l < 4; ++l) {
        if (l == 3) for (const Sh& x : att) all.push_back({ &x, l });
        else for (const Sh& x : gdn) all.push_back({ &x, l });
        if (l == 0 || l == 2) all.push_back({ &ab, l });
        for (const Sh& x : mlp) all.push_back({ &x, l });
    }
    for (auto& e : all) {
        char nm[96];
        std::snprintf(nm, sizeof nm, e.first->fmt, e.second);
        names.push_back(nm);
        keys_s.push_back(std::string("exl3.act.") + nm);
    }
    for (size_t i = 0; i < all.size(); ++i) {
        const Sh& x = *all[i].first;
        v3.push_back({ names[i].c_str(), 64, EXL3I_CB_MAD, x.stacked, x.rows, x.cols });
        keys.push_back(keys_s[i].c_str());
        vals.push_back("a8g128");
    }
    V3_WEIGHTS = v3.data(); N_V3 = (int)v3.size();
    RadModelMeta m = make_qwen35_meta(4);
    m.n_kv = (int)keys.size(); m.kv_key = keys.data(); m.kv_val = vals.data();
    RadBuildCtx ctx = make_ctx(0, 1, 4096, 262144);
    RadBuilder b{};
    assert(qwen35_exl3::declare(&b, &m, &ctx) == RAD_OK);

    auto pint = [](const RecOp& op, const char* k, long long dflt) {
        for (const auto& p : op.p) if (p.key == k) return (long long)p.ival;
        return dflt;
    };
    auto buf = [&](const char* n) {
        for (size_t i = 0; i < b.bufs.size(); ++i) if (b.bufs[i].name == n) return (rad_buf)(i + 1);
        return (rad_buf)0;
    };
    const rad_buf gab = buf("gdn_ab"), ga = buf("gdn_a"), gb = buf("gdn_b"), h = buf("h");
    assert(gab && ga && gb && h);
    assert(b.bufs[gab - 1].d.shape[0] == 4096 && b.bufs[gab - 1].d.shape[1] == 128);
    /* the weights: ssm_ab in layers 0 and 2, plain a / b in layer 1 only */
    int n_abw = 0, n_aw = 0;
    for (const auto& w : b.weights) {
        if (w.name.find(".ssm_ab.weight.record") != std::string::npos) {
            ++n_abw;
            assert(w.d.shape[0] == 128 / 16 && w.d.shape[1] == exl3i_row_words(5120, 64));
        }
        if (w.name == "blk.1.ssm_a.weight" || w.name == "blk.1.ssm_b.weight") ++n_aw;
        assert(w.name != "blk.0.ssm_a.weight" && w.name != "blk.2.ssm_b.weight");
    }
    assert(n_abw == 2 && n_aw == 2);
    /* the ops: one exl3i_dense of N 128 a fused layer, on plane 2 of 3; qkv / z on 0 / 1 of 3 there, of 2 in layer 1 */
    int n_ab = 0, n_gemm_ab = 0, n_site3 = 0;
    for (const auto& op : b.ops) {
        if (op.op == "gemm_nt" && pint(op, "N", 0) == 48) ++n_gemm_ab;
        if (op.op == EXL3I_OP_ACTQ && pint(op, "K", 0) == 5120) {
            const long long pl = pint(op, "planes", 0);
            if (pl == 3 && op.writes.size() == 1) ++n_site3;   /* the aq rows only: no h */
        }
        if (op.op != qwen35_exl3::kExl3iDenseOp) continue;
        const std::string w = b.weights[op.w[0] - 1].name;
        const long long p0 = pint(op, "aq_plane", -1), np = pint(op, "aq_planes", 0);
        const bool fused = w.rfind("blk.0.", 0) == 0 || w.rfind("blk.2.", 0) == 0;
        if (w.find("ssm_ab") != std::string::npos) {
            ++n_ab;
            assert(pint(op, "N", 0) == 128 && pint(op, "K", 0) == 5120 && p0 == 2 && np == 3);
            assert(op.reads.size() == 3 && op.reads[0] == h && op.writes.size() == 1 && op.writes[0] == gab);
        } else if (w.find("ssm_qkv") != std::string::npos) assert(p0 == 0 && np == (fused ? 3 : 2));
        else if (w.find("ssm_z") != std::string::npos) assert(p0 == 1 && np == (fused ? 3 : 2));
    }
    assert(n_ab == 2 && n_gemm_ab == 2 && n_site3 == 3);   /* the two fused GDN norms and the attention norm */
    for (const auto& op : b.ops)
        if (op.op == "gdn_conv_prep" || op.op == "gdn_recurrent_update") {
            const bool rd_ab = std::find(op.reads.begin(), op.reads.end(), gab) != op.reads.end();
            const bool rd_a = std::find(op.reads.begin(), op.reads.end(), ga) != op.reads.end();
            assert(rd_ab != rd_a);
        }

    auto run = [&](int64_t T, bool decode, RadCtx& c, int rank = 0) {
        static std::vector<int32_t> toks;
        toks.assign((size_t)T, 0);
        static int32_t out1[1] = { 0 };
        static int32_t cu[3];
        cu[0] = 0; cu[1] = (int32_t)T; cu[2] = (int32_t)T;
        c.rank = rank;
        RadBatch batch{};
        batch.n_tok = T; batch.n_seq = 1; batch.n_out = 1;
        batch.token_ids = toks.data(); batch.out_ids = out1; batch.cu_seqlens = cu;
        if (decode) { batch.phase = RAD_PHASE_DECODE; batch.max_q_len = 1; }
        else batch.phase = RAD_PHASE_PREFILL;
        qwen35_exl3::step(&c, &batch);
    };
    /* the a / b operands of the GDN ops at one step: column views of gdn_ab (fused) or the plain buffers */
    auto check_gdn = [&](const RadCtx& c, const char* opn, size_t ia, size_t ib, int64_t rows, int a_off, int b_off) {
        int n_fused = 0, n_plain = 0;
        for (const auto& is : c.issues) {
            const auto* d = b.op_of(is.op);
            if (!d || d->op != opn) continue;
            const RadOperand& a = is.opd[ia];
            const RadOperand& bb = is.opd[ib];
            if (a.handle == gab) {
                ++n_fused;
                assert(bb.handle == gab && a.offset == a_off && a.cols == 48 && bb.offset == b_off && bb.cols == 48);
                assert(a.rows == rows && bb.rows == rows);
            } else {
                ++n_plain;
                assert(a.handle == ga && bb.handle == gb && a.cols == 0 && bb.cols == 0);
            }
        }
        assert(n_fused == 2 && n_plain == 1);
    };
    auto count = [&](const RadCtx& c, const char* opn, int64_t N) {
        int n = 0;
        for (const auto& is : c.issues) {
            const auto* d = b.op_of(is.op);
            if (d && d->op == opn && (N == 0 || pint(*d, "N", 0) == N)) ++n;
        }
        return n;
    };
    for (int64_t T : { (int64_t)16, (int64_t)1024 }) {
        RadCtx c{};
        run(T, false, c);
        check_gdn(c, "gdn_conv_prep", 7, 8, T, 0, 48);
        assert(count(c, qwen35_exl3::kExl3iDenseOp, 128) == 2 && count(c, "gemm_nt", 48) == 2);
        assert(count(c, EXL3I_OP_ACTQ, 0) == (T >= 512 ? 13 : 0));   /* 3 GDN + 1 attention + 4 MLP norms, 4 silu_mul, 1 mul */
        const auto lint = lint_step(b, { &c });
        assert(report_lint(lint) == 0);
    }
    {
        RadCtx c{};
        run(1, true, c);
        check_gdn(c, "gdn_conv_recurrent_update", 5, 6, 1, 0, 48);   /* exl3_fuse.h gdn: the fold reads a / b at decode */
        assert(count(c, "gdn_recurrent_update", 0) == 0);
        assert(count(c, qwen35_exl3::kExl3iDenseOp, 128) == 2 && count(c, "gemm_nt", 48) == 2);
        const auto lint = lint_step(b, { &c });
        assert(report_lint(lint) == 0);
    }
    printf("  -> OK (2 fused layers: one exl3i_dense N 128 on AQ plane 2 of 3, the site writes no h, a / b column views "
           "of gdn_ab at prefill 16 / 1024 and decode (the GDN fold); the plain layer as before; lint clean)\n");

    /* refusals */
    {
        vals[2 * 0 + 0] = "f16";   /* layer 0's ssm_qkv f16 against z / ab a8g128 */
        RadBuilder b1{};
        assert(qwen35_exl3::declare(&b1, &m, &ctx) == RAD_E_FORMAT);
        vals[0] = "a8g128";
        for (auto& w : v3) if (std::strstr(w.name, "ssm_ab")) w.rows = 256;
        RadBuilder b2{};
        assert(qwen35_exl3::declare(&b2, &m, &ctx) == RAD_E_SHAPE);
        for (auto& w : v3) if (std::strstr(w.name, "ssm_ab")) w.rows = 128;
        printf("  -> OK (refused: an act mode other than qkv / z's, a 256-row ssm_ab)\n");
    }

    /* TP2: ssm_ab serves whole on every rank (its a / b rows are not a slice of the trellis record); the rank's gate and
     * beta are its own value heads' columns -- a at rank * 24, b at 48 + rank * 24 of the full 128-wide output. */
    {
        RadBuildCtx ctx2 = make_ctx(0, 2, 4096, 262144);
        RadBuilder b2{};
        assert(qwen35_exl3::declare(&b2, &m, &ctx2) == RAD_OK);
        int n_whole = 0;
        for (const auto& w : b2.weights)
            if (w.name.find(".ssm_ab.weight.record") != std::string::npos) {
                ++n_whole;
                assert(w.d.shard == RAD_SHARD_NONE && w.d.shape[0] == 128 / 16 &&
                       w.d.shape[1] == exl3i_row_words(5120, 64));
            }
        assert(n_whole == 2);
        const RecBuf* ab2 = nullptr;
        for (const auto& bf : b2.bufs) if (bf.name == "gdn_ab") ab2 = &bf;
        assert(ab2 && ab2->d.shape[1] == 128);   /* the full a | b | padding on every rank */
        int nar = 0;
        for (const auto& o : b2.ops) if (o.op == "all_reduce" && o.resolved) ++nar;
        assert(nar == 9);   /* 3 GDN out + 1 attention o + 4 MLP down, plus the row-sharded bf16 embed (this test's table) */
        for (int rank : { 0, 1 }) {
            RadBuildCtx ctxr = make_ctx(rank, 2, 4096, 262144);
            RadBuilder br{};
            assert(qwen35_exl3::declare(&br, &m, &ctxr) == RAD_OK);
            rad_buf gabr = 0;
            for (size_t i = 0; i < br.bufs.size(); ++i) if (br.bufs[i].name == "gdn_ab") gabr = (rad_buf)(i + 1);
            assert(gabr);
            RadCtx c{};
            run(16, false, c, rank);
            int n_fused = 0;
            for (const auto& is : c.issues) {
                const auto* d = br.op_of(is.op);
                if (!d || d->op != "gdn_conv_prep") continue;
                const RadOperand& a = is.opd[7];
                const RadOperand& bb = is.opd[8];
                if (a.handle != gabr) continue;   /* the plain layer's */
                ++n_fused;
                assert(bb.handle == gabr && a.offset == rank * 24 && a.cols == 24 &&
                       bb.offset == 48 + rank * 24 && bb.cols == 24 && a.rows == 16 && bb.rows == 16);
            }
            assert(n_fused == 2);
            const auto lint = lint_step(br, { &c });
            assert(report_lint(lint) == 0);
        }
        printf("  -> OK (TP2: ssm_ab whole on both ranks, a / b the rank's columns (offsets 0 / 48 and 24 / 72 of 24))\n");
    }
    V3_WEIGHTS = nullptr; N_V3 = 0;
}

/* Media and DFlash2: an 8-layer trunk with the Qwen3.8 vision tower cut to two blocks and a DFlash2
 * drafter merged under `draft.` (two layers reading trunk layers 1, 3, 5; block 8; selector rank 64 over 8), its
 * linears MXFP4, the trunk exl3i. Declared: the encoder "vision" and the scatter, the taps into dflash_src (one `cast`),
 * the block drafter at depth 7, and the probe's fixed depth 7. Every pass an engine step runs -- decode, prefill, a
 * mixed step, a verify followed by the drafter's context and query passes, an encoder pass -- runs without a step
 * failure and lints clean. Without --max-spec no drafter or taps; without media no tower. */
struct EncKit {
    std::vector<uint16_t> pix;
    std::vector<int32_t>  coord, cu;
    RadBatch b{};
};
static std::unique_ptr<EncKit> enc_pass(int32_t mod, int64_t n, int64_t patch_dim) {
    auto k = std::make_unique<EncKit>();
    k->pix.assign((size_t)(n * patch_dim), 0);
    k->coord.assign((size_t)(4 * n), 1);
    k->cu = { 0, (int32_t)n };
    k->b.enc = mod; k->b.enc_n_patch = n; k->b.enc_pixels = k->pix.data(); k->b.enc_coord = k->coord.data();
    k->b.enc_n_seg = 1; k->b.enc_cu = k->cu.data(); k->b.enc_max_seg = (int32_t)n;
    return k;
}

void test_media_and_dflash2() {
    printf("[TEST] Checking the vision tower and the DFlash2 drafter (stock blocks over the exl3i trunk)...\n");
    MetaKV mk(make_qwen35_meta(8));
    mk.set("rope_parameters.mrope_section", "11 11 10").set("rope_parameters.mrope_interleaved", "1")
      .set("vision_config.depth", "2").set("vision_config.hidden_size", "1152").set("vision_config.num_heads", "16")
      .set("vision_config.intermediate_size", "4304").set("vision_config.patch_size", "16")
      .set("vision_config.temporal_patch_size", "2").set("vision_config.in_channels", "3")
      .set("vision_config.spatial_merge_size", "2").set("vision_config.out_hidden_size", "5120")
      .set("vision_config.num_position_embeddings", "2304").set("vision_config.deepstack_visual_indexes", "")
      .set("vision_config.hidden_act", "gelu_pytorch_tanh")
      .set("draft.architectures", "DFlash2DraftModel").set("draft.num_hidden_layers", "2")
      .set("draft.head_dim", "128").set("draft.num_attention_heads", "32").set("draft.num_key_value_heads", "8")
      .set("draft.intermediate_size", "8192").set("draft.rms_norm_eps", "1e-06").set("draft.rope_theta", "10000000")
      .set("draft.vocab_size", "248320").set("draft.dflash_config.target_layer_ids", "1 3 5")
      .set("draft.dflash_config.block_size", "8").set("draft.dflash_config.mask_token_id", "248070")
      .set("draft.dflash_config.selector_rank", "64").set("draft.dflash_config.selector_top_k", "8")
      .set("draft.dflash_config.conv_kernel_size", "2").set("draft.dflash_config.conv_group_size", "16");
    reg_all_exl3i(*mk.meta(), false);   /* the 8-layer trunk exl3i (whole-weight extents); the head stays plain */
    const RadEncoding mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, rad::arch::RAD_MX_GROUP);
    auto build = [&](RadBuilder& b, int max_spec, int64_t patches) {
        b.encs = { { "v.", rad_enc_plain(RAD_BF16) }, { "dflash.draft_head", rad_enc_plain(RAD_BF16) },
                   { "dflash.", mx } };
        RadBuildCtx ctx = make_ctx(0, 1, 2048, 262144);
        ctx.max_seqs = 4;
        ctx.max_spec = max_spec;
        ctx.max_enc_patches = patches;
        return qwen35_exl3::declare(&b, mk.meta(), &ctx);
    };
    RadBuilder b;
    assert(build(b, 7, 1024) == RAD_OK);
    assert(b.encoder_name == "vision" && b.encoder.max_patches == 1024);
    {   /* every declared name mapped once (the builder refuses a second COPY map of one weight; the fakes record it) */
        std::map<std::string, int> nmap;
        for (const auto& mp : b.maps) ++nmap[mp.declared];
        for (const auto& [k, n] : nmap) { if (n != 1) fprintf(stderr, "    '%s' mapped %d times\n", k.c_str(), n); assert(n == 1); }
        assert(nmap.count("v.patch.weight") == 1);
    }
    assert(b.drafter.kind == RAD_DRAFT_BLOCK && b.drafter.depth == 7 && b.drafter.block == 8 && b.drafter.mask_token == 248070);
    int n_cast = 0, n_sel = 0, n_scatter = 0;
    for (const auto& o : b.ops) {
        n_cast += o.op == "cast";
        n_sel += o.op == "dflash_select";
        n_scatter += o.op == "scatter_rows";
    }
    assert(n_cast == 1 && n_sel >= 1 && n_scatter >= 1);
    bool src = false;
    for (const auto& bf : b.bufs)
        if (bf.name == "dflash_src") { src = true; assert(bf.d.shape[0] == 2048 && bf.d.shape[1] == 3 * 5120); }
    assert(src);
    {
        RadArchProbe pr{};
        assert(qwen35_exl3::probe(mk.meta(), &pr) == RAD_OK && pr.draft_depth == 7 && pr.draft_depth_fixed == 1);
        RadModelMeta plain = make_qwen35_meta(8);
        RadArchProbe p0{};
        assert(qwen35_exl3::probe(&plain, &p0) == RAD_OK && p0.draft_depth == 0);
    }
    /* the engine's passes, linted */
    const auto dec = kit(b, { 1, 1, 1, 1 }, {});
    const auto pre = kit(b, {}, { 32, 32 });
    const auto mix = kit(b, { 1, 8 }, { 6 });
    const auto ver = kit(b, std::vector<int>(4, 8), {});
    for (const Kit* k : { dec.get(), pre.get(), mix.get() }) {
        const auto v = run_passes(b, qwen35_exl3::step, { k });
        assert(steps_ok(v) && report_lint(lint_step(b, view(v))) == 0);
    }
    const auto cx = context_pass(b, *ver);
    const auto qp = query_pass(b, 4, 8, 7);
    const auto v = run_passes(b, qwen35_exl3::step, { ver.get(), cx.get(), qp.get() });
    assert(steps_ok(v) && report_lint(lint_step(b, view(v))) == 0);
    int taps = 0;
    for (const auto& is : v[0]->issues) { const auto* o = b.op_of(is.op); taps += o && o->op == "cast"; }
    assert(taps == 3);   /* a trunk step writes each tap once */
    const auto en = enc_pass(RAD_MM_IMAGE, 256, 3 * 2 * 16 * 16);
    {
        std::vector<std::unique_ptr<RadCtx>> ve;
        ve.push_back(std::make_unique<RadCtx>());
        ve[0]->b = &b; ve[0]->batch = &en->b; ve[0]->rank = 0; ve[0]->world = 1;
        qwen35_exl3::step(ve[0].get(), &en->b);
        assert(steps_ok(ve) && !ve[0]->issues.empty() && report_lint(lint_step(b, view(ve))) == 0);
    }
    printf("  -> OK (encoder 'vision', taps 1 3 5 into dflash_src, block drafter depth 7; decode / prefill / mixed / "
           "verify + context + query / encoder passes clean)\n");
    {
        RadBuilder b0;
        assert(build(b0, 0, 0) == RAD_OK);
        assert(b0.encoder_name.empty() && b0.drafter.kind == 0);
        for (const auto& bf : b0.bufs) assert(bf.name != "dflash_src");
        for (const auto& w : b0.weights) assert(w.name.rfind("v.", 0) != 0 && w.name.rfind("dflash.", 0) != 0);
        printf("  -> OK (max_spec 0, no media: no drafter, no taps, no tower)\n");
    }
    {   /* media asked for, but the container packed text-only (vision_config kept, no v.* weights): text only */
        RadBuilder b1;
        b1.encs = { { "dflash.draft_head", rad_enc_plain(RAD_BF16) }, { "dflash.", mx } };
        RadBuildCtx ctx = make_ctx(0, 1, 2048, 262144);
        ctx.max_seqs = 4; ctx.max_spec = 7; ctx.max_enc_patches = 1024;
        assert(qwen35_exl3::declare(&b1, mk.meta(), &ctx) == RAD_OK);
        assert(b1.encoder_name.empty());
        int n_scatter = 0;
        for (const auto& o : b1.ops) n_scatter += o.op == "scatter_rows";
        for (const auto& w : b1.weights) assert(w.name.rfind("v.", 0) != 0);
        assert(n_scatter == 0 && b1.drafter.kind == RAD_DRAFT_BLOCK);
        const auto pre1 = kit(b1, {}, { 32, 32 });
        const auto v1 = run_passes(b1, qwen35_exl3::step, { pre1.get() });
        assert(steps_ok(v1) && report_lint(lint_step(b1, view(v1))) == 0);
        printf("  -> OK (a text-only container with vision_config: no tower, no scatter, the drafter still declared)\n");
    }
    /* the rescoring head: absent from a container without the recipe line (above: mapped, not declared, no
     * logit_rerank, candidates sel_topk wide); an int8 copy of lm_head turns logit_rerank on at kRerankR candidates */
    auto cand_width = [](const RadBuilder& bb) {
        for (const auto& bf : bb.bufs) if (bf.name == "df_cand") return bf.d.shape[1];
        return (int64_t)-1;
    };
    auto n_op = [](const RadBuilder& bb, const char* name) {
        int n = 0;
        for (const auto& o : bb.ops) n += o.op == name;
        return n;
    };
    {
        bool rh = false;
        for (const auto& w : b.weights) rh |= w.name == "dflash.rerank_head.weight";
        int nmap = 0;
        for (const auto& mp : b.maps) nmap += mp.declared == "dflash.rerank_head.weight";
        assert(!rh && nmap == 1 && n_op(b, "logit_rerank") == 0 && cand_width(b) == 8);
    }
    for (const RadEncoding& re : { rad_enc_affine(RAD_I8, RAD_F16, 1, 128), rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128) }) {
        RadBuilder b2;
        b2.encs = { { "v.", rad_enc_plain(RAD_BF16) }, { "dflash.rerank_head", re },
                    { "dflash.draft_head", rad_enc_plain(RAD_BF16) }, { "dflash.", mx } };
        RadBuildCtx ctx = make_ctx(0, 1, 2048, 262144);
        ctx.max_seqs = 4; ctx.max_spec = 7; ctx.max_enc_patches = 1024;
        assert(qwen35_exl3::declare(&b2, mk.meta(), &ctx) == RAD_OK);
        bool rh = false;
        for (const auto& w : b2.weights)
            if (w.name == "dflash.rerank_head.weight") {
                rh = true;
                assert(w.d.dtype == re.plane[0].dtype && w.d.shape[0] == 248320 && w.d.shape[1] == 5120);
            }
        assert(rh && n_op(b2, "logit_rerank") == 1 && cand_width(b2) == qwen35_exl3::Dflash2::kRerankR);
        const auto ver2 = kit(b2, std::vector<int>(4, 8), {});
        const auto cx2  = context_pass(b2, *ver2);
        const auto qp2  = query_pass(b2, 4, 8, 7);
        const auto v2 = run_passes(b2, qwen35_exl3::step, { ver2.get(), cx2.get(), qp2.get() });
        assert(steps_ok(v2) && report_lint(lint_step(b2, view(v2))) == 0);
        int reranks = 0;
        for (const auto& is : v2[2]->issues) { const auto* o = b2.op_of(is.op); reranks += o && o->op == "logit_rerank"; }
        assert(reranks == 1);
    }
    printf("  -> OK (rescoring head: none without the recipe line; an int8 / block-fp8 copy of lm_head -> logit_rerank "
           "over 32 candidates in the query pass)\n");
    /* an exl3i head and no copy -> exl3i_rerank over the head's own record / suh / svh (the handles exl3i_dense
     * reads), 32 candidates, issued once in the query pass; a copy wins (logit_rerank); the rerank_off pin, an exl3i
     * library without the op, or no kernel for it -> the coarse head's own scores, sel_topk wide */
    {
        RadEncoding he{};
        assert(exl3i_encoding_make_r16(64, EXL3I_CB_MAD, 0, 248320, 5120, &he) == RAD_OK);
        enum { PLAIN, OFF, UNDECLARED, NOKERNEL, COPY };
        for (int mode : { PLAIN, OFF, UNDECLARED, NOKERNEL, COPY }) {
            RadBuilder b3;
            /* the trunk stays exl3i (the wrapper's table); the head answers the R16 encoding he */
            b3.encs = { { "v.", rad_enc_plain(RAD_BF16) }, { "dflash.draft_head", rad_enc_plain(RAD_BF16) }, { "dflash.", mx },
                        { "output.", he } };
            if (mode == COPY) b3.encs.insert(b3.encs.begin(), { "dflash.rerank_head", rad_enc_affine(RAD_I8, RAD_F16, 1, 128) });
            if (mode == UNDECLARED) b3.undeclared.insert(EXL3I_OP_RERANK);
            if (mode == NOKERNEL) b3.refuse.insert(EXL3I_OP_RERANK);
            if (mode == OFF) qwen35_exl3::exl3_test_pins().rerank_off = true;
            RadBuildCtx ctx = make_ctx(0, 1, 2048, 262144);
            ctx.max_seqs = 4; ctx.max_spec = 7; ctx.max_enc_patches = 1024;
            assert(qwen35_exl3::declare(&b3, mk.meta(), &ctx) == RAD_OK);
            qwen35_exl3::exl3_test_pins().rerank_off = false;
            const RecOp* rr = nullptr;
            int nrr = 0;
            for (const auto& o : b3.ops) if (o.op == EXL3I_OP_RERANK && o.resolved) { rr = &o; ++nrr; }
            if (mode == PLAIN) {
                assert(nrr == 1 && n_op(b3, "logit_rerank") == 0 && cand_width(b3) == qwen35_exl3::Dflash2::kRerankR);
                std::string rate, cbn;
                int64_t R = 0, N = 0, K = 0;
                for (const auto& p : rr->p) {
                    if (p.key == "rate") rate = p.sval;
                    if (p.key == "codebook") cbn = p.sval;
                    if (p.key == "R") R = p.ival;
                    if (p.key == "N") N = p.ival;
                    if (p.key == "K") K = p.ival;
                }
                assert(rate == "4" && cbn == "mad" && R == 32 && N == 248320 && K == 5120);
                /* the head's own planes: the weights exl3i_dense (the logits op) reads */
                const RecOp* lg = nullptr;
                for (const auto& o : b3.ops)
                    if (o.op == "exl3i_dense") for (const auto& p : o.p) if (p.key == "N" && p.ival == 248320) lg = &o;
                assert(lg && rr->w.size() == 3 && lg->w.size() >= 3);
                for (int i = 0; i < 3; ++i) assert(rr->w[i] == lg->w[i]);
            } else if (mode == COPY) {
                assert(nrr == 0 && n_op(b3, "logit_rerank") == 1 && cand_width(b3) == qwen35_exl3::Dflash2::kRerankR);
            } else {
                assert(nrr == 0 && n_op(b3, "logit_rerank") == 0 && cand_width(b3) == 8);
            }
            const auto ver3 = kit(b3, std::vector<int>(4, 8), {});
            const auto cx3  = context_pass(b3, *ver3);
            const auto qp3  = query_pass(b3, 4, 8, 7);
            const auto v3 = run_passes(b3, qwen35_exl3::step, { ver3.get(), cx3.get(), qp3.get() });
            assert(steps_ok(v3) && report_lint(lint_step(b3, view(v3))) == 0);
            int issued = 0;
            for (const auto& is : v3[2]->issues) { const auto* o = b3.op_of(is.op); issued += o && o->op == EXL3I_OP_RERANK; }
            assert(issued == (mode == PLAIN ? 1 : 0));
        }
    }
    printf("  -> OK (an exl3i head without a copy -> exl3i_rerank over output.weight's own planes, 32 candidates, issued "
           "once in the query pass; a copy -> logit_rerank; off / undeclared / no kernel -> no rescoring)\n");
    /* an exl3i trellis token table (K2.5 mad) and the drafter -> the query pass embeds its block through
     * embed_exl3i over the trunk's own record / suh / svh (M up to the query rows, 4 x 8), writing df_x, no embed_lookup
     * anywhere and no all-reduce; issued once in the query pass, lint clean */
    {
        RadEncoding te{};
        assert(exl3i_encoding_make_r16(40, EXL3I_CB_MAD, 0, 248320, 5120, &te) == RAD_OK);
        RadBuilder b4;
        b4.encs = { { "v.", rad_enc_plain(RAD_BF16) }, { "dflash.draft_head", rad_enc_plain(RAD_BF16) }, { "dflash.", mx },
                    { "token_embd.", te } };
        RadBuildCtx ctx = make_ctx(0, 1, 2048, 262144);
        ctx.max_seqs = 4; ctx.max_spec = 7; ctx.max_enc_patches = 1024;
        assert(qwen35_exl3::declare(&b4, mk.meta(), &ctx) == RAD_OK);
        assert(b4.drafter.kind == RAD_DRAFT_BLOCK && n_op(b4, "embed_lookup") == 0);
        std::vector<const RecOp*> em;
        for (const auto& o : b4.ops) if (o.op == "embed_exl3i") em.push_back(&o);
        assert(em.size() == 2);
        int64_t mmax[2] = {};
        for (int i = 0; i < 2; ++i) {
            int64_t mm = 0, nv = 0;
            std::string cbn;
            for (const auto& p : em[i]->p) {
                if (p.key == "M") mmax[i] = p.ihi;
                if (p.key == "m") mm = p.ival;
                if (p.key == "n_vocab") nv = p.ival;
                if (p.key == "codebook") cbn = p.sval;
            }
            assert(mm == 40 && nv == 248320 && cbn == "mad" && em[i]->w.size() == 3);
        }
        for (int i = 0; i < 3; ++i) assert(em[0]->w[i] == em[1]->w[i]);   /* one table, the trunk's */
        const RecOp* dq = mmax[0] == 32 ? em[0] : em[1];
        assert((mmax[0] == 32) != (mmax[1] == 32) && dq->writes.size() == 1);
        rad_buf dfx = 0;
        for (size_t i = 0; i < b4.bufs.size(); ++i) if (b4.bufs[i].name == "df_x") dfx = (rad_buf)(i + 1);
        assert(dfx && dq->writes[0] == dfx);
        const auto ver4 = kit(b4, std::vector<int>(4, 8), {});
        const auto cx4  = context_pass(b4, *ver4);
        const auto qp4  = query_pass(b4, 4, 8, 7);
        const auto v4 = run_passes(b4, qwen35_exl3::step, { ver4.get(), cx4.get(), qp4.get() });
        assert(steps_ok(v4) && report_lint(lint_step(b4, view(v4))) == 0);
        int issued = 0;
        for (const auto& is : v4[2]->issues) { const auto* o = b4.op_of(is.op); issued += o == dq; }
        assert(issued == 1);
    }
    printf("  -> OK (an exl3i token table: the drafter's block embedded by embed_exl3i over the trunk's planes, no "
           "embed_lookup, issued once in the query pass)\n");
}

/* The whole container at tensor parallelism (world 2 and 4): every trunk weight exl3i mad a8g128 (the one
 * codebook), ssm_ab fused in layers 0 and 2, the head exl3i R16 (serves whole, whole-vocab logits), the token table an
 * exl3i trellis (whole on every rank, no all-reduce after the embed). The trunk's record / suh / svh are the rank's
 * extents (the loader cuts them), one all_reduce on every row-parallel output, the step lint clean at both ranks. */
void test_tp() {
    printf("[TEST] Checking the whole exl3i container at TP2 and TP4 (sharded trunk, whole R16 head, whole trellis embed)...\n");
    static std::vector<V3Weight> v3;
    static std::vector<std::string> names, keys_s;
    static std::vector<const char*> keys, vals;
    auto reset = [&]() {
        v3.clear(); names.clear(); keys_s.clear(); keys.clear(); vals.clear();
        names.reserve(64);
    };
    struct Sh { const char* fmt; int64_t rows, cols; int stacked; };
    const Sh gdn[] = { { "blk.%d.ssm_qkv.weight", 10240, 5120, 0 }, { "blk.%d.ssm_z.weight", 6144, 5120, 0 },
                       { "blk.%d.ssm_out.weight", 5120, 6144, 0 } };
    const Sh ab = { "blk.%d.ssm_ab.weight", 128, 5120, 0 };
    const Sh att[] = { { "blk.%d.attn_qg.weight", 12288, 5120, 0 }, { "blk.%d.attn_k.weight", 1024, 5120, 0 },
                       { "blk.%d.attn_v.weight", 1024, 5120, 0 }, { "blk.%d.attn_output.weight", 5120, 6144, 0 } };
    const Sh mlp[] = { { "blk.%d.ffn_gate_up.weight", 34816, 5120, 1 }, { "blk.%d.ffn_down.weight", 5120, 17408, 0 } };
    for (int world : { 2, 4 }) {
        reset();
        std::vector<std::pair<const Sh*, int>> all;
        for (int l = 0; l < 4; ++l) {
            if (l == 3) for (const Sh& x : att) all.push_back({ &x, l });
            else for (const Sh& x : gdn) all.push_back({ &x, l });
            if (l == 0 || l == 2) all.push_back({ &ab, l });
            for (const Sh& x : mlp) all.push_back({ &x, l });
        }
        for (auto& e : all) {
            char nm[96];
            std::snprintf(nm, sizeof nm, e.first->fmt, e.second);
            names.push_back(nm);
            keys_s.push_back(std::string("exl3.act.") + nm);
        }
        for (size_t i = 0; i < all.size(); ++i) {
            const Sh& x = *all[i].first;
            v3.push_back({ names[i].c_str(), 64, EXL3I_CB_MAD, x.stacked, x.rows, x.cols });
            keys.push_back(keys_s[i].c_str());
            vals.push_back("a8g128");
        }
        v3.push_back({ "output.weight", 56, EXL3I_CB_MAD, 0, 248320, 5120, 1 });   /* the R16 head, K3.5 */
        v3.push_back({ "token_embd.weight", 30, EXL3I_CB_MAD, 0, 248320, 5120, 0 });
        V3_WEIGHTS = v3.data(); N_V3 = (int)v3.size();

        RadModelMeta m = make_qwen35_meta(4);
        m.n_kv = (int)keys.size(); m.kv_key = keys.data(); m.kv_val = vals.data();

        RadBuildCtx ctx = make_ctx(0, world, 4096, 262144);
        RadBuilder b{};
        assert(qwen35_exl3::declare(&b, &m, &ctx) == RAD_OK);

        auto recw = [&](const char* n) -> const RecWeight* {
            for (const auto& w : b.weights) if (w.name == n) return &w;
            return nullptr;
        };
        const int64_t W = world;
        {   /* the rank's extents: row shards cut the rows (record / 16, suh whole, svh with the rows), column shards
             * the K (record whole, suh with the K, svh whole) */
            const struct { const char* w; int64_t rr, rc; bool col; } want[] = {
                { "blk.0.ssm_qkv.weight", 10240, 5120, false },
                { "blk.0.ssm_z.weight",   6144, 5120, false },
                { "blk.0.ssm_out.weight", 5120, 6144, true },
                { "blk.3.attn_qg.weight", 12288, 5120, false },
                { "blk.3.attn_k.weight",  1024, 5120, false },
                { "blk.0.ffn_gate_up.weight", 34816, 5120, false },
                { "blk.0.ffn_down.weight", 5120, 17408, true },
            };
            for (const auto& x : want) {
                char n[128];
                std::snprintf(n, sizeof n, "%s.record", x.w);
                const RecWeight* c = recw(n);
                const int64_t nr = x.col ? x.rr : x.rr / W;
                const int64_t nk = x.col ? x.rc / W : x.rc;
                assert(c && c->d.shard == (x.col ? RAD_SHARD_COL : RAD_SHARD_ROW) && c->d.shape[0] == nr / 16 &&
                       c->d.shape[1] == exl3i_row_words(nk, 64));
                std::snprintf(n, sizeof n, "%s.suh", x.w);
                const RecWeight* s = recw(n);
                assert(s && s->d.shard == (x.col ? RAD_SHARD_COL : RAD_SHARD_NONE) && s->d.shape[1] == nk);
                std::snprintf(n, sizeof n, "%s.svh", x.w);
                const RecWeight* v = recw(n);
                assert(v && v->d.shard == (x.col ? RAD_SHARD_NONE : RAD_SHARD_ROW) && v->d.shape[0] == nr);
            }
        }
        {   /* row parts: a rank owns a slice of each stacked projection (the record by 16-row groups, the svh rows) --
             * without the parts the loader cuts one contiguous slice and the rank's k / v rows are the q rows' */
            char n[128];
            std::snprintf(n, sizeof n, "blk.0.ssm_qkv.weight.record");
            const RecWeight* c = recw(n);
            assert(c && c->d.n_row_parts == 3 && c->d.row_parts[0] == 2048 / W / 16 && c->d.row_parts[1] == 2048 / W / 16 &&
                   c->d.row_parts[2] == 6144 / W / 16);
            std::snprintf(n, sizeof n, "blk.0.ssm_qkv.weight.svh");
            const RecWeight* v = recw(n);
            assert(v && v->d.n_row_parts == 3 && v->d.row_parts[0] == 2048 / W && v->d.row_parts[1] == 2048 / W &&
                   v->d.row_parts[2] == 6144 / W);
            const int64_t ff = 17408 / W;
            std::snprintf(n, sizeof n, "blk.0.ffn_gate_up.weight.record");
            const RecWeight* gu = recw(n);
            assert(gu && gu->d.n_row_parts == 2 && gu->d.row_parts[0] == ff / 16 && gu->d.row_parts[1] == ff / 16);
        }
        {   /* ssm_ab whole on every rank; the head R16 whole with the whole-vocab logits; the trellis table whole */
            for (int l : { 0, 2 }) {
                char n[128];
                std::snprintf(n, sizeof n, "blk.%d.ssm_ab.weight.record", l);
                const RecWeight* c = recw(n);
                assert(c && c->d.shard == RAD_SHARD_NONE && c->d.shape[0] == 128 / 16 &&
                       c->d.shape[1] == exl3i_row_words(5120, 64));
            }
            const RecWeight* h = recw("output.weight.record");
            assert(h && h->d.shard == RAD_SHARD_NONE && h->d.shape[0] == 248320 / 16 &&
                   h->d.shape[1] == exl3i_row_words(5120, 56));
            const RecWeight* e = recw("token_embd.record");
            assert(e && e->d.shard == RAD_SHARD_NONE && e->d.shape[0] == 248320 / 16 &&
                   e->d.shape[1] == exl3i_row_words(5120, 30));
            const RecBuf* lg = nullptr;
            for (const auto& bf : b.bufs) if (bf.name == "logits") lg = &bf;
            assert(lg && lg->d.shape[1] == 248320 && lg->d.dtype == RAD_F32);
            int n_ar = 0;
            for (const auto& o : b.ops) if (o.op == "all_reduce" && o.resolved) ++n_ar;
            assert(n_ar == 8);   /* 3 GDN out + 1 attention o + 4 MLP down; the embed (whole table) reduces nothing */
        }
        for (int rank : { 0, (int)W - 1 }) {
            RadBuildCtx ctxr = make_ctx(rank, world, 4096, 262144);
            RadBuilder br{};
            assert(qwen35_exl3::declare(&br, &m, &ctxr) == RAD_OK);
            RadCtx c{};
            static std::vector<int32_t> toks;
            toks.assign(16, 0);
            static int32_t out1[1] = { 0 };
            static int32_t cu[3] = { 0, 16, 16 };
            c.rank = rank;
            RadBatch batch{};
            batch.n_tok = 16; batch.n_seq = 1; batch.n_out = 1;
            batch.token_ids = toks.data(); batch.out_ids = out1; batch.cu_seqlens = cu;
            batch.phase = RAD_PHASE_PREFILL;
            qwen35_exl3::step(&c, &batch);
            const auto lint = lint_step(br, { &c });
            assert(report_lint(lint) == 0);
        }
        printf("  -> OK (world %d: the trunk's planes at the rank's extents, ssm_ab / R16 head / trellis table whole, "
               "8 all_reduces, lint clean at both edge ranks)\n", world);
    }
    V3_WEIGHTS = nullptr; N_V3 = 0;
}

int main() {
    /* the decode band's xb sites (AQ version 4) are on for mad consumers; the tests below were written for the graph without
     * them (before the trunk was all mad), so they run with the aq_xb_off pin. test_aq's mad-consumer block turns it on. */
    qwen35_exl3::exl3_test_pins().aq_xb_off = true;
    test_64_layers_declare_and_blocks();
    test_step_execution_and_ordering();
    test_act_modes();
    test_exl3i();
    test_aq();
    test_fuse();
    test_ab();
    test_tp();
    test_media_and_dflash2();
    printf("ALL TESTS PASSED.\n");
    return 0;
}
