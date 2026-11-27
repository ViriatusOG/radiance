/* lora_test.cpp -- core/lora: sites from the declared Program, a PEFT adapter packed per rank,
 * declare's refusals, and the adapter slots' bookkeeping (slots.h).
 *
 * Host only, and it never reaches the device layer: no kernel plugin is loaded, because a Registry
 * that loads one brackets the dlopen with the device backend's code mark, which in a HIP build
 * starts the AQL backend. The Programs are made by the core Builder over an empty Registry, driven
 * by the arch/common code that declares the weights: LinearQ::declare (rad_linear.h) with the
 * arguments the blocks pass (rad_block_attn_q.h, rad_block_mlp_q.h, rad_block_attn_gated_q.h,
 * rad_block_gdn_q.h), over a Geom from geom_from(), and the delta net's uneven head split as
 * qwen4exp_fp8.cpp computes it (remainder_place). With no schema in the Registry the GEMM ops do
 * not resolve, so each is appended to the Program as decl_op would record it (op name, parameters,
 * weights, schema operand positions); everything a site is read from is what the builder records.
 *
 * The dense reference is the test's own: the full delta scale * B @ A of every module, placed by
 * model semantics (which heads and ff columns a rank owns), not by anything lora.cpp computes.
 */
#include "rad_test.h"

#include "build/rad_build.h"
#include "lora/declare.h"
#include "lora/lora.h"
#include "lora/slots.h"
#include "plugin/registry.h"
#include "rad_plugin.h"

#include <arch/rad_arch.h>
#include <arch/rad_linear.h>

#include "util/json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

using namespace rad;
namespace AR = rad::arch;
using ojson = Json;

namespace {

Registry& empty_registry() {
    static Registry r;   /* no plugin is ever loaded into it */
    return r;
}

/* ------------------------------------------------------------------ temp dirs */
/* Under the harness's directory (radtest::tmpdir), which it removes when the binary passes. */
std::string new_dir(const std::string& name) {
    const std::string d = radtest::tmp_path(name);
    mkdir(d.c_str(), 0755);
    return d;
}

/* ------------------------------------------------------------------ deterministic values */
float val(uint32_t seed, uint64_t i) {
    uint64_t x = (uint64_t)seed * 0x9E3779B97F4A7C15ull + i * 0xBF58476D1CE4E5B9ull + 1;
    x ^= x >> 31; x *= 0x94D049BB133111EBull; x ^= x >> 29;
    return (float)((double)(x >> 11) / (double)(1ull << 53) - 0.5) * 0.5f;
}

/* What the stored element is, read back the way any reader would, and rounded to bf16 RNE -- the
 * value a packed slot can hold. */
float stored_then_bf16(const std::string& dt, float f) {
    float v = f;
    if (dt == "F16") v = rad_f16_to_f32(rad_f32_to_f16(f));
    else if (dt == "BF16") v = rad_bf16_to_f32(rad_f32_to_bf16(f));
    return rad_bf16_to_f32(rad_f32_to_bf16(v));
}

void put(std::vector<uint8_t>* b, const std::string& dt, float f) {
    if (dt == "F32") {
        uint8_t x[4]; std::memcpy(x, &f, 4); b->insert(b->end(), x, x + 4);
    } else {
        const uint16_t h = dt == "F16" ? rad_f32_to_f16(f) : rad_f32_to_bf16(f);
        uint8_t x[2]; std::memcpy(x, &h, 2); b->insert(b->end(), x, x + 2);
    }
}

/* ------------------------------------------------------------------ a safetensors writer */
struct TW {
    std::string name, dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};
void write_st(const std::string& path, const std::vector<TW>& ts) {
    ojson h = ojson::object();
    uint64_t off = 0;
    for (const TW& t : ts) {
        h[t.name] = { { "dtype", t.dtype }, { "shape", t.shape },
                      { "data_offsets", { off, off + t.bytes.size() } } };
        off += t.bytes.size();
    }
    h["__metadata__"] = { { "format", "pt" } };
    std::string s = h.dump();
    while (s.size() % 8) s += ' ';
    std::ofstream f(path, std::ios::binary);
    const uint64_t n = s.size();
    f.write((const char*)&n, 8);
    f.write(s.data(), (std::streamsize)s.size());
    for (const TW& t : ts) f.write((const char*)t.bytes.data(), (std::streamsize)t.bytes.size());
}

/* ------------------------------------------------------------------ an adapter */
struct ModSpec {
    std::string module;     /* as the adapter spells it, without base_model.model. */
    int64_t r = 8, rows = 0, cols = 0;
    std::string dtA = "BF16", dtB = "BF16";
    uint32_t seed = 1;
    bool default_spelling = false;   /* .lora_A.default.weight */
    float scale = 0.f;               /* what the config says it is (checked) */
};

struct Adapter {
    std::string dir;
    std::vector<ModSpec> mods;
};

void write_adapter(Adapter* ad, const std::string& name, const ojson& cfg) {
    ad->dir = new_dir(name);
    std::ofstream(ad->dir + "/adapter_config.json") << cfg.dump(1);
    std::vector<TW> ts;
    for (const ModSpec& m : ad->mods) {
        const std::string sfx = m.default_spelling ? ".default.weight" : ".weight";
        TW a{ "base_model.model." + m.module + ".lora_A" + sfx, m.dtA, { m.r, m.cols }, {} };
        TW b{ "base_model.model." + m.module + ".lora_B" + sfx, m.dtB, { m.rows, m.r }, {} };
        for (int64_t i = 0; i < m.r * m.cols; ++i) put(&a.bytes, m.dtA, val(m.seed, (uint64_t)i));
        for (int64_t i = 0; i < m.rows * m.r; ++i) put(&b.bytes, m.dtB, val(m.seed + 7777, (uint64_t)i));
        ts.push_back(std::move(a));
        ts.push_back(std::move(b));
    }
    write_st(ad->dir + "/adapter_model.safetensors", ts);
}

ojson base_cfg(int64_t r, double alpha, std::vector<std::string> targets) {
    ojson c = ojson::object();
    c["peft_type"] = "LORA";
    c["base_model_name_or_path"] = "test/base-model";
    c["r"] = r;
    c["lora_alpha"] = alpha;
    c["target_modules"] = targets;
    c["bias"] = "none";
    c["use_rslora"] = false;
    c["use_dora"] = false;
    c["fan_in_fan_out"] = false;
    c["modules_to_save"] = nullptr;
    return c;
}

/* The full dense delta of one module, from the values as a slot holds them. [rows, cols] */
std::vector<double> module_delta(const ModSpec& m) {
    std::vector<double> d((size_t)(m.rows * m.cols), 0.0);
    for (int64_t i = 0; i < m.rows; ++i)
        for (int64_t j = 0; j < m.r; ++j) {
            const double b = stored_then_bf16(m.dtB, val(m.seed + 7777, (uint64_t)(i * m.r + j)));
            for (int64_t k = 0; k < m.cols; ++k)
                d[(size_t)(i * m.cols + k)] +=
                    b * stored_then_bf16(m.dtA, val(m.seed, (uint64_t)(j * m.cols + k)));
        }
    for (double& x : d) x *= (double)m.scale;
    return d;
}

/* ------------------------------------------------------------------ the declared programs */
/* Model semantics of one declared weight on one rank: the full weight's rows this rank's y holds,
 * in y order, and the full K columns its x holds. `sources` are the checkpoint modules in concat
 * order with their full row counts. Absent on this rank: `present` false. */
struct Truth {
    bool present = false;
    int shard = RAD_SHARD_ROW;
    int64_t N_full = 0, K_full = 0;
    std::vector<std::pair<std::string, int64_t>> sources;
    std::vector<int64_t> rows, cols;
};
void add_range(std::vector<int64_t>* v, int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; ++i) v->push_back(i);
}

struct RankProg {
    Program prog;
    std::map<std::string, Truth> truth;   /* by declared weight name */
};

OpInfo gemm_op(const char* name, int64_t N, int64_t K, const char* dtype, int64_t lo, int64_t hi,
               std::vector<rad_weight> w, std::vector<int32_t> opd, bool grouped) {
    OpInfo o;
    o.op = name;
    o.base.set_i("N", N);
    o.base.set_i("K", K);
    if (grouped) o.base.set_i("group", AR::RAD_SCALE_GROUP);
    o.base.set_s("dtype", dtype);
    o.ranged_key = "M";
    o.range_lo = lo;
    o.range_hi = hi;
    o.weights = std::move(w);
    o.weight_opd = std::move(opd);
    return o;
}
uint32_t push_op(Builder& b, OpInfo o) {
    o.index = (int32_t)b.program().ops.size();
    b.program().ops.push_back(std::move(o));
    return (uint32_t)b.program().ops.size() - 1;
}

/* LinearQ::declare as a block calls it, then the GEMM it would have declared over the two
 * views: gemm_nt_q fp8a8 (`b`, `b_scale` at schema positions 2, 3). `m1` adds the decode matvec
 * LinearQ declares beside it when fp8_matvec_rows() > 0 (the same two views, a lower band). */
int linear(Builder& b, AR::Names& nm, const AR::Geom& g, const char* base, int64_t N, int64_t K,
           int shard, int layer, std::initializer_list<const char*> src,
           std::initializer_list<int64_t> parts = {}, int64_t span_lo = 0, int64_t span_hi = 0,
           AR::LinearQ* keep = nullptr, bool m1 = false) {
    AR::LinearQ lin{};
    AR::ActQ act{};
    const int st = lin.declare(&b, nm, g, base, N, K, shard, AR::grp_layer(layer), src, act, 0, 0,
                               parts, span_lo, span_hi);
    if (st < 0 || !lin.w || !lin.ws) return st < 0 ? st : RAD_E_INVAL;
    if (m1)
        push_op(b, gemm_op("gemm_nt_q", N, K, "fp8a16", 1, 16, { lin.w, lin.ws }, { 2, 3 }, true));
    push_op(b, gemm_op("gemm_nt_q", N, K, "fp8a8", m1 ? 17 : 1, g.max_tok, { lin.w, lin.ws },
                       { 2, 3 }, true));
    if (keep) *keep = lin;
    return RAD_OK;
}

RadBuildCtx ctx_for(int rank, int world) {
    RadBuildCtx c{};
    c.rank = rank;
    c.world_size = world;
    c.max_tok = 256;
    c.max_seqs = 8;
    c.max_ctx = 4096;
    c.scope = "";
    return c;
}

/* ---- LLaMA fp8 (MiniCPM5's family; arch/llama_fp8): fused [q|k|v], o, fused [gate|up], down. */
RadModelMeta llama_meta() {
    RadModelMeta m{};
    m.arch_id = "llama"; m.name = "lora-test-llama"; m.quant = "fp8_e4m3";
    m.n_layers = 2; m.n_embd = 256; m.n_head = 12; m.n_head_kv = 12; m.head_dim = 128;
    m.n_ff = 1536; m.n_vocab = 4096; m.n_ctx_train = 4096;
    m.rms_eps = 1e-5f; m.rope_theta = 5e5f; m.rope_scale = 1.0f;
    return m;
}

int build_llama(int rank, int world, RankProg* out) {
    const RadModelMeta meta = llama_meta();
    const RadBuildCtx ctx = ctx_for(rank, world);
    AR::Geom g{};
    RAD_ARCH_TRY(AR::geom_from(g, &meta, &ctx, "bf16", RAD_BF16));   /* llama_fp8.cpp */
    Builder b(empty_registry(), meta, ctx);
    AR::Names nm("");
    const char* pre = "model.layers";                                  /* llama_fp8.cpp */
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();
    const int64_t q_all = g.n_head_all * g.head_dim, kv_all = g.n_head_kv_all * g.head_dim;
    for (int l = 0; l < (int)meta.n_layers; ++l) {
        auto ck = [&](const char* leaf) { return nm.ckpt("%s.%d.%s.weight", pre, l, leaf); };
        /* rad_block_attn_q.h */
        RAD_ARCH_TRY(linear(b, nm, g, nm.f("blk.%d.attn_qkv", l), q_dim + 2 * kv_dim, g.n_embd,
                            RAD_SHARD_ROW, l, { ck("self_attn.q_proj"), ck("self_attn.k_proj"),
                                                ck("self_attn.v_proj") },
                            { q_dim, kv_dim, kv_dim }));
        /* rad_block_attn_q.h */
        RAD_ARCH_TRY(linear(b, nm, g, nm.f("blk.%d.attn_output", l), g.n_embd, q_dim,
                            RAD_SHARD_COL, l, { ck("self_attn.o_proj") }));
        /* rad_block_mlp_q.h */
        AR::LinearQ gu{};
        RAD_ARCH_TRY(linear(b, nm, g, nm.f("blk.%d.ffn_gate_up", l), 2 * g.n_ff, g.n_embd,
                            RAD_SHARD_ROW, l, { ck("mlp.gate_proj"), ck("mlp.up_proj") },
                            { g.n_ff, g.n_ff }, 0, 0, &gu));
        /* rad_block_mlp_q.h */
        RAD_ARCH_TRY(linear(b, nm, g, nm.f("blk.%d.ffn_down", l), g.n_embd, g.n_ff, RAD_SHARD_COL,
                            l, { ck("mlp.down_proj") }));

        const std::string L = std::to_string(l), P = std::string(pre) + "." + L + ".";
        Truth t;
        t.present = true; t.shard = RAD_SHARD_ROW; t.N_full = q_all + 2 * kv_all; t.K_full = g.n_embd;
        t.sources = { { P + "self_attn.q_proj", q_all }, { P + "self_attn.k_proj", kv_all },
                      { P + "self_attn.v_proj", kv_all } };
        add_range(&t.rows, rank * q_dim, (rank + 1) * q_dim);
        add_range(&t.rows, q_all + rank * kv_dim, q_all + (rank + 1) * kv_dim);
        add_range(&t.rows, q_all + kv_all + rank * kv_dim, q_all + kv_all + (rank + 1) * kv_dim);
        add_range(&t.cols, 0, g.n_embd);
        out->truth["blk." + L + ".attn_qkv.weight"] = t;

        Truth o;
        o.present = true; o.shard = RAD_SHARD_COL; o.N_full = g.n_embd; o.K_full = q_all;
        o.sources = { { P + "self_attn.o_proj", g.n_embd } };
        add_range(&o.rows, 0, g.n_embd);
        add_range(&o.cols, rank * q_dim, (rank + 1) * q_dim);
        out->truth["blk." + L + ".attn_output.weight"] = o;

        Truth f;
        f.present = true; f.shard = RAD_SHARD_ROW; f.N_full = 2 * g.n_ff_all; f.K_full = g.n_embd;
        f.sources = { { P + "mlp.gate_proj", g.n_ff_all }, { P + "mlp.up_proj", g.n_ff_all } };
        add_range(&f.rows, rank * g.n_ff, (rank + 1) * g.n_ff);
        add_range(&f.rows, g.n_ff_all + rank * g.n_ff, g.n_ff_all + (rank + 1) * g.n_ff);
        add_range(&f.cols, 0, g.n_embd);
        out->truth["blk." + L + ".ffn_gate_up.weight"] = f;

        Truth d;
        d.present = true; d.shard = RAD_SHARD_COL; d.N_full = g.n_embd; d.K_full = g.n_ff_all;
        d.sources = { { P + "mlp.down_proj", g.n_embd } };
        add_range(&d.rows, 0, g.n_embd);
        add_range(&d.cols, rank * g.n_ff, (rank + 1) * g.n_ff);
        out->truth["blk." + L + ".ffn_down.weight"] = d;
    }
    out->prog = std::move(b.program());
    return RAD_OK;
}

/* ---- The hybrid trunk (Qwen3.5 / Qwen4-Exp family, VLM-spelled checkpoint): layer 0 gated delta
 * net, layer 1 gated full attention, a dense MLP in both. Two KV heads, five K heads: at three ranks
 * attention runs on two and rank 2 is attention-zero, and the delta net splits 2/2/1 with the extra
 * head on rank 2 first (remainder_place); at four ranks the KV heads replicate. */
const int64_t kNK = 5, kNV = 10, kHK = 128, kHV = 128;
RadModelMeta hybrid_meta() {
    RadModelMeta m{};
    m.arch_id = "qwen4exp"; m.name = "lora-test-hybrid"; m.quant = "fp8_e4m3";
    m.n_layers = 2; m.n_embd = 256; m.n_head = 8; m.n_head_kv = 2; m.head_dim = 128;
    m.n_ff = 1536; m.n_vocab = 4096; m.n_ctx_train = 4096;
    m.rms_eps = 1e-6f; m.rope_theta = 1e7f; m.rope_scale = 1.0f;
    return m;
}

struct GdnSplit { int64_t nk, kf, nv, vf; bool uneven; };
/* qwen4exp_fp8.cpp, verbatim in substance. */
GdnSplit gdn_split(const AR::Geom& g) {
    GdnSplit s{ kNK, 0, kNV, 0, false };
    if (g.world > 1 && (kNK % g.world != 0 || kNV % g.world != 0)) {
        const int64_t ratio = kNV / kNK, base = kNK / g.world, rem = kNK % g.world;
        const int64_t pl = AR::remainder_place(g, g.rank, 0);
        s.nk = base + (pl < rem ? 1 : 0);
        s.kf = pl * base + (pl < rem ? pl : rem);
        s.nv = ratio * s.nk;
        s.vf = ratio * s.kf;
        s.uneven = true;
    } else {
        s.nk = kNK / g.world; s.nv = kNV / g.world;
        s.kf = g.rank * s.nk; s.vf = g.rank * s.nv;
    }
    return s;
}

int build_hybrid(int rank, int world, RankProg* out, bool m1_on_inz = true) {
    const RadModelMeta meta = hybrid_meta();
    const RadBuildCtx ctx = ctx_for(rank, world);
    AR::Geom g{};
    /* qwen4exp_fp8.cpp */
    RAD_ARCH_TRY(AR::geom_from(g, &meta, &ctx, "bf16", RAD_BF16, true, true));
    Builder b(empty_registry(), meta, ctx);
    AR::Names nm("");
    const char* pre = "model.language_model.layers";
    const std::string VP = std::string(pre);
    const GdnSplit sp = gdn_split(g);
    const int64_t k_dim = sp.nk * kHK, v_dim = sp.nv * kHV, k_all = kNK * kHK, v_all = kNV * kHV;

    /* ---- layer 0: the gated delta net (rad_block_gdn_q.h) */
    {
        const int l = 0;
        auto ck = [&](const char* leaf) { return nm.ckpt("%s.%d.%s.weight", pre, l, leaf); };
        RAD_ARCH_TRY(AR::map_concat(&b, nm.f("blk.%d.ssm_ab.weight", l),
                                   { ck("linear_attn.in_proj_a"), ck("linear_attn.in_proj_b") }, 0));
        const rad_weight w_ab = AR::decl_w(&b, nm.f("blk.%d.ssm_ab.weight", l), RAD_BF16,
                                          { 2 * sp.nv, g.n_embd }, RAD_ACCESS_PER_TOKEN,
                                          RAD_SHARD_ROW, AR::grp_layer(l), 0, { sp.nv, sp.nv });
        if (!w_ab) return RAD_E_INVAL;
        AR::LinearQ in{};
        AR::ActQ act{};
        RAD_ARCH_TRY(in.declare(&b, nm, g, nm.f("blk.%d.ssm_inz", l), 2 * k_dim + 2 * v_dim,
                                g.n_embd, RAD_SHARD_ROW, AR::grp_layer(l),
                                { ck("linear_attn.in_proj_qkv"), ck("linear_attn.in_proj_z") }, act,
                                0, 0, { k_dim, k_dim, v_dim, v_dim }, 0, 0));
        if (sp.uneven) {
            const int64_t kl = sp.kf * kHK, vl = sp.vf * kHV;
            const int64_t lo[4] = { kl, kl, vl, vl };
            const int64_t hi[4] = { kl + k_dim, kl + k_dim, vl + v_dim, vl + v_dim };
            const int64_t tot[4] = { k_all, k_all, v_all, v_all };
            RAD_ARCH_TRY(in.span_parts(&b, lo, hi, tot, 4));
            const int64_t alo[2] = { sp.vf, sp.vf }, ahi[2] = { sp.vf + sp.nv, sp.vf + sp.nv };
            const int64_t atot[2] = { kNV, kNV };
            RAD_ARCH_TRY(rad_weight_shard_span_parts(&b, w_ab, alo, ahi, atot, 2));
        }
        if (m1_on_inz)
            push_op(b, gemm_op("gemm_nt_q", in.N, in.K, "fp8a16", 1, 16, { in.w, in.ws }, { 2, 3 }, true));
        push_op(b, gemm_op("gemm_nt_q", in.N, in.K, "fp8a8", m1_on_inz ? 17 : 1, g.max_tok,
                           { in.w, in.ws }, { 2, 3 }, true));
        push_op(b, gemm_op("gemm_nt", 2 * sp.nv, g.n_embd, "bf16", 1, g.max_tok, { w_ab }, { 1 }, false));
        RAD_ARCH_TRY(linear(b, nm, g, nm.f("blk.%d.ssm_out", l), g.n_embd, v_dim, RAD_SHARD_COL, l,
                            { ck("linear_attn.out_proj") }, {}, sp.uneven ? sp.vf * kHV : 0,
                            sp.uneven ? sp.vf * kHV + v_dim : 0));

        const std::string P = VP + ".0.";
        Truth t;
        t.present = true; t.shard = RAD_SHARD_ROW; t.N_full = 2 * k_all + 2 * v_all; t.K_full = g.n_embd;
        t.sources = { { P + "linear_attn.in_proj_qkv", 2 * k_all + v_all },
                      { P + "linear_attn.in_proj_z", v_all } };
        add_range(&t.rows, sp.kf * kHK, sp.kf * kHK + k_dim);
        add_range(&t.rows, k_all + sp.kf * kHK, k_all + sp.kf * kHK + k_dim);
        add_range(&t.rows, 2 * k_all + sp.vf * kHV, 2 * k_all + sp.vf * kHV + v_dim);
        add_range(&t.rows, 2 * k_all + v_all + sp.vf * kHV, 2 * k_all + v_all + sp.vf * kHV + v_dim);
        add_range(&t.cols, 0, g.n_embd);
        out->truth["blk.0.ssm_inz.weight"] = t;
        Truth ab;
        ab.present = true; ab.shard = RAD_SHARD_ROW; ab.N_full = 2 * kNV; ab.K_full = g.n_embd;
        ab.sources = { { P + "linear_attn.in_proj_a", kNV }, { P + "linear_attn.in_proj_b", kNV } };
        add_range(&ab.rows, sp.vf, sp.vf + sp.nv);
        add_range(&ab.rows, kNV + sp.vf, kNV + sp.vf + sp.nv);
        add_range(&ab.cols, 0, g.n_embd);
        out->truth["blk.0.ssm_ab.weight"] = ab;
        Truth so;
        so.present = true; so.shard = RAD_SHARD_COL; so.N_full = g.n_embd; so.K_full = v_all;
        so.sources = { { P + "linear_attn.out_proj", g.n_embd } };
        add_range(&so.rows, 0, g.n_embd);
        add_range(&so.cols, sp.vf * kHV, sp.vf * kHV + v_dim);
        out->truth["blk.0.ssm_out.weight"] = so;
    }

    /* ---- layer 1: gated attention (rad_block_attn_gated_q.h) */
    {
        const int l = 1;
        auto ck = [&](const char* leaf) { return nm.ckpt("%s.%d.%s.weight", pre, l, leaf); };
        const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim(), qg_dim = 2 * q_dim;
        const bool a_spanned = g.attn_world != g.world;
        const bool kv_spanned = g.world > g.n_head_kv_all || a_spanned;
        const int64_t kv_lo = kv_spanned ? g.kv_off : 0, kv_hi = kv_spanned ? g.kv_off + kv_dim : 0;
        const int64_t qg_lo = a_spanned ? g.attn_rank * qg_dim : 0;
        const int64_t qg_hi = a_spanned ? (g.attn_rank + 1) * qg_dim : 0;
        const int64_t o_lo = a_spanned ? g.attn_rank * q_dim : 0;
        const int64_t o_hi = a_spanned ? (g.attn_rank + 1) * q_dim : 0;
        const std::string P = VP + ".1.";
        const int64_t q_all = g.n_head_all * g.head_dim, kv_all = g.n_head_kv_all * g.head_dim;
        if (!g.attn_zero) {
            RAD_ARCH_TRY(linear(b, nm, g, nm.f("blk.%d.attn_qg", l), qg_dim, g.n_embd, RAD_SHARD_ROW,
                                l, { ck("self_attn.q_proj") }, {}, qg_lo, qg_hi));
            RAD_ARCH_TRY(linear(b, nm, g, nm.f("blk.%d.attn_k", l), kv_dim, g.n_embd, RAD_SHARD_ROW,
                                l, { ck("self_attn.k_proj") }, {}, kv_lo, kv_hi));
            RAD_ARCH_TRY(linear(b, nm, g, nm.f("blk.%d.attn_v", l), kv_dim, g.n_embd, RAD_SHARD_ROW,
                                l, { ck("self_attn.v_proj") }, {}, kv_lo, kv_hi));
            RAD_ARCH_TRY(linear(b, nm, g, nm.f("blk.%d.attn_output", l), g.n_embd, q_dim,
                                RAD_SHARD_COL, l, { ck("self_attn.o_proj") }, {}, o_lo, o_hi));
        }
        /* Model semantics: this rank's query heads, and the KV head(s) those heads read. */
        const int ar = g.attn_rank;
        const int64_t qh0 = ar * g.n_head, group = g.n_head_all / g.n_head_kv_all;
        const int64_t kvh0 = qh0 / group;
        Truth qg;
        qg.present = !g.attn_zero; qg.N_full = 2 * q_all; qg.K_full = g.n_embd;
        qg.sources = { { P + "self_attn.q_proj", 2 * q_all } };
        add_range(&qg.rows, qh0 * 2 * g.head_dim, (qh0 + g.n_head) * 2 * g.head_dim);
        add_range(&qg.cols, 0, g.n_embd);
        out->truth["blk.1.attn_qg.weight"] = qg;
        for (const char* kv : { "k", "v" }) {
            Truth t;
            t.present = !g.attn_zero; t.N_full = kv_all; t.K_full = g.n_embd;
            t.sources = { { P + "self_attn." + kv + "_proj", kv_all } };
            add_range(&t.rows, kvh0 * g.head_dim, (kvh0 + g.n_head_kv) * g.head_dim);
            add_range(&t.cols, 0, g.n_embd);
            out->truth[std::string("blk.1.attn_") + kv + ".weight"] = t;
        }
        Truth o;
        o.present = !g.attn_zero; o.shard = RAD_SHARD_COL; o.N_full = g.n_embd; o.K_full = q_all;
        o.sources = { { P + "self_attn.o_proj", g.n_embd } };
        add_range(&o.rows, 0, g.n_embd);
        add_range(&o.cols, qh0 * g.head_dim, (qh0 + g.n_head) * g.head_dim);
        out->truth["blk.1.attn_output.weight"] = o;
    }

    /* ---- the dense MLP of both layers (rad_block_mlp_q.h) */
    for (int l = 0; l < 2; ++l) {
        auto ck = [&](const char* leaf) { return nm.ckpt("%s.%d.%s.weight", pre, l, leaf); };
        RAD_ARCH_TRY(linear(b, nm, g, nm.f("blk.%d.ffn_gate_up", l), 2 * g.n_ff, g.n_embd,
                            RAD_SHARD_ROW, l, { ck("mlp.gate_proj"), ck("mlp.up_proj") },
                            { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(linear(b, nm, g, nm.f("blk.%d.ffn_down", l), g.n_embd, g.n_ff, RAD_SHARD_COL,
                            l, { ck("mlp.down_proj") }));
        const std::string L = std::to_string(l), P = VP + "." + L + ".";
        Truth f;
        f.present = true; f.N_full = 2 * g.n_ff_all; f.K_full = g.n_embd;
        f.sources = { { P + "mlp.gate_proj", g.n_ff_all }, { P + "mlp.up_proj", g.n_ff_all } };
        add_range(&f.rows, rank * g.n_ff, (rank + 1) * g.n_ff);
        add_range(&f.rows, g.n_ff_all + rank * g.n_ff, g.n_ff_all + (rank + 1) * g.n_ff);
        add_range(&f.cols, 0, g.n_embd);
        out->truth["blk." + L + ".ffn_gate_up.weight"] = f;
        Truth d;
        d.present = true; d.shard = RAD_SHARD_COL; d.N_full = g.n_embd; d.K_full = g.n_ff_all;
        d.sources = { { P + "mlp.down_proj", g.n_embd } };
        add_range(&d.rows, 0, g.n_embd);
        add_range(&d.cols, rank * g.n_ff, (rank + 1) * g.n_ff);
        out->truth["blk." + L + ".ffn_down.weight"] = d;
    }
    out->prog = std::move(b.program());
    return RAD_OK;
}

/* ------------------------------------------------------------------ the dense comparison */
struct Full { int64_t N = 0, K = 0; std::vector<double> d; int shard = RAD_SHARD_ROW; };

/* The full delta of every declared weight the adapter touches, keyed by weight name, placing each
 * module at its source's rows in concat order. `alias` maps the adapter's spelling to the base's. */
std::map<std::string, Full> full_deltas(const std::map<std::string, Truth>& truth,
                                        const Adapter& ad, const std::string& from,
                                        const std::string& to) {
    std::map<std::string, Full> out;
    for (const auto& kv : truth) {
        const Truth& t = kv.second;
        Full f;
        f.N = t.N_full; f.K = t.K_full; f.shard = t.shard;
        f.d.assign((size_t)(f.N * f.K), 0.0);
        int64_t off = 0;
        bool any = false;
        for (const auto& s : t.sources) {
            for (const ModSpec& m : ad.mods) {
                std::string base = m.module;
                if (!from.empty() && base.compare(0, from.size(), from) == 0)
                    base = to + base.substr(from.size());
                if (base != s.first) continue;
                CHECK_EQ(m.rows, s.second);
                CHECK_EQ(m.cols, f.K);
                if (m.rows != s.second || m.cols != f.K) continue;
                const std::vector<double> md = module_delta(m);
                for (int64_t i = 0; i < m.rows; ++i)
                    for (int64_t k = 0; k < f.K; ++k)
                        f.d[(size_t)((off + i) * f.K + k)] = md[(size_t)(i * f.K + k)];
                any = true;
            }
            off += s.second;
        }
        if (any) out[kv.first] = std::move(f);
    }
    return out;
}

/* The rank's slot product scale ⊙ (B @ A), [N_r, K_r]. */
std::vector<double> slot_delta(const lora::Site& s, const lora::PackedSite& p) {
    std::vector<double> d((size_t)(s.N_r * s.K_r), 0.0);
    for (int64_t y = 0; y < s.N_r; ++y) {
        if (p.scale[(size_t)y] == 0.0f) continue;
        for (int64_t j = 0; j < s.R; ++j) {
            const double b = rad_bf16_to_f32(p.b[(size_t)(y * s.R + j)]);
            if (b == 0.0) continue;
            for (int64_t k = 0; k < s.K_r; ++k)
                d[(size_t)(y * s.K_r + k)] += b * rad_bf16_to_f32(p.a[(size_t)(j * s.K_r + k)]);
        }
        for (int64_t k = 0; k < s.K_r; ++k) d[(size_t)(y * s.K_r + k)] *= p.scale[(size_t)y];
    }
    return d;
}

/* Every rank of one world: sites, the adapter packed, each pool's product against the reference
 * slice, the COL ranks summing to the whole delta, and the ROW ranks covering every row. */
void check_world(const char* label, int world,
                 const std::function<int(int, int, RankProg*)>& build, const Adapter& ad,
                 const lora::TargetSet& targets, int max_rank, const std::string& from,
                 const std::string& to, int* n_pools_checked) {
    std::vector<RankProg> rp((size_t)world);
    std::vector<lora::AdapterInfo> infos((size_t)world);
    std::map<std::string, Full> want;
    std::map<std::string, std::vector<double>> col_sum;
    std::map<std::string, std::vector<int>> row_seen;
    for (int r = 0; r < world; ++r) {
        REQUIRE_EQ(build(r, world, &rp[(size_t)r]), RAD_OK);
        if (r == 0) want = full_deltas(rp[0].truth, ad, from, to);
        std::vector<lora::Site> sites;
        std::vector<std::string> pools;
        std::string err;
        const int st = lora::build_sites(rp[(size_t)r].prog, targets, max_rank, &sites, &err, &pools);
        if (st != RAD_OK) fprintf(stderr, "    %s world %d rank %d: %s\n", label, world, r, err.c_str());
        REQUIRE_EQ(st, RAD_OK);
        std::vector<lora::UnsupportedUse> un;
        CHECK_EQ(lora::find_unsupported_uses(rp[(size_t)r].prog, targets, &un), 0);
        std::vector<lora::PackedSite> packed;
        const int ls = lora::load_adapter(ad.dir, sites, max_rank, &infos[(size_t)r], &packed, &err);
        if (ls != RAD_OK) fprintf(stderr, "    %s world %d rank %d: %s\n", label, world, r, err.c_str());
        REQUIRE_EQ(ls, RAD_OK);
        REQUIRE_EQ(packed.size(), pools.size());
        /* one representative site per pool; every site of a pool names it */
        std::vector<const lora::Site*> rep(pools.size(), nullptr);
        for (const lora::Site& s : sites) {
            REQUIRE(s.pool >= 0 && (size_t)s.pool < pools.size());
            CHECK_EQ(pools[(size_t)s.pool], s.weight);
            if (!rep[(size_t)s.pool]) rep[(size_t)s.pool] = &s;
        }
        /* every truth weight present on this rank whose sources are targeted has a pool */
        for (const auto& kv : rp[(size_t)r].truth) {
            bool targeted = false;
            for (const auto& src : kv.second.sources) {
                const size_t d = src.first.rfind('.');
                if (targets.count(src.first.substr(d + 1))) targeted = true;
            }
            bool pooled = false;
            for (const std::string& p : pools) if (p == kv.first) pooled = true;
            CHECK_EQ(pooled, kv.second.present && targeted);
        }
        for (size_t p = 0; p < pools.size(); ++p) {
            const lora::Site& s = *rep[p];
            const Truth& t = rp[(size_t)r].truth.at(s.weight);
            REQUIRE(t.present);
            REQUIRE_EQ((int64_t)t.rows.size(), s.N_r);
            REQUIRE_EQ((int64_t)t.cols.size(), s.K_r);
            const lora::PackedSite& pk = packed[p];
            REQUIRE_EQ((int64_t)pk.a.size(), s.R * s.K_r);
            REQUIRE_EQ((int64_t)pk.b.size(), s.N_r * s.R);
            REQUIRE_EQ((int64_t)pk.scale.size(), s.N_r);
            auto it = want.find(s.weight);
            if (it == want.end()) {   /* targeted, but this adapter has none of its modules */
                CHECK(pk.empty);
                for (float x : pk.scale) CHECK_EQ(x, 0.0f);
                continue;
            }
            const Full& f = it->second;
            const std::vector<double> d = slot_delta(s, pk);
            int bad = 0;
            for (int64_t y = 0; y < s.N_r; ++y)
                for (int64_t k = 0; k < s.K_r; ++k) {
                    const double ref = f.d[(size_t)(t.rows[(size_t)y] * f.K + t.cols[(size_t)k])];
                    if (std::fabs(d[(size_t)(y * s.K_r + k)] - ref) > 1e-6 * (1.0 + std::fabs(ref)))
                        ++bad;
                }
            if (bad) fprintf(stderr, "    %s world %d rank %d: %s has %d wrong elements\n", label,
                             world, r, s.weight.c_str(), bad);
            CHECK_EQ(bad, 0);
            ++*n_pools_checked;
            if (s.shard == RAD_SHARD_COL && world > 1) {
                std::vector<double>& cs = col_sum[s.weight];
                if (cs.empty()) cs.assign(f.d.size(), 0.0);
                for (int64_t y = 0; y < s.N_r; ++y)
                    for (int64_t k = 0; k < s.K_r; ++k)
                        cs[(size_t)(t.rows[(size_t)y] * f.K + t.cols[(size_t)k])] += d[(size_t)(y * s.K_r + k)];
            } else {
                std::vector<int>& rs = row_seen[s.weight];
                if (rs.empty()) rs.assign((size_t)f.N, 0);
                for (int64_t y = 0; y < s.N_r; ++y) rs[(size_t)t.rows[(size_t)y]] = 1;
            }
            /* zero padding: rows of A past a module's rank, columns of B outside its block */
            for (const lora::SiteSource& src : s.sources) {
                if (!src.targeted) continue;
                int64_t rk = 0;
                for (const auto& m : infos[(size_t)r].modules)
                    if (m.base == src.module) rk = m.rank;
                for (int64_t j = rk; j < max_rank; ++j)
                    for (int64_t k = 0; k < s.K_r; ++k)
                        CHECK_EQ(pk.a[(size_t)((src.slot * max_rank + j) * s.K_r + k)], (uint16_t)0);
            }
        }
    }
    for (const auto& kv : col_sum) {
        const Full& f = want.at(kv.first);
        int bad = 0;
        for (size_t i = 0; i < f.d.size(); ++i)
            if (std::fabs(kv.second[i] - f.d[i]) > 1e-6 * (1.0 + std::fabs(f.d[i]))) ++bad;
        if (bad) fprintf(stderr, "    %s world %d: the COL ranks of %s do not sum to its delta (%d)\n",
                         label, world, kv.first.c_str(), bad);
        CHECK_EQ(bad, 0);
    }
    for (const auto& kv : row_seen) {
        int missing = 0;
        for (int x : kv.second) missing += x ? 0 : 1;
        if (missing) fprintf(stderr, "    %s world %d: %d rows of %s on no rank\n", label, world,
                             missing, kv.first.c_str());
        CHECK_EQ(missing, 0);
    }
    std::string err;
    CHECK_OK(lora::check_placed(infos, &err, &rp[0].prog, &targets));
    if (!err.empty()) fprintf(stderr, "    %s world %d: %s\n", label, world, err.c_str());
}

/* ------------------------------------------------------------------ the adapters */
Adapter llama_adapter() {
    Adapter ad;
    uint32_t seed = 10;
    /* k_proj is not targeted: the fused qkv's middle source stays base-only. */
    for (int l = 0; l < 2; ++l) {
        const std::string P = "model.layers." + std::to_string(l) + ".";
        ModSpec q{ P + "self_attn.q_proj", 16, 1536, 256, "F32", "BF16", seed++, false, 1.0f };
        ModSpec v{ P + "self_attn.v_proj", 8, 1536, 256, "BF16", "F16", seed++, false, 2.0f };
        ModSpec o{ P + "self_attn.o_proj", 8, 256, 1536, "F16", "F32", seed++, false, 2.0f };
        ModSpec g{ P + "mlp.gate_proj", 8, 1536, 256, "BF16", "BF16", seed++, true, 2.0f };
        ModSpec u{ P + "mlp.up_proj", 16, 1536, 256, "BF16", "F32", seed++, false, 1.0f };
        ModSpec d{ P + "mlp.down_proj", 8, 256, 1536, "F32", "F16", seed++, false, 4.0f };
        ad.mods.insert(ad.mods.end(), { q, v, o, g, u, d });
    }
    ojson c = base_cfg(8, 16.0, { "q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj" });
    c["rank_pattern"] = { { "self_attn\\.q_proj", 16 }, { "up_proj", 16 } };
    c["alpha_pattern"] = { { "down_proj$", 32 } };
    write_adapter(&ad, "llama", c);
    return ad;
}

/* Spelled the text-model way (`model.layers.`) against a VLM base (`model.language_model.layers.`). */
Adapter hybrid_adapter() {
    Adapter ad;
    uint32_t seed = 100;
    const std::string P0 = "model.layers.0.", P1 = "model.layers.1.";
    ad.mods = {
        { P0 + "linear_attn.in_proj_qkv", 16, 2 * kNK * kHK + kNV * kHV, 256, "BF16", "F32", seed++, false, 1.0f },
        { P0 + "linear_attn.in_proj_z", 8, kNV * kHV, 256, "F16", "BF16", seed++, false, 0.5f },
        { P0 + "linear_attn.in_proj_a", 8, kNV, 256, "F32", "F32", seed++, false, 2.0f },
        { P0 + "linear_attn.in_proj_b", 8, kNV, 256, "BF16", "BF16", seed++, false, 2.0f },
        { P0 + "linear_attn.out_proj", 8, 256, kNV * kHV, "BF16", "F16", seed++, false, 2.0f },
        { P1 + "self_attn.q_proj", 16, 2 * 8 * 128, 256, "BF16", "BF16", seed++, false, 1.0f },
        { P1 + "self_attn.k_proj", 8, 2 * 128, 256, "F32", "BF16", seed++, false, 2.0f },
        { P1 + "self_attn.v_proj", 8, 2 * 128, 256, "BF16", "F32", seed++, false, 2.0f },
        { P1 + "self_attn.o_proj", 8, 256, 8 * 128, "F16", "F16", seed++, false, 2.0f },
        { P0 + "mlp.gate_proj", 8, 1536, 256, "BF16", "BF16", seed++, false, 2.0f },
        { P0 + "mlp.up_proj", 8, 1536, 256, "BF16", "BF16", seed++, false, 2.0f },
        { P0 + "mlp.down_proj", 8, 256, 1536, "BF16", "BF16", seed++, false, 4.0f },
        { P1 + "mlp.down_proj", 8, 256, 1536, "F32", "BF16", seed++, false, 4.0f },
    };
    ojson c = base_cfg(8, 16.0, { "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
                                  "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                  "down_proj" });
    c["rank_pattern"] = { { "in_proj_qkv", 16 }, { "q_proj", 16 } };
    c["alpha_pattern"] = { { "in_proj_z", 4 }, { "down_proj$", 32 } };
    write_adapter(&ad, "hybrid", c);
    return ad;
}

/* Only in_proj_z and k_proj, spelled the base's own way: the fused ssm_inz is laid out from z's
 * rows alone. */
Adapter hybrid_adapter2() {
    Adapter ad;
    ad.mods = {
        { "model.language_model.layers.0.linear_attn.in_proj_z", 4, kNV * kHV, 256, "F32", "F32", 500, false, 8.0f },
        { "model.language_model.layers.1.self_attn.k_proj", 4, 2 * 128, 256, "BF16", "BF16", 501, false, 8.0f },
    };
    ojson c = base_cfg(4, 32.0, { "in_proj_z", "k_proj" });
    write_adapter(&ad, "hybrid2", c);
    return ad;
}

const lora::TargetSet kLlamaTargets{ "q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj" };
const lora::TargetSet kHybridTargets{ "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
                                      "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                      "down_proj" };

/* A load that must refuse, with `needle` in the message. */
void expect_refusal(const std::string& dir, const std::vector<lora::Site>& sites, int max_rank,
                    const char* needle) {
    lora::AdapterInfo info;
    std::vector<lora::PackedSite> packed;
    std::string err;
    const int st = lora::load_adapter(dir, sites, max_rank, &info, &packed, &err);
    CHECK(st < 0);
    if (!has(err, needle))
        fprintf(stderr, "    expected a refusal naming '%s', got (%d) '%s'\n", needle, st, err.c_str());
    CHECK(has(err, needle));
}

std::vector<lora::Site> llama_sites(int rank, int world, const lora::TargetSet& t, RankProg* rp) {
    std::vector<lora::Site> s;
    std::string err;
    build_llama(rank, world, rp);
    lora::build_sites(rp->prog, t, 16, &s, &err);
    return s;
}

}  /* namespace */

/* ================================================================== the dense reference */
TEST(llama_fused_qkv_and_gate_up_every_world) {
    const Adapter ad = llama_adapter();
    int n = 0;
    for (int world : { 1, 2, 3, 4 })
        check_world("llama", world, build_llama, ad, kLlamaTargets, 16, "", "", &n);
    CHECK(n > 0);
    fprintf(stderr, "    llama: %d pool slots checked against the dense delta\n", n);
}

TEST(hybrid_gated_attention_and_uneven_delta_net_every_world) {
    const Adapter ad = hybrid_adapter();
    int n = 0;
    auto b = [](int r, int w, RankProg* p) { return build_hybrid(r, w, p); };
    for (int world : { 1, 2, 3, 4 })
        check_world("hybrid", world, b, ad, kHybridTargets, 16, "model.", "model.language_model.", &n);
    CHECK(n > 0);
    fprintf(stderr, "    hybrid: %d pool slots checked against the dense delta\n", n);
}

TEST(hybrid_partial_adapter_lays_out_the_fused_inz_from_z_alone) {
    const Adapter ad = hybrid_adapter2();
    int n = 0;
    auto b = [](int r, int w, RankProg* p) { return build_hybrid(r, w, p); };
    for (int world : { 1, 2, 3, 4 })
        check_world("hybrid2", world, b, ad, kHybridTargets, 16, "", "", &n);
    CHECK(n > 0);
}

/* ================================================================== the geometry itself */
TEST(sites_record_the_uneven_split_and_attention_zero_rank) {
    /* world 3: rank 2 is attention-zero and holds delta-net K heads [0, 2) of 5 (remainder first). */
    RankProg r2;
    REQUIRE_EQ(build_hybrid(2, 3, &r2), RAD_OK);
    std::vector<lora::Site> s;
    std::vector<std::string> pools;
    std::string err;
    REQUIRE_EQ(lora::build_sites(r2.prog, kHybridTargets, 16, &s, &err, &pools), RAD_OK);
    for (const lora::Site& x : s) CHECK(!has(x.weight, "attn_"));
    const lora::Site* inz = nullptr;
    int inz_sites = 0;
    for (const lora::Site& x : s) if (x.weight == "blk.0.ssm_inz.weight") { inz = inz ? inz : &x; ++inz_sites; }
    REQUIRE(inz != nullptr);
    CHECK_EQ(inz_sites, 2);                 /* the decode matvec and the GEMM: one pool */
    CHECK_EQ(inz->n_modules, 2);
    CHECK_EQ(inz->R, 32);
    CHECK(!inz->parts_resolved);            /* two sources over four parts: B rows decide */
    REQUIRE_EQ(inz->part_lo.size(), (size_t)4);
    CHECK_EQ(inz->part_lo[0], 0);
    CHECK_EQ(inz->part_hi[0], 256);
    CHECK_EQ(inz->part_lo[2], 0);
    CHECK_EQ(inz->part_hi[2], 512);
    CHECK_EQ(inz->part_total[0], 640);
    CHECK_EQ(inz->part_total[3], 1280);
    int n_inz_pools = 0;
    for (const std::string& p : pools) n_inz_pools += p == "blk.0.ssm_inz.weight";
    CHECK_EQ(n_inz_pools, 1);
    /* rank 1 holds K heads [4, 5) */
    RankProg r1;
    REQUIRE_EQ(build_hybrid(1, 3, &r1), RAD_OK);
    REQUIRE_EQ(lora::build_sites(r1.prog, kHybridTargets, 16, &s, &err), RAD_OK);
    for (const lora::Site& x : s)
        if (x.weight == "blk.0.ssm_out.weight") {
            CHECK_EQ(x.shard, (int)RAD_SHARD_COL);
            CHECK_EQ(x.k_lo, 8 * 128);
            CHECK_EQ(x.K_r, 2 * 128);
            CHECK_EQ(x.K_full, 0);          /* a COL span: the container width is lora_A's */
        }
    /* llama even split, world 4 rank 3: the fused qkv's targeted q and v land at y 0 and y 768 */
    RankProg l3;
    REQUIRE_EQ(build_llama(3, 4, &l3), RAD_OK);
    REQUIRE_EQ(lora::build_sites(l3.prog, kLlamaTargets, 16, &s, &err), RAD_OK);
    bool seen = false;
    for (const lora::Site& x : s)
        if (x.weight == "blk.0.attn_qkv.weight") {
            seen = true;
            CHECK(x.parts_resolved);
            CHECK_EQ(x.n_modules, 2);
            REQUIRE_EQ(x.parts.size(), (size_t)2);
            CHECK_EQ(x.parts[0].module, std::string("model.layers.0.self_attn.q_proj"));
            CHECK_EQ(x.parts[0].src_lo, 3 * 384);
            CHECK_EQ(x.parts[0].src_hi, 4 * 384);
            CHECK_EQ(x.parts[0].y_col, 0);
            CHECK_EQ(x.parts[0].t_off, 0);
            CHECK_EQ(x.parts[1].module, std::string("model.layers.0.self_attn.v_proj"));
            CHECK_EQ(x.parts[1].src_lo, 3 * 384);
            CHECK_EQ(x.parts[1].y_col, 2 * 384);
            CHECK_EQ(x.parts[1].t_off, 16);
        }
    CHECK(seen);
}

/* ================================================================== the predicate, mid-declare */
TEST(weight_targeted_reads_only_what_exists_at_the_moment) {
    const RadModelMeta meta = llama_meta();
    const RadBuildCtx ctx = ctx_for(0, 1);
    Builder b(empty_registry(), meta, ctx);
    const lora::TargetSet t{ "q_proj" };
    RadWeightDecl d{};
    d.dtype = RAD_BF16; d.rank = 2; d.shape[0] = 128; d.shape[1] = 256;
    d.access = RAD_ACCESS_PER_TOKEN; d.shard = RAD_SHARD_ROW; d.group = AR::grp_layer(0);
    const rad_weight w = rad_decl_weight(&b, "blk.0.attn_q.weight", &d);
    REQUIRE(w != 0);
    CHECK(!lora::weight_targeted(b.program(), w, t));        /* no name map yet */
    CHECK(lora::source_targeted(b.program(), t, "model.layers.0.self_attn.q_proj.weight"));
    b.set_lora_targets(&t);
    CHECK_EQ(rad_lora_source_targeted(&b, "model.layers.0.self_attn.q_proj.weight"), 1);
    CHECK_EQ(rad_lora_source_targeted(&b, "model.layers.0.self_attn.k_proj.weight"), 0);
    CHECK_EQ(rad_lora_source_targeted(&b, "mtp.layers.0.self_attn.q_proj.weight"), 0);
    CHECK_EQ(rad_lora_source_targeted(&b, "model.visual.q_proj.weight"), 0);
    const lora::TargetSet head{ "embed_tokens", "lm_head" };
    b.set_lora_targets(&head);
    CHECK_EQ(rad_lora_source_targeted(&b, "model.embed_tokens.weight"), 0);
    CHECK_EQ(rad_lora_source_targeted(&b, "lm_head.weight"), 0);
    b.set_lora_targets(&t);
    CHECK_EQ(rad_lora_source_targeted(&b, nullptr), 0);
    b.set_lora_targets(nullptr);
    CHECK_EQ(rad_lora_source_targeted(&b, "model.layers.0.self_attn.q_proj.weight"), 0);
    REQUIRE_EQ(AR::map_copy(&b, "blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"), RAD_OK);
    CHECK(lora::weight_targeted(b.program(), w, t));
    CHECK(!lora::weight_targeted(b.program(), w, { "k_proj" }));
    CHECK(!lora::weight_targeted(b.program(), w, {}));
    CHECK(!lora::weight_targeted(b.program(), 0, t));
    CHECK(!lora::weight_targeted(b.program(), 999, t));
}

TEST(lora_source_targeted_on_a_generator_is_the_declared_root) {
    const RadModelMeta meta = llama_meta();
    const RadBuildCtx ctx = ctx_for(0, 1);
    Builder b(empty_registry(), meta, ctx);
    const lora::TargetSet star{ "*" };
    b.set_lora_targets(&star);
    CHECK_EQ(rad_lora_source_targeted(&b, "diffusion_model.blocks.0.attn.q.weight"), 0);
    REQUIRE_EQ(rad_lora_root(&b, "diffusion_model", "-"), RAD_OK);
    CHECK_EQ(rad_lora_source_targeted(&b, "diffusion_model.blocks.0.attn.q.weight"), 1);
    CHECK_EQ(rad_lora_source_targeted(&b, "text_encoder.blocks.0.attn.q.weight"), 0);
    const lora::TargetSet q{ "q" };
    b.set_lora_targets(&q);
    CHECK_EQ(rad_lora_source_targeted(&b, "diffusion_model.blocks.0.attn.q.weight"), 1);
    CHECK_EQ(rad_lora_source_targeted(&b, "diffusion_model.blocks.0.attn.k.weight"), 0);
}

TEST(scopes_that_are_never_targets) {
    const RadModelMeta meta = llama_meta();   /* n_layers 2 */
    const RadBuildCtx ctx = ctx_for(0, 1);
    Builder b(empty_registry(), meta, ctx);
    auto decl = [&](const char* name, const char* src, int access, int layer) {
        AR::map_copy(&b, name, src);
        RadWeightDecl d{};
        d.dtype = RAD_BF16; d.rank = 2; d.shape[0] = 128; d.shape[1] = 256;
        d.access = access; d.shard = RAD_SHARD_NONE; d.group = AR::grp_layer(layer);
        const rad_weight w = rad_decl_weight(&b, name, &d);
        push_op(b, gemm_op("gemm_nt", 128, 256, "bf16", 1, 64, { w }, { 1 }, false));
        return w;
    };
    const lora::TargetSet t{ "q_proj", "qkv", "lm_head", "embed_tokens" };
    /* the MTP head: declared at layer n_layers, sourced from mtp.* */
    const rad_weight mtp = decl("blk.2.attn_qg.weight", "mtp.layers.0.self_attn.q_proj.weight",
                                RAD_ACCESS_PER_TOKEN, 2);
    const rad_weight mtp2 = decl("blk.1.attn_mtp.weight", "mtp.layers.0.self_attn.q_proj.weight",
                                 RAD_ACCESS_PER_TOKEN, 1);
    const rad_weight vis = decl("v.blk.0.qkv.weight", "model.visual.blocks.0.attn.qkv.weight",
                                RAD_ACCESS_PER_REQUEST, 0);
    const rad_weight vis2 = decl("blk.0.vt.weight", "model.vision_tower.layers.0.self_attn.q_proj.weight",
                                 RAD_ACCESS_PER_TOKEN, 0);
    const rad_weight head = decl("output.weight", "lm_head.weight", RAD_ACCESS_VOCAB, -1);
    const rad_weight emb = decl("token_embd.weight", "model.embed_tokens.weight", RAD_ACCESS_PER_TOKEN, -1);
    const rad_weight drf = decl("draft.blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight",
                                RAD_ACCESS_PER_TOKEN, 0);
    const rad_weight trunk = decl("blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight",
                                  RAD_ACCESS_PER_TOKEN, 0);
    for (rad_weight w : { mtp, mtp2, vis, vis2, head, emb, drf })
        CHECK(!lora::weight_targeted(b.program(), w, t));
    CHECK(lora::weight_targeted(b.program(), trunk, t));
    std::vector<lora::Site> s;
    std::string err;
    REQUIRE_EQ(lora::build_sites(b.program(), t, 16, &s, &err), RAD_OK);
    REQUIRE_EQ(s.size(), (size_t)1);
    CHECK_EQ(s[0].weight, std::string("blk.0.attn_q.weight"));
    CHECK_EQ(lora::find_unsupported_uses(b.program(), t, nullptr), 0);
    CHECK(has(lora::explain_module(b.program(), t, "lm_head"), "non-trunk"));
    CHECK(has(lora::explain_module(b.program(), t, "mtp.layers.0.self_attn.q_proj"), "non-trunk"));
    CHECK_EQ(lora::explain_module(b.program(), t, "model.layers.0.self_attn.q_proj"), std::string());
}

/* ================================================================== startup refusals */
TEST(non_capable_consumers_are_reported_never_skipped) {
    const RadModelMeta meta = llama_meta();
    const RadBuildCtx ctx = ctx_for(0, 1);
    Builder b(empty_registry(), meta, ctx);
    AR::Geom g{};
    REQUIRE_EQ(AR::geom_from(g, &meta, &ctx, "bf16", RAD_BF16), RAD_OK);
    AR::Names nm("");
    AR::LinearQ gu{};
    REQUIRE_EQ(linear(b, nm, g, "blk.0.ffn_gate_up", 2 * g.n_ff, g.n_embd, RAD_SHARD_ROW, 0,
                      { "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight" },
                      { g.n_ff, g.n_ff }, 0, 0, &gu), RAD_OK);
    /* the fused gated fold over the same views (rad_block_mlp_q.h): live -> reported */
    OpInfo gated;
    gated.op = "gemm_nt_q_gated";
    gated.base.set_i("N", gu.N); gated.base.set_i("K", gu.K); gated.base.set_s("act", "silu");
    gated.weights = { gu.w, gu.ws }; gated.weight_opd = { 2, 3 };
    const uint32_t gi = push_op(b, gated);
    std::vector<lora::UnsupportedUse> un;
    const lora::TargetSet t{ "gate_proj" };
    REQUIRE_EQ(lora::find_unsupported_uses(b.program(), t, &un), 1);
    CHECK_EQ(un[0].op, std::string("gemm_nt_q_gated"));
    CHECK_EQ(un[0].op_index, gi);
    CHECK_EQ(un[0].weight, std::string("blk.0.ffn_gate_up.weight"));
    CHECK(has(un[0].module, "model.layers.0.mlp.gate_proj"));
    CHECK_EQ(lora::find_unsupported_uses(b.program(), { "down_proj" }, nullptr), 0);
    /* ...and the same op left unresolved (the builder's fallback: recorded, never issued) */
    Band dead;
    b.program().ops[gi].bands.push_back(dead);
    CHECK_EQ(lora::find_unsupported_uses(b.program(), t, nullptr), 0);
    std::vector<lora::Site> s;
    std::string err;
    REQUIRE_EQ(lora::build_sites(b.program(), t, 16, &s, &err), RAD_OK);
    CHECK_EQ(s.size(), (size_t)1);

    /* a routed expert (moe_gemm_q over a stacked slice): targeted, so reported */
    RadNameMap ex = AR::map_base("blk.0.ffn_gate_up_exps.3.weight", RAD_MAP_CONCAT);
    ex.n_src = 2;
    ex.src[0] = "model.layers.0.mlp.experts.gate_proj.weight"; ex.src_index[0] = 3;
    ex.src[1] = "model.layers.0.mlp.experts.up_proj.weight";   ex.src_index[1] = 3;
    REQUIRE_EQ(rad_decl_name_map(&b, &ex), RAD_OK);
    RadWeightDecl d{};
    d.dtype = RAD_F8E4M3; d.rank = 2; d.shape[0] = 256; d.shape[1] = 256;
    d.access = RAD_ACCESS_CONDITIONAL; d.shard = RAD_SHARD_NONE; d.group = AR::grp_expert(0, 3);
    const rad_weight xw = rad_decl_weight(&b, "blk.0.ffn_gate_up_exps.3.weight", &d);
    REQUIRE(xw != 0);
    OpInfo moe;
    moe.op = "moe_gemm_q";
    moe.weights = { xw };
    push_op(b, moe);
    un.clear();
    REQUIRE_EQ(lora::find_unsupported_uses(b.program(), t, &un), 1);
    CHECK_EQ(un[0].op, std::string("moe_gemm_q"));
}

TEST(a_transformed_or_rotated_weight_is_not_a_site) {
    const RadModelMeta meta = llama_meta();
    const RadBuildCtx ctx = ctx_for(0, 1);
    Builder b(empty_registry(), meta, ctx);
    /* The weight's encoding as the model holds it, through the builder's own source lookup: plain
     * bf16 in the "perm" basis. */
    b.set_sources([](const std::string& name, const Program&, WeightSource* out) {
        if (name != "blk.0.attn_q.weight") return (int)RAD_E_NOTFOUND;
        out->enc = rad_enc_plain(RAD_BF16);
        rad_enc_copy_str(out->enc.transform, "perm");
        out->rank = 2; out->shape[0] = 128; out->shape[1] = 256;
        return (int)RAD_OK;
    });
    auto decl = [&](const char* name, const char* src) {
        AR::map_copy(&b, name, src);
        RadWeightDecl d{};
        d.dtype = RAD_BF16; d.rank = 2; d.shape[0] = 128; d.shape[1] = 256;
        d.access = RAD_ACCESS_PER_TOKEN; d.shard = RAD_SHARD_NONE; d.group = AR::grp_layer(0);
        return rad_decl_weight(&b, name, &d);
    };
    const rad_weight wq = decl("blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight");
    const rad_weight wk = decl("blk.0.attn_k.weight", "model.layers.0.self_attn.k_proj.weight");
    const rad_weight wv = decl("blk.0.attn_v.weight", "model.layers.0.self_attn.v_proj.weight");
    const rad_weight wo = decl("blk.0.attn_o.weight", "model.layers.0.self_attn.o_proj.weight");
    const rad_weight wu = decl("blk.0.ffn_up.weight", "model.layers.0.mlp.up_proj.weight");
    REQUIRE(wq && wk && wv && wo && wu);
    REQUIRE(b.program().weights[wq].enc_known);
    /* A quantised GEMM over the transformed weight: fed the rotated codes of its activation. */
    push_op(b, gemm_op("gemm_nt_q", 128, 256, "mxfp4a8", 1, 64, { wq, wq }, { 2, 3 }, true));
    OpInfo parts = gemm_op("gemm_nt_q", 128, 256, "mxfp4a8", 1, 64, { wk, wk }, { 2, 3 }, true);
    parts.base.set_i("pb1", 64);
    push_op(b, parts);
    push_op(b, gemm_op("gemm_nt_q", 128, 512, "mxfp4a8", 1, 64, { wv, wv }, { 2, 3 }, true));
    OpInfo act = gemm_op("gemm_nt_bias", 128, 256, "bf16", 1, 64, { wo, wo }, { 1, 2 }, false);
    act.base.set_s("act", "gelu");
    push_op(b, act);
    push_op(b, gemm_op("gemm_nt_q", 128, 256, "w4a8", 1, 64, { wu, wu }, { 2, 3 }, true));
    const lora::TargetSet t{ "q_proj", "k_proj", "v_proj", "o_proj", "up_proj" };
    std::vector<lora::UnsupportedUse> un;
    REQUIRE_EQ(lora::find_unsupported_uses(b.program(), t, &un), 5);
    CHECK(has(un[0].why, "perm"));
    CHECK_EQ(un[0].weight, std::string("blk.0.attn_q.weight"));
    CHECK(has(un[0].module, "q_proj"));
    CHECK(has(un[1].why, "pb1"));
    CHECK(has(un[2].why, "K=512"));
    CHECK(has(un[3].why, "act=gelu"));
    CHECK(has(un[4].why, "step order"));
    std::vector<lora::Site> s;
    std::string err;
    REQUIRE_EQ(lora::build_sites(b.program(), t, 16, &s, &err), RAD_OK);
    CHECK_EQ(s.size(), (size_t)0);
    CHECK(has(lora::explain_module(b.program(), t, "model.layers.0.self_attn.q_proj"), "perm"));
}

/* Adapters serve the token steps of a model that writes logits, or a generator that names its
 * adapter keys: declare refuses a generator that names none, an embedding model and a program with
 * no logits by name, and lets a text model through. */
/* A plain fp GEMM over a transform-basis weight is a site: it is fed plain activations, and the
 * kernel library that serves it presents the weight in the plain basis at load (a plain row
 * without such a relayout refuses the encoding at declare). */
TEST(a_plain_gemm_over_a_transformed_weight_is_a_site) {
    const RadModelMeta meta = llama_meta();
    const RadBuildCtx ctx = ctx_for(0, 1);
    Builder b(empty_registry(), meta, ctx);
    b.set_sources([](const std::string& name, const Program&, WeightSource* out) {
        if (name != "blk.0.attn_q.weight") return (int)RAD_E_NOTFOUND;
        out->enc = rad_enc_plain(RAD_BF16);
        rad_enc_copy_str(out->enc.transform, "fwht128");
        out->rank = 2; out->shape[0] = 128; out->shape[1] = 256;
        return (int)RAD_OK;
    });
    AR::map_copy(&b, "blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight");
    RadWeightDecl d{};
    d.dtype = RAD_BF16; d.rank = 2; d.shape[0] = 128; d.shape[1] = 256;
    d.access = RAD_ACCESS_PER_TOKEN; d.shard = RAD_SHARD_NONE; d.group = AR::grp_layer(0);
    const rad_weight wq = rad_decl_weight(&b, "blk.0.attn_q.weight", &d);
    REQUIRE(wq);
    REQUIRE(b.program().weights[wq].enc_known);
    push_op(b, gemm_op("gemm_nt", 128, 256, "bf16", 1, 64, { wq }, { 1 }, false));
    const lora::TargetSet t{ "q_proj" };
    std::vector<lora::UnsupportedUse> un;
    CHECK_EQ(lora::find_unsupported_uses(b.program(), t, &un), 0);
    std::vector<lora::Site> s;
    std::string err;
    REQUIRE_EQ(lora::build_sites(b.program(), t, 16, &s, &err), RAD_OK);
    CHECK_EQ(s.size(), (size_t)1);
}

TEST(declare_refuses_a_model_that_does_not_generate_text) {
    const RadModelMeta meta = llama_meta();
    const RadBuildCtx ctx = ctx_for(0, 1);
    LoraConfig cfg;
    cfg.max_loras = 1;
    cfg.targets = { "q_proj" };
    LoraDeclare d;
    d.cfg = &cfg;
    d.targets = { "q_proj" };
    const struct { const char* what; void (*set)(Program&); } kinds[] = {
        { "declares no adapter key names",
          [](Program& p) { p.logits_buf = 1; p.generator.layout = 1; } },
        { "makes embeddings", [](Program& p) { p.pooled_buf = 1; } },
        { "declares no logits", [](Program&) {} },
    };
    for (const auto& k : kinds) {
        Builder b(empty_registry(), meta, ctx);
        k.set(b.program());
        int st = RAD_OK;
        const std::string log = stderr_of([&] { st = lora_declare(b, d); });
        CHECK_EQ(st, (int)RAD_E_UNSUPPORTED);
        CHECK(has(log, k.what));
    }
    Builder text(empty_registry(), meta, ctx);
    text.program().logits_buf = 1;
    CHECK_OK(lora_declare(text, d));
}

/* ================================================================== load refusals */
TEST(load_refusals_name_module_expected_and_found) {
    RankProg rp;
    const std::vector<lora::Site> sites = llama_sites(0, 1, kLlamaTargets, &rp);
    REQUIRE(!sites.empty());
    const std::string q = "model.layers.0.self_attn.q_proj";
    auto one = [&](const char* name, ojson cfg, std::vector<ModSpec> mods) {
        Adapter ad;
        ad.mods = std::move(mods);
        write_adapter(&ad, name, cfg);
        return ad.dir;
    };
    const ModSpec good{ q, 8, 1536, 256, "BF16", "BF16", 1, false, 2.f };
    const ojson ok = base_cfg(8, 16, { "q_proj" });

    { ojson c = ok; c["use_dora"] = true;
      expect_refusal(one("dora", c, { good }), sites, 16, "DoRA"); }
    { ojson c = ok; c["peft_type"] = "LOHA";
      expect_refusal(one("loha", c, { good }), sites, 16, "peft_type is LOHA"); }
    { ojson c = ok; c["modules_to_save"] = { "lm_head" };
      expect_refusal(one("mts", c, { good }), sites, 16, "modules_to_save"); }
    { ojson c = ok; c["bias"] = "all";
      expect_refusal(one("bias", c, { good }), sites, 16, "trained biases"); }
    { ojson c = ok; c["fan_in_fan_out"] = true;
      expect_refusal(one("fifo", c, { good }), sites, 16, "fan_in_fan_out"); }
    { ojson c = ok; c["alpha_pattern"] = { { "(?P<x>q_proj)", 4 } };
      expect_refusal(one("regex", c, { good }), sites, 16, "(?P<x>q_proj)"); }
    { ojson c = ok; c["lora_alpha"] = 1e300;
      expect_refusal(one("alpha", c, { good }), sites, 16, "no finite f32 scale"); }
    { ojson c = ok; c["rank_pattern"] = { { "q_proj", (int64_t)1 << 40 } };
      expect_refusal(one("bigrank", c, { good }), sites, 16, "rank_pattern[q_proj]"); }
    { ojson c = ok; c["lora_bias"] = true;
      expect_refusal(one("lbias", c, { good }), sites, 16, "lora_bias"); }
    { ojson c = ok; c["target_parameters"] = { "experts.gate_up_proj" };
      expect_refusal(one("tparams", c, { good }), sites, 16, "target_parameters"); }
    { ojson c = ok; c["trainable_token_indices"] = { 1, 2 };
      expect_refusal(one("ttok", c, { good }), sites, 16, "trainable_token_indices"); }
    { ojson c = ok; c["layer_replication"] = { { 0, 2 }, { 1, 2 } };
      expect_refusal(one("lrep", c, { good }), sites, 16, "layer_replication"); }
    { ojson c = ok; c["use_qalora"] = true;
      expect_refusal(one("qalora", c, { good }), sites, 16, "use_qalora"); }
    { ojson c = ok; c["alora_invocation_tokens"] = { 7, 8 };
      expect_refusal(one("alora", c, { good }), sites, 16, "alora_invocation_tokens"); }
    { ojson c = ok; c["use_qalora"] = false; c["layer_replication"] = nullptr;
      Adapter ad; ad.mods = { good }; write_adapter(&ad, "plainopts", c);
      lora::AdapterInfo info; std::vector<lora::PackedSite> packed; std::string e;
      CHECK_OK(lora::load_adapter(ad.dir, sites, 16, &info, &packed, &e)); }
    /* tensors that are not linear LoRA pairs */
    {
        Adapter ad;
        ad.mods = { good };
        write_adapter(&ad, "emb", ok);
        std::vector<TW> ts;
        TW e{ "base_model.model.model.embed_tokens.lora_embedding_A", "BF16", { 8, 4 }, std::vector<uint8_t>(64, 0) };
        ts.push_back(e);
        write_st(ad.dir + "/adapter_model.safetensors", ts);
        expect_refusal(ad.dir, sites, 16, "embedding LoRA");
        TW bias{ "base_model.model." + q + ".lora_B.bias", "BF16", { 1536 }, std::vector<uint8_t>(3072, 0) };
        write_st(ad.dir + "/adapter_model.safetensors", { bias });
        expect_refusal(ad.dir, sites, 16, "trained bias");
        TW mts{ "base_model.model.lm_head.modules_to_save.default.weight", "BF16", { 4, 4 }, std::vector<uint8_t>(32, 0) };
        write_st(ad.dir + "/adapter_model.safetensors", { mts });
        expect_refusal(ad.dir, sites, 16, "modules_to_save");
        TW lone{ "base_model.model." + q + ".lora_A.weight", "BF16", { 8, 256 }, std::vector<uint8_t>(8 * 256 * 2, 0) };
        write_st(ad.dir + "/adapter_model.safetensors", { lone });
        expect_refusal(ad.dir, sites, 16, "has lora_A and no lora_B");
        TW i8{ "base_model.model." + q + ".lora_A.weight", "I8", { 8, 256 }, std::vector<uint8_t>(8 * 256, 0) };
        TW i8b{ "base_model.model." + q + ".lora_B.weight", "BF16", { 1536, 8 }, std::vector<uint8_t>(1536 * 16, 0) };
        write_st(ad.dir + "/adapter_model.safetensors", { i8, i8b });
        expect_refusal(ad.dir, sites, 16, "expected BF16, F16 or F32");
    }
    /* shapes */
    { ModSpec m = good; m.cols = 255;
      expect_refusal(one("acols", ok, { m }), sites, 16, "lora_A is [8, 255]; site 'blk.0.attn_qkv.weight' expects [8, 256]"); }
    { ModSpec m = good; m.rows = 1535;
      expect_refusal(one("brows", ok, { m }), sites, 16, "lora_B is [1535, 8]; site 'blk.0.attn_qkv.weight' expects [1536, 8]"); }
    { ModSpec m = good; m.r = 4;
      expect_refusal(one("rank", ok, { m }), sites, 16, "expected rank 8 (r / rank_pattern), found lora_A [4, 256]"); }
    { ModSpec m = good; m.r = 32; ojson c = base_cfg(32, 32, { "q_proj" });
      expect_refusal(one("toobig", c, { m }), sites, 16, "rank 32 is above --max-lora-rank 16"); }
    { ojson c = ok; c["rank_pattern"] = { { "q_proj", 16 } };
      expect_refusal(one("pattern", c, { good }), sites, 16, "expected rank 16"); }
    /* a module the base lacks / a module that is not a site: not placed on any rank */
    {
        ModSpec m = good;
        m.module = "model.layers.0.self_attn.nope_proj";
        Adapter ad;
        ad.mods = { good, m };
        write_adapter(&ad, "lacks", ok);
        lora::AdapterInfo info;
        std::vector<lora::PackedSite> packed;
        std::string err;
        REQUIRE_EQ(lora::load_adapter(ad.dir, sites, 16, &info, &packed, &err), RAD_OK);
        REQUIRE_EQ(info.absent.size(), (size_t)1);
        CHECK(lora::check_placed({ info }, &err, &rp.prog, &kLlamaTargets) < 0);
        CHECK(has(err, "nope_proj"));
        CHECK(has(err, "the base has no tensor model.layers.0.self_attn.nope_proj.weight"));
    }
    {
        RankProg rq;
        const std::vector<lora::Site> qonly = llama_sites(0, 1, { "q_proj" }, &rq);
        ModSpec g{ "model.layers.0.mlp.gate_proj", 8, 1536, 256, "BF16", "BF16", 3, false, 2.f };
        Adapter ad;
        ad.mods = { good, g };
        write_adapter(&ad, "notsite", ok);
        lora::AdapterInfo info;
        std::vector<lora::PackedSite> packed;
        std::string err;
        REQUIRE_EQ(lora::load_adapter(ad.dir, qonly, 16, &info, &packed, &err), RAD_OK);
        const lora::TargetSet qt{ "q_proj" };
        CHECK(lora::check_placed({ info }, &err, &rq.prog, &qt) < 0);
        CHECK(has(err, "model.layers.0.mlp.gate_proj"));
        CHECK(has(err, "not in the target module list"));
    }
    /* the sites were built for another max rank */
    expect_refusal(one("maxrank", ok, { good }), sites, 8, "max rank 16, the load asks 8");
}

TEST(an_ambiguous_vlm_spelling_is_refused) {
    const RadModelMeta meta = llama_meta();
    const RadBuildCtx ctx = ctx_for(0, 1);
    Builder b(empty_registry(), meta, ctx);
    auto decl = [&](const char* name, const char* src) {
        AR::map_copy(&b, name, src);
        RadWeightDecl d{};
        d.dtype = RAD_BF16; d.rank = 2; d.shape[0] = 128; d.shape[1] = 256;
        d.access = RAD_ACCESS_PER_TOKEN; d.shard = RAD_SHARD_NONE; d.group = AR::grp_layer(0);
        const rad_weight w = rad_decl_weight(&b, name, &d);
        push_op(b, gemm_op("gemm_nt", 128, 256, "bf16", 1, 64, { w }, { 1 }, false));
    };
    decl("blk.0.a.weight", "model.layers.0.mlp.down_proj.weight");
    decl("blk.0.b.weight", "model.language_model.layers.0.mlp.down_proj.weight");
    std::vector<lora::Site> s;
    std::string err;
    REQUIRE_EQ(lora::build_sites(b.program(), { "down_proj" }, 16, &s, &err), RAD_OK);
    REQUIRE_EQ(s.size(), (size_t)2);
    Adapter ad;
    ad.mods = { { "model.layers.0.mlp.down_proj", 8, 128, 256, "BF16", "BF16", 9, false, 2.f } };
    write_adapter(&ad, "ambig", base_cfg(8, 16, { "down_proj" }));
    expect_refusal(ad.dir, s, 16, "ambiguous");
    /* one base module named twice by the adapter, once in each spelling */
    std::vector<lora::Site> one;
    REQUIRE_EQ(lora::build_sites(b.program(), { "down_proj" }, 16, &one, &err), RAD_OK);
    one.resize(1);   /* only blk.0.a.weight: model.layers.0.mlp.down_proj */
    Adapter twice;
    twice.mods = { { "model.layers.0.mlp.down_proj", 8, 128, 256, "BF16", "BF16", 9, false, 2.f },
                   { "model.language_model.layers.0.mlp.down_proj", 8, 128, 256, "BF16", "BF16", 10,
                     false, 2.f } };
    write_adapter(&twice, "twice", base_cfg(8, 16, { "down_proj" }));
    expect_refusal(twice.dir, one, 16, "both name the base's model.layers.0.mlp.down_proj");
}

/* ================================================================== the adapter's facts */
TEST(adapter_info_and_scales) {
    RankProg rp;
    const std::vector<lora::Site> sites = llama_sites(0, 1, kLlamaTargets, &rp);
    const Adapter ad = llama_adapter();
    lora::AdapterInfo info;
    std::vector<lora::PackedSite> packed;
    std::string err;
    REQUIRE_EQ(lora::load_adapter(ad.dir, sites, 16, &info, &packed, &err), RAD_OK);
    CHECK_EQ(info.base_model, std::string("test/base-model"));
    CHECK_EQ(info.r, 8);
    CHECK_EQ(info.alpha, 16.0);
    CHECK_EQ(info.max_module_rank, 16);
    CHECK_EQ(info.modules.size(), ad.mods.size());
    CHECK(info.absent.empty());
    CHECK(info.leaves.count("q_proj") && info.leaves.count("down_proj") && !info.leaves.count("k_proj"));
    for (const lora::AdapterInfo::Module& m : info.modules)
        for (const ModSpec& s : ad.mods)
            if (s.module == m.name) {
                CHECK_EQ(m.rank, s.r);
                CHECK_EQ(m.scale, s.scale);
                CHECK(m.pool >= 0);
            }
    /* rsLoRA: alpha / sqrt(r) */
    Adapter rs;
    rs.mods = { { "model.layers.0.self_attn.q_proj", 16, 1536, 256, "BF16", "BF16", 4, false, 4.f } };
    ojson c = base_cfg(16, 16, { "q_proj" });
    c["use_rslora"] = true;
    write_adapter(&rs, "rslora", c);
    lora::AdapterInfo ri;
    REQUIRE_EQ(lora::load_adapter(rs.dir, sites, 16, &ri, &packed, &err), RAD_OK);
    REQUIRE_EQ(ri.modules.size(), (size_t)1);
    CHECK_EQ(ri.modules[0].scale, 4.0f);
}

/* rank_pattern and alpha_pattern keys match as PEFT's get_pattern_key does: the end of the
 * module's name on a dot boundary, the first matching key in the file winning. A key found only
 * inside the name, or not on a dot boundary, leaves the module at r and lora_alpha. */
TEST(pattern_keys_match_the_end_of_the_module_name_as_peft_does) {
    RankProg rp;
    const std::vector<lora::Site> sites = llama_sites(0, 1, kLlamaTargets, &rp);
    const std::string q = "model.layers.0.self_attn.q_proj";
    const struct { const char* key; double alpha; float scale; } kCases[] = {
        { "self_attn", 64.0, 2.0f },
        { "_proj", 64.0, 2.0f },
        { "q_proj", 64.0, 8.0f },
        { "layers.0.self_attn.q_proj", 64.0, 8.0f },
        { "layers.1.self_attn.q_proj", 64.0, 2.0f },
        { "model.layers.0.self_attn.q_proj", 64.0, 8.0f },
        { "k_proj|q_proj", 32.0, 4.0f },
    };
    int i = 0;
    for (const auto& c : kCases) {
        Adapter ad;
        ad.mods = { { q, 8, 1536, 256, "BF16", "BF16", 1, false, c.scale } };
        ojson cfg = base_cfg(8, 16.0, { "q_proj" });
        cfg["alpha_pattern"] = ojson::object();
        cfg["alpha_pattern"][c.key] = c.alpha;
        cfg["alpha_pattern"]["0.self_attn.q_proj"] = 128.0;   /* matches too, but comes second */
        write_adapter(&ad, "pattern" + std::to_string(i++), cfg);
        lora::AdapterInfo info;
        std::vector<lora::PackedSite> packed;
        std::string err;
        REQUIRE_EQ(lora::load_adapter(ad.dir, sites, 16, &info, &packed, &err), RAD_OK);
        REQUIRE_EQ(info.modules.size(), (size_t)1);
        const float want = c.scale == 2.0f ? 16.0f : c.scale;   /* no match: the second key's */
        if (info.modules[0].scale != want)
            fprintf(stderr, "    alpha_pattern key '%s': scale %g, PEFT's %g\n", c.key,
                    info.modules[0].scale, want);
        CHECK_EQ(info.modules[0].scale, want);
    }
}

/* ================================================================== the slot table
 * lora/slots.h: which adapter a slot holds, the holds requests take on it, and draining. */

/* An unload with no request on the adapter frees its slot at once, and the slot is written again
 * only after the step loop has gone round kSlotCoolRounds times. */
TEST(an_idle_adapter_unloads_at_once_and_its_slot_cools) {
    std::atomic<uint64_t> epoch{10};
    lora::SlotTable t;
    t.reset(1, &epoch);
    int slot = -1, n = -1;
    bool cooling = true;
    std::string err;
    CHECK_OK(t.reserve("a", &slot, &cooling, &err));
    CHECK_EQ(slot, 0);
    CHECK(!cooling);
    const uint64_t u1 = t.publish(slot, "a", "/a");
    CHECK(u1 != 0);
    CHECK_EQ(t.reserve("a", &slot, &cooling, &err), RAD_E_INVAL);   /* already loaded */
    CHECK_OK(t.unload("a", &n, &err));
    CHECK_EQ(n, 0);
    CHECK(t.names().empty());
    CHECK_EQ(t.unload("a", &n, &err), RAD_E_NOTFOUND);
    CHECK_EQ(t.reserve("b", &slot, &cooling, &err), RAD_E_FULL);
    CHECK(cooling);
    epoch += lora::kSlotCoolRounds;
    CHECK_OK(t.reserve("b", &slot, &cooling, &err));
    CHECK_EQ(slot, 0);
    const uint64_t u2 = t.publish(slot, "b", "/b");
    CHECK(u2 != 0 && u2 != u1);   /* a fresh uid: its prefix-cache blocks start cold */
}

/* An unload of an adapter requests hold drains it: new requests cannot name it, the ones holding
 * it keep it, a second unload reports the count as it is then, a load of the same name is refused,
 * and the last hold to go frees the slot -- which then cools like any freed slot. A hold copied
 * into every choice of one request counts once. */
TEST(an_adapter_in_use_drains_and_its_last_request_frees_it) {
    std::atomic<uint64_t> epoch{0};
    lora::SlotTable t;
    t.reset(1, &epoch);
    int slot = -1, n = -1;
    bool cooling = false;
    std::string err;
    CHECK_OK(t.reserve("a", &slot, &cooling, &err));
    const uint64_t u = t.publish(slot, "a", "/a");
    epoch += 10;

    int32_t s1 = -1, s2 = -1;
    uint64_t v1 = 0, v2 = 0;
    std::shared_ptr<void> h1, h2;
    CHECK_OK(t.acquire("a", &s1, &v1, &h1));
    CHECK_OK(t.acquire("a", &s2, &v2, &h2));
    CHECK_EQ(s1, 0);
    CHECK_EQ(v1, u);
    std::shared_ptr<void> h2_choice = h2;   /* a second choice of the same request */
    CHECK_EQ(t.in_flight("a"), 2);

    CHECK_OK(t.unload("a", &n, &err));
    CHECK_EQ(n, 2);
    CHECK(t.names().empty());
    std::shared_ptr<void> h3;
    int32_t s3 = -1;
    uint64_t v3 = 0;
    CHECK_EQ(t.acquire("a", &s3, &v3, &h3), RAD_E_NOTFOUND);   /* as for a name never loaded */
    CHECK(!h3);
    CHECK_EQ(t.reserve("a", &slot, &cooling, &err), RAD_E_STATE);
    CHECK(has(err, "being unloaded"));
    CHECK_OK(t.unload("a", &n, &err));                         /* asked again: the same */
    CHECK_EQ(n, 2);

    h1.reset();
    CHECK_OK(t.unload("a", &n, &err));
    CHECK_EQ(n, 1);
    h2.reset();
    CHECK_EQ(t.in_flight("a"), 1);                             /* the other choice still holds */
    h2_choice.reset();
    CHECK_EQ(t.in_flight("a"), 0);
    CHECK_EQ(t.unload("a", &n, &err), RAD_E_NOTFOUND);         /* gone, by itself */

    CHECK_EQ(t.reserve("a", &slot, &cooling, &err), RAD_E_FULL);
    CHECK(cooling);
    epoch += lora::kSlotCoolRounds;
    CHECK_OK(t.reserve("a", &slot, &cooling, &err));
    CHECK(t.publish(slot, "a", "/a2") != u);
}

/* Requests take and drop holds on several threads while the adapter is unloaded and loaded again
 * in a loop: a load is never handed a slot a live hold still names, and a request only ever gets
 * the slot of a published adapter. The loop starts once the users hold the adapter and runs until
 * they have taken thousands of holds during it, at most 30 s: on a loaded machine its 3000 rounds
 * alone can end before a user thread is first scheduled. */
TEST(a_slot_is_never_reloaded_under_a_live_hold) {
    std::atomic<uint64_t> epoch{0};
    lora::SlotTable t;
    t.reset(2, &epoch);
    int slot = -1;
    bool cooling = false;
    std::string err;
    REQUIRE_EQ(t.reserve("a", &slot, &cooling, &err), RAD_OK);
    t.publish(slot, "a", "/a");
    std::atomic<int> live[2] = { 0, 0 };
    std::atomic<bool> stop{ false };
    std::atomic<long> holds{ 0 };
    std::vector<std::thread> users;
    for (int i = 0; i < 4; ++i)
        users.emplace_back([&] {
            while (!stop.load()) {
                int32_t s = -1;
                uint64_t u = 0;
                std::shared_ptr<void> h;
                if (t.acquire("a", &s, &u, &h) != RAD_OK) { std::this_thread::yield(); continue; }
                ++live[s];
                ++holds;
                std::this_thread::yield();
                --live[s];
                h.reset();
            }
        });
    while (holds.load() == 0) std::this_thread::yield();
    const long before = holds.load();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    int bad = 0, loads = 0;
    for (int round = 0; round < 3000 || (holds.load() - before < 3000 &&
                                         std::chrono::steady_clock::now() < deadline); ++round) {
        int n = 0;
        (void)t.unload("a", &n, &err);
        epoch += lora::kSlotCoolRounds;
        if (t.reserve("a", &slot, &cooling, &err) != RAD_OK) continue;   /* still draining */
        if (live[slot].load() != 0) ++bad;
        t.publish(slot, "a", "/a");
        ++loads;
    }
    stop.store(true);
    for (std::thread& u : users) u.join();
    CHECK_EQ(bad, 0);
    CHECK(loads > 0);
    CHECK(holds.load() > before);
}

TEST(nothing_here_touches_a_device) {
    /* A guard on the guard: the Registry these programs were declared against has no plugin. */
    CHECK_EQ(empty_registry().plugins().size(), (size_t)0);
}


TEST(a_matching_published_combination_wins_over_a_full_slot_table) {
    std::atomic<uint64_t> epoch{0};
    lora::SlotTable table;
    table.reset(1, &epoch);
    int32_t slot = -1;
    uint64_t uid = 0;
    std::shared_ptr<void> request;
    bool all_held = true;
    CHECK_EQ(table.acquire("same", &slot, &uid, &request, &all_held), RAD_E_NOTFOUND);
    CHECK(!all_held);
    // Another request publishes after the first cache miss. A single atomic lookup must
    // acquire that exact combination, even though it occupies the only slot and is held.
    int reserved = -1;
    bool cooling = false;
    std::string why;
    REQUIRE_EQ(table.reserve("same", &reserved, &cooling, &why), RAD_OK);
    const uint64_t published = table.publish(reserved, "same", "");
    std::shared_ptr<void> other;
    REQUIRE_EQ(table.acquire("same", &slot, &uid, &other), RAD_OK);
    REQUIRE_EQ(table.acquire("same", &slot, &uid, &request, &all_held), RAD_OK);
    CHECK(!all_held);
    CHECK_EQ(uid, published);
    CHECK_EQ(slot, reserved);
    std::shared_ptr<void> missing;
    CHECK_EQ(table.acquire("different", &slot, &uid, &missing, &all_held), RAD_E_NOTFOUND);
    CHECK(all_held);
}

RAD_TEST_MAIN()
