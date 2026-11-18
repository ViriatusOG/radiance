/* gen_lora_test.cpp -- runtime LoRA for generators: adapter files in every spelling and kind
 * (core/lora/adapter.h), their keys mapped through a generator's declared names, stacked at
 * strengths, packed per rank; and the request fields that name them (core/server/loras.h).
 *
 * Host only, no kernel plugin: the Program is made by the core Builder over an empty Registry, as
 * lora_test.cpp makes its own, with the generator's names declared through the builder (the calls
 * rad_lora_root / rad_lora_alias make). The reference is the test's own dense delta of every
 * checkpoint module, placed by model semantics -- which rows of which tensor a rank's declared
 * weight holds -- never by anything adapter.cpp or lora.cpp computes.
 */
#include "rad_test.h"

#include "build/rad_build.h"
#include "lora/adapter.h"
#include "lora/lora.h"
#include "plugin/registry.h"
#include "rad_plugin.h"
#include "server/loras.h"

#include <arch/rad_arch.h>
#include <arch/rad_fp8.h>

#include "util/json.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <tuple>
#include <map>
#include <string>
#include <vector>
#include <sys/stat.h>

using namespace rad;
namespace AR = rad::arch;
using ojson = Json;

namespace {

Registry& empty_registry() {
    static Registry r;
    return r;
}

float val(uint32_t seed, uint64_t i) {
    uint64_t x = (uint64_t)seed * 0x9E3779B97F4A7C15ull + i * 0xBF58476D1CE4E5B9ull + 1;
    x ^= x >> 31; x *= 0x94D049BB133111EBull; x ^= x >> 29;
    return (float)((double)(x >> 11) / (double)(1ull << 53) - 0.5) * 0.5f;
}

/* ------------------------------------------------------------------ safetensors */
struct TW {
    std::string name, dtype;
    std::vector<int64_t> shape;
    std::vector<float> v;        /* written as dtype */
};
void write_st(const std::string& path, const std::vector<TW>& ts) {
    ojson h = ojson::object();
    std::vector<std::vector<uint8_t>> bytes;
    uint64_t off = 0;
    for (const TW& t : ts) {
        std::vector<uint8_t> b;
        if (t.dtype == "I64") { const int64_t v = 4; b.resize(8); std::memcpy(b.data(), &v, 8); }
        for (float f : t.v) {
            if (t.dtype == "F32") { uint8_t x[4]; std::memcpy(x, &f, 4); b.insert(b.end(), x, x + 4); }
            else {
                const uint16_t u = t.dtype == "F16" ? rad_f32_to_f16(f) : rad_f32_to_bf16(f);
                uint8_t x[2]; std::memcpy(x, &u, 2); b.insert(b.end(), x, x + 2);
            }
        }
        h[t.name] = { { "dtype", t.dtype }, { "shape", t.shape },
                      { "data_offsets", { off, off + b.size() } } };
        off += b.size();
        bytes.push_back(std::move(b));
    }
    h["__metadata__"] = { { "format", "pt" } };
    std::string s = h.dump();
    while (s.size() % 8) s += ' ';
    std::ofstream f(path, std::ios::binary);
    const uint64_t n = s.size();
    f.write((const char*)&n, 8);
    f.write(s.data(), (std::streamsize)s.size());
    for (const auto& b : bytes) f.write((const char*)b.data(), (std::streamsize)b.size());
}
/* A tensor of seeded values, and the values as the file holds them. */
TW tensor(const std::string& name, const std::string& dt, std::vector<int64_t> shape, uint32_t seed,
          float mul = 1.0f) {
    TW t{ name, dt, shape, {} };
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    for (int64_t i = 0; i < n; ++i) {
        float f = val(seed, (uint64_t)i) * mul;
        if (dt == "F16") f = rad_f16_to_f32(rad_f32_to_f16(f));
        else if (dt == "BF16") f = rad_bf16_to_f32(rad_f32_to_bf16(f));
        t.v.push_back(f);
    }
    return t;
}
TW scalar(const std::string& name, float f) { return TW{ name, "F32", {}, { f } }; }

/* ------------------------------------------------------------------ dense helpers */
struct Dense {
    int64_t N = 0, K = 0;
    std::vector<double> d;
    void add(const Dense& o, double w) {
        if (d.empty()) { N = o.N; K = o.K; d.assign(o.d.size(), 0.0); }
        for (size_t i = 0; i < d.size(); ++i) d[i] += w * o.d[i];
    }
};
Dense mul(const std::vector<float>& B, const std::vector<float>& A, int64_t N, int64_t r, int64_t K,
          double s) {
    Dense o{ N, K, std::vector<double>((size_t)(N * K), 0.0) };
    for (int64_t n = 0; n < N; ++n)
        for (int64_t j = 0; j < r; ++j)
            for (int64_t k = 0; k < K; ++k)
                o.d[(size_t)(n * K + k)] += s * (double)B[(size_t)(n * r + j)] * A[(size_t)(j * K + k)];
    return o;
}

/* ------------------------------------------------------------------ the generator */
/* A two-block DiT under "transformer.", as an image generator's plugin declares one: a fused
 * [q|k|v] (row parts, ROW), the output (COL), a SwiGLU re-laid from one checkpoint tensor whose
 * value half comes first (RAD_MAP_ROWS: the gate half moved first), the down projection (COL); a
 * text encoder's linear outside the root; and a DiT weight a fused GEMM reads. */
const int64_t kD = 128, kHD = 32, kNH = 8, kF = 512, kQ = kNH * kHD;

RadModelMeta gen_meta() {
    RadModelMeta m{};
    m.arch_id = "gentest"; m.name = "lora-test-gen"; m.quant = "fp8_e4m3";
    m.n_layers = 2; m.n_embd = kD; m.n_head = kNH; m.n_head_kv = kNH; m.head_dim = kHD;
    m.n_ff = kF; m.n_vocab = 512; m.n_ctx_train = 1024;
    m.rms_eps = 1e-5f; m.rope_theta = 1e4f; m.rope_scale = 1.0f;
    return m;
}

OpInfo gemm_op(const char* name, int64_t N, int64_t K, const char* dtype, int64_t hi,
               std::vector<rad_weight> w, std::vector<int32_t> opd) {
    OpInfo o;
    o.op = name;
    o.base.set_i("N", N);
    o.base.set_i("K", K);
    o.base.set_i("group", AR::RAD_FP8_BLOCK);
    o.base.set_s("dtype", dtype);
    o.ranged_key = "M";
    o.range_lo = 1;
    o.range_hi = hi;
    o.weights = std::move(w);
    o.weight_opd = std::move(opd);
    return o;
}
void push_op(Builder& b, OpInfo o) {
    o.index = (int32_t)b.program().ops.size();
    b.program().ops.push_back(std::move(o));
}

int linear(Builder& b, AR::Names& nm, const AR::Geom& g, const char* base, int64_t N, int64_t K,
           int shard, int layer, std::initializer_list<const char*> src,
           std::initializer_list<int64_t> parts = {}, bool mapped = false,
           const char* op = "gemm_nt_q") {
    AR::LinearFP8 lin{};
    AR::ActFP8 act{};
    lin.mapped = mapped;
    const int st = lin.declare((RadBuilder*)&b, nm, g, base, N, K, shard, AR::grp_layer(layer), src,
                               act, 0, 0, parts);
    if (st < 0 || !lin.w || !lin.ws) return st < 0 ? st : RAD_E_INVAL;
    push_op(b, gemm_op(op, N, K, "fp8a8", g.max_tok, { lin.w, lin.ws }, { 2, 3 }));
    return RAD_OK;
}

RadBuildCtx ctx_for(int rank, int world) {
    RadBuildCtx c{};
    c.rank = rank;
    c.world_size = world;
    c.max_tok = 256;
    c.max_seqs = 8;
    c.max_ctx = 1024;
    c.scope = "";
    return c;
}

/* Where a declared weight's rows on one rank come from: (checkpoint module, its full rows) per
 * row of y, and the full K columns of x. */
struct Truth {
    std::vector<std::pair<std::string, int64_t>> rows;
    std::vector<int64_t> cols;
};

int build_gen(int rank, int world, Program* out, std::map<std::string, Truth>* truth) {
    const RadModelMeta meta = gen_meta();
    const RadBuildCtx ctx = ctx_for(rank, world);
    AR::Geom g{};
    RAD_ARCH_TRY(AR::geom_from(g, &meta, &ctx, "bf16", RAD_BF16));
    Builder b(empty_registry(), meta, ctx);
    b.program().generator.layout = RAD_GEN_LAYOUT_JOINT_STEP;
    RAD_ARCH_TRY(AR::lora_names((RadBuilder*)&b, "transformer.", AR::kLoraDiffusionSpellings,
                                { { "blocks.#.attn.qkv", "transformer_blocks.#.attn.to_q", 0, kQ, 0 },
                                  { "blocks.#.attn.qkv", "transformer_blocks.#.attn.to_k", kQ, 2 * kQ, 0 },
                                  { "blocks.#.attn.qkv", "transformer_blocks.#.attn.to_v", 2 * kQ, 3 * kQ, 0 },
                                  { "blocks.#.attn.out_proj", "transformer_blocks.#.attn.to_out.0" },
                                  { "blocks.#.mlp.fc1", "transformer_blocks.#.ff.net.0.proj", 0, kF, kF },
                                  { "blocks.#.mlp.fc1", "transformer_blocks.#.ff.net.0.proj", kF, 2 * kF, 0 },
                                  { "blocks.#.mlp.fc2", "transformer_blocks.#.ff.net.2" } }));
    AR::Names nm("");
    const int64_t qr = g.q_dim(), fr = g.n_ff;
    for (int l = 0; l < 2; ++l) {
        const std::string P = "transformer.transformer_blocks." + std::to_string(l) + ".";
        auto ck = [&](const char* leaf) { return nm.ckpt("%s%s.weight", P.c_str(), leaf); };
        RAD_ARCH_TRY(linear(b, nm, g, nm.f("dit.%d.qkv", l), 3 * qr, kD, RAD_SHARD_ROW, l,
                            { ck("attn.to_q"), ck("attn.to_k"), ck("attn.to_v") }, { qr, qr, qr }));
        RAD_ARCH_TRY(linear(b, nm, g, nm.f("dit.%d.o", l), kD, qr, RAD_SHARD_COL, l,
                            { ck("attn.to_out.0") }));
        const char* fsrc = ck("ff.net.0.proj");
        RAD_ARCH_TRY(AR::map_rows((RadBuilder*)&b, nm.f("dit.%d.gate_up.weight", l),
                                  { { fsrc, (int)kF, kF }, { fsrc, 0, kF } }));
        RAD_ARCH_TRY(linear(b, nm, g, nm.f("dit.%d.gate_up", l), 2 * fr, kD, RAD_SHARD_ROW, l,
                            { fsrc }, { fr, fr }, true));
        RAD_ARCH_TRY(linear(b, nm, g, nm.f("dit.%d.down", l), kD, fr, RAD_SHARD_COL, l,
                            { ck("ff.net.2") }));

        Truth q, o, gu, dn;
        const char* part[3] = { "attn.to_q", "attn.to_k", "attn.to_v" };
        for (const char* p : part)
            for (int64_t i = rank * qr; i < (rank + 1) * qr; ++i) q.rows.push_back({ P + p, i });
        for (int64_t k = 0; k < kD; ++k) q.cols.push_back(k);
        for (int64_t i = 0; i < kD; ++i) o.rows.push_back({ P + "attn.to_out.0", i });
        for (int64_t k = rank * qr; k < (rank + 1) * qr; ++k) o.cols.push_back(k);
        for (int64_t i = rank * fr; i < (rank + 1) * fr; ++i) gu.rows.push_back({ P + "ff.net.0.proj", kF + i });
        for (int64_t i = rank * fr; i < (rank + 1) * fr; ++i) gu.rows.push_back({ P + "ff.net.0.proj", i });
        for (int64_t k = 0; k < kD; ++k) gu.cols.push_back(k);
        for (int64_t i = 0; i < kD; ++i) dn.rows.push_back({ P + "ff.net.2", i });
        for (int64_t k = rank * fr; k < (rank + 1) * fr; ++k) dn.cols.push_back(k);
        const std::string L = std::to_string(l);
        (*truth)["dit." + L + ".qkv.weight"] = q;
        (*truth)["dit." + L + ".o.weight"] = o;
        (*truth)["dit." + L + ".gate_up.weight"] = gu;
        (*truth)["dit." + L + ".down.weight"] = dn;
    }
    /* Outside the root: a text encoder's projection. */
    RAD_ARCH_TRY(linear(b, nm, g, "te.0.q", kD, kD, RAD_SHARD_NONE, 0,
                        { "text_encoder.layers.0.q_proj.weight" }));
    /* Under the root, read by a fused GEMM: it stays the base. */
    RAD_ARCH_TRY(linear(b, nm, g, "dit.fused", kD, kD, RAD_SHARD_NONE, 0,
                        { "transformer.proj_out.weight" }, {}, false, "gemm_nt_q_swiglu"));
    *out = std::move(b.program());
    return RAD_OK;
}

std::string new_dir(const std::string& name) {
    const std::string d = radtest::tmp_path(name);
    mkdir(d.c_str(), 0755);
    return d;
}

/* ------------------------------------------------------------------ the adapter files */
/* Two files, every spelling and kind, and the dense delta of every checkpoint module they reach
 * (strength 1). File 1: kohya LoRA on block 0's fused qkv; ComfyUI LoRA on block 0's fc1 (gate
 * half first); PEFT LoRA on block 1's down projection under the checkpoint's own name; LoHa on
 * block 1's out_proj; a full delta on block 0's fc2; a module of no model; a bias delta. File 2:
 * LoKr on block 1's qkv; a bare-keyed LoRA on block 0's qkv, stacking with file 1's. */
struct Files {
    std::string f1, f2;
    std::map<std::string, Dense> d1, d2;   /* by checkpoint module, full rows */
};

void place(std::map<std::string, Dense>* m, const std::string& mod, int64_t N_full, const Dense& d,
           int64_t from_lo, int64_t n, int64_t to_lo) {
    Dense& t = (*m)[mod];
    if (t.d.empty()) { t.N = N_full; t.K = d.K; t.d.assign((size_t)(N_full * d.K), 0.0); }
    for (int64_t r = 0; r < n; ++r)
        for (int64_t k = 0; k < d.K; ++k)
            t.d[(size_t)((to_lo + r) * d.K + k)] += d.d[(size_t)((from_lo + r) * d.K + k)];
}

Files make_files() {
    Files f;
    const std::string dir = new_dir("genlora");
    const std::string B0 = "transformer.transformer_blocks.0.", B1 = "transformer.transformer_blocks.1.";
    std::vector<TW> ts;
    {   /* kohya LoRA, rank 8, alpha 4: qkv of block 0 */
        TW dn = tensor("lora_unet_blocks_0_attn_qkv.lora_down.weight", "F16", { 8, kD }, 1);
        TW up = tensor("lora_unet_blocks_0_attn_qkv.lora_up.weight", "BF16", { 3 * kQ, 8 }, 2);
        const Dense d = mul(up.v, dn.v, 3 * kQ, 8, kD, 4.0 / 8.0);
        place(&f.d1, B0 + "attn.to_q", kQ, d, 0, kQ, 0);
        place(&f.d1, B0 + "attn.to_k", kQ, d, kQ, kQ, 0);
        place(&f.d1, B0 + "attn.to_v", kQ, d, 2 * kQ, kQ, 0);
        ts.push_back(dn); ts.push_back(up);
        TW al{ "lora_unet_blocks_0_attn_qkv.alpha", "I64", {}, {} };   /* as kohya writes some */
        ts.push_back(al);
    }
    {   /* ComfyUI LoRA, rank 4, no alpha: fc1 of block 0, [gate; up] */
        TW a = tensor("diffusion_model.blocks.0.mlp.fc1.lora_A.weight", "BF16", { 4, kD }, 3);
        TW b = tensor("diffusion_model.blocks.0.mlp.fc1.lora_B.weight", "F32", { 2 * kF, 4 }, 4);
        const Dense d = mul(b.v, a.v, 2 * kF, 4, kD, 1.0);
        place(&f.d1, B0 + "ff.net.0.proj", 2 * kF, d, 0, kF, kF);
        place(&f.d1, B0 + "ff.net.0.proj", 2 * kF, d, kF, kF, 0);
        ts.push_back(a); ts.push_back(b);
    }
    {   /* PEFT LoRA under the checkpoint's own name, rank 6, alpha 12 */
        TW a = tensor(B1 + "ff.net.2.lora_A.default.weight", "F32", { 6, kF }, 5);
        TW b = tensor(B1 + "ff.net.2.lora_B.default.weight", "F32", { kD, 6 }, 6);
        place(&f.d1, B1 + "ff.net.2", kD, mul(b.v, a.v, kD, 6, kF, 12.0 / 6.0), 0, kD, 0);
        ts.push_back(a); ts.push_back(b); ts.push_back(scalar(B1 + "ff.net.2.alpha", 12.0f));
    }
    {   /* LoHa, ranks 4 and 3, alpha 2: out_proj of block 1 */
        const std::string k = "diffusion_model.blocks.1.attn.out_proj.";
        TW w1a = tensor(k + "hada_w1_a", "F32", { kD, 4 }, 7), w1b = tensor(k + "hada_w1_b", "F32", { 4, kQ }, 8);
        TW w2a = tensor(k + "hada_w2_a", "F32", { kD, 3 }, 9), w2b = tensor(k + "hada_w2_b", "F32", { 3, kQ }, 10);
        const Dense p1 = mul(w1a.v, w1b.v, kD, 4, kQ, 1.0), p2 = mul(w2a.v, w2b.v, kD, 3, kQ, 1.0);
        Dense d{ kD, kQ, std::vector<double>(p1.d.size()) };
        for (size_t i = 0; i < d.d.size(); ++i) d.d[i] = 0.5 * p1.d[i] * p2.d[i];
        place(&f.d1, B1 + "attn.to_out.0", kD, d, 0, kD, 0);
        ts.insert(ts.end(), { w1a, w1b, w2a, w2b, scalar(k + "alpha", 2.0f) });
    }
    {   /* a full delta: fc2 of block 0 */
        TW df = tensor("diffusion_model.blocks.0.mlp.fc2.diff", "F32", { kD, kF }, 11, 0.1f);
        Dense d{ kD, kF, std::vector<double>(df.v.begin(), df.v.end()) };
        place(&f.d1, B0 + "ff.net.2", kD, d, 0, kD, 0);
        ts.push_back(df);
    }
    ts.push_back(tensor("diffusion_model.nowhere.lora_A.weight", "F32", { 2, 8 }, 12));
    ts.push_back(tensor("diffusion_model.nowhere.lora_B.weight", "F32", { 8, 2 }, 13));
    ts.push_back(tensor("diffusion_model.blocks.0.mlp.fc2.diff_b", "F32", { kD }, 14));
    f.f1 = dir + "/one.safetensors";
    write_st(f.f1, ts);

    ts.clear();
    {   /* LoKr: w1 [3, 2] full, w2 = w2_a [kQ, 4] @ w2_b [4, kD / 2], alpha 8 (scale 8 / 4) */
        const std::string k = "lora_unet_blocks_1_attn_qkv.";
        TW w1 = tensor(k + "lokr_w1", "F32", { 3, 2 }, 15);
        TW w2a = tensor(k + "lokr_w2_a", "F32", { kQ, 4 }, 16), w2b = tensor(k + "lokr_w2_b", "F32", { 4, kD / 2 }, 17);
        const Dense w2 = mul(w2a.v, w2b.v, kQ, 4, kD / 2, 1.0);
        Dense d{ 3 * kQ, kD, std::vector<double>((size_t)(3 * kQ * kD)) };
        for (int64_t i1 = 0; i1 < 3; ++i1)
            for (int64_t j1 = 0; j1 < 2; ++j1)
                for (int64_t i2 = 0; i2 < kQ; ++i2)
                    for (int64_t j2 = 0; j2 < kD / 2; ++j2)
                        d.d[(size_t)((i1 * kQ + i2) * kD + j1 * (kD / 2) + j2)] =
                            2.0 * w1.v[(size_t)(i1 * 2 + j1)] * w2.d[(size_t)(i2 * (kD / 2) + j2)];
        place(&f.d2, B1 + "attn.to_q", kQ, d, 0, kQ, 0);
        place(&f.d2, B1 + "attn.to_k", kQ, d, kQ, kQ, 0);
        place(&f.d2, B1 + "attn.to_v", kQ, d, 2 * kQ, kQ, 0);
        ts.insert(ts.end(), { w1, w2a, w2b, scalar(k + "alpha", 8.0f) });
    }
    {   /* bare keys, rank 8: qkv of block 0, stacked on file 1's */
        TW a = tensor("blocks.0.attn.qkv.lora_A.weight", "BF16", { 8, kD }, 18);
        TW b = tensor("blocks.0.attn.qkv.lora_B.weight", "BF16", { 3 * kQ, 8 }, 19);
        const Dense d = mul(b.v, a.v, 3 * kQ, 8, kD, 1.0);
        place(&f.d2, B0 + "attn.to_q", kQ, d, 0, kQ, 0);
        place(&f.d2, B0 + "attn.to_k", kQ, d, kQ, kQ, 0);
        place(&f.d2, B0 + "attn.to_v", kQ, d, 2 * kQ, kQ, 0);
        ts.push_back(a); ts.push_back(b);
    }
    f.f2 = dir + "/two.safetensors";
    write_st(f.f2, ts);
    return f;
}

/* A packed slot's product, [N_r, K_r]. */
std::vector<double> slot_delta(const lora::Site& s, const lora::PackedSite& p) {
    std::vector<double> d((size_t)(s.N_r * s.K_r), 0.0);
    for (int64_t y = 0; y < s.N_r; ++y) {
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

}  /* namespace */

/* ================================================================== the files */
TEST(adapter_files_group_by_module_and_refuse_what_they_cannot_reproduce) {
    const Files f = make_files();
    lora::AdapterFile a;
    std::string err;
    REQUIRE_EQ(a.open(f.f1, &err), RAD_OK);
    CHECK_EQ(a.modules().size(), (size_t)6);
    CHECK_EQ(a.skipped().size(), (size_t)1);
    CHECK_EQ(std::string(a.kind_name()), std::string("LoRA"));
    const std::string dir = new_dir("genlora_bad");
    write_st(dir + "/dora.safetensors", { tensor("m.lora_A.weight", "F32", { 2, 4 }, 1),
                                          tensor("m.lora_B.weight", "F32", { 4, 2 }, 2),
                                          tensor("m.dora_scale", "F32", { 4, 1 }, 3) });
    CHECK_EQ(a.open(dir + "/dora.safetensors", &err), (int)RAD_E_UNSUPPORTED);
    CHECK(err.find("DoRA") != std::string::npos);
    write_st(dir + "/half.safetensors", { tensor("m.lora_A.weight", "F32", { 2, 4 }, 1) });
    CHECK_EQ(a.open(dir + "/half.safetensors", &err), (int)RAD_E_FORMAT);
}

/* A delta given as an operator truncates to its best rank-r form: exact at or above its rank, and
 * otherwise keeping what the sketch says it keeps. */
TEST(truncation_is_exact_within_the_rank_and_honest_past_it) {
    const int64_t N = 96, K = 80;
    std::vector<float> D((size_t)(N * K));
    /* rank 12 */
    std::vector<float> U((size_t)(N * 12)), V((size_t)(12 * K));
    for (size_t i = 0; i < U.size(); ++i) U[i] = val(21, i);
    for (size_t i = 0; i < V.size(); ++i) V[i] = val(22, i);
    const Dense ref = mul(U, V, N, 12, K, 1.0);
    for (size_t i = 0; i < D.size(); ++i) D[i] = (float)ref.d[i];
    lora::Delta d;
    d.N = N; d.K = K; d.rank_bound = std::min(N, K);
    d.apply = [&](const float* X, int64_t l, float* Y) {
        for (int64_t n = 0; n < N; ++n)
            for (int64_t j = 0; j < l; ++j) {
                double a = 0;
                for (int64_t k = 0; k < K; ++k) a += (double)D[(size_t)(n * K + k)] * X[k * l + j];
                Y[n * l + j] = (float)a;
            }
    };
    d.apply_t = [&](const float* Y, int64_t l, float* X) {
        for (int64_t k = 0; k < K; ++k)
            for (int64_t j = 0; j < l; ++j) {
                double a = 0;
                for (int64_t n = 0; n < N; ++n) a += (double)D[(size_t)(n * K + k)] * Y[n * l + j];
                X[k * l + j] = (float)a;
            }
    };
    std::vector<float> B, A;
    double kept = 0;
    lora::truncate(d, 16, &B, &A, &kept);
    const Dense got = mul(B, A, N, 16, K, 1.0);
    double e = 0, t = 0;
    for (size_t i = 0; i < got.d.size(); ++i) { e += std::pow(got.d[i] - ref.d[i], 2); t += ref.d[i] * ref.d[i]; }
    CHECK(e / t < 1e-8);
    CHECK_NEAR(kept, 1.0, 1e-6);
    /* rank 6 of a rank-12 matrix: what is lost is what the report says is lost */
    lora::truncate(d, 6, &B, &A, &kept);
    const Dense g6 = mul(B, A, N, 6, K, 1.0);
    e = 0;
    for (size_t i = 0; i < g6.d.size(); ++i) e += std::pow(g6.d[i] - ref.d[i], 2);
    CHECK(kept < 0.99 && kept > 0.2);
    CHECK_NEAR(1.0 - e / t, kept, 0.02);
}

/* ================================================================== the names */
TEST(keys_resolve_through_spellings_aliases_and_kohya) {
    Program P;
    std::map<std::string, Truth> truth;
    REQUIRE_EQ(build_gen(0, 1, &P, &truth), RAD_OK);
    std::vector<lora::Site> sites;
    std::string err;
    REQUIRE_EQ(lora::build_sites(P, { "*" }, 16, &sites, &err), RAD_OK);
    /* 2 blocks x 4 weights; the text encoder's and the fused GEMM's are not sites */
    CHECK_EQ(sites.size(), (size_t)8);
    for (const lora::Site& s : sites) CHECK(s.weight.rfind("dit.", 0) == 0 && s.weight != "dit.fused.weight");
    const lora::KeyMap km(P, sites);
    std::string why;
    auto one = [&](const std::string& k) { return km.resolve(k, &why); };
    std::vector<lora::Target> t = one("lora_unet_blocks_1_attn_qkv");
    REQUIRE_EQ(t.size(), (size_t)3);
    CHECK_EQ(t[1].module, std::string("transformer.transformer_blocks.1.attn.to_k"));
    CHECK_EQ(t[1].from_lo, kQ);
    CHECK_EQ(t[1].from_hi, 2 * kQ);
    t = one("diffusion_model.blocks.0.mlp.fc2");
    REQUIRE_EQ(t.size(), (size_t)1);
    CHECK_EQ(t[0].module, std::string("transformer.transformer_blocks.0.ff.net.2"));
    t = one("transformer.transformer_blocks.0.attn.to_out.0");
    REQUIRE_EQ(t.size(), (size_t)1);
    t = one("lora_unet_transformer_blocks_0_attn_to_out_0");
    REQUIRE_EQ(t.size(), (size_t)1);
    CHECK_EQ(t[0].module, std::string("transformer.transformer_blocks.0.attn.to_out.0"));
    t = one("blocks.1.mlp.fc1");
    CHECK_EQ(t.size(), (size_t)2);
    CHECK(one("diffusion_model.proj_out").empty());
    CHECK(why.find("no adapter path") != std::string::npos);
    CHECK(one("text_encoder.layers.0.q_proj").empty());
    CHECK(one("diffusion_model.blocks.7.attn.qkv").empty());
}

/* ================================================================== stacked, packed */
/* Packed compact, as the engine packs a generator's slot: every pool's product is the dense delta,
 * and its ranks are taken from 0. */
TEST(stacked_adapters_of_every_kind_pack_to_their_dense_delta_on_every_rank) {
    const Files f = make_files();
    auto a1 = std::make_shared<lora::AdapterFile>(), a2 = std::make_shared<lora::AdapterFile>();
    std::string err;
    REQUIRE_EQ(a1->open(f.f1, &err), RAD_OK);
    REQUIRE_EQ(a2->open(f.f2, &err), RAD_OK);
    const double w1 = 0.8, w2 = -0.5;
    std::map<std::string, Dense> want;
    for (const auto& [k, d] : f.d1) want[k].add(d, w1);
    for (const auto& [k, d] : f.d2) want[k].add(d, w2);
    const int R = 24;
    for (int world : { 1, 2 }) {
        for (int rank = 0; rank < world; ++rank) {
            Program P;
            std::map<std::string, Truth> truth;
            REQUIRE_EQ(build_gen(rank, world, &P, &truth), RAD_OK);
            std::vector<lora::Site> sites;
            std::vector<std::string> pools;
            REQUIRE_EQ(lora::build_sites(P, { "*" }, R, &sites, &err, &pools), RAD_OK);
            const lora::KeyMap km(P, sites);
            std::map<std::string, lora::Composite> comp;
            lora::ComposeReport rep;
            const int cs = lora::compose(km, { { "one", a1, w1 }, { "two", a2, w2 } }, R, &comp, &rep, &err);
            if (cs != RAD_OK) fprintf(stderr, "    world %d rank %d: %s\n", world, rank, err.c_str());
            REQUIRE_EQ(cs, RAD_OK);
            CHECK_EQ(rep.unplaced.size(), (size_t)1);
            CHECK_EQ(rep.skipped.size(), (size_t)1);
            /* block 0's fc2 (a full delta, rank 128) is the one module past rank 24 */
            CHECK_EQ(rep.modules - rep.exact, (int64_t)1);
            CHECK_EQ(rep.worst_module, std::string("transformer.transformer_blocks.0.ff.net.2"));
            const std::map<std::string, lora::AdModule> mods = lora::as_modules(comp, km);
            lora::AdapterInfo info;
            std::vector<lora::PackedSite> packed;
            const int ps = lora::pack_modules(sites, R, mods, &info, &packed, &err, /*compact=*/true);
            if (ps != RAD_OK) fprintf(stderr, "    world %d rank %d: %s\n", world, rank, err.c_str());
            REQUIRE_EQ(ps, RAD_OK);
            REQUIRE_EQ(packed.size(), pools.size());
            for (size_t p = 0; p < pools.size(); ++p) {
                const lora::Site* sp = nullptr;
                for (const lora::Site& x : sites) if (x.pool == (int32_t)p && !sp) sp = &x;
                REQUIRE(sp != nullptr);
                const lora::Site& s = *sp;
                const Truth& t = truth.at(s.weight);
                /* compact: the ranks taken from 0, within the pool's R */
                const lora::PackedSite& pk = packed[(size_t)s.pool];
                CHECK(pk.empty || (pk.r_used > 0 && pk.r_used <= s.R));
                for (int64_t y = 0; y < s.N_r && !pk.empty; ++y)
                    for (int64_t j = pk.r_used; j < s.R; ++j)
                        CHECK_EQ(pk.b[(size_t)(y * s.R + j)], (uint16_t)0);
                REQUIRE_EQ((int64_t)t.rows.size(), s.N_r);
                REQUIRE_EQ((int64_t)t.cols.size(), s.K_r);
                const std::vector<double> got = slot_delta(s, packed[(size_t)s.pool]);
                double e = 0, tot = 0, mx = 0;
                bool truncated = false;
                for (int64_t y = 0; y < s.N_r; ++y) {
                    const auto& [mod, row] = t.rows[(size_t)y];
                    truncated = truncated || mod == "transformer.transformer_blocks.0.ff.net.2";
                    auto it = want.find(mod);
                    for (int64_t k = 0; k < s.K_r; ++k) {
                        const double ref = it == want.end() ? 0.0
                                           : it->second.d[(size_t)(row * it->second.K + t.cols[(size_t)k])];
                        const double g = got[(size_t)(y * s.K_r + k)];
                        e += (g - ref) * (g - ref);
                        tot += ref * ref;
                        mx = std::max(mx, std::fabs(g - ref) / (std::fabs(ref) + 1e-3));
                    }
                }
                const double rel = tot > 0 ? std::sqrt(e / tot) : std::sqrt(e);
                if (rel > (truncated ? 0.9 : 0.01))
                    fprintf(stderr, "    world %d rank %d %s: relative error %.4g\n", world, rank,
                            s.weight.c_str(), rel);
                /* bf16 factors hold a delta to ~0.5%; the truncated one to what its rank keeps */
                CHECK(rel <= (truncated ? 0.9 : 0.01));
                if (truncated && world == 1)
                    CHECK_NEAR(1.0 - rel * rel, rep.worst_kept, 0.03);
            }
        }
    }
}

/* ================================================================== the request fields */
TEST(requests_name_adapters_in_a_field_or_in_prompt_tags) {
    std::vector<std::pair<std::string, double>> out;
    std::string field, why;
    std::string prompt = "a cat, <lora:pixel art:0.7>, sitting <lora:detail>";
    Json body = { { "loras", { "style", "ink:0.5", { { "name", "x:y" }, { "weight", 2 } } } } };
    REQUIRE_EQ(server::parse_loras(body, &prompt, &out, &field, &why), RAD_OK);
    REQUIRE_EQ(out.size(), (size_t)5);
    CHECK_EQ(out[0].first, std::string("style"));
    CHECK_EQ(out[0].second, 1.0);
    CHECK_EQ(out[1].first, std::string("ink"));
    CHECK_EQ(out[1].second, 0.5);
    CHECK_EQ(out[2].first, std::string("x:y"));
    CHECK_EQ(out[2].second, 2.0);
    CHECK_EQ(out[3].first, std::string("pixel art"));
    CHECK_NEAR(out[3].second, 0.7, 1e-12);
    CHECK_EQ(out[4].first, std::string("detail"));
    CHECK_EQ(prompt, std::string("a cat, sitting"));
    body = { { "lora", { { "a", 0.25 }, { "b", 1 } } } };
    prompt = "p";
    REQUIRE_EQ(server::parse_loras(body, &prompt, &out, &field, &why), RAD_OK);
    CHECK_EQ(out.size(), (size_t)2);
    body = { { "loras", { 3 } } };
    CHECK_EQ(server::parse_loras(body, &prompt, &out, &field, &why), (int)RAD_E_INVAL);
    CHECK_EQ(field, std::string("loras"));
}

TEST(adapter_composition_compresses_a_known_spectrum_with_concurrent_callers) {
    const std::string path = new_dir("genlora-compose-spectrum") + "/one.safetensors";
    const std::string key = "transformer.transformer_blocks.0.attn.to_q";
    constexpr int rank = 10, keep = 8;
    TW A{key + ".lora_A.weight", "F32", {rank, kD}, std::vector<float>(rank * kD)};
    TW B{key + ".lora_B.weight", "F32", {kQ, rank}, std::vector<float>(kQ * rank)};
    for (int i = 0; i < rank; ++i) { A.v[i * kD + i] = 1; B.v[i * rank + i] = rank - i; }
    write_st(path, {A, B});
    auto adapter = std::make_shared<lora::AdapterFile>();
    std::string why;
    REQUIRE_EQ(adapter->open(path, &why), RAD_OK);
    Program program;
    std::map<std::string, Truth> truth;
    REQUIRE_EQ(build_gen(0, 1, &program, &truth), RAD_OK);
    std::vector<lora::Site> sites;
    REQUIRE_EQ(lora::build_sites(program, {"*"}, keep, &sites, &why), RAD_OK);
    const lora::KeyMap keys(program, sites);
    auto compose = [&] {
        std::map<std::string, lora::Composite> modules;
        lora::ComposeReport report;
        std::string error;
        const int status = lora::compose(keys, {{"one", adapter, 1.0}}, keep, &modules, &report, &error);
        return std::tuple{status, std::move(modules), report.worst_kept};
    };
    std::vector<std::future<decltype(compose())>> requests;
    for (int i = 0; i < 4; ++i) requests.push_back(std::async(std::launch::async, compose));
    for (auto& request : requests) {
        const auto [status, modules, retained] = request.get();
        REQUIRE_EQ(status, RAD_OK);
        REQUIRE_EQ(modules.size(), size_t(1));
        const auto& result = modules.at(key);
        CHECK_EQ(result.r, keep);
        CHECK_NEAR(retained, 380.0 / 385.0, 1e-5);
        const auto dense = mul(result.B, result.A, kQ, keep, kD, 1.0);
        for (int row = 0; row < kQ; ++row)
            for (int col = 0; col < kD; ++col)
                CHECK_NEAR(dense.d[(size_t)row * kD + col], row == col && row < keep ? rank - row : 0, 1e-4);
    }
}

TEST(scalar_truncation_deflates_dependent_columns_without_changing_scale) {
    for (const int64_t rows : {12, 256})
        for (const float scale : {1e-4f, 1.0f, 1e4f}) {
            constexpr int64_t cols = 128, rank = 12, keep = 8;
            lora::Delta delta;
            delta.N = rows; delta.K = cols; delta.r = rank;
            delta.rank_bound = rank; delta.low_rank = true;
            delta.B.assign((size_t)rows * rank, 0.0f);
            delta.A.assign(rank * cols, 0.0f);
            for (int64_t i = 0; i < 10; ++i) {
                delta.B[(size_t)i * rank + i] = scale;
                delta.A[i * cols + i] = 10 - i;
            }
            delta.energy = 385.0 * scale * scale;
            std::vector<float> B, A;
            double retained = 0.0;
            lora::truncate(delta, keep, &B, &A, &retained);
            CHECK_NEAR(retained, 380.0 / 385.0, 1e-5);
            const auto result = mul(B, A, rows, keep, cols, 1.0);
            double error = 0.0, reference = 0.0;
            for (int64_t row = 0; row < rows; ++row)
                for (int64_t col = 0; col < cols; ++col) {
                    const double expected = row == col && row < keep ? scale * (10 - row) : 0.0;
                    const double difference = result.d[(size_t)row * cols + col] - expected;
                    error += difference * difference;
                    reference += expected * expected;
                }
            REQUIRE(reference > 0);
            CHECK(error / reference < 1e-8);
        }
}

TEST(an_adapter_snapshot_keeps_factors_after_the_source_is_replaced) {
    const std::string path = new_dir("genlora-snapshot") + "/one.safetensors";
    const TW first = tensor("layer.lora_A.weight", "F32", {1, 8}, 1);
    const TW second = tensor("layer.lora_A.weight", "F32", {1, 8}, 9);
    const TW up = tensor("layer.lora_B.weight", "F32", {4, 1}, 2);
    write_st(path, {first, up});
    lora::AdapterFile snapshot;
    std::string why;
    REQUIRE_EQ(snapshot.open_snapshot(path, 1024 * 1024, &why), RAD_OK);
    write_st(path, {second, up});
    REQUIRE_EQ(snapshot.modules().size(), 1u);
    lora::Delta old_delta;
    REQUIRE_EQ(lora::module_delta(snapshot.modules()[0], 1.0, &old_delta, &why), RAD_OK);
    CHECK(old_delta.A == first.v);
    lora::AdapterFile current;
    REQUIRE_EQ(current.open_snapshot(path, 1024 * 1024, &why), RAD_OK);
    lora::Delta new_delta;
    REQUIRE_EQ(lora::module_delta(current.modules()[0], 1.0, &new_delta, &why), RAD_OK);
    CHECK(new_delta.A == second.v);
    lora::AdapterFile bounded;
    CHECK_EQ(bounded.open_snapshot(path, 8, &why), RAD_E_FULL);
}


TEST(a_snapshot_is_bound_to_the_expected_file_descriptor_revision) {
    const std::string dir = new_dir("genlora-revision");
    const std::string path = dir + "/one.safetensors", replacement = dir + "/replacement.safetensors";
    const TW first = tensor("layer.lora_A.weight", "F32", {1, 8}, 1);
    const TW second = tensor("layer.lora_A.weight", "F32", {1, 8}, 9);
    const TW up = tensor("layer.lora_B.weight", "F32", {4, 1}, 2);
    write_st(path, {first, up});
    FileRevision expected;
    REQUIRE_EQ(read_file_revision(path, &expected), RAD_OK);
    lora::AdapterFile old;
    std::string why;
    REQUIRE_EQ(old.open_snapshot(path, 1024 * 1024, &why, &expected), RAD_OK);
    CHECK(old.snapshot_revision() == expected);
    write_st(replacement, {second, up});
    FileRevision newer;
    REQUIRE_EQ(read_file_revision(replacement, &newer), RAD_OK);
    CHECK_EQ(expected.bytes, newer.bytes);
    CHECK(expected != newer);
    /* Deterministic descriptor mismatch: no rename timing is needed to exercise the ABA guard. */
    lora::AdapterFile wrong;
    CHECK_EQ(wrong.open_snapshot(replacement, 1024 * 1024, &why, &expected), RAD_E_STATE);
    CHECK_EQ(wrong.size_bytes(), 0u);
    CHECK(wrong.modules().empty());
    REQUIRE_EQ(::rename(replacement.c_str(), path.c_str()), 0);
    REQUIRE_EQ(read_file_revision(path, &newer), RAD_OK);
    lora::AdapterFile current;
    REQUIRE_EQ(current.open_snapshot(path, 1024 * 1024, &why, &newer), RAD_OK);
    CHECK(current.snapshot_revision() == newer);
    lora::Delta old_delta, new_delta;
    REQUIRE_EQ(lora::module_delta(old.modules()[0], 1.0, &old_delta, &why), RAD_OK);
    REQUIRE_EQ(lora::module_delta(current.modules()[0], 1.0, &new_delta, &why), RAD_OK);
    CHECK(old_delta.A == first.v);
    CHECK(new_delta.A == second.v);
}

TEST(adapter_snapshots_reject_non_regular_sources_without_blocking) {
    const std::string pipe = new_dir("genlora-nonregular") + "/fifo";
    REQUIRE_EQ(::mkfifo(pipe.c_str(), 0600), 0);
    lora::AdapterFile file;
    std::string why;
    CHECK_EQ(file.open_snapshot(pipe, 1024 * 1024, &why), RAD_E_FORMAT);
    CHECK_EQ(file.size_bytes(), 0u);
}

RAD_TEST_MAIN()
