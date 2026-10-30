/* Run the new plugin's actual issued graph on host kernels, independently of core/runtime.
 * This is a numerical architecture test, not a substitute for full-engine or GPU validation.
 * Goldens are produced offline by the official Transformers model; CTest never runs Python. */
#define RAD_ARCH_NO_EXPORTS 1
#include "arch_fakes.h"
#include "../arch/nemotron35_embed/nemotron35_embed.cpp"
#include "rad_plugin.h"
#include "mm/processor.h"
#include "server/nemotron35.h"
#include "server/oai.h"
#include "text/tokenizer.h"
#include <nlohmann/json.hpp>
#include <dlfcn.h>
#include <fstream>
#include <stdexcept>

using json = nlohmann::json;

namespace {
std::vector<const char*> host_libraries() {
    std::vector<const char*> out{"libref"};
#ifdef RAD_NEMOTRON_TEST_AVX
    out.push_back("libavx");
#endif
    return out;
}
MetaKV meta(bool real = false) {
    RadModelMeta m{};
    m.arch_id = "mistral3";
    m.quant = "";
    m.n_layers = real ? 34 : 2;
    m.n_embd = real ? 4096 : 16;
    m.n_head = real ? 32 : 4;
    m.n_head_kv = real ? 8 : 1;
    m.head_dim = 128;
    m.n_ff = real ? 14336 : 32;
    m.n_vocab = real ? 131072 : 32;
    m.n_ctx_train = 262144;
    m.rms_eps = 1e-5f;
    m.rope_theta = 1e6f;
    MetaKV mk(m);
    mk.set("model_type", "mistral3")
        .set("text_config.model_type", "ministral3")
        .set("is_causal", "0")
        .set("text_config.is_causal", "0")
        .set("text_config.hidden_act", "silu")
        .set("text_config.sliding_window", "null")
        .set("text_config.rope_parameters.rope_type", "yarn")
        .set("text_config.rope_parameters.factor", "16")
        .set("text_config.rope_parameters.original_max_position_embeddings", "16384")
        .set("text_config.rope_parameters.beta_fast", "32")
        .set("text_config.rope_parameters.beta_slow", "1")
        .set("text_config.rope_parameters.mscale", "1")
        .set("text_config.rope_parameters.mscale_all_dim", "1")
        .set("text_config.rope_parameters.llama_4_scaling_beta", "0.1");
    return mk;
}

struct Kernels {
    void* handle;
    std::vector<const RadKernelInfo*> rows;
    explicit Kernels(const char* lib) {
        const std::string path = home() + "/kernels/" + lib + ".so";
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) throw std::runtime_error(path + ": " + dlerror());
        const auto count = (int (*)())dlsym(handle, "rad_kernel_count");
        const auto at = (const RadKernelInfo* (*)(int))dlsym(handle, "rad_kernel_at");
        if (!count || !at) throw std::runtime_error("kernel exports missing");
        for (int i = 0; i < count(); ++i)
            rows.push_back(at(i));
    }
    ~Kernels() { dlclose(handle); }
    const RadKernelInfo* get(const std::string& op) const {
        for (const auto* k : rows)
            if (k->op && op == k->op) return k;
        throw std::runtime_error("no host kernel for " + op);
    }
};

RadTensor tensor(void* data, uint32_t dtype, const std::vector<int64_t>& shape) {
    RadTensor t{};
    t.data = data;
    t.dtype = dtype;
    t.rank = (uint32_t)shape.size();
    for (size_t i = 0; i < shape.size(); ++i)
        t.shape[i] = shape[i];
    rad_tensor_pack(&t);
    return t;
}

json golden(const char* file = "nemotron35.json") {
    std::ifstream f(std::string(RAD_GOLDEN_FIXTURES) + "/" + file);
    if (!f) throw std::runtime_error("Nemotron golden fixture missing");
    json j;
    f >> j;
    return j;
}

/* The recording ABI is also used by every old architecture's graph/step tests. Execute the
 * recorded operations with their own views: losing a qkv column pitch can otherwise pass a
 * graph test while producing the wrong embedding. This executor intentionally has no scheduler. */
std::map<std::string, std::vector<float>> execute_graph(const Kernels& kernels, const json& source,
                                                        RadBuilder& b, RadCtx& c,
                                                        const RadBatch& batch) {
    std::vector<std::vector<uint16_t>> weights;
    for (const auto& w : b.weights) {
        const RecMap* map = find_map(b, w.name);
        if (!map) throw std::runtime_error("unmapped weight " + w.name);
        std::vector<uint16_t> values;
        for (const auto& name : map->src) {
            const auto v = source.at(name).at("bf16").get<std::vector<uint16_t>>();
            values.insert(values.end(), v.begin(), v.end());
        }
        int64_t n = 1;
        for (uint32_t i = 0; i < w.d.rank; ++i)
            n *= w.d.shape[i];
        if ((int64_t)values.size() != n) throw std::runtime_error("weight shape mismatch");
        weights.push_back(std::move(values));
    }
    std::vector<std::vector<uint64_t>> buffers;
    for (const auto& buf : b.bufs) {
        int64_t n = 1;
        for (uint32_t i = 0; i < buf.d.rank; ++i)
            n *= buf.d.shape[i];
        buffers.emplace_back((size_t)(rad_dtype_bytes(buf.d.dtype, n) + 7) / 8, 0);
    }
    CHECK_LINT(b, &c);
    for (const auto& is : c.issues) {
        const RecOp& op = b.ops.at(is.op - 1);
        std::vector<RadParam> params;
        for (const auto& p : op.p) {
            RadParam v{};
            v.key = p.key.c_str();
            v.kind = p.kind;
            v.ival = p.ival;
            v.ihi = p.ihi;
            v.dval = p.dval;
            v.sval = p.sval.c_str();
            // Like the real ABI, freeze ranges at capacity; active rows live in issued views.
            if (v.kind == RAD_P_RANGE) {
                v.kind = RAD_P_INT;
                v.ival = p.ihi;
            }
            params.push_back(v);
        }
        std::vector<RadTensor> ts;
        for (const auto& od : is.opd) {
            if (od.kind == RAD_OPK_RAW) {
                ts.push_back(tensor(od.raw, od.dtype,
                                    od.cols > 0 ? std::vector<int64_t>{od.rows, od.cols}
                                                : std::vector<int64_t>{od.rows}));
                continue;
            }
            void* data;
            uint32_t dtype, rank;
            const int64_t* shape;
            if (od.kind == RAD_OPK_BUF) {
                const auto& d = b.bufs.at(od.handle - 1).d;
                data = buffers.at(od.handle - 1).data();
                dtype = d.dtype;
                rank = d.rank;
                shape = d.shape;
            } else {
                const auto& d = b.weights.at(od.handle - 1).d;
                data = weights.at(od.handle - 1).data();
                dtype = d.dtype;
                rank = d.rank;
                shape = d.shape;
            }
            RadTensor t = tensor(data, dtype, std::vector<int64_t>(shape, shape + rank));
            t.data = (char*)data + rad_dtype_bytes(dtype, od.offset);
            if (od.rows > 0) t.shape[0] = od.rows;
            if (od.cols > 0) t.shape[t.rank - 1] = od.cols;
            ts.push_back(t);
        }
        RadArgs args{};
        args.t = ts.data();
        args.n_t = (int)ts.size();
        args.p = params.data();
        args.n_p = (int)params.size();
        args.world_size = 1;
        const int rc = kernels.get(op.op)->launch(&args, nullptr);
        if (rc != RAD_OK) throw std::runtime_error(op.op + " returned " + std::to_string(rc));
    }
    std::map<std::string, std::vector<float>> result;
    for (size_t i = 0; i < b.bufs.size(); ++i) {
        const auto& d = b.bufs[i].d;
        int64_t count = 1;
        for (uint32_t j = 0; j < d.rank; ++j)
            count *= d.shape[j];
        auto& values = result[b.bufs[i].name];
        values.resize((size_t)count);
        if (d.dtype == RAD_F32)
            std::copy_n((const float*)buffers[i].data(), count, values.data());
        else if (d.dtype == RAD_BF16)
            for (int64_t j = 0; j < count; ++j)
                values[(size_t)j] = rad_bf16_to_f32(((const uint16_t*)buffers[i].data())[j]);
        else
            throw std::runtime_error("unsupported test buffer dtype");
    }
    return result;
}

std::vector<float> execute(const Kernels& kernels, const json& source, const json& test,
                           bool rerank = false) {
    auto mk = meta();
    if (rerank)
        mk.set("architectures", "Mistral3ForSequenceClassification")
            .set("pooling", "avg")
            .set("text_config.pooling", "avg");
    auto ctx = build_ctx();
    ctx.max_tok = 32;
    ctx.max_seqs = 4;
    RadBuilder b;
    if (nemotron35_embed::declare(&b, mk.meta(), &ctx) != RAD_OK)
        throw std::runtime_error("declaration failed");
    std::vector<int32_t> ids, positions, cu{0};
    int64_t longest = 0;
    for (size_t s = 0; s < test.at("inputs").size(); ++s) {
        const auto input = test["inputs"][s].get<std::vector<int32_t>>();
        const int off = test["offsets"][s];
        longest = std::max(longest, (int64_t)input.size());
        ids.insert(ids.end(), input.begin(), input.end());
        for (size_t p = 0; p < input.size(); ++p)
            positions.push_back(off + (int)p);
        cu.push_back((int32_t)ids.size());
    }
    RadBatch batch{};
    batch.n_tok = (int64_t)ids.size();
    batch.n_seq = (int64_t)cu.size() - 1;
    batch.max_q_len = longest;
    batch.token_ids = ids.data();
    batch.positions = positions.data();
    batch.cu_seqlens = cu.data();
    RadCtx c{.b = &b, .batch = &batch};
    nemotron35_embed::step(&c, &batch);
    if (!c.step_fail.empty()) throw std::runtime_error(c.step_fail);
    json mapped = json::object();
    for (const auto& entry : source.items())
        mapped[(rerank && entry.key() != "language_model.score.weight" ? "model." : "") +
               entry.key()] = entry.value();
    auto result = execute_graph(kernels, mapped, b, c, batch).at(b.bufs[b.pooled - 1].name);
    result.resize((size_t)batch.n_seq * (rerank ? 1 : 16));
    return result;
}

MetaKV vision_meta(bool real = false) {
    auto mk = meta(real);
    mk.set("vision_config.model_type", "pixtral")
        .set("vision_config.hidden_act", "silu")
        .set("vision_config.hidden_size", real ? "1024" : "64")
        .set("vision_config.intermediate_size", real ? "4096" : "96")
        .set("vision_config.num_hidden_layers", real ? "24" : "2")
        .set("vision_config.num_attention_heads", real ? "16" : "1")
        .set("vision_config.head_dim", "64")
        .set("vision_config.patch_size", "14")
        .set("vision_config.num_channels", "3")
        .set("vision_config.image_size", "1540")
        .set("vision_config.rope_parameters.rope_type", "default")
        .set("vision_config.rope_parameters.rope_theta", "10000")
        .set("spatial_merge_size", "2")
        .set("vision_feature_layer", "-1")
        .set("multimodal_projector_bias", "0")
        .set("projector_hidden_act", "gelu")
        .set("image_token_index", "10");
    return mk;
}

rad::mm::Frame image_frame(int f, int h, int w) {
    rad::mm::Frame out;
    out.h = h;
    out.w = w;
    out.rgb.resize((size_t)h * w * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                uint64_t z = (uint64_t)f * 0x9E3779B97F4A7C15ull +
                             (uint64_t)y * 0xBF58476D1CE4E5B9ull +
                             (uint64_t)x * 0x94D049BB133111EBull +
                             (uint64_t)c * 0x2545F4914F6CDD1Dull + 12345ull;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
                out.rgb[((size_t)y * w + x) * 3 + c] = (uint8_t)((z ^ (z >> 31)) & 255);
            }
    return out;
}

rad::mm::Processor image_processor() {
    auto mk = vision_meta();
    rad::mm::VisionConfig cfg;
    std::string why;
    if (cfg.from_meta(*mk.meta(), &why) != RAD_OK) throw std::runtime_error(why);
    cfg.max_patches = 6400;
    rad::mm::Processor p;
    if (p.init(cfg, {}, [](const std::string& s) {
            return s == "[IMG]" ? 10 : s == "[IMG_BREAK]" ? 12 : s == "[IMG_END]" ? 13 : -1;
        }) != RAD_OK)
        throw std::runtime_error("Pixtral processor init failed");
    return p;
}

std::shared_ptr<rad::mm::Item> processed(const rad::mm::Processor& p, const json& spec) {
    auto frame = image_frame(spec["f"], spec["h"], spec["w"]);
    std::shared_ptr<rad::mm::Item> out;
    std::string why;
    const int rc = p.image_from(frame, &out, &why);
    if (spec.contains("error")) {
        CHECK(rc < 0);
        return nullptr;
    }
    if (rc != RAD_OK) throw std::runtime_error(why);
    return out;
}

std::map<std::string, std::vector<float>> execute_vision(const Kernels& kernels,
                                                         const json& fixture, const json& indices) {
    auto mk = vision_meta();
    auto ctx = build_ctx();
    ctx.max_enc_patches = 64;
    ctx.max_tok = 32;
    ctx.max_seqs = 4;
    RadBuilder b;
    if (nemotron35_embed::declare(&b, mk.meta(), &ctx) != RAD_OK)
        throw std::runtime_error("vision declaration failed");
    auto processor = image_processor();
    std::vector<uint16_t> pixels;
    std::vector<int32_t> cu{0}, hh, ww, coord;
    std::vector<std::shared_ptr<rad::mm::Item>> items;
    int64_t longest = 0;
    for (const auto& i : indices) {
        auto item = processed(processor, fixture["images"][i.get<size_t>()]);
        pixels.insert(pixels.end(), item->pixels.begin(), item->pixels.end());
        cu.push_back(cu.back() + (int32_t)item->n_patch);
        hh.push_back(item->grid_h);
        ww.push_back(item->grid_w);
        longest = std::max(longest, item->n_patch);
        items.push_back(std::move(item));
    }
    const int64_t n = cu.back();
    coord.resize((size_t)n * 4);
    for (size_t i = 0; i < items.size(); ++i) {
        std::vector<int32_t> own((size_t)items[i]->n_patch * 4);
        items[i]->coords(0, items[i]->n_patch, own.data());
        for (int plane = 0; plane < 4; ++plane)
            std::copy_n(own.data() + plane * items[i]->n_patch, items[i]->n_patch,
                        coord.data() + plane * n + cu[i]);
    }
    RadBatch batch{};
    batch.enc = RAD_MM_IMAGE;
    batch.enc_n_patch = n;
    batch.enc_n_seg = (int64_t)items.size();
    batch.enc_pixels = pixels.data();
    batch.enc_coord = coord.data();
    batch.enc_cu = cu.data();
    batch.enc_seg_h = hh.data();
    batch.enc_seg_w = ww.data();
    batch.enc_max_seg = (int32_t)longest;
    RadCtx c{.b = &b, .batch = &batch};
    nemotron35_embed::step(&c, &batch);
    if (!c.step_fail.empty()) throw std::runtime_error(c.step_fail);
    auto weights = golden().at("weights");
    for (const auto& entry : fixture.at("weights").items())
        weights[entry.key()] = entry.value();
    return execute_graph(kernels, weights, b, c, batch);
}

/* The real tokenizer, with the fixture vocabulary's original ids and merge ordering. Force its
 * optional BOS/EOS on: the retrieval path must explicitly suppress both, as the template does. */
struct InputTokenizer final : rad::server::ITokenizer {
    rad::Tokenizer tok;
    mutable bool added = false;
    InputTokenizer() {
        rad::VocabBuild vb;
        const std::string path =
            std::string(RAD_CHAT_FIXTURES) + "/nemotron35/tokenizer-subset.json";
        if (rad::parse_tokenizer_json(path.c_str(), vb) != RAD_OK)
            throw std::runtime_error("fixture tokenizer parse failed");
        vb.bos = 1;
        vb.eos = 2;
        vb.add_bos = vb.add_eos = true;
        auto vocab = std::make_shared<rad::Vocab>();
        if (vocab->load(std::move(vb)) != RAD_OK || tok.init(vocab) != RAD_OK)
            throw std::runtime_error("fixture tokenizer init failed");
    }
    std::vector<int32_t> encode(const std::string& text, bool add, bool special) const override {
        added |= add;
        std::vector<int32_t> ids;
        if (tok.encode(text, ids, add, special) != RAD_OK)
            throw std::runtime_error("tokenization failed");
        return ids;
    }
    std::string decode(const std::vector<int32_t>& ids) const override {
        return tok.decode(ids, true);
    }
    int32_t bos() const override { return 1; }
    int32_t eos() const override { return 2; }
    VocabInfo vocab_info() const override {
        VocabInfo out;
        out.n_tokens = tok.vocab().n_tokens();
        out.bos = 1;
        out.eos = 2;
        out.add_bos = out.add_eos = true;
        return out;
    }
    std::unique_ptr<rad::server::IDetokenizer>
    detokenizer(const std::vector<std::string>&, const std::vector<std::string>&) const override {
        return {};
    }
};
struct InputEmbedder final : rad::server::IEmbedder {
    bool selected = true, rerank = false;
    int64_t limit = 32768;
    mutable MetaKV config = vision_meta();
    int64_t dim() const override { return rerank ? 1 : 16; }
    int64_t max_input() const override { return limit; }
    const RadModelMeta* metadata() const override { return selected ? config.meta() : nullptr; }
    bool prompt(const std::string& name, std::string* out) const override {
        if (name == "query")
            *out = "query: ";
        else if (name == "document")
            *out = "passage: ";
        else
            return false;
        return true;
    }
};
/* Only file decoding is replaced: one decoded fixture byte names a generated RGB frame. The
 * production processor still performs the resize, normalization, patching and placeholder map. */
struct InputImages final : rad::server::IMultimodal {
    rad::mm::Processor processor = image_processor();
    json fixture = golden("nemotron35_vision.json");
    std::vector<int> order;
    bool accepts(rad::server::MediaKind k) const override {
        return k == rad::server::MediaKind::Image;
    }
    int prepare(rad::server::MediaKind, const std::vector<uint8_t>& bytes,
                const rad::server::MediaOpts& opts, std::shared_ptr<rad::mm::Item>* out,
                std::string* why) override {
        if (bytes.size() != 1 || bytes[0] >= 4 || opts.image_tokens) return RAD_E_INVAL;
        const auto& spec = fixture["images"][bytes[0]];
        order.push_back(bytes[0]);
        return processor.image_from(image_frame(spec["f"], spec["h"], spec["w"]), out, why);
    }
};

std::vector<float> execute_multimodal(const Kernels& kernels, const json& test,
                                      bool rerank = false) {
    const auto vision = golden("nemotron35_vision.json");
    auto processor = image_processor();
    json indices = json::array();
    for (const auto& input : test["inputs"])
        for (const auto& p : input["placements"])
            indices.push_back(p["image"]);
    const auto encoded = indices.empty() ? std::vector<float>()
                                         : execute_vision(kernels, vision, indices).at("v.out");
    std::vector<rad::mm::PromptMedia> prompts;
    size_t encoded_at = 0;
    for (const auto& input : test["inputs"]) {
        rad::mm::PromptMedia media;
        const auto tokens = input["tokens"].get<std::vector<int32_t>>();
        for (const auto& p : input["placements"]) {
            auto item = processed(processor, vision["images"][p["image"].get<size_t>()]);
            const int32_t at = p["tok"];
            CHECK(std::equal(item->expand.begin(), item->expand.end(), tokens.begin() + at));
            item->embd.resize((size_t)item->n_rows() * 16);
            item->encoded = true;
            for (auto& x : item->embd)
                x = rad_f32_to_bf16(encoded.at(encoded_at++));
            media.items.push_back({std::move(item), at});
        }
        media.layout();
        prompts.push_back(std::move(media));
    }
    auto mk = vision_meta();
    if (rerank)
        mk.set("architectures", "Mistral3ForSequenceClassification")
            .set("pooling", "avg")
            .set("text_config.pooling", "avg");
    auto ctx = build_ctx();
    ctx.max_tok = 64;
    ctx.max_seqs = 4;
    ctx.max_enc_patches = 64;
    RadBuilder b;
    if (nemotron35_embed::declare(&b, mk.meta(), &ctx) != RAD_OK)
        throw std::runtime_error("combined declaration failed");
    std::vector<int32_t> ids, pos, cu{0}, mm_rows;
    std::vector<uint16_t> mm_embd;
    int64_t longest = 0;
    for (size_t s = 0; s < test["inputs"].size(); ++s) {
        const auto tokens = test["inputs"][s]["tokens"].get<std::vector<int32_t>>();
        longest = std::max(longest, (int64_t)tokens.size());
        for (size_t j = 0; j < tokens.size(); ++j) {
            const auto* run = prompts[s].rope.run_at((int64_t)j);
            if (run) {
                CHECK_EQ(tokens[j], 10);
                mm_rows.push_back((int32_t)ids.size());
                const auto& item = *prompts[s].items[(size_t)run->item].item;
                const int64_t row = run->row0 + (int64_t)j - run->tok;
                mm_embd.insert(mm_embd.end(), item.embd.begin() + row * 16,
                               item.embd.begin() + (row + 1) * 16);
            } else
                CHECK(tokens[j] != 10);
            ids.push_back(tokens[j]);
            pos.push_back((int32_t)j);
        }
        cu.push_back((int32_t)ids.size());
    }
    RadBatch batch{};
    batch.n_tok = (int64_t)ids.size();
    batch.n_seq = (int64_t)prompts.size();
    batch.token_ids = ids.data();
    batch.positions = pos.data();
    batch.cu_seqlens = cu.data();
    batch.max_q_len = longest;
    batch.n_mm_rows = (int64_t)mm_rows.size();
    batch.mm_rows = mm_rows.data();
    batch.mm_embd = mm_embd.data();
    RadCtx c{.b = &b, .batch = &batch};
    nemotron35_embed::step(&c, &batch);
    if (!c.step_fail.empty()) throw std::runtime_error(c.step_fail);
    auto weights = golden().at("weights");
    for (const auto& entry : vision["weights"].items())
        weights[entry.key()] = entry.value();
    if (rerank)
        weights["language_model.score.weight"] = golden("nemotron35_rerank.json")["head"];
    json mapped = json::object();
    for (const auto& entry : weights.items())
        mapped[(rerank && entry.key() != "language_model.score.weight" ? "model." : "") +
               entry.key()] = entry.value();
    auto out = execute_graph(kernels, mapped, b, c, batch).at("pooled");
    out.resize((size_t)batch.n_seq * (rerank ? 1 : 16));
    return out;
}
} // namespace

TEST(nemotron35_real_geometry_declares_bidirectional_pooling_without_kv_or_logits) {
    auto mk = meta(true);
    const auto ctx = build_ctx();
    RadBuilder b;
    CHECK_OK(nemotron35_embed::declare(&b, mk.meta(), &ctx));
    CHECK_EQ(b.logits, (rad_buf)0);
    CHECK(b.kv_groups.empty());
    REQUIRE(b.pooled);
    CHECK_EQ(b.bufs[b.pooled - 1].d.shape[1], 4096);
    CHECK_EQ(count_op(b, "embed_attn"), 34);
    CHECK_EQ(count_op(b, "position_scale"), 34);
    CHECK_EQ(count_op(b, "seg_mean"), 1);
    CHECK_EQ(count_op(b, "qk_norm_rope"), 0);
    for (const auto& op : b.ops) {
        if (op.op == "embed_attn") {
            CHECK_EQ(pint(&op, "causal"), 0);
            CHECK_EQ(pint(&op, "gqa"), 4);
            CHECK_NEAR(pflt(&op, "scale"), 1 / std::sqrt(128.), 1e-12);
        }
        if (op.op == "ministral_rope") CHECK_NEAR(pflt(&op, "yarn_magnitude"), 1.0, 0);
    }
    for (const auto& w : b.weights)
        CHECK(find_map(b, w.name));
}

TEST(nemotron35_1b_native_geometry_keeps_query_width_separate_from_hidden_width) {
    auto mk = vision_meta(true);
    mk.m.n_layers = 16;
    mk.m.n_embd = 2048;
    mk.m.n_head = 24;
    mk.m.n_ff = 6144;
    auto ctx = build_ctx();
    ctx.max_ctx = 32768;
    ctx.max_tok = 131072;
    ctx.max_enc_patches = 6400;
    RadBuilder b;
    REQUIRE_EQ(nemotron35_embed::declare(&b, mk.meta(), &ctx), RAD_OK);
    REQUIRE(b.pooled);
    CHECK_EQ(b.bufs[b.pooled - 1].d.shape[1], 2048);
    CHECK_EQ(count_op(b, "position_scale"), 16);
    CHECK_EQ(count_op(b, "pixtral_rope"), 24);
    CHECK_EQ(b.encoder.n_embd, 2048);
    CHECK_EQ(b.logits, (rad_buf)0);
    CHECK(b.kv_groups.empty());
    for (const auto& op : b.ops)
        if (op.op == "embed_attn" && pint(&op, "head_dim") == 128) CHECK_EQ(pint(&op, "gqa"), 3);
    bool output = false, projector = false;
    for (const auto& w : b.weights) {
        CHECK(find_map(b, w.name));
        if (w.name == "blk.0.attn_o.weight") {
            output = true;
            CHECK_EQ(w.d.shape[0], 2048);
            CHECK_EQ(w.d.shape[1], 3072);
        }
        if (w.name == "v.linear2.weight") {
            projector = true;
            CHECK_EQ(w.d.shape[0], 2048);
            CHECK_EQ(w.d.shape[1], 2048);
        }
    }
    CHECK(output);
    CHECK(projector);
}

TEST(nemotron35_attention_context_limit_applies_per_sequence_not_to_the_whole_batch) {
    const int M = 32769;
    std::vector<float> q(M, 0.5f), k(M, 0.5f), v(M), out(M, -9.0f);
    std::vector<int32_t> cu(M + 1);
    for (int i = 0; i <= M; ++i)
        cu[(size_t)i] = i;
    for (int i = 0; i < M; ++i)
        v[(size_t)i] = (float)(i % 31);
    for (const char* name : host_libraries()) {
        Kernels kernels(name);
        RadTensor ts[] = {tensor(q.data(), RAD_F32, {M, 1}), tensor(k.data(), RAD_F32, {M, 1}),
                          tensor(v.data(), RAD_F32, {M, 1}), tensor(cu.data(), RAD_I32, {M + 1}),
                          tensor(out.data(), RAD_F32, {M, 1})};
        RadParam p[] = {RAD_INT("M", M), RAD_INT("head_dim", 1), RAD_INT("gqa", 1),
                        RAD_INT("causal", 0), RAD_STR("dtype", "f32")};
        RadArgs a{};
        a.t = ts;
        a.n_t = 5;
        a.p = p;
        a.n_p = 5;
        // Many one-token sequences exceed 32K in total without a long attention segment.
        CHECK_OK(kernels.get("embed_attn")->launch(&a, nullptr));
        CHECK(out == v);
        std::fill(out.begin(), out.end(), -9.0f);
        const auto before = out;
        int32_t too_long[] = {0, M};
        ts[3] = tensor(too_long, RAD_I32, {2});
        CHECK_EQ(kernels.get("embed_attn")->launch(&a, nullptr), RAD_E_INVAL);
        CHECK(out == before);
    }
}

TEST(nemotron35_refuses_other_models_and_unimplemented_modes) {
    auto rc = [](MetaKV mk, RadBuildCtx ctx) {
        RadBuilder b;
        return stderr_of_rc([&] { return nemotron35_embed::declare(&b, mk.meta(), &ctx); });
    };
    auto ctx = build_ctx();
    CHECK_EQ(rc(meta().set("is_causal", "1"), ctx), RAD_E_UNSUPPORTED);
    CHECK_EQ(rc(meta().set("text_config.model_type", "llama"), ctx), RAD_E_UNSUPPORTED);
    CHECK_EQ(rc(meta().set("text_config.sliding_window", "4096"), ctx), RAD_E_UNSUPPORTED);
    CHECK_EQ(rc(meta().drop("text_config.rope_parameters.mscale"), ctx), RAD_E_INVAL);
    CHECK_EQ(rc(meta(), build_ctx(0, 2)), RAD_E_UNSUPPORTED);
    ctx.max_enc_patches = 6400;
    CHECK_EQ(rc(meta(), ctx), RAD_E_UNSUPPORTED);
    ctx = build_ctx();
    ctx.max_ctx = 262144;
    CHECK_EQ(rc(meta(), ctx), RAD_E_INVAL);
    {
        ctx = build_ctx();
        auto mk = meta();
        RadBuilder b;
        b.encs = {{"qkv.weight", rad_enc_plain(RAD_I8)}};
        CHECK_EQ(stderr_of_rc([&] { return nemotron35_embed::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
}

TEST(nemotron35_issued_graph_matches_transformers_bf16_on_host_libraries) {
    const auto fixture = golden();
    for (const char* name : host_libraries()) {
        Kernels kernels(name);
        double min_cosine = 1.0, max_error = 0, max_relative_l2 = 0;
        for (const auto& test : fixture.at("cases")) {
            const auto actual = execute(kernels, fixture.at("weights"), test);
            size_t k = 0;
            for (const auto& row : test.at("pooled")) {
                double dot = 0, aa = 0, bb = 0, squared_error = 0;
                for (const auto& value : row) {
                    const double want = value.get<double>(), got = actual.at(k++);
                    CHECK(std::isfinite(got));
                    max_error = std::max(max_error, std::abs(got - want));
                    squared_error += (got - want) * (got - want);
                    dot += got * want;
                    aa += got * got;
                    bb += want * want;
                }
                /* Host kernels fuse some BF16 rounding seams that Transformers materialises
                 * (norm gain and rotary products), and seg_mean rounds once after division.
                 * Bound vector error relative to its norm, including magnitude, rather than
                 * dividing by individual components that can be arbitrarily close to zero. */
                const double relative_l2 = std::sqrt(squared_error / bb);
                const double cosine = dot / std::sqrt(aa * bb);
                max_relative_l2 = std::max(max_relative_l2, relative_l2);
                min_cosine = std::min(min_cosine, cosine);
                CHECK(relative_l2 <= .01);
                CHECK(cosine >= .9995);
            }
        }
        std::fprintf(
            stderr,
            "    %s vs Transformers BF16: min cosine %.9f, max relative L2 %.6g, max component error %.6g\n",
            name, min_cosine, max_relative_l2, max_error);
    }
}

TEST(nemotron35_ragged_inputs_match_the_same_inputs_embedded_alone) {
    const auto fixture = golden();
    const auto& test = fixture.at("cases").at(1);
    for (const char* name : host_libraries()) {
        Kernels kernels(name);
        const auto batched = execute(kernels, fixture.at("weights"), test);
        for (size_t s = 0; s < test["inputs"].size(); ++s) {
            const json alone = {{"inputs", json::array({test["inputs"][s]})},
                                {"offsets", json::array({test["offsets"][s]})}};
            const auto single = execute(kernels, fixture.at("weights"), alone);
            for (size_t j = 0; j < single.size(); ++j)
                CHECK_NEAR(single[j], batched[s * 16 + j], 0.001);
        }
    }
}

TEST(nemotron35_rejects_media_without_an_encoder_and_decode_before_issuing_operations) {
    auto mk = meta();
    const auto ctx = build_ctx();
    RadBuilder b;
    REQUIRE_EQ(nemotron35_embed::declare(&b, mk.meta(), &ctx), RAD_OK);
    auto input = kit(b, {}, {5});
    for (int mode : {0, 1}) {
        auto batch = input->b;
        if (mode == 0)
            batch.n_mm_rows = 1;
        else
            batch.phase = RAD_PHASE_DECODE;
        RadCtx c{.b = &b, .batch = &batch};
        nemotron35_embed::step(&c, &batch);
        CHECK(!c.step_fail.empty());
        CHECK(c.issues.empty());
    }
}

TEST(nemotron35_position_scale_preserves_pitched_keys_and_checks_boundary_rounding) {
    for (const char* name : host_libraries()) {
        Kernels kernels(name);
        std::vector<uint16_t> x(4 * 6, rad_f32_to_bf16(2));
        int32_t pos[] = {0, 16383, 16384, 32767};
        auto view = tensor(x.data(), RAD_BF16, {4, 6});
        view.shape[1] = 4;
        RadTensor ts[] = {view, tensor(pos, RAD_I32, {4})};
        RadParam p[] = {RAD_INT("M", 64), RAD_INT("n", 4), RAD_INT("original", 16384),
                        RAD_F64("beta", .1), RAD_STR("dtype", "bf16")};
        RadArgs a{};
        a.t = ts;
        a.n_t = 2;
        a.p = p;
        a.n_p = 5;
        const auto* op = kernels.get("position_scale");
        CHECK_OK(op->launch(&a, nullptr));
        for (int r = 0; r < 4; ++r)
            for (int j = 0; j < 6; ++j)
                CHECK_EQ(x[(size_t)r * 6 + j], rad_f32_to_bf16(j >= 4 || r < 2 ? 2.f : 2.140625f));
        const auto before = x;
        pos[3] = -1;
        CHECK_EQ(op->launch(&a, nullptr), RAD_E_INVAL);
        CHECK(x == before);
    }
}

TEST(nemotron35_pixtral_processor_matches_official_pil_pixels_and_patch_order) {
    const auto fixture = golden("nemotron35_vision.json");
    auto p = image_processor();
    for (const auto& image : fixture["images"]) {
        const auto item = processed(p, image);
        if (!item) continue;
        CHECK_EQ(item->grid_h * 14, image["size"][0].get<int>());
        CHECK_EQ(item->grid_w * 14, image["size"][1].get<int>());
        CHECK(item->expand == image["expand"].get<std::vector<int32_t>>());
        CHECK(item->rope_1d);
        CHECK(item->patch_raster);
        uint64_t hash = 0xcbf29ce484222325ull;
        for (uint16_t value : item->pixels) {
            hash = (hash ^ (value & 255)) * 0x100000001b3ull;
            hash = (hash ^ (value >> 8)) * 0x100000001b3ull;
        }
        CHECK_EQ(hash, std::stoull(image["fnv"].get<std::string>(), nullptr, 16));
        if (!image["patches"].empty())
            CHECK(item->pixels == image["patches"].get<std::vector<uint16_t>>());
    }
}

TEST(nemotron35_pixtral_layout_keeps_separators_as_text_and_maps_only_image_rows) {
    auto p = image_processor();
    auto fixture = golden("nemotron35_vision.json");
    auto a = processed(p, fixture["images"][2]), b = processed(p, fixture["images"][3]);
    rad::mm::PromptMedia media;
    media.items = {{a, 5}, {b, 5 + (int32_t)a->expand.size() + 3}};
    media.layout();
    for (size_t i = 0; i < media.items.size(); ++i) {
        const auto& placed = media.items[i];
        int row = 0;
        for (size_t j = 0; j < placed.item->expand.size(); ++j) {
            const int pos = placed.tok + (int)j;
            const auto* run = media.rope.run_at(pos);
            if (placed.item->expand[j] == 10) {
                REQUIRE(run);
                CHECK_EQ(run->item, (int)i);
                CHECK_EQ(run->row0 + pos - run->tok, row++);
            } else
                CHECK(!run);
            int32_t xyz[3];
            media.rope.at(pos, xyz);
            for (int c : xyz)
                CHECK_EQ(c, pos);
        }
        CHECK_EQ(row, placed.item->n_rows());
        CHECK_EQ(placed.item->expand.back(), 13);
        CHECK_EQ((int)std::count(placed.item->expand.begin(), placed.item->expand.end(), 12),
                 placed.item->rows_h - 1);
    }
}

TEST(nemotron35_pixtral_real_geometry_and_configuration_refusals) {
    auto mk = vision_meta(true);
    auto ctx = build_ctx();
    ctx.max_enc_patches = 6400;
    RadBuilder b;
    RadArchProbe probe{};
    CHECK_OK(nemotron35_embed::probe(mk.meta(), &probe));
    CHECK_EQ(probe.enc_merge, 4);
    REQUIRE_EQ(nemotron35_embed::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ(count_op(b, "pixtral_rope"), 24);
    CHECK_EQ(count_op(b, "pixtral_merge"), 1);
    CHECK_EQ(b.encoder.merge, 4);
    CHECK_EQ(b.encoder.patch_dim, 588);
    CHECK_EQ(b.encoder.n_embd, 4096);
    for (const auto& w : b.weights)
        CHECK(find_map(b, w.name));
    for (const auto& mutation : std::vector<std::pair<const char*, const char*>>{
             {"vision_feature_layer", "-2"},
             {"projector_hidden_act", "silu"},
             {"multimodal_projector_bias", "1"},
             {"vision_config.rope_parameters.rope_type", "yarn"}}) {
        auto bad = vision_meta();
        bad.set(mutation.first, mutation.second);
        RadBuilder other;
        CHECK_EQ(nemotron35_embed::declare(&other, bad.meta(), &ctx), RAD_E_UNSUPPORTED);
    }
    auto small = vision_meta();
    RadBuilder quant;
    quant.encs = {{"v.patch.weight", rad_enc_plain(RAD_I8)}};
    CHECK_EQ(nemotron35_embed::declare(&quant, small.meta(), &ctx), RAD_E_UNSUPPORTED);
}

TEST(nemotron35_pixtral_graph_matches_official_vision_and_projector_stages) {
    const auto fixture = golden("nemotron35_vision.json");
    const std::pair<const char*, const char*> stages[] = {{"v.x", "hidden"},
                                                          {"v.h", "post_norm"},
                                                          {"v.packed", "packed"},
                                                          {"v.merged", "merged"},
                                                          {"v.out", "out"}};
    for (const char* name : host_libraries()) {
        Kernels kernels(name);
        double worst = 0;
        for (const auto& test : fixture["cases"]) {
            const auto got = execute_vision(kernels, fixture, test["images"]);
            for (const auto& stage : stages) {
                double error = 0, norm = 0;
                size_t j = 0;
                for (const auto& row : test[stage.second])
                    for (const auto& value : row) {
                        const double want = value.get<double>(),
                                     actual = got.at(stage.first).at(j++);
                        CHECK(std::isfinite(actual));
                        error += (actual - want) * (actual - want);
                        norm += want * want;
                    }
                const double rel = std::sqrt(error / norm);
                worst = std::max(worst, rel);
                if (rel > .01)
                    std::fprintf(stderr, "%s %s %s relative L2 %.9g\n", name,
                                 test["name"].get<std::string>().c_str(), stage.second, rel);
                CHECK(rel <= .01);
            }
        }
        std::fprintf(stderr, "    %s Pixtral stages vs Transformers BF16: max relative L2 %.9g\n",
                     name, worst);
    }
}

TEST(nemotron35_pixtral_mixed_images_match_individual_encoder_passes) {
    const auto fixture = golden("nemotron35_vision.json");
    const auto& indices = fixture["cases"][3]["images"];
    for (const char* name : host_libraries()) {
        Kernels kernels(name);
        const auto combined = execute_vision(kernels, fixture, indices).at("v.out");
        size_t offset = 0;
        for (const auto& idx : indices) {
            const auto alone = execute_vision(kernels, fixture, json::array({idx})).at("v.out");
            const auto& size = fixture["images"][idx.get<size_t>()]["size"];
            const size_t count = (size[0].get<size_t>() / 28) * (size[1].get<size_t>() / 28) * 16;
            for (size_t j = 0; j < count; ++j)
                CHECK_NEAR(alone[j], combined[offset + j], 1e-5);
            offset += count;
        }
    }
}

TEST(nemotron35_pixtral_rotary_matches_official_bf16_products_and_preserves_values) {
    const auto fixture = golden("nemotron35_vision.json");
    const auto& test = fixture["rotary"];
    auto coord = test["coord"].get<std::vector<int32_t>>();
    const auto want = test["output"].get<std::vector<uint16_t>>();
    for (const char* name : host_libraries()) {
        Kernels kernels(name);
        auto data = test["input"].get<std::vector<uint16_t>>();
        RadTensor ts[] = {tensor(data.data(), RAD_BF16, {6, 3 * 64}),
                          tensor(coord.data(), RAD_I32, {2, 6})};
        RadParam p[] = {RAD_INT("M", 64), RAD_INT("head_dim", 64), RAD_INT("n_head", 1),
                        RAD_F64("theta", 10000), RAD_STR("dtype", "bf16")};
        RadArgs a{};
        a.t = ts;
        a.n_t = 2;
        a.p = p;
        a.n_p = 5;
        CHECK_OK(kernels.get("pixtral_rope")->launch(&a, nullptr));
        int mismatches = 0;
        for (size_t i = 0; i < data.size(); ++i)
            if (data[i] != want[i] && ++mismatches <= 6)
                std::fprintf(stderr, "rotary %s index %zu got %.9g want %.9g\n", name, i,
                             rad_bf16_to_f32(data[i]), rad_bf16_to_f32(want[i]));
        CHECK(data == want);
        auto before = data;
        coord[0] = -1;
        CHECK_EQ(kernels.get("pixtral_rope")->launch(&a, nullptr), RAD_E_INVAL);
        CHECK(data == before);
        coord[0] = 0;
    }
}

TEST(nemotron35_pixtral_merger_preserves_channel_order_and_rejects_bad_grids_without_writes) {
    const int M = 16, N = 3; // a 2x4 image followed by a 4x2 image
    std::vector<int32_t> coords(4 * M);
    for (int start : {0, 8}) {
        const int h = start == 0 ? 2 : 4, w = start == 0 ? 4 : 2;
        for (int j = 0; j < 8; ++j) {
            coords[start + j] = j / w;
            coords[M + start + j] = j % w;
            coords[2 * M + start + j] = h;
            coords[3 * M + start + j] = w;
        }
    }
    for (const char* name : host_libraries())
        for (uint32_t dtype : {RAD_BF16, RAD_F32}) {
            Kernels kernels(name);
            std::vector<uint32_t> source((size_t)M * (N + 2), 0),
                dest((size_t)(M / 4) * (4 * N + 2), 0);
            RadTensor x = tensor(source.data(), dtype, {M, N + 2});
            x.shape[1] = N;
            RadTensor y = tensor(dest.data(), dtype, {M / 4, 4 * N + 2});
            y.shape[1] = 4 * N;
            auto put = [&](int row, int col, float value) {
                const size_t i = (size_t)row * (N + 2) + col;
                if (dtype == RAD_BF16)
                    ((uint16_t*)x.data)[i] = rad_f32_to_bf16(value);
                else
                    ((float*)x.data)[i] = value;
            };
            auto get = [&](int row, int col) {
                const size_t i = (size_t)row * (4 * N + 2) + col;
                return dtype == RAD_BF16 ? rad_bf16_to_f32(((uint16_t*)y.data)[i])
                                         : ((float*)y.data)[i];
            };
            for (int r = 0; r < M; ++r)
                for (int c = 0; c < N; ++c)
                    put(r, c, (float)(r * N + c));
            RadTensor ts[] = {x, tensor(coords.data(), RAD_I32, {4, M}), y};
            RadParam p[] = {RAD_INT("M", 64), RAD_INT("n", N), RAD_INT("merge", 2),
                            RAD_STR("dtype", dtype == RAD_BF16 ? "bf16" : "f32")};
            RadArgs a{};
            a.t = ts;
            a.n_t = 3;
            a.p = p;
            a.n_p = 4;
            CHECK_OK(kernels.get("pixtral_merge")->launch(&a, nullptr));
            for (int r = 0; r < M / 4; ++r) {
                const int w = r < 2 ? 4 : 2, base = r < 2 ? r * 2 : 8 + (r - 2) * 2 * w;
                for (int c = 0; c < N; ++c)
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx)
                            CHECK_EQ(get(r, c * 4 + dy * 2 + dx),
                                     (float)((base + dy * w + dx) * N + c));
                CHECK_EQ(get(r, 4 * N), 0.f);
                CHECK_EQ(get(r, 4 * N + 1), 0.f);
            }
            const auto before = dest;
            coords[M + 7] = 99;
            CHECK_EQ(kernels.get("pixtral_merge")->launch(&a, nullptr), RAD_E_INVAL);
            CHECK(dest == before);
            coords[M + 7] = 3;
        }
}

TEST(nemotron35_pixtral_rejects_incomplete_encoder_passes_and_unsupported_processor_options) {
    auto mk = vision_meta();
    auto ctx = build_ctx();
    ctx.max_enc_patches = 64;
    nemotron35_embed::Pixtral vision;
    rad::arch::Names nm("");
    RadBuilder b;
    REQUIRE_EQ(vision.declare(&b, mk.meta(), &ctx, nm, nm, 16, 2), RAD_OK);
    int32_t hh = 2, ww = 3;
    uint16_t pixel = 0;
    int32_t coord[16]{}, cu[] = {0, 4};
    RadBatch batch{};
    batch.enc = RAD_MM_IMAGE;
    batch.enc_n_patch = 4;
    batch.enc_n_seg = 1;
    batch.enc_max_seg = 4;
    batch.enc_pixels = &pixel;
    batch.enc_coord = coord;
    batch.enc_cu = cu;
    batch.enc_seg_h = &hh;
    batch.enc_seg_w = &ww;
    RadCtx c{.b = &b, .batch = &batch};
    vision.step(&c, &batch);
    CHECK(!c.step_fail.empty());
    CHECK(c.issues.empty());
    for (const auto& mutation : std::vector<std::pair<const char*, const char*>>{
             {"processor.image_processor.do_resize", "0"},
             {"processor.image_processor.resample", "2"},
             {"processor.image_processor.rescale_factor", "1"},
             {"processor.image_processor.size.longest_edge", "1119"}}) {
        auto bad = vision_meta();
        bad.set(mutation.first, mutation.second);
        rad::mm::VisionConfig cfg;
        std::string why;
        CHECK(cfg.from_meta(*bad.meta(), &why) < 0);
    }
    auto processor = image_processor();
    rad::mm::Video video;
    std::shared_ptr<rad::mm::Item> out;
    std::string why;
    CHECK_EQ(processor.video_from(video, &out, &why), RAD_E_UNSUPPORTED);
}

TEST(nemotron35_retrieval_selection_is_explicit_and_preserves_other_models) {
    auto mk = vision_meta();
    CHECK(rad::server::nemotron35_retrieval_meta(mk.meta()));
    CHECK(!rad::server::nemotron35_retrieval_meta(nullptr));
    for (const auto& field :
         std::vector<std::pair<const char*, const char*>>{{"model_type", "llama"},
                                                          {"text_config.model_type", "llama"},
                                                          {"vision_config.model_type", "qwen3_vl"},
                                                          {"is_causal", "1"},
                                                          {"text_config.is_causal", "true"}}) {
        auto other = vision_meta();
        other.set(field.first, field.second);
        CHECK(!rad::server::nemotron35_retrieval_meta(other.meta()));
    }
}

TEST(nemotron35_template_override_policy_preserves_other_embedding_models) {
    InputEmbedder embed;
    std::string why;
    CHECK_EQ(rad::server::check_embedding_template(&embed, false, &why), RAD_OK);
    CHECK_EQ(rad::server::check_embedding_template(&embed, true, &why), RAD_E_UNSUPPORTED);
    CHECK(why.find("--override-chat-template") != std::string::npos);
    embed.selected = false;
    CHECK_EQ(rad::server::check_embedding_template(&embed, true, &why), RAD_OK);
    CHECK_EQ(rad::server::check_embedding_template(nullptr, true, &why), RAD_OK);
    embed.selected = true;
    embed.config.set("model_type", "embeddinggemma2");
    CHECK_EQ(rad::server::check_embedding_template(&embed, true, &why), RAD_OK);
}

TEST(nemotron35_typed_formatter_and_real_tokenizer_match_reference_messages_and_inputs) {
    const auto fixture = golden("nemotron35_input.json");
    InputTokenizer tok;
    InputEmbedder embed;
    InputImages images;
    rad::server::OaiDeps deps;
    deps.tok = &tok;
    deps.embed = &embed;
    deps.mm = &images;
    rad::server::OaiLimits lim;
    lim.max_ctx = 32768;
    lim.allow_image = true;
    for (const auto& test : fixture["cases"]) {
        const auto messages = rad::server::json::parse(test["messages"].dump());
        std::vector<rad::server::Nemotron35Piece> pieces;
        std::string why, rendered;
        REQUIRE_EQ(rad::server::nemotron35_format(messages, &pieces, &why), RAD_OK);
        for (const auto& p : pieces)
            rendered += p.image ? "[IMG]" : p.text;
        CHECK_EQ(rendered, test["rendered"].get<std::string>());
        rad::server::OaiRequest request;
        rad::server::ApiError err;
        images.order.clear();
        const std::string body = json({{"messages", test["messages"]}}).dump();
        REQUIRE_EQ(rad::server::parse_embeddings_request(body, deps, lim, &request, &err), RAD_OK);
        REQUIRE_EQ(request.prompts.size(), (size_t)1);
        const auto& p = request.prompts[0];
        CHECK_EQ(p.text, rendered);
        CHECK(p.tokens == test["expanded_ids"].get<std::vector<int32_t>>());
        CHECK(images.order == test["image_order"].get<std::vector<int>>());
        CHECK(!tok.added);
        for (size_t j = 0; j < p.tokens.size(); ++j) {
            const auto* run = p.media ? p.media->rope.run_at((int64_t)j) : nullptr;
            CHECK((run != nullptr) == (p.tokens[j] == 10));
        }
        const auto name = test["name"].get<std::string>();
        if (name.starts_with("system_blocks/")) continue;
        json input = {{"input", test["messages"].back()["content"]}};
        if (name.starts_with("query/")) input["prompt_name"] = "query";
        if (name.starts_with("document/")) input["prompt_name"] = "document";
        rad::server::OaiRequest raw;
        images.order.clear();
        REQUIRE_EQ(rad::server::parse_embeddings_request(input.dump(), deps, lim, &raw, &err),
                   RAD_OK);
        CHECK(raw.prompts[0].tokens == p.tokens);
        CHECK(!tok.added);
    }
}

TEST(nemotron35_raw_ids_and_default_embedding_routes_keep_their_contracts) {
    InputTokenizer tok;
    InputEmbedder embed;
    rad::server::OaiDeps deps;
    deps.tok = &tok;
    deps.embed = &embed;
    rad::server::OaiLimits lim;
    rad::server::OaiRequest ids;
    rad::server::ApiError err;
    CHECK_OK(
        rad::server::parse_embeddings_request(R"({"input":[1,10,13,2]})", deps, lim, &ids, &err));
    CHECK(ids.prompts[0].tokens == std::vector<int32_t>({1, 10, 13, 2}));
    CHECK(!tok.added);
    embed.selected = false;
    const auto fixture = golden("nemotron35_input.json");
    for (const auto& test : fixture["cases"])
        if (test.contains("legacy_text")) {
            rad::server::OaiRequest native;
            CHECK_OK(rad::server::parse_embeddings_request(
                json({{"input", test["legacy_text"]}}).dump(), deps, lim, &native, &err));
            auto expected = test["legacy_ids"].get<std::vector<int32_t>>();
            expected.insert(expected.begin(), 1);
            expected.push_back(2);
            CHECK(native.prompts[0].tokens == expected);
            CHECK(tok.added);
        }
    rad::server::OaiRequest no_template;
    CHECK(rad::server::parse_embeddings_request(R"({"messages":[{"role":"user","content":"x"}]})",
                                                deps, lim, &no_template, &err) < 0);
}

TEST(nemotron35_input_batches_apply_the_prompt_and_media_layout_independently) {
    const auto fixture = golden("nemotron35_input.json");
    const json* text = nullptr;
    const json* media = nullptr;
    for (const auto& test : fixture["cases"]) {
        if (test["name"] == "query/literal_controls") text = &test;
        if (test["name"] == "query/text_before_images") media = &test;
    }
    REQUIRE(text);
    REQUIRE(media);
    InputTokenizer tok;
    InputEmbedder embed;
    InputImages images;
    rad::server::OaiDeps deps;
    deps.tok = &tok;
    deps.embed = &embed;
    deps.mm = &images;
    rad::server::OaiLimits lim;
    lim.allow_image = true;
    rad::server::OaiRequest out;
    rad::server::ApiError err;
    const json body = {{"input", json::array({(*text)["messages"].back()["content"],
                                              (*media)["messages"].back()["content"]})},
                       {"prompt_name", "query"}};
    REQUIRE_EQ(rad::server::parse_embeddings_request(body.dump(), deps, lim, &out, &err), RAD_OK);
    REQUIRE_EQ(out.prompts.size(), (size_t)2);
    CHECK(out.prompts[0].tokens == (*text)["expanded_ids"].get<std::vector<int32_t>>());
    CHECK(!out.prompts[0].media);
    CHECK(out.prompts[1].tokens == (*media)["expanded_ids"].get<std::vector<int32_t>>());
    REQUIRE(out.prompts[1].media);
    CHECK(images.order == (*media)["image_order"].get<std::vector<int>>());
    CHECK(!tok.added);
}

TEST(nemotron35_input_rejects_invalid_structures_and_checks_limits_after_image_expansion) {
    InputTokenizer tok;
    InputEmbedder embed;
    InputImages images;
    rad::server::OaiDeps deps;
    deps.tok = &tok;
    deps.embed = &embed;
    deps.mm = &images;
    rad::server::OaiLimits lim;
    lim.allow_image = true;
    for (
        const char* body :
        {R"({"messages":[{"role":"assistant","content":"x"}]})",
         R"({"messages":[{"role":"user","content":[{"type":"text","text":4}]}]})",
         R"({"messages":[{"role":"user","content":[{"type":"video_url","video_url":"data:video/mp4;base64,AA=="}]}]})",
         R"({"messages":[{"role":"user","content":null}]})",
         R"({"messages":[{"role":"user","content":""}]})", R"({"input":""})",
         R"({"messages":[{"role":"system","content":[{"type":"image_url"}]},{"role":"user","content":"x"}]})",
         R"({"messages":[{"role":"user","content":"x"}],"input":"y"})",
         R"({"messages":[{"role":"user","content":"x"}],"prompt_name":"query"})",
         R"({"input":[1,2],"prompt_name":"query"})", R"({"input":"x","prompt_name":"unknown"})",
         R"({"input":[{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]})",
         R"({"messages":[{"role":"system","content":"[IMG]"},{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]})"}) {
        rad::server::OaiRequest request;
        rad::server::ApiError err;
        CHECK(rad::server::parse_embeddings_request(body, deps, lim, &request, &err) < 0);
    }
    for (bool messages : {false, true}) {
        const json content = json::array(
            {{{"type", "image_url"}, {"image_url", {{"url", "data:image/png;base64,Ag=="}}}}});
        const json body =
            messages ? json({{"messages", json::array({{{"role", "user"}, {"content", content}}})}})
                     : json({{"input", content}});
        rad::server::OaiRequest request;
        rad::server::ApiError err;
        embed.limit = 3;
        CHECK(rad::server::parse_embeddings_request(body.dump(), deps, lim, &request, &err) < 0);
        CHECK_EQ(err.code, std::string("context_length_exceeded"));
        CHECK_EQ(err.n_prompt_tokens, 8);
        CHECK_EQ(err.n_ctx, 3);
    }
    embed.limit = 100000;
    lim.max_ctx = 262144;
    json body = {{"input", std::vector<int32_t>(32769, 3)}};
    rad::server::OaiRequest too_long;
    rad::server::ApiError err;
    CHECK(rad::server::parse_embeddings_request(body.dump(), deps, lim, &too_long, &err) < 0);
    CHECK_EQ(err.n_ctx, 32768);
    body["input"].erase(body["input"].end() - 1);
    rad::server::OaiRequest boundary;
    CHECK_OK(rad::server::parse_embeddings_request(body.dump(), deps, lim, &boundary, &err));
    CHECK_EQ(boundary.prompts[0].tokens.size(), (size_t)32768);
}

TEST(nemotron35_combined_image_text_graph_matches_official_mistral3_and_preserves_batch_isolation) {
    const auto fixture = golden("nemotron35_multimodal.json");
    for (const char* name : host_libraries()) {
        Kernels kernels(name);
        double worst_l2 = 0, worst_cos = 1;
        for (const auto& test : fixture["cases"]) {
            const auto actual = execute_multimodal(kernels, test);
            size_t at = 0;
            for (const auto& row : test["pooled"]) {
                double error = 0, aa = 0, bb = 0, dot = 0;
                for (const auto& x : row) {
                    const double want = x.get<double>(), got = actual.at(at++);
                    CHECK(std::isfinite(got));
                    error += (got - want) * (got - want);
                    aa += got * got;
                    bb += want * want;
                    dot += got * want;
                }
                const double rel = std::sqrt(error / bb), cos = dot / std::sqrt(aa * bb);
                worst_l2 = std::max(worst_l2, rel);
                worst_cos = std::min(worst_cos, cos);
                if (rel > .01 || cos < .9995)
                    std::fprintf(stderr, "%s %s combined relL2 %.9g cosine %.9g\n", name,
                                 test["name"].get<std::string>().c_str(), rel, cos);
                CHECK(rel <= .01);
                CHECK(cos >= .9995);
            }
            if (test["inputs"].size() > 1)
                for (size_t s = 0; s < test["inputs"].size(); ++s) {
                    const auto single =
                        execute_multimodal(kernels, {{"inputs", json::array({test["inputs"][s]})}});
                    for (size_t j = 0; j < 16; ++j)
                        CHECK_NEAR(single[j], actual[s * 16 + j], .001);
                }
        }
        std::fprintf(stderr,
                     "    %s combined graph vs Mistral3 BF16: max relL2 %.9g, min cosine %.9g\n",
                     name, worst_l2, worst_cos);
    }
}

TEST(nemotron35_image_injection_rejects_missing_or_out_of_bounds_batch_fields_before_issuing) {
    auto mk = vision_meta();
    auto ctx = build_ctx();
    ctx.max_enc_patches = 64;
    RadBuilder b;
    REQUIRE_EQ(nemotron35_embed::declare(&b, mk.meta(), &ctx), RAD_OK);
    auto input = kit(b, {}, {5});
    int32_t row = 0;
    uint16_t value = 0;
    for (int mode : {0, 1, 2, 3}) {
        auto batch = input->b;
        batch.n_mm_rows = 1;
        batch.mm_rows = &row;
        batch.mm_embd = &value;
        if (mode == 0) batch.mm_rows = nullptr;
        if (mode == 1) batch.mm_embd = nullptr;
        if (mode == 2) batch.n_mm_rows = batch.n_tok + 1;
        if (mode == 3) batch.n_mm_rows = -1;
        RadCtx c{.b = &b, .batch = &batch};
        nemotron35_embed::step(&c, &batch);
        CHECK(!c.step_fail.empty());
        CHECK(c.issues.empty());
    }
}

TEST(nemotron35_reranker_declares_a_scalar_head_without_changing_embedding_pooling) {
    auto mk = vision_meta(true);
    mk.set("architectures", "Mistral3ForSequenceClassification")
        .set("pooling", "avg")
        .set("text_config.pooling", "avg");
    auto ctx = build_ctx();
    ctx.max_seqs = 16;
    ctx.max_tok = 131072;
    ctx.max_enc_patches = 64;
    RadBuilder b;
    CHECK_OK(nemotron35_embed::declare(&b, mk.meta(), &ctx));
    REQUIRE(b.pooled);
    CHECK_EQ(b.bufs[b.pooled - 1].d.shape[1], 1);
    CHECK_EQ(b.logits, (rad_buf)0);
    CHECK(b.kv_groups.empty());
    CHECK_EQ(count_op(b, "seg_mean_bf16_sum"), 1);
    CHECK_EQ(count_op(b, "seg_mean"), 0);
    CHECK_EQ(count_op(b, "logits_gemm"), 1);
    const auto* map = find_map(b, "score.weight");
    REQUIRE(map);
    CHECK_EQ(map->src[0], std::string("language_model.score.weight"));
    REQUIRE(find_map(b, "token_embd.weight"));
    CHECK_EQ(find_map(b, "token_embd.weight")->src[0],
             std::string("model.language_model.embed_tokens.weight"));
    REQUIRE(find_map(b, "v.patch.weight"));
    CHECK_EQ(find_map(b, "v.patch.weight")->src[0],
             std::string("model.vision_tower.patch_conv.weight"));
    CHECK(rad::server::nemotron35_rerank_meta(mk.meta()));
    CHECK(!rad::server::nemotron35_retrieval_meta(mk.meta()));
    RadBuilder quantized;
    quantized.encs.emplace_back("score.weight", rad_enc_plain(RAD_I8));
    CHECK_EQ(nemotron35_embed::declare(&quantized, mk.meta(), &ctx), RAD_E_UNSUPPORTED);
    auto plain = vision_meta(true);
    RadBuilder embedding;
    CHECK_OK(nemotron35_embed::declare(&embedding, plain.meta(), &ctx));
    CHECK_EQ(embedding.bufs[embedding.pooled - 1].d.shape[1], 4096);
    CHECK_EQ(count_op(embedding, "seg_mean"), 1);
    CHECK_EQ(count_op(embedding, "seg_mean_bf16_sum"), 0);
    CHECK_EQ(count_op(embedding, "logits_gemm"), 0);
    mk.set("pooling", "last");
    RadBuilder bad;
    CHECK_EQ(nemotron35_embed::declare(&bad, mk.meta(), &ctx), RAD_E_UNSUPPORTED);
}

TEST(nemotron35_reranker_scores_match_independent_native_text_and_image_oracles) {
    const auto fixture = golden("nemotron35_rerank.json");
    auto weights = golden()["weights"];
    weights["language_model.score.weight"] = fixture["head"];
    for (const char* name : host_libraries()) {
        Kernels kernels(name);
        for (const auto& c : fixture["cases"]) {
            const auto actual = execute(kernels, weights, c, true);
            REQUIRE_EQ(actual.size(), c["scores"].size());
            for (size_t i = 0; i < actual.size(); ++i)
                CHECK_NEAR(actual[i], c["scores"][i].get<float>(), .01);
        }
        for (const auto& c : fixture["multimodal"]) {
            const auto actual = execute_multimodal(kernels, c, true);
            REQUIRE_EQ(actual.size(), c["scores"].size());
            for (size_t i = 0; i < actual.size(); ++i)
                CHECK_NEAR(actual[i], c["scores"][i].get<float>(), .01);
        }
    }
}

TEST(nemotron35_reranker_pool_rounds_the_sum_before_dividing_and_preserves_pitches) {
    const auto f = golden("nemotron35_rerank.json")["pool"];
    const auto bits = f["bf16"].get<std::vector<uint16_t>>();
    const auto want = f["expected"].get<std::vector<uint16_t>>();
    for (const char* name : host_libraries()) {
        Kernels kernels(name);
        std::vector<uint16_t> x(10 * 9, 0x4321), y(2 * 10, 0x4321);
        for (int r = 0; r < 10; ++r)
            std::copy_n(bits.data() + r * 7, 7, x.data() + r * 9);
        auto tx = tensor(x.data(), RAD_BF16, {10, 9});
        tx.shape[1] = 7;
        auto ty = tensor(y.data(), RAD_BF16, {2, 10});
        ty.shape[1] = 7;
        int32_t cu[] = {0, 3, 10};
        RadTensor ts[] = {tx, tensor(cu, RAD_I32, {3}), ty};
        RadParam ps[] = {RAD_INT("M", 64), RAD_INT("n", 7), RAD_STR("dtype", "bf16")};
        RadArgs a{};
        a.t = ts;
        a.n_t = 3;
        a.p = ps;
        a.n_p = 3;
        const auto* op = kernels.get("seg_mean_bf16_sum");
        CHECK_OK(op->launch(&a, nullptr));
        for (int r = 0; r < 2; ++r)
            for (int i = 0; i < 10; ++i)
                CHECK_EQ(y[r * 10 + i], i < 7 ? want[r * 7 + i] : (uint16_t)0x4321);
        const auto before = y;
        cu[1] = 0;
        CHECK_EQ(op->launch(&a, nullptr), RAD_E_INVAL);
        CHECK(y == before);
        cu[1] = 11;
        CHECK_EQ(op->launch(&a, nullptr), RAD_E_INVAL);
        CHECK(y == before);
        cu[1] = 3;
        ts[0].stride[0] = INT64_MAX;
        CHECK_EQ(op->launch(&a, nullptr), RAD_E_SHAPE);
        CHECK(y == before);
        ts[0] = tx;
        ts[0].data = nullptr;
        CHECK_EQ(op->launch(&a, nullptr), RAD_E_INVAL);
        CHECK(y == before);
    }
}

TEST(nemotron35_reranker_parser_matches_the_native_pair_template_and_context_boundary) {
    InputTokenizer tok;
    InputEmbedder embed;
    embed.rerank = true;
    embed.config.set("architectures", "Mistral3ForSequenceClassification");
    InputImages images;
    rad::server::OaiDeps deps;
    deps.tok = &tok;
    deps.embed = &embed;
    deps.mm = &images;
    rad::server::OaiLimits lim;
    lim.max_ctx = 32768;
    lim.allow_image = true;
    for (const auto& c : golden("nemotron35_rerank.json")["http"]) {
        json doc = c["document"];
        if (!c["image"].is_null()) {
            json parts = json::array();
            if (!doc.is_null())
                parts.push_back({{"type", "text"}, {"text", doc}});
            const uint8_t byte = c["image"].get<uint8_t>();
            parts.push_back(
                {{"type", "image_url"},
                 {"image_url",
                  {{"url", "data:image/png;base64," + rad::server::base64_encode(&byte, 1)}}}});
            doc = {{"content", parts}};
        }
        const auto body = json({{"query", c["query"]}, {"documents", json::array({doc})}}).dump();
        rad::server::OaiRequest out;
        rad::server::ApiError err;
        CHECK_OK(rad::server::parse_rerank_request(body, deps, lim, &out, &err));
        REQUIRE_EQ(out.prompts.size(), (size_t)1);
        CHECK_EQ(out.prompts[0].text, c["rendered"].get<std::string>());
        CHECK(!tok.added);
        const int64_t length = (int64_t)out.prompts[0].tokens.size();
        for (int delta : {-1, 0, 1}) {
            auto bounded = lim;
            bounded.max_ctx = length + delta;
            rad::server::OaiRequest r;
            const int rc = rad::server::parse_rerank_request(body, deps, bounded, &r, &err);
            if (delta < 0)
                CHECK_EQ(rc, RAD_E_INVAL);
            else
                CHECK_OK(rc);
        }
        rad::server::OaiRequest wrong;
        CHECK_EQ(rad::server::parse_embeddings_request(R"({"input":"x"})", deps, lim, &wrong, &err),
                 RAD_E_UNSUPPORTED);
    }
}

TEST(nemotron35_reranker_head_keeps_the_native_fp32_score) {
    const auto f = golden("nemotron35_rerank.json");
    auto head = f["head"]["bf16"].get<std::vector<uint16_t>>();
    for (const char* name : host_libraries()) {
        Kernels kernels(name);
        for (const auto& c : f["cases"]) {
            std::vector<uint16_t> pooled;
            for (const auto& row : c["pooled"])
                for (const auto& v : row)
                    pooled.push_back(rad_f32_to_bf16(v.get<float>()));
            const int64_t M = (int64_t)c["pooled"].size();
            std::vector<float> output((size_t)M, 0);
            RadTensor ts[] = {tensor(pooled.data(), RAD_BF16, {M, 16}),
                              tensor(head.data(), RAD_BF16, {1, 16}),
                              tensor(output.data(), RAD_F32, {M, 1})};
            RadParam ps[] = {RAD_INT("M", M), RAD_INT("n_embd", 16), RAD_INT("n_vocab", 1),
                             RAD_STR("dtype", "bf16")};
            RadArgs a{};
            a.t = ts;
            a.n_t = 3;
            a.p = ps;
            a.n_p = 4;
            CHECK_OK(kernels.get("logits_gemm")->launch(&a, nullptr));
            for (int64_t i = 0; i < M; ++i) {
                const double want = c["head_only_scores"][i];
                CHECK_NEAR(output[(size_t)i], want, 1e-5 + 1e-5 * std::abs(want));
            }
        }
    }
}

RAD_TEST_MAIN()
