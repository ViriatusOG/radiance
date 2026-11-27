/* The vision-only extension, for Gemma 4 MoE and Inkling 2. The base container supplies the entire
 * decoder and drafter. The architecture's checkpoint maps are used only for the new encoder
 * weights, which are the checkpoint's bf16 tensors (an Inkling 2 tower's q and k rows permuted by
 * its row-piece maps). */
#include "iface.h"
#include "format/checkpoint.h"
#include "mm/processor.h"
#include "text/tokenizer.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>

namespace rad {
namespace {

bool vision_metadata(std::string_view k) {
    return (k.starts_with("vision_config.") && k != "vision_config._name_or_path") ||
           k.starts_with("processor.image_processor.") ||
           k.starts_with("processor.video_processor.") ||
           k == "vision_soft_tokens_per_image" || k == "image_token_id" ||
           k == "video_token_id" || k == "boi_token_id" || k == "eoi_token_id" ||
           k == "text_config.use_bidirectional_attention" || k == "use_bidirectional_attention";
}

std::string meta_value(const RadFile& f, const RadFileKV& kv) {
    if (kv.type == RAD_P_INT) return std::to_string(kv.v.i);
    if (kv.type == RAD_KV_F64) return fmt("%.17g", kv.v.f);
    return f.str(kv.v.s);
}

/* The temporary directory is private and beside the destination. A hard link publishes the
 * validated file atomically, failing even if the destination appeared after the initial check. */
struct PendingOutput {
    std::string dir, path;
    ~PendingOutput() {
        if (dir.empty()) return;
        ::unlink(path.c_str());
        ::unlink((path + ".blob.tmp").c_str());
        ::rmdir(dir.c_str());
    }
    int begin(const std::string& out) {
        std::string pattern = out + ".tmp-XXXXXX";
        if (!::mkdtemp(pattern.data())) {
            RAD_ERR("%s: cannot stage output: %s", out.c_str(), std::strerror(errno));
            return RAD_E_IO;
        }
        dir = std::move(pattern);
        path = dir + "/model.rad";
        return RAD_OK;
    }
};

struct VisionJob {
    RadWriter::Weight weight;
    const RadFileEntry* base = nullptr;
    const CkptWeight* vision = nullptr;
    int64_t index = -1;
};

int copy_vocab(RadWriter& writer, const RadFile& base, bool aux) {
    const VocabSection section = aux ? base.aux_vocab() : base.vocab();
    if (!section) return RAD_OK;
    if (section.h->flags & ~uint64_t(RAD_VOCAB_IGNORE_MERGES)) {
        RAD_ERR("--add-vision: the container's %svocabulary carries flags 0x%llx this tool "
                "does not copy", aux ? "auxiliary " : "", (unsigned long long)section.h->flags);
        return RAD_E_UNSUPPORTED;
    }
    VocabBuild vocab;
    RAD_TRY(vocab_deserialize(base.base(), base.size(), section.h, vocab));
    struct Sink : StringSink {
        RadWriter& writer;
        explicit Sink(RadWriter& w) : writer(w) {}
        rad_stroff intern(std::string_view s) override { return writer.intern_string(s); }
    } sink(writer);
    std::vector<uint8_t> blob;
    RAD_TRY(vocab_serialize(vocab, sink, 0, blob));
    return aux ? writer.set_aux_vocab_section(blob.data(), blob.size())
               : writer.set_vocab_section(blob.data(), blob.size());
}

}  /* namespace */

int rad_add_gemma4_vision(const std::string& base_path, const std::string& input,
                         const std::string& config_dir, const std::string& out,
                         const std::string& home, const std::vector<std::string>& hierarchy,
                         bool plan_only) {
    struct stat st{};
    if (::lstat(out.c_str(), &st) == 0 || errno != ENOENT) {
        RAD_ERR("%s: --add-vision requires a new output path", out.c_str());
        return RAD_E_INVAL;
    }

    RadFile base;
    RAD_TRY(base.open(base_path.c_str()));
    const RadFileHeader& h = base.header();
    /* The families this tool extends, each with its tower's vision_config.model_type. */
    const std::string arch = base.str(h.arch_id);
    const bool inkling = arch == "inkling2";
    const char* tower = inkling ? "inkling2_vision" : "gemma4_vision";
    if ((arch != "gemma4" && !inkling) || *base.str(h.quant) ||
        *base.get_s("radiance.encoder", "") || !base.vocab()) {
        RAD_ERR("--add-vision requires a text-only gemma4 MoE or inkling2 container with its "
                "original vocab");
        return RAD_E_UNSUPPORTED;
    }
    Checkpoint ck;
    if (!config_dir.empty()) ck.set_meta_dir(config_dir);
    RAD_TRY(ck.open(input));
    const MetaOwn& source = ck.meta();
    /* Inkling 2 ships its experts NVFP4 and MXFP8 (the source states a quant); its tower is bf16
     * all the same, which `provide` below insists on, weight by weight. */
    if (ck.is_gguf() || source.arch != arch || (!source.quant.empty() && !inkling) ||
        source.m.n_expert <= 0 ||
        std::string_view(rad_meta_gets(&source.m, "vision_config.model_type", "")) != tower) {
        RAD_ERR("--add-vision requires a %s safetensors source with %s config",
                inkling ? "Inkling 2" : "Gemma 4 MoE", tower);
        return RAD_E_UNSUPPORTED;
    }

    MetaOwn meta;
    meta.arch = base.str(h.arch_id);
    meta.name = base.str(h.model_name);
    meta.m = source.m;
    for (const auto& [key, expected] : std::initializer_list<std::pair<const char*, int64_t>>{
             {"n_layers", source.m.n_layers}, {"n_embd", source.m.n_embd},
             {"n_head", source.m.n_head}, {"n_head_kv", source.m.n_head_kv},
             {"head_dim", source.m.head_dim}, {"n_ff", source.m.n_ff},
             {"n_vocab", source.m.n_vocab}, {"n_expert", source.m.n_expert}}) {
        int64_t n = 0;
        if (!base.get_i(key, &n) || n <= 0 || n != expected) {
            RAD_ERR("--add-vision: base and source have different or missing '%s'", key);
            return RAD_E_INVAL;
        }
    }
    /* Inkling 2's config states no trained context (the container holds 0); the declare below
     * only needs a positive one to size its buffers. */
    if (!base.get_i("n_ctx_train", &meta.m.n_ctx_train) || meta.m.n_ctx_train < 0 ||
        (meta.m.n_ctx_train == 0 && !inkling))
        return RAD_E_FORMAT;
    base.get_i("n_expert_used", &meta.m.n_expert_used);
    base.get_i("n_expert_shared", &meta.m.n_expert_shared);
    double value = 0;
    if (base.get_f("rms_eps", &value)) meta.m.rms_eps = (float)value;
    if (base.get_f("rope_theta", &value)) meta.m.rope_theta = (float)value;
    for (int64_t i = 0; i < base.meta_count(); ++i) {
        const RadFileKV& kv = base.meta_at(i);
        const char* key = base.str(kv.key);
        if (kv.type != RAD_P_STR && kv.type != RAD_P_INT && kv.type != RAD_KV_F64) {
            RAD_ERR("--add-vision: cannot preserve the unknown metadata type of '%s'", key);
            return RAD_E_FORMAT;
        }
        if (std::strcmp(key, "radiance.encoder")) meta.kv(key, meta_value(base, kv));
    }
    std::vector<size_t> added_meta;
    for (size_t i = 0; i < source.keys.size(); ++i) {
        const std::string& key = source.keys[i];
        if (!vision_metadata(key)) continue;
        const char* old = meta.get(key);
        if (old && source.vals[i] != old) {
            RAD_ERR("--add-vision: metadata '%s' differs between the base and vision source",
                    key.c_str());
            return RAD_E_INVAL;
        }
        if (!old) {
            meta.kv(key, source.vals[i]);
            added_meta.push_back(i);
        }
    }
    meta.finish();
    mm::VisionConfig media;
    std::string why;
    if (media.from_meta(meta.m, &why) < 0) {
        RAD_ERR("--add-vision: invalid processor metadata: %s", why.c_str());
        return RAD_E_INVAL;
    }
    /* Gemma 4 numbers its media tokens in the config; Inkling 2's are vocabulary words the
     * engine resolves at startup (mm::Processor::init). */
    for (int32_t id : {media.tok_image, media.tok_video, media.tok_start, media.tok_end}) {
        if (inkling) break;
        if (id < 0 || (uint32_t)id >= base.vocab().h->n_tokens) {
            RAD_ERR("--add-vision: media token %d is outside the base vocabulary", id);
            return RAD_E_INVAL;
        }
    }

    LoadedPlugins plugins;
    RAD_TRY(rad_tools_load_plugins(home, hierarchy, meta.arch, meta.quant, &plugins));
    RadArchProbe probe{};
    RAD_TRY(rad_tools_probe(plugins, meta.m, &probe));
    if (probe.enc_merge <= 0 || probe.enc_merge > 16384) return RAD_E_INVAL;
    RadBuildCtx ctx{};
    ctx.world_size = 1;
    ctx.max_tok = 2048;
    ctx.max_seqs = 1;
    ctx.max_ctx = meta.m.n_ctx_train > 0 ? meta.m.n_ctx_train : 8192;
    ctx.max_spec = probe.draft_depth;
    ctx.max_enc_patches = 16384 - 16384 % probe.enc_merge;
    ctx.scope = "";

    const auto stored = rad_container_sources(base);
    const RadEncoding bf16 = rad_enc_plain(RAD_BF16);
    std::map<std::string, CkptWeight> vision;
    auto provide = [&](const std::string& name, const Program& p, WeightSource* ws) {
        if (!name.starts_with("v.")) return stored(name, p, ws);
        auto it = vision.find(name);
        if (it == vision.end()) {
            CkptWeight cw;
            std::string error;
            const int rc = ckpt_resolve(ck, p, name, false, &cw, &error);
            if (rc < 0) {
                RAD_ERR("--add-vision: %s: %s", name.c_str(), error.c_str());
                return rc;
            }
            if (!cw.trivial || !rad_enc_equal(&cw.enc, &bf16)) {
                RAD_ERR("--add-vision: '%s' must come from BF16 vision tensors", name.c_str());
                return (int)RAD_E_FORMAT;
            }
            it = vision.emplace(name, std::move(cw)).first;
        }
        ws->enc = it->second.enc;
        ws->rank = it->second.rank;
        std::copy_n(it->second.shape, RAD_MAX_RANK, ws->shape);
        return (int)RAD_OK;
    };
    Program prog;
    RAD_TRY(rad_tools_declare(plugins, meta.m, ctx, &prog, {}, nullptr, provide, false));
    if (prog.encoder_name != "vision" ||
        prog.encoder.modalities != (inkling ? RAD_MM_IMAGE : (RAD_MM_IMAGE | RAD_MM_VIDEO)) ||
        prog.encoder.n_embd != meta.m.n_embd || prog.encoder.patch_dim != media.patch_dim() ||
        prog.encoder.merge != (int64_t)media.merge * media.merge ||
        (int64_t)std::max(media.image_soft_tokens, media.video_soft_tokens) * prog.encoder.merge >
            prog.encoder.max_patches) {
        RAD_ERR("--add-vision: declared encoder does not match the processor metadata");
        return RAD_E_FORMAT;
    }
    for (size_t i = 1; i < prog.ops.size(); ++i) {
        const OpInfo& op = prog.ops[i];
        if (std::none_of(op.bands.begin(), op.bands.end(), [](const Band& b) {
                return b.dom[RAD_DOMAIN_HOST] || b.dom[RAD_DOMAIN_DEVICE];
            })) {
            RAD_ERR("--add-vision: no kernel resolved for '%s'", op.op.c_str());
            return RAD_E_UNSUPPORTED;
        }
    }

    std::vector<VisionJob> jobs;
    for (int64_t i = 0; i < base.entry_count(); ++i) {
        const RadFileEntry& e = base.entry(i);
        RadWriter::Weight w{base.str(e.name), base.str(e.quantizer), base.str(e.options),
                            base.encoding(e), e.rank};
        if (w.name.starts_with("v.")) {
            RAD_ERR("--add-vision: the base already contains vision weight '%s'", w.name.c_str());
            return RAD_E_INVAL;
        }
        std::copy_n(e.shape, RAD_MAX_RANK, w.shape);
        w.layer = e.layer; w.expert = e.expert; w.slot = e.slot;
        jobs.push_back({std::move(w), &e});
    }
    for (size_t i = 1; i < prog.weights.size(); ++i) {
        const WeightInfo& w = prog.weights[i];
        if (!w.source.starts_with("v.")) {
            const RadFileEntry* e = base.find(w.source);
            if (!e && !w.decl.optional) {
                RAD_ERR("--add-vision: base is missing '%s'", w.source.c_str());
                return RAD_E_NOTFOUND;
            }
            const auto& g = w.decl.group;
            if (e && (e->expert >= 0 || g.expert >= 0) &&
                (e->layer != g.layer || e->expert != g.expert || e->slot != g.slot)) {
                RAD_ERR("--add-vision: expert identity for '%s' disagrees with the declaration",
                        w.source.c_str());
                return RAD_E_FORMAT;
            }
            continue;
        }
        const auto it = vision.find(w.source);
        const CkptWeight* cw = it == vision.end() ? nullptr : &it->second;
        if (!cw || !w.enc_known || cw->rank != w.decl.rank ||
            !std::equal(cw->shape, cw->shape + cw->rank, w.decl.shape)) {
            RAD_ERR("--add-vision: missing or wrong shape for '%s'", w.source.c_str());
            return cw ? RAD_E_SHAPE : RAD_E_NOTFOUND;
        }
        const auto& g = w.decl.group;
        RadWriter::Weight new_weight{w.source, "", "", cw->enc, cw->rank};
        std::copy_n(cw->shape, RAD_MAX_RANK, new_weight.shape);
        new_weight.layer = g.layer; new_weight.expert = g.expert; new_weight.slot = g.slot;
        jobs.push_back({std::move(new_weight), nullptr, cw});
    }
    if (jobs.size() == (size_t)base.entry_count()) {
        RAD_ERR("--add-vision: the checkpoint has no vision weights the container lacks");
        return RAD_E_FORMAT;
    }
    RAD_INFO("--add-vision: %lld base weights to copy, %zu BF16 vision weights to add",
             (long long)base.entry_count(), jobs.size() - (size_t)base.entry_count());
    if (plan_only) return RAD_OK;

    std::sort(jobs.begin(), jobs.end(), [](const VisionJob& a, const VisionJob& b) {
        const auto key = [](const RadWriter::Weight& w) {
            return std::tie(w.layer, w.expert, w.slot, w.name);
        };
        return key(a.weight) < key(b.weight);
    });
    PendingOutput pending;
    RAD_TRY(pending.begin(out));
    RadWriter writer;
    RAD_TRY(writer.begin(pending.path.c_str()));
    writer.set_arch(meta.arch);
    writer.set_model_name(meta.name);
    writer.set_quant(meta.quant);
    writer.set_recipe(base.str(h.recipe));
    writer.set_created_by(base.str(h.created_by));
    for (int64_t i = 0; i < base.meta_count(); ++i) {
        const RadFileKV& kv = base.meta_at(i);
        const char* key = base.str(kv.key);
        if (!std::strcmp(key, "radiance.encoder")) continue;
        if (kv.type == RAD_P_INT) writer.meta_i(key, kv.v.i);
        else if (kv.type == RAD_KV_F64) writer.meta_f(key, kv.v.f);
        else writer.meta_s(key, base.str(kv.v.s));
    }
    for (size_t i : added_meta) writer.meta_s(source.keys[i], source.vals[i]);
    writer.meta_s("radiance.encoder", prog.encoder_name);
    for (int64_t i = 0; i < base.profile_count(); ++i) {
        const RadFileProfile& p = base.profile()[i];
        writer.add_profile(p.layer, p.expert, p.share);
    }
    RAD_TRY(copy_vocab(writer, base, false));
    RAD_TRY(copy_vocab(writer, base, true));
    for (VisionJob& job : jobs) {
        job.index = writer.add_weight(job.weight);
        if (job.index < 0) return (int)job.index;
    }
    RAD_TRY(writer.seal());
    constexpr int64_t chunk = 4 << 20;
    std::vector<uint8_t> buf((size_t)chunk);
    for (const VisionJob& job : jobs) {
        if (job.base) {
            for (uint32_t k = 0; k < job.base->n_planes; ++k) {
                const RadFilePlane& plane = base.plane(*job.base, (int)k);
                if (writer.plane_bytes(job.index, (int)k) != (int64_t)plane.bytes)
                    return RAD_E_FORMAT;
                for (int64_t off = 0; off < (int64_t)plane.bytes; off += chunk) {
                    const int64_t n = std::min(chunk, (int64_t)plane.bytes - off);
                    RAD_TRY(base.read_at(buf.data(), n, plane.offset + off));
                    RAD_TRY(writer.write_plane(job.index, (int)k, off, buf.data(), n));
                }
                base.drop_range(plane.offset, plane.bytes);
            }
        } else {
            const CkptWeight& cw = *job.vision;
            const int64_t rb = cw.cols * 2;
            const int64_t step = std::max<int64_t>(1, chunk / rb);
            for (int64_t r = 0; r < cw.rows; r += step) {
                const int64_t nr = std::min(step, cw.rows - r);
                buf.resize((size_t)std::max(chunk, nr * rb));
                RAD_TRY(ckpt_plane_rows(cw, 0, r, nr, buf.data(), &why));
                RAD_TRY(writer.write_plane(job.index, 0, r * rb, buf.data(), nr * rb));
            }
        }
    }
    RAD_TRY(writer.finish());
    RadFile result;
    RAD_TRY(result.open(pending.path.c_str()));
    if (result.entry_count() != (int64_t)jobs.size() ||
        std::strcmp(result.get_s("radiance.encoder", ""), "vision"))
        return RAD_E_FORMAT;
    if (::link(pending.path.c_str(), out.c_str()) != 0) {
        RAD_ERR("%s: cannot publish a new container: %s", out.c_str(), std::strerror(errno));
        return RAD_E_IO;
    }
    RAD_INFO("--add-vision: wrote %s", out.c_str());
    return RAD_OK;
}

}  /* namespace rad */
