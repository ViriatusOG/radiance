/* Native Wan 2.2 TI2V-5B video generation. */
#include "wan_denoiser.h"
#include "wan_vae_decoder.h"
#include "wan_vae_encoder.h"
#include "rad_unipc_math.h"

#include <cmath>
#include <cstdlib>

namespace wan {

using namespace rad::arch;

static constexpr int64_t kChannels = 48, kScale = 16, kMaxChunks = 31;
static constexpr int64_t kMaxArea = 1280 * 704, kMaxPlane = kMaxArea / (kScale * kScale);
static constexpr int64_t kMaxPatchRows = kMaxChunks * kMaxPlane / 4;
static constexpr int64_t kMaxVoxels = 4 * kMaxPatchRows;
static constexpr int64_t kMaxPassRows = kMaxPatchRows + TextConditioning::prompt_rows;
static constexpr int64_t kLanes = 1;

struct Model {
    Names names{};
    TextConditioning text;
    Denoiser denoiser;
    VaeEncoder encoder;
    VaeDecoder decoder;
    rad_buf video = 0, history = 0, velocities = 0, first = 0, noise_offsets = 0;
    rad_buf input_patches = 0, encode_cu = 0, mean = 0;
    rad_buf decode_geom = 0, denormalized = 0, decode_latent = 0, rgb = 0;
    rad_op patch_rgb = 0, make_encode_cu = 0, widen_mean = 0, normalize = 0;
    rad_op make_noise_cu = 0, noise = 0, clear_history = 0, pin_first = 0, sample = 0;
    rad_op make_decode_geom = 0, denormalize = 0, narrow_latent = 0;
};

static Model models[MAX_SLOTS];

static bool latent_stat(const char* values, bool positive) {
    if (!values) return false;
    for (int c = 0; c < kChannels; ++c) {
        char* end = nullptr;
        const double value = std::strtod(values, &end);
        if (end == values || !std::isfinite(value) || (positive && value <= 0.0)) return false;
        values = end;
    }
    while (*values == ' ') ++values;
    return *values == '\0';
}

static int check_format(const RadModelMeta* meta) {
    if (rad_meta_geti(meta, "transformer.num_layers", 0) != 30 ||
        rad_meta_geti(meta, "transformer.num_attention_heads", 0) != 24 ||
        rad_meta_geti(meta, "transformer.attention_head_dim", 0) != 128 ||
        rad_meta_geti(meta, "transformer.in_channels", 0) != kChannels ||
        rad_meta_geti(meta, "transformer.out_channels", 0) != kChannels ||
        rad_meta_geti(meta, "vae.z_dim", 0) != kChannels ||
        rad_meta_geti(meta, "vae.base_dim", 0) != 160 ||
        rad_meta_geti(meta, "vae.decoder_base_dim", 0) != 256 ||
        rad_meta_geti(meta, "vae.patch_size", 0) != 2 ||
        !latent_stat(rad_meta_gets(meta, "vae.latents_mean", nullptr), false) ||
        !latent_stat(rad_meta_gets(meta, "vae.latents_std", nullptr), true)) {
        std::fprintf(stderr, "radiance: wan: expected the Wan 2.2 TI2V-5B single-transformer package\n");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

static int64_t positions(int which, const RadVideoGeom* g, float* pos, int64_t capacity) {
    PatchGrid grid{};
    if (!g || !patch_grid(g->lat_frames, g->lat_h, g->lat_w, grid) ||
        grid.rows > kMaxPatchRows || g->lat_h * (int64_t)g->lat_w > kMaxPlane)
        return RAD_E_SHAPE;
    if (which == RAD_VPOS_DECODE) return (int64_t)g->lat_h * g->lat_w;
    if (which != RAD_VPOS_STEP || g->n_text < 0 || g->n_text > TextConditioning::prompt_rows)
        return RAD_E_SHAPE;
    const int64_t rows = grid.rows + g->n_text;
    if (!pos) return rows;
    if (capacity < rows) return RAD_E_SHAPE;
    const int64_t plane = grid.height * grid.width;
    for (int lane = 0; lane < 3; ++lane)
        std::fill_n(pos + lane * rows, g->n_text, 0.0f);
    for (int64_t r = 0; r < grid.rows; ++r) {
        const int64_t j = g->n_text + r;
        pos[j] = static_cast<float>(r / plane);
        pos[rows + j] = static_cast<float>(r % plane / grid.width);
        pos[2 * rows + j] = static_cast<float>(r % grid.width);
    }
    return rows;
}

static rad_op cast_channels(RadBuilder* b, rad_buf from, rad_buf to, int64_t rows,
                            const char* from_dtype, const char* to_dtype) {
    return rw(b, RAD_OP(b, "cast",
                       RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", kChannels),
                                  RAD_STR("from", from_dtype), RAD_STR("to", to_dtype)),
                       RAD_NOWEIGHTS), {from}, {to});
}

static rad_op affine_latent(RadBuilder* b, const RadModelMeta* meta, rad_buf from, rad_buf to,
                            const char* mode) {
    return rw(b, RAD_OP(b, "chan_affine",
                       RAD_PARAMS(RAD_RANGE("M", 1, kMaxPlane), RAD_INT("n", kChannels),
                                  RAD_STR("mul", rad_meta_gets(meta, "vae.latents_std", "")),
                                  RAD_STR("add", rad_meta_gets(meta, "vae.latents_mean", "")),
                                  RAD_STR("mode", mode), RAD_INT("round", 0), RAD_STR("dtype", "f32")),
                       RAD_NOWEIGHTS), {from}, {to});
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || rad_build_slot(ctx) >= MAX_SLOTS) return RAD_E_INVAL;
    if (ctx->is_draft || ctx->world_size != 1 || ctx->rank != 0) return RAD_E_UNSUPPORTED;
    RAD_ARCH_TRY(check_format(meta));
    if (ctx->max_tok < kMaxPassRows) return RAD_E_SHAPE;

    Model& m = models[rad_build_slot(ctx)];
    m = Model{};
    m.names = Names(ctx->scope);
    Names& nm = m.names;
    m.video = decl_b(b, nm.f("lane.video"), RAD_F32, {kMaxVoxels, kChannels}, RAD_BUF_PERSIST);
    m.history = decl_b(b, nm.f("lane.history"), RAD_F32, {kMaxVoxels, 3 * kChannels}, RAD_BUF_PERSIST);
    m.first = decl_b(b, nm.f("lane.first"), RAD_F32, {kMaxPlane, kChannels}, RAD_BUF_PERSIST);
    // The second forward reuses DiT scratch. Keep the first velocity out of that arena.
    m.velocities = decl_b(b, nm.f("dit.velocities"), RAD_F32, {2 * kMaxVoxels, kChannels}, RAD_BUF_PERSIST);
    m.noise_offsets = decl_b(b, nm.f("noise.offsets"), RAD_I32, {2});
    if (!m.video || !m.history || !m.first || !m.velocities || !m.noise_offsets) return RAD_E_INVAL;

    RAD_ARCH_TRY(m.text.declare(b, nm, kLanes));
    m.input_patches = decl_b(b, nm.f("vae.encode.input"), RAD_BF16, {kMaxPlane * 64, 12});
    m.encode_cu = decl_b(b, nm.f("vae.encode.offsets"), RAD_I32, {2});
    m.mean = decl_b(b, nm.f("vae.encode.mean"), RAD_F32, {kMaxPlane, kChannels});
    if (!m.input_patches || !m.encode_cu || !m.mean) return RAD_E_INVAL;
    m.patch_rgb = rw(b, RAD_OP(b, "pix_patch",
        RAD_PARAMS(RAD_RANGE("M", 1, 8 * kMaxPlane), RAD_INT("p", 2), RAD_INT("C", 3),
                   RAD_STR("order", "wh"), RAD_STR("dtype", "bf16")), RAD_NOWEIGHTS),
        {}, {m.input_patches});
    m.make_encode_cu = rw(b, RAD_OP(b, "int_affine",
        RAD_PARAMS(RAD_RANGE("M", 2, 2), RAD_STR("src", "0 1"),
                   RAD_STR("mul", "0 4"), RAD_STR("add", "0 0")), RAD_NOWEIGHTS), {}, {m.encode_cu});
    RAD_ARCH_TRY(m.encoder.declare(b, nm, kMaxPlane, m.input_patches, m.encode_cu, 30));
    m.widen_mean = cast_channels(b, m.encoder.output(), m.mean, kMaxPlane, "bf16", "f32");
    m.normalize = affine_latent(b, meta, m.mean, m.first, "inv");

    m.make_noise_cu = rw(b, RAD_OP(b, "int_affine",
        RAD_PARAMS(RAD_RANGE("M", 2, 2), RAD_STR("src", "0 1"),
                   RAD_STR("mul", "4 4"), RAD_STR("add", "0 0")), RAD_NOWEIGHTS), {}, {m.noise_offsets});
    m.noise = rw(b, RAD_OP(b, "randn_lanes",
        RAD_PARAMS(RAD_RANGE("M", 1, kMaxVoxels), RAD_INT("n", kChannels),
                   RAD_INT("stride", kMaxVoxels), RAD_STR("dtype", "f32")), RAD_NOWEIGHTS),
        {m.video, m.noise_offsets}, {m.video});
    m.clear_history = rw(b, RAD_OP(b, "fill",
        RAD_PARAMS(RAD_RANGE("M", 1, kMaxVoxels), RAD_INT("n", 3 * kChannels),
                   RAD_STR("dtype", "f32"), RAD_F64("value", 0.0)), RAD_NOWEIGHTS), {}, {m.history});
    m.pin_first = cast_channels(b, m.first, m.video, kMaxPlane, "f32", "f32");
    RAD_ARCH_TRY(m.denoiser.declare(b, nm, m.video, m.text.buffer(), m.velocities, kMaxPatchRows));
    m.sample = rw(b, RAD_OP(b, "flow_unipc_step",
        RAD_PARAMS(RAD_RANGE("M", 1, kMaxVoxels), RAD_INT("n", kChannels), RAD_STR("dtype", "f32")),
        RAD_NOWEIGHTS), {m.video, m.velocities, m.history}, {m.video, m.history});
    // The first-frame copy runs before the first forward and after every sampler update.
    // Both buffers it uses persist for the request, so the two issue sites share one declaration.

    m.decode_geom = decl_b(b, nm.f("vae.decode.geom"), RAD_I32, {3});
    m.denormalized = decl_b(b, nm.f("vae.decode.denormalized"), RAD_F32, {kMaxPlane, kChannels});
    m.decode_latent = decl_b(b, nm.f("vae.decode.latent"), RAD_BF16, {kMaxPlane, kChannels});
    m.rgb = decl_b(b, nm.f("vae.decode.out_rgb"), RAD_U8, {4 * kMaxArea, 3}, RAD_BUF_PERSIST);
    if (!m.decode_geom || !m.denormalized || !m.decode_latent || !m.rgb) return RAD_E_INVAL;
    m.make_decode_geom = rw(b, RAD_OP(b, "int_affine",
        RAD_PARAMS(RAD_RANGE("M", 3, 3), RAD_STR("src", "-1 1 2"),
                   RAD_STR("mul", "0 1 1"), RAD_STR("add", "1 0 0")), RAD_NOWEIGHTS), {}, {m.decode_geom});
    m.denormalize = affine_latent(b, meta, m.video, m.denormalized, "fma");
    m.narrow_latent = cast_channels(b, m.denormalized, m.decode_latent, kMaxPlane, "f32", "bf16");
    RAD_ARCH_TRY(m.decoder.declare(b, nm, kMaxPlane, kLanes, m.decode_latent, m.decode_geom, grp_layer(31)));
    RAD_ARCH_TRY(m.decoder.declare_rgb(b, nm, m.rgb));

    if (!m.patch_rgb || !m.make_encode_cu || !m.widen_mean || !m.normalize ||
        !m.make_noise_cu || !m.noise || !m.clear_history || !m.pin_first || !m.sample ||
        !m.make_decode_geom || !m.denormalize || !m.narrow_latent) return RAD_E_NOKERNEL;

    RadVideoGenDecl d{};
    d.name = "wan-2.2-ti2v-5b";
    d.tmpl = "{prompt}</s>";
    d.max_text = TextConditioning::prompt_rows;
    d.fps_num = 24;
    d.fps_den = 1;
    d.frames_per_chunk = 4;
    d.frames_tail = -3;
    d.latents_per_chunk = d.dec_latents = 1;
    d.min_chunks = 1;
    d.max_chunks = kMaxChunks;
    d.latent_scale = kScale;
    d.latent_channels = kChannels;
    d.patch = 2;
    d.size_multiple = d.min_side = 32;
    d.max_area = kMaxArea;
    d.max_aspect = 4;
    d.default_width = 512;
    d.default_height = 288;
    d.default_chunks = 13;
    d.default_steps = 50;
    d.max_steps = 100;
    d.sched_kind = RAD_VIDEO_SCHEDULE_UNIPC;
    d.shift_video = 5.0;
    d.step_text_rows = TextConditioning::prompt_rows;
    d.guidance = RAD_VIDEO_GUIDANCE_CFG;
    // Wan-Video/Wan2.2 at 1ea34ff4, wan/configs/shared_config.py.
    d.default_negative = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
                         "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
                         "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
                         "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";
    d.guidance_default[0] = 5.0;
    d.guidance_default[2] = 1.0;
    d.image_cond = 1;
    d.max_lanes = kLanes;
    d.frames_out = m.rgb;
    return rad_declare_video_generator(b, &d);
}

static void step(RadCtx* c, const RadBatch* batch) {
    if (!batch || !batch->vgen_b) {
        rad_step_fail(c, "wan: expected a video generator pass");
        return;
    }
    const auto& m = models[rad_slot(c)];
    const auto& v = *batch->vgen_b;
    PatchGrid grid{};
    if (v.lane != 0 || !v.geom_d ||
        !patch_grid(v.geom.lat_frames, v.geom.lat_h, v.geom.lat_w, grid) ||
        grid.rows > kMaxPatchRows || (int64_t)v.geom.lat_h * v.geom.lat_w > kMaxPlane) {
        rad_step_fail(c, "wan: invalid video pass geometry");
        return;
    }
    const int64_t h = v.geom.lat_h, w = v.geom.lat_w, plane = h * w, voxels = 4 * grid.rows;
    const auto geom = praw(v.geom_d + RAD_VGEOM_LAT_FRAMES, RAD_I32, 3);
    const auto lat_cu = praw(v.lat_cu, RAD_I32, 2);
    const auto state = brows(m.video, voxels);
    const auto clean_first = brows(m.first, plane);
    const auto first_frame = brows(m.video, plane);
    switch (batch->gen) {
    case RAD_GEN_PASS_VCOND:
        m.text.step(c, *batch);
        if (!v.has_image) return;
        if (!v.image || !v.lat_cu) {
            rad_step_fail(c, "wan: missing first-frame conditioning pixels");
            return;
        }
        RAD_ISSUE_N(c, m.patch_rgb, h * 8, praw2(v.image, RAD_BF16, h * 16, w * 16 * 3),
                    praw2(rad_buf_ptr(c, m.input_patches), RAD_BF16, h * 8, w * 8 * 12));
        RAD_ISSUE_N(c, m.make_encode_cu, 2, praw(v.geom_d + RAD_VGEOM_FIRST_CU, RAD_I32, 2),
                    brows(m.encode_cu, 2));
        if (m.encoder.step(c, h, w) != RAD_OK) return;
        RAD_ISSUE_N(c, m.widen_mean, plane, m.encoder.mean(plane), brows(m.mean, plane));
        RAD_ISSUE_N(c, m.normalize, plane, brows(m.mean, plane), clean_first);
        return;

    case RAD_GEN_PASS_VSTEP: {
        if (!v.lat_cu || !v.sched || !v.guidance_h || !v.guidance ||
            (v.step == 0 && (!v.seed || !v.first_d || !v.lane_d))) {
            rad_step_fail(c, "wan: missing denoising pass fields");
            return;
        }
        const auto history = brows(m.history, voxels);
        if (v.step == 0) {
            RAD_ISSUE_N(c, m.make_noise_cu, 2, lat_cu, brows(m.noise_offsets, 2));
            RAD_ISSUE_N(c, m.noise, voxels, praw(v.first_d, RAD_I32, 1), praw(v.seed, RAD_U32, 2),
                        praw(v.lane_d, RAD_I32, 1), brows(m.noise_offsets, 2), RAD_B(m.video));
            RAD_ISSUE_N(c, m.clear_history, voxels, history);
            if (v.has_image) RAD_ISSUE_N(c, m.pin_first, plane, clean_first, first_frame);
        }
        const auto positive = brows(m.velocities, voxels);
        const bool guided = v.guidance_h[0] != 1.0f;
        const auto negative = guided
            ? brow_slice(m.velocities, kMaxVoxels, voxels, kChannels) : RAD_NONE;
        if (m.denoiser.step(c, v, state, m.text.context(0, 0), positive) != RAD_OK) return;
        if (guided && m.denoiser.step(c, v, state, m.text.context(0, 1), negative) != RAD_OK) return;
        RAD_ISSUE_N(c, m.sample, voxels, positive,
                    praw(v.sched + RAD_VIDEO_UNIPC_COEFF_OFFSET, RAD_F32, rad_unipc::COUNT),
                    history, state, negative, guided ? praw(v.guidance, RAD_F32, 1) : RAD_NONE);
        if (v.has_image) RAD_ISSUE_N(c, m.pin_first, plane, clean_first, first_frame);
        return;
    }

    case RAD_GEN_PASS_VDECODE:
        if (v.chunk < 0 || v.chunk >= v.geom.lat_frames ||
            v.n_frames_out != (v.chunk == 0 ? 1 : 4)) {
            rad_step_fail(c, "wan: invalid decoder chunk");
            return;
        }
        RAD_ISSUE_N(c, m.make_decode_geom, 3, geom, brows(m.decode_geom, 3));
        RAD_ISSUE_N(c, m.denormalize, plane, brow_slice(m.video, v.chunk * plane, plane, kChannels),
                    brows(m.denormalized, plane));
        RAD_ISSUE_N(c, m.narrow_latent, plane, brows(m.denormalized, plane), brows(m.decode_latent, plane));
        m.decoder.step(c, h, w, v.lane, v.chunk);
        return;

    default:
        rad_step_fail(c, "wan: unsupported video generator pass");
    }
}

static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    RAD_ARCH_TRY(check_format(meta));
    out->gen_rows = kMaxPassRows;
    out->gen_video = 1;
    return RAD_OK;
}

}  // namespace wan

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(wan)
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(wan, "wan", "", "0.1.0", "Wan 2.2 TI2V-5B text-to-video and image-to-video")

extern "C" int64_t rad_arch_video_positions(int which, const RadVideoGeom* g, float* pos, int64_t cap) {
    return wan::positions(which, g, pos, cap);
}
#endif
