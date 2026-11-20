/* Deployment bounds without allocating weights or touching a GPU. */
#define RAD_ARCH_NO_EXPORTS 1
#include "arch_fakes.h"
#include "../arch/minimaxh3/minimaxh3.cpp"
#include "../arch/minimaxh3/rad_ref2va_pack.cpp"
#include "../arch/minimaxh3/rad_vae_geom.cpp"

TEST(h3_output_limits_cover_every_accepted_canvas) {
    using namespace minimaxh3;
    for (const int64_t requested : {int64_t(256 * 256), int64_t(864 * 480), int64_t(kMaxArea), INT64_MAX}) {
        const int64_t area = video_area(requested);
        REQUIRE(area <= kMaxArea);
        const int64_t cap = max_tiles(area);
        REQUIRE(cap > 0);
        for (int64_t width = kMinSide; width <= area / kMinSide; width += kMult) {
            for (int64_t height = kMinSide; height <= area / width; height += kMult) {
                if (!size_ok(width, height, area)) continue;
                CHECK(tiles_of(height / kScale, width / kScale) <= cap);
                CHECK((height / kMult) * (width / kMult) * max_lat_frames() <= max_video_rows(area));
            }
        }
    }
    CHECK_EQ(video_area(0), kMaxArea);
    CHECK_EQ(video_area(-1), kMaxArea);
    CHECK(max_tiles(864 * 480) < max_tiles(kMaxArea));
}

TEST(h3_shorter_clips_reduce_the_required_step_budget) {
    using namespace minimaxh3;
    for (int64_t chunks = kMinChunks; chunks < kMaxChunks; ++chunks)
        CHECK(max_step_rows(864 * 480, chunks) < max_step_rows(864 * 480, chunks + 1));
}

TEST(h3_probe_reports_deployment_bounds_and_rejects_impossible_canvases) {
    RadModelMeta meta{};
    RadArchProbe baseline{};
    REQUIRE_EQ(minimaxh3::probe(&meta, &baseline), RAD_OK);
    CHECK_EQ(baseline.gen_video, 1);
    CHECK(baseline.gen_rows > 0);
    RadArchProbe capped{};
    capped.gen_max_pixels = 864 * 480;
    REQUIRE_EQ(minimaxh3::probe(&meta, &capped), RAD_OK);
    CHECK(capped.gen_rows < baseline.gen_rows);
    // The planner must cover complete video passes, independently of tensor-parallel rank count.
    CHECK(capped.gen_rows >= minimaxh3::max_tiles(864 * 480) * minimaxh3::kPeriod);
    CHECK(capped.gen_rows >= minimaxh3::max_step_rows(864 * 480));
    for (const int64_t area : {int64_t(1), int64_t(256 * 256 - 1)}) {
        RadArchProbe impossible{};
        impossible.gen_max_pixels = area;
        CHECK_EQ(minimaxh3::probe(&meta, &impossible), RAD_E_INVAL);
    }
    RadArchProbe excessive{};
    excessive.gen_max_pixels = INT64_MAX;
    REQUIRE_EQ(minimaxh3::probe(&meta, &excessive), RAD_OK);
    CHECK_EQ(excessive.gen_rows, baseline.gen_rows);
    CHECK_EQ(minimaxh3::probe(nullptr, &baseline), RAD_E_INVAL);
    CHECK_EQ(minimaxh3::probe(&meta, nullptr), RAD_E_INVAL);
}
namespace {
RadVideoGenDecl recorded_video{};
int recorded_video_calls = 0;

std::string repeated_values(int count, int value) {
    std::string out = "[";
    for (int i = 0; i < count; ++i) out += (i ? "," : "") + std::to_string(value);
    return out + "]";
}
MetaKV h3_fixture() {
    RadModelMeta base{};
    base.arch_id = "minimaxh3"; base.name = "synthetic-h3-limits";
    base.n_layers = 51; base.n_embd = 256; base.n_head = 2; base.n_head_kv = 2;
    base.head_dim = 128; base.n_ff = 256; base.n_vocab = 256;
    base.n_ctx_train = 32768; base.rms_eps = 1e-6; base.rope_theta = 10000;
    MetaKV meta(base);
    meta.set("pipeline._class_name", "MiniMaxH3Pipeline")
        .set("transformer.in_channels", "24").set("transformer.audio_in_channels", "32")
        .set("transformer.rope_freq_dim", "16").set("transformer.num_layers", "1")
        .set("transformer.hidden_size", "256").set("transformer.text_dim", "256").set("transformer.num_attention_heads", "2")
        .set("transformer.attention_head_dim", "128").set("transformer.ffn_dim", "256")
        .set("vae.latent_channels", "24").set("vae.clip_length", "17").set("vae.token_drop", "3")
        .set("vae.decoder_num_register_tokens", "4").set("vae.decoder_num_layers", "1")
        .set("vae.block_out_channels", "[128,256,512,512,1024,1024]")
        .set("vae.spatial_downsample_factors", "[1,2,2,2,2,1]")
        .set("vae.temporal_downsample_factors", "[1,1,2,2,1,1]")
        .set("audio_vae.latent_channels", "32").set("audio_vae.sampling_rate", "32000")
        .set("audio_vae.decoder_rates", "[2,2,2,2,2,5,5]")
        .set("audio_vae.decoder_kernel_sizes", "[4,4,4,4,4,10,10]")
        .set("audio_vae.resblock_kernel_sizes", "[3,7,11]");
    meta.set("vae.latents_mean", repeated_values(24, 0).c_str());
    meta.set("vae.latents_std", repeated_values(24, 1).c_str());
    meta.set("audio_vae.latents_mean", repeated_values(32, 0).c_str());
    meta.set("audio_vae.latents_std", repeated_values(32, 1).c_str());
    return meta;
}
void h3_encodings(RadBuilder& builder) {
    builder.encs.push_back({"dit.blk.", rad_enc_affine(RAD_I8, RAD_BF16, 1, 128)});
    builder.encs.push_back({"venc.cin.weight", rad_enc_plain(RAD_F32)});
}
const RecBuf* named_buffer(const RadBuilder& builder, const char* name) {
    for (const auto& buffer : builder.bufs) if (buffer.name == name) return &buffer;
    return nullptr;
}
}
extern "C" int rad_declare_video_generator(RadBuilder*, const RadVideoGenDecl* d) {
    if (!d) return RAD_E_INVAL;
    recorded_video = *d;
    // Only scalar geometry is inspected after declare; its borrowed strings are not retained.
    recorded_video.name = recorded_video.tmpl = nullptr;
    recorded_video.rungs = nullptr;
    ++recorded_video_calls;
    return RAD_OK;
}

TEST(h3_declaration_preserves_defaults_and_caps_output_without_capping_reference_encoder) {
    auto meta = h3_fixture();
    RadBuilder baseline;
    h3_encodings(baseline);
    RadBuildCtx full = build_ctx();
    full.max_tok = 1000000;
    const int before = recorded_video_calls;
    REQUIRE_EQ(minimaxh3::declare(&baseline, meta.meta(), &full), RAD_OK);
    REQUIRE_EQ(recorded_video_calls, before + 1);
    CHECK_EQ(recorded_video.default_width, 1344);
    CHECK_EQ(recorded_video.default_height, 768);
    CHECK_EQ(recorded_video.default_chunks, 7);
    CHECK_EQ(recorded_video.max_chunks, 20);
    const RecBuf* full_ref = named_buffer(baseline, "venc.tplan");
    const RecBuf* full_decode = named_buffer(baseline, "vae.x");
    REQUIRE(full_ref); REQUIRE(full_decode);
    const int64_t ref_tiles = full_ref->d.shape[0], full_decode_rows = full_decode->d.shape[0];
    RadBuilder capped;
    h3_encodings(capped);
    RadBuildCtx small = full;
    small.gen_max_pixels = 864 * 480;
    REQUIRE_EQ(minimaxh3::declare(&capped, meta.meta(), &small), RAD_OK);
    CHECK(recorded_video.default_width * recorded_video.default_height <= small.gen_max_pixels);
    CHECK_EQ(recorded_video.max_area, small.gen_max_pixels);
    CHECK_EQ(recorded_video.default_chunks, 7);
    CHECK_EQ(recorded_video.max_chunks, 20);
    const RecBuf* small_ref = named_buffer(capped, "venc.tplan");
    const RecBuf* small_decode = named_buffer(capped, "vae.x");
    REQUIRE(small_ref); REQUIRE(small_decode);
    CHECK_EQ(small_ref->d.shape[0], ref_tiles);
    CHECK(small_decode->d.shape[0] < full_decode_rows);
}

TEST(h3_declaration_obeys_chunk_thresholds_on_both_tensor_parallel_ranks) {
    auto meta = h3_fixture();
    const int64_t area = 864 * 480;
    const int64_t threshold = minimaxh3::max_step_rows(area, 10);
    for (const int rank : {0, 1}) {
        RadBuilder builder;
        h3_encodings(builder);
        auto ctx = build_ctx(rank, 2);
        ctx.gen_max_pixels = area;
        ctx.max_tok = threshold;
        REQUIRE_EQ(minimaxh3::declare(&builder, meta.meta(), &ctx), RAD_OK);
        CHECK_EQ(recorded_video.max_chunks, 10);
        const auto* decoder = named_buffer(builder, "vae.x");
        const auto* rank_pos = named_buffer(builder, "vae.pos");
        REQUIRE(decoder); REQUIRE(rank_pos);
        CHECK_EQ(decoder->d.shape[0], minimaxh3::max_tiles(area) * minimaxh3::kPeriod);
        CHECK_EQ(rank_pos->d.shape[1], ((minimaxh3::max_tiles(area) + 1) / 2) * minimaxh3::kPeriod);
        RadBuilder below;
        h3_encodings(below);
        --ctx.max_tok;
        REQUIRE_EQ(minimaxh3::declare(&below, meta.meta(), &ctx), RAD_OK);
        CHECK_EQ(recorded_video.max_chunks, 9);
    }
    RadBuilder insufficient;
    h3_encodings(insufficient);
    auto ctx = build_ctx(0, 2);
    ctx.gen_max_pixels = area;
    ctx.max_tok = minimaxh3::max_tiles(area) * minimaxh3::kPeriod - 1;
    REQUIRE(ctx.max_tok >= minimaxh3::max_step_rows(area, minimaxh3::kMinChunks));
    CHECK_EQ(minimaxh3::declare(&insufficient, meta.meta(), &ctx), RAD_E_INVAL);
}
/* A song guide is a keyframe of the soundtrack alone, after the first and last frames: the plugin
 * lays it out, reads it from the guide's own soundtrack slot, and takes one. */
TEST(h3_a_song_guide_keyframe_reads_its_own_soundtrack_slot) {
    using namespace minimaxh3;
    const RadVideoGeom g{ 64, 124, 37, 48, 84, 207 };   /* n_text, frames, latents, sound */
    /* A standalone soundtrack reference, then a first frame and the guide at frame 100, no pads. */
    std::vector<int32_t> tab = { 1, 2, RAD_VREF_AUDIO, 1, 0, 0, 0, 40, 0, 1, 0, 100, 0, 40, 0 };
    Ref2VAPack p;
    REQUIRE(ref_pack_of(&g, tab.data(), &p, false));
    std::vector<RefPart> parts;
    ref_parts(tab.data(), &parts);
    REQUIRE_EQ(parts.size(), (size_t)3);
    CHECK_EQ(parts[0].audio, 0);
    CHECK_EQ(parts[0].row, venc_slot_row(RAD_ENC_KEYFRAME, 1));
    CHECK_EQ(parts[1].audio, 1);
    CHECK_EQ(parts[1].row, (int64_t)6 * kEncSlotRowsA);
    CHECK_EQ(parts[2].row, aenc_slot_row(RAD_ENC_AUDIO, 1));
    /* The guide's slot is past the six reference soundtracks' and inside the buffer. */
    CHECK(aenc_slot_row(RAD_ENC_GUIDE, 1) + kEncSlotRowsA <= (int64_t)kEncSlotsA * kEncSlotRowsA);
    CHECK_EQ(aenc_slot_row(RAD_ENC_GUIDE, 2), (int64_t)-1);
    /* A second soundtrack keyframe has no slot. */
    tab = { 0, 2, 0, 0, 40, 100, 0, 40, 0 };
    CHECK(!ref_pack_of(&g, tab.data(), &p, false));
}

RAD_TEST_MAIN()
