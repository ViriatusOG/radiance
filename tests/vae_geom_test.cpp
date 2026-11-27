/* vae_geom_test.cpp -- the encode geometry (arch/minimaxh3/rad_vae_geom.cpp) against golden vectors
 * produced by the reference VAEs themselves.
 *
 * The TU is pure host arithmetic, so it is compiled straight in here -- no rad_core, no device.
 * tests/data/vae_geom/golden.json comes from gen_golden.py, which runs the reference's encoders on
 * CPU (video: EncoderFCN3D and the wrapper's own encode(), encode_temporal() and tiled_encode()
 * with the reference's constructor arguments; audio: the DAC encoder). The latent counts below are
 * the networks' own output shapes and the constants the reference's own tables, so a mismatch is
 * the port disagreeing with the model.
 */
#include "rad_test.h"
#include "rad_vae_geom.h"

#include "util/json.h"

using Json = rad::Json;

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace minimaxh3;

namespace {

Json& golden() {
    static Json g = [] {
        std::ifstream f(std::string(RAD_VAE_GOLDEN) + "/golden.json");
        if (!f.is_open()) {
            fprintf(stderr, "  FAIL golden.json missing -- regenerate with "
                            "tests/data/vae_geom/gen_golden.py\n");
            std::exit(1);
        }
        return Json::parse(f);
    }();
    return g;
}

}   /* namespace */

TEST(vae_encode_latent_counts) {
    /* Every frame count goes through the reference encode() end to end: chunking, the padded last
     * clip, the trailing drop. The spatial entries pin the ratio (generated at 32x32) and the
     * latent width pins z_channels. */
    for (const auto& c : golden()["video"]["encode"]) {
        const int64_t t = c["T"];
        const auto& z = c["z_shape"];
        CHECK_EQ(vae_latent_frames(t), z[2].get<int64_t>());
        CHECK_EQ(z[1].get<int64_t>(), 24);
        CHECK_EQ(z[3].get<int64_t>() * kVaeRatio, 32);
        CHECK_EQ(z[4].get<int64_t>() * kVaeRatio, 32);
    }
}

TEST(vae_clip_chain_composes_with_encode) {
    /* A full clip's latents times the clip count, less the drop, is the encode count; and a raw
     * clip through the causal chain (the golden's tiled_encode cases: no chunking, no drop) has
     * vae_clip_latents of its pixel frames. */
    for (const auto& c : golden()["video"]["encode"]) {
        const int64_t t = c["T"];
        if (t < 2) continue;
        const int64_t clips = (t + kVaeClipLength - 1) / kVaeClipLength;
        CHECK_EQ(clips * vae_clip_latents(kVaeClipLength) - kVaeTokenDrop, c["z_shape"][2].get<int64_t>());
    }
    for (const auto& c : golden()["video"]["tiled_encode"])
        CHECK_EQ(vae_clip_latents(c["x_shape"][2].get<int64_t>()), c["moments_shape"][2].get<int64_t>());
    /* A single frame truncates its temporal taps instead of padding them. */
    CHECK_EQ(vae_clip_latents(1), 1);
}

TEST(vae_audio_latents_match_encoder) {
    /* The wrapper right-pads to the hop and the stride chain divides exactly: the reference
     * encoder's own frame count is ceil(samples / 800). */
    for (const auto& c : golden()["audio"]["latents"]) {
        const int64_t s = c["samples"];
        CHECK_EQ(vae_audio_latents(s), c["T"].get<int64_t>());
        CHECK_EQ(vae_audio_latents(s) * kAudioHop, c["padded"].get<int64_t>());
    }
}

TEST(vae_constants_are_the_reference_tables) {
    /* The de-normalization the latents carry: doubles compare bit for bit, since the golden
     * carries the same decimal literals. */
    const auto& k = golden()["video"]["constants"];
    const auto mean = k["latents_mean"].get<std::vector<double>>();
    const auto stdv = k["latents_std"].get<std::vector<double>>();
    REQUIRE_EQ(mean.size(), 24u);
    REQUIRE_EQ(stdv.size(), 24u);
    for (size_t i = 0; i < mean.size() && i < 24; ++i) {
        CHECK(kVaeLatentsMean[i] == mean[i]);
        CHECK(kVaeLatentsStd[i] == stdv[i]);
    }
    const auto pm = k["pixel_mean"].get<std::vector<double>>();
    const auto ps = k["pixel_std"].get<std::vector<double>>();
    for (size_t i = 0; i < 3; ++i) {
        CHECK(kVaePixelMean[i] == pm[i]);
        CHECK(kVaePixelStd[i] == ps[i]);
    }
}

TEST(vae_geom_rejects_nonsense) {
    CHECK_EQ(vae_latent_frames(0), -1);
    CHECK_EQ(vae_latent_frames(-5), -1);
    CHECK_EQ(vae_clip_latents(0), -1);
    CHECK_EQ(vae_audio_latents(-1), -1);
    CHECK_EQ(vae_audio_latents(0), 0);
}

RAD_TEST_MAIN()
