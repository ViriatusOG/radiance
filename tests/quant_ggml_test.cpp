/* quant_ggml_test.cpp -- libquant's `ggml` quantiser through the plugin registry, read back
 * through the core's decoder.
 *
 * The quantiser claims its codes and scales are llama.cpp's own, at the commit vendor/UPSTREAM
 * names (06938ac12, ggml-quants.c built as ggml-base builds it): the same weights and imatrix
 * through llama.cpp's quantize_<type> and through this port give identical GGUF blocks for all
 * nineteen types, and llama.cpp's dequantize_row_<type> equals enc_decode_rows of these planes
 * value for value. llama.cpp's quantisers are not in this build, so bit_exact_against_llama_cpp
 * pins hashes of those planes for this file's inputs; any code or scale that moves fails there.
 * The error bounds carry margin and catch a wrong plane layout, which decodes to noise.
 */
#include "quant_fixture.h"
#include "format/ggml_planes.h"

#include <cstdint>

using namespace rad;
using namespace qfix;

namespace {

struct Type {
    const char* name;
    const char* enc;        /* the canonical spelling of its encoding */
    bool        needs_im;   /* llama.cpp's search has no path without an imatrix */
    int64_t     block;      /* values one GGUF block covers */
    double      bound;      /* rel_err on every_type_decodes_within_its_bound's weight */
};

/* Bounds are 1.3x the larger of the plain and imatrix errors on that weight (its first 8 rows for
 * the iq1_* and iq2_* types, all 32 for the rest). Its 0.2% outliers
 * at 10x widen every block they land in, hence q4_0's error of 0.11 rather than the 0.07 a
 * clean Gaussian gives. */
const Type kTypes[] = {
    { "q4_0", "i4*f16[1x32]", false, 32, 0.141 },
    { "q4_1", "affine:codes=u4[1x1],scale=f16[1x32],min=f16[1x32]", false, 32, 0.110 },
    { "q5_0", "i5*f16[1x32]", false, 32, 0.073 },
    { "q5_1", "affine:codes=u5[1x1],scale=f16[1x32],min=f16[1x32]", false, 32, 0.053 },
    { "q8_0", "i8*f16[1x32]", false, 32, 0.0090 },
    { "q2_K", "affine:codes=u2[1x1],scale=u4[1x16],scale.1=f16[1x256],min=u4[1x16],"
              "min.1=f16[1x256]", false, 256, 0.407 },
    { "q3_K", "i3*i6[1x16]*f16[1x256]", false, 256, 0.223 },
    { "q4_K", "affine:codes=u4[1x1],scale=u6[1x32],scale.1=f16[1x256],min=u6[1x32],"
              "min.1=f16[1x256]", false, 256, 0.103 },
    { "q5_K", "affine:codes=u5[1x1],scale=u6[1x32],scale.1=f16[1x256],min=u6[1x32],"
              "min.1=f16[1x256]", false, 256, 0.052 },
    { "q6_K", "i6*i8[1x16]*f16[1x256]", false, 256, 0.028 },
    { "iq4_nl", "affine:codes=u4[1x1],table=i8{1x16},scale=f16[1x32]", false, 32, 0.116 },
    { "iq4_xs", "affine:codes=u4[1x1],table=i8{1x16},scale=i6[1x32],scale.1=f16[1x256]", false,
      256, 0.118 },
    { "iq2_xxs", "affine:codes=u8[1x8],grid=u8{256x8},signs=u1[1x1],scale=u5[1x32],"
                 "scale.1=f16[1x256],scale.2=f32[*x*]", true, 256, 0.558 },
    { "iq2_xs", "affine:codes=u9[1x8],grid=u8{512x8},signs=u1[1x1],scale=u5[1x16],"
                "scale.1=f16[1x256],scale.2=f32[*x*]", true, 256, 0.468 },
    { "iq2_s", "affine:codes=u10[1x8],grid=u8{1024x8},signs=u1[1x1],scale=u5[1x16],"
               "scale.1=f16[1x256],scale.2=f32[*x*]", false, 256, 0.429 },
    { "iq3_xxs", "affine:codes=u8[1x4],grid=u8{256x4},signs=u1[1x1],scale=u5[1x32],"
                 "scale.1=f16[1x256],scale.2=f32[*x*]", false, 256, 0.318 },
    { "iq3_s", "affine:codes=u9[1x4],grid=u8{512x4},signs=u1[1x1],scale=u5[1x32],"
               "scale.1=f16[1x256]", false, 256, 0.242 },
    { "iq1_s", "affine:codes=u11[1x8],grid=i8{2048x8},scale=u4[1x32],scale.1=f16[1x256],"
               "zero=bf16[1x32]", true, 256, 0.751 },
    { "iq1_m", "affine:codes=u11[1x8],grid=i8{2048x8},scale=u4[1x16],scale.1=f16[1x256],"
               "zero=bf16[1x8]", false, 256, 0.770 },
};

/* llama.cpp's iq1_* and iq2_* types, whose searches are exhaustive over a neighbour grid. */
bool grid_search(const Type& t) {
    return !std::strncmp(t.name, "iq1_", 4) || !std::strncmp(t.name, "iq2_", 4);
}

/* One `ggml` type through quant_fixture.h's quantise, `blk` rows per call (0: the quantiser's
 * own row_block, one row for `ggml`). */
Quantised quantise_ggml(const char* type, int64_t rows, int64_t cols, const std::vector<float>& w,
                        const float* imp, int64_t blk = 0) {
    Opts o;
    o.s("type", type);
    return quantise(quantizer("ggml"), o, "blk.0.ffn_down.weight", rows, cols, w, imp, blk);
}

/* A column importance as an imatrix holds one: positive, and uneven enough to steer a search. */
std::vector<float> importance(int64_t cols, uint64_t seed) {
    const std::vector<float> d = draw(cols, seed);
    std::vector<float> imp((size_t)cols);
    for (int64_t c = 0; c < cols; ++c) imp[(size_t)c] = 0.05f + 40.0f * std::fabs(d[(size_t)c]);
    return imp;
}

uint64_t fnv1a(const Quantised& q) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (const auto& p : q.planes) h = fnv(p, h);
    return h;
}

/* The relative error under the imatrix's weighting: what a search given the imatrix minimises. */
double weighted_err(const std::vector<float>& w, const std::vector<float>& d,
                    const std::vector<float>& imp, int64_t cols) {
    double num = 0, den = 0;
    for (size_t i = 0; i < w.size(); ++i) {
        const double wi = imp[i % (size_t)cols], e = (double)w[i] - d[i];
        num += wi * e * e;
        den += wi * (double)w[i] * w[i];
    }
    return std::sqrt(num / den);
}

int encoding_status(const char* type, int64_t rows, int64_t cols, uint32_t rank = 2) {
    const RadQuantizerInfo* q = quantizer("ggml");
    Opts o;
    if (type) o.s("type", type);
    RadQuantWeight qw = weight_of("blk.0.attn_q.weight", rows, cols, nullptr);
    if (rank == 1) {
        qw.rank = 1;
        qw.shape[0] = cols;
    }
    RadEncoding e;
    return q->encoding(o.data(), o.n(), &qw, &e);
}

}  /* namespace */

TEST(ggml_writes_each_type_as_its_affine_planes) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* q = quantizer("ggml");
    REQUIRE(q != nullptr);
    for (const Type& t : kTypes) {
        Opts o;
        o.s("type", t.name);
        const RadQuantWeight qw = weight_of("w", 4, 512, nullptr);
        RadEncoding e;
        REQUIRE_EQ(q->encoding(o.data(), o.n(), &qw, &e), RAD_OK);
        CHECK(rad_enc_valid(&e));
        CHECK_EQ(enc_name(e), std::string(t.enc));
        CHECK_EQ(q->row_block(o.data(), o.n(), &qw), (int64_t)1);
    }
}

/* The core cannot call libquant, so a GGUF is read through its own copy of the encodings
 * (core/format/ggml_planes.h); this holds every type of the copy to the quantiser's. */
TEST(the_reader_names_the_planes_the_quantiser_writes) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* q = quantizer("ggml");
    REQUIRE(q != nullptr);
    for (const ggml_planes::Type& t : ggml_planes::kTypes) {
        Opts o;
        o.s("type", gguf_dtype(t.ggml)->name);
        const RadQuantWeight qw = weight_of("w", 4, 512, nullptr);
        RadEncoding written, read;
        REQUIRE_EQ(q->encoding(o.data(), o.n(), &qw, &written), RAD_OK);
        REQUIRE(ggml_plane_encoding(t.ggml, &read));
        CHECK_EQ(enc_name(read), enc_name(written));
        CHECK(rad_enc_equal(&read, &written));
    }
}

TEST(every_type_decodes_within_its_bound) {
    NEED_LIBQUANT();
    const int64_t C = 1024;
    const std::vector<float> w32 = draw(32 * C, 11, 10.0f);
    const std::vector<float> imp = importance(C, 12);
    for (const Type& t : kTypes) {
        /* The 1- and 2-bit grid types search llama.cpp's neighbour grids exhaustively, so they are
         * measured on the first 8 rows (8192 values) and the rest on all 32. */
        const int64_t R = grid_search(t) ? 8 : 32;
        const std::vector<float> w(w32.begin(), w32.begin() + R * C);
        double e[2] = { 0, 0 };
        for (int m = 0; m < 2; ++m) {
            if (m == 0 && t.needs_im) continue;
            Quantised qz = quantise_ggml(t.name, R, C, w, m ? imp.data() : nullptr, 8);
            REQUIRE_EQ(qz.status, RAD_OK);
            const std::vector<float> d = decode(qz, R, C);
            REQUIRE(!d.empty());
            e[m] = rel_err(w, d);
            CHECK(e[m] < t.bound);
        }
        std::fprintf(stderr, "    %-8s rel_err plain %.4f, imatrix %.4f\n", t.name, e[0], e[1]);
    }
}

TEST(the_imatrix_reaches_the_search) {
    NEED_LIBQUANT();
    /* With importances the search minimises the importance-weighted error, so every type that
     * has a search must lower it measurably (2.5-25% on this weight). q8_0 has no search
     * (llama.cpp rounds to nearest either way) and must not change at all. */
    const int64_t R = 32, C = 1024;
    const std::vector<float> w = draw(R * C, 11, 10.0f);
    const std::vector<float> imp = importance(C, 12);
    for (const Type& t : kTypes) {
        if (t.needs_im) continue;
        Quantised a = quantise_ggml(t.name, R, C, w, nullptr, 8);
        Quantised b = quantise_ggml(t.name, R, C, w, imp.data(), 8);
        REQUIRE_EQ(a.status, RAD_OK);
        REQUIRE_EQ(b.status, RAD_OK);
        if (!std::strcmp(t.name, "q8_0")) {
            CHECK(a.planes == b.planes);
            continue;
        }
        const double ea = weighted_err(w, decode(a, R, C), imp, C);
        const double eb = weighted_err(w, decode(b, R, C), imp, C);
        CHECK(eb < 0.99 * ea);
    }
}

TEST(q8_0_is_round_to_nearest_against_amax_over_127) {
    NEED_LIBQUANT();
    /* One format restated from its definition rather than from the port: without an imatrix,
     * llama.cpp's q8_0 is d = amax/127, code roundf(x * (1/d)), d stored as f16. */
    const int64_t R = 4, C = 256;
    const std::vector<float> w = draw(R * C, 21, 6.0f);
    Quantised q = quantise_ggml("q8_0", R, C, w, nullptr);
    REQUIRE_EQ(q.status, RAD_OK);
    int off = 0;
    for (int64_t r = 0; r < R; ++r)
        for (int64_t b = 0; b < C / 32; ++b) {
            float amax = 0;
            const float* x = w.data() + r * C + 32 * b;
            for (int64_t j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[j]));
            const float d = amax / 127, id = d ? 1.0f / d : 0.0f;
            uint16_t s;
            std::memcpy(&s, q.planes[1].data() + 2 * (r * (C / 32) + b), 2);
            off += s != rad_f32_to_f16(d);
            for (int64_t j = 0; j < 32; ++j) {
                const int8_t code = (int8_t)q.planes[0][(size_t)(r * C + 32 * b + j)];
                off += code != (int8_t)std::roundf(x[j] * id);
            }
        }
    CHECK_EQ(off, 0);
}

TEST(quantisation_is_deterministic_and_rows_are_independent) {
    NEED_LIBQUANT();
    /* A row's planes do not depend on which call carried it, nor on the thread that ran it. */
    const int64_t R = 12, C = 768;
    const std::vector<float> w = draw(R * C, 31, 10.0f);
    const std::vector<float> imp = importance(C, 32);
    for (const char* t : { "q4_K", "q5_1", "iq4_xs", "iq2_xxs", "iq3_s", "iq1_m" }) {
        Quantised one = quantise_ggml(t, R, C, w, imp.data(), 1);
        Quantised odd = quantise_ggml(t, R, C, w, imp.data(), 5);
        Quantised all = quantise_ggml(t, R, C, w, imp.data(), R);
        Quantised again = quantise_ggml(t, R, C, w, imp.data(), R);
        REQUIRE_EQ(one.status, RAD_OK);
        CHECK(one.planes == odd.planes);
        CHECK(one.planes == all.planes);
        CHECK(all.planes == again.planes);
    }
}

TEST(bit_exact_against_llama_cpp) {
    NEED_LIBQUANT();
    /* FNV-1a of every plane's bytes, in plane order, for the [8 x 512] weight below without (0)
     * and with (1) the importances below; each confirmed equal to llama.cpp's own blocks and
     * dequantisation (see the file header). */
    struct Pin { const char* type; int imatrix; uint64_t fnv; };
    const Pin kPins[] = {
        { "q4_0", 0, 0xafebf0225eb23c08ull },    { "q4_0", 1, 0x249e07415953dc83ull },
        { "q4_1", 0, 0x249f8507ccb28e84ull },    { "q4_1", 1, 0x861511021f1487d2ull },
        { "q5_0", 0, 0x68804fa0a750a161ull },    { "q5_0", 1, 0xe6de81cdcfa44e86ull },
        { "q5_1", 0, 0x2d8c51fe36496843ull },    { "q5_1", 1, 0xbd0ad8622f10a79eull },
        { "q8_0", 0, 0xe7fcb3f3b4ac3c32ull },    { "q8_0", 1, 0xe7fcb3f3b4ac3c32ull },
        { "q2_K", 0, 0x4126a37ae0eee6b7ull },    { "q2_K", 1, 0x6787af0669a459deull },
        { "q3_K", 0, 0xeda92060a51bdbd8ull },    { "q3_K", 1, 0x6bfc5154fab3c858ull },
        { "q4_K", 0, 0x1d7b53e854a01425ull },    { "q4_K", 1, 0x162638ae5f75f636ull },
        { "q5_K", 0, 0xd3ad81aef94faccdull },    { "q5_K", 1, 0x35ede462bc992c26ull },
        { "q6_K", 0, 0x27555bd1e0400234ull },    { "q6_K", 1, 0xa66725529adadf63ull },
        { "iq4_nl", 0, 0x9a5857bd0a27cf6eull },  { "iq4_nl", 1, 0xeda63216d5ce2c6bull },
        { "iq4_xs", 0, 0xe563a372ddd7506full },  { "iq4_xs", 1, 0x9b31075d3303262bull },
        { "iq2_xxs", 1, 0xd86fa2fa5fd2ecb3ull }, { "iq2_xs", 1, 0x5dcb565c62932519ull },
        { "iq2_s", 0, 0x892bc05ac8bbf27bull },   { "iq2_s", 1, 0x3a60de31653861e1ull },
        { "iq3_xxs", 0, 0xb7bb19b90de20079ull }, { "iq3_xxs", 1, 0xc0f103a573450456ull },
        { "iq3_s", 0, 0x7feeb3c566708bebull },   { "iq3_s", 1, 0xae2be8d72969b54aull },
        { "iq1_s", 1, 0xd527fc0b9d1bbb0aull },
        { "iq1_m", 0, 0xba4b5b012fb32412ull },   { "iq1_m", 1, 0x691e880c1d50e84dull },
    };
    const int64_t R = 8, C = 512;
    const std::vector<float> w = draw(R * C, 0x6a6, 8.0f);
    const std::vector<float> imp = importance(C, 0x1b1);
    for (const Pin& p : kPins) {
        Quantised q = quantise_ggml(p.type, R, C, w, p.imatrix ? imp.data() : nullptr);
        REQUIRE_EQ(q.status, RAD_OK);
        const uint64_t h = fnv1a(q);
        if (h != p.fnv)
            std::fprintf(stderr, "    %s imatrix=%d: fnv 0x%016llx, pinned 0x%016llx\n", p.type,
                         p.imatrix, (unsigned long long)h, (unsigned long long)p.fnv);
        CHECK(h == p.fnv);
    }
}

TEST(the_shared_planes_hold_llama_cpps_tables) {
    NEED_LIBQUANT();
    const int64_t R = 2, C = 256;
    const std::vector<float> w = draw(R * C, 41);
    const std::vector<float> imp = importance(C, 42);
    /* kvalues_iq4nl */
    Quantised nl = quantise_ggml("iq4_nl", R, C, w, nullptr);
    REQUIRE_EQ(nl.status, RAD_OK);
    const int8_t kv[16] = { -127, -104, -83, -65, -49, -35, -22, -10,
                            1, 13, 25, 38, 53, 69, 89, 113 };
    REQUIRE_EQ(nl.planes[1].size(), (size_t)16);
    CHECK(std::memcmp(nl.planes[1].data(), kv, 16) == 0);
    /* iq2xxs_grid[0] and [255], iq3xxs_grid[1], iq1s_grid[0] and [2047], and the per-tensor 1/8
     * and 1/4 */
    Quantised x2 = quantise_ggml("iq2_xxs", R, C, w, imp.data());
    Quantised x3 = quantise_ggml("iq3_xxs", R, C, w, nullptr);
    Quantised s1 = quantise_ggml("iq1_s", R, C, w, imp.data());
    REQUIRE_EQ(x2.status, RAD_OK);
    REQUIRE_EQ(x3.status, RAD_OK);
    REQUIRE_EQ(s1.status, RAD_OK);
    const uint8_t g2_0[8] = { 8, 8, 8, 8, 8, 8, 8, 8 };
    const uint8_t g2_255[8] = { 8, 25, 8, 8, 25, 43, 43, 43 };
    CHECK(std::memcmp(x2.planes[1].data(), g2_0, 8) == 0);
    CHECK(std::memcmp(x2.planes[1].data() + 255 * 8, g2_255, 8) == 0);
    const uint8_t g3_1[4] = { 20, 4, 4, 4 };
    CHECK(std::memcmp(x3.planes[1].data() + 4, g3_1, 4) == 0);
    const int8_t g1_0[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
    const int8_t g1_last[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
    CHECK(std::memcmp(s1.planes[1].data(), g1_0, 8) == 0);
    CHECK(std::memcmp(s1.planes[1].data() + 2047 * 8, g1_last, 8) == 0);
    float k2, k3;
    std::memcpy(&k2, x2.planes[5].data(), 4);
    std::memcpy(&k3, x3.planes[5].data(), 4);
    CHECK_EQ(k2, 0.125f);
    CHECK_EQ(k3, 0.25f);
}

TEST(a_weight_the_type_cannot_tile_is_declined) {
    NEED_LIBQUANT();
    CHECK_EQ(encoding_status("q4_0", 4, 96), RAD_OK);
    CHECK_EQ(encoding_status("q4_0", 4, 100), RAD_E_UNSUPPORTED);
    CHECK_EQ(encoding_status("q4_K", 4, 512), RAD_OK);
    CHECK_EQ(encoding_status("q4_K", 4, 512 + 32), RAD_E_UNSUPPORTED);
    CHECK_EQ(encoding_status("iq4_nl", 4, 2880), RAD_OK);
    CHECK_EQ(encoding_status("iq4_xs", 4, 2880), RAD_E_UNSUPPORTED);
    /* a vector is declined, not an error */
    CHECK_EQ(encoding_status("q8_0", 1, 256, 1), RAD_E_UNSUPPORTED);
}

TEST(a_type_is_named_in_any_case_and_an_unknown_one_is_an_error) {
    NEED_LIBQUANT();
    CHECK_EQ(encoding_status("Q4_K", 4, 256), RAD_OK);
    CHECK_EQ(encoding_status("IQ2_XXS", 4, 256), RAD_OK);
    CHECK_EQ(encoding_status("q4_9", 4, 256), RAD_E_INVAL);
    CHECK_EQ(encoding_status("q4", 4, 256), RAD_E_INVAL);
    CHECK_EQ(encoding_status(nullptr, 4, 256), RAD_E_INVAL);
    const int64_t R = 3, C = 256;
    const std::vector<float> w = draw(R * C, 51);
    Quantised a = quantise_ggml("q4_K", R, C, w, nullptr);
    Quantised b = quantise_ggml("Q4_k", R, C, w, nullptr);
    REQUIRE_EQ(a.status, RAD_OK);
    REQUIRE_EQ(b.status, RAD_OK);
    CHECK(rad_enc_equal(&a.enc, &b.enc));
    CHECK(a.planes == b.planes);
}

TEST(the_imatrix_types_refuse_to_run_without_one) {
    NEED_LIBQUANT();
    const int64_t R = 2, C = 256;
    const std::vector<float> w = draw(R * C, 61);
    for (const Type& t : kTypes) {
        /* the encoding is asked at declare, where there is never an imatrix, so it answers */
        CHECK_EQ(encoding_status(t.name, R, C), RAD_OK);
        Quantised q = quantise_ggml(t.name, R, C, w, nullptr);
        CHECK_EQ(q.status, t.needs_im ? RAD_E_INVAL : RAD_OK);
    }
}

TEST(quantize_refuses_rows_outside_the_weight) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* q = quantizer("ggml");
    Opts o;
    o.s("type", "q8_0");
    const RadQuantWeight qw = weight_of("w", 4, 64, nullptr);
    std::vector<float> src(5 * 64, 0.5f);
    std::vector<uint8_t> codes(5 * 64), scale(5 * 2 * 2);
    void* pp[2] = { codes.data(), scale.data() };
    CHECK_EQ(q->quantize(o.data(), o.n(), &qw, src.data(), 2, 3, pp), RAD_E_SHAPE);
    CHECK_EQ(q->quantize(o.data(), o.n(), &qw, src.data(), -1, 1, pp), RAD_E_SHAPE);
    CHECK_EQ(q->quantize(o.data(), o.n(), &qw, src.data(), 0, 4, pp), RAD_OK);
}

RAD_TEST_MAIN()
