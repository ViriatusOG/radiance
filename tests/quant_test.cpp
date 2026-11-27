/* quant_test.cpp -- libquant through the plugin registry, read back through the core's decoder.
 *
 * Every case quantises through the quantiser's own hooks, as rad-convert does, and checks the
 * result through enc_decode_rows, never through anything libquant reports about itself: an
 * encoding must let a reader that never saw the quantiser recover the weight. Exact rules (a
 * block scale equal to the bf16 of amax/448, a power-of-two MX exponent, a fixed scale) are
 * checked exactly; approximate ones are bounded by the grid's step, which a wrong block, a wrong
 * zero or a missing rotation all exceed.
 */
#include "quant_fixture.h"
#include "format/checkpoint.h"
#include "format/recipe.h"

#include <deque>
#include <fstream>
#include <sstream>

#include <algorithm>
#include <functional>
#include <sys/stat.h>
#include <unistd.h>

using namespace rad;
using namespace qfix;

TEST(libquant_offers_its_quantisers_through_the_registry) {
    NEED_LIBQUANT();
    CHECK(quantizer("rtn") != nullptr);
    CHECK(quantizer("gptq") != nullptr);
    CHECK(quantizer("no-such-quantiser") == nullptr);
    /* a second copy of the same plugin is refused, not merged */
    Registry r2;
    CHECK_EQ(r2.load_plugin(home() + "/quantizers/libquant.so", 0), RAD_OK);
    CHECK(r2.load_plugin(home() + "/quantizers/libquant.so", 1) < 0);
}

TEST(block_fp8_is_the_bf16_of_amax_over_448_and_codes_against_it) {
    NEED_LIBQUANT();
    const int64_t R = 256, C = 384;
    const std::vector<float> w = draw(R * C, 1, 20.0f);
    Opts o;
    o.s("codes", "fp8_e4m3").s("block", "128x128").s("scale", "bf16");
    Quantised q = quantise(quantizer("rtn"), o, "blk.0.attn_q.weight", R, C, w);
    REQUIRE_EQ(q.status, RAD_OK);
    CHECK_EQ(enc_name(q.enc), std::string("fp8_e4m3*bf16[128x128]"));
    REQUIRE_EQ(q.planes[1].size(), (size_t)(2 * 3 * 2));
    /* block (1, 2) */
    float amax = 0;
    for (int64_t r = 128; r < 256; ++r)
        for (int64_t c = 256; c < 384; ++c)
            amax = std::fmax(amax, std::fabs(w[(size_t)(r * C + c)]));
    const uint16_t want = rad_f32_to_bf16(amax * (1.0f / 448.0f));
    uint16_t got;
    std::memcpy(&got, q.planes[1].data() + (1 * 3 + 2) * 2, 2);
    CHECK_EQ((int)got, (int)want);
    const std::vector<float> d = decode(q, R, C);
    REQUIRE(!d.empty());
    CHECK(rel_err(w, d) < 0.04);
}

TEST(a_rotated_int4_group_decodes_back_through_the_hadamard) {
    NEED_LIBQUANT();
    const int64_t R = 32, C = 512;
    const std::vector<float> w = draw(R * C, 2, 30.0f);
    Opts plain, rot;
    plain.s("codes", "i4").i("group", 128).s("scale", "bf16");
    rot.s("codes", "i4").i("group", 128).s("scale", "bf16").s("transform", "fwht128");
    Quantised a = quantise(quantizer("rtn"), plain, "x", R, C, w);
    Quantised b = quantise(quantizer("rtn"), rot, "x", R, C, w);
    REQUIRE_EQ(a.status, RAD_OK);
    REQUIRE_EQ(b.status, RAD_OK);
    CHECK_EQ(enc_name(b.enc), std::string("i4*bf16[1x128]/fwht128"));
    const double ea = rel_err(w, decode(a, R, C)), eb = rel_err(w, decode(b, R, C));
    std::fprintf(stderr, "    int4 g128: plain %.4f, rotated %.4f\n", ea, eb);
    /* Both errors are a 4-bit grid's. The rotated one is decoded through the rotation, so a
     * reader that skipped it would see noise (~1.0). With outliers the rotation is also the
     * better grid: it spreads one large value over the group instead of letting it set the scale
     * everything else rounds against. */
    CHECK(ea < 0.35);
    CHECK(eb < 0.15);
    CHECK(eb < ea);
}

TEST(a_rotated_int8_group_decodes_back_through_the_hadamard) {
    NEED_LIBQUANT();
    /* What gemm_nt_q reads at dtype i8a8h: i8 codes with a bf16 scale a (row, 128), each group
     * rotated by the unnormalised Hadamard and the 1/128 in the stored scale. A scale that kept
     * the factor, or lost it twice, decodes 128 times off, and a skipped rotation to noise. */
    const int64_t R = 32, C = 512;
    const std::vector<float> w = draw(R * C, 2, 30.0f);
    Opts plain, rot;
    plain.s("codes", "i8").i("group", 128).s("scale", "bf16");
    rot.s("codes", "i8").i("group", 128).s("scale", "bf16").s("transform", "fwht128");
    Quantised a = quantise(quantizer("rtn"), plain, "x", R, C, w);
    Quantised b = quantise(quantizer("rtn"), rot, "x", R, C, w);
    REQUIRE_EQ(a.status, RAD_OK);
    REQUIRE_EQ(b.status, RAD_OK);
    CHECK_EQ(enc_name(b.enc), std::string("i8*bf16[1x128]/fwht128"));
    const double ea = rel_err(w, decode(a, R, C)), eb = rel_err(w, decode(b, R, C));
    std::fprintf(stderr, "    int8 g128: plain %.5f, rotated %.5f\n", ea, eb);
    /* An int8 grid's error is a few parts in a thousand; with outliers the rotation is again the
     * better grid. */
    CHECK(ea < 0.03);
    CHECK(eb < 0.01);
    CHECK(eb < ea);
}

TEST(the_w4nl_table_is_exact_in_e4m3_and_beats_minus_8_to_7_after_the_rotation) {
    NEED_LIBQUANT();
    /* The sixteen levels libr4d's w4nla8h kernel decodes, in its code order, as a container
     * stores them. Against -8..7 at the same bytes, rotation and scale search, they are the
     * better grid for the near-normal weight they were fitted to. Not with outliers: the rotation
     * turns one outlier into a +-o pattern across its group, which is bimodal. */
    const int64_t R = 32, C = 512;
    const std::vector<float> w = draw(R * C, 2);
    Opts i4, nl;
    i4.s("codes", "i4").i("group", 128).s("scale", "bf16").s("transform", "fwht128")
      .s("rule", "search");
    nl.s("table", "w4nl").i("group", 128).s("scale", "bf16").s("transform", "fwht128")
      .s("rule", "search");
    Quantised a = quantise(quantizer("rtn"), i4, "x", R, C, w);
    Quantised b = quantise(quantizer("rtn"), nl, "x", R, C, w);
    REQUIRE_EQ(a.status, RAD_OK);
    REQUIRE_EQ(b.status, RAD_OK);
    CHECK_EQ(enc_name(b.enc),
             std::string("affine:codes=u4[1x1],scale=bf16[1x128],table=f32{1x16}/fwht128"));
    const int t = rad_enc_find(&b.enc, "table");
    REQUIRE(t >= 0);
    const float want[16] = { 1.125f, 3.25f, 5.5f, 8.0f, 11.0f, 14.0f, 18.0f, 22.0f,
                             -1.125f, -3.25f, -5.5f, -8.0f, -11.0f, -14.0f, -18.0f, -22.0f };
    float got[16];
    std::memcpy(got, b.planes[(size_t)t].data(), sizeof got);
    for (int i = 0; i < 16; ++i) {
        CHECK_EQ(got[i], want[i]);
        CHECK_EQ(rad_fp8e4m3_to_f32(rad_f32_to_fp8e4m3(want[i])), want[i]);
    }
    const double ea = rel_err(w, decode(a, R, C)), eb = rel_err(w, decode(b, R, C));
    std::fprintf(stderr, "    rotated g128, searched scale: -8..7 %.4f, w4nl %.4f\n", ea, eb);
    CHECK(eb < ea * 0.95);
}

TEST(an_asymmetric_two_bit_grid_keeps_its_zero_in_range) {
    NEED_LIBQUANT();
    const int64_t R = 16, C = 256;
    const std::vector<float> w = draw(R * C, 3);
    Opts o;
    o.s("codes", "u2").i("group", 128).s("scale", "f16").s("zero", "u8");
    Quantised q = quantise(quantizer("rtn"), o, "head", R, C, w);
    REQUIRE_EQ(q.status, RAD_OK);
    CHECK_EQ(enc_name(q.enc), std::string("u2*f16[1x128]-u8[1x128]"));
    int bad = 0;
    for (uint8_t z : q.planes[2]) bad += z > 3;
    CHECK_EQ(bad, 0);
    CHECK(rel_err(w, decode(q, R, C)) < 0.6);
}

TEST(mxfp4_shares_a_power_of_two_a_32) {
    NEED_LIBQUANT();
    const int64_t R = 8, C = 256;
    const std::vector<float> w = draw(R * C, 4, 10.0f);
    Opts o;
    o.s("codes", "fp4_e2m1").i("group", 32).s("scale", "e8m0");
    Quantised q = quantise(quantizer("rtn"), o, "x", R, C, w);
    REQUIRE_EQ(q.status, RAD_OK);
    CHECK_EQ(enc_name(q.enc), std::string("fp4_e2m1*e8m0[1x32]"));
    /* the OCP rule: floor(log2 amax) - 2, so amax / 2^e lands in [4, 8) */
    for (int64_t r = 0; r < R; ++r)
        for (int64_t g = 0; g < C / 32; ++g) {
            float amax = 0;
            for (int64_t c = g * 32; c < g * 32 + 32; ++c)
                amax = std::fmax(amax, std::fabs(w[(size_t)(r * C + c)]));
            const float s = rad_e8m0_to_f32(q.planes[1][(size_t)(r * (C / 32) + g)]);
            CHECK(amax / s >= 4.0f && amax / s < 8.0f);
        }
    CHECK(rel_err(w, decode(q, R, C)) < 0.2);
}

TEST(a_fixed_per_tensor_scale_streams_a_row_at_a_time) {
    NEED_LIBQUANT();
    const int64_t R = 64, C = 160;
    const std::vector<float> w = draw(R * C, 5);
    Opts o;
    o.s("codes", "fp8_e4m3").s("block", "*x*").s("scale", "bf16").s("rule", "fixed")
     .f("scale_value", 1.99317932128906e-4);
    const RadQuantizerInfo* q = quantizer("rtn");
    RadQuantWeight qw{};
    qw.name = "t";
    qw.rank = 2; qw.shape[0] = R; qw.shape[1] = C; qw.rows = R; qw.cols = C;
    CHECK_EQ(q->row_block(o.data(), o.n(), &qw), 1);
    Quantised a = quantise(q, o, "t", R, C, w);
    REQUIRE_EQ(a.status, RAD_OK);
    CHECK_EQ(enc_name(a.enc), std::string("fp8_e4m3*bf16[*x*]"));
    uint16_t s;
    std::memcpy(&s, a.planes[1].data(), 2);
    CHECK_EQ((int)s, 0x3951);
    /* the rule's reciprocal is the unrounded value's, in double */
    const double inv = 1.0 / 1.99317932128906e-4;
    int off = 0;
    for (int64_t i = 0; i < R * C; ++i)
        off += a.planes[0][(size_t)i] != rad_f32_to_fp8e4m3((float)((double)w[(size_t)i] * inv));
    CHECK_EQ(off, 0);
}

TEST(gptq_without_a_hessian_is_round_to_nearest_byte_for_byte) {
    NEED_LIBQUANT();
    const int64_t R = 64, C = 256;
    const std::vector<float> w = draw(R * C, 6, 20.0f);
    Opts a, b;
    a.s("codes", "i4").i("group", 128).s("scale", "bf16").s("transform", "fwht128");
    b.s("codes", "i4").i("group", 128).s("scale", "bf16").s("transform", "fwht128");
    Quantised r = quantise(quantizer("rtn"), a, "blk.0.ffn_down_exps.0.weight", R, C, w);
    Quantised g = quantise(quantizer("gptq"), b, "blk.0.ffn_down_exps.0.weight", R, C, w);
    REQUIRE_EQ(r.status, RAD_OK);
    REQUIRE_EQ(g.status, RAD_OK);
    for (int i = 0; i < r.enc.n_planes; ++i) CHECK(r.planes[(size_t)i] == g.planes[(size_t)i]);
}

TEST(gptq_with_a_hessian_lowers_the_output_error) {
    NEED_LIBQUANT();
    /* Correlated activations: x = A z, so H = sum x x^T is far from diagonal and error feedback
     * can beat rounding each column alone. */
    const int64_t R = 48, C = 128, T = 2048;
    const std::vector<float> w = draw(R * C, 7, 10.0f);
    const std::vector<float> z = draw(T * C, 8);
    const std::vector<float> mix = draw(C * C, 9);
    std::vector<float> x((size_t)(T * C), 0.0f);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < C; ++i) {
            double s = z[(size_t)(t * C + i)] * 4.0;
            for (int64_t j = 0; j < 8; ++j)
                s += mix[(size_t)(i * C + j)] * z[(size_t)(t * C + j)] * 40.0;
            x[(size_t)(t * C + i)] = (float)s;
        }
    std::vector<float> h((size_t)(C * C), 0.0f);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < C; ++i)
            for (int64_t j = 0; j < C; ++j)
                h[(size_t)(i * C + j)] += x[(size_t)(t * C + i)] * x[(size_t)(t * C + j)];

    const TempDir td("rad_quant_test");
    const char* dir = td.path.c_str();
    const std::string path = std::string(dir) + "/gram.r0.blk.3.ffn_down_exps.bin";
    REQUIRE(write_gram(path, C, T, h));

    Opts a, b;
    a.s("codes", "i4").i("group", 32).s("scale", "f32");
    b.s("codes", "i4").i("group", 32).s("scale", "f32").s("calib", dir);
    Quantised r = quantise(quantizer("rtn"), a, "blk.3.ffn_down_exps.17.weight", R, C, w);
    Quantised g = quantise(quantizer("gptq"), b, "blk.3.ffn_down_exps.17.weight", R, C, w);
    REQUIRE_EQ(r.status, RAD_OK);
    REQUIRE_EQ(g.status, RAD_OK);
    const std::vector<float> dr = decode(r, R, C), dg = decode(g, R, C);
    auto out_err = [&](const std::vector<float>& d) {
        double e = 0, n = 0;
        for (int64_t t = 0; t < T; t += 4)
            for (int64_t o = 0; o < R; ++o) {
                double a1 = 0, a2 = 0;
                for (int64_t i = 0; i < C; ++i) {
                    a1 += (double)w[(size_t)(o * C + i)] * x[(size_t)(t * C + i)];
                    a2 += (double)d[(size_t)(o * C + i)] * x[(size_t)(t * C + i)];
                }
                e += (a1 - a2) * (a1 - a2);
                n += a1 * a1;
            }
        return std::sqrt(e / n);
    };
    const double er = out_err(dr), eg = out_err(dg);
    std::fprintf(stderr, "    output error: rtn %.4f, gptq %.4f\n", er, eg);
    CHECK(eg < er * 0.9);

    /* Coordinate descent afterwards never raises the quadratic form it descends (the output
     * error on the corpus the Gram came from), and on a correlated Hessian it finds codes to move. */
    Opts c;
    c.s("codes", "i4").i("group", 32).s("scale", "f32").s("calib", dir).i("cd", 3);
    Quantised gc = quantise(quantizer("gptq"), c, "blk.3.ffn_down_exps.17.weight", R, C, w);
    REQUIRE_EQ(gc.status, RAD_OK);
    const double ec = out_err(decode(gc, R, C));
    std::fprintf(stderr, "    output error: gptq + cd=3 %.4f\n", ec);
    CHECK(ec < eg);
    /* The scales are GPTQ's, untouched: only codes move. */
    CHECK(gc.planes[1] == g.planes[1]);
}

TEST(w6nl_gptq_and_coordinate_descent_accept_bf16_domain_hessians) {
    NEED_LIBQUANT();
    const int64_t R = 16, C = 128, T = 4096;
    std::vector<float> h((size_t)(C * C), (float)T * 0.25f);
    for (int64_t i = 0; i < C; ++i) h[(size_t)(i * C + i)] += (float)T;
    const TempDir td("rad_w6nl_gptq");
    REQUIRE(write_gram(td.path + "/gram.r0.blk.3.ffn_down_exps.bin", C, T, h));
    Opts o;
    o.s("table", "w6nl").i("group", 64).s("scale", "fp8_e4m3").s("scale2", "f32")
     .s("block2", "*x*").f("scale2_value", 0x1p-13).s("transform", "fwht128")
     .s("rule", "search").i("cd", 3).s("calib", td.path.c_str());
    const std::vector<float> w = draw(R * C, 770, 1.0f);
    const Quantised q = quantise(quantizer("gptq"), o, "blk.3.ffn_down_exps.17.weight", R, C, w);
    REQUIRE_EQ(q.status, RAD_OK);
    CHECK_EQ(q.enc.plane[0].dtype, (uint32_t)RAD_U6);
    CHECK_EQ(q.planes[0].size(), (size_t)(R * C * 6 / 8));
    CHECK_EQ(q.enc.plane[3].extent[1], 64LL);
    for (int i = 0; i < 64; ++i) {
        float value = 0.0f;
        std::memcpy(&value, q.planes[3].data() + i * sizeof(float), sizeof value);
        const uint8_t fp8 = rad_f32_to_fp8e4m3(value);
        CHECK_EQ(rad_load_f32(&fp8, RAD_F8E4M3, 0), value);
        if (i >= 32) {
            float positive = 0.0f;
            std::memcpy(&positive, q.planes[3].data() + (i - 32) * sizeof(float), sizeof positive);
            CHECK_EQ(value, -positive);
        }
    }
    const std::vector<float> decoded = decode(q, R, C);
    CHECK(rel_err(w, decoded) < 0.2);
}

/* ------------------------------------------------------------------ recipes, compiled
 *
 * A rule's options are typed by its quantiser's own table before declare runs, so a misspelt key or
 * a value of the wrong type is refused before a checkpoint of hundreds of GB is opened. */
TEST(a_rule_compiles_its_options_by_the_quantisers_types) {
    NEED_LIBQUANT();
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_text("a  gptq codes=i4 group=010 damp=0.02 calib=/x\n"
                          "b  rtn  codes=i4 codez=i8\n"
                          "c  rtn  codes=i4 group=0x80\n"
                          "d  rtn  codes=i4 group=\n"
                          "e  gptq codes=i4 damp=nan\n"
                          "f  gptq codes=i4 damp=1e999\n"
                          "g  rtn  codes=i4 group=99999999999999999999\n"
                          "h  nope codes=i4\n", "t", &why), RAD_OK);
    const std::vector<RecipeRule>& rules = r.rules();
    REQUIRE_EQ(rules.size(), (size_t)8);

    /* Decimal, so the text the container records is the number the quantiser was given. */
    CompiledRule a;
    REQUIRE_EQ(a.build(rules[0], reg(), &why), RAD_OK);
    REQUIRE_EQ(a.params.size(), (size_t)4);
    CHECK_EQ(a.params[1].kind, (int)RAD_P_INT);
    CHECK_EQ(a.params[1].ival, 10ll);
    CHECK_EQ(std::string(a.params[1].key), std::string("group"));
    CHECK_EQ(a.params[2].kind, (int)RAD_P_F64);
    CHECK_EQ(a.params[2].dval, 0.02);
    CHECK_EQ(std::string(a.params[3].sval), std::string("/x"));

    CompiledRule b;
    CHECK_EQ(b.build(rules[1], reg(), &why), RAD_E_INVAL);
    CHECK(has(why, "t:2") && has(why, "takes no option 'codez'"));
    for (size_t i = 2; i < 7; ++i) {
        CompiledRule c;
        CHECK_EQ(c.build(rules[i], reg(), &why), RAD_E_INVAL);
        CHECK(has(why, rules[i].origin));
    }
    CompiledRule h;
    CHECK_EQ(h.build(rules[7], reg(), &why), RAD_E_NOTFOUND);
    CHECK(has(why, "no quantiser 'nope'") && has(why, "rtn"));
}

/* Every rule of every shipped recipe names a loaded quantiser and options it takes, and the
 * quantiser accepts the kind of weight its pattern names, or declines it; none is refused as
 * malformed. Recipes ship beside the published containers, and a broken one would otherwise be
 * found only hours into a conversion. */
TEST(every_shipped_recipe_compiles_and_its_grids_are_accepted) {
    NEED_LIBQUANT();
    std::string dir = __FILE__;
    dir = dir.substr(0, dir.find_last_of('/')) + "/../data/recipes";
    const TempDir calib("rad_quant_recipes");
    const char* old = std::getenv("CALIB");
    const std::string saved = old ? old : "";
    const char* old_mtp = std::getenv("MTP_CALIB");
    const std::string saved_mtp = old_mtp ? old_mtp : "";
    setenv("CALIB", calib.path.c_str(), 1);
    setenv("MTP_CALIB", calib.path.c_str(), 1);

    int files = 0, rules = 0;
    std::error_code ec;
    /* Codebook files (vq=) are named from the repository's root, where rad-convert runs them. */
    const std::filesystem::path cwd = std::filesystem::current_path(ec);
    std::filesystem::current_path(dir + "/../..", ec);
    for (const auto& ent : std::filesystem::directory_iterator(dir, ec)) {
        if (ent.path().extension() != ".recipe") continue;
        ++files;
        std::ifstream f(ent.path());
        std::stringstream ss;
        ss << f.rdbuf();
        Recipe r;
        std::string why;
        const int st = r.add_text(ss.str(), ent.path().filename().string(), &why);
        if (st != RAD_OK) std::fprintf(stderr, "    %s\n", why.c_str());
        CHECK_EQ(st, RAD_OK);
        std::deque<CompiledRule> compiled;
        for (const RecipeRule& rule : r.rules()) {
            ++rules;
            /* rad-convert's own import rule: its options must parse. */
            if (rule.quantizer == kImportRule) {
                ImportForm f;
                const int ps = ckpt_import_form_parse(rule.options, &f, &why);
                if (ps != RAD_OK) std::fprintf(stderr, "    %s: %s\n", rule.origin.c_str(), why.c_str());
                CHECK_EQ(ps, RAD_OK);
                continue;
            }
            compiled.emplace_back();
            CompiledRule& c = compiled.back();
            if (c.build(rule, reg(), &why) != RAD_OK) {
                std::fprintf(stderr, "    %s\n", why.c_str());
                CHECK(false);
                continue;
            }
            /* A weight the pattern names: every wildcard a layer or expert 0. */
            std::string name = rule.pattern;
            for (char& ch : name) if (ch == '*' || ch == '?') ch = '0';
            const RadQuantWeight qw = weight_of(name.c_str(), 256, 4096);
            RadEncoding e;
            const int es = c.q->encoding(c.params.data(), (int)c.params.size(), &qw, &e);
            if (es != RAD_OK && es != RAD_E_UNSUPPORTED)
                std::fprintf(stderr, "    %s (%s %s): %s\n", rule.origin.c_str(),
                             rule.quantizer.c_str(), rule.options_text().c_str(), rad_strerror(es));
            CHECK(es == RAD_OK || es == RAD_E_UNSUPPORTED);
            if (es == RAD_OK) CHECK(rad_enc_valid(&e));
        }
    }
    std::fprintf(stderr, "    %d recipe(s), %d rule(s)\n", files, rules);
    CHECK(files >= 20);
    CHECK(rules >= 250);
    std::filesystem::current_path(cwd, ec);
    if (old) setenv("CALIB", saved.c_str(), 1); else unsetenv("CALIB");
    if (old_mtp) setenv("MTP_CALIB", saved_mtp.c_str(), 1); else unsetenv("MTP_CALIB");
}

/* ------------------------------------------------------------------ GPTQ against the textbook
 *
 * Frantar et al.'s GPTQ, written plainly in double: damp the Hessian (lambda = damp x mean
 * diagonal, a dead column's diagonal set to that mean), take U, the upper Cholesky factor of its
 * inverse, then for each column in order round it under its block's fixed scale and spread
 * (w_i - q_i) / U_ii over every later column j with weight U_ij. libquant's loop is blocked
 * (128 columns at a time, a lazy update across blocks) and in f32, so the codes are checked
 * column by column against the textbook target with libquant's earlier codes taken as given: a
 * code must be the nearest under its scale to that target, up to a tie. A wrong index in the
 * cross-block update, a transposed U or damping applied twice moves targets across rounding
 * boundaries and shows as codes that are not nearest. */
namespace {

/* Lower Cholesky of the SPD `a` [k x k] in place; false if it is not positive definite. */
bool ref_chol(std::vector<double>& a, int64_t k) {
    for (int64_t j = 0; j < k; ++j) {
        double d = a[(size_t)(j * k + j)];
        for (int64_t t = 0; t < j; ++t) d -= a[(size_t)(j * k + t)] * a[(size_t)(j * k + t)];
        if (!(d > 0)) return false;
        d = std::sqrt(d);
        a[(size_t)(j * k + j)] = d;
        for (int64_t i = j + 1; i < k; ++i) {
            double v = a[(size_t)(i * k + j)];
            for (int64_t t = 0; t < j; ++t) v -= a[(size_t)(i * k + t)] * a[(size_t)(j * k + t)];
            a[(size_t)(i * k + j)] = v / d;
        }
        for (int64_t i = 0; i < j; ++i) a[(size_t)(i * k + j)] = 0;
    }
    return true;
}

/* U with U^T U = (H + damping)^-1, upper, in double: the inverse by solving L L^T X = I column by
 * column, then the lower factor of the inverse transposed. */
std::vector<double> ref_gptq_u(const std::vector<float>& h, int64_t k, double damp) {
    std::vector<double> a(h.begin(), h.end());
    double tr = 0;
    for (int64_t i = 0; i < k; ++i) tr += a[(size_t)(i * k + i)];
    const double mean = tr / (double)k;
    for (int64_t i = 0; i < k; ++i) {
        double& d = a[(size_t)(i * k + i)];
        if (!(d > 0)) d = mean;
        d += damp * mean;
    }
    if (!ref_chol(a, k)) return {};
    std::vector<double> inv((size_t)(k * k)), y((size_t)k);
    for (int64_t c = 0; c < k; ++c) {
        for (int64_t i = 0; i < k; ++i) {                      /* L y = e_c */
            double v = i == c ? 1.0 : 0.0;
            for (int64_t t = 0; t < i; ++t) v -= a[(size_t)(i * k + t)] * y[(size_t)t];
            y[(size_t)i] = v / a[(size_t)(i * k + i)];
        }
        for (int64_t i = k - 1; i >= 0; --i) {                 /* L^T x = y */
            double v = y[(size_t)i];
            for (int64_t t = i + 1; t < k; ++t) v -= a[(size_t)(t * k + i)] * inv[(size_t)(t * k + c)];
            inv[(size_t)(i * k + c)] = v / a[(size_t)(i * k + i)];
        }
    }
    if (!ref_chol(inv, k)) return {};
    std::vector<double> u((size_t)(k * k), 0.0);
    for (int64_t i = 0; i < k; ++i)
        for (int64_t j = i; j < k; ++j) u[(size_t)(i * k + j)] = inv[(size_t)(j * k + i)];
    return u;
}

/* The codes of `w` [n x k] (stored column order) that are not the nearest i4 code to the textbook
 * GPTQ target, with libquant's own earlier codes fed forward. `scale(r, j)` is column j's stored
 * scale. Each mismatch is printed. */
int ref_gptq_mismatches(const std::vector<float>& w, const std::vector<int>& codes,
                        const std::function<double(int64_t, int64_t)>& scale,
                        const std::vector<double>& u, int64_t n, int64_t k) {
    int bad = 0;
    for (int64_t r = 0; r < n; ++r) {
        std::vector<double> t(w.begin() + r * k, w.begin() + (r + 1) * k);
        for (int64_t i = 0; i < k; ++i) {
            const double s = scale(r, i);
            const double q = t[(size_t)i] / s;
            const double nearest = std::min(7.0, std::max(-8.0, std::round(q)));
            const int c = codes[(size_t)(r * k + i)];
            /* A tie, to rounding in f32 against double: within 1e-3 of a code step. */
            if (c != (int)nearest && std::fabs(q - c) > std::fabs(q - nearest) + 1e-3) {
                if (bad < 5)
                    std::fprintf(stderr, "    row %lld col %lld: code %d, the target is %.4f steps\n",
                                 (long long)r, (long long)i, c, q);
                ++bad;
            }
            const double e = (t[(size_t)i] - c * s) / u[(size_t)(i * k + i)];
            for (int64_t j = i + 1; j < k; ++j) t[(size_t)j] -= e * u[(size_t)(i * k + j)];
        }
    }
    return bad;
}

}  /* namespace */

TEST(gptq_chooses_the_codes_of_textbook_gptq) {
    NEED_LIBQUANT();
    /* K = 320 is two whole 128-column blocks and a partial third; column 37 is dead (no input
     * reaches it), so its Hessian row and column are zero. */
    const int64_t N = 8, K = 320, T = 384, dead = 37;
    const std::vector<float> w = draw(N * K, 21, 8.0f);
    const std::vector<float> z = draw(T * K, 22);
    std::vector<float> x((size_t)(T * K));
    for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < K; ++i) {
            /* Each input shares most of its energy with its neighbour, so H is far from diagonal. */
            const float own = z[(size_t)(t * K + i)], next = z[(size_t)(t * K + (i + 1) % K)];
            x[(size_t)(t * K + i)] = i == dead ? 0.0f : (own + 3.0f * next) * (1.0f + (float)(i % 7));
        }
    std::vector<float> h((size_t)(K * K), 0.0f);
    for (int64_t t = 0; t < T; ++t) {
        const float* xr = x.data() + t * K;
        for (int64_t i = 0; i < K; ++i)
            for (int64_t j = 0; j < K; ++j) h[(size_t)(i * K + j)] += xr[i] * xr[j];
    }
    const TempDir td("rad_quant_gptq");
    REQUIRE(write_gram(td.path + "/gram.r0.blk.4.ffn_up.bin", K, T, h));
    const char* name = "blk.4.ffn_up.weight";

    /* The stored codes as values: decode, then divide by the scale the plane holds. */
    auto codes_of = [&](const Quantised& q, const std::vector<int32_t>* perm) {
        const std::vector<float> d = decode(q, N, K);
        std::vector<int> c((size_t)(N * K));
        for (int64_t r = 0; r < N; ++r)
            for (int64_t j = 0; j < K; ++j) {
                const int64_t col = perm ? (*perm)[(size_t)j] : j;   /* stored column j */
                float s;
                std::memcpy(&s, q.planes[1].data() + (r * (K / 32) + j / 32) * 4, 4);
                c[(size_t)(r * K + j)] = (int)std::lround(d[(size_t)(r * K + col)] / s);
            }
        return c;
    };
    auto scale_of = [&](const Quantised& q) {
        return [&q, K](int64_t r, int64_t j) {
            float s;
            std::memcpy(&s, q.planes[1].data() + (r * (K / 32) + j / 32) * 4, 4);
            return (double)s;
        };
    };

    Opts o;
    o.s("codes", "i4").i("group", 32).s("scale", "f32").s("calib", td.path.c_str());
    const Quantised g = quantise(quantizer("gptq"), o, name, N, K, w);
    REQUIRE_EQ(g.status, RAD_OK);
    const std::vector<double> u = ref_gptq_u(h, K, 0.01);
    REQUIRE(!u.empty());
    CHECK_EQ(ref_gptq_mismatches(w, codes_of(g, nullptr), scale_of(g), u, N, K), 0);

    /* The check has teeth: against a factor damped twice as hard, codes stop being nearest. */
    int stronger = 0;
    (void)stderr_of([&] {
        stronger = ref_gptq_mismatches(w, codes_of(g, nullptr), scale_of(g), ref_gptq_u(h, K, 0.02),
                                       N, K);
    });
    CHECK(stronger > 0);

    /* Act order: columns by descending input energy, ties in column order (the dead column last),
     * stored in that order with the permutation as t.perm; the same textbook loop on the permuted
     * weight and Hessian. */
    Opts a;
    a.s("codes", "i4").i("group", 32).s("scale", "f32").s("calib", td.path.c_str()).i("act_order", 1);
    const Quantised ga = quantise(quantizer("gptq"), a, name, N, K, w);
    REQUIRE_EQ(ga.status, RAD_OK);
    std::vector<int32_t> want((size_t)K);
    for (int64_t i = 0; i < K; ++i) want[(size_t)i] = (int32_t)i;
    std::stable_sort(want.begin(), want.end(), [&](int32_t p, int32_t q) {
        return h[(size_t)p * (size_t)K + (size_t)p] > h[(size_t)q * (size_t)K + (size_t)q];
    });
    const int tp = rad_enc_find(&ga.enc, "t.perm");
    REQUIRE(tp >= 0);
    std::vector<int32_t> perm((size_t)K);
    std::memcpy(perm.data(), ga.planes[(size_t)tp].data(), (size_t)K * 4);
    CHECK(perm == want);
    CHECK_EQ(perm[(size_t)(K - 1)], (int32_t)dead);
    std::vector<float> wp((size_t)(N * K)), hp((size_t)(K * K));
    for (int64_t r = 0; r < N; ++r)
        for (int64_t j = 0; j < K; ++j) wp[(size_t)(r * K + j)] = w[(size_t)(r * K + perm[(size_t)j])];
    for (int64_t i = 0; i < K; ++i)
        for (int64_t j = 0; j < K; ++j)
            hp[(size_t)(i * K + j)] = h[(size_t)perm[(size_t)i] * (size_t)K + (size_t)perm[(size_t)j]];
    const std::vector<double> up = ref_gptq_u(hp, K, 0.01);
    REQUIRE(!up.empty());
    CHECK_EQ(ref_gptq_mismatches(wp, codes_of(ga, &perm), scale_of(ga), up, N, K), 0);
}

/* ------------------------------------------------------------------ round to nearest, every grid
 *
 * Round-to-nearest must store, for every value, a code whose decoded value is nearest to it among
 * every code the grid has, given the block's stored scale (and zero, codebook and second level).
 * The expectation is the core decoder's, not libquant's: each candidate code is written into every
 * element of the codes plane and the weight decoded, so each element's value under every code comes
 * from the reader. A wrong rounding mode, a wrong fp6 encoding or an off-by-one table index decodes
 * to a plausible value inside an error bound, and fails here. Each grid is also quantised a row
 * block at a time and in one call, which must give the same bytes: the converter bands what it
 * quantises. Under a transform the comparison is in the rotated domain, the decoder's own before
 * it applies the Hadamard: H x / n for the value x. */
namespace {

/* "codes=i4,group=32" as typed options: the integer and number keys rtn takes, strings for the
 * rest. */
Opts opts_of(const std::string& spec) {
    Opts o;
    size_t at = 0;
    while (at < spec.size()) {
        size_t c = spec.find(',', at);
        if (c == std::string::npos) c = spec.size();
        const std::string kv = spec.substr(at, c - at);
        const size_t eq = kv.find('=');
        const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        if (k == "group" || k == "group2") o.i(k.c_str(), std::atoll(v.c_str()));
        else if (k == "scale_value" || k == "scale2_value") o.f(k.c_str(), std::atof(v.c_str()));
        else o.s(k.c_str(), v.c_str());
        at = c + 1;
    }
    return o;
}

}  /* namespace */

TEST(round_to_nearest_stores_the_nearest_code_on_every_grid) {
    NEED_LIBQUANT();
    const char* const kGrids[] = {
        /* integer codes, symmetric and asymmetric, every width the options name */
        "codes=i2,group=32,scale=bf16",
        "codes=i3,group=64,scale=f16",
        "codes=i4,group=128,scale=f32",
        "codes=i4,group=32,scale=bf16,clamp=sym",
        "codes=i5,group=32,scale=f16",
        "codes=i6,block=4x64,scale=bf16",
        "codes=i8,group=64,scale=f32",
        "codes=i8,block=*x*,scale=bf16",
        "codes=u1,group=32,scale=f16,zero=u8",
        "codes=u2,group=64,scale=bf16,zero=u4",
        "codes=u4,group=32,scale=f16,zero=u8",
        "codes=u8,group=128,scale=f32,zero=u8",
        "codes=i4,group=32,scale=fp8_e4m3",
        "codes=i4,group=32,scale=bf16,rule=search",
        "codes=u4,group=64,scale=f16,zero=u4,rule=search",
        "codes=i4,group=64,scale=bf16,transform=fwht128",
        "codes=i8,group=128,scale=f16,transform=fwht128",
        "codes=i4,block=8x128,scale=bf16",
        "codes=i4,group=64,rule=fixed,scale_value=0.0078125,scale=f32",
        /* float codes */
        "codes=fp8_e4m3,block=8x128,scale=bf16",
        "codes=fp8_e5m2,group=32,scale=bf16",
        "codes=fp6_e2m3,group=32,scale=f16",
        "codes=fp6_e3m2,group=64,scale=bf16",
        "codes=fp4_e2m1,group=32,scale=e8m0",
        "codes=fp6_e2m3,group=32,scale=e8m0",
        "codes=fp8_e4m3,group=32,scale=e8m0",
        "codes=fp4_e2m1,group=16,scale=fp8_e4m3,scale2=f32",
        "codes=fp8_e5m2,group=64,scale=f16,transform=fwht128",
        "codes=fp4_e2m1,group=32,scale=e8m0,transform=fwht32",
        "codes=i8,group=32,scale=fp8_e4m3,transform=fwht64",
        /* second levels */
        "codes=i4,group=32,scale=bf16,scale2=e8m0,group2=256",
        "codes=i4,group=32,scale=fp8_e4m3,scale2=fp8_e4m3,group2=128",
        "codes=i4,group=16,scale=u6,scale2=f16",
        "codes=i4,group=16,scale=i8,scale2=bf16,group2=128",
        "codes=u4,group=32,scale=u6,scale2=f16,zero=u4",
        "codes=fp8_e5m2,group=32,scale=bf16,scale2=f16,group2=256,transform=fwht128",
        "codes=i4,group=32,scale=fp8_e4m3,scale2=fp8_e4m3,group2=128,transform=fwht128",
        /* codebooks */
        "table=nf4,group=64,scale=bf16",
        "table=iq4nl,group=32,scale=f16",
        "table=w4nl,group=128,scale=bf16,transform=fwht128",
        "table=w5nl,group=64,scale=fp8_e4m3,scale2=f32",
        "table=w6nl,group=64,scale=fp8_e4m3,scale2=f32,transform=fwht128,rule=search",
        "table=binary,group=32,scale=f16,rule=search",
        "table=ternary,group=32,scale=bf16",
        "table=-1.5;-0.25;0;0.5;2,group=32,scale=f32",
        "table=nf4,group=32,scale=bf16,rule=search",
    };
    const int64_t R = 8, C = 256;
    const std::vector<float> w = draw(R * C, 31, 12.0f);
    std::vector<float> imp((size_t)C);
    for (int64_t c = 0; c < C; ++c) imp[(size_t)c] = 0.25f + (float)((c * 37) % 11);
    const RadQuantizerInfo* q = quantizer("rtn");
    int grids = 0;
    for (const char* spec : kGrids) {
        Opts o = opts_of(spec), ob = opts_of(spec);
        const bool search = std::strstr(spec, "rule=search") != nullptr;
        Quantised a = quantise(q, o, "blk.0.ffn_down.weight", R, C, w,
                               search ? imp.data() : nullptr, R);
        if (a.status != RAD_OK) std::fprintf(stderr, "    refused: %s\n", spec);
        REQUIRE_EQ(a.status, RAD_OK);
        ++grids;

        /* A row block at a time gives the same bytes as one call. */
        const RadQuantWeight qw = weight_of("blk.0.ffn_down.weight", R, C);
        const int64_t rb = q->row_block(o.data(), o.n(), &qw);
        if (rb > 0 && rb < R) {
            const Quantised b = quantise(q, ob, "blk.0.ffn_down.weight", R, C, w,
                                         search ? imp.data() : nullptr, rb);
            REQUIRE_EQ(b.status, RAD_OK);
            bool same = true;
            for (size_t k = 0; k < a.planes.size(); ++k) same = same && a.planes[k] == b.planes[k];
            if (!same) std::fprintf(stderr, "    banded differs: %s\n", spec);
            CHECK(same);
        }

        /* Decode in the decoder's own domain, before any transform. */
        RadEncoding e = a.enc;
        const int64_t n = rad_enc_fwht(&e);
        e.transform[0] = 0;
        std::vector<float> t = w;
        if (n > 0)
            for (int64_t r = 0; r < R; ++r)
                for (int64_t c = 0; c < C; c += n) {
                    fwht_inplace(t.data() + r * C + c, n);
                    for (int64_t i = 0; i < n; ++i) t[(size_t)(r * C + c + i)] /= (float)n;
                }
        auto decode_as = [&](const std::vector<std::vector<uint8_t>>& planes, std::vector<float>* out) {
            std::vector<const void*> d;
            for (const auto& pl : planes) d.push_back(pl.data());
            EncodedView v;
            enc_view(e, R, C, d.data(), &v);
            out->assign((size_t)(R * C), 0.0f);
            return enc_decode_rows(v, 0, R, out->data()) == RAD_OK;
        };
        std::vector<float> got;
        REQUIRE(decode_as(a.planes, &got));
        /* Nearest is relative to the stored scale, which a scale lost to rounding (zero, or off by
         * the transform's order) would still satisfy; the whole decode must also be the weight. */
        const double whole = rel_err(w, decode(a, R, C));
        if (!(whole < 1.0)) std::fprintf(stderr, "    %s: decodes at relative error %g\n", spec, whole);
        CHECK(whole < 1.0);

        /* Every code the plane's dtype holds, written into every element. */
        const uint32_t cdt = e.plane[0].dtype;
        const int bits = rad_dtype_bits(cdt);
        REQUIRE(bits >= 1 && bits <= 8);
        std::vector<float> best((size_t)(R * C), INFINITY), cand;
        std::vector<std::vector<uint8_t>> pl = a.planes;
        const int64_t row_bytes = rad_enc_row_bytes(cdt, C);
        for (int code = 0; code < (1 << bits); ++code) {
            for (int64_t r = 0; r < R; ++r)
                for (int64_t c = 0; c < C; ++c)
                    rad_store_code(pl[0].data() + r * row_bytes, cdt, c, code);
            if (!decode_as(pl, &cand)) continue;               /* past a codebook's end */
            for (size_t i = 0; i < cand.size(); ++i)
                if (std::isfinite(cand[i]))
                    best[i] = std::min(best[i], std::fabs(t[i] - cand[i]));
        }
        int bad = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const float err = std::fabs(t[i] - got[i]);
            /* A tie, or an f32 rounding of the division away from one. */
            if (err > best[i] + 1e-6f * (std::fabs(t[i]) + std::fabs(got[i])) + 1e-30f) {
                if (bad < 3)
                    std::fprintf(stderr, "    %s: element %zu is %.8g, stored as %.8g, but a code "
                                 "decodes %.8g from it\n", spec, i, t[i], got[i], best[i]);
                ++bad;
            }
        }
        CHECK_EQ(bad, 0);
    }
    CHECK_EQ(grids, (int)(sizeof kGrids / sizeof kGrids[0]));
}

TEST(w4nl_at_64_with_e4m3_scales_under_a_row_scale_beats_128_with_bf16_at_the_same_bytes) {
    NEED_LIBQUANT();
    /* The rotated expert grid two ways at the same bytes per row: a bf16 scale per 128 columns,
     * or an E4M3 scale per 64 under a bf16 scale per row. The second searches its sub-scales as
     * stored (rounded to E4M3 relative to the row scale), so rounding does not give back what the
     * finer groups won. Row magnitude varies along K, so 64-column groups pay off. */
    const int64_t R = 64, C = 1024;
    std::vector<float> w = draw(R * C, 21, 1.0f);
    for (int64_t r = 0; r < R; ++r)
        for (int64_t c = 0; c < C; ++c)
            w[(size_t)(r * C + c)] *= 1.0f + 3.0f * (float)((c / 64 + r) % 5) / 4.0f;
    Opts a, b;
    a.s("table", "w4nl").i("group", 128).s("scale", "bf16").s("transform", "fwht128")
     .s("rule", "search");
    b.s("table", "w4nl").i("group", 64).s("scale", "fp8_e4m3").s("scale2", "bf16")
     .s("block2", "1x*").s("transform", "fwht128").s("rule", "search");
    Quantised qa = quantise(quantizer("rtn"), a, "blk.0.ffn_gate_up_exps.0.weight", R, C, w);
    Quantised qb = quantise(quantizer("rtn"), b, "blk.0.ffn_gate_up_exps.0.weight", R, C, w);
    REQUIRE_EQ(qa.status, RAD_OK);
    REQUIRE_EQ(qb.status, RAD_OK);
    std::fprintf(stderr, "    %s\n    %s\n", enc_name(qa.enc).c_str(), enc_name(qb.enc).c_str());
    size_t ba = 0, bb = 0;
    for (const auto& p : qa.planes) ba += p.size();
    for (const auto& p : qb.planes) bb += p.size();
    const double ea = rel_err(w, decode(qa, R, C)), eb = rel_err(w, decode(qb, R, C));
    std::fprintf(stderr, "    g128 bf16: %.4f (%zu bytes)   g64 e4m3 + row bf16: %.4f (%zu bytes)\n",
                 ea, ba, eb, bb);
    CHECK(eb < ea * 0.97);
    CHECK(bb <= ba + (size_t)R * 2 + 64);

    /* The same grid with the second level fixed to one f32 for the tensor: the E4M3 sub-scales
     * carry each row's magnitude themselves (this weight's scales sit well inside E4M3's range),
     * and the error matches the per-row form at two bytes less per row. */
    Opts c;
    c.s("table", "w4nl").i("group", 64).s("scale", "fp8_e4m3").s("scale2", "f32")
     .s("block2", "*x*").f("scale2_value", 0x1p-13).s("transform", "fwht128").s("rule", "search");
    Quantised qc = quantise(quantizer("rtn"), c, "blk.0.ffn_gate_up_exps.0.weight", R, C, w);
    REQUIRE_EQ(qc.status, RAD_OK);
    std::fprintf(stderr, "    %s\n", enc_name(qc.enc).c_str());
    float s2 = 0.0f;
    std::memcpy(&s2, qc.planes[2].data(), 4);
    CHECK_EQ(s2, 0x1p-13f);
    const double ec = rel_err(w, decode(qc, R, C));
    std::fprintf(stderr, "    g64 e4m3 under a fixed 2^-13: %.4f\n", ec);
    CHECK(ec < ea * 0.97);
    CHECK(std::fabs(ec - eb) < 0.002);
    /* A value the second level's dtype cannot hold exactly is refused. */
    Opts d;
    d.s("table", "w4nl").i("group", 64).s("scale", "fp8_e4m3").s("scale2", "bf16")
     .s("block2", "*x*").f("scale2_value", 0.1).s("transform", "fwht128");
    CHECK(quantise(quantizer("rtn"), d, "x", R, C, w).status != RAD_OK);
}

TEST(gptq_cd_and_act_order_are_refused_together) {
    NEED_LIBQUANT();
    const int64_t R = 16, C = 128;
    const std::vector<float> w = draw(R * C, 12, 4.0f);
    Opts o;
    o.s("codes", "i4").i("group", 32).s("scale", "f32").i("act_order", 1).i("cd", 2);
    Quantised q = quantise(quantizer("gptq"), o, "blk.0.ffn_down_exps.0.weight", R, C, w);
    CHECK(q.status != RAD_OK);
}

TEST(a_codebook_indexes_its_table_and_the_scale_multiplies_it) {
    NEED_LIBQUANT();
    const int64_t R = 32, C = 256;
    const std::vector<float> w = draw(R * C, 11, 8.0f);
    /* NF4: sixteen f32 quantiles, a u4 code each, the extremes at +-1 so the scale is absmax */
    {
        Opts o;
        o.s("table", "nf4").i("group", 64).s("scale", "bf16");
        Quantised q = quantise(quantizer("rtn"), o, "x", R, C, w);
        REQUIRE_EQ(q.status, RAD_OK);
        CHECK_EQ(enc_name(q.enc),
                 std::string("affine:codes=u4[1x1],scale=bf16[1x64],table=f32{1x16}"));
        float t[16];
        std::memcpy(t, q.planes[2].data(), sizeof t);
        CHECK_EQ(t[0], -1.0f);
        CHECK_EQ(t[7], 0.0f);
        CHECK_EQ(t[15], 1.0f);
        const double e = rel_err(w, decode(q, R, C));
        std::fprintf(stderr, "    nf4 g64: %.4f\n", e);
        CHECK(e < 0.15);
    }
    /* IQ4_NL's values have unequal ends, so a block's scale takes its largest value's sign and
     * that value lands exactly on -127 (to the bf16 of the scale) */
    {
        Opts o;
        o.s("table", "iq4nl").i("group", 32).s("scale", "f32");
        Quantised q = quantise(quantizer("rtn"), o, "x", R, C, w);
        REQUIRE_EQ(q.status, RAD_OK);
        const std::vector<float> d = decode(q, R, C);
        REQUIRE(!d.empty());
        int off = 0;
        for (int64_t r = 0; r < R; ++r)
            for (int64_t g = 0; g < C / 32; ++g) {
                int64_t at = g * 32;
                for (int64_t c = g * 32; c < g * 32 + 32; ++c)
                    if (std::fabs(w[(size_t)(r * C + c)]) > std::fabs(w[(size_t)(r * C + at)]))
                        at = c;
                off += std::fabs(d[(size_t)(r * C + at)] - w[(size_t)(r * C + at)]) >
                       1e-6f * std::fabs(w[(size_t)(r * C + at)]);
            }
        CHECK_EQ(off, 0);
        CHECK(rel_err(w, d) < 0.15);
    }
    /* one bit: every value decodes to plus or minus its block's scale */
    {
        Opts o;
        o.s("table", "binary").i("group", 128).s("scale", "bf16");
        Quantised q = quantise(quantizer("rtn"), o, "x", R, C, w);
        REQUIRE_EQ(q.status, RAD_OK);
        CHECK_EQ(q.enc.plane[0].dtype, (uint32_t)RAD_U1);
        const std::vector<float> d = decode(q, R, C);
        int bad = 0;
        for (int64_t r = 0; r < R; ++r)
            for (int64_t c = 0; c < C; ++c) {
                const float s = rad_bf16_to_f32(((const uint16_t*)q.planes[1].data())[r * 2 + c / 128]);
                bad += std::fabs(d[(size_t)(r * C + c)]) != s ||
                       (w[(size_t)(r * C + c)] < 0) != (d[(size_t)(r * C + c)] < 0);
            }
        CHECK_EQ(bad, 0);
    }
}

/* A vq= codebook file is supplied data. Its entry count is checked while the loader appends
 * (the widest code is u12, so codewords past 4096 could not be indexed), and its "# unit" line
 * names the unit only with a finite positive value: `inf` parses through %lf and passes a
 * `!(unit > 0.0)` guard, where a NaN falls back to the entries' rms. */
TEST(the_codebook_loader_caps_while_reading_and_refuses_an_infinite_unit) {
    NEED_LIBQUANT();
    TempDir dir("vqcap");
    auto write_cb = [&](const char* name, const char* head, int lines) -> std::string {
        const std::string p = dir.path + "/" + name;
        /* No REQUIRE in here: its failure spelling returns void, and this lambda returns a
         * path. An unopenable path yields an empty/short file, which the loaders refuse --
         * the surrounding CHECKs catch it either way. */
        std::FILE* f = std::fopen(p.c_str(), "wb");
        if (f != nullptr) {
            if (head) std::fprintf(f, "%s\n", head);
            for (int i = 0; i < lines; ++i) std::fprintf(f, "%d %d %d %d\n", i % 5 - 2, 3, -4, 5);
            std::fclose(f);
        }
        return p;
    };
    const int64_t R = 8, C = 128;
    const std::vector<float> w = draw(R * C, 5, 0.0f);

    /* vq= is gptq's path -- rtn refuses it ("vq= is quantised by gptq") -- so every case here
     * goes through gptq and every refusal below is the LOADER's, not a quantizer mismatch. */
    /* One past the cap: refused by name. The u12 codes of an encoding with 4097 entries could
     * not index its own last codeword. */
    {
        Opts o;
        o.s("vq", write_cb("big.txt", "# unit 1.0", 4097).c_str()).i("group", 128).s("scale", "bf16");
        CHECK(quantise(quantizer("gptq"), o, "x", R, C, w).status != RAD_OK);
    }
    /* The cap itself is the largest honest codebook: it reads. */
    {
        Opts o;
        o.s("vq", write_cb("cap.txt", "# unit 1.0", 4096).c_str()).i("group", 128).s("scale", "bf16");
        CHECK_EQ(quantise(quantizer("gptq"), o, "x", R, C, w).status, RAD_OK);
    }
    /* "# unit inf" would multiply every decoded value by infinity. The line is ignored, the
     * entries' own rms takes over and the decode stays finite. */
    {
        Opts o;
        o.s("vq", write_cb("inf.txt", "# unit inf", 256).c_str()).i("group", 128).s("scale", "bf16");
        Quantised q = quantise(quantizer("gptq"), o, "x", R, C, w);
        REQUIRE_EQ(q.status, RAD_OK);
        const std::vector<float> d = decode(q, R, C);
        REQUIRE(!d.empty());
        int bad = 0;
        for (float x : d) bad += !std::isfinite(x);
        CHECK_EQ(bad, 0);
    }
    /* tcq's sibling flow: a zero tcq_gain would divide the unit by zero and turn every code
     * into inf; the loader refuses it by name. */
    {
        Opts o;
        o.s("codes", "u2").i("tcq", 10).i("tcq_rows", 256).i("group", 128).s("scale", "bf16")
            .f("tcq_gain", 0.0);
        CHECK(quantise(quantizer("gptq"), o, "x", R, C, w).status != RAD_OK);
    }
}

/* A trellis grid (tcq=L): each column's codes are the bits a 2^L-state bitshift trellis shifts in,
 * row by row, and a value is its state's table entry times the block's scale. Its own scheme,
 * decoded by libquant's hook: the decode is the values the Viterbi search chose, and at the same
 * two bits a weight it is nearer the weight than the scalar grid. */
TEST(a_trellis_grid_decodes_through_its_states_and_beats_the_scalar_grid) {
    NEED_LIBQUANT();
    const int64_t R = 512, C = 256;
    const std::vector<float> w = draw(R * C, 21, 8.0f);
    Opts t, s;
    t.s("codes", "u2").i("tcq", 10).i("tcq_rows", 256).i("group", 128).s("scale", "bf16")
        .s("transform", "fwht128");
    s.s("codes", "i2").i("group", 128).s("scale", "bf16").s("transform", "fwht128")
        .s("rule", "search");
    Quantised q = quantise(quantizer("gptq"), t, "x", R, C, w);
    Quantised r = quantise(quantizer("gptq"), s, "x", R, C, w);
    REQUIRE_EQ(q.status, RAD_OK);
    REQUIRE_EQ(r.status, RAD_OK);
    CHECK_EQ(std::string(q.enc.scheme), std::string("trellis"));
    const RadEncPlane* tp = rad_enc_plane(&q.enc, "trellis");
    REQUIRE(tp != nullptr);
    CHECK_EQ(tp->extent[0], (int64_t)1024);
    CHECK_EQ(tp->block[0], (int64_t)256);
    const std::vector<float> d = decode(q, R, C), dr = decode(r, R, C);
    REQUIRE(!d.empty());
    REQUIRE(!dr.empty());
    const double et = rel_err(w, d), es = rel_err(w, dr);
    std::fprintf(stderr, "    two bits: trellis %.4f, scalar %.4f\n", et, es);
    CHECK(et < es * 0.9);

    /* Its scale in two levels, E4M3 a block under one f32 for the tensor: the decode reads a block
     * at the bf16 of their product, which the codes were chosen against, and the error barely
     * moves for a byte a block less. */
    Opts t2;
    t2.s("codes", "u2").i("tcq", 10).i("tcq_rows", 256).i("group", 128).s("scale", "fp8_e4m3")
        .s("scale2", "f32").s("block2", "*x*").s("transform", "fwht128");
    Quantised q2 = quantise(quantizer("gptq"), t2, "x", R, C, w);
    REQUIRE_EQ(q2.status, RAD_OK);
    CHECK_EQ(q2.enc.n_planes, 4);
    CHECK(rad_enc_plane(&q2.enc, "scale.1") != nullptr);
    const std::vector<float> d2 = decode(q2, R, C);
    REQUIRE(!d2.empty());
    const double e2 = rel_err(w, d2);
    std::fprintf(stderr, "    two bits, E4M3 under f32: trellis %.4f\n", e2);
    CHECK(e2 < et * 1.02);

    /* Its values on the grid an int8 GEMM reads with one scale per 8 rows (tcq_rb=8): the table
     * plane carries the rows, the decode rounds each row onto its block's largest scale as the
     * search did, and the error stays the trellis's. */
    Opts t3;
    t3.s("codes", "u2").i("tcq", 10).i("tcq_rows", 256).i("tcq_rb", 8).i("group", 128)
        .s("scale", "fp8_e4m3").s("scale2", "f32").s("block2", "*x*").s("transform", "fwht128");
    Quantised q3 = quantise(quantizer("gptq"), t3, "x", R, C, w);
    REQUIRE_EQ(q3.status, RAD_OK);
    const RadEncPlane* tp3 = rad_enc_plane(&q3.enc, "trellis");
    REQUIRE(tp3 != nullptr);
    CHECK_EQ(tp3->block[1], (int64_t)8);
    CHECK_EQ(tp->block[1], (int64_t)1);
    const std::vector<float> d3 = decode(q3, R, C);
    REQUIRE(!d3.empty());
    const double e3 = rel_err(w, d3);
    std::fprintf(stderr, "    two bits, one int8 scale per 8 rows: trellis %.4f\n", e3);
    CHECK(e3 < e2 * 1.02);
    /* tcq_rb must divide the run */
    Opts t4;
    t4.s("codes", "u2").i("tcq", 10).i("tcq_rows", 256).i("tcq_rb", 24).i("group", 128)
        .s("scale", "bf16").s("transform", "fwht128");
    CHECK_EQ(quantise(quantizer("gptq"), t4, "x", R, C, w).status, RAD_E_INVAL);

    /* Coordinate descent over a trellis searches whole columns again against the Hessian's
     * metric; with no calibration there is none, and GPTQ's codes stand. */
    Opts c = t;
    c.i("cd", 1);
    const Quantised qc = quantise(quantizer("gptq"), c, "x", R, C, w);
    REQUIRE_EQ(qc.status, RAD_OK);
    CHECK(decode(qc, R, C) == d);
    /* A trellis's codes are unsigned bits, fewer than its state's. */
    Opts i;
    i.s("codes", "i2").i("tcq", 10).i("group", 128).s("scale", "bf16").s("transform", "fwht128");
    CHECK_EQ(quantise(quantizer("gptq"), i, "x", R, C, w).status, RAD_E_INVAL);
}

TEST(nvfp4_is_two_scale_levels_the_second_a_tensor) {
    NEED_LIBQUANT();
    const int64_t R = 16, C = 256;
    const std::vector<float> w = draw(R * C, 12, 30.0f);
    Opts o;
    o.s("codes", "fp4_e2m1").i("group", 16).s("scale", "fp8_e4m3").s("scale2", "f32");
    Quantised q = quantise(quantizer("rtn"), o, "x", R, C, w);
    REQUIRE_EQ(q.status, RAD_OK);
    CHECK_EQ(enc_name(q.enc), std::string("fp4_e2m1*fp8_e4m3[1x16]*f32[*x*]"));
    /* the second level is the largest block's ideal scale (amax / 6) over E4M3's top, so that
     * block's sub-scale is 448 */
    float amax = 0;
    for (float v : w) amax = std::fmax(amax, std::fabs(v));
    float s2;
    std::memcpy(&s2, q.planes[2].data(), 4);
    CHECK_EQ(s2, (amax / 6.0f) / 448.0f);
    int top = 0;
    for (uint8_t b : q.planes[1]) top += b == 0x7e;
    CHECK(top >= 1);
    const double e = rel_err(w, decode(q, R, C));
    std::fprintf(stderr, "    nvfp4: %.4f\n", e);
    CHECK(e < 0.15);
}

TEST(an_integer_sub_scale_under_a_superblock_scale) {
    NEED_LIBQUANT();
    /* a K-quant's shape: 4-bit codes, a six-bit sub-scale per 32, an f16 per 256 */
    const int64_t R = 8, C = 512;
    const std::vector<float> w = draw(R * C, 13, 6.0f);
    Opts o;
    o.s("codes", "i4").i("group", 32).s("scale", "u6").s("scale2", "f16").i("group2", 256);
    const RadQuantizerInfo* rtn = quantizer("rtn");
    RadQuantWeight qw{};
    qw.name = "x";
    qw.rank = 2; qw.shape[0] = R; qw.shape[1] = C; qw.rows = R; qw.cols = C;
    CHECK_EQ(rtn->row_block(o.data(), o.n(), &qw), 1);
    Quantised q = quantise(rtn, o, "x", R, C, w);
    REQUIRE_EQ(q.status, RAD_OK);
    CHECK_EQ(enc_name(q.enc), std::string("i4*u6[1x32]*f16[1x256]"));
    /* each superblock's widest group sits at 63 and none is 0 */
    for (int64_t r = 0; r < R; ++r)
        for (int64_t sb = 0; sb < 2; ++sb) {
            int mx = 0, mn = 64;
            for (int64_t g = 0; g < 8; ++g) {
                const int v = (int)rad_load_f32(q.planes[1].data() + r * 12, RAD_U6, sb * 8 + g);
                mx = std::max(mx, v);
                mn = std::min(mn, v);
            }
            CHECK_EQ(mx, 63);
            CHECK(mn >= 1);
        }
    CHECK(rel_err(w, decode(q, R, C)) < 0.2);
}

TEST(sixteen_and_thirty_two_bit_codes_are_nearly_exact) {
    NEED_LIBQUANT();
    const int64_t R = 8, C = 128;
    const std::vector<float> w = draw(R * C, 14, 5.0f);
    Opts a, b;
    a.s("codes", "i16").s("scale", "f32");
    b.s("codes", "i32").s("scale", "f32");
    Quantised qa = quantise(quantizer("rtn"), a, "x", R, C, w);
    Quantised qb = quantise(quantizer("rtn"), b, "x", R, C, w);
    REQUIRE_EQ(qa.status, RAD_OK);
    REQUIRE_EQ(qb.status, RAD_OK);
    CHECK_EQ(enc_name(qb.enc), std::string("i32*f32[1x*]"));
    const double ea = rel_err(w, decode(qa, R, C)), eb = rel_err(w, decode(qb, R, C));
    std::fprintf(stderr, "    i16 %.3g, i32 %.3g\n", ea, eb);
    CHECK(ea < 1e-4);
    CHECK(eb < 1e-6);
}

TEST(the_searched_scale_never_loses_to_absmax_and_weighs_by_importance) {
    NEED_LIBQUANT();
    const int64_t R = 32, C = 256;
    const std::vector<float> w = draw(R * C, 15, 12.0f);
    std::vector<float> imp((size_t)C);
    for (int64_t c = 0; c < C; ++c) imp[(size_t)c] = (c % 7 == 0) ? 50.0f : 1.0f;
    for (const char* codes : { "i4", "i3" }) {
        Opts a, b;
        a.s("codes", codes).i("group", 32).s("scale", "bf16");
        b.s("codes", codes).i("group", 32).s("scale", "bf16").s("rule", "search");
        Quantised qa = quantise(quantizer("rtn"), a, "x", R, C, w, imp.data());
        Quantised qb = quantise(quantizer("rtn"), b, "x", R, C, w, imp.data());
        REQUIRE_EQ(qa.status, RAD_OK);
        REQUIRE_EQ(qb.status, RAD_OK);
        const std::vector<float> da = decode(qa, R, C), db = decode(qb, R, C);
        /* per block, in the importance-weighted error the search minimises */
        int worse = 0;
        double ta = 0, tb = 0;
        for (int64_t r = 0; r < R; ++r)
            for (int64_t g = 0; g < C / 32; ++g) {
                double ea = 0, eb = 0;
                for (int64_t c = g * 32; c < g * 32 + 32; ++c) {
                    const size_t i = (size_t)(r * C + c);
                    ea += imp[(size_t)c] * (double)(w[i] - da[i]) * (w[i] - da[i]);
                    eb += imp[(size_t)c] * (double)(w[i] - db[i]) * (w[i] - db[i]);
                }
                worse += eb > ea * (1 + 1e-9);
                ta += ea;
                tb += eb;
            }
        std::fprintf(stderr, "    %s weighted error: absmax %.4g, search %.4g\n", codes, ta, tb);
        CHECK_EQ(worse, 0);
        CHECK(tb < ta * 0.9);
    }
    /* and asymmetric */
    Opts a, b;
    a.s("codes", "u3").s("zero", "u4").i("group", 32).s("scale", "f16");
    b.s("codes", "u3").s("zero", "u4").i("group", 32).s("scale", "f16").s("rule", "search");
    Quantised qa = quantise(quantizer("rtn"), a, "x", R, C, w);
    Quantised qb = quantise(quantizer("rtn"), b, "x", R, C, w);
    REQUIRE_EQ(qa.status, RAD_OK);
    REQUIRE_EQ(qb.status, RAD_OK);
    CHECK(rel_err(w, decode(qb, R, C)) <= rel_err(w, decode(qa, R, C)));
}

TEST(cast_converts_to_nearest_even_with_no_scale) {
    NEED_LIBQUANT();
    const int64_t R = 4, C = 64;
    const std::vector<float> w = draw(R * C, 16, 100.0f);
    const struct { const char* dt; uint32_t code; } cases[] = {
        { "bf16", RAD_BF16 }, { "f16", RAD_F16 }, { "fp8_e4m3", RAD_F8E4M3 }, { "f32", RAD_F32 } };
    for (const auto& k : cases) {
        Opts o;
        o.s("dtype", k.dt);
        Quantised q = quantise(quantizer("cast"), o, "x", R, C, w);
        REQUIRE_EQ(q.status, RAD_OK);
        CHECK_EQ(enc_name(q.enc), std::string(k.dt));
        int off = 0;
        for (int64_t i = 0; i < R * C; ++i) {
            std::vector<uint8_t> one(8);     /* rad_store_f32 writes up to eight bytes */
            rad_store_f32(one.data(), k.code, 0, w[(size_t)i]);
            off += std::memcmp(one.data(), q.planes[0].data() + i * rad_dtype_bytes(k.code, 1),
                               (size_t)rad_dtype_bytes(k.code, 1)) != 0;
        }
        CHECK_EQ(off, 0);
    }
    Opts bad;
    bad.s("dtype", "i4");
    CHECK(quantise(quantizer("cast"), bad, "x", R, C, w).status != RAD_OK);
}

TEST(a_grid_the_options_do_not_describe_is_refused) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* q = quantizer("rtn");
    RadQuantWeight qw{};
    qw.name = "x";
    qw.rank = 2; qw.shape[0] = 4; qw.shape[1] = 256; qw.rows = 4; qw.cols = 256;
    RadEncoding e;
    auto refused = [&](Opts o) { return q->encoding(o.data(), o.n(), &qw, &e) != RAD_OK; };
    { Opts o; o.s("codes", "u4").i("group", 32);                          CHECK(refused(o)); }
    { Opts o; o.s("codes", "i4").s("zero", "u8");                         CHECK(refused(o)); }
    { Opts o; o.s("codes", "fp4_e2m1").s("scale", "e8m0").s("rule", "absmax"); CHECK(refused(o)); }
    { Opts o; o.s("codes", "i4").i("group", 32).s("block", "1x32");       CHECK(refused(o)); }
    { Opts o; o.s("codes", "i4").s("transform", "fwht96");                CHECK(refused(o)); }
    { Opts o; o.s("codes", "bf16");                                       CHECK(refused(o)); }
    { Opts o; o.s("table", "nf4").s("zero", "u8");                        CHECK(refused(o)); }
    { Opts o; o.s("table", "nf4").s("codes", "u8");                       CHECK(refused(o)); }
    { Opts o; o.s("table", "1;2;x");                                      CHECK(refused(o)); }
    { Opts o; o.s("table", "iq4nl").s("scale", "u6").s("scale2", "f16");  CHECK(refused(o)); }
    { Opts o; o.s("codes", "i4").s("scale", "u6");                        CHECK(refused(o)); }
    { Opts o; o.s("codes", "i4").i("group", 32).s("scale2", "f16").i("group2", 48);
                                                                          CHECK(refused(o)); }
    { Opts o; o.s("codes", "i4").i("group2", 256);                        CHECK(refused(o)); }
    { Opts o; o.s("codes", "fp4_e2m1").s("scale", "e8m0").s("scale2", "f32"); CHECK(refused(o)); }
    { Opts o; o.s("codes", "i4").s("rule", "fixed").f("scale_value", 1.0).s("scale2", "f32");
                                                                          CHECK(refused(o)); }
    /* a vector is declined, not an error */
    RadQuantWeight v = qw;
    v.rank = 1; v.shape[0] = 256; v.rows = 1;
    {
        Opts o;
        o.s("codes", "i4");
        CHECK_EQ(q->encoding(o.data(), o.n(), &v, &e), RAD_E_UNSUPPORTED);
    }
}

/* A scheme a quantiser defines is decoded through that quantiser. The core decodes plain and
 * affine only; for any other scheme enc_decode_rows asks the loaded quantisers' decode hooks
 * (abi/rad_quant.h), which is how the oracle, rad-kbench and rad-convert's error report read a
 * format the engine does not know. Once the plugin is unloaded, the weight is refused by name. */
TEST(a_quantisers_own_scheme_decodes_through_its_hook_and_only_while_it_is_loaded) {
    const std::string so = home() + "/testquant/radtest_quant.so";
    struct stat st {};
    if (stat(so.c_str(), &st) != 0) ARTIFACT_MISSING(so);
    const int64_t R = 8, C = 64;
    const std::vector<float> w = draw(R * C, 7, 4.0f);
    Quantised q;
    {
        Registry r;
        REQUIRE_EQ(r.load_plugin(so, 0), RAD_OK);
        const Registry::QuantRow* qr = r.quantizer("radtest-halves");
        REQUIRE(qr && qr->info);
        Opts o;
        q = quantise(qr->info, o, "blk.0.ffn_down.weight", R, C, w);
        REQUIRE_EQ(q.status, RAD_OK);
        CHECK_EQ(std::string(q.enc.scheme), std::string("radtest-halves"));
        const std::vector<float> d = decode(q, R, C);
        REQUIRE_EQ(d.size(), w.size());
        for (size_t i = 0; i < w.size(); ++i) CHECK(std::fabs(d[i] - w[i]) <= 0.25f + 1e-6f);
        /* rows from the middle, as the oracle and rad-convert ask for them */
        std::vector<const void*> pd;
        for (auto& p : q.planes) pd.push_back(p.data());
        EncodedView v;
        enc_view(q.enc, R, C, pd.data(), &v);
        std::vector<float> part((size_t)(2 * C));
        REQUIRE_EQ(enc_decode_rows(v, 3, 2, part.data()), RAD_OK);
        for (int64_t i = 0; i < 2 * C; ++i) CHECK_EQ(part[(size_t)i], d[(size_t)(3 * C + i)]);
    }
    /* the registry closed: its decoder went with it */
    std::vector<const void*> pd;
    for (auto& p : q.planes) pd.push_back(p.data());
    EncodedView v;
    enc_view(q.enc, R, C, pd.data(), &v);
    std::vector<float> out((size_t)(R * C));
    std::string why;
    CHECK_EQ(enc_decode_rows(v, 0, R, out.data(), &why), RAD_E_UNSUPPORTED);
    CHECK(why.find("radtest-halves") != std::string::npos);
}

/* ------------------------------------------------------------------ bf16x12
 *
 * A bf16 weight packed losslessly (lq_bf16x12.cpp): the sign and mantissa a byte, the exponent a
 * nibble below its 32-block's largest, and each block that cannot pack whole in an escape table.
 * Lossless means every bf16 code, so the cases run all 65536 through packed and raw blocks and
 * compare bits, NaN payloads included. The planes are what a kernel library's relayout reads, so
 * they are also checked as the format states them, not only through the decoder, which shares the
 * quantiser's idea of them. */
namespace {

uint16_t bf16_code(unsigned sign, unsigned field, unsigned mant) {
    return (uint16_t)(sign << 15 | field << 7 | mant);
}

/* f32 values holding bf16 codes, bit for bit. */
std::vector<float> from_bf16(const std::vector<uint16_t>& h) {
    std::vector<float> v(h.size());
    for (size_t i = 0; i < h.size(); ++i) {
        const uint32_t u = (uint32_t)h[i] << 16;
        std::memcpy(&v[i], &u, 4);
    }
    return v;
}

/* The decoded values whose bits are not their code's; -1 for a decode of the wrong size. */
int64_t bits_off(const std::vector<float>& d, const std::vector<uint16_t>& h) {
    if (d.size() != h.size()) return -1;
    int64_t off = 0;
    for (size_t i = 0; i < h.size(); ++i) {
        uint32_t u;
        std::memcpy(&u, &d[i], 4);
        off += u != (uint32_t)h[i] << 16;
    }
    return off;
}

/* A weight whose blocks all pack: exponent fields 118-127, splitmix signs and mantissas. */
std::vector<uint16_t> narrow_bf16(int64_t n, uint64_t seed) {
    std::vector<uint16_t> h((size_t)n);
    uint64_t z = seed;
    for (uint16_t& v : h) {
        z += 0x9E3779B97F4A7C15ull;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        x ^= x >> 31;
        v = bf16_code((unsigned)(x >> 63), 118u + (unsigned)((x >> 8) % 10u), (unsigned)x & 0x7Fu);
    }
    return h;
}

/* The u32 in a block's first four d bytes: a raw block's row of the escape table. */
uint32_t esc_index(const Quantised& q, int64_t K, int64_t r, int64_t b) {
    uint32_t idx;
    std::memcpy(&idx, q.planes[1].data() + r * (K / 2) + b * 16, 4);
    return idx;
}

/* The planes against the format as it is stated, block by block in row-major order: a packed
 * block's E, codes and offsets; a raw block's 0xFF, zero codes, index (the count of raw blocks
 * before it) and table row; and the table's unused rows zero. Returns the elements that differ,
 * and the raw blocks in `raw`. */
int64_t x12_planes_off(const Quantised& q, int64_t R, int64_t K, const std::vector<uint16_t>& h,
                       int64_t* raw) {
    const uint8_t* codes = q.planes[0].data();
    const uint8_t* d = q.planes[1].data();
    const uint8_t* e = q.planes[2].data();
    const uint8_t* esc = q.planes[3].data();
    const int64_t nb = K / 32, cap = q.enc.plane[3].extent[0];
    int64_t off = 0, n = 0;
    for (int64_t r = 0; r < R; ++r)
        for (int64_t b = 0; b < nb; ++b) {
            const uint16_t* x = &h[(size_t)(r * K + b * 32)];
            const uint8_t* c = codes + r * K + b * 32;
            const uint8_t* db = d + r * (K / 2) + b * 16;
            unsigned hi = 0, lo = 255;
            for (int i = 0; i < 32; ++i) {
                hi = std::max(hi, (x[i] >> 7) & 0xFFu);
                lo = std::min(lo, (x[i] >> 7) & 0xFFu);
            }
            if (hi == 0xFFu || hi - lo > 15) {
                off += e[r * nb + b] != 0xFF;
                off += esc_index(q, K, r, b) != (uint32_t)n;
                for (int i = 4; i < 16; ++i) off += db[i] != 0;
                for (int i = 0; i < 32; ++i) {
                    uint16_t t = 0;
                    if (n < cap) std::memcpy(&t, esc + (n * 32 + i) * 2, 2);
                    off += c[i] != 0;
                    off += n >= cap || t != x[i];
                }
                ++n;
                continue;
            }
            off += e[r * nb + b] != hi;
            for (int i = 0; i < 32; ++i) {
                off += c[i] != ((x[i] >> 15) << 7 | (x[i] & 0x7F));
                off += ((db[i / 2] >> (4 * (i & 1))) & 0xFu) != hi - ((x[i] >> 7) & 0xFFu);
            }
        }
    for (int64_t t = n; t < cap; ++t)
        for (int i = 0; i < 64; ++i) off += esc[t * 64 + i] != 0;
    *raw = n;
    return off;
}

}  /* namespace */

TEST(bf16x12_round_trips_every_bf16_code_through_packed_and_raw_blocks) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* x12 = quantizer("bf16x12");
    REQUIRE(x12 != nullptr);
    const int64_t R = 64, C = 1024;
    int64_t raw = -1;

    /* In code order a block holds one sign and one exponent field, so it packs with every offset
     * 0, field 0 included (the zeros and subnormals, under E = 0), except the eight blocks of field
     * 0xFF, the infinities and NaNs, which are raw. */
    std::vector<uint16_t> h((size_t)(R * C));
    for (size_t v = 0; v < h.size(); ++v) h[v] = (uint16_t)v;
    Opts o;
    Quantised q = quantise(x12, o, "blk.0.ffn_down.weight", R, C, from_bf16(h));
    REQUIRE_EQ(q.status, RAD_OK);
    CHECK_EQ(enc_name(q.enc),
             std::string("bf16x12:codes=u8[1x1],d=u4[1x1],e=u8[1x32],esc=bf16{64x32}"));
    CHECK_EQ(bits_off(decode(q, R, C), h), (int64_t)0);
    CHECK_EQ(x12_planes_off(q, R, C, h, &raw), (int64_t)0);
    CHECK_EQ(raw, (int64_t)8);

    /* Each block two codes of each of 16 consecutive fields, so every offset 0-15 occurs, at
     * different places in different blocks, and fields 0-15 hold the zeros and subnormals under
     * E = 15. The 128 blocks over fields 240-255 are raw, and esc = 128/2048 fills the table. */
    {
        int64_t i = 0;
        for (unsigned g = 0; g < 16; ++g)
            for (unsigned k = 0; k < 128; ++k)
                for (unsigned j = 0; j < 32; ++j) {
                    const unsigned sm = 2 * k + j / 16;
                    h[(size_t)i++] = bf16_code(sm >> 7, 16 * g + (j % 16 + k) % 16, sm & 0x7F);
                }
        std::vector<uint8_t> seen(65536, 0);
        for (uint16_t v : h) seen[v]++;
        CHECK(std::all_of(seen.begin(), seen.end(), [](uint8_t s) { return s == 1; }));
    }
    Opts full;
    full.f("esc", 128.0 / 2048.0);
    q = quantise(x12, full, "blk.0.ffn_down.weight", R, C, from_bf16(h));
    REQUIRE_EQ(q.status, RAD_OK);
    CHECK_EQ(q.enc.plane[3].extent[0], (int64_t)128);
    CHECK_EQ(bits_off(decode(q, R, C), h), (int64_t)0);
    CHECK_EQ(x12_planes_off(q, R, C, h, &raw), (int64_t)0);
    CHECK_EQ(raw, (int64_t)128);
}

/* The edges of a packed block, a block a row: 16 binades pack (offsets 0-15) and 17 do not; a zero
 * packs under a largest field of 15 and makes its block raw above that; an infinity makes its
 * block raw, and the largest finite field packs. Raw blocks take the table's rows in order. */
TEST(bf16x12_packs_sixteen_binades_and_escapes_seventeen) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* x12 = quantizer("bf16x12");
    REQUIRE(x12 != nullptr);
    const int64_t R = 6, C = 32;
    std::vector<uint16_t> h((size_t)(R * C));
    for (unsigned j = 0; j < 32; ++j) {
        const unsigned s = j & 1, m = (j * 37) & 0x7F;
        h[0 * 32 + j] = bf16_code(s, 100 + j % 16, m);
        h[1 * 32 + j] = bf16_code(s, 100 + j % 17, m);
        h[2 * 32 + j] = j == 5 ? 0x0000 : bf16_code(s, 15 - j % 3, m);
        h[3 * 32 + j] = j == 5 ? 0x8000 : bf16_code(s, 16, m);
        h[4 * 32 + j] = j == 9 ? 0xFF80 : bf16_code(s, 127, m);
        h[5 * 32 + j] = bf16_code(s, j == 30 ? 239 : 254, m);
    }
    Opts o;
    Quantised q = quantise(x12, o, "blk.0.attn_k.weight", R, C, from_bf16(h));
    REQUIRE_EQ(q.status, RAD_OK);
    CHECK_EQ(q.enc.plane[3].extent[0], (int64_t)6);       /* at most a row a block */
    CHECK_EQ(bits_off(decode(q, R, C), h), (int64_t)0);
    int64_t raw = -1;
    CHECK_EQ(x12_planes_off(q, R, C, h, &raw), (int64_t)0);
    CHECK_EQ(raw, (int64_t)3);
    const unsigned want_e[6] = { 115, 0xFF, 15, 0xFF, 0xFF, 254 };
    for (int64_t r = 0; r < R; ++r) CHECK_EQ((unsigned)q.planes[2][(size_t)r], want_e[r]);
    CHECK_EQ(esc_index(q, C, 1, 0), 0u);
    CHECK_EQ(esc_index(q, C, 3, 0), 1u);
    CHECK_EQ(esc_index(q, C, 4, 0), 2u);
}

/* A weight with more raw blocks than its table holds is refused at convert, naming the weight and
 * the option that sizes the table, rather than written wrong. At exactly the table's size it
 * converts, and a larger esc= takes the weight that was refused. */
TEST(bf16x12_refuses_by_name_a_weight_with_more_raw_blocks_than_its_table) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* x12 = quantizer("bf16x12");
    REQUIRE(x12 != nullptr);
    const int64_t R = 32, C = 1024;                   /* 1024 blocks; esc=0.125 holds 128 */
    const std::vector<uint16_t> h0 = narrow_bf16(R * C, 21);
    /* the first n blocks raw: one value 16 fields below the narrowest block's largest */
    auto with_raw = [&](int64_t n) {
        std::vector<uint16_t> h = h0;
        for (int64_t b = 0; b < n; ++b) h[(size_t)(b * 32 + 7)] = bf16_code(0, 102, 5);
        return h;
    };
    Opts o;
    o.f("esc", 0.125);
    const std::vector<uint16_t> at_cap = with_raw(128);
    Quantised q = quantise(x12, o, "blk.59.attn_k.weight", R, C, from_bf16(at_cap));
    REQUIRE_EQ(q.status, RAD_OK);
    int64_t raw = -1;
    CHECK_EQ(x12_planes_off(q, R, C, at_cap, &raw), (int64_t)0);
    CHECK_EQ(raw, (int64_t)128);

    const std::vector<uint16_t> over = with_raw(129);
    const std::string err = stderr_of([&] {
        q = quantise(x12, o, "blk.59.attn_k.weight", R, C, from_bf16(over));
    });
    CHECK_EQ(q.status, (int)RAD_E_FULL);
    CHECK(has(err, "blk.59.attn_k.weight") && has(err, "129 of its 1024 blocks are raw"));
    CHECK(has(err, "esc=0.125 gives its escape table 128"));

    Opts more;
    more.f("esc", 0.2);
    q = quantise(x12, more, "blk.59.attn_k.weight", R, C, from_bf16(over));
    REQUIRE_EQ(q.status, RAD_OK);
    CHECK_EQ(q.enc.plane[3].extent[0], (int64_t)205);
    CHECK_EQ(bits_off(decode(q, R, C), over), (int64_t)0);
}

/* Lossless only for bf16: an f32 with low bits set is refused, the value named. */
TEST(bf16x12_refuses_a_value_that_is_not_a_bf16) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* x12 = quantizer("bf16x12");
    REQUIRE(x12 != nullptr);
    const int64_t R = 4, C = 64;
    std::vector<float> w = from_bf16(narrow_bf16(R * C, 5));
    w[(size_t)(2 * C + 37)] = 0.1f;                   /* 0x3DCCCCCD */
    Opts o;
    Quantised q;
    const std::string err = stderr_of([&] { q = quantise(x12, o, "blk.3.ffn_down.weight", R, C, w); });
    CHECK_EQ(q.status, (int)RAD_E_INVAL);
    CHECK(has(err, "blk.3.ffn_down.weight") && has(err, "row 2, column 37 is not a bf16"));
}

/* The decode hook reads any rows, and a rank's share alone: a column half or a row range of each
 * tiled plane with the whole table, as tensor parallelism cuts them, since an index counts from the
 * weight's first row and every share carries the whole table. A malformed block is RAD_E_FORMAT,
 * not a read past the table, and another scheme is left to its own decoder. */
TEST(bf16x12_decodes_through_its_hook_any_rows_and_a_ranks_share) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* x12 = quantizer("bf16x12");
    REQUIRE(x12 != nullptr);
    const int64_t R = 48, C = 256, nb = C / 32;
    std::vector<uint16_t> h = narrow_bf16(R * C, 9);
    /* 55 raw blocks of 384, in both halves of the rows and of the columns; the table holds 64 */
    for (int64_t b = 0; b < R * nb; b += 7) h[(size_t)(b * 32 + b % 32)] = bf16_code(1, 90, 3);
    Opts o;
    const Quantised q = quantise(x12, o, "blk.0.ssm_out.weight", R, C, from_bf16(h));
    REQUIRE_EQ(q.status, RAD_OK);
    CHECK_EQ(bits_off(decode(q, R, C), h), (int64_t)0);

    std::vector<const void*> pd;
    for (auto& p : q.planes) pd.push_back(p.data());
    EncodedView v;
    enc_view(q.enc, R, C, pd.data(), &v);
    std::vector<float> part((size_t)(4 * C));
    REQUIRE_EQ(enc_decode_rows(v, 5, 4, part.data()), RAD_OK);
    CHECK_EQ(bits_off(part, std::vector<uint16_t>(h.begin() + 5 * C, h.begin() + 9 * C)),
             (int64_t)0);

    /* rows [r0, r1) and columns [c0, c1) of each tiled plane, and the table whole */
    auto share = [&](int64_t r0, int64_t r1, int64_t c0, int64_t c1) {
        const int64_t per[3] = { 1, 2, 32 };          /* values a byte of codes, d and e */
        Quantised s;
        s.enc = q.enc;
        for (int k = 0; k < 3; ++k) {
            const int64_t row = C / per[k], b0 = c0 / per[k], b1 = c1 / per[k];
            std::vector<uint8_t> p;
            for (int64_t r = r0; r < r1; ++r)
                p.insert(p.end(), q.planes[(size_t)k].begin() + r * row + b0,
                         q.planes[(size_t)k].begin() + r * row + b1);
            s.planes.push_back(std::move(p));
        }
        s.planes.push_back(q.planes[3]);
        std::vector<uint16_t> want;
        for (int64_t r = r0; r < r1; ++r)
            want.insert(want.end(), h.begin() + r * C + c0, h.begin() + r * C + c1);
        return bits_off(decode(s, r1 - r0, c1 - c0), want);
    };
    CHECK_EQ(share(0, R, C / 2, C), (int64_t)0);
    CHECK_EQ(share(0, R, 0, C / 2), (int64_t)0);
    CHECK_EQ(share(R / 2, R, 0, C), (int64_t)0);

    /* a raw block's index past the table, and a packed block's offset above its E */
    auto refused = [&](Quantised m) {
        std::vector<const void*> md;
        for (auto& p : m.planes) md.push_back(p.data());
        EncodedView mv;
        enc_view(m.enc, R, C, md.data(), &mv);
        std::vector<float> out((size_t)(R * C));
        std::string why;
        return enc_decode_rows(mv, 0, R, out.data(), &why) == RAD_E_FORMAT && has(why, "bf16x12");
    };
    {
        Quantised m = q;
        const uint32_t past = (uint32_t)q.enc.plane[3].extent[0];
        REQUIRE_EQ((int)m.planes[2][0], 0xFF);
        std::memcpy(m.planes[1].data(), &past, 4);
        CHECK(refused(m));
    }
    {
        Quantised m = q;
        REQUIRE((int)m.planes[2][1] != 0xFF);
        m.planes[2][1] = 0;
        CHECK(refused(m));
    }

    RadEncoding other = rad_enc_affine(RAD_I8, RAD_BF16, 1, 32);
    const RadQuantWeight qw = weight_of("x", R, C);
    std::vector<float> out((size_t)(R * C));
    CHECK_EQ(x12->decode(&other, &qw, pd.data(), 0, R, out.data()), (int)RAD_E_UNSUPPORTED);
}

/* What it takes: a weight of whole 32-blocks, with a table of max(64, ceil(blocks x esc)) rows and
 * at most a row a block (Qwen3.8-27B's attn_k and gate_up at the default 0.004 among them); a
 * vector or a ragged row is declined and kept as it is, and an esc= outside [0, 1] refused. */
TEST(bf16x12_sizes_its_table_from_the_shape_and_declines_what_it_cannot_pack) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* x12 = quantizer("bf16x12");
    REQUIRE(x12 != nullptr);
    auto cap = [&](int64_t rows, int64_t cols, double esc) -> int64_t {
        Opts o;
        if (esc >= 0) o.f("esc", esc);
        const RadQuantWeight w = weight_of("blk.0.attn_k.weight", rows, cols);
        RadEncoding e;
        if (x12->encoding(o.data(), o.n(), &w, &e) != RAD_OK) return -1;
        return e.plane[3].extent[0];
    };
    CHECK_EQ(cap(256, 4096, -1), (int64_t)132);
    CHECK_EQ(cap(1024, 5120, -1), (int64_t)656);
    CHECK_EQ(cap(34816, 5120, -1), (int64_t)22283);
    CHECK_EQ(cap(64, 1024, 0.0), (int64_t)64);
    CHECK_EQ(cap(4, 64, -1), (int64_t)8);
    CHECK_EQ(cap(16, 64, 1.0), (int64_t)32);

    RadEncoding e;
    Opts none;
    RadQuantWeight vec = weight_of("blk.0.attn_norm.weight", 1, 5120);
    vec.rank = 1;
    vec.shape[0] = 5120;
    CHECK_EQ(x12->encoding(none.data(), none.n(), &vec, &e), (int)RAD_E_UNSUPPORTED);
    const RadQuantWeight ragged = weight_of("blk.0.ssm_conv1d.weight", 10240, 4);
    CHECK_EQ(x12->encoding(none.data(), none.n(), &ragged, &e), (int)RAD_E_UNSUPPORTED);
    for (double bad : { -0.1, 1.5 }) {
        Opts o;
        o.f("esc", bad);
        const RadQuantWeight w = weight_of("blk.0.attn_k.weight", 64, 1024);
        int st = RAD_OK;
        const std::string err = stderr_of([&] { st = x12->encoding(o.data(), o.n(), &w, &e); });
        CHECK_EQ(st, (int)RAD_E_INVAL);
        CHECK(has(err, "blk.0.attn_k.weight") && has(err, "esc="));
    }
}

RAD_TEST_MAIN()
