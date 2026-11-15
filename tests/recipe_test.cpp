/* recipe_test.cpp -- the rules rad-convert quantises by (core/format/recipe.h).
 *
 * A recipe decides every weight of a conversion that can take hours, and a rule that matches the
 * wrong weights (or none) shows only when the container is measured afterwards. These cases pin
 * the matching, the order and the two spellings to spec §4.4.
 */
#include "rad_test.h"
#include "format/recipe.h"

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

using namespace rad;

TEST(a_glob_matches_the_whole_name) {
    CHECK(recipe_glob("blk.*.ffn_*_exps.*.weight", "blk.12.ffn_down_exps.511.weight"));
    CHECK(recipe_glob("*", "output.weight"));
    CHECK(recipe_glob("output.weight", "output.weight"));
    CHECK(recipe_glob("blk.?.attn_q.weight", "blk.7.attn_q.weight"));
    CHECK(!recipe_glob("blk.?.attn_q.weight", "blk.17.attn_q.weight"));   /* ? is one character */
    CHECK(recipe_glob("blk.*.weight", "blk.1.2.3.weight"));               /* * crosses dots */
    CHECK(!recipe_glob("blk.*.weight", "blk.1.scale"));
    CHECK(!recipe_glob("output", "output.weight"));                        /* not a prefix match */
    CHECK(recipe_glob("*_exps.*", "blk.0.ffn_gate_up_exps.3.weight"));
    CHECK(recipe_glob("a*b*c", "aXXbYYbZc"));                              /* backtracks */
    CHECK(!recipe_glob("a*b*c", "aXXbYY"));
}

TEST(the_first_rule_that_matches_decides) {
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_text("blk.*.ffn_down_exps.*.weight  gptq codes=i4 group=128\n"
                          "blk.*.weight                  rtn  codes=fp8_e4m3 block=128x128\n",
                          "r", &why), RAD_OK);
    const RecipeRule* a = r.match("blk.3.ffn_down_exps.7.weight");
    const RecipeRule* b = r.match("blk.3.attn_q.weight");
    REQUIRE(a && b);
    CHECK_EQ(a->quantizer, std::string("gptq"));
    CHECK_EQ(b->quantizer, std::string("rtn"));
    CHECK(r.match("output.weight") == nullptr);                 /* kept as the checkpoint holds it */
}

TEST(a_command_line_rule_goes_ahead_of_the_file) {
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_flag("output.weight=rtn:codes=fp8_e4m3,block=128x128,scale=bf16", &why),
               RAD_OK);
    REQUIRE_EQ(r.add_text("* rtn codes=i8 group=128\n", "file", &why), RAD_OK);
    const RecipeRule* m = r.match("output.weight");
    REQUIRE(m != nullptr);
    CHECK_EQ(m->origin, std::string("--quant"));
    CHECK_EQ(m->options.size(), (size_t)3);
    CHECK_EQ(m->options[0].first, std::string("codes"));
    CHECK_EQ(m->options[2].second, std::string("bf16"));
    CHECK_EQ(r.match("blk.0.attn_q.weight")->origin, std::string("file:1"));
}

TEST(comments_and_blank_lines_are_not_rules_and_a_bad_line_is_named) {
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_text("# a recipe\n\n   \nout* rtn codes=i8  # trailing\n", "x.recipe", &why),
               RAD_OK);
    CHECK_EQ(r.rules().size(), (size_t)1);
    CHECK_EQ(r.rules()[0].options.size(), (size_t)1);
    CHECK_EQ(r.rules()[0].origin, std::string("x.recipe:4"));

    Recipe bad;
    CHECK_EQ(bad.add_text("a rtn\nlonely\n", "y.recipe", &why), RAD_E_FORMAT);
    CHECK(why.find("y.recipe:2") != std::string::npos);
    CHECK_EQ(bad.add_text("a rtn group\n", "z", &why), RAD_E_FORMAT);   /* not key=value */
    CHECK_EQ(bad.add_text("a rtn k=1 k=2\n", "z", &why), RAD_E_FORMAT); /* given twice */
    CHECK(why.find("twice") != std::string::npos);
    CHECK_EQ(bad.add_flag("noquantiser", &why), RAD_E_FORMAT);
    CHECK_EQ(bad.add_flag("p=:k=v", &why), RAD_E_FORMAT);
}

/* $NAME expands from the environment, so one recipe serves every machine. An unset name is
 * refused: quantising against an empty calibration directory would silently do something else. */
TEST(a_variable_is_the_environments_and_an_unset_one_is_refused) {
    ::setenv("RAD_RECIPE_TEST_DIR", "/data/calib", 1);
    ::unsetenv("RAD_RECIPE_TEST_UNSET");
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_text("* gptq calib=$RAD_RECIPE_TEST_DIR/q38 tag=${RAD_RECIPE_TEST_DIR}x\n",
                          "v", &why), RAD_OK);
    CHECK_EQ(r.rules()[0].options[0].second, std::string("/data/calib/q38"));
    CHECK_EQ(r.rules()[0].options[1].second, std::string("/data/calibx"));
    CHECK_EQ(r.add_text("* gptq calib=$RAD_RECIPE_TEST_UNSET\n", "v", &why), RAD_E_FORMAT);
    CHECK(why.find("RAD_RECIPE_TEST_UNSET") != std::string::npos);
    CHECK_EQ(r.add_text("* gptq calib=${RAD_RECIPE_TEST_DIR\n", "v", &why), RAD_E_FORMAT);
}

/* A container records the options in one canonical spelling, so --reuse compares two
 * conversions by what they asked for, not by how the rule was typed. */
TEST(the_recorded_options_do_not_depend_on_how_they_were_written) {
    Recipe a, b;
    std::string why;
    REQUIRE_EQ(a.add_text("* rtn scale=bf16 codes=i4 group=128\n", "a", &why), RAD_OK);
    REQUIRE_EQ(b.add_flag("*=rtn:group=128,codes=i4,scale=bf16", &why), RAD_OK);
    CHECK_EQ(a.rules()[0].options_text(), b.rules()[0].options_text());
    CHECK_EQ(a.rules()[0].options_text(), std::string("codes=i4,group=128,scale=bf16"));
}

/* The recipe a container carries reads back as the same rules. */
TEST(the_recorded_recipe_reads_back_as_the_same_rules) {
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_flag("output.weight=rtn:codes=fp8_e4m3,block=128x128", &why), RAD_OK);
    REQUIRE_EQ(r.add_text("blk.*.weight gptq codes=i4 group=128 damp=0.02\n", "f", &why), RAD_OK);
    Recipe back;
    REQUIRE_EQ(back.add_text(r.text(), "header", &why), RAD_OK);
    REQUIRE_EQ(back.rules().size(), r.rules().size());
    for (size_t i = 0; i < r.rules().size(); ++i) {
        CHECK_EQ(back.rules()[i].pattern, r.rules()[i].pattern);
        CHECK_EQ(back.rules()[i].quantizer, r.rules()[i].quantizer);
        CHECK(back.rules()[i].options == r.rules()[i].options);
    }
}

namespace {
bool has_opt(const RecipeRule* r, const char* k, const char* v) {
    for (const auto& [kk, vv] : r->options)
        if (kk == k && vv == v) return true;
    return false;
}
bool has_key(const RecipeRule* r, const char* k) {
    for (const auto& kv : r->options)
        if (kv.first == k) return true;
    return false;
}
}  /* namespace */

TEST(the_q38_native_mtp_overlay_only_changes_draft_weights) {
    const std::string src = __FILE__;
    std::ifstream in(src.substr(0, src.find_last_of('/')) + "/../data/recipes/q38-27b-native-mtp-u2.recipe");
    REQUIRE(in.good());
    Recipe r;
    std::string why;
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    REQUIRE_EQ(r.add_text(text, "mtp.recipe", &why), RAD_OK);
    CHECK_EQ(r.rules().size(), (size_t)7);
    for (const char* linear : {"attn_qg", "attn_k", "attn_v", "attn_output", "ffn_gate_up", "ffn_down"}) {
        const auto* rule = r.match(std::string("blk.64.") + linear + ".weight");
        REQUIRE(rule != nullptr);
        CHECK_EQ(rule->quantizer, std::string("rtn"));
        CHECK_EQ(rule->options_text(), std::string("codes=i8,group=128,scale=bf16,transform=fwht128"));
    }
    const auto* head = r.match("mtp.draft_head.weight");
    REQUIRE(head != nullptr);
    CHECK_EQ(head->quantizer, std::string("rtn"));
    CHECK_EQ(head->options_text(), std::string("codes=u2,group=128,scale=f16,zero=u8"));
    for (const char* n : {"blk.63.attn_qg.weight", "output.k4.weight", "output.weight",
                          "token_embd.weight", "blk.64.attn_norm.weight", "mtp.fc.weight"})
        CHECK(r.match(n) == nullptr);
}

/* The calibrated native-MTP overlay keeps the int8 block at RTN and fits only the U2 draft head. */
TEST(the_q38_native_mtp_gptq_overlay_uses_the_draft_head_gram) {
    const std::string src = __FILE__;
    std::ifstream in(src.substr(0, src.find_last_of('/')) +
                     "/../data/recipes/q38-27b-native-mtp-u2-gptq.recipe");
    REQUIRE(in.good());
    Recipe r;
    std::string why;
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const char* old_calib = std::getenv("MTP_CALIB");
    const std::string saved_calib = old_calib ? old_calib : "";
    setenv("MTP_CALIB", "/tmp/qwen35-mtp-calib", 1);
    const int parse_rc = r.add_text(text, "mtp-gptq.recipe", &why);
    if (old_calib) setenv("MTP_CALIB", saved_calib.c_str(), 1);
    else unsetenv("MTP_CALIB");
    REQUIRE_EQ(parse_rc, RAD_OK);
    CHECK_EQ(r.rules().size(), (size_t)7);
    for (const char* linear : {"attn_qg", "attn_k", "attn_v", "attn_output", "ffn_gate_up", "ffn_down"}) {
        const auto* rule = r.match(std::string("blk.64.") + linear + ".weight");
        REQUIRE(rule != nullptr);
        CHECK_EQ(rule->quantizer, std::string("rtn"));
        CHECK(has_opt(rule, "codes", "i8"));
        CHECK(has_opt(rule, "group", "128"));
        CHECK(has_opt(rule, "scale", "bf16"));
        CHECK(has_opt(rule, "transform", "fwht128"));
    }
    const auto* head = r.match("mtp.draft_head.weight");
    REQUIRE(head != nullptr);
    CHECK_EQ(head->quantizer, std::string("gptq"));
    CHECK(has_opt(head, "codes", "u2"));
    CHECK(has_opt(head, "zero", "u8"));
    CHECK(has_opt(head, "group", "128"));
    CHECK(has_opt(head, "scale", "f16"));
    CHECK(has_opt(head, "calib", "/tmp/qwen35-mtp-calib"));
    CHECK(has_opt(head, "damp", "0.01"));
    CHECK(r.match("mtp.fc.weight") == nullptr);
    CHECK(r.match("output.k4.weight") == nullptr);
}

/* The Sage-A1.5 recipe as it ships: every trunk linear -- attention, the delta net's in- and
 * out-projections, the routed and shared experts of all 40 blocks and the MTP layer at its own
 * index 40 -- goes to block fp8, the head likewise, and the routers, the shared-expert gates,
 * the delta net's parameters, the norms, the embedding and the vision tower keep the
 * checkpoint's form. */
TEST(the_sage_recipe_decides_the_model_it_was_written_for) {
    std::string dir = __FILE__;
    dir = dir.substr(0, dir.find_last_of('/')) + "/../data/recipes";
    std::ifstream in(dir + "/sage-a1.5-fp8.recipe");
    REQUIRE(in.good());
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_text(text, "sage-a1.5-fp8.recipe", &why), RAD_OK);
    CHECK_EQ(r.rules().size(), (size_t)12);

    const char* linears[] = { "blk.0.attn_qg.weight", "blk.3.attn_qg.weight", "blk.39.attn_qg.weight",
                              "blk.40.attn_qg.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight",
                              "blk.39.attn_output.weight", "blk.0.ssm_inz.weight", "blk.39.ssm_out.weight",
                              "blk.0.ffn_gate_up_exps.0.weight", "blk.19.ffn_down_exps.255.weight",
                              "blk.40.ffn_gate_up_exps.0.weight", "blk.40.ffn_down_exps.255.weight",
                              "blk.0.ffn_gate_up_shexp.weight", "blk.40.ffn_down_shexp.weight" };
    for (const char* n : linears) {
        const RecipeRule* m = r.match(n);
        REQUIRE(m != nullptr);
        CHECK_EQ(m->quantizer, std::string("rtn"));
        CHECK(has_opt(m, "codes", "fp8_e4m3"));
        CHECK(has_opt(m, "block", "128x128"));
        CHECK(has_opt(m, "scale", "bf16"));
    }
    const RecipeRule* head = r.match("output.weight");
    REQUIRE(head != nullptr);
    CHECK(has_opt(head, "codes", "fp8_e4m3"));
    CHECK(has_opt(head, "block", "128x128"));
    const RecipeRule* dh = r.match("mtp.draft_head.weight");
    REQUIRE(dh != nullptr);
    CHECK(has_opt(dh, "codes", "u2"));
    CHECK(has_opt(dh, "zero", "u8"));
    CHECK(has_opt(dh, "group", "128"));
    CHECK(has_opt(dh, "scale", "f16"));

    /* kept as the checkpoint ships them: the routers, the gates, the delta net's parameters,
     * the norms, the embedding and the whole vision tower */
    const char* kept[] = { "blk.0.ffn_norm.weight", "blk.0.attn_norm.weight", "blk.0.ssm_ab.weight",
                           "blk.0.ssm_conv1d.weight", "blk.0.ssm_a_log", "blk.0.ssm_dt.bias",
                           "blk.0.ssm_norm.weight", "blk.0.ffn_gate_inp.weight",
                           "blk.0.ffn_gate_shexp.weight", "blk.40.ffn_gate_inp.weight",
                           "token_embd.weight", "output_norm.weight", "mtp.fc.weight",
                           "mtp.norm.weight", "v.blk.0.qkv.weight", "v.merger.fc1.weight" };
    for (const char* n : kept)
        CHECK(r.match(n) == nullptr);
}

/* The two Sage-A1.5 containers with the DFlash2 drafter in place of the MTP head: the trunk's
 * rules unchanged in form (block fp8, or MXFP4 for the 4-bit sibling, whose lm_head stays block
 * fp8), the drafter's linears in the trunk's form, its draft head the 2-bit copy, and no rule for
 * an MTP head the container does not carry. The trunk's `blk.*` patterns do not reach the
 * drafter's `dflash.blk.*` names, and the drafter's norms, selector and convolution stay as the
 * checkpoint holds them. */
TEST(the_sage_dflash2_recipes_decide_the_trunk_and_the_drafter) {
    std::string dir = __FILE__;
    dir = dir.substr(0, dir.find_last_of('/')) + "/../data/recipes";
    struct Case { const char* file; const char* codes; const char* grid_key; const char* grid; };
    const Case cases[] = { { "sage-a1.5-fp8-df2.recipe", "fp8_e4m3", "block", "128x128" },
                           { "sage-a1.5-mxfp4-df2.recipe", "fp4_e2m1", "group", "32" } };
    for (const Case& c : cases) {
        std::ifstream in(dir + "/" + c.file);
        REQUIRE(in.good());
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        Recipe r;
        std::string why;
        REQUIRE_EQ(r.add_text(text, c.file, &why), RAD_OK);
        CHECK_EQ(r.rules().size(), (size_t)19);

        const char* linears[] = { "blk.0.attn_qg.weight", "blk.39.attn_k.weight",
                                  "blk.3.attn_v.weight", "blk.39.attn_output.weight",
                                  "blk.0.ssm_inz.weight", "blk.39.ssm_out.weight",
                                  "blk.0.ffn_gate_up_exps.0.weight",
                                  "blk.39.ffn_down_exps.255.weight",
                                  "blk.0.ffn_gate_up_shexp.weight", "blk.39.ffn_down_shexp.weight",
                                  "dflash.fc.weight", "dflash.blk.0.attn_q.weight",
                                  "dflash.blk.5.attn_k.weight", "dflash.blk.5.attn_v.weight",
                                  "dflash.blk.5.attn_output.weight",
                                  "dflash.blk.0.ffn_gate_up.weight",
                                  "dflash.blk.5.ffn_down.weight" };
        for (const char* n : linears) {
            const RecipeRule* m = r.match(n);
            REQUIRE(m != nullptr);
            CHECK_EQ(m->quantizer, std::string("rtn"));
            CHECK(has_opt(m, "codes", c.codes));
            CHECK(has_opt(m, c.grid_key, c.grid));
        }
        /* A drafter linear is decided by its own rule, not a trunk pattern. */
        CHECK_EQ(r.match("dflash.blk.0.attn_k.weight")->pattern,
                 std::string("dflash.blk.*.attn_k.weight"));

        const RecipeRule* head = r.match("output.weight");
        REQUIRE(head != nullptr);
        CHECK(has_opt(head, "codes", "fp8_e4m3"));
        CHECK(has_opt(head, "block", "128x128"));
        const RecipeRule* dh = r.match("dflash.draft_head.weight");
        REQUIRE(dh != nullptr);
        CHECK(has_opt(dh, "codes", "u2"));
        CHECK(has_opt(dh, "zero", "u8"));
        CHECK(has_opt(dh, "group", "128"));
        CHECK(has_opt(dh, "scale", "f16"));

        const char* kept[] = { "blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_shexp.weight",
                               "blk.0.ssm_ab.weight", "blk.0.attn_norm.weight",
                               "token_embd.weight", "output_norm.weight",
                               "dflash.blk.0.attn_norm.weight", "dflash.norm.weight",
                               "dflash.hidden_norm.weight", "dflash.sel_proj.weight",
                               "dflash.blk.0.attn_conv_proj.weight",
                               "dflash.blk.0.ffn_conv_proj.weight", "mtp.draft_head.weight",
                               "v.blk.0.qkv.weight" };
        for (const char* n : kept)
            CHECK(r.match(n) == nullptr);
    }
}

/* The Muse Glimmer recipes as they ship. The bf16 one is documentation only -- no rule, so every
 * weight keeps the checkpoint's form. The w4a8 one routes every trunk linear to int4 on the
# 128-column grid (the value projection to int8), and nothing else: the layer norms, the
 * embedding and the head match no rule. The patterns are the plugin's declared names:
 * separate q and gate projections (this family's q_proj/gate_proj are two tensors), the gate|up
 * concat, and the sliding/full two-group trunk. */
TEST(the_museglimmer_recipes_decide_the_model_they_were_written_for) {
    const std::string dir = std::string(__FILE__).substr(
                                0, std::string(__FILE__).find_last_of('/')) + "/../data/recipes";
    {
        std::ifstream in(dir + "/museglimmer-30b-bf16.recipe");
        REQUIRE(in.good());
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        Recipe r;
        std::string why;
        REQUIRE_EQ(r.add_text(text, "museglimmer-30b-bf16.recipe", &why), RAD_OK);
        CHECK_EQ(r.rules().size(), (size_t)0);
        CHECK(r.match("blk.3.attn_q.weight") == nullptr);
    }
    {
        std::ifstream in(dir + "/museglimmer-30b-w4a8.recipe");
        REQUIRE(in.good());
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        Recipe r;
        std::string why;
        REQUIRE_EQ(r.add_text(text, "museglimmer-30b-w4a8.recipe", &why), RAD_OK);
        CHECK_EQ(r.rules().size(), (size_t)7);
        const char* i4[] = { "blk.0.attn_q.weight", "blk.51.attn_k.weight", "blk.7.attn_gate.weight",
                             "blk.3.attn_output.weight", "blk.3.ffn_gate_up.weight",
                             "blk.3.ffn_down.weight" };
        for (const char* n : i4) {
            const RecipeRule* m = r.match(n);
            REQUIRE(m != nullptr);
            CHECK_EQ(m->quantizer, std::string("rtn"));
            CHECK(has_opt(m, "codes", "i4"));
            CHECK(has_opt(m, "group", "128"));
            CHECK(has_opt(m, "scale", "f16"));
            CHECK(has_opt(m, "rule", "search"));
        }
        const RecipeRule* v = r.match("blk.3.attn_v.weight");
        REQUIRE(v != nullptr);
        CHECK(has_opt(v, "codes", "i8"));
        const char* kept[] = { "blk.3.attn_norm.weight", "blk.3.attn_post.weight",
                               "blk.3.ffn_norm.weight", "blk.3.ffn_post.weight",
                               "token_embd.weight", "output_norm.weight", "output.weight",
                               "model.vision_tower.layers.3.attn.q_proj.weight" };
        for (const char* n : kept)
            CHECK(r.match(n) == nullptr);
    }
}

/* museglimmer-30b-iq8r as it ships: all seven trunk linears of every layer IQ8R, the head plain
 * int8 (no rotation), and nothing else: the norms, the embedding, the vision tower and a merged
 * drafter's `dspark.*` matrices match no rule. */
TEST(the_museglimmer_iq8r_recipe_decides_the_model_it_was_written_for) {
    const std::string dir = std::string(__FILE__).substr(
                                0, std::string(__FILE__).find_last_of('/')) + "/../data/recipes";
    std::ifstream in(dir + "/museglimmer-30b-iq8r.recipe");
    REQUIRE(in.good());
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_text(text, "museglimmer-30b-iq8r.recipe", &why), RAD_OK);
    CHECK_EQ(r.rules().size(), (size_t)8);
    const char* iq8r[] = { "blk.0.attn_q.weight", "blk.7.attn_gate.weight", "blk.51.attn_k.weight",
                           "blk.3.attn_v.weight", "blk.3.attn_output.weight",
                           "blk.0.ffn_gate_up.weight", "blk.51.ffn_down.weight" };
    for (const char* n : iq8r) {
        const RecipeRule* m = r.match(n);
        REQUIRE(m != nullptr);
        CHECK_EQ(m->quantizer, std::string("rtn"));
        CHECK(has_opt(m, "codes", "i8"));
        CHECK(has_opt(m, "group", "256"));
        CHECK(has_opt(m, "scale", "u8"));
        CHECK(has_opt(m, "scale2", "bf16"));
        CHECK(has_opt(m, "block2", "1x*"));
        CHECK(has_opt(m, "transform", "fwht128"));
        CHECK(has_opt(m, "clamp", "sym"));
        CHECK(has_opt(m, "rule", "search"));
    }
    const RecipeRule* head = r.match("output.weight");
    REQUIRE(head != nullptr);
    CHECK(has_opt(head, "codes", "i8"));
    CHECK(has_opt(head, "group", "128"));
    CHECK(has_opt(head, "scale", "bf16"));
    CHECK(!has_opt(head, "transform", "fwht128"));
    const char* kept[] = { "blk.3.attn_norm.weight", "blk.3.attn_post.weight",
                           "blk.3.ffn_norm.weight", "blk.3.ffn_post.weight", "token_embd.weight",
                           "output_norm.weight", "v.blk.0.attn.q_proj.weight", "v.proj.weight",
                           "dspark.fc.weight", "dspark.blk.0.attn_q.weight",
                           "dspark.blk.4.ffn_down.weight",
                           "model.vision_tower.layers.3.attn.q_proj.weight" };
    for (const char* n : kept)
        CHECK(r.match(n) == nullptr);
}

/* Muse Glimmer IQ4R-V4 uses computed-MAD trellises across the trunk and a compact U3/U4
 * DFlash2. The target head, embedding, norms, vision, selector and convolutions remain BF16. */
TEST(the_museglimmer_iq4r_v4_recipe_selects_the_trellis_trunk) {
    const std::string dir = std::string(__FILE__).substr(
                                0, std::string(__FILE__).find_last_of('/')) + "/../data/recipes";
    std::ifstream in(dir + "/museglimmer-30b-iq4r-v4.recipe");
    REQUIRE(in.good());
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    Recipe r;
    std::string why;
    const char* old_calib = std::getenv("CALIB");
    const std::string saved_calib = old_calib ? old_calib : "";
    setenv("CALIB", "/tmp/muse-glimmer-calib", 1);
    const int parse_rc = r.add_text(text, "museglimmer-30b-iq4r-v4.recipe", &why);
    if (old_calib) setenv("CALIB", saved_calib.c_str(), 1);
    else unsetenv("CALIB");
    REQUIRE_EQ(parse_rc, RAD_OK);
    CHECK_EQ(r.rules().size(), (size_t)14);
    for (const char* n : { "blk.0.attn_q.weight", "blk.51.attn_gate.weight",
                           "blk.12.attn_k.weight", "blk.12.attn_v.weight",
                           "blk.0.ffn_gate_up.weight", "blk.51.ffn_down.weight" }) {
        const RecipeRule* m = r.match(n);
        REQUIRE(m != nullptr);
        CHECK_EQ(m->quantizer, std::string("gptq"));
        CHECK(has_opt(m, "tcq", "16"));
        CHECK(has_opt(m, "tcq_code", "mad"));
        CHECK(has_opt(m, "transform", "fwht128"));
        if (std::string(n).find("ffn_") != std::string::npos)
            CHECK(has_opt(m, "calib", "/tmp/muse-glimmer-calib"));
        else
            CHECK(!has_key(m, "calib"));
        CHECK(has_opt(m, "tcq_order", "1"));
    }
    const RecipeRule* kv = r.match("blk.1.attn_k.weight");
    REQUIRE(kv != nullptr);
    CHECK(has_opt(kv, "tcq_rows", "128"));
    for (const char* n : { "dflash.fc.weight", "dflash.blk.0.attn_q.weight",
                           "dflash.blk.4.attn_output.weight", "dflash.blk.0.ffn_gate_up.weight",
                           "dflash.blk.4.ffn_down.weight" }) {
        const RecipeRule* m = r.match(n);
        REQUIRE(m != nullptr);
        CHECK_EQ(m->quantizer, std::string("gptq"));
        CHECK(has_opt(m, "tcq", "16"));
        CHECK(has_opt(m, "tcq_code", "mad"));
        CHECK(has_opt(m, "tcq_rows", "256"));
        CHECK(has_opt(m, "transform", "fwht128"));
        CHECK(!has_key(m, "calib"));
    }
    CHECK(has_opt(r.match("dflash.blk.0.ffn_down.weight"), "codes", "u3"));
    CHECK(has_opt(r.match("dflash.blk.0.attn_q.weight"), "codes", "u4"));
    const RecipeRule* o = r.match("blk.3.attn_output.weight");
    REQUIRE(o != nullptr);
    CHECK(has_opt(o, "codes", "u5"));
    CHECK(has_opt(o, "scale", "bf16"));
    CHECK(!has_opt(o, "tcq_order", "1"));
    for (const char* n : { "output.weight", "token_embd.weight", "blk.3.attn_norm.weight",
                           "dflash.blk.0.attn_norm.weight", "dflash.sel_pred", "v.proj.weight",
                           "dspark.fc.weight" })
        CHECK(r.match(n) == nullptr);
}

/* S-V4 text rules plus int8 vision. The patch stem is not named, so it stays BF16. */
TEST(the_museglimmer_iq4r_s_v4_i8vision_recipe_quantises_the_tower_linears) {
    const std::string dir = std::string(__FILE__).substr(
                                0, std::string(__FILE__).find_last_of('/')) + "/../data/recipes";
    std::ifstream in(dir + "/museglimmer-30b-iq4r-s-v4-i8vision.recipe");
    REQUIRE(in.good());
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    Recipe r;
    std::string why;
    const char* old_calib = std::getenv("CALIB");
    const std::string saved_calib = old_calib ? old_calib : "";
    setenv("CALIB", "/tmp/muse-glimmer-calib", 1);
    const int parse_rc = r.add_text(text, "museglimmer-30b-iq4r-s-v4-i8vision.recipe", &why);
    if (old_calib) setenv("CALIB", saved_calib.c_str(), 1);
    else unsetenv("CALIB");
    REQUIRE_EQ(parse_rc, RAD_OK);
    for (const char* n : { "v.blk.0.qkv.weight", "v.blk.49.fc2.weight", "v.mlp.fc1.weight",
                           "v.proj.weight" }) {
        const RecipeRule* m = r.match(n);
        REQUIRE(m != nullptr);
        CHECK_EQ(m->quantizer, std::string("rtn"));
        CHECK(has_opt(m, "codes", "i8"));
        CHECK(has_opt(m, "group", "128"));
        CHECK(has_opt(m, "scale", "bf16"));
    }
    CHECK(r.match("v.patch.weight") == nullptr);
    CHECK(r.match("v.blk.0.norm1.weight") == nullptr);
    CHECK(r.match("v.blk.0.qkv.bias") == nullptr);
    CHECK(has_opt(r.match("blk.0.ffn_down.weight"), "codes", "u3"));
    CHECK(has_opt(r.match("blk.0.ffn_down.weight"), "calib", "/tmp/muse-glimmer-calib"));
}

/* The Flint recipes as they ship, over arch/flint's declared names. flint-iq8r: every trunk linear
 * (attention, the two dense layers, the shared expert's gate/up) IQ8R, the routed experts and the
 * shared expert's down projection rotated int8 at a bf16 scale a (row, 128 of K), the head plain
 * int8. flint-bf16-q4: u4 attention, signed int4 experts, plain int8 dense, shared and head. In
 * both the norms, the sinks, the router and its zero correction, and the embedding match no rule. */
TEST(the_flint_recipes_decide_the_model_they_were_written_for) {
    const std::string dir = std::string(__FILE__).substr(
                                0, std::string(__FILE__).find_last_of('/')) + "/../data/recipes";
    const char* kept[] = { "blk.0.attn_norm.weight", "blk.0.attn_q_norm.weight",
                           "blk.7.attn_k_norm.weight", "blk.0.attn_sinks", "blk.3.ffn_norm.weight",
                           "blk.3.glm_router.weight", "blk.3.glm_router.correction",
                           "token_embd.weight", "output_norm.weight" };
    {
        std::ifstream in(dir + "/flint-iq8r.recipe");
        REQUIRE(in.good());
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        Recipe r;
        std::string why;
        REQUIRE_EQ(r.add_text(text, "flint-iq8r.recipe", &why), RAD_OK);
        CHECK_EQ(r.rules().size(), (size_t)13);
        const char* iq8r[] = { "blk.0.attn_q.weight", "blk.48.attn_k.weight", "blk.3.attn_v.weight",
                               "blk.5.attn_output.weight", "blk.0.ffn_gate_up.weight",
                               "blk.1.ffn_down.weight", "blk.2.ffn_gate_up_shexp.weight" };
        for (const char* n : iq8r) {
            const RecipeRule* m = r.match(n);
            REQUIRE(m != nullptr);
            CHECK_EQ(m->quantizer, std::string("rtn"));
            CHECK(has_opt(m, "codes", "i8"));
            CHECK(has_opt(m, "group", "256"));
            CHECK(has_opt(m, "scale", "u8"));
            CHECK(has_opt(m, "scale2", "bf16"));
            CHECK(has_opt(m, "transform", "fwht128"));
        }
        const char* i8h[] = { "blk.2.ffn_gate_up_exps.0.weight", "blk.48.ffn_down_exps.255.weight",
                              "blk.17.ffn_down_shexp.weight" };
        for (const char* n : i8h) {
            const RecipeRule* m = r.match(n);
            REQUIRE(m != nullptr);
            CHECK(has_opt(m, "codes", "i8"));
            CHECK(has_opt(m, "group", "128"));
            CHECK(has_opt(m, "scale", "bf16"));
            CHECK(has_opt(m, "transform", "fwht128"));
        }
        const RecipeRule* head = r.match("output.weight");
        REQUIRE(head != nullptr);
        CHECK(has_opt(head, "codes", "i8"));
        CHECK(!has_opt(head, "transform", "fwht128"));
        for (const char* n : kept) CHECK(r.match(n) == nullptr);
    }
    {
        std::ifstream in(dir + "/flint-bf16-q4.recipe");
        REQUIRE(in.good());
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        Recipe r;
        std::string why;
        REQUIRE_EQ(r.add_text(text, "flint-bf16-q4.recipe", &why), RAD_OK);
        const RecipeRule* a = r.match("blk.9.attn_output.weight");
        REQUIRE(a != nullptr);
        CHECK(has_opt(a, "codes", "u4"));
        CHECK(has_opt(a, "zero", "u8"));
        const RecipeRule* e = r.match("blk.30.ffn_down_exps.7.weight");
        REQUIRE(e != nullptr);
        CHECK(has_opt(e, "codes", "i4"));
        for (const char* n : { "blk.1.ffn_gate_up.weight", "blk.40.ffn_down_shexp.weight",
                               "output.weight" }) {
            const RecipeRule* m = r.match(n);
            REQUIRE(m != nullptr);
            CHECK(has_opt(m, "codes", "i8"));
        }
        for (const char* n : kept) CHECK(r.match(n) == nullptr);
    }
}

/* The Ling-3.0-tiny IQ8R recipe as it ships, over arch/bailing_hybrid's declared names: the KDA
 * projections (fused q|k|v, forget, output gate, out), the MLA out projection, the dense layer's
 * pair and the shared expert's gate/up IQ8R; the routed experts and the shared expert's down
 * projection rotated int8 at a bf16 scale a (row, 128 of K). The KDA beta, the MLA's low-rank q
 * and kv projections, its query parts and its head gate, the convolution taps, the router, the
 * embedding and the head match no rule and keep the checkpoint's bf16; the norms, the KDA
 * parameters and the router are cast to the f32 the plugin declares. */
TEST(the_ling_tiny_iq8r_recipe_decides_the_model_it_was_written_for) {
    const std::string dir = std::string(__FILE__).substr(
                                0, std::string(__FILE__).find_last_of('/')) + "/../data/recipes";
    std::ifstream in(dir + "/ling-3.0-tiny-iq8r.recipe");
    REQUIRE(in.good());
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_text(text, "ling-3.0-tiny-iq8r.recipe", &why), RAD_OK);
    CHECK_EQ(r.rules().size(), (size_t)20);
    const char* iq8r[] = { "blk.0.attn_qkv.weight", "blk.22.attn_forget.weight",
                           "blk.1.attn_gate.weight", "blk.2.attn_output.weight",
                           "blk.3.attn_output.weight", "blk.0.ffn_gate_up.weight",
                           "blk.0.ffn_down.weight", "blk.1.ffn_gate_up_shexp.weight",
                           "blk.23.ffn_gate_up_shexp.weight" };
    for (const char* n : iq8r) {
        const RecipeRule* m = r.match(n);
        REQUIRE(m != nullptr);
        CHECK_EQ(m->quantizer, std::string("rtn"));
        CHECK(has_opt(m, "codes", "i8"));
        CHECK(has_opt(m, "group", "256"));
        CHECK(has_opt(m, "scale", "u8"));
        CHECK(has_opt(m, "scale2", "bf16"));
        CHECK(has_opt(m, "transform", "fwht128"));
    }
    const char* i8h[] = { "blk.1.ffn_gate_up_exps.0.weight", "blk.23.ffn_down_exps.127.weight",
                          "blk.7.ffn_down_shexp.weight" };
    for (const char* n : i8h) {
        const RecipeRule* m = r.match(n);
        REQUIRE(m != nullptr);
        CHECK(has_opt(m, "codes", "i8"));
        CHECK(has_opt(m, "group", "128"));
        CHECK(has_opt(m, "scale", "bf16"));
        CHECK(has_opt(m, "transform", "fwht128"));
    }
    const char* f32[] = { "blk.0.attn_norm.weight", "blk.3.attn_q_a_norm.weight",
                          "blk.3.attn_kv_a_norm.weight", "blk.1.ffn_norm.weight",
                          "blk.0.ssm_norm.weight", "blk.0.ssm_a", "blk.0.ssm_dt_bias",
                          "blk.1.glm_router.weight", "blk.1.glm_router.correction",
                          "output_norm.weight" };
    for (const char* n : f32) {
        const RecipeRule* m = r.match(n);
        REQUIRE(m != nullptr);
        CHECK_EQ(m->quantizer, std::string("cast"));
        CHECK(has_opt(m, "dtype", "f32"));
    }
    const char* kept[] = { "blk.0.attn_beta.weight", "blk.3.attn_q_a.weight",
                           "blk.3.attn_q_rope.weight", "blk.3.attn_q_nope.weight",
                           "blk.3.attn_kv_a.weight", "blk.3.attn_kv_b.weight",
                           "blk.3.attn_head_gate.weight", "blk.0.ssm_conv.weight",
                           "token_embd.weight", "output.weight" };
    for (const char* n : kept) CHECK(r.match(n) == nullptr);
}

/* The Ling-3.0-flash recipe as it ships, over arch/bailing_hybrid's declared names: the trunk as the
 * tiny IQ8R recipe (KDA projections, MLA out, the two dense layers' pair, the shared expert's
 * gate/up IQ8R; the shared expert's down projection rotated int8 at a bf16 scale a (row, 128 of
 * K)), the 512 routed experts of a layer rotated four-bit codes into the w4nl codebook at an E4M3
 * scale a (row, 64 of K). Flash's MLA has no query LoRA, so no q_a norm is cast. The KDA beta, the
 * MLA's q parts, kv projections and head gate, the convolution taps, the embedding and the head
 * match no rule and keep the checkpoint's bf16. The experts name their statistics through $CALIB,
 * so the test sets it. */
TEST(the_ling_flash_iq8r_w4nl64_recipe_decides_the_model_it_was_written_for) {
    const char* old_calib = std::getenv("CALIB");
    const std::string saved_calib = old_calib ? old_calib : "";
    struct RestoreCalib {
        bool had;
        const std::string& value;
        ~RestoreCalib() {
            if (had) ::setenv("CALIB", value.c_str(), 1);
            else ::unsetenv("CALIB");
        }
    } restore_calib{ old_calib != nullptr, saved_calib };
    ::setenv("CALIB", "/data/calib", 1);
    const std::string dir = std::string(__FILE__).substr(
                                0, std::string(__FILE__).find_last_of('/')) + "/../data/recipes";
    std::ifstream in(dir + "/ling-3.0-flash-iq8r-w4nl64.recipe");
    REQUIRE(in.good());
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_text(text, "ling-3.0-flash-iq8r-w4nl64.recipe", &why), RAD_OK);
    CHECK_EQ(r.rules().size(), (size_t)18);
    const char* iq8r[] = { "blk.0.attn_qkv.weight", "blk.40.attn_forget.weight",
                           "blk.1.attn_gate.weight", "blk.2.attn_output.weight",
                           "blk.5.attn_output.weight", "blk.0.ffn_gate_up.weight",
                           "blk.1.ffn_down.weight", "blk.2.ffn_gate_up_shexp.weight",
                           "blk.41.ffn_gate_up_shexp.weight" };
    for (const char* n : iq8r) {
        const RecipeRule* m = r.match(n);
        REQUIRE(m != nullptr);
        CHECK_EQ(m->quantizer, std::string("rtn"));
        CHECK(has_opt(m, "codes", "i8"));
        CHECK(has_opt(m, "group", "256"));
        CHECK(has_opt(m, "scale", "u8"));
        CHECK(has_opt(m, "scale2", "bf16"));
        CHECK(has_opt(m, "transform", "fwht128"));
    }
    const char* w4nl[] = { "blk.2.ffn_gate_up_exps.0.weight", "blk.41.ffn_down_exps.511.weight" };
    for (const char* n : w4nl) {
        const RecipeRule* m = r.match(n);
        REQUIRE(m != nullptr);
        CHECK_EQ(m->quantizer, std::string("gptq"));
        CHECK(has_opt(m, "table", "w4nl"));
        CHECK(has_opt(m, "group", "64"));
        CHECK(has_opt(m, "scale", "fp8_e4m3"));
        CHECK(has_opt(m, "transform", "fwht128"));
        CHECK(has_opt(m, "calib", "/data/calib"));
    }
    {
        const RecipeRule* m = r.match("blk.7.ffn_down_shexp.weight");
        REQUIRE(m != nullptr);
        CHECK_EQ(m->quantizer, std::string("rtn"));
        CHECK(has_opt(m, "codes", "i8"));
        CHECK(has_opt(m, "group", "128"));
        CHECK(has_opt(m, "scale", "bf16"));
        CHECK(has_opt(m, "transform", "fwht128"));
    }
    const char* f32[] = { "blk.0.attn_norm.weight", "blk.5.attn_kv_a_norm.weight",
                          "blk.2.ffn_norm.weight", "blk.0.ssm_norm.weight", "blk.0.ssm_a",
                          "blk.0.ssm_dt_bias", "blk.2.glm_router.weight",
                          "blk.2.glm_router.correction", "output_norm.weight" };
    for (const char* n : f32) {
        const RecipeRule* m = r.match(n);
        REQUIRE(m != nullptr);
        CHECK_EQ(m->quantizer, std::string("cast"));
        CHECK(has_opt(m, "dtype", "f32"));
    }
    const char* kept[] = { "blk.0.attn_beta.weight", "blk.5.attn_q_rope.weight",
                           "blk.5.attn_q_nope.weight", "blk.5.attn_kv_a.weight",
                           "blk.5.attn_kv_b.weight", "blk.5.attn_head_gate.weight",
                           "blk.0.ssm_conv.weight", "token_embd.weight", "output.weight" };
    for (const char* n : kept) CHECK(r.match(n) == nullptr);
}

/* The bf16 recipe with the drafter as it ships: only the drafter's weights have rules, so the trunk,
 * the embedding and the lm_head keep the checkpoint's bf16, and qwen35_bf16 serves the container. */
TEST(the_bf16_df2_recipe_quantises_the_drafter_only) {
    std::string dir = __FILE__;
    dir = dir.substr(0, dir.find_last_of('/')) + "/../data/recipes";
    std::ifstream in(dir + "/q38-27b-bf16-df2.recipe");
    REQUIRE(in.good());
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_text(text, "q38-27b-bf16-df2.recipe", &why), RAD_OK);
    CHECK_EQ(r.rules().size(), (size_t)8);

    const char* kept[] = { "blk.3.attn_qg.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight",
                           "blk.3.attn_output.weight", "blk.47.ssm_inz.weight",
                           "blk.47.ssm_ab.weight", "blk.47.ssm_out.weight",
                           "blk.47.ffn_gate_up.weight", "blk.47.ffn_down.weight", "output.weight",
                           "token_embd.weight", "output_norm.weight" };
    for (const char* n : kept)
        CHECK(r.match(n) == nullptr);
    for (const char* n : { "dflash.fc.weight", "dflash.blk.1.attn_q.weight",
                           "dflash.blk.1.attn_output.weight", "dflash.blk.1.ffn_down.weight" }) {
        const RecipeRule* m = r.match(n);
        REQUIRE(m != nullptr);
        CHECK(has_opt(m, "codes", "fp4_e2m1"));
        CHECK(has_opt(m, "scale", "e8m0"));
    }
    const RecipeRule* dh = r.match("dflash.draft_head.weight");
    REQUIRE(dh != nullptr);
    CHECK(has_opt(dh, "codes", "u2"));
    CHECK(has_opt(dh, "zero", "u8"));
}

/* The bf16x12 recipe as it ships: the nine trunk linears qwen35_bf16 declares are packed
 * losslessly (bf16x12, at its default table size), the drafter is decided by the bf16-df2
 * recipe's own rules, so --reuse can take its weights from that container, and the head, the
 * embedding, the norms, the delta net's other parameters and the vision tower keep the
 * checkpoint's bf16. */
TEST(the_bf16x12_df2_recipe_packs_the_trunk_and_quantises_the_drafter_as_bf16_df2_does) {
    std::string dir = __FILE__;
    dir = dir.substr(0, dir.find_last_of('/')) + "/../data/recipes";
    auto load = [&](const char* file, Recipe* r) {
        std::ifstream in(dir + "/" + file);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::string why;
        return in.good() && r->add_text(text, file, &why) == RAD_OK;
    };
    Recipe r, df2;
    REQUIRE(load("q38-27b-bf16x12-df2.recipe", &r));
    REQUIRE(load("q38-27b-bf16-df2.recipe", &df2));
    CHECK_EQ(r.rules().size(), (size_t)17);

    const char* trunk[] = { "blk.3.attn_qg.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight",
                            "blk.3.attn_output.weight", "blk.0.ssm_in.weight",
                            "blk.47.ssm_z.weight", "blk.62.ssm_out.weight",
                            "blk.0.ffn_gate_up.weight", "blk.63.ffn_down.weight" };
    for (const char* n : trunk) {
        const RecipeRule* m = r.match(n);
        REQUIRE(m != nullptr);
        CHECK_EQ(m->quantizer, std::string("bf16x12"));
        CHECK(m->options.empty());
    }
    /* A drafter linear is decided by its own rule, not a trunk pattern. */
    const char* drafter[] = { "dflash.draft_head.weight", "dflash.fc.weight",
                              "dflash.blk.0.attn_q.weight", "dflash.blk.4.attn_k.weight",
                              "dflash.blk.4.attn_v.weight", "dflash.blk.4.attn_output.weight",
                              "dflash.blk.0.ffn_gate_up.weight", "dflash.blk.4.ffn_down.weight" };
    for (const char* n : drafter) {
        const RecipeRule* a = r.match(n);
        const RecipeRule* b = df2.match(n);
        REQUIRE(a != nullptr && b != nullptr);
        CHECK_EQ(a->pattern, b->pattern);
        CHECK_EQ(a->quantizer, b->quantizer);
        CHECK(a->options == b->options);
    }
    const char* kept[] = { "output.weight", "token_embd.weight", "output_norm.weight",
                           "blk.3.attn_norm.weight", "blk.3.attn_q_norm.weight",
                           "blk.3.attn_k_norm.weight", "blk.47.ffn_norm.weight",
                           "blk.47.ssm_conv1d.weight", "blk.47.ssm_a_log", "blk.47.ssm_dt.bias",
                           "blk.47.ssm_norm.weight", "v.blk.0.qkv.weight", "v.blk.0.proj.weight",
                           "dflash.norm.weight", "dflash.hidden_norm.weight",
                           "dflash.blk.0.attn_norm.weight", "dflash.sel_proj.weight",
                           "dflash.blk.0.attn_conv_proj.weight" };
    for (const char* n : kept)
        CHECK(r.match(n) == nullptr);
}

RAD_TEST_MAIN()
