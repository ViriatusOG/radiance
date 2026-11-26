/* json_test.cpp -- rad::Json's observable contract.
 *
 * The DOM (util/json.h) is the tree's one JSON reader and writer: a yyjson read, ordered
 * objects, integer/float typing where non-negative integers are unsigned and an integer past
 * uint64 is a double, duplicate keys keeping the first position with the last value, and
 * shortest-round-trip float spellings. The curated expectations below are that contract
 * written down -- they were recorded against the DOM this one replaced, while the two still
 * stood side by side (type-exact fingerprints, dump bytes, boundary behaviour), and they now
 * hold the reader to it alone: change one only with the same deliberateness as changing a
 * wire format.
 */
#include "rad_test.h"

#include "util/json.h"

#include <charconv>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using rad::Json;

#define CHECK_THROWS(expr)                                                     \
    do {                                                                       \
        bool threw_ = false;                                                   \
        try {                                                                  \
            expr;                                                              \
        } catch (const std::exception&) {                                      \
            threw_ = true;                                                     \
        }                                                                      \
        CHECK(threw_);                                                         \
    } while (0)

namespace {

/* Type-exact fingerprint, bar one dump round trip's known loss: a non-negative integer
 * prints as its digits and re-reads unsigned, so signedness below 2^63 is not part of the
 * fingerprint (it is asserted directly where it matters). Float bit patterns are exact. */
void fp(const Json& j, std::string& out) {
    if (j.is_null()) out += 'n';
    else if (j.is_boolean()) out += j.get<bool>() ? 'T' : 'F';
    else if (j.is_number_unsigned() && j.get<uint64_t>() > (uint64_t)INT64_MAX) {
        out += 'u';
        out += std::to_string(j.get<uint64_t>());
    }
    else if (j.is_number_integer()) { out += 'i'; out += std::to_string(j.get<int64_t>()); }
    else if (j.is_number_float()) {
        double d = j.get<double>();
        uint64_t bits;
        memcpy(&bits, &d, 8);
        char buf[32];
        snprintf(buf, sizeof buf, "f%016llx", (unsigned long long)bits);
        out += buf;
    }
    else if (j.is_string()) { out += 's'; out += j.get<std::string>(); out += '\x01'; }
    else if (j.is_array()) { out += '['; for (const auto& e : j) fp(e, out); out += ']'; }
    else {
        out += '{';
        for (auto it = j.begin(); it != j.end(); ++it) { out += it.key(); out += '='; fp(*it, out); }
        out += '}';
    }
}

}  /* namespace */

TEST(json_parse_accepts_and_refuses_the_recorded_texts) {
    /* input -> the canonical dump of what it reads as. */
    const std::pair<const char*, const char*> accepted[] = {
        { "{\"a\":1e-999}", "{\"a\":0.0}" },
        { "{\"a\":1e-400}", "{\"a\":0.0}" },
        { "{\"a\":0e999}", "{\"a\":0.0}" },
        { "{\"a\":1.7976931348623157e308}", "{\"a\":1.7976931348623157e+308}" },
        { "{\"a\":2.2250738585072011e-308}", "{\"a\":2.225073858507201e-308}" },
        { "{\"a\":5e-324}", "{\"a\":5e-324}" },
        { "{\"a\":4e-324}", "{\"a\":5e-324}" },
        { "{\"a\":0.1}", "{\"a\":0.1}" },
        { "{\"a\":123456789012345678901234567890}", "{\"a\":1.2345678901234568e+29}" },
        { "{\"a\":9223372036854775807}", "{\"a\":9223372036854775807}" },
        { "{\"a\":9223372036854775808}", "{\"a\":9223372036854775808}" },
        { "{\"a\":-9223372036854775808}", "{\"a\":-9223372036854775808}" },
        { "{\"a\":-9223372036854775809}", "{\"a\":-9.223372036854776e+18}" },
        { "{\"a\":18446744073709551615}", "{\"a\":18446744073709551615}" },
        { "{\"a\":18446744073709551616}", "{\"a\":1.8446744073709552e+19}" },
        { "{\"a\":-0}", "{\"a\":0}" },
        { "{\"a\":-0.0}", "{\"a\":-0.0}" },
        { "{\"a\":0.0}", "{\"a\":0.0}" },
        { "{\"a\":1E7}", "{\"a\":10000000.0}" },
        { "{\"a\":1e+7}", "{\"a\":10000000.0}" },
        { "{\"a\":\"\\u0000\"}", "{\"a\":\"\\u0000\"}" },
        { "{\"a\":\"\\/\"}", "{\"a\":\"/\"}" },
        { "{\"a\":\"\"}", "{\"a\":\"\"}" },
        { "{\"a\":1,\"b\":0,\"a\":2,\"b\":3}", "{\"a\":2,\"b\":3}" },
        { "{\"\\u0061\":1}", "{\"a\":1}" },
        { "[]", "[]" }, { "{}", "{}" }, { "null", "null" }, { "true", "true" },
        { "42", "42" }, { "-17", "-17" }, { "3.25", "3.25" }, { "\"hi\"", "\"hi\"" },
        { "  {\"a\":1}  ", "{\"a\":1}" },
        { "{\"x\":[1,\"two\",3.0,null,true,{\"y\":[]},{}]}",
          "{\"x\":[1,\"two\",3.0,null,true,{\"y\":[]},{}]}" },
        { "{\"a\":\"line\\nbreak\\t\\\"quote\\\" \\\\ slash\"}",
          "{\"a\":\"line\\nbreak\\t\\\"quote\\\" \\\\ slash\"}" },
    };
    for (const auto& c : accepted) {
        const Json j = Json::parse(c.first, nullptr, false);
        CHECK(!j.is_discarded());
        CHECK_EQ(j.dump(), std::string(c.second));
        /* Every spelling of a document reads back to the same tree. */
        for (const std::string& text : { j.dump(), j.dump(2), j.dump(-1, ' ', true) }) {
            std::string a, b;
            fp(j, a);
            fp(Json::parse(text), b);
            CHECK_EQ(b, a);
        }
    }
    /* Non-ASCII is carried as UTF-8 bytes, escapes in and raw out. */
    {
        const Json j = Json::parse("{\"a\":\"caf\\u00e9 \\ud83d\\ude00\"}");
        CHECK_EQ(j["a"].get<std::string>(), std::string("caf\xc3\xa9 \xf0\x9f\x98\x80"));
        CHECK_EQ(j.dump(), std::string("{\"a\":\"caf\xc3\xa9 \xf0\x9f\x98\x80\"}"));
        CHECK_EQ(j.dump(-1, ' ', true), std::string("{\"a\":\"caf\\u00e9 \\ud83d\\ude00\"}"));
        const Json bom = Json::parse(std::string("\xef\xbb\xbf") + "{\"a\":1}", nullptr, false);
        CHECK(!bom.is_discarded());
        CHECK_EQ(bom.dump(), std::string("{\"a\":1}"));
        CHECK(!Json::parse("{\"a\":\"\xc3\xa9\"}", nullptr, false).is_discarded());
        CHECK(!Json::parse("{\"a\":\"\xe2\x82\xac\"}", nullptr, false).is_discarded());
    }
    /* Typing: parsed non-negative integers are unsigned, -0 signed, past-uint64 double. */
    CHECK(Json::parse("3").is_number_unsigned());
    CHECK(Json::parse("-0").is_number_integer() && !Json::parse("-0").is_number_unsigned());
    CHECK(Json::parse("-9223372036854775809").is_number_float());
    /* Very long number texts read (or refuse) without incident. */
    CHECK(!Json::parse("{\"a\":0." + std::string(5000, '7') + "}", nullptr, false).is_discarded());
    CHECK(Json::parse("{\"a\":" + std::string(10000, '9') + "}", nullptr, false).is_discarded());

    const char* refused[] = {
        "{\"a\":1e309}", "{\"a\":-1e309}", "{\"a\":1.7976931348623159e308}",
        "{\"a\":0x10}", "{\"a\":.5}", "{\"a\":1.}", "{\"a\":1e}", "{\"a\":01}", "{\"a\":+1}",
        "{\"a\":NaN}", "{\"a\":Infinity}",
        "{\"a\":\"\\ude00\"}", "{\"a\":\"\\ud800\"}", "{\"a\":\"\\x\"}", "{\"a\":\"\\u12g4\"}",
        "[1,2,]", "{,}", "{\"a\"}", "{\"a\":}", "[}", "nul", "{\"a\":\"unterminated}",
        "", "   ", "{\"a\":1} x", "{\"a\":1} {\"b\":2}",
        "{\"a\":\"\xff\"}", "{\"a\":\"\xc3\x28\"}", "{\"a\":\"\xed\xa0\x80\"}",
    };
    for (const char* c : refused) CHECK(Json::parse(c, nullptr, false).is_discarded());
}

TEST(json_dump_float_spellings_are_the_shortest_round_trip) {
    const std::pair<double, const char*> known[] = {
        { 1.0, "1.0" }, { -0.0, "-0.0" }, { 0.1, "0.1" }, { 1e20, "1e+20" },
        { 1e-7, "1e-07" }, { 1e16, "1e+16" }, { 1e15, "1e+15" },
        { 123456789.25, "123456789.25" },
        { 4.767695833215205e86, "4.767695833215205e+86" },
        { 2.2250738585072014e-308, "2.2250738585072014e-308" },
        { 5e-324, "5e-324" },
        { 1.7976931348623157e308, "1.7976931348623157e+308" },
        { 3.141592653589793, "3.141592653589793" },
        /* Fixed notation while the point is within (-4, 15] places of the first digit, as the
         * predecessor wrote them: these reach grammars through a schema's const and enum. */
        { 100000.0, "100000.0" }, { 3e6, "3000000.0" }, { 0.0001, "0.0001" },
        { 0.001234, "0.001234" }, { 1e-5, "1e-05" }, { 1.5e-5, "1.5e-05" },
        { 123456789012345.0, "123456789012345.0" },
        { 1234567890123456.0, "1.234567890123456e+15" },
        { -9223372036854775809.0, "-9.223372036854776e+18" },
        { 1.5e300, "1.5e+300" }, { -2.5, "-2.5" },
    };
    for (const auto& k : known) CHECK_EQ(Json(k.first).dump(), std::string(k.second));

    std::mt19937_64 rng(4242);
    int checked = 0;
    for (int i = 0; i < 100000 && checked < 60000; ++i) {
        uint64_t bits = rng();
        double d;
        memcpy(&d, &bits, 8);
        if (!std::isfinite(d)) continue;
        const double back = Json::parse(Json(d).dump()).get<double>();
        uint64_t b2;
        memcpy(&b2, &back, 8);
        CHECK_EQ(b2, bits);
        ++checked;
    }
    CHECK(checked > 50000);
    /* Non-finite values dump as null, integral floats keep their point. */
    CHECK_EQ(Json(HUGE_VAL).dump(), std::string("null"));
    CHECK_EQ(Json(NAN).dump(), std::string("null"));
}

TEST(json_random_structures_round_trip_identically) {
    std::mt19937_64 rng(7);
    std::function<Json(int)> build = [&](int depth) -> Json {
        switch (rng() % (depth > 3 ? 6 : 9)) {
        case 0: return nullptr;
        case 1: return (bool)(rng() & 1);
        case 2: return (int64_t)rng();
        case 3: return rng();
        case 4: {
            double d;
            uint64_t bits = rng();
            memcpy(&d, &bits, 8);
            return std::isfinite(d) ? Json(d) : Json(0.5);
        }
        case 5: {
            static const char* words[] = { "plain", "with \"quotes\"", "uni\xc3\xa9", "\xe2\x82\xac",
                                           "ctrl\x01\x1f", "", "back\\slash", "nl\n" };
            return words[rng() % 8];
        }
        case 6: case 7: {
            Json a = Json::array();
            const int n = (int)(rng() % 5);
            for (int i = 0; i < n; ++i) a.push_back(build(depth + 1));
            return a;
        }
        default: {
            Json o = Json::object();
            const int n = (int)(rng() % 5);
            for (int i = 0; i < n; ++i) o["k" + std::to_string(rng() % 3)] = build(depth + 1);
            return o;
        }
        }
    };
    for (int i = 0; i < 2000; ++i) {
        const Json built = build(0);
        std::string want, got;
        fp(built, want);
        fp(Json::parse(built.dump()), got);
        CHECK_EQ(got, want);
    }
}

TEST(json_dom_api_carries_the_shapes_callers_use) {
    Json j = { { "name", "edit" }, { "count", 3 }, { "ratio", 0.5 }, { "on", true },
               { "tags", Json::array({ "a", "b" }) }, { "none", nullptr } };
    CHECK(j.is_object());
    REQUIRE_EQ(j.size(), 6u);
    CHECK_EQ(j["name"].get<std::string>(), std::string("edit"));
    CHECK_EQ(j["count"].get<int>(), 3);
    CHECK(j["count"].is_number_integer() && !j["count"].is_number_unsigned());
    CHECK_NEAR(j["ratio"].get<double>(), 0.5, 1e-12);
    CHECK_EQ(j.value("missing", 42), 42);
    CHECK_EQ(j.value("name", std::string("?")), std::string("edit"));
    CHECK_EQ(j.value("missing", "fallback"), std::string("fallback"));
    CHECK(j.contains("tags"));
    CHECK(!j.contains("nope"));
    CHECK_EQ(j.at("tags").at(1).get<std::string>(), std::string("b"));
    CHECK_EQ(j.dump(),
             std::string("{\"name\":\"edit\",\"count\":3,\"ratio\":0.5,\"on\":true,"
                         "\"tags\":[\"a\",\"b\"],\"none\":null}"));

    /* Self-subtree assignment (schema $ref resolution writes n = n[sel]): the source must
     * survive the target's old value being destroyed. */
    {
        Json t = Json::parse("{\"a\":{\"b\":{\"c\":[1,2,3]}},\"d\":4}");
        t = t["a"];
        CHECK_EQ(t.dump(), std::string("{\"b\":{\"c\":[1,2,3]}}"));
        t = std::move(t["b"]);
        CHECK_EQ(t.dump(), std::string("{\"c\":[1,2,3]}"));
        Json arr = Json::parse("[[9,8],7]");
        arr = arr[0];
        CHECK_EQ(arr.dump(), std::string("[9,8]"));
    }

    /* Insertion through operator[] on null, and duplicate keys. */
    Json k;
    k["a"] = 1;
    k["b"]["c"] = "deep";
    k["a"] = 2;
    CHECK_EQ(k.dump(), std::string("{\"a\":2,\"b\":{\"c\":\"deep\"}}"));

    /* Array growth through operator[] fills gaps with null. */
    Json a;
    a[2] = "x";
    CHECK_EQ(a.dump(), std::string("[null,null,\"x\"]"));

    /* Initializer-list disambiguation: pairs make an object, anything else an array. */
    Json arr = { 1, "two", 3.0 };
    CHECK(arr.is_array());
    Json empty_list = Json::array({});
    CHECK(empty_list.is_array() && empty_list.empty());

    /* Iteration order and keys. */
    std::string keys;
    for (auto it = j.begin(); it != j.end(); ++it) keys += it.key() + ";";
    CHECK_EQ(keys, std::string("name;count;ratio;on;tags;none;"));
    std::string via_items;
    for (auto kv : j.items()) via_items += kv.key() + ";";
    CHECK_EQ(via_items, keys);
    std::string via_bindings;
    for (const auto& [key, val] : j.items()) { (void)val; via_bindings += key + ";"; }
    CHECK_EQ(via_bindings, keys);

    /* erase, update, equality across number kinds. */
    CHECK_EQ(k.erase("a"), 1u);
    CHECK_EQ(k.erase("a"), 0u);
    Json u = { { "x", { { "p", 1 } } } };
    u.update(Json::parse("{\"x\":{\"q\":2},\"y\":3}"), true);
    CHECK_EQ(u.dump(), std::string("{\"x\":{\"p\":1,\"q\":2},\"y\":3}"));
    CHECK(Json(1) == Json(1.0));
    CHECK(Json(1) == Json::parse("1"));
    /* Objects compare member by member in order: key order is part of what is written. */
    CHECK(Json::parse("{\"a\":1,\"b\":2}") == Json::parse("{\"a\":1,\"b\":2}"));
    CHECK(Json::parse("{\"a\":1,\"b\":2}") != Json::parse("{\"b\":2,\"a\":1}"));
    CHECK(Json::parse("18446744073709551615") != Json::parse("18446744073709551616"));
    CHECK(Json("s") == "s");
    CHECK(Json(nullptr) == nullptr);

    /* Booleans compare as booleans and read as 0/1 numbers. */
    CHECK(Json(true) == true);
    CHECK(Json(false) == false);
    CHECK(Json(true) != false);
    CHECK_EQ(Json(true).get<int>(), 1);
    CHECK_EQ(Json(false).get<int>(), 0);
    CHECK_NEAR(Json(true).get<double>(), 1.0, 1e-12);
    CHECK_THROWS(Json(1).get<bool>());

    /* get<> across kinds throws; numbers convert. */
    CHECK_THROWS(j["name"].get<int>());
    CHECK_THROWS(j["count"].get<std::string>());
    CHECK_THROWS(j.at("absent"));
    CHECK_EQ(Json(3.75).get<int>(), 3);
    CHECK_NEAR(Json(2).get<double>(), 2.0, 1e-12);
    const std::vector<std::string> tags = j["tags"].get<std::vector<std::string>>();
    CHECK_EQ(tags.size(), 2u);
    std::set<std::string> tagset;
    j["tags"].get_to(tagset);
    CHECK_EQ(tagset.size(), 2u);

    /* Implicit conversions. */
    std::string s = j["name"];
    int64_t n = j["count"];
    bool on = j["on"];
    std::vector<std::string> tv = j["tags"];
    CHECK_EQ(s, std::string("edit"));
    CHECK_EQ(n, (int64_t)3);
    CHECK(on);
    CHECK_EQ(tv.size(), 2u);

    /* JSON Pointer reads. */
    CHECK_EQ(j.value(Json::json_pointer("/tags/1"), std::string("?")), std::string("b"));
    CHECK_EQ(j.value(Json::json_pointer("/nope"), 7), 7);
    CHECK(j.contains(Json::json_pointer("/tags/0")));
    CHECK(!j.contains(Json::json_pointer("/tags/9")));

    /* Pretty printing: two-space nesting, "key": value, empty containers stay closed. */
    CHECK_EQ(Json::parse("{\"a\":[1,{}],\"b\":[]}").dump(2),
             std::string("{\n  \"a\": [\n    1,\n    {}\n  ],\n  \"b\": []\n}"));
}

TEST(json_replace_handler_places_replacements_like_the_dfa) {
    const std::string rep = "\xef\xbf\xbd";
    auto dumped = [](const std::string& s, Json::error_handler_t eh) {
        return Json(s).dump(-1, ' ', false, eh);
    };
    /* One replacement per ill-formed maximal subpart: byte-at-a-time where every byte is
     * ill-formed on its own, one for a merely truncated tail. */
    const std::pair<std::string, std::string> cases[] = {
        { "\xed\xa0\x80", rep + rep + rep }, { "\xc0\xaf", rep + rep },
        { "\xe0\x80\xaf", rep + rep + rep }, { "\xf4\x90\x80\x80", rep + rep + rep + rep },
        { "\xe2\x82", rep }, { "\xe2", rep }, { "\xc2", rep }, { "\xf0\x9f\x98", rep },
        { "\x82", rep }, { "\x82\x82", rep + rep },
        { "\xf0\x28\x8c\xbc", rep + "(" + rep + rep },
        { "\xe2\x82\x28", rep + "(" }, { "\xc3\x28", rep + "(" },
        { "\xff\xfe", rep + rep }, { "a\xed\xa0\x80" "b", "a" + rep + rep + rep + "b" },
        { "a\xff\xfe b\xc3\x28 c\xe2\x82 d", "a" + rep + rep + " b" + rep + "( c" + rep + " d" },
    };
    for (const auto& c : cases) {
        CHECK_THROWS(dumped(c.first, Json::error_handler_t::strict));
        CHECK_EQ(dumped(c.first, Json::error_handler_t::replace), "\"" + c.second + "\"");
    }
    /* The valid sequence dumps under strict too; ignore drops ill-formed bytes silently. */
    CHECK_EQ(dumped("\xed\x9f\xbf", Json::error_handler_t::strict), std::string("\"\xed\x9f\xbf\""));
    CHECK_EQ(dumped("a\xff" "b", Json::error_handler_t::ignore), std::string("\"ab\""));
}

TEST(json_parse_errors_and_bounds) {
    CHECK_THROWS(Json::parse("{oops"));
    CHECK(Json::parse("{oops", nullptr, false).is_discarded());
    CHECK(Json::accept("{\"a\":[1,2]}"));
    CHECK(!Json::accept("{\"a\":}"));
    CHECK(Json::accept("{/*c*/\"a\":1}", true));
    CHECK(!Json::accept("{/*c*/\"a\":1}"));

    bool over = false, deep = false;
    const Json j = Json::parse_bounded("[1,2,3]", 3, 8, &over, &deep);
    CHECK(over && !deep);  /* array + 3 scalars = 4 values */
    const Json k = Json::parse_bounded("[[[[1]]]]", 100, 3, &over, &deep);
    CHECK(!over && deep);
    const Json ok = Json::parse_bounded("[[[1]]]", 100, 3, &over, &deep);
    CHECK(!over && !deep && ok.is_array());
    CHECK(Json::parse_bounded("{bad", 100, 3, &over, &deep).is_discarded());
    /* A refused body is refused without first being collected: this swarm carries no scalar at
     * all, and a budget counted in a second pass, after every frame of the document sits in a
     * scratch vector, would let the refusal grow memory with the body. Containers are values
     * too. */
    std::string swarm = "[";
    for (int i = 0; i < 200000; ++i) swarm += "[],";
    swarm.back() = ']';
    over = deep = false;
    CHECK(Json::parse_bounded(swarm, 1000, 64, &over, &deep).is_discarded());
    CHECK(over && !deep);
}

TEST(json_parse_duplicate_keys_and_large_flat_objects) {
    /* A parsed duplicate keeps its first position and takes the last value (see the header). */
    const Json d = Json::parse("{\"a\":1,\"b\":2,\"a\":3,\"c\":4,\"b\":5}");
    CHECK_EQ(d.size(), 3u);
    CHECK_EQ(d.dump(), std::string("{\"a\":3,\"b\":5,\"c\":4}"));

    /* A tokenizer.json-shaped flat object: tens of thousands of members must read back with
     * every value at its key, in file order, a trailing duplicate included. */
    std::string text = "{";
    for (int i = 0; i < 20000; ++i) {
        if (i) text += ',';
        text += "\"tok" + std::to_string(i) + "\":" + std::to_string(i * 7 % 100000);
    }
    text += ",\"tok0\":-1}";
    const Json big = Json::parse(text);
    REQUIRE_EQ(big.size(), 20000u);
    CHECK_EQ(big["tok0"].get<int64_t>(), -1LL);
    CHECK_EQ(big["tok12345"].get<int64_t>(), (long long)(12345 * 7 % 100000));
    CHECK_EQ(big["tok19999"].get<int64_t>(), (long long)(19999 * 7 % 100000));
    auto it = big.begin();
    CHECK_EQ(it.key(), std::string("tok0"));
    ++it;
    CHECK_EQ(it.key(), std::string("tok1"));
}

RAD_TEST_MAIN()
