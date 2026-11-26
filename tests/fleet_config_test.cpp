/* fleet_config_test.cpp -- the YAML configuration (core/fleet/config.h, docs/MULTIMODEL.md §4).
 *
 *   Taken     each level's keys reach the Config an engine is built from, with the YAML path as
 *             their source in the startup report
 *   Merged    defaults sit under every model; a model's own key wins; anchors and merge keys work
 *   Refused   unknown, misplaced, duplicated and contradictory keys stop the load by name
 *   Migrated  a command line turns into a YAML file that loads to the same Config
 *   Placed    card choice, eviction order and the flex claim (core/fleet/policy.h), from plain
 *             numbers
 *
 * No device, no model, no container: text in, a FleetConfig out.
 */
#include "rad_test.h"

#include "rad_core.h"
#include "fleet/config.h"
#include "fleet/policy.h"

#include <map>
#include <string>
#include <vector>

using namespace rad;

namespace {

int yload(const std::string& text, fleet::FleetConfig* out) { return fleet::parse(text, "fleet.yaml", out); }

bool refused(const std::string& text) {
    fleet::FleetConfig fc;
    return yload(text, &fc) != RAD_OK;
}

/* The YAML file a command line becomes. */
std::string yaml_of(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (std::string& a : args) argv.push_back(a.data());
    return fleet::yaml_from_args((int)argv.size(), argv.data());
}

}  // namespace

/* ------------------------------------------------------------------ taken */

TEST(one_model_is_an_engine_config) {
    fleet::FleetConfig fc;
    CHECK_EQ(yload("models:\n  qwen:\n    model: /m/q.rad\n    tp: 2\n    max-num-seqs: 8\n", &fc), RAD_OK);
    REQUIRE_EQ(fc.models.size(), 1u);
    const fleet::ModelEntry& m = fc.models[0];
    CHECK_EQ(m.name, "qwen");
    CHECK_EQ(m.cfg.model, "/m/q.rad");
    CHECK_EQ(m.cfg.tp, 2);
    CHECK_EQ(m.cfg.max_seqs, 8);
    CHECK_EQ(m.cfg.served_model_name, "qwen");
    CHECK_EQ(m.cfg.given.at("--tp"), "models.qwen.tp");
    CHECK_EQ(m.cfg.given.at("--served-model-name"), "models.qwen");
}

TEST(server_and_memory_keys_reach_every_model) {
    fleet::FleetConfig fc;
    CHECK_EQ(yload("server:\n  host: \"::\"\n  port: 8123\n  log: debug\n  allow-host: [a.example, b.example]\n"
                  "memory:\n  gpu-headroom-mib: 160\n  host-mib: 4096\n  disk-tier: /var/cache/r\n"
                  "models:\n  a:\n    model: /m/a.rad\n  b:\n    model: /m/b.rad\n", &fc), RAD_OK);
    CHECK_EQ(fc.models.size(), 2u);
    for (const fleet::ModelEntry& m : fc.models) {
        CHECK_EQ(m.cfg.host, "::");
        CHECK_EQ(m.cfg.port, 8123);
        CHECK_EQ(m.cfg.allow_hosts.size(), 2u);
        CHECK_EQ(m.cfg.gpu_headroom_mib, 160);
        CHECK_EQ(m.cfg.given.at("--port"), "server.port");
        CHECK_EQ(m.cfg.given.at("--gpu-headroom-mib"), "memory.gpu-headroom-mib");
    }
    CHECK(fc.log == Log::Debug);
    CHECK_EQ(fc.host_mib, 4096);
    CHECK_EQ(fc.disk_tier, "/var/cache/r");
    CHECK(fc.card_headroom_mib.empty());
    CHECK_EQ(fc.queue_reserve, 0);   /* unset: fleet.cpp's own default stands */
}

/* memory.queue-reserve: the hardware queues a card keeps ready, for a fleet whose entries claim
 * whole cards (docs/MULTIMODEL.md §7.1). Its bound is the device's ceiling on queues a card holds
 * ahead of its streams, so 24 is the last accepted number and 25 is refused by name. */
TEST(queue_reserve_key) {
    fleet::FleetConfig fc;
    CHECK_EQ(yload("memory:\n  queue-reserve: 2\nmodels:\n  a:\n    model: /m/a.rad\n", &fc), RAD_OK);
    CHECK_EQ(fc.queue_reserve, 2);
    CHECK(refused("memory:\n  queue-reserve: 0\nmodels:\n  a:\n    model: /m/a.rad\n"));
    CHECK(refused("memory:\n  queue-reserve: 25\nmodels:\n  a:\n    model: /m/a.rad\n"));
    CHECK(refused("memory:\n  queue-reserve: eight\nmodels:\n  a:\n    model: /m/a.rad\n"));
}

/* memory.card-headroom-mib: one card's reserve by ordinal, the others keep gpu-headroom-mib. */
TEST(a_card_keeps_its_own_headroom) {
    fleet::FleetConfig fc;
    CHECK_EQ(yload("memory:\n  gpu-headroom-mib: 160\n  card-headroom-mib: {0: 4096, 2: 0}\n"
                   "models:\n  a: {model: /m/a.rad}\n", &fc), RAD_OK);
    CHECK_EQ(fc.card_headroom_mib.size(), 2u);
    CHECK_EQ(fc.card_headroom_mib.at(0), 4096);
    CHECK_EQ(fc.card_headroom_mib.at(2), 0);
    CHECK_EQ(fc.models.front().cfg.gpu_headroom_mib, 160);
    CHECK(refused("memory:\n  card-headroom-mib: 4096\nmodels:\n  a: {model: /m/a.rad}\n"));
    CHECK(refused("memory:\n  card-headroom-mib: {gpu0: 4096}\nmodels:\n  a: {model: /m/a.rad}\n"));
    CHECK(refused("memory:\n  card-headroom-mib: {0: -1}\nmodels:\n  a: {model: /m/a.rad}\n"));
    CHECK(refused("models:\n  a: {model: /m/a.rad, card-headroom-mib: {0: 1}}\n"));
}

/* memory.pinned-numa-node: the node pinned host memory prefers, -1 (unset) for each card's own. */
TEST(pinned_host_memory_can_name_its_node) {
    fleet::FleetConfig fc;
    CHECK_EQ(yload("memory:\n  host-mib: 4096\n  pinned-numa-node: 1\n"
                   "models:\n  a: {model: /m/a.rad}\n", &fc), RAD_OK);
    CHECK_EQ(fc.pinned_numa_node, 1);
    CHECK_EQ(yload("models:\n  a: {model: /m/a.rad}\n", &fc), RAD_OK);
    CHECK_EQ(fc.pinned_numa_node, -1);
    CHECK(refused("memory:\n  pinned-numa-node: -1\nmodels:\n  a: {model: /m/a.rad}\n"));
    CHECK(refused("memory:\n  pinned-numa-node: 1024\nmodels:\n  a: {model: /m/a.rad}\n"));
    CHECK(refused("memory:\n  pinned-numa-node: node0\nmodels:\n  a: {model: /m/a.rad}\n"));
    /* The fleet's own key: a model entry that names it is the misplaced message, not a misspelling. */
    CHECK(refused("models:\n  a: {model: /m/a.rad, pinned-numa-node: 0}\n"));
}

TEST(structured_values_are_spelled_for_their_flag) {
    fleet::FleetConfig fc;
    CHECK_EQ(yload("models:\n  q:\n    model: /m/q.rad\n"
                  "    weights-disk-tier: true\n"
                  "    deterministic: false\n"
                  "    chat-template-kwargs: {enable_thinking: false, style: \"terse\"}\n"
                  "    lora: {sql: /a/sql, chat: /a/chat}\n"
                  "    reasoning-effort-map: {minimal: low}\n"
                  "    temp: 0.7\n", &fc), RAD_OK);
    const Config& c = fc.models[0].cfg;
    CHECK(c.weights_disk_tier);
    CHECK(!c.deterministic);
    CHECK_EQ(c.chat_template_kwargs.at("enable_thinking"), "false");
    CHECK_EQ(c.chat_template_kwargs.at("style"), "\"terse\"");
    REQUIRE_EQ(c.lora.adapters.size(), 2u);
    CHECK_EQ(c.lora.adapters[0].first, "sql");
    CHECK_EQ(c.reasoning_effort_map.at("minimal"), "low");
    CHECK(c.sample_temp > 0.69f && c.sample_temp < 0.71f);
}

/* The diagnose mode is a model key like the KL mode's: the entry's Config is what main() tests to
 * run it offline, and a sequence spells the sweeps' lists. */
TEST(the_diagnose_keys_are_model_keys) {
    fleet::FleetConfig fc;
    CHECK_EQ(yload("models:\n  qwen:\n    model: /m/q.rad\n    diagnose: /out\n    diagnose-runs: 5\n"
                  "    diagnose-concurrency: [1, 2]\n    diagnose-prefill: [1024, 4096]\n", &fc), RAD_OK);
    REQUIRE_EQ(fc.models.size(), 1u);
    const Config& c = fc.models[0].cfg;
    CHECK(c.diagnose_mode());
    CHECK_EQ(c.diagnose, "/out");
    CHECK_EQ(c.diagnose_runs, 5);
    CHECK_EQ(c.diagnose_concurrency.size(), 2u);
    CHECK_EQ(c.diagnose_prefill.size(), 2u);
    CHECK_EQ(c.given.at("--diagnose"), "models.qwen.diagnose");
    CHECK(refused("models:\n  qwen:\n    model: /m/q.rad\n    diagnose-runs: 5\n    diagnose: /out\n"
                  "    kld-ref: /r\n"));
    CHECK(refused("models:\n  qwen:\n    model: /m/q.rad\n    diagnose-baseline: /b.json\n"));
}

TEST(model_ids_may_hold_dots) {
    fleet::FleetConfig fc;
    CHECK_EQ(yload("models:\n  qwen3.8-27b:\n    model: /m/q.rad\n    tp: 2\n", &fc), RAD_OK);
    CHECK_EQ(fc.models[0].name, "qwen3.8-27b");
    CHECK_EQ(fc.models[0].cfg.tp, 2);
    CHECK_EQ(fc.models[0].cfg.given.at("--tp"), "models.qwen3.8-27b.tp");
}

/* A model stating its part of the card replaces the memory level's card pair; the others keep it. */
TEST(a_model_divides_its_card_with_gpu_memory_mib) {
    fleet::FleetConfig fc;
    CHECK_EQ(yload("memory: {gpu-headroom-mib: 160}\n"
                   "models:\n  a: {model: /m/a.rad, gpu-memory-mib: 12000}\n  b: {model: /m/b.rad}\n", &fc),
             RAD_OK);
    CHECK_EQ(fc.models[0].cfg.gpu_memory_mib, 12000);
    CHECK(!fc.models[0].cfg.given.count("--gpu-headroom-mib"));
    CHECK_EQ(fc.models[0].cfg.given.at("--gpu-memory-mib"), "models.a.gpu-memory-mib");
    CHECK_EQ(fc.models[1].cfg.gpu_headroom_mib, 160);
    CHECK(!fc.models[1].cfg.given.count("--gpu-memory-mib"));
}

/* Card choice and eviction (fleet/policy.h): a model goes where it displaces least, models leave
 * lowest priority first and idle before busy, and pinned, kept and higher-priority models stay. */
TEST(a_model_loads_where_it_displaces_least) {
    using fleet::CardRoom;
    using fleet::Resident;
    const int64_t G = (int64_t)1 << 30;
    fleet::RoomPlan p;
    std::string why;
    /* Room on card 1 already: nothing leaves. */
    std::vector<CardRoom> cards = { { 0, 2 * G }, { 1, 20 * G } };
    std::vector<Resident> res = { { 0, { 0 }, 28 * G, 0, false, false, false, 10, 5 } };
    CHECK(fleet::plan_room(cards, res, 1, 10 * G, 0, {}, {}, nullptr, &p, &why));
    CHECK(p.cards == std::vector<int>({ 1 }) && p.victims.empty());
    /* Two cards needed: the resident on card 0 leaves. */
    CHECK(fleet::plan_room(cards, res, 2, 10 * G, 0, {}, {}, nullptr, &p, &why));
    CHECK(p.cards == std::vector<int>({ 0, 1 }) && p.victims == std::vector<size_t>({ 0 }));
    /* It may not leave for a lower-priority model, nor when pinned. */
    CHECK(!fleet::plan_room(cards, res, 2, 10 * G, -1, {}, {}, nullptr, &p, &why));
    res[0].pin = true;
    CHECK(!fleet::plan_room(cards, res, 2, 10 * G, 5, {}, {}, nullptr, &p, &why));
    res[0].pin = false;
    /* The idle one leaves before the busy one, then the cheaper one to bring back. */
    cards = { { 0, 0 } };
    res = { { 0, { 0 }, 10 * G, 0, false, false, true, 100, 5 },
            { 1, { 0 }, 10 * G, 0, false, false, false, 10, 5 },
            { 2, { 0 }, 10 * G, 0, false, false, false, 10, 1 } };
    CHECK(fleet::plan_room(cards, res, 1, 10 * G, 0, {}, {}, nullptr, &p, &why));
    CHECK(p.victims == std::vector<size_t>({ 2 }));
    CHECK(fleet::plan_room(cards, res, 1, 25 * G, 0, {}, {}, nullptr, &p, &why));
    CHECK(p.victims == std::vector<size_t>({ 2, 1, 0 }));
    /* A parked model leaves last, after every resident one. */
    {
        std::vector<Resident> pr = { { 0, { 0 }, 1 * G, 0, false, false, false, 1000, 5 },
                                     { 1, { 0 }, 10 * G, 0, false, false, true, 1, 5 } };
        pr[0].parked = true;
        CHECK(fleet::plan_room(cards, pr, 1, 10 * G, 0, {}, {}, nullptr, &p, &why));
        CHECK(p.victims == std::vector<size_t>({ 1 }));
    }
    /* Lower priority leaves first whatever its idle time. */
    res[0].priority = -5;
    CHECK(fleet::plan_room(cards, res, 1, 10 * G, 0, {}, {}, nullptr, &p, &why));
    CHECK(p.victims == std::vector<size_t>({ 0 }));
    /* A forced member (exclusive group) leaves even when there is room. */
    cards = { { 0, 30 * G } };
    CHECK(fleet::plan_room(cards, res, 1, 10 * G, 0, { 1 }, {}, nullptr, &p, &why));
    CHECK(p.victims == std::vector<size_t>({ 1 }));
    /* With room everywhere, the card no model is on. */
    cards = { { 0, 30 * G }, { 1, 30 * G } };
    res = { { 0, { 0 }, 2 * G, 0, false, false, false, 10, 5 } };
    CHECK(fleet::plan_room(cards, res, 1, 10 * G, 0, {}, {}, nullptr, &p, &why));
    CHECK(p.cards == std::vector<int>({ 1 }) && p.victims.empty());
    /* Among equal sets, the cards it was on before. */
    cards = { { 0, 30 * G }, { 1, 30 * G } };
    CHECK(fleet::plan_room(cards, {}, 1, 10 * G, 0, {}, { 1 }, nullptr, &p, &why));
    CHECK(p.cards == std::vector<int>({ 1 }));
    /* A unit (a together group) gives back each member's bytes on its own card, and its members
     * leave together. */
    cards = { { 0, 0 }, { 1, 0 } };
    res = { { 7, { 0, 1 }, 0, 0, false, false, false, 10, 5, false, { 6 * G, 2 * G }, { 7, 8 } } };
    CHECK(fleet::plan_room(cards, res, 1, 5 * G, 0, {}, {}, nullptr, &p, &why));
    CHECK(p.cards == std::vector<int>({ 0 }) && p.victims == std::vector<size_t>({ 7, 8 }));
    CHECK(!fleet::plan_room(cards, res, 1, 7 * G, 0, {}, {}, nullptr, &p, &why));
    /* A unit is forced out when any member is, not only the one that names it. */
    cards = { { 0, 30 * G }, { 1, 30 * G } };
    CHECK(fleet::plan_room(cards, res, 1, G, 0, { 8 }, {}, nullptr, &p, &why));
    CHECK(p.victims == std::vector<size_t>({ 7, 8 }));
    cards = { { 0, 0 }, { 1, 0 } };
    /* Too few cards. */
    CHECK(!fleet::plan_room(cards, {}, 3, G, 0, {}, {}, nullptr, &p, &why));
    /* A returning model of two ranks, all out on card 1 and mostly still on card 0: card 0 needs
     * only its own part, so the model on card 1 leaves and the one on card 0 stays. */
    {
        cards = { { 0, 4 * G }, { 1, 2 * G } };
        res = { { 0, { 0 }, 8 * G, 0, false, false, false, 10, 5 },
                { 1, { 1 }, 20 * G, 0, false, false, false, 10, 5 } };
        const std::vector<int> both = { 0, 1 };
        const std::map<int, int64_t> need = { { 0, 3 * G }, { 1, 20 * G } };
        CHECK(!fleet::plan_room(cards, res, 2, 20 * G, 0, {}, {}, &both, &p, &why));
        CHECK(fleet::plan_room(cards, res, 2, 20 * G, 0, {}, {}, &both, &p, &why, &need));
        CHECK(p.victims == std::vector<size_t>({ 1 }));
    }
}

/* The claim: need, raised by the flex floors, never above want. */
TEST(a_claim_is_the_need_raised_by_the_flex_floor) {
    const int64_t M = (int64_t)1 << 20;
    /* kv: 100 MiB fixed state, 900 MiB for 9000 tokens; one context (1000 tokens) = 200 MiB. */
    CHECK_EQ(fleet::flex_claim(1000 * M, 5000 * M, 100 * M, 1000 * M, 200 * M, 9000, 0, 0, -1, -1),
             1000 * M);
    CHECK_EQ(fleet::flex_claim(1000 * M, 5000 * M, 100 * M, 1000 * M, 200 * M, 9000, 0, 0, 4500, -1),
             1000 * M + 350 * M);
    CHECK_EQ(fleet::flex_claim(1000 * M, 1100 * M, 100 * M, 1000 * M, 200 * M, 9000, 0, 0, 9000, -1),
             1100 * M);
    CHECK_EQ(fleet::flex_claim(1000 * M, 9000 * M, 0, 0, 0, 0, 4000 * M, 1000 * M, -1, 3000),
             3000 * M);
}

TEST(fleet_keys_on_a_model) {
    fleet::FleetConfig fc;
    CHECK_EQ(yload("models:\n  q:\n    model: /m/q.rad\n    aliases: [default, chat]\n    priority: 100\n"
                  "    pin: true\n    ttl: 30m\n"
                  "    flex: {kv: {min-tokens: 200000, weight: 2}, experts: {max-mib: 4096}}\n"
                  "    park: {vram: statics, ram: required, restore: live}\n", &fc), RAD_OK);
    const fleet::ModelEntry& m = fc.models[0];
    CHECK_EQ(m.aliases.size(), 2u);
    CHECK_EQ(m.priority, 100);
    CHECK(m.pin);
    CHECK_EQ(m.ttl_ms, 30 * 60 * 1000);
    CHECK_EQ(m.flex.kv_min_tokens, 200000);
    CHECK(m.flex.kv_weight == 2.0);
    CHECK_EQ(m.flex.experts_max_mib, 4096);
    {
        /* 0 keeps a tiered model's experts at its claim: no growth into free VRAM. */
        fleet::FleetConfig z;
        CHECK_EQ(yload("models:\n  q:\n    model: /m/q.rad\n    flex: {experts: {max-mib: 0}}\n", &z), RAD_OK);
        CHECK_EQ(z.models[0].flex.experts_max_mib, 0);
    }
    CHECK(m.park.vram == fleet::ParkSpec::Vram::Statics);
    CHECK(m.park.ram == fleet::ParkSpec::Ram::Required);
    CHECK(m.park.restore == fleet::ParkSpec::Restore::Live);
    CHECK(fc.find("chat") == &m);
    CHECK(fc.find("default") == &m);
    CHECK(fc.find("nope") == nullptr);

    CHECK_EQ(yload("models:\n  q:\n    model: /m/q.rad\n    park: {vram: 2048}\n", &fc), RAD_OK);
    CHECK(fc.models[0].park.vram == fleet::ParkSpec::Vram::Mib);
    CHECK_EQ(fc.models[0].park.vram_mib, 2048);
    CHECK(fc.models[0].park.restore == fleet::ParkSpec::Restore::Auto);
    CHECK_EQ(yload("models:\n  q:\n    model: /m/q.rad\n    park: {restore: full}\n", &fc), RAD_OK);
    CHECK(fc.models[0].park.restore == fleet::ParkSpec::Restore::Full);
    CHECK(refused("models:\n  q:\n    model: /m/q.rad\n    park: {restore: whole}\n"));
}

/* A model's own cards: a list among the fleet's, long enough for its ranks. They are not part of
 * the engine's flag list, so two entries that differ only in them configure the same engine. */
TEST(a_model_names_the_cards_it_may_use) {
    fleet::FleetConfig fc;
    CHECK_EQ(yload("models:\n  a: {model: /m/a.rad, cards: [1]}\n  b: {model: /m/b.rad, tp: 2, cards: [1, 0]}\n"
                  "  c: {model: /m/c.rad}\n  d: {model: /m/d.rad, cards: all}\n", &fc), RAD_OK);
    REQUIRE_EQ(fc.models[0].cards.size(), 1u);
    CHECK_EQ(fc.models[0].cards[0], 1);
    REQUIRE_EQ(fc.models[1].cards.size(), 2u);
    CHECK_EQ(fc.models[1].cards[0], 1);
    CHECK(fc.models[2].cards.empty());
    CHECK(fc.models[3].cards.empty());
    fleet::FleetConfig plain;
    CHECK_EQ(yload("models:\n  a: {model: /m/a.rad}\n", &plain), RAD_OK);
    CHECK(plain.models[0].args == fc.models[0].args);

    CHECK_EQ(yload("cards: [0, 2]\nmodels:\n  a: {model: /m/a.rad, cards: [2]}\n", &fc), RAD_OK);
    /* outside the fleet's cards, fewer than its ranks, twice, not a list, under defaults */
    CHECK(refused("cards: [0, 2]\nmodels:\n  a: {model: /m/a.rad, cards: [1]}\n"));
    CHECK(refused("models:\n  a: {model: /m/a.rad, tp: 2, cards: [1]}\n"));
    CHECK(refused("models:\n  a: {model: /m/a.rad, cards: [1, 1]}\n"));
    CHECK(refused("models:\n  a: {model: /m/a.rad, cards: 1}\n"));
    CHECK(refused("models:\n  a: {model: /m/a.rad, cards: []}\n"));
    CHECK(refused("defaults: {cards: [0]}\nmodels:\n  a: {model: /m/a.rad}\n"));
}

TEST(scheduler_cards_and_groups) {
    fleet::FleetConfig fc;
    CHECK_EQ(yload("cards: [0, 1]\nscheduler: {load-timeout: 90s, drain-timeout: 500ms, keepalive: 5s}\n"
                  "models:\n  a: {model: /m/a.rad}\n  b: {model: /m/b.rad}\n  c: {model: /m/c.rad}\n"
                  "groups:\n  rag: {together: [a, b]}\n  big: {exclusive: [a, c]}\n", &fc), RAD_OK);
    CHECK_EQ(fc.cards.size(), 2u);
    CHECK_EQ(fc.load_timeout_ms, 90000);
    CHECK_EQ(fc.drain_timeout_ms, 500);
    CHECK_EQ(fc.keepalive_ms, 5000);
    CHECK_EQ(fc.stuck_exit_ms, 60000);   /* a minute unless the file says otherwise */
    CHECK(fc.warm_up);   /* on unless the file turns it off */
    REQUIRE_EQ(fc.groups.size(), 2u);
    CHECK(fc.groups[1].kind == fleet::Group::Kind::Exclusive);
    fleet::FleetConfig off;
    CHECK_EQ(yload("scheduler: {stuck-exit: 0}\nmodels:\n  a: {model: /m/a.rad}\n", &off), RAD_OK);
    CHECK_EQ(off.stuck_exit_ms, 0);
    fleet::FleetConfig slow;
    CHECK_EQ(yload("scheduler: {stuck-exit: 5m}\nmodels:\n  a: {model: /m/a.rad}\n", &slow), RAD_OK);
    CHECK_EQ(slow.stuck_exit_ms, 300000);
    CHECK_EQ(yload("cards: all\nmodels:\n  a: {model: /m/a.rad}\n", &fc), RAD_OK);
    CHECK(fc.cards.empty());
    CHECK_EQ(yload("scheduler: {warm-up: false}\nmodels:\n  a: {model: /m/a.rad}\n", &fc), RAD_OK);
    CHECK(!fc.warm_up);
    CHECK(refused("scheduler: {warm-up: soon}\nmodels:\n  a: {model: /m/a.rad}\n"));
}

/* ------------------------------------------------------------------ merged */

TEST(defaults_sit_under_models) {
    fleet::FleetConfig fc;
    CHECK_EQ(yload("defaults:\n  kv-cache-dtype: i8\n  max-model-len: 65536\n  priority: 5\n"
                  "models:\n  a:\n    model: /m/a.rad\n  b:\n    model: /m/b.rad\n    max-model-len: 4096\n", &fc), RAD_OK);
    CHECK_EQ(fc.models[0].cfg.kv_cache_dtype, "i8");
    CHECK_EQ(fc.models[0].cfg.max_ctx, 65536);
    CHECK_EQ(fc.models[0].cfg.given.at("--max-model-len"), "defaults.max-model-len");
    CHECK_EQ(fc.models[1].cfg.max_ctx, 4096);
    CHECK_EQ(fc.models[1].cfg.given.at("--max-model-len"), "models.b.max-model-len");
    CHECK_EQ(fc.models[0].priority, 5);
    CHECK_EQ(fc.models[1].priority, 5);
}

/* The variant's name is listed per model, as MODEL:NAME, so one name under defaults: gives every
 * model a variant of its own without a collision: the switch for a whole fleet at once. */
TEST(a_control_vector_variant_under_defaults_reaches_every_model) {
    fleet::FleetConfig fc;
    CHECK_EQ(yload("defaults:\n  control-vector: off\n  control-vector-variant: uncensored\n"
                   "models:\n  a:\n    model: /m/a.rad\n  b:\n    model: /m/b.rad\n", &fc), RAD_OK);
    CHECK_EQ(fc.models[0].cfg.control_vector_variant, "uncensored");
    CHECK_EQ(fc.models[1].cfg.control_vector_variant, "uncensored");
    CHECK_EQ(fc.models[1].cfg.given.at("--control-vector-variant"),
             "defaults.control-vector-variant");
}

TEST(anchors_and_merge_keys) {
    fleet::FleetConfig fc;
    /* Top-level keys starting with x- are ignored, so they can hold anchors. */
    CHECK_EQ(yload("x-common: &common\n  tp: 2\n  max-num-seqs: 4\n"
                  "models:\n  a:\n    <<: *common\n    model: /m/a.rad\n", &fc), RAD_OK);
    CHECK_EQ(fc.models[0].cfg.tp, 2);
    CHECK_EQ(yload("defaults: &d\n  max-num-seqs: 4\n"
                  "models:\n  a:\n    <<: *d\n    model: /m/a.rad\n    tp: 2\n", &fc), RAD_OK);
    CHECK_EQ(fc.models[0].cfg.max_seqs, 4);
    CHECK_EQ(fc.models[0].cfg.tp, 2);
}

TEST(args_order_is_stable_for_reload) {
    fleet::FleetConfig a, b;
    CHECK_EQ(yload("models:\n  q:\n    model: /m/q.rad\n    tp: 2\n    max-num-seqs: 8\n", &a), RAD_OK);
    CHECK_EQ(yload("models:\n  q:\n    max-num-seqs: 8\n    tp: 2\n    model: /m/q.rad\n", &b), RAD_OK);
    CHECK(a.models[0].args == b.models[0].args);
}

/* ------------------------------------------------------------------ refused */

TEST(refusals) {
    CHECK(refused(""));
    CHECK(refused("server: {port: 8000}\n"));                                     /* no models */
    CHECK(refused("models: {}\n"));
    CHECK(refused("models:\n  a: {model: /m/a.rad, max-num-seq: 8}\n"));           /* unknown key */
    CHECK(refused("models:\n  a: {model: /m/a.rad, port: 9000}\n"));               /* server key on a model */
    CHECK(refused("models:\n  a: {model: /m/a.rad, served-model-name: x}\n"));
    CHECK(refused("server: {tp: 2}\nmodels:\n  a: {model: /m/a.rad}\n"));          /* model key on server */
    CHECK(refused("tp: 2\nmodels:\n  a: {model: /m/a.rad}\n"));                    /* model key at the top */
    CHECK(refused("models:\n  a:\n    model: /m/a.rad\n    tp: 2\n    tp: 4\n"));  /* given twice */
    CHECK(refused("models:\n  a: {model: /m/a.rad, tp: 0}\n"));                    /* config_parse refuses */
    CHECK(refused("models:\n  a: {tp: 2}\n"));                                     /* no model source */
    CHECK(refused("models:\n  \"a:b\": {model: /m/a.rad}\n"));                    /* ':' is base:adapter */
    CHECK(refused("models:\n  a: {model: /m/a.rad}\n  b: {model: /m/b.rad, aliases: [a]}\n"));
    CHECK(refused("models:\n  a: {model: /m/a.rad, weights-disk-tier: yes}\n"));   /* true or false */
    CHECK(refused("models:\n  a: {model: /m/a.rad, tp: [1, 2]}\n"));
    CHECK(refused("models:\n  a: {model: /m/a.rad, ttl: soon}\n"));
    CHECK(refused("models:\n  a: {model: /m/a.rad, park: {ram: maybe}}\n"));
    CHECK(refused("models:\n  a: {model: /m/a.rad, park: {restore: soon}}\n"));
    CHECK(refused("models:\n  a: {model: /m/a.rad, flex: {kv: {min-tokens: 9, max-tokens: 3}}}\n"));
    CHECK(refused("defaults: {aliases: [x]}\nmodels:\n  a: {model: /m/a.rad}\n"));
    CHECK(refused("server: {log: loud}\nmodels:\n  a: {model: /m/a.rad}\n"));
    CHECK(refused("cards: [0, 0]\nmodels:\n  a: {model: /m/a.rad}\n"));
    CHECK(refused("models:\n  a: {model: /m/a.rad}\ngroups:\n  g: {together: [a, nope]}\n"));
    CHECK(refused("models:\n  a: {model: /m/a.rad}\n  b: {model: /m/b.rad}\n"
                  "groups:\n  g: {together: [a, b]}\n  h: {exclusive: [b, a]}\n"));
    CHECK(refused("models:\n  a: {model: /m/a.rad}\ngroups:\n  g: {together: [a]}\n"));
    CHECK(refused("models:\n  a: [unclosed\n"));                                   /* YAML syntax */
    CHECK(refused("---\nmodels: {a: {model: /m/a.rad}}\n---\nmodels: {b: {model: /m/b.rad}}\n"));
}

/* ------------------------------------------------------------------ migrated */

TEST(command_line_becomes_yaml) {
    std::vector<std::string> args{ "radiance", "--model", "/models/q38-27b.rad", "--host", "::", "--port", "8000",
                                   "--tp", "2", "--kv-cache-dtype", "i8", "--gpu-headroom-mib", "160",
                                   "--weights-disk-tier", "--allow-host", "a.example", "--allow-host", "b.example",
                                   "--chat-template-kwargs", "{\"enable_thinking\": false}", "-v" };
    const std::string y = yaml_of(args);
    fleet::FleetConfig fc;
    CHECK_EQ(yload(y, &fc), RAD_OK);
    REQUIRE_EQ(fc.models.size(), 1u);
    const fleet::ModelEntry& m = fc.models[0];
    CHECK_EQ(m.name, "q38-27b");
    CHECK_EQ(m.cfg.model, "/models/q38-27b.rad");
    CHECK_EQ(m.cfg.host, "::");
    CHECK_EQ(m.cfg.tp, 2);
    CHECK_EQ(m.cfg.kv_cache_dtype, "i8");
    CHECK_EQ(m.cfg.gpu_headroom_mib, 160);
    CHECK(m.cfg.weights_disk_tier);
    CHECK_EQ(m.cfg.allow_hosts.size(), 2u);
    CHECK_EQ(m.cfg.chat_template_kwargs.at("enable_thinking"), "false");
    CHECK(fc.log == Log::Debug);
    CHECK(fc.log_set);

    /* A cache this release refuses is written as the command line gave it, with what replaces it
     * as a comment beside it; the file still loads as YAML, and the refusal comes from the key. */
    const std::string packed = yaml_of({ "radiance", "--model", "/m/a.rad", "--kv-cache-dtype", "k6_v4" });
    CHECK(packed.find("kv-cache-dtype: k6_v4   # not served: i8 is the 8-bit cache") != std::string::npos);
    const std::string e4m3 = yaml_of({ "radiance", "--model", "/m/a.rad", "--kv-cache-dtype", "fp8" });
    CHECK(e4m3.find("kv-cache-dtype: fp8   # served for few models") != std::string::npos);
    fleet::FleetConfig fe;
    CHECK_EQ(yload(e4m3, &fe), RAD_OK);
    CHECK_EQ(fe.models[0].cfg.kv_cache_dtype, "fp8");
    CHECK(y.find('#') == std::string::npos);
}

TEST(entry_name_keeps_a_checkpoint_name_whole) {
    fleet::FleetConfig fc;
    CHECK_EQ(yload(yaml_of({ "radiance", "--model", "/hf/Qwen3.8-27B-FP8", "--tp", "1" }), &fc), RAD_OK);
    CHECK_EQ(fc.models[0].name, "Qwen3.8-27B-FP8");
    CHECK(!fc.log_set);
}

TEST(durations) {
    int64_t ms = 0;
    CHECK(fleet::parse_duration_ms("250ms", &ms) && ms == 250);
    CHECK(fleet::parse_duration_ms("30", &ms) && ms == 30000);
    CHECK(fleet::parse_duration_ms("2h", &ms) && ms == 7200000);
    CHECK(!fleet::parse_duration_ms("", &ms));
    CHECK(!fleet::parse_duration_ms("1.5s", &ms));
    CHECK(!fleet::parse_duration_ms("5d", &ms));
}

/* Every key the loader places at the server and memory levels is an option in the table, so a
 * renamed flag cannot leave a level key that nothing accepts. */
TEST(level_keys_are_options) {
    size_t n = 0;
    const ConfigFlag* f = config_flags(&n);
    for (const char* level : { "server", "memory" }) {
        CHECK(!fleet::level_keys(level).empty());
        for (const std::string& k : fleet::level_keys(level)) {
            bool found = false;
            for (size_t i = 0; i < n; ++i) found |= ("--" + k) == f[i].name;
            if (!found) radtest::fail(__FILE__, __LINE__, std::string(level) + "." + k + " is not an option");
        }
    }
}

RAD_TEST_MAIN()
