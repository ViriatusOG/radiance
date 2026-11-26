/* r4d_ar_links_test.cpp -- the eight-rank two-shot's shard weights from the PCIe links
 * (libr4d/r4d_ar_links.h).
 *
 * Each case builds sysfs under a root directory, laid out like the eight-R9700 box the weights
 * were measured on: each card behind its on-card switch (upstream and downstream port at Gen5
 * x16, endpoint at Gen5 x16) under a root port whose link is the real slot. The slot shows only
 * at the root port, so a reader that stops at the card's own link sees eight equal cards.
 *
 *   Bottleneck  the slowest hop of a card's path, root port included
 *   Weights     round(4 * (rate / max)^2): two Gen4 x8 links give 4,1,0,4,0,1,4,4, the
 *               repaired ones 4 and 1; a card whose path says nothing leaves the weights unknown
 */
#include "rad_test.h"

#include "r4d_ar_links.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

struct Root {
    std::string dir;
    Root() : dir(radtest::make_tmp_dir("root")) {}
    void put(const std::string& rel, const std::string& text) const {
        const fs::path p = fs::path(dir) / rel.substr(1);
        fs::create_directories(p.parent_path());
        std::ofstream(p) << text;
    }
    void link(const std::string& rel, const std::string& target) const {
        const fs::path p = fs::path(dir) / rel.substr(1);
        fs::create_directories(p.parent_path());
        fs::create_symlink(target, p);
    }
};

/* A device under /sys/devices at `path` with its /sys/bus/pci/devices link; `speed` "" writes no
 * link files, as some bridges inside a switch do. */
void pci(const Root& r, const std::string& path, const std::string& speed, int width) {
    const std::string bdf = path.substr(path.rfind('/') + 1);
    const std::string d = "/sys/devices/" + path;
    r.put(d + "/class", "0x060400\n");
    if (!speed.empty()) {
        r.put(d + "/current_link_speed", speed + "\n");
        r.put(d + "/current_link_width", std::to_string(width) + "\n");
    }
    r.link("/sys/bus/pci/devices/" + bdf, "../../../devices/" + path);
}

std::string bdf_on(int bus) {
    char b[16];
    std::snprintf(b, sizeof b, "0000:%02x:00.0", bus);
    return b;
}

/* Card `i` behind root port 0000:00:0i.1 running `speed` x`width`; returns the card's BDF. */
std::string card(const Root& r, int i, const std::string& speed, int width) {
    const std::string rp = "pci0000:00/0000:00:0" + std::to_string(i) + ".1";
    const std::string up = rp + "/" + bdf_on(0x10 + 3 * i);
    const std::string down = up + "/" + bdf_on(0x11 + 3 * i);
    const std::string ep = down + "/" + bdf_on(0x12 + 3 * i);
    pci(r, rp, speed, width);
    pci(r, up, "32.0 GT/s PCIe", 16);
    pci(r, down, "32.0 GT/s PCIe", 16);
    pci(r, ep, "32.0 GT/s PCIe", 16);
    return bdf_on(0x12 + 3 * i);
}

struct Slot { const char* speed; int width; };

/* The weights the eight cards give, or "unknown", by the quadratic rule or the duplex model. */
std::string weights(const Root& r, const Slot* slots, bool duplex = false) {
    std::string bdf[8];
    for (int i = 0; i < 8; ++i) bdf[i] = card(r, i, slots[i].speed, slots[i].width);
    double gbs[8] = { 0 };
    for (int i = 0; i < 8; ++i) {
        R4DArLink l;
        if (r4d_ar_link_bottleneck(r.dir, bdf[i], &l)) gbs[i] = r4d_ar_link_gbs(l);
    }
    int w[8];
    if (!(duplex ? r4d_ar_weights_duplex(gbs, 8, w) : r4d_ar_weights_from_rates(gbs, 8, w)))
        return "unknown";
    std::string s;
    for (int i = 0; i < 8; ++i) s += (i ? "," : "") + std::to_string(w[i]);
    return s;
}

const char* G5 = "32.0 GT/s PCIe";
const char* G4 = "16.0 GT/s PCIe";

}  // namespace

TEST(the_slowest_hop_is_the_root_port_behind_an_on_card_switch) {
    Root r;
    const std::string bdf = card(r, 2, G4, 8);
    R4DArLink l;
    REQUIRE(r4d_ar_link_bottleneck(r.dir, bdf, &l));
    CHECK_EQ(l.bdf, std::string("0000:00:02.1"));
    CHECK(l.gts == 16.0);
    CHECK_EQ(l.width, 8);
    CHECK(std::fabs(r4d_ar_link_gbs(l) - 16.0 * 8 * 128.0 / 130.0 / 8.0) < 1e-9);
}

TEST(two_degraded_links_give_the_measured_weights) {
    Root r;
    const Slot s[8] = { { G5, 16 }, { G5, 8 }, { G4, 8 }, { G5, 16 },
                        { G4, 8 }, { G5, 8 }, { G5, 16 }, { G5, 16 } };
    CHECK_EQ(weights(r, s), std::string("4,1,0,4,0,1,4,4"));
}

TEST(repaired_links_give_four_and_one) {
    Root r;
    const Slot s[8] = { { G5, 16 }, { G5, 8 }, { G5, 16 }, { G5, 16 },
                        { G5, 8 }, { G5, 8 }, { G5, 16 }, { G5, 16 } };
    CHECK_EQ(weights(r, s), std::string("4,1,4,4,1,1,4,4"));
}

/* The duplex model with four Gen5 x16 and four Gen5 x8 links: x16 holds 2/9 of the message and
 * x8 1/36, so each direction of every link carries the same time's bytes. With two of the x8
 * links at Gen4 those two hold nothing (they would need T B > N) and Gen5 x8 one seventh of x16. */
TEST(the_duplex_model_balances_both_directions_of_every_link) {
    Root r;
    const Slot fixed[8] = { { G5, 16 }, { G5, 8 }, { G5, 8 }, { G5, 16 },
                            { G5, 8 }, { G5, 8 }, { G5, 16 }, { G5, 16 } };
    CHECK_EQ(weights(r, fixed, true), std::string("64,8,8,64,8,8,64,64"));
    Root r2;
    const Slot older[8] = { { G5, 16 }, { G5, 8 }, { G4, 8 }, { G5, 16 },
                            { G4, 8 }, { G5, 8 }, { G5, 16 }, { G5, 16 } };
    CHECK_EQ(weights(r2, older, true), std::string("64,9,0,64,0,9,64,64"));
    Root r3;
    Slot even[8];
    for (Slot& x : even) x = { G5, 16 };
    CHECK_EQ(weights(r3, even, true), std::string("64,64,64,64,64,64,64,64"));
    /* The time each rank's bytes take, (1 + 6 x) / B, is the same for every rank with a shard. */
    const double gbs[8] = { 63, 31.5, 31.5, 63, 31.5, 31.5, 63, 63 };
    int w[8];
    REQUIRE(r4d_ar_weights_duplex(gbs, 8, w));
    double sum = 0;
    for (int i = 0; i < 8; ++i) sum += w[i];
    const double tf = (1 + 6 * w[0] / sum) / gbs[0], ts = (1 + 6 * w[1] / sum) / gbs[1];
    CHECK(std::fabs(tf - ts) / tf < 1e-9);
}

TEST(equal_links_give_equal_weights) {
    Root r;
    Slot s[8];
    for (Slot& x : s) x = { G5, 16 };
    CHECK_EQ(weights(r, s), std::string("4,4,4,4,4,4,4,4"));
}

/* A Gen2 link moves 8 of every 10 bits, Gen3 on 128 of 130: 5.0 GT/s x16 is 8.0 GB/s, an eighth
 * of Gen5 x16 at 63.0, so its weight rounds to 0. */
TEST(the_line_code_is_counted) {
    R4DArLink l;
    l.gts = 5.0; l.width = 16;
    CHECK(std::fabs(r4d_ar_link_gbs(l) - 8.0) < 1e-9);
    l.gts = 8.0; l.width = 16;
    CHECK(std::fabs(r4d_ar_link_gbs(l) - 8.0 * 16 * 128.0 / 130.0 / 8.0) < 1e-9);
}

/* A hop without link files (a bridge inside a switch) is skipped; a path where no hop says
 * anything has no rate, and one such card leaves all weights unknown, not guessed. */
TEST(a_silent_path_has_no_rate_and_no_weights) {
    Root r;
    const Slot s[8] = { { G5, 16 }, { G5, 8 }, { G4, 8 }, { G5, 16 },
                        { G4, 8 }, { G5, 8 }, { G5, 16 }, { G5, 16 } };
    std::string bdf[8];
    for (int i = 0; i < 8; ++i) bdf[i] = card(r, i, s[i].speed, s[i].width);
    pci(r, "pci0000:00/0000:00:09.1", "", 0);
    pci(r, "pci0000:00/0000:00:09.1/0000:40:00.0", "", 0);
    R4DArLink l;
    CHECK(!r4d_ar_link_bottleneck(r.dir, "0000:40:00.0", &l));
    CHECK(!r4d_ar_link_bottleneck(r.dir, "0000:41:00.0", &l));
    double gbs[8] = { 0 };
    for (int i = 0; i < 7; ++i) {
        R4DArLink x;
        if (r4d_ar_link_bottleneck(r.dir, bdf[i], &x)) gbs[i] = r4d_ar_link_gbs(x);
    }
    int w[8];
    CHECK(!r4d_ar_weights_from_rates(gbs, 8, w));
}

RAD_TEST_MAIN()
