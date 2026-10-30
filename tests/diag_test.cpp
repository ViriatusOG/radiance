/* diag_test.cpp -- the startup report (core/diag/diag.h).
 *
 * The collector reads /proc and /sys under a root directory, so each case builds its machine:
 * two R9700s behind on-card switches, one on a root port running at half speed under load, an
 * integrated GPU, and the usual Linux host files. The layout copies a real gfx1201 host's sysfs
 * (KFD topology location ids, pp_dpm_pcie's table, pp_od_clk_voltage's offset block), so a parser
 * that only handles a simplified fixture fails.
 *
 *   Collected   each fact reaches its field, including every hop of a card's PCIe path
 *   Checked     each finding fires on a machine with the problem and stays quiet otherwise;
 *               an idle link parked low is not a finding
 *   Reported    every option in the command-line table reaches the report; secrets do not
 */
#include "rad_test.h"

#include "rad_core.h"
#include "diag/diag.h"
#include "gen_config.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace rad;
namespace fs = std::filesystem;

namespace {

/* A machine of its own: an empty directory under the test's temporary directory. */
struct Root {
    std::string dir;
    Root() {
        std::string tmpl = radtest::tmp_path("root.XXXXXX");
        dir = mkdtemp(tmpl.data()) ? tmpl : "";
    }
    void put(const std::string& rel, const std::string& text) const {
        const fs::path p = fs::path(dir) / rel.substr(1);
        fs::create_directories(p.parent_path());
        std::ofstream(p) << text;
    }
    void mkdir(const std::string& rel) const { fs::create_directories(fs::path(dir) / rel.substr(1)); }
    /* A relative symlink, as sysfs's are, so it resolves inside the root. */
    void link(const std::string& rel, const std::string& target) const {
        const fs::path p = fs::path(dir) / rel.substr(1);
        fs::create_directories(p.parent_path());
        fs::create_symlink(target, p);
    }
};

/* A PCI device directory under /sys/devices and its /sys/bus/pci/devices link. `path` is the
 * device's position under the root complex, "pci0000:00/0000:00:01.1/0000:01:00.0". */
void pci(const Root& r, const std::string& path, const std::string& cls, const std::string& cur,
         int curw, const std::string& max, int maxw) {
    const std::string bdf = path.substr(path.rfind('/') + 1);
    const std::string d = "/sys/devices/" + path;
    r.put(d + "/class", cls + "\n");
    r.put(d + "/current_link_speed", cur + "\n");
    r.put(d + "/current_link_width", std::to_string(curw) + "\n");
    r.put(d + "/max_link_speed", max + "\n");
    r.put(d + "/max_link_width", std::to_string(maxw) + "\n");
    r.link("/sys/bus/pci/devices/" + bdf, "../../../devices/" + path);
}

/* An R9700 behind its on-card switch, under root port `rp`. */
void r9700(const Root& r, const std::string& rp, const std::string& up, const std::string& down,
           const std::string& bdf, const std::string& rp_cur, int busy, int64_t cap_uw,
           const char* drm, int dpm_cur, int64_t kfd_loc, int node) {
    const std::string base = "pci0000:00/0000:00:" + rp;
    pci(r, base, "0x060400", rp_cur, 8, "32.0 GT/s PCIe", 8);
    pci(r, base + "/" + up, "0x060400", "32.0 GT/s PCIe", 16, "32.0 GT/s PCIe", 16);
    pci(r, base + "/" + up + "/" + down, "0x060400", "32.0 GT/s PCIe", 16, "32.0 GT/s PCIe", 16);
    const std::string path = base + "/" + up + "/" + down + "/" + bdf;
    pci(r, path, "0x030000", "32.0 GT/s PCIe", 16, "32.0 GT/s PCIe", 16);
    const std::string d = "/sys/devices/" + path;
    r.put(d + "/vendor", "0x1002\n");
    r.put(d + "/device", "0x7551\n");
    r.mkdir("/sys/bus/pci/drivers/amdgpu");
    r.link(d + "/driver", "../../../../../../../bus/pci/drivers/amdgpu");
    r.put(d + "/numa_node", "-1\n");
    r.put(d + "/power/runtime_status", "active\n");
    r.put(d + "/power/control", "auto\n");
    r.put(d + "/iommu_group/type", "identity\n");
    r.mkdir(d + "/drm/" + drm);
    r.put(d + "/mem_info_vram_total", "34208743424\n");
    r.put(d + "/mem_info_vis_vram_total", "34208743424\n");
    r.put(d + "/gpu_busy_percent", std::to_string(busy) + "\n");
    std::string pcie;
    const char* lv[] = { "0: 2.5GT/s, x8 250Mhz", "1: 16.0GT/s, x8 616Mhz", "2: 32.0GT/s, x8 1143Mhz" };
    for (int i = 0; i < 3; ++i) pcie += std::string(lv[i]) + (i == dpm_cur ? " *" : "") + "\n";
    r.put(d + "/pp_dpm_pcie", pcie);
    r.put(d + "/pp_dpm_sclk", "0: 500Mhz \n1: 1768Mhz *\n2: 2350Mhz \n");
    r.put(d + "/pp_dpm_mclk", "0: 96Mhz \n1: 456Mhz \n2: 772Mhz \n3: 875Mhz *\n4: 1124Mhz \n5: 1258Mhz \n");
    r.put(d + "/hwmon/hwmon2/name", "amdgpu\n");
    r.put(d + "/hwmon/hwmon2/power1_cap", std::to_string(cap_uw) + "\n");
    r.put(d + "/hwmon/hwmon2/power1_cap_default", "300000000\n");
    r.put(d + "/hwmon/hwmon2/power1_cap_max", "330000000\n");
    r.put(d + "/pp_od_clk_voltage",
          "OD_SCLK_OFFSET:\n0Mhz\nOD_MCLK:\n0: 97Mhz\n1: 1259MHz\nOD_RANGE:\n"
          "SCLK_OFFSET:    -500Mhz       1000Mhz\nVDDGFX_OFFSET:    -200mv          0mv\n"
          "OD_VDDGFX_OFFSET:\n-75mV\n");
    r.put("/sys/class/kfd/kfd/topology/nodes/" + std::to_string(node) + "/properties",
          "cpu_cores_count 0\nsimd_count 128\nlocation_id " + std::to_string(kfd_loc) +
              "\ndomain 0\ngfx_target_version 120001\n");
}

/* A 16-core desktop part, two R9700s (the second on a root port at Gen4 while busy, with a lower
 * power cap) and an integrated GPU. */
void machine(const Root& r) {
    std::string cpu;
    for (int i = 0; i < 4; ++i)
        cpu += "processor\t: " + std::to_string(i) + "\nmodel name\t: AMD Ryzen 9 7950X3D 16-Core Processor\n"
               "physical id\t: 0\ncore id\t\t: " + std::to_string(i / 2) + "\n\n";
    r.put("/proc/cpuinfo", cpu);
    r.put("/proc/meminfo", "MemTotal:       64669548 kB\nMemFree:        14000000 kB\n"
                           "MemAvailable:   53000000 kB\nSwapTotal:             0 kB\nSwapFree:              0 kB\n");
    r.put("/proc/sys/kernel/osrelease", "7.2.7-arch1-1\n");
    r.put("/etc/os-release", "NAME=\"Arch Linux\"\nPRETTY_NAME=\"Arch Linux\"\nID=arch\n");
    r.put("/proc/cmdline", "initrd=\\initramfs-linux.img root=PARTUUID=x rw iommu=pt amdgpu.ppfeaturemask=0xfff7ffff\n");
    r.put("/proc/self/cgroup", "0::/user.slice\n");
    r.put("/sys/fs/cgroup/user.slice/memory.max", "max\n");
    r.put("/sys/kernel/mm/transparent_hugepage/enabled", "[always] madvise never\n");
    r.put("/sys/kernel/mm/transparent_hugepage/defrag", "always defer defer+madvise [madvise] never\n");
    r.put("/sys/kernel/mm/transparent_hugepage/shmem_enabled", "always within_size [advise] never deny force\n");
    r.put("/sys/module/ttm/parameters/pages_limit", "8083693\n");
    r.put("/sys/module/amdgpu/parameters/cwsr_enable", "1\n");
    r.put("/proc/sys/kernel/io_uring_disabled", "0\n");
    r.put("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "powersave\n");
    r.put("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference", "balance_performance\n");
    r.put("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver", "amd-pstate-epp\n");
    r.mkdir("/sys/class/iommu/ivhd0");
    r.mkdir("/sys/kernel/iommu_groups/0");
    r9700(r, "01.1", "0000:01:00.0", "0000:02:00.0", "0000:03:00.0", "32.0 GT/s PCIe", 3,
          300000000, "card0", 2, 768, 1);
    r9700(r, "01.3", "0000:05:00.0", "0000:06:00.0", "0000:07:00.0", "16.0 GT/s PCIe", 90,
          280000000, "card1", 1, 1792, 2);
    /* The integrated GPU: on the root bus, a display device with its own KFD node. */
    pci(r, "pci0000:00/0000:00:08.1/0000:1b:00.0", "0x030000", "16.0 GT/s PCIe", 16, "16.0 GT/s PCIe", 16);
    r.put("/sys/devices/pci0000:00/0000:00:08.1/0000:1b:00.0/vendor", "0x1002\n");
    r.put("/sys/devices/pci0000:00/0000:00:08.1/0000:1b:00.0/device", "0x13c0\n");
    r.put("/sys/class/kfd/kfd/topology/nodes/3/properties",
          "simd_count 4\nlocation_id 6912\ndomain 0\ngfx_target_version 100306\n");
    r.put("/sys/class/kfd/kfd/topology/nodes/0/properties",
          "cpu_cores_count 32\nsimd_count 0\nlocation_id 0\ndomain 0\ngfx_target_version 0\n");
}

const diag::CardFacts* card(const diag::Facts& f, const char* bdf) {
    for (const diag::CardFacts& c : f.cards) if (c.bdf == bdf) return &c;
    return nullptr;
}

/* The engine's half: the two cards are ranks 0 and 1. */
diag::Facts served_machine(const Root& r) {
    diag::Facts f = diag::collect(r.dir);
    for (diag::CardFacts& c : f.cards) {
        if (c.bdf == "0000:03:00.0") { c.rank = 0; c.device = 0; }
        if (c.bdf == "0000:07:00.0") { c.rank = 1; c.device = 1; }
    }
    return f;
}

bool has(const std::vector<diag::Finding>& v, const char* id, diag::Finding::Level lv) {
    for (const diag::Finding& x : v) if (x.id == id && x.level == lv) return true;
    return false;
}
bool has(const std::vector<diag::Finding>& v, const char* id) {
    for (const diag::Finding& x : v) if (x.id == id) return true;
    return false;
}

}  /* namespace */

/* ------------------------------------------------------------------ collected */

TEST(host_facts_reach_their_fields) {
    Root r;
    machine(r);
    const diag::Facts f = diag::collect(r.dir);
    const diag::HostFacts& h = f.host;
    CHECK_EQ(h.cpu_model, std::string("AMD Ryzen 9 7950X3D 16-Core Processor"));
    CHECK_EQ(h.threads, 4);
    CHECK_EQ(h.cores, 2);
    CHECK_EQ(h.mem_total, (int64_t)64669548 * 1024);
    CHECK_EQ(h.mem_available, (int64_t)53000000 * 1024);
    CHECK_EQ(h.swap_total, 0);
    CHECK_EQ(h.kernel, std::string("7.2.7-arch1-1"));
    CHECK_EQ(h.distro, std::string("Arch Linux"));
    CHECK_EQ(h.cgroup_mem_limit, -2);
    CHECK_EQ(h.thp_enabled, std::string("always"));
    CHECK_EQ(h.thp_defrag, std::string("madvise"));
    CHECK_EQ(h.thp_shmem, std::string("advise"));
    CHECK_EQ(h.iommu, std::string("passthrough"));
    CHECK_EQ(h.cmdline, std::string("iommu=pt amdgpu.ppfeaturemask=0xfff7ffff"));
    CHECK_EQ(h.ttm_pages_limit, 8083693);
    CHECK_EQ(h.cwsr_enable, 1);
    CHECK_EQ(h.io_uring, std::string("available"));
    CHECK_EQ(h.cpu_governor, std::string("powersave"));
    CHECK_EQ(h.cpu_epp, std::string("balance_performance"));
}

TEST(every_display_device_is_listed_with_its_kfd_target) {
    Root r;
    machine(r);
    const diag::Facts f = diag::collect(r.dir);
    CHECK_EQ((int)f.cards.size(), 3);
    const diag::CardFacts* a = card(f, "0000:03:00.0");
    const diag::CardFacts* i = card(f, "0000:1b:00.0");
    CHECK(a && i);
    if (!a || !i) return;
    CHECK_EQ(a->gfx, std::string("gfx1201"));
    CHECK_EQ(a->simd, 128);
    CHECK_EQ(i->gfx, std::string("gfx1036"));
    CHECK_EQ(a->vendor, std::string("0x1002"));
    CHECK_EQ(a->pci_device, std::string("0x7551"));
    CHECK_EQ(a->driver, std::string("amdgpu"));
    CHECK_EQ(a->drm, std::string("card0"));
    CHECK_EQ(a->iommu_type, std::string("identity"));
    CHECK_EQ(a->vram_total, (int64_t)34208743424);
    CHECK_EQ(a->busy_percent, 3);
    CHECK_EQ(a->sclk, std::string("500-2350 MHz"));
    CHECK_EQ(a->mclk, std::string("96-1258 MHz"));
    CHECK_EQ(a->power_cap_uw, (int64_t)300000000);
    CHECK_EQ(a->power_cap_max_uw, (int64_t)330000000);
    CHECK_EQ(a->vddgfx_offset, std::string("-75 mV"));
    CHECK_EQ((int)a->dpm_pcie.size(), 3);
    CHECK_EQ(a->dpm_pcie_cur, 2);
    CHECK_EQ(a->dpm_pcie[1], std::string("1: 16.0GT/s, x8 616Mhz"));
}

/* The root port is part of the path: both cards' own links read Gen5 x16, and only the second
 * card's root port shows 16 GT/s. */
TEST(the_whole_pcie_path_is_read_root_port_first) {
    Root r;
    machine(r);
    const diag::Facts f = diag::collect(r.dir);
    const diag::CardFacts* b = card(f, "0000:07:00.0");
    CHECK(b);
    if (!b) return;
    CHECK_EQ((int)b->chain.size(), 4);
    if (b->chain.size() != 4) return;
    CHECK_EQ(b->chain[0].bdf, std::string("0000:00:01.3"));
    CHECK_EQ(b->chain[0].cur_speed, std::string("16.0 GT/s PCIe"));
    CHECK_EQ(b->chain[0].max_speed, std::string("32.0 GT/s PCIe"));
    CHECK_EQ(b->chain[0].max_width, 8);
    CHECK_EQ(b->chain[3].bdf, std::string("0000:07:00.0"));
    CHECK_EQ(b->chain[3].cur_speed, std::string("32.0 GT/s PCIe"));
    CHECK_EQ(b->chain[3].cur_width, 16);
    CHECK(diag::link_gts(b->chain[0].cur_speed) == 16.0);
}

TEST(an_empty_root_is_all_unknown_and_does_not_fail) {
    Root r;
    const diag::Facts f = diag::collect(r.dir);
    CHECK(f.cards.empty());
    CHECK(f.host.cpu_model.empty());
    CHECK_EQ(f.host.mem_total, -1);
    CHECK_EQ(f.host.iommu, std::string("off"));
    const std::string text = diag::render({}, {}, f, diag::RunFacts{}, {});
    CHECK(text.find("unknown") != std::string::npos);
    CHECK(text.find("no display-class PCI device") != std::string::npos);
}

/* A limit on an ancestor cgroup binds the process. The leaf says max, its parent 8 GiB and the
 * grandparent 16 GiB, so the process gets 8 GiB. */
TEST(the_tightest_cgroup_limit_on_the_way_up_is_reported) {
    Root r;
    machine(r);
    r.put("/proc/self/cgroup", "0::/system.slice/radiance.slice/radiance.service\n");
    r.put("/sys/fs/cgroup/system.slice/radiance.slice/radiance.service/memory.max", "max\n");
    r.put("/sys/fs/cgroup/system.slice/radiance.slice/memory.max", "8589934592\n");
    r.put("/sys/fs/cgroup/system.slice/memory.max", "17179869184\n");
    CHECK_EQ(diag::collect(r.dir).host.cgroup_mem_limit, (int64_t)8 << 30);
    /* Every level unlimited is unlimited, not unknown. */
    r.put("/sys/fs/cgroup/system.slice/radiance.slice/memory.max", "max\n");
    r.put("/sys/fs/cgroup/system.slice/memory.max", "max\n");
    CHECK_EQ(diag::collect(r.dir).host.cgroup_mem_limit, -2);
    /* Inside a cgroup namespace the process is at "/", and the mount's root is its group. */
    r.put("/proc/self/cgroup", "0::/\n");
    r.put("/sys/fs/cgroup/memory.max", "4294967296\n");
    CHECK_EQ(diag::collect(r.dir).host.cgroup_mem_limit, (int64_t)4 << 30);
}

/* unique_id is a full 64-bit hash that overflows a signed 64-bit read. It precedes the keys the
 * target is read from and must not stop the read. */
TEST(a_kfd_value_too_large_for_a_signed_read_does_not_hide_the_rest) {
    Root r;
    machine(r);
    r.put("/sys/class/kfd/kfd/topology/nodes/1/properties",
          "cpu_cores_count 0\nunique_id 17293822569102704640\nsimd_count 128\nlocation_id 768\n"
          "domain 0\ngfx_target_version 120001\n");
    const diag::Facts f = diag::collect(r.dir);
    const diag::CardFacts* a = card(f, "0000:03:00.0");
    CHECK(a);
    if (!a) return;
    CHECK_EQ(a->gfx, std::string("gfx1201"));
    CHECK_EQ(a->simd, 128);
}

/* A serving process reads the SMU-backed files of its own cards only. Other cards are still
 * listed from their PCI and memory files. */
TEST(only_the_managed_cards_power_files_are_read) {
    Root r;
    machine(r);
    const std::vector<std::string> mine = { "0000:03:00.0" };
    const diag::Facts f = diag::collect(r.dir, &mine);
    CHECK_EQ((int)f.cards.size(), 3);
    const diag::CardFacts* a = card(f, "0000:03:00.0");
    const diag::CardFacts* b = card(f, "0000:07:00.0");
    CHECK(a && b);
    if (!a || !b) return;
    CHECK_EQ(a->busy_percent, 3);
    CHECK_EQ(a->power_cap_uw, (int64_t)300000000);
    CHECK_EQ(a->vddgfx_offset, std::string("-75 mV"));
    CHECK_EQ(b->busy_percent, -1);
    CHECK_EQ(b->power_cap_uw, -1);
    CHECK(b->dpm_pcie.empty());
    CHECK(b->sclk.empty());
    CHECK(b->vddgfx_offset.empty());
    CHECK_EQ(b->vram_total, (int64_t)34208743424);
    CHECK_EQ(b->gfx, std::string("gfx1201"));
    CHECK_EQ((int)b->chain.size(), 4);
}

TEST(environment_names_and_never_their_values) {
    char a[] = "RADIANCE_TUNE=gemm:tile=128";
    char b[] = "PATH=/usr/bin";
    char c[] = "RADIANCE_LOG=debug";
    char d[] = "RADIANCE_SECRETISH=sk-not-for-the-log";
    char* env[] = { a, b, c, d, nullptr };
    const std::vector<std::string> n = diag::env_names(env, "RADIANCE_");
    CHECK_EQ(n.size(), (size_t)3);
    if (n.size() != 3) return;
    CHECK_EQ(n[0], std::string("RADIANCE_LOG"));
    CHECK_EQ(n[1], std::string("RADIANCE_SECRETISH"));
    CHECK_EQ(n[2], std::string("RADIANCE_TUNE"));
    diag::RunFacts run;
    run.environment = n;
    const std::string text = diag::render({}, {}, diag::Facts{}, run, {});
    const std::string json = diag::to_json({}, {}, diag::Facts{}, run, {});
    CHECK(text.find("  RADIANCE_TUNE\n") != std::string::npos);
    CHECK(json.find("RADIANCE_SECRETISH") != std::string::npos);
    CHECK(text.find("sk-not-for-the-log") == std::string::npos);
    CHECK(json.find("sk-not-for-the-log") == std::string::npos);
    CHECK(text.find("gemm:tile") == std::string::npos);
}

/* ------------------------------------------------------------------ checked */

TEST(a_link_below_max_under_load_is_a_warning) {
    Root r;
    machine(r);
    const auto found = diag::check(served_machine(r), diag::RunFacts{});
    CHECK(has(found, "link-below-max", diag::Finding::Warn));
    CHECK(has(found, "link-dpm-below-top", diag::Finding::Warn));
    bool names_port = false;
    for (const diag::Finding& x : found)
        if (x.id == "link-below-max") names_port = x.what.find("0000:00:01.3") != std::string::npos &&
                                                   x.what.find("0000:07:00.0") != std::string::npos;
    CHECK(names_port);
}

/* A link parks low at rest, so the same parked root port on an idle card is not a finding. */
TEST(an_idle_card_with_a_parked_link_is_not_a_finding) {
    Root r;
    machine(r);
    diag::Facts f = served_machine(r);
    for (diag::CardFacts& c : f.cards) c.busy_percent = 0;
    const auto found = diag::check(f, diag::RunFacts{});
    CHECK(!has(found, "link-below-max"));
    CHECK(!has(found, "link-dpm-below-top"));
}

TEST(cards_set_up_differently_are_named) {
    Root r;
    machine(r);
    const auto found = diag::check(served_machine(r), diag::RunFacts{});
    CHECK(has(found, "cards-differ", diag::Finding::Warn));
    for (const diag::Finding& x : found)
        if (x.id == "cards-differ") {
            CHECK(x.what.find("power cap") != std::string::npos);
            CHECK(x.what.find("280 W") != std::string::npos);
        }
    CHECK(has(found, "power-cap-below-default", diag::Finding::Note));
    CHECK(has(found, "od-runtime-suspend", diag::Finding::Note));
    CHECK(has(found, "cpu-governor", diag::Finding::Note));
}

/* A power cap is judged against the default, not the maximum. Both cards allow 330 W and default
 * to 300 W; the first is at its default, the second is lowered to 280 W and is named. */
TEST(a_power_cap_is_judged_against_the_cards_default) {
    Root r;
    machine(r);
    diag::Facts f = served_machine(r);
    auto named = [](const std::vector<diag::Finding>& v, const char* bdf) {
        for (const diag::Finding& x : v)
            if (x.id == "power-cap-below-default" && x.what.find(bdf) != std::string::npos)
                return true;
        return false;
    };
    auto found = diag::check(f, diag::RunFacts{});
    CHECK(!named(found, "0000:03:00.0"));
    CHECK(named(found, "0000:07:00.0"));
    /* A card that does not report its default is not judged. */
    for (diag::CardFacts& c : f.cards) c.power_cap_default_uw = -1;
    CHECK(!has(diag::check(f, diag::RunFacts{}), "power-cap-below-default"));
}

TEST(cards_alike_say_nothing_about_each_other) {
    Root r;
    machine(r);
    diag::Facts f = served_machine(r);
    for (diag::CardFacts& c : f.cards) {
        c.power_cap_uw = c.power_cap_max_uw;
        c.runtime_control = "on";
        c.busy_percent = 0;
        if (!c.chain.empty()) c.chain.front().max_width = 8;
    }
    const auto found = diag::check(f, diag::RunFacts{});
    CHECK(!has(found, "cards-differ"));
    CHECK(!has(found, "power-cap-below-default"));
    CHECK(!has(found, "od-runtime-suspend"));
}

TEST(the_unserved_igpu_is_not_checked_as_a_rank) {
    Root r;
    machine(r);
    const auto found = diag::check(served_machine(r), diag::RunFacts{});
    for (const diag::Finding& x : found) CHECK(x.what.find("1b:00.0") == std::string::npos);
}

TEST(host_findings) {
    Root r;
    machine(r);
    diag::Facts f = served_machine(r);
    diag::RunFacts run;
    CHECK(!has(diag::check(f, run), "cwsr-off"));
    f.host.cwsr_enable = 0;
    f.host.thp_enabled = "never";
    for (diag::CardFacts& c : f.cards) c.iommu_type = "DMA-FQ";
    f.host.mem_available = (int64_t)1 << 30;
    f.host.swap_total = (int64_t)8 << 30;
    f.host.swap_free = (int64_t)6 << 30;
    run.host_pool_mib = 12288;
    run.pinned_bytes = f.host.ttm_pages_limit * f.host.page_size - ((int64_t)1 << 20);
    const auto found = diag::check(f, run);
    CHECK(has(found, "cwsr-off", diag::Finding::Incompatible));
    CHECK(has(found, "thp-never", diag::Finding::Warn));
    CHECK(has(found, "iommu-translated", diag::Finding::Note));
    CHECK(has(found, "memory-low", diag::Finding::Warn));
    CHECK(has(found, "swap-in-use", diag::Finding::Note));
    /* Refusals, then warnings, then notes, so the section leads with what to act on. */
    bool order_ok = true;
    for (size_t i = 1; i < found.size(); ++i)
        if (found[i].level > found[i - 1].level) order_ok = false;
    CHECK(order_ok);
    /* THP off on a passthrough IOMMU is only a note: the page size has not measured slower. */
    for (diag::CardFacts& c : f.cards) c.iommu_type = "identity";
    CHECK(has(diag::check(f, run), "thp-never", diag::Finding::Note));
}

/* CWSR off is refused where cards serve, with its fix; the host backend beside a loaded driver is
 * only told. */
TEST(what_startup_refuses_is_named_with_its_fix) {
    Root r;
    machine(r);
    diag::Facts f = served_machine(r);
    CHECK(diag::incompatible(f).empty());
    f.host.cwsr_enable = 0;
    auto bad = diag::incompatible(f);
    CHECK(has(bad, "cwsr-off", diag::Finding::Incompatible));
    for (const diag::Finding& x : bad)
        if (x.id == "cwsr-off") CHECK(x.fix.find("cwsr_enable=1") != std::string::npos);
    /* The settings main only reports are not refusals. */
    for (diag::CardFacts& c : f.cards) {
        c.iommu_type = "DMA-FQ";
        if (c.rank == 0) c.vram_visible = (int64_t)256 << 20;
    }
    bad = diag::incompatible(f);
    CHECK_EQ(bad.size(), (size_t)1);
    for (diag::CardFacts& c : f.cards) c.rank = -1;
    bad = diag::incompatible(f);
    for (const diag::Finding& x : bad) CHECK(x.level != diag::Finding::Incompatible);
    CHECK(has(bad, "cwsr-off", diag::Finding::Warn));
}

TEST(a_run_started_anyway_says_so_at_the_top) {
    Root r;
    machine(r);
    diag::Facts f = served_machine(r);
    f.host.cwsr_enable = 0;
    diag::RunFacts run;
    run.unsupported = true;
    const auto found = diag::check(f, run);
    const std::string text = diag::render({}, {}, f, run, found);
    CHECK(text.rfind(diag::unsupported_banner().front(), 0) == 0);
    CHECK(text.find("--proceed-with-incompatible-configuration") != std::string::npos);
    CHECK(text.find("1 incompatible (started anyway)") != std::string::npos);
    CHECK(text.find("INCOMPATIBLE cwsr-off") != std::string::npos);
    const nlohmann::json j =
        nlohmann::json::parse(diag::to_json({}, {}, f, run, found), nullptr, false);
    CHECK(!j.is_discarded());
    if (j.is_discarded()) return;
    CHECK(j["unsupported_configuration"] == true);
    CHECK(j["findings"][0]["level"] == "incompatible");
    /* Without the flag the report carries no banner. */
    CHECK(diag::render({}, {}, f, diag::RunFacts{}, found).rfind("STARTUP REPORT", 0) == 0);
}

TEST(engine_findings) {
    Root r;
    machine(r);
    diag::Facts f = served_machine(r);
    diag::RunFacts run;
    run.tp = 2;
    run.tp_wire_lossy = true;
    run.tp_wire_min_kb = 128;
    run.prefix_cache = false;
    run.host_weight_bytes = (int64_t)20 << 30;
    run.p2p = 0;
    run.p2p_blocked = "0->1";
    run.ignored.push_back("using 2 of 3 [0:gfx1201 1:gfx1201], ignoring [2:gfx1036]");
    run.reference_kernels = true;
    run.profile_ops = true;
    const auto found = diag::check(f, run);
    CHECK(has(found, "tp-wire-lossy", diag::Finding::Note));
    CHECK(has(found, "prefix-cache-off", diag::Finding::Note));
    /* Passthrough is not advised for peer access: on an Intel VT-d host it disabled P2P. */
    for (const diag::Finding& x : found)
        if (x.id == "p2p-off") CHECK(x.fix.find("IOMMU in passthrough") == std::string::npos);
    CHECK(has(found, "experts-from-host", diag::Finding::Note));
    CHECK(has(found, "p2p-off", diag::Finding::Note));
    CHECK(has(found, "device-ignored", diag::Finding::Note));
    CHECK(has(found, "reference-kernels", diag::Finding::Warn));
    CHECK(has(found, "profile-ops", diag::Finding::Warn));
    CHECK(!has(found, "host-backend"));
    run.p2p_forced_off = true;
    CHECK(!has(diag::check(f, run), "p2p-off"));
    /* A cache the engine turned off itself (KL mode, an embedding model) is not a setting to
     * undo; its reason goes on the option's line instead. */
    run.prefix_cache_off_why = "an embedding model: an encoder keeps no KV to reuse";
    CHECK(!has(diag::check(f, run), "prefix-cache-off"));
    /* A card is present but the engine came up on the host backend. */
    for (diag::CardFacts& c : f.cards) { c.rank = -1; }
    run.host_backend = true;
    CHECK(has(diag::check(f, run), "host-backend", diag::Finding::Warn));
    /* An integrated GPU with a few hundred MiB of carve-out is not a card the engine would serve on. */
    for (diag::CardFacts& c : f.cards) c.vram_total = (int64_t)512 << 20;
    CHECK(!has(diag::check(f, run), "host-backend"));
}

/* The IOMMU mode comes from the serving cards: an identity-mapped integrated GPU beside a
 * translated discrete card still means the serving card is translated. */
TEST(the_iommu_mode_is_read_from_the_serving_cards) {
    Root r;
    machine(r);
    r.put("/proc/cmdline", "root=PARTUUID=x rw\n");
    r.put("/sys/devices/pci0000:00/0000:00:01.3/0000:05:00.0/0000:06:00.0/0000:07:00.0/iommu_group/type",
          "DMA-FQ\n");
    diag::Facts f = served_machine(r);
    CHECK_EQ(f.host.iommu, std::string("mixed"));
    CHECK(f.host.iommu_why.find("DMA-FQ for 0000:07:00.0") != std::string::npos);
    /* The groups beat iommu=pt on the command line, which the platform may not honour. */
    r.put("/proc/cmdline", "root=PARTUUID=x rw iommu=pt\n");
    CHECK_EQ(served_machine(r).host.iommu, std::string("mixed"));
    diag::RunFacts run;
    run.host_pool_mib = 12288;
    CHECK(has(diag::check(f, run), "iommu-translated", diag::Finding::Note));
    for (diag::CardFacts& c : f.cards) c.iommu_type = "identity";
    CHECK(!has(diag::check(f, run), "iommu-translated"));
}

/* ------------------------------------------------------------------ reported */

/* Every option in the table reaches the report with a value, not the placeholder an option
 * without a formatter gets. An option added without one fails here by name. */
TEST(every_option_in_the_table_is_reported) {
    Config c;
    Argv a({ "--model", "/m/x.rad" });
    CHECK_EQ(config_parse(a.argc(), a.argv.data(), &c), 0);
    const std::vector<diag::ConfigRow> rows = diag::config_rows(c);
    const std::vector<std::string> names = diag::config_option_names();
    CHECK_EQ(rows.size(), names.size());
    CHECK(names.size() > 80);
    for (size_t i = 0; i < rows.size() && i < names.size(); ++i) {
        CHECK_EQ(rows[i].flag, names[i]);
        if (rows[i].value == "(not shown)") {
            radtest::fail(__FILE__, __LINE__, "no value shown for " + rows[i].flag);
        }
    }
    diag::RunFacts all;
    all.all_options = true;
    const std::string text = diag::render(rows, {}, diag::Facts{}, all, {});
    for (const std::string& n : names)
        if (text.find("  " + n + " ") == std::string::npos)
            radtest::fail(__FILE__, __LINE__, "the report has no line for " + n);
}

/* Without -v the section lists only what is not at its default, and counts the rest. */
TEST(the_report_lists_the_options_not_at_their_defaults) {
    Config c;
    Argv a({ "--model", "/m/x.rad", "--tp", "2" });
    CHECK_EQ(config_parse(a.argc(), a.argv.data(), &c), 0);
    const std::vector<diag::ConfigRow> rows = diag::config_rows(c);
    int defaults = 0;
    for (const diag::ConfigRow& x : rows) defaults += x.source == "default";
    CHECK(defaults > 0);
    const std::string text = diag::render(rows, {}, diag::Facts{}, diag::RunFacts{}, {});
    CHECK(text.find("  --tp ") != std::string::npos);
    CHECK(text.find("  --model ") != std::string::npos);
    CHECK(text.find("  --max-num-seqs ") == std::string::npos);
    CHECK(text.find(std::to_string(defaults) + " options at their default") != std::string::npos);
}

TEST(sources_and_secrets) {
    setenv("RADIANCE_KERNELS", "libr4d,libref", 1);
    Config c;
    Argv a({ "--model", "/m/x.rad", "--tp", "2", "--api-key", "sk-very-secret-value",
             "--gpu-headroom", "128" });
    CHECK_EQ(config_parse(a.argc(), a.argv.data(), &c), 0);
    unsetenv("RADIANCE_KERNELS");
    std::vector<diag::ConfigRow> rows = diag::config_rows(c);
    auto row = [&](const char* f) -> const diag::ConfigRow* {
        for (const diag::ConfigRow& x : rows) if (x.flag == f) return &x;
        return nullptr;
    };
    CHECK(row("--tp") && row("--tp")->source == "command line" && row("--tp")->value == "2");
    CHECK(row("--max-num-seqs") && row("--max-num-seqs")->source == "default");
    /* The alias reports as the option it is. */
    CHECK(row("--gpu-headroom-mib") && row("--gpu-headroom-mib")->source == "command line" &&
          row("--gpu-headroom-mib")->value == "128");
    CHECK(row("--kernels") && row("--kernels")->source == "environment (RADIANCE_KERNELS)" &&
          row("--kernels")->value == "libr4d,libref");
    CHECK(row("--api-key") && row("--api-key")->value == "set");

    /* A decision supplies the value and the reason. A default it resolves becomes `auto`, a given
     * option keeps its source, and a topic with no option is returned as derived. */
    std::vector<diag::Decision> d;
    diag::decide(&d, "--num-speculative-tokens", "2", "the drafter's own depth");
    diag::decide(&d, "--tp", "2", "as given");
    diag::decide(&d, "vram split", "weights 20 GiB", "the report");
    diag::decide(&d, "--num-speculative-tokens", "3", "revised");
    CHECK_EQ(d.size(), (size_t)3);
    std::vector<diag::Decision> derived;
    diag::merge(&rows, d, &derived);
    CHECK(row("--num-speculative-tokens")->value == "3");
    CHECK(row("--num-speculative-tokens")->source == "auto");
    CHECK(row("--num-speculative-tokens")->why == "revised");
    CHECK(row("--tp")->source == "command line");
    CHECK_EQ(derived.size(), (size_t)1);
    CHECK(!derived.empty() && derived[0].key == "vram split");

    const std::string text = diag::render(rows, derived, diag::Facts{}, diag::RunFacts{}, {});
    const std::string json = diag::to_json(rows, derived, diag::Facts{}, diag::RunFacts{}, {});
    CHECK(text.find("sk-very-secret-value") == std::string::npos);
    CHECK(json.find("sk-very-secret-value") == std::string::npos);
    CHECK(text.find("DERIVED") != std::string::npos);
}

/* Bytes that are not UTF-8 are replaced, not thrown on. A --model path is arbitrary operator
 * bytes, and a strict serialiser would fail every /diagnostics request with a 500. */
TEST(a_value_that_is_not_utf8_does_not_break_the_json) {
    Config c;
    Argv a({ "--model", "/m/caf\xe9.rad" });
    CHECK_EQ(config_parse(a.argc(), a.argv.data(), &c), 0);
    c.sample_dry_seq_breakers = std::vector<std::string>{ "\xff\xfe" };
    std::vector<diag::ConfigRow> rows;
    bool threw = false;
    try { rows = diag::config_rows(c); } catch (...) { threw = true; }
    CHECK(!threw);
    diag::Facts f;
    diag::CardFacts card;
    card.bdf = "0000:03:00.0";
    card.name = "R9700 \xc3\x28";
    f.cards.push_back(card);
    std::string out;
    try { out = diag::to_json(rows, {}, f, diag::RunFacts{}, {}); } catch (...) { threw = true; }
    CHECK(!threw);
    const nlohmann::json j = nlohmann::json::parse(out, nullptr, false);
    CHECK(!j.is_discarded());
    if (j.is_discarded()) return;
    CHECK(j["configuration"][0]["value"].get<std::string>() == "/m/caf\xef\xbf\xbd.rad");
}

/* Each sampler field names its own source. A generation_config.json stating only temperature and
 * top_p supplied those two; top_k and min_p are SamplingParams' defaults and must not cite it. */
TEST(sampler_defaults_carry_a_source_per_field) {
    Config c;
    Argv a({ "--model", "/m/x.rad", "--min-p", "0.05" });
    CHECK_EQ(config_parse(a.argc(), a.argv.data(), &c), 0);
    GenSampling g;
    CHECK(gen_sampling_parse(R"({"temperature": 0.6, "top_p": 0.95})", "test", &g));
    SamplingParams sp = g.onto(SamplingParams{});
    sp.min_p = c.sample_min_p;
    const std::vector<GenFieldSource> s = gen_sampling_sources(g, "/m/gen.json", sp, c);
    auto find = [&](const char* flag) -> const GenFieldSource* {
        for (const GenFieldSource& x : s) if (x.flag == flag) return &x;
        return nullptr;
    };
    CHECK(find("--temp") && find("--temp")->why == "from /m/gen.json" &&
          find("--temp")->value == "0.6");
    CHECK(find("--top-p") && find("--top-p")->why == "from /m/gen.json");
    CHECK(find("--top-k") && find("--top-k")->why.find("/m/gen.json states no top_k") == 0);
    /* --min-p was given: its own row says "command line" and is not overridden here. */
    CHECK(!find("--min-p"));
    /* With no source, every field says the model stated nothing. */
    for (const GenFieldSource& x : gen_sampling_sources(GenSampling{}, "", SamplingParams{}, c))
        CHECK(x.why.find("the model states no sampling of its own") == 0);
}

/* A value the server resolves is recorded where it is resolved, not recomputed by the report.
 * Until then, the option's line says only how it will be resolved. */
TEST(auto_server_bounds_are_not_recomputed_by_the_report) {
    Config c;
    Argv a({ "--model", "/m/x.rad" });
    CHECK_EQ(config_parse(a.argc(), a.argv.data(), &c), 0);
    std::vector<diag::ConfigRow> rows = diag::config_rows(c);
    for (const diag::ConfigRow& x : rows)
        if (x.flag == "--http-threads" || x.flag == "--max-n" || x.flag == "--max-queued-requests")
            CHECK(x.value.rfind("auto (", 0) == 0);
    std::vector<diag::Decision> d, derived;
    diag::decide(&d, "--http-threads", "32", "one a hardware thread, at least 4");
    diag::merge(&rows, d, &derived);
    for (const diag::ConfigRow& x : rows)
        if (x.flag == "--http-threads") CHECK(x.value == "32" && x.source == "auto");
}

TEST(the_json_carries_every_section) {
    Root r;
    machine(r);
    const diag::Facts f = served_machine(r);
    const auto found = diag::check(f, diag::RunFacts{});
    Config c;
    Argv a({ "--model", "/m/x.rad" });
    CHECK_EQ(config_parse(a.argc(), a.argv.data(), &c), 0);
    const nlohmann::json j = nlohmann::json::parse(
        diag::to_json(diag::config_rows(c), {}, f, diag::RunFacts{}, found), nullptr, false);
    CHECK(!j.is_discarded());
    if (j.is_discarded()) return;
    CHECK(j["configuration"].is_array() && j["configuration"].size() == diag::config_option_names().size());
    CHECK(j["hardware"]["cards"].size() == 3);
    CHECK(j["hardware"]["cards"][1]["link_chain"].size() == 4);
    CHECK(j["hardware"]["host"]["iommu"]["mode"] == "passthrough");
    CHECK(j["findings"].size() == found.size());
    const std::string text = diag::render(diag::config_rows(c), {}, f, diag::RunFacts{}, found);
    CHECK(text.find("00:01.3 16.0 GT/s x8 (max 32.0 GT/s x8)") != std::string::npos);
    CHECK(text.find("WARN link-below-max") != std::string::npos);
}

RAD_TEST_MAIN()
