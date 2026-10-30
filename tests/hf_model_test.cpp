/* hf_model_test.cpp -- a --model that names a Hugging Face repo (core/hf_model.h).
 *
 * Both questions are answered before anything is fetched; a wrong answer costs a download or a
 * confusing refusal.
 *
 *   Is it a repo   a path that exists is never looked up, and a missing model file is reported
 *                  as missing rather than sent to the hub as a repo name
 *   What to serve  a model repo holds one container beside a README, a recipe and a licence;
 *                  a checkpoint repo holds none and is served as the directory
 *
 * No network: where the fetch itself runs (scripts/hfget.sh, from the build tree's libexec/), a
 * `curl` placed first on PATH answers for the hub.
 */
#include "rad_test.h"

#include "hf_model.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace rad;
namespace fs = std::filesystem;

namespace {

/* An empty directory of its own under the test's temporary directory. */
struct TempDir {
    std::string path;
    TempDir() {
        std::string t = radtest::tmp_path("dir.XXXXXX");
        path = mkdtemp(t.data()) ? t : "";
    }
    void touch(const std::string& name) const {
        if (FILE* f = std::fopen((path + "/" + name).c_str(), "w")) std::fclose(f);
    }
};

}  /* namespace */

TEST(a_repo_id_is_two_names_of_the_hubs_characters) {
    CHECK(hf_repo_id("StillDeadcode/minicpm5-2b-fp8"));
    CHECK(hf_repo_id("StillDeadcode/qwen3.8-27b-mxfp4"));
    CHECK(hf_repo_id("Qwen/Qwen3.8-Flash-Next"));
    CHECK(hf_repo_id("hf-internal-testing/tiny_random.v2"));

    CHECK(!hf_repo_id(""));
    CHECK(!hf_repo_id("minicpm5-2b-fp8"));               /* no owner */
    CHECK(!hf_repo_id("/abs/model"));                     /* absolute */
    CHECK(!hf_repo_id("a/b/c"));                          /* a path, not a repo */
    CHECK(!hf_repo_id("owner/"));
    CHECK(!hf_repo_id("/name"));
    CHECK(!hf_repo_id("../x"));                           /* a relative path, not a name */
    CHECK(!hf_repo_id("./x"));
    CHECK(!hf_repo_id("owner/.hidden"));
    CHECK(!hf_repo_id("-x/y"));                           /* would read as an option */
    CHECK(!hf_repo_id("owner/name with space"));
    CHECK(!hf_repo_id("owner/na$me"));
    CHECK(!hf_repo_id("C:/models"));
}

TEST(a_missing_model_file_is_not_sent_to_the_hub) {
    CHECK(!hf_repo_id("models/qwen.rad"));
    CHECK(!hf_repo_id("ckpt/model.safetensors"));
    CHECK(!hf_repo_id("ckpt/model.gguf"));
    CHECK(!hf_repo_id("ckpt/model.safetensors.index.json"));
}

TEST(a_path_that_exists_is_never_a_repo) {
    TempDir t;
    CHECK(!t.path.empty());
    const std::string cwd = fs::current_path().string();
    CHECK_EQ(chdir(t.path.c_str()), 0);
    fs::create_directories("owner/name");
    CHECK(!hf_repo_id("owner/name"));                     /* exists: a checkpoint directory */
    CHECK(hf_repo_id("owner/other"));                     /* does not: a repo */
    CHECK_EQ(chdir(cwd.c_str()), 0);
}

TEST(the_one_container_in_a_repo_is_what_is_served) {
    TempDir t;
    t.touch("README.md");
    t.touch("LICENSE");
    t.touch("q38-27b.recipe");
    t.touch(".hfget-manifest");
    std::string why;
    CHECK_EQ(hf_served_path(t.path, &why), t.path);       /* no container: a checkpoint */

    t.touch("qwen3.8-27b-mxfp4.rad");
    CHECK_EQ(hf_served_path(t.path, &why), t.path + "/qwen3.8-27b-mxfp4.rad");

    fs::create_directories(t.path + "/sub");
    t.touch("sub/other.rad");                             /* only the top level counts */
    CHECK_EQ(hf_served_path(t.path, &why), t.path + "/qwen3.8-27b-mxfp4.rad");
}

TEST(two_containers_are_refused_by_name) {
    TempDir t;
    t.touch("a.rad");
    t.touch("b.rad");
    std::string why;
    CHECK_EQ(hf_served_path(t.path, &why), std::string());
    CHECK(why.find("a.rad") != std::string::npos);
    CHECK(why.find("b.rad") != std::string::npos);
    CHECK(why.find("--model") != std::string::npos);
}

TEST(the_download_dir_defaults_to_the_xdg_cache) {
    const char* x = std::getenv("XDG_CACHE_HOME");
    const std::string saved = x ? x : "";
    setenv("XDG_CACHE_HOME", "/c", 1);
    CHECK_EQ(hf_default_download_dir(), std::string("/c/radiance/hf"));
    unsetenv("XDG_CACHE_HOME");
    if (const char* h = std::getenv("HOME"); h && *h)
        CHECK_EQ(hf_default_download_dir(), std::string(h) + "/.cache/radiance/hf");
    if (x) setenv("XDG_CACHE_HOME", saved.c_str(), 1);
}

/* ================================================================== a restart with the hub out of reach */
namespace {

/* Where resolve_hf_model looks for the downloader: libexec/radiance/ beside this binary's
 * directory, where the build copies it. */
std::string hfget_beside_this_binary() {
    std::error_code ec;
    const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    return (exe.parent_path() / ".." / "libexec" / "radiance" / "hfget.sh").lexically_normal();
}

bool on_path(const char* tool) {
    const char* path = std::getenv("PATH");
    std::string all = path ? path : "";
    size_t at = 0;
    while (at <= all.size()) {
        size_t c = all.find(':', at);
        if (c == std::string::npos) c = all.size();
        const std::string d = all.substr(at, c - at);
        if (!d.empty() && ::access((d + "/" + tool).c_str(), X_OK) == 0) return true;
        at = c + 1;
    }
    return false;
}

/* PATH with a directory holding a `curl` script first, restored on destruction. */
struct FakeCurl {
    TempDir bin;
    std::string saved;
    explicit FakeCurl(const std::string& script) {
        const char* p = std::getenv("PATH");
        saved = p ? p : "";
        const std::string f = bin.path + "/curl";
        if (FILE* o = std::fopen(f.c_str(), "w")) {
            std::fputs(script.c_str(), o);
            std::fclose(o);
        }
        ::chmod(f.c_str(), 0755);
        setenv("PATH", (bin.path + ":" + saved).c_str(), 1);
    }
    ~FakeCurl() { setenv("PATH", saved.c_str(), 1); }
};

/* A hub that cannot be reached: every request fails as curl does without a network. hfget.sh
 * exits 3 (nothing on disk changed). */
const char* kOffline = "#!/bin/sh\nexit 6\n";
/* A hub that lists a file and then fails to send it: hfget.sh exits 1 (a file may be partial). */
const char* kBrokenTransfer =
    "#!/bin/sh\n"
    "for a in \"$@\"; do case \"$a\" in *api/models*)\n"
    "  printf '[{\"type\":\"file\",\"path\":\"new.bin\",\"size\":5}]'; exit 0;; esac; done\n"
    "exit 22\n";

}  /* namespace */

/* hfget.sh exits 3 when it changed nothing on disk, and a copy a complete fetch left (it has the
 * success manifest) is then served, with a warning, so a restart does not need the hub. Without
 * the manifest the copy may be incomplete; after any other failure a file may be partial; either
 * way nothing is served. */
TEST(a_restart_serves_the_complete_copy_when_the_hub_cannot_be_reached) {
    NEED_ARTIFACT(hfget_beside_this_binary());
    if (!on_path("jq")) SKIP("no jq on PATH, which hfget.sh needs");
    TempDir root;
    REQUIRE(!root.path.empty());
    const std::string repo = "radiance-test-owner/radiance-test-model";
    const std::string dest = root.path + "/" + repo;
    fs::create_directories(dest);
    if (FILE* f = std::fopen((dest + "/m.rad").c_str(), "w")) std::fclose(f);

    auto resolve = [&](std::string* model, std::string* log) {
        Config c;
        c.model = repo;
        c.download_dir = root.path;
        bool ok = false;
        *log = stderr_of([&] { ok = resolve_hf_model(c); });
        *model = c.model;
        return ok;
    };
    std::string model, log;
    {
        FakeCurl curl(kOffline);
        /* No manifest: an interrupted first fetch, refused. */
        CHECK(!resolve(&model, &log));
        CHECK(has(log, "exit 3"));

        if (FILE* f = std::fopen((dest + "/.hfget-manifest").c_str(), "w")) {
            std::fputs("0\tm.rad\n", f);
            std::fclose(f);
        }
        CHECK(resolve(&model, &log));
        CHECK_EQ(model, dest + "/m.rad");
        CHECK(has(log, "serving the copy fetched before"));
    }
    {
        FakeCurl curl(kBrokenTransfer);
        CHECK(!resolve(&model, &log));
        CHECK(has(log, "exit 1"));
    }
}

/* A hub whose repo holds safetensors beside a torch copy of the same weights and a codec that only
 * ships as a .pth: the copies (pytorch_model*.pth, a pickle .bin) are skipped, the codec is
 * fetched, since the checkpoint reader reads it (core/format/torch_zip.h). */
const char* kCodecRepo =
    "#!/bin/sh\n"
    "out=''; url=''; prev=''\n"
    "for a in \"$@\"; do\n"
    "  [ \"$prev\" = -o ] && out=\"$a\"\n"
    "  case \"$a\" in https://*) url=\"$a\";; esac\n"
    "  prev=\"$a\"\n"
    "done\n"
    "case \"$url\" in *api/models*)\n"
    "  printf '[{\"type\":\"file\",\"path\":\"model.safetensors\",\"size\":3},"
    "{\"type\":\"file\",\"path\":\"codec.pth\",\"size\":3},"
    "{\"type\":\"file\",\"path\":\"pytorch_model.pth\",\"size\":3},"
    "{\"type\":\"file\",\"path\":\"pytorch_model.bin\",\"size\":3}]'; exit 0;; esac\n"
    "[ -n \"$out\" ] && printf abc > \"$out\"\n"
    "exit 0\n";

TEST(a_pth_that_is_not_the_weights_again_is_fetched_beside_the_safetensors) {
    NEED_ARTIFACT(hfget_beside_this_binary());
    if (!on_path("jq")) SKIP("no jq on PATH, which hfget.sh needs");
    TempDir root;
    REQUIRE(!root.path.empty());
    FakeCurl curl(kCodecRepo);
    const std::string cmd = "sh '" + hfget_beside_this_binary() +
                            "' radiance-test-owner/radiance-test-model '" + root.path +
                            "' >/dev/null 2>&1";
    CHECK_EQ(std::system(cmd.c_str()), 0);
    CHECK(fs::exists(root.path + "/model.safetensors"));
    CHECK(fs::exists(root.path + "/codec.pth"));
    CHECK(!fs::exists(root.path + "/pytorch_model.pth"));
    CHECK(!fs::exists(root.path + "/pytorch_model.bin"));
}

/* A listing that names a file outside the destination -- absolute, or a `..` segment -- is not
 * a listing of a repo at all, so hfget.sh refuses the whole thing rather than half-obeying it:
 * silently skipping the entry would leave a directory that reads as complete and is not, and
 * the manifest would certify it. */
const char* kEscapeListing =
    "#!/bin/sh\n"
    "for a in \"$@\"; do case \"$a\" in *api/models*)\n"
    "  printf '[{\"type\":\"file\",\"path\":\"../escape.bin\",\"size\":1},"
    "{\"type\":\"file\",\"path\":\"kept.bin\",\"size\":1}]'; exit 0;; esac; done\n"
    "exit 22\n";

TEST(a_listing_naming_a_path_outside_the_destination_is_refused) {
    NEED_ARTIFACT(hfget_beside_this_binary());
    if (!on_path("jq")) SKIP("no jq on PATH, which hfget.sh needs");
    TempDir root;
    REQUIRE(!root.path.empty());
    FakeCurl curl(kEscapeListing);
    const std::string cmd = "sh '" + hfget_beside_this_binary() +
                            "' radiance-test-owner/radiance-test-model '" + root.path +
                            "' >/dev/null 2>'" + root.path + "/err'";
    CHECK(std::system(cmd.c_str()) != 0);
    CHECK(!fs::exists(root.path + "/escape.bin"));
    CHECK(!fs::exists(root.path + "/kept.bin"));          /* the whole listing is refused */
    CHECK(!fs::exists(root.path + "/.hfget-manifest"));
    std::string err;
    if (FILE* e = std::fopen((root.path + "/err").c_str(), "r")) {
        char buf[256];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf - 1, e)) > 0) { buf[n] = '\0'; err += buf; }
        std::fclose(e);
    }
    CHECK(has(err, "does not land inside the destination"));
}

/* An LFS file's oid is the hub's claim about its bytes; the download is kept only once those
 * bytes hash to it -- the size check alone would take a swapped file of the same length, and a
 * spliced resume at any length the hub likes. A mismatch deletes the file and fails the run;
 * a match lands in the manifest, which is what makes "warm" mean verified-warm after. */
const char* kBadOid =
    "#!/bin/sh\n"
    "out=''; url=''; prev=''\n"
    "for a in \"$@\"; do\n"
    "  [ \"$prev\" = -o ] && out=\"$a\"\n"
    "  case \"$a\" in https://*) url=\"$a\";; esac\n"
    "  prev=\"$a\"\n"
    "done\n"
    "case \"$url\" in *api/models*)\n"
    "  printf '[{\"type\":\"file\",\"path\":\"big.bin\",\"size\":10,"
    "\"lfs\":{\"oid\":\"0000000000000000000000000000000000000000000000000000000000000000\"}}]';"
    " exit 0;; esac\n"
    "[ -n \"$out\" ] && printf 0123456789 > \"$out\"\n"
    "exit 0\n";

const char* kGoodOid =
    "#!/bin/sh\n"
    "out=''; url=''; prev=''\n"
    "for a in \"$@\"; do\n"
    "  [ \"$prev\" = -o ] && out=\"$a\"\n"
    "  case \"$a\" in https://*) url=\"$a\";; esac\n"
    "  prev=\"$a\"\n"
    "done\n"
    "case \"$url\" in *api/models*)\n"
    "  oid=$(printf 0123456789 | sha256sum | cut -d' ' -f1)\n"
    "  printf '[{\"type\":\"file\",\"path\":\"big.bin\",\"size\":10,\"lfs\":{\"oid\":\"%s\"}}]'"
    " \"$oid\"; exit 0;; esac\n"
    "[ -n \"$out\" ] && printf 0123456789 > \"$out\"\n"
    "exit 0\n";

TEST(an_lfs_file_that_does_not_hash_to_its_oid_is_not_kept) {
    NEED_ARTIFACT(hfget_beside_this_binary());
    if (!on_path("jq")) SKIP("no jq on PATH, which hfget.sh needs");
    if (!on_path("sha256sum")) SKIP("no sha256sum on PATH, which hfget.sh needs");
    TempDir root;
    REQUIRE(!root.path.empty());
    const std::string fetch = "sh '" + hfget_beside_this_binary() +
                              "' radiance-test-owner/radiance-test-model '" + root.path +
                              "' >/dev/null 2>'" + root.path + "/err'";
    {
        FakeCurl curl(kBadOid);
        CHECK(std::system(fetch.c_str()) != 0);
        CHECK(!fs::exists(root.path + "/big.bin"));        /* deleted, not left to be resumed */
        CHECK(!fs::exists(root.path + "/.hfget-manifest"));
        std::string err;
        if (FILE* e = std::fopen((root.path + "/err").c_str(), "r")) {
            char buf[256];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof buf - 1, e)) > 0) { buf[n] = '\0'; err += buf; }
            std::fclose(e);
        }
        CHECK(has(err, "hashes to"));
    }
    {
        FakeCurl curl(kGoodOid);
        CHECK_EQ(std::system(fetch.c_str()), 0);
        CHECK(fs::exists(root.path + "/big.bin"));
        CHECK_EQ(fs::file_size(root.path + "/big.bin"), 10u);
        CHECK(fs::exists(root.path + "/.hfget-manifest"));  /* the verified copy is certified */
    }
}

/* --model-revision names the revision the fetch reads: the pinned ref is the one listed, and
 * an unpinned fetch reads the repo's moving default branch -- which a deployment must be able
 * to refuse outright rather than restart onto. */
const char* kPinnedOnly =
    "#!/bin/sh\n"
    "out=''; url=''; prev=''\n"
    "for a in \"$@\"; do\n"
    "  [ \"$prev\" = -o ] && out=\"$a\"\n"
    "  case \"$a\" in https://*) url=\"$a\";; esac\n"
    "  prev=\"$a\"\n"
    "done\n"
    "case \"$url\" in *api/models*/tree/refs/pr/42*)\n"
    "  printf '[{\"type\":\"file\",\"path\":\"m.rad\",\"size\":10}]'; exit 0;; esac\n"
    "case \"$url\" in *api/models*) exit 22;; esac\n"
    "[ -n \"$out\" ] && printf 0123456789 > \"$out\"\n"
    "exit 0\n";

TEST(the_pinned_revision_is_the_revision_fetched) {
    NEED_ARTIFACT(hfget_beside_this_binary());
    if (!on_path("jq")) SKIP("no jq on PATH, which hfget.sh needs");
    TempDir root;
    REQUIRE(!root.path.empty());
    FakeCurl curl(kPinnedOnly);
    const std::string repo = "radiance-test-owner/radiance-test-model";
    auto resolve = [&](const char* rev, std::string* model) {
        Config c;
        c.model = repo;
        c.download_dir = root.path;
        if (rev) c.model_revision = rev;
        bool ok = false;
        stderr_of([&] { ok = resolve_hf_model(c); });
        *model = c.model;
        return ok;
    };
    std::string model;
    CHECK(!resolve(nullptr, &model));              /* main: this hub has nothing there */
    CHECK(resolve("refs/pr/42", &model));
    CHECK_EQ(model, root.path + "/" + repo + "/m.rad");
}

/* With no --download-dir and neither XDG_CACHE_HOME nor HOME there is nowhere to fetch to, which
 * is refused by name before anything runs. */
TEST(a_repo_id_with_nowhere_to_fetch_to_is_refused) {
    const char* x = std::getenv("XDG_CACHE_HOME");
    const char* h = std::getenv("HOME");
    const std::string sx = x ? x : "", sh = h ? h : "";
    unsetenv("XDG_CACHE_HOME");
    unsetenv("HOME");
    Config c;
    c.model = "radiance-test-owner/radiance-test-model";
    bool ok = true;
    const std::string log = stderr_of([&] { ok = resolve_hf_model(c); });
    CHECK(!ok);
    CHECK(has(log, "--download-dir"));
    CHECK_EQ(c.model, std::string("radiance-test-owner/radiance-test-model"));
    if (x) setenv("XDG_CACHE_HOME", sx.c_str(), 1);
    if (h) setenv("HOME", sh.c_str(), 1);
}

RAD_TEST_MAIN()
