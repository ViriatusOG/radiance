/* projection_test.cpp -- the per-request switch that feeds the control vector.
 *
 * Two halves, both host-side:
 *
 *   1. the derivation (core/sched/derive.cpp, `projection_on_rows`) that turns the step's requests
 *      into the int32 row mask the projection op reads. Getting this wrong is silent in both
 *      directions: a mask that is never set serves the stock model to a client that asked for the
 *      vector, and one that is always set applies it to every request, including those that
 *      explicitly turned it off. Mixed steps are the case that matters (two conversations in one
 *      step, one projected), and that is what these cases build.
 *
 *   2. the reference operation's arithmetic (libref/ref_hc.cpp, `hc_project`), called through the
 *      plugin the way the engine calls it, checked against the operation the vector's package
 *      documents: h -= scale * (h . v) v per hyper-connection stream, and an untouched row where
 *      the mask is 0.
 *
 * The kernel parity check (libavx against libref) is libavx/avx_cases.cpp, fed to rad-avx-check;
 * libr4d's device kernel has no in-tree parity case yet (the rad-kbench fixture is still missing).
 * This file pins the meaning.
 */
#include "rad_test.h"

#include "rad_core.h"
#include "sched/derive.h"

#include "../arch/common/rad_arch.h"

#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

namespace {

/* A step of `n_seq` sequences of `tokens_each` tokens, each with its own request, and the two
 * arrays a derivation reads. */
struct Step {
    std::vector<rad::Request>  reqs;
    std::vector<rad::StepEntry> entries;
    std::vector<int32_t>       cu;
    rad::StepPlan              plan;

    Step(int n_seq, int tokens_each) {
        reqs.resize((size_t) n_seq);
        entries.resize((size_t) n_seq);
        cu.push_back(0);
        for (int i = 0; i < n_seq; ++i) {
            entries[(size_t) i].req = &reqs[(size_t) i];
            entries[(size_t) i].n_tokens = tokens_each;
            cu.push_back(cu.back() + tokens_each);
        }
        plan.e = entries;
        plan.n_tok = cu.back();
    }
};

int run_derive(const Step& s, int32_t* out, int64_t cap, int64_t* n_out) {
    const int idx = rad::derive_index("projection_on_rows");
    if (idx < 0) return -999;
    rad::DeriveInput in;
    in.plan = &s.plan;
    in.cu_seqlens = s.cu.data();
    in.n_tok = s.plan.n_tok;
    in.n_seq = (int64_t) s.plan.e.size();
    return rad::derive_row(idx)->fn(in, out, cap, n_out);
}

/* libref's kernel table, dlopen'd from the build's radiance home (ctest sets RADIANCE_HOME), the
 * same way ref_test reaches it. */
const RadKernelInfo* libref_kernel(const char* op) {
    static int (*count)(void) = nullptr;
    static const RadKernelInfo* (*at)(int) = nullptr;
    if (!count) {
        std::vector<std::string> paths;
        const std::string home = ::home();
        paths.push_back(home + "/kernels/libref.so");
        paths.push_back("build/radiance_home/kernels/libref.so");
        paths.push_back("./libref.so");
        for (const std::string& p : paths) {
            void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!h) continue;
            count = (int (*)(void)) dlsym(h, "rad_kernel_count");
            at = (const RadKernelInfo* (*)(int)) dlsym(h, "rad_kernel_at");
            if (count && at) break;
            count = nullptr;
            at = nullptr;
        }
        if (!count) return nullptr;
    }
    for (int i = 0; i < count(); ++i) {
        const RadKernelInfo* k = at(i);
        if (k && k->op && std::strcmp(k->op, op) == 0) return k;
    }
    return nullptr;
}

}  // namespace

TEST(the_mask_follows_each_requests_own_choice) {
    Step s(2, 3);
    s.reqs[0].sp.use_control_vector = true;    /* this conversation asked for the vector */
    s.reqs[1].sp.use_control_vector = false;   /* this one did not */

    int32_t mask[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
    int64_t n = 0;
    CHECK_OK(run_derive(s, mask, 8, &n));
    CHECK_EQ(n, (int64_t) 6);
    CHECK_EQ(mask[0], 1);
    CHECK_EQ(mask[2], 1);
    /* The second sequence's rows: untouched. A mixed step is what a naive "any request in the
     * step asked for it" would get wrong. */
    CHECK_EQ(mask[3], 0);
    CHECK_EQ(mask[5], 0);
}

TEST(the_mask_is_all_zero_when_no_request_asked) {
    Step s(3, 2);
    int32_t mask[8] = { 9, 9, 9, 9, 9, 9, 9, 9 };
    int64_t n = 0;
    CHECK_OK(run_derive(s, mask, 8, &n));
    CHECK_EQ(n, (int64_t) 6);
    for (int i = 0; i < 6; ++i) CHECK_EQ(mask[i], 0);
}

TEST(the_derivation_refuses_a_buffer_too_small) {
    Step s(2, 4);
    int32_t mask[4] = {};
    int64_t n = 0;
    CHECK(run_derive(s, mask, 4, &n) < 0);   /* 8 rows in 4 slots */
}

TEST(the_reference_projection_removes_the_direction_per_stream) {
    /* Directly the operation the package's README states, on a synthetic stream: after one
     * application every stream's component along v is (1 - scale) times what it was, and with a
     * unit direction at the stored scale 1 it is exactly zero, i.e. h -= (h.v) v. Run at two
     * widths: a gated stream (hc 2) and a single plain one (hc 1 -- inkling2's bf16 residual,
     * where the stream loop is one pass). */
    const int64_t n = 8, rows = 2;
    for (const int64_t hc : { (int64_t) 2, (int64_t) 1 }) {
    std::vector<float> h((size_t) (rows * hc * n));
    std::vector<float> v((size_t) n);
    std::vector<int32_t> on{ 1, 0 };
    for (int64_t i = 0; i < n; ++i) v[(size_t) i] = (i % 2) ? 0.6f : 0.8f;
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < hc; ++c)
            for (int64_t i = 0; i < n; ++i)
                h[(size_t) ((r * hc + c) * n + i)] = (float) (0.1 * (r + 1) + 0.05 * (double) c +
                                                             0.01 * (double) i);

    /* The expected result, computed here independently of the kernel. */
    std::vector<float> want = h;
    for (int64_t c = 0; c < hc; ++c) {
        double dot = 0.0;
        for (int64_t i = 0; i < n; ++i) dot += (double) h[(size_t) ((0 * hc + c) * n + i)] * v[(size_t) i];
        for (int64_t i = 0; i < n; ++i)
            want[(size_t) ((0 * hc + c) * n + i)] =
                (float) ((double) h[(size_t) ((0 * hc + c) * n + i)] - dot * (double) v[(size_t) i]);
    }

    /* Call libref's own launch, the way the engine does: through the plugin's kernel table. */
    const RadKernelInfo* k = libref_kernel("hc_project");
    if (!k || !k->launch) {
        ARTIFACT_MISSING("libref.so with an hc_project kernel");
    }

    RadTensor t[3]{};
    t[0].data = h.data(); t[0].dtype = RAD_F32; t[0].rank = 2;
    t[0].shape[0] = rows; t[0].shape[1] = hc * n;
    t[0].stride[0] = hc * n; t[0].stride[1] = 1;
    t[1].data = v.data(); t[1].dtype = RAD_F32; t[1].rank = 1;
    t[1].shape[0] = n; t[1].stride[0] = 1;
    t[2].data = on.data(); t[2].dtype = RAD_I32; t[2].rank = 1;
    t[2].shape[0] = rows; t[2].stride[0] = 1;

    const RadParam ps[] = {
        RAD_RANGE("M", 1, (int64_t) rows),
        RAD_INT("n", n),
        RAD_INT("hc", hc),
        RAD_F64("scale", 1.0),
        RAD_STR("dtype", "f32"),
    };
    RadArgs a{};
    a.t = t; a.n_t = 3;
    a.p = ps; a.n_p = (int)(sizeof(ps) / sizeof(ps[0]));

    const int rc = k->launch(&a, nullptr);
    CHECK_OK(rc);

    for (size_t i = 0; i < h.size(); ++i) CHECK_NEAR(h[i], want[i], 1e-5);

    /* The row with a zero mask is untouched, bit for bit: that is what makes an unprojected
     * request identical to a deployment with no vector loaded. */
    for (int64_t c = 0; c < hc; ++c)
        for (int64_t i = 0; i < n; ++i) {
            const size_t k = (size_t) ((1 * hc + c) * n + i);
            CHECK_EQ(std::memcmp(&h[k], &want[k], sizeof(float)), 0);
        }
    }
}

RAD_TEST_MAIN()
