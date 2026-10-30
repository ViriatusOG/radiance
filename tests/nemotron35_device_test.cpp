// Independent operation checks on the engine's actual AQL queue, against libref. Ordinary host
// CTest skips these; the explicit card entry is the hardware gate. No model weights required.
#include "rad_test.h"
#include "rad_abi.h"
#include "rad_device.h"
#include "rad_plugin.h"
#include <dlfcn.h>
#include <memory>
#include <random>
#define REQUIRE_OK(expr) REQUIRE_EQ((expr), RAD_OK)

namespace {
struct Library {
    void* h;
    explicit Library(const char* name) {
        const char* home = std::getenv("RADIANCE_HOME");
        h = dlopen((std::string(home ? home : "") + "/kernels/" + name + ".so").c_str(),
                   RTLD_NOW | RTLD_LOCAL);
        if (!h) throw std::runtime_error(dlerror());
    }
    ~Library() { dlclose(h); }
    const RadKernelInfo* get(const char* op, int dim) {
        auto count = (int (*)())dlsym(h, "rad_kernel_count");
        auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
        for (int i = 0; i < count(); ++i) {
            auto k = at(i);
            if (std::strcmp(k->op, op)) continue;
            bool match = true;
            for (int j = 0; j < k->n_constraints; ++j) {
                auto& c = k->constraints[j];
                if (!std::strcmp(c.key, "head_dim") && c.op == RAD_C_EQ && c.ival != dim)
                    match = false;
                if (!std::strcmp(op, "gemm_nt") &&
                    std::strcmp(k->name, "gemm_bf16_pixtral_patch") &&
                    k->domain == RAD_DOMAIN_DEVICE)
                    match = false;
            }
            if (match) return k;
        }
        throw std::runtime_error(std::string("missing row ") + op);
    }
};
struct Buffer {
    std::vector<uint8_t> bytes;
    RadTensor t{};
    Buffer(uint32_t dt, int rows, int cols, int pitch = 0, int colstride = 1)
        : bytes(256 + (size_t)rows * (pitch ? pitch : cols * colstride) * (dt == RAD_I32 ? 4 : 2),
                0x5a) {
        t.data = bytes.data() + 128;
        t.dtype = dt;
        t.rank = cols ? 2 : 1;
        t.shape[0] = rows;
        t.stride[0] = pitch ? pitch : (cols ? cols * colstride : 1);
        if (cols) {
            t.shape[1] = cols;
            t.stride[1] = colstride;
        }
    }
    static std::unique_ptr<Buffer> ids(const std::vector<int>& ids, int planes = 1) {
        auto b = std::make_unique<Buffer>(RAD_I32, (int)ids.size(), 1);
        b->t.rank = planes == 1 ? 1 : 2;
        b->t.shape[0] = planes == 1 ? ids.size() : planes;
        b->t.shape[1] = ids.size() / planes;
        b->t.stride[0] = planes == 1 ? 1 : ids.size() / planes;
        b->t.stride[1] = 1;
        std::memcpy(b->t.data, ids.data(), ids.size() * 4);
        return b;
    }
    void fill(int seed) {
        std::mt19937 gen(seed);
        auto* p = (uint16_t*)t.data;
        for (int64_t r = 0; r < t.shape[0]; ++r)
            for (int64_t c = 0; c < t.shape[1]; ++c)
                p[r * t.stride[0] + c * t.stride[1]] =
                    rad_f32_to_bf16(((int)(gen() % 2001) - 1000) / 1700.f);
    }
};
RadParam pi(const char* k, int64_t v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_INT;
    p.ival = v;
    return p;
}
RadParam pf(const char* k, double v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_F64;
    p.dval = v;
    return p;
}
RadParam ps(const char* k, const char* v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_STR;
    p.sval = v;
    return p;
}
bool device() {
    RadDeviceProps p{};
    return rad_dev_props(0, &p) == RAD_OK && !p.is_host_backend && rad_dev_set(0) == RAD_OK;
}
void compare(const char* op, std::vector<Buffer*> b, std::vector<RadParam> params, int output,
             int dim = 128, bool exact = true) {
    Library ref("libref"), gpu("libr4d");
    auto kr = ref.get(op, dim), kg = gpu.get(op, dim);
    RadStream stream = nullptr;
    REQUIRE_OK(rad_stream_create(&stream, 0));
    std::vector<RadTensor> ht, dt;
    std::vector<std::vector<uint8_t>> oracle;
    std::vector<void*> alloc;
    for (auto* x : b)
        oracle.push_back(x->bytes);
    for (size_t i = 0; i < b.size(); ++i) {
        auto h = b[i]->t;
        h.data = oracle[i].data() + 128;
        ht.push_back(h);
        void* p = rad_dev_alloc(b[i]->bytes.size(), RAD_MEM_DEVICE);
        REQUIRE(p);
        alloc.push_back(p);
        REQUIRE_OK(rad_memcpy_async(p, b[i]->bytes.data(), b[i]->bytes.size(), stream));
        auto d = b[i]->t;
        d.data = (uint8_t*)p + 128;
        dt.push_back(d);
    }
    void* scratch = rad_dev_alloc(256, RAD_MEM_DEVICE);
    REQUIRE(scratch);
    RadArgs a{};
    a.t = ht.data();
    a.n_t = ht.size();
    a.p = params.data();
    a.n_p = params.size();
    REQUIRE_OK(kr->launch(&a, nullptr));
    if (kg->init) REQUIRE_OK(kg->init(params.data(), params.size(), 0, 1, &a.instance));
    a.t = dt.data();
    a.scratch = scratch;
    a.scratch_bytes = 256;
    // Multiple launches on the same queue also check scratch reuse and ordering.
    REQUIRE_OK(kg->launch(&a, stream));
    REQUIRE_OK(rad_stream_sync(stream));
    for (size_t i = 0; i < b.size(); ++i) {
        std::vector<uint8_t> got(b[i]->bytes.size());
        REQUIRE_OK(rad_memcpy_async(got.data(), alloc[i], got.size(), stream));
        REQUIRE_OK(rad_stream_sync(stream));
        if ((int)i != output) {
            CHECK(got == b[i]->bytes);
            continue;
        }
        const auto& t = b[i]->t;
        double err = 0, norm = 0, maxerr = 0;
        size_t different = 0;
        auto expected = b[i]->bytes;
        auto* e = (uint16_t*)(oracle[i].data() + 128);
        auto* g = (uint16_t*)(got.data() + 128);
        for (int64_t r = 0; r < t.shape[0]; ++r)
            for (int64_t c = 0; c < t.shape[1]; ++c) {
                size_t j = r * t.stride[0] + c * t.stride[1];
                double x = rad_bf16_to_f32(e[j]), y = rad_bf16_to_f32(g[j]);
                REQUIRE(std::isfinite(y));
                err += (x - y) * (x - y);
                norm += x * x;
                maxerr = std::max(maxerr, std::abs(x - y));
                different += e[j] != g[j];
                std::memcpy(expected.data() + 128 + 2 * j, got.data() + 128 + 2 * j, 2);
            }
        CHECK(got == expected); // All padding and both 128-byte canaries are unchanged.
        const double rel = std::sqrt(err / std::max(norm, 1e-30));
        fprintf(stderr, "    %s dim=%d relL2=%.9g maxAbs=%.9g bf16_differences=%zu\n", op, dim, rel,
                maxerr, different);
        if (exact)
            CHECK_EQ(different, (size_t)0);
        else {
            CHECK(rel <= 0.01);
            CHECK(maxerr <= 0.015);
        }
    }
    // Invalid dimensions fail synchronously before writing any operand.
    std::vector<uint8_t> before(b[output]->bytes.size()), after(before.size());
    REQUIRE_OK(rad_memcpy_async(before.data(), alloc[output], before.size(), stream));
    REQUIRE_OK(rad_stream_sync(stream));
    auto invalid = dt;
    invalid[0].shape[0] = 0;
    a.t = invalid.data();
    CHECK(kg->launch(&a, stream) < 0);
    invalid = dt;
    invalid[0].stride[1] = INT64_MAX;
    a.t = invalid.data();
    CHECK(kg->launch(&a, stream) < 0);
    if (kg->scratch) {
        a.t = dt.data();
        a.scratch_bytes = 0;
        CHECK_EQ(kg->launch(&a, stream), RAD_E_SCRATCH);
    }
    REQUIRE_OK(rad_memcpy_async(after.data(), alloc[output], after.size(), stream));
    REQUIRE_OK(rad_stream_sync(stream));
    CHECK(before == after);
    if (kg->fini) kg->fini(a.instance);
    for (void* p : alloc)
        rad_dev_free(p, RAD_MEM_DEVICE);
    rad_dev_free(scratch, RAD_MEM_DEVICE);
    rad_stream_destroy(stream);
}
}
TEST(nemotron35_device_position_scale_boundary_and_row_grid_tail) {
    if (!device()) SKIP("explicit R9700 card run required");
    for (int M : {4, 65539}) {
        Buffer x(RAD_BF16, M, 5, 19, 2);
        x.fill(3);
        std::vector<int> p(M);
        for (int i = 0; i < M; ++i)
            p[i] = i % 4 == 0 ? 0 : i % 4 == 1 ? 16383 : i % 4 == 2 ? 16384 : 32767;
        auto pos = Buffer::ids(p);
        compare("position_scale", {&x, pos.get()},
                {pi("M", M + 128), pi("n", 5), pi("original", 16384), pf("beta", 0.1),
                 ps("dtype", "bf16")},
                0);
    }
}
TEST(nemotron35_device_rotary_pitches_active_rows_and_reference_seams) {
    if (!device()) SKIP("explicit R9700 card run required");
    for (bool pix : {false, true})
        for (int D : {64, 128}) {
            const int M = 17, H = 4, K = pix ? 4 : 1;
            Buffer x(RAD_BF16, M, (H + 2 * K) * D, (H + 2 * K) * D * 2 + 13, 2);
            x.fill(44);
            std::vector<int> p(pix ? 2 * M : M);
            for (size_t i = 0; i < p.size(); ++i)
                p[i] = pix          ? (int)(i * 7 % 110)
                       : i % 4 == 0 ? 0
                       : i % 4 == 1 ? 16383
                       : i % 4 == 2 ? 16384
                                    : 32767;
            auto pos = Buffer::ids(p, pix ? 2 : 1);
            std::vector<RadParam> pars = {pi("M", 32),
                                          pi("head_dim", D),
                                          pi("n_head", H),
                                          pi("n_head_kv", K),
                                          pf("theta", pix ? 10000 : 1000000),
                                          ps("dtype", "bf16")};
            if (!pix) {
                pars.push_back(pf("yarn_factor", 16));
                pars.push_back(pf("yarn_low", 1));
                pars.push_back(pf("yarn_high", 27));
                pars.push_back(pf("yarn_magnitude", 1));
            }
            compare(pix ? "pixtral_rope" : "ministral_rope", {&x, pos.get()}, pars, 0, D, true);
        }
}
TEST(nemotron35_device_merger_ragged_rectangular_images) {
    if (!device()) SKIP("explicit R9700 card run required");
    const int M = 32, N = 37;
    Buffer x(RAD_BF16, M, N, 91, 2), y(RAD_BF16, M / 4, N * 4, 307, 2);
    x.fill(91);
    std::vector<int> p(4 * M);
    int off = 0;
    for (auto hw : {std::pair{2, 4}, std::pair{4, 6}}) {
        auto [h, w] = hw;
        for (int j = 0; j < h * w; ++j) {
            p[off + j] = j / w;
            p[M + off + j] = j % w;
            p[2 * M + off + j] = h;
            p[3 * M + off + j] = w;
        }
        off += h * w;
    }
    auto pos = Buffer::ids(p, 4);
    compare("pixtral_merge", {&x, pos.get(), &y},
            {pi("M", 64), pi("n", N), pi("merge", 2), ps("dtype", "bf16")}, 2);
}
TEST(nemotron35_device_attention_ragged_pitches_and_partial_tiles) {
    if (!device()) SKIP("explicit R9700 card run required");
    for (int D : {64, 128})
        for (int G : {1, 3, 4})
            for (int M : {1, 17, 73, 129}) {
                const int H = G * 2;
                Buffer q(RAD_BF16, M, H * D, H * D + 11), k(RAD_BF16, M, 2 * D, 2 * D + 7),
                    v(RAD_BF16, M, 2 * D, 2 * D + 19), o(RAD_BF16, M, H * D, H * D + 5);
                q.fill(17);
                k.fill(35);
                v.fill(83);
                std::vector<int> cu =
                    M > 17 ? std::vector<int>{0, 1, 17, M} : std::vector<int>{0, M};
                auto c = Buffer::ids(cu);
                compare("embed_attn", {&q, &k, &v, c.get(), &o},
                        {pi("M", M + 256), pi("head_dim", D), pi("gqa", G), pi("causal", 0),
                         pi("max_seqlen", M), ps("dtype", "bf16")},
                        4, D, false);
            }
}
TEST(nemotron35_device_1b_exact_attention_head_geometry) {
    if (!device()) SKIP("explicit R9700 card run required");
    const int M = 49, D = 128, H = 24, G = 3;
    Buffer q(RAD_BF16, M, H * D, H * D + 13), k(RAD_BF16, M, 8 * D, 8 * D + 7),
        v(RAD_BF16, M, 8 * D, 8 * D + 11), o(RAD_BF16, M, H * D, H * D + 5);
    q.fill(19);
    k.fill(37);
    v.fill(85);
    auto cu = Buffer::ids({0, 1, 18, M});
    compare("embed_attn", {&q, &k, &v, cu.get(), &o},
            {pi("M", M), pi("head_dim", D), pi("gqa", G), pi("causal", 0), pi("max_seqlen", 31),
             ps("dtype", "bf16")},
            4, D, false);
}
TEST(nemotron35_device_combined_tokens_exceed_context_without_splitting_segments) {
    if (!device()) SKIP("explicit R9700 card run required");
    const int M = 32770, D = 64;
    Buffer q(RAD_BF16, M, D, D + 3), k(RAD_BF16, M, D, D + 5), v(RAD_BF16, M, D, D + 7),
        o(RAD_BF16, M, D, D + 11);
    q.fill(32);
    k.fill(53);
    v.fill(81);
    std::vector<int> cu(M + 1);
    for (int i = 0; i <= M; ++i)
        cu[i] = i;
    auto c = Buffer::ids(cu);
    compare("embed_attn", {&q, &k, &v, c.get(), &o},
            {pi("M", 65536), pi("head_dim", D), pi("gqa", 1), pi("causal", 0), pi("max_seqlen", 1),
             ps("dtype", "bf16")},
            4, D, true);
}
TEST(nemotron35_device_patch_projection_588_tail_and_pitched_rows) {
    if (!device()) SKIP("explicit R9700 card run required");
    for (int M : {1, 17, 65})
        for (int N : {37, 64, 1024}) {
            Buffer a(RAD_BF16, M, 588, 599), w(RAD_BF16, N, 588, 607), y(RAD_BF16, M, N, N + 5);
            a.fill(17);
            w.fill(91);
            compare("gemm_nt", {&a, &w, &y},
                    {pi("M", 128), pi("N", N), pi("K", 588), ps("dtype", "bf16")}, 2, 128, true);
        }
}

TEST(nemotron35_device_rerank_pool_bf16_sum_rounding_ragged_strides_and_tails) {
    if (!device())
        SKIP("explicit R9700 card run required");
    for (int N : {7, 37, 257, 4096}) {
        Buffer x(RAD_BF16, 10, N, 2 * N + 3, 2), y(RAD_BF16, 3, N, 2 * N + 7, 2);
        x.fill(3508);
        auto cu = Buffer::ids({0, 3, 9, 10});
        compare("seg_mean_bf16_sum", {&x, cu.get(), &y},
                {pi("M", 64), pi("n", N), ps("dtype", "bf16")}, 2, 0, true);
    }
}

TEST(nemotron35_device_rerank_pool_full_context_and_sequence_grid_boundary) {
    if (!device())
        SKIP("explicit R9700 card run required");
    for (bool many : {false, true}) {
        const int M = many ? 65539 : 32768, S = many ? M : 3, N = 7;
        Buffer x(RAD_BF16, M, N, 11), y(RAD_BF16, S, N, 13);
        x.fill(3509);
        std::vector<int> lengths = {0, 257, 16384, M};
        if (many) {
            lengths.resize(M + 1);
            for (int i = 0; i <= M; ++i)
                lengths[i] = i;
        }
        auto cu = Buffer::ids(lengths);
        compare("seg_mean_bf16_sum", {&x, cu.get(), &y},
                {pi("M", M), pi("n", N), ps("dtype", "bf16")}, 2, 0, true);
    }
}

RAD_TEST_MAIN()
