/* avx_ggml.cpp -- gemm_nt_ggml, moe_gemm_ggml, logits_gemm_ggml and embed_lookup_ggml over
 * llama.cpp's own blocks (libref/ref_ggml.cpp), at this level.
 *
 * The weight reaches the kernel as the native blocks this library's layout hook packed from the
 * type's planes (avx_registry.cpp, the same bytes as libref's hook), [N, K / block * block_bytes]
 * bytes, or as the float [N, K] logical form a tool passes. A native row is dequantised with
 * llama.cpp's own dequantize_row_<type> (libquant's lq_ggml_dequant, compiled once at the baseline
 * as libref compiles it), so every weight value is libref's bit for bit; the products then run on
 * this level's tile driver (avx_gemm_tile.cpp), which asks a row decoder for whole blocks of a row
 * at a time. The sums differ from libref's index-order sum only in their order.
 *
 * Built only when libquant's sources are beside libavx's (AVX_HAVE_GGML), as libref's rows are;
 * without them the ops have no row. */
#include "avx_gemm.h"

#ifdef AVX_HAVE_GGML
#include "lq_ggml_dequant.h"
#include "lq_ggml_planes.h"

#include <vector>

using namespace avx;

namespace {

const lq::ggml::PlaneKind* kind_of(const RadArgs* a) {
    const char* s = p_str(a, "type", nullptr);
    if (!s) return nullptr;
    for (uint32_t t = 0; t < 64; ++t) {
        const lq::ggml::PlaneKind* k = lq::ggml::plane_kind(t);
        if (k && !std::strcmp(k->name, s) && lq::ggml::pack_of(t) && lq::ggml::dequant_of(t))
            return k;
    }
    return nullptr;
}

/* Rows of native blocks, `rb` bytes apart from `base`: whole blocks of a row dequantised to f32. */
struct GgmlDec final : RowDecoder {
    const uint8_t* base = nullptr;
    int64_t rb = 0, block = 0, bbytes = 0, K = 0;
    lq::ggml::DequantFn dq = nullptr;
    void row(int64_t r, float* dst) const override { dq(base + r * rb, dst, K); }
    void cols(int64_t r, int64_t k0, int64_t k1, float* dst) const override {
        dq(base + r * rb + k0 / block * bbytes, dst, k1 - k0);
    }
    int64_t align() const override { return block; }
    void codes(int64_t r, int64_t k0, int64_t k1, const void** p, int64_t* bytes) const override {
        *p = base + r * rb + k0 / block * bbytes;
        *bytes = (k1 - k0) / block * bbytes;
    }
};

GgmlDec make_dec(const lq::ggml::PlaneKind* k, const void* base, int64_t K) {
    GgmlDec d;
    d.base = (const uint8_t*)base;
    d.block = k->block;
    d.bbytes = (int64_t)k->block_bytes;
    d.rb = K / k->block * d.bbytes;
    d.K = K;
    d.dq = lq::ggml::dequant_of(k->type);
    return d;
}

/* One row of the weight as f32: dequantised, or read from the logical form. */
void weight_row(const RadTensor& w, const GgmlDec* dec, int64_t n, int64_t K, float* dst) {
    if (dec) { dec->row(n, dst); return; }
    const float* r = in_row(&w, rowoff(&w, n, K), K, dst);
    if (r != dst) std::memcpy(dst, r, (size_t)K * sizeof(float));
}

/* Y[m, n] = A[m] . W[n] for every (m, n): the tile driver, or, if it declines, a dot per output
 * (each weight row decoded once). */
void product(const Mat& ma, const Mat& mb, const RadTensor& w, const GgmlDec* dec,
             const RadTensor* Y, int64_t M, int64_t N, int64_t K) {
    if (gemm_tile_f32(ma, mb, nullptr, nullptr, Y, nullptr, M, N, K)) return;
    float* apack = (float*)scratch_raw((size_t)(M * K) * sizeof(float) + 64);
    for (int64_t m = 0; m < M; ++m) widen_cols(ma, m, 0, K, apack + m * K);
    const int64_t ys_r = row_stride(Y, N), ys_c = laststride(Y);
    AVX_PARALLEL_FOR_IF(M * N * K >= AVX_PAR_MIN)
    for (int64_t n = 0; n < N; ++n) {
        float* wr = scratch_f32((size_t)K);
        weight_row(w, dec, n, K, wr);
        for (int64_t m = 0; m < M; ++m) stt(Y, m * ys_r + n * ys_c, dotf(apack + m * K, wr, K));
    }
}

/* gemm_nt_ggml with N and K under their own names (logits_gemm_ggml's n_vocab and n_embd). */
int gemm_ggml_impl(const RadArgs* a, const char* nN, const char* nK) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &w = a->t[1], &y = a->t[2];
    const lq::ggml::PlaneKind* k = kind_of(a);
    const int64_t M = band_rows(x.rank >= 1 ? x.shape[0] : 0, p_int(a, "M", 0));
    const int64_t N = p_int(a, nN, 0), K = p_int(a, nK, 0);
    if (!k || M <= 0 || N <= 0 || K <= 0 || K % k->block || x.rank != 2 || x.shape[1] != K ||
        y.rank != 2 || y.shape[1] != N || y.shape[0] < M || w.rank != 2 || w.shape[0] != N)
        return RAD_E_SHAPE;
    const int64_t rb = K / k->block * (int64_t)k->block_bytes;
    const bool native = w.dtype == RAD_U8;
    if (native ? (w.shape[1] != rb && w.shape[1] != K) : w.shape[1] != K) return RAD_E_SHAPE;
    /* The row stride of native blocks is computed, never read: the engine hands a relayouted
     * weight at its declared shape [N, K] (libref's reading). */
    const GgmlDec dec = make_dec(k, w.data, K);
    const Mat ma = as_mat(&x, K);
    const Mat mb = native ? Mat{ &w, K, 1, &dec } : as_mat(&w, K);
    product(ma, mb, w, native ? &dec : nullptr, &y, M, N, K);
    return RAD_OK;
}

}  // namespace

AVX_KERNEL(gemm_nt_ggml) { return gemm_ggml_impl(a, "N", "K"); }

/* logits_gemm over a head in llama.cpp's blocks: gemm_nt_ggml with N and K spelled n_vocab and
 * n_embd, the logits f32. */
AVX_KERNEL(logits_gemm_ggml) { return gemm_ggml_impl(a, "n_vocab", "n_embd"); }

/* embed_lookup from a table in llama.cpp's blocks (libref's embed_lookup_ggml): the id's row
 * dequantised, times `mult`, rounded once; a padded id or one outside the rank's slice is a zero
 * row, and on a rank holding the whole vocabulary an id at or past n_vocab refuses the call. */
AVX_KERNEL(embed_lookup_ggml) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor &tok = a->t[0], &w = a->t[1], &x = a->t[2];
    const lq::ggml::PlaneKind* k = kind_of(a);
    const int64_t E = p_int(a, "n_embd", 0), V = p_int(a, "n_vocab", 0);
    const int64_t voff = p_int(a, "vocab_offset", 0);
    const float mult = p_f32(a, "mult", 1.0f);
    if (!k || E <= 0 || V <= 0 || voff < 0 || E % k->block || w.rank != 2 || w.shape[0] < V)
        return RAD_E_SHAPE;
    const int64_t rb = E / k->block * (int64_t)k->block_bytes;
    const bool native = w.dtype == RAD_U8;
    if (native ? (w.shape[1] != rb && w.shape[1] != E) : w.shape[1] != E) return RAD_E_SHAPE;
    const int64_t M = numel(&x) / E;
    if (M <= 0 || numel(&tok) < M) return RAD_E_SHAPE;
    if (voff == 0 && a->world_size <= 1)
        for (int64_t m = 0; m < M; ++m)
            if (ldi(&tok, offlin(&tok, m)) >= V) return RAD_E_INVAL;
    const GgmlDec dec = make_dec(k, w.data, E);
    const vf vm = vf_set1(mult);
    AVX_PARALLEL_FOR_IF(M * E >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        float* row = scratch_f32((size_t)E);
        const int64_t id = ldi(&tok, offlin(&tok, m)) - voff;
        if (id < 0 || id >= V) {
            std::memset(row, 0, (size_t)E * sizeof(float));
        } else {
            weight_row(w, native ? &dec : nullptr, id, E, row);
            int64_t i = 0;
            for (; i + VF_N <= E; i += VF_N) vf_storeu(row + i, vf_mul(vf_loadu(row + i), vm));
            for (; i < E; ++i) row[i] = row[i] * mult;
        }
        out_row(&x, rowoff(&x, m, E), E, row);
    }
    return RAD_OK;
}

namespace {

/* The s with off[s] <= i < off[s+1], as libref's binary search over the expert offsets. */
int64_t expert_of(const RadTensor& off, int64_t ne, int64_t i) {
    int64_t lo = 0, hi = ne;
    while (lo + 1 < hi) {
        const int64_t mid = (lo + hi) / 2;
        if (ldi(&off, offlin(&off, mid)) <= i) lo = mid; else hi = mid;
    }
    return lo;
}

/* Expert e of a RAD_OPD_WTAB operand (avx_moe.cpp's reading): an entry of the pointer table, or a
 * slab of a stacked tensor. */
RadTensor expert_entry(const RadTensor* w, int64_t e) {
    RadTensor t{};
    if (!w || !w->data || w->rank < 2 || e < 0 || e >= w->shape[0]) return t;
    t.dtype = w->dtype;
    t.rank = w->rank - 1;
    for (uint32_t d = 0; d + 1 < w->rank; ++d) {
        t.shape[d] = w->shape[d + 1];
        t.stride[d] = w->stride[d + 1];
    }
    if (w->stride[0] == 0) {
        t.data = ((void* const*)w->data)[e];
    } else {
        const int64_t bits = rad_dtype_bits(w->dtype);
        const int64_t nbits = bits * w->stride[0] * e;
        if (bits <= 0 || (nbits & 7)) return t;
        t.data = (void*)((const char*)w->data + (nbits >> 3));
    }
    return t;
}

}  // namespace

/* moe_gemm over llama.cpp's blocks (libref's moe_gemm_ggml): y[i] = a[row(i)] . w[e]^T in sorted
 * order, e the expert whose offset range holds i, row(i) per `a_order`; a dropped slot or a token
 * past `a` is a row of zeros. Each run of an expert's rows is one job of the tile driver, the
 * expert's weight read through its own row decoder. */
AVX_KERNEL(moe_gemm_ggml) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor &A = a->t[0], &W = a->t[1], &sorted = a->t[2], &off = a->t[3], &Y = a->t[4];
    const lq::ggml::PlaneKind* k = kind_of(a);
    const int64_t N = p_int(a, "N", 0), K = p_int(a, "K", 0);
    const int64_t ne = p_int(a, "n_expert", 0), top_k = p_int(a, "top_k", 0);
    const char* ord = p_str(a, "a_order", "token");
    const bool sorted_a = ord && !std::strcmp(ord, "sorted");
    if (ord && !sorted_a && std::strcmp(ord, "token")) return RAD_E_UNSUPPORTED;
    if (!k || N <= 0 || K <= 0 || K % k->block || ne <= 0 || top_k <= 0 || A.rank != 2 ||
        A.shape[1] != K || Y.rank != 2 || Y.shape[1] != N || W.rank != 3 || W.shape[0] < ne ||
        W.shape[1] != N)
        return RAD_E_SHAPE;
    const int64_t T = Y.shape[0], AR = A.shape[0];
    if (T <= 0 || numel(&sorted) < T || numel(&off) < ne + 1) return RAD_E_SHAPE;
    const int64_t rb = K / k->block * (int64_t)k->block_bytes;
    const bool native = W.dtype == RAD_U8;
    if (native ? (W.shape[2] != rb && W.shape[2] != K) : W.shape[2] != K) return RAD_E_SHAPE;
    /* Each expert's weight: a table entry, or a slab of a stacked tensor (native rows rb bytes
     * apart, so a stacked native expert is N * rb bytes on from the last). */
    std::vector<RadTensor> we((size_t)ne);
    std::vector<GgmlDec> dec((size_t)ne);
    std::vector<Mat> mw((size_t)ne);
    for (int64_t e = 0; e < ne; ++e) {
        we[(size_t)e] = expert_entry(&W, e);
        if (native && W.stride[0] != 0) we[(size_t)e].data = (uint8_t*)W.data + e * N * rb;
        if (!we[(size_t)e].data) return RAD_E_STRIDE;
    }
    for (int64_t e = 0; e < ne; ++e) {
        dec[(size_t)e] = make_dec(k, we[(size_t)e].data, K);
        mw[(size_t)e] = native ? Mat{ &we[(size_t)e], K, 1, &dec[(size_t)e] }
            : as_mat(&we[(size_t)e], K);
    }
    /* Each row's activation row (-1: zeros) and expert. */
    std::vector<int64_t> atok((size_t)T), aexp((size_t)T);
    for (int64_t i = 0; i < T; ++i) {
        const int64_t flat = ldi(&sorted, offlin(&sorted, i));
        aexp[(size_t)i] = expert_of(off, ne, i);
        const int64_t tok = flat < 0 ? -1 : (sorted_a ? i : flat / top_k);
        atok[(size_t)i] = tok >= 0 && tok < AR ? tok : -1;
    }
    const Mat ma = as_mat(&A, K);
    std::vector<TileJob> jobs;
    for (int64_t i0 = 0; i0 < T;) {
        int64_t i1 = i0 + 1;
        while (i1 < T && aexp[(size_t)i1] == aexp[(size_t)i0]) ++i1;
        TileJob jb{};
        jb.mb = mw[(size_t)aexp[(size_t)i0]];
        jb.a0 = i0; jb.M = i1 - i0; jb.N = N; jb.K = K;
        jb.y0 = i0; jb.yc0 = 0;
        jobs.push_back(jb);
        i0 = i1;
    }
    const int64_t y_rs = row_stride(&Y, N), y_cs = laststride(&Y);
    if (!gemm_tile_f32_jobs(ma, atok.data(), T, nullptr, jobs.data(), (int64_t)jobs.size(), &Y, N)) {
        /* The driver declined: a dot per output, each weight row decoded once per expert run. */
        float* apack = (float*)scratch_raw((size_t)(T * K) * sizeof(float) + 64);
        for (int64_t i = 0; i < T; ++i)
            if (atok[(size_t)i] >= 0) widen_cols(ma, atok[(size_t)i], 0, K, apack + i * K);
        AVX_PARALLEL_FOR_IF(T * N * K >= AVX_PAR_MIN)
        for (int64_t n = 0; n < N; ++n) {
            float* wr = scratch_f32((size_t)K);
            int64_t have = -1;
            for (int64_t i = 0; i < T; ++i) {
                if (atok[(size_t)i] < 0) continue;
                const int64_t e = aexp[(size_t)i];
                if (e != have) {
                    weight_row(we[(size_t)e], native ? &dec[(size_t)e] : nullptr, n, K, wr);
                    have = e;
                }
                stt(&Y, i * y_rs + n * y_cs, dotf(apack + i * K, wr, K));
            }
        }
    }
    /* A dropped row is exactly zero, as libref writes it, whatever the weight holds. */
    for (int64_t i = 0; i < T; ++i)
        if (atok[(size_t)i] < 0)
            for (int64_t n = 0; n < N; ++n) stt(&Y, i * y_rs + n * y_cs, 0.0f);
    return RAD_OK;
}

#else  /* !AVX_HAVE_GGML: the ops have no row (avx_registry.cpp), so these never run. */

AVX_KERNEL(gemm_nt_ggml)      { (void)a; return RAD_E_UNSUPPORTED; }
AVX_KERNEL(logits_gemm_ggml)  { (void)a; return RAD_E_UNSUPPORTED; }
AVX_KERNEL(embed_lookup_ggml) { (void)a; return RAD_E_UNSUPPORTED; }
AVX_KERNEL(moe_gemm_ggml)     { (void)a; return RAD_E_UNSUPPORTED; }

#endif /* AVX_HAVE_GGML */
