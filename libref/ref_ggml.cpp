/* ref_ggml.cpp -- gemm_nt_ggml, moe_gemm_ggml, logits_gemm_ggml and embed_lookup_ggml: linears,
 * grouped expert GEMMs, the head and the embedding over llama.cpp's own blocks, read at native
 * density.
 *
 * The weight is stored in the container as the affine planes of its GGUF type
 * (libquant/lq_ggml_planes: every block field in a plane, exactly). This row's layout hook asks for
 * all of the type's planes and relayouts them back into the type's native blocks with pack_of,
 * [N, K / block * block_bytes] bytes, so the card or host holds GGUF density (IQ1_S's planes are a
 * third larger than its blocks); the unrelayout unpacks them again. `type` names the GGUF type as
 * llama.cpp spells it ("iq1_s", "q6_K", ...).
 *
 * The arithmetic is the reference every faster row is checked against: each weight row is
 * dequantised to f32 with llama.cpp's own dequantize_row_<type> (lq_ggml_dequant, equal bit for bit
 * to the core's decode of the planes), dotted with the activation row in f32 in index order, and
 * rounded once to `dtype`. A plain float weight [N, K] (the logical form the op oracle and
 * rad-kbench pass) is read as it is, so both readings go through one dot.
 *
 * Built only when libquant's sources are beside libref's (REF_HAVE_GGML): an out-of-tree libref
 * has no ggml row.
 */
#include "ref_common.h"
#include "ref_gemm.h"
#include "ref_ops.h"

#ifdef REF_HAVE_GGML
#include "lq_ggml_dequant.h"
#include "lq_ggml_planes.h"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace ref;

namespace {

const lq::ggml::PlaneKind* kind_named(const char* s) {
    if (!s) return nullptr;
    for (uint32_t t = 0; t < 64; ++t) {
        const lq::ggml::PlaneKind* k = lq::ggml::plane_kind(t);
        if (k && !std::strcmp(k->name, s) && lq::ggml::pack_of(t) && lq::ggml::dequant_of(t))
            return k;
    }
    return nullptr;
}

int64_t pint(const RadParam* p, int n_p, const char* key) {
    const RadParam* q = rad_param_find(p, n_p, key);
    if (!q) return 0;
    if (q->kind == RAD_P_RANGE) return (int64_t)q->ihi;
    return q->kind == RAD_P_INT ? (int64_t)q->ival : 0;
}

const lq::ggml::PlaneKind* kind_of(const RadParam* p, int n_p) {
    const RadParam* t = rad_param_find(p, n_p, "type");
    return kind_named(t && t->kind == RAD_P_STR ? t->sval : nullptr);
}

/* The selection must be exactly the type's encoding, every plane in order. */
int ggml_check(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
               int n_planes, const lq::ggml::PlaneKind** out) {
    const lq::ggml::PlaneKind* k = kind_of(p, n_p);
    if (operand != 1 || !k || !enc || !sel) return RAD_E_DTYPE;
    const RadEncoding want = lq::ggml::encoding_of(k->type);
    if (!rad_enc_equal(enc, &want) || n_planes != want.n_planes) return RAD_E_DTYPE;
    for (int i = 0; i < n_planes; ++i)
        if (sel[i] != i) return RAD_E_SHAPE;
    *out = k;
    return RAD_OK;
}

/* Plane i's row r, or null for a TABLE or per-tensor plane (pack and unpack do not touch those). */
uint8_t* plane_row(const RadEncoding* enc, const RadTensor* planes, int i, int64_t r) {
    const RadEncPlane& pl = enc->plane[i];
    if (pl.kind == RAD_PLANE_TABLE || (pl.block[0] == 0 && pl.block[1] == 0)) return nullptr;
    const int64_t cols = planes[i].rank >= 2 ? planes[i].shape[1] : 0;
    return (uint8_t*)planes[i].data + r * rad_enc_row_bytes(pl.dtype, cols);
}

int64_t logical_rows(const RadTensor* planes) {
    return planes[0].rank >= 1 ? planes[0].shape[0] : 0;
}

/* The values a weight row holds: `K` for the GEMMs, `n_embd` for the vocabulary ops. */
int64_t row_len(const RadParam* p, int n_p) {
    const int64_t K = pint(p, n_p, "K");
    return K > 0 ? K : pint(p, n_p, "n_embd");
}

}  // namespace

extern "C" int ref_ggml_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                               const int* sel, const RadTensor* planes, int n_planes,
                                   RadLayout* out) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const lq::ggml::PlaneKind* k = nullptr;
    const int rc = ggml_check(p, n_p, operand, enc, sel, n_planes, &k);
    if (rc < 0) return rc;
    const int64_t K = row_len(p, n_p), rows = logical_rows(planes);
    if (K <= 0 || K % k->block || rows <= 0) return RAD_E_SHAPE;
    static thread_local char tag[32];
    snprintf(tag, sizeof tag, "ggml_%s", k->name);
    std::memset(out, 0, sizeof *out);
    out->tag = tag;
    out->dtype = RAD_U8;
    out->rank = 2;
    out->shape[0] = rows;
    out->shape[1] = K / k->block * (int64_t)k->block_bytes;
    out->align = 64;
    out->bytes = out->shape[0] * out->shape[1];
    return RAD_OK;
}

extern "C" int ref_ggml_relayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                 const int* sel, const RadTensor* planes, int n_planes, void* dst,
                                 int64_t dst_bytes) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const lq::ggml::PlaneKind* k = nullptr;
    const int rc = ggml_check(p, n_p, operand, enc, sel, n_planes, &k);
    if (rc < 0) return rc;
    const int64_t K = row_len(p, n_p), rows = logical_rows(planes);
    const int64_t nb = K / k->block, rb = nb * (int64_t)k->block_bytes;
    if (K <= 0 || K % k->block || rows <= 0 || dst_bytes < rows * rb) return RAD_E_SHAPE;
    const lq::ggml::PackFn pack = lq::ggml::pack_of(k->type);
    std::vector<const uint8_t*> row((size_t)n_planes);
    for (int64_t r = 0; r < rows; ++r) {
        for (int i = 0; i < n_planes; ++i) row[(size_t)i] = plane_row(enc, planes, i, r);
        pack(row.data(), nb, (uint8_t*)dst + r * rb);
    }
    return RAD_OK;
}

extern "C" int ref_ggml_unrelayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                   const int* sel, const void* src, int64_t src_bytes,
                                   const RadTensor* planes, int n_planes) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const lq::ggml::PlaneKind* k = nullptr;
    const int rc = ggml_check(p, n_p, operand, enc, sel, n_planes, &k);
    if (rc < 0) return rc;
    const int64_t K = row_len(p, n_p), rows = logical_rows(planes);
    const int64_t nb = K / k->block, rb = nb * (int64_t)k->block_bytes;
    if (K <= 0 || K % k->block || rows <= 0 || src_bytes < rows * rb) return RAD_E_SHAPE;
    std::vector<uint8_t*> row((size_t)n_planes);
    std::vector<void*> all((size_t)n_planes);
    for (int i = 0; i < n_planes; ++i) all[(size_t)i] = planes[i].data;
    for (int64_t r = 0; r < rows; ++r) {
        for (int i = 0; i < n_planes; ++i) row[(size_t)i] = plane_row(enc, planes, i, r);
        k->unpack((const uint8_t*)src + r * rb, nb, row.data());
    }
    lq::ggml::write_shared(k->type, *enc, all.data());
    return RAD_OK;
}

/* y[m, n] = sum_k a[m, k] w[n, k]: w native blocks dequantised a row at a time, or a plain float
 * [N, K]; f32 accumulation in index order, rounded once.
 *
 * Native blocks are u8, each row K / block * block_bytes bytes, rows back to back as the relayout
 * wrote them. The engine hands a relayouted weight at its stored dtype and its DECLARED shape,
 * [N, K] (core/runtime/ctx.cpp: a re-laying kernel derives its own stride), so the row stride is
 * computed here, never read from the operand; [N, row bytes] (a tool holding the stored bytes) is
 * read the same way. */
static int gemm_ggml_impl(const RadArgs* a, const char* nN, const char* nK) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &w = a->t[1], &y = a->t[2];
    const lq::ggml::PlaneKind* k = kind_of(a->p, a->n_p);
    const int64_t M = band_rows(x.rank >= 1 ? x.shape[0] : 0, p_int(a, "M", 0));
    const int64_t N = p_int(a, nN, 0), K = p_int(a, nK, 0);
    if (!k || M <= 0 || N <= 0 || K <= 0 || K % k->block || x.rank != 2 || x.shape[1] != K ||
        y.rank != 2 || y.shape[1] != N || y.shape[0] < M || w.rank != 2 || w.shape[0] != N)
        return RAD_E_SHAPE;
    const int64_t rb = K / k->block * (int64_t)k->block_bytes;
    const bool native = w.dtype == RAD_U8;
    if (native ? (w.shape[1] != rb && w.shape[1] != K) : w.shape[1] != K) return RAD_E_SHAPE;
    const lq::ggml::DequantFn dq = lq::ggml::dequant_of(k->type);
#pragma omp parallel
    {
        std::vector<float> wr((size_t)K);
#pragma omp for schedule(static)
        for (int64_t n = 0; n < N; ++n) {
            if (native)
                dq((const uint8_t*)w.data + n * rb, wr.data(), K);
            else
                for (int64_t i = 0; i < K; ++i)
                    wr[(size_t)i] = ldt(&w, rowoff(&w, n, K) + i * laststride(&w));
            for (int64_t m = 0; m < M; ++m) {
                const int64_t xb = rowoff(&x, m, K), xs = laststride(&x);
                float acc = 0.0f;
                for (int64_t i = 0; i < K; ++i) acc += ldt(&x, xb + i * xs) * wr[(size_t)i];
                stt(&y, rowoff(&y, m, N) + n * laststride(&y), acc);
            }
        }
    }
    return RAD_OK;
}

int ref_gemm_nt_ggml(const RadArgs* a, RadStream s) { (void)s; return gemm_ggml_impl(a, "N", "K"); }

/* logits_gemm over a head in llama.cpp's blocks: gemm_nt_ggml's arithmetic with N and K spelled
 * n_vocab and n_embd, the logits f32 (the head is its own access class, so its own op). */
int ref_logits_gemm_ggml(const RadArgs* a, RadStream s) {
    (void)s;
    return gemm_ggml_impl(a, "n_vocab", "n_embd");
}

/* embed_lookup from a table in llama.cpp's blocks: the token's row dequantised as llama.cpp does,
 * times `mult` when given, rounded once. Ids, padding and a vocab-parallel slice behave as
 * embed_lookup's: a negative id or one outside [vocab_offset, vocab_offset + n_vocab) writes a
 * zero row, and on a rank holding the whole vocabulary an id at or above n_vocab fails. */
int ref_embed_lookup_ggml(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor &tok = a->t[0], &w = a->t[1], &x = a->t[2];
    const lq::ggml::PlaneKind* k = kind_of(a->p, a->n_p);
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
            if (ld_int(tok.dtype, tok.data, offlin(&tok, m)) >= V) return RAD_E_INVAL;
    const lq::ggml::DequantFn dq = lq::ggml::dequant_of(k->type);
#pragma omp parallel
    {
        std::vector<float> wr((size_t)E);
#pragma omp for schedule(static)
        for (int64_t m = 0; m < M; ++m) {
            const int64_t id = ld_int(tok.dtype, tok.data, offlin(&tok, m)) - voff;
            const int64_t xb = rowoff(&x, m, E), xs = laststride(&x);
            if (id < 0 || id >= V) {
                for (int64_t i = 0; i < E; ++i) stt(&x, xb + i * xs, 0.0f);
                continue;
            }
            if (native)
                dq((const uint8_t*)w.data + id * rb, wr.data(), E);
            else
                for (int64_t i = 0; i < E; ++i)
                    wr[(size_t)i] = ldt(&w, rowoff(&w, id, E) + i * laststride(&w));
            for (int64_t i = 0; i < E; ++i) stt(&x, xb + i * xs, wr[(size_t)i] * mult);
        }
    }
    return RAD_OK;
}

/* moe_gemm over llama.cpp's blocks: y[i] = a[row(i)] . w[e]^T in sorted order, e the expert whose
 * offset range holds i, row(i) as moe_gemm's `a_order` says ("token": a[sorted_tok[i] / top_k];
 * "sorted": a[i]); a dropped slot (sorted_tok < 0) or a token past `a` is a row of zeros. `w` is a
 * table of n_expert entries, each [N, K] read as gemm_nt_ggml reads its weight (native blocks of
 * `type`, or the float logical form), so every output equals gemm_nt_ggml's for that row and
 * expert bit for bit. Each expert's rows are dequantised once a call, only for an expert with a
 * live row. */
int ref_moe_gemm_ggml(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const RadTensor &A = a->t[0], &W = a->t[1], &sorted = a->t[2], &off = a->t[3], &Y = a->t[4];
    const lq::ggml::PlaneKind* k = kind_of(a->p, a->n_p);
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
    /* Each expert's weight: a table entry (the engine's form, one pointer per expert), or a slab of
     * a stacked tensor. Native rows are rb bytes apart (gemm_nt_ggml), so a stacked native
     * expert is N * rb bytes on from the last. */
    std::vector<RadTensor> we((size_t)ne);
    for (int64_t e = 0; e < ne; ++e) {
        we[(size_t)e] = expert_entry(&W, e);
        if (native && W.stride[0] != 0) we[(size_t)e].data = (uint8_t*)W.data + e * N * rb;
        if (!we[(size_t)e].data) return RAD_E_STRIDE;
    }
    /* Each row's expert and activation row, or -1 for a row of zeros; then the live rows grouped
     * by expert in ascending row order. A live row outside [off[0], off[n_expert]) (malformed
     * routing) belongs to no expert's run and is left as it is, as the device rows leave it. */
    std::vector<int64_t> ex((size_t)T, -1), tk((size_t)T, 0);
    std::vector<bool> outside((size_t)T, false);
    const int64_t run_lo = ld_int(off.dtype, off.data, offlin(&off, 0));
    const int64_t run_hi = ld_int(off.dtype, off.data, offlin(&off, ne));
    for (int64_t i = 0; i < T; ++i) {
        const int64_t flat = ld_int(sorted.dtype, sorted.data, offlin(&sorted, i));
        if (flat < 0) continue;
        if (i < run_lo || i >= run_hi) { outside[(size_t)i] = true; continue; }
        int64_t lo = 0, hi = ne;
        while (lo + 1 < hi) {
            const int64_t mid = (lo + hi) / 2;
            if (ld_int(off.dtype, off.data, offlin(&off, mid)) <= i) lo = mid; else hi = mid;
        }
        const int64_t tok = sorted_a ? i : flat / top_k;
        if (tok < AR) { ex[(size_t)i] = lo; tk[(size_t)i] = tok; }
    }
    std::vector<std::vector<int64_t>> rows((size_t)ne);
    for (int64_t i = 0; i < T; ++i) {
        if (ex[(size_t)i] >= 0) rows[(size_t)ex[(size_t)i]].push_back(i);
        else if (!outside[(size_t)i])
            for (int64_t n = 0; n < N; ++n) stt(&Y, rowoff(&Y, i, N) + n * laststride(&Y), 0.0f);
    }
    /* One item per (expert, weight row): the row dequantised, dotted with each of its rows. */
    std::vector<std::pair<int64_t, int64_t>> items;
    for (int64_t e = 0; e < ne; ++e)
        if (!rows[(size_t)e].empty())
            for (int64_t n = 0; n < N; ++n) items.push_back({ e, n });
    const lq::ggml::DequantFn dq = lq::ggml::dequant_of(k->type);
#pragma omp parallel
    {
        std::vector<float> wr((size_t)K);
#pragma omp for schedule(dynamic, 16)
        for (int64_t it = 0; it < (int64_t)items.size(); ++it) {
            const int64_t e = items[(size_t)it].first, n = items[(size_t)it].second;
            const RadTensor& w = we[(size_t)e];
            if (native)
                dq((const uint8_t*)w.data + n * rb, wr.data(), K);
            else
                for (int64_t c = 0; c < K; ++c)
                    wr[(size_t)c] = ldt(&w, rowoff(&w, n, K) + c * laststride(&w));
            for (int64_t i : rows[(size_t)e]) {
                const int64_t xb = rowoff(&A, tk[(size_t)i], K), xs = laststride(&A);
                float acc = 0.0f;
                for (int64_t c = 0; c < K; ++c) acc += ldt(&A, xb + c * xs) * wr[(size_t)c];
                stt(&Y, rowoff(&Y, i, N) + n * laststride(&Y), acc);
            }
        }
    }
    return RAD_OK;
}

/* Values one block of GGUF type `type` covers, or 0 for a type this row does not read. */
extern "C" int64_t ref_ggml_block(const char* type) {
    const lq::ggml::PlaneKind* k = kind_named(type);
    return k ? k->block : 0;
}

#endif /* REF_HAVE_GGML */
