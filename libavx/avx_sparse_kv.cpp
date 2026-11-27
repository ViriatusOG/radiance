/* avx_sparse_kv.cpp -- DeepSeek-V4.1's compressed-KV ops, libref/ref_sparse_kv.cpp at this level:
 *
 *   quant_roundtrip    the block-scaled quantise-dequantise round trip of a cached key
 *   softmax_pool       `ratio` rows folded into one compressed entry by a channel-wise softmax
 *   attn_sparse_sink   latent attention over a window and the selected compressed entries
 *   index_score        the indexer's relu-gated head sum over a paged index cache
 *   index_candidates   level one of the two-level top-k: the kept blocks
 *   index_topk         level two: the selected positions, in position order
 *   entry_store        one compressed entry a paged block
 *   token_map          ids through an id table (Engram's compressed vocabulary)
 *   stream_gate_add    Engram's normalised-dot gate into the hc residual streams
 *
 * Each kernel computes libref's expression. Where libref's result is exact (a selection's ids, the
 * round trip's codes and scales) this one is the same bits; elsewhere it differs from libref only
 * in summation order: rows are widened to f32 once and reduced with this level's vector dot.
 * Parallel regions are over rows and every output has one owner, so the bytes do not depend on the
 * team size. */
#include "avx_vec.h"
#include "avx_common.h"
#include "rad_kv_fp8.h"

#include <algorithm>
#include <vector>

using namespace avx;

namespace {

/* ================================================================== the round trip
 * libref's block_scale and roundtrip, value for value (ref_sparse_kv.cpp). */
int ceil_log2_bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return (int)((u >> 23) & 0xffu) - 127 + ((u & 0x7fffffu) ? 1 : 0);
}
float pow2_bits(int e) {
    const uint32_t u = (uint32_t)(e + 127) << 23;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

enum class QFmt { Fp8E4m3, Fp4E2m1 };
enum class QScale { Ue8m0, E4m3, F32 };

float block_scale(float amax, QFmt f, QScale s) {
    if (f == QFmt::Fp8E4m3) {
        amax = std::max(amax, 1e-4f);
        const float inv = 1.0f / 448.0f;
        return s == QScale::Ue8m0 ? pow2_bits(ceil_log2_bits(amax * inv)) : amax * inv;
    }
    if (s == QScale::E4m3) {
        amax = std::max(amax, 6.0f * 0x1p-9f);
        return avx_fp8e4m3_tab[rad_f32_to_fp8e4m3(amax / 6.0f)];
    }
    amax = std::max(amax, 6.0f * 0x1p-126f);
    const float inv = 1.0f / 6.0f;
    return s == QScale::Ue8m0 ? pow2_bits(ceil_log2_bits(amax * inv)) : amax * inv;
}

/* max over |v[i]| as libref's std::max(amax, fabs(v)): a NaN never replaces the running maximum.
 * vf_max(x, acc) is (x > acc) ? x : acc lane by lane, which keeps acc for a NaN x, and the
 * maximum of the lanes is the same whatever the order. */
float block_amax(const float* v, int64_t n) {
    vf acc = vf_zero();
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N) acc = vf_max(vf_abs(vf_loadu(v + i)), acc);
    float m = vf_hmax(acc);
    for (; i < n; ++i) m = std::max(m, std::fabs(v[i]));
    return m;
}

/* v / scale clamped to [-lim, lim] with a NaN kept (std::clamp's), into q. */
void scaled_clamp(const float* v, float scale, float lim, float* q, int64_t n) {
    const vf vs = vf_set1(scale), lo = vf_set1(-lim), hi = vf_set1(lim);
    int64_t i = 0;
    /* vf_max(lo, x) is x for a NaN x, and vf_min(hi, y) is y for a NaN y. */
    for (; i + VF_N <= n; i += VF_N)
        vf_storeu(q + i, vf_min(hi, vf_max(lo, vf_div(vf_loadu(v + i), vs))));
    for (; i < n; ++i) q[i] = std::clamp(v[i] / scale, -lim, lim);
}

const float kE2m1[16] = { 0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                          -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };

/* One block of `n` values in place: the scale from its maximum, each value to a code and back. */
void roundtrip_block(float* v, int64_t n, QFmt f, QScale s, float* q, uint8_t* codes) {
    const float sc = block_scale(block_amax(v, n), f, s);
    if (f == QFmt::Fp8E4m3) {
        scaled_clamp(v, sc, 448.0f, q, n);
        cvt_f32_to_fp8e4m3(q, codes, n);
        cvt_fp8e4m3_to_f32(codes, q, n);
    } else {
        scaled_clamp(v, sc, 6.0f, q, n);
        for (int64_t i = 0; i < n; ++i) q[i] = kE2m1[e2m1_code(q[i])];
    }
    const vf vs = vf_set1(sc);
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N) vf_storeu(v + i, vf_mul(vf_loadu(q + i), vs));
    for (; i < n; ++i) v[i] = q[i] * sc;
}

}  // namespace

AVX_KERNEL(quant_roundtrip) {
    if (!rad_args_have(a, 1)) return RAD_E_INVAL;
    const RadTensor& x = a->t[0];
    const int64_t n = p_int(a, "n", 0), block = p_int(a, "block", 0);
    const char* fs = p_str(a, "format", "");
    const char* ss = p_str(a, "scale", "");
    QFmt f;
    QScale s;
    if (!std::strcmp(fs, "fp8_e4m3")) f = QFmt::Fp8E4m3;
    else if (!std::strcmp(fs, "fp4_e2m1")) f = QFmt::Fp4E2m1;
    else return RAD_E_UNSUPPORTED;
    if (!std::strcmp(ss, "ue8m0")) s = QScale::Ue8m0;
    else if (!std::strcmp(ss, "e4m3") && f == QFmt::Fp4E2m1) s = QScale::E4m3;
    else if (!std::strcmp(ss, "f32") && f == QFmt::Fp8E4m3) s = QScale::F32;
    else return RAD_E_UNSUPPORTED;
    if (n < 1 || block < 1 || n % block || !x.data || x.rank != 2 || x.shape[0] < 1 ||
        x.shape[1] != n)
        return RAD_E_SHAPE;
    AVX_PARALLEL_FOR_IF(x.shape[0] * n >= AVX_PAR_MIN)
    for (int64_t m = 0; m < x.shape[0]; ++m) {
        float* buf = scratch_f32((size_t)(2 * n + (n + 3) / 4 + 16));
        float* row = buf;
        float* q = buf + n;
        uint8_t* codes = (uint8_t*)(q + n);
        const int64_t xb = rowoff(&x, m, n);
        const float* in = in_row(&x, xb, n, row);
        if (in != row) std::memcpy(row, in, (size_t)n * sizeof(float));
        for (int64_t b0 = 0; b0 < n; b0 += block) roundtrip_block(row + b0, block, f, s, q, codes);
        out_row(&x, xb, n, row);
    }
    return RAD_OK;
}

/* out[w, c] = sum_i softmax_i(score[i, c]) value[i, c] over the entry's `ratio` rows, in f32:
 * the max subtracted, exp, the sum, then each weight p / den times the value, as libref. Vectors
 * run along the columns, so every column keeps libref's order over the rows; only exp differs
 * (this level's vf_exp against libref's expf). */
AVX_KERNEL(softmax_pool) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor &st = a->t[0], &out = a->t[1];
    const RadTensor* page = a->n_t > 2 && a->t[2].data ? &a->t[2] : nullptr;
    const int64_t n = p_int(a, "n", 0), ratio = p_int(a, "ratio", 0);
    if (n < 1 || ratio < 1 || ratio > 64 || !st.data || !out.data || st.rank != 2 ||
        out.rank != 2 || out.shape[0] < 1 || out.shape[1] != n ||
        st.shape[0] != out.shape[0] * ratio || st.shape[1] != 2 * n ||
        (page && numel(page) < out.shape[0]))
        return RAD_E_SHAPE;
    const int64_t W = out.shape[0];
    AVX_PARALLEL_FOR_IF(W * ratio * n >= AVX_PAR_MIN)
    for (int64_t w = 0; w < W; ++w) {
        if (page && ldi(page, offlin(page, w)) < 0) continue;
        /* The entry's rows widened: values then scores, `ratio` rows of 2n. */
        float* buf = scratch_f32((size_t)(ratio * 2 * n + n));
        float* acc = buf + ratio * 2 * n;
        for (int64_t i = 0; i < ratio; ++i) {
            float* r = buf + i * 2 * n;
            const float* in = in_row(&st, rowoff(&st, w * ratio + i, 2 * n), 2 * n, r);
            if (in != r) std::memcpy(r, in, (size_t)(2 * n) * sizeof(float));
        }
        int64_t c = 0;
        for (; c + VF_N <= n; c += VF_N) {
            vf mx = vf_set1(-INFINITY);
            for (int64_t i = 0; i < ratio; ++i) mx = vf_max(vf_loadu(buf + i * 2 * n + n + c), mx);
            vf d = vf_zero();
            for (int64_t i = 0; i < ratio; ++i) {
                float* sp = buf + i * 2 * n + n + c;
                const vf e = vf_exp(vf_sub(vf_loadu(sp), mx));
                vf_storeu(sp, e);
                d = vf_add(d, e);
            }
            vf s = vf_zero();
            for (int64_t i = 0; i < ratio; ++i)
                s = vf_add(s, vf_mul(vf_div(vf_loadu(buf + i * 2 * n + n + c), d),
                                     vf_loadu(buf + i * 2 * n + c)));
            vf_storeu(acc + c, s);
        }
        for (; c < n; ++c) {
            float mx = -INFINITY, d = 0, s = 0;
            for (int64_t i = 0; i < ratio; ++i) mx = std::max(mx, buf[i * 2 * n + n + c]);
            for (int64_t i = 0; i < ratio; ++i) {
                float& sp = buf[i * 2 * n + n + c];
                sp = std::exp(sp - mx);
                d += sp;
            }
            for (int64_t i = 0; i < ratio; ++i)
                s += buf[i * 2 * n + n + c] / d * buf[i * 2 * n + c];
            acc[c] = s;
        }
        out_row(&out, rowoff(&out, w, n), n, acc);
    }
    return RAD_OK;
}

namespace {
/* The s with cu[s] <= row < cu[s+1] (libref's seq_of_row). */
int64_t seq_of_row(const RadTensor& cu, int64_t n_seq, int64_t row) {
    int64_t lo = 0, hi = n_seq;
    while (hi - lo > 1) {
        const int64_t mid = (lo + hi) / 2;
        if (ldi(&cu, offlin(&cu, mid)) <= row) lo = mid;
        else hi = mid;
    }
    return lo;
}

/* `n` elements of t from flat offset `off`, as f32 (a cache entry: dense, libref's flat read). */
void widen_flat(const RadTensor& t, int64_t off, int64_t n, float* dst) {
    row_to_f32(t.dtype, byte_at(t.data, t.dtype, off), dst, n);
}
}  // namespace

/* Latent attention with a sink over a window and the selected compressed entries (libref's
 * attn_sparse_sink, the same keys): each row's keys are found and widened once, then per head the
 * scores are vector dots, p = exp(s - max) as libref (expf), and the output a vector sum of p k. */
AVX_KERNEL(attn_sparse_sink) {
    if (!a || a->n_t < 11 || !a->t) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &wc = a->t[1], &wbt = a->t[2], &wused = a->t[3];
    const RadTensor *cc = a->t[4].data ? &a->t[4] : nullptr, *cbt = a->t[5].data ? &a->t[5]
        : nullptr;
    const RadTensor* sel = a->t[6].data ? &a->t[6] : nullptr;
    const RadTensor &pos = a->t[7], &cu = a->t[8], &sink = a->t[9], &out = a->t[10];
    const int64_t H = p_int(a, "heads", 0), D = p_int(a, "dim", 0), win = p_int(a, "window", 0);
    const int64_t ratio = p_int(a, "ratio", 0), topk = p_int(a, "topk", 0);
    const float scale = p_f32(a, "scale", 0.0f);
    const bool comp = ratio > 0;
    /* latent_format: bf16 or block-scaled E4M3 entries (rad_kv_fp8.h), cslots elements an entry */
    const int lf = rad_kvf_parse(p_str(a, "latent_format", "bf16"));
    if (lf < 0) return RAD_E_UNSUPPORTED;
    const int64_t cslots = lf ? rad_kvf_bytes(lf, D) : D;
    if (H < 1 || D < 2 || D > 4096 || win < 1 || ratio < 0 || topk < 0 || !std::isfinite(scale) ||
        scale <= 0 || (comp && (!cc || !cbt || !sel || topk < 1 || D % ratio || cslots < 1 ||
                                cslots % ratio)))
        return RAD_E_SHAPE;
    if (!q.data || !wc.data || !wbt.data || !wused.data || !pos.data || !cu.data || !sink.data ||
        !out.data || q.rank != 2 || q.shape[0] < 1 || q.shape[1] != H * D || out.rank != 2 ||
        out.shape[0] != q.shape[0] || out.shape[1] != H * D || wc.rank != 4 || wc.shape[1] != 1 ||
        wc.shape[3] != D || wbt.rank != 2 || numel(&pos) != q.shape[0] || numel(&sink) != H ||
        sink.dtype != RAD_F32 || numel(&cu) < 2 || numel(&wused) < numel(&cu) - 1)
        return RAD_E_SHAPE;
    const int64_t cbs = comp ? cc->shape[2] : 0;
    if (comp && (cc->rank != 4 || cc->shape[1] != 1 || cc->shape[3] * ratio != cslots || cbs % ratio ||
                 cbt->rank != 2 || sel->rank != 2 || sel->shape[0] != q.shape[0] ||
                 sel->shape[1] != topk || cbt->shape[0] != wbt.shape[0]))
        return RAD_E_SHAPE;
    if (comp && lf && (cc->dtype != RAD_F8E4M3 || !rad_tensor_is_contiguous(cc)))
        return RAD_E_SHAPE;
    const int64_t M = q.shape[0], S = numel(&cu) - 1, bs = wc.shape[2];
    const int64_t nkeys = win + (comp ? topk : 0);
    const int64_t qs = laststride(&q), os = laststride(&out);
    AVX_PARALLEL_FOR_IF(M * H * nkeys * D >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t s = seq_of_row(cu, S, m);
        const int64_t p = ldi(&pos, offlin(&pos, m));
        const int64_t last = ldi(&pos, offlin(&pos, ldi(&cu, offlin(&cu, s + 1)) - 1));
        const int64_t first = last + 1 - ldi(&wused, offlin(&wused, s));
        /* The row's keys widened, in libref's key order (the window, then the selection); a slot
         * with no key is left out, as every one of libref's sums skips it. */
        float* buf = scratch_f32((size_t)(nkeys * D + 2 * D + nkeys + 16));
        float* keys = buf;
        float* qh = keys + nkeys * D;
        float* acc = qh + D;
        float* sc = acc + D;
        int64_t nk = 0;
        for (int64_t i = 0; i < win; ++i) {
            const int64_t j = p - win + 1 + i, r = j - first;
            if (j < 0 || r < 0 || r / bs >= wbt.shape[1]) continue;
            const int64_t blk = ldi(&wbt, rowoff(&wbt, s, wbt.shape[1]) + (r / bs) * laststride(&wbt));
            if (blk < 0 || blk >= wc.shape[0]) continue;
            widen_flat(wc, ((blk * bs) + r % bs) * D, D, keys + nk * D);
            ++nk;
        }
        for (int64_t t = 0; comp && t < topk; ++t) {
            const int64_t c = ldi(sel, rowoff(sel, m, topk) + t * laststride(sel));
            if (c < 0 || (c * ratio) / cbs >= cbt->shape[1]) continue;
            const int64_t blk =
                ldi(cbt, rowoff(cbt, s, cbt->shape[1]) + ((c * ratio) / cbs) * laststride(cbt));
            if (blk < 0 || blk >= cc->shape[0]) continue;
            const int64_t ko = (blk * cbs + (c * ratio) % cbs) * (cslots / ratio);
            if (lf) {
                /* code x scale, the f32 libref reads */
                const uint8_t* e = (const uint8_t*)cc->data + ko;
                for (int64_t d = 0; d < D; ++d) keys[nk * D + d] = rad_kvf_value(lf, e, D, d);
            } else {
                widen_flat(*cc, ko, D, keys + nk * D);
            }
            ++nk;
        }
        for (int64_t h = 0; h < H; ++h) {
            const int64_t qb = rowoff(&q, m, H * D) + h * D * qs;
            const float* qv = in_row(&q, qb, D, qh);
            float mx = -1e30f;
            for (int64_t k = 0; k < nk; ++k) {
                sc[k] = dotf(qv, keys + k * D, D) * scale;
                mx = std::max(mx, sc[k]);
            }
            float den = 0;
            for (int64_t d = 0; d < D; ++d) acc[d] = 0.0f;
            for (int64_t k = 0; k < nk; ++k) {
                const float pk = std::exp(sc[k] - mx);
                den += pk;
                const float* kv = keys + k * D;
                const vf vp = vf_set1(pk);
                int64_t d = 0;
                for (; d + VF_N <= D; d += VF_N)
                    vf_storeu(acc + d, vf_fma(vp, vf_loadu(kv + d), vf_loadu(acc + d)));
                for (; d < D; ++d) acc[d] += pk * kv[d];
            }
            den += std::exp(ldt(&sink, h * laststride(&sink)) - mx);
            const vf vd = vf_set1(den);
            int64_t d = 0;
            for (; d + VF_N <= D; d += VF_N) vf_storeu(acc + d, vf_div(vf_loadu(acc + d), vd));
            for (; d < D; ++d) acc[d] = acc[d] / den;
            out_row(&out, rowoff(&out, m, H * D) + h * D * os, D, acc);
        }
    }
    return RAD_OK;
}

/* The indexer's score of each compressed position for each row (libref's index_score): per head
 * the dot q_h . k_j rounded to q's dtype, relu(dot) times w_h = round(w * wscale) rounded, the head
 * sum in f32 rounded. The row's queries and head weights are read once; each live position's key
 * is widened once and dotted with every head. With `row_part` p, score holds one run of R rows:
 * score row i is step row p R + i of q, w, positions and cu_seqlens, and a row past the step
 * scores -inf, as in libref. */
AVX_KERNEL(index_score) {
    if (!rad_args_have(a, 7)) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &w = a->t[1], &kc = a->t[2], &bt = a->t[3];
    const RadTensor &pos = a->t[4], &cu = a->t[5], &sc = a->t[6];
    const int64_t H = p_int(a, "heads", 0), Dm = p_int(a, "dim", 0), ratio = p_int(a, "ratio", 0);
    const int64_t part = p_int(a, "row_part", -1);
    const float wscale = p_f32(a, "wscale", 0.0f);
    /* key_format: bf16 or block-scaled E4M3 entries (rad_kv_fp8.h), kslots elements an entry */
    const int kf = rad_kvf_parse(p_str(a, "key_format", "bf16"));
    if (kf < 0) return RAD_E_UNSUPPORTED;
    const int64_t kslots = kf ? rad_kvf_bytes(kf, Dm) : Dm;
    if (H < 1 || Dm < 1 || ratio < 1 || Dm % ratio || kslots < 1 || kslots % ratio ||
        !std::isfinite(wscale) || q.rank != 2 ||
        q.shape[0] < 1 || q.shape[1] != H * Dm || w.rank != 2 || w.shape[0] != q.shape[0] ||
        w.shape[1] != H || kc.rank != 4 || kc.shape[1] != 1 || kc.shape[3] * ratio != kslots ||
        kc.shape[2] % ratio || bt.rank != 2 || numel(&pos) != q.shape[0] || numel(&cu) < 2 ||
        sc.rank != 2 || sc.shape[0] < 1 || (part < 0 && sc.shape[0] != q.shape[0]) ||
        sc.shape[1] < 1 || sc.dtype != RAD_F32)
        return RAD_E_SHAPE;
    if (kf && (kc.dtype != RAD_F8E4M3 || !rad_tensor_is_contiguous(&kc))) return RAD_E_SHAPE;
    const int64_t Mq = q.shape[0], M = sc.shape[0], C = sc.shape[1], S = numel(&cu) - 1;
    const int64_t cbs = kc.shape[2], row0 = part > 0 ? part * M : 0;
    const uint32_t dt = q.dtype;
    const bool exact = dt == RAD_F32;
    AVX_PARALLEL_FOR_IF(M * C * H * Dm >= AVX_PAR_MIN)
    for (int64_t i = 0; i < M; ++i) {
        const int64_t m = row0 + i;
        if (m >= Mq) {
            const int64_t sb = rowoff(&sc, i, C), ss = laststride(&sc);
            for (int64_t j = 0; j < C; ++j) stt(&sc, sb + j * ss, -INFINITY);
            continue;
        }
        float* buf = scratch_f32((size_t)(H * Dm + Dm + H + C + 16));
        float* qv = buf;
        float* kv = qv + H * Dm;
        float* wh = kv + Dm;
        float* row = wh + H;
        const float* qr = in_row(&q, rowoff(&q, m, H * Dm), H * Dm, qv);
        for (int64_t h = 0; h < H; ++h) {
            const float v = ldt(&w, rowoff(&w, m, H) + h * laststride(&w)) * wscale;
            wh[h] = exact ? v : narrow1(dt, v);
        }
        const int64_t s = seq_of_row(cu, S, m);
        const int64_t reach = (ldi(&pos, offlin(&pos, m)) + 1) / ratio;
        for (int64_t j = 0; j < C; ++j) {
            float v = -INFINITY;
            const int64_t col = (j * ratio) / cbs;
            if (j < reach && col < bt.shape[1]) {
                const int64_t blk = ldi(&bt, rowoff(&bt, s, bt.shape[1]) + col * laststride(&bt));
                if (blk >= 0 && blk < kc.shape[0]) {
                    const int64_t ko = (blk * cbs + (j * ratio) % cbs) * (kslots / ratio);
                    if (kf) {
                        const uint8_t* e = (const uint8_t*)kc.data + ko;
                        for (int64_t d = 0; d < Dm; ++d) kv[d] = rad_kvf_value(kf, e, Dm, d);
                    } else {
                        widen_flat(kc, ko, Dm, kv);
                    }
                    float acc = 0;
                    for (int64_t h = 0; h < H; ++h) {
                        float dot = dotf(qr + h * Dm, kv, Dm);
                        if (!exact) dot = narrow1(dt, dot);
                        const float t = std::max(dot, 0.0f) * wh[h];
                        acc += exact ? t : narrow1(dt, t);
                    }
                    v = exact ? acc : narrow1(dt, acc);
                }
            }
            row[j] = v;
        }
        out_row(&sc, rowoff(&sc, i, C), C, row);
    }
    return RAD_OK;
}

namespace {
/* libref's order: higher value first, ties to the lower index; a strict total order, so any
 * selection under it picks libref's set. */
struct ByScore {
    const float* v;
    bool operator()(int64_t x, int64_t y) const {
        return v[x] > v[y] || (v[x] == v[y] && x < y);
    }
};
/* The k first indices of [0, n) under ByScore, in that order (fewer when n < k). */
void top_indices(const float* v, int64_t n, int64_t k, std::vector<int64_t>* idx) {
    idx->resize((size_t)n);
    for (int64_t i = 0; i < n; ++i) (*idx)[(size_t)i] = i;
    k = std::min(k, n);
    if (k < n) std::nth_element(idx->begin(), idx->begin() + k, idx->end(), ByScore{ v });
    idx->resize((size_t)k);
    std::sort(idx->begin(), idx->end(), ByScore{ v });
}
/* index_candidates' keep words a row: a bit a block of `block` positions over `width`, 32 to an
 * i32 word (libref's ref_keep_words). */
int64_t keep_words(int64_t width, int64_t block) {
    return block < 1 || width < 1 ? 0 : ((width + block - 1) / block + 31) / 32;
}
}  // namespace

/* Level one of the two-level top-k (libref's index_candidates): blocks of `block` positions
 * scored by their maximum, the newest reachable block pinned to +inf, the topk_blocks best kept
 * unless -inf. keep is a bit a block, i32 [M, ceil(ceil(width / block) / 32)]: bit b % 32 of word
 * b / 32 set when block b is kept, the bits past the last block 0. */
AVX_KERNEL(index_candidates) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor &sc = a->t[0], &pos = a->t[1], &keep = a->t[2];
    const int64_t block = p_int(a, "block", 0), kb = p_int(a, "topk_blocks", 0);
    const int64_t ratio = p_int(a, "ratio", 0);
    if (block < 1 || kb < 1 || ratio < 1 || sc.rank != 2 || sc.shape[0] < 1 || sc.shape[1] < 1 ||
        sc.dtype != RAD_F32 || numel(&pos) != sc.shape[0] || keep.rank != 2 ||
        keep.shape[0] != sc.shape[0] || keep.shape[1] != keep_words(sc.shape[1], block))
        return RAD_E_SHAPE;
    if (keep.dtype != RAD_I32) return RAD_E_DTYPE;
    const int64_t M = sc.shape[0], C = sc.shape[1], nb = (C + block - 1) / block;
    const int64_t KW = keep.shape[1];
    AVX_PARALLEL_FOR_IF(M * C >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        float* buf = scratch_f32((size_t)(C + nb + 16));
        float* row = buf;
        float* bsc = row + C;
        const float* r = in_row(&sc, rowoff(&sc, m, C), C, row);
        /* std::max(running, v): a NaN score never wins a block, as in libref. */
        for (int64_t b = 0; b < nb; ++b) {
            float mx = -INFINITY;
            for (int64_t j = b * block, e = std::min(C, j + block); j < e; ++j)
                mx = std::max(mx, r[j]);
            bsc[b] = mx;
        }
        const int64_t reach = (ldi(&pos, offlin(&pos, m)) + 1) / ratio;
        if (reach >= 1 && (reach - 1) / block < nb) bsc[(reach - 1) / block] = INFINITY;
        static thread_local std::vector<int64_t> idx;
        static thread_local std::vector<uint32_t> bits;
        top_indices(bsc, nb, kb, &idx);
        bits.assign((size_t)KW, 0u);
        for (int64_t b : idx)
            if (bsc[b] > -INFINITY) bits[(size_t)(b / 32)] |= 1u << (b % 32);
        const int64_t kbase = rowoff(&keep, m, KW), ks = laststride(&keep);
        for (int64_t wd = 0; wd < KW; ++wd)
            st_int_dt(keep.dtype, keep.data, kbase + wd * ks, (int32_t)bits[(size_t)wd]);
    }
    return RAD_OK;
}

/* Level two (libref's index_topk): the topk best positions the row has passed, inside the kept
 * blocks when `keep` is given (index_candidates' bits over blocks of `block`), never -inf, ties to
 * the lower position, written in position order then -1. */
AVX_KERNEL(index_topk) {
    if (!a || a->n_t < 4 || !a->t) return RAD_E_INVAL;
    const RadTensor &sc = a->t[0], &pos = a->t[2], &sel = a->t[3];
    const RadTensor* keep = a->t[1].data ? &a->t[1] : nullptr;
    const int64_t topk = p_int(a, "topk", 0), ratio = p_int(a, "ratio", 0);
    const int64_t block = p_int(a, "block", 0);
    if (topk < 1 || ratio < 1 || !sc.data || !pos.data || !sel.data || sc.rank != 2 ||
        sc.shape[0] < 1 || sc.dtype != RAD_F32 || numel(&pos) != sc.shape[0] || sel.rank != 2 ||
        sel.shape[0] != sc.shape[0] || sel.shape[1] != topk ||
        (keep && (block < 1 || keep->rank != 2 || keep->shape[0] != sc.shape[0] ||
                  keep->shape[1] != keep_words(sc.shape[1], block))))
        return RAD_E_SHAPE;
    if (keep && keep->dtype != RAD_I32) return RAD_E_DTYPE;
    const int64_t M = sc.shape[0], C = sc.shape[1], KW = keep ? keep->shape[1] : 0;
    AVX_PARALLEL_FOR_IF(M * C >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t reach = std::min<int64_t>(C, (ldi(&pos, offlin(&pos, m)) + 1) / ratio);
        const int64_t n = std::max<int64_t>(reach, 0);
        float* v = scratch_f32((size_t)(C + 16));
        const float* r = in_row(&sc, rowoff(&sc, m, C), C, v);
        if (r != v) std::memcpy(v, r, (size_t)n * sizeof(float));
        if (keep) {
            /* A block at a time: its word's bit clears the whole block. */
            const int64_t kbase = rowoff(keep, m, KW), ks = laststride(keep);
            for (int64_t b = 0, j0 = 0; j0 < n; ++b, j0 += block) {
                const uint32_t wd =
                    (uint32_t)ld_int_dt(keep->dtype, keep->data, kbase + (b / 32) * ks);
                if ((wd >> (b % 32)) & 1u) continue;
                for (int64_t j = j0, e = std::min(n, j0 + block); j < e; ++j) v[j] = -INFINITY;
            }
        }
        static thread_local std::vector<int64_t> idx;
        top_indices(v, n, topk, &idx);
        int64_t np = 0;
        for (int64_t j : idx)
            if (v[j] > -INFINITY) idx[(size_t)np++] = j;
        std::sort(idx.begin(), idx.begin() + np);
        const int64_t sb = rowoff(&sel, m, topk), ss = laststride(&sel);
        for (int64_t t = 0; t < topk; ++t)
            st_int_dt(sel.dtype, sel.data, sb + t * ss, t < np ? idx[(size_t)t] : -1);
    }
    return RAD_OK;
}

/* One entry a block (libref's entry_store): x[w] into the whole of block page[w] of the cache,
 * rounded once to its dtype; a negative page writes nothing. Every page is checked before any
 * entry is written, where libref writes the entries before a bad one; the code is libref's. */
AVX_KERNEL(entry_store) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &page = a->t[1], &cache = a->t[2];
    const int64_t n = p_int(a, "n", 0), bs = p_int(a, "block_size", 0);
    /* format: bf16, or each entry packed as block-scaled E4M3 (rad_kv_fp8.h, libref's packing) */
    const int f = rad_kvf_parse(p_str(a, "format", "bf16"));
    if (f < 0) return RAD_E_UNSUPPORTED;
    const int64_t ne = f ? rad_kvf_bytes(f, n) : n;
    if (n < 1 || ne < 1 || bs < 1 || x.rank != 2 || x.shape[0] < 1 || x.shape[1] != n ||
        numel(&page) < x.shape[0] || cache.rank != 4 || cache.shape[2] != bs ||
        cache.shape[1] * cache.shape[2] * cache.shape[3] != ne)
        return RAD_E_SHAPE;
    if (f && (cache.dtype != RAD_F8E4M3 || !rad_tensor_is_contiguous(&cache))) return RAD_E_SHAPE;
    const int64_t W = x.shape[0];
    for (int64_t w = 0; w < W; ++w)
        if (ldi(&page, offlin(&page, w)) >= cache.shape[0]) return RAD_E_INVAL;
    /* Serial, as libref: two entries naming one page leave the later one. */
    for (int64_t w = 0; w < W; ++w) {
        const int64_t pg = ldi(&page, offlin(&page, w));
        if (pg < 0) continue;
        float* row = scratch_f32((size_t)n);
        const float* in = in_row(&x, rowoff(&x, w, n), n, row);
        if (f) rad_kvf_pack(f, in, n, (uint8_t*)cache.data + pg * ne);
        else row_from_f32(cache.dtype, in, (void*)byte_at(cache.data, cache.dtype, pg * n), n);
    }
    return RAD_OK;
}

/* mapped[m] = table[ids[m]], -1 outside [0, n_vocab) (libref's token_map). */
AVX_KERNEL(token_map) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &tab = a->t[1], &y = a->t[2];
    const int64_t nv = p_int(a, "n_vocab", 0);
    if (nv < 1 || numel(&tab) != nv || numel(&x) < 1 || numel(&y) != numel(&x)) return RAD_E_SHAPE;
    const int64_t M = numel(&x);
    for (int64_t m = 0; m < M; ++m) {
        const int64_t id = ldi(&x, offlin(&x, m));
        const int64_t v = id < 0 || id >= nv ? -1 : ldi(&tab, offlin(&tab, id));
        st_int_dt(y.dtype, y.data, offlin(&y, m), v);
    }
    return RAD_OK;
}

/* Engram's gate (libref's stream_gate_add): per row and stream the three sums in one vector pass
 * over the widened rows, the gate as libref computes it, then out = h + gate v. */
AVX_KERNEL(stream_gate_add) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor &h = a->t[0], &k = a->t[1], &v = a->t[2], &w = a->t[3], &out = a->t[4];
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0);
    const float eps = p_f32(a, "eps", -1.0f), ge = p_f32(a, "gate_eps", -1.0f);
    if (n < 1 || hc < 1 || hc > 16 || !(eps >= 0) || !(ge >= 0) || h.rank != 2 || h.shape[0] < 1 ||
        h.shape[1] != hc * n || k.rank != 2 || k.shape[0] != h.shape[0] || k.shape[1] != hc * n ||
        v.rank != 2 || v.shape[0] != h.shape[0] || v.shape[1] != n || numel(&w) != hc * n ||
        w.dtype != RAD_F32 || out.rank != 2 || out.shape[0] != h.shape[0] || out.shape[1] != hc * n)
        return RAD_E_SHAPE;
    const int64_t M = h.shape[0], width = hc * n;
    const bool wdense = laststride(&w) == 1;
    AVX_PARALLEL_FOR_IF(M * width >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        float* buf = scratch_f32((size_t)(4 * width + n + 16));
        float* hv = buf;
        float* kv = hv + width;
        float* vv = kv + width;
        float* wv = vv + n;
        float* ov = wv + width;
        const float* hr = in_row(&h, rowoff(&h, m, width), width, hv);
        const float* kr = in_row(&k, rowoff(&k, m, width), width, kv);
        const float* vr = in_row(&v, rowoff(&v, m, n), n, vv);
        const float* wr = wdense ? (const float*)w.data : wv;
        if (!wdense)
            for (int64_t i = 0; i < width; ++i) wv[i] = ldt(&w, i * laststride(&w));
        for (int64_t s = 0; s < hc; ++s) {
            const float* hs = hr + s * n;
            const float* ks = kr + s * n;
            const float* ws = wr + s * n;
            vf ah = vf_zero(), ak = vf_zero(), ad = vf_zero();
            int64_t i = 0;
            for (; i + VF_N <= n; i += VF_N) {
                const vf x = vf_loadu(hs + i), y = vf_loadu(ks + i);
                ah = vf_fma(x, x, ah);
                ak = vf_fma(y, y, ak);
                ad = vf_fma(vf_mul(x, vf_loadu(ws + i)), y, ad);
            }
            float sh = vf_hsum(ah), sk = vf_hsum(ak), dot = vf_hsum(ad);
            for (; i < n; ++i) {
                sh += hs[i] * hs[i];
                sk += ks[i] * ks[i];
                dot += hs[i] * ws[i] * ks[i];
            }
            const float rstd = (1.0f / std::sqrt(sh / (float)n + eps)) *
                               (1.0f / std::sqrt(sk / (float)n + eps));
            dot = dot * rstd * (1.0f / std::sqrt((float)n));
            const float gate =
                1.0f / (1.0f + std::exp(-std::copysign(std::sqrt(std::max(std::fabs(dot), ge)), dot)));
            const vf vg = vf_set1(gate);
            float* os = ov + s * n;
            i = 0;
            for (; i + VF_N <= n; i += VF_N)
                vf_storeu(os + i, vf_fma(vg, vf_loadu(vr + i), vf_loadu(hs + i)));
            for (; i < n; ++i) os[i] = hs[i] + gate * vr[i];
        }
        out_row(&out, rowoff(&out, m, width), width, ov);
    }
    return RAD_OK;
}
