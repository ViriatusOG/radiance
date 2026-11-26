/* avx_wan.cpp -- Wan 2.2 TI2V's ops (docs/WAN.md): the UMT5 text encoder's attention and gelu, the
 * diffusion transformer's timestep embedding, rotary table and rotation, modulated norm and gated
 * residual, its patch unfold and UniPC sampler step, and the VAE decoder's frame offsets, causal
 * history and temporal shuffle.
 *
 * The contracts are libref's (libref/ref_video.cpp, ref_gen.cpp, ref_attn.cpp), validated with the
 * same abi/ helpers in the same order, so both libraries refuse a geometry with the same code. The
 * layout and arithmetic shared with libr4d live in abi/ (rad_wan_dit.h, rad_wan_vae.h,
 * rad_umt5_attention.h, rad_umt5_gelu.h, rad_unipc_math.h); this file reads them rather than
 * restating them.
 *
 * Bit-exact with libref: the copies (wan_unpatch, wan_vae_history, wan_vae_expand,
 * wan_vae_frame_cu), the f64 trigonometry of wan_rope_table and timestep_embed_f64 (the same
 * double expressions, each evaluated once per distinct argument instead of once per element),
 * wan_rope, wan_gated_add and flow_unipc_step, where `keep` holds every product to its own
 * rounding as libref (built without contraction) rounds it. wan_mod_norm and attn_umt5 reduce in
 * another order, and gelu_umt5 takes its tanh from vf_exp; those are held to the tolerance table.
 */
#include "avx_vec.h"
#include "avx_common.h"
#include "rad_umt5_attention.h"
#include "rad_unipc_math.h"
#include "rad_wan_dit.h"
#include "rad_wan_vae.h"

#include <algorithm>

using namespace avx;

namespace {

#define WAN_TRY(expr) do { const int rc_ = (expr); if (rc_ < 0) return rc_; } while (0)

/* A value the compiler may not fuse into the next add. At the scalar level a vector is a float. */
#if AVX_LEVEL == 0
inline float keep(float x) { __asm__("" : "+x"(x)); return x; }
#elif AVX_LEVEL == 3
inline float keep(float x) { __asm__("" : "+v"(x)); return x; }
inline vf keep(vf x) { __asm__("" : "+v"(x)); return x; }
#else
inline float keep(float x) { __asm__("" : "+x"(x)); return x; }
inline vf keep(vf x) { __asm__("" : "+x"(x)); return x; }
#endif

/* Each lane rounded to bf16 (to nearest even, NaN kept quiet) and widened back: a store and load
 * of a bf16 tensor, as avx_vec.h's converters round. */
inline vf vf_rbf(vf x) {
#if AVX_LEVEL == 0
    return rad_bf16_to_f32(rad_f32_to_bf16(x));
#elif AVX_LEVEL == 3
    return _mm512_castsi512_ps(_mm512_slli_epi32(bf16_round_bits(x), 16));
#elif AVX_LEVEL == 2
    return _mm256_castsi256_ps(_mm256_slli_epi32(bf16_round_bits(x), 16));
#else
    const __m128i lo = _mm_slli_epi32(bf16_round_bits128(_mm256_castps256_ps128(x)), 16);
    const __m128i hi = _mm_slli_epi32(bf16_round_bits128(_mm256_extractf128_ps(x, 1)), 16);
    return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(lo)),
                                _mm_castsi128_ps(hi), 1);
#endif
}
inline float rbf(float x) { return rad_bf16_to_f32(rad_f32_to_bf16(x)); }

}  // namespace

/* ------------------------------------------------------------------ timestep_embed_f64 */
/* y[m] = [cos(t f_0) .. cos(t f_{h-1}) | sin(t f_0) .. sin(t f_{h-1})], f_i = period^(-i / h), all
 * in f64 and rounded once to f32, as libref evaluates them: the sampler's sigmas reach the model
 * unscaled, and an f32 angle would lose the phase at t ~ 1000. The frequencies are taken once a
 * launch. */
AVX_KERNEL(timestep_embed_f64) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *t = &a->t[0], *y = &a->t[1];
    if (t->dtype != RAD_F32 || y->dtype != RAD_F32) return RAD_E_DTYPE;
    const int64_t n = p_int(a, "n", 0);
    const RadParam* max_period = p_find(a, "max_period");
    const double period = max_period && max_period->kind == RAD_P_F64 ? max_period->dval : 10000.0;
    const int64_t rows = y->shape[0], pitch = y->stride[0], half = n / 2;
    if (n <= 0 || n > INT32_MAX || (n & 1) || !(period > 0.0) || !std::isfinite(period) ||
        y->rank != 2 || y->shape[1] != n || rows <= 0 || rows > INT32_MAX ||
        (t->rank != 1 && (t->rank != 2 || t->shape[1] != 1)) || t->shape[0] < rows)
        return RAD_E_SHAPE;
    if ((rows > 1 && t->stride[0] != 1) || y->stride[1] != 1 || pitch < n ||
        pitch > (INT64_MAX / 4 - n) / rows) return RAD_E_STRIDE;
    double* freq = (double*)scratch_raw((size_t)half * sizeof(double));
    for (int64_t i = 0; i < half; ++i) freq[i] = std::pow(period, -static_cast<double>(i) / half);
    const auto* times = static_cast<const float*>(t->data);
    auto* output = static_cast<float*>(y->data);
    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t m = 0; m < rows; ++m) {
        const double position = times[m];
        float* out = output + m * pitch;
        for (int64_t i = 0; i < half; ++i) {
            const double angle = position * freq[i];
            out[i] = static_cast<float>(std::cos(angle));
            out[half + i] = static_cast<float>(std::sin(angle));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ flow_unipc_step */
/* One UniPC BH2 step per element (abi/rad_unipc_math.h: rad_unipc::advance): the clean estimate
 * from the velocity, the corrector over the history when its sample coefficient is positive, then
 * the predictor; the history rows [last sample | last clean | older clean] advance in place. With
 * `negative` and `guidance` the velocity is first u + g (v - u). Each product is kept to its own
 * rounding and the sums run in advance()'s order, so the bytes are libref's. */
AVX_KERNEL(flow_unipc_step) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *v = &a->t[0], *coeff = &a->t[1], *history = &a->t[2], *x = &a->t[3];
    if ((v->dtype != RAD_F32 && v->dtype != RAD_BF16) || coeff->dtype != RAD_F32 ||
        history->dtype != RAD_F32 || x->dtype != RAD_F32) return RAD_E_DTYPE;
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0 || n > INT32_MAX / 3 || coeff->rank != 1 ||
        coeff->shape[0] != rad_unipc::COUNT) return RAD_E_SHAPE;
    if (v->rank != 2 || x->rank != 2 || history->rank != 2 ||
        v->shape[1] != n || x->shape[1] != n || history->shape[1] != 3 * n)
        return RAD_E_SHAPE;
    if (v->stride[1] != 1 || x->stride[1] != 1 || history->stride[1] != 1 ||
        v->stride[0] < n || x->stride[0] < n || history->stride[0] < 3 * n ||
        coeff->stride[0] != 1) return RAD_E_STRIDE;
    const int64_t rows = x->shape[0];
    if (rows <= 0 || v->shape[0] < rows || history->shape[0] < rows ||
        rows > (INT64_MAX - n) / x->stride[0] ||
        rows > (INT64_MAX - n) / v->stride[0] ||
        rows > (INT64_MAX - 3 * n) / history->stride[0]) return RAD_E_SHAPE;
    const RadTensor *negative = rad_arg_in(a, 4), *guidance = rad_arg_in(a, 5);
    if (bool(negative) != bool(guidance)) return RAD_E_INVAL;
    if (negative) {
        if (negative->dtype != v->dtype || guidance->dtype != RAD_F32) return RAD_E_DTYPE;
        if (negative->rank != 2 || negative->shape[0] < rows || negative->shape[1] != n ||
            guidance->rank != 1 || guidance->shape[0] != 1) return RAD_E_SHAPE;
        if (negative->stride[1] != 1 || negative->stride[0] < n) return RAD_E_STRIDE;
        if (rows > (INT64_MAX - n) / negative->stride[0]) return RAD_E_SHAPE;
    }
    using namespace rad_unipc;
    const float g = guidance ? ldt(guidance, 0) : 1.0f;
    const float* k = static_cast<const float*>(coeff->data);
    const bool correct = k[CORRECT_SAMPLE] > 0.0f;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)(2 * n + 32));
        float *vs = buf, *us = buf + n + 16;
        const float* vel = row_f32_in(v->dtype, byte_at(v->data, v->dtype, r * v->stride[0]), vs,
                                      n);
        const float* unc = negative ? row_f32_in(negative->dtype,
                                                 byte_at(negative->data, negative->dtype,
                                                         r * negative->stride[0]), us, n)
                                    : nullptr;
        float* xr = static_cast<float*>(x->data) + r * x->stride[0];
        float* ls = static_cast<float*>(history->data) + r * history->stride[0];
        float *lc = ls + n, *oc = ls + 2 * n;
        int64_t c = 0;
#if AVX_LEVEL > 0
        const vf vg = vf_set1(g), sigma = vf_set1(k[SIGMA]);
        const vf cs = vf_set1(k[CORRECT_SAMPLE]), cc = vf_set1(k[CORRECT_CLEAN]);
        const vf cp = vf_set1(k[CORRECT_PREVIOUS_DELTA]), cd = vf_set1(k[CORRECT_CURRENT_DELTA]);
        const vf ps = vf_set1(k[PREDICT_SAMPLE]), pc = vf_set1(k[PREDICT_CLEAN]);
        const vf pd = vf_set1(k[PREDICT_DELTA]);
        for (; c + VF_N <= n; c += VF_N) {
            vf velocity = vf_loadu(vel + c);
            if (unc) {
                const vf u = vf_loadu(unc + c);
                velocity = vf_add(u, keep(vf_mul(vg, vf_sub(velocity, u))));
            }
            vf sample = vf_loadu(xr + c);
            const vf last_s = vf_loadu(ls + c), last_c = vf_loadu(lc + c);
            const vf older_c = vf_loadu(oc + c);
            const vf clean = vf_sub(sample, keep(vf_mul(sigma, velocity)));
            if (correct)
                sample = vf_add(vf_add(vf_add(keep(vf_mul(cs, last_s)), keep(vf_mul(cc, last_c))),
                                       keep(vf_mul(cp, vf_sub(older_c, last_c)))),
                                keep(vf_mul(cd, vf_sub(clean, last_c))));
            const vf next = vf_add(vf_add(keep(vf_mul(ps, sample)), keep(vf_mul(pc, clean))),
                                   keep(vf_mul(pd, vf_sub(last_c, clean))));
            vf_storeu(xr + c, next);
            vf_storeu(ls + c, sample);
            vf_storeu(lc + c, clean);
            vf_storeu(oc + c, last_c);
        }
#endif
        for (; c < n; ++c) {
            float velocity = vel[c];
            if (unc) velocity = unc[c] + keep(g * (velocity - unc[c]));
            float sample = xr[c];
            const float last_s = ls[c], last_c = lc[c], older_c = oc[c];
            const float clean = sample - keep(k[SIGMA] * velocity);
            if (correct)
                sample = keep(k[CORRECT_SAMPLE] * last_s) + keep(k[CORRECT_CLEAN] * last_c) +
                         keep(k[CORRECT_PREVIOUS_DELTA] * (older_c - last_c)) +
                         keep(k[CORRECT_CURRENT_DELTA] * (clean - last_c));
            xr[c] = keep(k[PREDICT_SAMPLE] * sample) + keep(k[PREDICT_CLEAN] * clean) +
                    keep(k[PREDICT_DELTA] * (last_c - clean));
            ls[c] = sample;
            lc[c] = clean;
            oc[c] = last_c;
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ wan_rope_table */
/* Each latent voxel's three-axis rotary coefficients, [cos, sin] per pair: time takes the first
 * 22 pairs, height and width 21 each (abi/rad_wan_dit.h). A pair's angle depends only on the
 * voxel's coordinate along the pair's axis, so each axis's (coordinate, pair) values are evaluated
 * once in libref's f64 expression and the rows are assembled from them. */
AVX_KERNEL(wan_rope_table) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    if (p_int(a, "M", 0) <= 0) return RAD_E_SHAPE;
    using namespace rad_wan_dit;
    const RadTensor *grid = &a->t[0], *table = &a->t[1];
    WAN_TRY(check_grid(grid));
    const auto* dims = static_cast<const int32_t*>(grid->data);
    int64_t height, width, rows;
    if (!latent_grid(dims, height, width, rows)) return RAD_E_SHAPE;
    WAN_TRY(check_matrix(table, RAD_F32, rows, head_width));
    double inv[pairs];
    for (int pair = 0; pair < pairs; ++pair) {
        const int axis = pair_axis(pair);
        const int count = axis == 0 ? time_pairs : spatial_pairs;
        inv[pair] = 1.0 / std::pow(10000.0, (double)pair_in_axis(pair, axis) / count);
    }
    /* axis_cs[axis] holds, per coordinate, [cos, sin] for that axis's pairs in order. */
    const int64_t extent[3] = { dims[0], height, width };
    const int npairs[3] = { time_pairs, spatial_pairs, spatial_pairs };
    float* cs = (float*)scratch_raw((size_t)(2 * (extent[0] * time_pairs +
                                                  (extent[1] + extent[2]) * spatial_pairs)) *
                                    sizeof(float));
    float* axis_cs[3] = { cs, cs + 2 * extent[0] * time_pairs, nullptr };
    axis_cs[2] = axis_cs[1] + 2 * extent[1] * spatial_pairs;
    for (int axis = 0, first = 0; axis < 3; first += npairs[axis], ++axis)
        for (int64_t at = 0; at < extent[axis]; ++at)
            for (int i = 0; i < npairs[axis]; ++i) {
                const double angle = at * inv[first + i];
                float* dst = axis_cs[axis] + 2 * (at * npairs[axis] + i);
                dst[0] = static_cast<float>(std::cos(angle));
                dst[1] = static_cast<float>(std::sin(angle));
            }
    auto* out = static_cast<float*>(table->data);
    AVX_PARALLEL_FOR_IF(rows * head_width >= AVX_PAR_MIN)
    for (int64_t row = 0; row < rows; ++row) {
        float* dst = out + row * table->stride[0];
        for (int axis = 0; axis < 3; ++axis) {
            const int64_t at = coordinate(row, axis, height, width);
            std::memcpy(dst, axis_cs[axis] + 2 * at * npairs[axis],
                        (size_t)(2 * npairs[axis]) * sizeof(float));
            dst += 2 * npairs[axis];
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ wan_rope */
/* q and k rotated in place by their row's coefficients, each head's pairs (re, im) as complex
 * products: re' = re c - im s, im' = im c + re s. A row's coefficients are spread once into
 * per-lane factors C = [c, c] and S = [-s, s], so a lane computes x C + swap(x) S, swap exchanging
 * each pair's two lanes; the products are kept apart, so the bytes are libref's. */
AVX_KERNEL(wan_rope) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    if (p_int(a, "M", 0) <= 0) return RAD_E_SHAPE;
    using namespace rad_wan_dit;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *table = &a->t[2];
    const int64_t rows = q->shape[0], heads = p_int(a, "heads", 0);
    if (heads < 1 || heads > 128) return RAD_E_SHAPE;
    if (q->dtype != RAD_F32 && q->dtype != RAD_BF16) return RAD_E_DTYPE;
    WAN_TRY(check_matrix(q, q->dtype, rows, heads * head_width));
    WAN_TRY(check_matrix(k, q->dtype, rows, heads * head_width));
    WAN_TRY(check_matrix(table, RAD_F32, rows, head_width));
    const int64_t width = heads * head_width;
    AVX_PARALLEL_FOR_IF(rows * width >= AVX_PAR_MIN)
    for (int64_t row = 0; row < rows; ++row) {
        const float* cs = static_cast<const float*>(table->data) + row * table->stride[0];
        float* buf = scratch_f32((size_t)(width + 2 * head_width + 16));
        float *C = buf, *S = buf + head_width, *xs = S + head_width;
        for (int i = 0; i < head_width; i += 2) {
            C[i] = C[i + 1] = cs[i];
            S[i] = -cs[i + 1];
            S[i + 1] = cs[i + 1];
        }
        for (const RadTensor* t : { q, k }) {
            void* at = (void*)byte_at(t->data, t->dtype, row * t->stride[0]);
            row_to_f32(t->dtype, at, xs, width);
            for (int64_t h = 0; h < heads; ++h) {
                float* x = xs + h * head_width;
#if AVX_LEVEL == 0
                for (int i = 0; i < head_width; i += 2) {
                    const float re = x[i], im = x[i + 1];
                    x[i] = keep(re * C[i]) + keep(im * S[i]);
                    x[i + 1] = keep(im * C[i + 1]) + keep(re * S[i + 1]);
                }
#else
                for (int i = 0; i < head_width; i += VF_N) {
                    const vf v = vf_loadu(x + i);
#if AVX_LEVEL == 3
                    const vf sw = _mm512_permute_ps(v, 0xB1);
#else
                    const vf sw = _mm256_permute_ps(v, 0xB1);
#endif
                    vf_storeu(x + i, vf_add(keep(vf_mul(v, vf_loadu(C + i))),
                                            keep(vf_mul(sw, vf_loadu(S + i)))));
                }
#endif
            }
            row_from_f32(t->dtype, xs, at, width);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ wan_mod_norm */
/* A gainless layer norm, then (1 + scale) and shift, each the learned table row plus the step's
 * time projection: rows from `split` on take the second time row. `group` picks the block's
 * shift/scale pair (0 or 3) or the head's (-1, two table rows against one time row). The two
 * time rows' factors are formed once a launch in libref's f32 steps; the statistics reduce in
 * avx_vec.h's fixed order, held to the tolerance table. */
AVX_KERNEL(wan_mod_norm) {
    if (!rad_args_have(a, 3) || !rad_arg_in(a, 4)) return RAD_E_INVAL;
    if (p_int(a, "M", 0) <= 0) return RAD_E_SHAPE;
    using namespace rad_wan_dit;
    const RadTensor *x = &a->t[0], *time = &a->t[1], *table = &a->t[2];
    const RadTensor *split = rad_arg_in(a, 3), *y = &a->t[4];
    const int64_t n = p_int(a, "n", 0), rows = x->shape[0];
    const int64_t group = p_int(a, "group", 0);
    const bool head = group == -1, round = p_int(a, "round_norm", 0) != 0;
    const float eps = p_f32(a, "eps", 1e-6f);
    if (n < 1 || n > 4096 || (group != -1 && group != 0 && group != 3) ||
        !(eps > 0) || !std::isfinite(eps)) return RAD_E_SHAPE;
    if (y->dtype != RAD_BF16 && y->dtype != RAD_F32) return RAD_E_DTYPE;
    WAN_TRY(check_matrix(x, RAD_F32, rows, n));
    WAN_TRY(check_matrix(y, y->dtype, rows, n));
    WAN_TRY(check_modulation(time, table, split, n, head));
    const int64_t first = split ? ldi(split, 0) : rows;
    if (first < 0 || first > rows) return RAD_E_SHAPE;
    const int64_t shift_row = head ? 0 : group;
    const int64_t scale_row = head ? 0 : group + 1;
    const auto* weights = static_cast<const float*>(table->data) + shift_row * n;
    /* fac[s] = [1 + scale | shift] for time row s. */
    float* fac = (float*)scratch_raw((size_t)(4 * n) * sizeof(float));
    for (int s = 0; s < (split ? 2 : 1); ++s) {
        const auto* mod = static_cast<const float*>(time->data) + s * time->stride[0];
        for (int64_t c = 0; c < n; ++c) {
            fac[2 * s * n + c] = 1.0f + (weights[n + c] + mod[scale_row * n + c]);
            fac[(2 * s + 1) * n + c] = weights[c] + mod[shift_row * n + c];
        }
    }
    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const float* row = static_cast<const float*>(x->data) + r * x->stride[0];
        const float* onep = fac + (r >= first ? 2 * n : 0);
        const float* shift = onep + n;
        float* buf = scratch_f32((size_t)(2 * n + 32));
        float *nv = buf, *tmp = buf + n + 16;
        const float mean = row_sum(row, n) / static_cast<float>(n);
        const vf vm = vf_set1(mean);
        vf acc = vf_zero();
        int64_t c = 0;
        for (; c + VF_N <= n; c += VF_N) {
            const vf d = vf_sub(vf_loadu(row + c), vm);
            vf_storeu(nv + c, d);
            acc = vf_fma(d, d, acc);
        }
        float variance = vf_hsum(acc);
        for (; c < n; ++c) {
            nv[c] = row[c] - mean;
            variance += nv[c] * nv[c];
        }
        const float inverse = 1.0f / std::sqrt(variance / static_cast<float>(n) + eps);
        const vf vi = vf_set1(inverse);
        for (c = 0; c + VF_N <= n; c += VF_N) vf_storeu(nv + c, vf_mul(vf_loadu(nv + c), vi));
        for (; c < n; ++c) nv[c] *= inverse;
        if (round) {
            row_from_f32(RAD_BF16, nv, tmp, n);
            row_to_f32(RAD_BF16, tmp, nv, n);
        }
        for (c = 0; c + VF_N <= n; c += VF_N)
            vf_storeu(nv + c, vf_fma(vf_loadu(nv + c), vf_loadu(onep + c), vf_loadu(shift + c)));
        for (; c < n; ++c) nv[c] = nv[c] * onep[c] + shift[c];
        row_from_f32(y->dtype, nv, (void*)byte_at(y->data, y->dtype, r * y->stride[0]), n);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ wan_gated_add */
/* The f32 residual stream += a bf16 branch output, times its gate (the table's `gate` row plus the
 * step's time projection; rows from `split` on take the second time row) when `gate` is 2 or 5.
 * The gates are formed once a launch; the product is kept apart from the add, so the bytes are
 * libref's. */
AVX_KERNEL(wan_gated_add) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    if (p_int(a, "M", 0) <= 0) return RAD_E_SHAPE;
    using namespace rad_wan_dit;
    const RadTensor *x = &a->t[0], *update = &a->t[1];
    const RadTensor *time = rad_arg_in(a, 2), *table = rad_arg_in(a, 3), *split = rad_arg_in(a, 4);
    const int64_t n = p_int(a, "n", 0), rows = x->shape[0], gate = p_int(a, "gate", -1);
    const bool gated = gate != -1;
    if (n < 1 || n > 4096 || (gated && gate != 2 && gate != 5)) return RAD_E_SHAPE;
    if (gated ? (!time || !table) : (time || table || split)) return RAD_E_INVAL;
    WAN_TRY(check_matrix(x, RAD_F32, rows, n));
    WAN_TRY(check_matrix(update, RAD_BF16, rows, n));
    if (gated) WAN_TRY(check_modulation(time, table, split, n, false));
    const int64_t first = split ? ldi(split, 0) : rows;
    if (first < 0 || first > rows) return RAD_E_SHAPE;
    float* gates = nullptr;
    if (gated) {
        gates = (float*)scratch_raw((size_t)(2 * n) * sizeof(float));
        const auto* w = static_cast<const float*>(table->data) + gate * n;
        for (int s = 0; s < (split ? 2 : 1); ++s) {
            const auto* mod = static_cast<const float*>(time->data) + s * time->stride[0] + gate * n;
            for (int64_t c = 0; c < n; ++c) gates[s * n + c] = w[c] + mod[c];
        }
    }
    auto* state = static_cast<float*>(x->data);
    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* u = scratch_f32((size_t)(n + 16));
        row_to_f32(RAD_BF16, byte_at(update->data, RAD_BF16, r * update->stride[0]), u, n);
        float* s = state + r * x->stride[0];
        int64_t c = 0;
        if (gated) {
            const float* gr = gates + (r >= first ? n : 0);
            for (; c + VF_N <= n; c += VF_N)
                vf_storeu(s + c, vf_add(vf_loadu(s + c),
                                        keep(vf_mul(vf_loadu(u + c), vf_loadu(gr + c)))));
            for (; c < n; ++c) s[c] += keep(u[c] * gr[c]);
        } else {
            for (; c + VF_N <= n; c += VF_N)
                vf_storeu(s + c, vf_add(vf_loadu(s + c), vf_loadu(u + c)));
            for (; c < n; ++c) s[c] += u[c];
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ wan_unpatch */
/* The head's 2x2 patches [patch, 4 * channels] back to f32 voxel rows [4 * patches, channels]: each
 * output row is one contiguous channel run of a patch (abi/rad_wan_dit.h: unpack_offset), widened
 * in one call. */
AVX_KERNEL(wan_unpatch) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    if (p_int(a, "M", 0) <= 0) return RAD_E_SHAPE;
    using namespace rad_wan_dit;
    const RadTensor *src = &a->t[0], *grid = &a->t[1], *dst = &a->t[2];
    const int64_t channels = p_int(a, "n", 0);
    WAN_TRY(check_grid(grid));
    if (channels < 1 || channels > 1024) return RAD_E_SHAPE;
    const auto* dims = static_cast<const int32_t*>(grid->data);
    int64_t height, width, rows;
    if (!latent_grid(dims, height, width, rows)) return RAD_E_SHAPE;
    if (src->dtype != RAD_F32 && src->dtype != RAD_BF16) return RAD_E_DTYPE;
    WAN_TRY(check_matrix(src, src->dtype, rows, 4 * channels));
    WAN_TRY(check_matrix(dst, RAD_F32, 4 * rows, channels));
    auto* out = static_cast<float*>(dst->data);
    AVX_PARALLEL_FOR_IF(4 * rows * channels >= AVX_PAR_MIN)
    for (int64_t row = 0; row < 4 * rows; ++row)
        row_to_f32(src->dtype,
                   byte_at(src->data, src->dtype,
                           unpack_offset(row, 0, height, width, channels, src->stride[0])),
                   out + row * dst->stride[0], channels);
    return RAD_OK;
}

/* ------------------------------------------------------------------ wan_vae_frame_cu */
/* One attention segment per decoded frame: cu[i] = i * H * W. */
AVX_KERNEL(wan_vae_frame_cu) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    if (p_int(a, "M", 0) <= 0) return RAD_E_SHAPE;
    const RadTensor *geom = &a->t[0], *y = &a->t[1];
    WAN_TRY(rad_wan_dit::check_grid(geom));
    if (y->dtype != RAD_I32) return RAD_E_DTYPE;
    if (y->rank != 1 || y->shape[0] < 2 || y->shape[0] > rad_wan_vae::max_chunk_frames + 1)
        return RAD_E_SHAPE;
    if (y->stride[0] != 1) return RAD_E_STRIDE;
    int64_t plane = 0;
    if (!rad_wan_vae::frame_plane(static_cast<const int32_t*>(geom->data), y->shape[0], plane))
        return RAD_E_SHAPE;
    auto* cu = static_cast<int32_t*>(y->data);
    for (int32_t i = 0; i < y->shape[0]; ++i) cu[i] = static_cast<int32_t>(i * plane);
    return RAD_OK;
}

/* ------------------------------------------------------------------ wan_vae_history */
/* A causal convolution's input with its two cached frames: per pixel the sequence
 * s = [cached older, cached newer, frame 0, frame 1, ...] (zeros for the cache under `reset`).
 * Plain mode writes s as rows; `temporal` writes frame f's three taps s[f], s[f + 1], s[f + 2]
 * interleaved per channel. The cache then keeps the sequence's last two. A pixel is one task:
 * its rows are whole channel runs, copied as bytes, and its cache rows are read before they are
 * rewritten. */
AVX_KERNEL(wan_vae_history) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    if (p_int(a, "M", 0) <= 0) return RAD_E_SHAPE;
    const RadTensor *x = &a->t[0], *history = &a->t[1], *out = &a->t[2];
    const int64_t mode = p_int(a, "temporal", 0), reset = p_int(a, "reset", 0);
    if ((mode != 0 && mode != 1) || (reset != 0 && reset != 1)) return RAD_E_INVAL;
    int64_t plane, frames;
    WAN_TRY(rad_wan_vae::check_history(x, history, out, mode != 0, plane, frames));
    const int64_t channels = x->shape[1];
    const size_t bytes = (size_t)channels * sizeof(uint16_t);
    const auto* src = static_cast<const uint16_t*>(x->data);
    auto* cache = static_cast<uint16_t*>(history->data);
    auto* dest = static_cast<uint16_t*>(out->data);
    const int64_t x_ld = x->stride[0], h_ld = history->stride[0], y_ld = out->stride[0];
    auto* zeros = static_cast<uint16_t*>(scratch_raw(bytes));
    std::memset(zeros, 0, bytes);
    AVX_PARALLEL_FOR_IF(plane * channels * frames >= AVX_PAR_MIN)
    for (int64_t pixel = 0; pixel < plane; ++pixel) {
        auto seq = [&](int64_t i) -> const uint16_t* {
            if (i < 2) return reset ? zeros : cache + (i * plane + pixel) * h_ld;
            return src + ((i - 2) * plane + pixel) * x_ld;
        };
        if (mode) {
            for (int64_t f = 0; f < frames; ++f) {
                const uint16_t *s0 = seq(f), *s1 = seq(f + 1), *s2 = seq(f + 2);
                uint16_t* tap = dest + (f * plane + pixel) * y_ld;
                for (int64_t c = 0; c < channels; ++c) {
                    tap[3 * c] = s0[c];
                    tap[3 * c + 1] = s1[c];
                    tap[3 * c + 2] = s2[c];
                }
            }
        } else {
            for (int64_t i = 0; i < frames + 2; ++i)
                std::memcpy(dest + (i * plane + pixel) * y_ld, seq(i), bytes);
        }
        /* Older first: with one frame it is the cached newer row, rewritten next. */
        std::memcpy(cache + pixel * h_ld, seq(frames), bytes);
        std::memcpy(cache + (plane + pixel) * h_ld, seq(frames + 1), bytes);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ wan_vae_expand */
/* The decoder's resampling as a gather (abi/rad_wan_vae.h: expand_offset): `temporal` deals the
 * channel halves into two frames, a channel run each; otherwise the shortcut's repeat over
 * time and both spatial axes, where an output channel's source depends on the voxel only through
 * (t % st, y % s, x % s). Those at most eight source-column maps are built once a launch, and a
 * row is then one gather. */
AVX_KERNEL(wan_vae_expand) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    if (p_int(a, "M", 0) <= 0) return RAD_E_SHAPE;
    const RadTensor *x = &a->t[0], *geom = &a->t[1], *out = &a->t[2];
    const int64_t cin = p_int(a, "cin", 0), cout = p_int(a, "cout", 0);
    const int64_t st = p_int(a, "st", 1), s = p_int(a, "s", 1), drop = p_int(a, "drop", 0);
    const int64_t mode = p_int(a, "temporal", 0);
    if (mode != 0 && mode != 1) return RAD_E_INVAL;
    WAN_TRY(rad_wan_dit::check_grid(geom));
    WAN_TRY(rad_wan_vae::check_expand(x, out, cin, cout, st, s, drop, mode != 0));
    const auto* dims = static_cast<const int32_t*>(geom->data);
    const int64_t rows = out->shape[0];
    if (!rad_wan_vae::check_expand_grid(dims, x->shape[0], rows, st, s, drop))
        return RAD_E_SHAPE;
    const int64_t h = dims[1], w = dims[2], out_w = w * s, out_h = h * s, pitch = x->stride[0];
    const auto* input = static_cast<const uint16_t*>(x->data);
    auto* output = static_cast<uint16_t*>(out->data);
    /* cols[((t % st) * s + y % s) * s + x % s][c]: output channel c's source column. */
    int64_t* cols = nullptr;
    if (!mode) {
        const int64_t repeat = cout * st * s * s / cin;
        cols = static_cast<int64_t*>(scratch_raw((size_t)(st * s * s * cout) * sizeof(int64_t)));
        for (int64_t k = 0; k < st * s * s; ++k)
            for (int64_t c = 0; c < cout; ++c)
                cols[k * cout + c] = ((c * st + k / (s * s)) * s * s + k % (s * s)) / repeat;
    }
    AVX_PARALLEL_FOR_IF(rows * cout >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xx = r % out_w, yy = (r / out_w) % out_h;
        const int64_t t = r / (out_w * out_h) + drop;
        const uint16_t* src = input + (((t / st) * h + yy / s) * w + xx / s) * pitch;
        uint16_t* dst = output + r * out->stride[0];
        if (mode) {
            std::memcpy(dst, src + (t % st) * cout, (size_t)cout * sizeof(uint16_t));
        } else {
            const int64_t* col = cols + (((t % st) * s + yy % s) * s + xx % s) * cout;
            for (int64_t c = 0; c < cout; ++c) dst[c] = src[col[c]];
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gelu_umt5 */
/* UMT5's tanh gelu as PyTorch evaluates it in bf16, every intermediate stored
 * (abi/rad_umt5_gelu.h): rd(rd(0.5 x) rd(1 + rd(tanh(rd(k0 rd(x + rd(0.044715 rd(rd(x x) x)))))))),
 * a lane at a time with vf_rbf for each rd. tanh is (1 - e^(-2u)) / (1 + e^(-2u)) on vf_exp,
 * whose error near zero is far below the bf16 step of the 1 + tanh it feeds. */
AVX_KERNEL(gelu_umt5) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    if (X->dtype != RAD_BF16 || Y->dtype != RAD_BF16) return RAD_E_DTYPE;
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || rows_of(X, n) < rows) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)(2 * n + 32));
        float *xs = buf, *y = buf + n + 16;
        const float* x = in_row(X, rowoff(X, r, n), n, xs);
        int64_t c = 0;
#if AVX_LEVEL > 0
        const vf one = vf_set1(1.0f), half = vf_set1(0.5f), m2 = vf_set1(-2.0f);
        const vf k0 = vf_set1(0.7978845608028654f), k1 = vf_set1(0.044715f);
        for (; c + VF_N <= n; c += VF_N) {
            const vf g = vf_loadu(x + c);
            const vf cube = vf_rbf(vf_mul(vf_rbf(vf_mul(g, g)), g));
            const vf inner = vf_rbf(vf_add(g, vf_rbf(vf_mul(k1, cube))));
            const vf u = vf_rbf(vf_mul(k0, inner));
            const vf e = vf_exp(vf_mul(m2, u));
            const vf act = vf_rbf(vf_div(vf_sub(one, e), vf_add(one, e)));
            vf_storeu(y + c, vf_rbf(vf_mul(vf_rbf(vf_mul(half, g)), vf_rbf(vf_add(one, act)))));
        }
#endif
        for (; c < n; ++c) {
            const float g = x[c];
            const float cube = rbf(rbf(g * g) * g);
            const float inner = rbf(g + rbf(0.044715f * cube));
            const float act = rbf(sc_tanh(rbf(0.7978845608028654f * inner)));
            y[c] = rbf(rbf(0.5f * g) * rbf(1.0f + act));
        }
        out_row(Y, rowoff(Y, r, n), n, y);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ attn_umt5 */
/* UMT5's bidirectional attention over packed sequences: an unscaled q.k rounded to bf16, plus the
 * head's learned bias for the signed distance's bucket (abi/rad_umt5_attention.h), rounded again;
 * a softmax whose probabilities are rounded to bf16; then their sum over v in f32. One task per
 * (head, sequence): its keys and values are widened once into f32 panels, and the bias by distance
 * laid out once so a query's biases are one contiguous run. Held to the tolerance table: the dot,
 * the exponentials and the sums are vector forms. */
AVX_KERNEL(attn_umt5) {
    if (!rad_args_have(a, 6)) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2];
    const RadTensor *cu = &a->t[3], *bias = &a->t[4], *out = &a->t[5];
    const RadTensor* matrices[] = { q, k, v, out };
    if (cu->rank != 1 || cu->shape[0] < 2) return RAD_E_SHAPE;
    const int64_t heads = p_int(a, "heads", 0), segments = cu->shape[0] - 1;
    int64_t rows = 0;
    WAN_TRY(rad_umt5::validate(matrices, cu, bias, heads, segments, rows));
    auto cu_at = [cu](int64_t i) { return ldi(cu, offlin(cu, i)); };
    if (cu_at(0) != 0 || cu_at(segments) != rows) return RAD_E_SHAPE;
    for (int64_t s = 0; s < segments; ++s) {
        const int64_t length = cu_at(s + 1) - cu_at(s);
        if (length < 1 || length > rad_umt5::max_tokens) return RAD_E_SHAPE;
    }
    constexpr int64_t hd = rad_umt5::head_width;
    AVX_PARALLEL_FOR_IF(heads * segments >= 2 && rows * heads * hd >= AVX_PAR_MIN)
    for (int64_t task = 0; task < heads * segments; ++task) {
        const int64_t h = task / segments, s = task % segments;
        const int64_t begin = cu_at(s), len = cu_at(s + 1) - begin;
        float* buf = scratch_f32((size_t)(2 * len * hd + 4 * len + 2 * hd + 64));
        float *kp = buf, *vp = kp + len * hd, *bd = vp + len * hd;
        float *sc = bd + 2 * len + 16, *qf = sc + len + 16, *acc = qf + hd;
        auto row_of = [h](const RadTensor* t, int64_t i) {
            return byte_at(t->data, RAD_BF16, i * t->stride[0] + h * hd);
        };
        for (int64_t j = 0; j < len; ++j) {
            row_to_f32(RAD_BF16, row_of(k, begin + j), kp + j * hd, hd);
            row_to_f32(RAD_BF16, row_of(v, begin + j), vp + j * hd, hd);
        }
        /* bd[d + len - 1] = the bias for key minus query distance d. */
        for (int64_t d = 1 - len; d < len; ++d)
            bd[d + len - 1] = ldt(bias, rad_umt5::bucket((int)d) * bias->stride[0] + h);
        for (int64_t i = 0; i < len; ++i) {
            row_to_f32(RAD_BF16, row_of(q, begin + i), qf, hd);
            for (int64_t j = 0; j < len; ++j) sc[j] = dotf(qf, kp + j * hd, hd);
            const float* bi = bd + (len - 1 - i);
            int64_t j = 0;
            for (; j + VF_N <= len; j += VF_N)
                vf_storeu(sc + j, vf_rbf(vf_add(vf_rbf(vf_loadu(sc + j)), vf_loadu(bi + j))));
            for (; j < len; ++j) sc[j] = rbf(rbf(sc[j]) + bi[j]);
            const float mx = row_max(sc, len);
            const vf vmx = vf_set1(mx);
            for (j = 0; j + VF_N <= len; j += VF_N)
                vf_storeu(sc + j, vf_exp(vf_sub(vf_loadu(sc + j), vmx)));
            for (; j < len; ++j) sc[j] = sc_exp(sc[j] - mx);
            const float sum = row_sum(sc, len);
            const vf vs = vf_set1(sum);
            for (j = 0; j + VF_N <= len; j += VF_N)
                vf_storeu(sc + j, vf_rbf(vf_div(vf_loadu(sc + j), vs)));
            for (; j < len; ++j) sc[j] = rbf(sc[j] / sum);
            for (int64_t c = 0; c < hd; ++c) acc[c] = 0.0f;
            for (j = 0; j < len; ++j) {
                const vf p = vf_set1(sc[j]);
                const float* vr = vp + j * hd;
                for (int64_t c = 0; c < hd; c += VF_N)
                    vf_storeu(acc + c, vf_fma(p, vf_loadu(vr + c), vf_loadu(acc + c)));
            }
            row_from_f32(RAD_BF16, acc, (void*)row_of(out, begin + i), hd);
        }
    }
    return RAD_OK;
}
