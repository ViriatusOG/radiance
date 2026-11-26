/* lq_ggml_gpu.hip -- GGML IQ2_XXS/IQ3_XXS on the card. See lq_ggml_gpu.h.
 *
 * One workgroup holds eight 256-value blocks; each block is eight lanes, one lane per 32-value
 * group. The lane runs llama.cpp's group search in the host's order, writes its grid/sign bytes,
 * and the workgroup reduces the eight group scales to the block's d and scale bits. The host then
 * unpacks the returned GGUF blocks into planes exactly as the host loop does.
 *
 * The search is a copy of lq_ggml_port.cpp's quantize_row_iq2_xxs_impl/iq3_xxs_impl, so the card
 * must write the host's bytes. libquant is built with -ffp-contract=off; this file additionally
 * builds without fast-math and uses the host's f16/nearest_int bit tricks.
 */
#include "hip/hip_runtime.h"

#include "lq_ggml_gpu.h"
#include "lq_ggml_port.h"
#include "lq_gpu.h"
#include "rad_plugin.h"

#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace lq::ggml {

namespace {

constexpr int kBlock = 256;
constexpr int kGroup = 32;
constexpr int kGroups = kBlock / kGroup;
constexpr int kBlocksPerWg = 8;
constexpr int kWgThreads = kBlocksPerWg * kGroups;

#define DMAX(a, b) ((a) > (b) ? (a) : (b))
#define FDIV(a, b) __fdiv_rn((a), (b))

__device__ __forceinline__ int dmin_i(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int dmax_i(int a, int b) { return a > b ? a : b; }

/* ggml-quants.c's nearest_int: the magic-number round lq_ggml_port.cpp uses. */
__device__ __forceinline__ int nearest_int_d(float fval) {
    const int i = __float_as_int(fval + 12582912.0f);
    return (i & 0x007fffff) - 0x00400000;
}

/* ggml-impl.h's fp32->fp16 round-to-nearest-even conversion, the same as lq_ggml_port.cpp's. */
__device__ __forceinline__ uint16_t fp16_d(float f) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = __float_as_uint(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) bias = 0x71000000u;

    base = __uint_as_float((bias >> 1) + 0x07800000u) + base;
    const uint32_t bits = __float_as_uint(base);
    const uint32_t exp_bits = (bits >> 13) & 0x00007C00u;
    const uint32_t mantissa_bits = bits & 0x00000FFFu;
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (uint16_t)((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

/* ggml-quants.c's make_qp_quants, exactly. */
__device__ __forceinline__ float make_qp_quants_d(int n, int nmax, const float * x, int8_t * L,
                                                  const float * quant_weights) {
    float max = 0;
    for (int i = 0; i < n; ++i) max = DMAX(max, x[i]);
    if (max < 1e-15f) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.0f;
    }
    float iscale = FDIV((float)nmax, max);
    for (int i = 0; i < n; ++i) L[i] = (int8_t)nearest_int_d(iscale * x[i]);
    float scale = FDIV(1.0f, iscale);
    float best_mse = 0;
    for (int i = 0; i < n; ++i) {
        /* Host make_qp_quants takes uint8_t L: nearest_int may return -1 and host wraps it to 255. */
        const float diff = x[i] - scale * (float)(uint8_t)L[i];
        const float w = quant_weights[i];
        best_mse += w * diff * diff;
    }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) continue;
        const float iscale_is = FDIV((0.1f * is + (float)nmax), max);
        const float scale_is = FDIV(1.0f, iscale_is);
        float mse = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_d(iscale_is * x[i]);
            l = dmin_i(nmax, l);
            const float diff = x[i] - scale_is * l;
            const float w = quant_weights[i];
            mse += w * diff * diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int_d(iscale * x[i]);
        l = dmin_i(nmax, l);
        L[i] = (int8_t)l;
        const float w = quant_weights[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            const float w = quant_weights[i];
            /* Host reads L through uint8_t here too; an int8 -1 would subtract the wrong residual. */
            float slx = sumlx - w * x[i] * (float)(uint8_t)L[i];
            float sl2 = suml2 - w * (float)(uint8_t)L[i] * (float)(uint8_t)L[i];
            if (slx > 0 && sl2 > 0) {
                int new_l = nearest_int_d(FDIV(x[i] * sl2, slx));
                new_l = dmin_i(nmax, new_l);
                // Host L is uint8_t, so its -1 is already 255 for this comparison too.
                if (new_l != (int)(uint8_t)L[i]) {
                    slx += w * x[i] * new_l;
                    sl2 += w * new_l * new_l;
                    if (slx * slx * suml2 > sumlx * sumlx * sl2) {
                        L[i] = (int8_t)new_l;
                        sumlx = slx;
                        suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) break;
    }
    return suml2 > 0.0f ? FDIV(sumlx, suml2) : 0.0f;
}

/* iq2_find_best_neighbour, exactly. */
__device__ __forceinline__ int find_best_iq2(const uint16_t * neighbours, const uint64_t * grid,
                                              const float * xval, const float * weight, float scale,
                                              int8_t * L) {
    const int num_neighbors = neighbours[0];
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            const float q = pg[i];
            const float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            grid_index = neighbours[j];
        }
    }
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; ++i) L[i] = (int8_t)((pg[i] - 1) / 2);
    return grid_index;
}

/* iq3_find_best_neighbour, exactly. */
__device__ __forceinline__ int find_best_iq3(const uint16_t * neighbours, const uint32_t * grid,
                                              const float * xval, const float * weight, float scale,
                                              int8_t * L) {
    const int num_neighbors = neighbours[0];
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 4; ++i) {
            const float q = pg[i];
            const float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            grid_index = neighbours[j];
        }
    }
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 4; ++i) L[i] = (int8_t)((pg[i] - 1) / 2);
    return grid_index;
}

/* One IQ2_XXS 32-value group. qs is the group's 8-byte q2 slot; returns the group scale and
 * leaves the scale nibble zero. */
__device__ __forceinline__ float iq2_group(const float * xb, const float * qw, float sigma2,
                                           const uint64_t * grid, const int * map,
                                           const uint16_t * neighbours, uint8_t * qs) {
    float weight[32];
    float xval[32];
    float waux[32];
    int8_t L[32];
    int8_t Laux[32];
    uint8_t signs[4];

    for (int i = 0; i < 32; ++i) weight[i] = qw[i] * __fsqrt_rn(sigma2 + xb[i] * xb[i]);
    for (int i = 0; i < 32; ++i) waux[i] = __fsqrt_rn(weight[i]);

    for (int k = 0; k < 4; ++k) {
        int nflip = 0;
        uint8_t s = 0;
        for (int i = 0; i < 8; ++i) {
            if (xb[8 * k + i] >= 0) {
                xval[8 * k + i] = xb[8 * k + i];
            } else {
                xval[8 * k + i] = -xb[8 * k + i];
                ++nflip;
                s |= (uint8_t)(1u << i);
            }
        }
        if (nflip % 2) {
            int imin = 0;
            float min = weight[8 * k] * xb[8 * k] * xb[8 * k];
            for (int i = 1; i < 8; ++i) {
                const float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                if (ax < min) {
                    min = ax;
                    imin = i;
                }
            }
            xval[8 * k + imin] = -xval[8 * k + imin];
            s ^= (uint8_t)(1u << imin);
        }
        signs[k] = s & 127;
    }

    float max = xval[0];
    for (int i = 1; i < 32; ++i) max = DMAX(max, xval[i]);

    if (max < 1e-15f) {
        for (int i = 0; i < 8; ++i) qs[i] = 0;
        return 0.0f;
    }

    float scale = make_qp_quants_d(32, 4, xval, L, weight);
    const float eff_max = scale * 3.0f;
    if (eff_max <= 0) {
        for (int i = 0; i < 8; ++i) qs[i] = 0;
        return 0.0f;
    }

    float best = 0;
    for (int is = -6; is <= 6; ++is) {
        const float id = FDIV((5.0f + (float)is * 0.1f), eff_max);
        const float this_scale = FDIV(1.0f, id);
        for (int k = 0; k < 4; ++k) {
            for (int i = 0; i < 8; ++i) {
                int l = nearest_int_d(0.5f * (id * xval[8 * k + i] - 1));
                Laux[8 * k + i] = (int8_t)dmax_i(0, dmin_i(2, l));
            }
            uint16_t u = 0;
            for (int i = 0; i < 8; ++i) u |= (uint16_t)((uint16_t)Laux[8 * k + i] << (2 * i));
            int grid_index = map[u];
            if (grid_index < 0) {
                const uint16_t * nb = neighbours - map[u] - 1;
                grid_index = find_best_iq2(nb, grid, xval + 8 * k, waux + 8 * k, this_scale,
                                           Laux + 8 * k);
            }
            (void)grid_index;
        }
        float sumqx = 0;
        float sumq2 = 0;
        for (int i = 0; i < 32; ++i) {
            const float w = weight[i];
            const float q = (float)(2 * Laux[i] + 1);
            sumqx += w * xval[i] * q;
            sumq2 += w * q * q;
        }
        if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
            scale = FDIV(sumqx, sumq2);
            best = scale * sumqx;
            for (int i = 0; i < 32; ++i) L[i] = Laux[i];
        }
    }

    if (scale > 0) {
        const float id = FDIV(1.0f, scale);
        for (int k = 0; k < 4; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 8; ++i) {
                int l = nearest_int_d(0.5f * (id * xval[8 * k + i] - 1));
                l = dmax_i(0, dmin_i(2, l));
                u |= (uint16_t)((uint16_t)l << (2 * i));
            }
            int grid_index = map[u];
            if (grid_index < 0) {
                const uint16_t * nb = neighbours - map[u] - 1;
                grid_index = find_best_iq2(nb, grid, xval + 8 * k, waux + 8 * k, scale, L + 8 * k);
            }
            const int8_t * pg = (const int8_t *)(grid + grid_index);
            for (int i = 0; i < 8; ++i) L[8 * k + i] = (int8_t)((pg[i] - 1) / 2);
        }
        float sumqx = 0;
        float sumq2 = 0;
        for (int i = 0; i < 32; ++i) {
            const float w = weight[i];
            const float q = (float)(2 * L[i] + 1);
            sumqx += w * xval[i] * q;
            sumq2 += w * q * q;
        }
        if (sumq2 > 0) scale = FDIV(sumqx, sumq2);
    }
    if (scale < 0) {
        scale = -scale;
        for (int k = 0; k < 4; ++k) signs[k] = (uint8_t)((~signs[k]) & 127);
    }

    uint32_t lo = 0;
    uint32_t hi = 0;
    for (int k = 0; k < 4; ++k) {
        uint16_t u = 0;
        for (int i = 0; i < 8; ++i) u |= (uint16_t)((uint16_t)L[8 * k + i] << (2 * i));
        const int grid_index = map[u];
        lo |= (uint32_t)grid_index << (8 * k);
        hi |= (uint32_t)signs[k] << (7 * k);
    }
    for (int b = 0; b < 4; ++b) {
        qs[b] = (uint8_t)((lo >> (8 * b)) & 255);
        qs[4 + b] = (uint8_t)((hi >> (8 * b)) & 255);
    }
    return scale;
}

/* One IQ3_XXS 32-value group. qs is the group's 8-byte q3 slot, sign_scale its packed sign/scale
 * word; returns the group scale. */
__device__ __forceinline__ float iq3_group(const float * xb, const float * qw, float sigma2,
                                           const uint32_t * grid, const int * map,
                                           const uint16_t * neighbours, uint8_t * qs,
                                           uint8_t * sign_scale) {
    float weight[32];
    float xval[32];
    float waux[32];
    int8_t L[32];
    int8_t Laux[32];
    uint8_t signs[4];
    bool on_grid[8];
    bool on_aux[8];

    // Host memset(L, 0, 32) before the IQ3 grid search; leave no lane undefined.
    for (int i = 0; i < 32; ++i) L[i] = 0;

    for (int i = 0; i < 32; ++i) {
        /* Host IQ3 uses sqrtf; __fsqrt_rn differs on real imatrix values and can change a grid pick. */
        if (qw)
            weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
        else
            weight[i] = xb[i] * xb[i];
    }
    for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);

    for (int k = 0; k < 4; ++k) {
        int nflip = 0;
        uint8_t s = 0;
        for (int i = 0; i < 8; ++i) {
            if (xb[8 * k + i] >= 0) {
                xval[8 * k + i] = xb[8 * k + i];
            } else {
                xval[8 * k + i] = -xb[8 * k + i];
                ++nflip;
                s |= (uint8_t)(1u << i);
            }
        }
        if (nflip % 2) {
            int imin = 0;
            float min = weight[8 * k] * xb[8 * k] * xb[8 * k];
            for (int i = 1; i < 8; ++i) {
                const float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                if (ax < min) {
                    min = ax;
                    imin = i;
                }
            }
            xval[8 * k + imin] = -xval[8 * k + imin];
            s ^= (uint8_t)(1u << imin);
        }
        signs[k] = s & 127;
    }

    float max = xval[0];
    for (int i = 1; i < 32; ++i) max = DMAX(max, xval[i]);
    if (max < 1e-8f) {
        for (int i = 0; i < 8; ++i) qs[i] = 0;
        for (int i = 0; i < 4; ++i) sign_scale[i] = 0;
        return 0.0f;
    }

    for (int k = 0; k < 8; ++k) on_grid[k] = true;
    float best = 0;
    float scale = max / 15.0f;
    for (int is = -15; is <= 15; ++is) {
        const float id = (15.0f + (float)is * 0.2f) / max;
        const float this_scale = 1.0f / id;
        for (int k = 0; k < 8; ++k) {
            for (int i = 0; i < 4; ++i) {
                int l = nearest_int_d(0.5f * (id * xval[4 * k + i] - 1));
                Laux[4 * k + i] = (int8_t)dmax_i(0, dmin_i(7, l));
            }
            uint16_t u = 0;
            for (int i = 0; i < 4; ++i) u |= (uint16_t)((uint16_t)Laux[4 * k + i] << (3 * i));
            const int grid_index = map[u];
            on_aux[k] = true;
            if (grid_index < 0) {
                on_aux[k] = false;
                const uint16_t * nb = neighbours - map[u] - 1;
                find_best_iq3(nb, grid, xval + 4 * k, waux + 4 * k, this_scale, Laux + 4 * k);
            }
        }
        float sumqx = 0;
        float sumq2 = 0;
        for (int i = 0; i < 32; ++i) {
            const float w = weight[i];
            const float q = (float)(2 * Laux[i] + 1);
            sumqx += w * xval[i] * q;
            sumq2 += w * q * q;
        }
        if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
            scale = FDIV(sumqx, sumq2);
            best = scale * sumqx;
            for (int i = 0; i < 32; ++i) L[i] = Laux[i];
            for (int k = 0; k < 8; ++k) on_grid[k] = on_aux[k];
        }
    }

    int n_not_ongrid = 0;
    for (int k = 0; k < 8; ++k)
        if (!on_grid[k]) ++n_not_ongrid;
    if (n_not_ongrid > 0 && scale > 0) {
        const float id = FDIV(1.0f, scale);
        for (int k = 0; k < 8; ++k) {
            if (on_grid[k]) continue;
            uint16_t u = 0;
            for (int i = 0; i < 4; ++i) {
                int l = nearest_int_d(0.5f * (id * xval[4 * k + i] - 1));
                l = dmax_i(0, dmin_i(7, l));
                u |= (uint16_t)((uint16_t)l << (3 * i));
                L[4 * k + i] = (int8_t)l;
            }
            const int grid_index = map[u];
            if (grid_index < 0) {
                const uint16_t * nb = neighbours - map[u] - 1;
                find_best_iq3(nb, grid, xval + 4 * k, waux + 4 * k, scale, L + 4 * k);
            }
        }
        float sumqx = 0;
        float sumq2 = 0;
        for (int i = 0; i < 32; ++i) {
            const float w = weight[i];
            const float q = (float)(2 * L[i] + 1);
            sumqx += w * xval[i] * q;
            sumq2 += w * q * q;
        }
        if (sumq2 > 0) scale = FDIV(sumqx, sumq2);
    }
    if (scale < 0) {
        scale = -scale;
        for (int k = 0; k < 4; ++k) signs[k] = (uint8_t)((~signs[k]) & 127);
    }

    for (int k = 0; k < 8; ++k) {
        uint16_t u = 0;
        for (int i = 0; i < 4; ++i) u |= (uint16_t)((uint16_t)L[4 * k + i] << (3 * i));
        const int grid_index = map[u];
        qs[k] = (uint8_t)grid_index;
    }
    uint32_t w = (uint32_t)signs[0] | ((uint32_t)signs[1] << 7) |
                 ((uint32_t)signs[2] << 14) | ((uint32_t)signs[3] << 21);
    for (int b = 0; b < 4; ++b) sign_scale[b] = (uint8_t)((w >> (8 * b)) & 255);
    return scale;
}

struct GgmlArgs {
    const float * src;
    void * dst;
    const float * imp;
    const void * grid;
    const int * map;
    const uint16_t * neighbours;
    int64_t nrow;
    int64_t n_per_row;
    int64_t nblocks_per_row;
    int64_t total_blocks;
};

__global__ void __launch_bounds__(kWgThreads) ggml_iq2_xxs_kernel(GgmlArgs a) {
    const int lane = threadIdx.x & 7;
    const int bg = threadIdx.x >> 3;
    const int64_t b = (int64_t)blockIdx.x * kBlocksPerWg + bg;
    const bool valid = b < a.total_blocks;

    __shared__ float sigma[kBlocksPerWg];
    __shared__ float group_scale[kBlocksPerWg][kGroups];
    __shared__ float block_d[kBlocksPerWg];

    const int64_t row = valid ? b / a.nblocks_per_row : 0;
    const int64_t blk = valid ? b % a.nblocks_per_row : 0;
    const float * xbl = valid ? a.src + row * a.n_per_row + blk * kBlock : nullptr;
    const float * qw = valid && a.imp ? a.imp + blk * kBlock + lane * kGroup : nullptr;
    uint8_t * qsblock = valid ? (uint8_t *)a.dst + b * 66 + 2 : nullptr;

    if (valid && lane == 0) {
        float sum = 0;
#pragma unroll 1
        for (int i = 0; i < kBlock; ++i) sum += xbl[i] * xbl[i];
        sigma[bg] = sum / (float)kBlock;
    }
    __syncthreads();

    if (valid) {
        const float s = iq2_group(xbl + lane * kGroup, qw, sigma[bg],
                                  (const uint64_t *)a.grid, a.map, a.neighbours,
                                  qsblock + lane * 8);
        group_scale[bg][lane] = s;
    }
    __syncthreads();

    if (valid && lane == 0) {
        float m = 0;
        for (int i = 0; i < kGroups; ++i) m = DMAX(m, group_scale[bg][i]);
        if (m > 0) {
            block_d[bg] = FDIV(m, 31.0f);
            *(uint16_t *)((uint8_t *)a.dst + b * 66) = fp16_d(block_d[bg]);
        } else {
            block_d[bg] = 0.0f;
            *(uint16_t *)((uint8_t *)a.dst + b * 66) = fp16_d(0.0f);
        }
    }
    __syncthreads();

    if (valid && block_d[bg] > 0) {
        uint8_t * qs = qsblock + lane * 8;
        const float id = FDIV(1.0f, block_d[bg]);
        int l = nearest_int_d(0.5f * (id * group_scale[bg][lane] - 1));
        l = dmax_i(0, dmin_i(15, l));
        qs[7] |= (uint8_t)(l << 4);
    }
}

__global__ void __launch_bounds__(kWgThreads) ggml_iq3_xxs_kernel(GgmlArgs a) {
    const int lane = threadIdx.x & 7;
    const int bg = threadIdx.x >> 3;
    const int64_t b = (int64_t)blockIdx.x * kBlocksPerWg + bg;
    const bool valid = b < a.total_blocks;

    __shared__ float sigma[kBlocksPerWg];
    __shared__ float group_scale[kBlocksPerWg][kGroups];
    __shared__ float block_d[kBlocksPerWg];

    const int64_t row = valid ? b / a.nblocks_per_row : 0;
    const int64_t blk = valid ? b % a.nblocks_per_row : 0;
    const float * xbl = valid ? a.src + row * a.n_per_row + blk * kBlock : nullptr;
    const float * qw = valid && a.imp ? a.imp + blk * kBlock + lane * kGroup : nullptr;
    uint8_t * qsblock = valid ? (uint8_t *)a.dst + b * 98 + 2 : nullptr;

    if (valid && lane == 0) {
        float sum = 0;
#pragma unroll 1
        for (int i = 0; i < kBlock; ++i) sum += xbl[i] * xbl[i];
        sigma[bg] = (2.0f * sum) / (float)kBlock;
    }
    __syncthreads();

    if (valid) {
        const float s = iq3_group(xbl + lane * kGroup, qw, sigma[bg],
                                  (const uint32_t *)a.grid, a.map, a.neighbours,
                                  qsblock + lane * 8, qsblock + 64 + lane * 4);
        group_scale[bg][lane] = s;
    }
    __syncthreads();

    if (valid && lane == 0) {
        float m = 0;
        for (int i = 0; i < kGroups; ++i) m = DMAX(m, group_scale[bg][i]);
        if (m > 0) {
            block_d[bg] = FDIV(m, 31.0f);
            *(uint16_t *)((uint8_t *)a.dst + b * 98) = fp16_d(block_d[bg] * 1.0125f);
        } else {
            block_d[bg] = 0.0f;
            *(uint16_t *)((uint8_t *)a.dst + b * 98) = fp16_d(0.0f);
        }
    }
    __syncthreads();

    if (valid && block_d[bg] > 0) {
        uint8_t * sp = qsblock + 64 + lane * 4;
        const float id = FDIV(1.0f, block_d[bg]);
        int l = nearest_int_d(0.5f * (id * group_scale[bg][lane] - 1));
        l = dmax_i(0, dmin_i(15, l));
        sp[3] |= (uint8_t)(l << 4);
    }
}

template <typename T>
struct DevBuf {
    T * p = nullptr;
    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { if (p) (void)hipFree(p); }
    int alloc(size_t n) {
        if (n == 0) return 0;
        return hipMalloc((void **)&p, n) == hipSuccess ? 0 : -1;
    }
};

struct PinnedBuf {
    void * p = nullptr;
    PinnedBuf() = default;
    PinnedBuf(const PinnedBuf&) = delete;
    PinnedBuf& operator=(const PinnedBuf&) = delete;
    ~PinnedBuf() { if (p) (void)hipHostFree(p); }
    int alloc(size_t n) {
        if (n == 0) return 0;
        return hipHostMalloc(&p, n, hipHostMallocDefault) == hipSuccess ? 0 : -1;
    }
};

int card_for_thread() {
    static thread_local int card = -1;
    if (card < 0) {
        const int n = gpu_device_count();
        if (n <= 0) return -1;
        static std::atomic<unsigned> turn{0};
        card = gpu_device((int)(turn.fetch_add(1, std::memory_order_relaxed) % (unsigned)n));
    }
    return card;
}

bool fail(std::string * why, const char * msg, hipError_t e = hipSuccess) {
    if (why) {
        *why = msg;
        if (e != hipSuccess) *why += ": " + std::string(hipGetErrorString(e));
    }
    return false;
}

}  /* namespace */

bool gpu_available() { return gpu_enabled() && gpu_device_count() > 0; }

/* The entry the host wrapper calls. */
int quantize_gpu(enum ggml_type type, const float * src, void * dst, int64_t nrow,
                 int64_t n_per_row, const float * imp, std::string * why) {
    if (!gpu_available()) return RAD_E_UNSUPPORTED;
    if (type != GGML_TYPE_IQ2_XXS && type != GGML_TYPE_IQ3_XXS) return RAD_E_UNSUPPORTED;
    if (type == GGML_TYPE_IQ2_XXS && !imp) return RAD_E_UNSUPPORTED;
    if (nrow <= 0 || n_per_row <= 0 || n_per_row % kBlock) return RAD_E_UNSUPPORTED;

    SearchTables tb{};
    if (!ggml_search_tables(type, &tb) || !tb.grid || !tb.map || !tb.neighbours)
        return RAD_E_UNSUPPORTED;

    const int card = card_for_thread();
    if (card < 0) return RAD_E_UNSUPPORTED;
    hipError_t e = hipSetDevice(card);
    if (e != hipSuccess) { fail(why, "selecting card", e); return RAD_E_DEVICE; }

    const int64_t blocks = nrow * (n_per_row / kBlock);
    const int64_t row_bytes = (int64_t)ggml_row_size(type, n_per_row);
    const size_t in_bytes = (size_t)(nrow * n_per_row) * sizeof(float);
    const size_t out_bytes = (size_t)(nrow * row_bytes);
    const size_t imp_bytes = (size_t)n_per_row * sizeof(float);
    const size_t grid_bytes = (size_t)tb.grid_size * (type == GGML_TYPE_IQ2_XXS ? 8 : 4);
    const size_t map_bytes = (size_t)tb.map_size * sizeof(int);
    const size_t neigh_bytes = (size_t)tb.neighbours_size * sizeof(uint16_t);

    PinnedBuf pin_in, pin_out;
    DevBuf<uint8_t> d_in, d_out, d_grid, d_map, d_neigh, d_imp;
    if (pin_in.alloc(in_bytes) || pin_out.alloc(out_bytes) ||
        d_in.alloc(in_bytes) || d_out.alloc(out_bytes) || d_grid.alloc(grid_bytes) ||
        d_map.alloc(map_bytes) || d_neigh.alloc(neigh_bytes) ||
        (imp && d_imp.alloc(imp_bytes))) {
        fail(why, "allocating card buffers");
        return RAD_E_UNSUPPORTED;   /* host fallback */
    }

    std::memcpy(pin_in.p, src, in_bytes);
    hipStream_t stream = nullptr;
    e = hipStreamCreate(&stream);
    if (e != hipSuccess) { fail(why, "creating stream", e); return RAD_E_DEVICE; }

    auto fail_stream = [&](const char * msg) {
        fail(why, msg);
        (void)hipStreamDestroy(stream);
        return RAD_E_DEVICE;
    };

    e = hipMemcpyAsync(d_in.p, pin_in.p, in_bytes, hipMemcpyHostToDevice, stream);
    if (e != hipSuccess) return fail_stream("copying rows to card");
    if (imp) {
        e = hipMemcpyAsync(d_imp.p, imp, imp_bytes, hipMemcpyHostToDevice, stream);
        if (e != hipSuccess) return fail_stream("copying imatrix");
    }
    e = hipMemcpyAsync(d_grid.p, tb.grid, grid_bytes, hipMemcpyHostToDevice, stream);
    if (e != hipSuccess) return fail_stream("copying grid");
    e = hipMemcpyAsync(d_map.p, tb.map, map_bytes, hipMemcpyHostToDevice, stream);
    if (e != hipSuccess) return fail_stream("copying grid map");
    e = hipMemcpyAsync(d_neigh.p, tb.neighbours, neigh_bytes, hipMemcpyHostToDevice, stream);
    if (e != hipSuccess) return fail_stream("copying neighbours");

    GgmlArgs a{};
    a.src = (const float *)d_in.p;
    a.dst = d_out.p;
    a.imp = imp ? (const float *)d_imp.p : nullptr;
    a.grid = d_grid.p;
    a.map = (const int *)d_map.p;
    a.neighbours = (const uint16_t *)d_neigh.p;
    a.nrow = nrow;
    a.n_per_row = n_per_row;
    a.nblocks_per_row = n_per_row / kBlock;
    a.total_blocks = blocks;

    const int64_t wg = (blocks + kBlocksPerWg - 1) / kBlocksPerWg;
    if (type == GGML_TYPE_IQ2_XXS)
        hipLaunchKernelGGL(ggml_iq2_xxs_kernel, dim3((unsigned)wg), dim3(kWgThreads), 0, stream, a);
    else
        hipLaunchKernelGGL(ggml_iq3_xxs_kernel, dim3((unsigned)wg), dim3(kWgThreads), 0, stream, a);
    e = hipGetLastError();
    if (e != hipSuccess) return fail_stream("launching kernel");

    e = hipMemcpyAsync(pin_out.p, d_out.p, out_bytes, hipMemcpyDeviceToHost, stream);
    if (e != hipSuccess) return fail_stream("copying blocks back");
    e = hipStreamSynchronize(stream);
    if (e != hipSuccess) return fail_stream("waiting for card");
    e = hipStreamDestroy(stream);
    if (e != hipSuccess) { fail(why, "destroying stream", e); return RAD_E_DEVICE; }
    std::memcpy(dst, pin_out.p, out_bytes);
    return RAD_OK;
}

}  /* namespace lq::ggml */
