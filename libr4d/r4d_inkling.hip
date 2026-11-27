/* r4d_inkling.hip -- Inkling 2 device-only fusions:
 *   - loop_ctl: fuses all 12 core controller ops (two RMS norms, down projection into f32 scratch,
 *     gain/forget projections, tanh softcap, and gated blend) into two chained launches at M <= 16.
 *   - loop_ctl_blend: fuses the 7 elementwise ops after gain/forget GEMMs at M > 16.
 *   - plane_copy: copies the first M = ref->shape[0] rows of src to dst without host-side fills
 *     of ones, supporting rank-2 [M, n] and rank-3 [1, MT, n] planes.
 *   - kv_clip_norm_store: fuses k l2_clip + v weightless rmsnorm + kv_store (bf16/i8) in 1 wave/head.
 *   - qr_l2_clip: fuses q l2_clip + r l2_clip (with optional position tau) in 1 launch.
 *   - attn_res_mix3 / attn_res_mix4: multi-query block attention residuals reading states/cur once.
 */
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_launch.h"
#include "r4d_plugin.h"
#include "rad_plugin.h"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

#define R4D_INK_CHK() do { hipError_t e_ = hipGetLastError();                        \
                           if (e_ != hipSuccess) return R4D_NO(RAD_E_DEVICE); } while (0)

constexpr long long kLoopCtlMaxM    = 16;
constexpr long long kLoopCtlMaxN    = 8192;
constexpr long long kLoopCtlMaxRank = 256;

__device__ __forceinline__ uint4 ink_ld_u4(const void* p) {
#ifdef __HIP__
    return *(const uint4*)p;
#else
    uint4 v;
    __builtin_memcpy(&v, p, sizeof v);
    return v;
#endif
}

__device__ __forceinline__ void ink_st_u4(void* p, uint4 v) {
#ifdef __HIP__
    *(uint4*)p = v;
#else
    __builtin_memcpy(p, &v, sizeof v);
#endif
}

/* ---- loop_ctl: Kernel 1 (RMS norms into LDS + down projection to f32 scratch) ---------------- */

constexpr int kDownWaves = 8;
constexpr int kDownThr   = kDownWaves * 32;   /* 256 threads */
constexpr int kDownPre   = 4;                 /* weight chunks a lane loads ahead of the norms */

template <bool VEC8>
__device__ __forceinline__ void loop_ctl_down_body(
    const uint16_t* __restrict__ x, long long x_ld,
    const uint16_t* __restrict__ msg, long long msg_ld,
    const uint16_t* __restrict__ w_dx, long long wdx_ld,
    float* __restrict__ scratch, int n, int rank, float eps) {
    __shared__ float s_ws[2][kDownWaves];
    R4D_DYN_LDS(uint4, s_nm_u4);
    uint16_t* s_nx = (uint16_t*)s_nm_u4;
    uint16_t* s_nm = s_nx + n;

    const int m = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const uint16_t* xr = x + (long long)m * x_ld;
    const uint16_t* mr = msg + (long long)m * msg_ld;

    /* The wave's first kDownPre weight chunks load before the norms, which they do not depend on,
     * so their latency hides behind the two passes over x and msg. */
    const int r = (int)blockIdx.y * kDownWaves + wave;
    const uint16_t* wx = w_dx + (long long)(r < rank ? r : 0) * wdx_ld;
    const uint16_t* wm = wx + n;
    uint4 pw_x[kDownPre], pw_m[kDownPre];
    if constexpr (VEC8) {
#pragma unroll
        for (int p = 0; p < kDownPre; ++p) {
            const int c = lane * 8 + p * 256;
            if (r < rank && c < n) {
                pw_x[p] = ink_ld_u4(wx + c);
                pw_m[p] = ink_ld_u4(wm + c);
            }
        }
    }

    float sx = 0.0f, sm = 0.0f;
    if constexpr (VEC8) {
        for (int c = tid * 8; c < n; c += kDownThr * 8) {
            const uint4 vx = ink_ld_u4(xr + c);
            const uint4 vm = ink_ld_u4(mr + c);
            const uint32_t ax[4] = { vx.x, vx.y, vx.z, vx.w };
            const uint32_t am[4] = { vm.x, vm.y, vm.z, vm.w };
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float x0 = bf16w_lo(ax[p]), x1 = bf16w_hi(ax[p]);
                const float m0 = bf16w_lo(am[p]), m1 = bf16w_hi(am[p]);
                sx += x0 * x0 + x1 * x1;
                sm += m0 * m0 + m1 * m1;
            }
        }
    } else {
        for (int c = tid; c < n; c += kDownThr) {
            const float xv = bf16_to_f32(xr[c]);
            const float mv = bf16_to_f32(mr[c]);
            sx += xv * xv;
            sm += mv * mv;
        }
    }
    sx = r4d_wave_sum(sx);
    sm = r4d_wave_sum(sm);
    if (lane == 0) {
        s_ws[0][wave] = sx;
        s_ws[1][wave] = sm;
    }
    lds_barrier();

    float sx_tot = 0.0f, sm_tot = 0.0f;
#pragma unroll
    for (int w = 0; w < kDownWaves; ++w) {
        sx_tot += s_ws[0][w];
        sm_tot += s_ws[1][w];
    }
    const float rx = rsqrtf(sx_tot / (float)n + eps);
    const float rm = rsqrtf(sm_tot / (float)n + eps);
    if constexpr (VEC8) {
        for (int c = tid * 8; c < n; c += kDownThr * 8) {
            const uint4 vx = ink_ld_u4(xr + c);
            const uint4 vm = ink_ld_u4(mr + c);
            const uint32_t ax[4] = { vx.x, vx.y, vx.z, vx.w };
            const uint32_t am[4] = { vm.x, vm.y, vm.z, vm.w };
            uint32_t nx[4], nm[4];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                nx[p] = (uint32_t)f32_to_bf16(bf16w_lo(ax[p]) * rx) |
                        ((uint32_t)f32_to_bf16(bf16w_hi(ax[p]) * rx) << 16);
                nm[p] = (uint32_t)f32_to_bf16(bf16w_lo(am[p]) * rm) |
                        ((uint32_t)f32_to_bf16(bf16w_hi(am[p]) * rm) << 16);
            }
            ink_st_u4(s_nx + c, make_uint4(nx[0], nx[1], nx[2], nx[3]));
            ink_st_u4(s_nm + c, make_uint4(nm[0], nm[1], nm[2], nm[3]));
        }
    } else {
        for (int c = tid; c < n; c += kDownThr) {
            s_nx[c] = f32_to_bf16(bf16_to_f32(xr[c]) * rx);
            s_nm[c] = f32_to_bf16(bf16_to_f32(mr[c]) * rm);
        }
    }
    lds_barrier();

    if (r < rank) {
        float dot = 0.0f;
        if constexpr (VEC8) {
            auto acc = [&](int c, uint4 vw_x, uint4 vw_m) {
                const uint4 vx = ink_ld_u4(s_nx + c);
                const uint4 vm = ink_ld_u4(s_nm + c);
                const uint32_t ax[4] = { vw_x.x, vw_x.y, vw_x.z, vw_x.w };
                const uint32_t bx[4] = { vx.x, vx.y, vx.z, vx.w };
                const uint32_t am[4] = { vw_m.x, vw_m.y, vw_m.z, vw_m.w };
                const uint32_t bm[4] = { vm.x, vm.y, vm.z, vm.w };
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    dot += bf16w_lo(ax[p]) * bf16w_lo(bx[p]) +
                           bf16w_hi(ax[p]) * bf16w_hi(bx[p]) +
                           bf16w_lo(am[p]) * bf16w_lo(bm[p]) +
                           bf16w_hi(am[p]) * bf16w_hi(bm[p]);
                }
            };
#pragma unroll
            for (int p = 0; p < kDownPre; ++p) {
                const int c = lane * 8 + p * 256;
                if (c < n) acc(c, pw_x[p], pw_m[p]);
            }
            for (int c = lane * 8 + kDownPre * 256; c < n; c += 256)
                acc(c, ink_ld_u4(wx + c), ink_ld_u4(wm + c));
        } else {
            for (int c = lane; c < n; c += 32) {
                dot += bf16_to_f32(wx[c]) * bf16_to_f32(s_nx[c]);
                dot += bf16_to_f32(wm[c]) * bf16_to_f32(s_nm[c]);
            }
        }
        dot = r4d_wave_sum(dot);
        if (lane == 0) scratch[(long long)m * rank + r] = bf16_to_f32(f32_to_bf16(dot));
    }
}

template <bool VEC8, class CH>
__global__ __launch_bounds__(kDownThr) void loop_ctl_down_kernel(
    const uint16_t* __restrict__ x, long long x_ld,
    const uint16_t* __restrict__ msg, long long msg_ld,
    const uint16_t* __restrict__ w_dx, long long wdx_ld,
    float* __restrict__ scratch, int n, int rank, float eps) {
    if (CH::on) r4d_chain_wait();
    loop_ctl_down_body<VEC8>(x, x_ld, msg, msg_ld, w_dx, wdx_ld, scratch, n, rank, eps);
    if (CH::on) r4d_chain_done();
}

/* ---- loop_ctl: Kernel 2 (gain/forget projections + tanh + blend) ---------------------------- */

constexpr int kApplyCols  = 16;
constexpr int kApplyLanes = 8;
constexpr int kApplyThr   = kApplyCols * kApplyLanes;   /* 128 threads */

template <bool VEC64>
__device__ __forceinline__ void loop_ctl_apply_body(
    const uint16_t* __restrict__ x, long long x_ld,
    const uint16_t* __restrict__ msg, long long msg_ld,
    const uint16_t* __restrict__ wg, long long wg_ld,
    const uint16_t* __restrict__ wf, long long wf_ld,
    const float* __restrict__ scratch,
    uint16_t* __restrict__ y, long long y_ld,
    int n, int rank) {
    __shared__ float s_g[kLoopCtlMaxRank];
    const int m = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    for (int k = tid; k < rank; k += kApplyThr) s_g[k] = scratch[(long long)m * rank + k];
    lds_barrier();

    const int col = (int)blockIdx.y * kApplyCols + (tid / kApplyLanes);
    const int lk = tid & (kApplyLanes - 1);
    if (col >= n) return;

    const uint16_t* wgr = wg + (long long)col * wg_ld;
    const uint16_t* wfr = wf + (long long)col * wf_ld;
    float gp = 0.0f, fp = 0.0f;

    if constexpr (VEC64) {
        for (int k0 = 0; k0 < rank; k0 += 64) {
            const uint4 vg = ink_ld_u4(wgr + k0 + lk * 8);
            const uint4 vf = ink_ld_u4(wfr + k0 + lk * 8);
            const uint32_t g4[4] = { vg.x, vg.y, vg.z, vg.w };
            const uint32_t f4[4] = { vf.x, vf.y, vf.z, vf.w };
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int k = k0 + lk * 8 + p * 2;
                const float g0 = s_g[k + 0], g1 = s_g[k + 1];
                gp += bf16_to_f32((uint16_t)(g4[p] & 0xffffu)) * g0 +
                      bf16_to_f32((uint16_t)(g4[p] >> 16)) * g1;
                fp += bf16_to_f32((uint16_t)(f4[p] & 0xffffu)) * g0 +
                      bf16_to_f32((uint16_t)(f4[p] >> 16)) * g1;
            }
        }
    } else {
        for (int k = lk; k < rank; k += kApplyLanes) {
            const float gv = s_g[k];
            gp += bf16_to_f32(wgr[k]) * gv;
            fp += bf16_to_f32(wfr[k]) * gv;
        }
    }

    gp = r4d_xor_sum<kApplyLanes>(gp);
    fp = r4d_xor_sum<kApplyLanes>(fp);
    if (lk == 0) {
        const float gp_r = bf16_to_f32(f32_to_bf16(gp));
        const float fp_r = bf16_to_f32(f32_to_bf16(fp));
        const float gt = bf16_to_f32(f32_to_bf16(tanhf(gp_r)));
        const float ft = bf16_to_f32(f32_to_bf16(tanhf(fp_r)));
        const float xv = bf16_to_f32(x[(long long)m * x_ld + col]);
        const float mv = bf16_to_f32(msg[(long long)m * msg_ld + col]);
        y[(long long)m * y_ld + col] = f32_to_bf16((1.0f - ft) * xv + gt * mv);
    }
}

template <bool VEC64, class CH>
__global__ __launch_bounds__(kApplyThr) void loop_ctl_apply_kernel(
    const uint16_t* __restrict__ x, long long x_ld,
    const uint16_t* __restrict__ msg, long long msg_ld,
    const uint16_t* __restrict__ wg, long long wg_ld,
    const uint16_t* __restrict__ wf, long long wf_ld,
    const float* __restrict__ scratch,
    uint16_t* __restrict__ y, long long y_ld,
    int n, int rank) {
    if (CH::on) r4d_chain_wait();
    loop_ctl_apply_body<VEC64>(x, x_ld, msg, msg_ld, wg, wg_ld, wf, wf_ld, scratch, y, y_ld, n, rank);
    if (CH::on) r4d_chain_done();
}

/* ---- loop_ctl_blend: prefill elementwise fusion --------------------------------------------- */

template <bool VEC8, class CH>
__global__ __launch_bounds__(256) void loop_ctl_blend_kernel(
    const uint16_t* __restrict__ x, long long x_ld,
    const uint16_t* __restrict__ msg, long long msg_ld,
    const uint16_t* __restrict__ gp, long long gp_ld,
    const uint16_t* __restrict__ fp, long long fp_ld,
    uint16_t* __restrict__ y, long long y_ld,
    long long M, int n) {
    if (CH::on) r4d_chain_wait();
    if constexpr (VEC8) {
        const int c8 = (int)(blockIdx.x * 256u + threadIdx.x) * 8;
        if (c8 < n) {
            for (long long m = (long long)blockIdx.y; m < M; m += (long long)gridDim.y) {
                const uint4 vx  = ink_ld_u4(x + m * x_ld + c8);
                const uint4 vm  = ink_ld_u4(msg + m * msg_ld + c8);
                const uint4 vgp = ink_ld_u4(gp + m * gp_ld + c8);
                const uint4 vfp = ink_ld_u4(fp + m * fp_ld + c8);
                const uint32_t xs[4] = { vx.x, vx.y, vx.z, vx.w };
                const uint32_t ms[4] = { vm.x, vm.y, vm.z, vm.w };
                const uint32_t gs[4] = { vgp.x, vgp.y, vgp.z, vgp.w };
                const uint32_t fs[4] = { vfp.x, vfp.y, vfp.z, vfp.w };
                uint32_t out[4];
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const float gt0 = bf16_to_f32(f32_to_bf16(tanhf(bf16w_lo(gs[p]))));
                    const float ft0 = bf16_to_f32(f32_to_bf16(tanhf(bf16w_lo(fs[p]))));
                    const float y0  = (1.0f - ft0) * bf16w_lo(xs[p]) + gt0 * bf16w_lo(ms[p]);
                    const float gt1 = bf16_to_f32(f32_to_bf16(tanhf(bf16w_hi(gs[p]))));
                    const float ft1 = bf16_to_f32(f32_to_bf16(tanhf(bf16w_hi(fs[p]))));
                    const float y1  = (1.0f - ft1) * bf16w_hi(xs[p]) + gt1 * bf16w_hi(ms[p]);
                    out[p] = bf16x2_rne(y0, y1);
                }
                ink_st_u4(y + m * y_ld + c8, make_uint4(out[0], out[1], out[2], out[3]));
            }
        }
    } else {
        const int c = (int)(blockIdx.x * 256u + threadIdx.x);
        if (c < n) {
            for (long long m = (long long)blockIdx.y; m < M; m += (long long)gridDim.y) {
                const float xv = bf16_to_f32(x[m * x_ld + c]);
                const float mv = bf16_to_f32(msg[m * msg_ld + c]);
                const float gt = bf16_to_f32(f32_to_bf16(tanhf(bf16_to_f32(gp[m * gp_ld + c]))));
                const float ft = bf16_to_f32(f32_to_bf16(tanhf(bf16_to_f32(fp[m * fp_ld + c]))));
                y[m * y_ld + c] = f32_to_bf16((1.0f - ft) * xv + gt * mv);
            }
        }
    }
    if (CH::on) r4d_chain_done();
}

/* ---- plane_copy: row-count-guided copy for [M, n] and [1, MT, n] planes --------------------- */

template <bool VEC8, class CH>
__global__ __launch_bounds__(256) void plane_copy_kernel(
    const uint16_t* __restrict__ src, long long s_ld,
    uint16_t* __restrict__ dst, long long d_ld,
    long long M, int n) {
    if (CH::on) r4d_chain_wait();
    if constexpr (VEC8) {
        const int c8 = (int)(blockIdx.x * 256u + threadIdx.x) * 8;
        if (c8 < n) {
            for (long long m = (long long)blockIdx.y; m < M; m += (long long)gridDim.y) {
                const uint4 v = ink_ld_u4(src + m * s_ld + c8);
                ink_st_u4(dst + m * d_ld + c8, v);
            }
        }
    } else {
        const int c = (int)(blockIdx.x * 256u + threadIdx.x);
        if (c < n) {
            for (long long m = (long long)blockIdx.y; m < M; m += (long long)gridDim.y)
                dst[m * d_ld + c] = src[m * s_ld + c];
        }
    }
    if (CH::on) r4d_chain_done();
}

bool plane_view(const RadTensor* t, long long n, long long* rows_out, long long* ld_out) {
    if (!t || !t->data || t->dtype != RAD_BF16) return false;
    if (t->rank == 2) {
        if (t->shape[1] != n || t->stride[1] != 1 || t->stride[0] < n) return false;
        *rows_out = t->shape[0];
        *ld_out = t->stride[0];
        return true;
    }
    if (t->rank == 3) {
        if (t->shape[0] != 1 || t->shape[2] != n || t->stride[2] != 1 || t->stride[1] < n)
            return false;
        *rows_out = t->shape[1];
        *ld_out = t->stride[1];
        return true;
    }
    return false;
}

/* ---- kv_clip_norm_store: K l2_clip + V weightless rmsnorm + paged KV store ------------------ */

template <bool KV_I8, int MAX_E>
__device__ __forceinline__ void kv_clip_norm_store_body(
    const uint16_t* __restrict__ k, long long k_ld,
    const uint16_t* __restrict__ v, long long v_ld,
    const int32_t* __restrict__ slot,
    void* __restrict__ kv, long long kv_s0, long long kv_s1, long long kv_s2,
    long long T, uint32_t head_dim, uint32_t n_blocks, uint32_t block_size,
    float threshold, float eps) {
    const int lane = (int)threadIdx.x;
    const uint32_t h = (uint32_t)blockIdx.x;
    const float inv_n = 1.0f / (float)head_dim;

    for (long long t = (long long)blockIdx.y; t < T; t += (long long)gridDim.y) {
        const int32_t s = slot[t];
        if (s < 0 || (uint32_t)s / block_size >= n_blocks) continue;

        const uint16_t* kr = k + t * k_ld + (long long)h * head_dim;
        const uint16_t* vr = v + t * v_ld + (long long)h * head_dim;
        float kv_f[MAX_E], vv_f[MAX_E];
        float ss_k = 0.0f, ss_v = 0.0f;
#pragma unroll
        for (int e = 0; e < MAX_E; ++e) {
            const int d = lane + e * 32;
            const bool active = (uint32_t)d < head_dim;
            const float kf = active ? bf16_to_f32(kr[d]) : 0.0f;
            const float vf = active ? bf16_to_f32(vr[d]) : 0.0f;
            kv_f[e] = kf;
            vv_f[e] = vf;
            ss_k += kf * kf;
            ss_v += vf * vf;
        }
        ss_k = r4d_wave_sum(ss_k);
        ss_v = r4d_wave_sum(ss_v);
        const float fk = fminf(1.0f, threshold / (sqrtf(ss_k) + 1e-6f));
        const float rv = rsqrtf(ss_v * inv_n + eps);

        const uint32_t blk = (uint32_t)s / block_size;
        const uint32_t off = (uint32_t)s % block_size;
        const size_t base = (size_t)blk * (size_t)kv_s0 + (size_t)h * (size_t)kv_s1 +
                            (size_t)off * (size_t)kv_s2;

        if constexpr (!KV_I8) {
            uint16_t* dst = (uint16_t*)kv + base;
#pragma unroll
            for (int e = 0; e < MAX_E; ++e) {
                const int d = lane + e * 32;
                if ((uint32_t)d < head_dim) {
                    dst[d]            = f32_to_bf16(kv_f[e] * fk);
                    dst[head_dim + d] = f32_to_bf16(vv_f[e] * rv);
                }
            }
        } else {
            float km = 0.0f, vm = 0.0f;
#pragma unroll
            for (int e = 0; e < MAX_E; ++e) {
                const int d = lane + e * 32;
                if ((uint32_t)d < head_dim) {
                    kv_f[e] = bf16_to_f32(f32_to_bf16(kv_f[e] * fk));
                    vv_f[e] = bf16_to_f32(f32_to_bf16(vv_f[e] * rv));
                    km = fmaxf(km, fabsf(kv_f[e]));
                    vm = fmaxf(vm, fabsf(vv_f[e]));
                }
            }
            km = r4d_wave_max(km);
            vm = r4d_wave_max(vm);
            const _Float16 k16 = (_Float16)fminf(km / 127.0f, 515.0f);
            const _Float16 v16 = (_Float16)fminf(vm / 127.0f, 515.0f);
            const float ksc = (float)k16;
            const float vsc = (float)v16;
            uint8_t* dst = (uint8_t*)kv + base;
#pragma unroll
            for (int e = 0; e < MAX_E; ++e) {
                const int d = lane + e * 32;
                if ((uint32_t)d < head_dim) {
                    const float kr_f = ksc > 0.0f ? rintf(kv_f[e] / ksc) : 0.0f;
                    const float vr_f = vsc > 0.0f ? rintf(vv_f[e] / vsc) : 0.0f;
                    dst[d]            = (uint8_t)(int8_t)(int)fminf(fmaxf(kr_f, -127.0f), 127.0f);
                    dst[head_dim + d] = (uint8_t)(int8_t)(int)fminf(fmaxf(vr_f, -127.0f), 127.0f);
                }
            }
            if (lane == 0) {
                *(uint16_t*)(dst + 2 * head_dim)     = __builtin_bit_cast(uint16_t, k16);
                *(uint16_t*)(dst + 2 * head_dim + 2) = __builtin_bit_cast(uint16_t, v16);
            }
        }
    }
}

template <bool KV_I8, int MAX_E, class CH>
__global__ __launch_bounds__(32) void kv_clip_norm_store_kernel(
    const uint16_t* __restrict__ k, long long k_ld,
    const uint16_t* __restrict__ v, long long v_ld,
    const int32_t* __restrict__ slot,
    void* __restrict__ kv, long long kv_s0, long long kv_s1, long long kv_s2,
    long long T, uint32_t head_dim, uint32_t n_blocks, uint32_t block_size,
    float threshold, float eps) {
    if (CH::on) r4d_chain_wait();
    kv_clip_norm_store_body<KV_I8, MAX_E>(k, k_ld, v, v_ld, slot, kv, kv_s0, kv_s1, kv_s2,
                                          T, head_dim, n_blocks, block_size, threshold, eps);
    if (CH::on) r4d_chain_done();
}

/* ---- qr_l2_clip: fused Q and R per-group L2 clip with optional position tau ----------------- */

__device__ __forceinline__ void qr_l2_clip_body(
    const uint16_t* __restrict__ xq, long long xq_ld, const uint16_t* __restrict__ wq,
    const uint16_t* __restrict__ xr, long long xr_ld, const uint16_t* __restrict__ wr,
    uint16_t* __restrict__ yq, long long yq_ld,
    uint16_t* __restrict__ yr, long long yr_ld,
    const int32_t* __restrict__ pos,
    long long T, int gq_cnt, int gq_sz, float thrq, int gr_sz, float thrr,
    float inv_floor, float alpha) {
    const int lane = (int)threadIdx.x;
    const int g_idx = (int)blockIdx.x;
    const bool is_q = g_idx < gq_cnt;
    const int g     = is_q ? g_idx : (g_idx - gq_cnt);
    const uint16_t* const src = is_q ? xq : xr;
    const long long s_ld      = is_q ? xq_ld : xr_ld;
    const uint16_t* const w   = (is_q ? wq : wr) + (long long)g * (is_q ? gq_sz : gr_sz);
    uint16_t* const dst       = is_q ? yq : yr;
    const long long d_ld      = is_q ? yq_ld : yr_ld;
    const int grp             = is_q ? gq_sz : gr_sz;
    const float thr           = is_q ? thrq : thrr;
    const long long off_c     = (long long)g * grp;

    for (long long t = (long long)blockIdx.y; t < T; t += (long long)gridDim.y) {
        float tau = 1.0f;
        if (pos && alpha != 0.0f) {
            const float arg = fmaxf(1.0f, ((float)pos[t] + 1.0f) * inv_floor);
            tau = 1.0f + alpha * logf(arg);
        }
        const uint16_t* srow = src + t * s_ld + off_c;
        uint16_t* drow       = dst + t * d_ld + off_c;
        if (grp <= 256) {
            float vals[8];
            float ss = 0.0f;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int i = lane + e * 32;
                const float v = i < grp ? bf16_to_f32(srow[i]) * bf16_to_f32(w[i]) : 0.0f;
                vals[e] = v;
                ss += v * v;
            }
            ss = r4d_wave_sum(ss);
            const float sc = fminf(1.0f, thr / (sqrtf(ss) + 1e-6f)) * tau;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int i = lane + e * 32;
                if (i < grp) drow[i] = f32_to_bf16(vals[e] * sc);
            }
        } else {
            float ss = 0.0f;
            for (int i = lane; i < grp; i += 32) {
                const float v = bf16_to_f32(srow[i]) * bf16_to_f32(w[i]);
                ss += v * v;
            }
            ss = r4d_wave_sum(ss);
            const float sc = fminf(1.0f, thr / (sqrtf(ss) + 1e-6f)) * tau;
            for (int i = lane; i < grp; i += 32) {
                const float v = bf16_to_f32(srow[i]) * bf16_to_f32(w[i]);
                drow[i] = f32_to_bf16(v * sc);
            }
        }
    }
}

template <class CH>
__global__ __launch_bounds__(32) void qr_l2_clip_kernel(
    const uint16_t* __restrict__ xq, long long xq_ld, const uint16_t* __restrict__ wq,
    const uint16_t* __restrict__ xr, long long xr_ld, const uint16_t* __restrict__ wr,
    uint16_t* __restrict__ yq, long long yq_ld,
    uint16_t* __restrict__ yr, long long yr_ld,
    const int32_t* __restrict__ pos,
    long long T, int gq_cnt, int gq_sz, float thrq, int gr_sz, float thrr,
    float inv_floor, float alpha) {
    if (CH::on) r4d_chain_wait();
    qr_l2_clip_body(xq, xq_ld, wq, xr, xr_ld, wr, yq, yq_ld, yr, yr_ld, pos,
                    T, gq_cnt, gq_sz, thrq, gr_sz, thrr, inv_floor, alpha);
    if (CH::on) r4d_chain_done();
}

/* ---- attn_res_mix3 / attn_res_mix4: multi-output AttnRes sharing states & cur reads --------- */

constexpr int kMixThr      = 256;
constexpr int kMixWaves    = kMixThr / 32;
constexpr int kMixFastK    = 5;
constexpr int kMixMaxK     = 16;
constexpr int kMixMaxElem  = 12;   /* 12 * 256 = 3072 */
constexpr int kMixVec8Elem = 2;    /* 2 * 8 * 256 = 4096 */

__device__ __forceinline__ uint16_t ink_tobf(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

template <int NQ>
struct MultiMixArgs {
    const uint16_t* st;
    long long st_ps, st_rs;
    const uint16_t* cur;
    long long cur_rs;
    const float* q[NQ];
    const void*  w[NQ];
    uint16_t*    y[NQ];
    long long    y_rs[NQ];
    int M, n, nS, has_w, w_bf16;
    float eps;
};

template <int NQ>
__device__ __forceinline__ float mix_load_w(const MultiMixArgs<NQ>& A, int qi, int c) {
    return A.w_bf16 ? bf16_to_f32(((const uint16_t*)A.w[qi])[c]) : ((const float*)A.w[qi])[c];
}

template <int NQ>
__device__ __forceinline__ float mix_val(const MultiMixArgs<NQ>& A, int t, int i, int c) {
    return i < A.nS ? bf16_to_f32(A.st[i * A.st_ps + t * A.st_rs + c])
                    : bf16_to_f32(A.cur[t * A.cur_rs + c]);
}

template <int NQ>
__device__ __forceinline__ void attn_res_multi_fast_body(MultiMixArgs<NQ> A) {
    constexpr int kRedCols = kMixFastK + NQ * kMixFastK;
    __shared__ float s_red[kMixWaves][kRedCols];
    __shared__ float s_pr[NQ][kMixFastK];
    __shared__ float s_wsm[NQ][kMixWaves];
    __shared__ float s_rq[NQ];

    const int t = (int)blockIdx.x, tid = (int)threadIdx.x;
    if (t >= A.M) return;
    const int lane = tid & 31, wave = tid >> 5;
    const int K = A.nS + 1, n = A.n;

    float val[kMixFastK][kMixMaxElem];
    float ss[kMixFastK];
    float dot[NQ][kMixFastK];
#pragma unroll
    for (int i = 0; i < kMixFastK; ++i) {
        ss[i] = 0.0f;
#pragma unroll
        for (int q = 0; q < NQ; ++q) dot[q][i] = 0.0f;
    }

#pragma unroll
    for (int e = 0; e < kMixMaxElem; ++e) {
        const int c = tid + e * kMixThr;
        if (c < n) {
            float qv[NQ];
#pragma unroll
            for (int q = 0; q < NQ; ++q) qv[q] = A.q[q][c];
#pragma unroll
            for (int i = 0; i < kMixFastK; ++i) {
                if (i < K) {
                    const float x = mix_val(A, t, i, c);
                    val[i][e] = x;
                    ss[i] += x * x;
#pragma unroll
                    for (int q = 0; q < NQ; ++q) dot[q][i] += x * qv[q];
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < kMixFastK; ++i) {
        if (i < K) {
            const float rs = r4d_wave_sum(ss[i]);
            if (lane == 0) s_red[wave][i] = rs;
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float ds = r4d_wave_sum(dot[q][i]);
                if (lane == 0) s_red[wave][kMixFastK + q * kMixFastK + i] = ds;
            }
        }
    }
    lds_barrier();

    if (tid == 0) {
        float rstd[kMixFastK];
        for (int i = 0; i < K; ++i) {
            float tot = 0.0f;
            for (int w = 0; w < kMixWaves; ++w) tot += s_red[w][i];
            rstd[i] = 1.0f / sqrtf(tot / (float)n + A.eps);
        }
        const float inv_sqrt_n = 1.0f / sqrtf((float)n);
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float sc[kMixFastK];
            for (int i = 0; i < K; ++i) {
                float tot = 0.0f;
                for (int w = 0; w < kMixWaves; ++w) tot += s_red[w][kMixFastK + q * kMixFastK + i];
                sc[i] = tot * rstd[i] * inv_sqrt_n;
            }
            float mx = sc[0];
            for (int i = 1; i < K; ++i) mx = fmaxf(mx, sc[i]);
            float den = 0.0f;
            for (int i = 0; i < K; ++i) {
                sc[i] = expf(sc[i] - mx);
                den += sc[i];
            }
            const float inv_den = 1.0f / den;
            for (int i = 0; i < K; ++i) s_pr[q][i] = (sc[i] * inv_den) * rstd[i];
        }
    }
    lds_barrier();

#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        if (A.has_w) {
            float mval[kMixMaxElem];
            float sm = 0.0f;
#pragma unroll
            for (int e = 0; e < kMixMaxElem; ++e) {
                const int c = tid + e * kMixThr;
                if (c < n) {
                    float m = 0.0f;
#pragma unroll
                    for (int i = 0; i < kMixFastK; ++i)
                        if (i < K) m += s_pr[q][i] * val[i][e];
                    mval[e] = m;
                    sm += m * m;
                }
            }
            sm = r4d_wave_sum(sm);
            if (lane == 0) s_wsm[q][wave] = sm;
            lds_barrier();
            if (tid == 0) {
                float tot = 0.0f;
                for (int w = 0; w < kMixWaves; ++w) tot += s_wsm[q][w];
                s_rq[q] = 1.0f / sqrtf(tot / (float)n + A.eps);
            }
            lds_barrier();
            const float rq = s_rq[q];
#pragma unroll
            for (int e = 0; e < kMixMaxElem; ++e) {
                const int c = tid + e * kMixThr;
                if (c < n)
                    A.y[q][t * A.y_rs[q] + c] = ink_tobf(mval[e] * rq * mix_load_w(A, q, c));
            }
        } else {
#pragma unroll
            for (int e = 0; e < kMixMaxElem; ++e) {
                const int c = tid + e * kMixThr;
                if (c < n) {
                    float m = 0.0f;
#pragma unroll
                    for (int i = 0; i < kMixFastK; ++i)
                        if (i < K) m += s_pr[q][i] * val[i][e];
                    A.y[q][t * A.y_rs[q] + c] = ink_tobf(m);
                }
            }
        }
    }
}

template <int NQ>
__device__ __forceinline__ void attn_res_multi_gen_body(MultiMixArgs<NQ> A) {
    __shared__ float s_w[kMixWaves];
    __shared__ float s_rstd[kMixMaxK];
    __shared__ float s_pr[NQ][kMixMaxK];
    __shared__ float s_rq[NQ];

    const int t = (int)blockIdx.x, tid = (int)threadIdx.x;
    if (t >= A.M) return;
    const int lane = tid & 31, wave = tid >> 5;
    const int K = A.nS + 1, n = A.n;
    const float inv_sqrt_n = 1.0f / sqrtf((float)n);

    for (int i = 0; i < K; ++i) {
        float ss = 0.0f;
        for (int c = tid; c < n; c += kMixThr) {
            const float x = mix_val(A, t, i, c);
            ss += x * x;
        }
        ss = r4d_wave_sum(ss);
        if (lane == 0) s_w[wave] = ss;
        lds_barrier();
        if (tid == 0) {
            float tot = 0.0f;
            for (int w = 0; w < kMixWaves; ++w) tot += s_w[w];
            s_rstd[i] = 1.0f / sqrtf(tot / (float)n + A.eps);
        }
        lds_barrier();
    }

    for (int q = 0; q < NQ; ++q) {
        float sc[kMixMaxK];
        for (int i = 0; i < K; ++i) {
            float d = 0.0f;
            for (int c = tid; c < n; c += kMixThr)
                d += mix_val(A, t, i, c) * A.q[q][c];
            d = r4d_wave_sum(d);
            if (lane == 0) s_w[wave] = d;
            lds_barrier();
            if (tid == 0) {
                float tot = 0.0f;
                for (int w = 0; w < kMixWaves; ++w) tot += s_w[w];
                sc[i] = tot * s_rstd[i] * inv_sqrt_n;
            }
            lds_barrier();
        }
        if (tid == 0) {
            float mx = sc[0];
            for (int i = 1; i < K; ++i) mx = fmaxf(mx, sc[i]);
            float den = 0.0f;
            for (int i = 0; i < K; ++i) {
                sc[i] = expf(sc[i] - mx);
                den += sc[i];
            }
            const float inv_den = 1.0f / den;
            for (int i = 0; i < K; ++i) s_pr[q][i] = (sc[i] * inv_den) * s_rstd[i];
        }
        lds_barrier();
    }

    for (int q = 0; q < NQ; ++q) {
        if (A.has_w) {
            float sm = 0.0f;
            for (int c = tid; c < n; c += kMixThr) {
                float m = 0.0f;
                for (int i = 0; i < K; ++i) m += s_pr[q][i] * mix_val(A, t, i, c);
                sm += m * m;
            }
            sm = r4d_wave_sum(sm);
            if (lane == 0) s_w[wave] = sm;
            lds_barrier();
            if (tid == 0) {
                float tot = 0.0f;
                for (int w = 0; w < kMixWaves; ++w) tot += s_w[w];
                s_rq[q] = 1.0f / sqrtf(tot / (float)n + A.eps);
            }
            lds_barrier();
            const float rq = s_rq[q];
            for (int c = tid; c < n; c += kMixThr) {
                float m = 0.0f;
                for (int i = 0; i < K; ++i) m += s_pr[q][i] * mix_val(A, t, i, c);
                A.y[q][t * A.y_rs[q] + c] = ink_tobf(m * rq * mix_load_w(A, q, c));
            }
        } else {
            for (int c = tid; c < n; c += kMixThr) {
                float m = 0.0f;
                for (int i = 0; i < K; ++i) m += s_pr[q][i] * mix_val(A, t, i, c);
                A.y[q][t * A.y_rs[q] + c] = ink_tobf(m);
            }
        }
    }
}

template <int NQ>
__device__ __forceinline__ void attn_res_multi_vec8_body(MultiMixArgs<NQ> A) {
    constexpr int kRedCols = kMixFastK + NQ * kMixFastK;
    __shared__ float s_red[kMixWaves][kRedCols];
    __shared__ float s_pr[NQ][kMixFastK];
    __shared__ float s_wsm[NQ][kMixWaves];
    __shared__ float s_rq[NQ];

    const int t = (int)blockIdx.x, tid = (int)threadIdx.x;
    if (t >= A.M) return;
    const int lane = tid & 31, wave = tid >> 5;
    const int K = A.nS + 1, n = A.n;

    uint4 raw[kMixFastK][kMixVec8Elem];
    float ss[kMixFastK];
    float dot[NQ][kMixFastK];
#pragma unroll
    for (int i = 0; i < kMixFastK; ++i) {
        ss[i] = 0.0f;
#pragma unroll
        for (int q = 0; q < NQ; ++q) dot[q][i] = 0.0f;
    }

#pragma unroll
    for (int e = 0; e < kMixVec8Elem; ++e) {
        const int c8 = (tid + e * kMixThr) * 8;
        if (c8 < n) {
            float4 q0[NQ], q1[NQ];
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                q0[q] = *(const float4*)(A.q[q] + c8);
                q1[q] = *(const float4*)(A.q[q] + c8 + 4);
            }
#pragma unroll
            for (int i = 0; i < kMixFastK; ++i) {
                if (i < K) {
                    const uint16_t* ptr = (i < A.nS)
                        ? (A.st + i * A.st_ps + t * A.st_rs + c8)
                        : (A.cur + t * A.cur_rs + c8);
                    const uint4 v = ink_ld_u4(ptr);
                    raw[i][e] = v;
                    const float x0 = bf16w_lo(v.x), x1 = bf16w_hi(v.x);
                    const float x2 = bf16w_lo(v.y), x3 = bf16w_hi(v.y);
                    const float x4 = bf16w_lo(v.z), x5 = bf16w_hi(v.z);
                    const float x6 = bf16w_lo(v.w), x7 = bf16w_hi(v.w);
                    ss[i] += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3 +
                             x4 * x4 + x5 * x5 + x6 * x6 + x7 * x7;
#pragma unroll
                    for (int q = 0; q < NQ; ++q) {
                        dot[q][i] += x0 * q0[q].x + x1 * q0[q].y + x2 * q0[q].z + x3 * q0[q].w +
                                     x4 * q1[q].x + x5 * q1[q].y + x6 * q1[q].z + x7 * q1[q].w;
                    }
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < kMixFastK; ++i) {
        if (i < K) {
            const float rs = r4d_wave_sum(ss[i]);
            if (lane == 0) s_red[wave][i] = rs;
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float ds = r4d_wave_sum(dot[q][i]);
                if (lane == 0) s_red[wave][kMixFastK + q * kMixFastK + i] = ds;
            }
        }
    }
    lds_barrier();

    if (tid == 0) {
        float rstd[kMixFastK];
        for (int i = 0; i < K; ++i) {
            float tot = 0.0f;
            for (int w = 0; w < kMixWaves; ++w) tot += s_red[w][i];
            rstd[i] = 1.0f / sqrtf(tot / (float)n + A.eps);
        }
        const float inv_sqrt_n = 1.0f / sqrtf((float)n);
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float sc[kMixFastK];
            for (int i = 0; i < K; ++i) {
                float tot = 0.0f;
                for (int w = 0; w < kMixWaves; ++w) tot += s_red[w][kMixFastK + q * kMixFastK + i];
                sc[i] = tot * rstd[i] * inv_sqrt_n;
            }
            float mx = sc[0];
            for (int i = 1; i < K; ++i) mx = fmaxf(mx, sc[i]);
            float den = 0.0f;
            for (int i = 0; i < K; ++i) {
                sc[i] = expf(sc[i] - mx);
                den += sc[i];
            }
            const float inv_den = 1.0f / den;
            for (int i = 0; i < K; ++i) s_pr[q][i] = (sc[i] * inv_den) * rstd[i];
        }
    }
    lds_barrier();

    auto mix8 = [&](int q, int e, float (&m)[8]) {
#pragma unroll
        for (int k = 0; k < 8; ++k) m[k] = 0.0f;
#pragma unroll
        for (int i = 0; i < kMixFastK; ++i) {
            if (i < K) {
                const float pr = s_pr[q][i];
                const uint4 v  = raw[i][e];
                m[0] += pr * bf16w_lo(v.x); m[1] += pr * bf16w_hi(v.x);
                m[2] += pr * bf16w_lo(v.y); m[3] += pr * bf16w_hi(v.y);
                m[4] += pr * bf16w_lo(v.z); m[5] += pr * bf16w_hi(v.z);
                m[6] += pr * bf16w_lo(v.w); m[7] += pr * bf16w_hi(v.w);
            }
        }
    };

#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        if (A.has_w) {
            float sm = 0.0f;
#pragma unroll
            for (int e = 0; e < kMixVec8Elem; ++e) {
                const int c8 = (tid + e * kMixThr) * 8;
                if (c8 < n) {
                    float m[8];
                    mix8(q, e, m);
#pragma unroll
                    for (int k = 0; k < 8; ++k) sm += m[k] * m[k];
                }
            }
            sm = r4d_wave_sum(sm);
            if (lane == 0) s_wsm[q][wave] = sm;
            lds_barrier();
            if (tid == 0) {
                float tot = 0.0f;
                for (int w = 0; w < kMixWaves; ++w) tot += s_wsm[q][w];
                s_rq[q] = 1.0f / sqrtf(tot / (float)n + A.eps);
            }
            lds_barrier();
            const float rq = s_rq[q];
#pragma unroll
            for (int e = 0; e < kMixVec8Elem; ++e) {
                const int c8 = (tid + e * kMixThr) * 8;
                if (c8 < n) {
                    float m[8];
                    mix8(q, e, m);
                    if (A.w_bf16) {
                        const uint4 vw = ink_ld_u4((const uint16_t*)A.w[q] + c8);
                        ink_st_u4(A.y[q] + t * A.y_rs[q] + c8, make_uint4(
                            bf16x2_rne(m[0] * rq * bf16w_lo(vw.x), m[1] * rq * bf16w_hi(vw.x)),
                            bf16x2_rne(m[2] * rq * bf16w_lo(vw.y), m[3] * rq * bf16w_hi(vw.y)),
                            bf16x2_rne(m[4] * rq * bf16w_lo(vw.z), m[5] * rq * bf16w_hi(vw.z)),
                            bf16x2_rne(m[6] * rq * bf16w_lo(vw.w), m[7] * rq * bf16w_hi(vw.w))));
                    } else {
                        const float4 w0 = *(const float4*)((const float*)A.w[q] + c8);
                        const float4 w1 = *(const float4*)((const float*)A.w[q] + c8 + 4);
                        ink_st_u4(A.y[q] + t * A.y_rs[q] + c8, make_uint4(
                            bf16x2_rne(m[0] * rq * w0.x, m[1] * rq * w0.y),
                            bf16x2_rne(m[2] * rq * w0.z, m[3] * rq * w0.w),
                            bf16x2_rne(m[4] * rq * w1.x, m[5] * rq * w1.y),
                            bf16x2_rne(m[6] * rq * w1.z, m[7] * rq * w1.w)));
                    }
                }
            }
        } else {
#pragma unroll
            for (int e = 0; e < kMixVec8Elem; ++e) {
                const int c8 = (tid + e * kMixThr) * 8;
                if (c8 < n) {
                    float m[8];
                    mix8(q, e, m);
                    ink_st_u4(A.y[q] + t * A.y_rs[q] + c8, make_uint4(
                        bf16x2_rne(m[0], m[1]), bf16x2_rne(m[2], m[3]),
                        bf16x2_rne(m[4], m[5]), bf16x2_rne(m[6], m[7])));
                }
            }
        }
    }
}

template <int NQ, class CH>
__global__ __launch_bounds__(kMixThr) void attn_res_multi_vec8_kernel(MultiMixArgs<NQ> A) {
    if (CH::on) r4d_chain_wait();
    attn_res_multi_vec8_body<NQ>(A);
    if (CH::on) r4d_chain_done();
}

template <int NQ, class CH>
__global__ __launch_bounds__(kMixThr) void attn_res_multi_fast_kernel(MultiMixArgs<NQ> A) {
    if (CH::on) r4d_chain_wait();
    attn_res_multi_fast_body<NQ>(A);
    if (CH::on) r4d_chain_done();
}

template <int NQ, class CH>
__global__ __launch_bounds__(kMixThr) void attn_res_multi_gen_kernel(MultiMixArgs<NQ> A) {
    if (CH::on) r4d_chain_wait();
    attn_res_multi_gen_body<NQ>(A);
    if (CH::on) r4d_chain_done();
}

bool ink_vec_1d(const RadTensor* t, long long n, uint32_t expected_dt, int allow_bf16_or_f32) {
    if (!t || !t->data) return false;
    if (allow_bf16_or_f32) {
        if (t->dtype != RAD_BF16 && t->dtype != RAD_F32) return false;
    } else if (t->dtype != expected_dt) {
        return false;
    }
    if (t->rank == 1) return t->shape[0] == n && t->stride[0] == 1;
    if (t->rank == 2) return t->shape[0] == 1 && t->shape[1] == n && t->stride[1] == 1;
    return false;
}

template <int NQ>
int attn_res_multi_launch(const RadArgs* a, RadStream stream, bool allow_w) {
    const long long n     = r4d_geti_or(a, "n", 0);
    const long long nS    = r4d_geti_or(a, "n_state", -1);
    const long long m_cap = r4d_geti_or(a, "M", 0);
    if (n <= 0 || nS < 0 || nS + 1 > kMixMaxK) return R4D_NO(RAD_E_SHAPE);

    const RadTensor* st  = nS > 0 ? r4d_opt(a, 0) : nullptr;
    const RadTensor* cur = r4d_opt(a, 1);
    if (!cur || !cur->data) return R4D_NO(RAD_E_INVAL);
    if (cur->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    long long cur_rows = 0;
    const long long cur_rs = r4d_rows_pitch(cur, n, &cur_rows);
    if (cur_rs < 0) return R4D_NO(RAD_E_STRIDE);
    const long long M = (m_cap > 0 && cur_rows > m_cap) ? m_cap : cur_rows;
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);

    long long st_ps = 0, st_rs = 0;
    if (nS > 0) {
        if (!st || !st->data) return R4D_NO(RAD_E_INVAL);
        if (st->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
        if (st->rank != 3 || st->stride[2] != 1 || st->shape[0] < nS ||
            st->shape[1] < M || st->shape[2] != n)
            return R4D_NO(RAD_E_SHAPE);
        st_ps = st->stride[0];
        st_rs = st->stride[1];
    }

    MultiMixArgs<NQ> A{};
    A.st     = nS > 0 ? (const uint16_t*)st->data : nullptr;
    A.st_ps  = st_ps;
    A.st_rs  = st_rs;
    A.cur    = (const uint16_t*)cur->data;
    A.cur_rs = cur_rs;
    A.M      = (int)M;
    A.n      = (int)n;
    A.nS     = (int)nS;
    A.eps    = (float)r4d_getf(a, "eps", 1e-6);

    for (int i = 0; i < NQ; ++i) {
        const RadTensor* q = r4d_opt(a, 2 + i);
        const RadTensor* y = r4d_opt(a, 2 + NQ + i);
        if (!q || !q->data || !y || !y->data) return R4D_NO(RAD_E_INVAL);
        if (q->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
        if (y->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
        if (!ink_vec_1d(q, n, RAD_F32, 0)) return R4D_NO(RAD_E_SHAPE);
        long long yr = 0;
        const long long y_rs = r4d_rows_pitch(y, n, &yr);
        if (y_rs < 0) return R4D_NO(RAD_E_STRIDE);
        if (yr < M) return R4D_NO(RAD_E_SHAPE);
        A.q[i]    = (const float*)q->data;
        A.y[i]    = (uint16_t*)y->data;
        A.y_rs[i] = y_rs;
    }

    if (allow_w) {
        int n_have = 0;
        uint32_t w_dt = 0;
        for (int i = 0; i < NQ; ++i) {
            const RadTensor* w = r4d_opt(a, 2 + 2 * NQ + i);
            if (!w || !w->data) continue;
            if (!ink_vec_1d(w, n, 0, 1)) return R4D_NO(RAD_E_SHAPE);
            if (n_have == 0) w_dt = w->dtype;
            else if (w->dtype != w_dt) return R4D_NO(RAD_E_DTYPE);
            A.w[i] = w->data;
            ++n_have;
        }
        if (n_have != 0 && n_have != NQ) return R4D_NO(RAD_E_INVAL);
        A.has_w  = (n_have == NQ) ? 1 : 0;
        A.w_bf16 = (w_dt == RAD_BF16) ? 1 : 0;
    }

    const bool chain = r4d_geti_or(a, "chain", 0) != 0;
    bool vec8_ok = (nS + 1) <= kMixFastK && (n % 8) == 0 && n <= kMixVec8Elem * 8 * kMixThr &&
                   rad_ptr_aligned(cur->data, 16) && (cur_rs % 8) == 0 &&
                   (nS == 0 || (rad_ptr_aligned(st->data, 16) && (st_ps % 8) == 0 && (st_rs % 8) == 0));
    for (int i = 0; i < NQ && vec8_ok; ++i) {
        vec8_ok = rad_ptr_aligned(A.q[i], 16) && rad_ptr_aligned(A.y[i], 16) &&
                  (A.y_rs[i] % 8) == 0 && (!A.has_w || rad_ptr_aligned(A.w[i], 16));
    }
    const bool fast  = (nS + 1) <= kMixFastK && n <= kMixMaxElem * kMixThr;
    int emit_rc;
    if (vec8_ok) {
        emit_rc = R4D_EMIT(stream,
                           (chain ? attn_res_multi_vec8_kernel<NQ, r4dchain>
                                  : attn_res_multi_vec8_kernel<NQ, R4dUnchained>),
                           dim3((unsigned)M), dim3(kMixThr), 0, A);
    } else if (fast) {
        emit_rc = R4D_EMIT(stream,
                           (chain ? attn_res_multi_fast_kernel<NQ, r4dchain>
                                  : attn_res_multi_fast_kernel<NQ, R4dUnchained>),
                           dim3((unsigned)M), dim3(kMixThr), 0, A);
    } else {
        emit_rc = R4D_EMIT(stream,
                           (chain ? attn_res_multi_gen_kernel<NQ, r4dchain>
                                  : attn_res_multi_gen_kernel<NQ, R4dUnchained>),
                           dim3((unsigned)M), dim3(kMixThr), 0, A);
    }
    if (emit_rc != RAD_OK) return emit_rc;
    R4D_INK_CHK();
    return RAD_OK;
}

}  // namespace

extern "C" int64_t r4d_rad_loop_ctl_scratch(const RadArgs* a) {
    const long long M    = r4d_geti_or(a, "M", 0);
    const long long rank = r4d_geti_or(a, "rank", 0);
    if (M <= 0 || M > kLoopCtlMaxM || rank <= 0 || rank > kLoopCtlMaxRank) return 0;
    return M * rank * (long long)sizeof(float);
}

extern "C" int r4d_rad_loop_ctl(const RadArgs* a, RadStream stream) {
    const RadTensor* x   = r4d_opt(a, 0);
    const RadTensor* msg = r4d_opt(a, 1);
    const RadTensor* wdx = r4d_opt(a, 2);
    const RadTensor* wg  = r4d_opt(a, 3);
    const RadTensor* wf  = r4d_opt(a, 4);
    const RadTensor* y   = r4d_opt(a, 5);
    if (!x || !msg || !wdx || !wg || !wf || !y) return R4D_NO(RAD_E_INVAL);
    if (!x->data || !msg->data || !wdx->data || !wg->data || !wf->data || !y->data)
        return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || msg->dtype != RAD_BF16 || wdx->dtype != RAD_BF16 ||
        wg->dtype != RAD_BF16 || wf->dtype != RAD_BF16 || y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);

    const long long n    = r4d_geti_or(a, "n", 0);
    const long long rank = r4d_geti_or(a, "rank", 0);
    if (n <= 0 || n > kLoopCtlMaxN || rank <= 0 || rank > kLoopCtlMaxRank)
        return R4D_NO(RAD_E_SHAPE);

    long long M = 0, Mm = 0, My = 0, Rw = 0, Ng = 0, Nf = 0;
    const long long x_ld   = r4d_rows_pitch(x, n, &M);
    const long long msg_ld = r4d_rows_pitch(msg, n, &Mm);
    const long long y_ld   = r4d_rows_pitch(y, n, &My);
    const long long wdx_ld = r4d_rows_pitch(wdx, 2 * n, &Rw);
    const long long wg_ld  = r4d_rows_pitch(wg, rank, &Ng);
    const long long wf_ld  = r4d_rows_pitch(wf, rank, &Nf);
    if (x_ld < 0 || msg_ld < 0 || y_ld < 0 || wdx_ld < 0 || wg_ld < 0 || wf_ld < 0)
        return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || M > kLoopCtlMaxM || Mm < M || My < M || Rw != rank || Ng != n || Nf != n)
        return R4D_NO(RAD_E_SHAPE);

    if (!a->scratch || a->scratch_bytes < M * rank * (long long)sizeof(float))
        return R4D_NO(RAD_E_SCRATCH);

    const float eps = (float)r4d_getf(a, "eps", 1e-6);
    const bool chain = r4d_geti_or(a, "chain", 0) != 0;
    const size_t lds = (size_t)(2 * n) * sizeof(uint16_t);
    const unsigned nr_blk = (unsigned)((rank + kDownWaves - 1) / kDownWaves);
    const bool vec8 = (n % 8) == 0 && (wdx_ld % 8) == 0 && rad_ptr_aligned(wdx->data, 16) &&
                      (x_ld % 8) == 0 && (msg_ld % 8) == 0 &&
                      rad_ptr_aligned(x->data, 16) && rad_ptr_aligned(msg->data, 16);
    int emit_rc;
    if (vec8) {
        emit_rc = R4D_EMIT(stream,
                           (chain ? loop_ctl_down_kernel<true, r4dchain>
                                  : loop_ctl_down_kernel<true, R4dUnchained>),
                           dim3((unsigned)M, nr_blk), dim3(kDownThr), lds,
                           (const uint16_t*)x->data, x_ld,
                           (const uint16_t*)msg->data, msg_ld,
                           (const uint16_t*)wdx->data, wdx_ld,
                           (float*)a->scratch, (int)n, (int)rank, eps);
    } else {
        emit_rc = R4D_EMIT(stream,
                           (chain ? loop_ctl_down_kernel<false, r4dchain>
                                  : loop_ctl_down_kernel<false, R4dUnchained>),
                           dim3((unsigned)M, nr_blk), dim3(kDownThr), lds,
                           (const uint16_t*)x->data, x_ld,
                           (const uint16_t*)msg->data, msg_ld,
                           (const uint16_t*)wdx->data, wdx_ld,
                           (float*)a->scratch, (int)n, (int)rank, eps);
    }
    if (emit_rc != RAD_OK) return emit_rc;

    const bool vec64 = (rank % 64) == 0 && (wg_ld % 8) == 0 && (wf_ld % 8) == 0 &&
                       rad_ptr_aligned(wg->data, 16) && rad_ptr_aligned(wf->data, 16);
    const unsigned ny = (unsigned)((n + kApplyCols - 1) / kApplyCols);
    if (vec64) {
        emit_rc = R4D_EMIT(stream,
                           (chain ? loop_ctl_apply_kernel<true, r4dchain>
                                  : loop_ctl_apply_kernel<true, R4dUnchained>),
                           dim3((unsigned)M, ny), dim3(kApplyThr), 0,
                           (const uint16_t*)x->data, x_ld,
                           (const uint16_t*)msg->data, msg_ld,
                           (const uint16_t*)wg->data, wg_ld,
                           (const uint16_t*)wf->data, wf_ld,
                           (const float*)a->scratch, (uint16_t*)y->data, y_ld,
                           (int)n, (int)rank);
    } else {
        emit_rc = R4D_EMIT(stream,
                           (chain ? loop_ctl_apply_kernel<false, r4dchain>
                                  : loop_ctl_apply_kernel<false, R4dUnchained>),
                           dim3((unsigned)M, ny), dim3(kApplyThr), 0,
                           (const uint16_t*)x->data, x_ld,
                           (const uint16_t*)msg->data, msg_ld,
                           (const uint16_t*)wg->data, wg_ld,
                           (const uint16_t*)wf->data, wf_ld,
                           (const float*)a->scratch, (uint16_t*)y->data, y_ld,
                           (int)n, (int)rank);
    }
    if (emit_rc != RAD_OK) return emit_rc;
    R4D_INK_CHK();
    return RAD_OK;
}

extern "C" int r4d_rad_loop_ctl_blend(const RadArgs* a, RadStream stream) {
    const RadTensor* x   = r4d_opt(a, 0);
    const RadTensor* msg = r4d_opt(a, 1);
    const RadTensor* gp  = r4d_opt(a, 2);
    const RadTensor* fp  = r4d_opt(a, 3);
    const RadTensor* y   = r4d_opt(a, 4);
    if (!x || !msg || !gp || !fp || !y) return R4D_NO(RAD_E_INVAL);
    if (!x->data || !msg->data || !gp->data || !fp->data || !y->data)
        return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || msg->dtype != RAD_BF16 || gp->dtype != RAD_BF16 ||
        fp->dtype != RAD_BF16 || y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);

    const long long n = r4d_geti_or(a, "n", 0);
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mm = 0, Mg = 0, Mf = 0, My = 0;
    const long long x_ld   = r4d_rows_pitch(x, n, &M);
    const long long msg_ld = r4d_rows_pitch(msg, n, &Mm);
    const long long gp_ld  = r4d_rows_pitch(gp, n, &Mg);
    const long long fp_ld  = r4d_rows_pitch(fp, n, &Mf);
    const long long y_ld   = r4d_rows_pitch(y, n, &My);
    if (x_ld < 0 || msg_ld < 0 || gp_ld < 0 || fp_ld < 0 || y_ld < 0)
        return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mm < M || Mg < M || Mf < M || My < M) return R4D_NO(RAD_E_SHAPE);

    const bool chain = r4d_geti_or(a, "chain", 0) != 0;
    const bool vec8 = (n % 8) == 0 && (x_ld % 8) == 0 && (msg_ld % 8) == 0 &&
                      (gp_ld % 8) == 0 && (fp_ld % 8) == 0 && (y_ld % 8) == 0 &&
                      rad_ptr_aligned(x->data, 16) && rad_ptr_aligned(msg->data, 16) &&
                      rad_ptr_aligned(gp->data, 16) && rad_ptr_aligned(fp->data, 16) &&
                      rad_ptr_aligned(y->data, 16);
    const unsigned gy = (unsigned)(M < 65535LL ? M : 65535LL);
    int emit_rc;
    if (vec8) {
        const unsigned nx = (unsigned)(((n / 8) + 255) / 256);
        emit_rc = R4D_EMIT(stream,
                           (chain ? loop_ctl_blend_kernel<true, r4dchain>
                                  : loop_ctl_blend_kernel<true, R4dUnchained>),
                           dim3(nx, gy), dim3(256), 0,
                           (const uint16_t*)x->data, x_ld,
                           (const uint16_t*)msg->data, msg_ld,
                           (const uint16_t*)gp->data, gp_ld,
                           (const uint16_t*)fp->data, fp_ld,
                           (uint16_t*)y->data, y_ld, M, (int)n);
    } else {
        const unsigned nx = (unsigned)((n + 255) / 256);
        emit_rc = R4D_EMIT(stream,
                           (chain ? loop_ctl_blend_kernel<false, r4dchain>
                                  : loop_ctl_blend_kernel<false, R4dUnchained>),
                           dim3(nx, gy), dim3(256), 0,
                           (const uint16_t*)x->data, x_ld,
                           (const uint16_t*)msg->data, msg_ld,
                           (const uint16_t*)gp->data, gp_ld,
                           (const uint16_t*)fp->data, fp_ld,
                           (uint16_t*)y->data, y_ld, M, (int)n);
    }
    if (emit_rc != RAD_OK) return emit_rc;
    R4D_INK_CHK();
    return RAD_OK;
}

extern "C" int r4d_rad_plane_copy(const RadArgs* a, RadStream stream) {
    const RadTensor* src = r4d_opt(a, 0);
    const RadTensor* ref = r4d_opt(a, 1);
    const RadTensor* dst = r4d_opt(a, 2);
    if (!src || !ref || !dst) return R4D_NO(RAD_E_INVAL);
    if (!src->data || !dst->data) return R4D_NO(RAD_E_INVAL);
    if (src->dtype != RAD_BF16 || dst->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);

    const long long n = r4d_geti_or(a, "n", 0);
    if (n <= 0 || ref->rank < 2 || ref->shape[0] <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long M = ref->shape[0];
    long long s_rows = 0, s_ld = 0, d_rows = 0, d_ld = 0;
    if (!plane_view(src, n, &s_rows, &s_ld) || !plane_view(dst, n, &d_rows, &d_ld))
        return R4D_NO(RAD_E_STRIDE);
    if (M > s_rows || M > d_rows) return R4D_NO(RAD_E_SHAPE);

    const bool chain = r4d_geti_or(a, "chain", 0) != 0;
    const bool vec8 = (n % 8) == 0 && (s_ld % 8) == 0 && (d_ld % 8) == 0 &&
                      rad_ptr_aligned(src->data, 16) && rad_ptr_aligned(dst->data, 16);
    const unsigned gy = (unsigned)(M < 65535LL ? M : 65535LL);
    int emit_rc;
    if (vec8) {
        const unsigned nx = (unsigned)(((n / 8) + 255) / 256);
        emit_rc = R4D_EMIT(stream,
                           (chain ? plane_copy_kernel<true, r4dchain>
                                  : plane_copy_kernel<true, R4dUnchained>),
                           dim3(nx, gy), dim3(256), 0,
                           (const uint16_t*)src->data, s_ld, (uint16_t*)dst->data, d_ld, M, (int)n);
    } else {
        const unsigned nx = (unsigned)((n + 255) / 256);
        emit_rc = R4D_EMIT(stream,
                           (chain ? plane_copy_kernel<false, r4dchain>
                                  : plane_copy_kernel<false, R4dUnchained>),
                           dim3(nx, gy), dim3(256), 0,
                           (const uint16_t*)src->data, s_ld, (uint16_t*)dst->data, d_ld, M, (int)n);
    }
    if (emit_rc != RAD_OK) return emit_rc;
    R4D_INK_CHK();
    return RAD_OK;
}

extern "C" int r4d_rad_kv_clip_norm_store(const RadArgs* a, RadStream stream) {
    const RadTensor* k    = r4d_opt(a, 0);
    const RadTensor* v    = r4d_opt(a, 1);
    const RadTensor* slot = r4d_opt(a, 2);
    const RadTensor* kv   = r4d_opt(a, 3);
    if (!k || !v || !slot || !kv) return R4D_NO(RAD_E_INVAL);
    if (!k->data || !v->data || !slot->data || !kv->data) return R4D_NO(RAD_E_INVAL);
    if (k->dtype != RAD_BF16 || v->dtype != RAD_BF16 || slot->dtype != RAD_I32)
        return R4D_NO(RAD_E_DTYPE);

    const long long head_dim   = r4d_geti_or(a, "head_dim", 0);
    const long long n_head_kv  = r4d_geti_or(a, "n_head_kv", 0);
    const long long block_size = r4d_geti_or(a, "block_size", 0);
    const char* kv_dtype       = r4d_gets(a, "kv_dtype");
    if (!kv_dtype) return R4D_NO(RAD_E_INVAL);
    const bool is_i8   = kv_dtype[0] == 'i' && kv_dtype[1] == '8' && kv_dtype[2] == '\0';
    const bool is_bf16 = kv_dtype[0] == 'b' && kv_dtype[1] == 'f' && kv_dtype[2] == '1' &&
                         kv_dtype[3] == '6' && kv_dtype[4] == '\0';
    if (!is_i8 && !is_bf16) return R4D_NO(RAD_E_DTYPE);
    if (head_dim <= 0 || head_dim > 256 || (head_dim % 4) != 0 || n_head_kv <= 0 || block_size <= 0)
        return R4D_NO(RAD_E_SHAPE);

    if (kv->dtype != (is_i8 ? (uint32_t)RAD_I8 : (uint32_t)RAD_BF16))
        return R4D_NO(RAD_E_DTYPE);
    if (kv->rank != 4 || kv->stride[3] != 1 || kv->shape[1] != n_head_kv ||
        kv->shape[2] != block_size ||
        kv->shape[3] != (is_i8 ? (2 * head_dim + 4) : (2 * head_dim)))
        return R4D_NO(RAD_E_SHAPE);
    if (kv->shape[0] <= 0) return R4D_NO(RAD_E_SHAPE);

    long long Tk = 0, Tv = 0;
    const long long cols = n_head_kv * head_dim;
    const long long k_ld = r4d_rows_pitch(k, cols, &Tk);
    const long long v_ld = r4d_rows_pitch(v, cols, &Tv);
    if (k_ld < 0 || v_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (Tk <= 0 || Tv < Tk) return R4D_NO(RAD_E_SHAPE);
    if (slot->rank < 1 || slot->stride[slot->rank - 1] != 1 || slot->shape[0] < Tk)
        return R4D_NO(RAD_E_SHAPE);

    const float threshold = (float)r4d_getf(a, "threshold", 0.0);
    const float eps       = (float)r4d_getf(a, "eps", 1e-6);
    const bool  chain     = r4d_geti_or(a, "chain", 0) != 0;
    const unsigned gy     = (unsigned)(Tk < 65535LL ? Tk : 65535LL);

    int emit_rc;
#define R4D_KV_CNS(I8_, ME_, CH_)                                                               \
    R4D_EMIT(stream, (kv_clip_norm_store_kernel<I8_, ME_, CH_>),                                \
             dim3((unsigned)n_head_kv, gy), dim3(32), 0,                                        \
             (const uint16_t*)k->data, k_ld, (const uint16_t*)v->data, v_ld,                    \
             (const int32_t*)slot->data, kv->data, kv->stride[0], kv->stride[1], kv->stride[2], \
             Tk, (uint32_t)head_dim, (uint32_t)kv->shape[0], (uint32_t)block_size,              \
             threshold, eps)
    if (head_dim <= 128) {
        if (is_i8) {
            emit_rc = chain ? R4D_KV_CNS(true, 4, r4dchain) : R4D_KV_CNS(true, 4, R4dUnchained);
        } else {
            emit_rc = chain ? R4D_KV_CNS(false, 4, r4dchain) : R4D_KV_CNS(false, 4, R4dUnchained);
        }
    } else {
        if (is_i8) {
            emit_rc = chain ? R4D_KV_CNS(true, 8, r4dchain) : R4D_KV_CNS(true, 8, R4dUnchained);
        } else {
            emit_rc = chain ? R4D_KV_CNS(false, 8, r4dchain) : R4D_KV_CNS(false, 8, R4dUnchained);
        }
    }
#undef R4D_KV_CNS
    if (emit_rc != RAD_OK) return emit_rc;
    R4D_INK_CHK();
    return RAD_OK;
}

extern "C" int r4d_rad_qr_l2_clip(const RadArgs* a, RadStream stream) {
    const RadTensor* xq  = r4d_opt(a, 0);
    const RadTensor* wq  = r4d_opt(a, 1);
    const RadTensor* xr  = r4d_opt(a, 2);
    const RadTensor* wr  = r4d_opt(a, 3);
    const RadTensor* yq  = r4d_opt(a, 4);
    const RadTensor* yr  = r4d_opt(a, 5);
    const RadTensor* pos = r4d_opt(a, 6);
    if (!xq || !wq || !xr || !wr || !yq || !yr) return R4D_NO(RAD_E_INVAL);
    if (!xq->data || !wq->data || !xr->data || !wr->data || !yq->data || !yr->data)
        return R4D_NO(RAD_E_INVAL);
    if (xq->dtype != RAD_BF16 || wq->dtype != RAD_BF16 || xr->dtype != RAD_BF16 ||
        wr->dtype != RAD_BF16 || yq->dtype != RAD_BF16 || yr->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);

    const long long nq = r4d_geti_or(a, "nq", 0);
    const long long gq = r4d_geti_or(a, "gq", 0);
    const long long nr = r4d_geti_or(a, "nr", 0);
    const long long gr = r4d_geti_or(a, "gr", 0);
    if (nq <= 0 || gq <= 0 || (nq % gq) != 0 || nr <= 0 || gr <= 0 || (nr % gr) != 0)
        return R4D_NO(RAD_E_SHAPE);

    long long T = 0, Tr = 0, Tyq = 0, Tyr = 0;
    const long long xq_ld = r4d_rows_pitch(xq, nq, &T);
    const long long xr_ld = r4d_rows_pitch(xr, nr, &Tr);
    const long long yq_ld = r4d_rows_pitch(yq, nq, &Tyq);
    const long long yr_ld = r4d_rows_pitch(yr, nr, &Tyr);
    if (xq_ld < 0 || xr_ld < 0 || yq_ld < 0 || yr_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (T <= 0 || Tr < T || Tyq < T || Tyr < T) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(wq) || r4d_numel(wq) != nq || !r4d_contig(wr) || r4d_numel(wr) != nr)
        return R4D_NO(RAD_E_SHAPE);
    if (pos) {
        if (pos->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
        if (!pos->data || pos->rank < 1 || pos->stride[pos->rank - 1] != 1 || pos->shape[0] < T)
            return R4D_NO(RAD_E_SHAPE);
    }

    const double floorv = r4d_getf(a, "floor", 1.0);
    const double alpha  = r4d_getf(a, "alpha", 0.0);
    if (!(floorv > 0.0)) return R4D_NO(RAD_E_INVAL);
    const float thrq      = (float)r4d_getf(a, "thrq", 0.0);
    const float thrr      = (float)r4d_getf(a, "thrr", 0.0);
    const float inv_floor = (float)(1.0 / floorv);
    const int   gq_cnt    = (int)(nq / gq);
    const int   gr_cnt    = (int)(nr / gr);
    const bool  chain     = r4d_geti_or(a, "chain", 0) != 0;
    const unsigned gy     = (unsigned)(T < 65535LL ? T : 65535LL);

    const int emit_rc = R4D_EMIT(
        stream, (chain ? qr_l2_clip_kernel<r4dchain> : qr_l2_clip_kernel<R4dUnchained>),
        dim3((unsigned)(gq_cnt + gr_cnt), gy), dim3(32), 0,
        (const uint16_t*)xq->data, xq_ld, (const uint16_t*)wq->data,
        (const uint16_t*)xr->data, xr_ld, (const uint16_t*)wr->data,
        (uint16_t*)yq->data, yq_ld, (uint16_t*)yr->data, yr_ld,
        pos ? (const int32_t*)pos->data : nullptr,
        T, gq_cnt, (int)gq, thrq, (int)gr, thrr, inv_floor, (float)alpha);
    if (emit_rc != RAD_OK) return emit_rc;
    R4D_INK_CHK();
    return RAD_OK;
}

extern "C" int r4d_rad_attn_res_mix3(const RadArgs* a, RadStream stream) {
    return attn_res_multi_launch<3>(a, stream, true);
}

extern "C" int r4d_rad_attn_res_mix4(const RadArgs* a, RadStream stream) {
    return attn_res_multi_launch<4>(a, stream, false);
}
