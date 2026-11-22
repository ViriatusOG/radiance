/* r4d_gated_quant_i8.hip -- gated_quant_i8: act(gate) * up, then per-group int8 codes against one f32
 * scale each, in one pass. libref/ref_quant.cpp's ref_gated_quant_i8 is the contract.
 *
 * gated_had_quant_i8 without the rotation. The NVFP4 experts' down projection reads these codes:
 * one scale per (row, group), row-major, the layout quant_act_i8g writes.
 *
 * One workgroup a row. Its threads take the row's groups in turn, and each group is evaluated twice,
 * once for its absmax and once for its codes, rather than staged: nothing lives in LDS and nothing
 * is held across the row, so the kernel has no size limit on n. That costs a second evaluation of
 * the gate, which is the price of the straightforward form; the rotated op stages its row because a
 * Hadamard needs the whole block at once, and this one does not.
 *
 * expf, not __expf: the oracle's sigmoid is exact, and an approximation moves a code that sits on a
 * rounding boundary.
 *
 * Gate and up arrive as two operands, or as one packed [M, 2n] gate_up with the gate first; the
 * kernel takes row pitches, so either form is one launch.
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include "r4d_plugin.h"
#include "r4d_launch.h"
#include "rad_plugin.h"

namespace r4d_gq {

constexpr int kThr = 256;

__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((uint32_t)v << 16); }

struct Args {
    const uint16_t* g; const uint16_t* u;   /* gate and up rows, bf16 */
    long long gs, us;                        /* their row pitches, in elements */
    signed char* q; float* s;               /* codes [M, n] and scales [M, ngrp], dense */
    int n, group, ngrp;
    int act;                                 /* 0 silu, 1 sigmoid, 2 situ */
    float gcap, ucap;
};

/* One element of act(gate) * up, in the oracle's order: (gate * sigmoid(gate)) * up for silu, and
 * for SiTU (capped gate * sigmoid(raw gate)) * capped up. The sigmoid always reads the raw gate. */
__device__ __forceinline__ float gated(const Args& A, const uint16_t* g, const uint16_t* u, int i) {
    const float gv = bf(g[i]), uv = bf(u[i]);
    const float sg = 1.0f / (1.0f + expf(-gv));
    if (A.act == 2) {
        const float ga = A.gcap > 0.0f ? A.gcap * tanhf(gv / A.gcap) : gv;
        const float ua = A.ucap > 0.0f ? A.ucap * tanhf(uv / A.ucap) : uv;
        return ga * sg * ua;
    }
    return (A.act == 1 ? sg : gv * sg) * uv;
}

__global__ __launch_bounds__(kThr) void gq_kernel(Args A) {
    const long long r = blockIdx.x;
    const uint16_t* g = A.g + r * A.gs;
    const uint16_t* u = A.u + r * A.us;
    signed char* q = A.q + r * A.n;
    float* s = A.s + r * A.ngrp;
    for (int gi = (int)threadIdx.x; gi < A.ngrp; gi += kThr) {
        const int k0 = gi * A.group;
        const int k1 = k0 + A.group < A.n ? k0 + A.group : A.n;
        float amax = 0.0f;
        for (int i = k0; i < k1; ++i) amax = fmaxf(amax, fabsf(gated(A, g, u, i)));
        /* A zero group keeps scale 1 so its codes (all zero) dequantise to zero, as libref does. */
        const float sc = amax > 0.0f ? amax * (1.0f / 127.0f) : 1.0f;
        const float inv = 1.0f / sc;
        for (int i = k0; i < k1; ++i) {
            const float v = gated(A, g, u, i) * inv;
            /* NaN stores as 0, the integer store rule of libref's st_dt; the clamp cannot see it. */
            int t = v != v ? 0 : (int)nearbyintf(fminf(fmaxf(v, -127.0f), 127.0f));
            q[i] = (signed char)t;
        }
        s[gi] = sc;
    }
}

/* act as libref spells it: the name, or libr4d's integer `mode` where the name is absent. */
int act_code(const RadArgs* a) {
    if (const char* s = rad_args_gets(a, "act")) {
        if (!std::strcmp(s, "silu")) return 0;
        if (!std::strcmp(s, "sigmoid")) return 1;
        if (!std::strcmp(s, "situ")) return 2;
        return -1;
    }
    const long long m = rad_args_geti_or(a, "mode", 0);
    return (m >= 0 && m <= 2) ? (int)m : -1;
}

}  // namespace r4d_gq

int r4d_rad_gated_quant_i8(const RadArgs* a, RadStream stream) {
    using namespace r4d_gq;
    if (!a) return RAD_E_INVAL;
    const RadTensor* ta = rad_arg_in(a, 0);
    const RadTensor* tb = rad_arg_in(a, 1);   /* absent = the packed gate_up form */
    const RadTensor* q  = rad_arg_in(a, 2);
    const RadTensor* sc = rad_arg_in(a, 3);
    if (!ta || !q || !sc) return RAD_E_INVAL;
    if (ta->dtype != RAD_BF16 || (tb && tb->dtype != RAD_BF16)) return RAD_E_DTYPE;
    if (q->dtype != RAD_I8 || sc->dtype != RAD_F32) return RAD_E_DTYPE;

    const long long n = rad_args_geti_or(a, "n", 0);
    if (n < 1) return RAD_E_SHAPE;
    /* libref's default and its clamp: a group wider than the row, or none, is one scale per row. */
    long long group = rad_args_geti_or(a, "group", 128);
    if (group <= 0 || group > n) group = n;
    const long long ngrp = (n + group - 1) / group;

    /* The rows are the codes': a launch sees the ranged `M` at its band's upper bound, so it is a
     * ceiling only (docs/OPS.md), and the inputs may hold more rows than the codes take. */
    if (q->rank != 2) return RAD_E_SHAPE;
    const long long m_cap = rad_args_geti_or(a, "M", 0);
    long long M = q->shape[0];
    if (m_cap > 0 && m_cap < M) M = m_cap;
    if (M < 1) return RAD_E_SHAPE;
    if (ta->rank != 2 || ta->stride[1] != 1 || ta->shape[0] < M) return RAD_E_SHAPE;
    const long long a_rs = ta->stride[0];
    long long b_rs = a_rs;
    if (tb) {
        if (tb->rank != 2 || tb->stride[1] != 1 || tb->shape[0] < M) return RAD_E_SHAPE;
        if (ta->shape[1] != n || tb->shape[1] != n) return RAD_E_SHAPE;
        b_rs = tb->stride[0];
    } else {
        if (ta->shape[1] != 2 * n) return RAD_E_SHAPE;
    }
    if (a_rs < 1 || b_rs < 1) return RAD_E_STRIDE;

    if (q->rank != 2 || q->shape[0] < M || q->shape[1] != n || q->stride[1] != 1 ||
        q->stride[0] != n) return RAD_E_SHAPE;
    if (sc->rank != 2 || sc->shape[0] < M || sc->shape[1] != ngrp || sc->stride[1] != 1 ||
        sc->stride[0] != ngrp) return RAD_E_SHAPE;

    const int act = act_code(a);
    if (act < 0) return RAD_E_INVAL;
    const double gcap = rad_args_getf_or(a, "gate_cap", 0.0), ucap = rad_args_getf_or(a, "up_cap", 0.0);
    if (!(gcap >= 0.0) || !(ucap >= 0.0) || !std::isfinite(gcap) || !std::isfinite(ucap))
        return RAD_E_INVAL;
    /* The caps are SiTU's. With silu or sigmoid they would be silently dropped, so they are refused:
     * a caller that asked for a cap gets an error, not the uncapped activation. */
    if (act != 2 && (gcap != 0.0 || ucap != 0.0)) return RAD_E_UNSUPPORTED;

    Args A{};
    A.g = (const uint16_t*)ta->data;
    A.u = tb ? (const uint16_t*)tb->data : (const uint16_t*)ta->data + n;
    A.gs = a_rs;
    A.us = tb ? b_rs : a_rs;
    A.q = (signed char*)q->data;
    A.s = (float*)sc->data;
    A.n = (int)n; A.group = (int)group; A.ngrp = (int)ngrp;
    A.act = act;
    A.gcap = (float)gcap; A.ucap = (float)ucap;
    R4D_KLAUNCH(R4D_KNAME(gq_kernel), dim3((unsigned)M), dim3(kThr), (hipStream_t)stream, A);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
