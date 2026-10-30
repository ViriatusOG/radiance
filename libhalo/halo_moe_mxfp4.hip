/* Derived from libr4d/r4d_moe_mxfp4.hip, then ported to gfx1151; see libhalo/README.md. */
// halo_moe_mxfp4.hip - moe_gemm_mxfp4a8: the grouped MXFP4 expert GEMM at every routed height,
// gfx1151.
//
// The stored weight is the canonical one: an e2m1 code per nibble along K (element 2i in the low
// nibble of byte i), one E8M0 exponent byte per 32 values, both in their source layout -- stacked,
// or as a table of per-expert pointers (stride[0] == 0, experts live in VRAM and, at `pooled`, in
// host-mapped memory). The activation is E4M3 with an f32 scale per (row, 128 of K), the output
// bf16, and `sorted`/`offset` are moe_scatter's routing.
//
// libr4d decodes each code into an exact E4M3 byte and runs the fp8 WMMA, whose product is exact
// and whose 32-group sums it folds `sa * e8m0` into. gfx1151 has no fp8 matrix instruction (see
// halo_fp8_wmma.h), so both operands dequantise to bf16 here and take halo::mma, whose products
// are equally exact: an e2m1 value has one mantissa bit, an e4m3 value three, and both are exact
// in bf16, so each product is exact in f32 and the fold order of the numbers is libr4d's.
//
// The 2^-120 that halo::deq16_e4m3 leaves in an e4m3 value (halo_wmma.h) is not folded into a
// scale: the code table here is decoded by arithmetic with `ediff = +120`, so the weight fragment
// carries 2^120 and multiplies back to the exact product. Nothing is exponent-folded -- the E8M0
// byte is applied per 32 values as `(sub * sa) * sb`, exactly libref's quant_dot order -- which is
// what lets an E8M0 code of 255 produce the reference's NaN and a code of 0 its 2^-127.
//
// One thing does NOT carry over from gfx1201: the fp8 WMMA's product sum is exact (03 mantissa
// bits times 1 is 4, and f32 holds the sum), while gfx1151's bf16 WMMA rounds at about one ulp of
// its own result. Against a real block scale (the checkpoint spread halo_gemm_mxfp4a8_nt_m64
// measured is E8M0 114..129) that is far inside any tolerance, but the format's ends are 2^-127
// and 2^128: a group there multiplies the last-bit residue by up to 2^128, and the selftest
// exercises exactly that (codes 0 and 255 beside 252..254). So the two 16-wide k steps accumulate
// separately and are added in F32 -- a pair that cancels cancels in the reference's terms -- and a
// group whose scale times this lane's activation scale exceeds 1 (the exact fold's gate, see
// mx_exact8) is summed in F32 software instead, which reproduces the reference's exact zero.
//
// One launch serves the whole band: a block is four waves over sixteen sorted rows and 64 output
// columns, a wave one 16x16 WMMA tile of them. A block's rows may belong to several experts (a
// run boundary, or a dropped slot inside one), so the block walks the expert runs its row range
// covers and recomputes per run; each row belongs to exactly one run, so it is written exactly
// once, zero when it is a dropped slot or a column past its class's width in this part.
//
// An expert is entry e/2 of the table for its parity (`w` for even experts, `w_odd` for odd), N/K
// differ by parity through N_odd/K_odd, and with `parts` > 1 output column n is column
// (n % (N/parts)) of part (n / (N/parts)), i.e. weight row part * (n_e/parts) + within.
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_wmma.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>

namespace {

/* The geometry libr4d's MxMeta carries, unchanged: two weight classes by expert parity, the
 * scale grids and the routing. */
struct MxMeta {
    long wstep[2], sstep[2];
    int wrow[2], srow[2], n[2], k[2];
    long ar, asr, yr;
    int T, Arows, N, K, experts, topk, sorted_a, parts;
};

/* An expert's weight or exponent plane: a table entry (stride[0] == 0, the shipped form) or its
 * slab of one stacked plane. */
__device__ __forceinline__ const uint8_t* mx_entry(const void* base, long step, int e) {
    return step ? (const uint8_t*)base + (long)e * step : ((const uint8_t* const*)base)[e];
}

/* One e2m1 code and its block's exponent shift as an exact bf16 bit pattern: value (2+m) * 2^(E-2)
 * with E = mag>>1 and m = mag&1, times 2^ediff. Zero keeps its sign and stays zero. */
__device__ __forceinline__ unsigned mx_e2m1_bf16(unsigned code, int ediff) {
    const unsigned mag = code & 7u;
    if (!mag) return (code & 8u) << 12;
    unsigned e = mag >> 1, mm = mag & 1u;
    if (mag == 1) mm = 0;   /* 0.5 is (1 + 0) * 2^-1, not (1 + 1/2) * 2^-1 */
    const unsigned ef = e + (unsigned)(ediff + 126);
    return ((code & 8u) << 12) | (ef << 7) | (mm << 6);
}

/* 2^120, the factor that cancels halo::deq16_e4m3's 2^-120 on the activation side. The largest
 * weight is 6 * 2^120, far inside bf16's range. */
#define MX_WSCALE 120

/* Sixteen consecutive k of this lane's column: the two packed dwords holding them, low nibble
 * first, to sixteen bf16. */
__device__ __forceinline__ v16bf mx_wfrag(unsigned dw0, unsigned dw1) {
    union { uint32_t u[8]; v16bf f; } q;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const unsigned dw = j < 8 ? dw0 : dw1;
        const unsigned code = (dw >> (4 * (j & 7))) & 15u;
        const unsigned w = mx_e2m1_bf16(code, MX_WSCALE);
        if ((j & 1) == 0)
            q.u[j >> 1] = w;
        else
            q.u[j >> 1] |= w << 16;
    }
    return q.f;
}

/* An E8M0 byte: 2^(e - 127), code 0 is 2^-127 (a subnormal f32), 255 is NaN -- libref's
 * e8m0_to_f32, and libr4d's bit pattern for both ends. */
__device__ __forceinline__ float mx_e8m0(unsigned e) {
    return __uint_as_float(e == 255u ? 0x7fc00000u : e ? (e << 23) : 0x00400000u);
}

/* The bits of a bf16 half as f32: bf16 is an f32's top half, so the widening is a shift. */
__device__ __forceinline__ float mx_bf(uint32_t bits) {
    return __uint_as_float((bits & 0xFFFFu) << 16);
}

/* One 32-value group of this lane's column against the eight rows it accumulates, the exact
 * product sum in F32 ascending k -- the chain libref's quant_dot forms.
 *
 * It exists because gfx1151's bf16 WMMA dot product rounds at about one ulp of its own result:
 * benign against any real scale, but a group whose E8M0 exponent is far from 127 multiplies that
 * residue by up to 2^128, and the reference -- whose 32-value sum is exact, the products of an
 * e2m1 code and an e4m3 byte always being exactly representable -- would hold a zero or a
 * subnormal there. The lanes' A values are one row a lane, so the eight rows come from the wave;
 * each lane's own column supplies the codes. Every product below is exact, so the sum is
 * too. */
__device__ __forceinline__ void mx_exact8(const uint32_t fa[2][8], const uint32_t fw[2][8], int h,
                                          float* sub) {
    float wf[32];
#pragma unroll
    for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int j = 0; j < 16; ++j) wf[16 * c + j] = mx_bf(fw[c][j >> 1] >> (16 * (j & 1)));
#pragma unroll
    for (int ee = 0; ee < 8; ++ee) {
        const int r = 2 * ee + h;
        float s = 0.f;
#pragma unroll
        for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                const uint32_t dw = __shfl_sync(0xFFFFFFFFFFFFFFFFull, fa[c][d], r);
                s = __fmaf_rn(mx_bf(dw), wf[16 * c + 2 * d], s);
                s = __fmaf_rn(mx_bf(dw >> 16), wf[16 * c + 2 * d + 1], s);
            }
        sub[ee] = s;
    }
}

constexpr int MX_AS_GROUP = 128;            /* K per activation f32 scale      */
constexpr int MX_BN = 64;                   /* output columns a block owns     */
constexpr int MX_RM = 16;                   /* sorted rows a block owns        */
constexpr int MX_WAVES = MX_BN / 16;        /* one 16x16 tile a wave           */

/* A block: MX_RM sorted rows by MX_BN columns, walking the expert runs its rows cover.
 *
 * Lane l of a wave holds row (l & 15) of the A fragment and column (l & 15) of the B fragment
 * (halo_wmma.h: gfx1151 keeps the whole 16-wide k run in a lane), so the A gather is one 16-byte
 * chunk of one activation row a lane and needs no cross-lane step; the D fragment hands that lane
 * eight rows -- 2*e + (l >> 4) -- of its one column, which is the shape of the epilogue below. */
__global__ __launch_bounds__(32 * MX_WAVES) void mx_moe_kernel(
    const uint8_t* __restrict__ aq, const float* __restrict__ as, const void* w0, const void* s0,
    const void* w1, const void* s1, const int* __restrict__ sorted, const int* __restrict__ off,
    uint16_t* __restrict__ y, MxMeta m) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int h = lane >> 4, c = lane & 15;
    const int r0 = blockIdx.x * MX_RM, rend = min(r0 + MX_RM, m.T);
    const int n = blockIdx.y * MX_BN + 16 * wave + c;
    /* This lane's column: which part it belongs to, which of the class's rows that part's tile
     * holds, and whether the column exists for each class at all. A tail block's columns past N
     * are simply not written; a column past a narrower class's width keeps the zero it is given. */
    const bool col_in = n < m.N;
    const int fy = m.N / m.parts;
    const int part = col_in ? n / fy : 0, within = col_in ? n % fy : 0;
    int fc[2], wn[2];
    bool cok[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        fc[q] = m.n[q] / m.parts;
        wn[q] = part * fc[q] + within;
        cok[q] = col_in && within < fc[q];
    }
    /* The expert whose run holds this block's first row: the last e with begin(e) <= r0, as
     * libref's routing search, which puts a row past the last offset in the last expert. */
    int lo = 0, hi = m.experts;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (off[mid] <= r0) lo = mid; else hi = mid;
    }
    for (int e = lo; e < m.experts; ++e) {
        const int s = e == 0 ? 0 : off[e];
        const int end = e + 1 < m.experts ? off[e + 1] : m.T;
        if (s >= rend) break;
        if (end <= r0) continue;
        const int rs = max(s, r0), re = min(end, rend);
        if (re <= rs) continue;
        const int cls = e & 1, kc = m.k[cls];
        /* The A fragment's row is (lane & 15) of the tile; a row outside this run, a dropped slot
         * and a row past the activation's own row count all read as zeros. */
        const int ra = r0 + c;
        int ar = -1;
        if (ra >= rs && ra < re) {
            const int sv = sorted[ra];
            if (sv >= 0) {
                const int t = m.sorted_a ? ra : sv / m.topk;
                if (t < m.Arows) ar = t;
            }
        }
        const uint8_t* ap = ar >= 0 ? aq + (long)ar * m.ar : nullptr;
        /* The activation scale of each of the eight rows this lane accumulates, per 128 of K. */
        int asrow[8];
#pragma unroll
        for (int ee = 0; ee < 8; ++ee) {
            const int r = r0 + 2 * ee + h;
            int t = -1;
            if (r >= rs && r < re) {
                const int sv = sorted[r];
                if (sv >= 0) {
                    const int tt = m.sorted_a ? r : sv / m.topk;
                    if (tt < m.Arows) t = tt;
                }
            }
            asrow[ee] = t;
        }
        const uint8_t* wrow = mx_entry(cls ? w1 : w0, m.wstep[cls], e >> 1) +
                              (long)(cok[cls] ? wn[cls] : 0) * m.wrow[cls];
        const uint8_t* srow = mx_entry(cls ? s1 : s0, m.sstep[cls], e >> 1) +
                              (long)(cok[cls] ? wn[cls] : 0) * m.srow[cls];
        float total[8];
#pragma unroll
        for (int ee = 0; ee < 8; ++ee) total[ee] = 0.f;
        for (int kb = 0; kb < kc; kb += 128) {
            float act[8];
#pragma unroll
            for (int ee = 0; ee < 8; ++ee)
                act[ee] = asrow[ee] >= 0 ? as[(long)asrow[ee] * m.asr + kb / 128] : 1.f;
            unsigned exps = 0x7f7f7f7fu;
            __builtin_memcpy(&exps, srow + kb / 32, 4);
            float amax = 0.f;
#pragma unroll
            for (int ee = 0; ee < 8; ++ee) amax = max(amax, act[ee]);
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                /* The fragments of the group's two 16-wide k steps, and its codes. */
                uint32_t fa[2][8], fw[2][8];
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    const int k = kb + 32 * g + 16 * t;
                    uint4 av = make_uint4(0, 0, 0, 0);
                    if (ap) __builtin_memcpy(&av, ap + k, 16);
                    const uint32_t a4[4] = { av.x, av.y, av.z, av.w };
                    unsigned dw0, dw1;
                    __builtin_memcpy(&dw0, wrow + (k >> 1), 4);
                    __builtin_memcpy(&dw1, wrow + (k >> 1) + 4, 4);
                    union { v16bf f; uint32_t u[8]; } qa, qw;
                    qa.f = halo::deq16_e4m3(a4);
                    qw.f = mx_wfrag(dw0, dw1);
#pragma unroll
                    for (int d = 0; d < 8; ++d) {
                        fa[t][d] = qa.u[d];
                        fw[t][d] = qw.u[d];
                    }
                }
                const float sb = mx_e8m0((exps >> (8 * g)) & 0xFFu);
                /* The two 16-wide k steps go to separate accumulators, added in F32 rather than
                 * chained through the mma's own C: the products are exact, and a pair that
                 * cancels then cancels in the reference's terms. A scale large enough to magnify
                 * the mma's last-bit rounding past the tolerance takes the exact fold. */
                v8f sub;
                if (__any(sb * amax > 1.f)) {
                    union { float f[8]; v8f v; } q;
                    mx_exact8(fa, fw, h, q.f);
                    sub = q.v;
                } else {
                    union { float f[8]; v8f v; } q;
                    q.v = (v8f)(0.f);
#pragma unroll
                    for (int t = 0; t < 2; ++t) {
                        union { uint32_t u[8]; v16bf f; } ua, uw;
#pragma unroll
                        for (int d = 0; d < 8; ++d) {
                            ua.u[d] = fa[t][d];
                            uw.u[d] = fw[t][d];
                        }
                        q.v += halo::mma(ua.f, uw.f, (v8f)(0.f));
                    }
                    sub = q.v;
                }
                /* libref's quant_dot order, (sub * sa) * sb, so the numbers are its numbers. */
#pragma unroll
                for (int ee = 0; ee < 8; ++ee) total[ee] += (sub[ee] * act[ee]) * sb;
            }
        }
#pragma unroll
        for (int ee = 0; ee < 8; ++ee) {
            const int r = r0 + 2 * ee + h;
            if (r < rs || r >= re || !col_in) continue;
            /* A dropped slot inside a live run keeps the zero: its fold is 0 * scale, and a
             * column whose E8M0 byte is 255 makes that 0 * NaN. Column past the class's width
             * in this part likewise. */
            y[(long)r * m.yr + n] = (cok[cls] && asrow[ee] >= 0) ? halo::f32_to_bf16(total[ee])
                                                                 : (uint16_t)0;
        }
    }
}

/* The expert-parity table a weight or exponent plane is stored in: [e][n][k] with the parity's
 * own n and k, `bits` wide, and either a pointer table or one plane per (ne + 1 - q) / 2 experts. */
int mx_table(const RadTensor* t, uint32_t dtype, int bits, long experts, long n, long k,
             long* step, int* row) {
    if (!t || !t->data || t->rank != 3 || t->dtype != dtype || t->shape[0] < experts ||
        t->shape[1] != n || t->shape[2] != k || t->stride[2] != 1 || t->stride[1] < k ||
        (t->stride[1] * bits) % 8 || t->stride[1] * bits / 8 > INT_MAX)
        return HALO_NO(RAD_E_SHAPE);
    if (t->stride[0] && (t->stride[0] < n * t->stride[1] || (t->stride[0] * bits) % 8))
        return HALO_NO(RAD_E_STRIDE);
    *step = t->stride[0] * bits / 8;
    *row = (int)(t->stride[1] * bits / 8);
    return RAD_OK;
}

int mx_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 9) return HALO_NO(RAD_E_INVAL);
    const RadTensor* aq = rad_arg_in(a, 0);
    const RadTensor* as = rad_arg_in(a, 1);
    const RadTensor* sorted = rad_arg_in(a, 4);
    const RadTensor* off = rad_arg_in(a, 5);
    const RadTensor* y = rad_arg_in(a, 8);
    if (!aq || !as || !sorted || !off || !y) return HALO_NO(RAD_E_INVAL);
    MxMeta m{};
    const long n0 = halo_geti_or(a, "N", 0), k0 = halo_geti_or(a, "K", 0);
    const long nn[2] = { n0, halo_geti_or(a, "N_odd", n0) },
               kk[2] = { k0, halo_geti_or(a, "K_odd", k0) };
    const long ne = halo_geti_or(a, "n_expert", 0), top = halo_geti_or(a, "top_k", 0),
               parts = halo_geti_or(a, "parts", 1);
    /* `group` is the activation scale's block along K (one f32 per 128 values), not the MX
     * format's 32-value exponent group, which is fixed and not a parameter. */
    if (ne < 2 || ne > 512 || top < 1 || top > ne || parts < 1 || parts > INT_MAX ||
        halo_geti_or(a, "group", 0) != MX_AS_GROUP)
        return HALO_NO(RAD_E_SHAPE);
    const RadTensor* w[2] = { rad_arg_in(a, 2), rad_arg_in(a, 6) };
    const RadTensor* s[2] = { rad_arg_in(a, 3), rad_arg_in(a, 7) };
    for (int q = 0; q < 2; ++q) {
        if (nn[q] < 1 || nn[q] > INT_MAX - 16 || kk[q] < 128 || kk[q] > INT_MAX || kk[q] % 128 ||
            nn[q] % parts)
            return HALO_NO(RAD_E_SHAPE);
        m.n[q] = (int)nn[q];
        m.k[q] = (int)kk[q];
        int rc = mx_table(w[q], RAD_FP4E2M1, 4, (ne + 1 - q) / 2, nn[q], kk[q], &m.wstep[q],
                          &m.wrow[q]);
        if (rc < 0) return rc;
        const uint32_t dt = s[q] ? s[q]->dtype : 0;
        if (dt != RAD_U8 && dt != RAD_E8M0) return HALO_NO(RAD_E_DTYPE);
        rc = mx_table(s[q], dt, 8, (ne + 1 - q) / 2, nn[q], kk[q] / 32, &m.sstep[q], &m.srow[q]);
        if (rc < 0) return rc;
    }
    m.N = (int)std::max(nn[0], nn[1]);
    m.K = (int)std::max(kk[0], kk[1]);
    if (aq->dtype != RAD_F8E4M3 || as->dtype != RAD_F32 || y->dtype != RAD_BF16 ||
        sorted->dtype != RAD_I32 || off->dtype != RAD_I32)
        return HALO_NO(RAD_E_DTYPE);
    if (aq->rank != 2 || aq->shape[0] > INT_MAX || aq->shape[1] != m.K || aq->stride[1] != 1 ||
        aq->stride[0] < m.K || as->rank != 2 || as->shape[0] != aq->shape[0] ||
        as->shape[1] != m.K / 128 || as->stride[1] != 1 || as->stride[0] < m.K / 128 ||
        y->rank != 2 || y->shape[0] < 1 || y->shape[0] > INT_MAX || y->shape[1] != m.N ||
        y->stride[1] != 1 || y->stride[0] < m.N || sorted->rank != 1 || sorted->stride[0] != 1 ||
        sorted->shape[0] < y->shape[0] || off->rank != 1 || off->stride[0] != 1 ||
        off->shape[0] < ne + 1)
        return HALO_NO(RAD_E_SHAPE);
    const char* ord = halo_gets(a, "a_order");
    if (!ord || (std::strcmp(ord, "token") && std::strcmp(ord, "sorted"))) return HALO_NO(RAD_E_INVAL);
    m.sorted_a = !std::strcmp(ord, "sorted");
    m.T = (int)y->shape[0];
    m.Arows = (int)aq->shape[0];
    if (m.T % top || m.Arows < (m.sorted_a ? m.T : m.T / top)) return HALO_NO(RAD_E_SHAPE);
    m.experts = (int)ne;
    m.topk = (int)top;
    m.parts = (int)parts;
    m.ar = aq->stride[0];
    m.asr = as->stride[0];
    m.yr = y->stride[0];
    const long gx = ((long)m.T + MX_RM - 1) / MX_RM, gy = ((long)m.N + MX_BN - 1) / MX_BN;
    if (gx < 1 || gx > 2147483647L || gy < 1 || gy > 65535L) return HALO_NO(RAD_E_SHAPE);
    hipStream_t st = (hipStream_t)stream;
    hipLaunchKernelGGL(mx_moe_kernel, dim3((unsigned)gx, (unsigned)gy), dim3(32 * MX_WAVES), 0,
                       st, (const uint8_t*)aq->data, (const float*)as->data, w[0]->data,
                       s[0]->data, w[1]->data, s[1]->data, (const int*)sorted->data,
                       (const int*)off->data, (uint16_t*)y->data, m);
    return RAD_OK;
}

}  // namespace

extern "C" int halo_rad_moe_mxfp4(const RadArgs* a, RadStream stream) { return mx_body(a, stream); }
