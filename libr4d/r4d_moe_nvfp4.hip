// R4D grouped GEMM over NVFP4 experts against an int8 activation: moe_gemm_q at dtype nvfp4a8i8.
//
// An NVFP4 expert is e2m1 codes, an E4M3 scale per (row, 16 of K) and one f32 level for the whole
// expert (amax / denominator). Its value is e2m1 * e4m3 * level. The e2m1 magnitudes are {0, 0.5,
// 1, 1.5, 2, 3, 4, 6}, so twice them are the integers {0, 1, 2, 3, 4, 6, 8, 12}: an int8 activation
// multiplies those exactly, and the 1/2 is folded into the level. Each 16-wide group of K is one
// int32 sum (at most 16 * 12 * 127 in magnitude), folded to f32 under its scale and the
// activation's scale for that group.
//
// The kernel is deliberately plain: one thread a output column, one block a sorted row, no WMMA
// and no staging. It is correct on every band (decode and prefill alike, since it takes no
// run map); tuning, if it is ever needed, is the performance PR's.
//
// Operands and parameters are moe_gemm_q's (r4d_rows.cpp, `moe_gemm_nvfp4a8i8`): `a` int8 [M, K]
// with an f32 scale per (row, a_group), `w` the even experts' e2m1 codes [E, N, K] (or a pointer
// table, stride 0), `w_scale` their stored scale blobs (r4d_layout.cpp, nvfp4_layout), and the
// odd class's `w_odd` and `w_odd_scale` at N_odd and K_odd.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_plugin.h"
#include "r4d_emit.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>

namespace {

#define R4D_NV_CHK() do { hipError_t e_ = hipGetLastError();                       \
                          if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

constexpr int kNvBlock = 64;   /* output columns a block, one thread each */

/* One operand's experts: `data` the stacked base when `s0` is non-zero (bytes an expert), else the
 * pointer table of the experts' planes. */
struct NvTab {
    const uint8_t* data;
    size_t         s0;
};
__device__ __forceinline__ const uint8_t* nv_entry(NvTab t, uint32_t e) {
    if (t.s0 == 0) return ((const uint8_t* const*)t.data)[e];
    return t.data + (size_t)e * t.s0;
}

/* e2m1's magnitudes doubled, by the low three bits of a code; bit 3 is the sign. */
__device__ __forceinline__ int nv_level2(uint32_t c) {
    const int mag2[8] = { 0, 1, 2, 3, 4, 6, 8, 12 };
    const int v = mag2[c & 7];
    return (c & 8) ? -v : v;
}

/* E4M3 (bias 7) to f32. The scales are finite and positive in practice; NaN decodes as NaN. */
__device__ __forceinline__ float nv_e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 0) v = (float)m * 0.001953125f;                     /* m * 2^-9 */
    else if (e == 15 && m == 7) v = __builtin_nanf("");
    else v = (1.0f + (float)m * 0.125f) * (float)(1 << e) * 0.0078125f;   /* 2^(e-7) */
    return s ? -v : v;
}

/* Round-to-nearest-even f32 to bf16 bits. */
__device__ __forceinline__ uint16_t nv_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);   /* NaN */
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

/* One output element: sorted row `i`, column `col`, in the class of expert `e`. */
struct NvArgs {
    const int8_t* a;  size_t a_ld;     /* activation codes, row pitch in elements */
    const float*  as; size_t as_ld;    /* activation scales, row pitch in elements */
    int64_t       AR;                  /* rows the activation addresses */
    int           a_sorted;            /* 1: row i of `a` is sorted row i; 0: its token */
    int           a_group;             /* K the activation scale covers */
    int           top_k;
    int           n_expert;
    int           N;                   /* output width, the wider class's */
    int           parts;
    int           N_c[2], K_c[2];      /* per class: weight rows and K */
    NvTab         w[2], ws[2];         /* per class: codes and scale blobs */
};

__global__ __launch_bounds__(kNvBlock) void nvfp4_moe_kernel(
    NvArgs p, const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
    uint16_t* __restrict__ y, size_t y_ld) {
    const int i = blockIdx.x;
    const int col = blockIdx.y * kNvBlock + threadIdx.x;
    if (col >= p.N) return;

    /* The row's expert and token, as libref's binary search over the offsets gives them: the last
     * expert whose offset is at most i, and no expert for a dropped slot or a token past `a`. */
    const int32_t flat = sorted[i];
    int e = -1;
    int64_t tok = 0;
    if (flat >= 0) {
        int lo = 0, hi = p.n_expert;
        while (lo + 1 < hi) {
            const int mid = (lo + hi) / 2;
            if (off[mid] <= i) lo = mid; else hi = mid;
        }
        tok = p.a_sorted ? (int64_t)i : (int64_t)flat / p.top_k;
        if (tok < p.AR) e = lo;
    }
    if (e < 0) { y[(size_t)i * y_ld + col] = 0; return; }

    /* The class (even experts in `w`, odd in `w_odd`), its entry, and the column's weight row: the
     * output's parts are stacked, a narrower class fills each part's leading columns and zeros
     * after them. */
    const int q = e & 1;
    const uint32_t ix = (uint32_t)(e >> 1);
    const int64_t n_q = p.N_c[q], k_q = p.K_c[q];
    const int64_t fy = p.N / p.parts, fc = n_q / p.parts;
    const int64_t pp = col / fy, j = col - pp * fy;
    if (j >= fc) { y[(size_t)i * y_ld + col] = 0; return; }
    const int64_t wr = pp * fc + j;

    const uint8_t* crow = nv_entry(p.w[q], ix) + (size_t)wr * (size_t)(k_q / 2);
    const uint8_t* sblob = nv_entry(p.ws[q], ix);
    const uint8_t* srow = sblob + (size_t)wr * (size_t)(k_q / 16);
    /* The expert's level follows its N_q rows of scales (the stored blob's last row, four bytes). */
    float level;
    __builtin_memcpy(&level, sblob + (size_t)n_q * (size_t)(k_q / 16), sizeof level);
    const float lv = 0.5f * level;   /* the doubled e2m1 levels' 1/2 */

    const int8_t* arow = p.a + (size_t)tok * p.a_ld;
    const float* sarow = p.as + (size_t)tok * p.as_ld;
    float acc = 0.0f;
    for (int64_t g = 0; g < k_q / 16; ++g) {
        int32_t sub = 0;
        for (int kk = 0; kk < 16; ++kk) {
            const int64_t k = g * 16 + kk;
            const uint32_t byte = crow[k >> 1];
            const uint32_t c = (k & 1) ? (byte >> 4) : (byte & 15u);
            sub += nv_level2(c) * (int32_t)arow[k];
        }
        const float sa = sarow[(g * 16) / p.a_group];
        acc += (float)sub * sa * (nv_e4m3(srow[g]) * lv);
    }
    y[(size_t)i * y_ld + col] = nv_bf16(acc);
}

/* A class's operand: a rank-3 stacked tensor [entries, rows, cols], or a pointer table with stride
 * 0 on its first axis. `bits` is the element width (4 for the e2m1 codes). */
bool nv_table(const RadTensor* t, uint32_t dtype, int64_t entries, int64_t rows, int64_t cols,
              int bits, NvTab* out) {
    if (!t || !t->data || t->rank != 3 || t->dtype != dtype) return false;
    if (t->shape[0] < entries || t->shape[1] < rows || t->shape[2] < cols) return false;
    if (t->stride[2] != 1 || t->stride[1] != cols) return false;
    out->data = (const uint8_t*)t->data;
    out->s0 = (size_t)((t->stride[0] * (int64_t)bits) / 8);
    /* a stacked entry is a whole number of bytes */
    if (t->stride[0] != 0 && (t->stride[0] * (int64_t)bits) % 8) return false;
    return true;
}

/* A class's scale blobs (r4d.nvfp4.s): `rows` rows of `cols` bytes, then the level's row. The pitch
 * is derived, not read, as moe_gemm_launch derives the codebook forms': the core hands the view at
 * the extent the plugin declared, the logical [N, K] with row stride K, while a stacked plane may
 * carry the blob's own [N + 1, K / 16]. The layout tag checked at load is what makes the bytes the
 * blob; here the entries need only be distinct. */
bool nv_scale_table(const RadTensor* t, int64_t entries, int64_t rows, int64_t cols, NvTab* out) {
    if (!t || !t->data || t->rank != 3 || t->dtype != RAD_U8) return false;
    if (t->shape[0] < entries || t->shape[1] < rows || t->stride[2] != 1) return false;
    if (t->stride[0] != 0 && t->stride[0] < (rows + 1) * cols) return false;
    out->data = (const uint8_t*)t->data;
    out->s0 = (size_t)t->stride[0];
    return true;
}

}  // namespace

extern "C" int r4d_rad_moe_gemm_nvfp4a8i8(const RadArgs* a, RadStream stream) {
    const RadTensor* Aq = r4d_opt(a, 0);
    const RadTensor* As = r4d_opt(a, 1);
    const RadTensor* Wc[2] = { r4d_opt(a, 2), r4d_opt(a, 6) };
    const RadTensor* Ws[2] = { r4d_opt(a, 3), r4d_opt(a, 7) };
    const RadTensor* sorted = r4d_opt(a, 4);
    const RadTensor* off = r4d_opt(a, 5);
    const RadTensor* Y = r4d_opt(a, 8);
    if (!Aq || !As || !Wc[0] || !Ws[0] || !Wc[1] || !Ws[1] || !sorted || !off || !Y)
        return R4D_NO(RAD_E_INVAL);
    if (!Aq->data || !As->data || !sorted->data || !off->data || !Y->data)
        return R4D_NO(RAD_E_INVAL);

    const long long N0 = r4d_geti_or(a, "N", 0), K0 = r4d_geti_or(a, "K", 0);
    const long long N1 = r4d_geti_or(a, "N_odd", N0), K1 = r4d_geti_or(a, "K_odd", K0);
    const long long parts = r4d_geti_or(a, "parts", 1);
    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long top_k = r4d_geti_or(a, "top_k", 0);
    const long long group = r4d_geti_or(a, "group", 128);
    const long long a_group = r4d_geti_or(a, "a_group", group);
    if (parts <= 0 || n_expert <= 0 || top_k <= 0 || N0 <= 0 || K0 <= 0 || N1 <= 0 || K1 <= 0)
        return R4D_NO(RAD_E_SHAPE);
    /* The layout is whole 16-wide groups, four to a word of scale bytes, and the activation's
     * scale covers whole groups. */
    if (K0 % 64 || K1 % 64 || N0 % parts || N1 % parts || a_group % 16 || group % 16 ||
        a_group <= 0)
        return R4D_NO(RAD_E_SHAPE);
    const char* ord = r4d_gets(a, "a_order");
    const bool a_sorted = ord && !std::strcmp(ord, "sorted");
    if (ord && !a_sorted && std::strcmp(ord, "token")) return R4D_NO(RAD_E_UNSUPPORTED);

    const long long N = N0 > N1 ? N0 : N1, K = K0 > K1 ? K0 : K1;
    if (K % a_group) return R4D_NO(RAD_E_SHAPE);
    if (Aq->dtype != RAD_I8 || As->dtype != RAD_F32 || Y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (sorted->dtype != RAD_I32 || off->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);

    long long T = 0;
    const long long y_ld = r4d_rows_pitch(Y, N, &T);
    if (y_ld < 0 || T <= 0) return R4D_NO(RAD_E_STRIDE);
    long long M = 0;
    const long long a_ld = r4d_rows_pitch(Aq, K, &M);
    if (a_ld < 0 || M <= 0) return R4D_NO(RAD_E_STRIDE);
    long long Ms = 0;
    const long long as_ld = r4d_rows_pitch(As, K / a_group, &Ms);
    if (as_ld < 0 || Ms < M) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(sorted) < T || r4d_numel(off) < n_expert + 1) return R4D_NO(RAD_E_SHAPE);
    const long long AR = a_sorted ? T : (T + top_k - 1) / top_k;
    if (M < AR) return R4D_NO(RAD_E_SHAPE);

    NvArgs p{};
    p.a = (const int8_t*)Aq->data;
    p.a_ld = (size_t)a_ld;
    p.as = (const float*)As->data;
    p.as_ld = (size_t)as_ld;
    p.AR = AR;
    p.a_sorted = a_sorted ? 1 : 0;
    p.a_group = (int)a_group;
    p.top_k = (int)top_k;
    p.n_expert = (int)n_expert;
    p.N = (int)N;
    p.parts = (int)parts;
    const long long Nc[2] = { N0, N1 }, Kc[2] = { K0, K1 };
    for (int q = 0; q < 2; ++q) {
        const long long entries = (n_expert + 1 - q) / 2;   /* ceil((n_expert - q) / 2) */
        p.N_c[q] = (int)Nc[q];
        p.K_c[q] = (int)Kc[q];
        /* codes: [entries, N_q, K_q] e2m1, two codes a byte; scales: [entries, N_q + 1, K_q / 16]
         * bytes, each expert's last row holding its f32 level */
        if (!nv_table(Wc[q], RAD_FP4E2M1, entries, Nc[q], Kc[q], 4, &p.w[q]))
            return R4D_NO(RAD_E_SHAPE);
        if (!nv_scale_table(Ws[q], entries, Nc[q], Kc[q] / 16, &p.ws[q]))
            return R4D_NO(RAD_E_SHAPE);
    }

    const unsigned groups = (unsigned)((N + kNvBlock - 1) / kNvBlock);
    const int emit_rc = R4D_EMIT(stream, nvfp4_moe_kernel, dim3((unsigned)T, groups),
                                 dim3(kNvBlock), 0, p, (const int32_t*)sorted->data,
                                 (const int32_t*)off->data, (uint16_t*)Y->data, (size_t)y_ld);
    if (emit_rc != RAD_OK) return emit_rc;
    R4D_NV_CHK();
    return RAD_OK;
}
