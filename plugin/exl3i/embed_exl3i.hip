/* embed_exl3i.hip -- the device row of embed_exl3i (embed_exl3i.h): exactly exl3i_decode_rows' row of the token, RNE bf16.
 * A workgroup of 256 threads does one (token, 128 columns) chunk at a time. The block's 64 tiles go to LDS as stored
 * (8 strips x 8 tiles, m / 2 words each). Then, 32 block rows a pass, 8 threads a row: each thread decodes 16 consecutive
 * columns of its row from the ring (exl3i_decode.h's state windows), the row's 128-point butterfly over k runs in the
 * oracle's stages (h = 1 .. 8 in registers, 16 / 32 / 64 across the row's 8 lanes by shuffles), x (float)(1/sqrt(128)),
 * x suh -- the oracle's f32 operations, each on the same operands; then 128 column threads fold the pass's 32 rows by the
 * output Hadamard's cone for the token's row j: at stage h the lower half's value plus or minus the upper half's (bit h
 * of j), adjacent rows first -- the exact additions the oracle's in-place butterfly over n makes for element j; the 4
 * passes' partials meet at h = 32 and 64; x (float)(1/sqrt(128)), x svh, one RNE to bf16. Every codebook's value is a
 * function of one byte sum alone: mad's T = bytesum((state A + B) & 0x3F3F3F3F) in [0, 252] (q = T - 126, the value
 * fp16(q 4 K_INV)); mad is the one codebook. Each workgroup first tabulates the values in LDS by the
 * oracle's arithmetic, and a value is a multiply-free index (a byte sum) and one LDS read.
 * A batch repeats tokens (spaces, punctuation, common words): from EMBED_EXL3I_DEDUP_M tokens on, a workgroup takes a
 * token, one whose id already occurs earlier in the batch returns at once, and a second launch copies each repeated
 * token's row from its first occurrence (the same bits: the row is a function of the id).
 * Nothing is expanded: the record, suh and svh are read where the container put them. */
#include <cstring>
#include <hip/hip_runtime.h>

#include "embed_exl3i.h"
#include "exl3dense_device.hip.h"
#include "exl3i_decode.h"
#include "rad_plugin.h"

namespace {

using namespace exl3dense_kernel;

/* exl3_tile_to_pos: tile row r (k axis), column c (n axis) -> ring position */
__device__ __forceinline__ int tile_pos(int r, int c) {
    const int lane = 4 * (c % 8) + (r % 8) / 2;
    const int j = 4 * (c / 8) + ((r & 1) | ((r >> 3) << 1));
    return 8 * lane + j;
}

/* exl3i_codebook_value's f32 value of a state whose index is S (embed_index): mad's fp16(q 4 K_INV), q = S - 126. */
template <int CB>
__device__ __forceinline__ float cb_value_s(unsigned S) {
    static_assert(CB == EXL3I_CB_MAD, "the embedding gather serves mad");
    const float s4 = 4.0f * half_to_float(kKInv);
    return half_to_float(f32_to_f16(__fmul_rn((float)((int)S - 126), s4)));   /* q 4 K_INV: exact, one rounding */
}

/* The LUT index of a state: mad's T = bytesum((state A + B) & 0x3F3F3F3F), the oracle's exl3i_mad_sum (the same 32-bit
 * arithmetic on the 16-bit state). sad_u8 against 0 is the byte sum. */
template <int CB>
__device__ __forceinline__ unsigned embed_index(unsigned state) {
    static_assert(CB == EXL3I_CB_MAD, "the embedding gather serves mad");
    return __builtin_amdgcn_sad_u8(((state * EXL3I_MAD_A) + EXL3I_MAD_B) & 0x3F3F3F3Fu, 0u, 0u);
}

/* From this many tokens a launch takes a token a workgroup and decodes each distinct id once (below it, a (token,
 * 128 columns) grid: decode steps need the 40-way parallelism of one token's row). */
constexpr long long EMBED_EXL3I_DEDUP_M = 64;

struct Lds {
    unsigned words[64 * (EXL3I_M_MAX / 2 + 1)];   /* 8 strips x 8 tiles, m / 2 + 1 words apart (an odd stride: the 8
                                                     * lanes of a row read 8 tiles in 8 different banks) */
    float    lut[1024];                 /* the codebook's value by byte sum S */
    float    R[32][129];                /* a pass's 32 rows after the k rotation and suh (padded) */
    int      red;
};

constexpr float kHad = 0.08838834764831844055f;   /* exl3_had128's (float)(1/sqrt(128)) */

/* The first position j <= m of tokens[m]'s id, the whole workgroup scanning [0, m) strided (nt threads); m when none. */
__device__ __forceinline__ long long first_of(const int32_t* tok, long long m, int t, int nt, int* red) {
    const int id = tok[m];
    int f = (int)m;
    for (long long j = t; j < m; j += nt)
        if (tok[j] == id) { f = (int)j; break; }
    if (t == 0) *red = (int)m;
    __syncthreads();
    if (f < m) atomicMin(red, f);
    __syncthreads();
    return *red;
}

template <int CB>
__device__ __forceinline__ void fill_lut(Lds& L, int t) {
    for (int i = t; i < 1021; i += 256) L.lut[i] = cb_value_s<CB>((unsigned)i);
}

/* Columns [128 g, 128 g + 128) of stored row n into xr (the token's output row), exactly exl3i_decode_rows' values: the
 * whole workgroup (256 threads). The LUT must be filled. */
template <int CB>
__device__ __forceinline__ void chunk(const EmbedExl3iCall& c, Lds& L, long long n, int g, unsigned short* xr, int t) {
    const long long b = n >> 7;
    const int j = (int)(n & 127), W = c.m >> 1, sw = 8 * W, TW = W + 1;
    for (int i = t; i < 8 * sw; i += 256) {   /* tile (st, tk) at (8 st + tk) TW */
        const int st = i / sw, o = i - st * sw;
        L.words[(8 * st + o / W) * TW + o % W] = c.record[(b * 8 + st) * c.row_words + (long long)g * sw + o];
    }
    /* phase A: thread t owns row lr of each 32-row pass and columns [16 s, 16 s + 16) of the chunk (tile s of its strip).
     * Its tile column ct = lr % 16 is the same in every pass, so its 16 ring windows (tile rows rt = 0..15) are fixed. */
    const int lr = t >> 3, s = t & 7, ct = lr & 15, ka = c.m >> 4, nb = c.m & 15, R = 16 * c.m;
    int wa[16], wb[16];
    unsigned sh[16], salt[16];
#pragma unroll
    for (int rt = 0; rt < 16; ++rt) {
        const int p = tile_pos(rt, ct), q = p & 15;
        const int S = ka * (p + 1) + (p >> 4) * nb + (((q + 1) * nb) >> 4);   /* ring bits read through p */
        int start = S - 16;
        if (start < 0) start += R;
        wa[rt] = start >> 5;
        wb[rt] = wa[rt] + 1 == W ? 0 : wa[rt] + 1;
        sh[rt] = 48u - (unsigned)(start & 31);
        const int d = ka + ((((q + 1) * nb) >> 4) - ((q * nb) >> 4));
        salt[rt] = d ? 0u : (0x9E37u * (unsigned)(q + 1)) & 0xFFFFu;
    }
    float su[16];
#pragma unroll
    for (int rt = 0; rt < 16; ++rt) su[rt] = half_to_float(c.suh[128 * g + 16 * s + rt]);
    __syncthreads();
    float T[4] = {0.f, 0.f, 0.f, 0.f};   /* phase B (t < 128: column t): each pass's 32-row partial */
    for (int pass = 0; pass < 4; ++pass) {
        const int r = 32 * pass + lr;
        const unsigned* tw = L.words + ((r >> 4) * 8 + s) * TW;
        float v[16];
#pragma unroll
        for (int rt = 0; rt < 16; ++rt) {
            const unsigned long long w2 = ((unsigned long long)tw[wa[rt]] << 32) | tw[wb[rt]];
            const unsigned state = ((unsigned)(w2 >> sh[rt]) & 0xFFFFu) ^ salt[rt];
            v[rt] = L.lut[embed_index<CB>(state)];
        }
        /* exl3_had128 over the row's 128 columns: element 16 s + rt; stages 1..8 inside the thread */
#pragma unroll
        for (int h = 1; h < 16; h <<= 1)
#pragma unroll
            for (int i = 0; i < 16; ++i)
                if (!(i & h)) {
                    const float x0 = v[i], x1 = v[i + h];
                    v[i] = __fadd_rn(x0, x1);
                    v[i + h] = __fsub_rn(x0, x1);
                }
        /* stages 16, 32, 64: the partner is lane s ^ (h / 16) of the row's 8 (consecutive lanes of one wave) */
#pragma unroll
        for (int q = 1; q < 8; q <<= 1)
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float o = __shfl_xor(v[i], q, 8);
                v[i] = (s & q) ? __fsub_rn(o, v[i]) : __fadd_rn(v[i], o);
            }
#pragma unroll
        for (int i = 0; i < 16; ++i) L.R[lr][16 * s + i] = __fmul_rn(__fmul_rn(v[i], kHad), su[i]);
        __syncthreads();
        if (t < 128) {   /* the cone of element j of the butterfly over the pass's 32 rows: stages 1, 2, 4, 8, 16 */
            float u[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float x0 = L.R[2 * i][t], x1 = L.R[2 * i + 1][t];
                u[i] = (j & 1) ? __fsub_rn(x0, x1) : __fadd_rn(x0, x1);
            }
#pragma unroll
            for (int h = 2, w = 8; h < 32; h <<= 1, w >>= 1)
#pragma unroll
                for (int i = 0; i < w; ++i) u[i] = (j & h) ? __fsub_rn(u[2 * i], u[2 * i + 1]) : __fadd_rn(u[2 * i], u[2 * i + 1]);
            T[pass] = u[0];
        }
        __syncthreads();   /* R is rewritten by the next pass */
    }
    if (t < 128) {
        const float l = (j & 32) ? __fsub_rn(T[0], T[1]) : __fadd_rn(T[0], T[1]);
        const float h = (j & 32) ? __fsub_rn(T[2], T[3]) : __fadd_rn(T[2], T[3]);
        const float y = (j & 64) ? __fsub_rn(l, h) : __fadd_rn(l, h);
        xr[128LL * g + t] = f32_to_bf16(__fmul_rn(__fmul_rn(y, kHad), half_to_float(c.svh[n])));
    }
    /* the next chunk's word loads only overwrite words, which every thread has finished reading (the last barrier) */
}

/* Small batches (decode): a workgroup a (token, 128 columns). */
template <int CB>
__global__ __launch_bounds__(256) void embed_exl3i_cols(EmbedExl3iCall c) {
    __shared__ Lds L;
    const int t = (int)threadIdx.x;
    const long long mi = blockIdx.x;
    const int g = (int)blockIdx.y;
    unsigned short* xr = c.x + mi * c.x_pitch;
    const long long id = c.tokens[mi];
    if (id < 0 || id >= c.n_vocab) {
        if (t < 128) xr[128LL * g + t] = 0;
        return;
    }
    fill_lut<CB>(L, t);
    chunk<CB>(c, L, embed_exl3i_row(id, c.n_vocab, c.r16), g, xr, t);
}

/* Large batches (prefill): a workgroup a token, its row's 128-column chunks in turn; a token whose id occurs earlier in
 * the batch returns at once (embed_exl3i_copy writes its row). */
template <int CB>
__global__ __launch_bounds__(256) void embed_exl3i_rows(EmbedExl3iCall c) {
    __shared__ Lds L;
    const int t = (int)threadIdx.x;
    const long long mi = blockIdx.x;
    unsigned short* xr = c.x + mi * c.x_pitch;
    const long long id = c.tokens[mi];
    if (id < 0 || id >= c.n_vocab) {
        for (long long k = t; k < c.n_embd; k += 256) xr[k] = 0;
        return;
    }
    if (mi > 0 && first_of(c.tokens, mi, t, 256, &L.red) != mi) return;
    fill_lut<CB>(L, t);
    const long long n = embed_exl3i_row(id, c.n_vocab, c.r16);
    for (int g = 0; g < (int)(c.n_embd / 128); ++g) chunk<CB>(c, L, n, g, xr, t);
}

/* A repeated token's row (an id seen earlier in the batch): copied from its first occurrence after embed_exl3i_rows. */
__global__ __launch_bounds__(256) void embed_exl3i_copy(EmbedExl3iCall c) {
    __shared__ int red;
    const int t = (int)threadIdx.x;
    const long long mi = blockIdx.x;
    const long long id = c.tokens[mi];
    if (mi == 0 || id < 0 || id >= c.n_vocab) return;
    const long long f = first_of(c.tokens, mi, t, 256, &red);
    if (f == mi) return;
    for (long long k = t; k < c.n_embd; k += 256) c.x[mi * c.x_pitch + k] = c.x[f * c.x_pitch + k];
}

}  /* namespace */

extern "C" int embed_exl3i_device_launch(const RadArgs* a, RadStream stream) {
    EmbedExl3iCall c;
    memset(&c, 0, sizeof c);   /* no stack garbage in the padding: a recorded launch replays byte for byte */
    const int rc = embed_exl3i_args(a, &c);
    if (rc != RAD_OK) return rc;
    if (c.M <= 0) return RAD_OK;
    if (c.M > 0x7fffffffLL || c.n_embd / 128 > 65535) return RAD_E_SHAPE;
    const hipStream_t st = (hipStream_t)stream;
    if (c.M < EMBED_EXL3I_DEDUP_M) {
        const dim3 grid((unsigned)c.M, (unsigned)(c.n_embd / 128));
        switch (c.cb) {
        case EXL3I_CB_MAD:    hipLaunchKernelGGL(embed_exl3i_cols<EXL3I_CB_MAD>, grid, dim3(256), 0, st, c); break;
        default: return RAD_E_INVAL;
        }
    } else {
        const dim3 grid((unsigned)c.M);
        switch (c.cb) {
        case EXL3I_CB_MAD:    hipLaunchKernelGGL(embed_exl3i_rows<EXL3I_CB_MAD>, grid, dim3(256), 0, st, c); break;
        default: return RAD_E_INVAL;
        }
        hipLaunchKernelGGL(embed_exl3i_copy, grid, dim3(256), 0, st, c);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
