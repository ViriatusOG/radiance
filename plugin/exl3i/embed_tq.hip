/* embed_tq.hip -- the device row of embed_tq (embed_tq.h): one workgroup of 128 threads a token, one thread a
 * column of each 128-column block. The decode and the butterfly are the host row's, step for step (the same
 * f32 products and sums in the same order), so the two give the same bits. */
#include <cstring>
#include <hip/hip_runtime.h>

#include "embed_tq.h"
#include "rad_plugin.h"

namespace {

__device__ __forceinline__ float bf16f(unsigned short h) { return __uint_as_float((unsigned)h << 16); }

__device__ __forceinline__ unsigned short f32bf16(float f) {
    const unsigned u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x40);
    return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__global__ __launch_bounds__(128) void embed_tq_kernel(EmbedTqCall c) {
    __shared__ float v[128];
    const int t = (int)threadIdx.x;
    const long long m = blockIdx.x;
    unsigned short* xr = c.x + m * c.x_pitch;
    const long long id = c.tokens[m];
    if (id < 0 || id >= c.n_vocab) {
        for (long long k = t; k < c.n_embd; k += 128) xr[k] = 0;
        return;
    }
    const unsigned char* cr = c.codes + id * c.codes_row_bytes;
    const unsigned short* sr = c.scale + id * (c.n_embd / 128);
    const unsigned mask = (1u << c.bits) - 1u;
    for (long long g = 0; g < c.n_embd; g += 128) {
        const long long bit = (g + t) * c.bits;
        unsigned w = cr[bit >> 3];
        if (((bit & 7) + c.bits) > 8) w |= (unsigned)cr[(bit >> 3) + 1] << 8;
        v[t] = c.table[(w >> (bit & 7)) & mask] * bf16f(sr[g / 128]);
        __syncthreads();
        for (int h = 1; h < 128; h <<= 1) {
            float p = 0.f, q = 0.f;
            const bool lo = (t & h) == 0;
            if (lo) { p = v[t]; q = v[t + h]; }
            __syncthreads();
            if (lo) { v[t] = p + q; v[t + h] = p - q; }
            __syncthreads();
        }
        xr[g + t] = f32bf16(v[t]);
        __syncthreads();   /* v is rewritten by the next block */
    }
}

}  /* namespace */

extern "C" int embed_tq_device_launch(const RadArgs* a, RadStream stream) {
    EmbedTqCall c;
    memset(&c, 0, sizeof c);   /* no stack garbage in the padding: a recorded launch replays byte for byte */
    const int rc = embed_tq_args(a, &c);
    if (rc != RAD_OK) return rc;
    if (c.M <= 0) return RAD_OK;
    if (c.M > 0x7fffffffLL) return RAD_E_SHAPE;
    hipLaunchKernelGGL(embed_tq_kernel, dim3((unsigned)c.M), dim3(128), 0, (hipStream_t)stream, c);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
