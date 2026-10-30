/* Derived from the copy helpers in libr4d/r4d_ar_oneshot_2rank_exact.hip by the
 * libhalo rename transform (2026-10-26), then ported to gfx1151; see
 * libhalo/README.md. These live apart from the all-reduce unit until that unit
 * lands, because the model ops' strided cast needs the local copy now. */

// halo_p2p_copy2d as a kernel with a bounded workgroup count.
//
// On the R9700 libr4d measured hipMemcpy2DAsync as `__amd_rocclr_copyBufferRectAligned`, one
// 16-thread workgroup per 16 bytes of a row (62080 workgroups for 64 x 248 KB); alone it reached
// link rate, but beside a GEMM it filled every scheduling slot. A peer copy needed only a small
// grid there: 32 workgroups of 64 threads was within 0.1% of the best configuration at every
// size. Neither measurement has been repeated on gfx1151 yet; the grid remains an argument and
// defaults to at most 64 workgroups.
//
// 16-byte accesses: every pitch, the width and both base addresses must be 16-byte aligned; the
// caller checks and falls back to halo_p2p_copy2d otherwise.
#include "halo.h"
#include <hip/hip_runtime.h>
#include <cstdint>

typedef unsigned halo_u4v __attribute__((ext_vector_type(4)));

__global__ __launch_bounds__(256) void halo_p2p_copy2d_k(const halo_u4v* __restrict__ src,
                                                         halo_u4v* __restrict__ dst, long spitch16,
                                                         long dpitch16, long wpr, int height) {
  const long cstride = (long)gridDim.x * blockDim.x;
  for (int r = blockIdx.y; r < height; r += gridDim.y) {
    const halo_u4v* s = src + (long)r * spitch16;
    halo_u4v* d = dst + (long)r * dpitch16;
    for (long c = (long)blockIdx.x * blockDim.x + threadIdx.x; c < wpr; c += cstride) d[c] = s[c];
  }
}

int halo_p2p_copy2d_kernel(long dst, long src, long dpitch, long spitch, long width, long height,
                           int bx, int by, int threads, long stream) {
  if ((dst | src | dpitch | spitch | width) & 15L) return -1;
  if (height <= 0 || width <= 0) return 0;
  if (threads <= 0 || threads > 256 || (threads & 31)) return -2;
  const long wpr = width / 16;
  if (bx <= 0) bx = (int)((wpr + threads - 1) / threads);
  if (bx > 64) bx = 64;
  if (by <= 0) by = (int)height;
  if (by > (int)height) by = (int)height;
  hipLaunchKernelGGL(halo_p2p_copy2d_k, dim3(bx, by), dim3(threads), 0, (hipStream_t)stream,
                     (const halo_u4v*)src, (halo_u4v*)dst, spitch / 16, dpitch / 16, wpr,
                     (int)height);
  hipError_t e = hipGetLastError();
  return e == hipSuccess ? 0 : -100 - (int)e;
}

// A system-scope release on `stream`, for when this device has written into peer memory that
// another agent will read. A peer write from a copy or kernel is posted and retires long before it
// lands; only a system-scope release orders it against a later handshake on a different path. One
// thread suffices, since the fence is a memory-system operation.
__global__ void halo_p2p_fence_kernel() { __threadfence_system(); }

int halo_p2p_fence(long stream) {
  hipLaunchKernelGGL(halo_p2p_fence_kernel, dim3(1), dim3(1), 0, (hipStream_t)stream);
  hipError_t e = hipGetLastError();
  return e == hipSuccess ? 0 : -100 - (int)e;
}

// A strided device-to-device copy: a two-rank all-gather once the peer's buffer is IPC-mapped.
// libr4d measured the runtime copy path fastest for peer traffic on the R9700; that comparison
// still has to be made on gfx1151. `dpitch`, `spitch` and `width` are bytes; `height` is rows.
int halo_p2p_copy2d(long dst, long src, long dpitch, long spitch, long width, long height,
                    long stream) {
  hipError_t e = hipMemcpy2DAsync((void*)dst, (size_t)dpitch, (const void*)src, (size_t)spitch,
                                  (size_t)width, (size_t)height, hipMemcpyDeviceToDevice,
                                  (hipStream_t)stream);
  return e == hipSuccess ? 0 : -100 - (int)e;
}
