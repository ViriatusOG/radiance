// exl3i_a8_gemm.hip -- the A8 path's GEMM (exl3i_a8.h): radiance's i8a8 prefill tile with EXL3's store
// (exl3i_8a8.hip.h) over one column chunk's int8 plane. Its own translation unit: the tile's helpers must not meet the
// helpers of exl3r4d_common.hip.h of the same names.
#include <hip/hip_runtime.h>
#include <stdint.h>

#include "rad_types.h"
#include "exl3i_8a8.hip.h"
#include "exl3i_a8.h"
#include "exl3i_a8_dev.h"

// radiance's prefill kernel's register budget (r4d_gemm_i8a8.hip: R4D_WAVES(6) on the prefill tile)
__global__ __launch_bounds__(exl3i_8a8::kPWaves * exl3i_8a8::kWave) __attribute__((amdgpu_waves_per_eu(6, 6)))
void exl3i_a8_gemm_kernel(exl3i_8a8::Tile p) {
  exl3i_8a8::exl3_8a8_body(p);
}

extern "C" int exl3i_a8_gemm(const Exl3iA8Args* a, void* stream) {
  using namespace exl3i_8a8;
  if (!a || a->M < 1 || a->K % 128 || a->N % 128 || a->c0 % 128 || a->nc % 128 || a->nc < 128 ||
      a->c0 + a->nc > a->N || (a->mats == 2 && a->N % 256) || (a->y_pitch * (a->y_dt == 1 ? 4 : 2)) % 16)
    return RAD_E_SHAPE;
  if (a->y_dt < 0 || a->y_dt > 2) return RAD_E_DTYPE;
  Tile p;
  p.A = (const uint8_t*)a->aq;
  p.Sa = a->sa;
  p.W = (const uint8_t*)a->wq;
  p.svh = (const uint16_t*)a->svh;
  p.y = a->y;
  p.M = (uint32_t)a->M;
  p.K = (uint32_t)a->K;
  p.N = (uint32_t)a->N;
  p.c0 = (uint32_t)a->c0;
  p.mats = (uint32_t)a->mats;
  p.y_pitch = (uint32_t)a->y_pitch;
  p.y_dt = (uint32_t)a->y_dt;
  // the int8 grid's step / sqrt(128): WS (mad's int8 grid)
  p.C = EXL3I_A8_WS * 0.08838834764831845f;
  // the activation planes: the arena's [mats][M][K] / [mats][M][K / 128], or AQ's packed rows (a_ld set)
  p.a_ld = a->a_ld ? (uint64_t)a->a_ld : (uint64_t)a->K;
  p.a_ms = a->a_ld ? (uint64_t)a->a_ms : (uint64_t)a->M * a->K;
  p.sa_ld = a->a_ld ? (uint64_t)a->sa_ld : (uint64_t)(a->K / 128);
  p.sa_ms = a->a_ld ? (uint64_t)a->sa_ms : (uint64_t)a->M * (a->K / 128);
  const dim3 grid((unsigned)((a->M + kPBM - 1) / kPBM), (unsigned)(a->nc / kPBN));
  if (grid.y > 65535u) return RAD_E_SHAPE;
  void* args[] = {&p};
  return hipLaunchKernel((const void*)exl3i_a8_gemm_kernel, grid, dim3(kPWaves * kWave), args, 0, (hipStream_t)stream) ==
                 hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
