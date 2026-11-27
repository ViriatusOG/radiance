// exl3i_a8q.hip.h -- the A8 activation quantiser's per-wave pieces (exl3i_a8.h's arithmetic on rotate128_math's values):
// what exl3i_a8_prep.hip's actq / AQ kernels and the mad GEMV (exl3i_mad.hip.h) share, so a row's int8 values are the
// same bytes on every path.
#pragma once
#include "exl3r4d_common.hip.h"
#include "exl3i_a8.h"

// The lane's four rotated halves as two v2h. NB: each dword is copied out of the vector before the bit cast:
// __builtin_bit_cast(v2h, v[1]) on an ext_vector element reads the vector's first element under the card's ROCm
// clang (core-10.1, 24.0.0git: the cast takes the base's address), so features 2, 3 came back as 0, 1.
static __device__ __forceinline__ void halves4(v2u v, v2h* a, v2h* b) {
  const unsigned w0 = v[0], w1 = v[1];
  *a = __builtin_bit_cast(v2h, w0);
  *b = __builtin_bit_cast(v2h, w1);
}

static __device__ __forceinline__ float amax4(v2u v) {
  v2h a, b;
  halves4(v, &a, &b);
  return fmaxf(fmaxf(fabsf((float)a[0]), fabsf((float)a[1])), fmaxf(fabsf((float)b[0]), fabsf((float)b[1])));
}

static __device__ __forceinline__ float wave_max(float m) {   // every lane gets the wave's max
  m = fmaxf(m, dpp_f<0x161>(m));
  m = fmaxf(m, dpp_f<0x162>(m));
  m = fmaxf(m, dpp_f<0x164>(m));
  m = fmaxf(m, dpp_f<0x168>(m));
  return fmaxf(m, x16_f(m));
}

static __device__ __forceinline__ unsigned quant4(v2u v, float inv) {
  v2h a, b;
  halves4(v, &a, &b);
  const int q0 = exl3i_a8_q((float)a[0], inv), q1 = exl3i_a8_q((float)a[1], inv);
  const int q2 = exl3i_a8_q((float)b[0], inv), q3 = exl3i_a8_q((float)b[1], inv);
  return (unsigned)(q0 & 0xFF) | (unsigned)(q1 & 0xFF) << 8 | (unsigned)(q2 & 0xFF) << 16 | (unsigned)(q3 & 0xFF) << 24;
}

