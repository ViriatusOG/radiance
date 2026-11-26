// exl3i_nt_mad.hip -- exl3i's dense linear, codebook mad (every rate): exl3i_nt.hip.h instantiated once, and the
// byte-sum GEMV (exl3i_mad.hip.h, act a8g128 at 1 .. 8 rows).
#include "exl3i_mad.hip.h"

extern "C" int exl3i_launch_nt_mad(const Exl3iGemm* a, int npw, void* stream) {
  return exl3i_launch_cb<0>(a, npw, stream);
}

extern "C" int exl3i_launch_mad_gemv(const Exl3iGemm* a, int npw, void* stream) {
  (void)npw;   // the device side's EXL3I_MAD_NPW (a build constant; nt.c passes its own default)
  return exl3i_mad_launch(a, EXL3I_MAD_NPW, stream);
}
