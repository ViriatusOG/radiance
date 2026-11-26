// exl3i_twopass_frag.cpp -- radiance's own definition of the i8a8 fragment order (libr4d/r4d_fp8_frag.h, plain C++),
// exported to the C host test (tests/exl3i_twopass_host_test.c section B) so the int8 plane is checked against it
// rather than against a second transcription.
#include "../../../libr4d/r4d_fp8_frag.h"

extern "C" void exl3i_test_frag_src(int64_t rt, int64_t ks, int64_t lane, int64_t t, int64_t j, int64_t* row, int64_t* k) {
    r4d_fp8::frag_src(rt, ks, lane, t, j, row, k);
}

extern "C" int exl3i_test_frag_consts(void) {
    return r4d_fp8::kSuperK == 64 && r4d_fp8::kTileBytes == 1024 && r4d_fp8::kLaneBytes == 32 && r4d_fp8::kFragRows == 16 &&
           r4d_fp8::kFragK == 16;
}
