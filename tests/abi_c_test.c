/* abi_c_test.c -- abi/ as a C plugin sees it.
 *
 * Every header is included in one translation unit compiled as strict C11 (-pedantic-errors, see
 * tests/CMakeLists.txt), so a header that only C++ accepts fails the build. The struct layouts a
 * C compiler gives them, and rad_yarn.h's C definitions, are handed to abi_layout_test.cpp, which
 * compares them with its own. */
#include "rad_abi.h"
#include "rad_builder.h"
#include "rad_device.h"
#include "rad_encoding.h"
#include "rad_format.h"
#include "rad_philox.h"
#include "rad_planes.h"
#include "rad_plugin.h"
#include "rad_quant.h"
#include "rad_runtime.h"
#include "rad_sample.h"
#include "rad_spechub.h"
#include "rad_types.h"
#include "rad_yarn.h"

#include "abi_layout.h"

const AbiRow abi_c_rows[] = {
#define ABI_STRUCT(T)   { #T, "", 0, sizeof(T) },
#define ABI_FIELD(T, f) { #T, #f, offsetof(T, f), sizeof(((T*)0)->f) },
#include "rad_abi_layout_rows.h"
#undef ABI_STRUCT
#undef ABI_FIELD
};
const size_t abi_c_n_rows = sizeof abi_c_rows / sizeof abi_c_rows[0];

float abi_c_yarn_frequency(float factor, float low, float high, float magnitude, float inv, int i) {
    RadYarn y;
    y.factor = factor;
    y.low = low;
    y.high = high;
    y.magnitude = magnitude;
    return rad_yarn_frequency(&y, inv, i);
}

int abi_c_yarn_valid(float factor, float low, float high, float magnitude) {
    RadYarn y;
    y.factor = factor;
    y.low = low;
    y.high = high;
    y.magnitude = magnitude;
    return rad_yarn_valid(&y);
}
