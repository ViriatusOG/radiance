/* inkling2_proc_golden.h -- written by tests/mkinkling2procgolden.py; do not edit. */
#pragma once
#include <cstdint>

struct GoldInk2Image { int32_t f, h, w, rows, cols; uint16_t head[16]; uint64_t fnv; };
static const GoldInk2Image kGoldInk2Image[] = {
    { 0, 33, 61, 1, 2, { 48827, 16144, 48465, 48984, 16119, 16178, 15633, 48859, 48992, 16204, 48803, 16200, 48914, 15941, 48932, 15785 }, 9704603189521691260ull },
    { 0, 40, 40, 1, 1, { 48827, 16144, 48465, 48984, 16119, 16178, 15633, 48859, 48992, 16204, 48803, 16200, 48914, 15941, 48932, 15785 }, 15301893525997476837ull },
    { 2, 81, 120, 3, 3, { 16220, 16103, 16180, 16142, 48693, 48964, 16170, 16182, 48988, 15957, 16075, 16083, 16067, 16152, 48950, 48851 }, 16342963968973679142ull },
    { 3, 1, 1, 1, 1, { 48966, 48863, 48807, 49025, 49025, 49025, 49025, 49025, 49025, 49025, 49025, 49025, 49025, 49025, 49025, 49025 }, 16900118132120918694ull },
    { 4, 7, 95, 1, 3, { 16150, 15925, 48823, 48741, 48193, 15957, 48863, 48803, 16150, 15785, 48968, 48906, 48994, 48922, 48954, 48771 }, 9752231269522530570ull },
};
static const int32_t kGoldInk2AlphaH = 37, kGoldInk2AlphaW = 29;
static const uint64_t kGoldInk2AlphaFnv = 1067654293403151565ull;
static const uint16_t kGoldInk2FlatIn[3] = { 48950, 15233, 16256 };
static const uint16_t kGoldInk2FlatPad[3] = { 49025, 49025, 49025 };
static const uint64_t kGoldInk2FileAlphaFnv = 4598454019427415288ull;
static const uint64_t kGoldInk2FileOpaqueFnv = 6156688943104315212ull;
