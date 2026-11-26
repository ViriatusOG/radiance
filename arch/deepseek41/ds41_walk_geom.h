/* ds41_walk_geom.h -- the planes and gather of deepseek41's full-vocabulary DSpark walk (dspark-walk;
 * dsp_declare / dsp_query), one definition the plugin and its test share (tests/r4d_emu/test_ds41walk.cpp
 * holds the issued gather to libr4d's all-gather admission, r4d_ar_gather_admit.h, at every sequence count
 * and step).
 *
 * A step's key is a u64 in a 16-byte slot (kSlot f32: the key, then 8 bytes nobody reads), because libr4d's
 * all-gather moves whole 16-byte words a row: ar_gather_4rank refuses an 8-byte row (the key alone, f32 row 2) at
 * the first full-vocabulary query pass. The planes, `wseqs` sequences a step, `block` steps:
 *   keys  [block * wseqs, kSlot]           f32, zeroed once a pass; step l's rows [l * wseqs, l * wseqs + S)
 *   gath  [block * wseqs, world * kSlot]   f32, all_gather(keys rows of step l, row kSlot) -> the same rows
 * The gather is declared with numel wseqs * kSlot (its bound) and issued over S rows. */
#ifndef DS41_WALK_GEOM_H
#define DS41_WALK_GEOM_H

#include <cstdint>

namespace ds41walk {

constexpr int64_t kSlot = 4;
inline int64_t keys_cols() { return kSlot; }
inline int64_t gath_cols(int world) { return (int64_t)world * kSlot; }
inline int64_t plane_rows(int64_t block, int64_t wseqs) { return block * wseqs; }
inline int64_t gather_numel(int64_t wseqs) { return wseqs * kSlot; }
inline int64_t gather_row() { return kSlot; }
/* step l's first row in keys and in gath */
inline int64_t step_row(int64_t l, int64_t wseqs) { return l * wseqs; }

}  // namespace ds41walk

#endif
