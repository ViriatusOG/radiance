/* lq_ggml_planes.h -- llama.cpp's block formats as radiance's affine planes (rad_encoding.h): the
 * encoding each type stores, and the unpacking of one row's blocks into those planes.
 *
 * Shared by the `ggml` quantiser (lq_ggml.cpp: llama.cpp's own search writes the blocks, these
 * unpack them) and the core's GGUF import (core/format/checkpoint.cpp: the blocks a GGUF ships,
 * unpacked the same way), so a container quantised here and one imported from a llama.cpp file hold
 * the same planes for the same blocks. It needs only lq_ggml_port.h's block layouts and tables and
 * abi/rad_encoding.h, so the core compiles it too. lq_ggml_planes.cpp lists each type's planes.
 */
#pragma once
#include "lq_ggml_port.h"
#include "rad_encoding.h"

#include <cstddef>
#include <cstdint>

namespace lq::ggml {

/* The `nb` blocks of one row -> plane i's row at row[i]. A TABLE or per-tensor plane's slot is
 * null; write_shared writes those. Every byte of each row is written exactly once. */
typedef void (*UnpackFn)(const void* blocks, int64_t nb, uint8_t* const* row);

struct PlaneKind {
    const char* name;          /* as llama.cpp spells it */
    ggml_type   type;
    int64_t     block;         /* values one block covers */
    size_t      block_bytes;   /* bytes one block occupies */
    UnpackFn    unpack;
};

/* The type with GGUF tensor-type number `type`, or null when it has no planes here (F32, F16 and
 * BF16 are plain dtypes, read as they are). */
const PlaneKind* plane_kind(uint32_t type);

/* The inverse of a type's UnpackFn: plane i's row at row[i] -> the `nb` native blocks of one row,
 * every byte written, so unpack then pack returns any block's bytes unchanged. A kernel that reads
 * llama.cpp's block layout relayouts a container's planes with it at load. Null for a type with no
 * pack (the types with one: q8_0, q5_K, q6_K, iq4_nl, iq2_xxs, iq3_xxs, iq1_s). */
typedef void (*PackFn)(const uint8_t* const* row, int64_t nb, void* blocks);
PackFn pack_of(uint32_t type);

/* The encoding a type writes, plane for plane as lq_ggml_planes.cpp lists them. */
RadEncoding encoding_of(ggml_type t);

/* The planes every row shares (an IQ codebook, IQ4's levels, IQ2/IQ3's 1/8 or 1/4), into planes[i]
 * for each such plane i of `e`; the other slots are not touched and may be null. */
void write_shared(ggml_type t, const RadEncoding& e, void* const* planes);

}  /* namespace lq::ggml */
