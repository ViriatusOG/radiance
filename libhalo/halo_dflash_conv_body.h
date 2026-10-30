/* Derived from libr4d/r4d_dflash_conv_body.h by the libhalo rename transform (2026-10-26);
 * see libhalo/README.md. Op schemas, row names and stored weight layouts are
 * shared with libr4d by design. Everything else in this file is libhalo's own
 * and diverges as it is measured on gfx1151. */
// DFlash2 grouped dynamic depthwise convolution -- device math only.
//
// Free of <hip/hip_runtime.h> so the same source compiles standalone to gfx1201 assembly with a
// stock clang (no ROCm) for ISA review, as well as into libhalo.so.
//
// The op, per DFlash2's _grouped_conv with taps=2, group_size=16:
//
//   out[t,h] = (base[0,h] + delta[t,0,g]) * x[t,h]
//            + (base[1,h] + delta[t,1,g]) * x[t-1,h] * ((t % block) >= 1)     g = h / group
//
// The reference uses ~6 [T,H] elementwise passes plus a materialised [T, taps, num_groups, group]
// coefficient tensor; the fused form reads x, delta and base once and writes out once.
#pragma once

typedef unsigned int   halo_u32;
typedef unsigned short halo_u16;
typedef halo_u32 halo_u32x4 __attribute__((ext_vector_type(4)));

// A 16-byte load addressed both ways. Subscript the union member, never a bit-cast vector:
// bit-casting a vector subscript silently yields element 0's bits.
union HaloV8 { halo_u32x4 v; halo_u16 h[8]; };

// Device functions under hipcc; plain always-inline functions under a standalone amdgcn clang.
#ifndef HALO_DEV
#ifdef __HIP__
#define HALO_DEV __device__ __forceinline__
#else
#define HALO_DEV static __attribute__((always_inline))
#endif
#endif

HALO_DEV float halo_bf2f(halo_u16 v) {
    return __builtin_bit_cast(float, (halo_u32)v << 16);
}

// f32 -> bf16 RTNE in software: gfx1201 has no bf16 convert instruction.
HALO_DEV halo_u16 halo_f2bf(float f) {
    halo_u32 u = __builtin_bit_cast(halo_u32, f);
    return (halo_u16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// One thread owns 8 contiguous channels (one dwordx4 in and out). The group size (16) is a multiple
// of 8, so the group index and both delta scalars are uniform per thread and loaded once.
template <int TAPS, int GROUP>
HALO_DEV void halo_dflash_conv_body(const halo_u16* __restrict__ x,
                                  const halo_u16* __restrict__ delta,
                                  const halo_u16* __restrict__ base,
                                  halo_u16* __restrict__ out,
                                  int H, int dpitch, int NG, int vpr,
                                  int blockmask, int t, int hv) {
    if (hv >= vpr) return;

    // Split each address into a wave-uniform pointer and a 32-bit lane offset (gfx12 saddr form).
    // Do not change either of these:
    //   * The mask: the backend folds a lane offset into saddr only if it can prove the byte offset
    //     fits 32 bits. Without it each load pays a 64-bit v_add_co_u32/v_add_co_ci_u32 pair;
    //     __builtin_assume does not propagate here, the AND does.
    //   * Pointer + offset, not a vector-typed array indexed by lane, which makes the backend
    //     materialise a 64-bit address per pointer.
    const unsigned off = ((unsigned)hv & 0xFFFFFu) * 8u;
    const unsigned g   = off / (unsigned)GROUP;

    const halo_u16* __restrict__ xrow = x + (long)t * H;
    halo_u16*       __restrict__ orow = out + (long)t * H;
    const halo_u16* __restrict__ drow = delta + (long)t * dpitch;

    HaloV8 x0; x0.v = *(const halo_u32x4*)(xrow + off);
    HaloV8 b0; b0.v = *(const halo_u32x4*)(base + off);
    const float d0 = halo_bf2f(drow[g]);

    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
        acc[j] = (halo_bf2f(b0.h[j]) + d0) * halo_bf2f(x0.h[j]);

    // Tap `tap` contributes only where the position within the block is >= tap. t is the
    // workgroup's y index, so this is a scalar branch and skipped rows issue no shifted load.
#pragma unroll
    for (int tap = 1; tap < TAPS; ++tap) {
        // Same saddr rule: uniform terms go into the pointer (adding tap*NG to the delta index
        // instead loses saddr). The pointers are formed outside the branch, because the backend
        // does not fold them into saddr from inside it; an unused address costs nothing.
        const halo_u16* __restrict__ xprev = xrow - (long)tap * H;
        const halo_u16* __restrict__ btap  = base + (long)tap * H;
        const halo_u16* __restrict__ dtap  = drow + (long)tap * NG;
        if ((t & blockmask) >= tap) {
            HaloV8 xs; xs.v = *(const halo_u32x4*)(xprev + off);
            HaloV8 bt; bt.v = *(const halo_u32x4*)(btap + off);
            const float dt = halo_bf2f(dtap[g]);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc[j] += (halo_bf2f(bt.h[j]) + dt) * halo_bf2f(xs.h[j]);
        }
    }

    HaloV8 o;
#pragma unroll
    for (int j = 0; j < 8; ++j) o.h[j] = halo_f2bf(acc[j]);
    *(halo_u32x4*)(orow + off) = o.v;
}
