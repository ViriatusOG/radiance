/* r3d_gemm_nt_q_w2a8.hip -- gemm_nt_q at dtype=w2a8: the vocabulary projection, decode-sized.
 *
 *   y[M,N] = a[M,K] @ b[N,K]^T over a 2-bit unsigned weight code with an f16 scale and an unsigned
 *   integer zero per (row, group of K), an int8 activation with one f32 scale per ROW, and the int8
 *   code sums `a_sum` the zero is removed with. f32 accumulate, bf16 store.
 *
 * This is the band that gates the server. MiniCPM5-2B's `output.weight` is N=130560, K=2048,
 * group=128, dtype=w2a8, and its stored form is the container's joined `rad.planes` operand: a
 * scale plane and a zero plane behind ONE operand (b_scale). libref declines w2a8 in
 * `gemm_q_shape`, so kbench has no case for the op the engine needs most -- the fixture cannot
 * draw the dtype, which is why this row's evidence is a driver that builds the joined operand with
 * `rad_planes_relayout` and compares every element against `ref_gemm_nt_q`'s own body. libref reads
 * the form in `gemm_zp`, so a host row serves it, at 135 GFLOP of CPU work per token; and a
 * placement without a reader is refused, because nothing in the loaded set can READ the joined form.
 * `r3d_layout_gemm_q_w2a8` below is that reader: it DESCRIBES the container's bytes (that is what
 * RadLayoutFn in rad_abi.h is), and the row carries no relayout and no repacking.
 *
 * ---- the law, and the order inside it ---------------------------------------------------------
 *
 * gemm_zp, the ZP_I8_ROW arm, is the spec, and this kernel runs its order:
 *
 *   q[k]   = rad_load_bits(row n of b, 2, k)                          bits [2k,2k+2) of byte k/4
 *   is(g)  = SUM_{k in group g} (int)a[m][i8_step_pos(k)] * (int)q[k]  integers: exact
 *   sub(g) = (float)is(g) - (float)z[n][g] * a_sum[g*mpad + m]         mpad = ceil(M/16)*16 = 16
 *   acc    = acc + (sub(g) * sa[m]) * sb[n][g]                         g ASCENDING
 *   y[m,n] = bf16(acc)
 *
 * `a_sum` is per row and per group and group-major, `sa` is one f32 per ROW (ZP_I8_ROW: quant_act_i8
 * scales a whole row, not a group), and `(sub * sa) * sb` is a parenthesised order, not a formula.
 *
 * The activation is in quant_act_i8's step order: column c sits at byte
 * i8_step_pos(c) = (c & ~12) | ((c & 4) << 1) | ((c & 8) >> 1), bits 2 and 3 of c swapped, the map
 * its own inverse. Swapping bits 2 and 3 of a column is swapping bits 0 and 1 of its DWORD index,
 * so a dword's four columns stay four contiguous bytes: staging reads a's dwords in a permuted
 * dword order, one 4-byte load per four codes, and nothing straddles a dword.
 *
 * `is` is a sum of integers, every partial sum under 2^24, so it is exact in f32 and its bracketing
 * is free. `sub`, the products and the g-ascending accumulation are float, and their order is the
 * reference's, so the store is bit-identical to the host row: bit-equality element by element is
 * the test, not a tolerance, because nothing else on this machine can draw the operands.
 *
 * ---- U2 to f16 through a bias, and the bias removed with a_sum ---------------------------------
 *
 * There is no int8 dot on this ISA. `__builtin_amdgcn_s32sd4`, `__builtin_amdgcn_s32ud4`, `__dp4a`
 * and the RDNA3 matrix ops `__builtin_amdgcn_madocx_i32_i8` / `_u32_u8` are all undeclared in AMD
 * clang 24.0.0git for gfx1102. So the codes widen to f16 and the dot runs on dot2_f16x2, whose
 * hardware form folds a pair into an f32 accumulator. Every product is
 * an integer and every partial sum is an integer, and "products and partials are integers below
 * 2^24" is a true statement about f32 and a false statement about this instruction: v_fdot2_f32_f16
 * carries a little under 24 bits of mantissa on integer operands -- about 22, from the ratio of the
 * worst error to its magnitude -- so any kernel that pushes integers through a pair-dot inherits a
 * tail once its partials reach 2^20. The instrument at the end of this file (one 64-code dot per
 * lane, the f32 it produces beside the exact integer it should have produced) shows: bias 1024,
 * 71027 of 262144 dots off, by up to 1/8. Bias 128, 70115 of 262144 off, by up to 1/64. The bias is 128 because the tail grows with the magnitude, and 128 is
 * the largest power of two that keeps a lane's partial sum under 2^21; shrinking it is a smaller
 * tail, not no tail. What the tail costs is one bit of the bf16 answer: 0.1-0.2% of outputs land one
 * bf16 ulp off and nothing else, rel_l2 5.0e-3 at the band, no non-finite anywhere. That is this
 * row's exactness, written out here because the law below is exact integers and the row reads like a
 * bit-exact one. It is not, and the row's prose says so.
 * The row without the dot unit -- four f32 words a quad in the tile, four __fmaf_rn per quad -- is
 * not in this file. Its arithmetic is exact: a probe of 262144 samples of the four fmas with no
 * tile, zero inexact dots, worst error exactly zero. Its staging is the defect: the f32 tile,
 * written through a vector pointer at -O2, comes up with content that was never written, and
 * typing the array `uint4` rules aliasing out, so the defect is in the tile staging and not in the
 * arithmetic.
 * At the band this row is not memory-bound. r3d_vocab_bf16.hip streams the same n_vocab x n_embd
 * head in 535 MB of bf16 in 3.29 ms at M=1, 163 GB/s, two thirds of the card's DRAM. The 66.8 MB
 * of codes this row reads has a streaming floor near 0.41 ms, not the 3.0 ms of one 22 GB/s pass
 * this row measures, so this row runs 6-10x above its own floor and the headroom is in its loop --
 * the group-term plane and the two barriers a chunk -- not in the memory system. An f32 row costs
 * nothing to run, and the tail goes with it.
 * The trick that pays for the widening is that f16 bits of (128 + x) are 0x5800 + 8x for 0 <= x < 8,
 * so a 2-bit code becomes an f16 with a mask, two shifts and an add -- no table, no compare per
 * code -- and the dot returns is' = SUM a[k]*(128 + q[k]) = is + 128*SUM a. SUM a over the group is
 * the plane the op already carries: a_sum[g*mpad + m], the very number the zero is removed with.
 * 128 * a_sum scales an integer-valued f32 by a power of two, so it is exact, is = is' - 128*a_sum
 * is exact, and the reference's own sub = (float)is - (float)z*a_sum follows unchanged. The de-bias
 * costs one exact f32 subtraction per (m, group) and is why a row with no int8 instruction on it is
 * memory-bound at the band instead of unpack-bound.
 *
 * ---- the lane map ------------------------------------------------------------------------------
 *
 * One warp per weight row n; a block of 8 waves covers 8 rows. MT, the template's row count, is the
 * power of two that covers M, and KL = 32/MT is the number of k-lanes per activation row. A chunk
 * is CHUNK = 2048/MT codes, so the staged tile is always MT*CHUNK/2 = 1024 dwords = 4 KiB: the
 * tile shrinks with the row count exactly as the row count grows, which is what keeps the block
 * resident at M=16. A chunk holds CHUNK/128 = 16/MT groups and every lane takes 64 codes -- one
 * 16-byte load of codes (four bytes, sixteen codes each, U2 being four codes a byte), thirty-two
 * dwords of activation from LDS and thirty-two dot2 in sixteen dot4_f16x4 -- so lane l owns row
 * m = l % MT, k-lane kl = l / MT, and group ch*(16/MT) + kl/2, whose two halves are the pair
 * (l, l^MT): the group folds with ONE shfl_xor, in integers.
 *
 * Each lane's group index ascends with the chunk index, so `acc += term` in the lane IS the
 * reference's ascending g loop over the groups that lane owns; the groups the other k-lanes own are
 * added by kl-1 ordered broadcasts per chunk, in ascending kl, which is ascending g. At M=16, where
 * a chunk is one group, there is no cross-lane float work at all. Reassociating the g sum is what
 * would break bit-equality, and the map is built so that no lane ever has to.
 *
 * LDS is touched a quad at a time (two dwords, 8 bytes), swizzled as `p_ix` below; unswizzled the
 * wavefront's 32 quads for one u all sit in the same bank pair.
 *
 * ---- refusals (codes; RAD_R3D_TRACE=1 prints the line) -----------------------------------------
 *
 * The shim mirrors gemm_ntq_impl's and gemm_zp's checks and declines two things the reference
 * accepts, each named at its site: M > 16 (the band is 16, the tile is 16 rows, mpad is the
 * reference's 16-row pad; the prefill vocab projection is `logits_gemm` and is a tiled bf16 row,
 * not a wider loop bound here), and a zero plane stored sub-byte (U2/U4/U5), where the kernel reads
 * a zero a byte at a time -- a sub-byte zero plane needs the bit-extract path,
 * `rad_load_bits(zb, bits, g)` is ten lines. w2a8's practical form, and this container's, is U8.
 */

#include <hip/hip_runtime.h>

#include "r3d_common.h"
#include "r3d_codecs.h"
#include "r3d_launch.h"
#include "r3d_args.h"

#include "rad_planes.h"

namespace {

constexpr int kBlockThreads = 256;
constexpr int kWaves        = kBlockThreads / kWave;   /* 8 weight rows per block */
constexpr int kMTMax        = 16;                      /* the band's M */
constexpr int kTileQuads    = 512;        /* MT * CHUNK/4, constant over MT: 512 quads, 4 KiB */
constexpr int kGroup        = 128;                     /* the block size w2a8 is stored at */
constexpr int kMpad         = 16;                      /* the reference's (M+15)/16*16, M <= 16 */

/* Column c sits at byte step_pos(c), bits 2 and 3 of c swapped, the map its own inverse. Applying
 * it to the four columns of one dword leaves them four contiguous bytes and turns the map on the
 * DWORD index into a swap of bits 0 and 1 -- the middle two dwords of a 16-byte step exchanged.
 * The byte a dword starts at is therefore 4 * dword_pos(d): NOT 4 * step_pos(d), which swaps the
 * wrong pair of bits and reads the step's dwords unexchanged. */
[[maybe_unused]] [[maybe_unused]] __device__ __forceinline__ int64_t step_pos(int64_t c)
{
    return (c & ~(int64_t)12) | ((c & 4) << 1) | ((c & 8) >> 1);
}
__device__ __forceinline__ int64_t dword_pos(int64_t d)
{
    return 4 * ((d & ~(int64_t)3) | ((d & 1) << 1) | ((d & 2) >> 1));
}

/* The LDS swizzle, at the granularity the tile is touched: a quad of four codes, two dwords, moved
 * as one 8-byte word. Quads are numbered q = r*(CHUNK/4) + c, and p_ix XORs q's low four bits with
 * bits 4..7 -- a bijection on 16-quad groups, so on any tile whose quad count is a multiple of 16.
 * A wavefront reads q = m*(CHUNK/4) + kl*16 + u for fixed u; CHUNK/4 is a multiple of 16 at every
 * MT, so q ^ (q >> 4) mod 16 is u ^ (4m + kl) mod 16 over the 32 lanes -- 16 distinct values, each
 * one 8-byte access spanning two banks: 32 banks, no conflict. The swizzle must leave bit 0 of the
 * quad index alone: the two dwords of a quad are adjacent only then, and the scalar form that
 * XORs bits 5..9 of the dword index does not. */
__device__ __forceinline__ int p_ix(int q) { return q ^ ((q >> 4) & 15); }

/* Four int8 codes -> two f16x2 dwords. Exact: an integer of |a| <= 127 needs no rounding. */
__device__ __forceinline__ void i8x4_to_f16x2(uint32_t w, uint32_t& lo, uint32_t& hi)
{
    lo = f16x2_pack((float)(int32_t)(int8_t)(w & 0xffu), (float)(int32_t)(int8_t)((w >> 8) & 0xffu));
    hi = f16x2_pack((float)(int32_t)(int8_t)((w >> 16) & 0xffu),
                    (float)(int32_t)(int8_t)((w >> 24) & 0xffu));
}

/* Four U2 codes (one byte) -> two f16x2 dwords, BIASED by 128: each half is the f16 bits of
 * (128 + code). An f16 at exponent 2^7 steps its mantissa by 1/8, so 128 + code is 0x5800 + 8*code
 * and stays an integer. The dot therefore returns is + 128 * the group's code sum, which the
 * epilogue takes off with the group's a_sum.
 *
 * The bias is 128 and not 1024 because of where the dot unit's precision runs out. a * (B + q) is an
 * integer the size of 128 * B, and a lane's 64 of them sum to 64 * 128 * (B + 3): at B = 1024 that is
 * 2^23, at B = 128 it is 2^20. The unit keeps a little under 24 bits of mantissa on integers, so the
 * tail scales with the magnitude -- 71027 of 262144 dots off by up to 1/8 at B = 1024, 70115 off by
 * up to 1/64 at B = 128, in the probe below. 128 is the largest power of two that keeps a lane's
 * partial sum under 2^21, which is as close as a pair-dot gets; the header carries the rest. */
__device__ __forceinline__ void u2x4_biased(uint32_t w, uint32_t& lo, uint32_t& hi)
{
    lo = 0x58005800u + ((w & 3u) << 3) + (((w >> 2) & 3u) << 19);
    hi = 0x58005800u + (((w >> 4) & 3u) << 3) + (((w >> 6) & 3u) << 19);
}

template <int MT>
__global__ void __launch_bounds__(kBlockThreads) R3D_WAVES(2)
gemm_w2a8_kernel(const int8_t* __restrict__ a, const float* __restrict__ sa, int sa_stride,
                 const uint8_t* __restrict__ b, const uint16_t* __restrict__ sb, int sb_rb,
                 const uint8_t* __restrict__ zq, int z_rb, const float* __restrict__ asum,
                 int asum_stride, uint16_t* __restrict__ y, int M, int N, int K, int chunk,
                 int a_pitch, int y_pitch)
{
    constexpr int KL = kWave / MT;                    /* k-lanes per activation row */
    constexpr int CG = (2048 / MT) / kGroup;          /* groups per chunk */
    constexpr int CL = (2048 / MT) / KL;              /* codes per lane per chunk: 64 */

    /* The tile: one quad of four codes per slot, two f16x2 dwords, typed as the quad the staging
     * writes and the dot reads. The tile is a `uint2[]` and not a `uint32_t[]` reached through a
     * vector pointer: the f32 variant of this row, written through `*(uint4*)&sA[]` at -O2, comes
     * up with content that was never written. */
    __shared__ __align__(16) uint2 sA[kTileQuads];       /* MT rows x CHUNK/2 f16, swizzled */
    __shared__ float sT[kWaves * 16];        /* each warp's chunk terms: warp*16 + m*CG + j */
    __shared__ float sSa[MT];

    const int lane = threadIdx.x & (kWave - 1);
    const int warp = (int)(threadIdx.x / kWave);
    const int n = blockIdx.x * kWaves + warp;
    /* A wavefront past the last row sits the chunk out, but it stages: the tile is block-wide, so
     * the warps that have no row to serve still write the rows their lanes own. Only a block whose
     * EIGHT rows are all past the last row leaves, which is a block-uniform test and cannot strand
     * a barrier; a guard on the row would, because the tail block's dead warps own tile rows and
     * staging them from never-written memory corrupts the tile. The store is guarded by `live`
     * below, so a dead wavefront writes nothing. */
    const bool live = n < N;
    if ((int)blockIdx.x * kWaves >= N) return;

    const int m = lane & (MT - 1);
    const int kl = lane / MT;
    const int row_bytes = K >> 2;                     /* rad_enc_row_bytes(RAD_U2, K), K % 16 == 0 */
    const int cq = chunk >> 2;                        /* the chunk's code bytes */
    const int nchunk = K / chunk;

    if (lane < MT) sSa[lane] = lane < M ? sa[(long long)lane * sa_stride] : 0.0f;
    __syncthreads();

    float acc = 0.0f;
    for (int ch = 0; ch < nchunk; ++ch) {
        /* Stage the chunk's activation rows, widened and step-unpermuted on the way in. Cells are
         * four codes: MT * CHUNK/4 = 512 of them at every MT, two per thread. */
        for (int i = threadIdx.x; i < MT * (chunk >> 2); i += kBlockThreads) {
            const int r = i / (chunk >> 2), c = i % (chunk >> 2);
            uint32_t w = 0;
            if (r < M)
                w = *(const uint32_t*)(a + (long long)r * a_pitch +
                                       dword_pos((int64_t)ch * (chunk >> 2) + c));
            const int q = r * (chunk >> 2) + c;
            uint32_t lo, hi;
            i8x4_to_f16x2(w, lo, hi);
            sA[p_ix(q)] = make_uint2(lo, hi);
        }
        __syncthreads();

        uint4 qw = make_uint4(0, 0, 0, 0);
        if (live)
            qw = *(const uint4*)(b + (long long)n * row_bytes + (long long)ch * cq + kl * (CL >> 2));
        const uint32_t w4[4] = { qw.x, qw.y, qw.z, qw.w };

        if (live) {
            const int qb = m * (chunk >> 2) + kl * (CL >> 2);
            float p = 0.0f;                            /* the lane's half-group, an exact integer */
#pragma unroll
            for (int v = 0; v < 4; ++v) {              /* four bytes, four quads, sixteen codes */
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    uint32_t ql, qh;
                    u2x4_biased((w4[v] >> (8 * b)) & 0xffu, ql, qh);
                    const uint2 av = sA[p_ix(qb + 4 * v + b)];
                    p = dot4_f16x4(av.x, av.y, ql, qh, p);
                }
            }
            /* Fold the pair: lanes l and l^MT are the same row, the two halves of one group. */
            const float isg = p + __shfl_xor(p, MT);

            const int g = ch * CG + (kl >> 1);
            const float as = asum[(long long)g * asum_stride + m];
            const float is = isg - 128.0f * as;                    /* exact: the de-bias */
            const float iz = (float)zq[(long long)n * z_rb + g];
            const float sc = f16_to_f32(sb[(long long)n * (sb_rb >> 1) + g]);
            const float term = ((is - iz * as) * sSa[m]) * sc;     /* gemm_zp's parentheses */
            /* One slot per (row, group-in-chunk): MT*CG is 16 at every MT, so this is 64 bytes.
             * The group's two half-lanes write the same value, which is why the pair fold above is
             * a fold and not a broadcast. */
            sT[warp * 16 + m * CG + (kl >> 1)] = term;
        }
        __syncthreads();
        /* Group ch*CG + j, for j ascending, into the lane that owns the row's answer. The reference
         * walks g in ascending order and f32 adds are not associative, so the order is part of the
         * answer, not a detail. The group gather is the one place in the kernel that crosses
         * k-lanes; it is a barrier and four LDS reads at 64 bytes, which is cheaper than a shuffle
         * round-trip. */
        if (live && kl == 0) {

            /* A chunk carries CG groups, and group ch*CG + j lives in the two k-lanes 2j and 2j+1:
             * they hold the same group's two halves, which is what the fold above combined. Every
             * lane adds all CG of them in ascending j, so each lane of a row ends on the same
             * partial sum, built in the same order the reference walks its groups in -- f32 adds are
             * not associative, and a lane that finished its own group last rounds the tail
             * differently. Reading a lane's `acc` would not do: by j = 2 it has already added a
             * group of its own. */
#pragma unroll
            for (int j = 0; j < CG; ++j)
                if (m < M) acc += sT[warp * 16 + m * CG + j];
        }
        __syncthreads();
    }

    if (live && kl == 0 && m < M) y[(long long)m * y_pitch + n] = f32_to_bf16(acc);
}

int mt_for(int M) { return M <= 1 ? 1 : M <= 2 ? 2 : M <= 4 ? 4 : M <= 8 ? 8 : kMTMax; }

}   // namespace

/* ================================================================== the launcher
 *
 * Codes: 0, -1 a required pointer is null, -2 a shape it does not read, -3 a pitch it cannot read,
 * -4 HIP. `nt` is the threads argument every libr3d launcher takes and this kernel ignores: the
 * work is one warp per weight row, and the wavefront count is the grid.
 *
 * `b_scale` is the joined operand's base. The launcher reads the two planes out of its own header
 * (rad_planes.h's rad_planes_find) and passes each plane's base, row pitch in bytes and the row
 * count down, so the kernel reads the container's bytes where they lie. */

/* The instrument behind the measurement in the header: the dot on its own, one 64-code sample per
 * lane, the f32 it produces beside the exact integer it should have produced. The driver runs it as
 * `dot [n]`; the bias is sized from it, and it shows the f32 path to be exact. Two extern "C"
 * symbols, no row, no cost at serve time. */
extern "C" __global__ void __launch_bounds__(256) probe_dot_kernel(const int8_t* a,
                                                                   const uint8_t* codes,
                                                                   float* out, long long* exact)
{
    const int g = blockIdx.x * 32 + (int)(threadIdx.x / kWave) * 4 + (int)(threadIdx.x & 3);
    const int lane4 = (int)(threadIdx.x & 31) & 3;
    (void)lane4;
    const uint32_t* w4 = (const uint32_t*)(codes + (long long)g * 16);
    float p = 0.0f;
    long long ex = 0;
#pragma unroll
    for (int v = 0; v < 4; ++v) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t aw = *(const uint32_t*)(a + (long long)g * 64 + 4 * (4 * v + b));
            uint32_t ql, qh;
            u2x4_biased((w4[v] >> (8 * b)) & 0xffu, ql, qh);
            uint32_t lo, hi;
            i8x4_to_f16x2(aw, lo, hi);
            p = dot4_f16x4(lo, hi, ql, qh, p);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                ex += (long long)(int)(int8_t)((aw >> (8 * j)) & 0xffu) *
                      (long long)(128 + ((w4[v] >> (8 * b + 2 * j)) & 3u));
        }
    }
    out[g] = p;
    exact[g] = ex;
}

extern "C" int r3d_w2a8_probe_launch(const int8_t* a, const uint8_t* codes, float* out,
                                     long long* exact, int n)
{
    probe_dot_kernel<<<(n + 31) / 32, 256, 0, 0>>>(a, codes, out, exact);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

extern "C" int r3d_gemm_nt_q_w2a8(long a, long a_scale, long b, long b_scale, long a_sum, long y,
                                  int M, int N, int K, int group, int a_ld, int y_ld, int nt,
                                  long stream)
{
    (void)nt;
    if (!a || !a_scale || !b || !b_scale || !a_sum || !y) return -1;
    if (M < 1 || N < 1 || K < 1) return -2;
    if (M > kMTMax) return -2;                        /* the row says the same: C_LE("M", 16) */
    if (group != kGroup) return -2;                   /* the tile is 128-wide: chunk = 128*16/MT */
    if (K % kGroup) return -2;
    if (K % 16) return -2;                            /* the step order is a 16-byte step */
    const int mt = mt_for(M);
    const int chunk = 2048 / mt;
    if (K % chunk) return -2;                         /* ragged chunk: the tile would read past K */
    const int G = K / group;
    if (G < 1) return -2;
    if (!a_ld) a_ld = K;
    if (!y_ld) y_ld = N;
    if (a_ld % 4) return -3;                          /* four codes a dword, staged whole */
    if (((uintptr_t)b & 15u) || ((uintptr_t)a & 3u)) return -3;

    const RadPlanesHeader* h = rad_planes_header((const void*)(uintptr_t)b_scale);
    if (!h) return -2;                                /* not the joined form: rad.planes is expected */
    const RadPlanesEntry* es = nullptr;
    const RadPlanesEntry* ez = nullptr;
    const void* sp = rad_planes_find((const void*)(uintptr_t)b_scale, "scale", &es);
    const void* zp = rad_planes_find((const void*)(uintptr_t)b_scale, "zero", &ez);
    if (!sp || !zp || !es || !ez) return -2;
    if (es->dtype != RAD_F16) return -2;              /* this kernel reads an f16 scale */
    if (ez->dtype != RAD_U8) return -2;               /* ...and a byte-wide zero; see the header */
    if (es->rows != N || es->cols != G || ez->rows != N || ez->cols != G) return -2;
    if ((es->row_bytes & 1) || es->row_bytes < 2 * G || ez->row_bytes < G) return -3;

#define R3D_W2A8_LAUNCH(MT)                                                                          \
    do {                                                                                             \
        const dim3 grid((N + kWaves - 1) / kWaves), block(kBlockThreads);                             \
        R3D_KLAUNCH(R3D_KNAME((gemm_w2a8_kernel<MT>)), grid, block, (hipStream_t)stream,             \
                    (const int8_t*)a, (const float*)a_scale, 1, (const uint8_t*)b,                   \
                    (const uint16_t*)sp, (int)es->row_bytes, (const uint8_t*)zp, (int)ez->row_bytes, \
                    (const float*)a_sum, kMpad, (uint16_t*)y, M, N, K, chunk, a_ld, y_ld);           \
    } while (0)

    switch (mt) {
        case 1:  R3D_W2A8_LAUNCH(1);  break;
        case 2:  R3D_W2A8_LAUNCH(2);  break;
        case 4:  R3D_W2A8_LAUNCH(4);  break;
        case 8:  R3D_W2A8_LAUNCH(8);  break;
        default: R3D_W2A8_LAUNCH(16); break;
    }
#undef R3D_W2A8_LAUNCH
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* ================================================================== the shim
 *
 * gemm_nt_q's slots, as gemm_ntq_impl takes them: 0 a (i8, step order), 1 a_scale (f32, one per
 * row), 2 b (the u2 codes), 3 b_scale (the joined scale+zero), 4 y (bf16), 5 a_sum (f32, group
 * major, mpad = 16 columns). */
extern "C" int r3d_rad_gemm_nt_q_w2a8(const RadArgs* a, RadStream s)
{
    if (!a) return R3D_NO(RAD_E_INVAL);
    const char* dt = r3d_gets(a, "dtype");
    if (dt && std::strcmp(dt, "w2a8") != 0) return R3D_NO(RAD_E_DTYPE);   /* the row is w2a8 only */

    const RadTensor* A = r3d_opt(a, 0);
    const RadTensor* AS = r3d_opt(a, 1);
    const RadTensor* B = r3d_opt(a, 2);
    const RadTensor* BS = r3d_opt(a, 3);
    const RadTensor* Y = r3d_opt(a, 4);
    const RadTensor* ASUM = r3d_opt(a, 5);
    if (!r3d_p(A) || !r3d_p(AS) || !r3d_p(B) || !r3d_p(BS) || !r3d_p(Y) || !r3d_p(ASUM))
        return R3D_NO(RAD_E_INVAL);

    if (A->dtype != RAD_I8) return R3D_NO(RAD_E_DTYPE);       /* quant_act_i8's codes */
    if (B->dtype != RAD_U2) return R3D_NO(RAD_E_DTYPE);       /* the grid's codes */
    if (Y->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f32(AS) || !r3d_is_f32(ASUM)) return R3D_NO(RAD_E_DTYPE);

    const long long N = r3d_geti_or(a, "N", 0), K = r3d_geti_or(a, "K", 0);
    const long long group = r3d_geti_or(a, "group", 0);
    if (N < 1 || K < 1) return R3D_NO(RAD_E_SHAPE);
    if (group != kGroup) return R3D_NO(RAD_E_SHAPE);          /* no defaulting: gemm_zp refuses 0 */
    if (K % group || K % 16) return R3D_NO(RAD_E_SHAPE);
    const long long G = K / group;

    const long long ynel = r3d_numel(Y);
    if (N < 1 || ynel % N) return R3D_NO(RAD_E_SHAPE);
    const long long M = ynel / N;
    if (M < 1 || M > kMTMax) return R3D_NO(RAD_E_SHAPE);      /* DECLINE, in the header: the band is
                                                               * 16 and the tile is 16 rows; the
                                                               * prefill vocab projection is
                                                               * logits_gemm, a tiled bf16 row. */
    if (r3d_numel(A) < M * K || r3d_numel(B) < N * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(AS) < M) return R3D_NO(RAD_E_INVAL);        /* gemm_zp's code for this one */
    if (r3d_numel(ASUM) < G * kMpad) return R3D_NO(RAD_E_INVAL);

    const long long ap = r3d_row_pitch(A);
    if (ap < K || ap % 4) return R3D_NO(RAD_E_STRIDE);        /* step order read four codes a dword */
    if (!r3d_contig(B)) return R3D_NO(RAD_E_STRIDE);          /* gemm_zp: contiguous codes only */
    long long yrows = 0;
    const long long yp = r3d_rows_pitch(Y, N, &yrows);
    if (yp < N) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_contig(ASUM)) return R3D_NO(RAD_E_STRIDE);

    const RadPlanesHeader* h = rad_planes_header(r3d_p(BS));
    if (!h) return R3D_NO(RAD_E_DTYPE);                       /* the joined form is the operand */
    const RadPlanesEntry* es = nullptr;
    const RadPlanesEntry* ez = nullptr;
    const void* sp = rad_planes_find(r3d_p(BS), "scale", &es);
    const void* zp = rad_planes_find(r3d_p(BS), "zero", &ez);
    if (!sp || !zp || !es || !ez) return R3D_NO(RAD_E_DTYPE);
    if (es->dtype != RAD_F16) return R3D_NO(RAD_E_DTYPE);
    if (ez->dtype != RAD_U8)
        return R3D_NO(RAD_E_DTYPE);   /* DECLINE: a sub-byte zero plane needs the bit-extract path;
                                       * rad_load_bits(zb, bits, g) is 10 lines. U8 is w2a8's form. */
    if (es->rows != N || es->cols != G || ez->rows != N || ez->cols != G) return R3D_NO(RAD_E_SHAPE);
    if ((es->row_bytes & 1) || es->row_bytes < 2 * G || ez->row_bytes < G) return R3D_NO(RAD_E_STRIDE);

    if (!r3d_aligned(r3d_p(A), 4) || !r3d_aligned(r3d_p(B), 16)) return R3D_NO(RAD_E_ALIGN);

    const int rc = r3d_gemm_nt_q_w2a8(
        r3d_addr(A), r3d_addr(AS), r3d_addr(B), r3d_addr(BS), r3d_addr(ASUM), r3d_addr(Y), (int)M,
        (int)N, (int)K, (int)group, (int)ap, (int)yp, 0, (long)s);
    /* The launcher's codes are its own (r3d_launch.h's set: null, shape, pitch, HIP); the row speaks
     * RAD codes, and the shim is where the two are named. Same mapping as the gated row's. */
    if (rc == 0) return RAD_OK;
    return R3D_NO(rc == -1 ? RAD_E_INVAL : rc == -2 ? RAD_E_SHAPE
                            : rc == -3 ? RAD_E_STRIDE : RAD_E_DEVICE);
}

/* ================================================================== the layout hook
 *
 * The scoped exception to libr3d's "no layout hooks, no repacking" rule, transcribed from libref's
 * `ref_layout_gemm_q` with the checks and their order unchanged:
 *
 *     if (n < 2 || operand != 3) return RAD_E_UNSUPPORTED;
 *     const ZpGrid* zg = zp_grid(dtype);  group = param "group";
 *     if (!zg || group <= 0 || !rad_enc_is(enc, "affine")) return RAD_E_DTYPE;
 *     transform empty, or beginning "givens";
 *     sel_roles(enc, sel, n, "scale", "zero") && enc->n_planes >= 3;
 *     plane[0]: role "codes", dtype == zg->codes, block {1,1};
 *     scale:    RAD_PLANE_TILED, RAD_F16, block {1, group};
 *     zero:     RAD_PLANE_TILED, zero_dt(...), block {1, group};
 *     return rad_planes_layout(enc, sel, planes, n, out);
 *
 * zp_grid, zero_dt and sel_roles are libref-internal statics, inlined below for w2a8: the grid is
 * { "w2a8", RAD_U2, ZP_I8_ROW }, zero_dt is U8|U2|U4|U5, and the practical form is U8 (the kernel
 * declines a sub-byte zero, at its site, with the fix named). The answer is rad_planes.h's own:
 * the tag is "rad.planes", the dtype the first plane's, the shape the byte count. NOTHING HERE
 * MOVES A BYTE: this hook is the reader the mapped tier needs and the row's relayout is null.
 */
/* The writer for the joined operand this hook names: codes, scale and zero as the plan's one
 * `rad.planes` weight, assembled by rad_planes_relayout -- header, entries, each plane's rows,
 * copied. The ABI's contract is that a query answering RAD_OK is a promise the loader keeps on the
 * row's word: "out describes the stored form, and relayout will produce it" (rad_abi.h), and a
 * weight with no hook is the only kind that can be read where it lies, which is why a joined weight
 * is placed in VRAM and a plain one can stay on the mapped tier. Nothing here transforms a code:
 * the bytes written are the bytes stored. */
extern "C" int r3d_relayout_gemm_nt_q_w2a8(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const RadTensor* planes, int n_planes, void* dst,
                                           int64_t dst_bytes)
{
    (void)p; (void)n_p;
    if (operand != 3) return RAD_E_UNSUPPORTED;
    return rad_planes_relayout(enc, sel, planes, n_planes, dst, dst_bytes);
}

/* A layout query answers in the engine's own vocabulary and leaves no trace, and a model that will
 * not load says only that two rows want one weight in two forms. This prints the question and the
 * answer -- operand, band dtype, planes offered, code returned -- under RAD_R3D_TRACE=1. */
static int r3d_layout_trace(const char* who, int operand, const char* dt, int n_planes, int rc) {
    if (r3d_trace_on())
        std::fprintf(stderr, "[r3d] %s: operand %d dtype %s planes %d -> %d\n", who, operand,
                     dt ? dt : "(none)", n_planes, rc);
    return rc;
}

extern "C" int r3d_layout_gemm_nt_q_w2a8(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel,
                                         const RadTensor* planes, int n, RadLayout* out)
{
    if (n < 2 || operand != 3) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_UNSUPPORTED);
    const char* dt = r3d_pgets(p, n_p, "dtype");
    /* The plan this hook describes -- the (scale, zero) pair joined on operand 3 -- is a statement
     * about the CONTAINER, not about the code width, and the engine forces every row that reads one
     * weight to answer with one form. So both packed grids come through, and the width a row can
     * unpack is the SHIM's business: gemm_nt_q_w2a8_m16 reads U2 and nothing else,
     * gemm_nt_q_w2a8_tiled reads U2 and U4, and each says so at its own line. */
    if (!dt || (std::strcmp(dt, "w2a8") != 0 && std::strcmp(dt, "w4a8_asym") != 0))
        return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);                                            /* the packed grids */
    const long long group = r3d_pgeti_or(p, n_p, "group", 0);
    if (group <= 0) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
    if (!enc || !rad_enc_is(enc, "affine")) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
    if (enc->transform[0] && std::strncmp(enc->transform, "givens", RAD_ENC_STR) != 0)
        return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
    if (!enc || enc->n_planes < 3 || !sel) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);

    /* sel_roles(enc, sel, n, "scale", "zero"): one role per selected plane, in order. */
    if (n != 2) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
    for (int i = 0; i < 2; ++i) {
        const char* want = i == 0 ? "scale" : "zero";
        if (sel[i] < 0 || sel[i] >= enc->n_planes) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
        if (std::strncmp(enc->plane[sel[i]].role, want, RAD_ENC_STR) != 0) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
    }
    /* The codes plane is the one this operand does NOT take: the weight's role "codes", the grid's
     * dtype, block {1,1}. libref finds it by role over enc->plane, not by sel. */
    int codes = -1;
    for (int i = 0; i < enc->n_planes; ++i)
        if (std::strncmp(enc->plane[i].role, "codes", RAD_ENC_STR) == 0) { codes = i; break; }
    if (codes < 0) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
    if (enc->plane[codes].block[0] != 1 || enc->plane[codes].block[1] != 1) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);

    for (int i = 0; i < 2; ++i) {
        const RadEncPlane& q = enc->plane[sel[i]];
        const RadTensor& t = planes[i];
        if (q.kind != RAD_PLANE_TILED) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
        if (q.block[0] != 1 || q.block[1] != group) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
        if (i == 0) {
            if (t.dtype != RAD_F16) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);               /* the scale is f16 */
        } else {
            const uint32_t zd = t.dtype;
            const bool ok = zd == RAD_U8 || zd == RAD_U2 || zd == RAD_U4 || zd == RAD_U5;
            if (!ok) return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);                              /* zero_dt: never signed */
        }
    }
    return r3d_layout_trace("r3d_layout_gemm_nt_q_w2a8", operand, r3d_pgets(p, n_p, "dtype"), n, rad_planes_layout(enc, sel, planes, n, out));
}
