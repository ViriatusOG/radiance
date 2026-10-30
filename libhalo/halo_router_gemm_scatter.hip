/* Derived from libr4d/r4d_moe.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_router_gemm_scatter.hip -- the router's GEMM, its exact top_k and the slot sort in ONE
// launch, plus, where the layer has a shared expert, that expert's one-column gate.
//
// At decode the router GEMM and router_topk_scatter are two dependent dispatches per routed layer
// with almost no work in the second. The selection needs a row's every logit, so it runs in the
// workgroup that finishes last, named by an arrival counter (the pattern of gemm_fp8a8's split-K
// finisher). Here the counter is the op instance's, taken at init, so it is never shared by
// concurrent launches and always starts zeroed. The shared expert's gate (gemm_nt at N = 1 over
// the same input) rides along as one more workgroup (`skg` > 0), removing its own launch.
//
// Byte-identical to the launches it replaces. The library has two bf16 skinny GEMM rows and
// halo_rows issues the m16 one up to M = 16 and the m64 one above, so this op carries BOTH
// reductions, selected by M:
//   - Above M = 16, halo_gemm_bf16_nt_m64's arithmetic: an output element is SK partial tiles, each
//     one wave's WMMA chain over a contiguous K slice, summed in ascending split order from 0.f and
//     narrowed by the language's conversion (not f32_to_bf16, which differs on NaN). Only SK
//     matters, so the split counts come from the same law (cfg_for in halo_gemm_args.h with the
//     same N-tile count), and a wave may run several (tile, split) items, each into its own LDS
//     slot. What the port had to change is the fragment layer and one output index, both noted
//     below.
//   - At M <= 16, halo_gemm_bf16_nt_m16's: a wave a column of the split, v_dot2_f32_bf16, a
//     __shfl_xor butterfly, splits summed ascending -- with that row's own split ladder, which is
//     not cfg_for's (see rg_m16_sk). rg_m16_phase below is that kernel's body, element for
//     element.
//   - The selection and sort are router_topk_scatter's body -- halo_router_topk.h's row function
//     and halo_moe_sort.h's placement -- so the last block's wave count changes no bit.
//
// NOTUNE on the router's shape: a tuned point for the unfused router would move SK. Decode only:
// at most kRgMaxM rows (four 16-row tiles); beyond that the architecture issues the pair.
//
// ---- M <= 16: why there are two phases, and why the m16 one is the right one --------------------
//
// The two rows are different reductions and their outputs differ in the last bit: measured on
// --case router_gemm_topk_scatter_bf16, 1 to 4 bytes of M * n_expert at M = 1..16 and none above.
// Which of the two is the weaker one is not a matter of taste. Against an exact f64 fold of the
// same bf16 products (the router's shape, N = 512, K = 2560, SK = 4, both reductions run on one
// draw) the m64/WMMA chain sits 19.3 f32 ulp from the truth on average (max 4802) and the
// m16/fdot2 chain 5.4 (max 1872) -- the WMMA 16-term dot is the lossy one, by 3.6x -- and on
// every one of the 9 elements that round to a different bf16 across M = 1..16, m16 is the closer.
// So this is not a tolerance argument: no tolerance was relaxed, and the fused op runs the row the
// engine serves in the band so that "byte-identical to the launches it replaces" holds for the
// same reason at every M.
//
// Cost: the m16 phase is a scalar-FMA reduction (no tensor cores), i.e. more issue slots per MAC
// than the WMMA one -- but that is exactly what the unfused pair pays at M <= 16 on this part, and
// halo_gemm_bf16_nt_m16.hip records it as the measured winner there. MT follows M (1, 2, 4, 8, 16)
// so a short row does not pay for sixteen.
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_emit.h"
#include "halo_gemm_args.h"
#include "halo_gdn_wmma.h"
#include "halo_moe_sort.h"
#include "halo_router_topk.h"
#include "halo_wmma.h"

#include <hip/hip_bf16.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>

#define HALO_RG_CHK() do { hipError_t e_ = hipGetLastError();                      \
                           if (e_ != hipSuccess) return HALO_NO(RAD_E_DEVICE); } while (0)

constexpr long long kRgMaxM = 64;

namespace {

using halo_moesort::kMaxExpert;
using halo_moesort::kScatterSmall;
using halo_moesort::Protect;
using halo_moesort::protect_of;
using halo_moesort::scatter_small_place;
using halo_moesort::ScatterLds;

constexpr uint32_t kWave = 32;
constexpr int kRgKU = 8;                  /* halo_gemm_bf16_nt_m64's KU: the unroll, not the order */

/* ---- the fragment layer, gfx1151 form -------------------------------------------------------
 *
 * libr4d's gfx12 fragment is two 8-byte halves sixteen bytes apart, so its `rg_fragU` reads
 * `p[4u]` and `p[4u+2]`. RDNA 3.5 gives a lane the row's whole 16-wide k run as 32 contiguous
 * bytes (halo_wmma.h), so one k step is `p[2u]` and `p[2u+1]` -- 32 bytes either way, which is why
 * the pointer arithmetic below is unchanged and only what one step loads differs.
 *
 * The pointer is explicitly global: a plain pointer held in an array across iterations becomes
 * generic, so loads become flat_load_b64, which on RDNA also increments dscnt and makes later LDS
 * waits wait on global memory. */
typedef unsigned rg_u4v __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) const rg_u4v* rg_gu4;

__device__ __forceinline__ v16bf rg_fragU(rg_gu4 p, int u) {
    union { rg_u4v w[2]; v16bf f; } x;
    x.w[0] = p[2 * u];
    x.w[1] = p[2 * u + 1];
    return x.f;
}

/* One (tile, split) item: the 16 x 16 partial tile halo_gemm_bf16_nt_m64_kernel's wave (n0, ks)
 * accumulates at MT = 1, with the same loads (rowGo, halo_gdn_wmma.h -- no half-wave k term on
 * gfx11) and the same WMMA chain from zero. */
__device__ __forceinline__ v8f rg_item(const uint16_t* __restrict__ A,
                                       const uint16_t* __restrict__ W, int M, int K, int N,
                                       int n0, int m0, int ks, int SK, int lane) {
    const int Ks = K / SK, ksteps = Ks / 16, kbase = ks * Ks;
    v8f acc = (v8f)(0.f);
    const unsigned woff = rowGo((unsigned)K, n0, N - 1, lane) + (unsigned)kbase;
    const unsigned aoff = rowGo((unsigned)K, m0, M - 1, lane) + (unsigned)kbase;
    if (n0 < N) {
        rg_gu4 pw = (rg_gu4)(const rg_u4v*)(W + woff);
        rg_gu4 pa = (rg_gu4)(const rg_u4v*)(A + aoff);
        int s = 0;
        for (; s + kRgKU <= ksteps; s += kRgKU) {
#pragma unroll
            for (int u = 0; u < kRgKU; u++) {
                const v16bf b = rg_fragU(pw, u);
                acc = mma(rg_fragU(pa, u), b, acc);
            }
            pw += kRgKU * 2;
            pa += kRgKU * 2;
        }
        for (; s < ksteps; s++) {
            const v16bf b = rg_fragU(pw, 0);
            pw += 2;
            acc = mma(rg_fragU(pa, 0), b, acc);
            pa += 2;
        }
    }
    return acc;
}

/* ---- the M <= 16 phase: halo_gemm_bf16_nt_m16.hip's arithmetic, element for element ----------
 *
 * The band has TWO reductions in this library and the fused op has to carry the one it stands in
 * for. halo_gemm_bf16_nt_m16_kernel is a column a wave (not a 16-column tile): a lane holds eight
 * contiguous K of its split, dots them with v_dot2_f32_bf16, the wave's 32 lanes reduce through a
 * __shfl_xor butterfly, and the splits are summed ascending from 0.f. m64's -- rg_item above -- is
 * a WMMA chain. Same SK, same K slices, different tree, so the emitted bf16 differs on the last
 * bit for the one element in two thousand whose exact value lands on a rounding boundary
 * (measured, router shape: 1 of 512 at M = 1, 4 of 8192 at M = 16).
 *
 * MT is M rounded up to 1, 2, 4, 8 or 16. Rows are independent -- a row's chain touches no other
 * row -- so a wider MT changes no bit of a row below M; rows at or past M are computed on the last
 * row's data (no read past A) and dropped, and a narrower MT is what keeps M = 1 from paying for
 * sixteen rows. One item is one (column, split); its partials land in MT floats of LDS, not 256.
 */
typedef __bf16 rg_bf162 __attribute__((ext_vector_type(2)));   /* v_dot2_f32_bf16 operand */

template <int MT>
__device__ __forceinline__ void rg_m16_phase(const uint16_t* __restrict__ A,
                                            const uint16_t* __restrict__ W, int M, int K, int N,
                                            int c0, int ncol, int SK, float* red, uint16_t* dst,
                                            int ld, int m0, bool gate, uint32_t tid,
                                            uint32_t wave, uint32_t lane, uint32_t NW) {
    const int Ks = K / SK;
    for (uint32_t it = wave; it < (uint32_t)ncol * (uint32_t)SK; it += NW) {
        const uint32_t j = it / (uint32_t)SK, ks = it % (uint32_t)SK;
        const int col = c0 + (int)j;
        float acc[MT];
#pragma unroll
        for (int i = 0; i < MT; ++i) acc[i] = 0.f;
        if (col < N) {
            const int kbase = (int)ks * Ks;
            const uint16_t* Wj = W + (size_t)col * K + kbase + (size_t)lane * 8u;
            const int niter = Ks / (32 * 8), rem = Ks - niter * (32 * 8);
            const bool has_tail = rem > 0 && (int)lane < rem / 8;
            for (int t = 0; t < niter; ++t) {
                const rg_bf162* wp = (const rg_bf162*)(Wj + (size_t)t * 32 * 8);
                const int kk = kbase + (int)lane * 8 + t * 32 * 8;
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    const int4 av = *(const int4*)(A + (size_t)(i < M ? i : M - 1) * K + kk);
                    const rg_bf162* ap = (const rg_bf162*)&av;
#pragma unroll
                    for (int p = 0; p < 4; ++p)
                        acc[i] = __builtin_amdgcn_fdot2_f32_bf16(ap[p], wp[p], acc[i], false);
                }
            }
            if (has_tail) {
                const rg_bf162* wp = (const rg_bf162*)(Wj + (size_t)niter * 32 * 8);
                const int kk = kbase + (int)lane * 8 + niter * 32 * 8;
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    const int4 av = *(const int4*)(A + (size_t)(i < M ? i : M - 1) * K + kk);
                    const rg_bf162* ap = (const rg_bf162*)&av;
#pragma unroll
                    for (int p = 0; p < 4; ++p)
                        acc[i] = __builtin_amdgcn_fdot2_f32_bf16(ap[p], wp[p], acc[i], false);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            float v = acc[i];
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
            if (lane == 0) red[(size_t)it * MT + i] = v;
        }
    }
    __syncthreads();
    for (uint32_t idx = tid; idx < (uint32_t)ncol * (uint32_t)MT; idx += NW * kWave) {
        const uint32_t j = idx / (uint32_t)MT, i = idx % (uint32_t)MT;
        float v = 0.f;
        for (int s = 0; s < SK; ++s) v += red[((size_t)j * (size_t)SK + (size_t)s) * MT + i];
        const int col = c0 + (int)j, m = m0 + (int)i;
        if (col < N && m < M) {
            if (gate) ((__hip_bfloat16*)dst)[m] = (__hip_bfloat16)v;
            else      ((__hip_bfloat16*)dst)[(size_t)m * ld + (size_t)col] = (__hip_bfloat16)v;
        }
    }
}

/* Launch geometry: `sk` the router's split, `tpb` its 16-column tiles per block, `nrb` the
 * router's blocks along x, `skg` the gate's split (0: no gate; its block is x = nrb), `m16` the
 * M <= 16 phase, `mt` its row count and `cpb` its columns a block. Rows are tiled along y, one
 * 16-row tile per block. */
struct RgGeom {
    uint32_t M, K, n_expert, n_local, top_k, lg_ld;
    int32_t  norm, base;
    uint32_t sk, tpb, nrb, skg, m16, mt, cpb;
};

/* `lg` is neither const nor __restrict__: every block writes it and the last one reads it back. */
template <uint32_t MAXOWN, uint32_t NW>
__global__ __launch_bounds__(NW * kWave) void router_gemm_topk_scatter_kernel(
    const uint16_t* __restrict__ A, const uint16_t* __restrict__ W,
    const uint16_t* __restrict__ Wg, uint16_t* lg, uint16_t* __restrict__ gate,
    int32_t* __restrict__ ids, uint16_t* __restrict__ w, int32_t* __restrict__ sorted,
    int32_t* __restrict__ off, int32_t* __restrict__ cnt, unsigned int* __restrict__ ctr,
    RgGeom g, Protect prot) {
    /* The GEMM's partial tiles, then (last block only, after two barriers) the sort's LDS; the
     * shim sizes it for the larger. */
    extern __shared__ __align__(16) unsigned char rg_smem[];
    float* const red = (float*)rg_smem;
    const uint32_t tid = threadIdx.x, wave = tid / kWave, lane = tid % kWave;

    const bool      gb  = g.skg > 0 && blockIdx.x == g.nrb;
    const int       SK  = (int)(gb ? g.skg : g.sk);
    const uint32_t  nti = gb ? 1u : g.tpb;
    const int       N   = gb ? 1 : (int)g.n_expert;
    const uint32_t  nt0 = gb ? 0u : blockIdx.x * g.tpb;
    const uint16_t* Wb  = gb ? Wg : W;
    const int       m0  = (int)(blockIdx.y * 16u);

    if (g.m16) {
        /* A column a wave, m16's tree. The switch is on a launch-wide value, so every thread of
         * every block takes the same arm and the phase's barrier is uniform.
         *
         * `ncol` is COLUMNS, and its per-column LDS is SK * MT floats (one partial an item), so a
         * block reserves ncol * SK * MT -- the shim sizes exactly that (the host's `lds`). The gate
         * block is one column: it runs the m16 row's gate launch (WV = 1, N = 1), and columns
         * 1..15 would be work above N plus SK * MT floats of `red` nobody reserved. */
        const int c0 = gb ? 0 : (int)(blockIdx.x * g.cpb), ncol = gb ? 1 : (int)g.cpb;
        uint16_t* const dst = gb ? gate : lg;
#define RG_M16(MT)                                                                          \
        rg_m16_phase<MT>(A, Wb, (int)g.M, (int)g.K, N, c0, ncol, SK, red, dst, (int)g.lg_ld,    \
                         m0, gb, tid, wave, lane, NW)
        switch (g.mt) {
          case 1:  RG_M16(1);  break;
          case 2:  RG_M16(2);  break;
          case 4:  RG_M16(4);  break;
          case 8:  RG_M16(8);  break;
          default: RG_M16(16); break;
        }
#undef RG_M16
    } else {
    const uint32_t items = nti * (uint32_t)SK;
    for (uint32_t it = wave; it < items; it += NW) {
        const uint32_t j = it / (uint32_t)SK, ks = it % (uint32_t)SK;
        const v8f acc = rg_item(A, Wb, (int)g.M, (int)g.K, N, (int)((nt0 + j) * 16u), m0,
                                (int)ks, SK, (int)lane);
        float* const slot = red + (size_t)it * 256;           /* it == j * SK + ks */
#pragma unroll
        for (int e = 0; e < 8; e++) slot[e * (int)kWave + (int)lane] = acc[e];
    }
    __syncthreads();
    /* The reduction, as in halo_gemm_bf16_nt_m64_kernel: splits in ascending order from 0.f, and
     * the language's float-to-bf16 conversion. The row an accumulator element belongs to is
     * gfx11's D mapping (accRow, halo_gdn_wmma.h) -- libr4d's `8 * (l >> 4) + (o >> 5)` is the
     * gfx12 one and writes every row twice on this part. */
    for (uint32_t idx = tid; idx < nti * 256u; idx += NW * kWave) {
        const uint32_t j = idx >> 8, o = idx & 255u;
        float v = 0.f;
        for (int s = 0; s < SK; s++) v += red[((size_t)j * (size_t)SK + (size_t)s) * 256 + o];
        const uint32_t l = o & 31u;
        const int n = (int)((nt0 + j) * 16u + (l & 15u));
        const int m = m0 + accRow((int)l, (int)(o >> 5));
        if (n < N && m < (int)g.M) {
            if (gb) ((__hip_bfloat16*)gate)[m] = (__hip_bfloat16)v;
            else    ((__hip_bfloat16*)lg)[(size_t)m * g.lg_ld + (size_t)n] = (__hip_bfloat16)v;
        }
    }
    }

    /* ---- the last block ranks. Every store is made visible device-wide before the arrival is
     * published; the block completing the count resets it for the next launch and acquires the
     * other blocks' logits. */
    __threadfence();
    __syncthreads();
    __shared__ unsigned int s_arrived;
    if (tid == 0)
        s_arrived = __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (s_arrived + 1u != gridDim.x * gridDim.y) return;
    if (tid == 0) __hip_atomic_store(ctr, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __threadfence();

    /* router_topk_scatter_kernel, from here to the end. */
    ScatterLds<NW>& sl = *reinterpret_cast<ScatterLds<NW>*>(rg_smem);
    for (uint32_t e = tid; e < g.n_local; e += blockDim.x) sl.cnt[e] = 0;
    const halo_routertopk::SlotSink snk{sl.e, g.base, g.n_local, prot};
    for (uint32_t m = wave; m < g.M; m += NW)
        halo_routertopk::router_topk_row<MAXOWN>(lg, ids, w, m, g.n_expert, g.top_k, g.lg_ld,
                                                g.norm, lane, snk);
    lds_barrier();
    scatter_small_place(sorted, off, cnt, g.M * g.top_k, g.n_local, sl);
}

/* The arrival counter belongs to the op instance and is taken at init, not on the launch path.
 *
 * It must persist between launches, zeroed (the last block resets it; a non-zero start would rank
 * before the GEMM finished), so scratch cannot hold it: other ops write the arena between
 * launches. Allocating on first use would escape the VRAM budget instead. Init runs once per
 * resolved instance during declare, before the engine measures free VRAM, so its allocation is
 * counted.
 *
 * One counter per instance (op, band, domain, rank). Each routed layer declares its own op, and an
 * instance's launches are ordered since they write the same declared outputs. Counters are 64-byte
 * slots of one zeroed block per device, freed with its last instance. */
constexpr uint32_t kRgCtrStride = 16;       /* uint32 words a slot: one 64-byte line each */
constexpr uint32_t kRgCtrSlots  = 4096;     /* instances a device; a model declares hundreds */

struct RgCtrBlock {
    unsigned int* base = nullptr;
    uint32_t next = 0;                      /* slots handed out since the block was made */
    uint32_t live = 0;                      /* instances holding one */
};
struct RgInst {
    int dev = 0;
    unsigned int* ctr = nullptr;
};
std::mutex g_rg_mu;
std::map<int, RgCtrBlock> g_rg_blocks;

}  /* namespace */

extern "C" int halo_rad_router_gemm_topk_scatter_init(const RadParam*, int, int, int,
                                                      void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return HALO_NO(RAD_E_DEVICE);
    std::lock_guard<std::mutex> lk(g_rg_mu);
    RgCtrBlock& blk = g_rg_blocks[dev];
    if (!blk.base) {
        const size_t bytes = (size_t)kRgCtrSlots * kRgCtrStride * sizeof(unsigned int);
        void* p = nullptr;
        if (hipMalloc(&p, bytes) != hipSuccess) return HALO_NO(RAD_E_NOMEM);
        /* Zero before any launch reads it. In the engine hipMemset is the AQL backend's and has
         * finished when it returns (core/device/aql.cpp); under HIP's own runtime (halo_selftest)
         * it runs on the null stream, which a non-blocking stream does not wait for, so wait. */
        if (hipMemset(p, 0, bytes) != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
            (void)hipFree(p);
            return HALO_NO(RAD_E_DEVICE);
        }
        blk.base = (unsigned int*)p;
        blk.next = 0;
    }
    /* A slot is never reused while its block lives: one returned by fini may not be zero if its
     * last launch did not finish. */
    if (blk.next >= kRgCtrSlots) return HALO_NO(RAD_E_FULL);
    RgInst* in = new RgInst;
    in->dev = dev;
    in->ctr = blk.base + (size_t)blk.next * kRgCtrStride;
    ++blk.next;
    ++blk.live;
    *out = in;
    return RAD_OK;
}

extern "C" void halo_rad_router_gemm_topk_scatter_fini(void* inst) {
    RgInst* in = (RgInst*)inst;
    if (!in) return;
    std::lock_guard<std::mutex> lk(g_rg_mu);
    auto it = g_rg_blocks.find(in->dev);
    if (it != g_rg_blocks.end() && it->second.live > 0 && --it->second.live == 0) {
        int prev = 0;
        const bool moved = hipGetDevice(&prev) == hipSuccess && prev != in->dev &&
                           hipSetDevice(in->dev) == hipSuccess;
        (void)hipFree(it->second.base);
        if (moved) (void)hipSetDevice(prev);
        g_rg_blocks.erase(it);
    }
    delete in;
}

/* ---- halo_rad_gemm_bf16_m16's split law, verbatim (halo_gemm_bf16_nt_m16.hip) ----------------
 *
 * The M <= 16 row decides its own SK, so the fused op cannot ask cfg_for and stay byte-identical:
 * at the router's shape the two laws happen to agree (4), the gate's do not (cfg_for answers 20,
 * the ladder 32), and only the ladder is the row's. The legality test is the kernel's eight-tile
 * register file: K % SK == 0, K/SK a multiple of 8 (VEC, so no lane holds a partial load) and at
 * most 8 tiles of 256 per split. */
static bool rg_m16_legal(long long K, int wv, int sk) {
    return sk >= 1 && wv >= 1 && (long long)wv * sk * 32 <= 1024 && K % sk == 0 &&
           ((K / sk) % 8) == 0 && (K / sk) / (32 * 8) <= 8;
}

/* A tuned split the row's register file accepts, else the smallest legal split that puts 1280
 * waves (one card) in flight, else the largest legal one. 0 means the row has no legal split and
 * forwards to _m64, so the fused op stays on the WMMA phase too. */
static int rg_m16_sk(long long K, long long N, int wv, int sk) {
    if (sk > 0 && rg_m16_legal(K, wv, sk)) return sk;
    static const int ladder[] = { 1, 2, 4, 5, 8, 10, 16, 20, 32 };
    int got = 0;
    for (int cand : ladder) {
        if (!rg_m16_legal(K, wv, cand)) continue;
        got = cand;
        if (N * cand >= 1280) break;
    }
    return got;
}

/* Operands: a, b (the router), gate_b? -> logits, expert_ids, expert_w, sorted_tok,
 * expert_offset, expert_count, gate?. The two optional ones come together or not at all. The
 * replaced shims' checks are all made here, plus the row bound. */
extern "C" int halo_rad_router_gemm_topk_scatter(const RadArgs* a, RadStream s) {
    const RadTensor* x      = halo_opt(a, 0);
    const RadTensor* wr     = halo_opt(a, 1);
    const RadTensor* wg     = halo_opt(a, 2);
    const RadTensor* lg     = halo_opt(a, 3);
    const RadTensor* ids    = halo_opt(a, 4);
    const RadTensor* w      = halo_opt(a, 5);
    const RadTensor* sorted = halo_opt(a, 6);
    const RadTensor* off    = halo_opt(a, 7);
    const RadTensor* cnt    = halo_opt(a, 8);
    const RadTensor* gate   = halo_opt(a, 9);
    if (!x || !wr || !lg || !ids || !w || !sorted || !off || !cnt) return HALO_NO(RAD_E_INVAL);
    if ((wg != nullptr) != (gate != nullptr)) return HALO_NO(RAD_E_INVAL);

    const long long n_expert = halo_geti_or(a, "n_expert", 0);
    const long long n_local  = halo_geti_or(a, "n_local", n_expert);
    const long long top_k    = halo_geti_or(a, "top_k", 0);
    const long long norm     = halo_geti_or(a, "norm", 0);
    const long long base     = halo_geti_or(a, "expert_base", 0);
    if (n_expert <= 0 || n_expert > (long long)kMaxExpert || top_k <= 0 || top_k > n_expert)
        return HALO_NO(RAD_E_SHAPE);
    if (top_k > (long long)kWave) return HALO_NO(RAD_E_SHAPE);
    if (n_local <= 0 || n_local > n_expert) return HALO_NO(RAD_E_SHAPE);
    if (base < 0 || base + n_local > n_expert) return HALO_NO(RAD_E_SHAPE);
    Protect prot;
    const int prc = protect_of(a, n_local, &prot);
    if (prc != RAD_OK) return prc;
    if (x->dtype != RAD_BF16 || wr->dtype != RAD_BF16 || lg->dtype != RAD_BF16 ||
        w->dtype != RAD_BF16) return HALO_NO(RAD_E_DTYPE);
    if (wg && (wg->dtype != RAD_BF16 || gate->dtype != RAD_BF16)) return HALO_NO(RAD_E_DTYPE);
    if (ids->dtype != RAD_I32 || sorted->dtype != RAD_I32 || off->dtype != RAD_I32 ||
        cnt->dtype != RAD_I32) return HALO_NO(RAD_E_DTYPE);

    /* The GEMM's operands, as gemm_ops reads them: A [M, K] and the output contiguous, A and the
     * weights 16-byte aligned. */
    long long M = 0, K = 0;
    if (halo_rows_of(x, &M, &K) != RAD_OK) return HALO_NO(RAD_E_SHAPE);
    if (!halo_contig(x)) return HALO_NO(RAD_E_STRIDE);
    if (halo_geti_or(a, "K", K) != K) return HALO_NO(RAD_E_SHAPE);
    if (M < 1 || M > kRgMaxM) return HALO_NO(RAD_E_SHAPE);
    if (!halo_aligned(x->data, 16) || !halo_aligned(wr->data, 16) ||
        (wg && !halo_aligned(wg->data, 16))) return HALO_NO(RAD_E_ALIGN);
    if (halo_numel(wr) < n_expert * K || (wg && halo_numel(wg) < K)) return HALO_NO(RAD_E_SHAPE);
    if (!halo_contig(lg) || halo_numel(lg) < M * n_expert) return HALO_NO(RAD_E_STRIDE);
    if (gate && (!halo_contig(gate) || halo_numel(gate) < M)) return HALO_NO(RAD_E_STRIDE);

    const long long T = M * top_k;
    if (T > (long long)kScatterSmall) return HALO_NO(RAD_E_SHAPE);
    if (!halo_contig(ids) || !halo_contig(w) || !halo_contig(sorted) || !halo_contig(off) ||
        !halo_contig(cnt)) return HALO_NO(RAD_E_STRIDE);
    if (halo_numel(ids) < T || halo_numel(w) < T) return HALO_NO(RAD_E_SHAPE);
    if (halo_numel(sorted) < T) return HALO_NO(RAD_E_SHAPE);
    if (halo_numel(off) < n_local + 1 || halo_numel(cnt) < n_local) return HALO_NO(RAD_E_SHAPE);

    /* WHICH ROW THIS IS. halo_rows serves the bf16 skinny GEMM with gemm_bf16_nt_m16 in this band
     * (cGemmNt: M <= 16; priority 60 over _m64's 50) and with _m64 above it, and the two are NOT
     * the same reduction -- fdot2 pairs and a wave butterfly against a WMMA chain -- so the fused
     * op has to run the one it replaces. Above, _m64: cfg_for with ceil(N/16) tiles, the router
     * asking with its width and the gate with one tile. Below, m16's own law
     * (halo_gemm_bf16_nt_m16.hip): a tuned split its eight-tile register file accepts, else the
     * smallest legal split from its ladder that keeps 1280 waves in flight; no legal split at all
     * and that row forwards to _m64, which this phase then does too. The selftest names the row by
     * M alone, and so does the band's one caller shape set (K 2560 and 2048, both K % 256 == 0,
     * which cGemmNt also requires), so M <= 16 is the test. */
    const long long ntiles = (n_expert + 15) / 16;
    Cfg cr{}, cg{};
    HaloGemmCfg cv{};
    halo_cfg_gemm(a, &cv);
    const int wv16 = cv.wv > 0 ? cv.wv : 1;
    bool m16 = false;
    long long skr = 0, skg = 0;
    if (M <= 16) {
        skr = rg_m16_sk(K, n_expert, wv16, cv.sk);
        skg = wg ? rg_m16_sk(K, 1, wv16, cv.sk) : 0;
        m16 = skr > 0 && (!wg || skg > 0);
    }
    int rc = RAD_OK;
    if (!m16) {
        rc = cfg_for(a, K, M, 16, BR_FLAT, &cr, ntiles);
        if (rc != RAD_OK) return rc;
        if (wg) {
            rc = cfg_for(a, K, M, 16, BR_FLAT, &cg, 1);
            if (rc != RAD_OK) return rc;
        }
    }
    const uint32_t mt = M <= 1 ? 1u : M <= 2 ? 2u : M <= 4 ? 4u : M <= 8 ? 8u : 16u;
    /* The instance's counter (see the arrival-counter note). Without init there is none and the
     * op is refused. */
    const RgInst* const inst = (const RgInst*)a->instance;
    if (!inst || !inst->ctr) return HALO_NO(RAD_E_STATE);
    unsigned int* const ctr = inst->ctr;

    /* The wave count follows router_topk_scatter's rule (a wave per row, up to 32); the GEMM takes
     * as many 16-column tiles per block as fill those waves at its split. In the m16 phase a wave
     * owns a column and its split, and a block takes as many columns as give each wave one item
     * (two at the served 512 x 2560, SK 4, eight waves), as halo_gemm_bf16_nt_m16 runs a column a
     * block, rather than a 16-column tile (33 blocks, eight items a wave in turn): cold, 18.5,
     * 19.4 and 22.2 us at M = 1, 4 and 8 against 20.1, 21.2 and 24.9. At 16 rows every block
     * would re-read sixteen A rows for its two columns (34.5 against 28.3 us), so wider rows keep
     * the tile. */
    const uint32_t nw = halo_routertopk::scatter_waves((uint32_t)M);
    long long tpb = m16 ? 1 : (long long)nw / cr.sk;
    if (tpb < 1) tpb = 1;
    if (tpb > ntiles) tpb = ntiles;
    long long cpb = 16;
    if (m16 && mt <= 8) {
        cpb = std::max(1LL, (long long)nw / skr);
        if (cpb > n_expert) cpb = n_expert;
    }
    const long long nrb = m16 ? (n_expert + cpb - 1) / cpb : (ntiles + tpb - 1) / tpb;
    size_t lds = (size_t)(tpb * cr.sk) * 256 * sizeof(float);
    if (wg) lds = std::max(lds, (size_t)cg.sk * 256 * sizeof(float));
    if (m16) {   /* MT floats an item (SK items a column), not an accumulator row of 256 */
        lds = std::max(lds, (size_t)cpb * (size_t)skr * (size_t)mt * sizeof(float));
        /* One column: the phase's ncol for the gate block. */
        if (wg) lds = std::max(lds, (size_t)skg * (size_t)mt * sizeof(float));
    }
    const size_t sort_lds = nw == 8 ? sizeof(ScatterLds<8>)
                          : nw == 16 ? sizeof(ScatterLds<16>) : sizeof(ScatterLds<32>);
    lds = std::max(lds, sort_lds);

    RgGeom g{};
    g.M = (uint32_t)M;               g.K = (uint32_t)K;
    g.n_expert = (uint32_t)n_expert; g.n_local = (uint32_t)n_local;
    g.top_k = (uint32_t)top_k;       g.lg_ld = (uint32_t)n_expert;
    g.norm = (int32_t)norm;          g.base = (int32_t)base;
    g.sk = m16 ? (uint32_t)skr : (uint32_t)cr.sk;   g.tpb = (uint32_t)tpb;
    g.nrb = (uint32_t)nrb;           g.skg = wg ? (uint32_t)(m16 ? skg : cg.sk) : 0u;
    g.m16 = m16 ? 1u : 0u;           g.mt = mt;
    g.cpb = (uint32_t)cpb;
    const dim3 grid((unsigned)(nrb + (wg ? 1 : 0)), (unsigned)((M + 15) / 16));

#define HALO_RG_L(OWN, NWV)                                                                     \
    HALO_EMIT(s, (router_gemm_topk_scatter_kernel<OWN, NWV>), grid, dim3((NWV) * kWave),         \
              (unsigned)lds, (const uint16_t*)x->data, (const uint16_t*)wr->data,               \
              wg ? (const uint16_t*)wg->data : (const uint16_t*)nullptr, (uint16_t*)lg->data,     \
              gate ? (uint16_t*)gate->data : (uint16_t*)nullptr, (int32_t*)ids->data,            \
              (uint16_t*)w->data, (int32_t*)sorted->data, (int32_t*)off->data,                   \
              (int32_t*)cnt->data, ctr, g, prot)
    if (n_expert <= (long long)(8 * kWave)) {
        if      (nw == 8)  rc = HALO_RG_L(8, 8);
        else if (nw == 16) rc = HALO_RG_L(8, 16);
        else               rc = HALO_RG_L(8, 32);
    } else {
        if      (nw == 8)  rc = HALO_RG_L(16, 8);
        else if (nw == 16) rc = HALO_RG_L(16, 16);
        else               rc = HALO_RG_L(16, 32);
    }
#undef HALO_RG_L
    if (rc != RAD_OK) return rc;
    HALO_RG_CHK();
    return RAD_OK;
}
