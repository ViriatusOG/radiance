// exl3r4d_prefill.hip -- the EXL3 routed-expert GEMMs (moe_exl3_gate_up, moe_exl3_down) on gfx1201, prefill form of
// the M > 16 band. The decode band and every shared device helper are in exl3r4d.hip and exl3r4d_common.hip.h; this
// file adds the run map and the MTX GEMM and nothing else.
//
// THE DESIGN. Run ownership: a workgroup owns up to MTX consecutive 16-row blocks of ONE expert, so one decoded weight
// fragment feeds up to MTX WMMAs. The trellis decode costs more than an MXFP4 fragment, so amortising it over more rows
// pays more here than in an MXFP4 GEMM (MTX 4 or 8 for prefill runs of tens of rows, ~80 typical). NPW column tiles a
// wave; no split-K (sk 1). The K loop, the decode, the prefetch and the epilogue are the decode band's
// (exl3r4d_common.hip.h), with MT A fragments a step instead of one.
//
// RADIANCE'S ROUTING CONTRACT (include/exl3_abi.h) shapes the rest:
//   * OWNERSHIP FROM A RUN-ALIGNED MAP, not vLLM's padded block-16 sort. Radiance hands sorted_tok + expert_offset:
//     expert e's rows are the contiguous run [off[e], off[e+1]), unpadded. One workgroup of one launch builds the map
//     in scratch (libr4d's moe_blockmap_kernel pattern): an entry {expert, r0, r1} for every 16 * MTX rows of every
//     run, and zero entries for the rows no run owns ([0, off[0]) and [off[n_expert], T): dropped slots, protected
//     experts' rows -- libr4d's rule). The GEMM's grid is (column block, entry); a workgroup past the map's count exits
//     at its head. Every row of [0, T) is in exactly one entry, so every output row is written once a column block.
//     The map is sized at declare from the band top (exl3r4d_pf_map_cap; the scratch hook).
//   * WEIGHTS ARE TABLES (planes_of): codes / suh / suh.1 / svh per parity class; the narrower class zero-fills its
//     columns past its width (the grid spans the wider class).
//   * A = the prep kernel's rows in scratch (exl3r4d_launch_prep: had128(f16(x * suh)) per (row, matrix)), the form
//     measured faster than rotating in the K loop at decode. At prefill the in-loop rotation is AM 1 / 3, measured
//     slower (the measurements at the end of this file).
//   * Epilogue per 16-row block = the decode band's epilogue_block (EPI 1: h = f16(silu(gate) * up); EPI 2: y = bf16),
//     so the two bands produce the same bits for the same row (the order of every sum is the same: one wave owns a
//     16x16 output tile and walks K in order, whatever MT is).
//
// NOT SUPPORTED HERE (each named): skg, the fused top-k sum, down pieces, dual pools, EPI 0, seg, the fused prep (its
// rows do not fit LDS at MTX > 1), MTX 8 with NPW 4 (its accumulators alone would be 256 VGPRs).
//
// The measured alternatives, readings and rejected forms are at the end of this file ("Measurements behind the
// defaults"); read them before changing a default here.
// THIS BAND'S ROW-STORE HINTS (the shared header's st_row; codes there), set before the header is read. The prefill
// band's kernels -- its GEMM epilogues (h, y) and its map-form prep (xr) -- take EXL3R4D_PF_STH_*; the decode band's
// translation unit keeps EXL3R4D_STH_* (and with it the decode band's prep_row kernel, which writes xr for the run form
// here too). All 0 (regular temporal stores): the measurements at the end of this file.
#ifndef EXL3R4D_PF_STH_XR
#define EXL3R4D_PF_STH_XR 0
#endif
#ifndef EXL3R4D_PF_STH_H
#define EXL3R4D_PF_STH_H 0
#endif
#ifndef EXL3R4D_PF_STH_Y
#define EXL3R4D_PF_STH_Y 0
#endif
#undef EXL3R4D_STH_XR
#undef EXL3R4D_STH_H
#undef EXL3R4D_STH_Y
#define EXL3R4D_STH_XR EXL3R4D_PF_STH_XR
#define EXL3R4D_STH_H EXL3R4D_PF_STH_H
#define EXL3R4D_STH_Y EXL3R4D_PF_STH_Y
#include "exl3r4d_common.hip.h"
#include "exl3r4d_prefill.h"
#include <stdio.h>
#include <string.h>

// ----------------------------------------------------------------------------- host-read codes: order and copy ring
// WHY. Under expert tensor parallel (DeepSeek V4.1), a prefill step over many newly routed experts streams ~5 GiB a
// rank of expert slices that sit in the engine's pinned host pool, read in place by this GEMM over the link. Per layer
// (32 new sequences of ~1.3k rows, the K = 2 layers), gate_up us = ~750 + 5.9 x (slices read from VRAM) + 48.3 x
// (slices read from the host): the host part (~2.7 ms of ~4.7) adds to the resident part instead of overlapping it,
// because the longest-first order (EXL3R4D_PF_LPT) puts the host-read experts -- the cold, short ones -- at the end of
// each launch, where the link runs alone.
// WHAT IT DOES, all inside the launch and the same bytes (fn4 switch "exl3_xr", prefill.c): (1) the map kernel orders
// each chunk's groups VRAM-read first (longest first), then host-read (longest first); (2) the first grid row of the
// GEMM launch (the copy row) copies the host-read groups' codes, in that order, into a ring in this op's scratch while
// the rest of the grid computes the resident ones; (3) a host-read group's workgroups wait for their copy's flag and
// read the copy. A copy that does not fit the ring, or whose flag does not come within EXL3R4D_XR_SPIN polls, is read
// in place. The order of entries does not change a row's bytes (every entry computes the same rows whatever order the
// workgroups run in), and a copy is the pool slot's bytes, so the output is identical to the launch without it.
// The residency test is the codes pointer against this rank's host pools (rad_host_pool_ranges at start; the launch
// carries them in its arguments). Two-class build only (the _mod4 forms keep xr_mode 0).
#define EXL3R4D_XR_ON (EXL3R4D_NCLS == 2)
#ifndef EXL3R4D_XR_SPIN   // flag polls of ~0.2 us (s_sleep 8) before a workgroup reads its group in place
#define EXL3R4D_XR_SPIN 200000
#endif
#if EXL3R4D_XR_ON
// The copy ring's counters (a reading; prefill.c prints them periodically): [0] chunks with a ring, [1] host-read
// heads in them, [2] heads copied, [3] 256-byte units copied, [4] workgroups that read a copy, [5] workgroups past the
// spin (read in place), [6] host-read groups the order saw (modes 1 / 2 / 5), [7] groups it saw. In the rank's device
// buffer (a.xr_cnt).
#define XR_CNT(i, v) do { if (a.xr_cnt) atomicAdd(&a.xr_cnt[i], (unsigned long long)(v)); } while (0)
#else
#define XR_CNT(i, v) ((void)0)
#endif
static __device__ __forceinline__ bool xr_host_codes(const Exl3r4dPrefill& a, const void* c) {
#if EXL3R4D_XR_ON
  const unsigned long long v = (unsigned long long)c;
  const int n = a.xr_n < EXL3R4D_XR_ARGPOOLS ? a.xr_n : EXL3R4D_XR_ARGPOOLS;
  for (int i = 0; i < n; i++)
    if (v >= a.xr_lo[i] && v < a.xr_hi[i]) return true;
#endif
  (void)a, (void)c;
  return false;
}
// The copy ring's tables, after the order array: xoff[cap] (ring offset in 256-byte units + 1, or 0), xhead[cap] (the
// entry's group head), xflag[cap] (copy workgroups done, by head), xlist[cap] (a chunk's copied heads in copy order, at
// the chunk's first entry), xcnt[64] (heads copied, by chunk); the ring after them. prefill.c sizes the same layout.
static __device__ __forceinline__ uint32_t* xr_tab(const Exl3r4dPrefill& a) {
  return (uint32_t*)((char*)a.order + (((size_t)4 * (size_t)a.cap + 255) / 256) * 256);
}
static __device__ __forceinline__ char* xr_ring(const Exl3r4dPrefill& a) {
  return (char*)xr_tab(a) + ((((size_t)4 * (size_t)a.cap + 64) * 4 + 255) / 256) * 256;
}

#ifndef EXL3R4D_PF_STAGES
#define EXL3R4D_PF_STAGES 2  // k steps of weights and activations in flight a wave (2 = double buffering, as decode)
#endif
// k steps of WEIGHTS in flight a wave (AM 0), a multiple of EXL3R4D_PF_STAGES (which stays the activations' depth), per
// op: gate_up 4 (-1.0/-2.4/-4.0/-9.0% at M=512/1024/2048/4096; the thread trace then shows its GEMM issue-bound, load
// waits 54% -> 7.5%); down stays 2 (4: +0.2..+3.2%, 208 VGPRs). EXL3R4D_PF_WSTAGES, if given, sets both (for
// measuring).
#ifdef EXL3R4D_PF_WSTAGES
#define EXL3R4D_PF_WSTAGES_GU EXL3R4D_PF_WSTAGES
#define EXL3R4D_PF_WSTAGES_DN EXL3R4D_PF_WSTAGES
#endif
#ifndef EXL3R4D_PF_WSTAGES_GU
#define EXL3R4D_PF_WSTAGES_GU 4
#endif
#ifndef EXL3R4D_PF_WSTAGES_DN
#define EXL3R4D_PF_WSTAGES_DN EXL3R4D_PF_STAGES
#endif
#ifndef EXL3R4D_PF_SGB   // an arm: the K step's schedule pinned with sched_group_barrier (1 grouped, 2 a tile at a time)
#define EXL3R4D_PF_SGB 0
#endif
// 1 (as in the decode band): the weight words and the A fragments as BUFFER loads -- a uniform resource, a per-lane
// offset fixed for the loop, the k-step's offset in an SGPR: no 64-bit address VALU a load. -2..-9% at prefill.
#ifndef EXL3R4D_PF_BUF
#define EXL3R4D_PF_BUF 1
#endif
#ifndef EXL3R4D_PF_KX_BUF   // K != 4 weight words through the buffer form (1) or the global form (0)
#define EXL3R4D_PF_KX_BUF 1
#endif
// The epilogue's svh loaded before the K loop (the shared header's early-svh form), per op: on for down
// (-0.9/-0.4/-0.7/-1.5% at M=512..4096, bit-identical); off for gate_up (+-0.2%). EXL3R4D_PF_SVH_EARLY, if given,
// sets both (for measuring).
#ifdef EXL3R4D_PF_SVH_EARLY
#define EXL3R4D_PF_SVH_EARLY_GU EXL3R4D_PF_SVH_EARLY
#define EXL3R4D_PF_SVH_EARLY_DN EXL3R4D_PF_SVH_EARLY
#else
#define EXL3R4D_PF_SVH_EARLY 0
#endif
#ifndef EXL3R4D_PF_SVH_EARLY_GU
#define EXL3R4D_PF_SVH_EARLY_GU 0
#endif
#ifndef EXL3R4D_PF_SVH_EARLY_DN
#define EXL3R4D_PF_SVH_EARLY_DN 1
#endif
#ifndef EXL3R4D_PF_NT
#define EXL3R4D_PF_NT 1  // 1: the weight words are nontemporal loads (read once a launch); 0: plain loads
#endif
// The th code of the GEMM's A-row loads (AM 0: gate_up's prep rows xr, h_rot down's h), gfx12 TH_LOAD_*: 0 RT, 1 NT,
// 3 LU (last use). A measuring arm; last-use loads did not change what the band leaves in L2 (the measurements at the
// end of this file), so 0.
#ifndef EXL3R4D_PF_ATH
#define EXL3R4D_PF_ATH 0
#endif

// ----------------------------------------------------------------------------- the run map

// ONE workgroup (1024 threads alone, 256 inside the fused map + prep kernel): each thread owns a contiguous range of
// expert ids (n_expert <= 4096 by the rows' constraint: at most 4 or 16 a thread), counts the entries its runs need, a
// wave prefix and a cross-wave prefix in LDS place them, and each thread writes its own. Thread 0 owns the head zero
// range [0, off[0]) and the last thread the tail [off[n_expert], T). Integers only: the map has one answer whatever
// order anything runs in. Writes past `cap` are dropped (only a malformed offset table can ask for more; its rows come
// out wrong, never written out of place).
static __device__ void pf_map_build(const Exl3r4dPrefill& a, int* s_w) {
  const Exl3r4dGemm& p = a.g;
  const int tid = threadIdx.x, nt = blockDim.x, lane = tid % WAVE32, wave = tid / WAVE32, nw = nt / WAVE32;
  const int T = p.T, fm = a.fm, cap = a.cap;
  const int lo = clamp_row(p.off[0], T), hi = clamp_row(p.off[p.n_expert], T);
  const int per = (p.n_expert + nt - 1) / nt, e0 = tid * per, e1 = min(e0 + per, p.n_expert);
  int cnt = 0;
  if (tid == 0) cnt += (lo + fm - 1) / fm;
  for (int e = e0; e < e1; e++) {
    const int a0 = clamp_row(p.off[e], T), a1 = clamp_row(p.off[e + 1], T);
    cnt += a1 > a0 ? (a1 - a0 + fm - 1) / fm : 0;
  }
  if (tid == nt - 1) cnt += (T - hi + fm - 1) / fm;
  int incl = cnt;
#pragma unroll
  for (int o = 1; o < WAVE32; o <<= 1) {
    const int v = __shfl_up(incl, o, WAVE32);
    if (lane >= o) incl += v;
  }
  if (lane == WAVE32 - 1) s_w[wave] = incl;
  __syncthreads();
  int base = 0, total = 0;
  for (int w = 0; w < nw; w++) {
    base += w < wave ? s_w[w] : 0;
    total += s_w[w];
  }
  int o = base + incl - cnt;
  uint4* ent = (uint4*)a.map + 1;
  auto put = [&](unsigned e, int r0, int r1) {
    if (o < cap) ent[o] = make_uint4(e, (unsigned)r0, (unsigned)r1, 0u);
    o++;
  };
  if (tid == 0)
    for (int r = 0; r < lo; r += fm) put(EXL3R4D_PF_ZERO, r, min(r + fm, lo));
  for (int e = e0; e < e1; e++) {
    const int a0 = clamp_row(p.off[e], T), a1 = clamp_row(p.off[e + 1], T);
    for (int r = a0; r < a1; r += fm) put((unsigned)e, r, min(r + fm, a1));
  }
  if (tid == nt - 1)
    for (int r = hi; r < T; r += fm) put(EXL3R4D_PF_ZERO, r, min(r + fm, T));
  if (tid == 0) a.map[0] = (unsigned)min(total, cap);
}

// EXL3R4D_PF_LPT (exl3r4d_prefill.h; gate_up's map form): the GEMM takes each launch's entries LONGEST FIRST --
// a.order[lo..hi) holds chunk [lo, hi)'s entry indices by 16-row blocks, most first (a counting sort, LDS atomics: the
// order inside a bucket is whatever the atomics gave; every entry computes the same rows whatever order the workgroups
// run in: the same bits).
// WHY: in expert order the grid mixes long and short entries, and a width-independent ~240 us of the tp2 M=2048 GEMM
// is per-launch ramp and the last round's tail of long workgroups (the measurements at the end of this file).
// LPT 1: entries by their own 16-row blocks. LPT 2: GROUPS (an expert's consecutive entries, or a zero range's) by the
// group's 16-row blocks (capped at 63), the group's entries kept together in map order -- an expert's entries share its
// weight tile, which a remainder entry then finds in cache.
// The copy ring for chunk c (xr_mode 5: order 1 + ring): the host-read groups at order positions [hs, hi), heads first
// in each group; a head's copy takes the next ring bytes while they fit (a prefix of the heads: offsets only grow).
// Every entry of [lo, hi) gets its xoff (0: read in place) and xhead; the flags start at 0; xlist / xcnt name the
// copies.
static __device__ void pf_xr_assign(const Exl3r4dPrefill& a, int c, int lo, int hs, int hi) {
  __shared__ unsigned s_u[1024], s_h[1024];
  __shared__ unsigned s_cu, s_ch;
  const int tid = threadIdx.x, nt = blockDim.x;
  const uint4* ent = (const uint4*)a.map + 1;
  uint32_t* const xt = xr_tab(a);
  uint32_t *const xoff = xt, *const xhead = xt + a.cap, *const xflag = xt + 2 * a.cap, *const xlist = xt + 3 * a.cap,
                  *const xcnt = xt + 4 * a.cap;
  const unsigned cap_u = (unsigned)(((long long)xr_ring_of(a) << 20) / 256);
  for (int i = lo + tid; i < hi; i += nt) xoff[i] = 0u, xhead[i] = (unsigned)i, xflag[i] = 0u;
  if (tid == 0) s_cu = 0u, s_ch = 0u;
  __syncthreads();
  for (int t0 = hs; t0 < hi; t0 += nt) {   // pass 1: the heads' ring offsets (two block scans: 256-byte units, heads)
    const int q = t0 + tid;
    int idx = -1;
    bool head = false;
    unsigned u = 0u;
    if (q < hi) {
      idx = (int)a.order[q];
      head = idx == lo || ent[idx - 1].x != ent[idx].x;
      if (head) {
        const Planes w = planes_of(a.g, (int)ent[idx].x);
        u = (unsigned)((((long long)w.rows * w.cols * xr_k2_of(a) / 16) + 255) / 256);
      }
    }
    s_u[tid] = u;
    s_h[tid] = head ? 1u : 0u;
    __syncthreads();
    for (int o = 1; o < nt; o <<= 1) {
      const unsigned vu = tid >= o ? s_u[tid - o] : 0u, vh = tid >= o ? s_h[tid - o] : 0u;
      __syncthreads();
      s_u[tid] += vu;
      s_h[tid] += vh;
      __syncthreads();
    }
    const unsigned ex_u = s_cu + s_u[tid] - u, ex_h = s_ch + s_h[tid] - (head ? 1u : 0u);
    if (head && ex_u + u <= cap_u) {
      xoff[idx] = ex_u + 1u;
      xlist[lo + (int)ex_h] = (unsigned)idx;
    }
    __syncthreads();
    if (tid == nt - 1) s_cu += s_u[tid], s_ch += s_h[tid];
    __syncthreads();
  }
  for (int t0 = hs; t0 < hi; t0 += nt) {   // pass 2: a group's other entries take their head's copy
    const int q = t0 + tid;
    if (q < hi) {
      const int idx = (int)a.order[q];
      if (!(idx == lo || ent[idx - 1].x != ent[idx].x)) {
        int h = idx;
        while (h > lo && ent[h - 1].x == ent[h].x) h--;
        xoff[idx] = xoff[h];
        xhead[idx] = (unsigned)h;
      }
    }
  }
  // heads copied: the count of heads whose xoff is set (a prefix, so the first unset one ends it)
  __shared__ unsigned s_n;
  if (tid == 0) s_n = 0u;
  __syncthreads();
  for (int t0 = hs; t0 < hi; t0 += nt) {
    const int q = t0 + tid;
    if (q < hi) {
      const int idx = (int)a.order[q];
      if ((idx == lo || ent[idx - 1].x != ent[idx].x) && xoff[idx]) atomicAdd(&s_n, 1u);
    }
  }
  __syncthreads();
  if (tid == 0) {
    xcnt[c] = s_n;
    XR_CNT(0, 1);
    XR_CNT(1, s_ch);
    XR_CNT(2, s_n);
    XR_CNT(3, s_n ? min(s_cu, cap_u) : 0u);
  }
  __syncthreads();
}

static __device__ void pf_order_build(const Exl3r4dPrefill& a) {
  constexpr int NB = 64;
  __shared__ int s_n[2][NB], s_o[2][NB];   // entries a bucket, then the bucket's next free position; [1]: host-read
  __shared__ int s_hs;                     // the chunk's first host-read position
  __shared__ int s_nv, s_nh;               // order 3: the chunk's VRAM-read / host-read entries
  const int E = (int)a.map[0], nch = a.chunk_n > 1 ? a.chunk_n : 1;
  const uint4* ent = (const uint4*)a.map + 1;
#if EXL3R4D_XR_ON
  const int mode = xr_mode_of(a) & 3, ring = (xr_mode_of(a) & 4) != 0 && xr_ring_of(a) > 0 && mode == 1 && nch <= 64;
  const long long f4 = 4 - ((xr_mode_of(a) >> 3) & 3);   // order 3's span: the host-read entries over f4 / 4 of the chunk
#else
  const int mode = 0, ring = 0;
  const long long f4 = 4;
#endif
  // ORDER 3 (interleave): each set longest first as in orders 1 / 2, then the two merged so the host-read entries sit
  // evenly over the first f4 / 4 of the chunk's positions -- VRAM-read entry i (of nv) at key (i + 1/2) / nv,
  // host-read entry j (of nh) at f4 / 4 x (j + 1/2) / nh, a tie host first; an entry's position = its rank in the
  // merge. A permutation of the same entries (each computes its rows alone, so the order changes no byte), so the
  // GEMM's workgroups that stream a host pool's codes run beside the resident ones instead of after them.
  auto merged = [&](int h, int q) -> int {
    const long long nv = s_nv, nh = s_nh;
    if (mode != 3 || !nv || !nh) return q;
    if (!h) {   // + host entries with key <= v_q: (2j + 1) nv f4 <= (2q + 1) nh 4
      const long long x = (2LL * q + 1) * nh * 4, r = x / (nv * f4);
      return q + (int)min(nh, (r + 1) / 2);
    }
    // + VRAM entries with key < h_q: (2i + 1) nh 4 < (2q + 1) nv f4
    const long long y = (2LL * q + 1) * nv * f4, r = (y - 1) / (nh * 4);
    return q + (int)min(nv, (r + 1) / 2);
  };
  auto blocks = [&](int i) { return ((int)(ent[i].z - ent[i].y) + 15) / 16; };
  // whether entry i's expert is read from a host pool (its codes in a pool the engine named); mode 0: never asked
  auto host = [&](int i) {
    return mode != 0 && ent[i].x != EXL3R4D_PF_ZERO && xr_host_codes(a, planes_of(a.g, (int)ent[i].x).codes) ? 1 : 0;
  };
  for (int c = 0; c < nch; c++) {
    const int lo = a.chunk_n > 1 ? (int)((const uint4*)a.ctab)[c].x : 0;
    const int hi = a.chunk_n > 1 ? (int)((const uint4*)a.ctab)[c].y : E;
    // group of entry i (LPT 2): it heads one when i == lo or the entry before is another expert's; n its entries
    auto group = [&](int i, int& n) {
      n = 1;
      if (EXL3R4D_PF_LPT != 2) return min(blocks(i), NB - 1);
      if (i > lo && ent[i - 1].x == ent[i].x) { n = 0; return 0; }
      int b = blocks(i);
      while (i + n < hi && ent[i + n].x == ent[i].x) b += blocks(i + n++);
      return min(b, NB - 1);
    };
    for (int t = threadIdx.x; t < 2 * NB; t += blockDim.x) s_n[t / NB][t % NB] = 0;
    __syncthreads();
    for (int i = lo + (int)threadIdx.x; i < hi; i += blockDim.x) {
      int n;
      const int b = group(i, n);
      if (n) {
        const int h = host(i);
        atomicAdd(&s_n[h][b], n);
        if (mode) {
          XR_CNT(7, 1);
          if (h) XR_CNT(6, 1);
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {   // set 0 (all of mode 0) first, or set 1 first (mode 2: host-read first)
      int o = lo;
      const int first = mode == 2 ? 1 : 0;
      for (int k = 0; k < 2; k++) {
        const int h = k ? 1 - first : first;
        if (k == 1) s_hs = o;
        if (mode == 3) o = 0;   // order 3: positions in each set from 0, merged below
        for (int b = NB - 1; b >= 0; b--) { s_o[h][b] = o; o += s_n[h][b]; }
        if (mode == 3) (h ? s_nh : s_nv) = o;
      }
    }
    __syncthreads();
    for (int i = lo + (int)threadIdx.x; i < hi; i += blockDim.x) {
      int n;
      const int b = group(i, n);
      if (!n) continue;
      const int h = host(i);
      const int o = atomicAdd(&s_o[h][b], n);
      if (mode == 3) {
        for (int j = 0; j < n; j++) a.order[lo + merged(h, o + j)] = (unsigned)(i + j);
      } else {
        for (int j = 0; j < n; j++) a.order[o + j] = (unsigned)(i + j);
      }
    }
    __syncthreads();
    if (ring) {
      __threadfence_block();
      pf_xr_assign(a, c, lo, s_hs, hi);
    }
  }
}

__global__ void __launch_bounds__(1024) exl3r4d_pf_map_kernel(Exl3r4dPrefill a) {
  EXL3R4D_KARG_PREFETCH();
  __shared__ int s_w[32];
  pf_map_build(a, s_w);
  if (a.chunk_n <= 1) {
    if (EXL3R4D_PF_LPT && a.order) {
      __threadfence_block();
      __syncthreads();
      pf_order_build(a);
    }
    return;
  }
  // THE CHUNK TABLE (the scratch cap): chunk c = the entries whose first row is in [c R, (c + 1) R) -- entries are in
  // row order (the zero head, the experts in id order, the zero tail), so a chunk is [lo, hi) of them, found by a
  // lower bound each end. Its rows start at its first entry's row (rbase) and end before rbase + R + fm.
  __threadfence_block();
  __syncthreads();
  const int E = (int)a.map[0], R = a.chunk_rows;
  const uint4* ent = (const uint4*)a.map + 1;
  auto first_at = [&](long long row) {   // the first entry whose first row is >= row
    int lo = 0, hi = E;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if ((long long)ent[mid].y < row) lo = mid + 1; else hi = mid;
    }
    return lo;
  };
  for (int c = threadIdx.x; c < a.chunk_n; c += blockDim.x) {
    const int lo = first_at((long long)c * R), hi = c == a.chunk_n - 1 ? E : first_at((long long)(c + 1) * R);
    ((uint4*)a.ctab)[c] = make_uint4((unsigned)lo, (unsigned)hi, lo < E ? ent[lo].y : 0u, 0u);
  }
  if (EXL3R4D_PF_LPT && a.order) {
    __threadfence_block();
    __syncthreads();
    pf_order_build(a);
  }
}

// Chunk c's entries [lo, hi) and the row its scratch rows start at (the chunk table, the scratch cap); one chunk: every
// entry, from row 0.
static __device__ __forceinline__ void pf_chunk(const Exl3r4dPrefill& a, int c, int& lo, int& hi, int& rbase) {
  if (a.chunk_n <= 1) {
    lo = 0, hi = (int)a.map[0], rbase = 0;
    return;
  }
  const uint4 t = ((const uint4*)a.ctab)[c];
  lo = (int)t.x, hi = (int)t.y, rbase = (int)t.z;
}

// ----------------------------------------------------------------------------- gate_up's epilogue, operands early
// EXL3R4D_PF_EPI_PRE: the shared header's epilogue_block<NPW, 2, PRE = true> (EPI 1, split-K 1, no EPIB / EPIC) copied
// statement for statement, with down's suh slice for the h_rot rotation passed in like svh (`sr`, loaded before the K
// loop) instead of a global load after the epilogue's barrier, which exposed two dependent round trips a call. Same
// helpers, same order of every sum: the same bits (checked by output hash against the shared path). Keep in step with
// epilogue_block (exl3r4d_common.hip.h).
#ifndef EXL3R4D_PF_EPI_PRE
#define EXL3R4D_PF_EPI_PRE 0
#endif
// wv / nwv: this wave's index in the waves that share `lds` and their count (the h_rot rows' split); a workgroup's
// whole wave set for AM 0 / 4, one column block's group of 8 for AM 5.
template <int NPW>
static __device__ __forceinline__ void pf_epi_gu(const Exl3r4dGemm& p, const Planes& w, const v8f (&acc)[NPW],
                                                 int col_local, int lane, int cb, int row, bool in_run, bool live,
                                                 float* lds, const v8h (*svp)[2], v2u sr, int wv, int nwv) {
  constexpr int MATS = 2, OPW = epi_opw<NPW, MATS>();
  constexpr float RS = 0.08838834764831845f;  // 1 / sqrt(128)
  const int h = lane >> 4;
  const bool writer = col_local * OPW < 8;
  float* xs = lds;  // [8 MATS tiles][32 lanes][8]
  const float sgn_h = h ? -1.f : 1.f;
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; e++) v[e] = acc[j][e];
    had16_lane(v, sgn_h);
    lds_put(xs + (col_local * NPW + j) * 256 + lane * 8, v);
  }
  lds_barrier();
  if (writer && in_run) {
#pragma unroll
    for (int o = 0; o < OPW; o++) {
      const int t = col_local * OPW + o;
      const int n0 = cb * 128 + 16 * t + 8 * h;
      float g[8], u[8], sg[8], su[8];
      had_tiles(xs, t, lane, g);
      had_tiles(xs + 8 * 256, t, lane, u);
#pragma unroll
      for (int e = 0; e < 8; e++) { sg[e] = (float)svp[o][0][e]; su[e] = (float)svp[o][1][e]; }
      union { _Float16 hv[8]; v4u u; } out;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const float gv0 = (g[e] * RS) * sg[e], uv0 = (u[e] * RS) * su[e];
        const float gv = p.limit > 0.f ? fminf(gv0, p.limit) : gv0;
        const float uv = p.limit > 0.f ? fminf(fmaxf(uv0, -p.limit), p.limit) : uv0;
        out.hv[e] = live ? (_Float16)((gv * (1.0f / (1.0f + expf(-gv)))) * uv) : (_Float16)0.f;
      }
      if (p.h_rot)
        *(v4u*)((_Float16*)((unsigned char*)lds + kHrotStage) + (lane & 15) * 128 + 16 * t + 8 * h) = out.u;
      else
        st_row<kStH>((_Float16*)p.y + (size_t)row * p.y_pitch + n0, out.u);
    }
  }
  if (p.h_rot) {
    lds_barrier();
    const _Float16* hst = (const _Float16*)((const unsigned char*)lds + kHrotStage);
    const int rc = row - (lane & 15);
    for (int i = wv; i < 16; i += nwv) {
      if (!__builtin_amdgcn_readlane((int)in_run, i)) continue;  // wave-uniform
      const bool li = __builtin_amdgcn_readlane((int)live, i);
      const v2u xr = *(const v2u*)(hst + i * 128 + 4 * lane);
      const v2u o = li ? rotate128_math(xr, sr, true, lane) : (v2u)(0u);
      st_row<kStH>((_Float16*)p.y + (size_t)(rc + i) * p.y_pitch + cb * 128 + 4 * lane, o);
    }
  }
  lds_barrier();  // xs (and, next pass, the rotated rows) are rewritten
}
constexpr int kGrpEpi = 16384 + 4096;   // bytes of LDS a group's epilogue takes (kHrotStage + the 16x128 h block)
// EXL3R4D_PF_EPI_DN2: down's epilogue (epilogue_block<NPW, 1, PRE = true>, EPI 2, split-K 1, no EPIB / EPIC)
// statement for statement, but over one of TWO LDS regions by the block's parity and without its trailing barrier:
// block i + 1 writes the other region, so the barrier that only kept block i's reads ahead of block i + 1's writes goes
// (down's GEMM spends 12.5% of its cycles at barriers in a thread trace); the caller ends with one. The same helpers
// and order of every sum: the same bits.
#ifndef EXL3R4D_PF_EPI_DN2
#define EXL3R4D_PF_EPI_DN2 0
#endif
template <int NPW>
static __device__ __forceinline__ void pf_epi_dn(const Exl3r4dGemm& p, const Planes& w, const v8f (&acc)[NPW],
                                                 int col_local, int lane, int cb, int row, bool in_run, bool live,
                                                 float* xs, const v8h (*svp)[1]) {
  constexpr int OPW = epi_opw<NPW, 1>();
  constexpr float RS = 0.08838834764831845f;  // 1 / sqrt(128)
  const int h = lane >> 4;
  const bool writer = col_local * OPW < 8;
  const float sgn_h = h ? -1.f : 1.f;
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; e++) v[e] = acc[j][e];
    had16_lane(v, sgn_h);
    lds_put(xs + (col_local * NPW + j) * 256 + lane * 8, v);
  }
  lds_barrier();
  if (writer && in_run) {
#pragma unroll
    for (int o = 0; o < OPW; o++) {
      const int t = col_local * OPW + o;
      const int n0 = cb * 128 + 16 * t + 8 * h;
      float r[8], sv[8];
      had_tiles(xs, t, lane, r);
#pragma unroll
      for (int e = 0; e < 8; e++) sv[e] = (float)svp[o][0][e];
      v4u out;
#pragma unroll
      for (int q = 0; q < 4; q++)
        out[q] = live ? (unsigned)f32_to_bf16((r[2 * q] * RS) * sv[2 * q]) |
                            ((unsigned)f32_to_bf16((r[2 * q + 1] * RS) * sv[2 * q + 1]) << 16)
                      : 0u;
      st_row<kStY>((unsigned short*)p.y + (size_t)row * p.y_pitch + n0, out);
    }
  }
}

// THE STREAMED y (down's map form, the kernels' YS form; prefill.c chooses it by y's size, EXL3R4D_PF_YNT_MB): down's
// epilogue (epilogue_block<NPW, 1, PRE = true>, EPI 2, split-K 1, no EPIB / EPIC) statement for statement, but the
// block's y goes through LDS and is stored a whole 256-byte row (128 bf16 columns) a wave instruction, as v3 EPI 1
// stores h, with store th code EXL3R4D_PF_YS_TH (1: nontemporal) -- the regular form stores 32 bytes of a row an
// instruction, 16 rows at once, each 128-byte line in four pieces from four waves, and nontemporal partial lines cost
// more than they save. Same values, same rows (a row not in the run is not written, a dropped row is zeros): the same
// bits. One more barrier a 16-row block (alone, with plain stores: +-0.4 us at tp2). WHY: a large y left dirty in L2
// slows the next op; the measurements at the end of this file.
#ifndef EXL3R4D_PF_YS_TH
#define EXL3R4D_PF_YS_TH 1
#endif
constexpr int kYStage = 8 * 256 * 4;   // bytes: after the epilogue's 8 tiles x 32 lanes x 8 f32 (MATS 1)
template <int NPW>
static __device__ __forceinline__ void pf_epi_dn_ys(const Exl3r4dGemm& p, const v8f (&acc)[NPW], int col_local,
                                                    int lane, int cb, int row, bool in_run, bool live, float* lds,
                                                    const v8h (*svp)[1]) {
  constexpr int OPW = epi_opw<NPW, 1>();
  constexpr float RS = 0.08838834764831845f;  // 1 / sqrt(128)
  const int h = lane >> 4;
  const bool writer = col_local * OPW < 8;
  const float sgn_h = h ? -1.f : 1.f;
  float* xs = lds;
  unsigned short* yst = (unsigned short*)((unsigned char*)lds + kYStage);   // [16 rows][128] bf16
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; e++) v[e] = acc[j][e];
    had16_lane(v, sgn_h);
    lds_put(xs + (col_local * NPW + j) * 256 + lane * 8, v);
  }
  lds_barrier();
  if (writer && in_run) {
#pragma unroll
    for (int o = 0; o < OPW; o++) {
      const int t = col_local * OPW + o;
      float r[8], sv[8];
      had_tiles(xs, t, lane, r);
#pragma unroll
      for (int e = 0; e < 8; e++) sv[e] = (float)svp[o][0][e];
      v4u out;
#pragma unroll
      for (int q = 0; q < 4; q++)
        out[q] = live ? (unsigned)f32_to_bf16((r[2 * q] * RS) * sv[2 * q]) |
                            ((unsigned)f32_to_bf16((r[2 * q + 1] * RS) * sv[2 * q + 1]) << 16)
                      : 0u;
      *(v4u*)(yst + (lane & 15) * 128 + 16 * t + 8 * h) = out;
    }
  }
  lds_barrier();
  const int nwv = (int)blockDim.x / WAVE32, wv = __builtin_amdgcn_readfirstlane((int)(threadIdx.x / WAVE32));
  const int rc = row - (lane & 15);
  for (int i = wv; i < 16; i += nwv) {
    if (!__builtin_amdgcn_readlane((int)in_run, i)) continue;  // wave-uniform
    st_th<EXL3R4D_PF_YS_TH>((unsigned short*)p.y + (size_t)(rc + i) * p.y_pitch + cb * 128 + 4 * lane,
                            *(const v2u*)(yst + i * 128 + 4 * lane));
  }
  lds_barrier();  // xs and the stage are rewritten by the next block
}

// Down's suh slice of column block cb for the h_rot rotation (zeros without h_rot: never read then).
static __device__ __forceinline__ v2u pf_dn_suh(const Exl3r4dGemm& p, const Planes& w, int cb, int lane) {
  typedef const __attribute__((address_space(1))) v2u* gv2;
  return p.h_rot && w.dn_suh ? *(gv2)((const unsigned char*)w.dn_suh + (size_t)(cb * 128 + 4 * lane) * 2) : (v2u)(0u);
}

// ----------------------------------------------------------------------------- the MTX GEMM

// One map entry: rows [r0, r1) of expert w, MT = ceil((r1 - r0) / 16) blocks, in column block cb = blockIdx.x of every
// matrix. WV = 8 MATS / NPW waves, wave w owns local tiles [w NPW, w NPW + NPW) of the block's 8 MATS tiles (matrix
// w NPW / 8). The A fragment of block i, lane l is row r0 + 16 i + l % 16 of the prep's rows (clamped into the entry;
// rows past it are computed and not stored).
template <int BITS, int MT, int NPW, int MATS, int AM, bool YS = false>
static __device__ __forceinline__ void pf_body(const Exl3r4dGemm& p, const Planes& w, int r0, int r1, float* lds,
                                               int rbase) {
  using P = Plan<BITS>;   // any K the shared decode serves (2..6 at mul1); the buffer form below is K = 4's
  constexpr int NW = P::NW, WORDS = P::WORDS;
  const int tid = threadIdx.x, lane = tid % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);  // keep tile bases in SGPRs
  const int h = lane >> 4, nl = lane & 15;
  const int width = MATS == 2 ? w.rows / 2 : w.rows;
  const int ntiles = width / 16, ksteps = w.cols / 16;
  const int cb = blockIdx.x;
  const int vl = wave * NPW;
  const int mat = vl / 8, nt0 = cb * 8 + vl % 8;

  // THE A OPERAND, three arms (AM):
  //   0  every wave loads its own fragments of the prep's rows from global memory each K step (the four waves of a
  //      matrix load the SAME rows -- ~4.9 GB of L2 traffic at gate_up tp2 M=2048, but L2 hits: staging them, AM 2,
  //      does not pay);
  //   1  the entry's rows rotated into LDS a 128-wide K block at a time (no prep kernel) -- measured slower;
  //   2  the prep's rows of a 128-wide K block copied ONCE a workgroup into LDS, 16 B a thread-chunk, loaded into
  //      registers one pass ahead (libr4d's staging pattern) -- every wave reads its fragments from LDS.
  // LDS rows are [row][mat][128 + 8] f16: the 8 halfs of padding put the 16 rows a fragment read touches in 16
  // different bank groups. `aoff` is the lane's offset in halfs.
  unsigned avo[MT];
  const _Float16* alds = (const _Float16*)lds;
  constexpr int APITCH = MATS * 128 + 8;   // halfs a row
  unsigned aoff = 0;
  gu8 abase = nullptr;
  if constexpr (AM != 0) {
    aoff = (unsigned)(nl * APITCH + mat * 128 + 4 * h);
  }
  if constexpr (AM == 2) {
    abase = (gu8)(const unsigned char*)p.xr;
  }
  if constexpr (AM == 0) {
#pragma unroll
    for (int i = 0; i < MT; i++) {
      const int row = r0 + 16 * i + nl;
      avo[i] = (unsigned)(((size_t)((row < r1 ? row : r1 - 1) - rbase) * p.xr_pitch + 4 * h) * 2);
    }
    abase = (gu8)(const unsigned char*)((const _Float16*)p.xr + (size_t)mat * w.cols);
  }

  const P plan(lane);
  unsigned wvo[2][NW];
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int x = 0; x < NW; x++) wvo[r][x] = 4u * (unsigned)plan.woff[r][x];
  // The record: matrix mat, tile (tk, tn) at (mat * ksteps * ntiles + tk * ntiles + tn) * WORDS u32 (layout.c).
  gu8 wt[NPW];
  const unsigned* wexp = w.codes + (size_t)mat * ksteps * ntiles * WORDS;
#pragma unroll
  for (int j = 0; j < NPW; j++) wt[j] = (gu8)(const unsigned char*)(wexp + (size_t)(nt0 + j) * WORDS);
  const size_t kbytes = (size_t)ntiles * WORDS * 4;

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);
  // EXL3R4D_PF_SVH_EARLY: the epilogue's svh in registers from the prologue, not a global round trip after the
  // epilogue's barrier in every 16-row block.
  constexpr bool SVE = MATS == 2 ? EXL3R4D_PF_SVH_EARLY_GU : EXL3R4D_PF_SVH_EARLY_DN;
  [[maybe_unused]] v8h svp[epi_opw<NPW, MATS>()][MATS];
  [[maybe_unused]] v2u dsr = {0u, 0u};
  if constexpr (SVE || EXL3R4D_PF_EPI_PRE || ((EXL3R4D_PF_EPI_DN2 || YS) && MATS == 1))
    svh_early<NPW, MATS>(w, wave, lane, cb, svp);
  if constexpr (EXL3R4D_PF_EPI_PRE && MATS == 2) dsr = pf_dn_suh(p, w, cb, lane);

  constexpr int PF = EXL3R4D_PF_STAGES, PW = MATS == 2 ? EXL3R4D_PF_WSTAGES_GU : EXL3R4D_PF_WSTAGES_DN;
  static_assert(PW >= PF && PW % PF == 0, "the weights' depth is a multiple of the activations'");
  unsigned wb[PW][NPW][2][NW];
  v8h ab[PF][MT];
  // Buffer form (EXL3R4D_PF_BUF): at K = 4 a lane's two runs touch three words of a tile -- woff[1][x] = woff[0][x] +
  // 1 -- so W[0] = {w-1, w0}, W[1] = {w0, w+1} (as in exl3r4d.hip). Descriptor word 3 0x31004000: gfx11/12 raw.
  // Every K takes the buffer form (the decode band's EXL3R4D_KX_BUF pattern): K = 4 three words a tile (its two runs
  // share the middle one), any other K four (each run's two), lane offsets fixed, the k-step in an SGPR -- the global
  // form's 64-bit address VALU a tile is what made K != 4 slow, with the shared header's packed-pair hash
  // (EXL3R4D_KX_PAIR).
  constexpr bool WBUF = EXL3R4D_PF_BUF && NW == 2 && (BITS == 4 || EXL3R4D_PF_KX_BUF);
  [[maybe_unused]] const __amdgpu_buffer_rsrc_t wrs =
      __builtin_amdgcn_make_buffer_rsrc((void*)wexp, (short)0, 0x7fffffff, 0x31004000);
  [[maybe_unused]] const __amdgpu_buffer_rsrc_t ars =
      __builtin_amdgcn_make_buffer_rsrc((void*)abase, (short)0, 0x7fffffff, 0x31004000);
  [[maybe_unused]] const unsigned bso = (unsigned)nt0 * WORDS * 4;
  [[maybe_unused]] const unsigned bwm = wvo[0][0], bw0 = wvo[0][1], bwp = wvo[1][NW - 1];
  constexpr int kAux = EXL3R4D_PF_NT ? 1 : 0;   // TH_LOAD_NT, as __builtin_nontemporal_load

#define PF_WLOAD(ks, W_)                                                                             \
  do {                                                                                               \
    if constexpr (WBUF) {                                                \
      const unsigned so_ = bso + (unsigned)(ks) * (unsigned)kbytes;                                  \
      _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                              \
        if constexpr (BITS == 4) {                                                                   \
          const unsigned m_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwm + j * 128, so_, kAux);   \
          const unsigned z_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bw0 + j * 128, so_, kAux);   \
          const unsigned q_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwp + j * 128, so_, kAux);   \
          W_[j][0][0] = m_; W_[j][0][1] = z_; W_[j][1][0] = z_; W_[j][1][1] = q_;                   \
        } else {                                                                                     \
          _Pragma("unroll") for (int r = 0; r < 2; r++)                                              \
            _Pragma("unroll") for (int x = 0; x < NW; x++)                                           \
              W_[j][r][x] = __builtin_amdgcn_raw_buffer_load_b32(wrs, wvo[r][x] + j * WORDS * 4, so_, kAux); \
        }                                                                                            \
      }                                                                                              \
      break;                                                                                         \
    }                                                                                                \
    _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                                \
      const gu8 tb_ = wt[j] + (size_t)(ks) * kbytes;                                                 \
      _Pragma("unroll") for (int r = 0; r < 2; r++)                                                  \
        _Pragma("unroll") for (int x = 0; x < NW; x++)                                               \
          W_[j][r][x] = EXL3R4D_PF_NT ? __builtin_nontemporal_load((gu32)(tb_ + wvo[r][x]))          \
                                      : *(gu32)(tb_ + wvo[r][x]);                                    \
    }                                                                                                \
  } while (0)
#define PF_ALOAD(ks, A_)                                                                             \
  do {                                                                                               \
    if constexpr (EXL3R4D_PF_BUF) {                                                                  \
      _Pragma("unroll") for (int i = 0; i < MT; i++) {                                               \
        union { v2u w[2]; v8h f; } x_;                                                               \
        x_.w[0] = __builtin_amdgcn_raw_buffer_load_b64(ars, avo[i], (unsigned)(ks) * 32u, EXL3R4D_PF_ATH); \
        x_.w[1] = __builtin_amdgcn_raw_buffer_load_b64(ars, avo[i] + 16u, (unsigned)(ks) * 32u, EXL3R4D_PF_ATH); \
        A_[i] = x_.f;                                                                                \
      }                                                                                              \
    } else {                                                                                         \
      const gu8 ab_ = abase + (size_t)(ks) * 32;                                                     \
      _Pragma("unroll") for (int i = 0; i < MT; i++) A_[i] = frag_a8(ab_, avo[i]);                   \
    }                                                                                                \
  } while (0)
#define PF_LOAD(ks, W_, A_) \
  do {                      \
    PF_WLOAD(ks, W_);       \
    PF_ALOAD(ks, A_);       \
  } while (0)
#define PF_COMPUTE(W_, A_)                                                                           \
  do {                                                                                               \
    _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                                \
      const v8h wf = decode_frag<BITS, NW>(W_[j], plan.shift);                                          \
      _Pragma("unroll") for (int i = 0; i < MT; i++)                                                 \
        acc[j][i] = mma(wf, A_[i], acc[j][i]);                                                       \
    }                                                                                                \
    if constexpr (EXL3R4D_PF_SGB == 1) {   /* the step's loads, its decode, its WMMAs (like EXL3R4D_SGB) */   \
      __builtin_amdgcn_sched_group_barrier(0x020, 3 * NPW + 2 * MT, 0);                              \
      __builtin_amdgcn_sched_group_barrier(0x002, 44 * NPW, 0);                                      \
      __builtin_amdgcn_sched_group_barrier(0x008, NPW * MT, 0);                                      \
    } else if constexpr (EXL3R4D_PF_SGB == 2) {   /* a tile at a time: its loads, decode, WMMAs */      \
      _Pragma("unroll") for (int j_ = 0; j_ < NPW; j_++) {                                           \
        __builtin_amdgcn_sched_group_barrier(0x020, 3 + 2 * MT / NPW, 0);                            \
        __builtin_amdgcn_sched_group_barrier(0x002, 44, 0);                                          \
        __builtin_amdgcn_sched_group_barrier(0x008, MT, 0);                                          \
      }                                                                                              \
    }                                                                                                \
  } while (0)

  if constexpr (AM == 0 && PW == PF) {
    // Unconditional (clamped) loads, PF stages in flight: the decode band's loop. With a branch around the prefetch
    // the compiler cannot count the loads in flight on the join and waits for all of them before the next WMMA.
#pragma unroll
    for (int u = 0; u < PF - 1; u++) PF_LOAD(u < ksteps ? u : ksteps - 1, wb[u], ab[u]);
    int ks = 0;
    for (; ks + PF <= ksteps; ks += PF) {
#pragma unroll
      for (int u = 0; u < PF; u++) {
        const int ls = ks + u + PF - 1 < ksteps ? ks + u + PF - 1 : ksteps - 1;
        PF_LOAD(ls, wb[(u + PF - 1) % PF], ab[(u + PF - 1) % PF]);
        PF_COMPUTE(wb[u], ab[u]);
      }
    }
#pragma unroll
    for (int u = 0; u < PF - 1; u++)
      if (ks + u < ksteps) PF_COMPUTE(wb[u], ab[u]);
  } else if constexpr (AM == 0) {
    // EXL3R4D_PF_WSTAGES > PF: the weights PW k-steps ahead, the activations PF (a thread trace shows the K loop
    // waiting on loads issued one step earlier -- too few weight bytes in flight a WGP to cover DRAM's loaded latency).
    // Steps past the loop's last whole PW run (ksteps % PW of them) take the same body under a uniform guard.
#pragma unroll
    for (int u = 0; u < PW - 1; u++) PF_WLOAD(u < ksteps ? u : ksteps - 1, wb[u]);
#pragma unroll
    for (int u = 0; u < PF - 1; u++) PF_ALOAD(u < ksteps ? u : ksteps - 1, ab[u]);
    auto step = [&](int ks, int u) __attribute__((always_inline)) {
      const int lw = ks + u + PW - 1 < ksteps ? ks + u + PW - 1 : ksteps - 1;
      const int la = ks + u + PF - 1 < ksteps ? ks + u + PF - 1 : ksteps - 1;
      PF_WLOAD(lw, wb[(u + PW - 1) % PW]);
      PF_ALOAD(la, ab[(u + PF - 1) % PF]);
      PF_COMPUTE(wb[u % PW], ab[u % PF]);
    };
    int ks = 0;
    for (; ks + PW <= ksteps; ks += PW) {
#pragma unroll
      for (int u = 0; u < PW; u++) step(ks, u);
    }
#pragma unroll
    for (int u = 0; u < PW - 1; u++)
      if (ks + u < ksteps) step(ks, u);
  } else if constexpr (AM == 3) {
    // THE ROTATION IN THE K LOOP, PIPELINED (AM 3: AM 1 done the way libr4d stages its operands): wave w rotates the
    // (row, matrix) pairs m = w % MATS, rows i = w / MATS + S j (S = WV / MATS) of every 128-wide K block -- RPW rows a
    // wave, all of one matrix, so its suh slice is one load a pass. The NEXT block's x and suh are loaded into
    // registers right after this block's rotation consumed them, so their round trip runs under the block's 8 K steps
    // (the first arm, AM 1, loaded them task by task between two barriers: +5..+64%). Buffer loads: the row's x base
    // and the pass's K offset are uniform (SGPR), the lane's 8 bytes a fixed voffset.
    constexpr int WV = 8 * MATS / NPW, S = WV / MATS, RPW = MT * 16 / S;
    static_assert(WV % MATS == 0 && (MT * 16) % S == 0, "rotation tasks divide over the waves");
    const int rm = wave % MATS, ri0 = wave / MATS, nkb = ksteps / 8, nrows = r1 - r0;
    const __amdgpu_buffer_rsrc_t xrs = __builtin_amdgcn_make_buffer_rsrc((void*)p.x, (short)0, 0x7fffffff, 0x31004000);
    const __amdgpu_buffer_rsrc_t srs =
        __builtin_amdgcn_make_buffer_rsrc((void*)(rm ? w.suh1 : w.suh), (short)0, 0x7fffffff, 0x31004000);
    unsigned xoff[RPW];   // the task's x row base in bytes (uniform): gate_up the token's row, down the sorted row
    bool rlive[RPW];
#pragma unroll
    for (int j = 0; j < RPW; j++) {
      const int i = ri0 + S * j, r = r0 + (i < nrows ? i : nrows - 1);
      const int t = __builtin_amdgcn_readfirstlane(p.sorted[r]);
      const int xrow = MATS == 2 ? (t >= 0 ? min(t / p.top_k, p.n_tok - 1) : 0) : r;
      xoff[j] = (unsigned)((size_t)xrow * p.x_pitch * 2);
      rlive[j] = t >= 0;
    }
    v2u xv[RPW], sv;
#define PF_XLOAD(kb_)                                                                                \
  do {                                                                                               \
    const unsigned ko_ = (unsigned)(kb_) * 256u;                                                     \
    _Pragma("unroll") for (int j = 0; j < RPW; j++)                                                  \
      xv[j] = __builtin_bit_cast(v2u, __builtin_amdgcn_raw_buffer_load_b64(xrs, (unsigned)lane * 8u, \
                                                                            xoff[j] + ko_, 0));     \
    sv = __builtin_bit_cast(v2u, __builtin_amdgcn_raw_buffer_load_b64(srs, (unsigned)lane * 8u, ko_, 0)); \
  } while (0)
    PF_XLOAD(0);
    PF_WLOAD(0, wb[0]);
    for (int kb = 0; kb < nkb; kb++) {
      if (kb) lds_barrier();   // every wave is done reading block kb - 1
#pragma unroll
      for (int j = 0; j < RPW; j++) {
        const v2u o = !rlive[j] ? (v2u)(0u) : rotate128_math(xv[j], sv, p.x_bf16 == 0, lane);
        *(v2u*)((_Float16*)lds + (ri0 + S * j) * APITCH + rm * 128 + 4 * lane) = o;
      }
      PF_XLOAD(kb + 1 < nkb ? kb + 1 : kb);
      lds_barrier();
#pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        const int ks = kb * 8 + kk, ln = ks + 1 < ksteps ? ks + 1 : ksteps - 1;
        PF_WLOAD(ln, wb[(kk + 1) % 2]);
        v8h af[MT];
#pragma unroll
        for (int i = 0; i < MT; i++) af[i] = frag_lds(alds + aoff + i * 16 * APITCH + kk * 16);
        PF_COMPUTE(wb[kk % 2], af);
      }
    }
#undef PF_XLOAD
    lds_barrier();   // the epilogue reuses the LDS
  } else if constexpr (AM == 2) {
    // A pass = one 128-wide K block = 8 K steps. Thread t owns chunks c = t + q NTHR of the pass's MT*16 rows x MATS
    // matrices x 16 chunks of 16 B; their global offsets (row clamped into the entry) and LDS offsets are fixed for the
    // whole loop, the pass adds 256 bytes. The stage registers for pass kb + 1 are loaded right after pass kb's are
    // stored, so the one wait for them lands at the next pass head (clamped, unconditional: a branch would make the
    // compiler wait for every load in flight at the join).
    constexpr int NTHR = 8 * MATS / NPW * WAVE32;
    constexpr int CH = MT * 16 * MATS * 16, CPT = CH / NTHR;
    static_assert(CH % NTHR == 0, "a pass's chunks divide over the workgroup");
    const int tid_ = threadIdx.x;
    unsigned goff[CPT], loff[CPT];
#pragma unroll
    for (int q = 0; q < CPT; q++) {
      const int c = tid_ + q * NTHR, i = c / (MATS * 16), m = (c / 16) % MATS, part = c % 16;
      const int row = r0 + i < r1 ? r0 + i : r1 - 1;
      goff[q] = (unsigned)(((size_t)(row - rbase) * p.xr_pitch + (size_t)m * w.cols + part * 8) * 2);
      loff[q] = (unsigned)(i * APITCH + m * 128 + part * 8);
    }
    const int nkb = ksteps / 8;
    uint4 stg[CPT];
#define PF_SLOAD(kb_)                                                                                \
  do {                                                                                               \
    const gu8 sb_ = abase + (size_t)(kb_) * 256;                                                     \
    _Pragma("unroll") for (int q = 0; q < CPT; q++) {                                                \
      const v4u v_ = *(__attribute__((address_space(1))) const v4u*)(sb_ + goff[q]);                 \
      stg[q] = make_uint4(v_[0], v_[1], v_[2], v_[3]);                                               \
    }                                                                                                \
  } while (0)
    PF_SLOAD(0);
    PF_WLOAD(0, wb[0]);
    for (int kb = 0; kb < nkb; kb++) {
      if (kb) lds_barrier();   // every wave is done reading pass kb - 1
#pragma unroll
      for (int q = 0; q < CPT; q++) *(uint4*)((_Float16*)lds + loff[q]) = stg[q];
      PF_SLOAD(kb + 1 < nkb ? kb + 1 : kb);
      lds_barrier();
#pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        const int ks = kb * 8 + kk, ln = ks + 1 < ksteps ? ks + 1 : ksteps - 1;
        PF_WLOAD(ln, wb[(kk + 1) % 2]);
        v8h af[MT];
#pragma unroll
        for (int i = 0; i < MT; i++) af[i] = frag_lds(alds + aoff + i * 16 * APITCH + kk * 16);
        PF_COMPUTE(wb[kk % 2], af);
      }
    }
#undef PF_SLOAD
    lds_barrier();   // the epilogue reuses the LDS
  } else {
    // One 128-wide K block at a time: every wave rotates a share of the entry's (row, matrix) pairs into LDS, a
    // barrier, then the block's 8 K steps read their A fragments from LDS. The weight words keep streaming across the
    // block boundary (double-buffered, clamped), so the barrier waits on the rotation, not on memory.
    const int nrows = r1 - r0, nw = (int)blockDim.x / WAVE32, tasks = nrows * MATS;
    PF_WLOAD(0, wb[0]);
    for (int kb = 0; kb < ksteps / 8; kb++) {
      if (kb) lds_barrier();   // every wave is done reading block kb - 1
      for (int t = wave; t < tasks; t += nw) {
        const int i = t / MATS, m = t - i * MATS;
        const v2u o = prep_rotate<MATS>(p, w, r0 + i, m, kb, lane);
        *(v2u*)((_Float16*)lds + i * APITCH + m * 128 + 4 * lane) = o;
      }
      lds_barrier();
#pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        const int ks = kb * 8 + kk, ln = ks + 1 < ksteps ? ks + 1 : ksteps - 1;
        PF_WLOAD(ln, wb[(kk + 1) % 2]);
        v8h af[MT];
#pragma unroll
        for (int i = 0; i < MT; i++) af[i] = frag_lds(alds + aoff + i * 16 * APITCH + kk * 16);
        PF_COMPUTE(wb[kk % 2], af);
      }
    }
    lds_barrier();   // the epilogue reuses the LDS
  }
#undef PF_WLOAD
#undef PF_LOAD
#undef PF_ALOAD
#undef PF_COMPUTE

  // The epilogue, a 16-row block at a time (two barriers each; the LDS holds one block's 8 MATS tiles).
#pragma unroll
  for (int i = 0; i < MT; i++) {
    const int row = r0 + 16 * i + nl;
    const bool in_run = row < r1;
    const int tok = p.sorted[in_run ? row : r0];   // unconditional, clamped into the entry
    v8f blk[NPW];
#pragma unroll
    for (int j = 0; j < NPW; j++) blk[j] = acc[j][i];
    if constexpr (YS && MATS == 1)
      pf_epi_dn_ys<NPW>(p, blk, wave, lane, cb, row, in_run, in_run && tok >= 0, lds, svp);
    else if constexpr (EXL3R4D_PF_EPI_DN2 && MATS == 1)
      pf_epi_dn<NPW>(p, w, blk, wave, lane, cb, row, in_run, in_run && tok >= 0, lds + (i & 1) * 2048, svp);
    else if constexpr (EXL3R4D_PF_EPI_PRE && MATS == 2)
      pf_epi_gu<NPW>(p, w, blk, wave, lane, cb, row, in_run, in_run && tok >= 0, lds, svp, dsr, wave,
                     (int)blockDim.x / WAVE32);
    else if constexpr (SVE || EXL3R4D_PF_EPI_PRE)
      epilogue_block<NPW, MATS, true>(p, w, blk, wave, 0, lane, cb, row, in_run, in_run && tok >= 0, lds, svp);
    else
      epilogue_block<NPW, MATS>(p, w, blk, wave, 0, lane, cb, row, in_run, in_run && tok >= 0, lds);
  }
  if constexpr (EXL3R4D_PF_EPI_DN2 && !YS && MATS == 1) lds_barrier();   // the regions are rewritten next pass
}

// DOWN, G COLUMN BLOCKS A WORKGROUP (EXL3R4D_PF_DN_G, measured slower: +0.3..+4%, kept as an arm). down's K is short
// (16 / 24 k-steps at tp2) and its GEMM is memory-latency bound (load waits 56% in a thread trace): every (entry, cb)
// workgroup pays its startup chain and its pipeline fill. Here a workgroup takes G consecutive column blocks of one
// entry and the weight / activation ring runs on across them: the next block's first loads are issued during this
// block's last steps and stay in flight through its epilogue. Each tile's sum is the regular one's (one wave, K in
// order), so the bits are the same.
#ifndef EXL3R4D_PF_DN_G
#define EXL3R4D_PF_DN_G 1
#endif
template <int BITS, int MT, int G>
static __device__ __forceinline__ void pf_body_dg(const Exl3r4dGemm& p, const Planes& w, int r0, int r1, float* lds,
                                                  int rbase) {
  using P = Plan<BITS>;
  constexpr int NW = P::NW, WORDS = P::WORDS, NPW = 2, PF = 2;
  static_assert(NW == 2, "the buffer form");
  const int tid = threadIdx.x, lane = tid % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);
  const int h = lane >> 4, nl = lane & 15;
  const int width = w.rows, ntiles = width / 16, ksteps = w.cols / 16;
  const int vl = wave * NPW, cb0 = (int)blockIdx.x * G;
  const int gv = min(G, max(0, width / 128 - cb0));   // valid column blocks of this group (the narrower class)
  unsigned avo[MT];
#pragma unroll
  for (int i = 0; i < MT; i++) {
    const int row = r0 + 16 * i + nl;
    avo[i] = (unsigned)(((size_t)((row < r1 ? row : r1 - 1) - rbase) * p.xr_pitch + 4 * h) * 2);
  }
  const __amdgpu_buffer_rsrc_t ars = __builtin_amdgcn_make_buffer_rsrc((void*)p.xr, (short)0, 0x7fffffff, 0x31004000);
  const P plan(lane);
  unsigned wvo[2][NW];
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int x = 0; x < NW; x++) wvo[r][x] = 4u * (unsigned)plan.woff[r][x];
  const __amdgpu_buffer_rsrc_t wrs = __builtin_amdgcn_make_buffer_rsrc((void*)w.codes, (short)0, 0x7fffffff, 0x31004000);
  const unsigned kbytes = (unsigned)ntiles * WORDS * 4;
  const unsigned bwm = wvo[0][0], bw0 = wvo[0][1], bwp = wvo[1][NW - 1];
  constexpr int kAux = EXL3R4D_PF_NT ? 1 : 0;
  unsigned wb[PF][NPW][2][NW];
  v8h ab[PF][MT];
  // Step (g, ks) of the group's concatenated stream: weights of column block cb0 + g, activations of k-step ks.
#define DG_LOAD(g_, ks_, W_, A_)                                                                     \
  do {                                                                                               \
    const unsigned so_ = (unsigned)((cb0 + (g_)) * 8 + vl) * WORDS * 4 + (unsigned)(ks_) * kbytes;   \
    _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                                \
      if constexpr (BITS == 4) {                                                                     \
        const unsigned m_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwm + j * 128, so_, kAux);     \
        const unsigned z_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bw0 + j * 128, so_, kAux);     \
        const unsigned q_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwp + j * 128, so_, kAux);     \
        W_[j][0][0] = m_; W_[j][0][1] = z_; W_[j][1][0] = z_; W_[j][1][1] = q_;                     \
      } else {                                                                                       \
        _Pragma("unroll") for (int r = 0; r < 2; r++)                                                \
          _Pragma("unroll") for (int x = 0; x < NW; x++)                                             \
            W_[j][r][x] = __builtin_amdgcn_raw_buffer_load_b32(wrs, wvo[r][x] + j * WORDS * 4, so_, kAux); \
      }                                                                                              \
    }                                                                                                \
    _Pragma("unroll") for (int i = 0; i < MT; i++) {                                                 \
      union { v2u w[2]; v8h f; } x_;                                                                 \
      x_.w[0] = __builtin_amdgcn_raw_buffer_load_b64(ars, avo[i], (unsigned)(ks_) * 32u, 0);         \
      x_.w[1] = __builtin_amdgcn_raw_buffer_load_b64(ars, avo[i] + 16u, (unsigned)(ks_) * 32u, 0);   \
      A_[i] = x_.f;                                                                                  \
    }                                                                                                \
  } while (0)
  if (gv > 0) DG_LOAD(0, 0, wb[0], ab[0]);   // PF - 1 = 1 step ahead
  for (int g = 0; g < gv; g++) {
    const int cb = cb0 + g;
    v8h svp[epi_opw<NPW, 1>()][1];
    svh_early<NPW, 1>(w, wave, lane, cb, svp);
    v8f acc[NPW][MT];
#pragma unroll
    for (int j = 0; j < NPW; j++)
#pragma unroll
      for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);
    for (int ks = 0; ks + PF <= ksteps; ks += PF) {   // ksteps even (the launch checks)
#pragma unroll
      for (int u = 0; u < PF; u++) {
        const int s = ks + u + 1;   // the step loaded: this block's next, else the next block's first
        const bool nxt = s >= ksteps, more = g + 1 < gv;
        const int lg = nxt && more ? g + 1 : g, lk = nxt ? (more ? s - ksteps : ksteps - 1) : s;
        DG_LOAD(lg, lk, wb[(u + 1) % PF], ab[(u + 1) % PF]);
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const v8h wf = decode_frag<BITS, NW>(wb[u][j], plan.shift);
#pragma unroll
          for (int i = 0; i < MT; i++) acc[j][i] = mma(wf, ab[u][i], acc[j][i]);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < MT; i++) {
      const int row = r0 + 16 * i + nl;
      const bool in_run = row < r1;
      const int tok = p.sorted[in_run ? row : r0];
      v8f blk[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++) blk[j] = acc[j][i];
      epilogue_block<NPW, 1, true>(p, w, blk, wave, 0, lane, cb, row, in_run, in_run && tok >= 0, lds, svp);
    }
  }
#undef DG_LOAD
  const int ncol = (int)(p.y_cols / 128);
  for (int g = gv; g < G && cb0 + g < ncol; g++) zero_rows(p, r0, r1, cb0 + g);   // the narrower class's zeros
}
template <int BITS, int M, int MTX, int G>
static __device__ __forceinline__ void pf_dg_dispatch(int mt, const Exl3r4dGemm& p, const Planes& w, int r0, int r1,
                                                      float* lds, int rbase) {
  if constexpr (M == MTX) {
    pf_body_dg<BITS, M, G>(p, w, r0, r1, lds, rbase);
  } else {
    if (mt <= M)
      pf_body_dg<BITS, M, G>(p, w, r0, r1, lds, rbase);
    else
      pf_dg_dispatch<BITS, M + 1, MTX, G>(mt, p, w, r0, r1, lds, rbase);
  }
}

// mt (1..MTX, workgroup-uniform) -> the body compiled for exactly that many blocks.
template <int BITS, int M, int MTX, int NPW, int MATS, int AM, bool YS = false>
static __device__ __forceinline__ void pf_dispatch(int mt, const Exl3r4dGemm& p, const Planes& w, int r0, int r1,
                                                   float* lds, int rbase) {
  if constexpr (M == MTX) {
    pf_body<BITS, M, NPW, MATS, AM, YS>(p, w, r0, r1, lds, rbase);
  } else {
    if (mt <= M)
      pf_body<BITS, M, NPW, MATS, AM, YS>(p, w, r0, r1, lds, rbase);
    else
      pf_dispatch<BITS, M + 1, MTX, NPW, MATS, AM, YS>(mt, p, w, r0, r1, lds, rbase);
  }
}

// One workgroup = (128-column block blockIdx.x, blockIdx.y): the map form (RUN 0) -- map entry blockIdx.y; the run form
// (RUN 1) -- the blockIdx.y-th live expert (the decode band's find_run, a scan of expert_offset; the workgroups past
// the live count zero the rows no run owns, its zero_dead_rows), its whole run in passes of 16 MTX rows.
// EXL3R4D_PF_WPE (an arm): the GEMM's minimum waves a SIMD (amdgpu_waves_per_eu), i.e. a VGPR cap -- 7: <= 219, 8:
// <= 192. 0: the compiler's choice (231-238 at MT 4: 6 waves); 7 and 8 spill and were measured slower.
#ifndef EXL3R4D_PF_WPE
#define EXL3R4D_PF_WPE 0
#endif
#if EXL3R4D_PF_WPE
#define PF_WPE_ATTR __attribute__((amdgpu_waves_per_eu(EXL3R4D_PF_WPE)))
#else
#define PF_WPE_ATTR
#endif
// EXL3R4D_PF_WPE_DN (an arm): down's GEMM alone held to that many waves a SIMD (a VGPR cap), 0: the compiler's choice.
#ifndef EXL3R4D_PF_WPE_DN
#define EXL3R4D_PF_WPE_DN 0
#endif
#ifndef EXL3R4D_PF_WPE_GU
#define EXL3R4D_PF_WPE_GU 1
#endif
#if EXL3R4D_PF_WPE_DN || EXL3R4D_PF_WPE_GU > 1
#define PF_WPE_DN_ATTR \
  __attribute__((amdgpu_waves_per_eu(MATS == 1 ? (EXL3R4D_PF_WPE_DN ? EXL3R4D_PF_WPE_DN : 1) : EXL3R4D_PF_WPE_GU)))
#else
#define PF_WPE_DN_ATTR
#endif
// EXL3R4D_PF_WGSTAMP (reading, timing only, never shipped): every GEMM workgroup's timeline in ISSUE order --
// a slot taken at entry (atomic counter, all launches of the op in sequence), the realtime counter (100 MHz) at entry
// and at exit (after a closing barrier), the hardware slot (HW_ID1) and what it did ({chunk, column block, rows, class
// or 3 = no expert / 2 = past its class's width}) -- read by perfmoe --pfstamps through the two exported functions at
// the end of this file. Names whether a launch's time is a tail of few workgroups or uniformly slow ones.
#ifndef EXL3R4D_PF_WGSTAMP
#define EXL3R4D_PF_WGSTAMP 0
#endif
#define EXL3R4D_PF_WGS_ON (EXL3R4D_PF_WGSTAMP && EXL3R4D_NCLS == 2)   // the two-class build only (one definition)
#if EXL3R4D_PF_WGS_ON
constexpr int kPfsMax = 65536;
__device__ unsigned long long g_pfs_t[kPfsMax][2];   // [slot][entry, exit]
__device__ unsigned g_pfs_hw[kPfsMax], g_pfs_info[kPfsMax], g_pfs_n;
struct PfStamp {   // entry in the constructor, exit (every return path: workgroup-uniform) in the destructor
  unsigned info = 0;
  __device__ PfStamp() {
    __shared__ unsigned s_slot;
    if (threadIdx.x == 0) {
      unsigned hw;
      asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw));
      s_slot = atomicAdd(&g_pfs_n, 1u);
      if (s_slot < kPfsMax) { g_pfs_t[s_slot][0] = __builtin_readsteadycounter(); g_pfs_hw[s_slot] = hw; }
    }
    __syncthreads();
    slot = s_slot;
  }
  __device__ ~PfStamp() {
    __syncthreads();
    if (threadIdx.x == 0 && slot < kPfsMax) { g_pfs_t[slot][1] = __builtin_readsteadycounter(); g_pfs_info[slot] = info; }
  }
  unsigned slot;
};
#define PF_STAMP_INFO(c, cb, rows, cls) \
  (pfst_.info = (unsigned)(c) | ((unsigned)(cb) << 8) | ((unsigned)(rows) << 16) | ((unsigned)(cls) << 28))
#else
#define PF_STAMP_INFO(c, cb, rows, cls) ((void)0)
#endif

// XR (the copy ring; gate_up's map form): grid row 0 is the copy row. Its gridDim.x workgroups walk chunk c's copy list
// in order (the host-read groups as the GEMM rows will reach them, after every resident group); each copies its share
// of the head's codes record (16-byte loads, 8 a thread in flight) into the ring, then fences and adds 1 to the head's
// flag. Nothing here waits, so a copy row that started always finishes.
static __device__ void pf_xr_copy(const Exl3r4dPrefill& a, int lo) {
  const Exl3r4dGemm& p = a.g;
  const uint32_t* const xt = xr_tab(a);
  const uint32_t *const xoff = xt, *const xlist = xt + 3 * a.cap, *const xcnt = xt + 4 * a.cap;
  uint32_t* const xflag = (uint32_t*)xt + 2 * a.cap;
  char* const ring = xr_ring(a);
  const uint4* ent = (const uint4*)a.map + 1;
  const int n = (int)xcnt[a.chunk_c], ncw = (int)gridDim.x, me = (int)blockIdx.x;
  const int tid = threadIdx.x, nt = blockDim.x;
  constexpr int U = 8;
  for (int j = 0; j < n; j++) {
    const int idx = (int)xlist[lo + j];
    const Planes w = planes_of(p, (int)ent[idx].x);
    const long long units = (long long)w.rows * w.cols * xr_k2_of(a) / 16 / 16;   // 16-byte units (rows: a multiple of 256)
    const long long per = (units + ncw - 1) / ncw, u0 = (long long)me * per, u1 = min(units, u0 + per);
    const uint4* src = (const uint4*)w.codes;
    uint4* dst = (uint4*)(ring + (size_t)(xoff[idx] - 1u) * 256);
    for (long long u = u0 + tid; u < u1; u += (long long)nt * U) {
      uint4 v[U];
#pragma unroll
      for (int k = 0; k < U; k++) {
        const long long q = u + (long long)k * nt;
        if (q < u1) v[k] = src[q];
      }
#pragma unroll
      for (int k = 0; k < U; k++) {
        const long long q = u + (long long)k * nt;
        if (q < u1) dst[q] = v[k];
      }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) atomicAdd(&xflag[idx], 1u);
  }
}
// XR: a workgroup of entry k whose group has a copy waits for the copy row's flag (gridDim.x workgroups done), then reads
// the copy; past EXL3R4D_XR_SPIN polls it reads its codes in place (the same bytes). Every wave takes the flag with an
// agent-scope acquire itself (a workgroup's waves may sit on both CUs of a WGP).
static __device__ __forceinline__ void pf_xr_redirect(const Exl3r4dPrefill& a, int k, Planes& w) {
  const uint32_t* const xt = xr_tab(a);
  const unsigned xo = xt[k];
  if (!xo) return;
  const uint32_t* f = xt + 2 * a.cap + xt[a.cap + k];
  __shared__ int s_xok;
  if (threadIdx.x == 0) {
    int ok = 0;
    for (int it = 0; it < EXL3R4D_XR_SPIN; it++) {
      if (__hip_atomic_load(f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) >= gridDim.x) { ok = 1; break; }
      __builtin_amdgcn_s_sleep(8);
    }
    s_xok = ok;
  }
  __syncthreads();
  if (threadIdx.x == 0) XR_CNT(s_xok ? 4 : 5, 1);
  if (__builtin_amdgcn_readfirstlane(s_xok)) {
    (void)__hip_atomic_load(f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    w.codes = (const unsigned*)(xr_ring(a) + (size_t)(__builtin_amdgcn_readfirstlane(xo) - 1u) * 256);
  }
}

// YS: down's map form with the streamed y (pf_epi_dn_ys; prefill.c's EXL3R4D_PF_YNT_MB decides a launch's form).
// XR: gate_up's map form with the copy ring's copy row (grid row 0) and redirect (above).
template <int BITS, int MTX, int NPW, int MATS, int AM, bool RUN, bool YS = false, bool XR = false>
__global__ void __launch_bounds__(512) PF_WPE_ATTR PF_WPE_DN_ATTR exl3r4d_pf_gemm_kernel(Exl3r4dPrefill a) {
  EXL3R4D_KARG_PREFETCH();
  extern __shared__ float lds[];
#if EXL3R4D_PF_WGS_ON
  PfStamp pfst_;
#endif
  const Exl3r4dGemm& p = a.g;
  int e, r0, r1, rbase = 0;
  [[maybe_unused]] int kx = -1;   // XR: the entry (map index)
  if constexpr (RUN) {
    __shared__ int s_find[33];
    const Run run = find_run(p, blockIdx.y, s_find);
    if (run.e < 0) {
      zero_dead_rows(p, (int)blockIdx.y - run.nlive, (int)gridDim.y - run.nlive, blockIdx.x);
      return;
    }
    e = run.e, r0 = run.r0, r1 = run.r1;
  } else {
    int lo, hi;
    pf_chunk(a, a.chunk_c, lo, hi, rbase);
    int y = (int)blockIdx.y;
    if constexpr (XR) {   // grid row 0: the copy row
      if (y == 0) {
        pf_xr_copy(a, lo);
        return;
      }
      y -= 1;
    }
    int k = lo + y;
    if (k >= hi) return;
    if (EXL3R4D_PF_LPT && MATS == 2 && a.order) k = (int)a.order[k];   // longest entries first (EXL3R4D_PF_LPT)
    kx = k;
    const uint4 ent = ((const uint4*)a.map)[1 + k];
    r0 = (int)ent.y, r1 = (int)ent.z;
    PF_STAMP_INFO(a.chunk_c, blockIdx.x, r1 - r0, ent.x == EXL3R4D_PF_ZERO ? 3u : ent.x % (unsigned)EXL3R4D_NCLS);
    constexpr int GD = MATS == 1 && AM == 0 ? EXL3R4D_PF_DN_G : 1;
    if (ent.x == EXL3R4D_PF_ZERO) {
      if constexpr (GD == 1)
        zero_rows(p, r0, r1, blockIdx.x);
      else
        for (int g = 0; g < GD && (int)blockIdx.x * GD + g < (int)(p.y_cols / 128); g++)
          zero_rows(p, r0, r1, (int)blockIdx.x * GD + g);
      return;
    }
    e = (int)ent.x;
    if constexpr (EXL3R4D_PF_PAIR && MATS == 2 && AM == 0)
      if ((r1 - r0 + 15) / 16 > MTX) return;   // PAIR: the pair kernel's entry
    if constexpr (GD > 1) {   // down, G column blocks a workgroup (entries are at most 16 MTX rows)
      const Planes w = planes_of(p, e);
      pf_dg_dispatch<BITS, 1, MTX, GD>((r1 - r0 + 15) / 16, p, w, r0, r1, lds, rbase);
      return;
    }
  }
  Planes w = planes_of(p, e);
  const int width = MATS == 2 ? w.rows / 2 : w.rows;
  if ((int)blockIdx.x * 128 >= width) {  // the narrower parity class: zeros past its own width
    PF_STAMP_INFO(a.chunk_c, blockIdx.x, r1 - r0, 2u);
    zero_rows(p, r0, r1, blockIdx.x);
    return;
  }
  if constexpr (XR && !RUN) pf_xr_redirect(a, kx, w);   // the copy, once it landed
  for (int rc = r0; rc < r1; rc += 16 * MTX) {   // one pass in the map form (an entry is at most 16 MTX rows)
    const int re = rc + 16 * MTX < r1 ? rc + 16 * MTX : r1;
    pf_dispatch<BITS, 1, MTX, NPW, MATS, AM, YS>((re - rc + 15) / 16, p, w, rc, re, lds, rbase);
  }
}

// ----------------------------------------------------------------------------- PAIR: one decode a tile an expert
// EXL3R4D_PF_PAIR (gate_up's map form; measured slower, kept as an arm). The default GEMM is issue-bound with the
// decode ~51% of its cycles (thread trace), and the map's 64-row entries decode every tile once AN ENTRY: an expert of
// 65..128 rows decodes it twice. With PAIR the map's entries are up to 128 rows; one of more than MTX blocks goes to
// this 16-wave workgroup (the regular kernel skips it): two sets of 8 waves split its blocks (set 0 the first
// ceil(mt / 2)); wave wl of each set owns the regular kernel's tile pair (nt0, nt0 + 1) for its own rows. Set s
// decodes tile nt0 + s only and the two trade the decoded fragment through LDS each k-step (one barrier a step,
// double-buffered slots). Every output tile is still one wave walking K in order with the same fragments: the same
// bits.
template <int BITS, int MT, int MTA>
static __device__ __forceinline__ void pf_pair_body(const Exl3r4dGemm& p, const Planes& w, int r0, int r1, float* lds,
                                                    int rbase, int set) {
  using P = Plan<BITS>;
  constexpr int NW = P::NW, WORDS = P::WORDS, MATS = 2, PF = 2, WS = 4;
  static_assert(NW == 2, "the buffer form");
  const int tid = threadIdx.x, lane = tid % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32), wl = wave & 7;
  const int h = lane >> 4, nl = lane & 15;
  const int width = w.rows / 2, ntiles = width / 16, ksteps = w.cols / 16, cb = blockIdx.x;
  const int vl = wl * 2, mat = vl / 8, nt0 = cb * 8 + vl % 8;
  const int rs = r0 + 16 * MTA * set, re = set ? r1 : min(r1, r0 + 16 * MTA);   // this set's rows
  unsigned avo[MT];
#pragma unroll
  for (int i = 0; i < MT; i++) {
    const int row = rs + 16 * i + nl;
    avo[i] = (unsigned)(((size_t)((row < re ? row : re - 1) - rbase) * p.xr_pitch + 4 * h) * 2);
  }
  const __amdgpu_buffer_rsrc_t ars = __builtin_amdgcn_make_buffer_rsrc(
      (void*)((const _Float16*)p.xr + (size_t)mat * w.cols), (short)0, 0x7fffffff, 0x31004000);
  const P plan(lane);
  unsigned wvo[2][NW];
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int x = 0; x < NW; x++) wvo[r][x] = 4u * (unsigned)plan.woff[r][x];
  const unsigned* wexp = w.codes + (size_t)mat * ksteps * ntiles * WORDS;
  const __amdgpu_buffer_rsrc_t wrs = __builtin_amdgcn_make_buffer_rsrc((void*)wexp, (short)0, 0x7fffffff, 0x31004000);
  const unsigned bso = (unsigned)(nt0 + set) * WORDS * 4, kbytes = (unsigned)ntiles * WORDS * 4;
  const unsigned bwm = wvo[0][0], bw0 = wvo[0][1], bwp = wvo[1][NW - 1];
  constexpr int kAux = EXL3R4D_PF_NT ? 1 : 0;
  // LDS: slot[k % 2][wl][set] of 32 lanes x 16 bytes (the decoded fragment)
  v4u* const xch = (v4u*)lds;
  const int mine = ((0 * 8 + wl) * 2 + set) * 32 + lane, theirs = ((0 * 8 + wl) * 2 + (set ^ 1)) * 32 + lane;
#define PR_WLOAD(ks, W_)                                                                             \
  do {                                                                                               \
    const unsigned so_ = bso + (unsigned)(ks) * kbytes;                                              \
    if constexpr (BITS == 4) {                                                                       \
      const unsigned m_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwm, so_, kAux);                 \
      const unsigned z_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bw0, so_, kAux);                 \
      const unsigned q_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwp, so_, kAux);                 \
      W_[0][0] = m_; W_[0][1] = z_; W_[1][0] = z_; W_[1][1] = q_;                                   \
    } else {                                                                                         \
      _Pragma("unroll") for (int r = 0; r < 2; r++)                                                  \
        _Pragma("unroll") for (int x = 0; x < NW; x++)                                               \
          W_[r][x] = __builtin_amdgcn_raw_buffer_load_b32(wrs, wvo[r][x], so_, kAux);                \
    }                                                                                                \
  } while (0)
#define PR_ALOAD(ks, A_)                                                                             \
  do {                                                                                               \
    _Pragma("unroll") for (int i = 0; i < MT; i++) {                                                 \
      union { v2u w[2]; v8h f; } x_;                                                                 \
      x_.w[0] = __builtin_amdgcn_raw_buffer_load_b64(ars, avo[i], (unsigned)(ks) * 32u, 0);          \
      x_.w[1] = __builtin_amdgcn_raw_buffer_load_b64(ars, avo[i] + 16u, (unsigned)(ks) * 32u, 0);    \
      A_[i] = x_.f;                                                                                  \
    }                                                                                                \
  } while (0)
  // acc[0] = tile nt0 (set 0 decodes it), acc[1] = tile nt0 + 1 (set 1)
  v8f acc[2][MT];
#pragma unroll
  for (int j = 0; j < 2; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);
  [[maybe_unused]] v8h svp[epi_opw<2, MATS>()][MATS];
  svh_early<2, MATS>(w, wl, lane, cb, svp);
  const v2u dsr = pf_dn_suh(p, w, cb, lane);
  unsigned wb[WS][2][NW];
  v8h ab[PF][MT];
#pragma unroll
  for (int u = 0; u < WS - 1; u++) PR_WLOAD(u < ksteps ? u : ksteps - 1, wb[u]);
  PR_ALOAD(0, ab[0]);
  for (int ks0 = 0; ks0 < ksteps; ks0 += 4) {   // ksteps is a multiple of 8
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int ks = ks0 + u;
      const int lw = ks + WS - 1 < ksteps ? ks + WS - 1 : ksteps - 1, la = ks + 1 < ksteps ? ks + 1 : ksteps - 1;
      PR_WLOAD(lw, wb[(u + WS - 1) % WS]);
      PR_ALOAD(la, ab[(u + 1) % PF]);
      union { v8h f; v4u u; } own, oth;
      own.f = decode_frag<BITS, NW>(wb[u % WS], plan.shift);
      xch[(u & 1) * 512 + mine] = own.u;
      lds_barrier();
      oth.u = xch[(u & 1) * 512 + theirs];
      const v8h w0 = set ? oth.f : own.f, w1 = set ? own.f : oth.f;
#pragma unroll
      for (int i = 0; i < MT; i++) acc[0][i] = mma(w0, ab[u % PF][i], acc[0][i]);
#pragma unroll
      for (int i = 0; i < MT; i++) acc[1][i] = mma(w1, ab[u % PF][i], acc[1][i]);
      __builtin_amdgcn_sched_barrier(0);
    }
  }
#undef PR_WLOAD
#undef PR_ALOAD
  lds_barrier();   // the epilogue reuses the LDS
  // Each set its own epilogue region; both sets make MTA calls (set 1's extra ones write nothing) so the
  // workgroup-wide barriers inside pair up.
  float* elds = (float*)((unsigned char*)lds + 16384 + set * kGrpEpi);
#pragma unroll
  for (int i = 0; i < MTA; i++) {
    const int row = rs + 16 * i + nl;
    const bool in_run = i < MT && row < re;
    const int tok = p.sorted[in_run ? row : r0];
    v8f blk[2];
#pragma unroll
    for (int j = 0; j < 2; j++) blk[j] = i < MT ? acc[j][i < MT ? i : 0] : (v8f)(0.f);
    pf_epi_gu<2>(p, w, blk, wl, lane, cb, row, in_run, in_run && tok >= 0, elds, svp, dsr, wl, 8);
  }
}

// Entries of MTX + 1 .. 2 MTX blocks; set 0 takes MTA = ceil(mt / 2), set 1 the rest (MTA or MTA - 1).
template <int BITS, int MTA, int MTX>
static __device__ __forceinline__ void pf_pair_dispatch(int mt, const Exl3r4dGemm& p, const Planes& w, int r0, int r1,
                                                        float* lds, int rbase, int set) {
  if constexpr (MTA > MTX) {
    return;
  } else {
    if ((mt + 1) / 2 == MTA) {
      if (set == 0 || mt % 2 == 0)
        pf_pair_body<BITS, MTA, MTA>(p, w, r0, r1, lds, rbase, set);
      else
        pf_pair_body<BITS, MTA - 1, MTA>(p, w, r0, r1, lds, rbase, set);
    } else {
      pf_pair_dispatch<BITS, MTA + 1, MTX>(mt, p, w, r0, r1, lds, rbase, set);
    }
  }
}

#ifndef EXL3R4D_PF_PAIR_WPE   // the pair kernel's minimum waves a SIMD (8: <= 192 VGPRs, two workgroups a WGP)
#define EXL3R4D_PF_PAIR_WPE 1
#endif
template <int BITS>
__global__ void __launch_bounds__(512) __attribute__((amdgpu_waves_per_eu(EXL3R4D_PF_PAIR_WPE)))
exl3r4d_pf_pair_kernel(Exl3r4dPrefill a) {
  EXL3R4D_KARG_PREFETCH();
  extern __shared__ float lds[];
  const Exl3r4dGemm& p = a.g;
  int lo, hi, rbase;
  pf_chunk(a, a.chunk_c, lo, hi, rbase);
  const int k = lo + (int)blockIdx.y;
  if (k >= hi) return;
  const uint4 ent = ((const uint4*)a.map)[1 + k];
  const int r0 = (int)ent.y, r1 = (int)ent.z, mt = (r1 - r0 + 15) / 16;
  if (ent.x == EXL3R4D_PF_ZERO || mt <= EXL3R4D_PF_MTX) return;   // the regular kernel's
  const Planes w = planes_of(p, (int)ent.x);
  if ((int)blockIdx.x * 128 >= w.rows / 2) {   // the narrower class: zeros past its own width
    zero_rows(p, r0, r1, blockIdx.x);
    return;
  }
  const int set = __builtin_amdgcn_readfirstlane((int)(threadIdx.x / WAVE32)) >> 3;
  pf_pair_dispatch<BITS, (EXL3R4D_PF_MTX + 2) / 2, EXL3R4D_PF_MTX>(mt, p, w, r0, r1, lds, rbase, set);
}
static const void* pf_pair_fn(int bits) {
  switch (bits) {
    case 2: return (const void*)exl3r4d_pf_pair_kernel<2>;
    case 3: return (const void*)exl3r4d_pf_pair_kernel<3>;
    case 4: return (const void*)exl3r4d_pf_pair_kernel<4>;
    case 5: return (const void*)exl3r4d_pf_pair_kernel<5>;
    case 6: return (const void*)exl3r4d_pf_pair_kernel<6>;
    case EXL3R4D_K25: return (const void*)exl3r4d_pf_pair_kernel<EXL3R4D_K25>;   // half-integer K = 2.5
    default: return nullptr;
  }
}

// ----------------------------------------------------------------------------- AM 4: gate_up fused, a row rotated once
//
// WHY (measured slower than the map form, kept as an arm): the map form's gate_up moves ~843 MB at tp2 M=2048 -- 423 MB
// of weights plus the prep's rotated rows written (210 MB) and read back (210 MB) -- at ~79% of DRAM peak; the prep
// alone is write-bandwidth bound (341 us, batching its loads changed nothing). Weights at ~85% of DRAM, the practical
// peak, needs the rows' round trip gone. Rotating in the column-block GEMM (AM 1 / 3) redoes each row's rotation in
// every column block (x3 even, x2 odd) and lost on VALU. Here ONE workgroup owns a map entry's rows across ALL N of its
// expert, so each (row, matrix, 128-wide K block) is rotated once, into LDS, and x (10.5 MB, cache-resident) is the
// only activation stream.
//
// Shape: 16 waves; wave w owns matrix w / 8 (gate, up) and that matrix's tile w % 8 of every 128-column block cb
// (NCB = width / 128 tiles a wave: 3 at tp2's 384, 2 at 256) x MT 16-row blocks -- so for each cb the 16 waves hold
// its 8 gate + 8 up tiles, one each: the decode band's epilogue_block<1, 2> as is (the same bits as every other path:
// one wave owns a 16x16 output tile and walks K in order). A K pass = 128-wide block: the waves rotate its rows into
// LDS ([row][gate 128 | up 128 | 8 pad] f16; wave w: matrix w / 8, rows w % 8 + 8 q), barrier, 8 K steps read the A
// fragments from LDS, the NEXT block's x and suh loads in flight under them, barrier, rotate, barrier. The weights
// stream one k-step ahead across the passes, in the buffer form, as the map form's.
#ifndef EXL3R4D_PF_FU_MID
#define EXL3R4D_PF_FU_MID 1
#endif
#ifndef EXL3R4D_PF_FU_WS   // AM 4: k-steps of weights in flight a wave (2, 4 or 8)
#define EXL3R4D_PF_FU_WS 2
#endif
template <int BITS, int MT, int NCB, bool XF>
static __device__ __forceinline__ void pf_fused_body(const Exl3r4dGemm& p, const Planes& w, int r0, int r1,
                                                     float* lds) {
  using P = Plan<BITS>;
  constexpr int NW = P::NW, WORDS = P::WORDS;
  constexpr int APITCH = 2 * 128 + 8, RPW = 2 * MT;
  static_assert(NW == 2, "the fused GEMM takes the buffer form (K = 2..6)");
  const int tid = threadIdx.x, lane = tid % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);
  const int h = lane >> 4, nl = lane & 15, mat = wave >> 3, wl = wave & 7;
  const int width = w.rows / 2, ntiles = width / 16, ksteps = w.cols / 16, nkb = ksteps / 8, nrows = r1 - r0;
  _Float16* const alds = (_Float16*)lds;
  const unsigned aoff = (unsigned)(nl * APITCH + mat * 128 + 4 * h);

  // The rotation tasks: rows wl + 8 q of the entry, matrix mat. The token row is uniform (an SGPR offset).
  const __amdgpu_buffer_rsrc_t xrs = __builtin_amdgcn_make_buffer_rsrc((void*)p.x, (short)0, 0x7fffffff, 0x31004000);
  const __amdgpu_buffer_rsrc_t srs =
      __builtin_amdgcn_make_buffer_rsrc((void*)(mat ? w.suh1 : w.suh), (short)0, 0x7fffffff, 0x31004000);
  unsigned xoff[RPW];
  bool rlive[RPW];
#pragma unroll
  for (int q = 0; q < RPW; q++) {
    const int i = wl + 8 * q, r = r0 + (i < nrows ? i : nrows - 1);
    const int t = __builtin_amdgcn_readfirstlane(p.sorted[r]);
    xoff[q] = (unsigned)((size_t)(t >= 0 ? min(t / p.top_k, p.n_tok - 1) : 0) * p.x_pitch * 2);
    rlive[q] = i < nrows && t >= 0;
  }
  v2u xv[RPW], sv;
  auto xload = [&](int kb) __attribute__((always_inline)) {
    const unsigned ko = (unsigned)kb * 256u;
#pragma unroll
    for (int q = 0; q < RPW; q++)
      xv[q] = __builtin_bit_cast(v2u, __builtin_amdgcn_raw_buffer_load_b64(xrs, (unsigned)lane * 8u, xoff[q] + ko, 0));
    sv = __builtin_bit_cast(v2u, __builtin_amdgcn_raw_buffer_load_b64(srs, (unsigned)lane * 8u, ko, 0));
  };
  // The rotation in two halves (EXL3R4D_PF_FU_MID): its math into registers in the middle of a pass, under the other
  // waves' WMMAs, and only the LDS stores between the pass's two barriers. xv and ro are never live at once.
  v2u ro[RPW];
  auto rot_math = [&]() __attribute__((always_inline)) {
#pragma unroll
    for (int q = 0; q < RPW; q++) {
      const v2u r_ = rotate128_math(xv[q], sv, XF, lane);   // XF: no branch
      ro[q] = rlive[q] ? r_ : (v2u)(0u);
    }
  };
  auto rot_put = [&]() __attribute__((always_inline)) {
#pragma unroll
    for (int q = 0; q < RPW; q++) *(v2u*)(alds + (wl + 8 * q) * APITCH + mat * 128 + 4 * lane) = ro[q];
  };

  // The weights: matrix mat, tile (tk, j * 8 + wl) at ((mat * ksteps + tk) * ntiles + j * 8 + wl) * WORDS u32.
  const P plan(lane);
  unsigned wvo[2][NW];
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int x = 0; x < NW; x++) wvo[r][x] = 4u * (unsigned)plan.woff[r][x];
  const unsigned* wexp = w.codes + (size_t)mat * ksteps * ntiles * WORDS;
  const __amdgpu_buffer_rsrc_t wrs = __builtin_amdgcn_make_buffer_rsrc((void*)wexp, (short)0, 0x7fffffff, 0x31004000);
  const unsigned bso = (unsigned)wl * WORDS * 4, kbytes = (unsigned)ntiles * WORDS * 4;
  constexpr unsigned TS = 8u * WORDS * 4u;   // bytes from a wave's tile in one column block to the next
  const unsigned bwm = wvo[0][0], bw0 = wvo[0][1], bwp = wvo[1][NW - 1];
  constexpr int kAux = EXL3R4D_PF_NT ? 1 : 0;
#define FU_WLOAD(ks, W_)                                                                             \
  do {                                                                                               \
    const unsigned so_ = bso + (unsigned)(ks) * kbytes;                                              \
    _Pragma("unroll") for (int j = 0; j < NCB; j++) {                                                \
      if constexpr (BITS == 4) {                                                                     \
        const unsigned m_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwm + j * TS, so_, kAux);      \
        const unsigned z_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bw0 + j * TS, so_, kAux);      \
        const unsigned q_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwp + j * TS, so_, kAux);      \
        W_[j][0][0] = m_; W_[j][0][1] = z_; W_[j][1][0] = z_; W_[j][1][1] = q_;                     \
      } else {                                                                                       \
        _Pragma("unroll") for (int r = 0; r < 2; r++)                                                \
          _Pragma("unroll") for (int x = 0; x < NW; x++)                                             \
            W_[j][r][x] = __builtin_amdgcn_raw_buffer_load_b32(wrs, wvo[r][x] + j * TS, so_, kAux);  \
      }                                                                                              \
    }                                                                                                \
  } while (0)

  v8f acc[NCB][MT];
#pragma unroll
  for (int j = 0; j < NCB; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);
  constexpr int WS = EXL3R4D_PF_FU_WS;
  static_assert(8 % WS == 0, "a pass's 8 steps hold whole weight rings");
  unsigned wb[WS][NCB][2][NW];

  xload(0);
#pragma unroll
  for (int u = 0; u < WS - 1; u++) FU_WLOAD(u < ksteps ? u : ksteps - 1, wb[u]);
  rot_math();
  rot_put();
  xload(nkb > 1 ? 1 : 0);
  lds_barrier();
  for (int kb = 0; kb < nkb; kb++) {
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
      if constexpr (EXL3R4D_PF_FU_MID)
        if (kk == 4) rot_math();   // block kb + 1's rows (kk is a constant: no branch)
      const int ks = kb * 8 + kk, ln = ks + WS - 1 < ksteps ? ks + WS - 1 : ksteps - 1;
      FU_WLOAD(ln, wb[(kk + WS - 1) % WS]);
      v8h af[MT];
#pragma unroll
      for (int i = 0; i < MT; i++) af[i] = frag_lds(alds + aoff + i * 16 * APITCH + kk * 16);
#pragma unroll
      for (int j = 0; j < NCB; j++) {
        const v8h wf = decode_frag<BITS, NW>(wb[kk % WS][j], plan.shift);
#pragma unroll
        for (int i = 0; i < MT; i++)
          acc[j][i] = mma(wf, af[i], acc[j][i]);
      }
      // A fence a k-step: unfenced, the scheduler hoists all 8 steps' loads of the pass to its top and spills them
      // (1,100 registers at MT 4 / NCB 3).
      __builtin_amdgcn_sched_barrier(0);
    }
    // Unconditional (the last pass rotates its own block again, unread): a branch here splits the pass into a block
    // of loads and decodes and a block of WMMAs, and the decoded fragments of all 8 steps spill.
    lds_barrier();   // every wave is done reading block kb
    if constexpr (!EXL3R4D_PF_FU_MID) rot_math();
    rot_put();       // block kb + 1, its loads issued a pass ago
    xload(kb + 2 < nkb ? kb + 2 : nkb - 1);
    lds_barrier();
  }
#undef FU_WLOAD
  lds_barrier();   // the epilogue reuses the LDS

  // The epilogue, a 16-row block and a column block at a time: the decode band's epilogue_block, 16 waves of one tile.
  // EXL3R4D_PF_SVH_EARLY: every column block's svh loaded once here (one round trip, not one a call).
  [[maybe_unused]] v8h svp[NCB][1][2];
  [[maybe_unused]] v2u dsr[NCB];
  if constexpr (EXL3R4D_PF_SVH_EARLY || EXL3R4D_PF_EPI_PRE)
#pragma unroll
    for (int j = 0; j < NCB; j++) svh_early<1, 2>(w, wave, lane, j, svp[j]);
  if constexpr (EXL3R4D_PF_EPI_PRE)
#pragma unroll
    for (int j = 0; j < NCB; j++) dsr[j] = pf_dn_suh(p, w, j, lane);
#pragma unroll
  for (int i = 0; i < MT; i++) {
    const int row = r0 + 16 * i + nl;
    const bool in_run = row < r1;
    const int tok = p.sorted[in_run ? row : r0];
#pragma unroll
    for (int j = 0; j < NCB; j++) {
      const v8f blk[1] = {acc[j][i]};
      if constexpr (EXL3R4D_PF_EPI_PRE)
        pf_epi_gu<1>(p, w, blk, wave, lane, j, row, in_run, in_run && tok >= 0, lds, svp[j], dsr[j], wave, 16);
      else if constexpr (EXL3R4D_PF_SVH_EARLY)
        epilogue_block<1, 2, true>(p, w, blk, wave, 0, lane, j, row, in_run, in_run && tok >= 0, lds, svp[j]);
      else
        epilogue_block<1, 2>(p, w, blk, wave, 0, lane, j, row, in_run, in_run && tok >= 0, lds);
    }
  }
}

template <int BITS, int M, int MTX, int NCB, bool XF>
static __device__ __forceinline__ void pf_fused_dispatch(int mt, const Exl3r4dGemm& p, const Planes& w, int r0, int r1,
                                                         float* lds) {
  if constexpr (M == MTX) {
    pf_fused_body<BITS, M, NCB, XF>(p, w, r0, r1, lds);
  } else {
    if (mt <= M)
      pf_fused_body<BITS, M, NCB, XF>(p, w, r0, r1, lds);
    else
      pf_fused_dispatch<BITS, M + 1, MTX, NCB, XF>(mt, p, w, r0, r1, lds);
  }
}
// x's format (f16 or bf16) as a template constant: a runtime one makes rotate128_math branch inside the K loop.
template <int BITS, int MLO, int MHI, int NCB>
static __device__ __forceinline__ void pf_fused_dispatch_x(int mt, const Exl3r4dGemm& p, const Planes& w, int r0,
                                                           int r1, float* lds) {
  if (p.x_bf16)
    pf_fused_dispatch<BITS, MLO, MHI, NCB, false>(mt, p, w, r0, r1, lds);
  else
    pf_fused_dispatch<BITS, MLO, MHI, NCB, true>(mt, p, w, r0, r1, lds);
}

// One workgroup = map entry blockIdx.y, every column of its expert (grid.x 1). Never chunked: no rows in scratch.
// The entries of MLO..MHI 16-row blocks (EXL3R4D_PF_FU_SPLIT: two launches over the same map, so the kernel of the
// short entries is allocated its own -- fewer -- registers; a kernel's VGPRs are its largest variant's). The launch with
// MLO = 1 also zeroes the rows no run owns. WPE: the minimum waves a SIMD asked of the compiler (0: its choice).
template <int BITS, int MLO, int MHI, int WPE>
__global__ void __launch_bounds__(512) __attribute__((amdgpu_waves_per_eu(WPE > 0 ? WPE : 1)))
exl3r4d_pf_fused_kernel(Exl3r4dPrefill a) {
  EXL3R4D_KARG_PREFETCH();
  extern __shared__ float lds[];
  const Exl3r4dGemm& p = a.g;
  const int k = (int)blockIdx.y;
  if (k >= (int)a.map[0]) return;
  const uint4 ent = ((const uint4*)a.map)[1 + k];
  const int r0 = (int)ent.y, r1 = (int)ent.z, ncol = (int)(p.y_cols / 128), mt = (r1 - r0 + 15) / 16;
  if (ent.x == EXL3R4D_PF_ZERO) {
    if (MLO == 1)
      for (int cb = 0; cb < ncol; cb++) zero_rows(p, r0, r1, cb);
    return;
  }
  if (mt < MLO || mt > MHI) return;   // the other launch's entry
  const Planes w = planes_of(p, (int)ent.x);
  const int ncb = w.rows / 256;
  if (ncb == 3)
    pf_fused_dispatch_x<BITS, MLO, MHI, 3>(mt, p, w, r0, r1, lds);
  else if (ncb == 2)
    pf_fused_dispatch_x<BITS, MLO, MHI, 2>(mt, p, w, r0, r1, lds);
  else
    pf_fused_dispatch_x<BITS, MLO, MHI, 1>(mt, p, w, r0, r1, lds);
  for (int cb = ncb; cb < ncol; cb++) zero_rows(p, r0, r1, cb);   // the narrower class: zeros past its width
}
#ifndef EXL3R4D_PF_FU_SPLIT   // AM 4: entries of <= SPLIT 16-row blocks in a launch of their own (0: one launch)
#define EXL3R4D_PF_FU_SPLIT 0
#endif
#ifndef EXL3R4D_PF_FU_WPE     // AM 4 split: the short entries' kernel's minimum waves a SIMD (8: <= 192 VGPRs)
#define EXL3R4D_PF_FU_WPE 0
#endif

// ----------------------------------------------------------------------------- AM 5: gate_up, column-block groups
//
// WHY (measured slower than the map form, kept as an arm): AM 4's one 16-wave workgroup a WGP at 246+ VGPRs leaves its
// prologue, epilogue and every wait exposed (deletions: skeleton ~0.5 ms + epilogue ~0.22 ms at M=2048). AM 5 keeps the
// regular map form's per-wave work (NPW 2 tiles x MT 16-row blocks, 6 waves a SIMD) and merges the expert's
// column-block workgroups into ONE workgroup of NCB groups of 8 waves (group g = column block g: the regular (entry,
// cb) workgroup's waves) that share the entry's rotated rows in LDS -- each (row, matrix, 128-K block) rotated once, no
// prep, no rows scratch. One launch a class width (NCB = width / 128: 3 and 2 at tp2), each skipping the other class's
// entries. LDS: the K pass's rows [64][264] f16 (33.8 KB), then each group's epilogue region (16 KB tiles + 4 KB h
// stage) side by side.
#ifndef EXL3R4D_PF_G_WS   // AM 5: k-steps of weights in flight a wave
#define EXL3R4D_PF_G_WS 4
#endif
template <int BITS, int MT, int NCB, bool XF>
static __device__ __forceinline__ void pf_grp_body(const Exl3r4dGemm& p, const Planes& w, int r0, int r1,
                                                   float* lds) {
  using P = Plan<BITS>;
  constexpr int NW = P::NW, WORDS = P::WORDS, NPW = 2, WV = 8 * NCB;
  constexpr int APITCH = 2 * 128 + 8, NT = 2 * 16 * MT, RPW = (NT + WV - 1) / WV;
  static_assert(NW == 2, "AM 5 takes the buffer form (K = 2..6)");
  const int tid = threadIdx.x, lane = tid % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);
  const int h = lane >> 4, nl = lane & 15, cb = wave >> 3, wl = wave & 7;
  const int vl = wl * NPW, mat = vl / 8, nt0 = cb * 8 + vl % 8;
  const int width = w.rows / 2, ntiles = width / 16, ksteps = w.cols / 16, nkb = ksteps / 8, nrows = r1 - r0;
  _Float16* const alds = (_Float16*)lds;
  const unsigned aoff = (unsigned)(nl * APITCH + mat * 128 + 4 * h);

  // Rotation tasks t = wave + WV q: row t / 2, matrix t % 2 (uniform; x row offsets in SGPRs).
  const __amdgpu_buffer_rsrc_t xrs = __builtin_amdgcn_make_buffer_rsrc((void*)p.x, (short)0, 0x7fffffff, 0x31004000);
  const __amdgpu_buffer_rsrc_t srs0 = __builtin_amdgcn_make_buffer_rsrc((void*)w.suh, (short)0, 0x7fffffff, 0x31004000);
  const __amdgpu_buffer_rsrc_t srs1 = __builtin_amdgcn_make_buffer_rsrc((void*)w.suh1, (short)0, 0x7fffffff, 0x31004000);
  unsigned xoff[RPW];
  bool rlive[RPW];
#pragma unroll
  for (int q = 0; q < RPW; q++) {
    const int t = wave + WV * q, i = t >> 1, r = r0 + (i < nrows ? i : nrows - 1);
    const int tk = __builtin_amdgcn_readfirstlane(p.sorted[r]);
    xoff[q] = (unsigned)((size_t)(tk >= 0 ? min(tk / p.top_k, p.n_tok - 1) : 0) * p.x_pitch * 2);
    rlive[q] = t < NT && i < nrows && tk >= 0;
  }
  v2u xv[RPW], sv0, sv1, ro[RPW];
  auto xload = [&](int kb) __attribute__((always_inline)) {
    const unsigned ko = (unsigned)kb * 256u;
#pragma unroll
    for (int q = 0; q < RPW; q++)
      xv[q] = __builtin_bit_cast(v2u, __builtin_amdgcn_raw_buffer_load_b64(xrs, (unsigned)lane * 8u, xoff[q] + ko, 0));
    sv0 = __builtin_bit_cast(v2u, __builtin_amdgcn_raw_buffer_load_b64(srs0, (unsigned)lane * 8u, ko, 0));
    sv1 = __builtin_bit_cast(v2u, __builtin_amdgcn_raw_buffer_load_b64(srs1, (unsigned)lane * 8u, ko, 0));
  };
  auto rot_math = [&]() __attribute__((always_inline)) {
#pragma unroll
    for (int q = 0; q < RPW; q++) {
      const int t = wave + WV * q;   // its matrix: t % 2 = wave % 2 (WV is even)
      const v2u r_ = rotate128_math(xv[q], (wave & 1) ? sv1 : sv0, XF, lane);
      ro[q] = rlive[q] ? r_ : (v2u)(0u);
      (void)t;
    }
  };
  // Unconditional (a branch here splits the pass and spills, as in AM 4): the surplus tasks t >= NT write rows
  // 64.. of the pass buffer, inside the launch's LDS (>= 72 rows' worth whatever NCB) and never read.
  static_assert(WV * RPW <= 2 * (16 * EXL3R4D_PF_MTX + 8), "the surplus rows fit the LDS the launch asks for");
  auto rot_put = [&]() __attribute__((always_inline)) {
#pragma unroll
    for (int q = 0; q < RPW; q++) {
      const int t = wave + WV * q;
      *(v2u*)(alds + (t >> 1) * APITCH + (t & 1) * 128 + 4 * lane) = ro[q];
    }
  };

  // The weights: the regular map form's (matrix mat, tiles nt0, nt0 + 1 of every k-step), buffer form.
  const P plan(lane);
  unsigned wvo[2][NW];
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int x = 0; x < NW; x++) wvo[r][x] = 4u * (unsigned)plan.woff[r][x];
  const unsigned* wexp = w.codes + (size_t)mat * ksteps * ntiles * WORDS;
  const __amdgpu_buffer_rsrc_t wrs = __builtin_amdgcn_make_buffer_rsrc((void*)wexp, (short)0, 0x7fffffff, 0x31004000);
  const unsigned bso = (unsigned)nt0 * WORDS * 4, kbytes = (unsigned)ntiles * WORDS * 4;
  const unsigned bwm = wvo[0][0], bw0 = wvo[0][1], bwp = wvo[1][NW - 1];
  constexpr int kAux = EXL3R4D_PF_NT ? 1 : 0;
#define GR_WLOAD(ks, W_)                                                                             \
  do {                                                                                               \
    const unsigned so_ = bso + (unsigned)(ks) * kbytes;                                              \
    _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                                \
      if constexpr (BITS == 4) {                                                                     \
        const unsigned m_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwm + j * 128, so_, kAux);     \
        const unsigned z_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bw0 + j * 128, so_, kAux);     \
        const unsigned q_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwp + j * 128, so_, kAux);     \
        W_[j][0][0] = m_; W_[j][0][1] = z_; W_[j][1][0] = z_; W_[j][1][1] = q_;                     \
      } else {                                                                                       \
        _Pragma("unroll") for (int r = 0; r < 2; r++)                                                \
          _Pragma("unroll") for (int x = 0; x < NW; x++)                                             \
            W_[j][r][x] = __builtin_amdgcn_raw_buffer_load_b32(wrs, wvo[r][x] + j * WORDS * 4, so_, kAux); \
      }                                                                                              \
    }                                                                                                \
  } while (0)

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);
  constexpr int WS = EXL3R4D_PF_G_WS;
  static_assert(8 % WS == 0, "a pass's 8 steps hold whole weight rings");
  unsigned wb[WS][NPW][2][NW];

  xload(0);
#pragma unroll
  for (int u = 0; u < WS - 1; u++) GR_WLOAD(u < ksteps ? u : ksteps - 1, wb[u]);
  rot_math();
  rot_put();
  xload(nkb > 1 ? 1 : 0);
  lds_barrier();
  for (int kb = 0; kb < nkb; kb++) {
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
      if (kk == 4) rot_math();   // block kb + 1's rows, under the other waves' work (kk a constant)
      const int ks = kb * 8 + kk, ln = ks + WS - 1 < ksteps ? ks + WS - 1 : ksteps - 1;
      GR_WLOAD(ln, wb[(kk + WS - 1) % WS]);
      v8h af[MT];
#pragma unroll
      for (int i = 0; i < MT; i++) af[i] = frag_lds(alds + aoff + i * 16 * APITCH + kk * 16);
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const v8h wf = decode_frag<BITS, NW>(wb[kk % WS][j], plan.shift);
#pragma unroll
        for (int i = 0; i < MT; i++) acc[j][i] = mma(wf, af[i], acc[j][i]);
      }
      __builtin_amdgcn_sched_barrier(0);   // AM 4's lesson: keep the pass's 8 steps from being hoisted and spilled
    }
    lds_barrier();   // every wave is done reading block kb
    rot_put();       // block kb + 1 (the last pass rewrites its own, unread)
    xload(kb + 2 < nkb ? kb + 2 : nkb - 1);
    lds_barrier();
  }
#undef GR_WLOAD

  // The epilogue: each group its own LDS region, the shared one's statements (pf_epi_gu) with svh and down's suh
  // loaded here, once.
  v8h svp[epi_opw<NPW, 2>()][2];
  svh_early<NPW, 2>(w, wl, lane, cb, svp);
  const v2u dsr = pf_dn_suh(p, w, cb, lane);
  float* elds = (float*)((unsigned char*)lds + cb * kGrpEpi);
#pragma unroll
  for (int i = 0; i < MT; i++) {
    const int row = r0 + 16 * i + nl;
    const bool in_run = row < r1;
    const int tok = p.sorted[in_run ? row : r0];
    v8f blk[NPW];
#pragma unroll
    for (int j = 0; j < NPW; j++) blk[j] = acc[j][i];
    pf_epi_gu<NPW>(p, w, blk, wl, lane, cb, row, in_run, in_run && tok >= 0, elds, svp, dsr, wl, 8);
  }
}

template <int BITS, int M, int MTX, int NCB, bool XF>
static __device__ __forceinline__ void pf_grp_dispatch(int mt, const Exl3r4dGemm& p, const Planes& w, int r0, int r1,
                                                       float* lds) {
  if constexpr (M == MTX) {
    pf_grp_body<BITS, M, NCB, XF>(p, w, r0, r1, lds);
  } else {
    if (mt <= M)
      pf_grp_body<BITS, M, NCB, XF>(p, w, r0, r1, lds);
    else
      pf_grp_dispatch<BITS, M + 1, MTX, NCB, XF>(mt, p, w, r0, r1, lds);
  }
}

// One workgroup = map entry blockIdx.y if its expert's class is NCB column blocks wide (the other class's launch takes
// the rest); the zero entries in the launch with ZERO = true.
template <int BITS, int MTX, int NCB, bool ZERO>
__global__ void __launch_bounds__(8 * NCB * WAVE32) exl3r4d_pf_grp_kernel(Exl3r4dPrefill a) {
  EXL3R4D_KARG_PREFETCH();
  extern __shared__ float lds[];
  const Exl3r4dGemm& p = a.g;
  const int k = (int)blockIdx.y;
  if (k >= (int)a.map[0]) return;
  const uint4 ent = ((const uint4*)a.map)[1 + k];
  const int r0 = (int)ent.y, r1 = (int)ent.z, ncol = (int)(p.y_cols / 128);
  if (ent.x == EXL3R4D_PF_ZERO) {
    if (ZERO)
      for (int cb = 0; cb < ncol; cb++) zero_rows(p, r0, r1, cb);
    return;
  }
  const Planes w = planes_of(p, (int)ent.x);
  if (w.rows / 256 != NCB) return;   // the other class's launch
  const int mt = (r1 - r0 + 15) / 16;
  if (p.x_bf16)
    pf_grp_dispatch<BITS, 1, MTX, NCB, false>(mt, p, w, r0, r1, lds);
  else
    pf_grp_dispatch<BITS, 1, MTX, NCB, true>(mt, p, w, r0, r1, lds);
  for (int cb = NCB; cb < ncol; cb++) zero_rows(p, r0, r1, cb);   // the narrower class: zeros past its width
}

// AM 5's launches: one a class width present (1..3 column blocks), the first also zeroing the rows no run owns.
template <int BITS>
static int pf_grp_launch_k(const Exl3r4dPrefill* a, void** args, hipStream_t st) {
  constexpr int X = EXL3R4D_PF_MTX;
  bool seen[4] = {false, false, false, false}, zero = true;
  for (int c = 0; c < a->g.n_class; c++) {
    const int ncb = (int)(a->g.cls[c].rows / 256);
    if (ncb < 1 || ncb > 3) return RAD_E_UNSUPPORTED;
    if (seen[ncb]) continue;
    seen[ncb] = true;
    const void* fn = ncb == 3 ? (zero ? (const void*)exl3r4d_pf_grp_kernel<BITS, X, 3, true>
                                      : (const void*)exl3r4d_pf_grp_kernel<BITS, X, 3, false>)
                   : ncb == 2 ? (zero ? (const void*)exl3r4d_pf_grp_kernel<BITS, X, 2, true>
                                      : (const void*)exl3r4d_pf_grp_kernel<BITS, X, 2, false>)
                              : (zero ? (const void*)exl3r4d_pf_grp_kernel<BITS, X, 1, true>
                                      : (const void*)exl3r4d_pf_grp_kernel<BITS, X, 1, false>);
    const size_t rows = (size_t)(16 * X + 8) * (2 * 128 + 8) * 2, epi = (size_t)ncb * kGrpEpi;   // + surplus rows
    if (hipLaunchKernel(fn, dim3(1u, (unsigned)a->cap), dim3((unsigned)(8 * ncb * WAVE32)), args,
                        rows > epi ? rows : epi, st) != hipSuccess)
      return RAD_E_DEVICE;
    zero = false;
  }
  return RAD_OK;
}
static int pf_grp_launch(int bits, const Exl3r4dPrefill* a, void** args, hipStream_t st) {
  switch (bits) {
    case 2: return pf_grp_launch_k<2>(a, args, st);
    case 3: return pf_grp_launch_k<3>(a, args, st);
    case 4: return pf_grp_launch_k<4>(a, args, st);
    case 5: return pf_grp_launch_k<5>(a, args, st);
    case 6: return pf_grp_launch_k<6>(a, args, st);
    case EXL3R4D_K25: return pf_grp_launch_k<EXL3R4D_K25>(a, args, st);
    default: return RAD_E_DTYPE;
  }
}

// ----------------------------------------------------------------------------- the prep, by map entry

// The prefill form's input rotation: the prep's rows (prep_rotate, the same bits as the decode band's prep kernel) for
// the rows the map's entries own. Workgroup = (8 work units, entry blockIdx.y); a unit = one (matrix, 128-feature
// block) task of the entry's rows r0 + rs, r0 + rs + PF_PREP_RS, ..: every task of every row is one wave's work,
// spread over the device by entry and task. The decode band's prep kernel takes one workgroup an expert, so at prefill
// a hot expert's hundreds of rows run on one workgroup: 743.5 us of gate_up tp2 M=2048 in a trace.
// EXL3R4D_PF_PREP_RS: rows a prep wave strides by (its task over rows rs, rs + RS, ..): RS waves share an entry's task.
#ifndef EXL3R4D_PF_PREP_RS
#define EXL3R4D_PF_PREP_RS 4
#endif
// Batches of 8 rows (their sorted entries in one lane-parallel load) with the lean rotation below (byte-identical):
// prep 330 -> 289 us at gate_up tp2 M=2048; gate_up -2.4..-2.6% at 2048, -4.7..-4.8% at 4096, +-0.3% at 512/1024. The
// batch without the lean rotation leaves the prep at 330 us; row stride 1 (322) and 2 (286, = 4) change little: the
// per-entry setup is not the prep's limit; it is near its own issue bound.
#ifndef EXL3R4D_PF_PREP_B   // rows a wave's prep batch (1: one row at a time)
#define EXL3R4D_PF_PREP_B 8
#endif
// EXL3R4D_PF_PREP_FAST: the prep's rotation as rotate128_math statement for statement (the same rounding
// points: x * suh rounded to f16, the transform in f32, x 1/sqrt(128) to f16) but its lane exchanges as DPP moves with
// no `old` operand (update_dpp(0, ..) makes the compiler zero a register before every one: ~14 of the rotation's ~84
// VALU) and x's format a template constant. The prep is bound by its own instructions (338 us at M=2048 with its rows
// cache-resident). Only this file's prep kernel uses it.
#ifndef EXL3R4D_PF_PREP_FAST   // on, with EXL3R4D_PF_PREP_B (see there)
#define EXL3R4D_PF_PREP_FAST 1
#endif
template <int CTRL>
static __device__ __forceinline__ float pf_dpp(float v) {   // row_xmask lane exchange, no `old` (every lane is fed)
  return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), CTRL, 0xF, 0xF, true));
}
template <int CTRL>
static __device__ __forceinline__ void pf_fwht_stage(float (&v)[4], float sgn) {
#pragma unroll
  for (int q = 0; q < 4; q++) v[q] = fmaf(v[q], sgn, pf_dpp<CTRL>(v[q]));
}
template <bool XF>
static __device__ __forceinline__ v2u pf_rotate128(v2u xr, v2u sr, int lane) {
  float v[4];
  const unsigned xs[2] = {xr[0], xr[1]}, ss[2] = {sr[0], sr[1]};
#pragma unroll
  for (int q = 0; q < 4; q++) {
    const unsigned short xb = (unsigned short)(xs[q >> 1] >> (16 * (q & 1)));
    const unsigned short sb = (unsigned short)(ss[q >> 1] >> (16 * (q & 1)));
    const float xv = XF ? (float)__builtin_bit_cast(_Float16, xb) : __builtin_bit_cast(float, (unsigned)xb << 16);
    v[q] = (float)(_Float16)(xv * (float)__builtin_bit_cast(_Float16, sb));
  }
  { const float a = v[0], b = v[1], c = v[2], d = v[3];
    v[0] = a + b; v[1] = a - b; v[2] = c + d; v[3] = c - d;
    const float e0 = v[0], e1 = v[1];
    v[0] = e0 + v[2]; v[1] = e1 + v[3]; v[2] = e0 - v[2]; v[3] = e1 - v[3]; }
  pf_fwht_stage<0x161>(v, (lane & 1) ? -1.f : 1.f);
  pf_fwht_stage<0x162>(v, (lane & 2) ? -1.f : 1.f);
  pf_fwht_stage<0x164>(v, (lane & 4) ? -1.f : 1.f);
  pf_fwht_stage<0x168>(v, (lane & 8) ? -1.f : 1.f);
  {
    const float sgn = (lane & 16) ? -1.f : 1.f;
#pragma unroll
    for (int q = 0; q < 4; q++) v[q] = fmaf(v[q], sgn, x16_f(v[q]));
  }
  constexpr float RS = 0.08838834764831845f;
  union { _Float16 h[4]; v2u u; } o;
#pragma unroll
  for (int q = 0; q < 4; q++) o.h[q] = (_Float16)(v[q] * RS);
  return o.u;
}
// EXL3R4D_PF_PREP_FLAT: the batch with NO branch inside it, so its B rotations are one block the compiler
// interleaves (each is a 7-stage dependent chain; split by a branch a row they ran one at a time, ~115 instructions and
// a dozen s_delay_alu a row). A row past the entry is clamped to the entry's last row -- the same inputs, so the same
// bytes written again; a dropped row selects zeros; x's format a template constant, dispatched once a kernel.
#ifndef EXL3R4D_PF_PREP_FLAT
#define EXL3R4D_PF_PREP_FLAT 0
#endif
template <int MATS, bool XF>
static __device__ __forceinline__ void pf_prep_flat(const Exl3r4dPrefill& a, int lo, int hi, int rbase) {
  const Exl3r4dGemm& p = a.g;
  constexpr int B = EXL3R4D_PF_PREP_B, RS = EXL3R4D_PF_PREP_RS;
  const int lane = threadIdx.x % WAVE32, nw = (int)blockDim.x / WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x / WAVE32));
  for (int k = lo + (int)blockIdx.y; k < hi; k += (int)gridDim.y) {
    const uint4 ent = ((const uint4*)a.map)[1 + k];
    if (ent.x == EXL3R4D_PF_ZERO) continue;
    const Planes w = planes_of(p, (int)ent.x);
    const int blocks = w.cols / 128, tasks = MATS * blocks, u = (int)blockIdx.x * nw + wave;
    const int t = u % tasks, rs = u / tasks;
    if (rs >= RS) continue;
    const int m = t / blocks, blk = t - m * blocks, z = (int)ent.z;
    const v2u sr = *(const v2u*)((m ? w.suh1 : w.suh) + blk * 128 + 4 * lane);
    _Float16* const dst = (_Float16*)p.xr + (size_t)m * w.cols + blk * 128 + 4 * lane;
    for (int rb = (int)ent.y + rs; rb < z; rb += RS * B) {
      const int tkl = p.sorted[min(rb + RS * lane, z - 1)];   // lane i: row rb + RS i, clamped (unconditional)
      int tk[B];
      v2u xv[B];
#pragma unroll
      for (int i = 0; i < B; i++) tk[i] = __builtin_amdgcn_readlane(tkl, i);
#pragma unroll
      for (int i = 0; i < B; i++) {
        const int xrow = MATS == 2 ? min(max(tk[i], 0) / p.top_k, p.n_tok - 1) : min(rb + RS * i, z - 1);
        xv[i] = *(const v2u*)((const unsigned short*)p.x + (size_t)xrow * p.x_pitch + blk * 128 + 4 * lane);
      }
#pragma unroll
      for (int i = 0; i < B; i++) {
        const v2u r_ = pf_rotate128<XF>(xv[i], sr, lane);
        const v2u o = tk[i] >= 0 ? r_ : (v2u)(0u);
        st_row<kStXr>(dst + (size_t)(min(rb + RS * i, z - 1) - rbase) * p.xr_pitch, o);
      }
    }
  }
}

template <int MATS>
__global__ void __launch_bounds__(256) exl3r4d_pf_prep_kernel(Exl3r4dPrefill a) {
  EXL3R4D_KARG_PREFETCH();
  const Exl3r4dGemm& p = a.g;
  int lo, hi, rbase;
  pf_chunk(a, a.chunk_c, lo, hi, rbase);
  if constexpr (EXL3R4D_PF_PREP_FLAT) {
    if (p.x_bf16) pf_prep_flat<MATS, false>(a, lo, hi, rbase);
    else          pf_prep_flat<MATS, true>(a, lo, hi, rbase);
    return;
  }
  const int lane = threadIdx.x % WAVE32, nw = (int)blockDim.x / WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x / WAVE32));
  // grid.y = the map's capacity unchunked (one entry a workgroup row), EXL3R4D_PF_PGY chunked (a loop over them)
  for (int k = lo + (int)blockIdx.y; k < hi; k += (int)gridDim.y) {
    const uint4 ent = ((const uint4*)a.map)[1 + k];
    if (ent.x == EXL3R4D_PF_ZERO) continue;
    const Planes w = planes_of(p, (int)ent.x);
    const int blocks = w.cols / 128, tasks = MATS * blocks, u = (int)blockIdx.x * nw + wave;
    const int t = u % tasks, rs = u / tasks;
    if (rs >= EXL3R4D_PF_PREP_RS) continue;
    const int m = t / blocks, blk = t - m * blocks;
    if constexpr (EXL3R4D_PF_PREP_B > 1 || EXL3R4D_PF_PREP_FAST) {
      // PREP_B rows a batch: their sorted entries, then their x loads, then the rotations (prep_rotate's math and
      // operands, so the same bits) -- in a thread trace, one row at a time a wave waits on every row's loads.
      const v2u sr = *(const v2u*)((m ? w.suh1 : w.suh) + blk * 128 + 4 * lane);
      for (int rb = (int)ent.y + rs; rb < (int)ent.z; rb += EXL3R4D_PF_PREP_RS * EXL3R4D_PF_PREP_B) {
        int tk[EXL3R4D_PF_PREP_B];
        v2u xv[EXL3R4D_PF_PREP_B];
        // The batch's sorted entries in ONE lane-parallel load (lane i: row rb + RS i), then read out per row: one
        // round trip a batch. Loaded row by row they compile to a vector load + s_wait_loadcnt 0 each -- B round
        // trips in series before the first x load issues (the prep's dominant wait in a thread trace).
        int tkl = -1;
        {
          const int r = rb + EXL3R4D_PF_PREP_RS * lane;
          if (lane < EXL3R4D_PF_PREP_B && r < (int)ent.z) tkl = p.sorted[r];
        }
#pragma unroll
        for (int i = 0; i < EXL3R4D_PF_PREP_B; i++) tk[i] = __builtin_amdgcn_readlane(tkl, i);
#pragma unroll
        for (int i = 0; i < EXL3R4D_PF_PREP_B; i++) {
          const int xrow = MATS == 2 ? min(max(tk[i], 0) / p.top_k, p.n_tok - 1) : rb + EXL3R4D_PF_PREP_RS * i;
          xv[i] = tk[i] >= 0 ? *(const v2u*)((const unsigned short*)p.x + (size_t)xrow * p.x_pitch + blk * 128 + 4 * lane)
                             : (v2u)(0u);
        }
#pragma unroll
        for (int i = 0; i < EXL3R4D_PF_PREP_B; i++) {
          const int r = rb + EXL3R4D_PF_PREP_RS * i;
          if (r >= (int)ent.z) break;
          v2u o = (v2u)(0u);
          if constexpr (EXL3R4D_PF_PREP_FAST) {
            if (tk[i] >= 0) o = p.x_bf16 ? pf_rotate128<false>(xv[i], sr, lane) : pf_rotate128<true>(xv[i], sr, lane);
          } else {
            o = tk[i] >= 0 ? rotate128_math(xv[i], sr, p.x_bf16 == 0, lane) : (v2u)(0u);
          }
          st_row<kStXr>((_Float16*)p.xr + (size_t)(r - rbase) * p.xr_pitch + (size_t)m * w.cols + blk * 128 +
                     4 * lane, o);
        }
      }
      continue;
    }
    for (int r = (int)ent.y + rs; r < (int)ent.z; r += EXL3R4D_PF_PREP_RS) {
      const v2u o = prep_rotate<MATS>(p, w, r, m, blk, lane);
      *(v2u*)((_Float16*)p.xr + (size_t)(r - rbase) * p.xr_pitch + (size_t)m * w.cols + blk * 128 + 4 * lane) =
          o;
    }
  }
}

// The map AND the prep in ONE launch (PREP 2): workgroup 0 builds the map (pf_map_build), every other workgroup
// rotates PF_ROWS consecutive sorted rows -- each row's expert found by a binary search of expert_offset (rows no run
// owns are left alone: the GEMM zeroes them), every (matrix, 128-feature block) task of the rows one wave's work.
// Saves the map's own dispatch and its dependent-launch gap (~3 + ~2.8 us a call, the M = 32 trace).
#define EXL3R4D_PF_ROWS 16
template <int MATS>
__global__ void __launch_bounds__(256) exl3r4d_pf_mapprep_kernel(Exl3r4dPrefill a) {
  EXL3R4D_KARG_PREFETCH();
  __shared__ int s_w[32];
  __shared__ int s_e[EXL3R4D_PF_ROWS];
  if (blockIdx.x == 0) {
    pf_map_build(a, s_w);
    return;
  }
  const Exl3r4dGemm& p = a.g;
  const int tid = threadIdx.x, lane = tid % WAVE32, nw = (int)blockDim.x / WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);
  const int base = ((int)blockIdx.x - 1) * EXL3R4D_PF_ROWS;
  if (tid < EXL3R4D_PF_ROWS) {
    const int r = base + tid;
    int e = -1;
    if (r < p.T && r >= clamp_row(p.off[0], p.T) && r < clamp_row(p.off[p.n_expert], p.T)) {
      int lo = 0, hi = p.n_expert;   // the largest e with off[e] <= r (r is below off[n_expert])
      while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (clamp_row(p.off[mid], p.T) <= r) lo = mid; else hi = mid;
      }
      e = lo;
    }
    s_e[tid] = e;
  }
  __syncthreads();
  const int64_t cmax = EXL3R4D_CLS_MAX(p, cols);
  const int tmax = MATS * (int)(cmax / 128);
  for (int u = wave; u < EXL3R4D_PF_ROWS * tmax; u += nw) {
    const int i = u / tmax, t = u - i * tmax, r = base + i;
    const int e = __builtin_amdgcn_readfirstlane(s_e[i]);
    if (e < 0) continue;
    const Planes w = planes_of(p, e);
    const int blocks = w.cols / 128;
    if (t >= MATS * blocks) continue;   // the narrower class's K
    const int m = t / blocks, blk = t - m * blocks;
    const v2u o = prep_rotate<MATS>(p, w, r, m, blk, lane);
    st_row<kStXr>((_Float16*)p.xr + (size_t)(r) * p.xr_pitch + (size_t)m * w.cols + blk * 128 + 4 * lane, o);
  }
}

// ----------------------------------------------------------------------------- host

// The kernels this build serves: the configured MTX and NPW (exl3r4d_prefill.h), both forms; at K = 4 every A arm,
// at the other K (2, 3, 5, 6) the default one (AM 0) -- the arms are K = 4 readings.
template <int BITS, int MATS, bool RUN>
static const void* pf_gemm_fn(int am, bool ys) {
  constexpr int X = EXL3R4D_PF_MTX, N = EXL3R4D_PF_NPW;
  if constexpr (MATS == 2 && !RUN)
    if (am == 4) return (const void*)exl3r4d_pf_fused_kernel<BITS, 1, X, EXL3R4D_PF_FU_WPE>;
  if constexpr (MATS == 1 && !RUN)   // down's map form, the streamed y
    if (ys) return am == 0 ? (const void*)exl3r4d_pf_gemm_kernel<BITS, X, N, 1, 0, false, true> : nullptr;
  if constexpr (BITS != 4) return am == 0 ? (const void*)exl3r4d_pf_gemm_kernel<BITS, X, N, MATS, 0, RUN> : nullptr;
  return am == 0 ? (const void*)exl3r4d_pf_gemm_kernel<BITS, X, N, MATS, 0, RUN>
       : am == 1 ? (const void*)exl3r4d_pf_gemm_kernel<BITS, X, N, MATS, 1, RUN>
       : am == 2 ? (const void*)exl3r4d_pf_gemm_kernel<BITS, X, N, MATS, 2, RUN>
       : am == 3 ? (const void*)exl3r4d_pf_gemm_kernel<BITS, X, N, MATS, 3, RUN> : nullptr;
}
template <int BITS>
static const void* pf_gemm_fn_k(int gate_up, int run, int am, bool ys) {
  return run ? (gate_up ? pf_gemm_fn<BITS, 2, true>(am, ys) : pf_gemm_fn<BITS, 1, true>(am, ys))
             : (gate_up ? pf_gemm_fn<BITS, 2, false>(am, ys) : pf_gemm_fn<BITS, 1, false>(am, ys));
}
// XR: gate_up's map form with the copy row, AM 0 (the default A form).
#if EXL3R4D_XR_ON
static const void* pf_gemm_select_xr(int bits) {
  constexpr int X = EXL3R4D_PF_MTX, N = EXL3R4D_PF_NPW;
  switch (bits) {
    case 2: return (const void*)exl3r4d_pf_gemm_kernel<2, X, N, 2, 0, false, false, true>;
    case 3: return (const void*)exl3r4d_pf_gemm_kernel<3, X, N, 2, 0, false, false, true>;
    case 4: return (const void*)exl3r4d_pf_gemm_kernel<4, X, N, 2, 0, false, false, true>;
    case EXL3R4D_K25: return (const void*)exl3r4d_pf_gemm_kernel<EXL3R4D_K25, X, N, 2, 0, false, false, true>;
    default: return nullptr;
  }
}
#endif
static const void* pf_gemm_select(int bits, int gate_up, int run, int am, bool ys) {
  switch (bits) {
    case 2: return pf_gemm_fn_k<2>(gate_up, run, am, ys);
    case 3: return pf_gemm_fn_k<3>(gate_up, run, am, ys);
    case 4: return pf_gemm_fn_k<4>(gate_up, run, am, ys);
    case 5: return pf_gemm_fn_k<5>(gate_up, run, am, ys);
    case 6: return pf_gemm_fn_k<6>(gate_up, run, am, ys);
    case EXL3R4D_K25: return pf_gemm_fn_k<EXL3R4D_K25>(gate_up, run, am, ys);
    default: return nullptr;
  }
}

// AM 4 split (EXL3R4D_PF_FU_SPLIT): the kernel of the short entries (h 0) or of the rest (h 1).
template <int BITS>
static const void* pf_fused_split_k(int h) {
  constexpr int S = EXL3R4D_PF_FU_SPLIT > 0 && EXL3R4D_PF_FU_SPLIT < EXL3R4D_PF_MTX ? EXL3R4D_PF_FU_SPLIT : 1;
  if constexpr (EXL3R4D_PF_FU_SPLIT == 0)
    return nullptr;
  else
    return h ? (const void*)exl3r4d_pf_fused_kernel<BITS, S + 1, EXL3R4D_PF_MTX < S + 1 ? S + 1 : EXL3R4D_PF_MTX, 0>
             : (const void*)exl3r4d_pf_fused_kernel<BITS, 1, S, EXL3R4D_PF_FU_WPE>;
}
static const void* pf_fused_split(int bits, int h) {
  switch (bits) {
    case 2: return pf_fused_split_k<2>(h);
    case 3: return pf_fused_split_k<3>(h);
    case 4: return pf_fused_split_k<4>(h);
    case 5: return pf_fused_split_k<5>(h);
    case 6: return pf_fused_split_k<6>(h);
    case EXL3R4D_K25: return pf_fused_split_k<EXL3R4D_K25>(h);
    default: return nullptr;
  }
}

// LDS a GEMM launch asks for: the epilogue's tiles of one 16-row block, or with AM 1 / 2 the rows of one 128-wide K
// block if larger (the two never live at once).
static int64_t pf_lds(int mats, int mtx, int am) {
  const int64_t epi = (int64_t)mats * 8 * 256 * 4, rows = am ? (int64_t)mtx * 16 * (mats * 128 + 8) * 2 : 0;
  return epi > rows ? epi : rows;
}

extern "C" int exl3r4d_launch_prefill(const Exl3r4dPrefill* a, int gate_up, int bits, int am, int prep, int run,
                                      void* stream) {
  constexpr int mtx = EXL3R4D_PF_MTX, npw = EXL3R4D_PF_NPW;
  const bool rot = am == 1 || am == 3 || am == 4 || am == 5;   // the rotation in the GEMM: no prep, no rows read
  const bool pair = EXL3R4D_PF_PAIR && gate_up && !run && am == 0;
  if (a->fm != 16 * mtx * (pair ? 2 : 1) || a->g.sk != 1 || (!rot && !a->g.xr) || (!run && (!a->map || a->cap < 1)))
    return RAD_E_INVAL;
  if (a->g.n_expert < 1 || a->g.n_expert > 4096) return RAD_E_SHAPE;   // the map kernel: <= 4 experts a thread
  const int mats = gate_up ? 2 : 1, wv = am == 4 ? 16 : 8 * mats / npw;
  if (am >= 4 && (!gate_up || run || a->chunk_n > 1)) return RAD_E_UNSUPPORTED;   // the fused GEMMs: gate_up's map form
  const bool ys = a->ystream != 0;
  if (ys && (gate_up || run || am != 0)) return RAD_E_INVAL;   // the streamed y: down's map form only
  // XR: the copy row's kernel when the launch carries a ring (prefill.c sets xr_mode 5 only with one); else as before
#if EXL3R4D_XR_ON
  const bool xr = gate_up && !run && am == 0 && !pair && (xr_mode_of(*a) & 4) && xr_ring_of(*a) > 0 && a->order &&
                  (a->chunk_n <= 1 ? 1 : a->chunk_n) <= 64;
  const void* fxr = xr ? pf_gemm_select_xr(bits) : nullptr;
#else
  const bool xr = false;
  const void* fxr = nullptr;
#endif
  const void* fn = am == 5 ? (const void*)exl3r4d_pf_map_kernel
                 : fxr     ? fxr
                           : pf_gemm_select(bits, gate_up, run, am, ys);
  const unsigned xrow = fxr ? 1u : 0u;   // the copy row
  if (!fn) return (bits < 2 || bits > 6) && bits != EXL3R4D_K25 ? RAD_E_DTYPE : RAD_E_UNSUPPORTED;
  (void)xr;
  int64_t smem = pf_lds(mats, mtx, am);
  if (!gate_up && EXL3R4D_PF_EPI_DN2 && smem < 2 * 8 * 256 * 4) smem = 2 * 8 * 256 * 4;   // two epilogue regions
  if (gate_up && a->g.h_rot && smem < kHrotStage + 4096) smem = kHrotStage + 4096;   // v3 EPI 1's staged h block
  if (ys && smem < kYStage + 4096) smem = kYStage + 4096;   // the streamed y's staged block
  if (smem > EXL3R4D_LDS_MAX) return RAD_E_SHAPE;
  Exl3r4dPrefill p = *a;
  void* args[] = {&p};
  hipStream_t st = (hipStream_t)stream;
  if (prep == 3 && (gate_up || !a->g.h_rot || rot || a->g.xr != a->g.x)) return RAD_E_INVAL;   // no prep: h_rot's down
  if (!run && prep == 2 && !rot) {   // the map and the prep by row in one launch
    const void* mk = gate_up ? (const void*)exl3r4d_pf_mapprep_kernel<2> : (const void*)exl3r4d_pf_mapprep_kernel<1>;
    if (hipLaunchKernel(mk, dim3((unsigned)(1 + (a->g.T + EXL3R4D_PF_ROWS - 1) / EXL3R4D_PF_ROWS)), dim3(256), args, 0,
                        st) != hipSuccess)
      return RAD_E_DEVICE;
  } else if (!run && hipLaunchKernel((const void*)exl3r4d_pf_map_kernel, dim3(1), dim3(1024), args, 0, st) !=
             hipSuccess) {
    return RAD_E_DEVICE;
  }
  if (!run && a->chunk_n > 1) {   // the scratch cap: the map form in chunks by row, prep then GEMM a chunk
    if (prep != 1 || rot || !a->ctab || a->chunk_rows < 1 || a->chunk_gy < 1) return RAD_E_UNSUPPORTED;
    const int64_t cols = EXL3R4D_CLS_MAX(a->g, cols);
    const int64_t units = (int64_t)mats * (cols / 128) * EXL3R4D_PF_PREP_RS;
    const void* pk = gate_up ? (const void*)exl3r4d_pf_prep_kernel<2> : (const void*)exl3r4d_pf_prep_kernel<1>;
    for (int c = 0; c < a->chunk_n; c++) {
      p.chunk_c = c;
      if (hipLaunchKernel(pk, dim3((unsigned)((units + 7) / 8), (unsigned)EXL3R4D_PF_PGY), dim3(256), args, 0, st) !=
              hipSuccess ||
          hipLaunchKernel(fn, dim3((unsigned)(a->g.y_cols / 128), (unsigned)a->chunk_gy + xrow),
                          dim3((unsigned)(wv * WAVE32)), args, (size_t)smem, st) != hipSuccess)
        return RAD_E_DEVICE;
      if (pair && hipLaunchKernel(pf_pair_fn(bits), dim3((unsigned)(a->g.y_cols / 128), (unsigned)a->chunk_gy),
                                  dim3(512u), args, (size_t)(16384 + 2 * kGrpEpi), st) != hipSuccess)
        return RAD_E_DEVICE;
    }
    return RAD_OK;
  }
  if ((!run && prep == 2 && !rot) || prep == 3) {
    // done above (2), or none (3)
  } else if (!rot && (run || prep == 0)) {   // the decode band's prep kernel: one workgroup a live expert (find_run)
    const int rc = exl3r4d_launch_prep(&a->g, gate_up, stream);
    if (rc != RAD_OK) return rc;
  } else if (!rot) {                     // by map entry
    const int64_t cols = EXL3R4D_CLS_MAX(a->g, cols);
    const int64_t units = (int64_t)mats * (cols / 128) * EXL3R4D_PF_PREP_RS;
    const void* pk = gate_up ? (const void*)exl3r4d_pf_prep_kernel<2> : (const void*)exl3r4d_pf_prep_kernel<1>;
    if (hipLaunchKernel(pk, dim3((unsigned)((units + 7) / 8), (unsigned)a->cap), dim3(256), args, 0, st) != hipSuccess)
      return RAD_E_DEVICE;
  }
  const unsigned gy = run ? (unsigned)a->g.nslot : (unsigned)a->cap;
  if (am == 5) return pf_grp_launch(bits, a, args, st);
  if (am == 4 && EXL3R4D_PF_FU_SPLIT > 0 && EXL3R4D_PF_FU_SPLIT < mtx) {   // short entries, then the rest
    const void* f2[2] = {pf_fused_split(bits, 0), pf_fused_split(bits, 1)};
    for (int h = 0; h < 2; h++)
      if (!f2[h] || hipLaunchKernel(f2[h], dim3(1u, gy), dim3(512u), args, (size_t)smem, st) != hipSuccess)
        return RAD_E_DEVICE;
    return RAD_OK;
  }
  // down, G column blocks a workgroup: the last group may be partial (K is a multiple of 128: even step counts)
  const int gdn = !gate_up && !run && am == 0 ? EXL3R4D_PF_DN_G : 1;
  if (pair && hipLaunchKernel(pf_pair_fn(bits), dim3((unsigned)(a->g.y_cols / 128), gy), dim3(512u), args,
                              (size_t)(16384 + 2 * kGrpEpi), st) != hipSuccess)
    return RAD_E_DEVICE;
  if (hipLaunchKernel(fn, dim3(am == 4 ? 1u : (unsigned)((a->g.y_cols / 128 + gdn - 1) / gdn), gy + xrow), dim3((unsigned)(wv * WAVE32)), args, (size_t)smem,
                      st) != hipSuccess)
    return RAD_E_DEVICE;
  return RAD_OK;
}

// The copy ring's counters as a reading (prefill.c calls exl3r4d_xr_note on every launch with an XR mode): every 256th
// such launch of a rank prints the rank's pinned slot (its last copy) and copies the rank's device counters into it on
// the launch's stream (the engine's queue takes the async copy). exl3r4d_xr_rank_init (prefill.c's
// rad_host_pool_ranges, at start, with the rank's card current) allocates the device counters and the pinned slots. No
// allocation or synchronisation on the launch path.
#if EXL3R4D_XR_ON
static unsigned long long* g_xr_hslot = nullptr;   // pinned [64 ranks][8]
static unsigned long long* g_xr_dcnt[64];
static unsigned long long g_xr_launch[64];
extern "C" unsigned long long* exl3r4d_xr_rank_init(int rank) {
  if (rank < 0 || rank >= 64) return nullptr;
  if (!g_xr_hslot) {
    if (hipHostMalloc((void**)&g_xr_hslot, 64 * 8 * sizeof(unsigned long long), hipHostMallocDefault) != hipSuccess) {
      g_xr_hslot = nullptr;
      return nullptr;
    }
    memset(g_xr_hslot, 0, 64 * 8 * sizeof(unsigned long long));
  }
  if (!g_xr_dcnt[rank]) {
    void* p = nullptr;
    if (hipMalloc(&p, 8 * sizeof(unsigned long long)) != hipSuccess) return nullptr;
    if (hipMemset(p, 0, 8 * sizeof(unsigned long long)) != hipSuccess) return nullptr;
    g_xr_dcnt[rank] = (unsigned long long*)p;
  }
  return g_xr_dcnt[rank];
}
extern "C" void exl3r4d_xr_note(int rank, void* stream) {
  if (rank < 0 || rank >= 64 || !g_xr_hslot || !g_xr_dcnt[rank]) return;
  if (g_xr_launch[rank]++ % 256 != 0) return;
  const unsigned long long* c = g_xr_hslot + 8 * rank;
  fprintf(stderr, "exl3r4d XR (reading): rank %d after %llu XR launches: chunks with a ring %llu, host-read heads %llu, copied %llu "
          "(%.1f MiB), workgroups reading a copy %llu, past the spin %llu; host-read groups %llu of %llu ordered\n", rank,
          g_xr_launch[rank] - 1, c[0], c[1], c[2], (double)c[3] * 256.0 / 1048576.0, c[4], c[5], c[6], c[7]);
  (void)hipMemcpyAsync(g_xr_hslot + 8 * rank, g_xr_dcnt[rank], 8 * sizeof(unsigned long long), hipMemcpyDeviceToHost,
                       (hipStream_t)stream);
}
#endif

// ============================================================================= Measurements behind the defaults
//
// What every default and every refused alternative of this band rests on. Read it before changing a default or
// proposing a form already listed here. Conditions: one gfx1201 card of 64 CUs, memory clock 1349 MHz under load,
// tools/perfmoe.cpp --exl3-row exl3r4d_moe_*_pf (48 routing draws over 2 real-size pools, 500 ms warm-up, libr4d's
// moe_gemm_w4a8 on the same routing in the same process as the baseline, the two interleaved ABAB), tp2 = rank 0's
// parity split (even experts 384 wide, odd 256), 512 experts, top-10, us a dispatch, median of 5. Every adopted change
// was checked byte-identical to the build before it by output hash. The decode band's counterpart is the measurements
// at the end of exl3r4d.hip.
//
// WHERE THE BAND STANDS against libr4d, h_rot 1 (interface v3), exl3r4d / libr4d us:
//   tp2      M=17         32           64           512          2048          4096
//   gate_up  197.0/195.6  305.2/304.9  458.8/463.1  811.1/724.9  1640.8/1004.6 2930.8/1736.2
//   down     103.7/97.0   151.6/145.3  225.5/217.9  399.1/428.5  698.7/750.1   1174.9/1196.5
//   Traced, the GEMMs are at or below libr4d's below M ~ 1024 (gate_up M=512 GEMM 672.7 vs 710.9, M=32 287.6 vs
//   302.1; down M=32 144.4 vs 142.0); what is over is gate_up's PREP: 15 / 110.6 / 329 us at M = 32 / 512 / 2048,
//   i.e. writing the rotated rows (52 / 210 MB) at DRAM speed, plus the GEMM reading them back at M=2048. The issue
//   floor at M=2048 (minimum WMMA 0.38 ms + decode 0.46 ms at 100% issue) is ~0.84 ms against libr4d's whole 0.95-1.0
//   ms: closing the prefill gate_up gap needs the rows' round trip gone, not tuning.
//
// THE FORMS (prefill.c chooses, by slots per expert):
//   run form  gate_up while 2 T <= 3 n_expert: one workgroup a (column block, live expert) via the decode band's
//             find_run, the whole run in passes of 16 MTX rows, the decode band's prep kernel. 2 launches.
//   map form  everything else: the map, the prep (gate_up: by map entry; down: map + prep by row in ONE launch), the
//             GEMM by map entry. 3 / 2 launches.
//
// WHERE THE TIME GOES (gate_up tp2 unless named):
//   trace, M=2048, the first form: GEMM 1329 us + a prep kernel of one workgroup an expert 744 us (the hottest expert's
//      443 rows x 40 tasks ran serially; hence the prep by map entry below) + map 3.4 us; libr4d 938 us.
//   trace with the prep by map entry and buffer loads: M=32 GEMM 283 us vs libr4d 301 (prep 22.5 + map 3.0 were the
//      excess); M=512 GEMM 674.5 vs 709.7 (prep 99.5); M=2048 GEMM 1291 vs 944, prep 314 = writing the rotated rows:
//      the prep writes T x 2 x 2560 x 2 B (210 MB at M=2048) and the GEMM reads them back -- 420 MB of DRAM beside 432
//      MB of weights. THAT ROUND TRIP AND THE GEMM'S ISSUE (~1.0 ms of decode + f16 WMMA by count at M=2048) ARE THE
//      GAP.
//   deletions: at M=2048 decode -3%, WMMA -7% (of the total); M=512 -11% / -9%; weight loads -12% at M=32/512/2048. At
//      h_rot 1: decode deleted -10.6% (M=2048) / -9.8% (4096) / -3.5% (512); WMMA -4.2 / -4.5 / +2.4%; weight loads
//      -8.8 / -12.0 / -13.3%. No single component is the cost.
//   rows folded onto a cache-resident region (a deletion build): M=2048 -15%, M=4096 -13%: the rows' round
//      trip is worth at most that even if free.
//   A loads deleted (a deletion build): +0.2% at 2048 (-5.3% at 4096): A traffic is NOT the cost; A + W loads
//      deleted -5.7% / -9.9%.
//   thread trace at 2048, before the deeper weight prefetch: s_wait_loadcnt 54% of summed cycles, WMMA 7.8%, VALU 15%
//      -- the waits are for loads issued one k-step earlier. With it (the defaults): the gate_up GEMM is ISSUE
//      bound (VALU 62% of cycles, the decode ~51%, WMMA 11%, epilogue barriers 11%, load waits 7.5%); down's GEMM is
//      memory-latency bound (load waits 56%).
//   f16 WMMA and VALU SERIALIZE on a SIMD at saturation (tests/gpu/prefill/mix.hip: the same waves take 0.94-0.99 of
//      the sum, split waves half the sum), so the floor is WMMA + decode + rotation VALU, not DRAM.
//
// ADOPTED (the defaults):
//   the map form's prep by map entry (EXL3R4D_PF_PREP gate_up digit 1): gate_up M=2048 -16%, M=4096 -20%.
//   one form for the band, the decode kernels off above M = 16 (EXL3R4D_PF_XN_* 0): down M=32 -15%, M=64 -20%; down on
//      the decode kernels with h_rot 1 also loses (M=32 +2.6%, M=64 +6.3%).
//   buffer loads (EXL3R4D_PF_BUF 1) and the shared header's state hash: -2..-9% each, both ops.
//   the run form for gate_up below the crossover (EXL3R4D_PF_RN_GU 3): M=17 -3.6%, M=32 -2.1%.
//   down: map + prep in one launch (EXL3R4D_PF_PREP down digit 2): -1.3..-1.7% (small, consistent).
//   gate_up's weights 4 k-steps ahead (EXL3R4D_PF_WSTAGES_GU 4): -1.0/-2.4/-4.0/-9.0% at M=512/1024/2048/4096,
//      replicated; down stays at 2 (4: +0.2..+3.2%, 208 VGPRs); 8 steps +13%, both operands 4 deep (PF_STAGES 4) +6%.
//   down's svh before the K loop (EXL3R4D_PF_SVH_EARLY_DN 1): -0.4..-1.5% at M=512..4096; gate_up +-0.2% (off). With
//      down's suh too (the local epilogue pf_epi_gu, EXL3R4D_PF_EPI_PRE, byte-identical to epilogue_block): gate_up
//      -0.3% at 2048. The two adopted together: gate_up 826/1053/1592/2978 -> 818/1027/1525/2700, down
//      388/461/671/1137 -> 385/460/667/1123 us at M=512/1024/2048/4096.
//   THE SCRATCH CAP (EXL3R4D_PF_SCRATCH_MB 46): above 46 MiB of rows the map form runs in balanced chunks by row (prep
//      then GEMM a chunk, one rows buffer; the map kernel writes the chunk table). Declared scratch at the 2048-token
//      band top falls from 200 to ~46 MiB a card (the baseline's: 47); that VRAM is KV the cache lends the expert slab.
//      Per op vs uncapped, tp2: M=512 +3.4%, 768 -6.8%, 1024 -4.0%, 2048 +0.2%, 4096 +3.6% (tp1 -0.4..+3.0%); 32 MiB
//      +5.6% at 2048. Chunks of RMAX rows with a short last one instead of balanced chunks: M=512 +6.0%. With
//      EXL3R4D_PF_LPT 2: uncapped tp2 +25.5% at 2048, 96 MiB +6.7%, 56 MiB tp1 M=512 +14.5%, 32 / 40 MiB slower: 46
//      stays (xr must stay in the 64 MB last-level cache).
//   EXL3R4D_PF_LPT 2 (exl3r4d_prefill.h; pf_order_build), each launch takes whole experts longest first: tp2 -2.0 /
//      -3.9 / -4.35%, tp1 -1.0 / -0.1 / -1.3% at M = 512 / 1024 / 2048; same bits. LPT 1 (entries longest first, an
//      expert's remainder split from its tile): tp1 +0.2..+0.4% at M >= 1024. See "gate_up's map form at M >= 512".
//   the prep in batches of 8 rows with the lean rotation (EXL3R4D_PF_PREP_B 8, EXL3R4D_PF_PREP_FAST 1): prep 330 ->
//      289 us at M=2048; gate_up -2.4..-2.6% at 2048, -4.7..-4.8% at 4096, +-0.3% at 512/1024.
//   THE STREAMED y (the YS kernels, pf_epi_dn_ys; prefill.c EXL3R4D_PF_YNT_MB 8): see "what the band leaves in L2".
//
// THE KERNELS ACROSS K (2..6, mul1): Plan<K> / decode_frag<K> are the shared decode's; the A forms AM 1..3 are
//   instantiated at K = 4 only. With global word loads and the general state hash, the K loop a wave (MT 4, 16 WMMA)
//   was K 2 219 instructions, 3 255, 4 240, 6 272, no spills -- but K=3 ran +24..31% SLOWER than K=4 at M=17..64 on
//   25% fewer bytes (48% of bandwidth). Hence the buffer form at every K (EXL3R4D_PF_KX_BUF) + the shared header's
//   packed-pair hash (EXL3R4D_KX_PAIR). With them, tp2 K=3 vs K=4: gate_up +2.7/+0.5/-1.7/+0.3/+2.3%, down
//   +3.5/+3.9/+2.3/+1.1/+3.3% at M=17/32/64/512/2048 -- K=3 matches K=4 but is issue-bound (decode 262 vs 240
//   instructions a step) on its 25% fewer bytes; K=2 0..-13%, K=5 +18..30%, K=6 +35..45%. gate_up's run form at K = 4
//   only (EXL3R4D_PF_RN_GU_KX 0): at K = 3 its MT 1 loop compiles to 256 instructions with 22 s_delay_alu (the map
//   form's: 236 / 1) and ran +35%.
//
// REFUSED (kept as arms where the code still has them; each with what would reopen it):
//   rotate in the K loop (AM 1): gate_up M=2048 +5%, down +21..+38%; in the run form +56..+60%; in down's map form
//      +25..+64%. As written the rotation sits between two barriers each 128-K block with its loads exposed.
//   AM 3, the rotation in the K loop pipelined one pass ahead: gate_up M=2048 +25%, run form M=32 +81%, down
//      +49..+62% (the compiler schedules such a loop dependency-bound).
//   WHY THE ROTATION IN THE GEMM LOSES -- its MATH: per wave per K pass (64 WMMA) AM 0 960 instructions, AM 3 3228 (490
//      s_delay_alu), AM 3 with the math deleted 1030 (18); timed, M=512 810 / 1099 / 878 us, M=2048 1642 / 2037 /
//      1497. ~137 instructions a 128-wide row block, done again in each of the 3 (2) column-block workgroups: a
//      schedule fix removes the stalls, not the work. Even FREE it is worth <= 9% at M=2048 and loses at 512; a ring
//      rotating each row once keeps 1/3 of it (~break-even). Reopen only with a rotation several times cheaper than
//      rotate128_math.
//   AM 4, gate_up fused (each row rotated once, no prep, no rows scratch): 1103/1344/1879/3085 us, the rotation math
//      mid-pass 1123/1363/1899/3113, + weights 4 deep 1045/1288/1817/3006 vs base 826/1055/1599/3005 at
//      M=512/1024/2048/4096. Its deletions at 2048 (1834 us): no WMMA 1326, no decode 1297, no weight loads 1760, no
//      barriers 1842, no epilogue 1615, all four K-loop terms 709 -- serial sums; an MT-class split (FU_SPLIT) -1%.
//   AM 5 (EXL3R4D_PF_AM_MAP=5: the (entry, cb) workgroups merged into one of NCB groups of 8 waves sharing each row's
//      rotation in LDS; 213 VGPRs): 1076/1313/1812/2950 vs base 826/1057/1595/2987. Byte-identical.
//   A staged through LDS once a workgroup (AM 2): -1..+4% (the four waves' redundant A loads are L2 hits); once a
//      workgroup a pass: +23.5%. Its rows are offset for the scratch cap (without that it reads past the rows buffer
//      and faults the card).
//   PAIR (EXL3R4D_PF_PAIR: 128-row entries, entries of > 4 blocks on a 16-wave workgroup of two sets sharing each
//      tile's decode through LDS, a barrier a k-step): 888/1215/1854/3527 vs 818/1032/1524/2700; capped at 192 VGPRs
//      (PAIR_WPE 8) no better. Byte-identical. With AM 4 and AM 5 the third 16-wave design that synchronises inside
//      the K loop to lose: decode -> LDS -> barrier -> WMMA serialises a step.
//   MTX 8 NPW 1: +26% (16-wave workgroups at 217 VGPRs: one a WGP). MTX 8 NPW 2 spills. MTX 6 / MTX 8 + prefetch 1:
//      +4.7% at M=512 (6 waves a SIMD, not 7), -3.3% at 4096; MTX 8 prefetch 1 +11..+35%. MTX 8 with one decode for a
//      128-row entry would save 14% of decodes at 2048 and 31% at 4096: <= -1.5% / -3.0% before its own LDS and
//      barrier costs, so not built.
//   tile shapes vs MTX 4 / NPW 2 (826/1595/3001 us at M=512/2048/4096): MTX 3 825/1614/2937, MTX 2 826/1707/3132, MTX
//      2 NPW 4 832/1691/3105, MTX 3 NPW 4 831/1604/2869 -- none beats MTX 4 / NPW 2.
//   more waves a SIMD: EXL3R4D_PF_WPE 7 (216 VGPRs, 14 spills) +0.6%, 8 (192, 96 spills) +5.0%.
//   the MUL24 state hash: -0.1..-2% (superseded by the shared header's hash). Prefetch 4: -3%; plain weight loads
//      (EXL3R4D_PF_NT 0): -1.5% -- not adopted (see the L2 section below for what NT 0 costs down).
//   the run form for down (EXL3R4D_PF_RN_DN 0): +8..+14% (20 column blocks x every live expert scanning
//      expert_offset).
//   the map form in expert chunks of 16 / 32 MiB of rows (EXL3R4D_PF_CHUNK_MB): +4.5..+21%: each chunk's prep -> GEMM
//      is serialised with a short-grid tail. Reopen only with prep and GEMM in ONE launch.
//   G 128-column blocks a GEMM workgroup: down run form G 4 +2.0%, G 10 +12.4%, and the loop around the body costs
//      gate_up's run form +30% even at G 1 (VGPRs 215 -> 241). Down G column blocks a workgroup with the ring across
//      them (EXL3R4D_PF_DN_G, 204 VGPRs): G2 +0.3..+1.4%, G4 +2..+3%, G5 +3..+4%.
//   the prep of chunk c + 1 by extra workgroups of chunk c's GEMM launch, two rows buffers in the cap: +26% (64
//      workgroup rows) .. +67% (16) at M=2048, the same at 512..4096. The prep needs ~1/5 of the device for its time
//      and the latency-bound GEMM leaves no idle slots; dispatched after the GEMM's, the prep workgroups extend every
//      launch's tail.
//   the prep's rows batched alone (EXL3R4D_PF_PREP_B 4 / 8 without EXL3R4D_PF_PREP_FAST): prep 341 -> 330 us, op
//      unchanged: the prep writes 210 MB at ~630 GB/s, write-bandwidth bound. With the lean rotation it pays (above).
//   sched_group_barrier (EXL3R4D_PF_SGB 1 grouped: 239 instructions / 0 s_delay_alu; 2 a tile at a time: 246 / 3):
//      nothing over the compiler's 240 / 0, so not timed.
//
// WHAT THE BAND LEAVES IN L2. The op after this band's _pf ops ran +3..9 us slower than after libr4d's (measured as one
//   libr4d dispatch of the next layer timed after ours minus after libr4d's: perfmoe --probe-gap -1).
//   the cause: this band's own dirty rows (xr, h, y) stay in L2 past the op because the K loop's weight stream is
//      NONTEMPORAL (EXL3R4D_PF_NT 1) and does not displace them; libr4d's temporal stream writes its own back while it
//      runs. EXL3R4D_PF_NT 0 removes the penalty at every cell but costs down +6..+13 us at M = 32, so NT stays 1.
//      (The decode band's EXL3R4D_NT is a separate switch and does not affect this band.)
//   refused: last-use A loads (EXL3R4D_PF_ATH 3, TH_LOAD_LU) of the dead rows: penalty unchanged. WB stores (STH 3):
//      penalty up (down 8..17 us faster at M = 64, lines kept longer). NT stores of xr in the decode band's unit
//      (EXL3R4D_STH_XR 1, which writes the run form's xr too): the _pf chain gate_up -> down + next op nets +2.0 / -2.4
//      us at tp2 M = 32 / 64 and the decode chain M = 16 +7.5 us (its GEMM reads xr right after the prep); the map
//      form's own xr (EXL3R4D_PF_STH_XR 1): gate_up +53 / +225 us at tp2 M = 512 / 2048 (the GEMM must find its rows in
//      cache). NT h (EXL3R4D_PF_STH_H 1): within +-0.5 us of y alone.
//   adopted, THE STREAMED y (the YS kernels, pf_epi_dn_ys; prefill.c EXL3R4D_PF_YNT_MB 8): y staged in LDS, stored a
//      whole 256-byte row a wave instruction with TH_STORE_NT, when y >= 8 MiB. NT on the regular epilogue's partial
//      lines cost down +7..9 us at M = 32; staged, the staging alone is +-0.4 us. Chain + next op, us a layer, tp2 /
//      tp1: M=256 -6.5 / -14.2, 512 -34.7 / -46.3, 1024 -34.9 / -35.1, 2048 -26.1 / -15.9; at M = 32 / 64 / 128 (tp2)
//      -2.3 / +1.2 / +0.4: no gain, so the verify band keeps the regular kernels. Same bits. Op alone: down -2.6 /
//      -1.8% (tp1) and -2.9 / -1.9% (tp2) at M = 512 / 2048, +-0.3% elsewhere. In the engine: prefill of 8K / 24K
//      tokens +0.09 / +0.06% (under the measurement's resolution; ~0.2% expected), TTFT at 8 streams -1.1%, MTP decode
//      at 8 / 16 streams unchanged, output identical. The engine numbers show the direction, not a resolved step-time
//      change; EXL3R4D_PF_YNT_MB 0 turns it off.
//   rotate-on-load (no xr) was not built for this: AM 1 / 3 / 4 / 5 above already measured it, +56..+81% in the run
//      form.
//   against libr4d (base op time): gate_up tp2 0.98 / 0.95 / 1.03 / 1.13 / 1.50 at M = 32 / 64 / 128 / 512 / 2048, tp1
//      0.93 / 0.93 / 0.98 / 0.89 / 1.18; down 0.89..0.97 everywhere. The band's remaining gap is gate_up at M >= 512
//      (the floor reading above), not the verify band.
//
// GATE_UP'S MAP FORM AT M >= 512, where tp2 reads 1.50x libr4d and tp1 1.18x (M = 2048):
//   not the class mix: ONE class at width 640 / 384 / 256 reads 1.18 / 1.25 / 1.65x libr4d. Time is linear in the
//      width -- this kernel 3.18 us a column + ~520 us, libr4d 3.55 us a column + ~0 (M = 2048): the width-proportional
//      GEMM is faster than libr4d's; the tp2 excess is a WIDTH-INDEPENDENT ~520 us (M=512 ~181, 1024 ~312) = the prep
//      (~287 us: xr is T x 2 x K whatever the width) + ~240 us inside the 5 scratch-cap GEMM launches + gaps.
//   workgroup stamps (EXL3R4D_PF_WGSTAMP): a launch at tp2 M = 2048 is ~3.8 rounds of 96 resident workgroups; 50-56 us
//      of each launch's 250-270 is lost to its tail (64-row workgroups ~85 us, the grid in expert order). A
//      workgroup's time is mostly fixed: ~38 us + ~12 us a 16-row block (its expert tile's stream and decode). Hence
//      EXL3R4D_PF_LPT 2 above.
//   deletions at tp2 M = 2048: decode -4.4%, WMMA -1.9%, weight loads -9.6%, A loads -2.0%, epilogue -3.2%: nothing
//      dominates. The f16 WMMA floor (22-26% of 199 TF/s reached) is a floor (no precision change).
//   what remains: the prep (~19% of the tp2 op, ~71% of its issue bound) and the fixed per-workgroup tile cost of
//      remainder entries (MTX 8 / PAIR, refused above on registers).

#if EXL3R4D_PF_WGS_ON
extern "C" __attribute__((visibility("default"))) int exl3r4d_pfstamp_clear(void* stream) {
  void *t = nullptr, *n = nullptr;
  if (hipGetSymbolAddress(&t, HIP_SYMBOL(g_pfs_t)) != hipSuccess ||
      hipGetSymbolAddress(&n, HIP_SYMBOL(g_pfs_n)) != hipSuccess)
    return RAD_E_DEVICE;
  if (hipMemsetAsync(t, 0, sizeof g_pfs_t, (hipStream_t)stream) != hipSuccess ||
      hipMemsetAsync(n, 0, sizeof g_pfs_n, (hipStream_t)stream) != hipSuccess)
    return RAD_E_DEVICE;
  return RAD_OK;
}
// up to n slots of {entry, exit, hw, info} as four u64 each; returns the slots taken (or a negative RAD_E_*)
extern "C" __attribute__((visibility("default"))) int exl3r4d_pfstamp_read(unsigned long long* out, int n) {
  static unsigned long long t[kPfsMax][2];
  static unsigned hw[kPfsMax], info[kPfsMax];
  unsigned cnt = 0;
  if (n > kPfsMax || hipMemcpyFromSymbol(t, HIP_SYMBOL(g_pfs_t), sizeof t) != hipSuccess ||
      hipMemcpyFromSymbol(hw, HIP_SYMBOL(g_pfs_hw), sizeof hw) != hipSuccess ||
      hipMemcpyFromSymbol(info, HIP_SYMBOL(g_pfs_info), sizeof info) != hipSuccess ||
      hipMemcpyFromSymbol(&cnt, HIP_SYMBOL(g_pfs_n), sizeof cnt) != hipSuccess)
    return RAD_E_DEVICE;
  for (int i = 0; i < n; ++i) {
    out[4 * i] = t[i][0], out[4 * i + 1] = t[i][1], out[4 * i + 2] = hw[i], out[4 * i + 3] = info[i];
  }
  return (int)(cnt < (unsigned)n ? cnt : (unsigned)n);
}
#endif

// EXL3R4D_OPSTAMP (READING ONLY, never shipped; for attributing time-to-first-token): each prefill-band op (prefill.c's
// exl3r4d_pf_launch) is bracketed on its stream by a one-wave kernel that takes the next slot of a device counter and
// writes {site, rows, device clock (s_memrealtime, 100 MHz)} into a pinned host ring: an UNSYNCHRONISED timeline of
// this band's ops and of the stretches of other ops between them, in execution order (rocprofv3 cannot trace this
// engine: libr4d's init fails under its queue interception). Every argument is constant per call site, so a recorded
// pass replays the stamps too (a replay calls no host code). Written to $EXL3R4D_OPSTAMP_OUT at exit.
#if EXL3R4D_OPSTAMP && EXL3R4D_NCLS == 2
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
namespace {
struct OpStamp { unsigned site, T; unsigned long long clk; };
constexpr unsigned kOsMax = 1u << 18;
OpStamp* g_os = nullptr;
unsigned* g_os_ctr = nullptr;
__global__ void exl3r4d_opstamp_kernel(OpStamp* ring, unsigned* ctr, unsigned site, unsigned T) {
  if (threadIdx.x == 0) {
    const unsigned long long c = __builtin_readsteadycounter();
    const unsigned i = atomicAdd(ctr, 1u) % kOsMax;
    ring[i].site = site; ring[i].T = T; ring[i].clk = c;
    __threadfence_system();
  }
}
void os_dump() {
  const char* f = getenv("EXL3R4D_OPSTAMP_OUT");
  if (!f || !g_os) return;
  FILE* o = fopen(f, "wb");
  if (!o) return;
  fwrite(g_os, sizeof(OpStamp), kOsMax, o);
  fclose(o);
}
}  // namespace
extern "C" int exl3r4d_opstamp(int site, long long T, void* stream) {
  if (!g_os) {
    if (hipHostMalloc((void**)&g_os, kOsMax * sizeof(OpStamp), hipHostMallocDefault) != hipSuccess) { g_os = nullptr; return -1; }
    memset(g_os, 0, kOsMax * sizeof(OpStamp));
    if (hipMalloc((void**)&g_os_ctr, 64) != hipSuccess || hipMemset(g_os_ctr, 0, 64) != hipSuccess) return -1;
    atexit(os_dump);
  }
  OpStamp* r = g_os; unsigned* c = g_os_ctr; unsigned st = (unsigned)site, t = (unsigned)T;
  void* args[] = {&r, &c, &st, &t};
  return hipLaunchKernel((const void*)exl3r4d_opstamp_kernel, dim3(1), dim3(32), args, 0, (hipStream_t)stream) == hipSuccess ? 0 : -1;
}
#endif
