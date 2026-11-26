// r4d_chain.h -- a chained launch's wait and completion (RadKernelChainFn, abi/rad_abi.h).
//
// Between two dependent kernels on one queue the packet processor waits for the first to finish
// before it launches the second (the AQL barrier bit): on 4x R9700, ~1.3 ms of a 14 ms DFlash2 step
// on the 31B. A chained launch drops the bit. The engine writes a record into the
// kernel's argument slot, and the kernel waits for its queue's completion counter to reach the
// blocks issued before it, then adds its own blocks as they finish. A kernel's chained form
// (`CHAIN` true) brackets its body with r4d_chain_wait / r4d_chain_done; the plain form never reads
// the record, so a launch from outside the engine (r4d_selftest, rad-kbench) is untouched.
//
// Every block must reach r4d_chain_done exactly once, so a kernel whose blocks return early keeps
// its body in a __device__ function called from the chained wrapper.
//
// The wait's acquire has to cover every read behind it, and an agent-scope fence does not: it
// invalidates the vector caches, while a load whose address every lane shares is a scalar load,
// served from the scalar data cache. A GEMM reads its rows' scales that way, the routed kernels
// their counts and offsets, the head its activation (r4dchain_lm_wmma, 16 words a load). What that
// cache holds is what a launch read from the same arena address a layer earlier. The packet
// processor's acquire empties it when it launches the kernel, which is before the kernels in front
// have written; r4d_chain_wait therefore empties it behind the fence itself (r4d_chain_acquire).
// With the packets' acquire on, a probe of 1024-block readers saw one launch in 1800 read a stale
// value without it.
#pragma once
#include <hip/hip_runtime.h>
#include "rad_abi.h"

/* A kernel template's chain parameter, for a kernel that brackets its own body: `r4dchain`
 * instantiates the chained form and puts the name the backend knows a chained kernel by into the
 * instantiation's symbol, R4dUnchained the plain form. */
struct r4dchain     { static constexpr bool on = true; };
struct R4dUnchained { static constexpr bool on = false; };

#if !defined(__HIP__)
/* The host SIMT emulator (tests/r4d_emu) compiles some of these sources as C++: no engine record
 * ever reaches a launch there, so a chained form is its plain form. */
__device__ __forceinline__ void r4d_chain_wait() {}
__device__ __forceinline__ void r4d_chain_wait_vec() {}
__device__ __forceinline__ void r4d_chain_done() {}
#else

struct R4dChainRec {
    unsigned long long  magic, target;
    unsigned long long* ctr;
    unsigned long long  check;
};

// The record, or null where it is not the engine's. Read from the argument segment (scalar loads)
// at both ends of the kernel, so nothing is held across the body: a register-heavy body (the
// split-KV decode attention) would otherwise pay for the pointer.
__device__ __forceinline__ const R4dChainRec* r4d_chain_rec() {
    const R4dChainRec* r = (const R4dChainRec*)((const char*)__builtin_amdgcn_kernarg_segment_ptr() +
                                                RAD_CHAIN_KARG_OFF);
    const unsigned long long magic = r->magic, target = r->target, check = r->check;
    const unsigned long long* ctr = r->ctr;
    if (magic != RAD_CHAIN_MAGIC ||
        check != (magic ^ target ^ (unsigned long long)(uintptr_t)ctr ^ RAD_CHAIN_CHECK))
        return nullptr;
    return r;
}

// Before the first read of an operand: one lane waits for the counter to reach the target.
// Every resident block polls the same word, so a wide kernel launched behind a long one (a GEMM
// behind a junction's all-reduce) would put hundreds of loads into one L2 channel every few dozen
// clocks for as long as the wait lasts, slowing whatever else that channel serves. The first
// R4D_CHAIN_FAST polls are ~64 clocks apart, the rest R4D_CHAIN_SLOW x 64: a long wait ends at most
// that much late and puts 1/R4D_CHAIN_SLOW of the load on the channel.
//
// A grid of a few blocks puts little on the channel however often it polls, and a decode step is
// mostly such grids behind one another (a reduction's four blocks, a quantiser's, a router's
// seventeen), each ending its wait up to a slow period late: so up to R4D_CHAIN_TINY blocks keep
// the fast period for the whole wait and up to R4D_CHAIN_SMALL poll every R4D_CHAIN_MID x 64 clocks,
// which is at most the load a 256-block grid puts there at the slow period. Qwen3.8-Flash-Next at
// tp 2, chain 3 with coupling, ~670 dependent launches a verify pass: -0.32% a step; every grid at
// 8 or 4 instead of 16 gave -0.16% and -0.25% for two and four times the polls of the wide ones.
#ifndef R4D_CHAIN_FAST
#define R4D_CHAIN_FAST 8
#endif
#ifndef R4D_CHAIN_SLOW
#define R4D_CHAIN_SLOW 16
#endif
// The acquire behind a chained wait: the vector caches by the fence, the scalar data cache by
// hand, which the fence leaves (see the head of this file). Every wave, for the compute unit it
// runs on.
__device__ __forceinline__ void r4d_chain_acquire(bool scalar = true) {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if (scalar) asm volatile("s_dcache_inv" ::: "memory");
}

#ifndef R4D_CHAIN_TINY
#define R4D_CHAIN_TINY 16
#endif
#ifndef R4D_CHAIN_SMALL
#define R4D_CHAIN_SMALL 64
#endif
#ifndef R4D_CHAIN_MID
#define R4D_CHAIN_MID 4
#endif
// The wait itself; `Scalar` false leaves the scalar data cache alone (see r4d_chain_wait_vec).
template <bool Scalar>
__device__ __forceinline__ void r4d_chain_wait_impl() {
    const R4dChainRec* r = r4d_chain_rec();
    if (!r) return;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        const unsigned long long target = r->target;
        unsigned long long* ctr = r->ctr;
        const unsigned nblk = gridDim.x * gridDim.y * gridDim.z;
        int n = 0;
        while (__hip_atomic_load(ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target) {
            if (n < R4D_CHAIN_FAST) {
                __builtin_amdgcn_s_sleep(1);
                ++n;
            } else if (nblk <= R4D_CHAIN_TINY) {
                __builtin_amdgcn_s_sleep(1);
            } else if (nblk <= R4D_CHAIN_SMALL) {
                __builtin_amdgcn_s_sleep(R4D_CHAIN_MID);
            } else {
                __builtin_amdgcn_s_sleep(R4D_CHAIN_SLOW);
            }
        }
    }
    __syncthreads();
    r4d_chain_acquire(Scalar);
}

// The wait every kernel uses unless it has been shown not to need the scalar half of the acquire.
#ifdef R4D_CHAIN_SURVEY_NO_SCALAR
// Audit builds only (tests/chain_scalar_audit.py): every wait skips the instruction, so the audit
// lists the kernels that do scalar loads behind it.
__device__ __forceinline__ void r4d_chain_wait() { r4d_chain_wait_impl<false>(); }
#else
__device__ __forceinline__ void r4d_chain_wait() { r4d_chain_wait_impl<true>(); }
#endif

// The wait of a kernel that reads nothing a launch in front wrote through a scalar load: no
// operand or scale of it is a value all lanes share and the compiler loads through the scalar
// data cache. Then the fence's vector-cache acquire is all the wait needs and `s_dcache_inv`
// (a scalar-cache flush and the refetch of every kernel argument after it) is left out. The
// kernel's instantiations all have to be free of such loads: tests/chain_scalar_audit.py
// compiles every unit and fails on a kernel that skips the instruction and has one, and
// R4D_CHAIN_SCINV_ALWAYS builds the library with the instruction behind every wait again.
#ifdef R4D_CHAIN_SCINV_ALWAYS
__device__ __forceinline__ void r4d_chain_wait_vec() { r4d_chain_wait_impl<true>(); }
#else
__device__ __forceinline__ void r4d_chain_wait_vec() { r4d_chain_wait_impl<false>(); }
#endif

// After the block's last write: release, then one add a block.
__device__ __forceinline__ void r4d_chain_done() {
    const R4dChainRec* r = r4d_chain_rec();
    if (!r) return;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        __hip_atomic_fetch_add(r->ctr, 1ull, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
#endif  /* __HIP__ */

/* A chained launch only at steps of up to 32 token rows: above that the waiting blocks hold room
 * a wide predecessor needs. */
constexpr long long kR4dChainMaxRows = 32;
inline bool r4d_chain_rows_ok(long long rows) { return rows <= kR4dChainMaxRows; }
