/* advance.hip -- the advance kernel: one thread per row of the batch it patches. See advance.h.
 *
 * Launched through hipLaunchKernel, which the AQL backend intercepts, so it is one dispatch on the
 * rank's own queue, in stream order with its pass.
 */
#include <hip/hip_runtime.h>

#include <cstdlib>

#include "advance_impl.h"
#include "rad_abi.h"

extern "C" int rad_dev_chain_enabled(void);   /* core/device/device.h */

namespace {

constexpr int kThreads = 64;

__global__ __launch_bounds__(kThreads) void rad_step_advance(rad::AdvanceArgs a) {
    const int i = (int)(blockIdx.x * kThreads + threadIdx.x);
    if (i < a.n_seq) rad::adv_row(a, i);
}

/* The chained form (RadKernelChainFn, abi/rad_abi.h): the backend gives a kernel named r4dchain*
 * a record at RAD_CHAIN_KARG_OFF and drops its barrier bit behind another chained kernel. Before
 * the first read one lane waits for the queue's counter to reach the record's target; after the
 * last write the block releases and adds 1. A record that does not check out means: run plain. */
struct ChainRec {
    unsigned long long  magic, target;
    unsigned long long* ctr;
    unsigned long long  check;
};
__device__ __forceinline__ const ChainRec* chain_rec() {
    const ChainRec* r = (const ChainRec*)((const char*)__builtin_amdgcn_kernarg_segment_ptr() +
                                          RAD_CHAIN_KARG_OFF);
    const unsigned long long magic = r->magic, target = r->target, check = r->check;
    if (magic != RAD_CHAIN_MAGIC ||
        check != (magic ^ target ^ (unsigned long long)(uintptr_t)r->ctr ^ RAD_CHAIN_CHECK))
        return nullptr;
    return r;
}

__global__ __launch_bounds__(kThreads) void r4dchain_step_advance(rad::AdvanceArgs a) {
    const ChainRec* r = chain_rec();
    if (r) {
        if (threadIdx.x == 0)
            while (__hip_atomic_load(r->ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < r->target)
                __builtin_amdgcn_s_sleep(1);
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    const int i = (int)(blockIdx.x * kThreads + threadIdx.x);
    if (i < a.n_seq) rad::adv_row(a, i);
    if (r) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __syncthreads();
        if (threadIdx.x == 0)
            __hip_atomic_fetch_add(r->ctr, 1ull, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

/* ADV_COMPACT: block (x, y) is work item x (a pool, layer and head) of batch row y; its threads
 * split the K|V row into 16-byte chunks. Chained like the advance where launches chain, every
 * block waiting for the kernels before it and counting itself in after its moves. */
constexpr int kCompactThreads = 64;
__device__ __forceinline__ void step_compact_body(const rad::AdvanceArgs& a) {
    int p = 0, l = 0, h = 0;
    if (!rad::adv_compact_item(a, (int)blockIdx.x, &p, &l, &h)) return;
    rad::adv_compact(a, (int)blockIdx.y, p, l, h, (int)threadIdx.x, kCompactThreads);
}
__global__ __launch_bounds__(kCompactThreads) void rad_step_compact(rad::AdvanceArgs a) {
    step_compact_body(a);
}
__global__ __launch_bounds__(kCompactThreads) void r4dchain_step_compact(rad::AdvanceArgs a) {
    const ChainRec* r = chain_rec();
    if (r) {
        if (threadIdx.x == 0)
            while (__hip_atomic_load(r->ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < r->target)
                __builtin_amdgcn_s_sleep(1);
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    step_compact_body(a);
    if (r) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __syncthreads();
        if (threadIdx.x == 0)
            __hip_atomic_fetch_add(r->ctr, 1ull, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

}  // namespace

extern "C" int rad_step_compact_launch(const rad::AdvanceArgs* args, void* stream) {
    if (!args || args->n_seq <= 0) return 0;
    const int items = rad::adv_compact_items(*args);
    if (items <= 0) return 0;
    if (!rad_dev_chain_enabled())
        hipLaunchKernelGGL(rad_step_compact, dim3((unsigned)items, (unsigned)args->n_seq),
                           dim3(kCompactThreads), 0, (hipStream_t)stream, *args);
    else
        hipLaunchKernelGGL(r4dchain_step_compact, dim3((unsigned)items, (unsigned)args->n_seq),
                           dim3(kCompactThreads), 0, (hipStream_t)stream, *args);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

extern "C" int rad_step_advance_launch(const rad::AdvanceArgs* args, void* stream) {
    if (!args || args->n_seq <= 0) return 0;
    const unsigned blocks = (unsigned)((args->n_seq + kThreads - 1) / kThreads);
    /* The chained form only where launches chain (rad_dev_chain_enabled): there the backend writes
     * its record; elsewhere a launch carries only these arguments, so the plain kernel runs. */
    if (!rad_dev_chain_enabled())
        hipLaunchKernelGGL(rad_step_advance, dim3(blocks), dim3(kThreads), 0, (hipStream_t)stream,
                           *args);
    else
        hipLaunchKernelGGL(r4dchain_step_advance, dim3(blocks), dim3(kThreads), 0,
                           (hipStream_t)stream, *args);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
