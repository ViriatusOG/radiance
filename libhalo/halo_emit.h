/* Derived from libr4d/r4d_emit.h by the libhalo rename transform (2026-10-26);
 * see libhalo/README.md. Op schemas, row names and stored weight layouts are
 * shared with libr4d by design. Everything else in this file is libhalo's own
 * and diverges as it is measured on gfx1151. */
// halo_emit.h -- the one launch path every libhalo kernel goes through.
//
// `hipLaunchKernel` takes the arguments as an array of pointers (kernelParams) and copies each
// slot's bytes without conversion, so a `long long` passed for a `uint32_t` parameter would fill a
// four-byte slot with eight bytes and corrupt every later argument. halo_emit takes the kernel
// typed, converts every argument to its declared parameter type, and checks the arity at compile
// time. It also traces each dispatch under RAD_HALO_TRACE and feeds halo_record_dispatches (halo.h).

#ifndef HALO_EMIT_H
#define HALO_EMIT_H

#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <tuple>

#include "rad_abi.h"
#include "halo_args.h"

/* halo_record_dispatches' log (halo_rows.cpp): while `halo_dispatch_log` is set, each launch appends
 * its function and counts it in `halo_dispatch_n`, which may run past `halo_dispatch_cap`. */
extern const void** halo_dispatch_log;
extern int          halo_dispatch_cap;
extern int          halo_dispatch_n;

/* `func` is the kernel's host stub; for a template, the instantiation's own symbol. */
template <class... P, class... A>
inline int halo_emit(RadStream stream, void (*kernel)(P...), dim3 grid, dim3 block,
                    uint32_t shared_bytes, const A&... args) {
    static_assert(sizeof...(P) == sizeof...(A),
                  "a kernel argument list must match the kernel's parameter list");
    const void* func = (const void*)kernel;
    std::tuple<P...> vals(static_cast<P>(args)...);
    void* params[sizeof...(P) + 1];
    std::apply([&](P&... v) {
        int i = 0;
        ((params[i++] = (void*)&v), ...);
    }, vals);
    params[sizeof...(P)] = nullptr;

    if (halo_dispatch_log) {
        if (halo_dispatch_n < halo_dispatch_cap) halo_dispatch_log[halo_dispatch_n] = func;
        ++halo_dispatch_n;
    }
    /* Trace the dispatch before submitting it: a fault inside the driver leaves only this frame,
     * with the geometry in reused registers and the kernel a stub address. An `emit` with no
     * `ret` after it is the faulting dispatch; flushed because the fault may be next. Each line is
     * one write, since every rank is a thread and piecewise fprintf calls would interleave. */
    if (halo_trace_on()) {
        static constexpr size_t kSize[] = { sizeof(P)..., 0 };
        char line[1024];
        int  o = snprintf(line, sizeof line,
                          "[r4d] emit %p grid=%ux%ux%u block=%ux%ux%u lds=%u np=%u",
                          func, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                          shared_bytes, (unsigned)sizeof...(P));
        for (size_t k = 0; k < sizeof...(P) && o > 0 && o < (int)sizeof line - 32; ++k) {
            unsigned long long v = 0;
            memcpy(&v, params[k], kSize[k] > 8 ? 8 : kSize[k]);
            o += snprintf(line + o, sizeof line - (size_t)o, " %u:%llx", (unsigned)kSize[k], v);
        }
        std::fprintf(stderr, "%s\n", line);
        std::fflush(stderr);
    }
    const int rc = hipLaunchKernel(func, grid, block, params, shared_bytes,
                                   (hipStream_t)stream) == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    if (halo_trace_on()) { std::fprintf(stderr, "[r4d] ret %p %d\n", func, rc); std::fflush(stderr); }
    return rc;
}

/* `kernel` is named, not passed as a value, so a template instantiation is spelled at the call
 * site exactly as the triple-chevron spelled it. A test may define HALO_EMIT to drive the same
 * sources through a host SIMT emulator, as with HALO_KLAUNCH (halo_launch.h). */
#ifndef HALO_EMIT
#define HALO_EMIT(stream, kernel, grid, block, lds, ...) \
    halo_emit((stream), (kernel), (grid), (block), (uint32_t)(lds), __VA_ARGS__)
#endif

#endif  /* HALO_EMIT_H */
