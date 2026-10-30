/* Derived from libr4d/r4d_launch.h by the libhalo rename transform (2026-10-26);
 * see libhalo/README.md. Op schemas, row names and stored weight layouts are
 * shared with libr4d by design. Everything else in this file is libhalo's own
 * and diverges as it is measured on gfx1151. */
/* halo_launch.h -- the kernel-launch spelling for the libhalo units tests/halo_emu runs on the host:
 * the windowed paged attention, pair_argmax, the sandwich-norm joins, expert_scale and the QSA
 * block selection.
 *
 * HALO_KNAME wraps a template kernel name containing commas, as HIP_KERNEL_NAME does for
 * hipLaunchKernelGGL. A test may define HALO_KLAUNCH to drive the same sources through a host SIMT
 * emulator; a build does not, and gets the triple-chevron launch, which the engine services via
 * hipLaunchKernel on its own queues (docs/PLUGIN.md §5).
 *
 * HALO_DYN_LDS declares a kernel's dynamic LDS, sized by the launch: `extern __shared__` in a
 * build. The emulator has no such storage class and defines its own. */
#ifndef HALO_LAUNCH_H
#define HALO_LAUNCH_H
#define HALO_KNAME(...) __VA_ARGS__
#ifndef HALO_KLAUNCH
#define HALO_KLAUNCH(kernel, grid, block, stream, ...) kernel<<<grid, block, 0, stream>>>(__VA_ARGS__)
#endif
#ifndef HALO_DYN_LDS
#define HALO_DYN_LDS(type, name) extern __shared__ type name[]
#endif
#endif
