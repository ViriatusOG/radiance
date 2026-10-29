/* r3d_launch.h -- kernel launch macros for libr3d units.
 *
 * R3D_KLAUNCH is the triple-chevron launch on the stream the engine hands the entry point, which is
 * a stream hipLaunchKernel accepts (docs/PLUGIN.md §5). It is a macro so a test can redefine it and
 * run the same sources through a host SIMT emulator.
 *
 * R3D_DYN_LDS declares a kernel's dynamic LDS, sized by the launch. RDNA3 gives a workgroup 64 KiB,
 * as RDNA4 does, so a tile sized for libr4d fits.
 *
 * R3D_KNAME wraps a kernel name that contains commas, like HIP_KERNEL_NAME.
 */
#ifndef R3D_LAUNCH_H
#define R3D_LAUNCH_H
#define R3D_KNAME(...) __VA_ARGS__
#ifndef R3D_KLAUNCH
#define R3D_KLAUNCH(kernel, grid, block, stream, ...) kernel<<<grid, block, 0, stream>>>(__VA_ARGS__)
#endif
#ifndef R3D_DYN_LDS
#define R3D_DYN_LDS(type, name) extern __shared__ type name[]
#endif
#endif
