/* r3d_launch.h -- the two lines a libr3d unit needs to launch a kernel.
 *
 * R3D_KLAUNCH is the triple-chevron form, which the engine services on its own queues because the
 * stream it hands a launch is the one hipLaunchKernel accepts (docs/PLUGIN.md §5). It is written as
 * a macro so a test can define it and drive the same sources through a host SIMT emulator; a build
 * does not, and gets the chevron.
 *
 * R3D_DYN_LDS declares a kernel's dynamic LDS, sized by the launch: `extern __shared__` in a build.
 * RDNA3 gives a workgroup the same 64 KiB RDNA4 does, so a tile sized for libr4d fits here too.
 *
 * R3D_KNAME names a kernel in a context where commas would split a macro argument, the way
 * HIP_KERNEL_NAME does for hipLaunchKernelGGL.
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
