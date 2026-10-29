# libr3d serves RDNA3 only, declared here and nowhere else. Its kernels rely on wave32 and on
# instructions RDNA3 has: v_dot2acc_f32_f16 for the matrix work, v_cvt_pkrtz_f16_f32, DPP16 and
# v_permlanex16. They do not use the WMMA fragment layouts (RDNA3's 16x16x16 wave-matrix family is
# a re-tile of the dot2 lane map, not a drop-in) or gfx12-only instructions such as
# global_load_tr_b128, and they decode every quantised weight in software: the lane map, not a
# missing instruction, is the reason.
#
# A build targeting other cards skips libr3d. The engine then has no device kernel library for that
# card and refuses ops only a reference library implements, unless
# --debug-accept-reference-kernels is passed (spec §17). The supported fix is a kernel library for
# that card; libr4d is that library for gfx12 and libr3d is it for gfx11.
#
# Built in radiance's tree, or standalone against an installed radiance
# (`cmake -S libr3d -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`, naming the HIP compiler as for
# the tree); see cmake/RadAddPlugin.cmake.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(libr3d LANGUAGES CXX HIP)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

# `^gfx11` is the family, not a chip: gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151 and the
# family target `gfx11-generic` all take these kernels, exactly as `^gfx12` covers RDNA4 for libr4d.
# The RDNA3 cards differ in CU count and clock, never in the instructions these kernels issue, so
# one code object per family serves the lot; a build that lists both a chip and the family gets
# both, and the engine picks the chip's object first (docs/PLUGIN.md §10).
rad_plugin_gpu_targets(R3D_GPU_TARGETS "^gfx11")
if(NOT R3D_GPU_TARGETS)
  message(STATUS
    "libr3d: RDNA3 (gfx11xx) only, and this build targets ${RAD_GPU_TARGETS} -- not built")
  return()
endif()

# R3D_UNITS is the definitive list of libr3d's device translation units, as "unit|extra flags".
#
# Every unit is compiled -ffp-contract=off, which rad_hipify puts on all HIP sources. It is not a
# style preference: libref accumulates in f32 with no FMA and no reassociation (ref_common.h), and
# a contracted device kernel disagrees with the oracle it is checked against. It also keeps the
# instruction counts quoted in kernel comments true.

# Kernels and libr3d's declaration (row table, operand descriptions) share one directory. There is
# no launch adapter: every entry point takes `(const RadArgs*, RadStream)` and sits beside its
# kernel. The plugin links rad_abi only, like one written out of tree.

set(R3D_UNITS
  # The model's small ops, one workgroup per row: rmsnorm, layernorm, the binary and unary element
  # ops, the activations, cast, softmax, embed_lookup, gather_rows and scatter_rows, rope and
  # kv_store. This is the set the engine refuses on a card with no device kernel library.
  "r3d_model_bf16|"
  # The sampler's pre-filters: penalties, DRY, the n-gram ban, the bitmask, and the logprobs
  # summary, which reads the logits plane once where the reference reads it three.
  "r3d_sample_pre_f32|"
  # The speculative decoder's accept step: a walk over the draft block's candidate lattice, which is
  # a gather out of two codebooks rather than arithmetic.
  "r3d_dflash_f32|"
  # The gated MLP band: GEMM, gate and output quantiser in one launch, so the [M,N] intermediate the
  # unfused chain writes and re-reads never exists.
  "r3d_gemm_fp8a8_gated|"
  # The vocab projection at decode: U2 codes, int8 activations, a joined scale+zero operand.
  "r3d_gemm_nt_q_w2a8|"
  # The vocabulary projection at both heights: the head streamed for M <= 16, tiled above it.
  "r3d_vocab_bf16|"
  # The quantised GEMM at prefill heights: the 176 bands a forward pass ran on the CPU.
  "r3d_gemm_fp8a8_tiled|"
  # The OCP-MXFP4 weight against the fp8 activation, both heights: e2m1 codes with one e8m0 scale
  # per 32 of K, the block's partials scaled at the block boundary.
  "r3d_gemm_mxfp4a8|"
  "r3d_gemm_inta8|"
  # Paged attention at head_dim 128: the decode leg (one head group a workgroup, 16-token tiles,
  # split-KV) and the prefill leg (a 16-token query tile staged in LDS), over a bf16 or an E4M3
  # cache. The walk is shared; the tile is what a workgroup stages once and every wave consumes.
  "r3d_attn_paged_h128|"
  # The sampler's spine: the row's top-R in one scan, and the stages that turn the candidate plane
  # the top-R produces into one token -- argmax, temp, top-k, the draw, and the four narrowings.
  "r3d_sample_f32|"
  # Activation quantisation: per-(row, group) symmetric E4M3 and int8, the gated form that
  # quantises a silu-gated pair, and the dequantise that reads a container's own quantised weights.
  "r3d_quant_fp8|"
  # The skinny bf16 GEMM, M <= 16: one output element a lane, k ascending, so the accumulation order
  # is libref's. The decode GEMM -- the op a token is made of.
  "r3d_gemm_bf16_nt_m16|"
  # The same shape over a block-scaled fp8 weight, decoded to f16 exactly (E4M3's four exponent bits
  # and three mantissa bits are a subset of f16's), and over int8 codes.
  "r3d_gemm_fp8a16_nt_m1|"
  # Landing next, in the same shape as libr4d's: the prefill GEMM (an 8x8 tile a wave over LDS, the
  # same dot2 inner product), paged causal attention (the decode leg split-KV, the prefill leg a
  # query tile a workgroup, bf16 or fp8 KV), the vocab (logits as a GEMM over the embedding, the row
  # top-k, the cross-rank top-1) and the sampler's stages. Their rows are not in r3d_rows.cpp yet,
  # because a row naming an entry point that is not in the library is an undefined symbol and the
  # plugin would fail to load, taking every other row with it.
)

set(R3D_SRC "")
foreach(u ${R3D_UNITS})
  string(REPLACE "|" ";" parts "${u}")
  list(GET parts 0 unit)
  list(LENGTH parts n)
  set(extra "")
  if(n GREATER 1)
    list(GET parts 1 extra)
  endif()
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${unit}.hip")
  if(NOT EXISTS "${src}")
    message(FATAL_ERROR "libr3d unit ${unit}.hip is missing from ${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  if(extra)
    set_source_files_properties("${src}" PROPERTIES COMPILE_OPTIONS "${extra}")
  endif()
  list(APPEND R3D_SRC "${src}")
endforeach()

# The host sources are declaration only: the row table, the operand descriptions and the fusion
# claims. They are pure host C++ that includes neither r3d.h nor hip_runtime.h, so these tables
# compile and can be inspected without ROCm. libr3d has no layout source: nothing it serves is
# repacked, because a dot2 GEMM reads a weight in the order it is stored (see r3d_rows.cpp).
# r3d_fuse.cpp joins the list with the first fused row: a `replaces` chain with no row naming it
# would be dead code, and the fused rmsnorm-then-quant kernel is the first one worth its bytes.
set(R3D_DECLARATION r3d_rows.cpp r3d_shapes.cpp)

rad_add_plugin(libr3d kernels
  SOURCES     ${R3D_DECLARATION}
  HIP_SOURCES ${R3D_SRC}
  GPU_TARGETS ${R3D_GPU_TARGETS}
  INCLUDES    ${CMAKE_CURRENT_SOURCE_DIR})

# rad_plugin_info's build_target is this build's own target list, spelled as the device reports its
# arch, because the loader warns when a plugin claims a card the running device is not
# (core/plugin/loader.cpp, target_covers). A library built for two archs says two.
string(REPLACE ";" " " R3D_BUILD_TARGET "${R3D_GPU_TARGETS}")
target_compile_definitions(libr3d PRIVATE R3D_BUILD_TARGET="${R3D_BUILD_TARGET}")

# Catalogue dump: prints every row and, on a card, calls each launch with a minimal valid input, to
# catch transcription errors in the constraint table. It dlopens the plugin so an undefined symbol
# in the .so is caught too.
add_executable(r3d_selftest ${CMAKE_CURRENT_SOURCE_DIR}/r3d_selftest.cpp)
# Links the HIP runtime to allocate the device buffers it feeds the kernels.
target_link_libraries(r3d_selftest PRIVATE radiance::abi ${CMAKE_DL_LIBS} ${RAD_HIP_RUNTIME})
target_include_directories(r3d_selftest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# The device phase of the selftest -- the per-row launch -- is built only when the HIP runtime is
# there, which in this plugin's build it always is.
target_compile_definitions(r3d_selftest PRIVATE RAD_HAVE_HIP)
set_target_properties(r3d_selftest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# Installed beside the tools, so the image can run the rows' own card checks without a build tree:
#   r3d_selftest <home>/kernels/libr3d.so [--case <row>]
install(TARGETS r3d_selftest RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# C++17, the standard every radiance plugin builds device code at: it is what the tree's kernels are
# written against and what the cost figures assume.
set_target_properties(libr3d PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
