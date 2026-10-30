# libhalo serves Strix Halo (gfx1151, RDNA 3.5) only, declared here and nowhere else. Its kernels
# rely on wave32, the gfx11 WMMA fragment layouts and unified memory; built for other targets they
# would compute garbage.
#
# A build targeting other cards skips libhalo. The engine then has no device kernel library for that
# card and refuses ops only a reference library implements, unless
# --debug-accept-reference-kernels is passed (spec §17). The supported fix is a kernel library for
# that card.
#
# Built in radiance's tree, or standalone against an installed radiance
# (`cmake -S libhalo -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`, naming the HIP compiler as for
# the tree); see cmake/RadAddPlugin.cmake.
#
# Declaration (schemas, rows, layouts) is shared with libr4d by design -- see libhalo/README.md.
# Kernel units land family by family in HALO_UNITS; until one does, halo_stubs.cpp answers its rows
# with RAD_E_UNSUPPORTED so the library loads and the gap is reported by name.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(libhalo LANGUAGES CXX HIP)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

rad_plugin_gpu_targets(HALO_GPU_TARGETS "^gfx1151$")
if(NOT HALO_GPU_TARGETS)
  message(STATUS
    "libhalo: Strix Halo (gfx1151) only, and this build targets ${RAD_GPU_TARGETS} -- not built")
  return()
endif()

# HALO_UNITS is generated: libhalo/gen_stubs.py writes the list below from the halo_*.hip files
# present (each "unit|extra flags", mirroring libr4d's R4D_UNITS), so add a unit by adding its
# .hip, not by editing the list. Every unit compiles -ffp-contract=off (rad_hipify): the
# quantised all-reduce requires it, for the reason libr4d/CMakeLists.txt gives.
set(HALO_UNITS
  "halo_ar_entry|"
  "halo_ar_gather_2rank|"
  "halo_ar_gather_Nrank|"
  "halo_ar_ln_had_quant_i8|"
  "halo_ar_oneshot_2rank_exact|"
  "halo_ar_oneshot_2rank_wht6|"
  "halo_ar_oneshot_Nrank_exact|"
  "halo_ar_twoshot_Nrank_exact|"
  "halo_ar_twoshot_Nrank_ti8|"
  "halo_attn_dense_bf16|"
  "halo_attn_gen|"
  "halo_attn_paged_gqa8|"
  "halo_attn_paged_h128_gqa1|"
  "halo_attn_paged_h128_gqa2|"
  "halo_attn_paged_h128_gqa2w|"
  "halo_attn_paged_h128_gqa4|"
  "halo_attn_paged_h256_gqa12|"
  "halo_attn_paged_h256_gqa2|"
  "halo_attn_paged_h256_gqa4|"
  "halo_attn_paged_h256_gqa6|"
  "halo_attn_paged_h256_gqa6_bs4|"
  "halo_attn_paged_window|"
  "halo_attn_vit_h72_bf16|"
  "halo_audio|"
  "halo_conv2d_nhwc|"
  "halo_conv_t1d|"
  "halo_dequant_w4_bf16|"
  "halo_dflash_conv_t2_g16_bf16|"
  "halo_dflash_select_bf16|"
  "halo_dsa|"
  "halo_encoder_bf16|"
  "halo_expert_scale_bf16|"
  "halo_fused_quant_fp8|"
  "halo_gated_had_quant_i8|"
  "halo_gdn_chunk_scan|"
  "halo_gdn_conv_w4_h128_bf16|"
  "halo_gdn_gated_norm_had_quant_i8|"
  "halo_gdn_gated_rmsnorm_h128_bf16|"
  "halo_gdn_kkt_solve|"
  "halo_gdn_recurrent_update_k128_v128_bf16_fp32state|"
  "halo_gelu_mul_quant_i8|"
  "halo_gemm_bf16_nt_m16|"
  "halo_gemm_bf16_nt_m64|"
  "halo_gemm_bf16_nt_tiled|"
  "halo_gemm_f32_nt_bias|"
  "halo_gemm_fp8_f32scale|"
  "halo_gemm_fp8a16_m1|"
  "halo_gemm_fp8a8|"
  "halo_gemm_mxfp4a8_decode|"
  "halo_gemm_mxfp4a8_gated|"
  "halo_gemm_mxfp4a8_gemv|"
  "halo_gemm_mxfp4a8_nt_m64|"
  "halo_gemm_mxfp4a8_tiled|"
  "halo_gemm_nt_bias|"
  "halo_gemm_nt_f32|"
  "halo_gemm_w2a8_nt|"
  "halo_gemm_w4a16_nt_m64|"
  "halo_gemm_w4a8_nt_m64|"
  "halo_gemm_w4a8_tiled|"
  "halo_gemm_w5a16_nt_m64|"
  "halo_gemm_w8a16_nt_m64|"
  "halo_gemm_w8a8_nt_m64|"
  "halo_gemm_w8a8_tiled|"
  "halo_gemm_wpa8i8_tiled|"
  "halo_gen|"
  "halo_givens_dense|"
  "halo_givens_rotate|"
  "halo_glm|"
  "halo_gram_accum_fp8|"
  "halo_had_quant_act_i8|"
  "halo_hc_bf16|"
  "halo_hc_read|"
  "halo_kda|"
  "halo_kv_len_trim|"
  "halo_kv_resolve|"
  "halo_logits_gemm|"
  "halo_lora_bf16|"
  "halo_ltx|"
  "halo_mhc|"
  "halo_mla|"
  "halo_model_bf16|"
  "halo_moe|"
  "halo_moe_gemm_bf16|"
  "halo_moe_gemm_q|"
  "halo_moe_mxfp4|"
  "halo_mtp_enter|"
  "halo_ngram_ids|"
  "halo_norm_join_bf16|"
  "halo_p2p_copy2d|"
  "halo_pair_argmax_f32|"
  "halo_ple_conv_bf16|"
  "halo_ple_gate_bf16|"
  "halo_prefetch|"
  "halo_qk_norm_rope_gate|"
  "halo_qkv_norm_rope|"
  "halo_qsa_block_key_bf16|"
  "halo_qsa_score_bf16|"
  "halo_qsa_select_bf16|"
  "halo_qsa_tail_store_bf16|"
  "halo_qsa_work_bf16|"
  "halo_quant_act_fp8|"
  "halo_quant_act_i8|"
  "halo_rmsnorm_had_quant_i8|"
  "halo_rmsnorm_quant_i8|"
  "halo_router_gemm_scatter|"
  "halo_router_scatter|"
  "halo_router_sigmoid_topk_f32|"
  "halo_router_topk_bf16|"
  "halo_rowtopk_bf16|"
  "halo_sam_attn_bf16|"
  "halo_sample_chain_f32|"
  "halo_sample_ngram_f32|"
  "halo_sample_stages_f32|"
  "halo_scale_rows_bf16|"
  "halo_upscaler|"
  "halo_video|"
  "halo_vit_norm_embed|"
  "halo_vsa|"
)

set(HALO_SRC "")
foreach(u ${HALO_UNITS})
  string(REPLACE "|" ";" parts "${u}")
  list(GET parts 0 unit)
  list(LENGTH parts n)
  set(extra "")
  if(n GREATER 1)
    list(GET parts 1 extra)
  endif()
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${unit}.hip")
  if(NOT EXISTS "${src}")
    message(FATAL_ERROR "libhalo unit ${unit}.hip is missing from ${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  if(extra)
    set_source_files_properties("${src}" PROPERTIES COMPILE_OPTIONS "${extra}")
  endif()
  list(APPEND HALO_SRC "${src}")
endforeach()

# The host sources are declaration only: the row table, weight layouts and relayouts, operand
# descriptions and fusion claims, plus the generated stubs for units not yet ported. They are pure
# host C++ that includes neither halo.h nor hip_runtime.h, so these tables compile and can be
# inspected without ROCm.
set(HALO_DECLARATION halo_rows.cpp halo_layout.cpp halo_fp8_layout.cpp halo_shapes.cpp halo_fuse.cpp
                     halo_stubs.cpp)

rad_add_plugin(libhalo kernels
  SOURCES     ${HALO_DECLARATION}
  HIP_SOURCES ${HALO_SRC}
  GPU_TARGETS ${HALO_GPU_TARGETS}
  INCLUDES    ${CMAKE_CURRENT_SOURCE_DIR})

# Catalogue dump: prints every row and, on a card, calls each launch with a minimal valid input, to
# catch transcription errors in the constraint table. Derived from libr4d's r4d_selftest.cpp (row
# names are shared, so its cases address libhalo's rows); it dlopens the plugin so an undefined
# symbol in the .so is caught too.
add_executable(halo_selftest ${CMAKE_CURRENT_SOURCE_DIR}/halo_selftest.cpp)
# Links the HIP runtime to allocate the device buffers it feeds the kernels.
target_link_libraries(halo_selftest PRIVATE radiance::abi ${CMAKE_DL_LIBS} ${RAD_HIP_RUNTIME})
target_include_directories(halo_selftest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(halo_selftest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
install(TARGETS halo_selftest RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Interface parity with libr4d (halo_parity.cpp): schemas, rows, constraints, tunables and the
# concurrency/collective promises, compared table against table. Only a build that produces both
# plugins (RAD_GPU_TARGETS naming a gfx12 part and gfx1151) can run it; tests/CMakeLists.txt adds
# the test then.
if(TARGET libr4d)
  add_executable(halo_parity ${CMAKE_CURRENT_SOURCE_DIR}/halo_parity.cpp)
  target_link_libraries(halo_parity PRIVATE radiance::abi ${CMAKE_DL_LIBS})
  set_target_properties(halo_parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

# C++17 for the same reason as libr4d: kernel cost figures are for C++17, and the ported units are
# written against it.
set_target_properties(libhalo PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
