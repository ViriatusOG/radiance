# exl3r4d -- the gfx1201 kernel library for the EXL3 routed-expert ops (moe_exl3_gate_up, moe_exl3_down and the
# ops in docs/OPS-exl3.md), decode and prefill bands. Built from the repository's top level, or on its own against
# an installed radiance:
#
#   cmake -S kernels/exl3r4d -B build-exl3r4d -DCMAKE_PREFIX_PATH=<radiance install prefix>
#
# DEVICE ONLY. The library exists when the radiance found has a device build (RAD_HIP_FOUND), -DRAD_WITH_HIP is
# not OFF, and the build targets include gfx1201 (rad_plugin_gpu_targets); otherwise it is left out and said so --
# a host-only build has nothing for it to serve (its host twin is kernels/exl3ref).
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(exl3r4d LANGUAGES C CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
  # Refuse at configure time what the loader would refuse at dlopen. Inside the tree the plugin
  # compiles against the tree's own abi/ headers and so always matches.
  if(NOT RADIANCE_ABI_VERSION EQUAL 44)
    message(FATAL_ERROR "exl3r4d is written against radiance plugin ABI 44; the radiance found has ABI "
                        "${RADIANCE_ABI_VERSION}")
  endif()
  enable_testing()
  set(EXL3R4D_STANDALONE ON)
endif()
enable_language(C)

get_filename_component(EXL3_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
foreach(_f include/exl3_abi.h include/exl3_encoding.h common/exl3_decode.c common/exl3_layout.c)
  if(NOT EXISTS "${EXL3_ROOT}/${_f}")
    message(FATAL_ERROR "exl3r4d needs ${EXL3_ROOT}/${_f} (the repository around kernels/exl3r4d)")
  endif()
endforeach()
set(EXL3R4D_C_FLAGS -ffp-contract=off -Wall -Wextra)

set(EXL3R4D_HIP OFF)
if(RAD_HIP_FOUND AND NOT (DEFINED RAD_WITH_HIP AND RAD_WITH_HIP STREQUAL "OFF"))
  rad_plugin_gpu_targets(EXL3R4D_TARGETS "^gfx1201$")
  if(EXL3R4D_TARGETS)
    set(EXL3R4D_HIP ON)
  else()
    message(STATUS "exl3r4d: left out -- this build targets '${RAD_GPU_TARGETS}', and the library is gfx1201 code")
  endif()
else()
  message(STATUS "exl3r4d: left out -- no device build (RAD_HIP_FOUND=${RAD_HIP_FOUND}, RAD_WITH_HIP=${RAD_WITH_HIP})")
endif()

if(EXL3R4D_HIP)
  if(EXL3R4D_STANDALONE AND NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${EXL3R4D_TARGETS}")
  endif()
  if(EXL3R4D_STANDALONE)
    enable_language(HIP)
  endif()
  rad_add_plugin(exl3r4d kernels
    SOURCES rows.c launch.c layout.c ${EXL3_ROOT}/common/exl3_decode.c ${EXL3_ROOT}/common/exl3_layout.c
            prefill.c                    # the M > 16 band's hooks
    HIP_SOURCES exl3r4d.hip exl3r4d_prefill.hip
                exl3r4d_mod4.hip exl3r4d_prefill_mod4.hip   # the same kernels at four classes
                exl3r4d_gd_mod4.hip                         # the fused gate_up_down at four classes
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${EXL3_ROOT}/include ${EXL3_ROOT}/common
    GPU_TARGETS ${EXL3R4D_TARGETS})
  set_target_properties(exl3r4d PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
  # -ffp-contract=off on the host code too (rad_hipify sets it for the device code): the layout code does no
  # arithmetic that could contract, and the flag keeps it that way.
  target_compile_options(exl3r4d PRIVATE $<$<COMPILE_LANGUAGE:C>:${EXL3R4D_C_FLAGS}>)
  # Every device function starts on a 4 KB boundary. Two builds with instruction-identical kernels that differed only
  # in where the linker placed them measured the tp1 M=3 split chain at 146.0 vs 155.2 us a layer (the gate_up GEMM
  # inside the chain 88 vs 101 us; equal alone and at every other shape tried); with 4 KB alignment the slow placement
  # reads 146.0. The cause is not known (it is not the K loop crossing a page). The cost is padding, < 4 KB a kernel.
  target_compile_options(exl3r4d PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-falign-functions=4096>)
  string(REPLACE ";" " " _exl3r4d_bt "${EXL3R4D_TARGETS}")
  target_compile_definitions(exl3r4d PRIVATE "EXL3R4D_BUILD_TARGET=\"${_exl3r4d_bt}\"")
  find_library(EXL3R4D_LIBM m)   # common/exl3_decode.c's nearbyint; part of libc where absent
  if(EXL3R4D_LIBM)
    target_link_libraries(exl3r4d PRIVATE ${EXL3R4D_LIBM})
  endif()
endif()

# ------------------------------------------------------------------ tests (label gpu)
# The device rows through the same harness and NumPy transcription as the reference (kernels/exl3ref's
# exl3ref_run, built with HIP: `--domain device`), within the suite's relative-L2 tolerance. They need a gfx1201
# card; without python3 and numpy the suite exits 77 and ctest reports it skipped. Registered where kernels/exl3ref
# is part of the same build, which provides the harness.
if(EXL3R4D_HIP AND TARGET exl3ref_run)
  find_program(EXL3R4D_PYTHON NAMES python3 python)
  if(EXL3R4D_PYTHON)
    set(_py ${EXL3R4D_PYTHON})
  else()
    set(_py sh -c "echo SKIPPED: no python3 interpreter; exit 77" --)
  endif()
  set(_suite ${_py} ${EXL3_ROOT}/tests/ref/test_exl3ref.py --harness $<TARGET_FILE:exl3ref_run>
             --plugin $<TARGET_FILE:exl3r4d> --domain device --golden ${EXL3_ROOT}/common/tests/data/exl3_golden.npz
             --work ${CMAKE_CURRENT_BINARY_DIR}/cases
             --probes $<TARGET_FILE:layout_probe> $<TARGET_FILE:layout_probe_badinv>)
  add_test(NAME exl3r4d_cases COMMAND ${_suite} --suite cases)
  add_test(NAME exl3r4d_band_tops COMMAND ${_suite} --suite band-tops)
  set_tests_properties(exl3r4d_cases PROPERTIES LABELS gpu SKIP_RETURN_CODE 77
                       ENVIRONMENT PYTHONDONTWRITEBYTECODE=1)
  set_tests_properties(exl3r4d_band_tops PROPERTIES LABELS "gpu;slow" SKIP_RETURN_CODE 77
                       ENVIRONMENT PYTHONDONTWRITEBYTECODE=1)
endif()
