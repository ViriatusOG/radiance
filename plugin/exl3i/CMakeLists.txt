# exl3i -- the gfx1201 kernel library for EXL3-V3's dense linear (exl3i_dense). Built from the repository's top level,
# or on its own against an installed radiance:
#
#   cmake -S plugin/exl3i -B build-exl3i -DCMAKE_PREFIX_PATH=<radiance install prefix>
#
# DEVICE ONLY (a device build targeting gfx1201). The kernel is the dense GEMM over the device headers kept in this
# directory (exl3r4d_common.hip.h, exl3r4d_dev.h, exl3r4d_gemm_fields.h) with the decode a rate policy; the one codebook
# (mad, the byte-sum GEMV in exl3i_mad.hip.h) compiles as one translation unit (exl3i_nt_mad.hip). The A8 path
# (exl3i_a8.h: mad with an int8 activation mode, from 512 rows) is one more: its producer passes (exl3i_a8_prep.hip) and
# its GEMM (exl3i_a8_gemm.hip, radiance's i8a8 prefill tile carried as exl3i_8a8.hip.h), apart because the two device
# libraries share helper names.
# The token-embedding gathers (embed_exl3i.*, embed_tq.*) are in this
# library too: their device rows in the HIP build (with their host rows, the references), their host test without a device.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(exl3i LANGUAGES C CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
  enable_testing()
  set(EXL3I_STANDALONE ON)
  # Against an installed radiance only: in the tree the plugin compiles against the tree's own abi/ headers, and
  # RADIANCE_ABI_VERSION is what find_package(radiance) defines.
  if(NOT RADIANCE_ABI_VERSION EQUAL 44)
    message(FATAL_ERROR "exl3i is written against radiance ABI 44; the radiance found is ABI ${RADIANCE_ABI_VERSION}")
  endif()
  set(EXL3I_TESTS ON)
else()
  # In radiance's tree the tests build with the suite's (RAD_BUILD_TESTS); a standalone build always builds them.
  set(EXL3I_TESTS ${RAD_BUILD_TESTS})
  if(EXL3I_TESTS)
    enable_testing()
  endif()
endif()
enable_language(C)


set(EXL3I_C_FLAGS -ffp-contract=off -Wall -Wextra)
set(EXL3I_COMMON ${CMAKE_CURRENT_SOURCE_DIR}/../exl3import/common)
set(EXL3I_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../exl3import/include ${EXL3I_COMMON})

find_library(EXL3I_LIBM m)

# The host tests carry `unit` and `fast` beside `host`, so in radiance's tree they run in the suite's fast tier
# (docs/GUIDE.md §1.3) and in the image's host-only test stage; the device tests carry `gpu` and the `card` lock.
if(EXL3I_TESTS)
  # The host plan test needs no device: the run-time plan against the oracle, every fractional rate and lane.
  add_executable(exl3i_plan_test tests/exl3i_plan_test.c ${EXL3I_COMMON}/exl3i_decode.c ${EXL3I_COMMON}/exl3_decode.c)
  target_include_directories(exl3i_plan_test PRIVATE ${EXL3I_INCLUDES})
  target_compile_options(exl3i_plan_test PRIVATE -ffp-contract=off)
  if(EXL3I_LIBM)
    target_link_libraries(exl3i_plan_test ${EXL3I_LIBM})
  endif()
  add_test(NAME exl3i_plan_test COMMAND exl3i_plan_test)
  set_tests_properties(exl3i_plan_test PROPERTIES LABELS "host;unit;fast")
  # The compile-time lane plans (exl3i_plan_tab.h, what RateFrac reads) against exl3i_plan(), every rate and lane.
  add_executable(exl3i_plan_tab_test tests/exl3i_plan_tab_test.cpp)
  target_include_directories(exl3i_plan_tab_test PRIVATE ${EXL3I_INCLUDES})
  set_target_properties(exl3i_plan_tab_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  add_test(NAME exl3i_plan_tab_test COMMAND exl3i_plan_tab_test)
  set_tests_properties(exl3i_plan_tab_test PROPERTIES LABELS "host;unit;fast")

  # The two-pass prefill's host test (tests/exl3i_twopass_host_test.c): the planes' layout, the passes' lane arithmetic, the
  # arena, nt.c / rows.c's dispatch with the device launchers stubbed. No device. With libr4d in the tree it also checks
  # the int8 plane against radiance's own r4d_fp8_frag.h.
  set(EXL3I_TP_SRC tests/exl3i_twopass_host_test.c tests/exl3i_twopass_ref.c nt.c rows.c rerank.c pin.c
      embed_exl3i.cpp embed_tq.cpp  # rows.c's table lists the two embedding gathers, so their objects link here too
      ${EXL3I_COMMON}/exl3i_decode.c ${EXL3I_COMMON}/exl3_decode.c)
  set(EXL3I_TP_DEFS "")
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../libr4d/r4d_fp8_frag.h)
    list(APPEND EXL3I_TP_SRC tests/exl3i_twopass_frag.cpp)
  else()
    list(APPEND EXL3I_TP_DEFS EXL3I_NO_LIBR4D)
  endif()
  add_executable(exl3i_twopass_host_test ${EXL3I_TP_SRC})
  target_include_directories(exl3i_twopass_host_test PRIVATE ${EXL3I_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_compile_definitions(exl3i_twopass_host_test PRIVATE ${EXL3I_TP_DEFS})
  target_compile_options(exl3i_twopass_host_test PRIVATE $<$<COMPILE_LANGUAGE:C>:-ffp-contract=off>)
  target_compile_options(exl3i_twopass_host_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  target_link_libraries(exl3i_twopass_host_test radiance::abi)
  if(EXL3I_LIBM)
    target_link_libraries(exl3i_twopass_host_test ${EXL3I_LIBM})
  endif()
  add_test(NAME exl3i_twopass_host_test COMMAND exl3i_twopass_host_test)
  set_tests_properties(exl3i_twopass_host_test PROPERTIES LABELS "host;unit;fast")

  # exl3i_rerank's host reference (rerank_ref.c) against the decode form's value in double and the oracle's weight
  # (tests/exl3i_rerank_host_test.c). No device.
  add_executable(exl3i_rerank_host_test tests/exl3i_rerank_host_test.c rerank_ref.c ${EXL3I_COMMON}/exl3i_decode.c
                 ${EXL3I_COMMON}/exl3_decode.c)
  target_include_directories(exl3i_rerank_host_test PRIVATE ${EXL3I_INCLUDES})
  target_compile_options(exl3i_rerank_host_test PRIVATE -ffp-contract=off)
  target_link_libraries(exl3i_rerank_host_test radiance::abi)
  if(EXL3I_LIBM)
    target_link_libraries(exl3i_rerank_host_test ${EXL3I_LIBM})
  endif()
  add_test(NAME exl3i_rerank_host_test COMMAND exl3i_rerank_host_test)
  set_tests_properties(exl3i_rerank_host_test PROPERTIES LABELS "host;unit;fast")

  # The embedding gathers' host rows against their oracles (exl3i_decode_rows + one RNE bf16), every codebook, rate and
  # row order; the layout and the malformed calls. No device.
  add_executable(embed_exl3i_test tests/embed_exl3i_test.cpp embed_exl3i.cpp ${EXL3I_COMMON}/exl3_decode.c
                 ${EXL3I_COMMON}/exl3i_decode.c)
  target_include_directories(embed_exl3i_test PRIVATE ${EXL3I_INCLUDES})
  target_compile_options(embed_exl3i_test PRIVATE -ffp-contract=off)
  target_link_libraries(embed_exl3i_test PRIVATE radiance::abi)
  if(EXL3I_LIBM)
    target_link_libraries(embed_exl3i_test PRIVATE ${EXL3I_LIBM})
  endif()
  add_test(NAME embed_exl3i_test COMMAND embed_exl3i_test)
  set_tests_properties(embed_exl3i_test PROPERTIES LABELS "host;unit;fast")
endif()

set(EXL3I_HIP OFF)
if(RAD_HIP_FOUND AND NOT (DEFINED RAD_WITH_HIP AND RAD_WITH_HIP STREQUAL "OFF"))
  rad_plugin_gpu_targets(EXL3I_TARGETS "^gfx1201$")
  if(EXL3I_TARGETS)
    set(EXL3I_HIP ON)
  else()
    message(STATUS "exl3i: left out -- this build targets '${RAD_GPU_TARGETS}', and the library is gfx1201 code")
  endif()
else()
  message(STATUS "exl3i: left out -- no device build (RAD_HIP_FOUND=${RAD_HIP_FOUND}, RAD_WITH_HIP=${RAD_WITH_HIP})")
endif()

if(EXL3I_HIP)
  if(EXL3I_STANDALONE AND NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${EXL3I_TARGETS}")
  endif()
  if(EXL3I_STANDALONE)
    enable_language(HIP)
  endif()
  rad_add_plugin(exl3i kernels
    SOURCES rows.c nt.c rerank.c pin.c embed_exl3i.cpp embed_tq.cpp ${EXL3I_COMMON}/exl3i_decode.c ${EXL3I_COMMON}/exl3_decode.c
    HIP_SOURCES exl3i_nt_mad.hip exl3i_a8_prep.hip exl3i_a8_gemm.hip
                exl3i_rerank.hip embed_exl3i.hip embed_tq.hip
    INCLUDES ${EXL3I_INCLUDES}
    GPU_TARGETS ${EXL3I_TARGETS})
  set_target_properties(exl3i PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
  target_compile_options(exl3i PRIVATE $<$<COMPILE_LANGUAGE:C>:${EXL3I_C_FLAGS}>)
  option(EXL3I_ALIGN_PAGE "start every exl3i device function on a 4 KB boundary " ON)
  if(EXL3I_ALIGN_PAGE)
    target_compile_options(exl3i PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-falign-functions=4096>)
  endif()
  string(REPLACE ";" " " _exl3i_bt "${EXL3I_TARGETS}")
  # The build id the plugin reports (rad_plugin_info's description): the tree's commit, "+dirty" with local changes.
  # A tree copied off its checkout has no git: -DEXL3I_BUILD_ID=<commit> says which it is.
  set(EXL3I_BUILD_ID "" CACHE STRING "the build id to report (default: git describe of the source tree)")
  set(_exl3i_id "${EXL3I_BUILD_ID}")
  if(NOT _exl3i_id)
    execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} describe --always --dirty=+dirty --abbrev=12
                    OUTPUT_VARIABLE _exl3i_id OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  endif()
  if(NOT _exl3i_id)
    set(_exl3i_id "nogit")
  endif()
  target_compile_definitions(exl3i PRIVATE "EXL3I_BUILD_TARGET=\"${_exl3i_bt}\"" "EXL3I_BUILD_ID=\"${_exl3i_id}\"")
  if(EXL3I_LIBM)
    target_link_libraries(exl3i PRIVATE ${EXL3I_LIBM})
  endif()

endif()

if(EXL3I_HIP AND EXL3I_TESTS)
  # The device selftest (tests/exl3i_gpu_test.hip): the row through its plugin ABI against the C oracle.
  add_executable(exl3i_gpu_test tests/exl3i_gpu_test.hip pin.c ${EXL3I_COMMON}/exl3i_decode.c ${EXL3I_COMMON}/exl3_decode.c)
  set_source_files_properties(tests/exl3i_gpu_test.hip PROPERTIES LANGUAGE HIP)
  target_include_directories(exl3i_gpu_test PRIVATE ${EXL3I_INCLUDES} ${RADIANCE_INCLUDE_DIR})
  find_library(EXL3I_DL dl)
  if(EXL3I_DL)
    target_link_libraries(exl3i_gpu_test ${EXL3I_DL})
  endif()
  if(EXL3I_LIBM)
    target_link_libraries(exl3i_gpu_test ${EXL3I_LIBM})
  endif()
  add_test(NAME exl3i_gpu_test COMMAND exl3i_gpu_test $<TARGET_FILE:exl3i>)
  set_tests_properties(exl3i_gpu_test PROPERTIES LABELS "gpu" RESOURCE_LOCK card)

  # The two-pass prefill's device test (tests/test_twopass.hip): both rows through the ABI, the scratch's planes against
  # the C reference byte for byte, the selftest matrix. And its timing driver (tests/bench_twopass.hip; not a test).
  add_executable(test_twopass tests/test_twopass.hip tests/exl3i_twopass_ref.c pin.c ${EXL3I_COMMON}/exl3i_decode.c
                 ${EXL3I_COMMON}/exl3_decode.c)
  set_source_files_properties(tests/test_twopass.hip PROPERTIES LANGUAGE HIP)
  set_source_files_properties(tests/exl3i_twopass_ref.c PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  target_include_directories(test_twopass PRIVATE ${EXL3I_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR}/tests ${RADIANCE_INCLUDE_DIR})
  # The decode form's K split across workgroups (tests/test_ks.hip): split vs unsplit at the trunk shapes, repeatability.
  add_executable(test_ks tests/test_ks.hip pin.c ${EXL3I_COMMON}/exl3i_decode.c ${EXL3I_COMMON}/exl3_decode.c)
  set_source_files_properties(tests/test_ks.hip PROPERTIES LANGUAGE HIP)
  target_include_directories(test_ks PRIVATE ${EXL3I_INCLUDES} ${RADIANCE_INCLUDE_DIR})
  # The R16 row order (tests/test_r16.hip): natural vs R16 launches of one record, every path, outputs at their R16 rows.
  add_executable(test_r16 tests/test_r16.hip ${EXL3I_COMMON}/exl3i_decode.c ${EXL3I_COMMON}/exl3_decode.c)
  set_source_files_properties(tests/test_r16.hip PROPERTIES LANGUAGE HIP)
  target_include_directories(test_r16 PRIVATE ${EXL3I_INCLUDES} ${RADIANCE_INCLUDE_DIR})
  # AQ (tests/test_aq.hip): exl3i_actq's planes vs the host reference, the dense op with aq vs quantising itself.
  add_executable(test_aq tests/test_aq.hip tests/exl3i_twopass_ref.c ${EXL3I_COMMON}/exl3i_decode.c ${EXL3I_COMMON}/exl3_decode.c)
  set_source_files_properties(tests/test_aq.hip PROPERTIES LANGUAGE HIP)
  target_include_directories(test_aq PRIVATE ${EXL3I_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR}/tests ${RADIANCE_INCLUDE_DIR})
  # exl3i_rerank (tests/test_rerank.hip): the device row == rerank_ref.c bit for bit, vs exl3i_dense's logits, its time.
  add_executable(test_rerank tests/test_rerank.hip rerank_ref.c ${EXL3I_COMMON}/exl3i_decode.c ${EXL3I_COMMON}/exl3_decode.c)
  set_source_files_properties(tests/test_rerank.hip PROPERTIES LANGUAGE HIP)
  set_source_files_properties(rerank_ref.c PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  target_include_directories(test_rerank PRIVATE ${EXL3I_INCLUDES} ${RADIANCE_INCLUDE_DIR})
  add_executable(bench_twopass tests/bench_twopass.hip exl3i_a8_prep.hip exl3i_a8_gemm.hip pin.c)
  set_source_files_properties(tests/bench_twopass.hip PROPERTIES LANGUAGE HIP)
  target_include_directories(bench_twopass PRIVATE ${EXL3I_INCLUDES} ${RADIANCE_INCLUDE_DIR})
  # its copies of the A8 passes (the phases rows) compiled as the plugin's are: the device level, -ffp-contract=off,
  # the VOPD setting (without, CMake's default HIP flags: -O2 and the compiler's VOPD pairing)
  rad_hipify(bench_twopass ${EXL3I_TARGETS})
  foreach(t test_twopass test_ks test_r16 test_aq test_rerank bench_twopass)
    target_link_libraries(${t} PRIVATE radiance::abi)
    if(EXL3I_DL)
      target_link_libraries(${t} PRIVATE ${EXL3I_DL})
    endif()
    if(EXL3I_LIBM)
      target_link_libraries(${t} PRIVATE ${EXL3I_LIBM})
    endif()
  endforeach()
  add_test(NAME exl3i_twopass COMMAND test_twopass $<TARGET_FILE:exl3i>)
  set_tests_properties(exl3i_twopass PROPERTIES LABELS "gpu" RESOURCE_LOCK card)
  # The A8 path's GEMM decodes its own weight tiles (exl3i_a8_prep.hip gemmd): exl3i_twopass runs it, exl3i_twopass_nofuse
  # the two passes it replaces (the EXL3I_PIN_A8_FUSE_OFF pin); a timeout, not a hung ctest, should the fused GEMM hang
  set_tests_properties(exl3i_twopass PROPERTIES TIMEOUT 600)
  add_test(NAME exl3i_twopass_nofuse COMMAND test_twopass $<TARGET_FILE:exl3i> --nofuse)
  set_tests_properties(exl3i_twopass_nofuse PROPERTIES LABELS "gpu" RESOURCE_LOCK card TIMEOUT 600)
  add_test(NAME exl3i_ks COMMAND test_ks $<TARGET_FILE:exl3i>)
  set_tests_properties(exl3i_ks PROPERTIES LABELS "gpu" RESOURCE_LOCK card)
  add_test(NAME exl3i_r16 COMMAND test_r16 $<TARGET_FILE:exl3i>)
  set_tests_properties(exl3i_r16 PROPERTIES LABELS "gpu" RESOURCE_LOCK card)
  add_test(NAME exl3i_aq COMMAND test_aq $<TARGET_FILE:exl3i>)
  set_tests_properties(exl3i_aq PROPERTIES LABELS "gpu" RESOURCE_LOCK card)
  add_test(NAME exl3i_rerank COMMAND test_rerank $<TARGET_FILE:exl3i> --quick)
  set_tests_properties(exl3i_rerank PROPERTIES LABELS "gpu" RESOURCE_LOCK card TIMEOUT 900)
  # The embedding gathers' device rows against their host rows, bit for bit, and their time at the 27B's table beside
  # embed_tq's (tests/embed_exl3i_gpu_test.hip; label gpu, needs the card).
  add_executable(embed_exl3i_gpu_test tests/embed_exl3i_gpu_test.hip embed_exl3i.hip embed_exl3i.cpp embed_tq.hip
                 embed_tq.cpp ${EXL3I_COMMON}/exl3_decode.c ${EXL3I_COMMON}/exl3i_decode.c)
  set_source_files_properties(tests/embed_exl3i_gpu_test.hip PROPERTIES LANGUAGE HIP)
  target_include_directories(embed_exl3i_gpu_test PRIVATE ${EXL3I_INCLUDES} ${RADIANCE_INCLUDE_DIR})
  target_link_libraries(embed_exl3i_gpu_test PRIVATE radiance::abi)
  if(EXL3I_DL)
    target_link_libraries(embed_exl3i_gpu_test PRIVATE ${EXL3I_DL})
  endif()
  if(EXL3I_LIBM)
    target_link_libraries(embed_exl3i_gpu_test PRIVATE ${EXL3I_LIBM})
  endif()
  add_test(NAME embed_exl3i_gpu_test COMMAND embed_exl3i_gpu_test)
  set_tests_properties(embed_exl3i_gpu_test PROPERTIES LABELS "gpu" RESOURCE_LOCK card)
  # The device tests are gfx1201 code like the library (the mad GEMV's sudot4 has no form on most
  # other targets), so they are compiled for the library's targets, not for every target of a tree
  # that also builds for other cards.
  set_target_properties(exl3i_gpu_test test_twopass test_ks test_r16 test_aq test_rerank embed_exl3i_gpu_test
                        PROPERTIES HIP_ARCHITECTURES "${EXL3I_TARGETS}")
endif()
