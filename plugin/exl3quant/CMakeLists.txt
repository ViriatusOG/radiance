# exl3quant -- the `exl3i` quantiser plugin: imports EXL3-V3 levels (the EXL3-V3 quantizer's trellises, any rate in 1/8
# steps, codebook mad) into the exl3i encoding. Host code only; builds the same with or without HIP.
#
# Standalone against an installed radiance at plugin ABI 44:
#   cmake -S plugin/exl3quant -B build-exl3quant -G Ninja -DCMAKE_PREFIX_PATH=/path/to/radiance-install
#   cmake --build build-exl3quant -j4
# In radiance's tree it is an ordinary quantiser plugin (rad_add_plugin is already defined).
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(exl3quant LANGUAGES C CXX)
  set(RAD_PLUGIN_INSTALL_HOME "${CMAKE_BINARY_DIR}/home" CACHE PATH "the $RADIANCE_HOME this build installs under")
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
  set(EXL3QUANT_ABI 44)
  if(NOT RADIANCE_ABI_VERSION EQUAL EXL3QUANT_ABI)
    message(FATAL_ERROR "exl3quant is written against radiance plugin ABI ${EXL3QUANT_ABI} "
                        "; the radiance found has ABI ${RADIANCE_ABI_VERSION}. "
                        "Point CMAKE_PREFIX_PATH at a matching install (RADIANCE_VERSION).")
  endif()
  enable_testing()
  set(EXL3QUANT_TESTS ON)
else()
  # In radiance's tree the test builds with the suite's (RAD_BUILD_TESTS); a standalone build always builds it.
  set(EXL3QUANT_TESTS ${RAD_BUILD_TESTS})
  if(EXL3QUANT_TESTS)
    enable_testing()
  endif()
endif()

enable_language(C)

set(EXL3IMPORT ${CMAKE_CURRENT_SOURCE_DIR}/../exl3import)
set(EXL3QUANT_COMMON
    ${CMAKE_CURRENT_SOURCE_DIR}/exl3_quant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/exl3_src.cpp
    ${EXL3IMPORT}/common/exl3_decode.c
    ${EXL3IMPORT}/common/exl3i_decode.c)
set(EXL3QUANT_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EXL3IMPORT}/common
    ${EXL3IMPORT}/include)

rad_add_plugin(exl3quant quantizers SOURCES ${EXL3QUANT_COMMON} INCLUDES ${EXL3QUANT_INCLUDES})

# exl3_decode.c's orderings are bit-exact: an FMA that fused a multiply into a Hadamard sum would
# move the oracle (and so the convert-time error report) off exllamav3's own decode.
set_source_files_properties(${EXL3IMPORT}/common/exl3_decode.c ${EXL3IMPORT}/common/exl3i_decode.c
                            PROPERTIES COMPILE_OPTIONS "-ffp-contract=off"
                                       C_STANDARD 99 C_STANDARD_REQUIRED ON)
target_compile_options(exl3quant PRIVATE -ffp-contract=off)

# The test carries `unit` and `fast` beside `host`, so in radiance's tree it runs in the suite's fast tier and in the
# image's host-only test stage.
if(EXL3QUANT_TESTS)
  # The quantiser through its plugin ABI: a synthetic level (fractional trellises, stamp.json), imported and decoded
  # against the fixture's own numbers.
  add_executable(exl3quant_test tests/exl3quant_test.cpp
      ${EXL3IMPORT}/common/exl3_decode.c
      ${EXL3IMPORT}/common/exl3i_decode.c)
  set_source_files_properties(${EXL3IMPORT}/common/exl3_decode.c
                              PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  find_library(RAD_DL_LIB dl)   # glibc >= 2.34 folds libdl into libc; link it only when it exists
  if(RAD_DL_LIB)
    target_link_libraries(exl3quant_test ${RAD_DL_LIB})
  endif()
  target_include_directories(exl3quant_test PRIVATE ${RADIANCE_INCLUDE_DIR} ${EXL3QUANT_INCLUDES})
  add_test(NAME exl3quant_test COMMAND exl3quant_test $<TARGET_FILE:exl3quant>)
  set_tests_properties(exl3quant_test PROPERTIES LABELS "host;unit;fast")
endif()
