# gemma4_exl3 architecture plugin: gemma4_bf16 with EXL3-V3 trunk linears
# Out-of-tree build against an installed Radiance at plugin ABI 44

if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(gemma4_exl3 LANGUAGES C CXX)
  set(RAD_PLUGIN_INSTALL_HOME "${CMAKE_BINARY_DIR}/home" CACHE PATH "the \$RADIANCE_HOME this build installs under")
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
  set(GEMMA4_EXL3_ABI 44)
  if(NOT RADIANCE_ABI_VERSION EQUAL GEMMA4_EXL3_ABI)
    message(FATAL_ERROR "gemma4_exl3 is written against radiance plugin ABI ${GEMMA4_EXL3_ABI}; "
                        "the radiance found has ABI ${RADIANCE_ABI_VERSION}. Point CMAKE_PREFIX_PATH at a matching "
                        "install (RADIANCE_VERSION).")
  endif()
  set(GEMMA4_EXL3_TESTS ON)
else()
  # In radiance's tree the tests build with the suite's (RAD_BUILD_TESTS); a standalone build always builds them.
  set(GEMMA4_EXL3_TESTS ${RAD_BUILD_TESTS})
endif()

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/RADIANCE_VERSION" _gemma4_exl3_pin LIMIT_COUNT 1)
string(REGEX MATCH "^([0-9]+\\.[0-9]+\\.[0-9]+)" _pin_m "${_gemma4_exl3_pin}")
set(GEMMA4_EXL3_PIN_VER "${CMAKE_MATCH_1}")

if(EXISTS "${RADIANCE_INCLUDE_DIR}/../bin/radiance")
  file(STRINGS "${RADIANCE_INCLUDE_DIR}/../bin/radiance" _rad_versions REGEX "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  if(NOT GEMMA4_EXL3_PIN_VER IN_LIST _rad_versions)
    message(FATAL_ERROR "gemma4_exl3 pinned to radiance ${GEMMA4_EXL3_PIN_VER} but install is ${_rad_versions}")
  endif()
endif()

# exl3_dense_op.h (the dense EXL3 projection helpers) is qwen35_exl3's: included, not copied.
set(GEMMA4_EXL3_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/../qwen35_exl3
    ${CMAKE_CURRENT_SOURCE_DIR}/../exl3import/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../exl3import/common
)

rad_add_plugin(gemma4_exl3 architectures
  SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/gemma4_exl3.cpp
  INCLUDES
    ${GEMMA4_EXL3_INCLUDES}
)

# The tests need no GPU and no model file. They carry `unit` and `fast` beside `host`, so in radiance's tree they run in
# the suite's fast tier and in the image's host-only test stage.
if(GEMMA4_EXL3_TESTS)
  enable_testing()
  # The declared graph.
  add_executable(gemma4_exl3_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/gemma4_exl3_test.cpp
  )
  target_include_directories(gemma4_exl3_test PRIVATE
    ${GEMMA4_EXL3_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RADIANCE_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../tests
  )
  target_link_libraries(gemma4_exl3_test PRIVATE stdc++)
  add_test(NAME gemma4_exl3_test COMMAND gemma4_exl3_test)
  set_tests_properties(gemma4_exl3_test PROPERTIES LABELS "host;unit;fast")

  # The token-table path in numbers: a quantised table's gather (embed_tq, embed_exl3i: plugin/exl3i's host rows,
  # linked here) scaled as the plugin declares it, against embed_lookup with the same mult. fill, mul and embed_lookup
  # are libref's, from $RADIANCE_HOME/kernels; the test exits 77 (skipped) without it.
  set(GEMMA4_EXL3_EXL3I ${CMAKE_CURRENT_SOURCE_DIR}/../exl3i)
  add_executable(gemma4_exl3_embed_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/gemma4_exl3_embed_test.cpp
    ${GEMMA4_EXL3_EXL3I}/embed_tq.cpp
    ${GEMMA4_EXL3_EXL3I}/embed_exl3i.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../exl3import/common/exl3_decode.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../exl3import/common/exl3i_decode.c
  )
  target_include_directories(gemma4_exl3_embed_test PRIVATE
    ${GEMMA4_EXL3_EXL3I}
    ${GEMMA4_EXL3_INCLUDES}
    ${RADIANCE_INCLUDE_DIR}
  )
  target_compile_options(gemma4_exl3_embed_test PRIVATE -ffp-contract=off)
  target_link_libraries(gemma4_exl3_embed_test PRIVATE ${CMAKE_DL_LIBS} m)
  if(RADIANCE_HOME)   # find_package(radiance): the installed home
    set(_gemma4_exl3_ref_home "${RADIANCE_HOME}")
  else()              # radiance's tree: the build's own home, where its libref is built before the test runs
    set(_gemma4_exl3_ref_home "${RAD_PLUGIN_OUT}")
    if(TARGET libref)
      add_dependencies(gemma4_exl3_embed_test libref)
    endif()
  endif()
  add_test(NAME gemma4_exl3_embed_test COMMAND gemma4_exl3_embed_test)
  set_tests_properties(gemma4_exl3_embed_test PROPERTIES LABELS "host;unit;fast" SKIP_RETURN_CODE 77
                       ENVIRONMENT "RADIANCE_HOME=${_gemma4_exl3_ref_home}")
endif()
