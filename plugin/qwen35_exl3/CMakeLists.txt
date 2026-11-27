# qwen35_exl3 architecture plugin
# Out-of-tree build against an installed Radiance at plugin ABI 44

if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(qwen35_exl3 LANGUAGES C CXX)
  set(RAD_PLUGIN_INSTALL_HOME "${CMAKE_BINARY_DIR}/home" CACHE PATH "the \$RADIANCE_HOME this build installs under")
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
  set(QWEN35_EXL3_ABI 44)
  if(NOT RADIANCE_ABI_VERSION EQUAL QWEN35_EXL3_ABI)
    message(FATAL_ERROR "qwen35_exl3 is written against radiance plugin ABI ${QWEN35_EXL3_ABI}; "
                        "the radiance found has ABI ${RADIANCE_ABI_VERSION}. Point CMAKE_PREFIX_PATH at a matching "
                        "install (RADIANCE_VERSION).")
  endif()
  set(QWEN35_EXL3_TESTS ON)
else()
  # In radiance's tree the tests build with the suite's (RAD_BUILD_TESTS); a standalone build always builds them.
  set(QWEN35_EXL3_TESTS ${RAD_BUILD_TESTS})
endif()

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/RADIANCE_VERSION" _qwen35_exl3_pin LIMIT_COUNT 1)
string(REGEX MATCH "^([0-9]+\\.[0-9]+\\.[0-9]+)" _pin_m "${_qwen35_exl3_pin}")
set(QWEN35_EXL3_PIN_VER "${CMAKE_MATCH_1}")

if(EXISTS "${RADIANCE_INCLUDE_DIR}/../bin/radiance")
  file(STRINGS "${RADIANCE_INCLUDE_DIR}/../bin/radiance" _rad_versions REGEX "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  if(NOT QWEN35_EXL3_PIN_VER IN_LIST _rad_versions)
    message(FATAL_ERROR "qwen35_exl3 pinned to radiance ${QWEN35_EXL3_PIN_VER} but install is ${_rad_versions}")
  endif()
endif()

rad_add_plugin(qwen35_exl3 architectures
  SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/qwen35_exl3.cpp
  INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../exl3import/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../exl3import/common
)

# The declaration test (no GPU, no model file). It carries `unit` and `fast` beside `host`, so in radiance's tree it
# runs in the suite's fast tier and in the image's host-only test stage.
if(QWEN35_EXL3_TESTS)
  enable_testing()
  add_executable(qwen35_exl3_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen35_exl3_test.cpp
  )
  target_include_directories(qwen35_exl3_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../exl3import/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../exl3import/common
    ${RADIANCE_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../tests
  )
  target_link_libraries(qwen35_exl3_test PRIVATE stdc++)
  add_test(NAME qwen35_exl3_test COMMAND qwen35_exl3_test)
  set_tests_properties(qwen35_exl3_test PROPERTIES LABELS "host;unit;fast")
endif()
