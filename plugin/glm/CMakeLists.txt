# glm -- GLM-5.3-Flash's own kernels, a kernel plugin beside libr4d.
#
# Ops only glm5next issues (and only where a loaded library declares them), each with a device row
# and a host reference row. The device sources include libr4d's device headers by their path in this
# tree (the Hadamard, the int8 quantiser, the emit helper), so the bytes are libr4d's; a host-only
# build needs abi/ alone. Load it ahead of libr4d: `kernels: glm,libr4d,libavx,libref`.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(glm LANGUAGES CXX)
  find_package(radiance REQUIRED)
  if(RAD_HIP_FOUND)
    enable_language(HIP)
  endif()
  rad_plugin_defaults()
endif()

set(GLM_COMMON
    ${CMAKE_CURRENT_SOURCE_DIR}/glm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ref_glm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/exports.cpp)

if(NOT RAD_HIP_FOUND)
  message(STATUS "glm: this radiance has no device build -- only the host reference is built")
  rad_add_plugin(glm kernels SOURCES ${GLM_COMMON})
else()
  get_property(glm_langs GLOBAL PROPERTY ENABLED_LANGUAGES)
  if(NOT "HIP" IN_LIST glm_langs)
    message(FATAL_ERROR "glm: HIP_SOURCES need `enable_language(HIP)` in this project (set "
                        "-DRAD_WITH_HIP=ON, or enable it after project()).")
  endif()
  rad_add_plugin(glm kernels SOURCES ${GLM_COMMON}
                 HIP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/glm.hip)
  target_compile_definitions(glm PRIVATE RAD_GLM_WITH_DEVICE=1)
endif()

# The host reference rounds as the device does: no FMA contraction.
target_compile_options(glm PRIVATE -ffp-contract=off)
