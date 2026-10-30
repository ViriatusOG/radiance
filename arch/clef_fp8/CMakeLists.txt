# Built in radiance's tree, or standalone against an installed radiance
# (`cmake -S arch/clef_fp8 -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`); see cmake/RadAddPlugin.cmake.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(clef_fp8 LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

rad_add_plugin(clef_fp8 architectures SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/clef_fp8.cpp)
