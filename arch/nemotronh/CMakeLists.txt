if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(nemotronh LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()
# Experts and linears stored as llama.cpp's blocks (NVIDIA's 2-bit release: IQ2_XXS and Q8_0;
# arch/common/rad_ggml.h): the plugin names a weight's block type through libquant's one definition
# of each type's encoding, compiled from its sources as libref compiles them. Out of tree, where
# libquant's sources are not beside the plugin, it names no type.
set(NEM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/nemotronh.cpp)
set(NEM_GGML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../libquant)
if(EXISTS ${NEM_GGML_DIR}/lq_ggml_planes.cpp)
  list(APPEND NEM_SRC ${NEM_GGML_DIR}/lq_ggml_planes.cpp ${NEM_GGML_DIR}/lq_ggml_tables.cpp)
  set(NEM_HAVE_GGML ON)
endif()
rad_add_plugin(nemotronh architectures SOURCES ${NEM_SRC})
if(NEM_HAVE_GGML)
  target_compile_definitions(nemotronh PRIVATE RAD_ARCH_HAVE_GGML=1)
  target_include_directories(nemotronh PRIVATE ${NEM_GGML_DIR})
endif()
