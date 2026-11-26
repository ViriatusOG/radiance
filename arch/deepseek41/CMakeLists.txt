if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(deepseek41 LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

# A GGUF's weights in llama.cpp's blocks (arch/common/rad_ggml.h): the plugin names a weight's type
# through libquant's one definition of each type's encoding, compiled from its sources as libref
# compiles them (nothing is linked, so abi/ stays the whole contract). Out of tree, without those
# sources, no weight is read as blocks.
set(DS41_SRC ${CMAKE_CURRENT_SOURCE_DIR}/deepseek41.cpp)
set(DS41_GGML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../libquant)
if(EXISTS ${DS41_GGML_DIR}/lq_ggml_planes.cpp)
  list(APPEND DS41_SRC ${DS41_GGML_DIR}/lq_ggml_planes.cpp ${DS41_GGML_DIR}/lq_ggml_tables.cpp)
  set(DS41_HAVE_GGML ON)
endif()
rad_add_plugin(deepseek41 architectures SOURCES ${DS41_SRC})
if(DS41_HAVE_GGML)
  target_compile_definitions(deepseek41 PRIVATE RAD_ARCH_HAVE_GGML=1)
  target_include_directories(deepseek41 PRIVATE ${DS41_GGML_DIR})
endif()
