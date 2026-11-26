# radiance-exl3 in the tree: EXL3 trellis experts (exllamav3's format) for radiance (Apache-2.0;
# LICENSE and NOTICE here). Built with the tree's own rad_add_plugin, like libr4d and libquant:
#   kernels/exl3r4d     the gfx1201 kernel library (moe_exl3_gate_up / moe_exl3_down and friends)
#   kernels/exl3ref     the host reference library for the same ops (declares itself a reference)
#   quantizers/exl3     the `exl3` quantiser, which imports packed EXL3 planes into a .rad
#   common/             the canonical EXL3 decode and layout, shared by all three, and its C tests
#   include/            the interface headers: op schemas, operand positions, EXL3_IFACE_VERSION
# arch/common/exl3/ vendors include/exl3_abi.h, include/exl3_encoding.h and common/exl3_decode.h for
# the architecture plugins (arch/common is installed on its own); exl3_vendor_check holds the copies
# to these.
#
# Each subdirectory also builds on its own against an installed radiance
# (`cmake -S exl3/kernels/exl3r4d -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`).
set(EXL3_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(EXL3_IN_TREE ON)
enable_language(C)
if(RAD_BUILD_TESTS)
  enable_testing()
endif()
# The docs check (exl3ref_ops_doc) reads tools/ops_doc_check.sh from this source tree.
set(RADIANCE_SRC ${PROJECT_SOURCE_DIR})

add_subdirectory(common)
add_subdirectory(quantizers/exl3)
add_subdirectory(kernels/exl3ref)
add_subdirectory(kernels/exl3r4d)

if(RAD_BUILD_TESTS)
  # The architecture plugins' copies of the interface headers must be these, apart from
  # exl3_encoding.h's one include path ("../common/exl3_decode.h" -> "exl3_decode.h").
  set(_v ${PROJECT_SOURCE_DIR}/arch/common/exl3)
  add_test(NAME exl3_vendor_check
           COMMAND sh -c "cmp \"$0/include/exl3_abi.h\" \"$1/exl3_abi.h\" && cmp \"$0/common/exl3_decode.h\" \"$1/exl3_decode.h\" && sed 's|\"../common/exl3_decode.h\"|\"exl3_decode.h\"|' \"$0/include/exl3_encoding.h\" | cmp - \"$1/exl3_encoding.h\""
                   ${EXL3_ROOT} ${_v})
  set_tests_properties(exl3_vendor_check PROPERTIES LABELS "host;unit")

  # The loader refuses a whole kernel library over one op whose schema disagrees with a library
  # loaded before it, or over another ABI version or layout fingerprint. rad-schemas applies the
  # same checks to the exl3 libraries beside the tree's own, in hierarchy order, and exits nonzero
  # on any of them. No card is needed: it only reads the static schema tables. Libraries defined
  # later in the configure (plugin/exl3i) are taken by TARGET_EXISTS, which is evaluated once every
  # directory has been read.
  set(_exl3_libs)
  foreach(_lib libr4d exl3r4d exl3i libavx libref exl3ref)
    list(APPEND _exl3_libs "$<$<TARGET_EXISTS:${_lib}>:${RAD_PLUGIN_OUT}/kernels/${_lib}.so>")
  endforeach()
  add_test(NAME exl3_schemas_agree COMMAND rad-schemas ${_exl3_libs} COMMAND_EXPAND_LISTS)
  set_tests_properties(exl3_schemas_agree PROPERTIES LABELS "host;unit"
                       ENVIRONMENT "ROCR_VISIBLE_DEVICES=99")
endif()
