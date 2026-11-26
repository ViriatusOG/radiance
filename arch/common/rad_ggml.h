/* rad_ggml.h -- which GGUF block type a stored weight is, for a plugin that serves llama.cpp's own
 * blocks (gemm_nt_ggml, moe_gemm_ggml; docs/OPS.md).
 *
 * A GGUF tensor is imported as the affine planes of its type (libquant/lq_ggml_planes.h), so the
 * only thing a plugin sees is that encoding (rad_weight_encoding). ggml_type_of names the type back
 * by comparing it with libquant's one definition of each type's encoding, compiled into the plugin
 * from its sources the way libref compiles them (nothing is linked, so abi/ stays the whole
 * contract). Each type's encoding is its own (checkpoint_test: ggml_encodings_are_distinct), so
 * the answer is unambiguous. A plugin built without those sources (RAD_ARCH_HAVE_GGML unset)
 * names no type, and serves such a weight however it served it before.
 */
#pragma once
#include "rad_encoding.h"

#ifdef RAD_ARCH_HAVE_GGML
#include "lq_ggml_planes.h"
#endif

namespace rad::arch {

/* llama.cpp's spelling of the GGUF type whose encoding `e` is exactly ("iq1_s", "q6_K"), the
 * spelling the ggml ops' `type` parameter takes, and in `block` the values one of its blocks
 * covers along K; null for any other encoding. Only types whose native blocks the loader can
 * rebuild from the planes (pack_of) are named. */
inline const char* ggml_type_of(const RadEncoding& e, int64_t* block = nullptr) {
#ifdef RAD_ARCH_HAVE_GGML
    for (uint32_t t = 0; t < 64; ++t) {
        const lq::ggml::PlaneKind* k = lq::ggml::plane_kind(t);
        if (!k || !lq::ggml::pack_of(t)) continue;
        const RadEncoding want = lq::ggml::encoding_of(k->type);
        if (rad_enc_equal(&e, &want)) {
            if (block) *block = k->block;
            return k->name;
        }
    }
#endif
    (void)e;
    (void)block;
    return nullptr;
}

/* The dtype a weight stored as `e`'s planes is declared at: its codes plane's
 * (rad_builder.h: RadWeightDecl, several planes selected). */
inline uint32_t ggml_codes_dtype(const RadEncoding& e) {
    const RadEncPlane* p = rad_enc_plane(&e, "codes");
    return p ? p->dtype : (uint32_t)RAD_DT_INVALID;
}

}  // namespace rad::arch
