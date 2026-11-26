/* r4d_ds41_ggml_shapes.inc -- operand descriptions of the ggml ops (the GEMMs over llama.cpp's
 * native blocks); included at the end of r4d_shapes.cpp for its helpers (opd, opd_idx, opd_none,
 * pick, R4D_PICK, pi, dt_act).
 *
 * libref's (ref_shapes.cpp, ref_*_ggml_shape) operand for operand: each weight in its float
 * logical form ([N, K] f32, [n_expert, N, K] f32 for the grouped GEMM), which every row reads as
 * it is; a real container's weights reach the rows through their layout hooks instead. A type the
 * rows do not read, or K not whole blocks of it, is a decline. */

namespace {
/* Values one block of GGUF type `type` covers, or 0 for a type these rows do not read (libref's
 * ref_ggml_block: the types with both a pack and a dequant in libquant). */
int64_t ds41_ggml_block(const RadParam* p, int n_p) {
    const char* s = r4d_pgets(p, n_p, "type");
    if (!s) return 0;
    if (!std::strcmp(s, "q8_0") || !std::strcmp(s, "iq4_nl")) return 32;
    if (!std::strcmp(s, "q5_K") || !std::strcmp(s, "q6_K") || !std::strcmp(s, "iq2_xxs") ||
        !std::strcmp(s, "iq1_s") || !std::strcmp(s, "iq3_xxs"))
        return 256;
    return 0;
}
}  // namespace

extern "C" int r4d_shape_gemm_nt_ggml(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t blk = ds41_ggml_block(p, n_p);
    const int64_t M = pi(p, n_p, "M"), N = pi(p, n_p, "N"), K = pi(p, n_p, "K");
    if (blk <= 0 || M <= 0 || N <= 0 || K <= 0 || K % blk) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, K),                       /* a */
        opd((uint32_t)RAD_F32, N, K),       /* w, the logical form */
        opd(D, M, N),                       /* y */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_moe_gemm_ggml(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t blk = ds41_ggml_block(p, n_p);
    const int64_t M = pi(p, n_p, "M"), N = pi(p, n_p, "N"), K = pi(p, n_p, "K");
    const int64_t ne = pi(p, n_p, "n_expert"), k = pi(p, n_p, "top_k");
    if (blk <= 0 || M <= 0 || N <= 0 || K <= 0 || K % blk || ne <= 0 || k <= 0)
        return RAD_E_UNSUPPORTED;
    const char* ord = r4d_pgets(p, n_p, "a_order");
    const int64_t AR = (ord && !std::strcmp(ord, "sorted")) ? M * k : M;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, AR, K),                                                   /* a -- see a_order */
        opd((uint32_t)RAD_F32, ne, N, K),                                /* w, the logical form */
        /* A valid routing, as moe_scatter produces it (libref's moe_gemm description). */
        opd_idx(M * k, 0, M * k, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE),   /* sorted_tok */
        opd_idx(ne + 1, 0, M * k + 1, -1, (uint32_t)RAD_OPD_F_IDX_PREFIX), /* expert_offset */
        opd(D, M * k, N),                                                /* y, sorted order */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_logits_gemm_ggml(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t blk = ds41_ggml_block(p, n_p);
    const int64_t M = pi(p, n_p, "M"), nv = pi(p, n_p, "n_vocab"), ne = pi(p, n_p, "n_embd");
    if (blk <= 0 || M <= 0 || nv <= 0 || ne <= 0 || ne % blk) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, ne),                      /* x */
        opd((uint32_t)RAD_F32, nv, ne),     /* lm_head, the logical form */
        opd((uint32_t)RAD_F32, M, nv),      /* logits */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_embed_lookup_ggml(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t blk = ds41_ggml_block(p, n_p);
    const int64_t M = pi(p, n_p, "M"), ne = pi(p, n_p, "n_embd"), nv = pi(p, n_p, "n_vocab");
    const int64_t vo = pi(p, n_p, "vocab_offset", 0);
    if (blk <= 0 || M <= 0 || ne <= 0 || nv <= 0 || vo < 0 || ne % blk) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd_idx(M, 0, vo + nv),             /* tokens (ids are global under a vocab shard) */
        opd((uint32_t)RAD_F32, nv, ne),     /* wte, the logical form */
        opd(D, M, ne),                      /* x */
    };
    R4D_PICK(l);
}
