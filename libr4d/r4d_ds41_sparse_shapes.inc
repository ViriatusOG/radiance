/* r4d_ds41_sparse_shapes.inc -- operand descriptions of DeepSeek-V4.1's sparse ops; included at the end
 * of r4d_shapes.cpp for its helpers (opd, opd_idx, opd_none, pick, R4D_PICK, pi, dt_act).
 *
 * Each is libref's description (libref/ref_shapes.cpp) operand for operand: one sequence, window
 * blocks of 4 slots covering every position a row may hold, one compressed or index entry a block. */

/* index_candidates' keep words a row: a bit a block of `block` positions over `width`. */
static int64_t keep_words(int64_t width, int64_t block) {
    return block < 1 || width < 1 ? 0 : ((width + block - 1) / block + 31) / 32;
}

extern "C" int r4d_shape_attn_sparse_sink(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), H = pi(p, n_p, "heads"), Dm = pi(p, n_p, "dim");
    const int64_t win = pi(p, n_p, "window"), ratio = pi(p, n_p, "ratio"), topk = pi(p, n_p, "topk");
    if (M <= 0 || H < 1 || Dm < 2 || win < 1 || ratio < 0 || (ratio > 0 && (topk < 1 || Dm % ratio)))
        return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const int64_t bs = 4, nb = (M + bs - 1) / bs + 1, ncb = M + 1;
    const bool comp = ratio > 0;
    /* latent_format: an E4M3 entry is its rad_kv_fp8.h bytes */
    const RadParam* lfp = rad_param_find(p, n_p, "latent_format");
    const int lf = rad_kvf_parse(lfp && lfp->kind == RAD_P_STR ? lfp->sval : nullptr);
    const int64_t cslots = lf > 0 ? rad_kvf_bytes(lf, Dm) : Dm;
    if (lf < 0 || cslots < 1 || (comp && cslots % ratio)) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd(D, M, H * Dm),                                  /* q */
        opd(D, nb, 1, bs, Dm),                              /* window cache */
        opd_idx(1, nb, nb),                                 /* window block table */
        opd_idx(1, 0, M + 1),                               /* window tokens held from the table's start */
        comp ? opd(lf > 0 ? (uint32_t)RAD_F8E4M3 : D, ncb, 1, ratio, cslots / ratio)
             : opd_none(),                          /* compressed cache: an entry a block */
        comp ? opd_idx(1, ncb, ncb) : opd_none(),           /* compressed block table */
        comp ? opd_idx(M, topk, ncb) : opd_none(),          /* selected compressed positions */
        opd_idx(M, 0, M),                                   /* positions */
        opd_idx(2, 0, M, -1, (uint32_t)RAD_OPD_F_IDX_CU),   /* cu_seqlens */
        opd((uint32_t)RAD_F32, H),                          /* sinks */
        opd(D, M, H * Dm),                                  /* out */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_index_score(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), H = pi(p, n_p, "heads"), Dm = pi(p, n_p, "dim");
    const int64_t ratio = pi(p, n_p, "ratio"), C = pi(p, n_p, "width");
    if (M <= 0 || H < 1 || Dm < 1 || ratio < 1 || Dm % ratio || C < 1) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const int64_t ncb = C + 1;
    /* key_format: an E4M3 entry is its rad_kv_fp8.h bytes */
    const RadParam* kfp = rad_param_find(p, n_p, "key_format");
    const int kf = rad_kvf_parse(kfp && kfp->kind == RAD_P_STR ? kfp->sval : nullptr);
    const int64_t kslots = kf > 0 ? rad_kvf_bytes(kf, Dm) : Dm;
    if (kf < 0 || kslots < 1 || kslots % ratio) return RAD_E_UNSUPPORTED;
    /* A run (row_part p) scores rows [p M, p M + M) of a step that holds them. */
    const int64_t part = pi(p, n_p, "row_part"), Mq = part > 0 ? (part + 1) * M : M;
    const RadOpdDesc l[] = {
        opd(D, Mq, H * Dm),                                 /* q */
        opd(D, Mq, H),                                      /* head weights */
        opd(kf > 0 ? (uint32_t)RAD_F8E4M3 : D, ncb, 1, ratio, kslots / ratio),  /* index cache: an entry a block */
        opd_idx(1, ncb, ncb),                               /* block table */
        opd_idx(Mq, 0, Mq),                                 /* positions */
        opd_idx(2, 0, Mq, -1, (uint32_t)RAD_OPD_F_IDX_CU),  /* cu_seqlens */
        opd((uint32_t)RAD_F32, M, C),                       /* score */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_index_candidates(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), C = pi(p, n_p, "width"), block = pi(p, n_p, "block");
    if (M <= 0 || C < 1 || block < 1 || pi(p, n_p, "topk_blocks") < 1 || pi(p, n_p, "ratio") < 1)
        return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M, C),                       /* score */
        opd_idx(M, 0, M),                                   /* positions */
        opd((uint32_t)RAD_I32, M, keep_words(C, block)),    /* keep: a bit a block */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_index_topk(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), C = pi(p, n_p, "width"), k = pi(p, n_p, "topk");
    const int64_t block = pi(p, n_p, "block");
    if (M <= 0 || C < 1 || k < 1 || pi(p, n_p, "ratio") < 1) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M, C),                       /* score */
        /* keep, optional: with `block`, index_candidates' bits (any word is a valid mask) */
        block > 0 ? opd_idx(M, keep_words(C, block), 0x7fffffff) : opd_none(),
        opd_idx(M, 0, M),                                   /* positions */
        opd((uint32_t)RAD_I32, M, k),                       /* sel */
    };
    R4D_PICK(l);
}
