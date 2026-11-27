/* exl3r4d_gemm_fields.h -- the fields of the dense GEMM's kernel argument block after its class records, included by
 * exl3r4d_dev.h into Exl3r4dGemm. No include guard: it is a struct body, included once a block. */
    const void*    x;        /* gate_up: a [n_tok][x_pitch] f16|bf16 (token order);
                              * down: h [>= T][x_pitch] f16 (sorted order) */
    void*          y;        /* gate_up: h f16 [T][y_pitch]; down: y bf16 [T][y_pitch] */
    const int32_t* sorted;   /* [>= T] flat (token, slot) index of sorted row i, -1 dropped */
    const int32_t* off;      /* [n_expert + 1] expert e's rows are [off[e], off[e+1]) */
    void*          xr;       /* the prep kernel's rotated rows, f16 [T][xr_pitch] (scratch); null
                              * when the GEMM rotates its own rows (fused prep) */
    int64_t        x_pitch, y_pitch, xr_pitch;   /* elements */
    int32_t        T;        /* sorted rows = the output's rows */
    int32_t        n_tok;    /* rows of x (gate_up: tokens; down: >= T) */
    int32_t        n_expert, top_k, n_class;
    int32_t        y_cols;   /* the output's width: the wider class's */
    int32_t        x_bf16;   /* gate_up's a is bf16 (else f16) */
    int32_t        sk;       /* LDS split-K: k-slices a column group of waves */
    int32_t        chunk;    /* fused prep: rows a pass (what fits in LDS, <= 16); else 16 */
    int32_t        nslot;    /* grid.y: workgroups a column block (live experts + spares) */
    int32_t        h_rot;    /* gate_up writes h rotated for down (dn_suh); down reads it so */
    int32_t        a_rot;    /* gate_up's `a` is the prep's xr (sorted rows, already rotated) */
    unsigned*      ctr;      /* single launch (gate_up): this launch's counter block -- [0] tickets, [1] done,
                              * [2 + slot] slot ready -- zero at entry, reset to zero by the last workgroup; null:
                              * the prep kernel and the GEMM as two dispatches */
    int32_t        tpw;      /* split epilogue: 16-column tiles a GEMM workgroup owns (of one matrix) */
    int32_t        y_dt;     /* the output's dtype: 0 bf16, 1 f32, 2 f16 (in tpw's padding, so the block's size is unchanged) */
    float*         raw;      /* split epilogue: the GEMM's raw f32 sums [T][raw_pitch] (scratch), which the finisher
                              * rotates, scales and writes out; null: the GEMM's own epilogue */
    int64_t        raw_pitch;
