/* exl3r4d_prefill_fields.h -- the prefill form's fields after its decode-band block, included by exl3r4d_prefill.h
 * into Exl3r4dPrefill and Exl3r4dPrefill4 (exl3r4d_gemm_fields.h's pattern). No include guard: a struct body. */
    uint32_t*   map;    /* scratch: words 0..3 a header (word 0 = entries written), then `cap` entries of four
                         * words {expert or EXL3R4D_PF_ZERO, first row, end row, 0} */
    int32_t     cap;    /* entries the map can hold = grid.y of the GEMM */
    int32_t     fm;     /* rows an entry at most: 16 * MTX */
    int32_t     chunk_c, chunk_n;   /* the map form in chunks (chunk c of n; 0 of 1: whole) -- EXL3R4D_PF_SCRATCH_MB */
    int32_t     chunk_rows;         /* R: chunk c holds the map entries whose first row is in [c R, (c + 1) R), so its
                                     * rows fit R + fm rows of scratch from its first entry's row (rbase) */
    int32_t     chunk_gy;           /* grid.y of a chunk's GEMM: an upper bound on a chunk's entries */
    uint32_t*   ctab;               /* scratch, after the map: chunk_n records {lo, hi, rbase, 0}, written by the map
                                     * kernel (the entries [lo, hi) of chunk c and the row its scratch rows start at) */
    uint32_t*   order;              /* scratch, after the chunk table: gate_up's entries longest first a chunk, or null
                                     * (EXL3R4D_PF_LPT; written by the map kernel) */
    int32_t     ystream;            /* down's map form: 1 = the streamed y (EXL3R4D_PF_YNT_MB, prefill.c) */
    /* The copy ring (fn4 exl3_xr; gate_up's map form, two-class build), after the earlier fields so none of them
     * moves. xr_mode bits 0-1: the order of a chunk's groups (0: longest first; 1: the groups whose codes the card
     * reads from VRAM longest first, then those it reads from a host pool (rad_host_pool_ranges) longest first;
     * 2: host-read first; 3: the two interleaved); bit 2: the ring copy (order 1, needs xr_ring_mb). 0 = the regular
     * launch. */
    /* One word, in the slot after ystream (so the four-class block stays within the 512 explicit kernarg bytes): bits
     * 0-7 xr_mode, bits 8-15 the codes' 2 K (an expert's codes record = rows x cols x 2K / 16 bytes), bits 16-31 the
     * MiB of ring after the copy ring's tables in scratch (the op's xring; 0: none). Read through xr_mode_of /
     * xr_k2_of / xr_ring_of. */
    int32_t     xr;
