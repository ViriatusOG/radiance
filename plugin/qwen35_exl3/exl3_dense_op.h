#ifndef EXL3_DENSE_OP_H
#define EXL3_DENSE_OP_H

#include "exl3i_encoding.h"
#include "exl3_act.h"
#include "exl3i_abi.h"   /* AQ: exl3i_actq's operands, EXL3I_AQ_* */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <initializer_list>
#include <rad_abi.h>
#include <rad_builder.h>
#include <vector>

namespace qwen35_exl3 {

/* The dense op for EXL3-V3's weights (include/exl3i_encoding.h: any rate in 1/8 steps, mad), served by the
 * exl3i kernel library (include/exl3i_abi.h). */
constexpr const char* kExl3iDenseOp = "exl3i_dense";

/* Holds declared weight handles for an EXL3 linear projection */
struct Exl3DenseWeight {
    rad_weight w_codes = 0;   /* exl3i: the record plane */
    rad_weight w_suh = 0;
    rad_weight w_svh = 0;
    rad_weight w_suh1 = 0; /* optional, present when stacked */

    /* BF16 fallback if unquantized */
    rad_weight w_bf16 = 0;

    bool is_exl3 = false;   /* an exl3i trellis weight with suh / svh, served by the dense op */
    bool is_exl3i = false;  /* the op is exl3i_dense (== is_exl3: exl3i is the only trellis scheme served) */
    int  r16 = 0;           /* exl3i: the rows are stored in R16 order (the head) */
    bool replicated = false; /* exl3i, R16 at world > 1: an R16 head's rows are permuted in stored form, so a rank's
                             * vocab slice is not a slice of the record; the weight serves whole on every rank
                             * (the engine's whole-vocab logits layout, read from rank 0) */
    char rate[16] = "";     /* exl3i: the rate's canonical text */
    int bits = 4;
    int codebook = EXL3I_CB_MAD;
    const char* cb_name = "mad";
    char act[EXL3_ACT_STR] = "f16";   /* the activation mode (exl3_act.h), from the container's metadata */
};

/* The container metadata the weights' activation modes are read from (exl3_act.h keys), set for the length of one
 * declare by Exl3ActScope: the block helpers declare weights without the model metadata in hand. */
inline const RadModelMeta*& exl3_act_meta() { static thread_local const RadModelMeta* m = nullptr; return m; }
struct Exl3ActScope {
    explicit Exl3ActScope(const RadModelMeta* m) { exl3_act_meta() = m; }
    ~Exl3ActScope() { exl3_act_meta() = nullptr; }
    Exl3ActScope(const Exl3ActScope&) = delete;
    Exl3ActScope& operator=(const Exl3ActScope&) = delete;
};

/* The mode of weight `name`: its key's value, "f16" when absent. RAD_E_FORMAT for a value exl3_act.h does not parse
 * (never read as f16: the weight was chosen for some mode). */
inline int exl3_act_of(const char* name, char* out) {
    char key[320];
    Exl3Act a{};
    std::snprintf(out, EXL3_ACT_STR, "f16");
    const RadModelMeta* m = exl3_act_meta();
    if (!m || exl3_act_key(name, key, sizeof key) != 0) return RAD_OK;
    const char* v = rad_meta_gets(m, key, nullptr);
    if (!v) return RAD_OK;
    if (exl3_act_parse(v, &a) != 0) {
        std::fprintf(stderr, "radiance: qwen35_exl3: '%s' = '%s': not an activation mode (f16, a8t, a8g<N>)\n", key, v);
        return RAD_E_FORMAT;
    }
    exl3_act_str(&a, out);
    return RAD_OK;
}

/* A shared-input group (attention q / k / v, GDN qkv / z; gate_up is one weight) runs one activation mode: the
 * mix writes one mode for the group, and a container whose metadata gives its members different
 * modes is refused here, by name, instead of serving a mode mix no search priced. */
inline int exl3_act_same(const char* group, int layer, std::initializer_list<const Exl3DenseWeight*> ws) {
    const Exl3DenseWeight* first = nullptr;
    for (const Exl3DenseWeight* w : ws) {
        if (!first) { first = w; continue; }
        if (std::strcmp(first->act, w->act) != 0) {
            std::fprintf(stderr, "radiance: qwen35_exl3: blk.%d %s: one activation mode a shared-input group, got '%s' and '%s'\n",
                         layer, group, first->act, w->act);
            return RAD_E_FORMAT;
        }
    }
    return RAD_OK;
}

/* Helper to declare weight planes for exl3i (record / suh / svh) or fallback to BF16. `world` is the serve's
 * tensor parallelism: a sharded exl3i weight's planes are this rank's slice of the container's (record rows /16,
 * svh rows, suh K along the weight's shard; the core cuts a single-plane declaration in the plane's own elements
 * at load), so `rows` / `cols` are the rank's extents and the container's are `rows * world` / `cols * world`
 * down the sharded dimension. An R16 weight (the head's row order) cannot be cut that way: at world > 1 it serves
 * whole on every rank (out->replicated) with the container's extents. */
inline int decl_exl3_weight(RadBuilder* b, const char* name, int64_t rows, int64_t cols,
                            int shard, RadWeightGroup grp, int n_parts, const int64_t* parts,
                            Exl3DenseWeight* out, int world = 1) {
    if (!b || !name || !out || world < 1) return RAD_E_INVAL;
    *out = Exl3DenseWeight{};

    RadEncoding enc{};
    int64_t logical_shape[RAD_MAX_RANK] = {0};
    uint32_t rank = 0;
    int stacked = 0;

    /* The encoding is the container entry's: at convert time the recipe decides it, at load time
     * it is the entry itself. Querying the checkpoint's source name would answer bf16 (the
     * checkpoint's own) and fall every weight back to a gemm_nt. */
    int status = rad_weight_encoding(b, name, &enc, logical_shape, &rank);
    if (status == RAD_OK && rad_enc_is(&enc, EXL3I_SCHEME)) {
        int m = 0, cb = 0, r16 = 0;
        int64_t er = 0, ec = 0;
        if (exl3i_encoding_read_r16(&enc, &m, &cb, &stacked, &er, &ec, &r16) != RAD_OK) {
            std::fprintf(stderr, "radiance: qwen35_exl3: '%s' is an exl3i weight with an encoding exl3i does not serve "
                                 "(mad, any rate in 1/8 steps): refused at load\n", name);
            return RAD_E_FORMAT;
        }
        out->r16 = r16;
        if (r16 && world > 1) {   /* the R16 head: its stored rows are not the vocab's, so a rank's slice is not one */
            rows = er;
            cols = ec;
            shard = RAD_SHARD_NONE;
            out->replicated = true;
        }
        const int64_t erw = (shard == RAD_SHARD_ROW) ? rows * world : rows;
        const int64_t ecw = (shard == RAD_SHARD_COL) ? cols * world : cols;
        if (er != erw || ec != ecw) {
            std::fprintf(stderr, "radiance: qwen35_exl3: '%s' is an exl3i weight [%lld, %lld] declared as [%lld, %lld] "
                                 "at %d rank(s): the record splits by row (and the scales follow it) only along the "
                                 "shard, and an R16 weight serves whole\n", name, (long long)er, (long long)ec,
                         (long long)rows, (long long)cols, world);
            return RAD_E_SHAPE;
        }
        out->is_exl3 = true;
        out->is_exl3i = true;
        exl3i_rate_str(m, out->rate);
        out->bits = m;   /* exl3i: 16 K */
        out->codebook = cb;
        out->cb_name = exl3i_codebook_name(cb);
        if (exl3_act_of(name, out->act) != RAD_OK) return RAD_E_FORMAT;

        /* exl3i_dense takes K and N in whole 128-feature blocks (the rank's extents at tp > 1) */
        if ((rows % 128 != 0) || (cols % 128 != 0)) {
            std::fprintf(stderr, "radiance: qwen35_exl3: '%s' is an exl3i weight of [%lld, %lld] on this rank: "
                                 "exl3i_dense takes rows and columns in multiples of 128\n", name, (long long)rows,
                         (long long)cols);
            return RAD_E_SHAPE;
        }

        /* The record table: one row a 16 weight rows, (cols / 16) m / 2 words. `source` names the logical
         * weight these planes are a view OF -- the builder answers the weight's encoding from it (looking up
         * the 'name.record' view itself would find nothing and hand the hook a plain single-plane encoding it
         * cannot read); the encoding is the container entry's. */
        RadWeightDecl dc{};
        dc.source = name;
        dc.planes = "record";
        dc.dtype = RAD_U32;
        dc.rank = 2;
        dc.shape[0] = rows / 16;
        dc.shape[1] = exl3i_row_words(cols, m);
        dc.access = RAD_ACCESS_PER_TOKEN;
        dc.shard = shard;
        dc.group = grp;
        if (n_parts > 0 && parts) {
            dc.n_row_parts = n_parts;
            for (int i = 0; i < n_parts; ++i) {
                if (parts[i] % 16) return RAD_E_SHAPE;
                dc.row_parts[i] = parts[i] / 16;
            }
        }
        char wname[256];
        snprintf(wname, sizeof wname, "%s.record", name);
        out->w_codes = rad_decl_weight(b, wname, &dc);

        /* 2. suh [1, K] */
        RadWeightDecl dsuh{};
        dsuh.source = name;
        dsuh.planes = "suh";
        dsuh.dtype = RAD_F16;
        dsuh.rank = 2;
        dsuh.shape[0] = 1;
        dsuh.shape[1] = cols;
        dsuh.access = RAD_ACCESS_PER_TOKEN;
        /* suh runs along K: a column-sharded weight's rank owns its K slice of it, a row-sharded one reads it whole */
        dsuh.shard = shard == RAD_SHARD_COL ? RAD_SHARD_COL : RAD_SHARD_NONE;
        dsuh.group = grp;
        snprintf(wname, sizeof wname, "%s.suh", name);
        out->w_suh = rad_decl_weight(b, wname, &dsuh);

        /* 3. svh [N, 1] */
        RadWeightDecl dsvh{};
        dsvh.source = name;
        dsvh.planes = "svh";
        dsvh.dtype = RAD_F16;
        dsvh.rank = 2;
        dsvh.shape[0] = rows;
        dsvh.shape[1] = 1;
        dsvh.access = RAD_ACCESS_PER_TOKEN;
        /* svh runs along N: split with the rows, whole when the weight is split by columns */
        dsvh.shard = shard == RAD_SHARD_ROW ? RAD_SHARD_ROW : RAD_SHARD_NONE;
        dsvh.group = grp;
        if (n_parts > 0 && parts && dsvh.shard == RAD_SHARD_ROW) {   /* svh's rows follow the codes' parts */
            dsvh.n_row_parts = n_parts;
            for (int i = 0; i < n_parts; ++i) dsvh.row_parts[i] = parts[i];
        }
        snprintf(wname, sizeof wname, "%s.svh", name);
        out->w_svh = rad_decl_weight(b, wname, &dsvh);

        /* 4. optional suh1 [1, K] if stacked */
        if (stacked) {
            RadWeightDecl dsuh1{};
            dsuh1.source = name;
            dsuh1.planes = "suh.1";
            dsuh1.dtype = RAD_F16;
            dsuh1.rank = 2;
            dsuh1.shape[0] = 1;
            dsuh1.shape[1] = cols;
            dsuh1.access = RAD_ACCESS_PER_TOKEN;
            dsuh1.shard = shard == RAD_SHARD_COL ? RAD_SHARD_COL : RAD_SHARD_NONE;
            dsuh1.group = grp;
            snprintf(wname, sizeof wname, "%s.suh1", name);
            out->w_suh1 = rad_decl_weight(b, wname, &dsuh1);
        }
    } else {
        out->is_exl3 = false;
        if (n_parts > 0 && parts) {
            out->w_bf16 = rad::arch::decl_w_n(b, name, RAD_BF16, {rows, cols},
                                              RAD_ACCESS_PER_TOKEN, shard, grp, 0, parts, n_parts);
        } else {
            out->w_bf16 = rad::arch::decl_w(b, name, RAD_BF16, {rows, cols},
                                            RAD_ACCESS_PER_TOKEN, shard, grp);
        }
    }
    return RAD_OK;
}

/* ---- AQ (exl3i_abi.h version 3): the producer writes its consumers' int8 planes.
 * At a prefill step of EXL3I_AQ_MIN_ROWS rows or more (the A8 path's threshold, EXL3I_A8_MIN_ROWS in exl3i) a producer whose consumers are all exl3i A8 weights (a8g128, mad: exl3_aq_weight) runs as exl3i_actq with its
 * stage folded in (pre rmsnorm / silu_mul / mul: the same bf16 values, exl3i_abi.h), writing each consumer's plane to the
 * shared row buffer `aq`; the consumers read their planes there instead of quantising the bf16 output themselves. Below
 * the threshold (decode) the producer and the consumers are issued as before -- except at EXL3I_AQ_XB_ISSUE_ROWS (1) row
 * when every consumer is mad (version 4): the site's xb op writes the GEMV's rows (form xb) to its own small buffer.
 * The test pins aq_off and aq_xb_off (Exl3TestPins) leave AQ out, or the decode band's alone. */

/* Test pins: the declarations a served container never chooses by itself, reachable from tests/qwen35_exl3_test.cpp
 * (which compiles this plugin's source) to compare a fused or AQ site with the chain it replaces. Nothing in the plugin
 * writes them, so a served engine declares every fusion and AQ site a loaded library resolves. Not synchronised: a
 * test sets them between declarations. */
struct Exl3TestPins {
    bool aq_off = false;      /* no AQ site: each consumer quantises the bf16 output itself */
    bool aq_xb_off = false;   /* no decode-band AQ site (form xb): each mad GEMV quantises its rows */
    bool rerank_off = false;  /* no exl3i_rerank: the drafter keeps the coarse head's scores */
    int  fuse_off = 0;        /* EXL3_FUSE_* bits (exl3_fuse.h): those step fusions are not declared */
};
inline Exl3TestPins& exl3_test_pins() {
    static Exl3TestPins p;
    return p;
}

inline bool exl3_aq_weight(const Exl3DenseWeight& w) {
    return w.is_exl3i && w.codebook == EXL3I_CB_MAD && std::strcmp(w.act, "a8g128") == 0;
}
inline bool exl3_aq_enabled() { return !exl3_test_pins().aq_off; }
/* The decode band's AQ (form xb, mad consumers). */
inline bool exl3_aq_xb_enabled() { return !exl3_test_pins().aq_xb_off; }
inline int64_t exl3_aq_min_rows() { return EXL3I_AQ_MIN_ROWS; }   /* exl3i's A8 threshold (exl3i_abi.h) */

/* The row buffer every site shares (u8 [max_tok, width]), declared by the first site that turns on: a container AQ
 * serves nowhere (f16, V2 weights) carries none. The decode band's (exl3i_abi.h version 4, form xb: the mad GEMV's
 * rows, at most EXL3I_AQ_XB_MAX_ROWS) is its own small buffer, [rows_xb, width_xb], declared likewise. */
struct Exl3AqBuf {
    const char* name = nullptr;
    int64_t     rows = 0, width = 0;
    rad_buf     buf = 0;
    const char* name_xb = nullptr;
    int64_t     rows_xb = 0, width_xb = 0;
    rad_buf     buf_xb = 0;
    rad_buf get(RadBuilder* b) {
        if (!buf && name && rows > 0 && width > 0) buf = rad::arch::decl_b(b, name, RAD_U8, { rows, width });
        return buf;
    }
    rad_buf get_xb(RadBuilder* b) {
        if (!buf_xb && name_xb && rows_xb > 0 && width_xb > 0) buf_xb = rad::arch::decl_b(b, name_xb, RAD_U8, { rows_xb, width_xb });
        return buf_xb;
    }
};

/* One producer site: exl3i_actq over x [T, K] (x2: mul's second factor; g: rmsnorm's gain; h: where the stage's bf16
 * value also goes, for a reader that is not a consumer) for consumers `ws` (a stacked weight takes two planes). */
struct Exl3Aq {
    rad_op  op = 0;   /* 0: AQ does not serve this site */
    rad_op  op_xb = 0;   /* the decode band's (form xb), when every consumer is mad: 0 off */
    rad_op  op_res = 0, op_xb_res = 0;   /* version 5 (declare_res): op / op_xb with the residual add x += res first */
    rad_buf aq = 0, aq_xb = 0, x = 0, x2 = 0, h = 0, res = 0;
    rad_weight g = 0;
    rad_weight s[EXL3I_AQ_MAX_PLANES] = {};
    int     planes = 0;
    int64_t min_rows = 0;
    int64_t max_tok = 0, K = 0;   /* declare's, for declare_res */
    double  eps = 0, wadd = 0;

    /* The plane consumer i's weight starts at (its aq_plane). */
    static int plane_of(std::initializer_list<const Exl3DenseWeight*> ws, const Exl3DenseWeight* w) {
        int p = 0;
        for (const Exl3DenseWeight* v : ws) {
            if (v == w) return p;
            p += v->w_suh1 ? 2 : 1;
        }
        return -1;
    }

    /* Declares the site when every consumer is an AQ weight and the planes fit; else leaves op 0 (the caller declares
     * its consumers without aq). */
    int declare(RadBuilder* b, int64_t max_tok, int64_t K, const char* pre, double eps, double wadd, uint32_t act_dtype,
                std::initializer_list<const Exl3DenseWeight*> ws, Exl3AqBuf* ab, rad_buf x_buf, rad_buf x2_buf,
                rad_weight gain, rad_buf h_buf) {
        *this = Exl3Aq{};
        if (!exl3_aq_enabled() || !ab || act_dtype != RAD_BF16 || K % 128) return RAD_OK;
        if (!rad_op_declared(b, EXL3I_OP_ACTQ)) return RAD_OK;   /* an exl3i from before version 3 */
        int n = 0;
        for (const Exl3DenseWeight* w : ws) {
            if (!exl3_aq_weight(*w)) return RAD_OK;
            if (n + (w->w_suh1 ? 2 : 1) > EXL3I_AQ_MAX_PLANES) return RAD_OK;
            s[n++] = w->w_suh;
            if (w->w_suh1) s[n++] = w->w_suh1;
        }
        if (n == 0) return RAD_OK;
        for (int i = n; i < EXL3I_AQ_MAX_PLANES; ++i) s[i] = s[0];   /* weights fill their slots in order: s0 stands in */
        const bool norm = std::strcmp(pre, "rmsnorm") == 0;
        RadParam p[] = { RAD_RANGE("M", 1, max_tok), RAD_INT("K", K), RAD_INT("planes", n), RAD_STR("pre", pre),
                         RAD_F64("eps", eps), RAD_F64("wadd", wadd) };
        std::vector<rad_weight> wts(s, s + EXL3I_AQ_MAX_PLANES);
        if (norm) wts.push_back(gain);
        rad_op o = rad_decl_op(b, EXL3I_OP_ACTQ, p, norm ? 6 : 4, wts.data(), (int)wts.size());
        if (!o || !rad_op_resolved(b, o)) return RAD_OK;   /* no kernel: the site stays off */
        const rad_buf aq_buf = ab->get(b);
        if (!aq_buf) return RAD_E_INVAL;
        if (x2_buf && h_buf) rad::arch::rw(b, o, { x_buf, x2_buf }, { aq_buf, h_buf });
        else if (x2_buf)     rad::arch::rw(b, o, { x_buf, x2_buf }, { aq_buf });
        else if (h_buf)      rad::arch::rw(b, o, { x_buf }, { aq_buf, h_buf });
        else                 rad::arch::rw(b, o, { x_buf }, { aq_buf });
        op = o; aq = aq_buf; x = x_buf; x2 = x2_buf; h = h_buf; g = norm ? gain : 0; planes = n;
        min_rows = exl3_aq_min_rows();
        this->max_tok = max_tok; this->K = K; this->eps = eps; this->wadd = wadd;
        /* version 4: the decode band's rows in the mad GEMV's form, when every consumer is mad (a consumer of another
         * codebook at decode reads the producer's bf16 value, which the site would not write) */
        bool all_mad = true;
        for (const Exl3DenseWeight* w : ws) all_mad = all_mad && w->codebook == EXL3I_CB_MAD;
        if (all_mad && exl3_aq_xb_enabled()) {
            const int64_t mx = std::min<int64_t>(max_tok, EXL3I_AQ_XB_ISSUE_ROWS);
            RadParam px[] = { RAD_RANGE("M", 1, mx), RAD_INT("K", K), RAD_INT("planes", n), RAD_STR("pre", pre),
                              RAD_STR("form", "xb"), RAD_F64("eps", eps), RAD_F64("wadd", wadd) };
            rad_op ox = rad_decl_op(b, EXL3I_OP_ACTQ, px, norm ? 7 : 5, wts.data(), (int)wts.size());
            const rad_buf xb_buf = ox && rad_op_resolved(b, ox) ? ab->get_xb(b) : 0;
            if (xb_buf) {
                if (x2_buf && h_buf) rad::arch::rw(b, ox, { x_buf, x2_buf }, { xb_buf, h_buf });
                else if (x2_buf)     rad::arch::rw(b, ox, { x_buf, x2_buf }, { xb_buf });
                else if (h_buf)      rad::arch::rw(b, ox, { x_buf }, { xb_buf, h_buf });
                else                 rad::arch::rw(b, ox, { x_buf }, { xb_buf });
                op_xb = ox; aq_xb = xb_buf;
            }
        }
        return RAD_OK;
    }

    /* Version 5, an rmsnorm site that is on: its forms (int8, and xb where op_xb is) with the residual add folded in --
     * x = bf16(x + res) written back to x, the norm of that (libr4d's add_rmsnorm; exl3i_abi.h res / xo). `res_buf`
     * the block output the previous block left unadded (exl3_fuse.h addnorm); h may be it. A form exl3i does not
     * serve stays 0 (the add is then issued alone before the site). */
    void declare_res(RadBuilder* b, rad_buf res_buf) {
        if (!op || !g || !res_buf) return;
        std::vector<rad_weight> wts(s, s + EXL3I_AQ_MAX_PLANES);
        wts.push_back(g);
        for (int xb = 0; xb < 2; ++xb) {
            if (xb && !op_xb) continue;
            const int64_t mx = xb ? std::min<int64_t>(max_tok, EXL3I_AQ_XB_ISSUE_ROWS) : max_tok;
            RadParam p[] = { RAD_RANGE("M", 1, mx), RAD_INT("K", K), RAD_INT("planes", planes), RAD_STR("pre", "rmsnorm"),
                             RAD_F64("eps", eps), RAD_F64("wadd", wadd), RAD_INT("res", 1), RAD_STR("form", "xb") };
            const rad_op o = rad_decl_op(b, EXL3I_OP_ACTQ, p, xb ? 8 : 7, wts.data(), (int)wts.size());
            if (!o || !rad_op_resolved(b, o)) continue;
            const rad_buf out = xb ? aq_xb : aq;
            if (h) rad::arch::rw(b, o, { x, res_buf }, { out, h, x });
            else   rad::arch::rw(b, o, { x, res_buf }, { out, x });
            (xb ? op_xb_res : op_res) = o;
        }
        if (op_res || op_xb_res) res = res_buf;
    }

    /* This step's rows take the decode band's form (the xb site): below the A8 threshold, at most
     * EXL3I_AQ_XB_ISSUE_ROWS rows (batch-1 decode: where every mad consumer takes the GEMV). */
    bool xb_at(int64_t T) const { return op_xb && T < min_rows && T <= EXL3I_AQ_XB_ISSUE_ROWS; }
    /* This step's rows are the site's (issue() returns it): the decode band's or the A8 path's. A caller that would
     * fold the producer's stage into another op (exl3_fuse.h) asks here first. */
    bool takes(int64_t T) const { return xb_at(T) || (op && T >= min_rows); }
    /* ... and the form it takes has the residual add (declare_res) */
    bool res_at(int64_t T) const { return xb_at(T) ? op_xb_res != 0 : op_res && op && T >= min_rows; }

    /* Issues the fused producer when this step takes it (T at the threshold or above) and returns the site for the
     * consumers' issue_exl3_proj; null: the caller issues the producer and the consumers as without AQ. `add`: with
     * the residual add folded in (the caller has checked res_at). */
    const Exl3Aq* issue(RadCtx* c, int64_t T, bool add = false) const {
        using rad::arch::brows;
        const bool xb = xb_at(T);
        if (!xb && (!op || T < min_rows)) return nullptr;
        const rad_op o = xb ? (add ? op_xb_res : op_xb) : (add ? op_res : op);
        const rad_buf out = xb ? aq_xb : aq;
        if (add)
            RAD_ISSUE_N(c, o, T, brows(x, T), RAD_W(s[0]), RAD_W(s[1]), RAD_W(s[2]), RAD_W(s[3]), brows(out, T),
                        RAD_NONE, RAD_W(g), h ? brows(h, T) : RAD_NONE, brows(res, T), brows(x, T));
        else
            RAD_ISSUE_N(c, o, T, brows(x, T), RAD_W(s[0]), RAD_W(s[1]), RAD_W(s[2]), RAD_W(s[3]), brows(out, T),
                        x2 ? brows(x2, T) : RAD_NONE, g ? RAD_W(g) : RAD_NONE, h ? brows(h, T) : RAD_NONE);
        return this;
    }
};

/* The shared row buffer's width: the widest site's row (exl3i_aq_row_bytes), in 16-byte steps. */
inline int64_t exl3_aq_row_width(std::initializer_list<std::pair<int64_t, int>> sites) {
    int64_t w = 0;
    for (const auto& s : sites) w = std::max<int64_t>(w, exl3i_aq_row_bytes(s.first, s.second));
    return (w + 15) / 16 * 16;
}
/* The decode band's (form xb): the widest site's exl3i_aq_xb_row_bytes (256-byte steps already). */
inline int64_t exl3_aq_xb_row_width(std::initializer_list<std::pair<int64_t, int>> sites) {
    int64_t w = 0;
    for (const auto& s : sites) w = std::max<int64_t>(w, exl3i_aq_xb_row_bytes(s.first, s.second));
    return w;
}

/* Helper to declare projection op: exl3i_dense (if exl3i) or gemm_nt (if BF16). aq (a site that is on, so every consumer
 * an AQ weight): the op also reads the producer-quantised rows, its planes from aq_plane (exl3i_abi.h). */
inline rad_op decl_exl3_proj(RadBuilder* b, int64_t max_tok, int64_t N, int64_t K,
                             const char* default_dtype, const Exl3DenseWeight& wt,
                             rad_buf in_act, rad_buf out_act,
                             const Exl3Aq* site = nullptr, int aq_plane = 0) {
    if (wt.is_exl3) {
        std::vector<rad_weight> wts = { wt.w_codes, wt.w_suh, wt.w_svh };
        if (wt.w_suh1) wts.push_back(wt.w_suh1);

        const bool aq = wt.is_exl3i && site && site->op;
        RadParam params[] = {
            RAD_RANGE("M", 1, max_tok),
            RAD_INT("N", N),
            RAD_INT("K", K),
            RAD_STR("rate", wt.rate),
            RAD_STR("codebook", wt.cb_name),
            RAD_STR("act", wt.act),         /* exl3i_abi.h: every declaration, f16 included */
            RAD_INT("aq_plane", aq_plane),
            RAD_INT("aq_planes", aq ? site->planes : 0)
        };
        rad_op op = rad_decl_op(b, kExl3iDenseOp, params, aq ? 8 : 6, wts.data(), (int)wts.size());
        /* with aq the rows `a` names are the site's input (Exl3Aq::rows): read for its row count only */
        if (aq && site->aq_xb) return rad::arch::rw(b, op, { in_act, site->aq, site->aq_xb, site->x }, { out_act });
        if (aq) return rad::arch::rw(b, op, { in_act, site->aq, site->x }, { out_act });
        return rad::arch::rw(b, op, { in_act }, { out_act });
    } else {
        RadParam params[] = {
            RAD_RANGE("M", 1, max_tok),
            RAD_INT("N", N),
            RAD_INT("K", K),
            RAD_STR("dtype", default_dtype)
        };
        rad_op op = rad_decl_op(b, "gemm_nt", params, 4, &wt.w_bf16, 1);
        return rad::arch::rw(b, op, { in_act }, { out_act });
    }
}

/* Helper to issue projection at step time. aq: the site quantised this step's rows (Exl3Aq::issue returned true), the op
 * reads its planes there, and `a` names the site's input (its rows; in_act the producer did not write); null: in_act. */
inline void issue_exl3_proj(RadCtx* c, rad_op op, int64_t T,
                            const Exl3DenseWeight& wt, rad_buf in_act, rad_buf out_act, const Exl3Aq* aq = nullptr) {
    /* RAD_ISSUE_N: the band is T's (the logits run at n_out rows, not the step's n_tok) */
    if (wt.is_exl3 && aq) {
        RAD_ISSUE_N(c, op, T, rad::arch::brows(aq->x, T),
                  RAD_W(wt.w_codes), RAD_W(wt.w_suh), RAD_W(wt.w_svh), wt.w_suh1 ? RAD_W(wt.w_suh1) : RAD_NONE,
                  rad::arch::brows(out_act, T), rad::arch::brows(aq->xb_at(T) ? aq->aq_xb : aq->aq, T));
    } else if (wt.is_exl3) {
        if (wt.w_suh1) {
            RAD_ISSUE_N(c, op, T, rad::arch::brows(in_act, T),
                      RAD_W(wt.w_codes), RAD_W(wt.w_suh), RAD_W(wt.w_svh), RAD_W(wt.w_suh1),
                      rad::arch::brows(out_act, T));
        } else {
            RAD_ISSUE_N(c, op, T, rad::arch::brows(in_act, T),
                      RAD_W(wt.w_codes), RAD_W(wt.w_suh), RAD_W(wt.w_svh), RAD_NONE,
                      rad::arch::brows(out_act, T));
        }
    } else {
        RAD_ISSUE_N(c, op, T, rad::arch::brows(in_act, T), RAD_W(wt.w_bf16), rad::arch::brows(out_act, T));
    }
}

} /* namespace qwen35_exl3 */

#endif /* EXL3_DENSE_OP_H */
