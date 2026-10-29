/* r3d_shapes.cpp -- the extent and content domain of each operand of an op at a geometry.
 *
 * libr3d's side of the `RadShapeFn` hook. The component that reads an operand knows its extent,
 * and a row that exists only here is invisible to the hand-written table in `tools/`. Correctness
 * checks, benchmarks and the coverage census size these rows through this hook.
 *
 * The shim is the authority for every extent: each `r3d_rad_*` entry point unpacks `RadArgs.t[i]`
 * into extents and pitches, and a change to a shim's contract must be mirrored here.
 *
 * Content matters as much as size. A token id gathers a codebook row, a KV slot indexes
 * n_blocks * block_size positions, a block-table entry indexes n_blocks. Drawn from the wrong
 * range, such an operand tests out-of-range handling instead of arithmetic, and colliding scatter
 * indices are a race. `RAD_FILL_INDEX` with an explicit `idx_max` states each range, and
 * `RAD_OPD_F_IDX_UNIQUE` marks scatters that need distinct destinations.
 *
 * ============================================================ rows with no description
 *
 * A decline is a real answer: the caller skips and counts the case, where an invented extent
 * yields a meaningless pass.
 *
 *   gemm_nt_q at a quantised `a` whose scale plane the row does not name
 *       Does not arise here (libr4d declines it): both quantised rows state their activation
 *       dtype in `dtype`, so `a` is bf16 against fp8 (fp8a16) or i8 against i8 (i8a8).
 *
 *   rope at a multi-component mode
 *       mrope, imrope and axial take one position per component, component-major, with the count
 *       from `sections`. libr3d's kernel implements neox and gptj only and its row's constraints
 *       say so, so a geometry with `sections` set is never routed here.
 *
 * No tuned axis is read: this plugin's axes (split counts, wave counts, the streaming mode) change
 * how a kernel walks its operands, not their extents. An axis that did would be read from `p`,
 * where the core appends the chosen point.
 *
 * Pure host code (no r3d.h, no HIP), like r3d_rows.cpp, so it compiles without ROCm.
 */

#include <cstdint>
#include <cstring>

#include "r3d_plugin.h"

namespace {

/* ==================================================================== building a description
 *
 * Each op builds its whole operand list in schema order and hands out the one asked for, so the
 * count can be checked against the schema in r3d_rows.cpp (the tool checks
 * `RadOpSchema::n_operands`).
 */

/* A plain operand of up to four extents. A trailing 0 ends the shape; zero is never a legal
 * extent (`pick` refuses one). */
RadOpdDesc opd(uint32_t dt, int64_t d0, int64_t d1 = 0, int64_t d2 = 0, int64_t d3 = 0) {
    RadOpdDesc d{};
    d.dtype     = dt;
    d.fill      = RAD_FILL_NORMAL;
    /* -1 is "draw"; set explicitly though the caller pre-initialises it. */
    d.idx_const = -1;
    const int64_t e[4] = { d0, d1, d2, d3 };
    for (int i = 0; i < 4 && e[i] != 0; ++i) d.shape[d.rank++] = e[i];
    return d;
}

/* An index operand. `idx_max` is the exclusive upper bound of the draw; 0 means the previous
 * operand's leading extent (the ABI default, right for a gather from the operand before it).
 * `konst >= 0` fills every element with that value. Always i32: every index libr3d reads is signed
 * 32-bit. */
RadOpdDesc opd_idx(int64_t d0, int64_t idx_max, int64_t konst = -1, uint32_t flags = 0,
                   int64_t d1 = 0) {
    RadOpdDesc d = opd((uint32_t)RAD_I32, d0, d1);
    d.fill      = RAD_FILL_INDEX;
    d.idx_max   = idx_max;
    d.idx_const = konst;
    d.flags     = flags;
    return d;
}

/* An optional operand this description does not pass; to the ABI it means a null RadTensor. */
RadOpdDesc opd_none() {
    RadOpdDesc d{};
    d.idx_const = -1;
    d.flags     = RAD_OPD_F_ABSENT;
    return d;
}

/* Restricts the fill to (0, 1); `RAD_FILL_SIGMOID` is the ABI's only such draw. A scale drawn from
 * a normal is negative half the time, and a negative descale inverts a whole quantised block. */
RadOpdDesc in_unit(RadOpdDesc d) { d.fill = RAD_FILL_SIGMOID; return d; }

/* Hands out operand `operand` after checking the whole list, so a geometry missing a parameter
 * declines every operand. A non-positive extent means a needed parameter was absent or zero
 * (RAD_E_UNSUPPORTED); an index outside the list is a bug here or in the schema (RAD_E_SHAPE). */
int pick(RadOpdDesc* out, int operand, const RadOpdDesc* list, int n) {
    if (!out || !list || n <= 0) return RAD_E_INVAL;
    for (int i = 0; i < n; ++i) {
        if (list[i].flags & RAD_OPD_F_ABSENT) continue;
        if (list[i].rank == 0 || list[i].rank > RAD_MAX_RANK) return RAD_E_UNSUPPORTED;
        for (uint32_t d = 0; d < list[i].rank; ++d)
            if (list[i].shape[d] <= 0) return RAD_E_UNSUPPORTED;
    }
    if (operand < 0 || operand >= n) return RAD_E_SHAPE;
    *out = list[operand];
    return RAD_OK;
}

#define R3D_PICK(list) return pick(out, operand, (list), (int)(sizeof(list) / sizeof((list)[0])))

/* ==================================================================== parameters
 *
 * `M` and the other ranged parameters read as the band's bound, not one step's row count: buffers
 * must hold the whole band the kernel was selected for.
 *
 * Not r3d_pgeti_or(), which reads RAD_P_INT only; this hook can see ranges still open, and
 * rad_param_getdim reads a RAD_P_RANGE as its high bound.
 */
int64_t gi(const RadParam* p, int n_p, const char* k, int64_t dflt) {
    return rad_param_getdim(p, n_p, k, dflt);
}

const char* gs(const RadParam* p, int n_p, const char* k) {
    return rad_param_gets(p, n_p, k, nullptr);
}

/* The activation dtype, from `dtype` or `q_dtype`; bf16 when the geometry names none. A
 * quantisation-scheme string ("fp8a16", "w8a16") is not a dtype and lands on bf16, which is what
 * the quantised rows mean by it: the scheme picks the weight's dtype. */
uint32_t act_dt(const RadParam* p, int n_p) {
    const char* s = gs(p, n_p, "dtype");
    if (!s) s = gs(p, n_p, "q_dtype");
    const uint32_t dt = rad_dtype_parse(s);
    return dt == (uint32_t)RAD_DT_INVALID ? (uint32_t)RAD_BF16 : dt;
}

/* The scale-stream group count, matching the quantisers' launches: `group` <= 0 or wider than the
 * row is one scale per row, otherwise ceil(n / group). */
int64_t scale_groups(int64_t n, int64_t group) {
    int64_t g = group;
    if (g <= 0 || g > n) g = n;
    return g > 0 ? (n + g - 1) / g : 0;
}

/* ==================================================================== elementwise and norms
 *  r3d_model_bf16.hip
 */

}  /* namespace */

extern "C" {

/* y = x * rsqrt(mean(x^2) + eps) * (w + wadd) * mult. The gain is described at full width; the
 * form with no gain is a hand case, since a null optional operand is not a fill domain. */
int r3d_shape_rmsnorm(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const RadOpdDesc list[] = { opd(D, M, n), opd(D, n), opd(D, M, n) };   /* x, w, y */
    R3D_PICK(list);
}

/* The two-pass form. `b` is optional and passed: the biased form covers strictly more code. */
int r3d_shape_layernorm(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const RadOpdDesc list[] = { opd(D, M, n), opd(D, n), opd(D, n), opd(D, M, n) };  /* x,w,b,y */
    R3D_PICK(list);
}

/* add / mul share a schema and a description. `b` is described at full extent; the row-broadcast
 * form is not, since a tool that sizes by shape alone cannot tell it from a shape bug. */
static int binary_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const RadOpdDesc list[] = { opd(D, M, n), opd(D, M, n), opd(D, M, n) };   /* a, b, y */
    return pick(out, operand, list, 3);
}

int r3d_shape_add(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return binary_shape(p, n_p, operand, out);
}
int r3d_shape_mul(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return binary_shape(p, n_p, operand, out);
}

/* y = act(gate) * up over two [M, n] operands, the halves as the kernel sees them (the model
 * issues the fused [M, 2n] form). */
int r3d_shape_act_mul(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return binary_shape(p, n_p, operand, out);
}

/* x -> y, one dtype to another. Operand dtypes come from `from` and `to`; an unparseable spelling
 * falls back to f32, the widest it could mean. */
int r3d_shape_cast(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    uint32_t from = rad_dtype_parse(gs(p, n_p, "from"));
    uint32_t to   = rad_dtype_parse(gs(p, n_p, "to"));
    if (from == (uint32_t)RAD_DT_INVALID) from = (uint32_t)RAD_F32;
    if (to   == (uint32_t)RAD_DT_INVALID) to   = (uint32_t)RAD_F32;
    const RadOpdDesc list[] = { opd(from, M, n), opd(to, M, n) };       /* x, y */
    R3D_PICK(list);
}

/* The unary activations and softmax: x and y at the same extent. */
static int unary_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const RadOpdDesc list[] = { opd(D, M, n), opd(D, M, n) };           /* x, y */
    return pick(out, operand, list, 2);
}

int r3d_shape_silu(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return unary_shape(p, n_p, operand, out);
}
int r3d_shape_gelu(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return unary_shape(p, n_p, operand, out);
}
int r3d_shape_sigmoid(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return unary_shape(p, n_p, operand, out);
}
int r3d_shape_softmax(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return unary_shape(p, n_p, operand, out);
}

/* y = x * s with `s` a single broadcast element. */
int r3d_shape_scale(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const RadOpdDesc list[] = { opd(D, M, n), opd(D, 1), opd(D, M, n) };   /* x, s, y */
    R3D_PICK(list);
}

/* y = value. The only operand is the output. */
int r3d_shape_fill(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const RadOpdDesc list[] = { opd(D, M, n) };
    R3D_PICK(list);
}

/* ==================================================================== the model's odds and ends
 *  r3d_model_bf16.hip
 */

/* In-place rotary over a fused qkv row: n_head q heads, then n_head_kv each of k and v. Positions
 * are drawn over the row count; positions contiguous from an arbitrary start are a hand case,
 * since no fill domain spells them. */
int r3d_shape_rope(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M   = gi(p, n_p, "M", 0);
    const int64_t hd  = gi(p, n_p, "head_dim", 0);
    const int64_t nh  = gi(p, n_p, "n_head", 0);
    const int64_t nkv = gi(p, n_p, "n_head_kv", nh);
    if (M <= 0 || hd <= 0 || nh <= 0) return RAD_E_UNSUPPORTED;
    const RadOpdDesc list[] = {
        opd((uint32_t)RAD_BF16, M, (nh + 2 * nkv) * hd),   /* qkv, INOUT */
        opd_idx(M, 0)                                     /* positions */
    };
    R3D_PICK(list);
}

/* K and V into a paged cache. Two blocks of slack over the tokens, so the slot mapping is not a
 * dense prefix. Only the cache takes `kv_dtype`; k and v are always bf16. */
int r3d_shape_kv_store(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M   = gi(p, n_p, "M", 0);
    const int64_t hd  = gi(p, n_p, "head_dim", 0);
    const int64_t kvh = gi(p, n_p, "n_head_kv", 0);
    const int64_t bs  = gi(p, n_p, "block_size", 0);
    if (M <= 0 || hd <= 0 || kvh <= 0 || bs <= 0) return RAD_E_UNSUPPORTED;
    const int64_t nb = (M + bs - 1) / bs + 2;
    const uint32_t kv = rad_dtype_parse(gs(p, n_p, "kv_dtype"));
    const uint32_t KV = kv == (uint32_t)RAD_DT_INVALID ? (uint32_t)RAD_BF16 : kv;
    const RadOpdDesc list[] = {
        opd((uint32_t)RAD_BF16, M, kvh, hd),                        /* k */
        opd((uint32_t)RAD_BF16, M, kvh, hd),                        /* v */
        opd_idx(M, nb * bs, -1, RAD_OPD_F_IDX_UNIQUE),              /* slot_mapping */
        opd(KV, nb, kvh, bs, 2 * hd)                                /* kv_cache, INOUT */
    };
    R3D_PICK(list);
}

/* x[r] = wte[ids[r] - vocab_offset]. Ids are global, so the draw's bound is `vocab_offset + n_vocab`:
 * under a shard the draw crosses the slice boundary and exercises the zero-row path. */
int r3d_shape_embed_lookup(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t ne = gi(p, n_p, "n_embd", 0);
    const int64_t nv = gi(p, n_p, "n_vocab", 0);
    const int64_t vo = gi(p, n_p, "vocab_offset", 0);
    if (M <= 0 || ne <= 0 || nv <= 0 || vo < 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    const RadOpdDesc list[] = {
        opd_idx(M, vo + nv),        /* tokens */
        opd(D, nv, ne),             /* wte */
        opd(D, M, ne)               /* x */
    };
    R3D_PICK(list);
}

/* The source is 4x the gathered height so the gather is not the identity; `idx` draws over those
 * 4M rows. */
int r3d_shape_gather_rows(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const RadOpdDesc list[] = { opd(D, 4 * M, n), opd_idx(M, 0), opd(D, M, n) };  /* x, idx, y */
    R3D_PICK(list);
}

/* The destination is 4x the scattered height, since onto exactly M rows a kernel ignoring `idx`
 * could pass; unnamed rows must come back unchanged. Indices are distinct, as a shared destination
 * is a race. */
int r3d_shape_scatter_rows(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const RadOpdDesc list[] = {
        opd(D, M, n), opd_idx(M, 4 * M, -1, RAD_OPD_F_IDX_UNIQUE), opd(D, 4 * M, n)
    };                                                                /* v, idx, x (INOUT) */
    R3D_PICK(list);
}

/* ================================================================== quantisation
 *  r3d_quant_fp8.hip
 */

/* Codes row-major at [M, n], one f32 scale per (row, `group` of n). `dtype` names the input; the
 * output's dtype is in the op name. */
int r3d_shape_quant_act_fp8(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
    const RadOpdDesc list[] = {
        opd(D, M, n), opd((uint32_t)RAD_F8E4M3, M, n), opd((uint32_t)RAD_F32, M, ng)
    };                                                                /* x, q, scale */
    R3D_PICK(list);
}

/* The per-row int8: one scale per row in the tightest [M, 1] plane (the reference writes it at
 * pitch 1). The optional `asum`, group-major as libref's quant_act_i8 writes it, is described
 * whenever `group` divides n so a case checks the row sums too. `q` is in the step order; a
 * description states size and content domain, not a permutation, so it is described by extent. */
int r3d_shape_quant_act_i8(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), g = gi(p, n_p, "group", 0);
    const RadOpdDesc list[] = {
        opd(D, M, n),
        opd((uint32_t)RAD_I8, M, n),
        opd((uint32_t)RAD_F32, M, 1),
        (g > 0 && n % g == 0) ? opd((uint32_t)RAD_F32, n / g, (M + 15) / 16 * 16) : opd_none()
    };                                                                /* x, q, scale, asum */
    R3D_PICK(list);
}

/* The per-group int8 the i8a8 GEMM reads: the fp8 quantiser's extents with int8 codes. */
int r3d_shape_quant_act_i8g(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
    const RadOpdDesc list[] = {
        opd(D, M, n), opd((uint32_t)RAD_I8, M, n), opd((uint32_t)RAD_F32, M, ng)
    };                                                                /* x, q, scale */
    R3D_PICK(list);
}

/* The gated form: gate_up is [M, 2n] with the gate first, and what is quantised is
 * act(gate) * up over n columns. */
int r3d_shape_gated_quant_fp8(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
    const RadOpdDesc list[] = {
        opd(D, M, 2 * n),
        opd((uint32_t)RAD_F8E4M3, M, n),
        opd((uint32_t)RAD_F32, M, ng),
        opd((uint32_t)RAD_BF16, M, n)
    };                                                       /* gate_up, q, scale, out_bf16 */
    R3D_PICK(list);
}

/* y = code * scale. `dtype` names the code: E4M3 or E5M2 (checkpoint fp8) or i8. The scale is
 * drawn in (0, 1), see in_unit(). */
int r3d_shape_dequant(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t g = gi(p, n_p, "group", 0);
    const uint32_t qd = rad_dtype_parse(gs(p, n_p, "dtype"));
    const uint32_t Q  = qd == (uint32_t)RAD_DT_INVALID ? (uint32_t)RAD_I8 : qd;
    const RadOpdDesc list[] = {
        opd(Q, M, n), in_unit(opd((uint32_t)RAD_F32, M, scale_groups(n, g))),
        opd((uint32_t)RAD_BF16, M, n)
    };                                                                /* q, scale, y */
    R3D_PICK(list);
}

/* ================================================================== gemm
 *  r3d_gemm_bf16_nt_m16.hip, r3d_gemm_bf16_nt_tiled.hip, r3d_gemm_fp8a16_nt_m1.hip
 */

/* y[M,N] = a[M,K] @ b[N,K]^T. Both operands are k-contiguous (the "nt" in the name). */
int r3d_shape_gemm_nt(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), N = gi(p, n_p, "N", 0), K = gi(p, n_p, "K", 0);
    const RadOpdDesc list[] = { opd(D, M, K), opd(D, N, K), opd(D, M, N) };   /* a, b, y */
    R3D_PICK(list);
}

/* The vocabulary projection, transcribed from ref_logits_gemm_shape: x [M, n_embd] and the head
 * [n_vocab, n_embd] in the activation's dtype, logits in f32 because every sampler stage reads
 * f32 (a prefill band's plane is 1.07 GB at M=2048). Unlike the reference row, which also places
 * the head (`ref_layout_head`), this row reads the head where it is stored, so it declares only a
 * shape hook. */
int r3d_shape_logits_gemm_bf16(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), nv = gi(p, n_p, "n_vocab", 0), ne = gi(p, n_p, "n_embd", 0);
    if (M <= 0 || nv <= 0 || ne <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    const RadOpdDesc list[] = { opd(D, M, ne), opd(D, nv, ne), opd(RAD_F32, M, nv) };
    R3D_PICK(list);                              /* x, lm_head, logits */
}

/* The quantised GEMM. `dtype` names the scheme, which fixes which operands are coded:
 *
 *   fp8a16   a bf16, b the checkpoint's E4M3 [N][K], b_scale one f32 per 128x128 block (the
 *            container's `weight_scale_inv`, whose dtype the container states: f32 at fp8a8f32,
 *            bf16 as DeepSeek's own scales are).
 *   fp8a8    a E4M3 against a_scale, one f32 per (row, `group` of K): quant_act_fp8's output, so
 *            the two rows state the same group.
 *   i8a8     i8 codes on both axes with an f32 scale per (row, `group`).
 *
 * `b_scale` is block-major [ceil(N/pb), ceil(K/pb)] for fp8 and row-major [N, ceil(K/group)] for
 * i8, as each checkpoint stores it. */
int r3d_shape_gemm_nt_q(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), N = gi(p, n_p, "N", 0), K = gi(p, n_p, "K", 0);
    if (M <= 0 || N <= 0 || K <= 0) return RAD_E_UNSUPPORTED;
    int64_t group = gi(p, n_p, "group", 0);
    if (group <= 0) group = R3D_FP8_BLOCK;      /* the block every block-scaled checkpoint uses */
    const int64_t kb = (K + group - 1) / group, nb = (N + group - 1) / group;
    const char* q = gs(p, n_p, "dtype");
    const bool i8a8 = q && !std::strcmp(q, "i8a8");
    const bool a8   = q && (!std::strcmp(q, "fp8a8") || i8a8);

    RadOpdDesc list[R3D_GEMM_Q_NOPERAND];
    list[0] = opd(i8a8 ? (uint32_t)RAD_I8 : (a8 ? (uint32_t)RAD_F8E4M3 : (uint32_t)RAD_BF16),
                  M, K);                                              /* a */
    list[1] = a8 ? in_unit(opd((uint32_t)RAD_F32, M, kb)) : opd_none();   /* a_scale */
    list[2] = opd(i8a8 ? (uint32_t)RAD_I8 : (uint32_t)RAD_F8E4M3, N, K);  /* b */
    list[3] = in_unit(opd((uint32_t)RAD_F32, i8a8 ? N : nb, kb));         /* b_scale */
    list[4] = opd((uint32_t)RAD_BF16, M, N);                             /* y */
    list[5] = opd_none();                                                /* a_sum */
    list[6] = opd_none();                                                /* b_ref */
    return pick(out, operand, list, R3D_GEMM_Q_NOPERAND);
}

/* The OCP-MXFP4 grid, as libref describes it: an e2m1 weight [N, K], two a byte, with one e8m0
 * exponent per 32 of K ([N, K/32]); `b_ref` is the same exponent plane, accepted and ignored. The
 * activation is the fp8a8 row's: E4M3 against one f32 per (row, 128 of K), drawn in (0, 1). Both
 * zero-point operands are absent since the scale is symmetric. */
int r3d_shape_gemm_nt_q_mxfp4a8(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), N = gi(p, n_p, "N", 0), K = gi(p, n_p, "K", 0);
    if (M <= 0 || N <= 0 || K <= 0) return RAD_E_UNSUPPORTED;
    if (K % 128) return RAD_E_UNSUPPORTED;   /* the activation's scale group and the 32-block */
    const char* q = gs(p, n_p, "dtype");
    if (!q || std::strcmp(q, "mxfp4a8") != 0) return RAD_E_UNSUPPORTED;
    const int64_t kb = K / 128;
    RadOpdDesc list[R3D_GEMM_Q_NOPERAND];
    list[0] = opd((uint32_t)RAD_F8E4M3, M, K);                          /* a */
    RadOpdDesc as = opd((uint32_t)RAD_F32, M, kb);                      /* a_scale */
    as.fill = RAD_FILL_SIGMOID;                                         /* a quantiser's (0, 1) */
    list[1] = as;
    list[2] = opd((uint32_t)RAD_FP4E2M1, N, K);                         /* b: e2m1, two a byte */
    /* The e8m0 scale is filled with 127, the factor 2^0. A normal draw of the u8 grid reaches
     * exponents near 227 (2^100) and overflows the output's finiteness check; 127 is where real
     * checkpoints cluster and is not a reserved value. */
    RadOpdDesc e8 = opd((uint32_t)RAD_E8M0, N, K / 32);                 /* b_scale: e8m0 */
    e8.fill = RAD_FILL_INDEX;
    e8.idx_const = 127;
    list[3] = e8;
    list[4] = opd((uint32_t)RAD_BF16, M, N);                            /* y */
    list[5] = opd_none();                                               /* a_sum */
    list[6] = e8;                                                       /* b_ref: the e8 plane */
    return pick(out, operand, list, R3D_GEMM_Q_NOPERAND);
}

/* The symmetric int4 / int8 weight grids, as libref describes them: int8 activations in the step
 * order against signed codes (I4 two a byte, or I8) with one f16 scale per (row, 128 of K) and one
 * f32 activation scale per row, drawn in (0, 1). No zero point and no a_sum: the dequant is code
 * times scale. */
static int shape_inta8(const RadParam* p, int n_p, int operand, RadOpdDesc* out, const char* q,
                       uint32_t bdt)
{
    const int64_t M = gi(p, n_p, "M", 0), N = gi(p, n_p, "N", 0), K = gi(p, n_p, "K", 0);
    if (M <= 0 || N <= 0 || K <= 0) return RAD_E_UNSUPPORTED;
    if (K % 128) return RAD_E_UNSUPPORTED;   /* the weight's scale group */
    const char* d = gs(p, n_p, "dtype");
    if (!d || std::strcmp(d, q) != 0) return RAD_E_UNSUPPORTED;
    const int64_t kb = K / 128;
    RadOpdDesc list[R3D_GEMM_Q_NOPERAND];
    list[0] = opd((uint32_t)RAD_I8, M, K);                            /* a, step order */
    RadOpdDesc as = opd((uint32_t)RAD_F32, M, 1);                     /* a_scale, per row */
    as.fill = RAD_FILL_SIGMOID;
    list[1] = as;
    list[2] = opd(bdt, N, K);                                         /* b: the signed codes */
    list[3] = opd((uint32_t)RAD_F16, N, kb);                          /* b_scale */
    list[4] = opd((uint32_t)RAD_BF16, M, N);                          /* y */
    list[5] = opd_none();                                             /* a_sum */
    list[6] = opd_none();                                             /* b_ref */
    return pick(out, operand, list, R3D_GEMM_Q_NOPERAND);
}

int r3d_shape_gemm_nt_q_w4a8(const RadParam* p, int n_p, int operand, RadOpdDesc* out)
{
    return shape_inta8(p, n_p, operand, out, "w4a8", (uint32_t)RAD_I4);
}

int r3d_shape_gemm_nt_q_w8a8(const RadParam* p, int n_p, int operand, RadOpdDesc* out)
{
    return shape_inta8(p, n_p, operand, out, "w8a8", (uint32_t)RAD_I8);
}

}  /* extern "C" */


/* ==================================================================== r3d_sample_f32.hip, r3d_attn_paged_h128.hip
 *
 * The vocabulary and the sampler. The schemas in r3d_rows.cpp are transcribed from libref, and
 * every extent, dtype, fill domain and flag below is libref's: a description that differs is a
 * case kbench skips with a one-line notice.
 *
 * The sampler's `params` operand is one plane with fill domain `RAD_FILL_SAMPLE_PARAMS`: the ABI's
 * `RadSampleParams`, 96 bytes (24 f32 words) a row. `fill_chunk` and `idx_max` place the history
 * window and the mask-row count inside it, so the draw yields a window that fits and a `mask_row`
 * that is a row of the bitmask plane (or -1, an unconstrained row).
 */

/* The sampler's parameter plane: [M, 24] f32, one RadSampleParams a row. */
RadOpdDesc opd_params(int64_t M, int64_t hist, int64_t mask_rows) {
    RadOpdDesc d = opd((uint32_t)RAD_F32, M, 24);
    d.fill       = (uint32_t)RAD_FILL_SAMPLE_PARAMS;
    d.fill_chunk = hist;
    d.idx_max    = mask_rows;
    return d;
}

/* libref's widths: a plausible deployment history window, and a candidate width narrower than the
 * vocabulary but wide enough for the narrowing stages. */
enum { kHist = 64, kCand = 64 };

/* Reads M and n_vocab, the extents every vocabulary-width stage opens with. */
int sample_vocab(const RadParam* p, int n_p, int64_t* M, int64_t* nv) {
    *M  = gi(p, n_p, "M", 0);
    *nv = gi(p, n_p, "n_vocab", 0);
    return (*M > 0 && *nv > 0) ? RAD_OK : RAD_E_UNSUPPORTED;
}

/* silu(gate) * up over one fused projection: the operand is [M, 2n] and the output [M, n]. */
int r3d_shape_silu_mul(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const RadOpdDesc list[] = { opd(D, M, 2 * n), opd(D, M, n) };   /* gate_up (gate first), y */
    R3D_PICK(list);
}

/* The fused add + norm + quantiser. `residual` is INOUT and holds the bf16 sum the variance reads,
 * so it has x's shape. The optional `out_bf16` is the value the codes were taken over, for tests. */
int r3d_shape_rmsnorm_quant_fp8(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
    const RadOpdDesc list[] = { opd(D, M, n), opd(D, M, n), opd(D, n),
                                opd((uint32_t)RAD_F8E4M3, M, n), opd((uint32_t)RAD_F32, M, ng),
                                opd((uint32_t)RAD_BF16, M, n) };
    R3D_PICK(list);                          /* x, residual, w, q, scale, out_bf16 */
}

/* Paged attention over one sequence. A context equal to the query length exercises no deep paging,
 * KV split or window, so `ctx` wins when given and is never below the query. `q_len` is the ranged
 * parameter; decode and prefill chunks are different kernels (docs/OPS.md). */
int r3d_shape_attn_paged(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M   = gi(p, n_p, "q_len", 0);
    const int64_t hd  = gi(p, n_p, "head_dim", 0);
    const int64_t bs  = gi(p, n_p, "block_size", 16);
    const int64_t gqa = gi(p, n_p, "gqa", 1);
    const int64_t nh  = gi(p, n_p, "n_head", gqa);
    if (M <= 0 || hd <= 0 || bs <= 0 || gqa <= 0 || nh <= 0 || nh % gqa) return RAD_E_UNSUPPORTED;
    const int64_t kvh = nh / gqa;
    const int64_t S   = 1;
    /* `ctx` as the reference spells it, not the band's `max_ctx`; absent, the context is the
     * query chunk. */
    int64_t ctx = gi(p, n_p, "ctx", 0);
    if (ctx < M) ctx = M;
    const int64_t max_blocks = (ctx + bs - 1) / bs;

    const uint32_t Q  = act_dt(p, n_p);
    const uint32_t kv = rad_dtype_parse(gs(p, n_p, "kv_dtype"));
    const uint32_t KV = kv == (uint32_t)RAD_DT_INVALID ? Q : kv;

    const RadOpdDesc list[] = {
        opd(Q,  S * M, nh, hd),                                 /* q */
        opd(KV, max_blocks, kvh, bs, 2 * hd),                   /* kv_cache */
        opd_idx(S, max_blocks, -1, 0, max_blocks),              /* block_table: indexes blocks,
                                                                   may repeat -- a page table */
        opd_idx(S, 0, ctx),                                     /* seqused: a constant, not a draw */
        opd_none(),                                             /* k_descale: no per-(seq,head) */
        opd_none(),                                             /* v_descale */
        opd_none(),                                             /* cu_seqlens: the rectangular
                                                                   reading at one sequence */
        opd(Q,  S * M, nh, hd)                                  /* out */
    };
    R3D_PICK(list);
}

/* Exact per-row top-R. `pairs` packs an int32 id's bits in the first slot beside its value, in an
 * f32 plane. */
int r3d_shape_row_topk(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), N = gi(p, n_p, "N", 0), R = gi(p, n_p, "R", 0);
    if (M <= 0 || N <= 0 || R <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    const RadOpdDesc list[] = {
        opd(D, M, N), opd((uint32_t)RAD_I32, M, R), opd((uint32_t)RAD_F32, M, R),
        opd((uint32_t)RAD_F32, M, R, 2)
    };                                                          /* x, idx, val, pairs */
    R3D_PICK(list);
}

int r3d_shape_sample_argmax(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t M = 0, nv = 0;
    if (sample_vocab(p, n_p, &M, &nv) < 0) return RAD_E_UNSUPPORTED;
    const RadOpdDesc list[] = {
        opd((uint32_t)RAD_F32, M, nv), opd_params(M, 0, 0), opd((uint32_t)RAD_I32, M)
    };                                                          /* logits, params, token */
    R3D_PICK(list);
}

int r3d_shape_sample_temp(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t M = 0, nv = 0;
    if (sample_vocab(p, n_p, &M, &nv) < 0) return RAD_E_UNSUPPORTED;
    const RadOpdDesc list[] = { opd((uint32_t)RAD_F32, M, nv), opd_params(M, 0, 0) };
    R3D_PICK(list);                                             /* logits INOUT, params */
}

/* The gated MLP band, slot for slot from libref's shape hook: n = N/2, the weight's scales per
 * (ceil(N/group), ceil(K/group)) in bf16 (the container's `weight_scale_inv`), the output's codes
 * per (M, n) and its scales per (M, ceil(n/group)). The checks mirror the kernel's: fp8a8 only,
 * group a power of two in [32, 256] no wider than n, and K a whole number of groups, so the hook
 * never describes a shape the kernel refuses. */
int r3d_shape_gemm_nt_q_gated(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M     = gi(p, n_p, "M", 0);
    const int64_t N     = gi(p, n_p, "N", 0);
    const int64_t K     = gi(p, n_p, "K", 0);
    const int64_t group = gi(p, n_p, "group", 0);
    const char* q = gs(p, n_p, "dtype");
    if (!q || std::strcmp(q, "fp8a8") != 0) return RAD_E_UNSUPPORTED;
    const int64_t n = N / 2;
    if (M <= 0 || M > 64 || N <= 1 || (N & 1) || K <= 0 || n <= 0) return RAD_E_UNSUPPORTED;
    if (group != 32 && group != 64 && group != 128 && group != 256) return RAD_E_UNSUPPORTED;
    if (group > n || (K % group)) return RAD_E_UNSUPPORTED;
    const int64_t kb = (K + group - 1) / group, nb = (N + group - 1) / group;
    const int64_t ng = (n + group - 1) / group;
    const RadOpdDesc list[] = {
        opd((uint32_t)RAD_F8E4M3, M, K),            /* a */
        opd((uint32_t)RAD_F32, M, kb),              /* a_scale */
        opd((uint32_t)RAD_F8E4M3, N, K),            /* b: gate rows [0,n), up rows [n,N) */
        opd((uint32_t)RAD_BF16, nb, kb),            /* b_scale: the container's form */
        opd((uint32_t)RAD_F8E4M3, M, n),            /* q */
        opd((uint32_t)RAD_F32, M, ng),              /* scale */
        opd((uint32_t)RAD_BF16, M, n),              /* out_bf16, optional */
    };
    R3D_PICK(list);
}

/* DFlash2's candidate walk, slot for slot from libref's shape hook. `cand` and `anchor` hold token
 * ids bounded by n_vocab so the codebook gather stays inside the plane; cand/unary/hp carry one
 * row per (sequence, step), M * steps. */
int r3d_shape_dflash_select(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M     = gi(p, n_p, "M", 0);
    const int64_t steps = gi(p, n_p, "steps", 0);
    const int64_t K     = gi(p, n_p, "top_k", 0);
    const int64_t R     = gi(p, n_p, "rank", 0);
    const int64_t nv    = gi(p, n_p, "n_vocab", 0);
    const int64_t astr  = gi(p, n_p, "anchor_stride", 1);
    if (M <= 0 || steps <= 0 || K <= 0 || R <= 0 || nv <= 0 || astr <= 0) return RAD_E_UNSUPPORTED;
    const int64_t rows = M * steps;
    const RadOpdDesc list[] = {
        opd_idx(rows, nv, -1, 0, K),                                 /* cand: drawn over the vocab */
        opd((uint32_t)RAD_F32, rows, K),                             /* unary */
        opd((uint32_t)RAD_BF16, rows, R),                            /* hp, absent is a plane of ones */
        opd_idx(M * astr, nv),                                       /* anchor */
        opd((uint32_t)RAD_BF16, nv, R),                              /* pred */
        opd((uint32_t)RAD_BF16, nv, R),                              /* succ */
        opd_idx(M, nv, -1, 0, steps),                                /* tokens */
    };
    R3D_PICK(list);
}

int r3d_shape_sample_mask(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t M = 0, nv = 0;
    if (sample_vocab(p, n_p, &M, &nv) < 0) return RAD_E_UNSUPPORTED;
    /* mask_rows is M: one bitmask row a sequence is the widest the plane gets, and an
     * unconstrained row shares it with mask_row -1. A set bit is an allowed token. */
    const RadOpdDesc list[] = { opd((uint32_t)RAD_F32, M, nv), opd_params(M, 0, M),
                                opd((uint32_t)RAD_I32, M, (nv + 31) / 32) };
    R3D_PICK(list);
}

int r3d_shape_sample_logprobs(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t M = 0, nv = 0;
    if (sample_vocab(p, n_p, &M, &nv) < 0) return RAD_E_UNSUPPORTED;
    const int64_t k = gi(p, n_p, "k", 0);
    if (k <= 0) return RAD_E_UNSUPPORTED;
    const RadOpdDesc list[] = { opd((uint32_t)RAD_F32, M, nv), opd_params(M, 0, 0),
                                opd((uint32_t)RAD_F32, M, 2 + 2 * k) };
    R3D_PICK(list);                                   /* logits, params, stats [M, 2+2k] */
}

int r3d_shape_sample_topk(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t M = 0, nv = 0;
    if (sample_vocab(p, n_p, &M, &nv) < 0) return RAD_E_UNSUPPORTED;
    const int64_t nc = nv < (int64_t)kCand ? nv : (int64_t)kCand;
    const RadOpdDesc list[] = {
        opd((uint32_t)RAD_F32, M, nv), opd_params(M, 0, 0),
        opd((uint32_t)RAD_I32, M, nc), opd((uint32_t)RAD_F32, M, nc),
        opd((uint32_t)RAD_I32, M, 2 * nc)
    };                            /* logits, params, cand_idx, cand_val (logits, not probs), pairs */
    R3D_PICK(list);
}

/* The four narrowing stages (top-p, min-p, typical, XTC), in place over the candidate set. The set
 * is drawn, not sorted: the ABI has no descending fill domain, and top-p is a prefix rule in
 * candidate order, which applies to any order. */
int r3d_shape_narrow(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t nc = gi(p, n_p, "n_cand", 0);
    if (M <= 0 || nc <= 0) return RAD_E_UNSUPPORTED;
    const RadOpdDesc list[] = { opd_idx(M, nc, -1, 0, nc), opd((uint32_t)RAD_F32, M, nc),
                                opd_params(M, 0, 0) };
    R3D_PICK(list);                          /* cand_idx INOUT, cand_val INOUT, params */
}

int r3d_shape_sample_pick(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t nc = gi(p, n_p, "n_cand", 0);
    if (M <= 0 || nc <= 0) return RAD_E_UNSUPPORTED;
    const RadOpdDesc list[] = { opd_idx(M, nc, -1, 0, nc), opd((uint32_t)RAD_F32, M, nc),
                                opd_params(M, 0, 0), opd((uint32_t)RAD_I32, M) };
    R3D_PICK(list);                          /* cand_idx, cand_val, params (seed and pos), token */
}

/* Penalties and DRY: logits, params with a history window, and the history of token ids. DRY's
 * breaker plane marks about half its positions, so the exemption runs as well as the penalty. */
int r3d_shape_sample_penalties(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t M = 0, nv = 0;
    if (sample_vocab(p, n_p, &M, &nv) < 0) return RAD_E_UNSUPPORTED;
    const RadOpdDesc list[] = {
        opd((uint32_t)RAD_F32, M, nv), opd_params(M, kHist, 0),
        opd_idx(M, nv, -1, RAD_OPD_F_IDX_HISTORY, kHist)
    };
    R3D_PICK(list);
}

int r3d_shape_sample_dry(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t M = 0, nv = 0;
    if (sample_vocab(p, n_p, &M, &nv) < 0) return RAD_E_UNSUPPORTED;
    RadOpdDesc brk = opd_idx(M, 2, -1, 0, kHist);
    brk.dtype = (uint32_t)RAD_U8;
    const RadOpdDesc list[] = {
        opd((uint32_t)RAD_F32, M, nv), opd_params(M, kHist, 0),
        opd_idx(M, nv, -1, RAD_OPD_F_IDX_HISTORY, kHist), brk
    };
    R3D_PICK(list);
}

/* The windowed n-gram ban adds a {n, window} pair a row and an optional exemption bitmask. */
int r3d_shape_sample_ngram_ban(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t M = 0, nv = 0;
    if (sample_vocab(p, n_p, &M, &nv) < 0) return RAD_E_UNSUPPORTED;
    const RadOpdDesc list[] = {
        opd((uint32_t)RAD_F32, M, nv), opd_params(M, kHist, 0),
        opd_idx(M, nv, -1, RAD_OPD_F_IDX_HISTORY, kHist),
        opd_idx(M, 64, -1, 0, 2),
        opd((uint32_t)RAD_I32, M, (nv + 31) / 32)
    };
    R3D_PICK(list);
}

/* The four narrowing stages share one description; the op names the predicate. */
int r3d_shape_sample_topp(const RadParam* p, int n_p, int o, RadOpdDesc* out) {
    return r3d_shape_narrow(p, n_p, o, out);
}
int r3d_shape_sample_minp(const RadParam* p, int n_p, int o, RadOpdDesc* out) {
    return r3d_shape_narrow(p, n_p, o, out);
}
int r3d_shape_sample_typical(const RadParam* p, int n_p, int o, RadOpdDesc* out) {
    return r3d_shape_narrow(p, n_p, o, out);
}
int r3d_shape_sample_xtc(const RadParam* p, int n_p, int o, RadOpdDesc* out) {
    return r3d_shape_narrow(p, n_p, o, out);
}
