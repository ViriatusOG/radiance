/* r3d_dflash_f32.hip -- DFlash2's candidate walk, `dflash_select`, on the card.
 *
 * A block-diffusion drafter proposes every position of a block in one backbone pass, so position
 * l+1 is not conditioned on what the drafter chose at l. This op restores that dependence: it walks
 * the `steps` positions in order and, at each, picks the candidate that maximises
 *
 *     score[l, c] = unary[l, c] + sum_r pred[pid(l)][r] * hp[l][r] * succ[cand[l, c]][r]
 *
 * where pid(0) is the anchor -- the token the target just committed, read at
 * `anchor[m * anchor_stride]` -- and pid(l) is the walk's own previous pick, `cand[l-1, chosen]`.
 * `hp` absent is a plane of ones, which drops the trilinear form to a low-rank bigram (DSpark's
 * Markov head). libref's `ref_dflash_select` is the law, and this file transcribes its arithmetic
 * and its order.
 *
 * ============================== why it is a kernel and not a GEMM ==============================
 *
 * The walk is serial in l: the predecessor of step l is not known until step l-1 has been decided,
 * so the [steps, K, K] edge tensor cannot be scored in advance -- scoring it computes K times the
 * edges that are ever read. What is parallel is the sequences (M of them) and, inside a step, the K
 * candidates. So a block takes `256 / top_k` sequences and gives each lane one (sequence,
 * candidate) pair: the lane runs the rank sum for its own edge, in the reference's order, and one
 * lane per sequence scans the K scores and takes the pick. The loop over `steps` stays serial, with
 * the two LDS round trips per step (the scores out, the pick back) that the dependence requires.
 *
 * The rank sum is the whole cost: at the band (steps=7, top_k=16, rank=256, M=256) a walk is
 * 7 * 16 * 256 = 28672 fused terms, and the bytes it touches are the K successor rows a step reads,
 * 16 * 512 B of bf16, so the op is a gather over `succ` [n_vocab, rank] -- 66.8 MiB of codebook at
 * the band, of which the walk reads M * steps * K = 28672 rows, about 14.7 MiB, plus one
 * predecessor row a step. There is no arithmetic to hide that behind, which is why the sum is a
 * plain ascending chain and not a tree: a tree buys nothing here and costs the agreement.
 *
 * ============================== agreement, and where this file declines ==============================
 *
 * Bit-exact by construction, not by tolerance:
 *
 *   - the rank sum is `sc = unary; for (i = 0; i < rank; ++i) sc += ax[i] * succ[i];` -- ascending
 *     in r, one accumulator, no reduction, no tree, with `-ffp-contract=off` so no FMA fusion
 *     changes the rounding of a term;
 *   - `ax[i] = pred[pid][i] * hp[l][i]` is rounded to f32 before it meets `succ`, because the
 *     reference writes it into a float array first (and with `hp` absent it is `pred * 1.0f`, which
 *     is the same value);
 *   - the argmax is a strict `>` over c ascending, so the earliest candidate wins a tie and a NaN
 *     never displaces the incumbent -- the reference's loop, statement for statement;
 *   - a predecessor outside the vocabulary contributes zero, and a candidate outside it is scored
 *     by `unary` alone: both are the reference's tests (`pid >= 0 && pid < nv`, `tok >= 0 &&
 *     tok < nv`), and both are why padding rows -- a shard's unused rows, a block shorter than
 *     `steps` -- are legal inputs;
 *   - the pick of step l is written to `tokens[m, l]` and becomes pid(l+1) unchanged, including
 *     when it is out of range for the codebooks.
 *
 * Declines, all of them the shim's, none reachable from a row that carries the constraints its row
 * states: a codebook (`pred`, `succ`, `hp`) that is not bf16, because this is the bf16
 * instantiation -- the band's `dtype` is the codebooks' width and a row asks for bf16 (RAD_E_DTYPE);
 * `unary` that is not f32, `cand`/`anchor`/`tokens` that are not i32, exactly as the schema's
 * operand list types them (RAD_E_DTYPE); a codebook whose rows are not contiguous -- the gather is
 * `token * rank + r` and a row stride inside the row is not that (RAD_E_STRIDE); `top_k` above 256,
 * which is one lane a candidate and no more candidates than lanes (RAD_E_UNSUPPORTED); and a `cand`
 * plane that describes more sequences than the band's `M`, which is `band_rows` refusing and is
 * RAD_E_SHAPE, not a truncation.
 */

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_codecs.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <rad_types.h>

namespace {

constexpr uint32_t kThr = 256;              /* 8 waves: the block's (sequence, candidate) lanes */
constexpr uint32_t kKMax = kThr;            /* one lane a candidate, so top_k <= 256 */

/* Whole-register loads of the planes the walk gathers through. bf16 is a 16-bit load and the
 * library's codec, which is the same decode libref's `ldt` performs. */
__device__ __forceinline__ int32_t ldg_i32(const int32_t* p) {
    return (int32_t)ldg_u32((const uint32_t*)p);
}
__device__ __forceinline__ float ldg_f32(const float* p) { return *(const float*)p; }
__device__ __forceinline__ uint16_t ldg_u16(const uint16_t* p) { return *(const uint16_t*)p; }

/* ------------------------------------------------------------------ the walk

 * `per` sequences per block, `K` candidates per sequence, so lane (s, c) owns the edge
 * (sequence s, candidate c) of the step in hand. `hp` may be null: then the `hp` factor is 1.0f,
 * exactly as the reference's `(hp ? ldt(hp, ...) : 1.0f)` writes it. */
__global__ __launch_bounds__(kThr) void dflash_select_kernel(
    const int32_t* __restrict__ cand, const float* __restrict__ un,
    const uint16_t* __restrict__ hp, const int32_t* __restrict__ anc,
    const uint16_t* __restrict__ pred, const uint16_t* __restrict__ succ,
    int32_t* __restrict__ out, uint32_t M, uint32_t steps, uint32_t K, uint32_t R, uint32_t nv,
    uint32_t astr, long long cand_ld, long long un_ld, long long hp_ld, long long pred_ld,
    long long succ_ld, long long tok_ld, uint32_t per) {

    __shared__ int32_t s_pid[kThr];         /* the walk's predecessor, one per sequence */
    __shared__ float   s_sc[kThr];          /* this step's K scores, one per lane */

    const uint32_t s = threadIdx.x / K;                 /* the sequence this lane walks for */
    const uint32_t c = threadIdx.x - s * K;             /* the candidate this lane scores */
    const uint32_t seq0 = blockIdx.x * per + s;
    /* A block of 256 lanes holds ceil(256 / K) sequences and the last group can be partial. A lane
     * with no sequence replays the last real one: it reaches every __syncthreads() and its stores
     * are suppressed, and the loads it performs are in range because the sequence it replays is. */
    const uint32_t seq = seq0 < M ? seq0 : M - 1;
    const bool mine = seq0 < M;

    /* pid(0): the anchor, at `m * anchor_stride` in a flat plane. The reference reads it through
     * `ld_int`, so a negative id is a legal "no predecessor" and not an error. */
    int32_t pid = ldg_i32(anc + (size_t)seq * (size_t)astr);
    const bool has_hp = (hp != nullptr);

    for (uint32_t l = 0; l < steps; ++l) {
        const uint32_t r = seq * steps + l;              /* cand/unary/hp are one row per (m, l) */
        const size_t   cb = (size_t)r * (size_t)cand_ld;
        const int32_t  tok = ldg_i32(cand + cb + c);
        float sc = ldg_f32(un + (size_t)r * (size_t)un_ld + c);

        /* A predecessor outside the vocabulary contributes nothing; a candidate outside it is
         * scored by `unary` alone. Both tests are the reference's, and both are why a shard's
         * padding rows and a block shorter than `steps` are legal inputs. */
        const bool have_p = pid >= 0 && (uint32_t)pid < nv;
        const bool have_s = tok >= 0 && (uint32_t)tok < nv;
        if (have_s) {
            const uint16_t* pp = pred + (size_t)(have_p ? pid : 0) * (size_t)pred_ld;
            const uint16_t* sp = succ + (size_t)tok * (size_t)succ_ld;
            const uint16_t* hpb = hp + (size_t)r * (size_t)hp_ld;
            for (uint32_t i = 0; i < R; ++i) {
                /* ax[i], rounded to f32 where the reference writes it into its float array, and
                 * then the ascending chain. 1.0f for a missing hp is the reference's own factor. */
                const float ax = have_p
                    ? bf16_to_f32(ldg_u16(pp + i)) * (has_hp ? bf16_to_f32(ldg_u16(hpb + i)) : 1.0f)
                    : 0.0f;
                sc += ax * bf16_to_f32(ldg_u16(sp + i));
            }
        }
        s_sc[threadIdx.x] = sc;                            /* == s_sc[s * K + c] */
        __syncthreads();

        if (c == 0) {
            /* The reference's argmax: strict `>`, c ascending, so the earliest candidate wins a
             * tie and a NaN never takes the lead from a number. */
            uint32_t best_c = 0;
            float best_s = -INFINITY;
            for (uint32_t j = 0; j < K; ++j) {
                const float v = s_sc[s * K + j];
                if (v > best_s) { best_s = v; best_c = j; }
            }
            const int32_t pick = ldg_i32(cand + cb + best_c);
            if (mine) out[(size_t)seq * (size_t)tok_ld + l] = pick;
            s_pid[s] = pick;                               /* pid(l+1), out-of-range included */
        }
        __syncthreads();
        pid = s_pid[s];
    }
}

/* ------------------------------------------------------------------ shim helpers */

/* The plane's row pitch, from the plane: `rows` is its row count and `ld` the pitch a row's
 * elements start at, K apart. A rank-1 plane is a run of rows K apart, which is what the
 * reference's rowoff does for one. */
static inline int rows_pitch(const RadTensor* t, long long K, long long* rows, long long* ld) {
    const long long p = r3d_rows_pitch(t, K, rows);
    if (p < 0) return RAD_E_STRIDE;
    if (K <= 0 || *rows <= 0 || *rows > 0x7fffffffLL) return RAD_E_SHAPE;
    if (p < K) return RAD_E_SHAPE;
    *ld = p;
    return RAD_OK;
}

/* A codebook: [n, rank] bf16 with contiguous rows, because the gather is `token * rank + r`. */
static inline int codebook_ok(const RadTensor* t, long long rank) {
    if (!t || !t->data) return RAD_E_INVAL;
    if (!r3d_is_bf16(t)) return RAD_E_DTYPE;
    if (!r3d_aligned(r3d_p(t), 2)) return RAD_E_ALIGN;
    if (rank > 0 && t->rank >= 1 && r3d_stride(t, (int)t->rank - 1) != 1) return RAD_E_STRIDE;
    return RAD_OK;
}

static inline int r3d_launch_status(int rc) {
    if (rc == 0) return RAD_OK;
    switch (rc) {
        case -1: return RAD_E_INVAL;
        case -2: return RAD_E_SHAPE;
        case -3: return RAD_E_STRIDE;
        default: return RAD_E_DEVICE;
    }
}

}  /* namespace */

#define R3D_NEED(x) \
    do { const int _s = (x); if (_s != RAD_OK) return R3D_NO(_s); } while (0)

/* ================================================================== launcher
 *
 *   cand    [M*steps][top_k]  i32, candidate token ids, -1 for padding
 *   unary   [M*steps][top_k]  f32, the candidate-independent half of the score
 *   hp      [M*steps][rank]   bf16, the step's factor; 0 (absent) is a plane of ones
 *   anchor  [M*anchor_stride] i32, the token each walk starts from, at m*anchor_stride
 *   pred    [n_vocab][rank]   bf16, the predecessor codebook
 *   succ    [n_vocab][rank]   bf16, the successor codebook the walk gathers with its picks
 *   tokens  [M][steps]        i32, out: the accepted prefix, one pick a step
 *
 * 0, -1 null, -2 bad extent, -3 pitch, -4 the HIP launch failed. No allocation, no synchronisation.
 */
extern "C" int r3d_dflash_select(long cand, long unary, long hp, long anchor, long pred, long succ,
                                long tokens, int M, int steps, int top_k, int rank, int n_vocab,
                                int anchor_stride, long long cand_ld, long long un_ld,
                                long long hp_ld, long long pred_ld, long long succ_ld,
                                long long tok_ld, long stream) {
    if (!cand || !unary || !anchor || !pred || !succ || !tokens) return -1;
    if (M <= 0 || steps <= 0 || top_k <= 0 || rank <= 0 || n_vocab <= 0 || anchor_stride <= 0)
        return -2;
    if ((unsigned)top_k > kKMax) return -2;               /* one lane a candidate */
    if (!cand_ld) cand_ld = top_k;
    if (!un_ld) un_ld = top_k;
    if (!hp_ld) hp_ld = rank;
    if (!pred_ld) pred_ld = rank;
    if (!succ_ld) succ_ld = rank;
    if (!tok_ld) tok_ld = steps;
    if (cand_ld < top_k || un_ld < top_k) return -3;
    if (hp_ld < rank || pred_ld < rank || succ_ld < rank) return -3;
    if (tok_ld < steps) return -3;
    const uint32_t per = (kThr + (uint32_t)top_k - 1) / (uint32_t)top_k;
    R3D_KLAUNCH(R3D_KNAME(dflash_select_kernel), dim3((M + per - 1) / per), dim3(kThr),
                (hipStream_t)(uintptr_t)stream, (const int32_t*)(uintptr_t)cand,
                (const float*)(uintptr_t)unary, (const uint16_t*)(uintptr_t)hp,
                (const int32_t*)(uintptr_t)anchor, (const uint16_t*)(uintptr_t)pred,
                (const uint16_t*)(uintptr_t)succ, (int32_t*)(uintptr_t)tokens, (uint32_t)M,
                (uint32_t)steps, (uint32_t)top_k, (uint32_t)rank, (uint32_t)n_vocab,
                (uint32_t)anchor_stride, cand_ld, un_ld, hp_ld, pred_ld, succ_ld, tok_ld, per);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* ================================================================== shim
 *
 * libref's operand list is {cand, unary, hp?, anchor, pred, succ, tokens} and its parameter list is
 * {M, steps, top_k, rank, n_vocab, anchor_stride?, dtype}. Sequences come from `cand` -- the plane
 * holds M * steps * top_k ids -- through `band_rows`, so a plane describing more sequences than the
 * band's M is refused, not truncated. `rank` and `n_vocab` fall back to the predecessor codebook's
 * own shape, as the reference's do, and `anchor_stride` defaults to 1.
 */
extern "C" int r3d_rad_dflash_select(const RadArgs* a, RadStream stream) {
    const RadTensor* cand = r3d_opt(a, 0);
    const RadTensor* un   = r3d_opt(a, 1);
    const RadTensor* hp   = r3d_opt(a, 2);          /* OPD_O: absent is a plane of ones */
    const RadTensor* anc  = r3d_opt(a, 3);
    const RadTensor* pred = r3d_opt(a, 4);
    const RadTensor* succ = r3d_opt(a, 5);
    const RadTensor* tok  = r3d_opt(a, 6);
    if (!cand || !cand->data || !un || !un->data || !anc || !anc->data || !pred || !pred->data ||
        !succ || !succ->data || !tok || !tok->data)
        return R3D_NO(RAD_E_INVAL);

    /* The band names the codebooks' width; this is the bf16 instantiation. */
    const char* dt = r3d_gets(a, "dtype");
    if (dt && !r3d_dt_named(RAD_BF16, dt)) return R3D_NO(RAD_E_DTYPE);

    const long long steps = r3d_geti_or(a, "steps", 0);
    const long long K = r3d_geti_or(a, "top_k", 0);
    const long long astr = r3d_geti_or(a, "anchor_stride", 1);
    long long R = r3d_geti_or(a, "rank", pred->rank >= 2 ? r3d_dim(pred, (int)pred->rank - 1) : 0);
    long long nv = r3d_geti_or(a, "n_vocab",
                              pred->rank >= 2 ? r3d_dim(pred, (int)pred->rank - 2) : 0);
    const long long band_M = r3d_geti_or(a, "M", 0);
    if (steps <= 0 || K <= 0 || astr <= 0 || R <= 0 || nv <= 0) return R3D_NO(RAD_E_SHAPE);
    if (K > (long long)kKMax) return R3D_NO(RAD_E_UNSUPPORTED);
    if (steps > 0x7fffffffLL / (K * 256) ) return R3D_NO(RAD_E_SHAPE);

    /* Sequences from the plane, the band as a ceiling -- band_rows refuses, it does not truncate. */
    const long long all = steps * K;
    if (all <= 0 || r3d_numel(cand) % all) return R3D_NO(RAD_E_SHAPE);
    const long long M = r3d_numel(cand) / all;
    if (M <= 0 || (band_M > 0 && M > band_M)) return R3D_NO(RAD_E_SHAPE);

    /* The reference's extent tests, in its order. */
    if (r3d_numel(un) < M * steps * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(anc) < (M - 1) * astr + 1) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(tok) < M * steps) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(pred) < nv * R || r3d_numel(succ) < nv * R) return R3D_NO(RAD_E_SHAPE);
    if (hp && hp->data && r3d_numel(hp) < M * steps * R) return R3D_NO(RAD_E_SHAPE);

    if (!r3d_is_i32(cand)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f32(un)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_i32(anc)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_i32(tok)) return R3D_NO(RAD_E_DTYPE);
    R3D_NEED(codebook_ok(pred, R));
    R3D_NEED(codebook_ok(succ, R));
    if (hp && hp->data) R3D_NEED(codebook_ok(hp, R));
    if (!r3d_aligned(r3d_p(cand), 4) || !r3d_aligned(r3d_p(un), 4) || !r3d_aligned(r3d_p(anc), 4) ||
        !r3d_aligned(r3d_p(tok), 4))
        return R3D_NO(RAD_E_ALIGN);

    long long cand_ld = 0, un_ld = 0, tok_ld = 0, rows = 0;
    R3D_NEED(rows_pitch(cand, K, &rows, &cand_ld));
    R3D_NEED(rows_pitch(un, K, &rows, &un_ld));
    R3D_NEED(rows_pitch(tok, steps, &rows, &tok_ld));
    long long hp_ld = R, pred_ld = R, succ_ld = R;
    if (hp && hp->data) R3D_NEED(rows_pitch(hp, R, &rows, &hp_ld));
    R3D_NEED(rows_pitch(pred, R, &rows, &pred_ld));
    R3D_NEED(rows_pitch(succ, R, &rows, &succ_ld));

    return r3d_launch_status(r3d_dflash_select(
        r3d_addr(cand), r3d_addr(un), (hp && hp->data) ? r3d_addr(hp) : 0L, r3d_addr(anc),
        r3d_addr(pred), r3d_addr(succ), r3d_addr(tok), (int)M, (int)steps, (int)K, (int)R, (int)nv,
        (int)astr, cand_ld, un_ld, hp_ld, pred_ld, succ_ld, tok_ld, (long)(uintptr_t)stream));
}
