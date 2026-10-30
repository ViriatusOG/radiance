/* clef_fp8 -- Cloudflare's clef and clef-flash decision models, block fp8: the qwen35_fp8 trunk
 * with the joint schema head on the card (arch/common/rad_block_clef.h), served on
 * /v1/systemone.
 *
 * This is arch/qwen35_fp8/qwen35_fp8.cpp built again under its own namespace and architecture
 * id, as qwen35_w8a8 and qwen35moe_fp8 are: the trunk's declare, step and probe are that file's,
 * untouched, and this file adds the head after them. A container selects it with `rad-convert
 * --arch clef` and the fp8 recipe; the lm_head stays bf16, which the head reads.
 */
#define QWEN35_NS      qwen35_clef_fp8
#define QWEN35_ARCH_ID "clef"
#define RAD_ARCH_NO_EXPORTS 1
#include "../qwen35_fp8/qwen35_fp8.cpp"
#undef RAD_ARCH_NO_EXPORTS

#include <arch/rad_block_clef.h>

#include <cstdio>

namespace clef_fp8 {

using namespace rad::arch;

static ClefHead g_head[MAX_RANKS];

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;
    if (!ClefHead::present(meta)) {
        std::fprintf(stderr, "radiance: clef_fp8 serves a container converted with "
                             "--set decision.type=clef and the head added (--add "
                             "joint_head.safetensors); this one declares no decision model\n");
        return RAD_E_INVAL;
    }
    RAD_ARCH_TRY(qwen35_clef_fp8::declare(b, meta, ctx));
    qwen35_clef_fp8::Model& m = qwen35_clef_fp8::g_model[ctx->rank];
    /* The lm_head's row would be sampled once and discarded: with this fill declared, the trunk
     * skips the lm_head on a step every sampled row of which carries a request's words
     * (plugin_words_cover_rows) and leaves zero rows instead. */
    m.op_logits_fill = clef_logits_fill(b, m.nm, m.g, m.b_logits);
    if (!m.op_logits_fill) return RAD_E_INVAL;
    ClefHead& h = g_head[ctx->rank];
    h = ClefHead{};
    return h.declare(b, meta, ctx, m.nm, m.g, m.a_h.x, m.b_logits, m.w_lm_head, m.kv_full,
                     (int)m.g.n_layer + 256);
}

static void step(RadCtx* c, const RadBatch* batch) {
    qwen35_clef_fp8::step(c, batch);
    /* An encoder pass leaves no rows for the head, and a drafter's pass is not a trunk step. */
    if (batch->enc || batch->draft_pass != 0) return;
    qwen35_clef_fp8::Model& m = qwen35_clef_fp8::g_model[rad_rank(c)];
    ClefHead& h = g_head[rad_rank(c)];
    /* The final norm's rows are in `a_h.x` after either of the trunk's step paths. */
    h.step_tokens(c, batch, m.a_h.x);
    if (batch->n_out > 0) h.step_heads(c, batch, m.b_logits, m.w_lm_head);
}

}  /* namespace clef_fp8 */

RAD_ARCH_PROBE(qwen35_clef_fp8)
RAD_ARCH_PLUGIN(clef_fp8, "clef", "fp8_e4m3", "0.1.0",
                "Cloudflare's clef / clef-flash (9B, 27B): the Qwen3.5 trunk of qwen35_fp8, "
                "block-scaled fp8, with the joint schema head on the card for /v1/systemone")
