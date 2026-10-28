/* clef_bf16 -- Cloudflare's clef and clef-flash decision models, bf16: the qwen35_bf16 trunk with
 * the joint schema head on the card (arch/common/rad_block_clef.h), served on /v1/systemone.
 *
 * This is arch/qwen35_bf16/qwen35_bf16.cpp built again under its own architecture id, as
 * qwen35_w8a8 is qwen35_fp8 built again: the trunk's declare and step are that file's, untouched,
 * and this file adds the head after them. A container selects it with `rad-convert --arch clef`,
 * so a plain qwen35 container never pays for the head and the qwen35 plugins carry no decision
 * code. The head needs `decision.type=clef` and the head's geometry in the container header,
 * which rad-convert writes from joint_head_config.json and joint_head.safetensors (--add).
 */
#define RAD_ARCH_NO_EXPORTS 1
#include "../qwen35_bf16/qwen35_bf16.cpp"
#undef RAD_ARCH_NO_EXPORTS

#include <arch/rad_block_clef.h>

#include <cstdio>

namespace clef_bf16 {

using namespace rad::arch;

/* One head per rank beside the trunk's own per-rank state (qwen35_bf16::g_model). */
static ClefHead g_head[MAX_RANKS];

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;
    if (!ClefHead::present(meta)) {
        std::fprintf(stderr, "radiance: clef_bf16 serves a container converted with "
                             "--set decision.type=clef and the head added (--add "
                             "joint_head.safetensors); this one declares no decision model\n");
        return RAD_E_INVAL;
    }
    RAD_ARCH_TRY(qwen35_bf16::declare(b, meta, ctx));
    /* The trunk's declare does not take part in sizing declares (no probe), so its state and the
     * head's are the real ones. The head goes after the vision tower: its buffers join the same
     * arena, and it reads the final norm's rows, the lm_head and the trunk's cache block size. */
    qwen35_bf16::Model& m = qwen35_bf16::g_model[ctx->rank];
    ClefHead& h = g_head[ctx->rank];
    h = ClefHead{};
    return h.declare(b, meta, ctx, m.nm, m.g, m.b_h, m.b_logits, m.w_lm_head, m.kv_full,
                     (int)m.g.n_layer + 256);
}

static void step(RadCtx* c, const RadBatch* batch) {
    qwen35_bf16::step(c, batch);
    /* An encoder pass runs the tower alone and leaves no rows for the head. */
    if (batch->enc) return;
    qwen35_bf16::Model& m = qwen35_bf16::g_model[rad_rank(c)];
    ClefHead& h = g_head[rad_rank(c)];
    /* The final norm's rows are still in `h` after the trunk's step; the logits row of each
     * sampled sequence has been written, and the head adds its values beside it. */
    h.step_tokens(c, batch, m.b_h);
    if (batch->n_out > 0) h.step_heads(c, batch, m.b_logits, m.w_lm_head);
}

}  /* namespace clef_bf16 */

RAD_ARCH_PLUGIN(clef_bf16, "clef", "", "0.1.0",
                "Cloudflare's clef / clef-flash (9B, 27B): the Qwen3.5 trunk of qwen35_bf16, bf16, "
                "with the joint schema head on the card for /v1/systemone")
