/* engine_draft.cpp -- the two drafter drivers (spec §10).
 *
 * A draft head runs as extra passes after the trunk step. SERIAL runs `depth` dependent passes;
 * BLOCK runs a context pass, then one pass that fills the whole block. The architecture plugin
 * declares which (rad_declare_drafter); nothing here depends on the model.
 */
#include "engine_priv.h"

#include "sample/mtp_couple_evidence.h"
#include "sched/mtp_couple_bind.h"
#include "sched/mtp_couple_plan.h"
#include "sched/scheduler.h"
#include "device/device.h"
#include "rad_gdn_anchor.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>


namespace rad {

/* ------------------------------------------------------------------ the card's step state */
/* Every rank's zeroed StepSlots and sorted end-of-generation ids, and the pinned host copy of rank
 * 0's slots the step reads back. Allocated once at start, so the step path never allocates. */
int Engine::step_state_init() {
    std::vector<int32_t> eog;
    if (vocab_) eog = vocab_->eog();
    std::sort(eog.begin(), eog.end());
    eog.erase(std::unique(eog.begin(), eog.end()), eog.end());
    /* Scheduler::ends_every_request reads the same ids. */
    sched_.set_eog(eog);
    const int64_t n_slot = cfg_.max_seqs > 0 ? cfg_.max_seqs : 1;
    const int64_t slot_bytes = n_slot * (int64_t)sizeof(StepSlot);
    for (auto& rk : ranks_) {
        RAD_TRY(rad_dev_set(rk->device));
        rk->step_slots = static_cast<StepSlot*>(rad_dev_alloc(slot_bytes, RAD_MEM_DEVICE));
        rk->n_eog = (int32_t)eog.size();
        rk->eog = static_cast<int32_t*>(
            rad_dev_alloc(std::max<int64_t>(1, (int64_t)eog.size()) * 4, RAD_MEM_DEVICE));
        if (!rk->step_slots || !rk->eog) {
            RAD_ERR("step state: no device memory for %lld slot(s) on card %d",
                    (long long)n_slot, rk->device);
            return RAD_E_NOMEM;
        }
        RAD_TRY(rad_memset_async(rk->step_slots, 0, slot_bytes, rk->ctx.stream()));
        if (!eog.empty())
            RAD_TRY(rad_memcpy_async(rk->eog, eog.data(), (int64_t)eog.size() * 4,
                                     rk->ctx.stream()));
        RAD_TRY(rad_stream_sync(rk->ctx.stream()));
    }
    if (!ranks_.empty()) RAD_TRY(rad_dev_set(ranks_[0]->device));
    slots_host_ = static_cast<StepSlot*>(rad_dev_alloc(slot_bytes, RAD_MEM_HOST_PINNED));
    if (!slots_host_) return RAD_E_NOMEM;

    /* --draft-coupling: the head's two planes, found by name in the declared program, and the
     * pinned staging beside them. A program without them is refused rather than served
     * uncoupled. */
    mtp_couple_ = cfg_.draft_coupling;
    if (cfg_.draft_verify_traversal && !mtp_couple_) {
        RAD_ERR("--draft-verify traversal requires --draft-coupling on: the walk needs the draft's "
                "law Q, which only a coupled draft carries");
        return RAD_E_INVAL;
    }
    if (mtp_couple_) {
        /* Coupling is defined on a linear chain of drafts: a serial head's rounds, or a block
         * drafter whose walk samples each position in turn (DFlash2, dflash_couple_propose), its
         * position l standing for round l + 1. Either way one draw a drafting row and one state a
         * (round, row). */
        if ((drafter_.kind != RAD_DRAFT_SERIAL && drafter_.kind != RAD_DRAFT_BLOCK) ||
            drafter_.depth <= 0) {
            RAD_ERR("--draft-coupling on requires a linear draft chain (a serial head or a block "
                    "drafter with a coupled walk, depth > 0); this model declares drafter '%s' "
                    "kind %d depth %d, which has no per-round draw to couple",
                    drafter_.name ? drafter_.name : "(none)", drafter_.kind, drafter_.depth);
            return RAD_E_UNSUPPORTED;
        }
        const Program& prog = ranks_[0]->program;
        /* By name in each rank's own program: handles index the program that declared them
         * (advance_all), and the draws are staged on every rank. */
        couple_draws_rank_.assign(ranks_.size(), 0);
        for (size_t r = 0; r < ranks_.size(); ++r) {
            const Program& pr = ranks_[r]->program;
            for (size_t i = 0; i < pr.buffers.size(); ++i) {
                const std::string& name = pr.buffers[i].name;
                auto ends = [&](const char* suffix) {
                    const size_t n = std::strlen(suffix);
                    return name.size() >= n && name.compare(name.size() - n, n, suffix) == 0;
                };
                if (ends("mtp_couple_draws")) couple_draws_rank_[r] = (rad_buf)i;
                if (r == 0 && ends("mtp_couple_state")) couple_state_buf_ = (rad_buf)i;
            }
        }
        bool all_draws = true;
        for (rad_buf h : couple_draws_rank_) all_draws = all_draws && h;
        if (!all_draws || !couple_state_buf_) {
            RAD_ERR("--draft-coupling on, but the architecture declared no coupled MTP planes; "
                    "this model's drafter cannot couple");
            return RAD_E_STATE;
        }
        const RadBufDecl& sd = prog.buffers[couple_state_buf_].decl;
        if (sd.rank != 3 || sd.shape[2] != (int64_t)RAD_MTP_COUPLE_STATE_WORDS ||
            sd.shape[0] < (int64_t)drafter_.depth) {
            RAD_ERR("--draft-coupling: 'mtp_couple_state' is not the [rounds, rows, %d]-word "
                    "plane the coupling reads", (int)RAD_MTP_COUPLE_STATE_WORDS);
            return RAD_E_STATE;
        }
        couple_rows_ = sd.shape[1];
        couple_planes_ = sd.shape[0];
        /* One entry per card slot, indexed by StepEntry::card as slots_host_ is. */
        couple_by_card_.assign((size_t)n_slot, RadMtpCoupleDraw{});
        couple_draw_stage_ = static_cast<RadMtpCoupleDraw*>(
            rad_dev_alloc(n_slot * (int64_t)sizeof(RadMtpCoupleDraw), RAD_MEM_HOST_PINNED));
        couple_readback_ = static_cast<RadMtpCoupleState*>(rad_dev_alloc(
            (int64_t)drafter_.depth * couple_rows_ * (int64_t)sizeof(RadMtpCoupleState),
            RAD_MEM_HOST_PINNED));
        if (!couple_draw_stage_ || !couple_readback_) return RAD_E_NOMEM;
        if (drafter_.kind == RAD_DRAFT_BLOCK)
            RAD_INFO("DFlash coupled LIVE: %d position(s), x~Q along the candidate walk, maximal "
                     "coupling at the verify; fixed-seed tokens are NOT preserved", drafter_.depth);
        else
            RAD_INFO("MTP coupled LIVE: %d linear round(s), x~Q from the draft shortlist, "
                     "maximal coupling at the verify; fixed-seed tokens are NOT preserved",
                     drafter_.depth);
    }

    /* --draft-depth adaptive: the rule, the per-round readback of the chain's states and one host
     * event a round. Refused wherever the chain it stops does not exist, rather than served fixed. */
    depth_cfg_ = DraftDepthCfg{};
    if (cfg_.draft_depth_adaptive) {
        if (!mtp_couple_ || drafter_.kind != RAD_DRAFT_SERIAL) {
            RAD_ERR("--draft-depth adaptive needs a coupled serial draft chain (--draft-coupling on, a "
                    "serial head); this deployment's drafter is '%s'",
                    drafter_.name ? drafter_.name : "(none)");
            return RAD_E_INVAL;
        }
        /* --draft-window adaptive picks a window from the acceptance and the cost of the steps
         * before (sched/draft.h); this rule picks the depth inside the step from the head's laws.
         * Two controllers on one chain would fight over it. */
        if (cfg_.draft_window == DraftWindow::Adaptive) {
            RAD_ERR("--draft-depth adaptive and --draft-window adaptive both choose the chain's "
                    "depth; serve one of them");
            return RAD_E_INVAL;
        }
        depth_cfg_.on   = true;
        depth_cfg_.max  = (int)std::min<int64_t>(drafter_.depth, couple_planes_);
        /* --draft-suffix widens the verify window (and the depth the plugin declares) past the head's
         * own rounds: the chain never goes deeper than the head's num-speculative-tokens. */
        if (cfg_.n_spec_head > 0) depth_cfg_.max = std::min(depth_cfg_.max, cfg_.n_spec_head);
        if (cfg_.draft_depth_max > 0) depth_cfg_.max = std::min(depth_cfg_.max, cfg_.draft_depth_max);
        depth_cfg_.max  = std::min(depth_cfg_.max, (int)RAD_MTP_COUPLE_MAX_ROUNDS);
        depth_cfg_.min  = std::min(std::max(cfg_.draft_depth_min, 1), depth_cfg_.max);
        depth_cfg_.lag  = cfg_.draft_depth_lag;
        depth_cfg_.stat = cfg_.draft_depth_stat;
        depth_cfg_.stop = cfg_.draft_depth_stop;
        depth_cfg_.lawless = std::min(3, depth_cfg_.max);
        /* --draft-depth-load: the rule's tables (built in, or the configured lists) and the online
         * calibration that keeps them. */
        if (cfg_.draft_depth_load) {
            draft_depth_load_defaults(&depth_cfg_);
            depth_cfg_.load      = true;
            depth_load_on_       = true;
            depth_cfg_.load_from = std::max(2, cfg_.draft_depth_load_from);
            depth_cfg_.load_min  = std::min(std::max(cfg_.draft_depth_load_min, 1), depth_cfg_.max);
            depth_cfg_.late      = cfg_.draft_depth_load_late;
            depth_cfg_.late_lag  = 1;
            const int kN = DraftDepthCfg::kLoadSeqs, kJ = DraftDepthCfg::kJumps;
            /* a list by drafting rows 1 .. 8; a short one repeats its last number */
            auto by_rows = [&](const std::string& l, double* dst) {
                double v[DraftDepthCfg::kLoadSeqs];
                const int k = l.empty() ? 0 : draft_depth_parse_list(l.c_str(), v, kN, 0.0, 1000.0);
                for (int i = 0; i < kN && k > 0; ++i) dst[1 + i] = v[std::min(i, k - 1)];
                dst[0] = dst[1];
            };
            by_rows(cfg_.draft_depth_load_tok_ms, depth_cfg_.load_lam);
            by_rows(cfg_.draft_depth_load_row_ms, depth_cfg_.load_row_ms);
            by_rows(cfg_.draft_depth_load_round_ms, depth_cfg_.load_round_ms);
            if (!cfg_.draft_depth_load_jump_ms.empty()) {
                double v[DraftDepthCfg::kJumps] = {};
                const int k = draft_depth_parse_list(cfg_.draft_depth_load_jump_ms.c_str(), v, kJ, 0.0, 1000.0);
                for (int nn = 0; nn <= kN && k > 0; ++nn)
                    for (int j = 0; j < kJ; ++j) depth_cfg_.load_jump_ms[nn][j] = j < k ? v[j] : 0.0;
            }
            depth_online_on_ = cfg_.draft_depth_load_online;
            draft_depth_online_begin(depth_cfg_, &depth_online_);
            depth_pend_.assign((size_t)n_slot, DepthPend{});
            if (const char* e = std::getenv("RADIANCE_DEBUG_CONC_LOG"); e && *e) {
                depth_log_ = std::fopen(e, "w");
                if (depth_log_) std::setvbuf(depth_log_, nullptr, _IOFBF, 1 << 20);
            }
        }
        depth_rb_ = static_cast<RadMtpCoupleState*>(rad_dev_alloc(
            (int64_t)RAD_MTP_COUPLE_MAX_ROUNDS * couple_rows_ * (int64_t)sizeof(RadMtpCoupleState),
            RAD_MEM_HOST_PINNED));
        if (!depth_rb_) return RAD_E_NOMEM;
        for (RadEvent& e : depth_ev_)
            RAD_TRY(rad_event_create_as(&e, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT));
        depth_row_.assign((size_t)n_slot, DraftDepthRow{});
        depth_of_.assign((size_t)n_slot, 0);
        RAD_INFO("draft depth adaptive: 1..%d draft(s) a step, the chain goes on while the product of "
                 "the head's %s probabilities is >= %.3f (decided %d round(s) behind, %d round(s) "
                 "always run); a row with no law drafts %d",
                 depth_cfg_.max, depth_cfg_.stat == DRAFT_DEPTH_PATH ? "drawn-token" : "top",
                 depth_cfg_.stop, depth_cfg_.lag, depth_cfg_.min, depth_cfg_.lawless);
        if (depth_cfg_.load)
            RAD_INFO("draft depth load-aware: a step drafting for %d sequence(s) or more runs a round "
                     "for the rows whose expected accepted drafts pay their verify rows, the tile "
                     "they open and the round (%d round(s) always run; late choice of the verify's "
                     "rows %s); tables %s (at 8 sequences: %.3f tok/ms, a row %.3f ms, a round "
                     "%.3f ms, past %d/%d/%d/%d rows +%.2f/%.2f/%.2f/%.2f ms)",
                     depth_cfg_.load_from, depth_cfg_.load_min, depth_cfg_.late ? "on" : "off",
                     depth_online_on_ ? "calibrated online from these" : "static",
                     depth_cfg_.load_lam[8], depth_cfg_.load_row_ms[8], depth_cfg_.load_round_ms[8],
                     depth_cfg_.load_jump_at[0], depth_cfg_.load_jump_at[1], depth_cfg_.load_jump_at[2],
                     depth_cfg_.load_jump_at[3], depth_cfg_.load_jump_ms[8][0],
                     depth_cfg_.load_jump_ms[8][1], depth_cfg_.load_jump_ms[8][2],
                     depth_cfg_.load_jump_ms[8][3]);
    }

    RAD_TRY(rad_event_create_as(&step_end_, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT));
    RAD_TRY(rad_event_create_as(&tok_end_, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT));
    return RAD_OK;
}

void Engine::step_state_fini() {
    for (auto& rk : ranks_) {
        (void)rad_dev_set(rk->device);
        if (rk->step_slots) rad_dev_free(rk->step_slots, RAD_MEM_DEVICE);
        if (rk->eog) rad_dev_free(rk->eog, RAD_MEM_DEVICE);
        rk->step_slots = nullptr;
        rk->eog = nullptr;
    }
    if (slots_host_) rad_dev_free(slots_host_, RAD_MEM_HOST_PINNED);
    slots_host_ = nullptr;
    if (couple_draw_stage_) rad_dev_free(couple_draw_stage_, RAD_MEM_HOST_PINNED);
    if (couple_readback_) rad_dev_free(couple_readback_, RAD_MEM_HOST_PINNED);
    couple_draw_stage_ = nullptr;
    couple_readback_ = nullptr;
    if (depth_rb_) rad_dev_free(depth_rb_, RAD_MEM_HOST_PINNED);
    depth_rb_ = nullptr;
    for (RadEvent& e : depth_ev_) { if (e) rad_event_destroy(e); e = nullptr; }
    if (depth_log_) std::fclose(depth_log_);
    depth_log_ = nullptr;
    if (step_end_) rad_event_destroy(step_end_);
    step_end_ = nullptr;
    if (tok_end_) rad_event_destroy(tok_end_);
    tok_end_ = nullptr;
}

/* ------------------------------------------------------------------ the serial driver
 *
 * RAD_DRAFT_SERIAL: one head pass per drafted token after the trunk step (MTP, EAGLE, Medusa).
 * Round 1 consumes the sampled token and the hidden state of the position it came from, which
 * depends on how many drafts were accepted; it runs in the history pass, whose rows include that
 * position (RadBatch::draft_out_ids).
 *
 * The card, not the host, resolves those positions (sched/advance.h): behind the trunk's sampler
 * every rank runs the acceptance into its StepSlots, and an advance before each draft pass fills
 * in its positions and tokens. So the host issues the whole step (trunk, history, every round)
 * without waiting and reads back once after the last round, instead of idling the card for a
 * readback, commit and batch build per step.
 *
 * Each pass is an ordinary RadBatch through the same barrier and sampler chain; `draft_pass`
 * tells the plugin to run the head. Tokens go out through set_draft(), like prompt lookup's; a
 * sequence the head declined keeps n_draft 0, leaving it to the n-gram drafter.
 */
bool Engine::serial_drafting() const {
    return drafter_.kind == RAD_DRAFT_SERIAL && drafter_.depth > 0;
}

/* One advance on every rank against the scheduler's latest build, over its first `rows` rows
 * when given, else all. */
int Engine::advance_all(int mode, int round, bool from_proposal, int rows) {
    for (size_t i = 0; i < ranks_.size(); ++i) {
        Rank& rk = *ranks_[i];
        AdvanceArgs a = sched_.advance_args((int)i);
        if (rows >= 0 && rows < a.n_seq) a.n_seq = rows;
        a.mode  = mode;
        a.round = round;
        a.st    = rk.step_slots;
        a.eog   = rk.eog;
        a.n_eog = rk.n_eog;
        if (from_proposal) {
            /* This rank's own handle: a buffer handle indexes the program that declared it, and a
             * plugin may declare a different buffer list on each rank (DeepSeek-V4.1's drafter
             * source rows exist on its attention ranks only), so rank 0's handle can name another
             * buffer here, and an attention-zero rank would verify garbage proposals at positions
             * of its own while the other ranks verify the head's. */
            const RadDrafterDecl& dd = rk.program.drafter;
            a.src = static_cast<const int32_t*>(dd.proposal ? rk.ctx.buf_ptr(dd.proposal)
                                                            : rk.ctx.buf_ptr(rk.tok_buf));
            a.src_pitch = (dd.proposal && dd.proposal_pitch > 0) ? dd.proposal_pitch : 1;
            /* A tree drafter's parents, at the proposal's pitch (rad_drafter_tree). */
            if (tree_ && mode == ADV_BTAKE) {
                a.par = static_cast<const int32_t*>(rk.ctx.buf_ptr(tree_parents_));
                if (!a.par) {
                    RAD_ERR("drafter '%s': rank %zu has no parents buffer", drafter_.name, i);
                    return RAD_E_STATE;
                }
            }
        } else {
            a.src = static_cast<const int32_t*>(rk.ctx.buf_ptr(rk.tok_buf));
        }
        if (!a.src || !a.st) {
            RAD_ERR("drafter '%s': rank %zu has no %s to advance from", drafter_.name, i,
                    a.st ? "proposal buffer" : "step state");
            return RAD_E_STATE;
        }
        RAD_TRY(step_advance(a, rk.ctx.stream()));
    }
    return RAD_OK;
}

/* ADV_COMPACT on every rank, against the scheduler's latest build (the tree step just accepted):
 * the accepted paths' K and V into place in every paged group, and the accepted paths' conv
 * window entries and anchor replay records to the front of every recurrent group's window and
 * factor block (AdvPoolKind), so the next step's `num_accepted` names a prefix again. Each rank's
 * pools are its own. */
int Engine::tree_compact(int rows) {
    for (size_t i = 0; i < ranks_.size(); ++i) {
        Rank& rk = *ranks_[i];
        AdvanceArgs a = sched_.advance_args((int)i);
        if (rows >= 0 && rows < a.n_seq) a.n_seq = rows;
        a.mode = 0;
        a.st   = rk.step_slots;
        const Program& p = rk.program;
        for (size_t g = 1; g < p.kv_groups.size() && (int)g < a.n_groups; ++g) {
            const RadKVGroupDecl& d = p.kv_groups[g].decl;
            const int kind = d.kind;
            if (kind != RAD_KV_FULL && kind != RAD_KV_WINDOW && kind != RAD_KV_CONV &&
                kind != RAD_KV_LINEAR)
                continue;
            const KVPoolBinding* b = rk.ctx.kv_binding((uint32_t)g);
            if (!b || !b->base) continue;   /* a zero group on this rank */
            const int bits = rad_dtype_bits(b->dtype);
            int32_t layers = 0;
            for (int32_t s : b->layer_slot) layers += s >= 0;
            if (a.n_pool >= kAdvMaxPools) {
                RAD_ERR("speculation: more than %d KV groups for a tree's acceptance",
                        (int)kAdvMaxPools);
                return RAD_E_UNSUPPORTED;
            }
            AdvPool q{};
            q.base         = static_cast<char*>(b->base);
            q.layer_stride = b->layer_stride;
            q.n_layers     = layers;
            q.grp          = (int32_t)g;
            if (kind == RAD_KV_FULL || kind == RAD_KV_WINDOW) {
                if (b->rank != 4) continue;
                const int64_t row = bits > 0 ? b->shape[3] * bits / 8 : 0;
                if (row <= 0 || row % 4 || (bits != 8 && bits != 16 && bits != 32) ||
                    (b->bytes && (int64_t)layers * b->layer_stride > b->bytes)) {
                    RAD_ERR("speculation: KV group %zu's cache (%s, %lld-byte rows) is not one a "
                            "tree's acceptance can move", g, rad_dtype_name(b->dtype),
                            (long long)row);
                    return RAD_E_UNSUPPORTED;
                }
                q.kind      = ADV_POOL_PAGED;
                q.n_head    = (int32_t)b->shape[1];
                q.row_bytes = (int32_t)row;
            } else if (kind == RAD_KV_CONV) {
                /* The rolling window (docs/OPS.md, gdn_conv_update): conv_width - 2 entries of
                 * history, then one per row of the step. */
                if (b->rank != 3 || d.conv_width < 2 || (bits != 16 && bits != 32)) {
                    RAD_ERR("speculation: conv group %zu's window (%s, width %lld) is not one a "
                            "tree's acceptance can move", g, rad_dtype_name(b->dtype),
                            (long long)d.conv_width);
                    return RAD_E_UNSUPPORTED;
                }
                q.kind      = ADV_POOL_CONV;
                q.row_bytes = bits / 8;
                q.hist      = (int32_t)(d.conv_width - 2);
                q.n_chan    = (int32_t)b->shape[1];
                q.n_ent     = (int32_t)b->shape[2];
                q.n_head    = (int32_t)((b->shape[1] + kAdvConvChunk - 1) / kAdvConvChunk);
            } else {
                /* A recurrent state under the anchor form (two copies: the state and the replay
                 * records, abi/rad_gdn_anchor.h). A group kept per candidate has no records. */
                const KVGroupPlan* pl = rk.kv.plan((int32_t)g);
                if (!pl || pl->state_copies != 2) continue;
                if (b->rank != 4 || (bits != 16 && bits != 32)) {
                    RAD_ERR("speculation: state group %zu (%s, rank %u) is not one a tree's "
                            "acceptance can move", g, rad_dtype_name(b->dtype), b->rank);
                    return RAD_E_UNSUPPORTED;
                }
                const int64_t H = b->shape[1], V = b->shape[2], K = b->shape[3];
                q.kind       = ADV_POOL_ANCHOR;
                q.row_bytes  = bits / 8;
                q.n_head     = (int32_t)H;
                q.copy_off   = (int32_t)pl->n_seq_states();
                q.rec_elems  = (int32_t)rad_gdn_anchor_fstr(K, V);
                q.head_elems = V * K;
                q.slot_elems = H * V * K;
            }
            a.pool[a.n_pool++] = q;
        }
        RAD_TRY(step_compact(a, rk.ctx.stream()));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ --draft-depth adaptive
 *
 * The chain's depth is decided while it is drafted (sched/draft_depth.h), so the rounds cannot all
 * be queued behind the step as a fixed chain's are: round j is issued once the statistic of round
 * j - 1 - lag is known. That costs the host one tiny readback a round (the round's state rows of
 * the drafting sequences, rank 0's copy; every rank holds the same) and a wait for it. With lag 1
 * the card is running round j - 1 meanwhile, so nothing idles as long as the host answers within a
 * round (about 0.5 ms; reading the row and issuing a recorded pass takes about 0.1 ms). The next
 * verify is issued the same way, right after the chain's last decision, with exactly the rows the
 * chain has (Engine::issue_ahead): each row count is its own recorded pass.
 *
 * The commit of the step in flight moves behind those decisions (the step loop collects after
 * issue_ahead), so its tokens are streamed up to the chain's rounds later than with a fixed chain;
 * the card is not held up by that. */
int Engine::depth_row_nodes(int i) const {
    if (!depth_cfg_.on) return serial_done_;
    if (i < 0 || (size_t)i >= depth_of_.size()) return 0;
    return depth_of_[(size_t)i];
}

/* Behind the pass of round `round`: its state rows of the `n` drafting sequences to the host, and
 * the event the driver waits on. */
int Engine::depth_readback_issue(int round, int n) {
    if (round < 1 || round > (int)RAD_MTP_COUPLE_MAX_ROUNDS || round > (int)couple_planes_ || n <= 0)
        return RAD_OK;
    Rank& r0 = *ranks_[0];
    const char* state = static_cast<const char*>(r0.ctx.buf_ptr(couple_state_buf_));
    if (!state) return RAD_E_STATE;
    const int64_t plane = couple_rows_ * (int64_t)sizeof(RadMtpCoupleState);
    RAD_TRY(rad_memcpy_async(depth_rb_ + (int64_t)(round - 1) * couple_rows_,
                             state + (int64_t)(round - 1) * plane,
                             (int64_t)n * (int64_t)sizeof(RadMtpCoupleState), r0.ctx.stream()));
    RAD_TRY(rad_event_record(depth_ev_[round - 1], r0.ctx.stream()));
    depth_issued_[round - 1] = true;
    return RAD_OK;
}

/* Waits for the statistics of every round up to `round` and feeds them to the rows. A round whose
 * readback was never issued (its pass did not run) leaves the rows without a law. */
int Engine::depth_observe_upto(int round, int n, const StepOut& so) {
    while (depth_seen_ < round) {
        const int r = depth_seen_ + 1;
        const bool have = r <= (int)RAD_MTP_COUPLE_MAX_ROUNDS && depth_issued_[r - 1];
        if (have) {
            RadEvent ev = depth_ev_[r - 1];
            const auto t0 = std::chrono::steady_clock::now();
            int q = rad_event_query(ev);
            if (q < 0) return q;
            if (q == 0) {
                /* The ids' copy sits ahead of every draft pass: block for it (the trunk's whole
                 * forward pass may be in front), then poll, which answers within a microsecond of
                 * the card where an interrupt takes tens. */
                if (so.early && depth_seen_ == 0) RAD_TRY(rad_event_sync(tok_end_));
                const auto t1 = std::chrono::steady_clock::now();
                bool blocked = false;
                for (uint32_t spin = 0;; ++spin) {
                    q = rad_event_query(ev);
                    if (q != 0) break;
                    if ((spin & 255u) == 255u &&
                        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1)
                                .count() > (double)kDepthSpinUs) {
                        blocked = true;
                        break;
                    }
#if defined(__x86_64__) || defined(__i386__)
                    __builtin_ia32_pause();
#endif
                }
                if (q < 0) return q;
                if (blocked) ++depth_stats_.blocked;
                ++depth_stats_.waits;
            } else {
                /* Already there: the host was behind the card by the time it asked. */
                ++depth_stats_.late;
            }
            /* The wait proper: what a host wait does to the fences of the dispatches that follow
             * (the copy wrote host memory), and the block when the poll gave up. */
            RAD_TRY(rad_event_sync(ev));
            const auto t2 = std::chrono::steady_clock::now();
            depth_stats_.wait_us += std::chrono::duration<double, std::micro>(t2 - t0).count();
            /* The card's cycle by the last step's depth: the host waits for every step's first
             * statistic, so two of them in a row are one cycle apart. A gap of 30 ms or more is
             * no decode cycle (an idle engine, or the next request's prefill, in between). */
            if (r == 1) {
                const double us = std::chrono::duration<double, std::micro>(t2 - depth_e1_).count();
                if (q == 1 && depth_last_ >= 1 && depth_last_ <= (int)RAD_MTP_COUPLE_MAX_ROUNDS &&
                    us > 0 && us < 30000.0) {
                    depth_stats_.cyc_us[depth_last_] += us;
                    ++depth_stats_.cyc_n[depth_last_];
                }
                depth_e1_ = t2;
            }
        }
        for (int i = 0; i < n; ++i) {
            DraftDepthRow& row = depth_row_[(size_t)i];
            double st = -1.0;
            if (have && !row.lawless) {
                const RadMtpCoupleState& cs = depth_rb_[(int64_t)(r - 1) * couple_rows_ + i];
                /* this step's draw for this row, or a stale plane: no law */
                if (!std::memcmp(&cs.draw, &couple_draw_stage_[(size_t)i], sizeof cs.draw))
                    st = draft_depth_stat(cs, r, depth_cfg_.stat);
            }
            draft_depth_observe(&row, r, st);
        }
        depth_seen_ = r;
    }
    return RAD_OK;
}

/* The adaptive chain's tuning line: the shares of each depth and what the host waits cost, then
 * by the drafts a step made, the card's cycle and the drafts its verify accepted. Logged every
 * 4096 drafting rows and at shutdown. */
void Engine::depth_log_stats(const char* when) {
    const DepthStats& ds = depth_stats_;
    if (!depth_cfg_.on || ds.rows == 0 || (depth_logged_ == ds.rows && !std::strcmp(when, "final"))) return;
    depth_logged_ = ds.rows;
    std::string h;
    for (int d2 = 1; d2 <= depth_cfg_.max; ++d2)
        h += fmt("%s%d:%.3f", d2 > 1 ? " " : "", d2, (double)ds.hist[d2] / (double)ds.rows);
    RAD_INFO("draft depth adaptive (%s): %llu drafting step(s), drafts/row %.3f, rounds/step %.3f, "
             "depth shares [%s], lawless rows %llu; host waits %llu (%.1f us each, %llu ended "
             "blocked), statistics already there %llu",
             when, (unsigned long long)ds.steps, (double)ds.drafts / (double)ds.rows,
             (double)ds.rounds / (double)ds.steps, h.c_str(), (unsigned long long)ds.lawless,
             (unsigned long long)ds.waits,
             ds.waits + ds.late ? ds.wait_us / (double)(ds.waits + ds.late) : 0.0,
             (unsigned long long)ds.blocked, (unsigned long long)ds.late);
    std::string by;
    for (int d2 = 1; d2 <= depth_cfg_.max; ++d2)
        by += fmt("%s%d: %.0f us (%llu), %.3f accepted (%llu)", d2 > 1 ? "; " : "", d2,
                 ds.cyc_n[d2] ? ds.cyc_us[d2] / (double)ds.cyc_n[d2] : 0.0,
                 (unsigned long long)ds.cyc_n[d2],
                 ds.acc_n[d2] ? (double)ds.acc[d2] / (double)ds.acc_n[d2] : 0.0,
                 (unsigned long long)ds.acc_n[d2]);
    RAD_INFO("draft depth adaptive (%s): by drafts made, the card's cycle and the drafts the "
             "verify accepted -- %s", when, by.c_str());
    if (depth_load_on_) depth_load_log(when);
}

/* --draft-depth-load, once a step, after its draft passes ended and its verify was committed:
 * the step's milliseconds (since the step before it ended, when both were pure decode steps of the
 * same drafting rows), its verify rows, rounds and tokens for the rule's rate and prices; every 64
 * steps the tables again from what was measured. Between steps: the next step's chain is decided
 * after this returns, from tables that nothing of that step went into (sched/draft_depth.h). */
void Engine::depth_step_note(const RadBatch* b, const StepOut& so) {
    const auto now = std::chrono::steady_clock::now();
    const StepPlan& plan = sched_.plan();
    bool pure = b->phase == RAD_PHASE_DECODE;
    int tokens = 0;
    for (size_t i = 0; i < plan.e.size(); ++i) {
        if (plan.e[i].is_prefill) pure = false;
        if (i < so.first_row.size() && so.first_row[i] >= 0)
            tokens += 1 + std::max(0, i < so.accepted.size() ? so.accepted[i] : 0);
    }
    const int n = serial_rows_;
    if (depth_online_on_ && pure && n >= 1 && serial_done_ >= 1 && tokens > 0 &&
        depth_note_step_ == b->step - 1 && depth_note_rows_ == n) {
        const double ms = std::chrono::duration<double, std::milli>(now - depth_note_t_).count();
        draft_depth_online_step(depth_cfg_, &depth_online_, n, (int)b->n_tok, serial_done_, ms, tokens);
    }
    depth_note_t_ = now;
    depth_note_step_ = pure ? b->step : -2;
    depth_note_rows_ = n;
    if (depth_log_) depth_log_step(b, so);
    if (depth_online_on_ && (++depth_note_n_ & 63u) == 0) {
        draft_depth_online_refresh(&depth_cfg_, &depth_online_);
        if ((depth_note_n_ & 32767u) == 0) depth_load_log("running");
        if (depth_log_) {
            /* K: the tables after a refresh, at the step's drafting rows */
            const int nn = draft_depth_seqs(std::max(n, 1));
            std::string k = fmt("K %d %d %.4f %.4f %.4f", b->step, nn, depth_cfg_.load_lam[nn],
                                depth_cfg_.load_row_ms[nn], depth_cfg_.load_round_ms[nn]);
            for (int j = 0; j < DraftDepthCfg::kJumps; ++j) k += fmt(" %.3f", depth_cfg_.load_jump_ms[nn][j]);
            for (int t = 0; t < 3; ++t)
                for (int j = 2; j <= depth_cfg_.max && j <= DraftDepthCfg::kCalRounds; ++j) {
                    k += fmt(" | %d,%d", t, j);
                    for (int x = 0; x < draft_depth_knots(depth_cfg_); ++x)
                        k += fmt(" %.3f", depth_cfg_.cal_p[t][j][x]);
                }
            std::fprintf(depth_log_, "%s\n", k.c_str());
        }
    }
}

/* What the load-aware rule's tables hold now, by drafting rows that were measured. */
void Engine::depth_load_log(const char* when) const {
    const DraftDepthOnline& o = depth_online_;
    std::string t;
    for (int n = 1; n <= DraftDepthCfg::kLoadSeqs; ++n) {
        if (o.reg[n].n < 64 && o.lam[n].n < 32) continue;
        t += fmt("%s%d seq: %.3f tok/ms, row %.3f ms, round %.3f ms, tiles +%.2f/%.2f/%.2f/%.2f ms "
                 "(%lld steps)", t.empty() ? "" : "; ", n, depth_cfg_.load_lam[n],
                 depth_cfg_.load_row_ms[n], depth_cfg_.load_round_ms[n], depth_cfg_.load_jump_ms[n][0],
                 depth_cfg_.load_jump_ms[n][1], depth_cfg_.load_jump_ms[n][2],
                 depth_cfg_.load_jump_ms[n][3], (long long)o.reg[n].n);
    }
    std::string c;
    for (int j = 2; j <= depth_cfg_.max && j <= DraftDepthCfg::kCalRounds; ++j) {
        c += fmt("%sdraft %d:", j > 2 ? "; " : "", j);
        for (int x = 0; x < draft_depth_knots(depth_cfg_); ++x) c += fmt(" %.2f", depth_cfg_.cal_p[2][j][x]);
    }
    RAD_INFO("draft depth load-aware (%s): %s calibration, %lld chain(s) and %lld step(s) counted "
             "(%lld stalled step(s) left out), %lld refresh(es); %s; acceptance by the confidence "
             "that includes the draft's own law, at %.2f .. %.2f -- %s", when,
             depth_online_on_ ? "online" : "static", (long long)o.chains, (long long)o.steps,
             (long long)o.outliers, (long long)o.refreshes, t.empty() ? "no step count measured yet" : t.c_str(),
             depth_cfg_.cal_x[0], depth_cfg_.cal_x[draft_depth_knots(depth_cfg_) - 1], c.c_str());
}

/* RADIANCE_DEBUG_CONC_LOG: the step just finished, one line.
 *   S step ns phase n_seq n_tok rounds drafting_rows load_step | entry;... | chain;...
 * An entry is a sequence of the step's verify: seq,is_prefill,n_tokens,drafts verified,drafts
 * accepted (-1: no token row),ctx_len. A chain is what a drafting row made behind the step, for
 * the next verify: seq,depth,the rule's round,rounds that ran for it,lawless, then each of those
 * rounds' top mass. Called after finish_serial_drafter, while plan() still describes the step. */
void Engine::depth_log_step(const RadBatch* b, const StepOut& so) {
    const StepPlan& plan = sched_.plan();
    std::string e, d;
    for (size_t i = 0; i < plan.e.size(); ++i) {
        const StepEntry& x = plan.e[i];
        e += fmt("%s%llu,%d,%d,%d,%d,%d", i ? ";" : "", (unsigned long long)x.seq, (int)x.is_prefill,
                 (int)x.n_tokens, (int)x.n_spec,
                 i < so.accepted.size() && so.first_row[i] >= 0 ? (int)so.accepted[i] : -1,
                 (int)x.ctx_len);
    }
    for (int i = 0; i < serial_rows_ && serial_done_ > 0; ++i) {
        const DraftDepthRow& row = depth_row_[(size_t)i];
        const int ran = serial_done_;
        d += fmt("%s%llu,%d,%d,%d,%d", i ? ";" : "", (unsigned long long)serial_row_[(size_t)i].seq,
                 depth_row_nodes(i), depth_load_step_ ? row.rule : 0, ran, (int)row.lawless);
        if (mtp_couple_ && couple_readback_)
            for (int r = 1; r <= ran && r <= (int)couple_planes_; ++r)
                d += fmt(",%.4f", draft_depth_stat(couple_readback_[(int64_t)(r - 1) * couple_rows_ + i],
                                                    r, DRAFT_DEPTH_TOP));
    }
    const uint64_t ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch()).count();
    std::fprintf(depth_log_, "S %d %llu %d %lld %lld %d %d %d | %s | %s\n", b->step,
                 (unsigned long long)ns, (int)b->phase, (long long)b->n_seq, (long long)b->n_tok,
                 serial_done_, serial_rows_, (int)depth_load_step_, e.c_str(), d.c_str());
}

/* Issue the step's draft passes behind it, without waiting. After the trunk's ranks join: the
 * acceptance on every rank, the sampled-id readback (before a round can overwrite them), the
 * history pass and every round, then the final proposal into the slots and their readback.
 * Nothing synchronises here: collect_step waits for the ids alone, and the step loop for the
 * passes after the commit, before finish_serial_drafter reads the slots. */
int Engine::issue_serial_drafter(int step, StepOut& so) {
    serial_rounds_ = 0;
    serial_rows_   = 0;
    serial_done_   = 0;
    const bool history_only = drafter_.kind == RAD_DRAFT_HISTORY;
    /* A step issued ahead continues from the card's acceptance, so run it even without drafts. */
    if (!serial_drafting() && !history_only) {
        if (pipelining_) RAD_TRY(advance_all(ADV_ACCEPT, 0, false));
        return RAD_OK;
    }

    RAD_TRY(advance_all(ADV_ACCEPT, 0, false));
    RAD_TRY(collect_issue(so, /*fence=*/true));

    /* serial_begin needs a pass budget even for history alone. The scheduler's speculative
     * depth is still zero there, so it stages history rows but never reserves a proposal row. */
    int rounds = history_only ? 1 : drafter_.depth;
    /* --draft-depth-max below the head's declared depth: no round past it is ever asked for. */
    if (depth_cfg_.on && rounds > depth_cfg_.max) rounds = depth_cfg_.max;
    serial_row_.resize((size_t)cfg_.max_seqs);
    /* `n` counts rows that can draft. It can be zero while the history pass still has work (a
     * mid-prompt sequence has hidden states the head needs but no token to propose from). */
    const int n = sched_.serial_begin(&rounds, serial_row_.data(), (int)serial_row_.size());
    if (rounds <= 0) return RAD_OK;
    serial_rounds_ = rounds;
    serial_rows_   = n;
    /* Before any serial pass reads a draw; round 1 is the history pass below. */
    if (mtp_couple_ && n > 0) RAD_TRY(stage_couple_draws(serial_row_.data(), n));

    /* RADIANCE_DEBUG_STAGE: running means of a draft pass's host time: batch + advance, greedy
     * sampler chain, barrier release and issue. They cost the step nothing until their sum exceeds
     * the previous pass's device time. */
    static const bool stg = std::getenv("RADIANCE_DEBUG_STAGE") != nullptr;
    static double a_batch = 0, a_samp = 0, a_rel = 0, a_iss = 0;
    static long   n_pass = 0;

    /* --draft-depth adaptive: every row starts a chain; a greedy row (no draw) has no law. */
    const bool adepth = depth_cfg_.on && n > 0;
    depth_load_step_ = false;
    if (adepth) {
        depth_seen_ = 0;
        for (bool& b : depth_issued_) b = false;
        for (int i = 0; i < n; ++i) {
            draft_depth_begin(depth_cfg_, &depth_row_[(size_t)i], couple_draw_stage_[(size_t)i].mode != 1);
            depth_of_[(size_t)i] = 0;
        }
        /* --draft-depth-load: the next verify's rows beside the drafting sequences' (the plan's
         * other token rows), and whether this step has the sequences for the load-aware rule. */
        depth_other_rows_ = 0;
        for (const StepEntry& pe : sched_.plan().e) depth_other_rows_ += pe.produces_token;
        depth_other_rows_ = std::max(0, depth_other_rows_ - n);
        depth_load_step_ = depth_cfg_.load && n >= depth_cfg_.load_from;
    }

    /* Pass 0, then rounds 2..rounds: round 1 has no pass of its own. */
    for (int rd = 0; rd <= rounds; rd += rd == 0 ? 2 : 1) {
        if (rd > 0 && n <= 0) break;
        /* The adaptive chain: round rd runs only for a row whose confidence up to round
         * rd - 1 - lag holds (sched/draft_depth.h); the pass runs while any row goes on. */
        if (adepth && rd >= 2) {
            const int need = draft_depth_need(depth_cfg_, rd);
            if (need >= 1) RAD_TRY(depth_observe_upto(need, n, so));
            bool any = false;
            if (depth_load_step_)
                any = draft_depth_go_load(depth_cfg_, depth_row_.data(), n, rd, depth_other_rows_);
            else
                for (int i = 0; i < n; ++i)
                    any = draft_depth_go(depth_cfg_, &depth_row_[(size_t)i], rd) || any;
            if (!any) break;
        }
        const auto tA = std::chrono::steady_clock::now();
        const RadBatch* db = sched_.serial_batch(rd, step);
        if (!db) {
            if (history_only) {
                RAD_ERR("'%s' calibration: missing history batch for step %d", drafter_.name, step);
                return RAD_E_STATE;
            }
            if (rd == 0) continue;
            break;
        }
        /* Pass 0 is the history pass, writing the head's own K and V (nothing else does). It is
         * also draft round 1 for drafting rows, whose inputs are the last row of their history run
         * (RadBatch::draft_out_ids), so it ends in the lm_head over those rows. Round r >= 2 embeds
         * round r-1's proposal, which the advance takes from the proposal buffer on the card. */
        const int drafted = rd == 0 ? (int)db->n_draft_out : n;
        if (history_only && (n != 0 || drafted != 0 || rd != 0)) {
            RAD_ERR("'%s' history-only calibration cannot reserve draft proposals", drafter_.name);
            return RAD_E_STATE;
        }
        RAD_TRY(advance_all(rd == 0 ? ADV_HIST : ADV_ROUND, rd, rd > 0));
        const auto tB = std::chrono::steady_clock::now();

        /* Greedy, not the request's sampler: the target verifies against its own draw
         * (core/sample/accept.h), so sampling a draft would spend the request's RNG stream and
         * lower acceptance. A head with its own proposal buffer samples nothing (a zero-row
         * chain). */
        for (auto& rk : ranks_) {
            int s = rk->sampler.build_greedy(drafter_.proposal ? 0 : drafted);
            if (s < 0) {
                RAD_ERR("drafter '%s': build_greedy: %s", drafter_.name, rad_strerror(s));
                return s;
            }
        }

        const auto tC = std::chrono::steady_clock::now();
        issue_what_.store(kIssueAll, std::memory_order_release);
        step_batch_.store(db, std::memory_order_release);
        /* Timed separately: the first crossing releases the parked ranks (the wake), the second
         * waits for them to finish issuing (the issue). */
        barrier_->arrive_and_wait();
        const auto tC2 = std::chrono::steady_clock::now();
        barrier_->arrive_and_wait();
        const auto tD = std::chrono::steady_clock::now();

        int st = step_status_.load(std::memory_order_acquire);
        if (st < 0) {
            RAD_ERR("drafter '%s': pass %d of step %d failed: %s -- the engine cannot continue",
                    drafter_.name, rd, step, rad_strerror(st));
            return st;
        }
        if (stg) {
            using us = std::chrono::duration<double, std::micro>;
            a_batch += us(tB - tA).count();
            a_samp  += us(tC - tB).count();
            a_rel   += us(tC2 - tC).count();
            a_iss   += us(tD - tC2).count();
            if (++n_pass % 400 == 0)
                RAD_INFO("draft: %ld passes; batch %.1f, sampler %.1f, release %.1f, issue %.1f us "
                         "a pass", n_pass, a_batch / (double)n_pass, a_samp / (double)n_pass,
                         a_rel / (double)n_pass, a_iss / (double)n_pass);
        }
        if (rd > 0) serial_done_ = rd;
        else if (drafted > 0) serial_done_ = 1;
        /* This round's states to the host, when a later round's decision reads them. */
        /* The late choice reads every round but the last. */
        if (adepth && serial_done_ == (rd > 0 ? rd : 1) &&
            (serial_done_ <= draft_depth_need(depth_cfg_, rounds) ||
             (depth_load_step_ && depth_cfg_.late)))
            RAD_TRY(depth_readback_issue(serial_done_, n));
    }
    if (serial_done_ <= 0 || n <= 0) return RAD_OK;
    /* The late choice (--draft-depth-load-late): every row's verified prefix again, from the
     * rounds read by now and the wait for all but the last (it is still running, so the card is
     * not held up as long as the wait and the next step's issue fit in it). */
    if (adepth && depth_load_step_ && depth_cfg_.late && serial_done_ >= 2) {
        const int upto = std::max(depth_seen_, serial_done_ - std::max(0, depth_cfg_.late_lag));
        if (upto > depth_seen_) RAD_TRY(depth_observe_upto(upto, n, so));
        draft_depth_select(depth_cfg_, depth_row_.data(), n, serial_done_, depth_other_rows_);
    }
    if (adepth) {
        DepthStats& ds = depth_stats_;
        ++ds.steps;
        ds.rounds += (uint64_t)serial_done_;
        for (int i = 0; i < n; ++i) {
            const DraftDepthRow& row = depth_row_[(size_t)i];
            const int d = std::max(1, std::min(row.depth, serial_done_));
            depth_of_[(size_t)i] = d;
            ++ds.rows;
            ds.drafts += (uint64_t)d;
            ds.lawless += row.lawless;
            ++ds.hist[std::min(d, (int)RAD_MTP_COUPLE_MAX_ROUNDS)];
        }
        depth_last_ = 0;
        for (int i = 0; i < n; ++i) depth_last_ = std::max(depth_last_, (int)depth_of_[(size_t)i]);
        if (ds.rows >= depth_logged_ + 4096) depth_log_stats("running");
    } else {
        depth_last_ = 0;
    }

    /* The last round's proposal into the slots, then rank 0's slots back to the host. Only the
     * drafting rows: at depth 1 the last pass is the history pass, whose later rows only carried
     * history (Scheduler::serial_batch). */
    RAD_TRY(advance_all(ADV_TAKE, serial_done_, true, n));
    Rank& r0 = *ranks_[0];
    if (mtp_couple_) {
        /* Rounds 1..done of the drafting rows, on the same stream as the slots below. */
        const void* state = r0.ctx.buf_ptr(couple_state_buf_);
        if (!state || serial_done_ > (int)drafter_.depth) return RAD_E_STATE;
        const int64_t pitch = couple_rows_ * (int64_t)sizeof(RadMtpCoupleState);
        RAD_TRY(rad_memcpy_2d_async(couple_readback_, pitch, state, pitch,
                                    (int64_t)n * (int64_t)sizeof(RadMtpCoupleState), serial_done_,
                                    r0.ctx.stream()));
    }
    RAD_TRY(rad_memcpy_async(slots_host_, r0.step_slots,
                             (int64_t)cfg_.max_seqs * (int64_t)sizeof(StepSlot), r0.ctx.stream()));
    return RAD_OK;
}

/* After commit: check the host's acceptance against the card's, publish the proposals through
 * set_draft, and move the history watermark. The card's acceptance positioned every draft pass,
 * so a disagreement is a defect and the step is refused. */
int Engine::finish_serial_drafter(const StepOut& so) {
    if (serial_rounds_ <= 0) return RAD_OK;
    if (drafter_.kind == RAD_DRAFT_HISTORY) {
        sched_.serial_end(0);  /* advance the history watermark, not the draft chain */
        return RAD_OK;
    }
    if (!serial_drafting()) return RAD_OK;
    const StepPlan& plan = sched_.plan();
    if (serial_done_ > 0 && serial_rows_ > 0) {
        for (size_t i = 0; i < plan.e.size(); ++i) {
            const StepEntry& e = plan.e[i];
            if (so.first_row[i] < 0) continue;
            const StepSlot& S = slots_host_[e.card];   /* in range: the builder refuses others */
            if (S.acc != so.accepted[i] || S.tok[0] != so.token[i]) {
                RAD_ERR("drafter '%s': the card accepted %d draft(s) ending in token %d for "
                        "sequence %llu and the host %d ending in %d; the draft passes ran from "
                        "the card's", drafter_.name, S.acc, S.tok[0],
                        (unsigned long long)e.seq, so.accepted[i], so.token[i]);
                return RAD_E_STATE;
            }
        }
    }
    /* --draft-depth adaptive: what the verify of d drafts accepted, by d (the tuning line). */
    if (depth_cfg_.on) {
        for (size_t i = 0; i < plan.e.size(); ++i) {
            const StepEntry& e = plan.e[i];
            if (so.first_row[i] < 0 || e.n_spec <= 0 || e.n_spec > (int)RAD_MTP_COUPLE_MAX_ROUNDS ||
                e.suffix || e.card < 0 || (size_t)e.card >= couple_by_card_.size())
                continue;
            const RadMtpCoupleDraw& d = couple_by_card_[(size_t)e.card];
            if (d.seq != e.seq || d.mode != 1) continue;
            /* --draft-depth-load: the chain this verify settled, with the confidences its
             * decisions read, for the online calibration. */
            if (depth_online_on_ && (size_t)e.card < depth_pend_.size()) {
                const DepthPend& pd = depth_pend_[(size_t)e.card];
                if (pd.depth > 0 && pd.seq == e.seq && pd.depth == e.n_spec)
                    draft_depth_online_chain(depth_cfg_, &depth_online_, pd.depth, pd.rule,
                                             std::max(0, so.accepted[i]), pd.s, pd.n_s);
            }
            depth_stats_.acc[e.n_spec] += (uint64_t)std::max(0, so.accepted[i]);
            ++depth_stats_.acc_n[e.n_spec];
        }
    }
    sched_.serial_end(serial_done_);
    for (DepthPend& pd : depth_pend_) pd.depth = 0;     /* settled above, or never verified */
    if (serial_done_ <= 0 || serial_rows_ <= 0) return RAD_OK;

    Rank& r0 = *ranks_[0];
    const int rounds = serial_rounds_, done = serial_done_;
    draft_out_.assign((size_t)serial_rows_ * (size_t)rounds, 0);
    for (int i = 0; i < serial_rows_; ++i) {
        const StepSlot& S = slots_host_[serial_row_[(size_t)i].card];
        for (int rd = 1; rd <= done; ++rd)
            draft_out_[(size_t)i * (size_t)rounds + (size_t)(rd - 1)] = S.tok[rd];
    }
    for (int i = 0; i < serial_rows_; ++i) {
        /* No draft for a request the sampler will not speculate on (a constraining grammar that
         * cannot snapshot its state); the sampler would refuse the next step otherwise. */
        if (!r0.sampler.allows_speculation(serial_row_[(size_t)i].seq)) continue;
        const uint64_t seq = serial_row_[(size_t)i].seq;
        /* Under --draft-depth adaptive the row's own chain: the drafts the step issued ahead was
         * planned with (issue_ahead's expect_draft), a prefix of the rounds that ran. */
        const int row_nodes = depth_cfg_.on ? depth_row_nodes(i) : done;
        const int rc = sched_.set_draft(seq, &draft_out_[(size_t)i * (size_t)rounds], row_nodes);
        /* --draft-depth-load: the chain as the next verify will settle it -- its depth, the round
         * the rule sent it to, and the statistics of the rounds that ran for it. */
        if (depth_online_on_ && rc == RAD_OK && mtp_couple_ && couple_readback_ && row_nodes >= 1 &&
            !depth_row_[(size_t)i].lawless) {
            const int32_t card = serial_row_[(size_t)i].card;
            if (card >= 0 && (size_t)card < depth_pend_.size()) {
                DepthPend& pd = depth_pend_[(size_t)card];
                pd.seq = seq;
                pd.depth = row_nodes;
                pd.rule = depth_load_step_ ? depth_row_[(size_t)i].rule : 0;
                pd.n_s = 0;
                const int ran = std::min(done, (int)std::min<int64_t>(RAD_MTP_COUPLE_MAX_ROUNDS, couple_planes_));
                for (int r = 1; r <= ran; ++r) {
                    const RadMtpCoupleState& cs = couple_readback_[(int64_t)(r - 1) * couple_rows_ + i];
                    if (std::memcmp(&cs.draw, &couple_draw_stage_[(size_t)i], sizeof cs.draw)) break;
                    const double st = draft_depth_stat(cs, r, DRAFT_DEPTH_TOP);
                    if (st < 0.0) break;
                    pd.s[r - 1] = (float)st;
                    pd.n_s = r;
                }
                if (pd.n_s < row_nodes) pd.depth = 0;   /* a round with no law: not counted */
            }
        }
        if (mtp_couple_)
            RAD_TRY(bind_couple(seq, i, row_nodes, rc == RAD_OK, serial_row_[(size_t)i].card));
    }
    return RAD_OK;
}

/* --draft-coupling, after the commit: bind drafting row `i`'s `done` states (its readback column)
 * to the positions the next build samples, and hand them to every rank's sampler.
 *
 * Bound only for a request still running: one this commit finished, or whose draft the scheduler
 * refused (`drafted` false), binds nothing, and set_mtp_couple(..., 0) clears an earlier step's
 * states (RAD_E_NOTFOUND for a request the sampler already dropped).
 *
 * The position is the committed output length after this commit, the n_out the next build_step
 * reads. It comes from the scheduler's commit snapshot, not the plan's Request pointer, which
 * reap() may already have freed (sched/mtp_couple_bind.h). */
int Engine::bind_couple(uint64_t seq, int i, int done, bool drafted, int32_t card) {
    MtpCoupleBindPoint bp;
    bp.found = sched_.committed_seq(seq, &bp.n_out, &bp.live);
    if (!bp.found)
        RAD_WARN("coupled draft: row for sequence %llu is not in the committed step; clearing its "
                 "coupling", (unsigned long long)seq);
    int n_bound = 0, n_malformed = 0;
    std::string why;
    const int bs = mtp_couple_bind_rounds(bp, couple_draw_stage_[(size_t)i],
                                          &couple_readback_[(size_t)i], (int)couple_rows_, done,
                                          slots_host_[card], meta_.n_vocab, couple_bound_,
                                          &n_bound, &why, &n_malformed);
    if (bs < 0) { RAD_ERR("%s", why.c_str()); return bs; }
    if (n_malformed > 0 && !couple_malformed_note_) {
        couple_malformed_note_ = true;
        RAD_WARN("coupled draft: sequence %llu: %d draft position(s) had a shortlist with no valid "
                 "Q (non-finite score, duplicate id or no live entry); those keep the greedy "
                 "proposal and the ordinary verify (logged once)",
                 (unsigned long long)seq, n_malformed);
    }
    for (auto& rk : ranks_) {
        const int st = rk->sampler.set_mtp_couple(seq, couple_bound_, drafted ? n_bound : 0);
        if (st < 0 && st != RAD_E_NOTFOUND) return st;
    }
    return RAD_OK;
}

/* One line, once, at the first coupled verify that completed. It reads the snapshot build_step
 * froze for the step whose wait just returned (sample/mtp_couple_evidence.h), which the
 * drafter's build_greedy does not clear, and checks that run() issued the verify for that build.
 * Sequences are re-checked as Running after the wait. Request numbers, the row count and the
 * kernel name only; a host-domain row is labelled as such. */
void Engine::couple_verify_evidence(int step) {
    if (!mtp_couple_ || couple_evidence_) return;
    Sampler& smp = ranks_[0]->sampler;
    const rad_op op = smp.staged_couple_op();
    const MtpCoupleEvidence ev = mtp_couple_evidence(
        smp.couple_build(), step, ranks_[0]->ctx.op_domain(op),
        [this](uint64_t seq) { return sched_.running(seq); });
    if (ev.rows <= 0 || smp.staged_couple_issued_step() != step) return;
    const char* kernel = ranks_[0]->ctx.ip_kernel_name(op);
    if (ev.device) {
        couple_evidence_ = true;
        RAD_INFO("MTP coupled verify completed: step %d, %lld bound row(s) over request(s) [%s], "
                 "kernel %s, domain device", step, (long long)ev.rows, ev.seqs.c_str(), kernel);
    } else if (!couple_host_note_) {
        couple_host_note_ = true;
        RAD_INFO("MTP coupled verify completed on the host domain: step %d, %lld bound row(s) over "
                 "request(s) [%s], kernel %s -- not device evidence", step, (long long)ev.rows,
                 ev.seqs.c_str(), kernel);
    }
}

/* --draft-coupling: the draw for every drafting row, before any draft pass reads it.
 *
 * The coordinate is the pre-commit output size, which strictly grows between two steps that draft
 * one request, so (base, round) never repeats a draft stream. The card is the slot the advance
 * writes; the propose op keeps the greedy proposal for a row whose draw names another card. The
 * rule is sched/mtp_couple_plan.h. */
template <class Row>
int Engine::stage_couple_draws(const Row* rows, int n) {
    std::string why;
    const int stg = mtp_couple_plan_stage(sched_.plan(),
                                          (int)(cfg_.max_seqs > 0 ? cfg_.max_seqs : 1),
                                          rows, n, &couple_by_card_, couple_draw_stage_, &why);
    if (stg < 0) {
        RAD_ERR("MTP coupled: %s", why.c_str());
        return stg;
    }
    for (auto& rk : ranks_) {
        void* dst = rk->ctx.buf_ptr(couple_draws_rank_[(size_t)rk->index]);
        if (!dst) return RAD_E_STATE;
        RAD_TRY(rad_memcpy_async(dst, couple_draw_stage_,
                                 n * (int64_t)sizeof(RadMtpCoupleDraw), rk->ctx.stream()));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the block driver
 *
 * RAD_DRAFT_BLOCK (block diffusion, e.g. DFlash2): one backbone pass over `block` positions (an
 * anchor and `block - 1` masked slots) proposes `depth` tokens at once, with no rounds. `block` is
 * the drafter's own: depth + 1 when the anchor row predicts nothing, depth when it does
 * (rad_builder.h). Two passes:
 *
 *   Context pass: mirrors the trunk step just run. The drafter projects the trunk's tapped hidden
 *   states and stores K and V for those positions in its own cache, over exactly the trunk's rows
 *   in order.
 *
 *   Query pass: the drafter's layers over the block fill every masked position. Its own selector
 *   writes the tokens to the plugin's `proposal` buffer, which is why RAD_DRAFT_BLOCK requires
 *   one.
 *
 * Proposals go out through set_draft() like every other.
 *
 * Both passes are queued behind the step before its readback, from the card's acceptance, as the
 * serial driver's are (sched/advance.h), so the next step can be issued ahead of the commit.
 */

/* One drafter pass through the barrier. Neither pass samples (no logits in the context pass; the
 * query pass has its own selector), so the sampler chain is zero rows. */
int Engine::block_pass(const RadBatch* db, const char* what, int step) {
    for (auto& rk : ranks_) {
        int s = rk->sampler.build_greedy(0);
        if (s < 0) {
            RAD_ERR("drafter '%s': build_greedy: %s", drafter_.name, rad_strerror(s));
            return s;
        }
    }
    issue_what_.store(kIssueAll, std::memory_order_release);
    step_batch_.store(db, std::memory_order_release);
    barrier_->arrive_and_wait();
    barrier_->arrive_and_wait();
    const int s = step_status_.load(std::memory_order_acquire);
    if (s < 0)
        RAD_ERR("drafter '%s': %s pass of step %d failed: %s -- the engine cannot continue",
                drafter_.name, what, step, rad_strerror(s));
    /* Probe after each pass; by the step loop's dump these buffers hold stale values. */
    if (s >= 0) dump_buffers(ranks_[0]->ctx, step, what);
    return s;
}

bool Engine::block_drafting() const {
    return drafter_.kind == RAD_DRAFT_BLOCK && drafter_.depth > 0;
}

/* Both passes behind the step, without waiting. After the trunk's ranks join, the acceptance runs
 * on every rank (ADV_ACCEPT), leaving each sequence's anchor and committed end in its StepSlot. The context pass mirrors the trunk step, whose positions the host knows.
 * The query pass's rows are device rows that the card positions at the committed end (ADV_BLOCK),
 * and its proposals go into the slots behind the anchor (ADV_BTAKE): exactly the token row the
 * next verify reads (ADV_TRUNK), so the step issued ahead verifies them with no host round trip.
 * collect_step waits for the sampled ids alone, so the commit streams the tokens while the card
 * runs both passes; the step loop waits for them (step_end_) before finish_block_drafter reads the
 * slots. */
void Engine::block_shape(int* block, int* steps) {
    *steps = drafter_.depth;
    /* The row count is the drafter's (defaulted and checked by declare_drafter). */
    *block = drafter_.block > 0 ? (int)drafter_.block : drafter_.depth + 1;
    /* Its short block under load (rad_drafter_short_block): the same anchor convention, so as many
     * fewer proposals as rows. The load is the step just run's, which the next step, the one these
     * proposals are for, mostly repeats (Scheduler::short_block_on). */
    if (drafter_short_ <= 0) return;
    short_on_ = Scheduler::short_block_on(short_on_, sched_.decoded_seqs(), short_from_);
    if (short_on_) {
        /* A tree (rad_drafter_tree) grows a tree of its own over the shorter pass, at most its
         * draft positions of nodes (DFlash2 attends across its block, so the short pass is not the
         * full pass's prefix); the verify takes that many. */
        if (tree_) *steps = std::min(*steps, (int)drafter_short_ - 1);
        else       *steps -= *block - (int)drafter_short_;
        *block  = (int)drafter_short_;
    }
}

int Engine::issue_block_drafter(int step, StepOut& so) {
    block_rows_ = 0;
    if (!block_drafting()) return RAD_OK;

    RAD_TRY(advance_all(ADV_ACCEPT, 0, false));
    /* A tree's accepted path into place in the trunk's cache, against the same build (the step
     * just accepted), before any later pass reads it. */
    if (tree_) RAD_TRY(tree_compact(-1));
    /* The sampled ids before either pass, as the serial driver reads them. */
    RAD_TRY(collect_issue(so, /*fence=*/true));

    /* The short block too (block_shape): its query pass is a shorter batch the card positions
     * the same way, its proposals the front of each row at the drafter's pitch (ADV_BTAKE), and
     * the step issued ahead verifies that many (issue_ahead). */
    int block = 0, steps = 0;
    block_shape(&block, &steps);
    block_steps_ = steps;
    block_row_.resize((size_t)cfg_.max_seqs);
    /* The head reads the block's draft rows. A tree (rad_drafter_tree; DFlash2's convention, the
     * anchor predicts nothing) reads all block - 1 of them whatever its node count. */
    const int head = tree_ ? block - 1 : steps;
    const int n = sched_.block_begin(block, head, (int)drafter_.window, drafter_.mask_token,
                                     block_row_.data(), (int)block_row_.size(), /*on_card=*/true);

    /* --draft-coupling: the draws before the query pass reads them (stage_couple_draws). */
    if (mtp_couple_ && n > 0) RAD_TRY(stage_couple_draws(block_row_.data(), n));

    /* The context pass runs even when nothing drafts: a mid-prompt sequence has hidden states the
     * drafter will need, and skipping would leave a permanent hole in its cache. A tree verify's
     * context rows store only the accepted path, at its committed positions (ADV_CTXTREE). */
    const RadBatch* const cb = sched_.block_context(step);
    if (cb) {
        if (tree_) RAD_TRY(advance_all(ADV_CTXTREE, 0, false));
        RAD_TRY(block_pass(cb, "context", step));
    }
    /* A snapshot carrying the drafter's window is saved now, with the step's context rows stored
     * and before the query pass and the step issued ahead (save_checkpoints). */
    if (drafter_snap_) {
        /* This skip applies only under --checkpoint-draft-window. No context pass behind a step
         * that saves a snapshot (every row reaped or cancelled, say) would leave the drafter's
         * window in the slot as the previous tenant's pages. Such a checkpoint is not committed
         * (Scheduler::skip_ckpt_commit), so no hit resumes from it, and is counted; warned once. */
        if (cfg_.checkpoint_draft_window && !cb) {
            for (const StepEntry& e : sched_.plan().e)
                if (e.ckpt_slot >= 0 && e.ckpt_tok >= 0) {
                    sched_.skip_ckpt_commit(e.seq, (int64_t)e.ctx_len + e.ckpt_tok + 1);
                    if (!ckpt_stale_warned_) {
                        ckpt_stale_warned_ = true;
                        RAD_WARN("step %d saves a checkpoint (slot %d) with no drafter context pass "
                                 "behind it, so the drafter's window in that slot would be stale: "
                                 "not committed (radiance:draft_window_stale_ckpt_total; "
                                 "--checkpoint-draft-window; warned once)", step, (int)e.ckpt_slot);
                    }
                }
        }
        ckpt_save_skip_ = cfg_.checkpoint_draft_window && !cb;
        const int ss = save_checkpoints();
        ckpt_save_skip_ = false;
        RAD_TRY(ss);
    }
    if (n <= 0) return RAD_OK;
    const RadBatch* qb = sched_.block_query(step);
    if (!qb) return RAD_OK;
    RAD_TRY(advance_all(ADV_BLOCK, 0, false));
    RAD_TRY(block_pass(qb, "query", step));
    RAD_TRY(advance_all(ADV_BTAKE, steps, true, n));
    Rank& r0 = *ranks_[0];
    if (mtp_couple_) {
        /* Positions 1..steps of the drafting rows, on the stream of the slots below. */
        const void* state = r0.ctx.buf_ptr(couple_state_buf_);
        if (!state || steps > (int)drafter_.depth) return RAD_E_STATE;
        const int64_t pitch = couple_rows_ * (int64_t)sizeof(RadMtpCoupleState);
        RAD_TRY(rad_memcpy_2d_async(couple_readback_, pitch, state, pitch,
                                    (int64_t)n * (int64_t)sizeof(RadMtpCoupleState), steps,
                                    r0.ctx.stream()));
    }
    RAD_TRY(rad_memcpy_async(slots_host_, r0.step_slots,
                             (int64_t)cfg_.max_seqs * (int64_t)sizeof(StepSlot), r0.ctx.stream()));
    block_rows_ = n;
    return RAD_OK;
}

/* After commit: check the host's acceptance against the card's, which positioned the query pass
 * (a disagreement is a defect and the step is refused, as in finish_serial_drafter), move the
 * drafter's filled range, and publish each row's proposal from the slots. */
int Engine::finish_block_drafter(const StepOut& so) {
    if (!block_drafting()) return RAD_OK;
    if (block_rows_ > 0) {
        const StepPlan& plan = sched_.plan();
        for (size_t i = 0; i < plan.e.size(); ++i) {
            const StepEntry& e = plan.e[i];
            if (so.first_row[i] < 0) continue;
            const StepSlot& S = slots_host_[e.card];   /* in range: the builder refuses others */
            if (S.acc != so.accepted[i] || S.tok[0] != so.token[i]) {
                RAD_ERR("drafter '%s': the card accepted %d draft(s) ending in token %d for "
                        "sequence %llu and the host %d ending in %d; the query pass ran from the "
                        "card's", drafter_.name, S.acc, S.tok[0], (unsigned long long)e.seq,
                        so.accepted[i], so.token[i]);
                sched_.block_end();
                return RAD_E_STATE;
            }
        }
    }
    sched_.block_end();
    if (block_rows_ <= 0) return RAD_OK;

    Rank& r0 = *ranks_[0];
    const int steps = block_steps_;
    draft_out_.assign((size_t)block_rows_ * (size_t)steps, 0);
    for (int i = 0; i < block_rows_; ++i) {
        const StepSlot& S = slots_host_[block_row_[(size_t)i].card];
        for (int j = 0; j < steps; ++j)
            draft_out_[(size_t)i * (size_t)steps + (size_t)j] = S.tok[1 + j];
    }
    /* -v: what each block pass proposed, a line a sequence, so two runs can be compared draft by
     * draft where acceptance alone cannot tell them apart (a drafter whose context is wrong still
     * drafts, and its rejected drafts change no token). */
    const bool show = (int)log_level() >= (int)Log::Debug;
    for (int i = 0; i < block_rows_; ++i) {
        const int32_t* d = &draft_out_[(size_t)i * (size_t)steps];
        if (show) {
            std::string t;
            for (int j = 0; j < steps; ++j) t += fmt(" %d", (int)d[j]);
            RAD_DEBUG("drafter '%s': sequence %llu proposes%s", drafter_.name,
                      (unsigned long long)block_row_[(size_t)i].seq, t.c_str());
        }
        /* As in finish_serial_drafter: the step issued ahead only carries a request whose answer
         * is fixed (Scheduler::step_next_locked), so this agrees with the count it verifies. A
         * tree's parents come from the same slot, as ADV_BTAKE took them. */
        if (!r0.sampler.allows_speculation(block_row_[(size_t)i].seq)) continue;
        const uint64_t seq = block_row_[(size_t)i].seq;
        const StepSlot& S = slots_host_[block_row_[(size_t)i].card];
        const int rc = sched_.set_draft(seq, d, steps, tree_ ? S.par : nullptr);
        if (mtp_couple_)
            RAD_TRY(bind_couple(seq, i, steps, rc == RAD_OK, block_row_[(size_t)i].card));
    }
    return RAD_OK;
}

}  /* namespace rad */
