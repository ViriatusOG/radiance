// A model's figures: a row of tiles, each a figure now, and the chosen one drawn large over the
// window. A figure the model has never produced (acceptance without a drafter) has no tile.
import { useMemo, useState, useSyncExternalStore } from "react"
import { Bars, Empty, Panel, PartBar, PartLegend, type Part } from "@/components/bits"
import { Chart, Legend, type Line, type Overlay } from "@/components/chart"
import { Switch } from "@/components/ui/switch"
import type { Experts, Stats } from "@/lib/api"
import { splitModel } from "@/lib/api"
import { DASH, bytesParts, fBytes, fDur, fNum, fPct, fRate, fSig, fin } from "@/lib/format"
import type { Scope, Series } from "@/lib/store"
import { useRange } from "@/lib/ui"
import { cn } from "@/lib/utils"

export type Kind = "chat" | "embed" | "gen"
const fMs = (v: number) => (v >= 100 ? v.toFixed(0) : v.toFixed(2)) + " ms"
const fPc = (v: number) => fPct(v, v >= 0.9995 ? 0 : 1) + "%"
const fTps = (v: number) => (v === 0 ? "0" : fSig(v))

interface Metric {
  id: string; kinds: Kind; label: string; unit?: string; draft?: boolean; always?: boolean
  show: (s: Stats, h: Series, slots: number | undefined, noun: string) => [string, string?, string?]
  lines: Line[]; fmt: (v: number) => string
  stack?: boolean; max?: number; tight?: boolean; integer?: boolean; bytes?: boolean; join?: boolean
  joinGap?: { seconds: number; while?: string }
}
const one = (key: string, label: string): Line[] => [{ key, label, color: 1 }]
// A rate has no value in a sample where work of its kind was under way but no step of it finished
// (a long prefill step holds decode back); such gaps are drawn through while the model is busy
// (history's `busy`: requests running, a step open or one finished) and stay gaps when it is idle.
const BUSY = { seconds: 60, while: "busy" }
// Measurements rather than rates (step time, acceptance, tokens per step, expert hits) exist only
// where their kind of step ran; their points are joined across every gap, idle ones included.
const JOIN_ALL = { seconds: 3600 }
const METRICS: Metric[] = [
  { id: "decode", kinds: "chat", label: "Decode", unit: "tok/s", show: (s) => [fin(s.decode_tps) ? fTps(s.decode_tps) : DASH], lines: one("decode", "tokens a second"), fmt: fTps, joinGap: BUSY },
  { id: "step", kinds: "chat", label: "Step", unit: "ms",
    // The newest sample only: with no decode step in it the figure is –, never an older step
    // shown as current. The detail is the engine's best decode step.
    show: (s, h) => { const c = h.col("step") ?? []; const v = c.length ? c[c.length - 1] : null;
                    // A bare number: the tile appends the unit, and fMs would add it again.
                    return [fin(v) ? (v >= 100 ? v.toFixed(0) : v.toFixed(2)) : DASH, (s.step_ns_decode_best ?? 0) > 0 ? "best " + fMs(s.step_ns_decode_best! / 1e6) : ""] },
    lines: [{ key: "step", label: "step", color: 1 }, { key: "stepBest", label: "best", dash: true }], fmt: fMs, tight: true,
    joinGap: JOIN_ALL },
  // The newest sample; an idle rate reads 0 and the dash is only for a ring with no sample yet. The
  // draft tiles below keep their last value instead, since acceptance is undefined, not 0, when a
  // sample drafted nothing.
  { id: "prefill", kinds: "chat", label: "Prefill", unit: "tok/s", show: (_s, h) => { const c = h.col("prefill") ?? []; if (!c.length) return [DASH]; const v = c[c.length - 1]; return [fin(v) ? fTps(v) : fTps(0)] }, lines: one("prefill", "tokens a second"), fmt: fTps, joinGap: BUSY },
  // Time in the vision encoder. A model reading an image can sit here for seconds with its cards
  // busy and nothing stepping, which the decode and prefill lines alone show as idle.
  { id: "encode", kinds: "chat", label: "Vision encode", unit: "%",
    // An encode takes well under a second, so a per-second rate only blinks; the count since the
    // model started and the mean time an image took hold still, and a queue is named only while
    // there is one.
    show: (s, h) => {
      const v = h.last("encBusy"), n = s.encode_items ?? 0, w = s.encode_pending ?? 0
      const each = n > 0 ? (s.encode_ns ?? 0) / n / 1e6 : NaN
      return [fin(v) ? fPct(v) : DASH, [n > 0 ? fNum(n) + " img · " + fMs(each) + " avg" : "", w > 0 ? fNum(w) + " pending" : ""].filter(Boolean).join(" · ")]
    },
    lines: one("encBusy", "of the time encoding"), fmt: fPc, max: 1 },
  { id: "reqs", kinds: "chat", always: true, label: "Requests",
    show: (s, _h, slots) => [String(s.running ?? 0), [slots ? "of " + slots + " slots" : "", s.waiting ? s.waiting + " waiting" : "",
      (s.step_open_phase === 0 || s.step_open_phase === 2) && (s.step_open_s ?? 0) >= 1 ? "prefill step running for " + fDur(s.step_open_s) : ""].filter(Boolean).join(" · ")],
    lines: [{ key: "running", label: "running", color: 1 }, { key: "waiting", label: "waiting", color: 5 }, { key: "preempted", label: "preempted", color: 2 }], fmt: (v) => fNum(v), stack: true, integer: true },
  { id: "kv", kinds: "chat", label: "KV cache",
    show: (s) => { const [v, u] = bytesParts(s.kv_bytes_used); return [v, "of " + fBytes(s.kv_bytes_total), u] },
    lines: [{ key: "kvRead", label: "in use", color: 1 }, { key: "kvKept", label: "cached for the next turn", color: 4 }], fmt: fBytes, stack: true, bytes: true },
  // The same pool counted in tokens: held now against what it holds at present and the most it can
  // grow to when every lendable block goes to the cache. Absent for a model with no paged pool.
  { id: "kvtok", kinds: "chat", label: "KV tokens",
    show: (s) => [fNum(s.kv_tokens_used), "of " + fNum(s.kv_tokens_total)
      + ((s.kv_tokens_max ?? 0) > (s.kv_tokens_total ?? 0) ? " · up to " + fNum(s.kv_tokens_max) : "")],
    lines: [{ key: "kvTokUsed", label: "in use", color: 1 }, { key: "kvTokCap", label: "capacity now", color: 2, dash: true }, { key: "kvTokMax", label: "capacity if all flexible space is used", color: 5, dash: true }],
    fmt: (v) => fNum(v), integer: true },
  { id: "hit", kinds: "chat", label: "Prefix cache hits", unit: "%", show: (s) => [fin(s.prefix_hit_rate) ? fPct(s.prefix_hit_rate) : DASH], lines: one("hit", "of prompt tokens"), fmt: fPc, max: 1 },
  { id: "accept", kinds: "chat", label: "Acceptance", unit: "%", draft: true, show: (_s, h) => [fPct(h.last("accept"))], lines: one("accept", "of drafted tokens"), fmt: fPc, max: 1, joinGap: JOIN_ALL },
  { id: "tok", kinds: "chat", label: "Tokens per step", draft: true, show: (_s, h) => { const v = h.last("tokStep"); return [fin(v) ? v.toFixed(2) : DASH] }, lines: one("tokStep", "tokens"), fmt: (v) => v.toFixed(2), tight: true, joinGap: JOIN_ALL },
  { id: "ethru", kinds: "embed", always: true, label: "Throughput", unit: "tok/s", show: (s) => [fin(s.prefill_tps) ? fTps(s.prefill_tps) : DASH], lines: one("prefill", "tokens a second"), fmt: fTps },
  { id: "erate", kinds: "embed", always: true, label: "Inputs", unit: "/s", show: (_s, h) => { const v = h.last("reqRate"); return [fin(v) ? fTps(v) : DASH] }, lines: one("reqRate", "inputs a second"), fmt: fTps },
  { id: "estep", kinds: "embed", always: true, label: "Step", unit: "ms", show: (_s, h) => { const c = h.col("stepAny") ?? []; const v = c.length ? c[c.length - 1] : null; return [fin(v) ? v.toFixed(2) : DASH] }, lines: one("stepAny", "step"), fmt: fMs, tight: true },
  { id: "eflight", kinds: "embed", always: true, label: "Active requests", show: (s) => [String(s.in_flight ?? 0)], lines: one("flight", "in flight"), fmt: (v) => fNum(v), integer: true },
  { id: "jrun", kinds: "gen", always: true, label: "Running", show: (s) => [String(s.jobs?.running ?? 0)], lines: one("jobRun", "running"), fmt: (v) => fNum(v), integer: true },
  { id: "jdone", kinds: "gen", always: true, label: "Completed", show: (s) => [fNum(s.jobs?.done), s.jobs?.failed ? s.jobs.failed + " failed" : ""], lines: one("jobDone", "made"), fmt: (v) => fNum(v), integer: true },
  { id: "jlast", kinds: "gen", always: true, label: "Last duration", show: (s) => [s.jobs?.done ? fDur(s.jobs.last_ms / 1000) : DASH], lines: one("jobTime", "took"), fmt: (v) => fDur(v), join: true },
  { id: "jmean", kinds: "gen", always: true, label: "Mean", show: (s) => [s.jobs?.done ? fDur(s.jobs.total_ms / s.jobs.done / 1000) : DASH], lines: one("jobTime", "took"), fmt: (v) => fDur(v), join: true },
]
// The hardware view: the model's cards, carried in the model's own history (open wherever the model is), drawn over the chosen chart and
// in tiles of their own. It is off until the viewer turns it on (and remembered where the browser
// lets the page keep anything), so the page reads and draws nothing more for those who never ask.
const HW_KEY = "radiance.hardware"
let hwOn = false
try { hwOn = localStorage.getItem(HW_KEY) === "1" } catch { /* storage may be blocked */ }
const hwSubs = new Set<() => void>()
function setHardware(on: boolean) {
  hwOn = on
  try { localStorage.setItem(HW_KEY, on ? "1" : "0") } catch { /* as above */ }
  hwSubs.forEach((f) => f())
}
const useHardware = () => useSyncExternalStore((f) => { hwSubs.add(f); return () => { hwSubs.delete(f) } }, () => hwOn)
export function HardwareSwitch() {
  const on = useHardware()
  return (
    <label className="ml-1.5 inline-flex items-center gap-2 text-xs text-muted-foreground">
      Hardware
      <Switch size="sm" checked={on} onCheckedChange={setHardware} />
    </label>
  )
}
// A card keeps its colour across the overlay and the tiles; the first is left to the model's own line.
const cardColor = (i: number) => i + 2
const fMhz = (v: number) => v.toFixed(0) + " MHz"
const fDeg = (v: number) => v.toFixed(0) + " °C"
interface HwTile { id: string; title: string; description: string; keys: [string, string, boolean?][]; fmt: (v: number) => string; max?: number; tight?: boolean }
const HW_TILES: HwTile[] = [
  { id: "busy", title: "GPU busy", description: "Share of the time each accelerator was busy. Every card of the model runs the whole vision tower, so all of them go busy during an encode.", keys: [["busy", "busy"]], fmt: fPc, max: 1 },
  { id: "memb", title: "Memory busy", description: "Share of the time each accelerator's memory was busy", keys: [["memb", "memory busy"]], fmt: fPc, max: 1 },
  { id: "clk", title: "Clocks", description: "Core clock, and memory clock dashed", keys: [["sclk", "core"], ["mclk", "memory", true]], fmt: fMhz, tight: true },
  { id: "temp", title: "Temperatures", description: "Junction, and memory dashed", keys: [["temp", "junction"], ["tmem", "memory", true]], fmt: fDeg, tight: true },
  { id: "fan", title: "Fan", description: "Share of full speed", keys: [["fan", "fan"]], fmt: (v) => v.toFixed(0) + "%", max: 100 },
]

const TRAFFIC: Line[] = [["lkAr", "between accelerators"], ["lkUp", "host to accelerator"], ["lkDn", "accelerator to host"], ["lkSsd", "drive"], ["lkStream", "streaming"]].map(([key, label], i) => ({ key, label, color: i + 1 }))

export function Performance({ scope, kind, slots, noun, model, cards }: { scope: Scope<Stats>; kind: Kind; slots?: number; noun: string; model: string; cards?: number[] }) {
  const range = useRange()
  const [chosen, setChosen] = useState<string | null>(null)
  const s = scope.live, h = scope.series
  /* A LoRA child page: what this adapter did, beside the tiles that stay model-wide. */
  const { adapter } = splitModel(model)
  const lora = adapter ? s?.loras?.find((l) => l.name === adapter) : undefined
  const hw = useHardware()
  // The spans where the encoder was busy, from the first busy sample to the next one after the run.
  const encSpans = useMemo(() => {
    if (!hw) return undefined
    const col = h.col("encBusy") ?? []
    const out: number[][] = []
    let from = -1
    for (let i = 0; i <= col.length; ++i) {
      const busy = i < col.length && (col[i] ?? 0) > 0
      if (busy && from < 0) from = i
      else if (!busy && from >= 0) { out.push([h.t[from], h.t[Math.min(i, h.t.length - 1)]]); from = -1 }
    }
    return out
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hw, h, scope.version])
  // The model's cards: those its own history carries figures for (it does for any viewer who may
  // read the model), else the ones the fleet names.
  const fromSeries = [...new Set([...h.c.keys()].flatMap((k) => { const m = /^busy(\d+)$/.exec(k); return m && h.has(k) ? [Number(m[1])] : [] }))].sort((a, b) => a - b)
  const mine = fromSeries.length ? fromSeries : (cards ?? [])
  const overlay: Overlay | undefined = hw && mine.length
    ? { data: h, version: scope.version, max: 1, lines: mine.map((c, i) => ({ key: "busy" + c, label: "accelerator " + c + " busy", color: cardColor(i), dash: true })).filter((l) => h.has(l.key)) }
    : undefined
  const shownOverlay = overlay && overlay.lines.length ? overlay : undefined
  const hasDraft = (s?.draft_positions?.length ?? 0) > 0
  const produced = (m: Metric) => m.always || m.lines.some((l) => !l.dash && (h.col(l.key) ?? []).some((v) => (v ?? 0) > 0))
  const tiles = METRICS.filter((m) => m.kinds === kind && (!m.draft || hasDraft || h.has(m.lines[0].key)) && (s ? produced(m) : !m.draft))
  const metric = tiles.find((m) => m.id === chosen) ?? tiles[0]
  if (!metric) return <Empty>No data for this model yet</Empty>
  const moved = TRAFFIC.filter((l) => (h.col(l.key) ?? []).some((v) => (v ?? 0) > 0))
  const ex = s?.experts

  return (
    <div className="flex flex-col gap-4">
      {adapter && (
        <div className="grid grid-cols-2 gap-3 rounded-xl border bg-card p-4 sm:grid-cols-4">
          <div><div className="text-xs text-muted-foreground">Requests served</div><div className="num text-lg font-semibold">{fNum(lora?.requests)}</div></div>
          <div><div className="text-xs text-muted-foreground">Prompt tokens</div><div className="num text-lg font-semibold">{fNum(lora?.prompt_tokens)}</div></div>
          <div><div className="text-xs text-muted-foreground">Tokens generated</div><div className="num text-lg font-semibold">{fNum(lora?.output_tokens)}</div></div>
          <div><div className="text-xs text-muted-foreground">Running now</div><div className="num text-lg font-semibold">{fNum(lora?.running)}</div></div>
          <div className="col-span-full text-xs text-muted-foreground">Attributed to {adapter}. Timings, cache and memory are the model's, shared by its adapters.</div>
        </div>
      )}
      <div className="overflow-hidden rounded-xl border bg-card">
        <div role="tablist" className="grid grid-cols-2 border-b sm:grid-cols-4 xl:grid-flow-col xl:auto-cols-fr xl:grid-cols-none">
          {tiles.map((m) => {
            const [v, det, unit] = s ? m.show(s, h, slots, noun) : [DASH]
            const on = m === metric
            return (
              <button key={m.id} role="tab" aria-selected={on} onClick={() => setChosen(m.id)}
                className={cn("relative -mb-px flex min-w-0 flex-col gap-1 border-r border-b px-4 py-3 text-left transition-colors outline-none last:border-r-0 hover:bg-accent/50 focus-visible:bg-accent/60", on && "bg-accent/40")}>
                <span className="truncate text-xs font-medium text-muted-foreground">{m.id === "jdone" && noun ? noun : m.label}</span>
                <span className="num flex items-baseline gap-1 text-xl leading-none font-semibold tracking-tight">
                  {/* Not `truncate`: a flex child with it shrinks before its unit sibling, so a five-character
                      figure came out as `12.91 ⋯`. This is the tile the Performance page actually draws -
                      `Kpi` is the Overview/Requests/Sessions one. */}
                  <span>{v}</span>
                  {(unit ?? m.unit) && <span className="text-xs font-normal text-muted-foreground">{unit ?? m.unit}</span>}
                </span>
                <span className="min-h-4 text-xs text-muted-foreground break-words [overflow-wrap:anywhere]">{det}</span>
                {on && <span className="absolute inset-x-0 bottom-0 h-0.5 bg-primary" />}
              </button>
            )
          })}
        </div>
        <div className="flex flex-col gap-2 p-4">
          {metric.lines.length > 1 && <Legend lines={metric.lines} />}
          {shownOverlay && <Legend lines={shownOverlay.lines} />}
          <Chart key={metric.id} data={h} version={scope.version} lines={metric.lines} range={range} fmt={metric.fmt} height={300}
            stack={metric.stack} max={metric.max} tight={metric.tight} integer={metric.integer} bytes={metric.bytes} join={metric.join} joinGap={metric.joinGap}
            overlay={shownOverlay} bands={encSpans?.length ? encSpans : undefined} />
        </div>
      </div>
      {hw && <HardwarePanels data={h} version={scope.version} cards={mine} range={range} />}

      {(moved.length > 0 || hasDraft) && (
        <div className="grid gap-4 lg:grid-cols-3">
          {moved.length > 0 && (
            <Panel title="Traffic" description="Bytes per second, by destination" className={hasDraft ? "lg:col-span-2" : "lg:col-span-3"}>
              <div className="flex flex-col gap-2">
                <Legend lines={moved} />
                <Chart data={h} version={scope.version} lines={moved} range={range} fmt={fRate} stack bytes height={190} />
              </div>
            </Panel>
          )}
          {hasDraft && s?.draft_positions && (
            <Panel title="Acceptance by position" description="Share of drafted tokens accepted, by position" className={moved.length ? "" : "lg:col-span-3"}>
              <Bars values={s.draft_positions.map((p) => p.rate || 0)} max={1} height={150}
                tip={(i, v) => `Position ${i + 1}\n${fPc(v)} accepted\n${fNum(s.draft_positions![i].reached)} reached`} labels={(i) => String(i + 1)} />
            </Panel>
          )}
        </div>
      )}
      {ex && <ExpertsPanel ex={ex} h={h} version={scope.version} range={range} />}
    </div>
  )
}

/** The hardware tiles, one line a card; a tile no card of the model reports a figure for is left out. */
function HardwarePanels({ data, version, cards, range }: { data: Series; version: number; cards: number[]; range: number }) {
  const tiles = HW_TILES.map((t) => ({
    t, lines: cards.flatMap((c, i) => t.keys.map(([k, what, dash]): Line => ({ key: k + c, label: "accelerator " + c + " " + what, color: cardColor(i), dash }))).filter((l) => data.has(l.key)),
  })).filter((x) => x.lines.length > 0)
  if (!cards.length) return <Empty>The model is on no accelerator right now</Empty>
  if (!tiles.length) return <Empty>No hardware figures for this model's accelerators yet</Empty>
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      {tiles.map(({ t, lines }) => (
        <Panel key={t.id} title={t.title} description={t.description}>
          <div className="flex flex-col gap-2">
            {lines.length > 1 && <Legend lines={lines} />}
            <Chart data={data} version={version} lines={lines} range={range} fmt={t.fmt} max={t.max} tight={t.tight} height={150} />
          </div>
        </Panel>
      ))}
    </div>
  )
}

const TIER: Record<string, string> = { active: "var(--series-1)", resident: "var(--series-1)", staged: "var(--series-4)", warm: "var(--series-3)", host: "var(--series-5)", ram: "var(--series-5)" }
function ExpertsPanel({ ex, h, version, range }: { ex: Experts; h: Series; version: number; range: number }) {
  const parts: Part[] = (ex.tiers ?? []).filter((t) => t.units > 0).map((t) => ({ key: t.name, label: t.name, value: t.bytes || t.units, color: TIER[t.name.toLowerCase()] ?? "var(--idle)", text: `${fNum(t.units)} units · ${fBytes(t.bytes)}` }))
  // With every unit in VRAM nothing is routed from elsewhere and nothing moves: the placement alone is shown.
  const tiered = ex.tiered === true
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Panel title="Experts" description={`${fNum(ex.n_units)} units by where they are`}>
        <div className="flex flex-col gap-3">
          <PartBar parts={parts} className="h-3" />
          <PartLegend parts={parts} />
          {ex.layer_units && ex.layer_resident && (
            <Bars values={ex.layer_resident} max={Math.max(...ex.layer_units, 1)} height={70}
              tip={(i, v) => `Layer ${i}\n${fNum(v)} of ${fNum(ex.layer_units![i])} on the accelerator`} />
          )}
        </div>
      </Panel>
      {tiered && <Panel title="Expert hit rate" description="Share of routed experts already on the accelerator">
        <Chart data={h} version={version} lines={[{ key: "expHit", label: "hits", color: 1 }]} range={range} fmt={fPc} max={1} height={170} joinGap={JOIN_ALL} />
      </Panel>}
      {tiered && <Panel title="Expert moves" description="Units per second">
        <div className="flex flex-col gap-2">
          <Legend lines={[{ key: "expIn", label: "onto the accelerator", color: 1 }, { key: "expOut", label: "off the accelerator", color: 2 }]} />
          <Chart data={h} version={version} lines={[{ key: "expIn", label: "onto the accelerator", color: 1 }, { key: "expOut", label: "off the accelerator", color: 2 }]} range={range} fmt={(v) => fTps(v) + "/s"} stack height={150} />
        </div>
      </Panel>}
    </div>
  )
}
