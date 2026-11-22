// The fleet: what is on each card, what every model is doing, and how that moved over the window.
import { useMemo } from "react"
import { Chart, Legend, type Line } from "@/components/chart"
import { Empty, Kpi, Panel, PartBar, PartLegend, StateBadge, StateDot, STATE_OF_CODE, stateOf, type Part } from "@/components/bits"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import type { Fleet, FleetModel } from "@/lib/api"
import { bytesParts, fBytes, fClock, fDur, fin, fNum } from "@/lib/format"
import { fleet as fleetScope, useScope, type Series } from "@/lib/store"
import { hrefOf, useRange } from "@/lib/ui"

const hue = (i: number) => `var(--series-${(i % 8) + 1})`
// What a model holds on one accelerator: the server's figure for it in any state, so a model that
// is loading or partly parked is counted as itself; from a server that sends none, its claim
// while it is loaded.
const onCard = (m: FleetModel, card: number) =>
  m.vram_bytes_by_card ? (m.vram_bytes_by_card[String(card)] ?? 0)
    : m.state === "resident" && m.cards?.includes(card) ? (m.vram_bytes_per_card ?? 0) : 0
const held = (m: FleetModel) =>
  m.vram_bytes_by_card ? Object.values(m.vram_bytes_by_card).reduce((a, b) => a + b, 0)
    : m.state === "resident" ? (m.vram_bytes_per_card ?? 0) * (m.cards?.length ?? 0) : 0

export function Overview() {
  const scope = useScope(fleetScope)
  const range = useRange()
  const live = scope.live
  if (!live) return <Empty className="m-6">{scope.error ?? "Waiting for data…"}</Empty>
  // A served adapter is listed as a child of its model, with the model's state and counters: the
  // overview counts and draws the models themselves.
  const f = { ...live, models: live.models.filter((m) => m.adapter !== true) }
  const hueOf = new Map(f.models.map((m, i) => [m.id, hue(i)]))
  const resident = f.models.filter((m) => m.state === "resident")
  const parked = f.models.filter((m) => m.state === "parked")
  const cold = f.models.filter((m) => m.state === "cold")
  const failed = f.models.filter((m) => m.state === "failed")
  const onCards = f.models.reduce((a, m) => a + held(m), 0)
  const vram = f.cards.reduce((a, c) => a + (c.vram_bytes ?? 0), 0)
  // All the process pins in host memory: the models' pools, tables and cache tiers as well as
  // the copies of parked ones. A server that does not say counts the copies alone.
  const copies = f.models.reduce((a, m) => a + (m.host_bytes ?? 0), 0)
  const hostBytes = f.host?.held_bytes ?? copies
  // LoRA children are rows too, and a child's in-flight is part of its parent's: counting both
  // twice would make one request look like two active ones. The KPI counts the work, in parents.
  const flight = f.models.filter((m) => !m.parent).reduce((a, m) => a + (m.in_flight ?? 0), 0)
  const waiting = f.models.filter((m) => !m.parent).reduce((a, m) => a + (m.waiting ?? 0), 0)
  const returns = f.models.reduce((a, m) => a + (m.restores ?? 0), 0)
  const power = f.cards.reduce((a, c) => a + (fin(c.power_w) ? c.power_w : 0), 0)
  const [cv, cu] = bytesParts(onCards)
  const [hv, hu] = bytesParts(hostBytes)

  return (
    <div className="flex flex-col gap-4 p-4 md:p-6">
      {failed.map((m) => (
        <a key={m.id} href={hrefOf(m.id)} className="flex flex-wrap items-baseline gap-x-2 rounded-lg border border-bad/40 bg-bad/10 px-3 py-2 text-sm">
          <span className="font-medium">{m.id} did not start</span>
          <span className="min-w-0 text-muted-foreground">{m.error ?? "no reason was given"}</span>
        </a>
      ))}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <Kpi label="VRAM in use" value={cv} unit={cu} detail={`${resident.length} of ${f.models.length} models · ${fBytes(vram)} in all`} />
        <Kpi label="Active requests" value={fNum(flight)} detail={waiting ? `${waiting} waiting` : "being served now"} />
        <Kpi label="Parked" value={fNum(parked.length)} detail={parked.length ? parked.map((m) => m.id).join(", ") : "none"} />
        <Kpi label="Host memory in use" value={hv} unit={hu} detail={(fin(f.host?.budget_bytes) ? `of ${fBytes(f.host.budget_bytes)} allowed` : "") + (copies > 0 ? ` · ${fBytes(copies)} parked copies` : "")} />
        <Kpi label="Restores" value={fNum(returns)} detail={cold.length ? `${cold.length} not loaded` : "from parked to loaded"} />
        <Kpi label="Power draw" value={power > 0 ? fNum(power) : "–"} unit={power > 0 ? "W" : undefined} detail={`across ${f.cards.length} accelerator${f.cards.length === 1 ? "" : "s"}`} />
      </div>

      <div className="grid gap-3 xl:grid-cols-2">
        {f.cards.map((c) => <CardPanel key={c.card} fleet={f} card={c.card} hueOf={hueOf} data={scope.series} version={scope.version} range={range} />)}
      </div>

      <Panel title="State over time" description="Each model's state across the time range">
        <StateStrip models={f.models} data={scope.series} version={scope.version} range={range} />
      </Panel>

      <Panel title="Models" flush>
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead className="pl-4">Model</TableHead>
              <TableHead>State</TableHead>
              <TableHead className="text-right">Priority</TableHead>
              <TableHead>Accelerators</TableHead>
              <TableHead className="text-right">VRAM</TableHead>
              <TableHead className="text-right">Host</TableHead>
              <TableHead className="text-right">Active</TableHead>
              <TableHead className="text-right">Loads</TableHead>
              <TableHead className="text-right">Parks</TableHead>
              <TableHead className="pr-4 text-right">Last load</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {f.models.map((m) => (
              <TableRow key={m.id} className="cursor-pointer" onClick={() => { location.hash = hrefOf(m.id) }}>
                <TableCell className="pl-4">
                  <a href={hrefOf(m.id)} className="flex items-center gap-2 font-medium" onClick={(e) => e.stopPropagation()}>
                    <i className="size-2 rounded-full" style={{ background: hueOf.get(m.id) }} />
                    {m.id}
                  </a>
                  <div className="mt-0.5 max-w-72 truncate pl-4 text-xs text-muted-foreground">{m.error ?? m.file}</div>
                </TableCell>
                <TableCell><StateBadge state={m.state} doing={m.doing} /></TableCell>
                <TableCell className="num text-right">{fNum(m.priority)}{m.pinned ? " · pinned" : ""}</TableCell>
                <TableCell className="num text-muted-foreground">{m.cards?.length ? m.cards.join(", ") : "–"}</TableCell>
                <TableCell className="num text-right">{held(m) > 0 ? fBytes(held(m)) : "–"}</TableCell>
                <TableCell className="num text-right">{(m.host_bytes ?? 0) + (m.host_pinned_bytes ?? 0) > 0 ? fBytes((m.host_bytes ?? 0) + (m.host_pinned_bytes ?? 0)) : "–"}</TableCell>
                <TableCell className="num text-right">{m.in_flight ? fNum(m.in_flight) : "–"}{m.waiting ? ` +${m.waiting}` : ""}</TableCell>
                <TableCell className="num text-right">{fNum(m.loads)}</TableCell>
                <TableCell className="num text-right">{fNum(m.parks)}</TableCell>
                <TableCell className="num pr-4 text-right">{fin(m.load_seconds as number) ? fDur(m.load_seconds as number) : "–"}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Panel>
    </div>
  )
}

function CardPanel({ fleet: f, card, hueOf, data, version, range }: { fleet: Fleet; card: number; hueOf: Map<string, string>; data: Series; version: number; range: number }) {
  const c = f.cards.find((x) => x.card === card)!
  const parts: Part[] = f.models.map((m) => ({ key: m.id, label: m.id, value: onCard(m, card), color: hueOf.get(m.id)!, text: fBytes(onCard(m, card)) }))
  const used = (c.vram_bytes ?? 0) - (c.free_bytes ?? 0)
  const named = parts.reduce((a, p) => a + p.value, 0)
  // What no model's ledger names: the open hardware queues, by the server's count, and the
  // rest, which is buffers the loaded models' kernels allocate themselves and the driver's own.
  const queues = Math.max(Math.min(c.queue_bytes ?? 0, used - named), 0)
  const rest = used - named - queues
  if (queues > 0) parts.push({ key: "~queues", label: `Queues (${c.queues ?? 0} open)`, value: queues, color: "var(--idle)", text: fBytes(queues) })
  if (rest > 64 << 20) parts.push({ key: "~other", label: "Model buffers and driver", value: rest, color: "var(--muted-foreground)", text: fBytes(rest) })
  const lines: Line[] = f.models.map((m, i) => ({ key: `c${card}:${m.id}`, label: m.id, color: i + 1 }))
  return (
    <Panel title={<span className="flex items-baseline gap-2">Accelerator {card}<span className="text-xs font-normal text-muted-foreground">{c.name}</span></span>}
      action={<span className="num text-xs text-muted-foreground">{fBytes(c.free_bytes)} free of {fBytes(c.vram_bytes)}</span>}>
      <div className="flex flex-col gap-3">
        <PartBar parts={parts} total={c.vram_bytes} className="h-3" />
        <PartLegend parts={parts} className="content-start xl:min-h-10" />
        <div className="grid grid-cols-3 gap-2">
          <Gauge label="Busy" value={fin(c.busy) ? Math.round(c.busy * 100) + "%" : "–"} data={data} k={`busy${card}`} max={1} />
          <Gauge label="Power" value={fin(c.power_w) ? Math.round(c.power_w) + " W" : "–"} data={data} k={`power${card}`} />
          <Gauge label="Temperature" value={fin(c.temp_c) ? Math.round(c.temp_c) + " °C" : "–"} data={data} k={`temp${card}`} />
        </div>
        <Chart data={data} version={version} lines={lines} range={range} fmt={fBytes} stack bytes height={170} max={c.vram_bytes} />
      </div>
    </Panel>
  )
}

/** A figure with its last five minutes drawn beside it. */
function Gauge({ label, value, data, k, max }: { label: string; value: string; data: Series; k: string; max?: number }) {
  const col = data.col(k) ?? []
  const lo = data.firstAfter(data.end - 300)
  const pts = col.slice(lo)
  const vals = pts.filter((v): v is number => v !== null)
  const top = max ?? Math.max(1, ...vals) * 1.05
  const base = max !== undefined ? 0 : Math.min(...vals, top) * 0.9
  const n = pts.length
  let d = ""
  let open = false
  pts.forEach((v, i) => {
    if (v === null) { open = false; return }
    d += (open ? "L" : "M") + ((100 * i) / Math.max(1, n - 1)).toFixed(2) + " " + (27 - (24 * (Math.min(v, top) - base)) / Math.max(1e-9, top - base)).toFixed(2)
    open = true
  })
  return (
    <div className="flex items-center gap-3 rounded-lg border bg-background/40 px-3 py-2">
      <div className="min-w-0">
        <div className="truncate text-xs text-muted-foreground">{label}</div>
        <div className="num text-base font-semibold whitespace-nowrap">{value}</div>
      </div>
      <svg viewBox="0 0 100 30" preserveAspectRatio="none" className="h-8 min-w-0 flex-1 text-primary">
        <path d={d} fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
      </svg>
    </div>
  )
}

/** One row a model: its state as coloured runs along the window. */
function StateStrip({ models, data, version, range }: { models: FleetModel[]; data: Series; version: number; range: number }) {
  const rows = useMemo(() => {
    const end = data.end, start = end - range, lo = data.firstAfter(start)
    return models.map((m) => {
      const col = data.col("s:" + m.id) ?? []
      const runs: { a: number; b: number; s: string }[] = []
      for (let i = lo; i < data.t.length; ++i) {
        const v = col[i]
        if (v == null) continue
        const s = STATE_OF_CODE[v] ?? "cold"
        const a = i === lo ? Math.max(start, data.t[i] - 1) : data.t[i - 1], b = data.t[i]
        const last = runs[runs.length - 1]
        if (last && last.s === s && a - last.b < 2.5) last.b = b
        else runs.push({ a, b, s })
      }
      return { id: m.id, runs }
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models.map((m) => m.id).join("|"), data, version, range])
  const end = data.end, start = end - range
  const ticks = [0, 0.25, 0.5, 0.75, 1]
  const seen = new Set(rows.flatMap((r) => r.runs.map((x) => x.s)))
  return (
    <div className="flex flex-col gap-2">
      <div className="grid grid-cols-[minmax(6rem,10rem)_1fr] items-center gap-x-3 gap-y-1.5">
        {rows.map((r) => (
          <div key={r.id} className="contents">
            <a href={hrefOf(r.id)} className="truncate text-sm">{r.id}</a>
            <div className="relative h-5 overflow-hidden rounded bg-muted/50">
              {r.runs.map((x, i) => (
                <Tooltip key={i}>
                  <TooltipTrigger render={<div className="absolute inset-y-0" style={{ left: (100 * (x.a - start)) / range + "%", width: Math.max(0.3, (100 * (x.b - x.a)) / range) + "%", background: stateOf(x.s).tone, opacity: x.s === "cold" ? 0.35 : 0.9 }} />} />
                  <TooltipContent>{stateOf(x.s).label} · {fClock(x.a, true)} to {fClock(x.b, true)} · {fDur(x.b - x.a)}</TooltipContent>
                </Tooltip>
              ))}
            </div>
          </div>
        ))}
        <div />
        <div className="num relative h-4 text-[11px] text-muted-foreground">
          {ticks.map((p) => (
            <span key={p} className={"absolute top-0 whitespace-nowrap" + (p === 0.25 || p === 0.75 ? " hidden lg:inline" : "")} style={p === 0 ? { left: 0 } : p === 1 ? { right: 0 } : { left: p * 100 + "%", transform: "translateX(-50%)" }}>
              {end ? fClock(start + p * range, range <= 600) : ""}
            </span>
          ))}
        </div>
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        {["resident", "loading", "waiting", "parked", "cold", "failed"].filter((s) => seen.has(s)).map((s) => (
          <span key={s} className="inline-flex items-center gap-1.5"><StateDot state={s} />{stateOf(s).label}</span>
        ))}
      </div>
    </div>
  )
}
export { Legend }
