// A model's requests: the ones running now, and the ones that finished within the hour the
// process remembers.
import { useMemo, useState } from "react"
import { Empty, Kpi, Panel } from "@/components/bits"
import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import type { FinishedRequest, Stats } from "@/lib/api"
import { splitModel } from "@/lib/api"
import { DASH, fAgo, fDur, fNum, fSig, fin } from "@/lib/format"
import type { Scope } from "@/lib/store"
import { Download } from "lucide-react"

const median = (a: number[]) => { if (!a.length) return null; const s = a.slice().sort((x, y) => x - y); return s[s.length >> 1] }
const STATE: Record<string, string> = { prefill: "Processing prompt", decode: "Generating", waiting: "Waiting", preempted: "Paused", done: "Done" }

export function Requests({ scope, model }: { scope: Scope<Stats>; model: string }) {
  const s = scope.live
  /* A LoRA child page shows only its adapter's rows; the base model's page keeps every request,
   * adapter included -- the model is the unit that did the work. */
  const { adapter } = splitModel(model)
  const lora = adapter ? s?.loras?.find((l) => l.name === adapter) : undefined
  const done = adapter ? scope.finished.filter((r) => r.lora === adapter) : scope.finished
  const [shown, setShown] = useState(50)
  const now = scope.series.end * 1000
  const speeds = done.map((r) => r.tps).filter((v): v is number => fin(v) && v > 0)
  const ttfts = done.map((r) => r.ttft).filter((v): v is number => fin(v))
  const mean = speeds.length ? speeds.reduce((a, b) => a + b, 0) / speeds.length : null
  const running = (s?.requests ?? []).filter((r) => r.state !== "done" && (!adapter || r.lora === adapter))
  const latest = useMemo(() => done.slice().reverse(), [done, scope.version]) // eslint-disable-line react-hooks/exhaustive-deps

  function save() {
    const blob = new Blob([JSON.stringify(done, null, 1)], { type: "application/json" })
    const a = document.createElement("a")
    a.href = URL.createObjectURL(blob)
    a.download = `${s?.model ?? "model"}-requests.json`
    a.click()
    setTimeout(() => URL.revokeObjectURL(a.href), 1000)
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Kpi label="Served" value={adapter ? fNum(lora?.requests) : fNum(s?.total_requests)} detail={adapter ? `requests completed under ${adapter}` : "since the model started"} />
        <Kpi label="Preempted" value={fNum(s?.preemptions)} detail={adapter ? "model-wide: pauses are the scheduler's, not the adapter's" : "requests paused for another and resumed"} />
        <Kpi label="Mean speed" value={fin(mean) ? fSig(mean) : DASH} unit="tok/s" detail={`over ${fNum(speeds.length)} finished`} />
        <Kpi label="Median time to first token" value={ttfts.length ? fDur(median(ttfts)) : DASH} detail={`over ${fNum(ttfts.length)} finished`} />
      </div>

      <Panel title="Running" description={running.length ? undefined : adapter ? `No active requests under ${adapter}` : "No active requests"} flush={running.length > 0}>
        {running.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="pl-4">Request</TableHead><TableHead>Status</TableHead>
                <TableHead className="w-[28%]">Progress</TableHead>
                <TableHead className="text-right">Prompt</TableHead><TableHead className="text-right">Reused</TableHead>
                <TableHead className="text-right">Written</TableHead><TableHead className="text-right">Speed</TableHead>
                <TableHead className="pr-4 text-right">Age</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {running.map((r) => {
                const reading = r.state === "prefill"
                const of = reading ? Math.max(1, r.prompt - r.cached) : Math.max(1, r.max)
                const at = reading ? Math.max(0, r.computed - r.cached) : r.output
                return (
                  <TableRow key={String(r.id)}>
                    <TableCell className="num pl-4 font-medium">{String(r.id)}</TableCell>
                    <TableCell>{STATE[r.state] ?? r.state}</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 min-w-12 flex-1 overflow-hidden rounded-full bg-muted"><div className="h-full rounded-full bg-primary transition-[width] duration-700" style={{ width: Math.min(100, (100 * at) / of) + "%", opacity: reading ? 0.55 : 1 }} /></div>
                        <span className="num w-24 shrink-0 text-right text-xs text-muted-foreground">{fNum(at)} / {fNum(of)}</span>
                      </div>
                    </TableCell>
                    <TableCell className="num text-right">{fNum(r.prompt)}</TableCell>
                    <TableCell className="num text-right text-muted-foreground">{r.cached ? fNum(r.cached) : DASH}</TableCell>
                    <TableCell className="num text-right">{fNum(r.output)}</TableCell>
                    <TableCell className="num text-right">{r.tps > 0 ? fSig(r.tps) : DASH}</TableCell>
                    <TableCell className="num pr-4 text-right text-muted-foreground">{fDur(r.age_s)}</TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        )}
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Speed by context" description="Tokens per second vs. final context length, per finished request">
          <Scatter pts={done.filter((r) => fin(r.tps) && r.tps! > 0 && r.ctx > 0).map((r) => ({ x: r.ctx, y: r.tps!, r }))} />
        </Panel>
        <Panel title="Time to first token" description="Distribution across finished requests">
          <Histogram values={ttfts} />
        </Panel>
      </div>

      <Panel title="Finished" description={done.length ? `${fNum(done.length)} in the last hour${adapter ? ` under ${adapter}` : ""}, latest first` : `No requests in the last hour${adapter ? ` under ${adapter}` : ""}`} flush={done.length > 0}
        action={done.length > 0 && <Button variant="outline" size="sm" onClick={save}><Download />Export</Button>}>
        {done.length > 0 && (
          <>
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead className="pl-4">Request</TableHead><TableHead className="text-right">Prompt</TableHead>
                  <TableHead className="text-right">Written</TableHead><TableHead className="text-right">Context</TableHead>
                  <TableHead className="text-right">First token</TableHead><TableHead className="text-right">Prefill</TableHead>
                  <TableHead className="text-right">Speed</TableHead><TableHead className="pr-4 text-right">Finished</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {latest.slice(0, shown).map((r) => (
                  <TableRow key={r.id + ":" + r.at}>
                    <TableCell className="num pl-4 font-medium">{r.id}</TableCell>
                    <TableCell className="num text-right">{fNum(r.prompt)}</TableCell>
                    <TableCell className="num text-right">{fNum(r.out)}</TableCell>
                    <TableCell className="num text-right text-muted-foreground">{fNum(r.ctx)}</TableCell>
                    <TableCell className="num text-right">{fin(r.ttft) ? fDur(r.ttft) : DASH}</TableCell>
                    <TableCell className="num text-right">{fin(r.pf) ? fSig(r.pf) + " tok/s" : DASH}</TableCell>
                    <TableCell className="num text-right">{fin(r.tps) && r.tps > 0 ? fSig(r.tps) + " tok/s" : DASH}</TableCell>
                    <TableCell className="num pr-4 text-right text-muted-foreground">{fAgo(r.at, now)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            {latest.length > shown && <div className="border-t p-2 text-center"><Button variant="ghost" size="sm" onClick={() => setShown(shown + 100)}>Show {Math.min(100, latest.length - shown)} more</Button></div>}
          </>
        )}
      </Panel>
    </div>
  )
}

const tick = (v: number) => (v >= 1000 ? fSig(v / 1000) + "k" : fSig(v))
function Scatter({ pts }: { pts: { x: number; y: number; r: FinishedRequest }[] }) {
  if (pts.length < 2) return <Empty>Available after two finished requests</Empty>
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y)
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y1 = Math.max(...ys) * 1.08
  const lx = (v: number) => (x1 > x0 ? (Math.log(v) - Math.log(x0)) / (Math.log(x1) - Math.log(x0)) : 0.5)
  return (
    <div className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-1">
      <div className="num flex flex-col justify-between text-right text-[11px] text-muted-foreground"><span>{fSig(y1)}</span><span>{fSig(y1 / 2)}</span><span>0</span></div>
      <div className="relative h-44 rounded-md border bg-background/40">
        <div className="absolute inset-x-0 top-1/2 border-t border-dashed border-grid" />
        {pts.map((p, i) => (
          <Tooltip key={i}>
            <TooltipTrigger render={<i className="absolute size-2 -translate-x-1/2 translate-y-1/2 rounded-full bg-primary/70 ring-1 ring-background hover:bg-primary" style={{ left: 2 + 96 * lx(p.x) + "%", bottom: (96 * p.y) / y1 + "%" }} />} />
            <TooltipContent>{p.r.id} · {fNum(p.x)} tokens of context · {fSig(p.y)} tok/s</TooltipContent>
          </Tooltip>
        ))}
      </div>
      <div />
      <div className="num flex justify-between text-[11px] text-muted-foreground"><span>{tick(x0)}</span><span>tokens of context</span><span>{tick(x1)}</span></div>
    </div>
  )
}

function Histogram({ values }: { values: number[] }) {
  if (values.length < 2) return <Empty>Available after two finished requests</Empty>
  const edges = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, Infinity]
  const used = edges.findIndex((e) => e > Math.max(...values))
  const bins = edges.slice(0, Math.max(6, used + 1))
  const counts = bins.slice(0, -1).map((lo, i) => values.filter((v) => v >= lo && v < bins[i + 1]).length)
  const top = Math.max(...counts, 1)
  const name = (i: number) => (bins[i + 1] === Infinity ? "over " + fDur(bins[i]) : "under " + fDur(bins[i + 1]))
  return (
    <div className="flex h-[12.25rem] flex-col justify-end gap-1.5">
      <div className="flex flex-1 items-end justify-center gap-1.5">
        {counts.map((c, i) => (
          <Tooltip key={i}>
            <TooltipTrigger render={<div className="flex h-full max-w-24 flex-1 flex-col justify-end"><span className="num mb-1 text-center text-[11px] text-muted-foreground">{c || ""}</span><div className="rounded-sm bg-primary/80" style={{ height: Math.max(c ? 3 : 0, (82 * c) / top) + "%" }} /></div>} />
            <TooltipContent>{c} {c === 1 ? "request" : "requests"} {name(i)}</TooltipContent>
          </Tooltip>
        ))}
      </div>
      <div className="num flex justify-center gap-1.5 border-t pt-1 text-[11px] text-muted-foreground">
        {counts.map((_, i) => <span key={i} className="max-w-24 flex-1 truncate text-center">{bins[i + 1] === Infinity ? "> " + fDur(bins[i]) : "< " + fDur(bins[i + 1])}</span>)}
      </div>
    </div>
  )
}
