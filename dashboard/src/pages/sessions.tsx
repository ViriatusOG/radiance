// The conversations a model keeps so their next turn need not read the prompt again: how many,
// how much of the cache they hold, and which would resume and which would start over.
import { Empty, Kpi, Panel, PartBar, type Part } from "@/components/bits"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { withModel, splitModel, type Session, type Sessions } from "@/lib/api"
import { bytesParts, fBytes, fDur, fNum } from "@/lib/format"
import { useFetched } from "@/lib/fetched"

export function SessionsView({ model }: { model: string }) {
  const { data: s, error } = useFetched<Sessions>(withModel("sessions", model), 5000)
  if (!s) return <Empty>{error ?? "Loading"}</Empty>
  /* A LoRA child page lists the conversations whose newest turn ran under that adapter. The
   * cache pools are the model's: every adapter's tokens sit in the same blocks. */
  const { adapter } = splitModel(model)
  const all = (s.sessions ?? []).slice().sort((a, b) => a.idle_s - b.idle_s)
  const list = adapter ? all.filter((r) => r.lora === adapter) : all
  const linear = (s.linear?.checkpoints_written ?? 0) > 0 || list.some((r) => r.snapshots > 0)
  const live = list.filter((r) => !linear || r.resume_tokens > 0 || r.snapshots > 0)
  const cold = list.filter((r) => !live.includes(r))
  const reuse = (r: Session) => (linear ? Math.min(r.resume_tokens || 0, r.tokens) : r.tokens)
  const [pv, pu] = bytesParts(s.pool?.bytes_used)
  const [hv, hu] = bytesParts(s.occupancy?.host_bytes)
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Kpi label="Resumable" value={fNum(live.length)} detail={cold.length ? `${cold.length} will restart` : "cached conversations"} />
        <Kpi label="Tokens cached" value={fNum(s.pool?.tokens_used)} detail={s.pool ? `of ${fNum(s.pool.tokens_total)}` : ""} />
        <Kpi label="VRAM in use" value={pv} unit={pu} detail={s.pool ? `of ${fBytes(s.pool.bytes_total)}` : ""} />
        {s.enabled
          ? <Kpi label="Host memory in use" value={hv} unit={hu} detail={s.occupancy?.host_cap_bytes ? `of ${fBytes(s.occupancy.host_cap_bytes)}` : ""} />
          : <Kpi label="Prefix cache hits" value={s.prefix_hit_rate != null ? (s.prefix_hit_rate * 100).toFixed(1) : "–"} unit="%" detail="of prompt tokens reused" />}
      </div>
      <Panel title="Will resume" description={live.length ? `Conversations whose next turn can reuse cached tokens${adapter ? ` (last turn under ${adapter})` : ""}, most recent first` : undefined} flush={live.length > 0}>
        {live.length ? <SessionTable rows={live} reuse={reuse} /> : <Empty>No cached conversations</Empty>}
      </Panel>
      {cold.length > 0 && (
        <Panel title={`${fNum(cold.length)} will restart`} description="Cached, but with no prefix the next turn can reuse" flush>
          <SessionTable rows={cold} />
        </Panel>
      )}
    </div>
  )
}

function SessionTable({ rows, reuse }: { rows: Session[]; reuse?: (r: Session) => number }) {
  const shown = rows.slice(0, 200)
  return (
    <Table>
      <TableHeader>
        <TableRow className="hover:bg-transparent">
          <TableHead className="pl-4">Conversation</TableHead>
          <TableHead className="text-right">Tokens</TableHead>
          {reuse && <TableHead className="text-right">Reusable</TableHead>}
          <TableHead className="text-right">Idle</TableHead>
          <TableHead className="w-[30%] pr-4">Where</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {shown.map((r) => {
          const blocks = r.on_device + r.on_host, unit = blocks > 0 ? r.bytes / blocks : 0
          const parts: Part[] = [
            { key: "d", label: "On accelerators", value: r.on_device, color: "var(--series-1)", text: fBytes(r.on_device * unit) },
            { key: "h", label: "Host memory", value: r.on_host, color: "var(--series-5)", text: fBytes(r.on_host * unit) },
          ]
          return (
            <TableRow key={String(r.id)}>
              <TableCell className="num pl-4 font-medium">{r.label || "#" + r.id}</TableCell>
              <TableCell className="num text-right">{fNum(r.tokens)}</TableCell>
              {reuse && <TableCell className="num text-right">{fNum(reuse(r))}</TableCell>}
              <TableCell className="num text-right text-muted-foreground">{fDur(r.idle_s)}</TableCell>
              <TableCell className="pr-4">
                <div className="flex items-center gap-2">
                  <PartBar parts={parts} className="h-1.5 flex-1" />
                  <span className="num w-28 shrink-0 text-right text-xs text-muted-foreground">{fBytes(r.bytes)}{r.moving ? " · moving" : ""}</span>
                </div>
              </TableCell>
            </TableRow>
          )
        })}
      </TableBody>
    </Table>
  )
}
