// What a model is and what the engine built for it: its shape, where its bytes are, the kernels
// its program runs, and every op with the kernel chosen for it.
import { useEffect, useMemo, useState } from "react"
import { Empty, Panel, PartBar, type Part } from "@/components/bits"
import { Input } from "@/components/ui/input"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { Switch } from "@/components/ui/switch"
import { api, withModel, splitModel, type Stats } from "@/lib/api"
import { fBytes, fNum } from "@/lib/format"
import { useFetched } from "@/lib/fetched"

interface Kernel { kernel: string; plugin: string; family?: string; computes?: string; shape?: string; dtypes?: string; geometry?: string; constraints?: string[] }
interface Weight { name: string; bytes: number; dtype: string; shape: string; tier?: string; access?: string }
interface Op {
  op: string; index: number; geometry: string; count?: number; ranged?: string; range_lo?: number; range_hi?: number
  schema?: { doc?: string }
  weights?: Weight[]
  bands?: { span: string; device?: Kernel; host?: Kernel }[]
}
interface Graph {
  model?: { arch?: string; quant?: string; n_layers?: number; n_embd?: number; n_vocab?: number; world?: number; max_seqs?: number }
  counts?: { ops?: number; weights?: number; weight_bytes?: number; arena_bytes?: number; host_arena_bytes?: number; scratch_bytes?: number }
  plugins?: { name: string; version?: string; kernels?: number; build_target?: string; description?: string }[]
  ops: Op[]; order?: number[]
  drafter?: { name?: string; kind?: string; depth?: number }
  notes?: string[]
}
interface Info { server?: { max_model_len?: number; max_num_seqs?: number }; engine?: Record<string, unknown>; endpoints?: string[]; request_defaults?: Record<string, unknown> }

const kernelOf = (o: Op): Kernel | null => { for (const b of o.bands ?? []) { const k = b.device ?? b.host; if (k?.kernel) return k } return null }
const familyOf = (o: Op) => kernelOf(o)?.family || o.op

export function Details({ model, stats }: { model: string; stats: Stats | null }) {
  const { adapter } = splitModel(model)
  const lora = adapter ? stats?.loras?.find((l) => l.name === adapter) : undefined
  const { data: g, error } = useFetched<Graph>(withModel("graph", model))
  const { data: info } = useFetched<Info>(withModel("server_info", model))
  const [query, setQuery] = useState("")
  const [open, setOpen] = useState<Op | null>(null)
  const fam = useMemo(() => {
    const n = new Map<string, number>()
    for (const o of g?.ops ?? []) n.set(familyOf(o), (n.get(familyOf(o)) ?? 0) + (o.count ?? 1))
    const ranked = [...n.entries()].sort((a, b) => b[1] - a[1])
    return { ranked, hue: new Map(ranked.map(([f], i) => [f, i < 7 ? `var(--series-${i + 1})` : "var(--idle)"])) }
  }, [g])
  if (!g) return <Empty>{error ? "Details are available once the model is loaded" : "Loading"}</Empty>
  const m = g.model ?? {}, c = g.counts ?? {}, e = (info?.engine ?? {}) as Record<string, any>
  const facts: [string, string][] = [
    ["Architecture", m.arch ?? e.arch ?? "–"], ["Quantization", m.quant ?? "none"], ["Layers", fNum(m.n_layers)], ["Width", fNum(m.n_embd)],
    ["Vocabulary", fNum(m.n_vocab)], ["Context", fNum(info?.server?.max_model_len)], ["Sequences", fNum(m.max_seqs)], ["Accelerators", String(m.world && m.world > 1 ? m.world : 1)],
    ["Ops", fNum(c.ops)], ["Weights", fNum(c.weights)],
  ]
  if (g.drafter?.name) facts.push(["Drafter", g.drafter.name], ["Draft depth", String(g.drafter.depth ?? "–")])
  if (adapter) facts.push(["LoRA adapter", adapter], ["Runs on", splitModel(model).base], ["Served under it", fNum(lora?.requests)])
  if (e.tp_wire && (m.world ?? 1) > 1) facts.push(["Link between accelerators", String(e.tp_wire)])
  const mem: Part[] = ([["Weights", c.weight_bytes, 1], ["Activations", c.arena_bytes, 3], ["Host arena", c.host_arena_bytes, 5], ["Scratch", c.scratch_bytes, 6], ["KV cache", stats?.kv_bytes_total, 4]] as [string, number | undefined, number][])
    .filter(([, v]) => (v ?? 0) > 0).map(([label, v, i]) => ({ key: label, label, value: v!, color: `var(--series-${i})`, text: fBytes(v) }))
  const byIndex = new Map(g.ops.map((o) => [o.index, o]))
  const q = query.trim().toLowerCase()
  const ops = g.ops.filter((o) => !q || o.op.toLowerCase().includes(q) || (kernelOf(o)?.kernel ?? "").toLowerCase().includes(q) || o.geometry.toLowerCase().includes(q))
  const top = fam.ranked.slice(0, 7), most = top[0]?.[1] ?? 1
  const strip = (g.order ?? []).slice(0, 2400)
  const libs = (g.plugins ?? []).filter((p) => (p.kernels ?? 0) > 0)

  return (
    <div className="flex flex-col gap-4">
      <div className="overflow-hidden rounded-xl border bg-card">
        <div className="-mr-px -mb-px grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6">
          {facts.map(([k, v]) => (
            <div key={k} className="min-w-0 border-r border-b px-4 py-3">
              <div className="truncate text-xs text-muted-foreground">{k}</div>
              <div className="num truncate text-sm font-medium" title={v}>{v}</div>
            </div>
          ))}
        </div>
      </div>

      <ControlVector model={model} />

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Memory" description="Per accelerator, by use">
          <div className="flex flex-col gap-4">
            <PartBar parts={mem} className="h-3" />
            <div className="grid grid-cols-[auto_1fr_auto] items-center gap-x-3 gap-y-1.5 text-xs">
              {mem.map((p) => (
                <div key={p.key} className="contents">
                  <span className="inline-flex items-center gap-1.5"><i className="size-2 rounded-full" style={{ background: p.color }} />{p.label}</span>
                  <div className="h-1.5 overflow-hidden rounded-full bg-muted"><div className="h-full rounded-full" style={{ width: (100 * p.value) / Math.max(...mem.map((q) => q.value), 1) + "%", background: p.color }} /></div>
                  <span className="num text-right text-muted-foreground">{p.text}</span>
                </div>
              ))}
            </div>
          </div>
        </Panel>
        <Panel title="Kernels in use" description="Launches per pass, by kernel type">
          <div className="grid grid-cols-[auto_1fr_auto] items-center gap-x-3 gap-y-1.5 text-xs">
            {top.map(([f, n]) => (
              <div key={f} className="contents">
                <span className="truncate">{f}</span>
                <div className="h-1.5 overflow-hidden rounded-full bg-muted"><div className="h-full rounded-full" style={{ width: (100 * n) / most + "%", background: fam.hue.get(f) }} /></div>
                <span className="num text-right text-muted-foreground">{fNum(n)}</span>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      {strip.length > 0 && (
        <Panel title="Program" description={`The ${fNum(g.order?.length)} launches of one pass, in order, colored by kernel type`}>
          <div className="flex flex-wrap gap-px">
            {strip.map((idx, i) => {
              const o = byIndex.get(idx)
              return o ? <button key={i} title={o.op} onClick={() => setOpen(o)} className="h-3.5 w-[3px] rounded-[1px] opacity-85 hover:opacity-100 hover:outline hover:outline-foreground" style={{ background: fam.hue.get(familyOf(o)) }} /> : null
            })}
          </div>
        </Panel>
      )}

      <Panel title="Ops" description={`${fNum(ops.length)} of ${fNum(g.ops.length)} distinct`} flush
        action={<Input value={query} onChange={(ev) => setQuery(ev.target.value)} placeholder="Filter by op, kernel or geometry" className="h-8 w-64 max-w-[50vw]" />}>
        <div className="max-h-[28rem] overflow-auto">
          <Table>
            <TableHeader className="sticky top-0 z-10 bg-card">
              <TableRow className="hover:bg-transparent">
                <TableHead className="pl-4">Op</TableHead><TableHead>Kernel</TableHead><TableHead>Library</TableHead>
                <TableHead className="text-right">Launches</TableHead><TableHead className="pr-4">Geometry</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {ops.map((o) => {
                const k = kernelOf(o)
                return (
                  <TableRow key={o.index} className="cursor-pointer" onClick={() => setOpen(o)}>
                    <TableCell className="pl-4 font-medium"><span className="inline-flex items-center gap-2"><i className="size-2 rounded-full" style={{ background: fam.hue.get(familyOf(o)) }} />{o.op}</span></TableCell>
                    <TableCell className="font-mono text-xs">{k?.kernel ?? "–"}</TableCell>
                    <TableCell className="text-muted-foreground">{k?.plugin ?? "–"}</TableCell>
                    <TableCell className="num text-right">{fNum(o.count ?? 1)}</TableCell>
                    <TableCell className="max-w-[28rem] truncate pr-4 font-mono text-xs text-muted-foreground">{o.geometry}</TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </div>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        {libs.length > 0 && (
          <Panel title="Kernel libraries" description="In lookup order" flush>
            <Table>
              <TableHeader><TableRow className="hover:bg-transparent"><TableHead className="pl-4">Library</TableHead><TableHead>Version</TableHead><TableHead className="text-right">Kernels</TableHead><TableHead className="pr-4">Built for</TableHead></TableRow></TableHeader>
              <TableBody>
                {libs.map((p) => (
                  <TableRow key={p.name}>
                    <TableCell className="pl-4 font-medium">
                      <Tooltip><TooltipTrigger render={<span>{p.name}</span>} /><TooltipContent className="max-w-80">{p.description}</TooltipContent></Tooltip>
                    </TableCell>
                    <TableCell className="num text-muted-foreground">{p.version ?? "–"}</TableCell>
                    <TableCell className="num text-right">{fNum(p.kernels)}</TableCell>
                    <TableCell className="pr-4 text-muted-foreground">{p.build_target ?? "–"}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Panel>
        )}
        {(g.notes?.length ?? 0) > 0 && (
          <Panel title="Notes" description="Messages from the model plugin during setup">
            <ul className="flex flex-col gap-2 text-sm text-muted-foreground">
              {g.notes!.map((n, i) => <li key={i} className="border-l-2 pl-3 leading-snug">{n}</li>)}
            </ul>
          </Panel>
        )}
      </div>

      <Sheet open={!!open} onOpenChange={(v) => { if (!v) setOpen(null) }}>
        <SheetContent className="w-full gap-0 overflow-y-auto sm:max-w-xl">
          {open && <OpDetail op={open} />}
        </SheetContent>
      </Sheet>
    </div>
  )
}

/** The switch of a model that carries a control vector: whether requests get it unless they say. */
function ControlVector({ model }: { model: string }) {
  const [cv, setCv] = useState<{ default?: boolean; name?: string } | null>(null)
  const load = () => { api<{ default?: boolean; name?: string }>(withModel("admin/control-vector", model)).then(setCv).catch(() => setCv(null)) }
  useEffect(load, [model]) // eslint-disable-line react-hooks/exhaustive-deps
  if (!cv) return null
  const set = (on: boolean) => {
    setCv({ ...cv, default: on })
    api(withModel("admin/control-vector", model), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ enabled: on }) }).catch(() => undefined).finally(load)
  }
  return (
    <label className="flex items-center gap-3 rounded-xl border bg-card px-4 py-3">
      <span className="text-sm font-medium">Control vector</span>
      <span className="min-w-0 flex-1 truncate text-sm text-muted-foreground">{cv.name ?? ""}</span>
      <Switch checked={!!cv.default} onCheckedChange={set} />
    </label>
  )
}

function OpDetail({ op }: { op: Op }) {
  const k = kernelOf(op)
  const rows: [string, string | undefined][] = [
    ["Geometry", op.geometry], ["Launches per pass", fNum(op.count ?? 1)],
    ["Range", op.ranged ? `${op.ranged} from ${op.range_lo} to ${op.range_hi}` : undefined],
    ["Kernel", k?.kernel], ["Library", k?.plugin], ["Computes", k?.computes], ["Shapes", k?.shape], ["Types", k?.dtypes],
    ["Requires", k?.constraints?.length ? k.constraints.join("; ") : undefined],
  ]
  return (
    <>
      <SheetHeader>
        <SheetTitle>{op.op}</SheetTitle>
        <SheetDescription>{op.schema?.doc ?? "Op " + op.index}</SheetDescription>
      </SheetHeader>
      <div className="flex flex-col gap-4 px-4 pb-6">
        <dl className="grid grid-cols-[8rem_1fr] gap-x-3 gap-y-2 text-sm">
          {rows.filter(([, v]) => v).map(([l, v]) => (
            <div key={l} className="contents"><dt className="text-muted-foreground">{l}</dt><dd className="min-w-0 break-words font-mono text-xs leading-5">{v}</dd></div>
          ))}
        </dl>
        {(op.weights?.length ?? 0) > 0 && (
          <div>
            <div className="mb-1.5 text-xs font-medium text-muted-foreground">Weights</div>
            <div className="overflow-hidden rounded-lg border">
              {op.weights!.slice(0, 24).map((w) => (
                <div key={w.name} className="flex items-baseline gap-3 border-b px-3 py-1.5 text-xs last:border-b-0">
                  <span className="min-w-0 flex-1 truncate font-mono">{w.name}</span>
                  <span className="text-muted-foreground">{w.dtype} · {w.shape}</span>
                  <span className="num w-20 text-right">{fBytes(w.bytes)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {(op.bands?.length ?? 0) > 1 && (
          <div>
            <div className="mb-1.5 text-xs font-medium text-muted-foreground">By size</div>
            <div className="overflow-hidden rounded-lg border">
              {op.bands!.map((b) => (
                <div key={b.span} className="flex items-baseline gap-3 border-b px-3 py-1.5 text-xs last:border-b-0">
                  <span className="num w-28 text-muted-foreground">{b.span}</span>
                  <span className="min-w-0 flex-1 truncate font-mono">{(b.device ?? b.host)?.kernel ?? "–"}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  )
}
