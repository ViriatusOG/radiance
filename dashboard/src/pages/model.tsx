// One model: what it is and how it is doing, in four views.
import { Empty, StateBadge } from "@/components/bits"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import type { FleetModel } from "@/lib/api"
import { fBytes, fDur } from "@/lib/format"
import { modelScope, useScope } from "@/lib/store"
import { hrefOf, type Tab } from "@/lib/ui"
import { Details } from "./details"
import { HardwareSwitch, Performance, type Kind } from "./performance"
import { Requests } from "./requests"
import { SessionsView } from "./sessions"

const NOUN: Record<string, string> = { image_generation: "Images generated", shape_generation: "Meshes generated", audio_generation: "Clips generated", video_generation: "Videos generated" }
function kindOf(c: Record<string, unknown> | undefined): [Kind, string] {
  for (const k of Object.keys(NOUN)) if (c?.[k]) return ["gen", NOUN[k]]
  if (c?.chat) return ["chat", ""]
  if (c?.embedding) return ["embed", ""]
  return ["chat", ""]
}
const DOES: [string, string][] = [["chat", "Chat"], ["tools", "Tools"], ["reasoning", "Reasoning"], ["image", "Reads images"], ["video", "Reads video"], ["audio", "Reads audio"], ["embedding", "Embeddings"],
  ["image_generation", "Makes images"], ["image_edit", "Edits images"], ["audio_generation", "Makes audio"], ["video_generation", "Makes video"], ["shape_generation", "Makes meshes"]]

export function ModelPage({ id, tab, model }: { id: string; tab: Tab; model: FleetModel | undefined }) {
  const scope = useScope(modelScope(id))
  const [kind, noun] = kindOf(model?.capabilities)
  const tabs: [Tab, string][] = kind === "chat" ? [["performance", "Performance"], ["requests", "Requests"], ["sessions", "Sessions"], ["details", "Details"]] : [["performance", "Performance"], ["details", "Details"]]
  const shown = tabs.some(([t]) => t === tab) ? tab : "performance"
  const s = scope.live
  const serving = model?.state === "resident" && !!s
  const slots = typeof s?.max_num_seqs === "number" ? s.max_num_seqs : undefined
  const where = model?.cards?.length ? `accelerator${model.cards.length > 1 ? "s" : ""} ${model.cards.join(" and ")}` : null
  return (
    <div className="flex flex-col gap-4 p-4 md:p-6">
      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <h1 className="text-xl font-semibold tracking-tight">{id}</h1>
          <StateBadge state={model?.state} doing={model?.doing} />
          <span className="text-sm text-muted-foreground">
            {[model?.file, where, model?.state === "resident" && model.vram_bytes_per_card ? fBytes(model.vram_bytes_per_card) + " per accelerator" : null, s?.uptime_s ? "up " + fDur(s.uptime_s) : null].filter(Boolean).join(" · ")}
          </span>
        </div>
        {model?.error && <div className="rounded-lg border border-bad/40 bg-bad/10 px-3 py-2 text-sm">{model.error}</div>}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Tabs value={shown} onValueChange={(v) => { location.hash = hrefOf(id, v as Tab) }}>
            <TabsList>{tabs.map(([t, label]) => <TabsTrigger key={t} value={t} className="px-3">{label}</TabsTrigger>)}</TabsList>
          </Tabs>
          <div className="flex flex-wrap gap-1.5">
            {DOES.filter(([k]) => model?.capabilities?.[k]).map(([k, label]) => <span key={k} className="rounded-md border px-1.5 py-0.5 text-xs text-muted-foreground">{label}</span>)}
            {model && shown === "performance" && <HardwareSwitch />}
          </div>
        </div>
      </div>
      {!model ? <Empty>This process serves no model named {id}</Empty>
        : shown === "performance" ? (
          <>
            {!serving && scope.series.t.length > 0 && <div className="text-sm text-muted-foreground">This model is not loaded right now. The charts show its earlier activity.</div>}
            {scope.series.t.length === 0 && !serving ? <Empty>{model.state === "failed" ? "The model failed to start, so there is no data" : model.state === "cold" ? "This model has not been loaded yet. It loads when a request asks for it." : "No data yet. Charts appear once the model has been loaded."}</Empty>
              : <Performance scope={scope} kind={kind} slots={slots} noun={noun} model={id} cards={model.cards} />}
          </>
        )        : shown === "requests" ? <Requests scope={scope} model={id} />
        : shown === "sessions" ? (serving ? <SessionsView model={id} /> : <Empty>Conversations are listed while the model is loaded</Empty>)
        : <Details model={id} stats={s} />}
    </div>
  )
}
