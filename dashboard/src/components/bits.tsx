// Small pieces every view uses: a figure tile, a model's state, a bar of parts, a row of bars.
import type { ReactNode } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardAction, CardDescription } from "@/components/ui/card"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

export function Kpi({ label, value, unit, detail, className }: { label: string; value: ReactNode; unit?: string; detail?: ReactNode; className?: string }) {
  return (
    <div className={cn("flex min-w-0 flex-col gap-1 rounded-xl border bg-card px-4 py-3.5", className)}>
      <span className="truncate text-xs font-medium text-muted-foreground">{label}</span>
      <span className="num flex items-baseline gap-1 text-2xl leading-none font-semibold tracking-tight">
        <span>{value}</span>
        {unit && <span className="text-sm font-normal text-muted-foreground">{unit}</span>}
      </span>
      <span className="min-h-4 text-xs text-muted-foreground break-words [overflow-wrap:anywhere]">{detail}</span>
    </div>
  )
}

export function Panel({ title, description, action, children, className, flush }: { title?: ReactNode; description?: ReactNode; action?: ReactNode; children: ReactNode; className?: string; flush?: boolean }) {
  return (
    <Card className={cn("min-w-0 gap-3 py-4 shadow-none", className)}>
      {(title || action) && (
        <CardHeader className="px-4">
          <CardTitle className="text-sm font-medium">{title}</CardTitle>
          {description && <CardDescription className="text-xs">{description}</CardDescription>}
          {action && <CardAction>{action}</CardAction>}
        </CardHeader>
      )}
      <CardContent className={flush ? "px-0" : "px-4"}>{children}</CardContent>
    </Card>
  )
}

const STATES: Record<string, { label: string; tone: string }> = {
  resident: { label: "Loaded", tone: "var(--ok)" },
  loading: { label: "Loading", tone: "var(--warn)" },
  waiting: { label: "Waiting for room", tone: "var(--series-4)" },
  restoring: { label: "Restoring", tone: "var(--warn)" },
  closing: { label: "Unloading", tone: "var(--warn)" },
  parked: { label: "Parked", tone: "var(--series-1)" },
  cold: { label: "Not loaded", tone: "var(--idle)" },
  failed: { label: "Failed", tone: "var(--bad)" },
}
export const stateOf = (s: string | undefined) => STATES[s ?? ""] ?? { label: s || "Unknown", tone: "var(--idle)" }
/** The state behind a history value of an `s:<id>` column. */
export const STATE_OF_CODE: Record<number, string> = { 0: "cold", 1: "parked", 2: "loading", 3: "resident", 4: "failed", 5: "waiting" }

export function StateDot({ state, busy, className }: { state?: string; busy?: boolean; className?: string }) {
  const s = stateOf(state)
  return (
    <span className={cn("relative inline-flex size-2 shrink-0", className)} aria-label={s.label}>
      {busy && <span className="absolute inline-flex size-full animate-ping rounded-full opacity-60" style={{ background: s.tone }} />}
      <span className="relative inline-flex size-2 rounded-full" style={{ background: s.tone }} />
    </span>
  )
}
export function StateBadge({ state, doing }: { state?: string; doing?: string }) {
  const s = stateOf(state)
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium whitespace-nowrap" style={{ borderColor: `color-mix(in oklab, ${s.tone} 35%, transparent)`, background: `color-mix(in oklab, ${s.tone} 10%, transparent)` }}>
      <span className="size-1.5 rounded-full" style={{ background: s.tone }} />
      {state === "waiting" ? s.label : doing || s.label}
    </span>
  )
}

export interface Part { key: string; label: string; value: number; color: string; text?: string }
/** One bar divided among parts, in proportion, with what is left of `total` shown empty. */
export function PartBar({ parts, total, className }: { parts: Part[]; total?: number; className?: string }) {
  const sum = parts.reduce((a, p) => a + Math.max(0, p.value), 0)
  const whole = Math.max(total ?? sum, sum, 1)
  return (
    <div className={cn("flex h-2.5 w-full overflow-hidden rounded-full bg-muted", className)}>
      {parts.filter((p) => p.value > 0).map((p) => (
        <Tooltip key={p.key}>
          <TooltipTrigger render={<div className="h-full min-w-[3px] border-r border-background/60 last:border-r-0 transition-[width] duration-500" style={{ width: (100 * p.value) / whole + "%", background: p.color }} />} />
          <TooltipContent>{p.label}{p.text ? " · " + p.text : ""}</TooltipContent>
        </Tooltip>
      ))}
    </div>
  )
}
export function PartLegend({ parts, className }: { parts: Part[]; className?: string }) {
  return (
    <div className={cn("flex flex-wrap gap-x-4 gap-y-1 text-xs", className)}>
      {parts.filter((p) => p.value > 0).map((p) => (
        <span key={p.key} className="inline-flex items-center gap-1.5">
          <i className="size-2 rounded-full" style={{ background: p.color }} />
          <span className="text-muted-foreground">{p.label}</span>
          {p.text && <span className="num">{p.text}</span>}
        </span>
      ))}
    </div>
  )
}

/** Upright bars of equal width, one a value, against `max`. */
export function Bars({ values, max, color = "var(--series-1)", height = 120, tip, labels }: { values: number[]; max?: number; color?: string; height?: number; tip?: (i: number, v: number) => string; labels?: (i: number) => string | null }) {
  const top = Math.max(max ?? 0, ...values, 1e-9)
  return (
    <div>
      <div className="flex items-end justify-center gap-[3px]" style={{ height }}>
        {values.map((v, i) => (
          <Tooltip key={i}>
            <TooltipTrigger render={<div className="flex h-full max-w-14 min-w-0 flex-1 items-end rounded-sm bg-muted/60"><div className="w-full rounded-sm transition-[height] duration-500" style={{ height: Math.max(v > 0 ? 2 : 0, (100 * v) / top) + "%", background: color }} /></div>} />
            {tip && <TooltipContent className="whitespace-pre-line">{tip(i, v)}</TooltipContent>}
          </Tooltip>
        ))}
      </div>
      {labels && (
        <div className="num mt-1.5 flex justify-center gap-[3px] text-[10px] text-muted-foreground">
          {values.map((_, i) => <span key={i} className="max-w-14 min-w-0 flex-1 truncate text-center">{labels(i)}</span>)}
        </div>
      )}
    </div>
  )
}

export function Empty({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn("grid min-h-28 place-items-center rounded-lg border border-dashed px-4 py-8 text-center text-sm text-muted-foreground", className)}>{children}</div>
}
