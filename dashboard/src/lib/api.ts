// What the page asks the engine for. Everything it draws every second comes from /history, which
// the process answers from its sampler's last read: an open page reads nothing of a running model.
// /sessions, /graph and /server_info are asked for once, when their view is opened.

export interface Columns { [name: string]: (number | null)[] }
/** A model served through a runtime LoRA reads as `base:adapter`; the tabs slice by adapter. */
export const splitModel = (id: string): { base: string; adapter: string | null } => {
  const i = id.lastIndexOf(":")
  return i < 0 ? { base: id, adapter: null } : { base: id.slice(0, i), adapter: id.slice(i + 1) }
}
export interface LoRAStat { name: string; requests: number; prompt_tokens: number; output_tokens: number; running: number }
export interface FinishedRequest {
  id: string; prompt: number; out: number; ctx: number
  ttft: number | null; pf: number | null; tps: number | null; at: number; lora?: string
}
export interface HistoryBody<Live> {
  keep_s: number
  t: number[]
  c: Columns
  requests?: FinishedRequest[]
  live: Live | null
}

export interface FleetModel {
  id: string; state: string; file?: string; aliases?: string[]
  priority?: number; pinned?: boolean; tp?: number; ttl_ms?: number
  parks?: number; restores?: number; loads?: number; unloads?: number
  in_flight?: number; waiting?: number
  cards?: number[]; vram_bytes_per_card?: number; parked_bytes_per_card?: number
  /** What the model holds on each accelerator now, in any state (loading and partly parked too). */
  vram_bytes_by_card?: Record<string, number>
  host_bytes?: number; host_pinned_bytes?: number; drive_bytes?: number; last_restore_s?: number; load_s?: number
  doing?: string; waiting_for?: string; error?: string; warm?: boolean; in_place?: boolean
  park?: { ram?: string; restore?: string; vram?: string }
  capabilities?: Record<string, unknown>
  [k: string]: unknown
}
export interface FleetCard {
  card: number; name?: string; vram_bytes?: number; free_bytes?: number; busy?: number; power_w?: number; temp_c?: number; queues?: number; queues_idle?: number; queue_bytes?: number
  /** Each only where the driver reports it: memory busy as a ratio, clocks in MHz, the memory's
   * temperature beside the junction's, and the fan as a percentage of its full speed. */
  mem_busy?: number; sclk_mhz?: number; mclk_mhz?: number; temp_mem_c?: number; fan_pct?: number
}
export interface Fleet {
  version: string; uptime_s: number
  models: FleetModel[]; cards: FleetCard[]
  groups: { name: string; kind: string; members: string[] }[]
  host: { budget_bytes?: number; copies_bytes?: number; held_bytes?: number; disk_tier?: string }
  warm_up: { queued?: number; loading?: string | null; drive_only_bytes?: number }
}

// What the page reads of each route's object. Every field read is declared here (the compiler
// refuses any other), and the engine's tests check that the routes emit each one
// (tools/dashboard_embed.py copies these names into the compiled-in header).
export interface RunningRequest {
  id: string | number; state: string; prompt: number; computed: number; cached: number; output: number
  max: number; age_s: number; tps: number; lora?: string
}
export interface DraftPosition { reached: number; rate: number }
export interface ExpertTier { name: string; units: number; bytes: number }
export interface Experts { n_units: number; tiered?: boolean; tiers: ExpertTier[]; layer_units?: number[]; layer_resident?: number[] }
export interface Stats {
  model: string; uptime_s: number
  in_flight: number; max_num_seqs?: number
  jobs: { running: number; done: number; failed: number; total_ms: number; last_ms: number }
  running?: number; waiting?: number
  kv_bytes_used?: number; kv_bytes_total?: number
  // The pool in tokens: held now, what it holds at present, and the most it could hold if every block
  // of its carve (VRAM lent to the experts included) were the cache's.
  kv_tokens_used?: number; kv_tokens_total?: number; kv_tokens_max?: number
  // The step running now (0 prefill, 1 decode, 2 mixed; -1 none) and its age in seconds.
  step_open_phase?: number; step_open_s?: number
  prefix_hit_rate?: number; prefill_tps?: number; decode_tps?: number
  total_requests?: number; preemptions?: number
  step_ns_decode_best?: number
  draft_positions?: DraftPosition[]
  requests?: RunningRequest[]
  experts?: Experts
  loras?: LoRAStat[]
  // The vision encoder: images and patches encoded and the time spent on them (counters since the
  // model loaded), and the images waiting for it now.
  encode_items?: number; encode_patches?: number; encode_ns?: number; encode_pending?: number
}
export interface Session { id: number | string; label?: string; tokens: number; bytes: number; idle_s: number; on_device: number; on_host: number; moving: boolean; snapshots: number; resume_tokens: number; lora?: string }
export interface Sessions {
  sessions: Session[]; enabled?: boolean
  pool?: { tokens_used: number; tokens_total: number; bytes_used: number; bytes_total: number }
  occupancy?: { host_bytes: number; host_cap_bytes: number }
  linear?: { checkpoints_written: number }
  prefix_hit_rate?: number
}

let apiKey: string | null = null
try { apiKey = localStorage.getItem("radiance.key") } catch { /* storage may be blocked */ }
const listeners = new Set<() => void>()
let needKey = false
export const auth = {
  get needed() { return needKey },
  get key() { return apiKey },
  set(k: string) {
    apiKey = k || null
    try { if (k) localStorage.setItem("radiance.key", k); else localStorage.removeItem("radiance.key") } catch { /* as above */ }
    needKey = false
    listeners.forEach((f) => f())
  },
  subscribe(f: () => void) { listeners.add(f); return () => { listeners.delete(f) } },
}

export class HttpError extends Error {
  status: number
  constructor(status: number, message: string) { super(message); this.status = status }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = { ...(init?.headers as Record<string, string> | undefined) }
  if (apiKey) headers.Authorization = "Bearer " + apiKey
  const r = await fetch(new URL(path, document.baseURI), { cache: "no-store", ...init, headers })
  if (r.status === 401) {
    if (!needKey) { needKey = true; listeners.forEach((f) => f()) }
    throw new HttpError(401, "an API key is required")
  }
  if (!r.ok) {
    let msg = r.statusText
    try { const j = await r.json(); msg = j?.error?.message ?? msg } catch { /* not JSON */ }
    throw new HttpError(r.status, msg)
  }
  return (await r.json()) as T
}
export const withModel = (path: string, model: string) => path + (path.includes("?") ? "&" : "?") + "model=" + encodeURIComponent(model)
