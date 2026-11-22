# Multi-model serving

One radiance process serves any number of models behind one HTTP endpoint. A YAML file names the
models; requests pick one by their `model` field; the process loads, parks and unloads models as
requests arrive, moving each model's weights between VRAM, pinned host RAM and disk. Models that fit
side by side stay resident together and run concurrently.

This document is the design. Source comments cite its section numbers. `spec.md` is still the engine
specification; this document says where multi-model serving changes it.

---

## 1. Decisions

| Question | Answer |
|---|---|
| Workload | Mixed: co-resident throughput and fast swaps both matter. |
| Process model | **One process, many `Engine`s.** No worker processes (§3.1). |
| Model kinds | Every kind radiance serves: text and vision LLMs, embedders and rerankers, image, video, speech, music and shape generators, ASR. |
| Configuration | **YAML only.** The command-line flags are removed from the server binary; their names live on as YAML keys (§4). Ships as 2.0.0. |
| Hot reload | Yes: SIGHUP or `POST /admin/reload` re-reads the file; changed models drain and reload, the rest keep serving (§10). |
| Memory | Each model has a **fixed** part and **flex** parts. All flex (KV blocks, offloadable experts) draws from one shared pool per card (§6). |
| Flex reclaim | **Strict priority.** A higher-priority model takes flex from a lower one even while it is busy; the victim's running sequences are preempted and recompute when they resume (§6.4). A tiered MoE's expert flex goes back first, to any model (§6.7). |
| Eviction | Pinned models never leave; then lower priority first; ties by least-recently-used weighted by reload cost; optional idle TTL (§8.3). |
| Parking | Per model: what stays in VRAM, and whether a RAM copy is required, allowed or never kept. RAM copies beyond that are kept opportunistically and dropped first under pressure (§7). |
| Cards | The YAML gives each model's TP degree; the scheduler picks the cards (§8.4). |
| Contention | A model leaving VRAM is drained first, bounded by a max wait. Flex reclaim preempts (§6.4). |
| Warm-up | Every model loads once at start, lowest priority first, giving way to requests (§8.6). A parked model can serve while its weights move back (§7.4), and an embedding model serves in place when its cards lack room (§7.5). |
| Shared cards | Concurrent queues: measured against time-sliced steps, which lost everywhere and were removed (§9). |
| Load wait | Requests wait up to a timeout, then 503; streaming clients get SSE keepalive comments meanwhile (§8.2). |
| Adapters | Addressed as `base:adapter` (§5.3). |
| Groups | `together` and `exclusive` co-residency rules (§8.5). |
| Hardware | Discrete multi-card hosts (two R9700 on the test host, §13), unified-memory APUs (gfx1151), and arbitrary public machines. |

---

## 2. Terms

- **Fleet**: the process-level owner of every model, card and budget. One per process.
- **Model entry**: one `models:` item in the YAML. It has a name, a config and a state.
- **Engine**: today's `rad::Engine`, one per *loaded* model entry. It owns its TP ranks, pools,
  scheduler and step loop.
- **Card ledger**: the per-card account of VRAM: who holds what, fixed or flex, committed or only
  reserved.
- **Fixed**: the bytes a model cannot serve without: non-expert weights, the activation arena, its
  queues and staging, and a floor of KV (one sequence at its context bound unless the YAML says
  more).
- **Flex**: bytes that make a model faster or wider but that it can give back while serving: KV
  blocks above the floor, and expert-slab slots above the planner's floor.
- **Park**: take a model out of VRAM but keep it ready to come back fast: its RAM copy, its compiled
  state and its address reservations stay.

---

## 3. Process architecture

### 3.1 Why one process

Process-per-model (the llama-swap shape) was the first idea. It loses on three points:

1. **Queue teardown on a busy card.** On RDNA4, a process exiting while another process keeps the
   same card busy can make the MES firmware fail `REMOVE_QUEUE`; the driver then mode1-resets the
   card and every process on it loses its VRAM. Unloading by exiting a process is exactly that path.
2. **Shared memory accounting.** Flex lending between models and same-address parking (§7.2) need one
   owner of each card's VRAM. Across processes that owner would need IPC handles and a VRAM broker.
3. **Fast swap.** A worker process has to redo plugin loading, kernel selection and code-object loads
   on every cold start. In one process they are done once per architecture.

The price: a GPU memory fault, which is not recoverable in-process (spec §17), ends every model at
once. The container's restart policy brings the fleet back (§11).

### 3.2 Shape

```
                         +-------------------------------------------+
  HTTP  ---->  Front     |  Fleet                                    |
              (one port, |   - model entries + state machine (§8.1)  |
               auth,     |   - router: model field -> entry (§5)     |
               routing)  |   - load queue, eviction, groups (§8)     |
                 |       |   - card ledgers + flex pools (§6)        |
                 |       |   - host RAM ledger, park store (§7)      |
                 v       +-------------------------------------------+
          per-model Front state          |  owns
          (limits, tokenizer,            v
           template, metrics)     Engine A (tp 2, cards 0,1)   Engine B (tp 1, card 1) ...
                                    rank threads + stepper        rank threads + stepper
```

Every engine keeps its own scheduler, step loop and rank threads, as today. The fleet adds no work to
the step path: it changes budgets between steps through the hooks the engines already poll
(`loan_tick`, `tier_tick`) and it drains and parks engines through explicit calls.

### 3.3 Process-wide state that must become per-engine

What a second `Engine` in the process needed made per-engine, by fix:

| State | Where | Fix |
|---|---|---|
| Kernel registry: load, select log, device facts, instances; `~Engine` closes it for everyone | `core/build/startup.cpp`, `core/plugin/registry.h`, `loader.cpp`, `select.cpp`, `engine.cpp` `~Engine` | Split into a **loaded set** (process service, loaded once, never closed while an engine lives) and a per-engine **selection context** (device facts, code device, log, instances to fini). |
| Arch plugin model state `g_model[MAX_RANKS]`, indexed by TP rank (30 plugins) | `arch/*/*.cpp`, `arch/common/rad_arch.h` | ABI: `RadBuildCtx::slot` and `rad_slot(RadCtx*)`, a process-unique instance slot. Plugins index their state by slot, not rank. A plugin that does not set `RadArchProbe::multi_instance_ok` may back one engine only; the fleet refuses a second engine on it by name. |
| libr4d all-reduce rendezvous `g_rv` and instance cache `g_cache`, keyed without engine identity | `libr4d/r4d_ar_entry.hip` | Kernel `init` receives a `group` parameter (process-unique TP group id) through the existing `RadParam` list; rendezvous and cache are keyed by it. No signature change. |
| libr4d/libr3d split-counter and scratch caches allocated with `hipMalloc`, never freed, some shared across streams | `libr4d/*.hip`, `libr3d/*.hip` | Keyed by stream, released by the kernel's `fini` when the owning engine goes; `s_scratch_pool` becomes per stream. |
| `hc_mix_ctr` four slots per device; `g_rg_blocks` one block per device | `libr4d/r4d_hc_bf16.hip`, `libr4d/r4d_moe.hip` | Pool per device with release on `fini`. |
| Device fault flag, single fault signal, global stall verdict | `core/device/aql.cpp` | Fault domain per card set; stall verdict per engine. A memory fault still ends the process (§11). |
| DMA link gate `held` flag per card | `core/device/aql.cpp` `a_link_hold` | Refcounted holders. |
| Per-thread HSA interrupt signals never destroyed | `core/device/aql.cpp` `thread_signal` | RAII `thread_local` owner. |
| Log rank tags global; no model tag | `core/util/rad_util.cpp` | `thread_local` model tag set on each engine's threads; lines read `[model/r0]`. |
| StallWatch: one last-line clock, one report slot | `core/util/stallwatch.cpp`, `rad_util.cpp` | Per-watch beats; a mutex around the report. |
| `g_signalled` + `sigaction` per `serve()` | `core/engine.cpp` | One process signal hub owned by the fleet. |
| Server built on `run_server`'s stack, one per engine | `core/engine.cpp`, `core/server/*` | One `Front` (§5); engines expose their bridges as objects. |
| `calib_staging` per-rank pinned map, never freed | `arch/common/rad_arch.h` | Keyed by slot, freed on close. |

---

## 4. Configuration

### 4.1 One file

```
radiance --config /etc/radiance/fleet.yaml
radiance --config fleet.yaml --check        # parse, validate, plan; serve nothing
```

The server binary takes no other flags (`--help` prints the key reference). The graph dump
(`debug-graph: true`) and the offline modes run one model from a one-model file (§4.4). Every former flag is a YAML key with the same name and no
leading dashes, at the level it belongs to (§4.3). Values keep their meaning, units and validation:
the YAML loader feeds the existing parser, so a key and its flag can never drift.

### 4.2 Example

```yaml
# fleet.yaml
server:
  host: "::"
  port: 8000
  api-key-env: RADIANCE_API_KEY

cards: all                 # or a list of card indices; models never leave this set
memory:
  gpu-headroom-mib: 160    # per card, as before
  card-headroom-mib: {0: 4096}        # card 0 (the desktop's) keeps 4 GiB free instead
  host-mib: 40960          # pinned host RAM the fleet may hold: parked weights, KV host tier, pools
  disk-tier: /var/cache/radiance      # prefix-cache disk tier and park files

defaults:                  # merged under every model; a model's own keys win
  kv-cache-dtype: i8
  max-model-len: 200000

models:
  qwen:
    model: /models/qwen3.8-27b-iq8r-dflash.rad
    aliases: [default]
    tp: 1
    priority: 100
    pin: true              # never leaves VRAM
    max-num-seqs: 8
    flex:
      kv: { min-tokens: 200000, weight: 1 }
    park: { ram: required }

  flashnext:
    model: /models/qwen3.8-next-flash-iqr-v4.rad
    tp: 2
    priority: 50
    placement: expert_tiered
    num-speculative-tokens: 3
    park: { vram: none, ram: allowed }
    ttl: 30m

  flux:
    model: /models/flux-3-klein-iq8r.rad
    tp: 1
    priority: 10
    park: { ram: never }   # reload from disk; never hold RAM for it

  embed:
    model: /models/embeddinggemma2.rad
    tp: 1
    priority: 60
    max-num-seqs: 64
    cards: [1]             # only ever on card 1, e.g. to keep it off the card that drives a display

groups:
  rag: { together: [qwen, embed] }          # load embed whenever qwen loads, and the reverse
  big: { exclusive: [flashnext, flux] }     # never resident at the same time

scheduler:
  load-timeout: 120s       # a request waits this long for its model, then 503
  drain-timeout: 30s       # a model being evicted gets this long to finish its requests
  keepalive: 10s           # SSE comment interval for streaming requests waiting on a load
  stuck-exit: 60s          # a device that advances nothing this long ends the process; 0 never
  warm-up: true            # fill free VRAM, then host RAM, with models at start (§8.6)
```

### 4.3 Levels

| Level | Keys |
|---|---|
| `server:` | `host`, `port`, `api-key`, `api-key-env`, `allow-host`, `trusted-proxy`, `http-threads`, `read-timeout`, `write-timeout`, `keep-alive-timeout`, `max-body-mib`, `retry-after`, `no-cors`, `live`, `log` (`info`/`debug`/`trace`), `radiance-home`, `kernels`, `download-dir`, `proceed-with-incompatible-configuration`, `debug-accept-reference-kernels` |
| `cards:` | `all` or a list of card indices |
| `memory:` | `gpu-headroom-mib`, `gpu-memory-mib` (per card), `card-headroom-mib` (`{CARD: MiB}`, a card's own reserve), `host-mib`, `host-headroom-mib`, `disk-tier` |
| `scheduler:` | `load-timeout`, `drain-timeout`, `keepalive`, `stuck-exit`, `warm-up` |
| `defaults:` / `models.<name>:` | every per-model flag: model source, TP, placement, KV, scheduler, spec decode, LoRA, request defaults and limits, media limits, offline modes, debug; plus the fleet keys `aliases`, `priority`, `pin`, `ttl`, `flex`, `park`, `cards` |
| `groups:` | `together` / `exclusive` lists |

A key at the wrong level is refused with the level it belongs to. An unknown key is refused with
the closest known key. `served-model-name` is gone: the entry's name is the model id, and `aliases`
adds more.

### 4.4 Single-model use

A one-model YAML is the only way to run one model too. `radiance --model X ...` is refused with a
message that prints the equivalent YAML, so a migration is copy and paste
([`GUIDE.md`](GUIDE.md) §7.3 covers what else a 1.x deployment changes). Offline modes (KL
recording, calibration, the diagnose workload) run from a one-model YAML with the `kld-*` /
`calib-*` / `diagnose` keys on that model;
the process does that job and exits, as today.

### 4.5 Provenance

The startup report keeps its per-option rows. Each row's source is the YAML path it came from
(`models.qwen.max-num-seqs`, `defaults.kv-cache-dtype`) or `auto` for a decision. Global rows print
once; each model's rows print under its name.

---

## 5. The front

### 5.1 Split

`ServerOptions` splits into **ListenOptions** (process: bind, auth, hosts, proxies, CORS, timeouts,
body size, threads) and **ModelFront** (per model: limits, default sampling, template kwargs,
tokenizer, chat template, grammar compiler, modality bridges, metrics, queue). `Deps` moves into
ModelFront. The bridges that `Engine::run_server` builds on its stack become engine-owned objects
the front borrows while the model is loaded.

### 5.2 Routing

1. The router reads the body's `model` before any parsing; multipart bodies (transcriptions) read the
   form field. The parsers then run against that model's `ModelFront`.
2. `model` resolves through names and aliases. A request without `model` goes to the model named
   `default` (by name or alias); with no such model it is refused.
3. A modality route checks the resolved model serves that modality (an image request to a text model
   is a 400 naming what the model serves), so `/v1/images/generations` with no `model` resolves to
   the only image model when there is exactly one.
4. The resolved entry is **acquired**: the request holds a reference that keeps the engine from being
   parked under it. If the entry is not resident, the request waits on the load (§8.2).

A finished video or shape is answered by the front itself, by its id: the model that made it may be
parked or unloaded the moment it is done, and a poll or a download is no reason to bring it back. It
stays until it expires, as a video kept by its server does.

### 5.3 Adapters

LoRA adapters are addressed `base:adapter`. `/v1/load_lora_adapter` takes `base` explicitly. The
`:` is reserved in model names and aliases. A model's control-vector variant
(`control-vector-variant`, GUIDE §4.10) is addressed and listed the same way, `base:NAME`.

### 5.4 Fleet-wide routes

- `/v1/models` lists every entry and alias with `state` (`resident`, `loading`, `closing`, `parked`,
  `cold`, `failed`), `kind`, `tp`, `cards`, and its adapters as `base:adapter` children.
  `/admin/fleet` adds `waiting` for a model that waits for room (§8.1).
- `/health` is the process (200 once the front listens); `/health?model=NAME` is that model.
- `/metrics` carries a `model` label on every per-model series, plus fleet series: per-card
  committed fixed/flex bytes, parked bytes, loads, parks, evictions and their durations.
- `/stats`, `/sessions` and the dashboard take `?model=`; the dashboard grows a fleet page.
- `/admin/models/NAME/{load,park,unload,pin,unpin}`, `/admin/reload`.

---

## 6. Memory: fixed, flex and the card ledger

### 6.1 Today

Each engine claims its card once: `vram_budget_resolve` splits the free VRAM into a weights pool
(one `hipMalloc`), a KV pool (a VMem reservation committed whole), and the activation arena (one
allocation). Pools are bump-allocated and never shrink. The mover lends unused KV and arena bytes to
the expert slab *at addresses the lender already owns* (`core/place/mover.h`). That lending is the
prototype of flex: it has one borrower (experts) and one model.

### 6.2 Address reservation, physical commit

Every pool of every engine becomes a VMem reservation sized for the engine's *largest* configuration.
Physical memory is committed in granules from the card's ledger and decommitted back to it. Virtual
address space is cheap and stays reserved for the model's whole life, loaded or parked, so:

- **parking keeps addresses**: weights decommitted and later recommitted land at the same device
  addresses, so weight tables, residency tables, recorded tapes and plugin pointers stay valid. A
  restore is commit plus copy, never a re-publish or re-declare;
- **flex is real**: a KV pool or expert slab grows by committing granules at its top and shrinks by
  decommitting them, and the bytes go to another model on the same card.

The fleet requires VMem on every card it shares between models. Where the driver lacks it, a card
can host exactly one model with today's static pools, and the startup report says so; there is no
silent fallback.

### 6.3 Ledger

Per card, the ledger holds `capacity − headroom − already_held` and accounts for every committed
granule as one of:

- **fixed(m)**: model m's fixed part, committed while m is resident;
- **flex(m)**: model m's flex above its floor;
- **transit**: staging for a load or park in flight.

A model is admitted to a card set when `fixed(m) + floor(m)` fits beside the resident models' fixed
parts and floors. Flex is never a reason to refuse a load; it is reclaimed (§6.4).

The flex floor and weight come from the YAML `flex:` block (`kv: {min-tokens, max-tokens, weight}`,
`experts: {min-mib, max-mib}`). The default floor is one sequence at the model's context bound, which
is today's KV minimum.

That bound is worth setting for an image generator. Unset, it is the generator's largest pass
(Qwen-Image 2.1: 33600 rows), so the floor is that many rows of conditioning keys and values, 8.2 GiB
a card at `tp: 2`, which every restore has to find room for. Text-to-image prompts need a few hundred
rows, and `max-model-len: 2048` takes the floor to about 0.5 GiB a card. Set it to cover the longest
conditioning the deployment sends: a 1024 x 1024 reference picture adds about 4600 rows. In the
§13.1 fleet (text-to-image at 768 x 768) this took the sequential plan from 89.8 to 81.9 s.

### 6.4 Reclaim by strict priority

When a model needs granules (a load, or its own flex growth), the ledger takes them in this order:

1. uncommitted capacity;
2. flex of models with **lower priority**, lowest first, idle models before busy ones at equal
   priority;
3. eviction of a lower-priority model's fixed part (§8.3).

A model never takes flex from an equal or higher priority model; it waits for them to release it.
Nor does it displace a higher-priority model, even an idle one (§8.3), so a model whose room only
higher-priority models hold waits out `load-timeout`: with priorities on the small models of a
fleet, MiniMax-Music3 and Hunyuan3D-2 did. A fleet in which every model should be reachable gives
its entries no `priority`, and the least recently used gives way.
The one exception is expert flex (§6.7): it grew into memory nobody held, and giving it back costs
an address change per unit, so any model's room comes out of it first, whatever its priority.

Taking flex from a model:

- **expert slots**: units in reclaimed slots drop to their host copy (`Mover::reclaim`, as the KV
  lender does today). This is an address change in the residency table, no copy;
- **free KV blocks**: decommitted directly. The block free list is a min-heap, so free blocks
  already sit at the top of the pool where decommit reaches them;
- **held KV blocks**: the sequences holding them are **parked**: the scheduler stops them at a step
  boundary, publishes their computed prefix into the prefix cache, the idle tier writes those blocks
  to pinned host (`TierExec` gather, outside the scheduler lock), and the VRAM blocks are freed. The
  sequence goes back to the queue head; on resume, admission restores the prefix from host like any
  cache hit and only the uncached tail is recomputed. Linear-state models (GDN, Mamba) park through
  the same path with a checkpoint snapshot at the parked boundary.

This replaces recompute-only preemption for reclaim. A prefix-cache hit already restores a host
entry in one batched pass per chain, so a parked sequence costs one D2H and one H2D of its own
blocks.

As built, the sequences holding reclaimed blocks are preempted and recompute when they resume
(`KVManager::set_flex`'s preemption); parking them through the prefix cache is not done (§12).

A reclaim takes from the prefix cache only what the floor lets the fleet have: a cache at its floor
gives nothing however empty it is, so it keeps its conversations. Entries go least recently used
first and stay restorable from their host copies.

### 6.5 Host RAM ledger

One pinned host budget (`memory.host-mib`) covers: parked weight copies (§7), the KV host tier, the
expert host pools of tiered models, and the transit buffers. Unset, it is three quarters of the
host's memory, or of the tightest memory cgroup the process sits in (a container's `--memory`),
whichever is less. The room a load or park may take at a moment is the host's available memory, or
less where such a cgroup leaves less, and a park never leaves less than 8 GiB of it. The KV host tier is shared between
models under the same eviction order as parked weights, so RAM goes to whatever is most likely to be
needed next. Host pinned memory is locked in pieces (`hsa_amd_memory_lock_to_pool`), so the ledger
grants and returns it in pieces too.

A load counts what it will pin before it waits for its VRAM: its pools, the tables placed in host
memory and the buffers startup pins beside them. Other loads and parks leave that room alone until
the load ends, and the load checks once more before it pins. An embedding left to its default goes on
the card when the host cannot hold it, and the VRAM ask includes it then; a placement asked for that
cannot be had fails the load, since pinning past the host's floor gets the process killed by the
kernel.

A load that waits for that room logs `model NAME: waiting for N of host memory to pin (...)` and
shows as `waiting` with `waiting_for: "host memory"` (§8.1). Once every other model is unloaded
and none is coming or going, nothing is left to give memory back, so it logs
`... is not there with every other model unloaded ...` and gives up the wait instead of sleeping
until `load-timeout`.

### 6.6 Unified memory (gfx1151)

On an APU the card ledger and the host ledger are one pool (the budget code already charges pinned
host memory to the VRAM ledger on unified devices). Parking to RAM frees nothing there, so a parked
model keeps its pages and only its *flex* is reclaimable; eviction goes straight to disk. The ledger
reads unified-ness from the device and reports it.

### 6.7 Expert flex

A tiered MoE's claim holds what the host pool cannot (Flash Next on the test host of §13: 23.65 GiB
a card of the 35.97 GiB it could use), and routing reads every other expert over the link or
promotes it into a slot first. The mover already fills slots the KV cache and the activation arena
lend; the fleet is a third lender (`Mover::kLendFleet`):

- at start each rank reserves address space for one slot per pooled expert (`Mover::lendable_bytes`,
  no VRAM) and carves it top down like the arena's top, in bytes;
- while the engine is busy (a step in the last 5 s), the step loop asks the broker for a GiB more
  (`flex_grow`: free past the headroom, nothing loading, restoring or waiting for room), backs it on
  every rank and raises the loan; the next GiB waits until the slab has filled what it holds, so no
  VRAM sits empty;
- any model needing room takes it back first (`reclaim_flex`'s first pass, any priority): the units
  past the new loan drop to their host copies between steps and the granules are unbacked. The slab
  does not grow again for 5 s, so the room is the taker's. A park recalls all of it and unbacks it,
  so nothing of it is copied;
- `flex.experts.max-mib` bounds the experts' VRAM on a card, the claim's share included; `0` keeps
  the slab at its claim.

When a busy slab finds no free memory (`expert_flex_wanted`), a neighbour on its cards that has been
idle for at least 30 s and thirty times its way back parks just what the slab wants (one neighbour
at a time, each at most once a minute; not a pinned model, not one of higher priority, not one that
may not park in RAM, not a together partner). Its next request restores it and takes the room back
from the slab first. The bar is that high because the neighbour's next request pays a restore, and
the slab's gain (§13.2) is a few percent at one client, more under load.

---

## 7. Parking and restore

### 7.1 What a model holds

| Piece | Resident | Parked | Cold |
|---|---|---|---|
| Arch selection, kernels, code objects, tapes, tokenizer, template | kept | kept | dropped (cheap) |
| VA reservations | kept | kept | released |
| Non-expert weights | VRAM | unmapped; read again from the container | disk |
| Experts | slab + host pool | slab unmapped, read again from the container; host pool kept | disk |
| KV pool | committed | decommitted, and the prefix cache reset: a conversation's next turn prefills again | released |
| Arena, staging, queues | committed | decommitted; queues kept | released |

Queues are kept across a park and an unload: destroying a HW queue next to a busy card is the risky
path (§3.1), so a closed stream's queue stays open for the card's next stream. An open queue holds
about 41 MiB of its card (the context save area), and opening one on a full card can stall the
process for minutes, so the fleet opens queues only while a card has room: eight to a card at the
start, and before each load as many as bring the kept ones back to eight, up to 24 open on a card.
Kept queues are closed in two places, both where no other model has anything on the card: down to
eight when an unload leaves the card empty, and down to four for a model that is short of exactly
their memory after every other model has left (MiniMax-H3 on a 32 GB card). `/admin/fleet` gives
each card's `queues` and `queues_idle`.

### 7.2 Park

An engine's large device allocations (`rad_dev_alloc` of `kPageMin` and up inside an engine's
scope) are reserved address space mapped in 64 MiB chunks (`core/device/paging.cpp`), so a park takes
off exactly what the room needs, card by card: a partial park unmaps only the chunks it takes and the
model stays parked with the rest still on its cards; a later park takes more ("parked further").

1. Stop admitting; drain running requests up to `drain-timeout`; hold the step loop.
2. Recall the expert slab's loans; drop slab units that have another copy; mark the KV cache, empty
   slab stretches and the activation arena above its fixed region (transients, dead between steps)
   as having nothing to keep (`dev_page_discard`).
3. Page out, cheapest chunks first: nothing to keep (unmapped), the weights pool (unmapped: its
   container is its copy, see below), a copy that is still exact (unmapped, or swapped), a copy
   exact but for a few dirty blocks (only those copied), then copies: to host memory while the host
   budget lasts, then straight to the disk tier (O_DIRECT, four writers). A chunk of any other
   allocation the load filled verbatim from the container is first read back and compared with it;
   when it matches, nothing is written and the container is its copy.

**Weights come back from the container, never from a copy.** A model loaded from a container marks
its weights pool as refilled by its owner (`dev_page_refilled`): a park unmaps the pool's chunks
with nothing copied, to host memory or to disk, and a restore maps them back empty and hands them to
the engine, which reads every weight with a byte in them from the container again and lays it out
for its kernels at its current address (`Engine::refill_weights`): the load's own sweep and
relayout pool, for those weights only, every write clipped to the memory paged back empty, so the restore
of a fully parked model costs about what its load's read did. The refill holds up to
`Engine::kRefillHost` (4 GiB) of host memory while it runs (staged inputs, output buffers, the
sweep's pinned windows), which the fleet keeps room for before a restore. An expert unit the mover moved into a slab slot is read for that slot (its
residency), and a unit the park dropped from the slab is not read at all. Nothing is written to the
drive for weights, and nothing waits for such a write. A model under runtime adapters (whose
pool is written after the load) keeps the copies described below.

**Host copies are mappings.** Every pageable allocation has a shadow reservation of its size. A
chunk's host copy is pinned host memory mapped at an address (`vmem_commit_host`): at the shadow
while the chunk is on the card, at the chunk's own address while it is out. A park copies the chunk
to the shadow and swaps the two mappings (`vmem_swap`); a chunk whose copy is kept leaves by the
swap alone; a restore copies back through the shadow and swaps again. A parked chunk therefore stays
mapped where the kernels expect it, readable and writable by the card over the link (§7.4).

Copies stay exact across parks for **sealed** memory. The mover calls `dev_page_invalidate` for
every slab slot it writes, which marks the 64 KiB blocks it touched dirty: the copy stays, exact
elsewhere, and the next park rewrites only those blocks.

The `park:` block says what stays:

- `vram:` `none` (default), `statics` (non-expert weights stay), or a size in MiB of the most
  first-used weights to keep (the head of the first-use order, which is what progressive warm-up
  needs first);
- `ram:` `required` (refuse a park that cannot hold a RAM copy; evict others' opportunistic copies
  for it), `allowed` (keep one while RAM is free, drop it first under pressure; the default), or
  `never`; a copy of what the park copies, never of the weights, which come from the container;
- `restore:` `auto` (the default: `full`, but a one-step model comes back in place when its cards
  lack the room, §7.5), `full` (every byte back on the card before it serves) or `live` (§7.4).

Park or unload: a park that would write more to disk (and read it back) than the model's measured
load takes, or a model whose last page-in plus what this park would write takes more than 1.2x its
load, unloads instead (a model whose container sits in the page cache reloads faster than it
round-trips the disk tier). Waits for room or for requests to drain are not counted: a load waits
for those as well.

### 7.3 Restore

Restore maps device memory for every chunk that is out, at its old address or (for a chunk read in
place from its host copy) at its shadow, then copies back: host copies at once and swapped in, disk
and container copies through one pipeline (eight readers, a ring of sixteen pinned buffers, each
freed when its own copy lands; the drive's rate, 6.9 GB/s on the test host). The weights pool's
chunks come back empty, and the engine reads the weights in them from the container again (§7.2)
before the model serves. Kernels, tapes and tables are untouched because the addresses are.

**Remapping an address needs a full translation flush first.** On kernel 7.1 with gfx1201, copies
into memory mapped again at an address that was mapped before landed nowhere: the chunk read back
its previous owner's bytes or zeros (every disk-tier restore came back corrupted). A map flushes the
cards' translation caches only lightly; unpinning host memory makes the driver flush them in full,
so `dev_settle_maps` pins and unpins one page after a batch of maps (page-in, VMem commits) before
anything writes them. `tests/device_test.cpp` covers the remap, host copies, the disk tier, many
chunks, several allocations of one owner, partial parks and container-backed chunks; each fails
without the flush. `granules_remapped_at_random_never_share_a_page` maps and unmaps granules at
random, so freed pages come back under other addresses: without the flush two live granules shared a
page within 14 to 32 cycles in every run; with it, 23 runs of 3000 cycles found nothing, 20 of them
beside a process polling `rocm-smi` and opening the render nodes, which makes the alias more
frequent.

### 7.4 Serving while it comes back (`park.restore: live`)

With `park: {restore: live}` a restore serves at once (`dev_page_in_live`): everything but the
sealed chunks parked in host memory comes back as in §7.3, and those stay mapped at their own
addresses, read in place over the link. Behind the first requests they move onto the card
(`dev_page_promote`): each is copied to device memory at its shadow while the engine runs, then
the mappings are swapped in a quiet window, a step boundary with every rank's streams drained
(`Engine::run_quiet`); a chunk the engine wrote meanwhile is copied again inside the window. The
fleet counts what is still in host memory as room owed to that card. Promotion that finds no device
memory leaves the rest where it is: the model serves on, slower, partly resident, and the fleet
tries again every 5 s until it is all on the card.

It is not the default, because the link is the bottleneck either way. Measured on the 27B (dense,
decoding, 128 tokens): the first reply after a park takes 5.38 s live against 2.52 s with a full
restore (1.33 s resident). Every decode step reads all the weights still in host memory over the
same link the promotion copies through (it ran at 3 GB/s instead of 27), while a full restore from
host memory already moves them at the link's rate. Live suits a model whose steps read a small part
of what is parked (a tiered MoE's experts: routing touches a few per layer) or that computes long
per byte it reads; a dense decoder is faster restored whole.

Not built: a first step that waits per layer for an upload in first-use order (TTFT
`max(upload, prefill)` instead of the sum; the gain is the first step's compute, large only for long
prompts).

### 7.5 One-step models served in place

An embedding model's request is one step: a pass over the prompt, nothing generated. Such a step
reads each weight once, so reading the weights in place over the link costs it about what a restore
would have cost to move them (EmbeddingGemma2's 1.1 GiB at the link's rate, some 40 ms on the test
host). Waiting for the room costs more: another model's quantum, or a neighbour parked for it.

So a one-step model (`Engine::one_step`) with `park.restore: auto` (the default) that comes back to
cards without room for all of it right now comes back in place: only what is not a sealed chunk
parked in host memory (the KV cache, the arena, written memory) needs room
(`dev_page_out_bytes(owner, live)`), and the weights stay mapped in host memory at their own
addresses, as with `park.restore: live` (§7.4). Unlike a live restore they are not room owed to the
card: the fleet does not hold room for them, a planner counts only what is on the card as what the
model would free, and they move onto the card (`dev_page_promote`) only into free memory, a pass
every 5 s. When the cards do have the room, the model comes back whole, as any other.

---

## 8. Scheduling

### 8.1 States

```
 cold --load--> loading --(first layers ready)--> partial --(all ready)--> resident
   ^                                                                        |   ^
   |                                               evict-to-disk            |   | restore
   +------------------------- parked <---------------park (drain)-----------+   |
                                 +----------------------------------------------+
 any --fault/refused load--> failed --(admin load or reload)--> cold
```

`GET /admin/fleet` reports each model's `state`: `cold`, `loading`, `resident`, `closing`,
`parked`, `failed`, or `waiting` for a model that is not serving and waits for room, whatever it was
doing when the wait began (a load, a return from its park). `waiting_for` names the room:
`accelerator` (memory on its cards, §6.3) or `host memory` (memory to pin, §6.5). The number
`waiting` beside them counts the requests queued for the model. The dashboard's history keeps the
state as the figure `s:<id>`: 0 cold, 1 parked, 2 loading or closing, 3 resident, 4 failed,
5 waiting; the page shows `waiting` as "Waiting for room". `vram_bytes_by_card` is what the model
holds on each card now, by card index: its claim and its cache's growth while it serves, what a
park left of it there, or what it has backed so far while it loads.

A wait for room on the cards lasts while their free memory still rises: memory given back is
counted free only as the driver's wipe reaches it, which after the tens of GiB a large model's room
takes is several seconds. When the ledger says the models that may leave do not hold enough, and
none must stay, every model on those cards is parked and then unloaded before the demand fails: a
model also holds memory outside its claim, which comes free only with the model.

### 8.2 Requests for a model that is not resident

The request takes a reference and waits on the entry's load, bounded by `load-timeout` (then 503
with `Retry-After`). A streaming request gets an SSE comment line every `keepalive`, so proxies and
clients keep the connection open. Requests queue per model in arrival order; a model's own
`max-queued-requests` still bounds its queue.

### 8.3 Eviction

When a load needs room, candidates are the resident models without `pin`, sorted by:

1. priority, lowest first;
2. then reload cost × recency: `idle_seconds / restore_seconds`, highest first. `restore_seconds` is
   the measured restore time from where the model would go (RAM copy or disk), so a model that comes
   back cheaply leaves before one that does not.

A model with an idle `ttl` parks after that long without requests, whether or not room is needed.
Groups bind (§8.5). A busy candidate is drained, bounded by `drain-timeout`.

A model serving requests keeps its cards for a **quantum** after it came on (four times what coming
back costs it, 5-30 s): another model's demand waits instead of evicting it, so a model under load
serves a batch rather than giving its cards back for each request of another. Measured on the c4
plan of §13: 313 s without it, ~230 s with it; aging it out after a waiter's 3 s, or serving
waiters strictly first come first served, were both slower.

A demand whose room is held by something passing (another model's load, restore or park, reserved
room, a model inside its quantum) waits for it rather than failing every request queued for it. A
grant that waits releases the bring-up lock meanwhile, and reserves its room in the same locked
section that found it, so two loads cannot both take it. Two parked models that each need the room
the other's remains hold do not wait on each other: one that has waited 2 s parks the rest of itself
and waits on.

A returning model's room is counted card by card: a model of several ranks that is all out on one
card and mostly still on another needs only its missing part on the second. Asking for the larger
part on every card cannot be met where its own rest still sits, so it waited out the 2 s and parked
the rest of itself to come back (Qwen-Image on §13's seq plan: 5.5 s back, 2.1 s counted per card).

Cold loads beside each other share the drive and the relayout workers, so each took as long as all
of them together (seven models at once: three to twenty times their time alone). One container
sweep over 2 GiB reads at a time, the smallest waiting first; small models never wait.

### 8.4 Card choice

A model asks for `tp: N`. The scheduler picks N cards from `cards:` that minimise evictions (fewest
fixed bytes of lower-priority models displaced), then flex displaced, then prefer cards the model was
resident on before (its RAM copy is per rank and per card's shard; a different card set reuses the
same shards). TP groups that need peer links prefer card sets with P2P. Within an engine's life its
card set is fixed; moving it is an unload and reload.

A model with its own `cards: [N, ...]` is placed on those cards only: the choice above runs over
its list instead of the fleet's, so it waits for room there rather than load elsewhere. The list is
among the top-level `cards:` and has at least `tp` entries. It is not an engine setting, but a
reload that changes it unloads the model, which then loads again on the new list. Without the key,
or with `cards: all`, a model may use any card of the fleet. On a machine whose cards differ, or
where one card also drives a display, this keeps a model off the card it should not take.

### 8.5 Groups

- `together: [a, b]`: loading one loads the others; parking one parks the others. A request for any
  member waits for all of them.
- `exclusive: [a, b]`: at most one member is resident; loading one parks the others first.

The config check refuses contradictions (a pair both together and exclusive, a together group whose
members cannot fit at once).

### 8.6 Warm-up and host fill

A first request should find its model parked, not cold: a restore from host memory takes a second
or two where a cold load takes the container read and relayout (12.9 s for the 27B).

**Warm-up** (`scheduler.warm-up`, on by default). Once the listener is open, the unpinned models
load one at a time, lowest priority first (later in the file first among equals), so the ones that
matter most are those still on the cards at the end. Each next load waits until the fleet is quiet:
no request waiting, no load, restore or park running. Displacing a warm model later is an unmap:
its weights come back from its container (§7.2). A warm load gives way to everything:

- it takes room only from free memory and from other warm models (a model a request has used
  stays, and no parked model unloads for it);
- it does not wait for room, and one that finds none leaves its model cold without an error (a
  request that came for it meanwhile asks for it again);
- a request for a warm model makes it an ordinary one.

The warm-up stops for good after a warm load that made another model unload. What a warm load
displaces needs no room elsewhere: a parked model's weights leave by an unmap, and its KV cache and
transients are dropped.

**Host fill.** A parked model whose chunks are only on the drive (a spill, its container) gets
them read into host memory while the fleet is quiet, a GiB at a time
(`dev_page_fill_host`): mapped at the chunks' own addresses, exactly as a park to host memory leaves
them, so its restore copies over the link instead of reading the drive (and a live restore serves
from them in place). The highest priority goes first, then the most recently used. A fill leaves
2 GiB of host room, and room for the refill it runs; these copies have their drive copy, so
`make_host_room` lets them go before anything else. The weights pool's chunks are filled by the
engine (`Engine::fill_host`): host memory is mapped at the chunks (`dev_page_fill_begin`) and the
weights in them are read from the container and laid out into it, as a restore's refill does, with
every write clipped to the batch's chunks (a weight running into the next batch is read for each).
New host-fill batches also wait while any model has an active response or retained background-job
lease. Continuous traffic can postpone warming; demand-driven restores still proceed. A request
arriving during an already-started batch can overlap that batch. The GiB limit bounds its bytes,
not its duration, and brief dashboard leases can also defer the next batch.
Such a copy is exact until the mover writes the chunk, so a park after the restore keeps it and
leaves by the swap; once written, or dropped for room, the chunk is refilled from the container
again. Nothing of it ever comes from the card.

Parked models also hold pinned host memory that is not weights: the linear-state snapshot store
(4.19 GiB for the 27B, 3.08 GiB for Flash Next), empty once the park has reset the prefix cache.
Under host pressure, and to make room for a fill, a parked model gives it back
(`Engine::release_host_spare`) and its next restore pins it again first: 4 GiB pins in 0.3 s on the
test host, against seconds to read the same GiBs of weights back from the drive. In `make_host_room`
these go after resident models' kept copies and before parked models' host copies.

`GET /admin/fleet` reports `warm_up: {queued, loading, drive_only_bytes}`.

---

## 9. Two models on one card

Two resident models with work on the same card can share it in two ways:

- **concurrent queues**: each engine's stepper submits on its own HW queues and the GPU interleaves
  them;
- **time-sliced steps**: a per-card arbiter grants whole steps to one engine at a time, weighted by
  priority, with no step issued ahead of the previous one's readback.

Both were built and measured on one R9700 (210 W): MiniCPM5-2B bf16 decoding one
stream, its inter-token latency beside a second workload, twice each, interleaved.

| beside | concurrent p50 / p99 | time-sliced p50 / p99 |
|---|---|---|
| nothing | 12.3 / 12.8 ms | 12.7 / 13.1 ms |
| the 27B IQR-V4 decoding two streams | 25.9 / 27.6 ms | 50.0 / 68.0 ms |
| the 27B prefilling 6K-token prompts | 37.4 / 43.1 ms | 33.1 / 1455 ms |
| Qwen-Image 2.1 generating 1024x1024 (TP 2) | 35.3 / 37.7 ms, 13.1 images/min | 326 / 329 ms, 9.3 images/min |

The interferer's throughput was the same or better under concurrent queues in every row. Taking
turns idles the card between steps and makes the small model wait out the large one's whole step
(a diffusion evaluation, a prefill chunk), so concurrent queues stay and the arbiter is gone; there
is no `co-exec` key.

### 9.1 Two multi-card models on the same cards

A model on more than one card exchanges its ranks' partial sums in collectives, and a block of a
collective on one card finishes only when the same block has run on the other card. A chained
launch (GUIDE §4.6, "Chained launches") holds its wave slots while it waits behind the kernel before
it, and the packet processor launches as far ahead as the card has slots. One engine cannot starve
itself that way: each rank's collective is launched before anything waits behind it. Two engines
on the same cards could. Engine A's collective on card 0 waits for its half on card 1; card 1 is
full of engine B's chained kernels waiting behind B's collective; B's half gets no slot on card 0
beside A's, because a prefill junction launches more blocks than a card runs at once. Nothing
advances from there. Qwen3.8-27B (IQ4R-S-V4) prefilling beside Gemma 4 12B decoding, both TP 2 on
the same two R9700, stopped within 12 s of load, and the process then ignored its stop signal.

So every engine claims its cards when it starts, including a single-rank engine that can occupy
wave slots beside another engine's collective. On a card with more than
one claim a chained launch keeps its barrier bit while a collective may still run on its queue.
Only the collectives' own blocks then wait on a card for another card, and those are few enough at
decode to leave the other engine its slots. The claim that makes a card shared returns once
everything issued on that card before it has finished, so nothing launched under the old rule is
left. A model alone on its cards keeps every chained launch. A claim
lasts as long as the engine does: a parked model still holds its own.

With the rule the same load ran 200 s without a stop (335 requests), as did both models prefilling
(467 requests) and both decoding (131). What it costs a model while it shares its cards, measured
by forcing it on a model that was alone (medians of three rounds): the 27B's step 20.30 to 20.38 ms
at one stream and 26.7 to 27.1 at four; Qwen3.8-Flash-Next under its draft head 12.46 to 12.56 and
23.3 to 23.3.

---

## 10. Hot reload

SIGHUP or `POST /admin/reload` re-reads the file and validates it completely before touching
anything. Then, per entry:

- unchanged (same normalised key set): untouched;
- changed only in request defaults and limits: applied in place to its `ModelFront`;
- changed otherwise: drained, unloaded, and re-registered as cold (loaded on the next request, or at
  once if pinned);
- removed: drained and unloaded; its name answers 404;
- added: registered cold.

`server:` changes other than `log` need a restart and are refused by reload with that reason.

---

## 11. Faults

- **GPU memory fault**: unrecoverable in-process. The handler names the buffer and the model, and the
  process exits; the container's restart policy brings the fleet back and pinned models reload.
- **Queue fault or stall**: confined to the engine's card set. Every engine on those cards is marked
  failed and refuses its requests by name; engines on other cards keep serving. A failed card set
  needs a process restart to recover, which the fleet reports on `/health`.
- **Load failure** (bad file, out of memory, refused plugin): the entry goes `failed` with the error,
  its waiting requests get it, and the rest of the fleet is unaffected.

---

## 12. Limitations

Everything §3-§11 describe is built and measured on the test host of §13 (two R9700), except:

- **§7.4** is opt-in: measured slower for a dense decoder. The fleet does not yet choose partial
  residency on its own (serving a tiered MoE with its slab in host memory instead of waiting for
  room). A dense model serves whole on its cards (the quantum, §8.3); a tiered MoE's expert slab
  grows into free VRAM and into the room of an idle neighbour (§6.7). On the test host's 61 GB Flash
  Next's own host pool (24 GiB) leaves room in RAM for only a few GiB of other models' parked
  copies; the rest restore from the drive.
- A model waiting for room does not start its prefill with part of its weights streamed over the
  link (measured in §13.4: it would prefill at 80-90% of its speed, but it needs the KV room it is
  waiting for, decode that way runs at a tenth, and a placement is fixed for an engine's life).
- A first step that starts on a model coming back from the drive before every byte has arrived
  (layer by layer) is not built: a dense step reads all its weights, so the first token waits for
  the drive either way, and the gain is the first step's compute.
- `park.vram` (statics or MiB kept on the card) and `flex.kv.weight` are refused as not acted on.
  The disk tier keeps a parked engine (its copies written to `memory.disk-tier` in the background,
  its host memory let go under pressure); a model unloaded outright still loads from its container.
- A tiered-MoE cache that lends to its expert slab does not run as flex: it keeps its claim (the
  slab grows through the fleet's loan instead, §6.7). Its claim is still sized by `flex.kv.min-tokens`
  (§6.3), and since the cache never grows past it, set it to the tokens the model must serve at
  once: unset, the floor is one sequence at the context bound. Qwen3.8-Flash-Next at 4 x 262144 in
  a fleet of four on two R9700: 2.27 GiB a card, one full-length session; with `min-tokens:
  1048576`, 7.30 GiB, 3.7 sessions (a model alone in its file claims the card and gets 4.0).
- Unified memory (gfx1151) parks nothing useful: host RAM is the card's memory. Not tested there.

---

## 13. Measured against llama-swap

The test host: two R9700 (32 GB each, PCIe 5.0 behind x8 root ports, 28.8 GB/s measured host to
card), a Ryzen 9 7950X3D with dual-channel DDR5, 61 GB RAM. Seven models configured at once:
EmbeddingGemma2, Unlimited-OCR, MiniCPM5-2B FP8, Qwen-Image 2.1 FP8 (TP 2), Qwen3.8-27B IQ4R-S-V4,
Flash Next FP8/IQ4R (TP 2, tiered experts, 12 GiB host pool a rank) and Hunyuan3D-2. Three workflows
use all of them: docqa (OCR, embed, 27B answer, MiniCPM summary), create (MiniCPM prompt, Qwen-Image
picture, Hunyuan3D mesh, Flash Next critique) and chat (Flash Next, 27B, embed, MiniCPM). llama-swap
v270 runs the same radiance binary, one process per model: in its default mode (one model at a
time), and with its capacity router given per-card budgets and card assignments by hand. The fleet
runs the plain file with `memory.disk-tier` and nothing hand-placed.

| plan | fleet | llama-swap default | llama-swap capacity (hand-placed) |
|---|---|---|---|
| 9 workflows one after another, wall | 168 s | 231 s | 186 s |
| 2 clients x 6 workflows, wall | 133-142 s over 3 runs, none failed | 352 s | 121 s, 1 failed; 207 s, 4 failed |
| 4 clients x 6 workflows, wall | 176-220 s over 3 runs, none failed | 517 s | 163 s, 1 failed (in each of 2 runs) |
| 27B / Flash Next alternating, per request (median) | 3.3 / 1.7 s | 15.1 / 17.8 s | 15.1 / 17.9 s |

The fleet beats llama-swap's default mode on every plan, and its hand-placed capacity mode on the
sequential plan and on swapping (a parked model comes back in 1-3 s instead of a 15-18 s process
start). On the concurrent plans it fails nothing where the capacity mode loses runs (a process
started before the one it replaced had handed its VRAM back), wins two clients against the
capacity mode's slower runs and loses four clients by 10-35%.

The concurrent loss is the first round. From the second round on every workflow runs at or below the
capacity mode's times (docqa 4-7 s, chat 8-16 s, create 20-42 s); the first round is seven cold
loads at once plus the two models that need both cards almost whole (Qwen-Image 2.1, 25 GiB a card,
and Flash Next) taking turns. Races there decide which model waits for which, and the 4-client wall
varies by 25% between runs of the same code: a run where Flash Next's first load waited behind the
27B's sweep, and Qwen-Image's restore read its page file while Flash Next's load filled the drive,
took 220 s; the two before it 176 and 178 s.

What got the concurrent plans here, from 351 s (c4) in the first measurement: a residency quantum
for models serving a batch (§8.3), demands that wait for passing room instead of failing, grants
that wait without the bring-up lock and reserve their room atomically, returning models that give
way to each other instead of deadlocking, park-or-unload by measured page-in and projected writes,
loads held to the host memory floor (one run that pinned past it was killed by the kernel), dirty
blocks instead of whole-chunk invalidation (Flash Next's parks wrote 7-25 GiB each, now megabytes),
the activation arena's transients dropped at a park (Qwen-Image copied 13 GiB of them), and one
large cold sweep at a time, smallest first.

llama-swap's capacity mode needed every model's VRAM and cards worked out by hand, and still lost
runs when a process started before the one it replaced had handed its VRAM back.

### 13.1 Warm starts

The table above starts every system cold. With the warm-up (§8.6) the fleet loads all seven models
once in about 26 s after it starts, and the plans start once it is done. llama-swap's capacity mode
gets the same chance with `hooks.on_startup.preload`: the five models its hand placement keeps
resident at once (27B and Hunyuan3D-2 on card 0, the three small ones on card 1), ready in 22 s.

| plan | fleet, warm | llama-swap capacity, preloaded |
|---|---|---|
| 9 workflows one after another, wall | 78.0 / 78.0 s over 2 runs | 165.3 s |
| 2 clients x 6 workflows, wall | 65.8 / 64.5 s over 2 runs, none failed | 144.5 s, none failed |
| 4 clients x 6 workflows, wall | 109.9 / 110.1 s over 2 runs, none failed | 202.3 s, none failed |
| 27B / Flash Next alternating, 6 rounds, wall | 38.8 s (per request 1.95 / 4.5 s) | |

Warm, every first request finds its model parked: q27, Flash Next and Qwen-Image come back in 2-7 s,
their weights read from the container again at the drive's rate (§7.2), instead of loading for
13-35 s. Parked models' snapshot stores (7.3 GiB for the 27B and Flash Next) go back to the host
under pressure, so a few GiB of parked weights fill into RAM even beside Flash Next's 24 GiB pool.
With the disk-tier copies of the weights an earlier design wrote after every load, the plans took
97.7, 71.8 and 113.9 s; the concurrent plans turn on which of two models waiting for the same cards
comes back first (§8.3), and a schedule where Flash Next and the 27B take turns on card 1 twice
more costs the 2-client plan 20-30 s. Before a returning model's room was counted card by card
(§8.3), Qwen-Image gave way on every return and the 2-client plan took 99.9 / 107.5 s. The fleet
runs Qwen-Image with `max-num-seqs: 4` and `max-model-len: 2048` (§6.3); with neither set its cache
floor was 8.2 GiB a card and the plans took 89.8 / 93.3, 75.5 / 75.5 and 117.0 / 151.3 s. With
`memory.host-mib` at 60% of the host, the default before 75%, they took 81.9 / 82.7, 68.4 / 72.6 and
119.5 / 128.7 s: more parked weights stay in host memory and come back from there, so the
sequential plan reads 308-316 GiB from the drive instead of 334 and its restores page in for 34 s
instead of 39; the 8 GiB host floor held (7.9 GiB available at the lowest). The test host's drive is 94% full (116 GB free), which holds
its writes to about 1.2 GB/s; its reads run at 7.3 GB/s.

### 13.2 Expert flex

Flash Next alone in a two-model fleet (MiniCPM configured, never asked for), its claim 23.65 GiB a
card; with expert flex the slab grew 7 GiB a card into the free memory within the first requests.
Fresh start for each run, runs alternating, 320-token answers, distinct prompts per stream:

| | no expert flex (`max-mib: 0`) | expert flex |
|---|---|---|
| 1 client, decode | 63.8 tok/s | 65.8 tok/s |
| 1 client, first token | 72 ms | 44 ms |
| 4 clients x 6 rounds, median stream | 34.4 / 35.5 tok/s | 40.5 / 40.7 tok/s |
| 4 clients x 6 rounds, slowest stream | 20.1 / 34.3 tok/s | 39.2 / 39.1 tok/s |
| 4 clients x 6 rounds, wall | 29.2 / 24.4 s | 21.6 / 21.2 s |
| mover traffic over one run (rank 0) | 35.0k promotions, 41 GiB in, 34 GiB out | 10.6k promotions, 12.5 GiB in, 4 GiB out |

The single stream is almost all hits already; four streams route to more experts than the claim
holds, and the slab stops trading them over the link. Beside a resident MiniCPM the slab grew into
the 3 GiB left free, MiniCPM parked after 30 s idle while Flash Next was busy and the slab grew to
7 GiB, and MiniCPM's next request took 3.34 GiB back and restored it; both models' temp-0 answers
were identical in every phase.

### 13.3 One-step models in place

One card (`cards: [0]`), the 27B at a 204,800-token context (its claim takes 30.4 GiB of the card)
decoding two streams without pause, and EmbeddingGemma2 with `ttl: 3s`, asked to embed eight chunks
every 4 s, so every request finds it parked. 90 s a run, runs alternating:

| EmbeddingGemma2 `park.restore` | embed latency p50 / p90 / max | 27B, both streams | 27B parked in part |
|---|---|---|---|
| `auto` (in place) | 51 / 53 / 2330 ms; 50 / 53 / 2333 ms | 50.4 / 50.6 tok/s | 2 / 2 times |
| `full` | 2242 / 6083 / 7290 ms; 2113 / 6022 / 6293 ms | 45.4 / 48.2 tok/s | 14 / 15 times |

Coming back whole, EmbeddingGemma2 needs 1.6 GiB the 27B's claim leaves no room for: each request
waits while the busy 27B drains and parks part of its cache, and the 27B takes the room back as soon
as the embedding is done. In place it needs the 0.4 GiB of its cache and arena, comes back in 0.01 s
and embeds in 50 ms with 1.1 GiB read over the link (448 MiB of it moved onto the card into free
memory). The one slow request in place is the first, while both models settle on the card.

### 13.4 Prefill with part of the weights streamed

Whether a model waiting for room could start its prefill with what is off the card streamed over
the link: the 27B alone, `placement: layer_offload` (each step stages the off-card layers onto the
card ahead of the ones that read them), against all weights on the card. One sequence's
204,800-token cache on the card in both, fresh prompt text for each request, two rounds:

| weights off the card | prefill, 2.7K-token prompt | 10.8K | 43K | decode |
|---|---|---|---|---|
| none (14.8 GiB on the card) | 2566 / 2497 tok/s | 2541 / 2481 | 2181 / 2152 | 25 tok/s |
| 4.2 GiB | 2023 | 2178 | 1951 | 3.2 tok/s |
| 7.6 GiB (half) | 1868 / 1856 | 2074 / 2052 | 1890 / 1906 | 2.3 tok/s |

The staging overlaps the compute: a long prefill keeps 85-90% of its speed with half the weights
off the card, a short one 75%, and the cost hardly depends on how much is off (the staging's fixed
cost per step, not the link, sets it). A decode step reads every weight for one token, and runs
at a tenth. Less than half on the card did not plan (the staging slabs of the layer classes did not
fit beside the cache). So a waiting model's prefill could run nearly at speed, but not where it
matters: it needs the cache room it is waiting for (13.9 GiB for one full-length sequence here,
against 7.6 GiB of weights), it would have to change placement while it serves (an engine's plan
is fixed for its life), and its decode after the prefill still waits for the whole restore. Not
built.
