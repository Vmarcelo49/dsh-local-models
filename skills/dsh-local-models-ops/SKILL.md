---
name: dsh-local-models-ops
description: Operate and maintain the dsh-local-models plugin that spawns llama-server from the dsh Web GUI. Verify the spawned args match the tuned upstream llama-server config (fixed MTP depth, KV cache K/V types from the tab, mmproj on CPU), audit and fix ~/.dsh/local-models/profiles.json (model/mmproj paths, MTP per profile, KV pairs), restart the dsh web after node-half changes, and diagnose a spawned server (cmdline, log, DRM VRAM, /slots fill). Use when the daily server is run via the Local Models tab or when plugin config/profiles diverge from the tuning. Only useful inside this project.
---

# dsh-local-models-ops

Operational skill for the dsh Local Models plugin (node half: `lib/index.js`,
client half: `lib/client.js`). It spawns upstream llama-server from the Web
GUI and provides the "Register in dsh" provider wiring.

## 1. Spawn parity checklist

The plugin spawn must match the tuned daily config. Verify against
`lib/index.js` (the `args` array in `run()`):

| Setting | Expected value |
|---|---|
| `-ngl 999 -c <ctx>` | full offload, ctx from the tab slider |
| `--flash-attn on --kv-unified` | always |
| `--cache-type-k <K> --cache-type-v <V>` | from the tab's K/V selectors (`kvTypeK`/`kvTypeV` in profiles), default `q5_0`/`q4_1`; accepted ids: `f32 f16 bf16 q8_0 q5_1 q5_0 q4_1 iq4_nl q4_0`. MLA models (DeepSeek-style latent KV, `isMla` in the GGUF) must use the same type for both |
| `--spec-draft-type-k q4_0 --spec-draft-type-v q4_0` | fixed (MTP draft KV is not user-selectable) |
| reasoning chain | `--reasoning auto --reasoning-format deepseek --reasoning-preserve/--no-reasoning-preserve --reasoning-effort medium` (preserve toggle in the tab, `preserveThinking` in profiles; default off = `--no-reasoning-preserve`, matching the plugin's historical behavior — upstream defaults to preserve ON) |
| MTP | `--spec-type draft-mtp --spec-draft-n-max N --spec-draft-p-min 0` whenever `mtp > 0` — no ctx ceiling any more (the old `ignoreCtxCap` checkbox is gone), depth 0-7 with 3 the tuned sweet spot (deeper collapses at large ctx) |
| idle eviction | `--sleep-idle-seconds <autoUnloadMins * 60>` when the setting is > 0 (default 30 min; omitted when 0). Sleeping server keeps `/health` + `/models` and reloads on next request |
| mmproj | `--mmproj <file> --image-min-tokens 1024` + `--no-mmproj-offload` when the tab checkbox is on (default) |
| spawn env | `RADV_PERFTEST=nogttspill` |

## 2. Profiles audit (`~/.dsh/local-models/profiles.json`)

Schema per profile: `{ id, name, modelPath, ctx, mtpHeads, mmprojPath,
mmprojCpu, effort, preserveThinking, kvTypeK, kvTypeV, splitMode, tensorSplit,
cpuMoe, nCpuMoe, expertUsed, arch, updatedAt }`
(`preserve_thinking` is accepted as an alias on read; router presets emit
`reasoning-preserve = 1/0`, a per-model `cache-type-k/-v` pair, and
`no-mmproj-offload = 1` unless the profile sets `mmprojCpu: false`). Known audit items:

- **Paths must exist and live on the fast mount**. Referencing the failing/
  legacy `/mnt/disco1` is a red flag - targets are under `/mnt/raid0/GGUF/`.
  A path outside the tab's roots (home + the Runtime card's Model folders + the
  folders saved profiles already declare) is refused on save and on launch, so
  a profile pointing somewhere else needs its folder added in the tab first.
- **MTP depth must be <= 7** (the tab's ceiling; upstream clamps the
  effective depth to the model's nextn depth) and **3 is the tuned value** —
  deeper collapses at large ctx. Old profiles may still carry 4-6 from the
  fork era; the tab keeps them now, so fix the stored value.
- **Daily qwen35 profile** should use `mtpHeads: 3` at `ctx <= 131072`
  (no KV streaming upstream, so 256K ctx is out of reach on 16 GB VRAM).
- **KV pair**: profiles without `kvTypeK`/`kvTypeV` run the default
  `q5_0`/`q4_1`; an unknown id is normalized to that default on save/launch.
  MLA (DeepSeek-style) profiles must have `kvTypeK === kvTypeV` or the launch
  is refused.
- A helper ships with this skill: `node scripts/audit-profiles.mjs`.

## 3. Tab / client defaults (`lib/client.js`)

- `_mtp = useState(2)` (fixed-MTP head default; the daily tune wants 3)
- `_kvTypeK = useState("q5_0")`, `_kvTypeV = useState("q4_1")` (the tuned
  16 GiB pair; the two selects live in the Model card and feed the estimate)
- `_mmCpu = useState(true)` (mmproj on CPU, frees ~0.87 GiB VRAM)
- no MTP softcap control: the draft is unconditional, and `ignoreCtxCap` in
  old profiles is ignored
- `_splitMode = useState("layer")` / `_tensorSplit = useState("")` — the
  multi-GPU row only renders with more than one detected GPU (or a non-default
  value carried by a profile), and the default emits no flags at all

## 4. Diagnostics

- Spawned server PID: `pgrep -x llama-server`; confirm the real cmdline with
  `ps -o args= -p <pid>` (two processes may exist: the router on :8080 with
  `--models-preset` + the model server on a random port).
- Server log: `~/.dsh/local-models/llama-server.log` — also viewable live in
  the GUI: Local Models tab → **Open terminal** (overlay tail of the same
  file, backed by `GET /local-models/logs?offset=<nextOffset>`). Each launch
  writes a header line (`starting <model> (ctx …, mtp …, effort …,
  kv <K>/<V>, experts …, mmproj …)`) — the fastest way to confirm what the
  tab actually asked for.
- VRAM: `/sys/class/drm/card*/device/mem_info_vram_used|total` (used/total).
- Current fill: `curl -s http://<port>/slots` -> `n_past` / slot ctx.

## 5. Restart flows

- **Node-half changes** (lib/index.js, inject list, new routes): restart the
  dsh web process - `pkill -f 'dsh web'` (or the profile's launcher) and
  relaunch; bundle composition picks up only at boot.
- **Client-half changes** (lib/client.js): page refresh is enough (the served
  bundle rev updates automatically).
- **Router autostart** (`settings.json: { autostartRouter: true }`, toggled
  from the tab's Router card): dsh boot launches the router from saved
  profiles and registers `local-router` once healthy. Watch
  `~/.dsh/local-models/llama-server.log` (`[autostart]` lines) after a
  restart; needs at least one saved profile, otherwise it logs and skips.

## 6. Register in dsh + opencode wiring

`POST /local-models/register` adds the ready server as an `llm-pi-ai`
provider route on the dsh webserver. Single-model mode needs the manual
**Register in dsh** press; router mode has no Register button — starting
the router auto-registers the `local-router` route once the server is
ready (a refresh poll fires it; stopping first cancels a pending one).
The registered model advertises `maxTokens: min(32768, floor(ctx/2))` — a
32K cap at half the window. The advertised value becomes the adapter's
`defaultMaxTokens`, which compaction uses as its reserved output `O` in
`W - O - headroom`: advertising the whole window (`O = W`) leaves no
message budget and proactive compaction never fires. Heavy-thinking models
at xhigh that need more than 32K must raise per-request maxTokens
explicitly (which honestly moves the compaction threshold earlier).
For opencode against the spawned server:
- the opencode config resolves per directory (project `opencode.json` beats
  nothing; the global `~/.config/opencode/opencode.json(c)` is authoritative
  from arbitrary cwds),
- `-m local/<model>` needs the model entry **in the config opencode loads**
  ("Model not found: ... Did you mean: <other model>?" = wrong config won),
- `opencode run ... -f <image>` consumes the NEXT argument as the file - put
  the message BEFORE `-f`,
- model must declare `"attachment": true` (+ image modality) for vision.

## 7. Pitfalls learned in the field

- `profiles.json` is overwritten wholesale by the plugin's profile save
  (merge is by id) - edit carefully, keep `updatedAt`.
- Legacy `disco1` references: the disk is failing; always resolve to
  `/mnt/raid0/GGUF/<author>/<model>/` and verify with `test -f`.
- Old profiles may carry fork-era fields (`kvStreamMib`, `ignoreCtxCap`,
  `mtpHeads` 4-6): harmless leftovers (the tab keeps depths up to 7 and ignores
  `ignoreCtxCap`), but clean them when auditing.