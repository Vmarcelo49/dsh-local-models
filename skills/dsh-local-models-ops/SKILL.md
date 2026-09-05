---
name: dsh-local-models-ops
description: Operate and maintain the dsh-local-models plugin that spawns llama-server from the dsh Web GUI. Verify the spawned args match the tuned upstream llama-server config (fixed MTP depth, q8_0/q4_0 KV, mmproj on CPU), audit and fix ~/.dsh/local-models/profiles.json (model/mmproj paths, MTP per profile), restart the dsh web after node-half changes, and diagnose a spawned server (cmdline, log, DRM VRAM, /slots fill). Use when the daily server is run via the Local Models tab or when plugin config/profiles diverge from the tuning. Only useful inside this project.
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
| `--cache-type-k q8_0 --cache-type-v q4_0` | fixed |
| reasoning chain | `--reasoning auto --reasoning-format deepseek --no-reasoning-preserve --reasoning-effort medium` |
| MTP | `--spec-type draft-mtp --spec-draft-n-max N --spec-draft-p-min 0.75` when `mtp > 0 && (ctx <= 131072 || ignoreCtxCap)` (draft dropped above the ctx ceiling; depth capped at 3 — fixed depth > 3 collapses at large ctx) |
| mmproj | `--mmproj <file> --image-min-tokens 1024` + `--no-mmproj-offload` when the tab checkbox is on (default) |
| spawn env | `RADV_PERFTEST=nogttspill` |

## 2. Profiles audit (`~/.dsh/local-models/profiles.json`)

Schema per profile: `{ id, name, modelPath, ctx, mtpHeads, mmprojPath,
effort, ignoreCtxCap, updatedAt }`. Known audit items:

- **Paths must exist and live on the fast mount**. Referencing the failing/
  legacy `/mnt/disco1` is a red flag - targets are under `/mnt/raid0/GGUF/`.
- **MTP depth must be <= 3** (upstream fixed draft; deeper collapses at
  large ctx). Old profiles may still carry 4-6 from the fork era — the tab
  clamps them on load, but fix the stored value.
- **Daily qwen35 profile** should use `mtpHeads: 3` at `ctx <= 131072`
  (no KV streaming upstream, so 256K ctx is out of reach on 16 GB VRAM).
- A helper ships with this skill: `node scripts/audit-profiles.mjs`.

## 3. Tab / client defaults (`lib/client.js`)

- `_mtp = useState(2)` (fixed-MTP head default; the daily tune wants 3)
- `_mmCpu = useState(true)` (mmproj on CPU, frees ~0.87 GiB VRAM)
- softcap checkbox unchecked by default (`ignoreCtxCap: false` unless the profile sets it)

## 4. Diagnostics

- Spawned server PID: `pgrep -x llama-server`; confirm the real cmdline with
  `ps -o args= -p <pid>` (two processes may exist: the router on :8080 with
  `--models-preset` + the model server on a random port).
- Server log: `~/.dsh/local-models/llama-server.log` — also viewable live in
  the GUI: Local Models tab → **Open terminal** (overlay tail of the same
  file, backed by `GET /local-models/logs?offset=<nextOffset>`).
- VRAM: `/sys/class/drm/card*/device/mem_info_vram_used|total` (used/total).
- Current fill: `curl -s http://<port>/slots` -> `n_past` / slot ctx.

## 5. Restart flows

- **Node-half changes** (lib/index.js, inject list, new routes): restart the
  dsh web process - `pkill -f 'dsh web'` (or the profile's launcher) and
  relaunch; bundle composition picks up only at boot.
- **Client-half changes** (lib/client.js): page refresh is enough (the served
  bundle rev updates automatically).

## 6. Register in dsh + opencode wiring

`POST /local-models/register` adds the ready server as an `llm-pi-ai`
provider route on the dsh webserver. Single-model mode needs the manual
**Register in dsh** press; router mode has no Register button — starting
the router auto-registers the `local-router` route once the server is
ready (a refresh poll fires it; stopping first cancels a pending one).
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
- Old profiles may carry fork-era fields (`kvStreamMib`, `mtpHeads` 4-6):
  harmless leftovers the plugin now ignores (MTP clamps to 3 on load), but
  clean them when auditing.