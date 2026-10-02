# AGENTS.md — contributor rules for dsh-local-models

Two halves, one contract: `lib/client.js` (Web GUI tab) owns the controls,
`lib/index.js` (node half) owns the launch. A launch-affecting control only
works if it round-trips through **every** layer below. A feature is DONE only
when the checklist in §1 passes for each new control.

Node-half changes need a dsh web restart (bundle composition at boot);
client-half changes need only a page refresh.

## 1. UI ↔ logic contract (mandatory per new control)

Every launch-affecting control must pass through all 8 layers:

1. **State default** (`useState` in `LocalModelsSection`) matches the
   historic launch behavior (what an omitted field produced before the
   feature existed).
2. **localStorage**: dedicated `LS_*` key, restore-on-mount in the mount
   effect, save in the persist effect **plus the key in its dep array**.
   Missing restore or a missing dep = "the tick doesn't stick after refresh".
3. **Status adopt** in `refresh()`: adopt the server value only for a fresh
   tab — guard with `!selectedPathRef.current` (model fields) or
   `!mmprojPathRef.current && !clearedMmprojRef.current` (mmproj fields).
   Never override unconditionally: the poll runs every 2 s and would fight
   live user edits.
4. **Launch path**: `doRun` sends the field; the `/local-models/run`
   handler reads it with the explicit-boolean pattern
   `typeof body.x === "boolean" ? body.x : DEFAULT` (env default only when
   omitted — `=== false ? false : DEFAULT` silently drops an explicit `true`
   when the env default is `0`).
5. **Manager**: `run()` normalizes into a `*Final` local, uses it in `args`
   **and** in the `logNote` header, stores it in the `launch()` info object,
   and `launch()` persists it to `state` so `GET /local-models/status`
   returns it for tab re-sync.
6. **Profiles**: `normalizeProfileConfig` carries the field (missing/legacy
   defaults to the historic launch value, never to the current tab state);
   `POST /local-models/profiles` forwards it.
7. **Profile round-trip**: `doSaveProfile` sends it; `doLoadProfile`
   restores it deterministically. No "leave the current selection alone"
   for legacy profiles — restore the value that reproduces the original
   launch (e.g. pre-MoE profiles → `gpu`, pre-placement profiles → CPU).
8. **Router + visibility**: `generateRouterPresets` honors the field
   per-profile (never hardcode one placement for all presets); the status
   card, `statusLine`, and the profile subtitle display it so the user can
   verify what launched without opening the terminal.

### Canonical field table (keep in sync when adding a control)

| Tab state | Profile field | `/run` body | `state` | Legacy default |
|---|---|---|---|---|
| `selectedPath` | `modelPath` | `path` | `modelPath` | — (required) |
| `ctx` | `ctx` | `ctx` | `ctx` | `8192` |
| `mtp` | `mtpHeads` | `mtp` | `mtpHeads` | `0` (tab offers 0-7) |
| `effort` | `effort` | `effort` | `reasoningEffort` | `"medium"` |
| `kvTypeK`/`kvTypeV` | `kvTypeK`/`kvTypeV` | `kvTypeK`/`kvTypeV` | `kvTypeK`/`kvTypeV` | `q5_0`/`q4_1` |
| `preserveThinking` | `preserveThinking` (`preserve_thinking` alias on read) | `preserveThinking` | `preserveThinking` | `false` |
| `mmprojPath` | `mmprojPath` | `mmproj` | `mmprojPath` | `null` |
| `mmprojCpu` | `mmprojCpu` | `mmprojCpu` | `mmprojCpu` | `LOCAL_MODELS_MMPROJ_CPU` (default `true` = historic CPU) |
| `moeMode`/`nCpuMoeText` | `cpuMoe`/`nCpuMoe` | `cpuMoe`/`nCpuMoe` | `cpuMoe`/`nCpuMoe` | `false`/`0` (all-GPU) |
| `expertUsedText` | `expertUsed` | `expertUsed` | `expertUsed` | `null` (stock top-k) |
| `splitMode`/`tensorSplitText` | `splitMode`/`tensorSplit` | `splitMode`/`tensorSplit` | `splitMode`/`tensorSplit` | `"layer"`/`null` (no flags; multi-GPU control only shows with >1 GPU) |

`arch` travels with the MoE payload (profile + launch) for the top-k
`--override-kv` key; on load it is re-derived from the fresh GGUF parse.

Removed controls: `ignoreCtxCap` (the old "ignore the MTP ctx softcap"
checkbox) is gone — the fixed MTP draft is emitted whenever `mtp > 0`, at any
ctx. Legacy profiles may still carry the key; nothing reads it any more.

### Machine-level settings (the exception: no profile field)

The Runtime card's `binPath` (llama.cpp binaries), `shortcuts` and `vramGb`
(the estimator's VRAM budget, 0 = auto-detect) belong to the machine, not to a
model, so they skip layers 6-7 on purpose: profiles must
stay portable, and the router launches from one binary anyway. They still owe
the rest of the contract — layer 1/2 (`useState` default `""` = auto-detect,
`LS_BIN` restore + dep), layer 3 (adopt `status.binConfigured` only while the
field is clean: `binDraft !== binSaved`), layer 4 (`POST /local-models/settings`,
which validates that a non-empty `binPath` resolves to a real `llama-server`),
layer 5 (`run()`/`startRouter()` re-resolve per launch via `requireBin()`;
`launch()` stores `state.binPath` and the `logNote` header names the binary)
and layer 8 (status returns `bin`/`binConfigured`/`binVersion`/`shortcuts`/
`shortcutsSource`/`vramTotalBytes`/`vramSource`/`vramOverrideGb`/`gpus`).

Binary resolution is the ordered `binCandidates()` chain — setting →
`LOCAL_MODELS_BIN` → `PATH` → home build dirs → system dirs — never a hardcoded
path at the spawn site, and never an absolute path of one particular machine in
the candidate list either. Two guards keep that chain from becoming a way
around the file sandbox, and both are load-bearing:

- **Identity** — `isLlamaServerPath()` requires an executable named
  `llama-server`, enforced on save (`POST /settings`), on the check verdict
  (`POST /runtime/check`) and again in `requireBin()`. Without it, any file the
  route accepted could be spawned and its directory would become a browse root.
- **Roots** — `shortcuts` IS the file-endpoint confinement (`allowedRoots()`),
  so it is validated on write: absolute, no `..`, never `/`, at most 12
  entries. The roots also carry the directories of the paths saved profiles
  declare, so a saved profile always stays loadable, and profile saves are
  confined to those roots — a profile cannot be used to widen them.

## 2. Verification (run before declaring done)

- `node --check lib/client.js && node --check lib/index.js`
- `npm test` (node built-in runner, `test/`) — extend it when the change
  touches a pure helper (`normalize*`, `generateRouterPresets`,
  `moeArgsFor`, …).
- Manual: Load → `Open terminal` → the `starting <model> (…)` header must
  show the new setting; refresh the page → the control keeps its value;
  save → delete → re-load the profile → same launch.

## 3. Layout

```
lib/index.js    node half: process manager, GGUF parser, routes, presets
lib/client.js   browser half: settings tab (single build-free bundle)
skills/         operator skill: spawn-parity checklist, profile audits
docs/           UI mockup (illustrative only — lib/client.js is authoritative)
test/           unit tests for the pure exported helpers
```
