# dsh-local-models

A `dsh` addon that adds a **Local Models** tab to the dsh Web GUI: pick a `.gguf` file, tune context and speculative decoding, watch a live VRAM estimate, and load it through `llama-server` — then register the running server as an LLM provider in dsh with one click.

Built against stock upstream `llama.cpp` (`llama-server`). No fork, no patches, no build step: the client bundle is hand-written `React.createElement` (no JSX toolchain) and the node half is dependency-free.

## Features

- **Model picker** — in-app file browser (directories + `.gguf` only) with a header-only GGUF parse (architecture, quant, layers, context length, MoE detection) behind `POST /local-models/gguf-meta`
- **Launch options** — context slider (8K steps, capped at the model's trained context) + fine-tune input, KV cache quantization selectors (one for K, one for V — every type `llama-server` accepts, with bytes-per-element shown), fixed MTP draft depth (0–7, upstream clamps to the model's nextn depth), thinking level (`off`/`low`/`medium`/`xhigh`) + preserve-thinking toggle (`--reasoning-preserve` vs `--no-reasoning-preserve`, default off), optional vision `mmproj` (GPU or CPU offload), MoE expert placement (`--cpu-moe` / `--n-cpu-moe` / top-k override) with a fit-to-VRAM helper
- **Live VRAM estimate** — weights + the selected K/V cache types + recurrent state + compute/graph + overhead against the detected GPU total (nvidia-smi / amdgpu sysfs, summed across GPUs, 16 GB assumed when unknown), with fits / safe-margin / max-ctx-that-fits rows (see [Known issues](./KNOWN_ISSUES.md) for Gemma-family accuracy)
- **Profiles** — save named launch configurations, reload in one click
- **Router mode** — serve all saved profiles from one OpenAI-compatible endpoint (`--models-preset`); models load on demand, one resident at a time by default. Starting the router automatically (re-)registers its models in dsh — no manual Register press.
- **Register in dsh** — writes the ready server as an `llm-pi-ai` provider route (vision modality + thinking levels included, max output advertised at 131K tokens so long xhigh thinking blocks aren't truncated)
- **Terminal overlay** — live tail of the `llama-server` log from the tab

## Requirements

- `dsh` with the `web` profile (the plugin composes into it)
- A `llama-server` binary (upstream `llama.cpp`, Vulkan/CUDA/CPU — whatever your machine uses)
- The VRAM budget is detected (`nvidia-smi` for NVIDIA, amdgpu sysfs for AMD, all visible GPUs summed) and can be pinned in the tab's Runtime card; the 16 GB fallback and the safety margin live at the top of `lib/client.js` (`TOTAL_VRAM_BYTES`, `SAFE_MARGIN_BYTES`)

## Install

```bash
cd ~/.dsh/profiles/web
dsh plugin --profile web add /path/to/dsh-local-models
# then add "dsh-local-models" to the "bundles" array in package.json
```

Restart the dsh web process (bundle composition picks up only at boot), refresh the browser, open Settings → **Local Models**.

> Node-half changes (routes, inject list) need a dsh restart; client-half changes only need a page refresh.

## Usage

1. **Choose GGUF…** — pick a model file (Home / Models shortcuts, Up navigation).
2. Tune **context**, **KV cache K / V**, **Max MTP head** (fixed draft; capped at 3 — deeper collapses at large ctx), **thinking level** + **preserve thinking** checkbox, optional **mmproj** and **MoE** settings.
3. **Load model**, watch the status card, inspect output via **Open terminal**.
4. **Register in dsh** — the route (default `local-<alias>`) appears in the Models picker.
5. Alternatively, save **profiles** and **Start router (from profiles)** for a multi-model endpoint.
6. Tick **"Start the router automatically when dsh starts"** (Router card) to launch the router at boot and register its `local-router` route once healthy — models stay usable without opening the tab. Needs at least one saved profile; progress lands in `llama-server.log` (`[autostart]` lines, visible via Open terminal).
7. **Idle eviction** (Router card, "Unload models after …", default 30 min idle) frees VRAM via upstream `--sleep-idle-seconds` on both single loads and the router; the sleeping server keeps answering `/health` and reloads automatically on the next request (one slow request). `0` disables it. Takes effect on the next start — the tab warns when the running server uses a different timer.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LOCAL_MODELS_PORT` | `8080` | `llama-server` port |
| `LOCAL_MODELS_BIN` | — (auto-detect) | server binary or the dir holding it; the Runtime card's setting wins over it |
| `LOCAL_MODELS_SHORTCUTS` | — (none) | colon-separated file-browser shortcut dirs (`name=path` for custom labels); the Runtime card's folder list takes over once saved |
| `LOCAL_MODELS_MMPROJ_CPU` | `1` | vision projector weights in RAM (`0` = offload to GPU) |
| `LOCAL_MODELS_ROUTER_MAX` | `1` | max simultaneously resident router models |
| `LOCAL_MODELS_MAX_IMAGE_BYTES` | `10485760` | vision image guard |
| `LOCAL_MODELS_IMAGE_PIXEL_BUDGET` | `4194304` | vision pixel budget |
| `DSH_HOME` | `~/.dsh` | data dir (`local-models/profiles.json`, `local-models/settings.json`, `llama-server.log`) |

The tab's VRAM budget is detected, not hardcoded: NVIDIA through `nvidia-smi`,
AMD through sysfs (`mem_info_vram_total`, with the product name resolved from
`pci.ids` when present), all visible GPUs summed, and `CUDA_VISIBLE_DEVICES` /
`HIP_VISIBLE_DEVICES` honored. Hardware that cannot be read falls back to the
historic 16 GiB, and the Runtime card's **VRAM budget** field pins the number
by hand (`settings.json` → `vramGb`, 0 = auto).

Launch flags are fixed to the validated daily config: full offload, `-b 2048 -ub 512 -t 4 -np 1`, `--flash-attn on --kv-unified`, reasoning `--reasoning auto --reasoning-format deepseek --reasoning-effort <level>` plus `--reasoning-preserve` when the preserve toggle (profile `preserveThinking`) is on else `--no-reasoning-preserve`, MTP `--spec-type draft-mtp --spec-draft-n-max N --spec-draft-p-min 0` (ungated — upstream's own default; the confidence gate only pays on bandwidth-starved cards, on this 16 GB card it costs ~32% decode at n-max 3 while *raising* acceptance 63.5% → 91.1%, see [bench/mtp_tuning.md](./bench/mtp_tuning.md); the tab offers depths 0-7, upstream clamps the effective depth to the model's nextn depth, and the draft is unconditional at any ctx — the old “ignore the MTP ctx softcap” checkbox is gone, so a deep draft at large ctx can still OOM or collapse decode), multi-GPU placement `--split-mode` / `--tensor-split` when a profile sets them (default: llama.cpp's own layer split, no flags — the control only appears when more than one GPU is detected), and the KV cache pair from the tab's K/V selectors (`--cache-type-k` / `--cache-type-v`, profile fields `kvTypeK` / `kvTypeV`). Every type this `llama-server` accepts is offered (`f32 f16 bf16 q8_0 q5_1 q5_0 q4_1 iq4_nl q4_0`, labeled with its bytes/element); the default `q5_0` K / `q4_1` V is the measured 16 GB sweet spot, and legacy profiles without the fields launch with exactly that pair. Quantized V needs flash-attn (always on here) and the MTP draft KV stays pinned to `q4_0`. MLA models (DeepSeek-style latent KV) reject mixed K/V types in llama.cpp, so the tab warns and keeps Load disabled until both match, and the `/run` route refuses such a launch with a clear error. Router presets carry the same per-profile KV pair and `reasoning-preserve = 1/0` choice.

## HTTP API (mounted under `/local-models`)

| Route | Meaning |
|---|---|
| `GET /local-models/browse?dir=` | dirs + `.gguf` files |
| `POST /local-models/gguf-meta` | `{path}` → parsed GGUF header (cached) |
| `GET /local-models/status` | state + fresh `/health` probe |
| `GET /local-models/logs?offset=&max=` | incremental tail of `llama-server.log` |
| `POST /local-models/run` | spawn the server |
| `POST /local-models/stop` | stop the child (or reap the port) |
| `POST /local-models/profiles` / `GET` | save (upsert) / list profiles |
| `POST /local-models/profiles/remove` | delete a profile |
| `GET /local-models/settings` / `POST` | read / update plugin settings (`autostartRouter`, `autoUnloadMins`, `binPath`, `shortcuts`, `vramGb`) |
| `POST /local-models/runtime/check` | `{binPath}` → resolve + `<bin> --version` (the Runtime card's Check) |
| `POST /local-models/router/start` | build presets from profiles + start router |
| `POST /local-models/router/unload` | unload one router model |
| `POST /local-models/router/unload-all` | unload all router models |
| `POST /local-models/register` | add the ready server as an `llm-pi-ai` route |

## Project layout

```
lib/index.js    node half: process manager, GGUF parser, routes, presets
lib/client.js   browser half: settings tab (single build-free bundle)
skills/         operator skill: spawn-parity checklist, profile audits
docs/           UI mockup
```

Pure, exported helpers (`normalizeEffort`, `moeArgsFor`, `generateRouterPresets`, `buildProviderProfile`, profiles store) are covered by `npm test` (node's built-in runner, `test/`); `node lib/index.js /path/to/model.gguf` dumps a parsed header as a self-test.

Host-provided modules: `@deepseek-ai/dsh-client-runtime` and `@deepseek-ai/dsh-client-ui-settings` are injected by the dsh host at bundle time (see the `dsh.client.inject` list in `package.json`) and are deliberately **not** in `dependencies` — they don't exist on npm and must not be installed.

## Known issues

See [KNOWN_ISSUES.md](./KNOWN_ISSUES.md) — most notably, the VRAM estimate is approximate for Gemma-family layouts.

## License

MIT — see [LICENSE](./LICENSE).
