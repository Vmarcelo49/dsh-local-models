# dsh-local-models

A `dsh` addon that adds a **Local Models** tab to the dsh Web GUI: pick a `.gguf` file, tune context and speculative decoding, watch a live VRAM estimate, and load it through `llama-server` — then register the running server as an LLM provider in dsh with one click.

Built against stock upstream `llama.cpp` (`llama-server`). No fork, no patches, no build step: the client bundle is hand-written `React.createElement` (no JSX toolchain) and the node half is dependency-free.

## Features

- **Model picker** — in-app file browser (directories + `.gguf` only) with a header-only GGUF parse (architecture, quant, layers, context length, MoE detection) behind `POST /local-models/gguf-meta`
- **Launch options** — context slider (8K steps, capped at the model's trained context) + fine-tune input, fixed MTP draft depth (0–3), thinking level (`off`/`low`/`medium`/`xhigh`), optional vision `mmproj` (GPU or CPU offload), MoE expert placement (`--cpu-moe` / `--n-cpu-moe` / top-k override) with a fit-to-VRAM helper
- **Live VRAM estimate** — weights + Q8_0/Q4_0 KV cache + recurrent state + compute/graph + overhead against 16 GB, with fits / safe-margin / max-ctx-that-fits rows (see [Known issues](./KNOWN_ISSUES.md) for Gemma-family accuracy)
- **Profiles** — save named launch configurations, reload in one click
- **Router mode** — serve all saved profiles from one OpenAI-compatible endpoint (`--models-preset`); models load on demand, one resident at a time by default
- **Register in dsh** — writes the ready server as an `llm-pi-ai` provider route (vision modality + thinking levels included)
- **Terminal overlay** — live tail of the `llama-server` log from the tab

## Requirements

- `dsh` with the `web` profile (the plugin composes into it)
- A `llama-server` binary (upstream `llama.cpp`, Vulkan/CUDA/CPU — whatever your machine uses)
- The VRAM estimate constants target a **16 GB GPU**; they live at the top of `lib/client.js` (`TOTAL_VRAM_BYTES`, `SAFE_MARGIN_BYTES`) if yours differs

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
2. Tune **context**, **Max MTP head** (fixed draft; capped at 3 — deeper collapses at large ctx), **thinking level**, optional **mmproj** and **MoE** settings.
3. **Load model**, watch the status card, inspect output via **Open terminal**.
4. **Register in dsh** — the route (default `local-<alias>`) appears in the Models picker.
5. Alternatively, save **profiles** and **Start router (from profiles)** for a multi-model endpoint.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LOCAL_MODELS_PORT` | `8080` | `llama-server` port |
| `LOCAL_MODELS_BIN` | `~/Projetos/llama.cpp/build/bin/llama-server` | server binary |
| `LOCAL_MODELS_MMPROJ_CPU` | `1` | vision projector weights in RAM (`0` = offload to GPU) |
| `LOCAL_MODELS_ROUTER_MAX` | `1` | max simultaneously resident router models |
| `LOCAL_MODELS_MAX_IMAGE_BYTES` | `10485760` | vision image guard |
| `LOCAL_MODELS_IMAGE_PIXEL_BUDGET` | `4194304` | vision pixel budget |
| `DSH_HOME` | `~/.dsh` | data dir (`local-models/profiles.json`, `llama-server.log`) |

Launch flags are fixed to the validated daily config: full offload, `-b 2048 -ub 512 -t 4 -np 1`, `--flash-attn on --kv-unified`, `--cache-type-k q8_0 --cache-type-v q4_0`, MTP `--spec-type draft-mtp --spec-draft-n-max N --spec-draft-p-min 0.75` (dropped above 131072 ctx).

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

Pure, exported helpers (`normalizeEffort`, `moeArgsFor`, `generateRouterPresets`, `buildProviderProfile`, profiles store) are unit-testable without a running server; `node lib/index.js /path/to/model.gguf` dumps a parsed header as a self-test.

## Known issues

See [KNOWN_ISSUES.md](./KNOWN_ISSUES.md) — most notably, the VRAM estimate is approximate for Gemma-family layouts.

## License

MIT — see [LICENSE](./LICENSE).
