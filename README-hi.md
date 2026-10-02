# dsh-local-models

एक `dsh` addon जो dsh Web GUI में **Local Models** tab जोड़ता है: कोई `.gguf` फ़ाइल चुनें, context और speculative decoding ट्यून करें, लाइव VRAM अनुमान देखें, और उसे `llama-server` के ज़रिए लोड करें — फिर चल रहे server को एक क्लिक में dsh में LLM provider के रूप में register कर दें।

stock upstream `llama.cpp` (`llama-server`) के विरुद्ध बनाया गया। कोई fork नहीं, कोई patch नहीं, कोई build step नहीं: client bundle हाथ से लिखा गया `React.createElement` है (कोई JSX toolchain नहीं) और node half dependency-free है।

## Features

- **Model picker** — in-app file browser (सिर्फ़ directories + `.gguf`) के साथ header-only GGUF parse (architecture, quant, layers, context length, MoE detection), `POST /local-models/gguf-meta` के पीछे
- **Launch options** — context slider (8K steps, model के trained context पर capped) + fine-tune input, KV cache quantization selectors (एक K के लिए, एक V के लिए — हर वह type जो `llama-server` स्वीकार करता है, bytes-per-element दिखाया गया), fixed MTP draft depth (0–7, upstream model की nextn depth पर clamp करता है), thinking level (`off`/`low`/`medium`/`xhigh`) + preserve-thinking toggle (`--reasoning-preserve` बनाम `--no-reasoning-preserve`, default off), वैकल्पिक vision `mmproj` (GPU या CPU offload), MoE expert placement (`--cpu-moe` / `--n-cpu-moe` / top-k override) fit-to-VRAM helper के साथ
- **Live VRAM estimate** — weights + चुने गए K/V cache types + recurrent state + compute/graph + overhead, detected GPU total के विरुद्ध (nvidia-smi / amdgpu sysfs, सभी GPUs पर summed, अज्ञात होने पर 16 GB माना गया), fits / safe-margin / max-ctx-that-fits पंक्तियों के साथ (Gemma-family सटीकता के लिए [Known issues](./KNOWN_ISSUES.md) देखें)
- **Profiles** — नामित launch configurations सहेजें, एक क्लिक में फिर लोड करें
- **Router mode** — सभी सहेजे गए profiles को एक OpenAI-compatible endpoint (`--models-preset`) से serve करें; models माँग पर लोड होते हैं, default रूप से एक बार में एक ही resident रहता है। router शुरू करने पर dsh में उसके models अपने आप (फिर से) register हो जाते हैं — manual Register दबाने की ज़रूरत नहीं।
- **Register in dsh** — तैयार server को `llm-pi-ai` provider route के रूप में लिखता है (vision modality + thinking levels शामिल, max output 131K tokens बताया गया ताकि लंबे xhigh thinking blocks कटें नहीं)
- **Terminal overlay** — tab से ही `llama-server` log का live tail

## Requirements

- `web` profile वाला `dsh` (plugin उसी में compose होता है)
- एक `llama-server` binary (upstream `llama.cpp`, Vulkan/CUDA/CPU — जो भी आपकी मशीन इस्तेमाल करती हो)
- VRAM budget detect होता है (NVIDIA के लिए `nvidia-smi`, AMD के लिए amdgpu sysfs, सभी दिखने वाले GPUs summed) और tab के Runtime card में pin किया जा सकता है; 16 GB fallback और safety margin `lib/client.js` के शीर्ष पर रहते हैं (`TOTAL_VRAM_BYTES`, `SAFE_MARGIN_BYTES`)

## Install

एक plugin dsh **profile** के अंदर रहता है, जो `$DSH_HOME/profiles/<name>` के नीचे एक pnpm project है; `dsh plugin` अपने arguments उसी directory में pnpm को अग्रेषित करता है।

```bash
# from the npm registry
dsh plugin --profile web add dsh-local-models

# straight from git (plain ESM, no build step)
dsh plugin --profile web add github:Vmarcelo49/dsh-local-models

# from a local clone, for development (symlinked: edits apply on reload)
dsh plugin --profile web add link:/path/to/dsh-local-models
```

`dsh plugin add` dependency लिखता है **और** package को `$DSH_HOME/profiles/web/package.json` में `dsh.profile.bundles` के आगे जोड़ देता है — यही array इसे mount करता है, इसलिए हाथ से संपादित करने को कुछ नहीं है। dsh web process को restart करें (bundle composition boot पर होता है), browser refresh करें और Settings → **Local Models** खोलें।

boot किए बिना composition जाँचें, और इसे हटाएँ:

```bash
dsh --profile web --dump-config | grep -A 2 dsh-local-models
dsh plugin --profile web remove dsh-local-models
```

- **pnpm का `PATH` पर होना ज़रूरी है।** `dsh` CLI खुद npm या bun से install हो सकता है, लेकिन profile के अंदर plugin management pnpm का है (`dsh plugin` उसी को shell out करता है और अन्यथा `pnpm was not found` छापता है)।
- **कोई version gate नहीं, कोई exemption नहीं।** package कोई `@deepseek-ai/*` peer dependency declare नहीं करता — यह सिर्फ़ injected services (`settings`, `credentials`, `webServer`) और client slots इस्तेमाल करता है — इसलिए `dsh plugin` dsh mismatch पर इसे कभी अस्वीकार नहीं करता और किसी `dsh plugin allow-version` की ज़रूरत नहीं पड़ती।
- **कोई build step नहीं।** कोई `prepare` script नहीं है, इसलिए वह pnpm `allowBuilds` gate, जिससे git-hosted plugins टकराते हैं, कभी trigger ही नहीं होता।
- **Manifest check.** [`dsh-plugin-dev check`](https://www.npmjs.com/package/dsh-plugin-guide) (`dsh-plugin-guide` से) bundle manifest validate करता है: `cordis.patch.yml`, `dsh.bundle.patch` pointer, `engines` और `files` whitelist।

> Node-half changes (routes, inject list) के लिए dsh restart चाहिए; client-half changes के लिए सिर्फ़ page refresh काफ़ी है।

## Usage

1. **Choose GGUF…** — कोई model फ़ाइल चुनें (Home / Models shortcuts, Up navigation)।
2. **context**, **KV cache K / V**, **Max MTP head** (fixed draft, 0-7; 3 ही ट्यून किया गया sweet spot है — इससे गहरा draft बड़े ctx पर collapse हो जाता है), **thinking level** + **preserve thinking** checkbox, वैकल्पिक **mmproj** और **MoE** settings ट्यून करें।
3. **Load model**, status card देखें, **Open terminal** से output जाँचें।
4. **Register in dsh** — route (default `local-<alias>`) Models picker में दिखने लगता है।
5. वैकल्पिक रूप से **profiles** सहेजें और multi-model endpoint के लिए **Start router (from profiles)** करें।
6. **"Start the router automatically when dsh starts"** (Router card) tick करें ताकि boot पर router शुरू हो और healthy होते ही उसका `local-router` route register हो जाए — tab खोले बिना भी models इस्तेमाल में रहते हैं। कम से कम एक सहेजा हुआ profile चाहिए; प्रगति `llama-server.log` में आती है (`[autostart]` पंक्तियाँ, Open terminal से दिखती हैं)।
7. **Idle eviction** (Router card, "Unload models after …", default 30 min idle) single loads और router दोनों पर upstream `--sleep-idle-seconds` के ज़रिए VRAM मुक्त करता है; सोया हुआ server `/health` का जवाब देता रहता है और अगली request पर अपने आप फिर लोड हो जाता है (एक धीमी request)। `0` इसे बंद कर देता है। अगले start पर लागू होता है — चल रहा server अलग timer इस्तेमाल कर रहा हो तो tab चेतावनी देता है।

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LOCAL_MODELS_PORT` | `8080` | `llama-server` port |
| `LOCAL_MODELS_BIN` | — (auto-detect) | server binary या उसे रखने वाली dir; Runtime card की setting इस पर भारी पड़ती है |
| `LOCAL_MODELS_SHORTCUTS` | — (none) | colon-separated file-browser shortcut dirs (custom labels के लिए `name=path`); सहेजे जाने के बाद Runtime card की folder list कार्यभार संभाल लेती है |
| `LOCAL_MODELS_MMPROJ_CPU` | `1` | vision projector weights RAM में (`0` = GPU पर offload) |
| `LOCAL_MODELS_ROUTER_MAX` | `1` | एक साथ resident router models की अधिकतम संख्या |
| `LOCAL_MODELS_MAX_IMAGE_BYTES` | `10485760` | vision image guard |
| `LOCAL_MODELS_IMAGE_PIXEL_BUDGET` | `4194304` | vision pixel budget |
| `DSH_HOME` | `~/.dsh` | data dir (`local-models/profiles.json`, `local-models/settings.json`, `llama-server.log`) |

tab का VRAM budget hardcoded नहीं, detect होता है: NVIDIA `nvidia-smi` से, AMD sysfs से (`mem_info_vram_total`, मौजूद होने पर product name `pci.ids` से resolve किया जाता है), सभी दिखने वाले GPUs summed, और `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` का सम्मान किया जाता है। जो hardware पढ़ा नहीं जा सकता वह ऐतिहासिक 16 GiB पर लौट आता है, और Runtime card का **VRAM budget** field संख्या हाथ से pin करता है (`settings.json` → `vramGb`, 0 = auto)।

Launch flags सत्यापित daily config पर स्थिर हैं: full offload, `-b 2048 -ub 512 -t 4 -np 1`, `--flash-attn on --kv-unified`, reasoning `--reasoning auto --reasoning-format deepseek --reasoning-effort <level>` तथा preserve toggle (profile `preserveThinking`) चालू होने पर `--reasoning-preserve`, वरना `--no-reasoning-preserve`, MTP `--spec-type draft-mtp --spec-draft-n-max N --spec-draft-p-min 0` (ungated — upstream का अपना default; confidence gate सिर्फ़ bandwidth-starved cards पर फ़ायदा देता है, इस 16 GB card पर यह n-max 3 पर decode का ~32% लेता है जबकि acceptance 63.5% → 91.1% *बढ़ाता* है, देखें [bench/mtp_tuning.md](./bench/mtp_tuning.md); tab 0-7 depths देता है, upstream प्रभावी depth को model की nextn depth पर clamp करता है, और draft किसी भी ctx पर बिना शर्त है — पुराना “ignore the MTP ctx softcap” checkbox हट चुका है, इसलिए बड़े ctx पर गहरा draft अब भी OOM कर सकता है या decode collapse कर सकता है), multi-GPU placement `--split-mode` / `--tensor-split` जब कोई profile उन्हें set करता हो (default: llama.cpp का अपना layer split, कोई flags नहीं — यह control सिर्फ़ तब दिखता है जब एक से ज़्यादा GPU detect हों), और tab के K/V selectors से KV cache pair (`--cache-type-k` / `--cache-type-v`, profile fields `kvTypeK` / `kvTypeV`)। यह `llama-server` जो भी type स्वीकार करता है वह सब दिया जाता है (`f32 f16 bf16 q8_0 q5_1 q5_0 q4_1 iq4_nl q4_0`, उसके bytes/element के साथ labeled); default `q5_0` K / `q4_1` V मापा गया 16 GB sweet spot है, और बिना इन fields वाले legacy profiles ठीक उसी pair के साथ launch होते हैं। Quantized V के लिए flash-attn चाहिए (यहाँ हमेशा on) और MTP draft KV `q4_0` पर pinned रहता है। MLA models (DeepSeek-style latent KV) llama.cpp में मिले-जुले K/V types अस्वीकार करते हैं, इसलिए tab चेतावनी देता है और दोनों के मेल खाने तक Load disabled रखता है, और `/run` route ऐसा launch स्पष्ट error के साथ मना कर देता है। Router presets वही per-profile KV pair और `reasoning-preserve = 1/0` चुनाव साथ ले जाते हैं।

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

शुद्ध, exported helpers (`normalizeEffort`, `moeArgsFor`, `generateRouterPresets`, `buildProviderProfile`, profiles store) `npm test` से covered हैं (node का built-in runner, `test/`); `node lib/index.js /path/to/model.gguf` self-test के रूप में parsed header dump करता है।

Host-provided modules: `@deepseek-ai/dsh-client-runtime` और `@deepseek-ai/dsh-client-ui-settings` को dsh host bundle time पर inject करता है (देखें `package.json` में `dsh.client.inject` list) और ये जान-बूझकर `dependencies` में **नहीं** हैं — ये npm पर मौजूद नहीं हैं और इन्हें install नहीं करना चाहिए।

## Known issues

देखें [KNOWN_ISSUES.md](./KNOWN_ISSUES.md) — सबसे उल्लेखनीय यह कि Gemma-family layouts के लिए VRAM अनुमान अनुमानित है।

## License

MIT — देखें [LICENSE](./LICENSE).
