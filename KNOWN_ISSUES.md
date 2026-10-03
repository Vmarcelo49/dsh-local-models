# Known issues

Current, observed limitations — read before trusting affected features.

## Mid-size windows compact far below the advertised context (131K loops at ~38K)

**Observed:** on the Swift 131K profile (xhigh + preserveThinking),
dsh starts compacting around 32-38K pressure tokens and then re-compacts
on nearly every step — each summarization replays the whole prefix on the
local 27B (minutes per attempt), so the session feels stuck in a
compaction loop while 2/3 of the window sits empty.

**Why:** the registered `maxTokens: 32768` becomes compaction's reserved
output `O`, and under the dsh-compaction-basic defaults
(`headroomTokens: 65536`, `thresholdRatio: 0.8`) the trigger is
`min(0.8 * W, W - O - headroom)`. For W = 131072 that is 32768 — the 32K
cap keeps a message budget alive but cannot buy a late threshold while the
64K default headroom (calibrated for 1M-token cloud models) stands. The
meter prices tools + system on top of the surface, so the visible trigger
lands a few K above 32768; xhigh thinking then re-adds 10-20K tokens per
turn and the threshold is crossed again immediately. A 98304 window is the
same cliff with the opposite symptom: pressure exactly 0, so proactive
compaction never fires at all (warn-once, overflow recovery only).

**Impact:** 96-131K profiles compact far too early (or never) with a stock
dsh profile. Loading and generation are unaffected — only the compaction
schedule is wrong.

**Workarounds / guidance:**
- Per-model headroom override, placed on the **agent preset row**
  (`- id: preset-standard`, wholesale `config` copied from the installed
  `dsh-web-app/presets/standard.patch.yml` plus `modelPolicies` on its
  nested `plugins → compaction → compaction-basic` engine), then restart
  `dsh web`. A patch on the top-level `compaction-basic` row does NOTHING
  (that row is disabled; the running engines live in the preset scopes).
  Because the preset `config` is replaced whole, re-generate it after any
  dsh upgrade touching the preset file.
  (~92K threshold for 131K via `headroomTokens: 6554`;
  `compactionBudgetFor(ctx).recommendedHeadroom` in `lib/index.js`
  computes the value per window — ~4K floor for 96K.)
  Do NOT shrink the advertised `maxTokens` instead: that truncates the
  xhigh thinking blocks the 32K cap exists to protect.
- Zero-config alternative: run the 200K/250K profiles — they threshold at
  ~106K/158K under defaults (verify VRAM: they use the lighter q4_0 KV).

## VRAM estimate is inaccurate for Gemma-family models

**Observed:** loading a Gemma model, the VRAM estimate in the Local Models tab
does not add up against what the loaded model actually uses (the totals are
visibly off from measurements/laid-out sizes).

**Why:** the estimate (client-side, in `lib/client.js`) is a port of the
Qwen/llama-family KV-cache and graph formulas: standard per-head KV bytes
(the tab's selected K/V cache types — `Q5_0 (K) / Q4_1 (V)` by default),
sliding-window local/global attention share, GDN
recurrent state, and a fixed compute/graph + 200 MB overhead. Gemma models use
different attention/layout assumptions (interleaved global-local attention
structure and head configs that do not match the formula's share or head-dim
arithmetic), so the derived `Max ctx that fits`, `Fits 16 GB`, and Total rows
are approximate for Gemma and must not be treated as exact.

**Impact:** the estimate card and the max-context suggestion may be wrong for
Gemma checkpoints. Loading is not affected — the server still starts with the
chosen context — only the estimation math is off.

**Workarounds / guidance:**
- Treat the VRAM card as advisory for Gemma models; verify against the running
  server's actual use (e.g. llama-server logs/VRAM monitor).
- Fine-tune manually via the context input below the slider; the estimate's
  "Fits" flags are hints, not guards.
- A proper fix requires reading Gemma's architecture metadata (attention
  intervals, head dims) from the GGUF header instead of assuming the Qwen
  layout — tracked as a future improvement; contributions welcome.

## Notes
- Thinking levels are probe-validated per model chat template (see README);
  the tab only offers levels the loaded template accepts.
- **Fixed MTP depth is capped at 3.** Upstream fixed draft depth above 3
  collapses at large ctx (measured ~12 tok/s at n=6/131K vs ~57 at n=3),
  and the draft is dropped entirely above ctx 131072 (upstream has no
  draft-KV auto-quant). The tab offers 0-3; the API clamps higher values.
- **KV cache types are per-model, not per-request.** `--cache-type-k` /
  `--cache-type-v` are launch settings: changing a selector in the tab only
  takes effect on the next Load (the running server keeps its pair; the status
  card shows it).
- **MLA models need matching K/V cache types.** llama.cpp rejects mixed types
  on DeepSeek-style latent-KV models; the tab flags it and disables Load, and
  `POST /run` refuses the launch with a clear error instead of letting the
  child die on its own.
- See README for architecture and the full configuration surface.

## MoE expert bytes are an offset-delta heuristic

**Observed:** nothing user-visible yet — but the **Experts in RAM** row and the
reduced weights total on MoE loads are derived, not measured.

**Why:** expert weights are sized from tensor offset deltas (next offset minus
own offset), which assumes the standard dense-packed GGUF data layout. Gaps,
padding, or non-monotonic layouts skew the split; shared experts (`*shexp*`,
always active) are intentionally excluded from the CPU-movable total.

**Impact:** the MoE VRAM total is advisory — same standing as the rest of the
estimate card. Launch flags themselves are exact (`--cpu-moe` /
`--n-cpu-moe` / `--override-kv` pass through verbatim).

**Guidance:** if the layout is non-standard the parser reports
`expertBytes: null` and the estimate falls back to full weights in VRAM with
a footnote. Verify against the running server's actual use before trusting
tight fits.
