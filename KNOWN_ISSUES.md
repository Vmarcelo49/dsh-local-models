# Known issues

Current, observed limitations — read before trusting affected features.

## VRAM estimate is inaccurate for Gemma-family models

**Observed:** loading a Gemma model, the VRAM estimate in the Local Models tab
does not add up against what the loaded model actually uses (the totals are
visibly off from measurements/laid-out sizes).

**Why:** the estimate (client-side, in `lib/client.js`) is a port of the
Qwen/llama-family KV-cache and graph formulas: standard per-head KV bytes
(`Q8_0 (K) / Q4_0 (V)`), sliding-window local/global attention share, GDN
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
