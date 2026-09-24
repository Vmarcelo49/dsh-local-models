#!/usr/bin/env python3
"""Turn results_raw.csv + max_ctx.csv into report tables."""
import argparse
import csv
import os
from collections import defaultdict

OUTDIR = "/home/marcelo/Projetos/dsh-local-models/bench"
RAW = os.path.join(OUTDIR, "results_raw.csv")

ORDER = [("f16", "f16"), ("q8_0", "q8_0"), ("q8_0", "q5_1"), ("q8_0", "q4_0"),
         ("q5_1", "q4_1"), ("q5_0", "q4_1"), ("q4_1", "q4_1"),
         ("q4_0", "q4_0"), ("iq4_nl", "iq4_nl")]

KV_BPT = {  # theoretical bytes per token across the 16 KV-carrying layers
    ("f16", "f16"): 65536, ("q8_0", "q8_0"): 34816, ("q8_0", "q5_1"): 28672,
    ("q8_0", "q4_0"): 26624, ("q5_1", "q4_1"): 21504, ("q5_0", "q4_1"): 21504,
    ("q4_1", "q4_1"): 20480, ("q4_0", "q4_0"): 18432, ("iq4_nl", "iq4_nl"): 18432,
}


def order_key(cfg):
    return ORDER.index(cfg) if cfg in ORDER else 99


def load_raw(tags):
    data = defaultdict(list)   # (ctk,ctv,kind) -> list of (depth, ts, sd)
    peaks = {}                 # (ctk,ctv) -> max vram peak seen
    if not os.path.exists(RAW):
        return data, peaks
    with open(RAW) as f:
        for r in csv.DictReader(f):
            if tags and r["tag"] not in tags:
                continue
            cfg = (r["type_k"], r["type_v"])
            # run_kv_sweep.sh rows carry no `kind` column (bench.py rows do):
            # infer it the same way bench.py does (prompt>0 = prefill).
            kind = r.get("kind") or ("pp" if int(r["n_prompt"]) > 0 else "tg")
            key = (cfg[0], cfg[1], kind)
            data[key].append((int(r["n_depth"]), float(r["avg_ts"]),
                              float(r["stddev_ts"])))
            p = int(r["vram_peak_mib"])
            peaks[cfg] = max(peaks.get(cfg, 0), p)
    return data, peaks


def pivot(data, kind):
    """(ctk,ctv) -> {depth: (mean_ts, max_sd)}"""
    out = {}
    for (ctk, ctv, k), rows in data.items():
        if k != kind:
            continue
        by_depth = defaultdict(list)
        for d, ts, sd in rows:
            by_depth[d].append((ts, sd))
        out[(ctk, ctv)] = {
            d: (sum(t for t, _ in v) / len(v), max(s for _, s in v))
            for d, v in by_depth.items()
        }
    return out


def load_residency():
    """Read the residency study output (measured, not theoretical)."""
    out = {}
    path = os.path.join(OUTDIR, "residency.csv")
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                out[(r["type_k"], r["type_v"])] = {
                    "kv_128k": int(r["kv_buffer_mib"]),
                    "measured_bpt": int(r["kv_buffer_mib"]) * 1048576 / int(r["test_ctx"]),
                    "needed_128k": int(r["needed_mib"]),
                    "free": int(r["free_mib"]),
                    "spill_128k": int(r["spill_mib"]),
                    "verdict_128k": r["verdict"],
                    "fit_ctx": int(r["fit_ctx"]) if r.get("fit_ctx") else None,
                    "fit_verified": r.get("fit_verified", ""),
                }
    return out


def fmt_depths(depths):
    return [d for d in sorted(depths)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="v1")
    ap.add_argument("--out", default=os.path.join(OUTDIR, "tables.md"))
    args = ap.parse_args()
    tags = set(args.tags.split(",")) if args.tags else None

    data, peaks = load_raw(tags)
    maxctx = load_residency()
    pp = pivot(data, "pp")
    tg = pivot(data, "tg")

    lines = []

    def w(s=""):
        lines.append(s)

    # ---- table 1: prefill ----
    if pp:
        depths = fmt_depths({d for v in pp.values() for d in v})
        w("### Prefill (prompt processing) — tok/s\n")
        hdr = "| KV (K/V) | KV MiB@32k | " + " | ".join(f"d={d//1024}k" for d in depths) + " |"
        w(hdr)
        w("|" + "---|" * (len(depths) + 2))
        base = pp.get(("f16", "f16"))
        for cfg in sorted(pp, key=order_key):
            bpt = KV_BPT.get(cfg, 0)
            mib = bpt * 32768 / 1048576
            cells = []
            for d in depths:
                if d in pp[cfg]:
                    ts = pp[cfg][d][0]
                    cells.append(f"{ts:,.0f}")
                else:
                    cells.append("—")
            w(f"| {cfg[0]}/{cfg[1]} | {mib:,.0f} | " + " | ".join(cells) + " |")

        w("\n### Prefill — relative to f16/f16 (same depth)\n")
        w("| KV (K/V) | " + " | ".join(f"d={d//1024}k" for d in depths) + " |")
        w("|" + "---|" * (len(depths) + 1))
        for cfg in sorted(pp, key=order_key):
            cells = []
            for d in depths:
                if base and d in pp[cfg] and d in base:
                    pct = (pp[cfg][d][0] / base[d][0] - 1) * 100
                    cells.append(f"{pct:+.1f}%")
                else:
                    cells.append("—")
            w(f"| {cfg[0]}/{cfg[1]} | " + " | ".join(cells) + " |")

    # ---- table 2: generation ----
    if tg:
        depths = fmt_depths({d for v in tg.values() for d in v})
        w("\n### Generation (decode) — tok/s\n")
        w("| KV (K/V) | " + " | ".join(f"d={d//1024}k" for d in depths) + " |")
        w("|" + "---|" * (len(depths) + 1))
        base = tg.get(("f16", "f16"))
        for cfg in sorted(tg, key=order_key):
            cells = []
            for d in depths:
                if d in tg[cfg]:
                    cells.append(f"{tg[cfg][d][0]:,.1f}")
                else:
                    cells.append("—")
            w(f"| {cfg[0]}/{cfg[1]} | " + " | ".join(cells) + " |")

        w("\n### Generation — relative to f16/f16\n")
        w("| KV (K/V) | " + " | ".join(f"d={d//1024}k" for d in depths) + " |")
        w("|" + "---|" * (len(depths) + 1))
        for cfg in sorted(tg, key=order_key):
            cells = []
            for d in depths:
                if base and d in tg[cfg] and d in base:
                    pct = (tg[cfg][d][0] / base[d][0] - 1) * 100
                    cells.append(f"{pct:+.1f}%")
                else:
                    cells.append("—")
            w(f"| {cfg[0]}/{cfg[1]} | " + " | ".join(cells) + " |")

    # ---- table 3: max ctx + vram ----
    w("\n### Context capacity and VRAM residency\n")
    w(f"Free VRAM available to llama.cpp: **{next((v['free'] for v in maxctx.values()), 0):,} MiB** "
      f"(RX 9070 XT 16,304 MiB less desktop). Model buffers 10,617 MiB + ~412 MiB "
      f"recurrent/compute overhead.\n")
    w("| KV (K/V) | KV B/token (measured) | KV @128k MiB | needs @128k | verdict @128k | "
      "spill @128k MiB | max resident ctx (512 MiB margin) | verified |")
    w("|---|---|---|---|---|---|---|---|")
    for cfg in sorted(set(list(KV_BPT) + list(maxctx)), key=order_key):
        m = maxctx.get(cfg)
        if not m:
            w(f"| {cfg[0]}/{cfg[1]} | — | — | — | (not probed) | — | — | — |")
            continue
        fit_s = f"**{m['fit_ctx']:,}** ({m['fit_ctx']/1024:.0f}k)" if m["fit_ctx"] else "—"
        w(f"| {cfg[0]}/{cfg[1]} | {m['measured_bpt']:,.0f} | {m['kv_128k']:,} | "
          f"{m['needed_128k']:,} | {m['verdict_128k']} | {m['spill_128k']:,} | "
          f"{fit_s} | {m['fit_verified']} |")

    txt = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(txt + "\n")
    print(txt)
    print(f"\n[written to {args.out}]")


if __name__ == "__main__":
    main()
