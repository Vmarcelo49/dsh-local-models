#!/usr/bin/env python3
"""Fast interleaved A/B prefill comparison.

Cost is dominated by the depth fill (llama-bench re-decodes `n_depth` tokens for
every measurement), so this keeps ONE deep point and many short interleaved
rounds rather than a few long ones. Round order rotates (Latin square) and the
analysis uses PAIRED within-round differences, which cancels drift.

Prints partial results after every round.
"""
import argparse
import statistics
import time

import bench as B


def report(results, configs, depths, base, reps, title):
    print(f"\n--- {title} ---")
    hdr = "| KV (K/V) | " + " | ".join(f"d={d//1024}k" for d in depths) + " |"
    print(hdr)
    print("|" + "---|" * (len(depths) + 1))
    for cfg in configs:
        cells = []
        for d in depths:
            vals = [v for _, v in results[cfg][d]]
            cells.append(f"{statistics.mean(vals):,.0f}" if vals else "—")
        tag = " (base)" if cfg == base else ""
        print(f"| {cfg[0]}/{cfg[1]}{tag} | " + " | ".join(cells) + " |")

    for cfg in configs:
        if cfg == base:
            continue
        cells = []
        for d in depths:
            bmap = dict(results[base][d])
            cmap = dict(results[cfg][d])
            diffs = [cmap[r] - bmap[r] for r in sorted(bmap) if r in cmap]
            if not diffs:
                cells.append("—")
            elif len(diffs) == 1:
                cells.append(f"{diffs[0]:+,.0f}")
            else:
                cells.append(f"{statistics.mean(diffs):+,.0f} "
                             f"±{statistics.stdev(diffs):,.0f}")
        print(f"| {cfg[0]}/{cfg[1]} vs base | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="f16:f16,q8_0:q8_0")
    ap.add_argument("--depths", default="0,32768")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--tag", default="fast")
    ap.add_argument("--baseline", default="f16:f16")
    args = ap.parse_args()

    configs = [tuple(c.split(":")) for c in args.configs.split(",")]
    base = tuple(args.baseline.split(":"))
    if base not in configs:
        raise SystemExit(
            f"--baseline {args.baseline} is not one of --configs "
            f"({args.configs}); report() compares every config against it.")
    depths = [int(d) for d in args.depths.split(",")]
    results = {c: {d: [] for d in depths} for c in configs}

    t_start = time.time()
    for r in range(args.rounds):
        order = configs[r % len(configs):] + configs[:r % len(configs)]
        print(f"\n### round {r}  order={['/'.join(c) for c in order]}  "
              f"{time.strftime('%H:%M:%S')}", flush=True)
        for cfg in order:
            rc, out, err, dt, peak = B.run_one(
                cfg[0], cfg[1], args.prompt, 0, args.depths, args.reps,
                extra=["--no-warmup"])
            if rc != 0:
                print(f"  !! {cfg[0]}/{cfg[1]} FAILED rc={rc}: "
                      f"{(err or '').strip().splitlines()[-1:]}")
                continue
            rows = B.parse_csv(out)
            B.append_rows(f"{args.tag}-r{r}", rows, peak)
            got = []
            for row in rows:
                if row["n_prompt"] > 0:
                    if row["n_depth"] in results[cfg]:
                        results[cfg][row["n_depth"]].append((r, row["avg_ts"]))
                        got.append(f"d={row['n_depth']//1024}k:{row['avg_ts']:.0f}")
                    else:
                        print(f"  [warn] ignoring unrequested depth "
                              f"{row['n_depth']} (not in --depths)")
            print(f"  {cfg[0]+'/'+cfg[1]:<12} {'  '.join(got)}  [{dt:.0f}s]",
                  flush=True)
        report(results, configs, depths, base, args.reps,
               f"partial after round {r}")

    print(f"\n\n========== FINAL ==========")
    report(results, configs, depths, base, args.reps,
           f"prefill pp{args.prompt}, {args.rounds} paired rounds x {args.reps} rep")
    print(f"\ntotal wall time {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
