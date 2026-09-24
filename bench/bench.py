#!/usr/bin/env python3
"""KV-cache quantization benchmark driver for Qwen3.8-27B-UD-IQ3_S (Vulkan, RX 9070 XT).

llama-bench sets n_ctx = n_prompt + n_gen + n_depth, so -d is the number of
already-cached tokens that the measured prefill must attend over.

Stages:
  prefill : -p N -n 0   -> prompt-processing tests only
  gen     : -p 0 -n N   -> text-generation tests only
"""
import argparse
import csv
import io
import os
import subprocess
import sys
import threading
import time

BENCH = "/home/marcelo/Projetos/llama.cpp/build/bin/llama-bench"
MODEL = "/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf"
VRAM = "/sys/class/drm/card1/device/mem_info_vram_used"
OUTDIR = "/home/marcelo/Projetos/dsh-local-models/bench"
RAW = os.path.join(OUTDIR, "results_raw.csv")

# The user's requested configs plus a few reference points.
CONFIGS = [
    ("f16", "f16"),        # full-precision baseline
    ("q8_0", "q8_0"),      # user ask
    ("q8_0", "q5_1"),      # midpoint
    ("q8_0", "q4_0"),      # user ask
    ("q5_1", "q4_1"),      # user ask ("q5 / q4.1")
    ("q5_0", "q4_1"),      # user ask, variant
    ("q4_1", "q4_1"),      # symmetric 4-bit with min
    ("q4_0", "q4_0"),      # most aggressive classic
    ("iq4_nl", "iq4_nl"),  # i-quant 4-bit
]

KEY_CONFIGS = [("f16", "f16"), ("q8_0", "q8_0"), ("q8_0", "q4_0"),
               ("q5_1", "q4_1"), ("q4_0", "q4_0")]


def vram_mib():
    try:
        with open(VRAM) as f:
            return int(f.read().strip()) // 1048576
    except Exception:
        return -1


class VramSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self._stop = False

    def run(self):
        while not self._stop:
            v = vram_mib()
            if v > self.peak:
                self.peak = v
            time.sleep(0.2)

    def stop(self):
        self._stop = True


def parse_csv(text):
    """Extract (type_k, type_v, n_prompt, n_gen, n_depth, avg_ts, stddev_ts, test_time)."""
    rows = []
    header = None
    idx = {}
    for rec in csv.reader(io.StringIO(text)):
        if not rec:
            continue
        if rec[0] == "build_commit":
            header = rec
            idx = {name: i for i, name in enumerate(header)}
            continue
        if header is None or len(rec) != len(header):
            continue
        try:
            rows.append({
                "type_k": rec[idx["type_k"]],
                "type_v": rec[idx["type_v"]],
                "n_prompt": int(rec[idx["n_prompt"]]),
                "n_gen": int(rec[idx["n_gen"]]),
                "n_depth": int(rec[idx["n_depth"]]),
                "avg_ts": float(rec[idx["avg_ts"]]),
                "stddev_ts": float(rec[idx["stddev_ts"]]),
                "test_time": rec[idx["test_time"]],
            })
        except (ValueError, KeyError):
            continue
    return rows


def run_one(ctk, ctv, p, n, depths, reps, extra=None, timeout=7200, fa="on"):
    cmd = [BENCH, "-m", MODEL, "-ngl", "99", "-fa", fa,
           "-ctk", ctk, "-ctv", ctv,
           "-p", str(p), "-n", str(n), "-d", depths,
           "-r", str(reps), "-t", "6", "-o", "csv"]
    if extra:
        cmd += extra
    sampler = VramSampler()
    sampler.start()
    t0 = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        rc, out, err = res.returncode, res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        rc, out, err = -9, "", "TIMEOUT"
    dt = time.time() - t0
    sampler.stop()
    sampler.join(timeout=2)
    return rc, out, err, dt, sampler.peak


def append_rows(tag, rows, peak):
    new = not os.path.exists(RAW)
    with open(RAW, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "type_k", "type_v", "kind", "n_prompt", "n_gen",
                        "n_depth", "avg_ts", "stddev_ts", "test_time", "vram_peak_mib"])
        for r in rows:
            kind = "pp" if r["n_prompt"] > 0 else "tg"
            w.writerow([tag, r["type_k"], r["type_v"], kind, r["n_prompt"],
                        r["n_gen"], r["n_depth"], f"{r['avg_ts']:.3f}",
                        f"{r['stddev_ts']:.3f}", r["test_time"], peak])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prefill", "gen", "custom"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--depths", default="0,16384,32768")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--configs", default=None,
                    help="e.g. 'q8_0:q4_0,f16:f16'; default = all")
    ap.add_argument("--key-only", action="store_true")
    ap.add_argument("--fa", default="on", choices=["on", "off", "auto"])
    args = ap.parse_args()

    if args.configs:
        configs = [tuple(c.split(":")) for c in args.configs.split(",")]
    elif args.key_only:
        configs = KEY_CONFIGS
    else:
        configs = CONFIGS

    os.makedirs(OUTDIR, exist_ok=True)
    tag = args.tag or args.stage
    p = args.prompt if args.stage == "prefill" else 0
    n = 0 if args.stage == "prefill" else args.gen

    print(f"### stage={args.stage} tag={tag} depths={args.depths} reps={args.reps} "
          f"p={p} n={n} configs={len(configs)}", flush=True)

    for ctk, ctv in configs:
        label = f"{ctk}/{ctv}"
        print(f"\n=== [{tag}] {label}  depths={args.depths}  {time.strftime('%H:%M:%S')} ===",
              flush=True)
        rc, out, err, dt, peak = run_one(ctk, ctv, p, n, args.depths, args.reps,
                                         fa=args.fa)
        if rc != 0:
            print(f"  !! FAILED rc={rc} ({dt:.0f}s)")
            for line in (err or "").strip().splitlines()[-6:]:
                print(f"     {line}")
            continue
        rows = parse_csv(out)
        if not rows:
            print(f"  !! no parsable rows ({dt:.0f}s)")
            continue
        append_rows(tag, rows, peak)
        for r in sorted(rows, key=lambda x: (x["n_depth"], x["n_prompt"])):
            kind = "pp" if r["n_prompt"] > 0 else "tg"
            print(f"  {kind}  d={r['n_depth']:>6}  {r['avg_ts']:>9.2f} tok/s  "
                  f"(sd {r['stddev_ts']:.2f})   [{dt:.0f}s, vram {peak} MiB]")
    print(f"\n### stage={tag} complete {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
