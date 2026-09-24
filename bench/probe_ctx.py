#!/usr/bin/env python3
"""Find the largest context that fits ENTIRELY in VRAM for each KV cache config.

Uses llama-server because it allocates the full KV cache at load time without
filling it (llama-bench would decode -d tokens, taking minutes at 256k).

--fit is forced OFF so llama.cpp cannot silently reduce n_gpu_layers or n_ctx.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
import urllib.request

SERVER = "/home/marcelo/Projetos/llama.cpp/build/bin/llama-server"
MODEL = "/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf"
VRAM = "/sys/class/drm/card1/device/mem_info_vram_used"
OUTDIR = "/home/marcelo/Projetos/dsh-local-models/bench"
OUT = os.path.join(OUTDIR, "max_ctx.csv")
PORT = 18099

KV_BYTES_PER_TOKEN = {
    ("f16", "f16"): 65536,
    ("q8_0", "q8_0"): 34816,
    ("q8_0", "q5_1"): 28672,
    ("q8_0", "q4_0"): 26624,
    ("q5_1", "q4_1"): 21504,
    ("q5_0", "q4_1"): 21504,
    ("q4_1", "q4_1"): 20480,
    ("q4_0", "q4_0"): 18432,
    ("iq4_nl", "iq4_nl"): 18432,
}

MODEL_MAX_CTX = 262144
GRANULARITY = 2048

# Seed-estimate budget: free-VRAM headroom measured on the reference box
# (RX 9070 XT). DESKTOP_MIB is the idle-desktop reservation subtracted from
# the baseline reading; FREE_MIB is the free VRAM llama.cpp reported at
# tuning time. Both are machine-specific — override with --budget-mib on
# any other box instead of trusting these numbers.
DESKTOP_MIB = 570
FREE_MIB = 15733


def vram_mib():
    try:
        with open(VRAM) as f:
            return int(f.read().strip()) // 1048576
    except Exception:
        return -1


def wait_vram_released(baseline, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if vram_mib() <= baseline + 400:
            return True
        time.sleep(0.5)
    return False


def probe(ctk, ctv, ctx, timeout=300):
    """Return (ok, peak_mib, elapsed)."""
    cmd = [SERVER, "-m", MODEL, "-c", str(ctx), "-ngl", "99", "-fa", "on",
           "-ctk", ctk, "-ctv", ctv, "--fit", "off", "--no-webui",
           "--host", "127.0.0.1", "--port", str(PORT)]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak = 0
    ok = False
    while time.time() - t0 < timeout:
        v = vram_mib()
        if v > peak:
            peak = v
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT}/health", timeout=1) as r:
                if r.status == 200 and b'"ok"' in r.read():
                    ok = True
                    break
        except Exception:
            pass
        time.sleep(0.4)
    v = vram_mib()
    if v > peak:
        peak = v
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
    return ok, peak, time.time() - t0


def find_max(ctk, ctv, baseline, budget_mib, log):
    bpt = KV_BYTES_PER_TOKEN.get((ctk, ctv))
    if bpt is None:
        raise SystemExit(
            f"unknown KV config {ctk}:{ctv} "
            f"(valid: {', '.join(f'{a}:{b}' for a, b in KV_BYTES_PER_TOKEN)})")
    headroom_bytes = max(0, budget_mib - (baseline - DESKTOP_MIB)) * 1048576
    est = int(headroom_bytes / bpt)
    est = max(GRANULARITY, min(est, MODEL_MAX_CTX))
    est = (est // GRANULARITY) * GRANULARITY

    log(f"  seed estimate = {est}")

    ok, peak, dt = probe(ctk, ctv, est)
    log(f"    probe c={est:>6} -> {'OK ' if ok else 'FAIL'} (peak {peak} MiB, {dt:.0f}s)")
    wait_vram_released(baseline)

    if ok:
        lo, hi = est, MODEL_MAX_CTX
        if est >= MODEL_MAX_CTX:
            return est, peak
        # grow until failure
        step = max(GRANULARITY, est // 4)
        cur = est
        while cur < MODEL_MAX_CTX:
            nxt = min(cur + step, MODEL_MAX_CTX)
            ok2, peak2, dt2 = probe(ctk, ctv, nxt)
            log(f"    probe c={nxt:>6} -> {'OK ' if ok2 else 'FAIL'} (peak {peak2} MiB, {dt2:.0f}s)")
            wait_vram_released(baseline)
            if ok2:
                lo, peak, cur = nxt, peak2, nxt
                if nxt >= MODEL_MAX_CTX:
                    return nxt, peak
                step *= 2
            else:
                hi = nxt
                break
        else:
            return lo, peak
    else:
        hi, lo = est, GRANULARITY
        # shrink until success
        cur = est
        while cur > GRANULARITY:
            nxt = max(GRANULARITY, cur // 2)
            nxt = (nxt // GRANULARITY) * GRANULARITY
            ok2, peak2, dt2 = probe(ctk, ctv, nxt)
            log(f"    probe c={nxt:>6} -> {'OK ' if ok2 else 'FAIL'} (peak {peak2} MiB, {dt2:.0f}s)")
            wait_vram_released(baseline)
            if ok2:
                lo, peak = nxt, peak2
                break
            cur = nxt
        else:
            return GRANULARITY, peak

    # binary search between lo (ok) and hi (fail)
    while hi - lo > GRANULARITY:
        mid = ((lo + hi) // 2 // GRANULARITY) * GRANULARITY
        if mid <= lo or mid >= hi:
            break
        ok3, peak3, dt3 = probe(ctk, ctv, mid)
        log(f"    probe c={mid:>6} -> {'OK ' if ok3 else 'FAIL'} (peak {peak3} MiB, {dt3:.0f}s)")
        wait_vram_released(baseline)
        if ok3:
            lo, peak = mid, peak3
        else:
            hi = mid
    return lo, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=None)
    ap.add_argument("--only", default=None, help="single config ctk:ctv")
    ap.add_argument("--budget-mib", type=int, default=FREE_MIB,
                    help="free-VRAM headroom in MiB for the seed estimate "
                         f"(default {FREE_MIB}, tuned on the reference box)")
    args = ap.parse_args()

    if args.only:
        configs = [tuple(args.only.split(":"))]
    elif args.configs:
        configs = [tuple(c.split(":")) for c in args.configs.split(",")]
    else:
        configs = list(KV_BYTES_PER_TOKEN.keys())

    baseline = vram_mib()
    print(f"VRAM baseline (desktop) = {baseline} MiB", flush=True)

    new = not os.path.exists(OUT)
    f = open(OUT, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["type_k", "type_v", "max_ctx", "probe_peak_mib",
                    "kv_bytes_per_token_theory", "when"])
    f.flush()

    def log(msg):
        print(msg, flush=True)

    for ctk, ctv in configs:
        log(f"\n=== {ctk}/{ctv} ({time.strftime('%H:%M:%S')}) ===")
        best, peak = find_max(ctk, ctv, baseline, args.budget_mib, log)
        log(f"  => MAX CTX {ctk}/{ctv} = {best}  (vram peak {peak} MiB)")
        w.writerow([ctk, ctv, best, peak, KV_BYTES_PER_TOKEN.get((ctk, ctv), ""),
                    time.strftime("%Y-%m-%dT%H:%M:%S")])
        f.flush()
    f.close()
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
