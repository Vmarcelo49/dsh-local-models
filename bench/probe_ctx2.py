#!/usr/bin/env python3
"""Find the largest context whose KV cache is genuinely RESIDENT in VRAM.

Why this is not just "does it load": the RADV driver over-commits allocations
into system RAM (GTT). llama-server will happily report success while 8+ GiB of
KV cache sits in host memory behind PCIe. So we must check the allocation
budget, not the exit code.

Method
  1. Calibrate fixed overhead (model + recurrent state + compute buffers) from a
     small context where residency is certain.
  2. budget_kv = free_vram - overhead.  max_ctx = budget_kv / bytes_per_token.
  3. Verify each config by loading at max_ctx and confirming VRAM actually grew
     to the predicted level (a spill shows up as a plateau well below it).
"""
import argparse
import csv
import os
import re
import subprocess
import time
import urllib.request

SERVER = "/home/marcelo/Projetos/llama.cpp/build/bin/llama-server"
MODEL = "/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf"
VRAM = "/sys/class/drm/card1/device/mem_info_vram_used"
OUTDIR = "/home/marcelo/Projetos/dsh-local-models/bench"
OUT = os.path.join(OUTDIR, "max_ctx.csv")
LOGDIR = os.path.join(OUTDIR, "logs")
PORT = 18097

MODEL_MAX_CTX = 262144
GRAN = 1024

KV_BPT = {
    ("f16", "f16"): 65536, ("q8_0", "q8_0"): 34816, ("q8_0", "q5_1"): 28672,
    ("q8_0", "q4_0"): 26624, ("q5_1", "q4_1"): 21504, ("q5_0", "q4_1"): 21504,
    ("q4_1", "q4_1"): 20480, ("q4_0", "q4_0"): 18432, ("iq4_nl", "iq4_nl"): 18432,
}
ORDER = [("f16", "f16"), ("q8_0", "q8_0"), ("q8_0", "q5_1"), ("q8_0", "q4_0"),
         ("q5_1", "q4_1"), ("q5_0", "q4_1"), ("q4_1", "q4_1"),
         ("q4_0", "q4_0"), ("iq4_nl", "iq4_nl")]


def vram_mib():
    try:
        with open(VRAM) as f:
            return int(f.read().strip()) // 1048576
    except Exception:
        return -1


def load_probe(ctk, ctv, ctx, timeout=420):
    """Load llama-server at ctx, return measurement dict."""
    log_path = os.path.join(LOGDIR, f"{ctk}_{ctv}_{ctx}.log")
    cmd = [SERVER, "-m", MODEL, "-c", str(ctx), "-np", "1", "-ngl", "99",
           "-fa", "on", "-ctk", ctk, "-ctv", ctv,
           "--fit", "off", "--no-webui", "--host", "127.0.0.1",
           "--port", str(PORT)]
    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
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
            time.sleep(0.3)
        v = vram_mib()
        if v > peak:
            peak = v
        try:
            proc.terminate()
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
    time.sleep(1.5)

    with open(log_path, errors="replace") as f:
        txt = f.read()
    def grab(pat, cast=float):
        m = re.search(pat, txt)
        return cast(m.group(1)) if m else None

    return {
        "ok": ok,
        "vram_peak": peak,
        "model_mib": grab(r"Vulkan0 model buffer size\s*=\s*([\d.]+) MiB"),
        "kv_mib": grab(r"Vulkan0 KV buffer size\s*=\s*([\d.]+) MiB"),
        "out_mib": grab(r"Vulkan_Host\s+output buffer size\s*=\s*([\d.]+) MiB"),
        "n_ctx": grab(r"llama_context: n_ctx\s+=\s*(\d+)", int),
        "free_mib": grab(r"MiB,\s*(\d+) MiB free", int),
        "dt": time.time() - t0,
        "log": log_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=None)
    args = ap.parse_args()
    configs = ([tuple(c.split(":")) for c in args.configs.split(",")]
               if args.configs else ORDER)
    for cfg in configs:
        if len(cfg) != 2 or cfg not in KV_BPT:
            raise SystemExit(
                f"unknown KV config {':'.join(cfg)} "
                f"(valid: {', '.join(f'{a}:{b}' for a, b in ORDER)})")

    os.makedirs(LOGDIR, exist_ok=True)
    desktop = vram_mib()
    print(f"desktop VRAM = {desktop} MiB", flush=True)

    # ---- calibrate overhead at a context that certainly fits ----
    CAL_CTX = 8192
    cal = load_probe("f16", "f16", CAL_CTX)
    if not cal["ok"]:
        raise SystemExit(f"calibration load failed: {cal['log']}")
    free = cal["free_mib"]
    # llama-only VRAM at calibration ctx (subtract desktop)
    llama_cal = cal["vram_peak"] - desktop
    kv_cal = cal["kv_mib"]
    overhead = llama_cal - kv_cal
    print(f"calibration @c={CAL_CTX}: llama_vram={llama_cal} MiB "
          f"(model={cal['model_mib']} kv={kv_cal} out={cal['out_mib']})")
    print(f"  => fixed overhead (model+recurrent+compute) = {overhead:.0f} MiB")
    print(f"  => free VRAM reported by llama.cpp = {free} MiB")
    budget_kv = free - overhead
    print(f"  => KV budget = {budget_kv:.0f} MiB\n")

    rows = []
    print(f"{'config':<16} {'B/tok':>7} {'analytic max':>13} "
          f"{'verified':>9} {'peak MiB':>9} {'predicted':>10} {'verdict':>10}")
    for cfg in configs:
        bpt = KV_BPT[cfg]
        est = int(budget_kv * 1048576 / bpt)
        est = min((est // GRAN) * GRAN, MODEL_MAX_CTX)

        m = load_probe(cfg[0], cfg[1], est)
        predicted = overhead + (m["kv_mib"] or 0)
        # A spill shows up as measured VRAM far below the predicted requirement.
        spilled = (m["kv_mib"] is not None and
                   (m["vram_peak"] - desktop) < predicted - 512)
        verdict = "SPILL" if spilled else ("ok" if m["ok"] else "FAIL")
        print(f"{cfg[0]+'/'+cfg[1]:<16} {bpt:>7} {est:>13,} "
              f"{str(m['ok']):>9} {m['vram_peak']:>9,} {predicted:>10,.0f} {verdict:>10}",
              flush=True)
        rows.append({
            "type_k": cfg[0], "type_v": cfg[1], "kv_bytes_per_token": bpt,
            "max_ctx": est if not spilled else "",
            "model_max_ctx": MODEL_MAX_CTX,
            "kv_buffer_mib": m["kv_mib"], "vram_peak_mib": m["vram_peak"],
            "predicted_llama_mib": round(predicted),
            "overhead_mib": round(overhead), "free_vram_mib": free,
            "verdict": verdict, "log": os.path.basename(m["log"]),
        })

    if not rows:
        raise SystemExit("no configs probed; nothing to write")
    # Fresh study per invocation ("w", not append): the write happens once
    # here for all configs — split --configs runs would clobber each other,
    # so pass the full set in a single invocation.
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
