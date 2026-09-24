#!/usr/bin/env python3
"""KV-cache VRAM residency study at a fixed test context (default 131072).

The RADV driver over-commits VRAM into system RAM (GTT), so llama-server
reports success even when gigabytes of KV cache sit behind PCIe. Residency is
therefore checked against the allocation budget, not the exit code.

VRAM is sampled while a real completion runs, so the compute buffers (which are
only allocated on first inference) are included in the measurement.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import threading
import time
import urllib.request

SERVER = "/home/marcelo/Projetos/llama.cpp/build/bin/llama-server"
MODEL = "/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf"
VRAM = "/sys/class/drm/card1/device/mem_info_vram_used"
OUTDIR = "/home/marcelo/Projetos/dsh-local-models/bench"
LOGDIR = os.path.join(OUTDIR, "logs")
PORT = 18097
MODEL_MAX_CTX = 262144

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


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self._stop = False

    def run(self):
        while not self._stop:
            v = vram_mib()
            if v > self.peak:
                self.peak = v
            time.sleep(0.15)

    def stop(self):
        self._stop = True


def post_completion(timeout=240):
    """Force a realistic prefill so compute buffers are sized as in real use."""
    prompt = "The quick brown fox jumps over the lazy dog. " * 220  # ~2000 tokens
    body = json.dumps({"prompt": prompt, "n_predict": 4,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion", data=body,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def load_probe(ctk, ctv, ctx, timeout=420, work=True):
    log_path = os.path.join(LOGDIR, f"{ctk}_{ctv}_{ctx}.log")
    cmd = [SERVER, "-m", MODEL, "-c", str(ctx), "-np", "1", "-ngl", "99",
           "-fa", "on", "-ctk", ctk, "-ctv", ctv, "-v",
           "--fit", "off", "--no-webui", "--host", "127.0.0.1",
           "--port", str(PORT)]
    t0 = time.time()
    s = Sampler()
    s.start()
    ok = False
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        while time.time() - t0 < timeout:
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
            time.sleep(0.25)
        if ok and work:
            post_completion()
            time.sleep(0.6)
        s.stop()
        s.join(timeout=2)
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
        "ok": ok, "vram_peak": s.peak,
        "model_mib": grab(r"Vulkan0 model buffer size\s*=\s*([\d.]+) MiB"),
        "kv_mib": grab(r"Vulkan0 KV buffer size\s*=\s*([\d.]+) MiB"),
        "out_mib": grab(r"Vulkan_Host\s+output buffer size\s*=\s*([\d.]+) MiB"),
        "n_ctx": grab(r"llama_context: n_ctx\s+=\s*(\d+)", int),
        "free_mib": grab(r"MiB,\s*(\d+) MiB free", int),
        "dt": time.time() - t0, "log": log_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--configs", default=None)
    ap.add_argument("--skip-verify", action="store_true")
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
    print(f"desktop/other VRAM = {desktop} MiB", flush=True)

    # ---------- calibrate fixed overhead with real inference ----------
    cal = load_probe("f16", "f16", 8192)
    if not cal["ok"] or cal["kv_mib"] is None:
        raise SystemExit(f"calibration failed; see {cal['log']}")
    free = cal["free_mib"]
    model_mib = cal["model_mib"]
    out_mib = cal["out_mib"]
    overhead = (cal["vram_peak"] - desktop) - model_mib - cal["kv_mib"] - out_mib
    fixed = model_mib + overhead + out_mib
    print(f"calibration @c=8192 (with real inference)")
    print(f"  model buffer (Vulkan0)            = {model_mib:,.0f} MiB")
    print(f"  KV buffer @8192 (f16)             = {cal['kv_mib']:,.0f} MiB")
    print(f"  recurrent state + compute buffers = {overhead:,.0f} MiB")
    print(f"  => fixed VRAM cost                = {fixed:,.0f} MiB")
    print(f"  free VRAM available to llama.cpp  = {free:,} MiB")
    print(f"  => KV budget                      = {free - fixed:,.0f} MiB\n")

    # ---------- phase 1 ----------
    ctx = args.ctx
    print(f"=== PHASE 1: residency at c={ctx:,} ===")
    print(f"{'config':<15} {'KV MiB':>8} {'needed':>8} {'free':>7} "
          f"{'SPILL MiB':>10}  verdict")
    rows = []
    for cfg in configs:
        m = load_probe(cfg[0], cfg[1], ctx)
        kv = m["kv_mib"] if m["kv_mib"] is not None else KV_BPT[cfg]*ctx/1048576
        need = fixed + kv
        spill = need - free
        actual = m["vram_peak"] - desktop
        # empirical cross-check: measured usage should reach `need` if resident
        short = need - actual
        verdict = "SPILLS" if (spill > 64 or short > 512) else "resident"
        note = f"  [measured {actual:,.0f}, short {short:,.0f}]" if short > 512 else ""
        print(f"{cfg[0]+'/'+cfg[1]:<15} {kv:>8,.0f} {need:>8,.0f} {free:>7,} "
              f"{max(0,spill):>10,.0f}  {verdict}{note}", flush=True)
        rows.append({
            "type_k": cfg[0], "type_v": cfg[1], "test_ctx": ctx,
            "kv_buffer_mib": round(kv), "fixed_mib": round(fixed),
            "needed_mib": round(need), "free_mib": free,
            "spill_mib": round(max(0, spill)), "verdict": verdict,
            "measured_llama_mib": round(actual), "vram_peak_mib": m["vram_peak"],
            "ok": m["ok"],
        })

    # ---------- phase 2 ----------
    # Use the KV buffer size llama.cpp ACTUALLY reported, not the theoretical
    # block-size arithmetic: some types get padding (q5_1 is +128 MiB @128k).
    MARGIN = 512  # MiB reserved for compute-buffer growth / driver overhead
    budget = free - fixed - MARGIN
    print(f"\n=== PHASE 2: largest context with {MARGIN} MiB safety margin "
          f"(budget {budget:,.0f} MiB) ===")
    print(f"{'config':<15} {'meas B/tok':>11} {'max ctx':>10} {'~k tokens':>10}")
    fit = {}
    for cfg, r in zip(configs, rows):
        bpt = r["kv_buffer_mib"] * 1048576 / ctx
        mx = min(int(budget * 1048576 / bpt) // 1024 * 1024, MODEL_MAX_CTX)
        fit[cfg] = mx
        print(f"{cfg[0]+'/'+cfg[1]:<15} {bpt:>11,.0f} {mx:>10,} {mx/1024:>9.0f}k")

    # ---------- phase 3 ----------
    if not args.skip_verify:
        print(f"\n=== PHASE 3: verify computed sizes stay resident ===")
        print(f"{'config':<15} {'ctx':>9} {'needed':>8} {'measured':>9}  verdict")
        for cfg in configs:
            mx = fit[cfg]
            m = load_probe(cfg[0], cfg[1], mx)
            kv = m["kv_mib"] if m["kv_mib"] is not None else KV_BPT[cfg]*mx/1048576
            need = fixed + kv
            actual = m["vram_peak"] - desktop
            ok = m["ok"] and need <= free + 64 and (need - actual) < 512
            print(f"{cfg[0]+'/'+cfg[1]:<15} {mx:>9,} {need:>8,.0f} {actual:>9,.0f}  "
                  f"{'OK resident' if ok else 'STILL SPILLS'}", flush=True)
            for r in rows:
                if (r["type_k"], r["type_v"]) == cfg:
                    r["fit_ctx"] = mx
                    r["fit_verified"] = "yes" if ok else "no"
                    r["fit_needed_mib"] = round(need)
                    r["fit_measured_mib"] = round(actual)

    out = os.path.join(OUTDIR, "residency.csv")
    if not rows:
        raise SystemExit("no configs probed; nothing to write")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
