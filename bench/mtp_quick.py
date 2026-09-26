#!/usr/bin/env python3
"""Quick MTP n-max probe: tiny ctx, short generation, read tg straight from the
server's own print_timing lines. No full prefill, no long task."""
import json, os, re, subprocess, sys, time, urllib.request

BIN = "/home/marcelo/Projetos/llama.cpp/build/bin/llama-server"
MODEL = "/mnt/raid0/GGUF/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
PORT = 8099
BASE = f"http://127.0.0.1:{PORT}"
CTX = int(os.environ.get("CTX", "8192"))
NPRED = int(os.environ.get("NPRED", "192"))
LOGDIR = "/home/marcelo/Projetos/dsh-local-models/bench/logs"
VRAM = "/sys/class/drm/card1/device/mem_info_vram_used"

PROMPT = ("// ===== bench/probe =====\n"
          "The dsh-local-models addon spawns llama-server from the Local Models tab. "
          "It computes a VRAM estimate, writes profiles, and registers the route. ") * 12


def post(path, obj, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def vram():
    try:
        return int(open(VRAM).read().strip()) // 1048576
    except Exception:
        return -1


def run(n_max, p_min):
    tag = f"mtp-n{n_max}"
    log = os.path.join(LOGDIR, f"quick-{tag}.log")
    args = [BIN, "-m", MODEL, "-ngl", "999", "-c", str(CTX),
            "-b", "2048", "-ub", "512", "-t", "4", "-np", "1", "--poll", "0",
            "--flash-attn", "on", "--kv-unified",
            "--cache-type-k", "q5_0", "--cache-type-v", "q4_1",
            "--spec-type", "draft-mtp", "--spec-draft-n-max", str(n_max),
            "--spec-draft-type-k", "q4_0", "--spec-draft-type-v", "q4_0",
            "--host", "127.0.0.1", "--port", str(PORT)]
    if p_min is not None:
        args += ["--spec-draft-p-min", str(p_min)]
    print(f"\n=== {tag}  p_min={p_min}  ctx={CTX} ===", flush=True)
    f = open(log, "w")
    p = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 300:
        if p.poll() is not None:
            f.close()
            print("  server died:", open(log).read().splitlines()[-6:])
            return None
        try:
            urllib.request.urlopen(BASE + "/health", timeout=3).read()
            break
        except Exception:
            time.sleep(0.7)
    print(f"  loaded in {time.time()-t0:.0f}s, VRAM {vram()} MiB", flush=True)
    post("/completion", {"prompt": PROMPT, "n_predict": 24, "temperature": 0.0,
                         "top_k": 1, "cache_prompt": False, "ignore_eos": True})
    r = post("/completion", {"prompt": PROMPT, "n_predict": NPRED, "temperature": 0.0,
                             "top_k": 1, "seed": 7, "cache_prompt": True,
                             "ignore_eos": True})
    tm = r.get("timings", {})
    peak = vram()
    f.close()
    p.terminate()
    try:
        p.wait(timeout=30)
    except Exception:
        p.kill()
    txt = open(log, errors="replace").read()
    for ln in txt.splitlines():
        if "n_gen =" in ln or "draft acceptance" in ln or "clamping" in ln or "n_max=" in ln:
            print("  | " + re.sub(r"^\S+ \S+ ", "", ln).strip(), flush=True)
    print(f"  -> tg={tm.get('predicted_per_second'):.2f} tok/s  "
          f"n={tm.get('predicted_n')}  draft={tm.get('draft_n')} "
          f"acc={tm.get('draft_n_accepted')}  wall_vram={peak} MiB", flush=True)
    time.sleep(3)
    return tm.get("predicted_per_second")


if __name__ == "__main__":
    out = {}
    for nm, pm in [(3, 0.75), (4, 0.75), (6, 0.75)]:
        out[nm] = run(nm, pm)
    print("\n===== QUICK MTP SUMMARY (ctx %d, %d tokens generated) =====" % (CTX, NPRED))
    for nm in sorted(out):
        print(f"  n-max {nm}: {out[nm]:.2f} tok/s" if out[nm] else f"  n-max {nm}: failed")
