#!/bin/bash
# KV-cache quantization sweep for Qwen3.8-27B-UD-IQ3_S on RX 9070 XT (Vulkan)
#
# llama-bench sets n_ctx = n_prompt + n_gen + n_depth, so -d is the number of
# already-cached tokens the measured prefill attends over.
#
# Usage: run_kv_sweep.sh <depths_csv> <reps> <tag> <ctk:ctv> [<ctk:ctv> ...]

set -uo pipefail

BENCH=/home/marcelo/Projetos/llama.cpp/build/bin/llama-bench
MODEL=/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf
OUTDIR=/home/marcelo/Projetos/dsh-local-models/bench
RAW=$OUTDIR/results_raw.csv
VRAM=/sys/class/drm/card1/device/mem_info_vram_used

DEPTHS=$1; shift
REPS=$1; shift
TAG=$1; shift
CONFIGS=("$@")

PROMPT=2048
GEN=128

mkdir -p "$OUTDIR"
[ -f "$RAW" ] || echo "tag,type_k,type_v,n_prompt,n_gen,n_depth,avg_ts,stddev_ts,test_time,vram_peak_mib" > "$RAW"

for cfg in "${CONFIGS[@]}"; do
    CTK="${cfg%%:*}"; CTV="${cfg##*:}"
    echo "=== [$TAG] ctk=$CTK ctv=$CTV depths=$DEPTHS reps=$REPS  $(date +%H:%M:%S) ==="

    # VRAM high-water sampler
    PEAK=$OUTDIR/.vram_peak.$$
    echo 0 > "$PEAK"
    (
      while true; do
        v=$(cat "$VRAM" 2>/dev/null || echo 0)
        p=$(cat "$PEAK" 2>/dev/null || echo 0)
        [ "$v" -gt "$p" ] && echo "$v" > "$PEAK"
        sleep 0.25
      done
    ) &
    SAMPLER=$!

    TMP=$OUTDIR/.bench_out.$$
    timeout 3600 "$BENCH" \
        -m "$MODEL" -ngl 99 -fa on \
        -ctk "$CTK" -ctv "$CTV" \
        -p $PROMPT -n $GEN -d "$DEPTHS" \
        -r "$REPS" -t 6 -o csv \
        > "$TMP" 2>"$TMP.err"
    RC=$?

    kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null
    PEAKVAL=$(( $(cat "$PEAK" 2>/dev/null || echo 0) / 1048576 ))
    rm -f "$PEAK"

    if [ $RC -ne 0 ]; then
        echo "  !! FAILED rc=$RC"
        tail -5 "$TMP.err" | sed 's/^/     /'
    else
        # Header-based parse, one python per config (not per line): immune
        # to llama-bench column reorderings. Columns, in RAW order:
        # tag,type_k,type_v,n_prompt,n_gen,n_depth,avg_ts,stddev_ts,
        # test_time,vram_peak_mib
        python3 - "$TAG" "$TMP" "$PEAKVAL" >> "$RAW" <<'PY'
import sys, csv
tag, path, peak = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, newline="") as f:
    rows = [r for r in csv.reader(f) if r]
header = next((r for r in rows if r[0] == "build_commit"), None)
if header is None:
    raise SystemExit("no CSV header in llama-bench output")
idx = {name: i for i, name in enumerate(header)}
need = ["type_k", "type_v", "n_prompt", "n_gen", "n_depth",
        "test_time", "avg_ts", "stddev_ts"]
missing = [n for n in need if n not in idx]
if missing:
    raise SystemExit(f"llama-bench CSV missing columns: {missing}")
out = []
for r in rows:
    if r[0] == "build_commit" or len(r) != len(header):
        continue
    out.append(",".join([tag, r[idx["type_k"]], r[idx["type_v"]],
                         r[idx["n_prompt"]], r[idx["n_gen"]], r[idx["n_depth"]],
                         r[idx["avg_ts"]], r[idx["stddev_ts"]],
                         r[idx["test_time"]], peak]))
sys.stdout.write("".join(l + "\n" for l in out))
PY
        grep -v '^$' "$TMP" | \
          python3 -c "
import sys,csv
rows=[r for r in csv.reader(sys.stdin) if r]
hdr=next((r for r in rows if r[0]=='build_commit'),None)
if hdr is None: raise SystemExit('no CSV header in llama-bench output')
idx={n:i for i,n in enumerate(hdr)}
for r in rows:
    if r is hdr or len(r)!=len(hdr): continue
    print(f'  d={r[idx[\"n_depth\"]]:>6s}  pp={r[idx[\"avg_ts\"]]} tok/s   (+-{r[idx[\"stddev_ts\"]]})')
"
    fi
    rm -f "$TMP" "$TMP.err"
done
echo "=== [$TAG] done $(date +%H:%M:%S) ==="
