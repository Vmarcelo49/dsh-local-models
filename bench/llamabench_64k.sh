#!/bin/bash
# llama-bench cross-check (no draft model) in the same methodology as
# bench/tables_v1.md: -d is the number of already-cached tokens the measured
# prefill attends over. Here d=63488 with -p 2048 -> n_ctx = 65664 (~64K),
# so the row slots straight into the v1 tables.
set -uo pipefail

BENCH=/home/marcelo/Projetos/llama.cpp/build/bin/llama-bench
MODEL=/mnt/raid0/GGUF/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf
OUTDIR=/home/marcelo/Projetos/dsh-local-models/bench
VRAM=/sys/class/drm/card1/device/mem_info_vram_used
GTT=/sys/class/drm/card1/device/mem_info_gtt_used

PEAK=$OUTDIR/.lb_vram.$$
echo 0 > "$PEAK"; echo 0 > "$PEAK.gtt"
(
  while true; do
    v=$(cat "$VRAM" 2>/dev/null || echo 0); p=$(cat "$PEAK" 2>/dev/null || echo 0)
    [ "$v" -gt "$p" ] && echo "$v" > "$PEAK"
    g=$(cat "$GTT" 2>/dev/null || echo 0); q=$(cat "$PEAK.gtt" 2>/dev/null || echo 0)
    [ "$g" -gt "$q" ] && echo "$g" > "$PEAK.gtt"
    sleep 0.2
  done
) &
SAMPLER=$!

echo "idle VRAM=$(( $(cat $VRAM)/1048576 )) MiB  GTT=$(( $(cat $GTT)/1048576 )) MiB"
echo "=== llama-bench q5_0/q4_1 d=0,63488 $(date +%H:%M:%S) ==="
"$BENCH" -m "$MODEL" -ngl 99 -fa on -ctk q5_0 -ctv q4_1 \
    -p 2048 -n 128 -d 0,63488 -r 3 -t 4 -b 2048 -ub 512 -o csv

kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null
echo "VRAM peak=$(( $(cat $PEAK)/1048576 )) MiB   GTT peak=$(( $(cat $PEAK.gtt)/1048576 )) MiB"
rm -f "$PEAK" "$PEAK.gtt"
