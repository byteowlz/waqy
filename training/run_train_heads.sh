#!/usr/bin/env bash
# Train one head per address form plus the bare head for a word pack, reusing extracted features.
#
# Recipe (see ADR 0007 and CHANGELOG 2026-10-29):
#   - addressed heads (hey_<word>) train with the other words' windows as negatives, otherwise every
#     head fires on "Hey <any word>";
#   - max_negative_weight stays at 1: the default ramp to 3000 with 1024 general negatives per batch
#     collapses these smaller sets to constant-negative (measured on 24 heads);
#   - balanced batches, small model, 6000 steps: ~45-75 s per head on a 4090.
#
# Usage: bash training/run_train_heads.sh <word> [<word> ...]
set -euo pipefail
R=${WAQY_ROOT:-$HOME/.local/share/foxline/wake-train-1}
P=$R/wordpack-1/output
cd "$R/code"
export WAKE_DEVICE=${WAKE_DEVICE:-cuda} WAKE_GPU_GB=1.2 OMP_NUM_THREADS=4
export WAKE_BATCH_JSON='{"positive": 200, "adversarial_negative": 200, "ACAV100M_sample": 200, "background_noise": 50}'
ADVERSARIAL=${ADVERSARIAL:-1}

for w in "${@:-elissa}"; do
  for m in "hey_$w" "$w"; do
    if [ "$ADVERSARIAL" = 1 ] && [ "$m" = "hey_$w" ]; then
      .venv/bin/python adv_negative.py "$w" "$R/wordpack-1" > /dev/null
      outdir=$P   # export reads <root>/output, so the augmented set lives there too
    else
      outdir=$P
    fi
    [ -d "$outdir/$m" ] || { echo "no features for $m"; continue; }
    if [ -f "$outdir/$m/$m.onnx" ]; then echo "skip $m (done)"; continue; fi
    echo "== $m $(date +%H:%M:%S)"
    .venv/bin/python pipeline.py "$R" "$m" config > /dev/null
    sed -i "s|^output_dir: .*|output_dir: $outdir|; s/^steps: .*/steps: 6000/; \
            s/^max_negative_weight: .*/max_negative_weight: 1/; s/model_size: .*/model_size: small/" \
      "$R/configs/$m.yaml"
    if .venv/bin/python pipeline.py "$R" "$m" train > "$R/logs/pack-$m.log" 2>&1 &&
       .venv/bin/python export_head.py "$R/configs/$m.yaml" "$outdir/$m/$m.pt" "$outdir/$m/$m.onnx" \
         >> "$R/logs/pack-$m.log" 2>&1; then
      # an export that predicts a constant looks fine and packages fine, so gate on the head's own data
      if ! .venv/bin/python verify_head.py "$outdir/$m/$m.onnx" "$outdir/$m/positive_features_test.npy"; then
        echo "COLLAPSED $m"; rm -f "$outdir/$m/$m.onnx"
      fi
    else
      echo "FAILED $m"; tail -3 "$R/logs/pack-$m.log" | cut -c1-120
    fi
  done
done
echo "TRAIN DONE $(date +%H:%M:%S)"
