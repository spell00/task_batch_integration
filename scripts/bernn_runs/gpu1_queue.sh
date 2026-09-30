#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/simonp/task_batch_integration_bernn
DATA=$ROOT/resources_full/task_batch_integration/datasets/cellxgene_census
OUT=$ROOT/results/bernn_openproblems
LOG=$OUT/logs
mkdir -p "$OUT" "$LOG"
wait_ready() {
  local f="$1"
  until python3 - "$f" <<'PY'
import sys, anndata as ad
try:
    a=ad.read_h5ad(sys.argv[1], backed="r")
    print(a.shape)
    a.file.close()
except Exception:
    raise SystemExit(1)
PY
  do sleep 30; done
}
for ds in gtex_v9 immune_cell_atlas mouse_pancreas_atlas; do
  f="$DATA/$ds/log_cp10k/dataset.h5ad"
  wait_ready "$f"
  mkdir -p "$OUT/$ds"
  echo "===== START $ds $(date -Is) GPU1 =====" | tee -a "$LOG/gpu1.log"
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python3 "$ROOT/scripts/bernn_runs/run_dataset.py" \
    --input "$f" --output "$OUT/$ds/integrated.h5ad" \
    --n-hvg 2000 --epochs 200 --warmup 50 --batch-size 1024 --n-layers 2 --layer1 512 \
    2>&1 | tee "$LOG/$ds.log"
  echo "===== DONE $ds $(date -Is) =====" | tee -a "$LOG/gpu1.log"
done
