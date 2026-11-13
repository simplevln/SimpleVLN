#!/bin/bash
# Fresh trajectory generation, followed by VLM instruction labeling.
set -euo pipefail
cd "$(dirname "$0")"
export VLN_DATA="${VLN_DATA:-/root/code/data}"
PY=${PY:-python3}

mp3d() {
  local OUT=${MP3D_OUT:-$VLN_DATA/datasets/r2r_rl_v3}
  local SHARDS=$OUT/_shards PAR=${PAR:-16} SEED=${SEED:-100} STALE=${STALE:-400}
  local SCENES CAT COUNT RATIOS
  SCENES=$("$PY" -c 'import gzip,json,os; d=json.load(gzip.open(os.path.join(os.environ["VLN_DATA"],"datasets/r2r/train/train.json.gz"),"rt")); print(" ".join(sorted({e["scene_id"].split("/")[1] for e in d["episodes"]})))')
  [ -n "$SCENES" ] || { echo "No MP3D train scenes" >&2; exit 1; }
  mkdir -p "$SHARDS"
  for CAT in r2r rxr; do
    if [ "$CAT" = r2r ]; then
      COUNT=${R2R_PER_SCENE:-100}; RATIOS=${R2R_STEP_RATIOS:-}
    else
      COUNT=${RXR_PER_SCENE:-100}; RATIOS=${RXR_STEP_RATIOS:-}
    fi
    "$PY" -c 'import sys; from quotas import ratio_caps; n=int(sys.argv[2]); assert n>0; print("step-bin quotas:",ratio_caps(sys.argv[1],n) or "random")' "$RATIOS" "$COUNT"
    export CAT COUNT RATIOS SHARDS SEED STALE PY
    printf "%s\n" $SCENES | xargs -P "$PAR" -I {} bash -c '
      set -euo pipefail
      S="$1"
      ARGS=()
      if [ -n "$RATIOS" ]; then ARGS+=(--step-ratios "$RATIOS"); fi
      "$PY" sample.py mp3d --category "$CAT" --scenes "$S" --total "$COUNT" \
        "${ARGS[@]}" --seed-real --path-overlap 0.75 --stale "$STALE" \
        --seed "$SEED" --out "$SHARDS/$CAT/$S" > "$SHARDS/$CAT-$S.log" 2>&1
      echo "[done] $CAT/$S"
    ' _ {}
    "$PY" sample.py merge --shards "$SHARDS/$CAT" --out "$OUT/$CAT"
  done
}

hm3d() {
  local SAMPLE=${SAMPLE_OUT:-$VLN_DATA/datasets/scalevln_rl_new}
  local OUT=${HM3D_OUT:-$VLN_DATA/datasets/scalevln_rl_v3}
  local COUNT=${SCALEVLN_PER_SCENE:-100} RATIOS=${SCALEVLN_STEP_RATIOS:-}
  local ATT=${HM3D_MAX_ATTEMPTS:-$((COUNT * 80))}
  local ARGS=()
  "$PY" -c 'import sys; from quotas import ratio_caps; n=int(sys.argv[2]); assert n>0; print("step-bin quotas:",ratio_caps(sys.argv[1],n) or "random")' "$RATIOS" "$COUNT"
  if [ -n "$RATIOS" ]; then ARGS+=(--bin-ratios "$RATIOS"); fi
  "$PY" sample.py hm3d --dataset hm3d --split train --workers "${WORKERS:-32}" \
    --per-scene "$COUNT" --min-m 1 --max-m 55 --max-attempts "$ATT" \
    "${ARGS[@]}" --ndtw-dedup 0.55 --ndtw-hi 0.55 \
    --seed "${HM3D_SEED:-0}" --out "$SAMPLE"
  "$PY" sample.py pack --new "$SAMPLE" --trim-ratio 0 --out "$OUT"
}

label() {
  local DS
  local sets=("$@")
  if [ "$#" -eq 0 ]; then sets=(r2r rxr scalevln); fi
  for DS in "${sets[@]}"; do
    "$PY" instruct.py label --dataset "$DS" --data-root "$VLN_DATA" --workers "${WORKERS:-32}" --conc "${CONC:-8}"
    "$PY" instruct.py assemble --dataset "$DS" --data-root "$VLN_DATA"
  done
}

case "${1:-all}" in
  all) mp3d; hm3d; label ;;
  mp3d) mp3d ;;
  hm3d) hm3d ;;
  label) shift; label "$@" ;;
  *) echo "Usage: bash run.sh [all|mp3d|hm3d|label]" >&2; exit 2 ;;
esac
