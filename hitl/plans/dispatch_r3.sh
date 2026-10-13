#!/usr/bin/env bash
# r3 多场景训练 run 派发器（探索件，登记 dev/INDEX.md）。
# 逐 child 跑 sequential_run.py run-child；attempt_dir 用
# multiscene/<tag>/runs/<id> 两层结构 → eval 落
# multiscene/<tag>/_shared_evaluation（runtime.py launch_root 语义）。
# 用法: bash hitl/plans/dispatch_r3.sh [qc|kn|sb|all] [并发数]
set -u
cd "$(dirname "$0")/../.."
PY="${PY:-python}"   # 需在项目运行环境（colight）下执行；PY= 可覆盖
PLAN="${PLAN:-data/output_data/hitl_probe/pilot5/plan_r3.json}"
TAGS="${1:-all}"
PAR="${2:-10}"
[ "$TAGS" = all ] && TAGS="qc kn sb"
export SUMO_HOME="${SUMO_HOME:-$($PY -c 'import sumo,os;print(os.path.dirname(sumo.__file__))' 2>/dev/null)}"
export CUDA_VISIBLE_DEVICES=""
for tag in $TAGS; do
  ids=$($PY - "$PLAN" "$tag" <<'EOF'
import json,sys
p=json.load(open(sys.argv[1]))
tag=sys.argv[2].upper()
for c in p['children']:
    rid=c['logical_run_id']
    if f'-{tag}-' in rid:
        print(rid)
EOF
)
  mkdir -p "data/output_data/hitl_probe/multiscene/$tag/runs" \
          "data/output_data/hitl_probe/multiscene/$tag/logs"
  echo "$ids" | xargs -P "$PAR" -I{} bash -c '
    id={}
    out="data/output_data/hitl_probe/multiscene/'"$tag"'/runs/$id"
    log="data/output_data/hitl_probe/multiscene/'"$tag"'/logs/$id.log"
    [ -f "$out/completed.json" ] && { echo "skip $id"; exit 0; }
    '"$PY"' sequential_run.py run-child \
      --manifest "'"$PLAN"'" \
      --logical-run-id "$id" \
      --attempt-dir "$out" >"$log" 2>&1 \
      && echo "done $id" || echo "FAIL $id"
  '
done
