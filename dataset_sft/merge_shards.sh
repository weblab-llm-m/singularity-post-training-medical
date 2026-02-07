#!/bin/bash
# ===========================================================
# 8シャードのSFT出力をマージ
# 使い方: bash merge_shards.sh [output_path]
# ===========================================================

set -euo pipefail

OUTPUT_PATH="${1:-output/sft_igakuqa.jsonl}"
STEM="${OUTPUT_PATH%.*}"
EXT="${OUTPUT_PATH##*.}"
NUM_SHARDS=8

echo "=== シャードマージ ==="

TOTAL=0
> "$OUTPUT_PATH"  # 初期化

for i in $(seq 0 $((NUM_SHARDS - 1))); do
    SHARD="${STEM}_shard${i}.${EXT}"
    if [ -f "$SHARD" ]; then
        COUNT=$(wc -l < "$SHARD")
        cat "$SHARD" >> "$OUTPUT_PATH"
        echo "  shard${i}: ${COUNT} 件"
        TOTAL=$((TOTAL + COUNT))
    else
        echo "  shard${i}: ファイルなし (${SHARD})"
    fi
done

# 重複排除（problem_idベース）
if [ $TOTAL -gt 0 ]; then
    python -c "
import json
seen, unique = set(), []
with open('${OUTPUT_PATH}', 'r') as f:
    for line in f:
        rec = json.loads(line.strip())
        pid = rec.get('problem_id')
        if pid not in seen:
            seen.add(pid)
            unique.append(line.strip())
with open('${OUTPUT_PATH}', 'w') as f:
    f.write('\n'.join(unique) + '\n')
print(f'  重複排除: {$TOTAL} → {len(unique)} 件')
"
fi

FINAL=$(wc -l < "$OUTPUT_PATH")
echo ""
echo "完了: ${FINAL} 件 → ${OUTPUT_PATH}"