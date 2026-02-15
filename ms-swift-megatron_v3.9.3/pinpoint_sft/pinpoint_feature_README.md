# ============================================================
#  PinPointTuning設定（スクリプト冒頭で定義）
# ============================================================
PINPOINT_LAYERS="5,10,15,20,25,30"
PINPOINT_EXPERTS="5:3,7_10:1,4_15:0,5"
PINPOINT_HEADS="5:0,1,2_10:3,4"

# ============================================================
#  学習実行
# ============================================================
torchrun \
    --nnodes ${SLURM_JOB_NUM_NODES} \
    --nproc_per_node ${NPROC_PER_NODE} \
    --node_rank ${NODE_RANK} \
    --rdzv_backend c10d \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    -m swift.cli._megatron.sft \
      --use_hf true \
      --model ${MODEL_PATH} \
      --model_type qwen3_moe \
      --dataset ${DATASET_JSONL} \
      --train_type full \
      --load_safetensors true \
      --no_initialization false \
      --pinpoint_tuning true \
      --pinpoint_trainable_layers "${PINPOINT_LAYERS}" \
      --pinpoint_trainable_experts "${PINPOINT_EXPERTS}" \
      --pinpoint_trainable_heads "${PINPOINT_HEADS}" \
      --pinpoint_freeze_mlp false \
      --pinpoint_freeze_attention false \
      --pinpoint_freeze_router true \
      --pinpoint_freeze_shared_expert true \
      --pinpoint_freeze_embed_lm_head true \
      --tensor_model_parallel_size 1 \
      --pipeline_model_parallel_size 4 \
      --expert_model_parallel_size 8 \
      # ... 残りのオプション
```

---

## 5. 機能サマリー

| 制御レベル | パラメータ | 形式 | 例 |
|-----------|-----------|------|-----|
| Layer | `--pinpoint_trainable_layers` | カンマ区切り | `"5,10,15,20"` |
| Expert (MoE) | `--pinpoint_trainable_experts` | `layer:expert,expert_layer:expert` | `"5:3,7_10:1,4"` |
| Attention Head | `--pinpoint_trainable_heads` | `layer:head,head_layer:head` | `"5:0,1,2_10:3,4"` |
| MLP Freeze | `--pinpoint_freeze_mlp` | boolean | `true` / `false` |
| Attention Freeze | `--pinpoint_freeze_attention` | boolean | `true` / `false` |
| Router Freeze | `--pinpoint_freeze_router` | boolean | `true` / `false` |
| Shared Expert Freeze | `--pinpoint_freeze_shared_expert` | boolean | `true` / `false` |
| Embed/LM_Head Freeze | `--pinpoint_freeze_embed_lm_head` | boolean | `true` / `false` |

---

## 6. パラメータ形式の詳細

シェルスクリプトとの互換性を考慮し、JSON形式ではなくシンプルな文字列形式を採用しています。

### Layers形式
```
"layer1,layer2,layer3"
例: "5,10,15"
```

### Experts形式
```
"layer1:expert1,expert2_layer2:expert1,expert2"
例: "5:3,7_10:1,4"
→ 解釈: {5: [3, 7], 10: [1, 4]}
```

### Heads形式
```
"layer1:head1,head2,head3_layer2:head1,head2"
例: "5:0,1,2_10:3,4"
→ 解釈: {5: [0, 1, 2], 10: [3, 4]}