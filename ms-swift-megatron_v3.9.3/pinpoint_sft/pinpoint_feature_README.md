## 使用例

修正後、以下のようにスクリプトで使用できます。

```bash
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
      --pinpoint_tuning true \
      --pinpoint_trainable_layers "5,10,15,20,25,30" \
      --pinpoint_trainable_experts '{"5": [3, 7], "10": [1, 4], "15": [0, 5]}' \
      --pinpoint_freeze_mlp true \
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

| 制御レベル | パラメータ | 例 |
|-----------|-----------|-----|
| Layer | `--pinpoint_trainable_layers` | `"5,10,15,20"` |
| Expert (MoE) | `--pinpoint_trainable_experts` | `'{"5": [3, 7], "10": [1, 4]}'` |
| Attention Head | `--pinpoint_trainable_heads` | `'{"5": [0, 1, 2], "10": [3, 4]}'` |
| MLP Freeze | `--pinpoint_freeze_mlp` | `true` / `false` |
| Attention Freeze | `--pinpoint_freeze_attention` | `true` / `false` |
| Router Freeze | `--pinpoint_freeze_router` | `true` / `false` |
| Shared Expert Freeze | `--pinpoint_freeze_shared_expert` | `true` / `false` |
| Embed/LM_Head Freeze | `--pinpoint_freeze_embed_lm_head` | `true` / `false` |

この実装により、PinPointTuning論文の手法をms-swift 3.9.3のMegatronモードで利用できるようになります。