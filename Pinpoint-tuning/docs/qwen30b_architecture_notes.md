# Qwen3-30B-A3B-Instruct-2507 アーキテクチャ詳細

## 基本情報

| パラメータ | 値 |
|-----------|-----|
| model_type | `qwen3_moe` |
| architectures | `Qwen3MoeForCausalLM` |
| transformers_version | 4.51.0+ |

## レイヤー構造

| パラメータ | 値 |
|-----------|-----|
| num_hidden_layers | 48 |
| hidden_size | 2048 |
| intermediate_size | 6144 |

## Attention構造

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| num_attention_heads (Query) | 32 | |
| num_key_value_heads (KV) | 4 | GQA |
| num_key_value_groups | 8 | 32 / 4 = 8 |
| head_dim | 128 | |
| 全Attentionヘッド数 | 1536 | 48 × 32 |

### Attention モジュール名

```
model.layers.{0-47}.self_attn.q_proj      # Query projection
model.layers.{0-47}.self_attn.k_proj      # Key projection
model.layers.{0-47}.self_attn.v_proj      # Value projection
model.layers.{0-47}.self_attn.o_proj      # Output projection
model.layers.{0-47}.self_attn.q_norm      # Query RMSNorm (Qwen3で追加)
model.layers.{0-47}.self_attn.k_norm      # Key RMSNorm (Qwen3で追加)
```

## MoE構造

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| num_experts | 128 | 全エキスパート数 |
| num_experts_per_tok | 8 | トークンあたりのアクティブエキスパート数 |
| moe_intermediate_size | 768 | 各エキスパートの中間サイズ |
| decoder_sparse_step | 1 | |
| norm_topk_prob | True | Top-K確率の正規化 |
| router_aux_loss_coef | 0.001 | Load balancing loss係数 |

### MoE モジュール名

```
model.layers.{0-47}.mlp.gate                              # Router層（エキスパート選択）
model.layers.{0-47}.mlp.experts.{0-127}.gate_proj         # Expert gate projection
model.layers.{0-47}.mlp.experts.{0-127}.up_proj           # Expert up projection
model.layers.{0-47}.mlp.experts.{0-127}.down_proj         # Expert down projection
```

**注意**: Shared Expert は存在しない（重みファイルに含まれていない）

## LayerNorm構造

```
model.layers.{0-47}.input_layernorm           # 入力層正規化
model.layers.{0-47}.post_attention_layernorm  # Attention後の正規化
```

## その他の構造

```
model.embed_tokens.weight    # 入力埋め込み
model.norm.weight            # 最終層正規化
lm_head.weight               # 言語モデルヘッド
```

## メモリ要件

| 設定 | 推定メモリ |
|------|-----------|
| 総パラメータ数 | 30.5B |
| アクティブパラメータ | 3.3B |
| BF16推論 | ~62GB（全パラメータロード時） |
| BF16学習（batch=1） | ~120GB+ |

**注意**: MoEモデルは全エキスパートをメモリにロードする必要があるため、アクティブパラメータ数（3.3B）に比してメモリ使用量が大きい。

## Qwen3-14B (Dense) との主な違い

| 項目 | Qwen3-14B | Qwen3-30B-A3B |
|------|-----------|---------------|
| アーキテクチャ | Dense | MoE |
| num_hidden_layers | 40 | 48 |
| num_attention_heads | 40 | 32 |
| num_key_value_heads | 8 | 4 |
| num_key_value_groups | 5 | 8 |
| MLP構造 | 単一MLP | 128 Experts + Router |
| q_norm/k_norm | なし | あり |

## Path Patching 実装向け情報

### フック対象モジュール

1. **Attention ヘッド分析**
   - 入力: `model.layers.{i}.input_layernorm`
   - 出力: `model.layers.{i}.self_attn.o_proj`
   - 分析対象: 48層 × 32ヘッド = 1536ヘッド

2. **Expert 分析（将来拡張）**
   - Router: `model.layers.{i}.mlp.gate`
   - Expert: `model.layers.{i}.mlp.experts.{j}`
   - 分析対象: 48層 × 128エキスパート = 6144エキスパート

## Pinpoint Tuning 実装向け情報

### freeze_modules 設定

```python
# 推奨設定
freeze_router = True      # Router層は安定性のためfreeze
freeze_experts = True     # Expert層もfreeze（Attentionのみ学習）
train_kv = False          # KV projectionはfreeze
precise_level = 5         # functional_classification mode
```

### GQA対応

```python
# Qwen3-30B-A3B のGQA比率
num_key_value_groups = 32 // 4 = 8

# 8つのQueryヘッドが1つのKVヘッドを共有
# KV unfreezeする際は group 単位で処理
```

## 参考リンク

- [Qwen3-30B-A3B-Instruct-2507 - Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
- [Qwen3 Technical Report](https://arxiv.org/abs/2501.15451)

---

**作成日**: 2026-02-03
**更新履歴**: 初版作成（モデル構造分析結果に基づく）
