# GLM-5 Path Patching 適用計画

作成日: 2026-02-19
対象: [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5)
参照: `Pinpoint-tuning/Qwen30B-A3B/Phase2_path_patching`

---

## 1. アーキテクチャ比較

### 基本スペック

| 項目 | Qwen3-30B-A3B | GLM-5 |
|---|---|---|
| model_type | `qwen3_moe` | `glm_moe_dsa` |
| 総パラメータ数 | ~30B | 744B |
| アクティブパラメータ | ~3B | 40B |
| 層数 | 48 | 78 |
| アテンションヘッド数 | 32 | 64 |
| KVヘッド数 (GQA) | 4 | 64（MLA圧縮） |
| hidden_size | 2,048 | 6,144 |
| head_dim (QK) | 128 | 256（nope 192 + rope 64） |
| **value_head_dim** | **128** | **256** |
| o_proj 入力次元 | 32 × 128 = 4,096 | 64 × 256 = 16,384 |
| o_proj 出力次元 | 2,048 | 6,144 |
| MoE 総エキスパート数 | 128 | 256 |
| MoE アクティブ数 | 8 | 8 |
| 共有エキスパート | なし | 1 |
| MoE 適用レイヤー | 全48層 | 全78層（先頭3層はDense） |
| 最大コンテキスト長 | 32,768 | 202,752 |
| モデルファイルサイズ | ~60GB (bf16) | ~1,510GB / 282シャード (bf16) |
| 1GPU最低必要VRAM | 80GB × 1 枚 | 80GB × 20 枚以上 |

### アテンション方式の違い（最重要）

Qwen3-30B-A3B は**標準 MHA（GQA）**、GLM-5 は **MLA（Multi-Head Latent Attention）**を採用。

```
【Qwen3-30B-A3B: 標準 GQA】
input_layernorm → Q/K/V proj → Attention → concat → o_proj
                                              ↑
                              32ヘッド × 128dim = 4,096dim

【GLM-5: MLA（DeepSeek-V2 系）】
input_layernorm → Q: q_a_proj(down/rank=2048) → q_b_proj(up)
               → KV: kv_a_proj(down/rank=512) → kv_b_proj(up)
               → Attention → concat → o_proj
                               ↑
                   64ヘッド × 256dim (value_head_dim) = 16,384dim
```

**Path Patching への影響:**
- hook点は `model.layers.{i}.self_attn.o_proj` で共通と推定
- ただし **パッチ対象のスライス幅が 128 → 256 に変わる**（head_dim の変更のみ）
- MLA の LoRA 分解（Q/KV の圧縮）はモデル内部の計算であり、o_proj フック後の処理には影響しない
- `hook_functions_moe.py` の `write_hook` は head_dim をパラメータで受け取る実装のため、**設定ファイルの変更だけで対応可能**（コード変更不要の可能性が高い）

### MoE 構造の違い

| 項目 | Qwen3-30B-A3B | GLM-5 |
|---|---|---|
| ルーター方式 | softmax + top-8 | sigmoid + top-8 (`noaux_tc`) |
| Dense レイヤー | なし（全層 MoE） | 先頭 3 層が Dense |
| エキスパートサイズ | - | intermediate_size = 2,048 |
| 共有エキスパート | なし | 1 つ（全サンプルで共通） |

**Path Patching への影響:** MoE ルーティング自体は attention とは独立しているため、現行の attention head 単位のパッチング手法に直接の影響はなし。先頭 3 層が Dense である点は configs に記載が必要。

---

## 2. ディレクトリ構成

### 現行（Qwen3-30B-A3B）

```
Phase2_path_patching/
├── dataset.py                        # データセットクラス
├── utils.py                          # compute_metric / 可視化ユーティリティ
├── hook_functions_moe.py             # forward hook 実装（read/write）
├── __init__.py
├── 1node8gpu/
│   ├── path_patching_medical.py      # メイン実行スクリプト
│   ├── merge_parallel_results.py     # 並列結果のマージ
│   ├── monitor_parallel.sh
│   └── configs/
│       └── qwen3_moe.json            # モデル設定（hook点・head_dim等）
└── 8node64gpu/
    ├── submit_all.sh                 # SLURMジョブ一括投入
    ├── node_job.sh                   # ノード単位のジョブスクリプト
    ├── monitor_all_nodes.sh
    ├── merge_64gpu_results.py        # 64プロセス結果のマージ
    └── results_overlap_{medical,reasoning}/
```

### GLM-5 向けの構成案

```
Pinpoint-tuning/GLM-5/
├── plan_phase2_path_patching.md      # 本ファイル
└── Phase2_path_patching/
    ├── dataset.py                    # ★ 流用
    ├── utils.py                      # ★ 流用
    ├── hook_functions_moe.py         # ★ 流用（head_dim はパラメータ渡しのため）
    ├── __init__.py                   # ★ 流用
    ├── 1node_verify/                 # 動作検証用（少数サンプル）
    │   ├── path_patching_glm5.py     # △ 改造（MODEL_TYPE_CONFIGS追加）
    │   ├── verify_hook_point.py      # ● 新規作成（MLA hook点の確認スクリプト）
    │   └── configs/
    │       └── glm_moe_dsa.json      # ● 新規作成
    └── 8node64gpu/
        ├── submit_all.sh             # △ 改造（データパス・ノード設定変更）
        ├── node_job.sh               # △ 改造（モデルパス・GPU割り当て変更）
        ├── monitor_all_nodes.sh      # ★ 流用（ノード名変更のみ）
        └── merge_64gpu_results.py    # ★ 流用
```

**凡例:** ★ 流用 ／ △ 改造 ／ ● 新規作成

---

## 3. ファイル別の改造内容

### ★ 流用（変更不要）

| ファイル | 理由 |
|---|---|
| `dataset.py` | 入力 JSONL 形式・フィールド名が同一 |
| `utils.py` | `compute_metric`（logit比率）・可視化関数はモデル非依存 |
| `hook_functions_moe.py` | head_dim をパラメータ引数で受け取る設計のため汎用 |
| `merge_64gpu_results.py` | pt ファイルの weighted average マージはモデル非依存 |
| `monitor_all_nodes.sh` | ノード名のみ変更で流用可能 |

### △ 改造（既存ファイルを改造）

#### `path_patching_glm5.py`（`path_patching_medical.py` をベースに改造）

```python
# 変更点 1: MODEL_TYPE_CONFIGS に GLM-5 を追加
MODEL_TYPE_CONFIGS = {
    "qwen3_moe": { ... },       # 既存
    "glm_moe_dsa": {            # 追加
        "module_input_name":  "model.layers.{i}.input_layernorm",
        "module_output_name": "model.layers.{i}.self_attn.o_proj",  # 要検証
        "is_moe": True,
    },
}

# 変更点 2: head_dim の読み取り
# Qwen3: model.config.head_dim
# GLM-5: model.config.value_head_dim = 256 (QK head_dim=256 とは別)
if hasattr(model.config, "value_head_dim"):
    head_dim = model.config.value_head_dim   # GLM-5: 256
else:
    head_dim = model.config.head_dim          # Qwen3: 128

# 変更点 3: num_attention_heads
# GLM-5: model.config.num_attention_heads = 64
```

#### `node_job.sh`（GPU割り当て変更）

```bash
# Qwen3-30B-A3B: 60GB → 1GPU に 1 モデルインスタンス
CUDA_VISIBLE_DEVICES=${gpu_idx} python path_patching_glm5.py ...

# GLM-5: 1.51TB → 1 モデルインスタンスに複数 GPU が必要
# → GPU_PER_INSTANCE=8 の場合: 8GPU × 80GB = 640GB（ギリギリ）
# → GPU_PER_INSTANCE=16 の場合: 16GPU × 80GB = 1,280GB（必要バッファあり）
# 具体的な数は verify_hook_point.py でメモリ計測後に決定
```

#### `submit_all.sh`

- `MODEL_PATH` を GLM-5 のパスに変更
- `NODES` リストをGLM-5 実行ノードに変更
- データパスを `dictionary_260212/` 配下に変更
- GPU_PER_INSTANCE に応じて並列数を調整

### ● 新規作成

#### `configs/glm_moe_dsa.json`

```json
{
  "model_type": "glm_moe_dsa",
  "model_name": "GLM-5",
  "module_input_name": "model.layers.{i}.input_layernorm",
  "module_output_name": "model.layers.{i}.self_attn.o_proj",
  "is_moe": true,
  "moe_config": {
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "has_shared_expert": true,
    "dense_layers": [0, 1, 2]
  },
  "attention_config": {
    "num_hidden_layers": 78,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "head_dim": 256,
    "value_head_dim": 256,
    "hidden_size": 6144,
    "attention_type": "mla"
  },
  "notes": "GLM-5 MLA 特有の value_head_dim=256 を使用。先頭 3 層は Dense Attention。"
}
```

#### `verify_hook_point.py`（動作確認スクリプト）

MLA の hook 点が `self_attn.o_proj` で正しいか、hook した tensor のシェイプが `(batch, seq_len, num_heads * value_head_dim)` となるかを数サンプルで確認する。また GPU メモリ使用量を計測し、最適な GPU_PER_INSTANCE を決定する。

---

## 4. GPU 環境と工数見積もり

### GPU 環境（実環境）

| 項目 | 値 |
|---|---|
| 利用可能ノード数 | 最大 8 ノード（ただし後述の制約あり） |
| GPU/ノード | 8 GPU（80GB × 8 = **640GB/ノード**） |
| 合計 GPU | 最大 64 GPU / 合計 5,120GB VRAM |

### ⚠️ 重大制約：モデルがノードをまたげない問題

GLM-5 の bf16 重量は約 **1,488GB**。1 ノード 640GB では **モデル全体を収められない**。

現行の Qwen3 では `CUDA_VISIBLE_DEVICES=X python ...` で 1 GPU に 1 モデルを乗せているが、GLM-5 では同じ方法は使えない。さらに **`device_map='auto'`（transformers）は単一ノード内の GPU 分散しかサポートしない**。マルチノードにまたがったモデル分散は不可。

#### 対応策の比較

| 方式 | 1ノード内VRAM | 並列インスタンス数 | 精度影響 | 実装コスト |
|---|---|---|---|---|
| **bf16 フル精度** | 1,488GB → 不可 | — | なし | — |
| **INT8 量子化** (`bitsandbytes`) | ~744GB → 不可 | — | 微小 | 低（`load_in_8bit=True`） |
| **INT4 量子化** (`bitsandbytes`) | ~372GB → **収まる** | 1 | 小 | 低（`load_in_4bit=True`） |
| **GPTQ/AWQ (INT4)** | ~372GB → **収まる** | 1 | 小 | 中（事前量子化が必要） |
| **FP8 量子化** | ~744GB → 不可 | — | 微小 | 中 |
| **マルチノードTP** (vLLM/Megatron) | 各ノード ~744GB | 複数 | なし | **非常に高い** |

#### 現実的な選択肢

**推奨: INT4 量子化（bitsandbytes `load_in_4bit=True`）**

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_4bit=True,           # ~372GB → 1ノード640GBに収まる
    bnb_4bit_compute_dtype=torch.bfloat16,
    device_map="auto",           # 8GPU に自動分散
    trust_remote_code=True,
)
```

- 1 ノード（8 GPU = 640GB）に 1 モデルインスタンスが収まる
- 同一ノード内での `device_map='auto'` を使うため、既存コードの変更は最小限
- Path Patching はモデルの「相対的な応答変化」を見るため、量子化による絶対値のずれは許容範囲と考えられる

**注意点:**
- bitsandbytes の量子化モデルは `generate()` は動くが、**forward hook が量子化レイヤーを通過する際の挙動を要検証**（`verify_hook_point.py` で確認必須）
- 特に o_proj が `Linear4bit` 型になるため、hook の `args[0]` のテンソル形状・dtype が変わる可能性あり

#### マルチノードTP（参考）

vLLM または Megatron-LM による tensor parallel を使えばマルチノードでも bf16 のまま動作するが、**forward hook の埋め込みが極めて困難**（TP 分割後のモジュールにアクセスする必要があり、既存 hook_functions_moe.py は使えない）。実装コストが大きく、現時点では推奨しない。

---

### 処理量の比較

| 項目 | Qwen3-30B-A3B | GLM-5 |
|---|---|---|
| パッチング対象 | 48層 × 32ヘッド = **1,536** | 78層 × 64ヘッド = **4,992** |
| 1サンプルのforward pass数 | 1,539（patch × 1,536 + overhead × 3） | 4,995（patch × 4,992 + overhead × 3） |
| データセット件数 | 5,003 | 5,003（同じ） |
| 1ノードあたりの並列インスタンス数 | 8（1GPU × 8） | 1（8GPU × 1、INT4） |

### 実行時間の見積もり

#### 前提
- Qwen3-30B-A3B 実績: 1 GPU × 1 インスタンス × 78 サンプル ≈ 6 時間
- 1 forward pass あたり: 6h / (78 × 1,539) ≈ **0.18 秒**

#### GLM-5 の推定（INT4 量子化、8GPU/インスタンス）

| 係数 | 内容 | 倍率 |
|---|---|---|
| head 数増加 | 4,992 / 1,536 | × 3.25 |
| アクティブパラメータ規模 | 40B / 3B | × 13 |
| 8GPU分散効率 | 8GPU ÷ 通信オーバーヘッド 2× | ÷ 4 |
| INT4量子化の高速化 | 演算量削減 | ÷ 1.5 |

推定 1 forward pass = 0.18s × 13 / 4 / 1.5 ≈ **0.39 秒**

| シナリオ | GPU/ノード | 並列インスタンス数 | ノード数 | 合計サンプル担当 | 推定時間 |
|---|---|---|---|---|---|
| INT4、1ノード | 8 | 1 | 1 | 5,003 | 5,003 × 4,995 × 0.39s / 3600 ≈ **約 2,706 時間（約 113 日）** |
| INT4、8ノード並列 | 8 | 1/ノード | 8 | 5,003 / 8 = 626 | 626 × 4,995 × 0.39s / 3600 ≈ **約 338 時間（約 14 日）** |
| INT4、8ノード + サブサンプル（500件） | 8 | 1/ノード | 8 | 63 | 63 × 4,995 × 0.39s / 3600 ≈ **約 34 時間** |

> **現実的な推奨:** まず **500 サンプルで 8 ノード並列（約 34 時間）** を実行して結果品質を確認し、フルスケール（14 日）実行の要否を判断する。

### 開発工数の見積もり

| タスク | 内容 | 工数 |
|---|---|---|
| INT4 量子化ロードの動作確認 | bitsandbytes での GLM-5 ロード確認 | 0.5 日 |
| `verify_hook_point.py` 作成・実行 | MLA hook点 + Linear4bit での hook 動作確認 | 1〜2 日 |
| `configs/glm_moe_dsa.json` 作成 | hook点・head_dim 設定 | 0.5 日 |
| `path_patching_glm5.py` 改造 | INT4 ロード対応、head_dim 読み取り変更 | 1 日 |
| `node_job.sh` / `submit_all.sh` 改造 | 1インスタンス＝8GPU 対応、モデルパス変更 | 0.5 日 |
| 小規模検証（200 サンプル） | 動作確認 + 実測スループット計測 | 1〜2 日 |
| デバッグ・修正 | MLA・INT4 特有の問題への対応 | 1〜2 日 |
| **合計** | | **5.5〜8.5 日** |

---

## 5. 実装上のリスクと対応方針

| リスク | 内容 | 対応方針 |
|---|---|---|
| **1ノード640GBに収まらない** | bf16（1,488GB）、INT8（744GB）はどちらも 640GB を超える | **INT4 量子化必須**（~372GB → 640GB に収まる）。マルチノードTP は hook 実装コストが高く現実的でない |
| **INT4 hook の挙動** | `load_in_4bit` 時に o_proj が `Linear4bit` 型になり、hook の `args[0]` dtype が変わる可能性 | `verify_hook_point.py` で shapeと dtype を確認。必要なら hook 内で `.to(bfloat16)` キャストを追加 |
| **MLA hook点の不一致** | GLM-5 の MLA 実装で `self_attn.o_proj` が存在しない、またはシェイプが想定外の可能性 | `verify_hook_point.py` で `model.named_modules()` を列挙し hook 点を特定 |
| **value_head_dim の確認** | 実装によっては concat 前の処理があり、スライス幅 256 が正しくない可能性 | hook した tensor のシェイプを確認（期待値: `[batch, seq, 64×256=16384]`） |
| **モデルロードの失敗** | `transformers` バージョンが `glm_moe_dsa` に未対応の可能性 | `trust_remote_code=True` + GLM-5 リポジトリの `modeling_glm_moe_dsa.py` を確認 |
| **処理時間超過** | 8ノード並列でも 14 日（SLURM 72h 制限を超える） | チェックポイント保存機能を `path_patching_glm5.py` に追加。500 サンプルのサブセットから開始 |
| **量子化による精度変化** | INT4 量子化でモデル出力の絶対値がずれ、impact の大小関係が変わる可能性 | 少数サンプルで bf16 ロード（CPU + GPU オフロード）と INT4 の結果を比較して許容誤差を確認 |

---

## 6. 推奨実施ステップ

1. **INT4 量子化ロード確認**: `bitsandbytes` で GLM-5 を 8GPU にロードし、VRAM 使用量・推論動作を確認
2. **hook点検証**: `verify_hook_point.py` で INT4 ロード時の o_proj シェイプ・dtype を確認
3. **設定ファイル作成**: `configs/glm_moe_dsa.json` を実測値で更新
4. **小規模検証**: 200 サンプルで `path_patching_glm5.py` を実行し、1 forward pass の実測時間を計測
5. **チェックポイント機能追加**: 72h 制限を超えるため、途中再開機能を実装
6. **スケールアップ**: 500 サンプル（推定 34h）から開始し、8 ノード投入
7. **マージ・可視化**: `merge_64gpu_results.py` → ヒートマップ生成・Qwen3 との比較

---

## 7. 出力データ（予定）

Qwen3-30B-A3B と同様の形式で出力。

```
GLM-5/Phase2_path_patching/8node64gpu/
├── results_overlap_medical/
│   ├── results_combined.pt           # (78, 64) 平均 impact
│   ├── results_per_sample_combined.pt # (5003, 78, 64) サンプル別 impact
│   └── heatmap_overlap_medical.png
└── results_overlap_reasoning/
    ├── results_combined.pt
    ├── results_per_sample_combined.pt
    └── heatmap_overlap_reasoning.png
```

Qwen3 との比較により、**医療問題の解答に関与する attention head がモデルアーキテクチャをまたいで共通するか**を検証できる。
