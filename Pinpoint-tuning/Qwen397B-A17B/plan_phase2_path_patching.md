# Qwen3.5-397B-A17B Path Patching 計画書

作成日: 2026-02-25
対象モデル: `Qwen/Qwen3.5-397B-A17B`
参照実装: `Pinpoint-tuning/Qwen30B-A3B/Phase2_path_patching/`
GPU環境: **1ノード 8GPU (H100 80GB × 8 = 640GB)**（使用可能上限）

---

## 1. アーキテクチャ比較

| パラメータ | Qwen3-30B-A3B | Qwen3.5-397B-A17B | 影響 |
|---|---|---|---|
| model_type | `qwen3_moe` | `qwen3_5_moe` | 設定ファイル新規作成が必要 |
| num_hidden_layers | 48 | 60 | フォワードパス回数増加 |
| num_attention_heads | 32 | 32 | 変わらず |
| num_key_value_heads | 4 | **2** | GQAさらに強化（patch対象はQ heads=32で変わらず） |
| hidden_size | 2048 | **4096** | 2倍 |
| head_dim | 64 | **256** | 4倍（hookのスライス幅が変わる） |
| num_experts | 128 | **512** | 4倍 |
| num_experts_per_tok | 8 | **10** | MoEルーティング変化 |
| moe_intermediate_size | 2048 | **1024** | 専門家1つあたりのサイズ縮小 |
| attn_output_gate | なし | **あり** | attention出力にゲートが追加 |
| 層タイプ | 全層 full_attention | **3 linear + 1 full の繰り返し** | ★最大の差異（後述） |
| full_attention_interval | N/A | **4** | 60層中 full attention は15層のみ |
| 総パラメータ数 | ~30B | **~397B** | メモリ要件が激変 |
| 活性化パラメータ | ~3B | **~17B** | 実計算量は~5.7倍 |
| 1GPU bf16 メモリ | ~60GB（1GPU収容可能） | **~794GB**（8GPU合計でも収まらない） | ★量子化必須 |
| max_position_embeddings | 32,768 | 262,144 | トークン化時は影響なし |

---

## 2. 重要な技術的差異と課題

### 2.1 ★ Linear Attention の存在（最大の差異）

397B-A17B は `full_attention_interval: 4` のため、4層ごとに1層だけ full attention が入る混合構造。

```
layer 0,1,2 → linear_attention
layer 3      → full_attention   ← hookの対象
layer 4,5,6 → linear_attention
layer 7      → full_attention   ← hookの対象
...（15回繰り返し）
layer 59     → full_attention
```

**full attention 層（hookの対象候補）: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59（計15層）**

#### Linear attention のアーキテクチャ

linear attention 層は full attention とは異なる Q/K/V ヘッド構成を持つ。

| パラメータ | full attention | linear attention |
|---|---|---|
| query heads | 32 | 32 |
| key heads | 2 | **16** |
| value heads | 2 | **64** |
| key head_dim | 256 | **128** |
| value head_dim | 256 | **128** |
| o_proj | 存在（32×256→4096） | モジュール名・構造要確認 |

linear attention 層に対して `model.layers.{i}.self_attn.o_proj` が存在するかは**実機確認が必要**。存在しても、ヘッド意味論が full attention とは異なるため、**初期フェーズでは full attention 15層のみを対象**とする方針を推奨する。

### 2.2 ★ メモリ制約（1ノード 8GPU = 640GB VRAM）

| 精度 | 397B モデルメモリ | 640GB に収まるか |
|---|---|---|
| bf16 | ~794 GB | **NG（154GB 超過）** |
| int8 | ~397 GB | **OK（243GB の余裕）** |
| int4 | ~199 GB | OK（余裕あり、精度低下リスク） |

**結論: INT8 量子化（bitsandbytes `load_in_8bit=True`）が唯一の現実的選択肢。**

int4は精度が低下しやすく、hook対象の `o_proj`（`Linear4bit`型）での挙動が不安定になる可能性があるため、INT8 を優先する。

#### INT8 量子化でのフック動作

既存の `add_pre_module_hook_single_head` は `module.register_forward_pre_hook` を使用しており、`Linear8bitLt` 型のモジュールにも適用可能。ただし、**hook が受け取るテンソルが bfloat16（dequantize後）か int8 のままかは事前確認が必要**。`verify_hook_points.py`（後述）で検証する。

### 2.3 データ並列化不可

Qwen30B-A3B では「1GPU = 1モデルインスタンス」として 8GPU × 8プロセスの**データ並列**が可能だった。

397B-A17B では **8GPU 全体で 1モデル**（`device_map='auto'` によるパイプライン並列）を構成するため、**データ並列化は不可能**。1プロセスが順番にサンプルを処理する。

#### 含意

- Qwen30B-A3B（8GPU並列）: 78 サンプル/GPU × 8 = 624 サンプル/job
- Qwen3.5-397B-A17B（1プロセス）: **N サンプル/job（1プロセスが全サンプルを処理）**

### 2.4 head_dim の増大

hook のスライス位置は `head_idx * head_dim : (head_idx+1) * head_dim` で決まる。
397B-A17B の full attention 層では `head_dim=256`（30B-A3Bの4倍）。
既存の `hook_functions_moe.py` は head_dim を引数で受け取るため、**コード修正不要**（設定値の変更のみ）。

### 2.5 attn_output_gate

`attn_output_gate: true` により、attention 出力に sigmoid ゲートが乗算される場合がある。
このゲートが `o_proj` の前か後に適用されるかで hook ポイントへの影響が変わる。
`verify_hook_points.py` で `o_proj` 前後の活性化を確認する。

---

## 3. ディレクトリ構成

```
Pinpoint-tuning/Qwen397B-A17B/
└── Phase2_path_patching/
    ├── ★ dataset.py              ← 【流用】変更なし
    ├── ★ utils.py                ← 【流用】変更なし
    ├── ★ hook_functions_moe.py   ← 【流用】変更なし
    ├── ★ __init__.py             ← 【流用】変更なし
    │
    ├── 1node8gpu/
    │   ├── △ path_patching_397b.py    ← 【改造】メイン処理スクリプト
    │   ├── △ run_path_patching.sh     ← 【改造】実行シェルスクリプト
    │   └── ● monitor.sh               ← 【新規】1プロセス監視スクリプト
    │
    ├── configs/
    │   └── ● qwen3_5_moe.json         ← 【新規】モデル設定ファイル
    │
    └── ● verify_hook_points.py        ← 【新規】フック動作確認スクリプト
```

### 3.1 流用するファイル（変更なし）★

Qwen30B-A3B の `Phase2_path_patching/` から**symlink または コピー**で利用する。

| ファイル | 理由 |
|---|---|
| `dataset.py` | jsonl の読み込み・トークン化ロジックは同一 |
| `utils.py` | `compute_metric`, `create_batch`, `show_path_patching_results` はモデル非依存 |
| `hook_functions_moe.py` | `register_forward_pre_hook` ベースで head_dim は引数受け取りのため汎用 |
| `__init__.py` | パッケージ初期化のみ |

### 3.2 改造が必要なファイル △

#### `1node8gpu/path_patching_397b.py`

Qwen30B-A3B の `1node8gpu/path_patching_medical.py` をベースに以下を変更。

**変更点 ①: MODEL_TYPE_CONFIGS への `qwen3_5_moe` 追加**

```python
MODEL_TYPE_CONFIGS = {
    "qwen3_moe": { ... },   # 既存
    "qwen3_5_moe": {        # ← 新規追加
        "module_input_name": "model.layers.{i}.input_layernorm",
        "module_output_name": "model.layers.{i}.self_attn.o_proj",
        "router_module_name": "model.layers.{i}.mlp.gate",
        "expert_module_pattern": "model.layers.{i}.mlp.experts.{j}",
        "is_moe": True,
        "full_attention_interval": 4,       # 4層に1層だけ full attention
        "full_attention_offset": 3,         # layer 3, 7, 11, ... が full
    }
}
```

**変更点 ②: full attention 層のみをパッチ対象にするフィルタリング**

```python
# full attention layers: i % full_attention_interval == full_attention_offset
full_attn_layers = [
    i for i in range(num_layers)
    if i % full_attention_interval == full_attention_offset
]
# → [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59] (15層)

# results テンソルも full_attention 層のみを記録
results = torch.zeros(size=(len(full_attn_layers), num_attention_heads), device=model.device)
```

**変更点 ③: INT8 量子化ロード**

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True,
    attn_implementation="eager",   # attention weight 出力に必要
)
```

**変更点 ④: チェックポイント保存**（長時間実行のため必須）

```python
# 定期的に途中結果を保存（10サンプルごと）
if cnt % 10 == 0:
    checkpoint = {
        'per_sample_results': per_sample_results,
        'processed_count': cnt,
    }
    torch.save(checkpoint, os.path.join(output_dir, f"checkpoint_{cnt:04d}.pt"))
```

#### `1node8gpu/run_path_patching.sh`

Qwen30B-A3B の `run_parallel_path_patching.sh` から大幅に変更。
**データ並列なし → 全8GPUを1プロセスに割り当て。**

```bash
#!/bin/bash
#SBATCH --job-name=pp_397b
#SBATCH --partition=P08317
#SBATCH --gres=gpu:8          # 全8GPU使用
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=72:00:00

BASE_DIR="/home/yuuki.nakamura/singularity-post-training-medical/Pinpoint-tuning/Qwen397B-A17B"
DATASET=${1:-"overlap_medical"}
DATA_PATH="...dictionary_260212/path_patching_data_${DATASET}.jsonl"
OUTPUT_DIR="${BASE_DIR}/Phase2_path_patching/1node8gpu/results_${DATASET}"

source ${BASE_DIR}/venv/bin/activate  # または Qwen30B と共有
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# 全8GPUを1モデルに使用（device_map='auto'が自動分散）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python \
    ${BASE_DIR}/Phase2_path_patching/1node8gpu/path_patching_397b.py \
    --model_path /home/yuuki.nakamura/downloads/models/Qwen_Qwen3.5-397B-A17B \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --sample_num ${2:--1} \
    > ${OUTPUT_DIR}/run.log 2>&1
```

### 3.3 新規作成するファイル ●

#### `configs/qwen3_5_moe.json`

```json
{
  "module_input_name": "model.layers.{i}.input_layernorm",
  "module_output_name": "model.layers.{i}.self_attn.o_proj",
  "router_module_name": "model.layers.{i}.mlp.gate",
  "expert_module_pattern": "model.layers.{i}.mlp.experts.{j}",
  "is_moe": true,
  "full_attention_interval": 4,
  "full_attention_offset": 3,
  "num_experts": 512,
  "num_experts_per_tok": 10,
  "attention_note": "full attention layers: i % 4 == 3 (layers 3,7,...,59)"
}
```

#### `verify_hook_points.py`

INT8 量子化環境で hook が正しく動作するかを確認するスクリプト。

確認項目:
1. `model.layers.3.self_attn.o_proj` が `Linear8bitLt` 型であること
2. `register_forward_pre_hook` が動作すること（hook が呼ばれること）
3. hook に渡される `args[0]` のテンソルが bf16/f32（dequantize 済み）であること
4. write hook でスライス書き換えが有効に機能すること
5. `model.layers.0.self_attn.o_proj` が存在するか（linear attention の o_proj 確認）
6. `attn_output_gate` の適用位置（o_proj の前か後か）の確認

#### `1node8gpu/monitor.sh`

単一プロセスの進捗を tail と grep で確認するシェルスクリプト。

---

## 4. 実行時間の見積もり

### 4.1 Qwen30B-A3B の実績値（推定）

| 項目 | 値 |
|---|---|
| GPU構成 | 1 H100 80GB / プロセス |
| フォワードパス数/サンプル | 2 + 48×32 = **1,538** |
| 処理速度（推定） | 1〜2 秒/フォワードパス |
| 処理時間/サンプル | **25〜50 分** |
| 72時間で処理可能数 | **85〜170 サンプル** |

### 4.2 Qwen3.5-397B-A17B の見積もり（full attention 15層のみ対象）

| 項目 | 値 | 備考 |
|---|---|---|
| GPU構成 | 8 H100 全て使用（パイプライン並列） | device_map='auto' |
| パッチ対象層 | full attention 15層（層 3,7,11,...,59） | linear attention 45層は除外 |
| フォワードパス数/サンプル | 2 + 15×32 = **482** | 30B比: **31%** |
| フォワードパス時間（推定） | 5〜20 秒 | パイプライン並列オーバーヘッド含む |
| 処理時間/サンプル（楽観） | 482 × 5 秒 = **40 分** | |
| 処理時間/サンプル（悲観） | 482 × 20 秒 = **161 分（約2.7時間）** | |
| 72時間で処理可能数（楽観） | **〜108 サンプル** | |
| 72時間で処理可能数（悲観） | **〜26 サンプル** | |

> **注意**: 上記はベンチマーク前の推定値。実際のフォワードパス時間は `verify_hook_points.py` または 10 サンプルのパイロット実行で確認すること。

### 4.3 全サンプル（5,003件）処理に必要な期間

| フォワードパス時間 | サンプル/run (72h) | 必要 run 数 | 総期間 |
|---|---|---|---|
| 5 秒/forward | 108 | 47 | **141 日** |
| 10 秒/forward | 54 | 93 | **280 日** |
| 20 秒/forward | 26 | 193 | **580 日** |

**→ 全 5,003 サンプルの処理は現実的ではない。**

### 4.4 推奨サンプル数

| フォワードパス時間 | 72h で処理可能数 | 推奨目標 |
|---|---|---|
| 5 秒 | 108 | 100〜200 サンプル（2〜4 run） |
| 10 秒 | 54 | 50〜100 サンプル（1〜2 run） |
| 20 秒 | 26 | 20〜50 サンプル（2 run） |

パイロット実行の結果に基づき、最終的なサンプル数目標を決定する。

---

## 5. 開発工数見積もり

| タスク | 工数 | 詳細 |
|---|---|---|
| `verify_hook_points.py` 作成・実行 | 1〜2 日 | INT8 環境でのフック動作確認。linear attention の o_proj 確認 |
| `configs/qwen3_5_moe.json` 作成 | 0.5 日 | 設定ファイル記述 |
| `path_patching_397b.py` 改造 | 1〜2 日 | full attention 層フィルタ、INT8 量子化、チェックポイント機能 |
| `run_path_patching.sh` 改造 | 0.5 日 | 単一プロセス・全8GPU使用に変更 |
| パイロット実行（10 サンプル） | 1〜2 日 | 実行時間ベンチマーク、動作確認 |
| 本番実行・監視 | 3〜7 日 | サンプル数による（72h × N run） |
| 結果マージ・可視化 | 0.5 日 | `merge_results.py` 流用 |
| **合計（開発）** | **3〜5 日** | パイロット完了まで |

---

## 6. 実行手順（推奨）

### Step 1: モデルダウンロード確認

```bash
ls /home/yuuki.nakamura/downloads/models/Qwen_Qwen3.5-397B-A17B/
```

### Step 2: フック動作確認

```bash
# 小さいサンプルで INT8 フック動作を確認（GPU 8枚使用）
sbatch --partition P08317 --gres=gpu:8 --mem=300G --time=2:00:00 \
  --wrap="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python \
    Phase2_path_patching/verify_hook_points.py \
    --model_path /home/yuuki.nakamura/downloads/models/Qwen_Qwen3.5-397B-A17B"
```

### Step 3: パイロット実行（10 サンプル）

```bash
sbatch Phase2_path_patching/1node8gpu/run_path_patching.sh overlap_medical 10
```

### Step 4: パイロット結果からスループット計測

```bash
grep "Processed\|seconds\|sample" Phase2_path_patching/1node8gpu/results_overlap_medical/run.log
```

### Step 5: 本番実行

パイロット結果をもとにサンプル数を決定し、`--sample_num N` を指定して実行。
必要なら複数 run を `--sample_start` と `--sample_num` で分割実行（要スクリプト改造）。

```bash
# 例: 100サンプル本番実行
sbatch Phase2_path_patching/1node8gpu/run_path_patching.sh overlap_medical 100
sbatch Phase2_path_patching/1node8gpu/run_path_patching.sh overlap_reasoning 100
```

---

## 7. リスクと対策

| リスク | 影響 | 対策 |
|---|---|---|
| フォワードパスが20秒超 | 72h で26サンプル以下 | サンプル数目標を下げる / linear attention を対象外に確定 |
| INT8 hook が動作しない | 実装不可 | INT4 (NF4) で再試行、または bfloat16 + CPU offload を検討 |
| linear attention の o_proj が異なる | hook 対象が不明 | verify_hook_points.py で確認後、full attention 15層に絞る |
| attn_output_gate がフック後に適用 | patch が無効化される | gate 後の出力にフック位置を変更（forward hook を使用） |
| OOM（量子化後も VRAM 不足） | 実行不可 | INT4 に切り替え、または batch_size=1 + gradient checkpointing 無効化確認 |
| 72h SLURM 制限 | 途中終了 | チェックポイント機能で resume 可能にする |

---

## 8. 参考: Qwen30B との主要な違いまとめ

```
Qwen30B-A3B                         Qwen3.5-397B-A17B
--------------------                --------------------
1 GPU / プロセス                    8 GPU / プロセス (device_map='auto')
データ並列 8プロセス                 データ並列 不可（1プロセスのみ）
48層 全 full attention              60層 = 45 linear + 15 full attention
head_dim = 64                       head_dim = 256
bf16 ロード (~60GB)                 INT8 ロード (~397GB)
1,538 forward/sample               482 forward/sample (full attn 15層のみ)
model_type = qwen3_moe             model_type = qwen3_5_moe
```
