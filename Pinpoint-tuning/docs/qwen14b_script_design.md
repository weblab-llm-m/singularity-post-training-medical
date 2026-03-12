# 産婦人科データ × Qwen14B × 翻訳論文手法：スクリプト構造設計書

**作成日**: 2025年10月23日
**目的**: 翻訳論文の手法で機能分類（Medical Term, Guideline, Reasoning）を行うための完全スクリプト設計
**対象**: 産婦人科診療ガイドライン2023データ + Qwen3-14Bモデル

---

## 目次

1. [全体アーキテクチャ](#1-全体アーキテクチャ)
2. [新規作成スクリプト詳細](#2-新規作成スクリプト詳細)
3. [既存改変スクリプト詳細](#3-既存改変スクリプト詳細)
4. [実装優先順位と依存関係](#4-実装優先順位と依存関係)
5. [最小実装セット](#5-最小実装セット)
6. [実行手順](#6-実行手順)

---

## 1. 全体アーキテクチャ

### 1.1 ディレクトリ構造

```
medical_path_patching/
│
├── 📁 Phase1_data_preparation/
│   ├── medical_term_annotator.py          【新規】
│   ├── counterfactual_generator.py        【新規】
│   ├── path_patching_data_builder.py      【改変】
│   └── medical_terms_dictionary.json      【新規データ】
│
├── 📁 Phase2_path_patching/
│   ├── path_patching_medical.py           【改変】
│   ├── hook_functions.py                  【既存・流用】
│   ├── dataset.py                         【既存・流用】
│   └── utils.py                           【部分改変】
│
├── 📁 Phase3_attention_analysis/
│   ├── attention_extractor.py             【新規】
│   ├── head_classifier.py                 【新規】
│   ├── medical_pattern_detector.py        【新規】
│   └── classification_criteria.json       【新規データ】
│
├── 📁 Phase4_visualization/
│   ├── heatmap_generator.py               【新規】
│   ├── statistical_analyzer.py            【新規】
│   ├── report_generator.py                【新規】
│   └── plot_templates/                    【新規ディレクトリ】
│
├── 📁 Phase5_pinpoint_tuning/（オプション）
│   ├── select_trainable_heads.py          【新規】
│   └── run_spt_medical.sh                 【改変】
│
├── 📁 configs/
│   ├── qwen.json                          【既存・流用】
│   ├── medical_config.yaml                【新規】
│   └── head_classification_params.yaml    【新規】
│
├── 📁 utils_common/
│   ├── tokenizer_utils.py                 【新規】
│   ├── medical_nlp_utils.py               【新規】
│   └── visualization_helpers.py           【新規】
│
└── 📁 scripts/
    ├── run_full_pipeline.sh               【新規】
    ├── run_phase1.sh                      【新規】
    ├── run_phase2.sh                      【新規】
    └── run_phase3.sh                      【新規】
```

### 1.2 スクリプト統計

| カテゴリ | 新規 | 改変 | 既存流用 | 合計 |
|---------|-----|-----|---------|-----|
| Phase 1: データ準備 | 3 | 1 | 0 | 4 |
| Phase 2: Path Patching | 0 | 2 | 2 | 4 |
| Phase 3: 注意解析 | 3 | 0 | 0 | 3 |
| Phase 4: 可視化 | 3 | 0 | 0 | 3 |
| Phase 5: Pinpoint Tuning | 1 | 1 | 0 | 2 |
| 共通ユーティリティ | 3 | 0 | 0 | 3 |
| 設定・データ | 3 | 0 | 0 | 3 |
| 実行スクリプト | 4 | 0 | 0 | 4 |
| **合計** | **20** | **4** | **7** | **31** |

---

## 2. 新規作成スクリプト詳細

### Phase 1: データ準備

#### 2.1.1 `medical_term_annotator.py`

**目的**: 産婦人科データから医療用語を自動抽出・アノテーション

**機能**:
- 医療用語の位置を特定（token index）
- 用語タイプを分類（疾患名、検査法、薬剤名など）
- Guideline指示語を特定（"産婦人科診療ガイドライン", "CQ"など）
- 推論キーワードを特定（"<think>", "選択肢", "正解は"など）

**入力**:
- `gynecology_guideline_2023.jsonl`

**出力**:
- `annotated_medical_data.jsonl`

**依存**:
- `transformers`, `spaCy`（医療用語抽出用）

**実装概要**:
```python
class MedicalTermAnnotator:
    def __init__(self, medical_dict_path):
        # 医療用語辞書を読み込み
        self.medical_terms = self.load_medical_dictionary(medical_dict_path)
        self.guideline_indicators = [
            "産婦人科診療ガイドライン",
            "婦人科外来編",
            "2023",
            "CQ"
        ]
        self.reasoning_keywords = [
            "<think>",
            "</think>",
            "選択肢",
            "正解は",
            "検討します"
        ]

    def annotate_sample(self, text, tokenizer):
        """
        Returns:
        {
            'medical_term_positions': [2, 5, 8, ...],
            'guideline_indicator_positions': [0, 1, ...],
            'reasoning_keyword_positions': [15, 20, ...],
            'medical_term_tokens': ["クラミジア", "IgA", ...],
            'term_types': {"クラミジア": "disease", "IgA": "biomarker"}
        }
        """
        tokens = tokenizer.tokenize(text)
        token_positions = {}

        # 医療用語の検出
        for i, token in enumerate(tokens):
            if token in self.medical_terms['diseases']:
                token_positions.setdefault('medical_term_positions', []).append(i)
            # ... 他のタイプも同様

        return token_positions
```

---

#### 2.1.2 `counterfactual_generator.py`

**目的**: 医療QAからCounterfactualデータを生成（翻訳論文方式）

**機能**:
- 医療用語を一般語に置換
- 文法構造は保持
- 複数の置換戦略をサポート

**入力**:
- `annotated_medical_data.jsonl`

**出力**:
- `counterfactual_medical_data.jsonl`

**依存**:
- `medical_term_annotator.py`

**実装概要**:
```python
class CounterfactualGenerator:
    def __init__(self):
        self.replacement_mapping = {
            # 疾患名 → 一般語
            "クラミジア子宮頸管炎": "感染症",
            "多嚢胞性卵巣症候群": "症候群",
            "PCOS": "病態",

            # 検査法 → 一般語
            "核酸増幅法": "検査法",
            "培養法": "診断法",

            # バイオマーカー → 一般語
            "IgA": "抗体A",
            "IgG": "抗体G",

            # ガイドライン → 一般語
            "産婦人科診療ガイドライン": "医療ガイドライン",
            "婦人科外来編2023": "外来編",
        }

    def generate_counterfactual(self, reference_text):
        """
        Strategy 1: 医療用語のみ置換
        Strategy 2: 医療用語 + ガイドライン置換
        Strategy 3: すべての専門語を置換
        """
        counterfactual = reference_text
        for medical_term, generic_term in self.replacement_mapping.items():
            counterfactual = counterfactual.replace(medical_term, generic_term)

        return counterfactual
```

---

#### 2.1.3 `medical_terms_dictionary.json`

**目的**: 医療用語の辞書データ

**内容**:
```json
{
  "diseases": [
    "クラミジア子宮頸管炎",
    "多嚢胞性卵巣症候群",
    "PCOS",
    "子宮内膜増殖症",
    "子宮内膜癌"
  ],
  "diagnostic_methods": [
    "核酸増幅法",
    "培養法",
    "抗体検査",
    "画像診断"
  ],
  "biomarkers": [
    "IgA",
    "IgG",
    "エストロゲン",
    "プロゲステロン"
  ],
  "guidelines": [
    "産婦人科診療ガイドライン",
    "婦人科外来編2023",
    "CQ"
  ],
  "treatments": [
    "アジスロマイシン",
    "レボフロキサシン",
    "経口避妊薬"
  ]
}
```

---

### Phase 3: 注意パターン解析

#### 2.3.1 `attention_extractor.py`

**目的**: Path patching実行中に各ヘッドの注意パターンを抽出

**機能**:
- 各レイヤー・各ヘッドの注意重みを記録
- END位置（最終生成位置）の注意パターンを抽出
- 保存形式: `[num_layers, num_heads, seq_len]`

**入力**:
- Path patching実行時にフック経由

**出力**:
- `attention_patterns.pt`

**依存**:
- `torch`, `transformers`

**実装概要**:
```python
class AttentionExtractor:
    def __init__(self, model, num_layers, num_heads):
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attention_patterns = {}

    def add_attention_hooks(self):
        """
        各レイヤーのattentionモジュールにフックを追加
        """
        hooks = []
        for layer_idx in range(self.num_layers):
            module = self.model.model.layers[layer_idx].self_attn

            def hook_fn(module, input, output, layer_idx=layer_idx):
                # Qwen3のattention_weightsを取得
                # output = (attn_output, attn_weights, past_key_value)
                attn_weights = output[1]  # [batch, num_heads, seq_len, seq_len]

                # END位置の注意パターンのみ保存
                self.attention_patterns[layer_idx] = attn_weights[:, :, -1, :]

            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

        return hooks

    def extract_and_save(self, input_ids, save_path):
        """
        実行して保存
        """
        hooks = self.add_attention_hooks()
        _ = self.model(input_ids)

        torch.save(self.attention_patterns, save_path)

        for hook in hooks:
            hook.remove()
```

---

#### 2.3.2 `head_classifier.py`

**目的**: 抽出された注意パターンから3種類のヘッドを分類

**機能**:
- Medical Term Headsの検出
- Guideline Indicator Headsの検出
- Reasoning Flow Headsの検出
- 分類基準は翻訳論文の手法を踏襲

**入力**:
- `attention_patterns.pt`
- `annotated_medical_data.jsonl`

**出力**:
- `head_classification_results.json`

**依存**:
- `attention_extractor.py`

**実装概要**:
```python
class HeadClassifier:
    def __init__(self, attention_patterns, annotation_data, criteria_config):
        self.attention_patterns = attention_patterns
        self.annotation_data = annotation_data
        self.criteria = self.load_criteria(criteria_config)

    def classify_all_heads(self):
        """
        全ヘッドを3種類に分類

        Returns:
        {
            'medical_term_heads': [(layer, head), ...],
            'guideline_indicator_heads': [(layer, head), ...],
            'reasoning_flow_heads': [(layer, head), ...],
            'unclassified': [(layer, head), ...]
        }
        """
        results = {
            'medical_term_heads': [],
            'guideline_indicator_heads': [],
            'reasoning_flow_heads': [],
            'unclassified': []
        }

        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                head_type = self.classify_single_head(layer, head)
                results[head_type].append((layer, head))

        return results

    def classify_single_head(self, layer, head):
        """
        翻訳論文の分類基準を適用
        """
        attn_pattern = self.attention_patterns[layer][head]  # [seq_len]

        # Medical Term Headsの判定
        if self.is_medical_term_head(attn_pattern):
            return 'medical_term_heads'

        # Guideline Indicator Headsの判定
        elif self.is_guideline_indicator_head(attn_pattern):
            return 'guideline_indicator_heads'

        # Reasoning Flow Headsの判定
        elif self.is_reasoning_flow_head(attn_pattern):
            return 'reasoning_flow_heads'

        else:
            return 'unclassified'

    def is_medical_term_head(self, attn_pattern):
        """
        医療用語位置への注意が高い
        翻訳論文のSource Heads検出と同じロジック
        """
        medical_positions = self.annotation_data['medical_term_positions']
        medical_attention_score = attn_pattern[medical_positions].mean()

        # 閾値: 医療用語位置への平均注意 > 0.3
        return medical_attention_score > self.criteria['medical_term_threshold']

    def is_guideline_indicator_head(self, attn_pattern):
        """
        ガイドライン指示語へのスパイク状注意
        翻訳論文のIndicator Heads検出と同じロジック
        """
        guideline_positions = self.annotation_data['guideline_indicator_positions']

        # スパイク検出
        max_attn_to_guideline = attn_pattern[guideline_positions].max()
        other_positions = [i for i in range(len(attn_pattern))
                          if i not in guideline_positions]
        mean_other_attn = attn_pattern[other_positions].mean()

        spike_ratio = max_attn_to_guideline / (mean_other_attn + 1e-10)

        # 閾値: 最大注意 > 0.7 AND スパイク比 > 5
        return (max_attn_to_guideline > self.criteria['spike_threshold'] and
                spike_ratio > self.criteria['spike_ratio'])

    def is_reasoning_flow_head(self, attn_pattern):
        """
        推論キーワード・隣接トークンへの均一注意
        翻訳論文のPositional Heads検出と同じロジック
        """
        reasoning_positions = self.annotation_data['reasoning_keyword_positions']

        # 隣接位置を含む
        adjacent_positions = list(range(len(attn_pattern) - 3, len(attn_pattern)))
        relevant_positions = list(set(reasoning_positions + adjacent_positions))

        # 均一性を測定
        relevant_attention = attn_pattern[relevant_positions]
        attention_std = relevant_attention.std()
        attention_mean = relevant_attention.mean()

        # 閾値: 標準偏差 < 0.1 AND 平均注意 > 0.4
        return (attention_std < self.criteria['uniformity_threshold'] and
                attention_mean > self.criteria['attention_mean_threshold'])
```

---

#### 2.3.3 `medical_pattern_detector.py`

**目的**: 医療QA特有のパターンを検出する補助スクリプト

**機能**:
- 思考プロセス（`<think>`）の解析
- 選択肢パターンの検出
- 正解到達経路の追跡

**入力**:
- `attention_patterns.pt`
- `medical_data`

**出力**:
- `medical_specific_patterns.json`

**実装概要**:
```python
class MedicalPatternDetector:
    def detect_think_pattern(self, attention_patterns):
        """
        <think>タグ内での注意パターンを解析
        """
        think_start_pos = self.find_token_position("<think>")
        think_end_pos = self.find_token_position("</think>")

        # <think>内での注意フロー
        think_attention = attention_patterns[:, :, think_start_pos:think_end_pos]

        return {
            'think_attention_flow': think_attention,
            'peak_attention_positions': self.find_peaks(think_attention)
        }
```

---

### Phase 4: 可視化・レポート

#### 2.4.1 `heatmap_generator.py`

**目的**: 3種類のヘッドごとにヒートマップを生成

**機能**:
- Medical Term Headsのヒートマップ
- Guideline Indicator Headsのヒートマップ
- Reasoning Flow Headsのヒートマップ
- Path patching結果との重ね合わせ

**入力**:
- `head_classification_results.json`
- `path_patching_results.pt`

**出力**:
- `heatmaps/*.png`

**依存**:
- `plotly`, `matplotlib`

**実装概要**:
```python
class HeatmapGenerator:
    def generate_classification_heatmap(self, classification_results, patching_results):
        """
        3種類のヒートマップを生成

        Color coding:
        - Medical Term Heads: 赤系
        - Guideline Indicator Heads: 青系
        - Reasoning Flow Heads: 緑系
        - Unclassified: 灰色
        """
        import plotly.graph_objects as go

        fig = go.Figure()

        # レイヤー × ヘッドのマトリックス
        data = np.zeros((num_layers, num_heads, 3))  # RGB

        for layer in range(num_layers):
            for head in range(num_heads):
                if (layer, head) in classification_results['medical_term_heads']:
                    data[layer, head] = [1, 0, 0]  # 赤
                elif (layer, head) in classification_results['guideline_indicator_heads']:
                    data[layer, head] = [0, 0, 1]  # 青
                elif (layer, head) in classification_results['reasoning_flow_heads']:
                    data[layer, head] = [0, 1, 0]  # 緑

        fig.add_trace(go.Heatmap(z=data))
        fig.write_image('heatmaps/classification_heatmap.png')

        return fig
```

---

#### 2.4.2 `statistical_analyzer.py`

**目的**: ヘッド分類の統計分析

**機能**:
- 各タイプの分布（レイヤーごと）
- Path patching impactとの相関
- 統計的有意性検定

**入力**:
- `head_classification_results.json`
- `path_patching_results.pt`

**出力**:
- `statistical_report.json`

**依存**:
- `scipy`, `numpy`

**実装概要**:
```python
class StatisticalAnalyzer:
    def analyze_distribution_by_layer(self, classification_results):
        """
        レイヤーごとのヘッドタイプ分布
        """
        distribution = {}
        for layer in range(num_layers):
            distribution[layer] = {
                'medical_term': 0,
                'guideline_indicator': 0,
                'reasoning_flow': 0
            }

            for head_type, heads in classification_results.items():
                count = sum(1 for l, h in heads if l == layer)
                distribution[layer][head_type] = count

        return distribution

    def correlation_with_patching_impact(self, classification_results, patching_results):
        """
        ヘッドタイプとPath patching impactの相関
        """
        from scipy.stats import spearmanr

        medical_impacts = [patching_results[l, h]
                          for l, h in classification_results['medical_term_heads']]
        guideline_impacts = [patching_results[l, h]
                            for l, h in classification_results['guideline_indicator_heads']]

        # 相関係数を計算
        correlation = {
            'medical_term_mean_impact': np.mean(medical_impacts),
            'guideline_mean_impact': np.mean(guideline_impacts),
            'correlation_coefficient': spearmanr(medical_impacts, guideline_impacts)
        }

        return correlation
```

---

#### 2.4.3 `report_generator.py`

**目的**: 統合レポート生成（Markdown/HTML）

**機能**:
- 発見されたヘッドのサマリー
- 各タイプの特徴
- 推奨Pinpoint Tuning対象

**入力**:
- すべての分析結果

**出力**:
- `medical_head_analysis_report.md/html`

**実装概要**:
```python
class ReportGenerator:
    def generate_markdown_report(self, all_results):
        """
        Markdownレポート生成
        """
        report = f"""
# 産婦人科データ Path Patching 分析レポート

## サマリー

- **総ヘッド数**: {num_layers * num_heads}
- **Medical Term Heads**: {len(all_results['classification']['medical_term_heads'])}
- **Guideline Indicator Heads**: {len(all_results['classification']['guideline_indicator_heads'])}
- **Reasoning Flow Heads**: {len(all_results['classification']['reasoning_flow_heads'])}

## レイヤーごとの分布

{self.generate_distribution_table(all_results['statistical']['distribution'])}

## 推奨Pinpoint Tuning対象

{self.generate_tuning_recommendations(all_results['selected_heads'])}
"""

        with open('medical_head_analysis_report.md', 'w') as f:
            f.write(report)
```

---

### Phase 5: Pinpoint Tuning

#### 2.5.1 `select_trainable_heads.py`

**目的**: 3種類のヘッド分類結果からPinpoint Tuning対象を選択

**機能**:
- Path patching impact上位 + 機能的に重要なヘッドを選択
- 優先順位: Medical Term > Guideline > Reasoning
- 出力: `trainable_heads.json`（SPTスクリプトで使用）

**入力**:
- `head_classification_results.json`
- `path_patching_results.pt`

**出力**:
- `trainable_heads.json`

**実装概要**:
```python
class TrainableHeadSelector:
    def select_heads(self, classification_results, patching_results, config):
        """
        選択戦略:
        1. Medical Term Headsで impact > threshold の全て
        2. Guideline Indicator Headsで impact > threshold の全て
        3. Reasoning Flow Headsで impact > threshold の上位50%

        合計: 全体の5-10%のヘッド
        """
        selected_heads = []

        # Medical Term Headsを優先
        for layer, head in classification_results['medical_term_heads']:
            impact = patching_results[layer][head].item()
            if impact > config['medical_threshold']:
                selected_heads.append({
                    'layer': layer,
                    'head': head,
                    'type': 'medical_term',
                    'impact': impact,
                    'priority': 'high'
                })

        # Guideline Indicator Heads
        for layer, head in classification_results['guideline_indicator_heads']:
            impact = patching_results[layer][head].item()
            if impact > config['guideline_threshold']:
                selected_heads.append({
                    'layer': layer,
                    'head': head,
                    'type': 'guideline_indicator',
                    'impact': impact,
                    'priority': 'high'
                })

        # Reasoning Flow Heads（上位50%のみ）
        reasoning_heads_with_impact = [
            (l, h, patching_results[l][h].item())
            for l, h in classification_results['reasoning_flow_heads']
        ]
        reasoning_heads_with_impact.sort(key=lambda x: x[2], reverse=True)

        top_reasoning = reasoning_heads_with_impact[:len(reasoning_heads_with_impact)//2]
        for layer, head, impact in top_reasoning:
            if impact > config['reasoning_threshold']:
                selected_heads.append({
                    'layer': layer,
                    'head': head,
                    'type': 'reasoning_flow',
                    'impact': impact,
                    'priority': 'medium'
                })

        # 保存
        with open('trainable_heads.json', 'w') as f:
            json.dump(selected_heads, f, indent=2)

        print(f"Selected {len(selected_heads)} heads for training")
        return selected_heads
```

---

### 共通ユーティリティ

#### 2.6.1 `tokenizer_utils.py`

**目的**: トークナイザー関連の共通処理

**機能**:
- Qwen3トークナイザーの初期化
- トークンIDから位置マッピング
- 医療用語のトークン分割処理

**実装概要**:
```python
class TokenizerUtils:
    @staticmethod
    def initialize_qwen3_tokenizer(model_path):
        """Qwen3トークナイザーの初期化"""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @staticmethod
    def find_token_positions(text, tokenizer, target_words):
        """
        特定の単語のトークン位置を検出
        """
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        positions = {}
        for word in target_words:
            word_tokens = tokenizer.tokenize(word)
            # トークン列のマッチング
            for i in range(len(tokens) - len(word_tokens) + 1):
                if tokens[i:i+len(word_tokens)] == word_tokens:
                    positions[word] = list(range(i, i+len(word_tokens)))

        return positions
```

---

#### 2.6.2 `medical_nlp_utils.py`

**目的**: 医療NLP処理

**機能**:
- 医療用語の正規化
- 同義語展開
- 用語タイプ判定

**実装概要**:
```python
class MedicalNLPUtils:
    @staticmethod
    def normalize_medical_term(term):
        """
        医療用語の正規化
        例: "PCOS" → "多嚢胞性卵巣症候群"
        """
        synonym_dict = {
            "PCOS": "多嚢胞性卵巣症候群",
            "IgA抗体": "IgA",
            "IgG抗体": "IgG"
        }
        return synonym_dict.get(term, term)

    @staticmethod
    def classify_term_type(term, medical_dict):
        """
        用語タイプの判定
        """
        for category, terms in medical_dict.items():
            if term in terms:
                return category
        return "unknown"
```

---

#### 2.6.3 `visualization_helpers.py`

**目的**: 可視化の共通処理

**機能**:
- カラースキーム定義
- プロット設定
- 凡例生成

**実装概要**:
```python
class VisualizationHelpers:
    @staticmethod
    def get_head_type_color(head_type):
        """
        ヘッドタイプごとの色定義
        """
        color_map = {
            'medical_term': 'red',
            'guideline_indicator': 'blue',
            'reasoning_flow': 'green',
            'unclassified': 'gray'
        }
        return color_map.get(head_type, 'gray')

    @staticmethod
    def create_legend(head_types):
        """
        凡例生成
        """
        import matplotlib.patches as mpatches

        patches = []
        for head_type in head_types:
            color = VisualizationHelpers.get_head_type_color(head_type)
            patch = mpatches.Patch(color=color, label=head_type)
            patches.append(patch)

        return patches
```

---

### 設定ファイル

#### 2.7.1 `medical_config.yaml`

```yaml
# 産婦人科データ設定
data:
  train_path: "/home/Competition2025/P05/shareP05/data/gynecology_guideline_2023_some_models_correct_formatted/train.parquet"
  test_path: "/home/Competition2025/P05/shareP05/data/gynecology_guideline_2023_some_models_correct_formatted/test.parquet"

model:
  path: "/home/Competition2025/P05/shareP05/models/Qwen3-14B"
  num_layers: 40
  num_heads: 40
  head_dim: 128

path_patching:
  batch_size: 2
  sample_num: 10

counterfactual:
  strategy: "medical_term_replacement"  # or "full_replacement"
  preserve_grammar: true
```

---

#### 2.7.2 `head_classification_params.yaml`

```yaml
# ヘッド分類基準（翻訳論文を参考に調整）
classification_criteria:
  medical_term:
    threshold: 0.30  # 医療用語位置への平均注意

  guideline_indicator:
    spike_threshold: 0.70  # 最大注意値
    spike_ratio: 5.0       # スパイク比率

  reasoning_flow:
    uniformity_threshold: 0.10  # 標準偏差
    attention_mean_threshold: 0.40  # 平均注意

selection:
  medical_priority_impact: 0.05  # impact > 5%で選択
  guideline_priority_impact: 0.08
  reasoning_priority_impact: 0.10
  max_total_heads: 64  # 全体の4%（40層×40ヘッド=1600の4%）
```

---

### 実行スクリプト

#### 2.8.1 `run_full_pipeline.sh`

```bash
#!/bin/bash
# 全Phase自動実行

set -e  # エラーで停止

echo "==================================================="
echo "Medical Path Patching Full Pipeline"
echo "==================================================="

echo ""
echo "Phase 1: Data Preparation"
echo "---------------------------------------------------"
bash scripts/run_phase1.sh

echo ""
echo "Phase 2: Path Patching Execution"
echo "---------------------------------------------------"
bash scripts/run_phase2.sh

echo ""
echo "Phase 3: Attention Analysis & Head Classification"
echo "---------------------------------------------------"
bash scripts/run_phase3.sh

echo ""
echo "Phase 4: Visualization & Reporting"
echo "---------------------------------------------------"
python Phase4_visualization/heatmap_generator.py
python Phase4_visualization/statistical_analyzer.py
python Phase4_visualization/report_generator.py

echo ""
echo "==================================================="
echo "Pipeline Completed!"
echo "==================================================="
echo ""
echo "Results:"
echo "  - Path Patching: Phase2_path_patching/results/"
echo "  - Classification: Phase3_attention_analysis/head_classification_results.json"
echo "  - Visualizations: Phase4_visualization/heatmaps/"
echo "  - Report: Phase4_visualization/medical_head_analysis_report.md"
```

---

#### 2.8.2 `run_phase1.sh`

```bash
#!/bin/bash
# Phase 1: データ準備

set -e

echo "Step 1: Medical Term Annotation"
python Phase1_data_preparation/medical_term_annotator.py \
    --input_path /home/Competition2025/P05/shareP05/data/gynecology_guideline_2023_some_models_correct_formatted/test.parquet \
    --output_path Phase1_data_preparation/annotated_medical_data.jsonl \
    --medical_dict Phase1_data_preparation/medical_terms_dictionary.json

echo "Step 2: Counterfactual Generation"
python Phase1_data_preparation/counterfactual_generator.py \
    --input_path Phase1_data_preparation/annotated_medical_data.jsonl \
    --output_path Phase1_data_preparation/counterfactual_medical_data.jsonl \
    --strategy medical_term_replacement

echo "Step 3: Path Patching Data Building"
python Phase1_data_preparation/path_patching_data_builder.py \
    --annotation_path Phase1_data_preparation/annotated_medical_data.jsonl \
    --counterfactual_path Phase1_data_preparation/counterfactual_medical_data.jsonl \
    --output_path Phase1_data_preparation/medical_path_patching_enhanced.jsonl

echo "Phase 1 Completed!"
```

---

#### 2.8.3 `run_phase2.sh`

```bash
#!/bin/bash
# Phase 2: Path Patching実行

set -e

echo "Running Path Patching with Attention Extraction"
python Phase2_path_patching/path_patching_medical.py \
    --model_path /home/Competition2025/P05/shareP05/models/Qwen3-14B \
    --data_path Phase1_data_preparation/medical_path_patching_enhanced.jsonl \
    --batch_size 2 \
    --sample_num 10 \
    --extract_attention true \
    --output_dir Phase2_path_patching/results/

echo "Phase 2 Completed!"
echo "Results saved to: Phase2_path_patching/results/"
```

---

#### 2.8.4 `run_phase3.sh`

```bash
#!/bin/bash
# Phase 3: 注意パターン解析とヘッド分類

set -e

echo "Running Head Classification"
python Phase3_attention_analysis/head_classifier.py \
    --attention_patterns Phase2_path_patching/results/attention_patterns.pt \
    --annotation_data Phase1_data_preparation/annotated_medical_data.jsonl \
    --criteria_config configs/head_classification_params.yaml \
    --output_path Phase3_attention_analysis/head_classification_results.json

echo "Running Medical Pattern Detection"
python Phase3_attention_analysis/medical_pattern_detector.py \
    --attention_patterns Phase2_path_patching/results/attention_patterns.pt \
    --medical_data Phase1_data_preparation/annotated_medical_data.jsonl \
    --output_path Phase3_attention_analysis/medical_specific_patterns.json

echo "Phase 3 Completed!"
```

---

## 3. 既存改変スクリプト詳細

### 3.1 `path_patching_medical.py`

**ベース**: `path_patching_hf.py`（sycophancy-interpretability）

**改変内容**:

```python
# 【追加】注意パターン抽出機能
from Phase3_attention_analysis.attention_extractor import AttentionExtractor

@torch.no_grad()
def path_patching_batch_with_attention(
    model, tokenizer, batch_data,
    module_input_name, module_output_name,
    num_layers, num_attention_heads, head_dim,
    extract_attention=False  # 【追加パラメータ】
):
    results = torch.zeros(size=(num_layers, num_attention_heads), device=model.device)

    # 【追加】注意パターン抽出器を初期化
    attention_extractor = None
    if extract_attention:
        attention_extractor = AttentionExtractor(model, num_layers, num_attention_heads)

    # Create path patching data
    xr_toks, xr_mask = create_batch(batch_data, split="xr_toks", pad_token_id=tokenizer.pad_token_id)
    xc_toks, xc_mask = create_batch(batch_data, split="xc_toks", pad_token_id=tokenizer.pad_token_id)

    # ... 既存のコード ...

    # 【追加】Reference実行時に注意パターンを記録
    if extract_attention:
        attention_hooks = attention_extractor.add_attention_hooks()

    # Forward A: Record the activation of Xr
    hooks = []
    for i in range(num_layers):
        # ... 既存のコード ...

    _ = model(xr_toks.to(model.device), attention_mask=xr_mask.to(model.device))

    # 【追加】注意パターンを保存
    if extract_attention:
        attention_extractor.extract_and_save('Phase2_path_patching/results/attention_patterns.pt')
        for hook in attention_hooks:
            hook.remove()

    # ... 既存のPath patchingループ ...

    return results, attention_extractor.attention_patterns if extract_attention else None


# 【変更】メイン関数
def main():
    parser = argparse.ArgumentParser("Path Patching Arguments")
    # ... 既存の引数 ...
    parser.add_argument("--extract_attention", type=bool, default=False,
                       help="Extract attention patterns")  # 【追加】

    args = parser.parse_args()

    # ... 既存のコード ...

    # 【追加】アノテーションデータを読み込み
    if args.extract_attention:
        with open('Phase1_data_preparation/annotated_medical_data.jsonl', 'r') as f:
            annotation_data = [json.loads(line) for line in f]

    # ... 既存のコード ...

    # 【変更】出力に注意パターンも含める
    results, attention_patterns = path_patching_batch_with_attention(
        model=model,
        tokenizer=tokenizer,
        batch_data=batch_data,
        # ... 既存パラメータ ...
        extract_attention=args.extract_attention
    )

    # 既存の保存処理
    torch.save(results, f"{output_dir}/results.pt")

    # 【追加】注意パターンも保存（既にextract_and_save内で保存済み）
    print(f"Attention patterns saved to: {output_dir}/attention_patterns.pt")
```

**改変行数**: 約50行追加

---

### 3.2 `utils.py`

**ベース**: `utils.py`（sycophancy-interpretability）

**改変内容**:

```python
# 【追加】医療QA特有のメトリクス計算
@torch.no_grad()
def compute_metric_medical(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_token_ids: torch.Tensor,
    record_token_ids: torch.Tensor,
    medical_term_positions: torch.Tensor = None,  # 【追加】
    weight_medical_terms: bool = False  # 【追加】
) -> torch.Tensor:
    """
    医療QA用のメトリクス計算

    オプション: 医療用語への注意が高い場合、メトリクスを調整
    """

    # 既存ロジック
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits.detach()

    batch_size = logits.size(0)
    logits_target = logits[torch.arange(batch_size), -1, target_token_ids]
    logits_sum = torch.zeros_like(logits_target).to(logits_target.device)

    for i in range(batch_size):
        logits_sum[i] = logits[i, -1, record_token_ids[i]].sum()

    # 【追加】医療用語位置の重み付け（オプション）
    if weight_medical_terms and medical_term_positions is not None:
        # 医療用語への注意が高い場合、メトリクスを調整
        # （実験的機能、必要に応じて実装）
        pass

    return logits_target / logits_sum


# 【追加】ヘッド分類結果を考慮した可視化
def show_path_patching_results_with_classification(
    m,
    classification_results=None,  # 【追加】
    xlabel="Head",
    ylabel="Layer",
    title="Path Patching Results with Head Classification",
    bartitle="Impact (%)",
    **kwargs,
):
    """
    Plot a heatmap with head classification overlay
    """
    import plotly.graph_objects as go

    # 基本のヒートマップ
    fig = px.imshow(
        m,
        title=title,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        **kwargs,
    )

    # 【追加】ヘッド分類結果を重ねる
    if classification_results is not None:
        # Medical Term Headsを赤丸でマーク
        medical_heads = classification_results.get('medical_term_heads', [])
        medical_x = [h for l, h in medical_heads]
        medical_y = [l for l, h in medical_heads]

        fig.add_scatter(
            x=medical_x,
            y=medical_y,
            mode='markers',
            marker=dict(color='red', size=10, symbol='circle-open', line=dict(width=2)),
            name='Medical Term Heads'
        )

        # Guideline Indicator Headsを青四角でマーク
        guideline_heads = classification_results.get('guideline_indicator_heads', [])
        guideline_x = [h for l, h in guideline_heads]
        guideline_y = [l for l, h in guideline_heads]

        fig.add_scatter(
            x=guideline_x,
            y=guideline_y,
            mode='markers',
            marker=dict(color='blue', size=10, symbol='square-open', line=dict(width=2)),
            name='Guideline Indicator Heads'
        )

        # Reasoning Flow Headsを緑三角でマーク
        reasoning_heads = classification_results.get('reasoning_flow_heads', [])
        reasoning_x = [h for l, h in reasoning_heads]
        reasoning_y = [l for l, h in reasoning_heads]

        fig.add_scatter(
            x=reasoning_x,
            y=reasoning_y,
            mode='markers',
            marker=dict(color='green', size=10, symbol='triangle-up-open', line=dict(width=2)),
            name='Reasoning Flow Heads'
        )

    fig.update_layout(
        yaxis_title=ylabel,
        xaxis_title=xlabel,
        showlegend=True,
        legend=dict(x=1.1, y=1.0)
    )

    return fig
```

**改変行数**: 約80行追加

---

### 3.3 `path_patching_data_builder.py`

**ベース**: `generate_gynecology_path_patching_data.py`

**改変内容**:

```python
#!/usr/bin/env python3
"""
拡張版Path Patchingデータ生成
アノテーション情報を含む
"""

import json
import argparse
import pandas as pd

def generate_enhanced_path_patching_data(
    annotation_path,
    counterfactual_path,
    output_path
):
    """
    既存のgenerate_gynecology_path_patching_data.pyに以下を追加:
    - medical_term_positions
    - guideline_indicator_positions
    - reasoning_keyword_positions
    """

    # アノテーションデータを読み込み
    with open(annotation_path, 'r') as f:
        annotations = [json.loads(line) for line in f]

    # Counterfactualデータを読み込み
    with open(counterfactual_path, 'r') as f:
        counterfactuals = [json.loads(line) for line in f]

    path_patching_data = []

    for i, (annot, cf) in enumerate(zip(annotations, counterfactuals)):
        # 既存ロジック
        path_item = {
            "id": i,
            "reference_data": annot['original_text'],  # 医療QA
            "counterfactual_data": cf['counterfactual_text'],  # 医療知識除外
            "predict_token": annot['correct_answer'],  # 例: "d"
            "record_tokens": ["a", "b", "c", "d", "e"],  # 全選択肢

            # 【追加】アノテーション情報
            "medical_term_positions": annot['medical_term_positions'],
            "medical_term_tokens": annot['medical_term_tokens'],
            "term_types": annot['term_types'],
            "guideline_indicator_positions": annot['guideline_indicator_positions'],
            "reasoning_keyword_positions": annot['reasoning_keyword_positions'],
        }

        path_patching_data.append(path_item)

    # 保存
    with open(output_path, 'w') as f:
        for item in path_patching_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Generated {len(path_patching_data)} enhanced path patching samples")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser("Enhanced Path Patching Data Builder")
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--counterfactual_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()

    generate_enhanced_path_patching_data(
        args.annotation_path,
        args.counterfactual_path,
        args.output_path
    )


if __name__ == '__main__':
    main()
```

**改変行数**: 約40行追加

---

### 3.4 `run_spt_medical.sh`

**ベース**: `scripts/spt_gynecology.sh`（産婦人科ガイドに記載）

**改変内容**:

```bash
#!/bin/bash
###
# Gynecology SPT Training Script with Functional Classification
###

# 実行例:
# WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=12345 RANK=0 NGPUS=1 \
# CUDA_VISIBLE_DEVICES=0 bash scripts/run_spt_medical.sh

# Model arguments
model_path=/home/Competition2025/P05/shareP05/models/Qwen3-14B
torch_dtype=bfloat16

# PEFT arguments (SPTではLoRAを使用しない)
peft_type=None
peft_config=None

# Dataset / Tokenization arguments
data_path=../prepare_training_data/finetuning_data/gynecology_instruction_data.jsonl
data_type=instruction_tuning
max_seq_len=2048
padding=False
padding_side=right
train_on_prompt=False

# Training arguments
training=True
deepspeed=configs/configs_deepspeed/deepspeed_config_stage1.json

num_epochs=3
max_steps=-1
save_steps=100
save_total_limit=5

learning_rate=3e-5
min_learning_rate=1e-7
lr_scheduler_type=polynomial
weight_decay=0.1

batch_size_per_device=1
global_batch_size=32

# Saving arguments
output_dir=outputs/gynecology_spt_functional
save_only_model=False
overwrite_output_dir=False
resume_from_checkpoint=None

# Pinpoint tuning arguments
path_patching_path=../Phase2_path_patching/results/Qwen3-14B

# 【追加】ヘッド選択方式を指定
head_selection_mode=functional_classification  # 新モード
trainable_heads_file=../Phase5_pinpoint_tuning/trainable_heads.json

# Precise level:
#   0: all parameters
#   1: qkv_proj + o_proj + mlp + wte/lm_head
#   2: qkv_proj + o_proj + mlp
#   3: qkv_proj + o_proj
#   4: qkv_proj only
#   5: functional_classification（Medical+Guideline+Reasoning） ← 【新規追加】
precise_level=5

# 【変更】train_topkを動的に決定
train_topk=$(python3 -c "import json; data=json.load(open('$trainable_heads_file')); print(len(data))")
echo "Training $train_topk heads based on functional classification"

# Whether to train key-value projection
train_kv=False

cmd="WORLD_SIZE=$WORLD_SIZE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT RANK=$RANK NGPUS=$NGPUS MODEL_PATH=$model_path TORCH_DTYPE=$torch_dtype PEFT_TYPE=$peft_type PEFT_CONFIG=$peft_config DATA_PATH=$data_path DATA_TYPE=$data_type MAX_SEQ_LEN=$max_seq_len PADDING=$padding PADDING_SIDE=$padding_side TRAIN_ON_PROMPT=$train_on_prompt TRAINING=$training DEEPSPEED=$deepspeed NUM_EPOCHS=$num_epochs MAX_STEPS=$max_steps SAVE_STEPS=$save_steps SAVE_TOTAL_LIMIT=$save_total_limit LEARNING_RATE=$learning_rate MIN_LEARNING_RATE=$min_learning_rate LR_SCHEDULER_TYPE=$lr_scheduler_type WEIGHT_DECAY=$weight_decay BATCH_SIZE_PER_DEVICE=$batch_size_per_device GLOBAL_BATCH_SIZE=$global_batch_size OUTPUT_DIR=$output_dir SAVE_ONLY_MODEL=$save_only_model OVERWRITE_OUTPUT_DIR=$overwrite_output_dir RESUME_FROM_CHECKPOINT=$resume_from_checkpoint PATH_PATCHING_PATH=$path_patching_path PRECISE_LEVEL=$precise_level TRAIN_TOPK=$train_topk TRAIN_KV=$train_kv HEAD_SELECTION_MODE=$head_selection_mode TRAINABLE_HEADS_FILE=$trainable_heads_file bash scripts/run_train.sh"

eval $cmd
```

**改変行数**: 約25行追加

---

## 4. 実装優先順位と依存関係

### 4.1 依存関係グラフ

```
                    ┌─────────────────┐
                    │ medical_terms_  │
                    │ dictionary.json │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ medical_term_   │
                    │ annotator.py    │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
┌────────▼────────┐ ┌────────▼────────┐ ┌───────▼────────┐
│counterfactual_  │ │ path_patching_  │ │ tokenizer_     │
│ generator.py    │ │ data_builder.py │ │ utils.py       │
└────────┬────────┘ └────────┬────────┘ └────────────────┘
         │                   │
         └───────────┬───────┘
                     │
            ┌────────▼────────┐
            │ path_patching_  │
            │ medical.py      │
            │ (+ 既存utils)   │
            └────────┬────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼────────┐    ┌─────────▼──────┐
│ Path Patching   │    │ Attention      │
│ Results.pt      │    │ Patterns.pt    │
└────────┬────────┘    └────────┬───────┘
         │                      │
         └──────────┬───────────┘
                    │
           ┌────────▼────────┐
           │ head_classifier │
           │ .py             │
           └────────┬────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
┌────────▼────────┐  ┌─────────▼──────┐
│ heatmap_        │  │ statistical_   │
│ generator.py    │  │ analyzer.py    │
└────────┬────────┘  └────────┬───────┘
         │                    │
         └────────┬───────────┘
                  │
         ┌────────▼────────┐
         │ report_         │
         │ generator.py    │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ select_         │
         │ trainable_heads │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ run_spt_        │
         │ medical.sh      │
         └─────────────────┘
```

### 4.2 実装優先順位（Phase順）

#### **Priority 1: 必須・基盤（Week 1）**

| 順位 | ファイル | タイプ | 見積時間 |
|-----|---------|-------|---------|
| 1 | `medical_terms_dictionary.json` | 新規・手動 | 2-3時間 |
| 2 | `tokenizer_utils.py` | 新規 | 3-4時間 |
| 3 | `medical_term_annotator.py` | 新規 | 6-8時間 |
| 4 | `counterfactual_generator.py` | 新規 | 4-6時間 |
| 5 | `path_patching_data_builder.py` | 改変 | 3-4時間 |

**Week 1合計**: 約18-25時間（3-4日）

#### **Priority 2: Path Patching実行（Week 1-2）**

| 順位 | ファイル | タイプ | 見積時間 |
|-----|---------|-------|---------|
| 6 | `attention_extractor.py` | 新規 | 6-8時間 |
| 7 | `path_patching_medical.py` | 改変 | 4-6時間 |
| 8 | `utils.py`（改変部分） | 部分改変 | 2-3時間 |
| 9 | `run_phase1.sh`, `run_phase2.sh` | 新規 | 1-2時間 |

**Week 2前半合計**: 約13-19時間（2-3日）

#### **Priority 3: 注意パターン解析（Week 2後半）**

| 順位 | ファイル | タイプ | 見積時間 |
|-----|---------|-------|---------|
| 10 | `head_classification_params.yaml` | 設定 | 1時間 |
| 11 | `head_classifier.py` | 新規 | 8-10時間 |
| 12 | `medical_pattern_detector.py` | 新規 | 4-6時間 |
| 13 | `run_phase3.sh` | 新規 | 1時間 |

**Week 2後半合計**: 約14-18時間（2-3日）

#### **Priority 4: 可視化・分析（Week 3）**

| 順位 | ファイル | タイプ | 見積時間 |
|-----|---------|-------|---------|
| 14 | `visualization_helpers.py` | 新規 | 2-3時間 |
| 15 | `heatmap_generator.py` | 新規 | 6-8時間 |
| 16 | `statistical_analyzer.py` | 新規 | 4-6時間 |
| 17 | `report_generator.py` | 新規 | 4-6時間 |

**Week 3前半合計**: 約16-23時間（2-3日）

#### **Priority 5: Pinpoint Tuning（Week 3後半、オプション）**

| 順位 | ファイル | タイプ | 見積時間 |
|-----|---------|-------|---------|
| 18 | `select_trainable_heads.py` | 新規 | 4-6時間 |
| 19 | `run_spt_medical.sh` | 改変 | 2-3時間 |

**Week 3後半合計**: 約6-9時間（1日）

---

## 5. 最小実装セット

最初に動かすための**必須5ファイル + 1設定**:

### 5.1 最小構成

```
medical_path_patching_minimal/
├── medical_term_annotator.py          【新規】
├── counterfactual_generator.py        【新規】
├── path_patching_medical.py           【改変】
├── attention_extractor.py             【新規】
├── head_classifier.py                 【新規】
└── medical_terms_dictionary.json      【データ】
```

### 5.2 最小実装の実行手順

```bash
# Step 0: 医療用語辞書を準備（手動）
# medical_terms_dictionary.json を作成

# Step 1: アノテーション
python medical_term_annotator.py \
    --input_path /path/to/gynecology_data.parquet \
    --output_path annotated_medical_data.jsonl \
    --medical_dict medical_terms_dictionary.json

# Step 2: Counterfactual生成
python counterfactual_generator.py \
    --input_path annotated_medical_data.jsonl \
    --output_path counterfactual_data.jsonl

# Step 3: Path Patching + 注意パターン抽出
python path_patching_medical.py \
    --model_path /path/to/Qwen3-14B \
    --data_path path_patching_data.jsonl \
    --extract_attention true

# Step 4: ヘッド分類
python head_classifier.py \
    --attention_patterns attention_patterns.pt \
    --annotation_data annotated_medical_data.jsonl \
    --output_path head_classification_results.json
```

### 5.3 最小実装の期待出力

```json
// head_classification_results.json
{
  "medical_term_heads": [
    [10, 5],
    [12, 8],
    [15, 3],
    ...
  ],
  "guideline_indicator_heads": [
    [8, 12],
    [9, 15],
    ...
  ],
  "reasoning_flow_heads": [
    [20, 7],
    [25, 10],
    ...
  ],
  "statistics": {
    "total_medical_term": 45,
    "total_guideline_indicator": 32,
    "total_reasoning_flow": 28,
    "total_unclassified": 1495
  }
}
```

---

## 6. 実行手順

### 6.1 環境構築

```bash
cd /home/Competition2025/P08/P08U023/model_analyze/medical_path_patching

# 仮想環境作成
python3 -m venv venv
source venv/bin/activate

# 依存パッケージインストール
pip install torch transformers pandas numpy scipy matplotlib plotly pyyaml
```

### 6.2 フル実行（推奨）

```bash
# 全Phaseを自動実行
bash scripts/run_full_pipeline.sh
```

### 6.3 Phase別実行

```bash
# Phase 1のみ
bash scripts/run_phase1.sh

# Phase 2のみ
bash scripts/run_phase2.sh

# Phase 3のみ
bash scripts/run_phase3.sh

# Phase 4のみ（可視化）
python Phase4_visualization/heatmap_generator.py
python Phase4_visualization/statistical_analyzer.py
python Phase4_visualization/report_generator.py
```

### 6.4 結果の確認

```bash
# Path Patching結果
python3 -c "
import torch
results = torch.load('Phase2_path_patching/results/Qwen3-14B/results.pt')
print('Results shape:', results.shape)
print('Top-10 important heads:')
topk_values, topk_indices = torch.topk(results.flatten(), 10)
for idx, val in zip(topk_indices, topk_values):
    layer = idx // results.shape[1]
    head = idx % results.shape[1]
    print(f'  Layer {layer}, Head {head}: {val:.2f}%')
"

# ヘッド分類結果
python3 -c "
import json
with open('Phase3_attention_analysis/head_classification_results.json') as f:
    results = json.load(f)
print('Medical Term Heads:', len(results['medical_term_heads']))
print('Guideline Indicator Heads:', len(results['guideline_indicator_heads']))
print('Reasoning Flow Heads:', len(results['reasoning_flow_heads']))
"

# レポート確認
cat Phase4_visualization/medical_head_analysis_report.md
```

---

## 付録

### A. トラブルシューティング

#### A.1 OOMエラー

**症状**: CUDA out of memory

**対策**:
```bash
# batch_sizeを削減
python path_patching_medical.py --batch_size 1

# または、gradient checkpointingを有効化
# path_patching_medical.pyに追加:
model.gradient_checkpointing_enable()
```

#### A.2 医療用語が検出されない

**症状**: アノテーション結果が空

**対策**:
```python
# medical_terms_dictionary.jsonを確認
# 産婦人科データに実際に出現する用語を追加
```

#### A.3 ヘッド分類が偏る

**症状**: ほぼ全てがunclassified

**対策**:
```yaml
# head_classification_params.yamlの閾値を調整
classification_criteria:
  medical_term:
    threshold: 0.20  # 0.30 → 0.20に緩和
```

---

### B. 参考文献

1. **翻訳メカニズム論文**
   Zhang et al. (2025) "Exploring Translation Mechanism of Large Language Models" arXiv:2502.11806

2. **Sycophancy論文**
   Chen et al. (2024) "From Yes-Men to Truth-Tellers" arXiv:2409.01658

3. **Path Patching原論文**
   Wang et al. (2022) "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small" arXiv:2211.00593

4. **産婦人科データ**
   team-suzuki/gynecology_guideline_2023_some_models_correct_formatted

---

**作成日**: 2025年10月23日
**最終更新**: 2025年10月23日
**バージョン**: 1.0
**ステータス**: 設計完了・実装準備中
