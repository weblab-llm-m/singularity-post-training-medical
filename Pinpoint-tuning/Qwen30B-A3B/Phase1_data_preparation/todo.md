### TODO: Qwen30B-A3B Phase1 データ準備パイプライン

#### ✅ 完了済み

| Step | ファイル | 説明 |
|------|---------|------|
| 0 | `extract_problem_texts.py` | igakuQAから`problem_text`を抽出 → `problem_texts.jsonl` |
| 1 | `generate_medical_dict_improved.py` | 医学用語辞書をLLMで生成（80B-A3Bモデル使用） |

#### 📋 今後のTODO

| Step | 作成するファイル | 説明 | 参考元（Qwen14B） |
|------|-----------------|------|------------------|
| 2 | `merge_dictionaries.py` | 複数seedで生成した辞書をマージ | `merge_dictionaries.py` |
| 3 | `medical_term_annotator.py` | igakuQAデータに医学用語位置をアノテーション | `medical_term_annotator.py` |
| 4 | `counterfactual_generator.py` | 反実仮想データ生成（医学用語を置換） | `counterfactual_generator.py` |
| 5 | `path_patching_data_builder.py` | アノテーション＋反実仮想を統合してPath Patching用データ作成 | `path_patching_data_builder.py` |
| 6 | `utils_common/` | 共通ユーティリティ（tokenizer_utils, medical_nlp_utils） | `utils_common/` |

#### 📋 各Stepの詳細

**Step 2: merge_dictionaries.py**
- 入力: `medical_terms_dictionary_qwen_generated_*.json`（複数seed）
- 出力: `medical_terms_dictionary.json`（統合版）
- 処理: 各カテゴリの用語を重複排除してマージ

**Step 3: medical_term_annotator.py**
- 入力: igakuQAデータ（HuggingFaceまたはjsonl）、`medical_terms_dictionary.json`
- 出力: `annotated_medical_data_full.jsonl`
- 処理: 各サンプルに対して `medical_term_positions`, `guideline_indicator_positions`, `reasoning_keyword_positions` を付与

**Step 4: counterfactual_generator.py**
- 入力: `annotated_medical_data_full.jsonl`
- 出力: `counterfactual_strategy*.jsonl`
- 処理: 医学用語を同カテゴリ別用語/一般用語に置換して反実仮想データ生成

**Step 5: path_patching_data_builder.py**
- 入力: `annotated_medical_data_full.jsonl`, `counterfactual_strategy*.jsonl`
- 出力: `path_patching_strategy*.jsonl`
- 処理: Path Patching用の (reference, counterfactual) ペアデータ作成

**Step 6: utils_common/**
- `tokenizer_utils.py`: Qwen30B-A3B用トークナイザー初期化
- `medical_nlp_utils.py`: 医学用語検出、`<think>`セクション分離など

#### 📁 最終的なファイル構成（目標）

```
Phase1_data_preparation/
├── extract_problem_texts.py      ✅
├── problem_texts.jsonl           ✅
├── generate_medical_dict_improved.py  ✅
├── merge_dictionaries.py         📋 TODO
├── medical_terms_dictionary.json 📋 TODO
├── medical_term_annotator.py     📋 TODO
├── annotated_medical_data_full.jsonl  📋 TODO
├── counterfactual_generator.py   📋 TODO
├── counterfactual_strategy*.jsonl     📋 TODO
├── path_patching_data_builder.py 📋 TODO
├── path_patching_strategy*.jsonl      📋 TODO
└── utils_common/                 📋 TODO
    ├── tokenizer_utils.py
    └── medical_nlp_utils.py
```

#### ⚠️ 注意点

1. **トークナイザー**: Qwen30B-A3B（MoE）用に調整が必要
2. **データ形式**: igakuQAは `problem_text` + `choices` の形式。Qwen14Bの産婦人科データ（`question` + `answer`）とは異なる
3. **反実仮想戦略**: `dataset_patching.py` で元の `problem_text` のみを使用する想定なので、反実仮想生成のロジックを調整