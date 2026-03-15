  3つのHeadタイプの検出方法の違い
  ┌───────────────────────────┬────────────────────────────────────┬────────────────────────────┐
  │        Head タイプ        │              検出方法              │    Counterfactual必要？    │
  ├───────────────────────────┼────────────────────────────────────┼────────────────────────────┤
  │ Medical Term Heads        │ 医療用語位置への平均注意値         │ ✓（医療→一般で変化を見る） │
  ├───────────────────────────┼────────────────────────────────────┼────────────────────────────┤
  │ Guideline Indicator Heads │ ガイドライン位置へのスパイク比率   │ ✓（医療→一般で変化を見る） │
  ├───────────────────────────┼────────────────────────────────────┼────────────────────────────┤
  │ Reasoning Flow Heads      │ 推論キーワード位置での注意の均一性 │ ✗（パターン分析のみ）      │
  └───────────────────────────┴────────────────────────────────────┴────────────────────────────┘
  Reasoning Flow Headsの具体的検出ロジック

  # Phase3_attention_analysis/head_classifier.py
  def is_reasoning_flow_head(self, attn_pattern):
      # 推論キーワード位置 + 末尾3トークンへの注意を取得
      relevant_attention = attn_pattern[reasoning_positions + adjacent_positions]

      # 均一性を測定
      std = relevant_attention.std()
      mean = relevant_attention.mean()
      relative_std = std / mean

      # 均一な注意分布 → Reasoning Flow Head
      return relative_std < 1.2 and mean > 0.0005

  つまり:
  - reasoning_keyword_positionsはアノテーションデータに含まれる
  - Phase3で「その位置への注意が均一か」を分析
  - Counterfactual比較ではなく、パターンの形状で判定

  ---
  replacement_mapping.jsonでreasoning_keywordsを除外した理由

  正しい判断でした。reasoning_keywordsは置換対象ではなく、位置情報としてのみ使用されます。