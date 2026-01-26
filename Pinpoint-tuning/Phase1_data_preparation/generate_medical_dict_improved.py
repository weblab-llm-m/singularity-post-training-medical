#!/usr/bin/env python3
"""
Generate medical_terms_dictionary.json using Qwen3-235B-A22B-Thinking-2507
Improved version with correct column structure and no unnecessary generation flags
"""

import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_random_parquet_samples(parquet_path, num_samples=20, random_seed=42):
    """Load random samples from parquet file"""
    print(f"Loading data from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Randomly sample rows
    sampled_df = df.sample(n=min(num_samples, len(df)), random_state=random_seed)
    print(f"Randomly sampled {len(sampled_df)} rows (random_seed={random_seed})")

    # Extract question and answer from extra_info
    sample_texts = []
    for idx, row in sampled_df.iterrows():
        if 'extra_info' in row and isinstance(row['extra_info'], dict):
            question = row['extra_info'].get('question', '')
            answer = row['extra_info'].get('answer', '')
            combined = f"【質問】\n{question}\n\n【回答】\n{answer}"
            sample_texts.append(combined)

    print(f"Loaded {len(sample_texts)} sample texts (question + answer)")
    return sample_texts

def generate_medical_dictionary(random_seed=42, output_suffix=""):
    """Generate medical terms dictionary using Qwen3-14B model"""

    # Load random 20 samples from parquet
    parquet_path = "/home/Competition2025/P05/shareP05/data/gynecology_guideline_2023_some_models_correct_formatted/train.parquet"
    sample_texts = load_random_parquet_samples(parquet_path, num_samples=20, random_seed=random_seed)

    if not sample_texts:
        print("ERROR: No texts loaded!")
        return None

    # Create context from sampled data
    context_text = "\n\n---サンプル---\n\n".join(sample_texts)

    # Load model and tokenizer
    model_path = "/home/Competition2025/P05/shareP05/models/Qwen3-14B"
    print(f"\nLoading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,  # Use 'dtype' instead of deprecated 'torch_dtype'
        device_map='auto',
        trust_remote_code=True
    )

    # Enhanced prompt based on user requirements (simplified, no example JSON)
    prompt = f"""# 目的
あなたは優秀な日本語医療NLP（自然言語処理）エンジニアです。提供された「産婦人科診療ガイドライン2023データ（日本語テキスト）」から、機能分類に特化した**医療用語辞書（JSON形式）**を作成してください。

この辞書は、Path Patching手法を用いた大規模言語モデル（Qwen14B）の内部挙動解析において、Medical Term Heads、Guideline Indicator Heads、Reasoning Flow Headsを正確に特定するための基盤データとして使用されます。

## 辞書の構造と抽出定義

以下の7つのカテゴリに分類し、それぞれ日本語の用語をリストとして抽出してください。

1. **diseases**: 疾患名、症候群名、病態名。例: クラミジア子宮頸管炎、多嚢胞性卵巣症候群、PCOS
2. **diagnostic_methods**: 検査法、診断手順、医療手技。略語も正式名称と併記する。例: 核酸増幅法、超音波検査、MRI検査
3. **biomarkers**: 血液、体液、組織などで測定される指標。略語も含む。例: IgA、CA125、エストロゲン
4. **guidelines**: ガイドラインのタイトル、発行年、重要な指示語、レベル分類。例: 産婦人科診療ガイドライン、婦人科外来編2023、CQ、推奨度
5. **treatments**: 薬剤名、手術法、その他の治療介入。例: アジスロマイシン、経口避妊薬、子宮全摘術
6. **anatomical_terms**: 産婦人科に関連する具体的な器官名、組織名。例: 子宮、卵巣、子宮頸部
7. **reasoning_keywords**: LLMの推論過程を示すための特殊なキーワードや論理接続詞。例: <think>、</think>、正解は、したがって、なぜなら、検討します

## 出力形式の制約

- 出力は、指定されたJSON形式（UTF-8エンコーディング）のテキストブロック一つのみとしてください
- JSONのキーは、上記で定義された7つのカテゴリ名（全て小文字）と完全に一致させること
- 各リストには、入力テキストから抽出された用語を**20個以上**（総量が少ない場合は全て）含めること
- 各用語リスト内は重複を避けること
- **以下の様なJSONのみを出力し、説明文やマークダウンは一切含めないこと**

```json
{{
  "diseases": [
    "多嚢胞性卵巣症候群",
    "PCOS",
    "骨盤内炎症性疾患",
    "子宮内膜増殖症"
  ],
  "diagnostic_methods": [
    "核酸増幅法",
    "超音波検査",
    "MRI検査",
    "血液培養"
  ],
  "biomarkers": [
    "IgA",
    "CA125",
    "エストロゲン",
    "FSH"
  ],
  "guidelines": [
    "産婦人科診療ガイドライン",
    "婦人科外来編2023",
    "CQ",
    "推奨度"
  ],
  "treatments": [
    "アジスロマイシン",
    "経口避妊薬",
    "レトロゾール",
    "ペニシリン系抗菌薬"
  ],
  "anatomical_terms": [
    "子宮",
    "卵巣",
    "子宮頸部",
    "卵管"
  ],
  "reasoning_keywords": [
    "<think>",
    "</think>",
    "正解は",
    "したがって",
    "なぜなら",
    "検討します"
  ]
}}
```

## 産婦人科診療ガイドライン2023データ

以下は、train.parquetからランダムに抽出した100サンプルのデータです（質問と回答のみ）:

{context_text}

上記のデータに基づいて、医療用語辞書をJSON形式で出力してください。JSONのみを出力し、他のテキストは含めないでください。"""

    # Generate response
    print("\nGenerating medical dictionary...")
    print("This may take several minutes for a 235B parameter model...")

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print(f"Input tokens: {model_inputs.input_ids.shape[1]}")
    print("Starting generation...")

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20
        )

    # Decode response
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("\n" + "="*60)
    print("Generated Response (first 3000 chars):")
    print("="*60)
    print(response[:3000])
    if len(response) > 3000:
        print(f"\n... (truncated, total length: {len(response)} characters)")
    print("="*60 + "\n")

    # Extract JSON from response
    try:
        # Find JSON content (look for { ... })
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]

            # Try to parse JSON
            medical_dict = json.loads(json_str)

            # Validate required keys
            required_keys = [
                "diseases", "diagnostic_methods", "biomarkers",
                "guidelines", "treatments", "anatomical_terms", "reasoning_keywords"
            ]

            missing_keys = [key for key in required_keys if key not in medical_dict]
            if missing_keys:
                print(f"⚠ Warning: Missing keys: {missing_keys}")

            # Save to file
            output_path = f"medical_terms_dictionary_qwen_generated{output_suffix}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(medical_dict, f, ensure_ascii=False, indent=2)

            print(f"✓ Medical dictionary saved to: {output_path}")
            print(f"\nDictionary statistics:")
            for category, terms in medical_dict.items():
                print(f"  - {category}: {len(terms)} terms")

            return medical_dict
        else:
            print("✗ Error: Could not find JSON in response")
            # Save raw response for debugging
            with open("medical_dict_raw_response.txt", 'w', encoding='utf-8') as f:
                f.write(response)
            print("  Raw response saved to: medical_dict_raw_response.txt")
            return None

    except json.JSONDecodeError as e:
        print(f"✗ Error parsing JSON: {e}")
        print(f"  Attempted to parse from char {start_idx} to {end_idx}")
        print(f"  First 500 chars: {json_str[:500]}...")
        # Save raw response for debugging
        with open("medical_dict_raw_response.txt", 'w', encoding='utf-8') as f:
            f.write(response)
        print("  Raw response saved to: medical_dict_raw_response.txt")
        return None

if __name__ == "__main__":
    import sys

    # Get random seed and suffix from command line args
    random_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    output_suffix = sys.argv[2] if len(sys.argv) > 2 else ""

    print("="*60)
    print("Generating Medical Terms Dictionary with Qwen3-14B")
    print(f"Source: train.parquet (20 random samples, seed={random_seed})")
    print("="*60 + "\n")

    result = generate_medical_dictionary(random_seed=random_seed, output_suffix=output_suffix)

    if result:
        print("\n✓ Generation completed successfully!")
    else:
        print("\n✗ Generation failed. Please check the raw response file.")
