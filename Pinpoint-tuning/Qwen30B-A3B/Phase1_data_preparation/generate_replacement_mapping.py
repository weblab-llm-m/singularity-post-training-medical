#!/usr/bin/env python3
"""
Generate replacement_mapping.json using Qwen3-Next-80B-A3B-Instruct
医療用語を一般用語に変換するマッピング辞書を生成
"""

import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_replacement_mapping():
    """Generate medical term to generic term mapping using Qwen3-Next-80B-A3B-Instruct"""

    script_dir = Path(__file__).parent
    dict_path = script_dir / "medical_terms_dictionary.json"

    if not dict_path.exists():
        print(f"ERROR: {dict_path} not found!")
        print("Please run merge_dictionaries.py first.")
        return None

    # Load medical dictionary
    print(f"Loading medical dictionary from: {dict_path}")
    with open(dict_path, 'r', encoding='utf-8') as f:
        medical_dict = json.load(f)

    # Count total terms
    total_terms = sum(len(terms) for terms in medical_dict.values())
    print(f"Total terms to process: {total_terms}")

    # Load model and tokenizer
    model_path = "/home/yuuki.nakamura/downloads/models/Qwen_Qwen3-Next-80B-A3B-Instruct"
    print(f"\nLoading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True
    )

    # Define category to generic term mapping guidance
    category_generic_guidance = {
        "diseases": "健康状態",
        "diagnostic_methods": "検査",
        "biomarkers": "検査値",
        "profile": "属性情報",
        "treatments": "治療",
        "anatomical_terms": "部位",
        "reasoning_keywords": "接続詞"
    }

    # Create the prompt
    dict_json_str = json.dumps(medical_dict, ensure_ascii=False, indent=2)

    prompt = f"""# 目的
あなたは優秀な日本語NLPエンジニアです。医療用語辞書の各用語を、医療的な意味を持たない一般的な表現に変換したマッピング辞書を作成してください。

このマッピングは、大規模言語モデルの解析において「Counterfactual（反事実）データ」を生成するために使用されます。医療用語を一般語に置き換えることで、モデルが医療知識に依存している部分を特定します。

## 変換ルール

各カテゴリの用語は、以下の一般語に変換してください：

- **diseases（疾患名）** → 「健康状態」「体調」「状態」など
- **diagnostic_methods（診断方法）** → 「検査」「確認」「調査」など
- **biomarkers（バイオマーカー）** → 「検査値」「数値」「測定値」など
- **profile（患者属性）** → 属性の種類に応じて「年齢」「性別」「背景」「生活習慣」など
- **treatments（治療法）** → 「治療」「対処」「処置」など
- **anatomical_terms（解剖学用語）** → 「部位」「箇所」「器官」など
- **reasoning_keywords（推論キーワード）** → そのまま保持（変換不要）

## 特別なルール

1. **profile カテゴリ**の場合：
   - 年齢（例: 20歳、65歳）→ 「年齢」
   - 性別（例: 男性、女性）→ 「性別」
   - 既往歴/生活習慣（例: 喫煙歴あり、飲酒歴なし）→ 「生活背景」
   - 職業（例: 公務員、事務職）→ 「職業」
   - 妊娠関連（例: 妊娠12週、初産婦）→ 「妊娠状態」
   - 身体情報（例: 身長168cm、体重58kg）→ 「身体情報」
   - ADL/生活状況 → 「生活状況」

2. **reasoning_keywords カテゴリ**は変換不要（nullまたは空文字）

3. 同じカテゴリ内でも、文脈に応じて適切な一般語を選んでください

## 出力形式

以下のJSON形式で出力してください。キーが医療用語、値が一般語です：

```json
{{
  "糖尿病": "健康状態",
  "心電図": "検査",
  "血糖値": "検査値",
  "20歳": "年齢",
  "男性": "性別",
  "インスリン": "治療",
  "心臓": "部位",
  "したがって": null
}}
```

## 入力データ（医療用語辞書）

{dict_json_str}

上記の全ての用語に対して、マッピング辞書をJSON形式で出力してください。JSONのみを出力し、説明文は含めないでください。"""

    # Generate response
    print("\nGenerating replacement mapping...")
    print("This may take several minutes...")

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
            max_new_tokens=8192,
            do_sample=True,
            temperature=0.3,  # Lower temperature for more consistent mapping
            top_p=0.9,
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
            replacement_mapping = json.loads(json_str)

            # Validate: check if we have most terms covered
            all_terms = []
            for category, terms in medical_dict.items():
                all_terms.extend(terms)

            covered = sum(1 for term in all_terms if term in replacement_mapping)
            coverage = covered / len(all_terms) * 100

            print(f"Coverage: {covered}/{len(all_terms)} terms ({coverage:.1f}%)")

            # Remove null values (reasoning_keywords that don't need replacement)
            replacement_mapping = {k: v for k, v in replacement_mapping.items() if v is not None}

            # Save to file
            output_path = script_dir / "replacement_mapping.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(replacement_mapping, f, ensure_ascii=False, indent=2)

            print(f"\n✓ Replacement mapping saved to: {output_path}")
            print(f"  Total mappings: {len(replacement_mapping)}")

            # Print sample mappings per category
            print("\nSample mappings by category:")
            for category, terms in medical_dict.items():
                mapped_terms = [(t, replacement_mapping.get(t, "N/A")) for t in terms[:3]]
                print(f"  {category}:")
                for orig, repl in mapped_terms:
                    print(f"    {orig} → {repl}")

            return replacement_mapping
        else:
            print("✗ Error: Could not find JSON in response")
            raw_path = script_dir / "replacement_mapping_raw_response.txt"
            with open(raw_path, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"  Raw response saved to: {raw_path}")
            return None

    except json.JSONDecodeError as e:
        print(f"✗ Error parsing JSON: {e}")
        raw_path = script_dir / "replacement_mapping_raw_response.txt"
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(response)
        print(f"  Raw response saved to: {raw_path}")
        return None


if __name__ == "__main__":
    print("="*60)
    print("Generating Replacement Mapping with Qwen3-Next-80B-A3B-Instruct")
    print("Source: medical_terms_dictionary.json")
    print("="*60 + "\n")

    result = generate_replacement_mapping()

    if result:
        print("\n✓ Generation completed successfully!")
    else:
        print("\n✗ Generation failed. Please check the raw response file.")
