#!/usr/bin/env python3
"""
Generate medical_terms_dictionary.json using Qwen3-Next-80B-A3B-Instruct
For igakuQA (医師国家試験) dataset
"""

import json
import random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_random_problem_texts(jsonl_path: str, num_samples: int = 20, random_seed: int = 42):
    """Load random samples from problem_texts.jsonl"""
    print(f"Loading data from: {jsonl_path}")

    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Total rows: {len(samples)}")

    # Randomly sample rows
    random.seed(random_seed)
    sampled = random.sample(samples, min(num_samples, len(samples)))
    print(f"Randomly sampled {len(sampled)} rows (random_seed={random_seed})")

    # Extract problem_text
    sample_texts = []
    for item in sampled:
        problem_text = item.get('problem_text', '')
        problem_id = item.get('problem_id', '')
        if problem_text:
            sample_texts.append(f"【{problem_id}】\n{problem_text}")

    print(f"Loaded {len(sample_texts)} sample texts")
    return sample_texts


def generate_medical_dictionary(random_seed: int = 42, output_suffix: str = ""):
    """Generate medical terms dictionary using Qwen3-Next-80B-A3B-Instruct model"""

    # Load random samples from problem_texts.jsonl
    script_dir = Path(__file__).parent
    jsonl_path = script_dir / "problem_texts.jsonl"

    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found!")
        print("Please run extract_problem_texts.py first.")
        return None

    sample_texts = load_random_problem_texts(str(jsonl_path), num_samples=30, random_seed=random_seed)

    if not sample_texts:
        print("ERROR: No texts loaded!")
        return None

    # Create context from sampled data
    context_text = "\n\n---サンプル---\n\n".join(sample_texts)

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

    # Enhanced prompt for igakuQA (医師国家試験)
    prompt = f"""# 目的
あなたは優秀な日本語医療NLP（自然言語処理）エンジニアです。提供された「医師国家試験問題データ（日本語テキスト）」から、機能分類に特化した**医療用語辞書（JSON形式）**を作成してください。

この辞書は、Path Patching手法を用いた大規模言語モデル（Qwen30B-A3B）の内部挙動解析において、Medical Term Heads、Guideline Indicator Heads、Reasoning Flow Headsを正確に特定するための基盤データとして使用されます。

## 辞書の構造と抽出定義

以下の7つのカテゴリに分類し、それぞれ日本語の用語をリストとして抽出してください。

1. **diseases**: 疾患名、症候群名、病態名、症状名。例: 糖尿病、心筋梗塞、肺炎、高血圧症、意識障害、発熱
2. **diagnostic_methods**: 検査法、診断手順、医療手技、診察所見。例: 心電図、CT、MRI、血液検査、打診、聴診、触診
3. **biomarkers**: 血液、体液、組織などで測定される指標、検査値。例: 血糖値、クレアチニン、GOT、GPT、白血球数、CRP
4. **profile**: 患者の属性を示す語句。例: 20歳、男性、既往歴なし、喫煙歴あり
5. **treatments**: 薬剤名、手術法、治療法、処置。例: インスリン、抗菌薬、胃洗浄、輸液、手術、放射線療法
6. **anatomical_terms**: 器官名、組織名、解剖学的部位。例: 心臓、肺、肝臓、腎臓、脳、胸部、腹部
7. **reasoning_keywords**: 推論過程を示すキーワード、論理接続詞、考察に使う語句。例: したがって、なぜなら、まず、次に、考えられる、可能性がある

## 出力形式の制約

- 出力は、指定されたJSON形式（UTF-8エンコーディング）のテキストブロック一つのみとしてください
- JSONのキーは、上記で定義された7つのカテゴリ名（全て小文字）と完全に一致させること
- 各リストには、入力テキストから抽出された用語を**20個以上**（総量が少ない場合は全て）含めること
- 各用語リスト内は重複を避けること
- **以下の様なJSONのみを出力し、説明文やマークダウンは一切含めないこと**

```json
{{
  "diseases": [
    "糖尿病",
    "心筋梗塞",
    "肺炎",
    "高血圧症"
  ],
  "diagnostic_methods": [
    "心電図",
    "CT",
    "血液検査",
    "打診"
  ],
  "biomarkers": [
    "血糖値",
    "クレアチニン",
    "GOT",
    "白血球数"
  ],
  "profile": [
    "20歳",
    "男性",
    "既往歴なし",
    "喫煙歴あり"
  ],
  "treatments": [
    "インスリン",
    "抗菌薬",
    "胃洗浄",
    "輸液"
  ],
  "anatomical_terms": [
    "心臓",
    "肺",
    "肝臓",
    "腎臓"
  ],
  "reasoning_keywords": [
    "したがって",
    "なぜなら",
    "考えられる",
    "可能性がある"
  ]
}}
```

## 医師国家試験問題データ

以下は、igakuQAデータセットからランダムに抽出した問題文のサンプルです:

{context_text}

上記のデータに基づいて、医療用語辞書をJSON形式で出力してください。JSONのみを出力し、他のテキストは含めないでください。"""

    # Generate response
    print("\nGenerating medical dictionary...")
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
                "profile", "treatments", "anatomical_terms", "reasoning_keywords"
            ]

            missing_keys = [key for key in required_keys if key not in medical_dict]
            if missing_keys:
                print(f"⚠ Warning: Missing keys: {missing_keys}")

            # Save to file
            output_path = script_dir / f"medical_terms_dictionary_qwen_generated{output_suffix}.json"
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
            raw_path = script_dir / "medical_dict_raw_response.txt"
            with open(raw_path, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"  Raw response saved to: {raw_path}")
            return None

    except json.JSONDecodeError as e:
        print(f"✗ Error parsing JSON: {e}")
        print(f"  Attempted to parse from char {start_idx} to {end_idx}")
        print(f"  First 500 chars: {json_str[:500]}...")
        # Save raw response for debugging
        raw_path = script_dir / "medical_dict_raw_response.txt"
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(response)
        print(f"  Raw response saved to: {raw_path}")
        return None


if __name__ == "__main__":
    import sys

    # Get random seed and suffix from command line args
    random_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    output_suffix = sys.argv[2] if len(sys.argv) > 2 else ""

    print("="*60)
    print("Generating Medical Terms Dictionary with Qwen3-Next-80B-A3B-Instruct")
    print(f"Source: problem_texts.jsonl (30 random samples, seed={random_seed})")
    print("="*60 + "\n")

    result = generate_medical_dictionary(random_seed=random_seed, output_suffix=output_suffix)

    if result:
        print("\n✓ Generation completed successfully!")
    else:
        print("\n✗ Generation failed. Please check the raw response file.")
