#!/usr/bin/env python3
"""
Generate expanded medical_terms_dictionary using Qwen3-Next-80B-A3B-Instruct
- CLI引数で seeds と num_samples を指定可能
- 各カテゴリ40語以上を要求
"""

import argparse
import json
import random
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_random_problem_texts(jsonl_path: str, num_samples: int = 30, random_seed: int = 42):
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    random.seed(random_seed)
    sampled = random.sample(samples, min(num_samples, len(samples)))
    texts = []
    for item in sampled:
        problem_text = item.get('problem_text', '')
        problem_id = item.get('problem_id', '')
        if problem_text:
            texts.append(f"【{problem_id}】\n{problem_text}")
    return texts


def generate_dictionary(model, tokenizer, context_text: str, output_path: Path):
    prompt = f"""# 目的
あなたは優秀な日本語医療NLPエンジニアです。提供された「医師国家試験問題データ」から、機能分類に特化した**医療用語辞書（JSON形式）**を作成してください。

この辞書は、Path Patching手法を用いた大規模言語モデルの内部挙動解析に使用されます。

## 辞書の構造と抽出定義

以下の7つのカテゴリに分類し、それぞれ日本語の用語をリストとして抽出してください。

1. **diseases**: 疾患名、症候群名、病態名、症状名。例: 糖尿病、心筋梗塞、肺炎、高血圧症、意識障害、発熱、嘔吐、下痢、浮腫、チアノーゼ
2. **diagnostic_methods**: 検査法、診断手順、医療手技、診察所見。例: 心電図、CT、MRI、血液検査、打診、聴診、触診、内視鏡検査、超音波検査
3. **biomarkers**: 血液、体液、組織などで測定される指標、検査値。例: 血糖値、クレアチニン、GOT、GPT、白血球数、CRP、HbA1c、BNP
4. **profile**: 患者の属性を示す語句。例: 20歳、男性、既往歴なし、喫煙歴あり、身長170cm、体重60kg
5. **treatments**: 薬剤名、手術法、治療法、処置。例: インスリン、抗菌薬、胃洗浄、輸液、手術、放射線療法、化学療法、ステロイド
6. **anatomical_terms**: 器官名、組織名、解剖学的部位。例: 心臓、肺、肝臓、腎臓、脳、胸部、腹部、子宮、膵臓
7. **reasoning_keywords**: 推論過程を示すキーワード、論理接続詞、問題の指示語。例: したがって、なぜなら、まず、次に、考えられる、正しいのは、誤っているのは

## 出力形式の制約

- 出力は、JSON形式のテキストブロック一つのみ
- 各リストには**40〜60個**の用語を含めること（**60個を超えないこと**）
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

上記のデータに基づいて、医療用語辞書をJSON形式で出力してください。各カテゴリ40〜60語（60語以内厳守）を含めてください。JSONのみを出力してください。"""

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print(f"  Input tokens: {model_inputs.input_ids.shape[1]}", flush=True)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Extract JSON
    start_idx = response.find('{')
    end_idx = response.rfind('}') + 1
    if start_idx == -1 or end_idx <= start_idx:
        print("  ERROR: Could not find JSON in response", flush=True)
        raw_path = output_path.with_suffix('.raw.txt')
        raw_path.write_text(response, encoding='utf-8')
        print(f"  Raw response saved to: {raw_path}", flush=True)
        return None

    json_str = response[start_idx:end_idx]
    try:
        medical_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  ERROR: JSON parse failed: {e}", flush=True)
        raw_path = output_path.with_suffix('.raw.txt')
        raw_path.write_text(response, encoding='utf-8')
        return None

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(medical_dict, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in medical_dict.values() if isinstance(v, list))
    print(f"  Saved: {output_path} ({total} terms)", flush=True)
    for cat, terms in medical_dict.items():
        if isinstance(terms, list):
            print(f"    {cat}: {len(terms)}", flush=True)
    return medical_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="List of random seeds")
    parser.add_argument("--num_samples", type=int, default=30,
                        help="Number of samples per seed")
    args = parser.parse_args()

    seeds = args.seeds
    num_samples = args.num_samples

    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    jsonl_path = parent_dir / "annotated_medical_data_full.jsonl"

    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found!")
        sys.exit(1)

    model_path = "/home/yuuki.nakamura/downloads/models/Qwen_Qwen3-Next-80B-A3B-Instruct"
    print("=" * 60, flush=True)
    print("Expanded Medical Dictionary Generator", flush=True)
    print(f"  Seeds: {seeds}", flush=True)
    print(f"  Samples per run: {num_samples}", flush=True)
    print(f"  Model: {model_path}", flush=True)
    print("=" * 60, flush=True)

    print(f"\nLoading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True
    )
    print("Model loaded.\n", flush=True)

    for seed in seeds:
        print(f"\n{'='*40}", flush=True)
        print(f"Generating with seed={seed}, num_samples={num_samples}", flush=True)
        print(f"{'='*40}", flush=True)

        sample_texts = load_random_problem_texts(str(jsonl_path), num_samples=num_samples, random_seed=seed)
        context_text = "\n\n---サンプル---\n\n".join(sample_texts)
        print(f"  Loaded {len(sample_texts)} problem texts", flush=True)

        output_path = script_dir / f"dict_seed{seed}.json"
        generate_dictionary(model, tokenizer, context_text, output_path)

    print(f"\n{'='*60}", flush=True)
    print("All dictionary generation complete!", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
