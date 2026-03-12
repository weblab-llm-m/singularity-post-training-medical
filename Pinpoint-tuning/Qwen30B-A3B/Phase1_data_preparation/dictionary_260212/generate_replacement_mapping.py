#!/usr/bin/env python3
"""
Generate replacement_mapping.json for expanded dictionary
using Qwen3-Next-80B-A3B-Instruct
"""

import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_replacement_mapping():
    script_dir = Path(__file__).parent
    dict_path = script_dir / "medical_terms_dictionary.json"

    if not dict_path.exists():
        print(f"ERROR: {dict_path} not found! Run merge_all_dicts.py first.")
        return None

    with open(dict_path, 'r', encoding='utf-8') as f:
        medical_dict = json.load(f)

    total_terms = sum(len(terms) for terms in medical_dict.values())
    print(f"Medical dictionary: {total_terms} terms")

    # 既存のマッピングをロード（差分のみ生成するため）
    parent_dir = script_dir.parent
    old_mapping_path = parent_dir / "replacement_mapping.json"
    old_mapping = {}
    if old_mapping_path.exists():
        with open(old_mapping_path) as f:
            old_mapping = json.load(f)
        print(f"Existing mapping: {len(old_mapping)} entries (will be reused)")

    # 新規用語のみ抽出
    new_terms_by_cat = {}
    for cat, terms in medical_dict.items():
        if cat == "reasoning_keywords":
            continue
        new_in_cat = [t for t in terms if t not in old_mapping]
        if new_in_cat:
            new_terms_by_cat[cat] = new_in_cat

    new_total = sum(len(v) for v in new_terms_by_cat.values())
    print(f"New terms to map: {new_total}")

    if new_total == 0:
        print("No new terms to map. Copying existing mapping.")
        output = dict(old_mapping)
        out_path = script_dir / "replacement_mapping.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Saved: {out_path} ({len(output)} entries)")
        return output

    # Load model
    model_path = "/home/yuuki.nakamura/downloads/models/Qwen_Qwen3-Next-80B-A3B-Instruct"
    print(f"\nLoading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True
    )
    print("Model loaded.\n")

    new_terms_json = json.dumps(new_terms_by_cat, ensure_ascii=False, indent=2)

    prompt = f"""# 目的
医療用語を一般的な表現に変換するマッピング辞書を作成してください。

## 変換ルール
- **diseases（疾患名）** → 「健康状態」「体調」「状態」など
- **diagnostic_methods（診断方法）** → 「検査」「確認」「調査」など
- **biomarkers（バイオマーカー）** → 「検査値」「数値」「測定値」など
- **profile（患者属性）** → 年齢→「年齢」、性別→「性別」、既往歴→「生活背景」、妊娠→「妊娠状態」、身体情報→「身体情報」
- **treatments（治療法）** → 「治療」「対処」「処置」など
- **anatomical_terms（解剖学用語）** → 「部位」「箇所」「器官」など

## 出力形式
キーが医療用語、値が一般語のJSONのみ出力:
```json
{{"糖尿病": "健康状態", "心電図": "検査", "20歳": "年齢"}}
```

## 変換対象の用語

{new_terms_json}

上記の全ての用語に対して、マッピング辞書をJSON形式で出力してください。JSONのみを出力してください。"""

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print(f"Input tokens: {model_inputs.input_ids.shape[1]}")
    print("Generating...")

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=8192,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            top_k=20,
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    start_idx = response.find('{')
    end_idx = response.rfind('}') + 1
    if start_idx == -1 or end_idx <= start_idx:
        print("ERROR: Could not find JSON in response")
        raw_path = script_dir / "replacement_mapping_raw.txt"
        raw_path.write_text(response, encoding='utf-8')
        return None

    try:
        new_mapping = json.loads(response[start_idx:end_idx])
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON parse failed: {e}")
        raw_path = script_dir / "replacement_mapping_raw.txt"
        raw_path.write_text(response, encoding='utf-8')
        return None

    # Remove nulls
    new_mapping = {k: v for k, v in new_mapping.items() if v is not None}
    print(f"New mappings generated: {len(new_mapping)}")

    # Merge with old
    combined = dict(old_mapping)
    combined.update(new_mapping)

    out_path = script_dir / "replacement_mapping.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"  Old: {len(old_mapping)}, New: {len(new_mapping)}, Combined: {len(combined)}")
    return combined


if __name__ == "__main__":
    print("=" * 60)
    print("Generating Expanded Replacement Mapping")
    print("=" * 60 + "\n")
    result = generate_replacement_mapping()
    if result:
        print(f"\nDone! Total mappings: {len(result)}")
    else:
        print("\nFailed. Check raw response file.")
