#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Model Evaluation Script for Medical QA (Fixed version)
Evaluates model accuracy on gynecology guideline dataset
'''

import argparse
import json
import re
from typing import List, Dict, Tuple, Set
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def extract_answer(text: str) -> Set[str]:
    """Extract answer(s) from model output.

    Looks for patterns like:
    - ['a', 'b', 'c']
    - ['a' 'b' 'c']
    - \\boxed{a}
    - Final answer: a
    - Answer: a

    Returns a set of answer letters.
    """
    # Pattern 1: Array format ['a', 'b'] or ['a' 'b']
    array_match = re.search(r'\[([^\]]+)\]', text)
    if array_match:
        content = array_match.group(1)
        # Extract all letters a-e from the array
        letters = re.findall(r'\b([a-e])\b', content, re.IGNORECASE)
        if letters:
            return set(l.lower() for l in letters)

    # Pattern 2: \boxed{...}
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        content = boxed_match.group(1).strip().lower()
        letters = re.findall(r'\b([a-e])\b', content, re.IGNORECASE)
        if letters:
            return set(l.lower() for l in letters)

    # Pattern 3: "Final answer:" or "Answer:" or "答え:" or "正解:"
    answer_patterns = [
        r'final\s+answer\s*[:：]\s*([a-e])',
        r'answer\s*[:：]\s*([a-e])',
        r'答え\s*[:：]\s*([a-e])',
        r'正解\s*[:：]\s*([a-e])',
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {match.group(1).strip().lower()}

    # Pattern 4: Just look for single letter a-e at the end
    last_letter_match = re.search(r'\b([a-e])\b(?!.*\b[a-e]\b)', text, re.IGNORECASE)
    if last_letter_match:
        return {last_letter_match.group(1).strip().lower()}

    return set()


def evaluate_model(
    model_path: str,
    data_path: str,
    output_path: str = None,
    max_samples: int = None,
    batch_size: int = 1
) -> Dict[str, float]:
    """Evaluate model on medical QA dataset.

    Parameters
    ----------
    model_path : str
        Path to model directory or checkpoint
    data_path : str
        Path to parquet file with questions
    output_path : str, optional
        Path to save detailed results
    max_samples : int, optional
        Maximum number of samples to evaluate
    batch_size : int, optional
        Batch size for inference

    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy metrics
    """
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
        attn_implementation="eager"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    if max_samples is not None:
        df = df.head(max_samples)

    print(f"Evaluating on {len(df)} samples...")

    correct = 0
    total = 0
    results = []

    model.eval()
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            # Extract question
            prompt_msgs = row['prompt']
            if len(prompt_msgs) == 0:
                continue

            question = prompt_msgs[0]['content']

            # Extract ground truth
            if 'reward_model' in row and isinstance(row['reward_model'], dict):
                ground_truth = row['reward_model'].get('ground_truth', [])

                # Parse ground truth - could be a list, tuple, ndarray, string representation of list, or single value
                if isinstance(ground_truth, (list, tuple, np.ndarray)):
                    gt_answers = set(str(x).strip().lower() for x in ground_truth)
                elif isinstance(ground_truth, str):
                    # Check if it's a string representation of a list like "['a', 'b']"
                    list_match = re.findall(r"'([a-e])'", ground_truth, re.IGNORECASE)
                    if list_match:
                        gt_answers = set(x.lower() for x in list_match)
                    else:
                        # Single letter
                        letter_match = re.search(r'\b([a-e])\b', ground_truth, re.IGNORECASE)
                        if letter_match:
                            gt_answers = {letter_match.group(1).lower()}
                        else:
                            continue
                else:
                    continue
            else:
                continue

            # Prepare messages
            messages = [
                {"role": "user", "content": question}
            ]

            # Tokenize
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            # Generate (using Qwen3 recommended settings for thinking mode)
            # Reference: https://huggingface.co/Qwen/Qwen3-14B README.md
            # Thinking mode: Temperature=0.6, TopP=0.95, TopK=20, MaxNewTokens=32768
            # DO NOT use greedy decoding (do_sample=False)
            outputs = model.generate(
                input_ids,
                max_new_tokens=32768,  # Qwen3 recommended for most queries
                do_sample=True,  # Must use sampling (NOT greedy decoding)
                temperature=0.6,  # Qwen3 recommended for thinking mode
                top_p=0.95,  # Qwen3 recommended for thinking mode
                top_k=20,  # Qwen3 recommended for thinking mode
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # Clear GPU cache after generation
            torch.cuda.empty_cache()

            # Decode
            generated_text = tokenizer.decode(
                outputs[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )

            # Extract answer
            predicted_answers = extract_answer(generated_text)

            # Check correctness - predicted answers should overlap with ground truth
            is_correct = len(predicted_answers & gt_answers) > 0 if predicted_answers else False
            if is_correct:
                correct += 1
            total += 1

            # Store result
            result = {
                'index': int(idx),
                'question': question[:200] + '...' if len(question) > 200 else question,
                'ground_truth': sorted(list(gt_answers)),
                'predicted': sorted(list(predicted_answers)),
                'correct': is_correct,
                'generated_text': generated_text[:500] + '...' if len(generated_text) > 500 else generated_text
            }
            results.append(result)

            # Print progress
            if (total % 10) == 0:
                current_acc = 100.0 * correct / total
                print(f"Progress: {total} samples, Accuracy: {current_acc:.2f}%")

    # Calculate metrics
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    metrics = {
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy
    }

    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"  Total samples: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"{'='*50}\n")

    # Save detailed results
    if output_path:
        output_data = {
            'model_path': model_path,
            'data_path': data_path,
            'metrics': metrics,
            'results': results
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on medical QA')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model directory')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to parquet file')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save results JSON')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
