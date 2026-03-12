"""
MCQ形式のSFTデータを生成する。
GRPO評価と同じ出力形式（[ans]x[/ans]）で、system prompt付きmessagesを出力。
"""
import os
import json
import re
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

import hydra
from omegaconf import DictConfig


# ============================================================
# SDG-Nexus YAML準拠: システムプロンプト定義
# ============================================================

SYSTEM_PROMPT = (
    "あなたは医師国家試験を受験している医学生です。"
    "与えられた問題に対して、まず日本語で考察を行い、"
    "最後に正解の選択肢を [ans][/ans] タグで囲んで回答してください。"
)

def build_prompt(problem_text: str, choices_text: str) -> str:
    """GRPO用データと同じプロンプト"""
    return f"""次の多肢選択問題について、日本語で考察したあと、
最後の1行で正しい選択肢を [ans][/ans] で囲んで答えてください。

問題:
{problem_text}

選択肢:
{choices_text}

出力フォーマット例:

ここに日本語で考察を書く。

[ans]a,c[/ans]
"""

# ============ [ans]タグ抽出 ============

def extract_ans_tag(text: str) -> str | None:
    m = re.search(r"\[ans\](.*?)\[/ans\]", text, re.DOTALL)
    return m.group(1).strip().lower() if m else None

# ============ 新しい生成関数（1ステップ） ============

async def generate_mcq_answer(client, item, model_name, cfg):
    prompt = build_prompt(item["problem_text"], item["choices_text"])
    try:
        resp = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=cfg.get("temperature", 0.3),
            top_p=cfg.get("top_p", 0.95),
            max_completion_tokens=cfg.get("max_completion_tokens", 4096),
            stream=False,
        )
        content = resp.choices[0].message.content
        if content is None or resp.choices[0].finish_reason == "length":
            return None, None
        usage = json.loads(resp.usage.model_dump_json())
        return content, usage
    except Exception as e:
        print(f"Generate Error [{item.get('problem_id')}]: {e}")
        return None, None

# ============ 新しい品質検証 ============

def validate_answer(answer_text: str, gold_answer: str) -> tuple[bool, str | None]:
    if not answer_text or len(answer_text) < 20:
        return False, None
    extracted = extract_ans_tag(answer_text)
    if extracted is None:
        return False, None
    gold_norm = ",".join(sorted(gold_answer.strip().lower().split(",")))
    pred_norm = ",".join(sorted(extracted.split(",")))
    return (pred_norm == gold_norm), extracted


# ============================================================
# データ読み込み・シャード
# ============================================================

def load_input(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def shard_data(items: list, shard_index: int, num_shards: int) -> list:
    return [x for i, x in enumerate(items) if i % num_shards == shard_index]


# ============================================================
# 既存結果の管理（中断再開対応）
# ============================================================

def load_existing(path: str) -> dict:
    result = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    pid = rec.get("problem_id")
                    if pid:
                        result[pid] = rec
    return result


def append_jsonl(path: str, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# 2ステップ統合
# ============================================================

async def process_one(client, item, cfg):
    model = cfg.model_name
    max_retries = cfg.get("max_retries_per_item", 3)
    prompt = build_prompt(item["problem_text"], item["choices_text"])

    for attempt in range(max_retries):
        answer_text, usage = await generate_mcq_answer(client, item, model, cfg)
        if answer_text is None:
            continue

        is_correct, extracted = validate_answer(answer_text, item["answer"])
        if is_correct:
            return {
                "problem_id": item["problem_id"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},  # ← 追加
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer_text},
                ],
                "solution": item["answer"],
                "extracted_answer": extracted,
                "gold_answer": item["answer"],
                "year": item.get("year"),
                "category": item.get("category", ""),
                "usage": usage,
                "attempts": attempt + 1,
            }

    print(f"  除外（{max_retries}回不正解）: {item.get('problem_id')}")
    return None

async def generate_all(
    client: AsyncOpenAI,
    items: list[dict],
    cfg: DictConfig,
    output_path: str,
    existing: dict,
):
    sem = asyncio.Semaphore(cfg.num_workers)

    async def bounded(item):
        async with sem:
            return await process_one(client, item, cfg)

    todo = [x for x in items if x["problem_id"] not in existing]
    print(f"  生成対象: {len(todo)} 件 (スキップ: {len(items) - len(todo)} 件)")

    if not todo:
        print("  全件処理済み。")
        return

    tasks = [bounded(item) for item in todo]
    results = await tqdm_asyncio.gather(*tasks)

    ok, ng = 0, 0
    for result in results:
        if result is not None:
            append_jsonl(output_path, result)
            ok += 1
        else:
            ng += 1

    print(f"  生成成功: {ok} 件 / 失敗・除外: {ng} 件")


# ============================================================
# メイン
# ============================================================

@hydra.main(config_path="conf", config_name="config_sft", version_base=None)
def main(cfg: DictConfig):
    shard_index = cfg.get("shard_index", 0)
    num_shards = cfg.get("num_shards", 1)

    base_output = cfg.output_path
    if num_shards > 1:
        stem = Path(base_output).stem
        suffix = Path(base_output).suffix
        parent = Path(base_output).parent
        output_path = str(parent / f"{stem}_shard{shard_index}{suffix}")
    else:
        output_path = base_output

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print("=== SFT DataGen (2-Step Pipeline) ===")
    print(f"  Model: {cfg.model_name}")
    print(f"  Input: {cfg.input_path}")
    print(f"  Output: {output_path}")
    print(f"  Shard: {shard_index}/{num_shards}")
    print(f"  Workers: {cfg.num_workers}")

    items = load_input(cfg.input_path)
    print(f"  全データ: {len(items)} 件")

    if num_shards > 1:
        items = shard_data(items, shard_index, num_shards)
        print(f"  シャード後: {len(items)} 件")

    existing = load_existing(output_path)
    if existing:
        print(f"  既存結果: {len(existing)} 件")

    client = AsyncOpenAI(
        base_url=cfg.base_url,
        api_key=cfg.get("api_key", "dummy"),
        timeout=600,
        max_retries=3,
    )

    asyncio.run(generate_all(client, items, cfg, output_path, existing))
    print("完了!")


if __name__ == "__main__":
    main()
