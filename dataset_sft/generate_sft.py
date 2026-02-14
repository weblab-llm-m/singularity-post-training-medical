"""
vLLMサーバーに対して2ステップパイプラインでSFTデータを生成する。

SDG-Nexus YAML (medical_qa_synthesis.yaml) と同等のフロー:
  Step 1: MCQ → 自然な質問に変換  (temp=0.0, JSON出力)
  Step 2: 自然な質問 → 構造化医療回答  (temp=0.3)

auto_eval_simple の predict_provider/vllm_pred.py を参考にした構造。

使い方:
  python generate_sft.py
  python generate_sft.py +shard_index=0 +num_shards=8
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

# Block 2: 質問生成（YAML generate_question と同一）
QUESTION_GEN_SYSTEM = """あなたは医学領域のデータ合成者です。
医学の選択式問題を、ユーザーが日常的にしそうな自然な質問に変換してください。

【ルール】
- 選択肢の列挙形式（A/B/C/D等）は使わない
- 正答を質問文に含めない
- 医学的テーマは保持する
- 入力と同じ言語で出力する

【出力形式】
{"generated_question": "変換後の質問"}"""

# Block 4: 回答生成（YAML generate_answer と同一）
ANSWER_GEN_SYSTEM = """あなたは医療情報を提供するアシスタントです。

【基本方針】
- 医師ではない。診断確定や個別治療指示はしない
- 安全最優先。緊急性が高い場合は受診を促す
- 不確実な点は明示する

【回答構造】
1) 要点（3行以内）
2) 考えられる原因（3〜5つ）
3) 自分でできる対応
4) 受診の目安
5) 追加確認質問（最大3つ）

最後に「医療機関での相談が必要な場合があります」と添える。"""


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
# Block 1: MCQフォーマット（YAML format_mcq と同一）
# ============================================================

def format_mcq_text(item: dict) -> str:
    lines = [
        "【Problem】",
        item["problem_text"].strip(),
        "",
        "【Choices】",
    ]
    for choice_line in item["choices_text"].strip().split("\n"):
        lines.append(choice_line.strip())
    lines.extend(["", "【Answer】", item["answer"].strip()])
    return "\n".join(lines)


# ============================================================
# Block 2+3: Step 1 — MCQ → 自然な質問（temp=0.0）
# ============================================================

async def step1_generate_question(
    client: AsyncOpenAI,
    item: dict,
    model_name: str,
) -> str | None:
    mcq_text = format_mcq_text(item)

    try:
        resp = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": QUESTION_GEN_SYSTEM},
                {"role": "user", "content": f"MCQ:\n{mcq_text}"},
            ],
            temperature=0.0,
            max_completion_tokens=2048,
            stream=False,
        )
        raw = resp.choices[0].message.content
        if raw is None:
            return None

        # JSON抽出（YAML extract_question と同一ロジック）
        try:
            parsed = json.loads(raw)
            return parsed.get("generated_question", raw)
        except json.JSONDecodeError:
            m = re.search(r'\{[^}]*"generated_question"\s*:\s*"([^"]+)"', raw)
            if m:
                return m.group(1)
            return raw.strip()

    except Exception as e:
        print(f"Step1 Error [{item.get('problem_id')}]: {e}")
        return None


# ============================================================
# Block 4: Step 2 — 自然な質問 → 構造化医療回答（temp=0.3）
# ============================================================

async def step2_generate_answer(
    client: AsyncOpenAI,
    question: str,
    model_name: str,
    cfg: DictConfig,
) -> tuple[str | None, dict | None]:
    try:
        resp = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": ANSWER_GEN_SYSTEM},
                {"role": "user", "content": question},
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
        print(f"Step2 Error: {e}")
        return None, None


# ============================================================
# 品質検証
# ============================================================

def validate_answer(answer_text: str) -> bool:
    """回答構造の最低品質チェック"""
    if not answer_text or len(answer_text) < 100:
        return False
    markers = ["要点", "原因", "対応", "受診", "医療機関"]
    found = sum(1 for m in markers if m in answer_text)
    return found >= 2


# ============================================================
# 2ステップ統合
# ============================================================

async def process_one(
    client: AsyncOpenAI,
    item: dict,
    cfg: DictConfig,
) -> dict | None:
    model = cfg.model_name

    # Step 1
    question = await step1_generate_question(client, item, model)
    if question is None:
        return None

    # Step 2
    answer, usage = await step2_generate_answer(client, question, model, cfg)
    if answer is None:
        return None

    # 品質検証
    if not validate_answer(answer):
        print(f"  品質不足で除外: {item.get('problem_id')}")
        return None

    # Block 5: 最終出力（YAML final と同一フィールド）
    return {
        "problem_id": item["problem_id"],
        "original_mcq": format_mcq_text(item),
        "question": question,
        "answer": answer,
        "gold_answer": item["answer"],
        "reasoning_effort": "high",
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "year": item.get("year"),
        "category": item.get("category", ""),
        "usage": usage,
    }


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
    print(f"  Step1: MCQ → 自然質問 (temp=0.0)")
    print(f"  Step2: 質問 → 構造化回答 (temp={cfg.get('temperature', 0.3)})")
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