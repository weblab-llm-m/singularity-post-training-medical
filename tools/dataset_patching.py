import ast
import json
import re
import os
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv

# envファイル読み込み
load_dotenv()

def clean_choice(raw):
    """
    文字列形式のリスト（例: "['a', 'b']"）を実際のリストに変換し、
    \u3000（全角スペース）を削除する
    """
    # 文字列がリスト表現の場合
    if isinstance(raw, str) and raw.strip().startswith('['):
        try:
            parsed = ast.literal_eval(raw)
            return [item.replace('\u3000', '') for item in parsed]
        except:
            return raw
    
    return raw

def build_prompt(problem_text: str, choices) -> str:
    # choicesが文字列（リスト表現）なら変換
    processed_choices = clean_choice(choices)
    
    if isinstance(processed_choices, list):
        choices_text = '\n'.join([f"{chr(97+i)}. {c}" for i, c in enumerate(processed_choices)])
    else:
        choices_text = str(processed_choices)

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

# 追加: 「◯つ選べ」から必要な解答数を取るヘルパ
def get_required_answer_count(problem_text: str):
    """
    問題文中の「2つ選べ」「２つ選べ」「二つ選べ」などから
    必要な解答数を推定する。
    見つからなければ None を返す。
    """

    text = problem_text

    # 全角数字 → 半角数字に変換しておく
    z2h = str.maketrans("０１２３４５６７８９", "0123456789")
    text_norm = text.translate(z2h)

    # 1) 数字ベース: 「2つ選」「2 つ選」「2つ選べ」「2つ選びなさい」など
    m = re.search(r'([0-9]+)\s*つ\s*選', text_norm)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass

    # 2) 漢数字ベース: 「二つ選べ」など（1〜5くらいだけ対応）
    kanji_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5}
    m2 = re.search(r'([一二三四五])\s*つ\s*選', text)
    if m2:
        ch = m2.group(1)
        return kanji_map.get(ch)

    # 取れなかった場合
    return None

def convert_orig_to_ms_swift_rl(
    dataset,
    out_path: str = "rl_ophtho.jsonl",
) -> None:
    out_path = Path(out_path)

    with out_path.open("w", encoding="utf-8") as fout:
        for ex in dataset:  # ファイル読み込みからdatasetイテレーションに変更
            # もし 2023 と 2024 両方除外したい場合は ex.get("year") in [2023, 2024, "2023", "2024"] とします
            if str(ex.get("year")) in [2023, 2024, "2023", "2024"]:
                continue

            if not ex["text_only"]:
                continue
            
            if not len(ex["answer"]) > 0:
                continue

            required = get_required_answer_count(ex["problem_text"])
            if required is not None and len(ex["answer"]) != required:
                # Debug用
                # print(f"skip mismatch: id={ex.get('problem_id')}, required={required}, got={len(ex['answer'])}")
                continue

            prompt = build_prompt(ex["problem_text"], ex["choices"])

            # リストを文字列に変換
            answer_list = clean_choice(ex["answer"])
            if isinstance(answer_list, list):
                solution_str = ",".join(answer_list)  # ["a","c"] → "a,c"
            else:
                solution_str = str(answer_list)
            
            item = {
                "problem_id": ex.get("problem_id"),
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "solution": solution_str,  # リストではなく文字列にする
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            

def convert_hf_to_ms_swift_rl(
    dataset_name: str,
    split: str = "train",
    out_path: str = "rl_ophtho.jsonl",
) -> None:
    """
    Hugging Faceからデータセットをダウンロードして前処理を行う
    
    Args:
        dataset_name: Hugging Faceのデータセット名 (例: "username/dataset-name")
        split: データセットのsplit名 (例: "train", "test")
        out_path: 出力ファイルのパス
        token: Hugging Faceのアクセストークン (Noneの場合は環境変数から読み込む)
        use_auth_token: 認証トークンを使用するかどうか
    """
    
    # アクセストークンの読み込み
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token is None:
        print("警告: アクセストークンが見つかりません。環境変数 HF_TOKEN または HUGGING_FACE_HUB_TOKEN を設定してください。")
        print("または、引数 token= で直接指定してください。")
    
    # Hugging Faceにログイン
    if token:
        login(token=token)
        print("Hugging Faceにログインしました")
    
    # データセットの読み込み
    print(f"データセット '{dataset_name}' の '{split}' splitを読み込み中...")
    dataset = load_dataset(dataset_name, split=split, token=token)
    print(f"読み込み完了: {len(dataset)} サンプル")
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    convert_orig_to_ms_swift_rl(dataset,out_path)

# 2023-2024を省く必要がある
if __name__ == "__main__":    
    convert_hf_to_ms_swift_rl(
        dataset_name="weblab-LLM-M/igakuqa-2001-2024-filtered",
        split="train",
        out_path=os.path.expandvars("$HOME/downloads/datasets/igakuqa.jsonl"),
    )
    