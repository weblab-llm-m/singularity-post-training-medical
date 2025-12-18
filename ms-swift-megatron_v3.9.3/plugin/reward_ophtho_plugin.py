# plugin.py
import re
from typing import Iterable, List, Optional, Set

from swift.plugin import ORM, orms

# [ans]...[/ans] を抜き出す正規表現
ANS_PATTERN = re.compile(r"\[ans\](.*?)\[/ans\]", re.IGNORECASE | re.DOTALL)


def parse_output(output: str) -> Optional[str]:
    """
    モデル出力のうち、最後の [ans]...[/ans] の「中身」だけを取り出す。
    ans_text が見つからない場合は None を返す。
    """
    matches = list(ANS_PATTERN.finditer(output))
    if not matches:
        return None
    last = matches[-1]
    ans_text = last.group(1).strip()
    return ans_text or None


def parse_ans_set(ans_text: Optional[str]) -> Optional[Set[str]]:
    """
    "a,c" や "a, c" を {"a","c"} に変換。
    ans_text が None のときは None を返す。
    """
    if ans_text is None:
        return None
    text = ans_text.strip()
    if not text:
        return None

    # 基本はカンマ区切り
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        # 念のためスペース区切りも許す
        parts = [p.strip() for p in text.split() if p.strip()]
    if not parts:
        return None
    return set(parts)


def answer_reward(pred: Optional[Set[str]], gold: Set[str]) -> float:
    """
    完全一致で 1.0、それ以外は 0.0。
    """
    if pred is None:
        return 0.0
    return 1.0 if pred == gold else 0.0


def total_reward_one(completion: str, gold_set: Set[str]) -> float:
    """
    1サンプル分のトータル reward（ここでは正解判定のみ）。
    """
    ans_text = parse_output(completion)
    pred_set = parse_ans_set(ans_text)
    return answer_reward(pred_set, gold_set)


class OphthoRewardFunction(ORM):
    """
    ms-swift の GRPO 用 reward 関数。
    dataset 側に "solution" カラムを持たせておく前提。
    """

    def __call__(self, completions: Iterable[str], solution: Iterable, **kwargs) -> List[float]:
        """
        completions: モデル生成テキストのリスト
        solution: dataset の "solution" カラム
                - [["a"], ["a","c"], ...] のような List[List[str]] を想定。
                - 文字列で来てもある程度は頑張ってパース。
        """
        rewards: List[float] = []

        for comp, gold in zip(completions, solution):
            # gold が ["a","c"] / "a,c" / "a" など何で来ても頑張ってセットにする
            if isinstance(gold, (list, tuple)):
                gold_set = {str(x).strip() for x in gold if str(x).strip()}
            else:
                text = str(gold)
                parts = [p.strip() for p in text.split(",") if p.strip()]
                if not parts:
                    parts = [p.strip() for p in text.split() if p.strip()]
                gold_set = set(parts)

            rewards.append(total_reward_one(comp, gold_set))

        return rewards


# ms-swift に "ophtho" という名前で登録
orms["ophtho"] = OphthoRewardFunction