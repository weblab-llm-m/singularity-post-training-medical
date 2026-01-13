# chinese_reward_plugin.py
import re
from typing import List
from swift.plugin import ORM, orms


def has_chinese_consecutive(text: str) -> bool:
    """
    ひらがな・カタカナを。で区切った文章で存在しなければTrue・存在すればFalse
    連続で中国語（True）と判定したらTrue
    """
    chinese_judge_list = [not re.search(r'[\u3040-\u30ff]', t) for t in text.split("。") if t]
    return any(a and b for a, b in zip(chinese_judge_list, chinese_judge_list[1:]))


class ChineseRewardFunction(ORM):
    """
    中国語が連続して検出された場合は 0.0（ペナルティ）
    中国語が連続して検出されなければ 1.0
    """

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        return [0.0 if has_chinese_consecutive(comp) else 1.0 for comp in completions]


# ms-swift に "chinese" という名前で登録
orms["chinese"] = ChineseRewardFunction