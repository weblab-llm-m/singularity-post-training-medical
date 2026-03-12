# 共通コードは shared/phase3/head_classifier.py に統合済み
# Dense/MoE両対応、num_layers/num_headsはコンストラクタで指定
# indicator_type='profile' で Qwen30B の profile_indicator_heads を使用
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'phase3'))
from head_classifier import *  # noqa: E402, F401, F403
