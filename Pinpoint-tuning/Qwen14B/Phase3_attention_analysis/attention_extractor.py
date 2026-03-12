# 共通コードは shared/phase3/attention_extractor.py に統合済み
# Dense/MoE両対応、num_layers/num_headsはコンストラクタで指定
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'phase3'))
from attention_extractor import *  # noqa: E402, F401, F403
