# 共通コードは shared/phase2/dataset.py に統合済み
# 既存のimportパスとの互換性を維持するための re-export
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'phase2'))
from dataset import *  # noqa: E402, F401, F403
