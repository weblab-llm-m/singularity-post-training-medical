# 共通コードは shared/phase3/medical_pattern_detector.py に統合済み
# 既存のimportパスとの互換性を維持するための re-export
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'phase3'))
from medical_pattern_detector import *  # noqa: E402, F401, F403
