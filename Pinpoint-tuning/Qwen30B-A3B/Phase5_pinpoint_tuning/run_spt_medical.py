# 共通コードは shared/phase5/run_spt_medical.py に統合済み
# Dense/MoE両対応の統合版を使用
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'phase5'))
from run_spt_medical import main  # noqa: E402, F401

if __name__ == '__main__':
    main()
