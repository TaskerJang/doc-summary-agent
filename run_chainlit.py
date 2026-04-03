# run_chainlit.py (프로젝트 루트에 생성)
import sys
from pathlib import Path

# 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from ui.app import *  # noqa: F401, F403