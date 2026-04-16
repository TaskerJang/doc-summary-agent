# run_chainlit.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── #69 SQLite data layer 초기화 — thumbs up/down UI 활성화 ─────────────────────
# config.toml [data_layer]는 Chainlit 2.11에서 지원 안 됨 — 코드로 직접 주입
import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

cl_data._data_layer = SQLAlchemyDataLayer(
    conninfo="sqlite+aiosqlite:///./chainlit.db",
    show_logger=True,
)

from ui.app import *  # noqa: F401, F403
