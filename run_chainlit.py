# run_chainlit.py
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── #69 SQLite data layer 초기화 + 테이블 자동 생성 ───────────────────
# SQLAlchemyDataLayer는 테이블 자동 생성 안 함 — create_all()로 선행 생성
import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

DB_URL = "sqlite+aiosqlite:///./chainlit.db"


async def _init_db() -> None:
    """SQLite DB 테이블을 없으면 생성한다 (idempotent)."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from chainlit.data.sql_alchemy import Base  # Chainlit 정의 메타데이터
    engine = create_async_engine(DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()


asyncio.run(_init_db())

cl_data._data_layer = SQLAlchemyDataLayer(
    conninfo=DB_URL,
    show_logger=True,
)

from ui.app import *  # noqa: F401, F403
