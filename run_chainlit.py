# run_chainlit.py
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── #69 SQLite data layer 초기화 + 테이블 자동 생성 ───────────────────
import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

DB_URL = "sqlite+aiosqlite:///./chainlit.db"

# Chainlit SQLAlchemy 공식 스키마
# ref: https://github.com/Chainlit/chainlit/blob/main/backend/chainlit/data/sql_alchemy.py
_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS users (
    "id"          TEXT PRIMARY KEY,
    "identifier"  TEXT NOT NULL UNIQUE,
    "metadata"    TEXT NOT NULL DEFAULT '{}',
    "createdAt"   TEXT
);

CREATE TABLE IF NOT EXISTS threads (
    "id"          TEXT PRIMARY KEY,
    "createdAt"   TEXT,
    "name"        TEXT,
    "userId"      TEXT,
    "userIdentifier" TEXT,
    "tags"        TEXT,
    "metadata"    TEXT,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS steps (
    "id"          TEXT PRIMARY KEY,
    "name"        TEXT NOT NULL,
    "type"        TEXT NOT NULL,
    "threadId"    TEXT NOT NULL,
    "parentId"    TEXT,
    "streaming"   INTEGER NOT NULL DEFAULT 0,
    "waitForAnswer" INTEGER DEFAULT 0,
    "isError"     INTEGER DEFAULT 0,
    "metadata"    TEXT,
    "tags"        TEXT,
    "input"       TEXT,
    "output"      TEXT,
    "createdAt"   TEXT,
    "start"       TEXT,
    "end"         TEXT,
    "generation"  TEXT,
    "showInput"   TEXT,
    "language"    TEXT,
    "indent"      INTEGER
);

CREATE TABLE IF NOT EXISTS elements (
    "id"          TEXT PRIMARY KEY,
    "threadId"    TEXT,
    "type"        TEXT,
    "url"         TEXT,
    "chainlitKey" TEXT,
    "name"        TEXT NOT NULL,
    "display"     TEXT,
    "objectKey"   TEXT,
    "size"        TEXT,
    "page"        INTEGER,
    "language"    TEXT,
    "forId"       TEXT,
    "mime"        TEXT
);

CREATE TABLE IF NOT EXISTS feedbacks (
    "id"          TEXT PRIMARY KEY,
    "forId"       TEXT NOT NULL,
    "threadId"    TEXT NOT NULL,
    "value"       INTEGER NOT NULL,
    "comment"     TEXT
);
"""


async def _init_db() -> None:
    """SQLite DB 테이블을 없으면 생성한다 (idempotent)."""
    import aiosqlite
    async with aiosqlite.connect("./chainlit.db") as db:
        await db.executescript(_CREATE_TABLES_SQL)
        await db.commit()


asyncio.run(_init_db())

cl_data._data_layer = SQLAlchemyDataLayer(
    conninfo=DB_URL,
    show_logger=True,
)

from ui.app import *  # noqa: F401, F403
