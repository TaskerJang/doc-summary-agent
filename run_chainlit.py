# run_chainlit.py
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

DB_URL = "sqlite+aiosqlite:///./chainlit.db"

# Chainlit 2.11 SQLAlchemy 스키마 — get_all_user_threads 소스 기준
_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS users (
    "id"          TEXT PRIMARY KEY,
    "identifier"  TEXT NOT NULL UNIQUE,
    "metadata"    TEXT NOT NULL DEFAULT '{}',
    "createdAt"   TEXT
);

CREATE TABLE IF NOT EXISTS threads (
    "id"             TEXT PRIMARY KEY,
    "createdAt"      TEXT,
    "name"           TEXT,
    "userId"         TEXT,
    "userIdentifier" TEXT,
    "tags"           TEXT,
    "metadata"       TEXT,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS steps (
    "id"            TEXT PRIMARY KEY,
    "name"          TEXT NOT NULL,
    "type"          TEXT NOT NULL,
    "threadId"      TEXT NOT NULL,
    "parentId"      TEXT,
    "streaming"     INTEGER NOT NULL DEFAULT 0,
    "waitForAnswer" INTEGER DEFAULT 0,
    "isError"       INTEGER DEFAULT 0,
    "metadata"      TEXT,
    "tags"          TEXT,
    "input"         TEXT,
    "output"        TEXT,
    "createdAt"     TEXT,
    "start"         TEXT,
    "end"           TEXT,
    "generation"    TEXT,
    "showInput"     TEXT,
    "language"      TEXT,
    "indent"        INTEGER,
    "defaultOpen"   INTEGER DEFAULT 0,
    "autoCollapse"  INTEGER DEFAULT 0
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
    "mime"        TEXT,
    "props"       TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS feedbacks (
    "id"       TEXT PRIMARY KEY,
    "forId"    TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "value"    INTEGER NOT NULL,
    "comment"  TEXT
);
"""

# 기존 DB 마이그레이션 — 누락 컬럼 추가 (idempotent)
_MIGRATIONS = [
    'ALTER TABLE steps ADD COLUMN "defaultOpen" INTEGER DEFAULT 0',
    'ALTER TABLE steps ADD COLUMN "autoCollapse" INTEGER DEFAULT 0',
    'ALTER TABLE elements ADD COLUMN "props" TEXT DEFAULT \'{}\'',
]


async def _init_db() -> None:
    """DB 테이블 생성 + 마이그레이션 (idempotent)."""
    import aiosqlite
    async with aiosqlite.connect("./chainlit.db") as db:
        await db.executescript(_CREATE_TABLES_SQL)
        for stmt in _MIGRATIONS:
            try:
                await db.execute(stmt)
            except Exception:
                pass  # 이미 존재하면 무시
        await db.commit()


asyncio.run(_init_db())

cl_data._data_layer = SQLAlchemyDataLayer(
    conninfo=DB_URL,
    show_logger=True,
)

from ui.app import *  # noqa: F401, F403
