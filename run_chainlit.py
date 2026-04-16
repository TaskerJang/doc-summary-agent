# run_chainlit.py
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

DB_URL = "sqlite+aiosqlite:///./chainlit.db"

# ── #69 Chainlit 프론트엔드 패치 ─────────────────────────────────────────────
# Chainlit 2.11 프론트엔드는 FirstUserInteraction이 항상 undefined라
# 새 대화에서 thumbs up/down 이 항상 disabled 됨.
# 해결: index.js 번들에서 f=kn(O6) → f=true 로 패치
def _patch_frontend() -> None:
    import os
    import glob

    cl_dir = os.path.dirname(__import__("chainlit").__file__)
    fe_dir = os.path.join(cl_dir, "frontend", "dist", "assets")
    pattern = os.path.join(fe_dir, "index-*.js")
    files = glob.glob(pattern)
    if not files:
        print("[patch] index-*.js 파일을 찾지 못함")
        return

    target = files[0]
    MARKER = "/* feedback-patch-applied */"

    with open(target, encoding="utf-8") as f:
        content = f.read()

    if MARKER in content:
        return  # 이미 패치 적용됨

    if "f=kn(O6)" not in content:
        print("[patch] 패치 대상 패턴을 찾지 못함 — Chainlit 버전 확인 필요")
        return

    patched = content.replace("f=kn(O6)", f"f=true{MARKER}")
    with open(target, "w", encoding="utf-8") as f:
        f.write(patched)
    print(f"[patch] 피드백 버튼 패치 완료: {os.path.basename(target)}")


_patch_frontend()

# ── Chainlit 2.11 SQLAlchemy 스키마 ─────────────────────────────────────────────
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

_MIGRATIONS = [
    'ALTER TABLE steps ADD COLUMN "defaultOpen" INTEGER DEFAULT 0',
    'ALTER TABLE steps ADD COLUMN "autoCollapse" INTEGER DEFAULT 0',
    'ALTER TABLE elements ADD COLUMN "props" TEXT DEFAULT \'{}\'',
]


async def _init_db() -> None:
    import aiosqlite
    async with aiosqlite.connect("./chainlit.db") as db:
        await db.executescript(_CREATE_TABLES_SQL)
        for stmt in _MIGRATIONS:
            try:
                await db.execute(stmt)
            except Exception:
                pass
        await db.commit()


asyncio.run(_init_db())

cl_data._data_layer = SQLAlchemyDataLayer(
    conninfo=DB_URL,
    show_logger=True,
)

from ui.app import *  # noqa: F401, F403
