# run_chainlit.py
import sys
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))

import chainlit.data as cl_data
from chainlit.data import BaseDataLayer
from chainlit.types import Feedback, PaginatedResponse, PageInfo
from chainlit.user import PersistedUser

logger = logging.getLogger(__name__)

FEEDBACK_FILE = Path("./feedback_log.jsonl")

# ── #69 Chainlit 프론트엔드 패치 ─────────────────────────────────────────────
def _patch_frontend() -> None:
    import os, glob
    cl_dir = os.path.dirname(__import__("chainlit").__file__)
    fe_dir = os.path.join(cl_dir, "frontend", "dist", "assets")
    files  = glob.glob(os.path.join(fe_dir, "index-*.js"))
    if not files:
        return
    target = files[0]
    MARKER = "/* feedback-patch-applied */"
    with open(target, encoding="utf-8") as f:
        content = f.read()
    if MARKER in content:
        return
    if "f=kn(O6)" not in content:
        print("[patch] 패치 대상 패턴을 찾지 못함")
        return
    with open(target, "w", encoding="utf-8") as f:
        f.write(content.replace("f=kn(O6)", f"f=true{MARKER}"))
    print(f"[patch] 피드백 버튼 패치 완료: {os.path.basename(target)}")


_patch_frontend()


# ── 하이브리드 DataLayer ──────────────────────────────────────────────────────
# - upsert_feedback: SQLite에 저장 (피드백 수집 목적)
# - get_user / create_user: PersistedUser 반환 (인증 통과)
# - create_step / update_step / update_thread 등: 즉시 반환 (no-op)
#   → SQLAlchemyDataLayer의 동기 DB I/O가 이벤트 루프를 막는 문제 해결
# - list_threads / get_all_user_threads: 사이드바 히스토리 미지원 (빈 응답)
# ────────────────────────────────────────────────────────────────────────────

async def _init_feedback_db() -> None:
    """feedbacks 테이블만 SQLite에 생성."""
    import aiosqlite
    async with aiosqlite.connect("./feedback.db") as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS feedbacks (
                "id"       TEXT PRIMARY KEY,
                "forId"    TEXT NOT NULL,
                "threadId" TEXT,
                "value"    INTEGER NOT NULL,
                "comment"  TEXT,
                "createdAt" TEXT
            )
        """)
        await db.commit()


asyncio.run(_init_feedback_db())


class HybridDataLayer(BaseDataLayer):
    """
    피드백만 SQLite에 저장하고 나머지는 no-op인 경량 data layer.

    SQLAlchemyDataLayer는 메시지 전송마다 create_step/update_thread를
    await하여 이벤트 루프를 블로킹하는 문제가 있음.
    이 레이어는 step/thread I/O를 전혀 하지 않으므로 블로킹 없음.
    피드백 버튼 활성화는 프론트엔드 JS 패치(_patch_frontend)로 해결.
    """

    async def get_user(self, identifier: str):
        return PersistedUser(
            id=identifier,
            identifier=identifier,
            createdAt=datetime.now(timezone.utc).isoformat(),
        )

    async def create_user(self, user):
        return PersistedUser(
            id=user.identifier,
            identifier=user.identifier,
            createdAt=datetime.now(timezone.utc).isoformat(),
        )

    async def upsert_feedback(self, feedback: Feedback) -> str:
        emoji = "👍" if feedback.value == 1 else "👎"
        comment = getattr(feedback, "comment", None)
        thread_id = getattr(feedback, "threadId", None)
        logger.info(
            "[Feedback] %s value=%s forId=%s comment=%r",
            emoji, feedback.value, feedback.forId, comment,
        )
        try:
            import aiosqlite
            async with aiosqlite.connect("./feedback.db") as db:
                await db.execute(
                    'INSERT OR REPLACE INTO feedbacks VALUES (?,?,?,?,?,?)',
                    (
                        feedback.id or feedback.forId,
                        feedback.forId,
                        thread_id,
                        feedback.value,
                        comment,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                await db.commit()
        except Exception as e:
            logger.warning("[Feedback] DB 저장 실패: %s", e)
            # jsonl fallback
            try:
                entry = {"forId": feedback.forId, "value": feedback.value, "comment": comment}
                with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass
        return feedback.id or ""

    # ── no-op 메서드들 (블로킹 없이 즉시 반환) ──
    async def update_thread(self, thread_id, name=None, user_id=None, metadata=None, tags=None): pass
    async def get_thread(self, thread_id): return None
    async def get_thread_author(self, thread_id): return ""
    async def delete_thread(self, thread_id): pass
    async def list_threads(self, pagination, filters):
        return PaginatedResponse(
            pageInfo=PageInfo(hasNextPage=False, startCursor=None, endCursor=None),
            data=[],
        )
    async def get_element(self, thread_id, element_id): return None
    async def create_element(self, element): pass
    async def delete_element(self, element_id, thread_id=None): pass
    async def create_step(self, step_dict): pass
    async def update_step(self, step_dict): pass
    async def delete_step(self, step_id): pass
    async def get_all_user_threads(self, user_id=None, thread_id=None): return None
    async def delete_feedback(self, feedback_id): return True
    async def build_debug_url(self) -> str: return ""
    async def close(self): pass
    async def get_favorite_steps(self): return []


cl_data._data_layer = HybridDataLayer()

from ui.app import *  # noqa: F401, F403
