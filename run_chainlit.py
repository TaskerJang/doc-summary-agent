# run_chainlit.py
import sys
import json
import logging
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import chainlit.data as cl_data
from chainlit.data import BaseDataLayer
from chainlit.types import Feedback

logger = logging.getLogger(__name__)

FEEDBACK_FILE = Path("./feedback_log.jsonl")


class FeedbackOnlyDataLayer(BaseDataLayer):
    """
    #69 Human Feedback 전용 미니 data layer.
    upsert_feedback()만 구현, 나머지는 no-op.
    피드백은 feedback_log.jsonl로 저장.
    """

    async def upsert_feedback(self, feedback: Feedback) -> str:
        entry = {
            "id":       feedback.id,
            "forId":    feedback.forId,
            "threadId": getattr(feedback, "threadId", None),
            "value":    feedback.value,
            "comment":  getattr(feedback, "comment", None),
        }
        emoji = "👍" if feedback.value == 1 else "👎"
        logger.info(
            "[Feedback] %s value=%s forId=%s comment=%r",
            emoji, feedback.value, feedback.forId,
            getattr(feedback, "comment", None),
        )
        try:
            with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("[Feedback] 파일 저장 실패: %s", e)
        return feedback.id or ""

    # 필수 추상 메서드 스튱 (no-op)
    async def get_user(self, identifier: str): return None
    async def create_user(self, user): return None
    async def update_thread(self, thread_id, name=None, user_id=None,
                            metadata=None, tags=None): pass
    async def get_thread(self, thread_id): return None
    async def get_thread_author(self, thread_id): return ""
    async def delete_thread(self, thread_id): pass
    async def list_threads(self, pagination, filters): return []
    async def get_element(self, thread_id, element_id): return None
    async def create_element(self, element): pass
    async def delete_element(self, element_id, thread_id=None): pass
    async def create_step(self, step_dict): pass
    async def update_step(self, step_dict): pass
    async def delete_step(self, step_id): pass
    async def get_all_user_threads(self, user_id=None, thread_id=None): return None
    async def delete_feedback(self, feedback_id): return True
    # Chainlit 2.11 추가 추상 메서드
    async def build_debug_url(self) -> str: return ""
    async def close(self): pass
    async def get_favorite_steps(self): return []


cl_data._data_layer = FeedbackOnlyDataLayer()

from ui.app import *  # noqa: F401, F403
