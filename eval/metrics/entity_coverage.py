"""eval/metrics/entity_coverage.py
RAGAS Context Entities Recall 변형 — 정답 entity 가 예측에 등장하는 비율.

doc-graph-agent 와 동일 메트릭.
"""
import logging
import re
from typing import Optional

from eval.metrics.numerical_accuracy import extract_numbers

logger = logging.getLogger(__name__)


def _extract_named_entities(text: str) -> list[str]:
    pattern = re.compile(r'[\uac00-\ud7a3]{2,10}')
    candidates = pattern.findall(text)
    STOPWORDS = {
        "이곳", "저곥", "그곳", "이것", "저것", "그것",
        "한다", "하는", "되는", "있다", "없다",
        "서말", "대비", "대해", "대한", "따릅", "위한",
        "목표", "결과", "수준", "기준",
        "N/A", "문서", "내용", "없음", "없는",
    }
    filtered = [c for c in candidates if c not in STOPWORDS]
    seen = set()
    result = []
    for c in filtered:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def compute_entity_coverage(
    prediction: str,
    reference: str,
    key_entities: Optional[list[str]] = None,
) -> dict:
    if not prediction or not reference:
        return {"entity_coverage": 0.0, "matched": [], "missed": [], "total": 0}

    if key_entities:
        target_entities = list(key_entities)
    else:
        target_entities = list(set(
            extract_numbers(reference) + _extract_named_entities(reference)
        ))

    if not target_entities:
        return {"entity_coverage": 1.0, "matched": [], "missed": [], "total": 0}

    matched = []
    missed = []
    for e in target_entities:
        e_norm = e.replace(",", "").replace(" ", "")
        pred_norm = prediction.replace(",", "").replace(" ", "")
        if e_norm in pred_norm or e in prediction:
            matched.append(e)
        else:
            missed.append(e)

    return {
        "entity_coverage": round(len(matched) / len(target_entities), 4),
        "matched": matched,
        "missed": missed,
        "total": len(target_entities),
    }
