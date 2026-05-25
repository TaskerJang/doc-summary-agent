"""eval/metrics/semantic_similarity.py
RAGAS Answer Semantic Similarity — 임베딩 cosine similarity.

doc-summary 측은 summarizer.embedder 의 _get_encoder() 재사용 가능하지만,
평가 메트릭 패키지는 독립적이어야 함 — 자체 sentence-transformers 사용.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _encoder = SentenceTransformer("BAAI/bge-m3")
            logger.info("semantic_similarity: bge-m3 로딩 완료")
        except ImportError as e:
            logger.error(
                "sentence-transformers 미설치 — 'uv add sentence-transformers' 필요: %s", e,
            )
            raise
    return _encoder


def compute_semantic_similarity(prediction: str, reference: str) -> dict:
    if not prediction or not reference:
        return {"semantic_similarity": 0.0}
    try:
        encoder = _get_encoder()
        embs = encoder.encode([prediction, reference], normalize_embeddings=True, show_progress_bar=False)
        sim = float(np.dot(embs[0], embs[1]))
        sim = max(0.0, min(1.0, sim))
        return {"semantic_similarity": round(sim, 4)}
    except Exception as e:
        logger.error("semantic_similarity 계산 실패: %s", e)
        return {"semantic_similarity": None, "error": str(e)}
