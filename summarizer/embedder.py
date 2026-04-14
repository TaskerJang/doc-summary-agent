"""
임베딩 + Qdrant 벡터 DB 레이어

역할:
- 청크 텍스트를 BAAI/bge-m3로 로컬 임베딩 (한국어 금융 문서 최적화)
- Qdrant 로컬 인스턴스에 저장/검색
- 세션별 컬렉션 격리 (collection_name = doc_id 해시)

bge-m3 선택 이유:
- 다국어 SOTA 임베딩 모델 (한국어 성능 우수, MIRACL 1위)
- sparse + dense + multi-vector 동시 지원 (향후 full hybrid 확장 가능)
- 로컬 실행으로 API 비용 없음, MIT 라이선스
- dim=1024, 최대 8192 토큰, 568M 파라미터

bge prefix 규칙 (공식 권고):
- 문서 인덱싱: "passage: {text}"
- 쿼리 검색:   "query: {text}"
  → prefix 없이 사용하면 검색 품질 저하됨

의존:
    uv add qdrant-client sentence-transformers
    docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
"""
from __future__ import annotations

import hashlib
import logging
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)

_QDRANT_URL  = os.getenv("QDRANT_URL", "http://localhost:6333")
_EMBED_MODEL = "BAAI/bge-m3"
_EMBED_DIM   = 1024
_BATCH_SIZE  = 32  # bge-m3는 모델이 커서 배치 작게

# bge 계열 prefix (공식 권고)
_DOC_PREFIX   = "passage: "
_QUERY_PREFIX = "query: "

_encoder: object | None = None
_qdrant_client: QdrantClient | None = None


def _get_encoder():
    """bge-m3 인코더 싱글턴 — 첫 호출 시 모델 로드 (~2GB, 최초 1회 다운로드)."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("bge-m3 모델 로딩 중... (최초 1회 다운로드 필요)")
        _encoder = SentenceTransformer(_EMBED_MODEL)
        logger.info("bge-m3 로딩 완료")
    return _encoder


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=_QDRANT_URL, timeout=10)
    return _qdrant_client


def _embed_documents(texts: list[str]) -> list[list[float]]:
    """문서 청크 임베딩 — 'passage:' prefix 적용."""
    encoder  = _get_encoder()
    prefixed = [_DOC_PREFIX + t for t in texts]
    vectors  = []
    for i in range(0, len(prefixed), _BATCH_SIZE):
        batch = prefixed[i: i + _BATCH_SIZE]
        embs  = encoder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        vectors.extend(embs.tolist())
    return vectors


def _embed_query(text: str) -> list[float]:
    """쿼리 임베딩 — 'query:' prefix 적용."""
    encoder = _get_encoder()
    emb = encoder.encode(
        _QUERY_PREFIX + text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return emb.tolist()


def _collection_name(doc_id: str) -> str:
    """doc_id -> Qdrant 컬렉션 이름 (영숫자+언더스코어만 허용)."""
    h = hashlib.md5(doc_id.encode()).hexdigest()[:8]
    return f"doc_{h}"


def index_chunks(chunks: list[str], doc_id: str) -> str:
    """
    청크 리스트를 bge-m3로 임베딩해 Qdrant에 저장.
    이미 동일 컬렉션이 존재하면 삭제 후 재생성 (문서 갱신 대응).

    Returns:
        컬렉션 이름
    """
    client   = _get_qdrant()
    col_name = _collection_name(doc_id)
    existing = {c.name for c in client.get_collections().collections}

    if col_name in existing:
        client.delete_collection(col_name)
        logger.info("기존 컬렉션 삭제: %s", col_name)

    client.create_collection(
        collection_name=col_name,
        vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
    )
    logger.info("컬렉션 생성: %s (%d청크)", col_name, len(chunks))

    vectors = _embed_documents(chunks)
    points  = [
        PointStruct(id=i, vector=v, payload={"text": t})
        for i, (t, v) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=col_name, points=points)
    logger.info("벡터 저장 완료: %s", col_name)
    return col_name


def search_chunks(question: str, doc_id: str, top_k: int = 5) -> list[str]:
    """
    Dense 검색: 질문 bge-m3 임베딩 → Qdrant cosine 유사도 검색.
    쿼리에 'query:' prefix 적용 (bge 공식 권고).

    Returns:
        유사도 높은 청크 텍스트 리스트 (최대 top_k개)
        Qdrant 미연결 또는 컬렉션 없으면 빈 리스트
    """
    try:
        client   = _get_qdrant()
        col_name = _collection_name(doc_id)
        existing = {c.name for c in client.get_collections().collections}
        if col_name not in existing:
            logger.warning("컬렉션 없음 — dense 검색 스킵: %s", col_name)
            return []

        q_vec   = _embed_query(question)
        results = client.query_points(
            collection_name=col_name,
            query=q_vec,
            limit=top_k,
            with_payload=True,
        ).points
        return [r.payload["text"] for r in results if r.payload]
    except Exception as e:
        logger.warning("Dense 검색 실패 (BM25 fallback): %s", e)
        return []


def collection_exists(doc_id: str) -> bool:
    """해당 문서의 Qdrant 컬렉션이 존재하는지 확인."""
    try:
        client   = _get_qdrant()
        col_name = _collection_name(doc_id)
        return col_name in {c.name for c in client.get_collections().collections}
    except Exception:
        return False
