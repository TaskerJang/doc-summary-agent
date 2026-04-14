"""
임베딩 + Qdrant 벡터 DB 레이어

역할:
- 청크 텍스트를 OpenAI text-embedding-3-small로 임베딩
- Qdrant 로컬 인스턴스에 저장/검색
- 세션별 컬렉션 격리 (collection_name = doc_id)

의존:
    uv add qdrant-client sentence-transformers
    docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import TYPE_CHECKING

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_QDRANT_URL   = os.getenv("QDRANT_URL", "http://localhost:6333")
_EMBED_MODEL  = "text-embedding-3-small"
_EMBED_DIM    = 1536
_BATCH_SIZE   = 64

_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_qdrant_client: QdrantClient | None = None


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=_QDRANT_URL, timeout=10)
    return _qdrant_client


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """OpenAI text-embedding-3-small 로 배치 임베딩."""
    vectors = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        resp  = _openai_client.embeddings.create(model=_EMBED_MODEL, input=batch)
        vectors.extend([d.embedding for d in resp.data])
    return vectors


def _collection_name(doc_id: str) -> str:
    """doc_id -> Qdrant 컬렉션 이름 (영숫자+언더스코어만 허용)."""
    h = hashlib.md5(doc_id.encode()).hexdigest()[:8]
    return f"doc_{h}"


def index_chunks(chunks: list[str], doc_id: str) -> str:
    """
    청크 리스트를 임베딩해 Qdrant에 저장.
    이미 동일 컬렉션이 존재하면 삭제 후 재생성 (문서 갱신 대응).

    Returns:
        컬렉션 이름
    """
    client     = _get_qdrant()
    col_name   = _collection_name(doc_id)
    existing   = {c.name for c in client.get_collections().collections}

    if col_name in existing:
        client.delete_collection(col_name)
        logger.info("기존 컬렉션 삭제: %s", col_name)

    client.create_collection(
        collection_name=col_name,
        vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
    )
    logger.info("컬렉션 생성: %s (%d청크)", col_name, len(chunks))

    vectors = _embed_texts(chunks)
    points  = [
        PointStruct(id=i, vector=v, payload={"text": t})
        for i, (t, v) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=col_name, points=points)
    logger.info("벡터 저장 완료: %s", col_name)
    return col_name


def search_chunks(question: str, doc_id: str, top_k: int = 5) -> list[str]:
    """
    Dense 검색: 질문 임베딩 → Qdrant cosine 유사도 검색.

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

        q_vec   = _embed_texts([question])[0]
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
