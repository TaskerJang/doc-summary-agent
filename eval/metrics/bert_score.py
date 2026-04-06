"""
eval/metrics/bert_score.py
BERTScore — 의미적 유사도 측정
모델: snunlp/KR-ELECTRA-discriminator (한국어 특화)

Note:
    최초 실행 시 HuggingFace에서 모델이 자동 다운로드됩니다.
    (snunlp/KR-ELECTRA-discriminator, 약 440MB)
"""
import logging

from bert_score import score as bert_score_fn

logger = logging.getLogger(__name__)


def compute_bert_score(predictions: list[str], references: list[str]) -> dict:
    """
    Args:
        predictions: 모델 생성 답변 리스트
        references:  정답 레퍼런스 리스트
    Returns:
        {'precision': float, 'recall': float, 'f1': float}
        입력이 비어 있으면 모든 값 0.0 반환
    """
    if not predictions or not references:
        logger.warning("compute_bert_score: 빈 입력, 0.0 반환")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    P, R, F = bert_score_fn(
        predictions,
        references,
        model_type="snunlp/KR-ELECTRA-discriminator",
        verbose=False,
    )
    return {
        'precision': round(P.mean().item(), 4),
        'recall':    round(R.mean().item(), 4),
        'f1':        round(F.mean().item(), 4),
    }


if __name__ == "__main__":
    preds = ["두산밥캣의 목표주가는 80,000원이다."]
    refs  = ["목표주가는 80,000원으로 유지된다."]
    print(compute_bert_score(preds, refs))