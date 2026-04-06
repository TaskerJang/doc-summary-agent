"""
eval/metrics/bert_score.py
BERTScore — 의미적 유사도 측정
모델: snunlp/KR-ELECTRA-discriminator (한국어 특화)
"""
from bert_score import score as bert_score_fn


def compute_bert_score(predictions: list[str], references: list[str]) -> dict:
    """
    Args:
        predictions: 모델 생성 답변 리스트
        references:  정답 레퍼런스 리스트
    Returns:
        {'precision': float, 'recall': float, 'f1': float}
    """
    P, R, F = bert_score_fn(
        predictions,
        references,
        model_type="snunlp/KR-ELECTRA-discriminator",
        lang="ko",
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
