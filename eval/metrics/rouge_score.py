"""
eval/metrics/rouge_score.py
ROUGE-1 / ROUGE-2 / ROUGE-L 계산
"""
from rouge_score import rouge_scorer


def compute_rouge(prediction: str, reference: str) -> dict:
    """
    Args:
        prediction: 모델 생성 답변
        reference:  정답 레퍼런스
    Returns:
        {'rouge1': f1, 'rouge2': f1, 'rougeL': f1}
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=False,
        tokenizer=None,
    )
    scores = scorer.score(reference, prediction)
    return {
        'rouge1': round(scores['rouge1'].fmeasure, 4),
        'rouge2': round(scores['rouge2'].fmeasure, 4),
        'rougeL': round(scores['rougeL'].fmeasure, 4),
    }


if __name__ == "__main__":
    pred = "두산밥캣의 목표주가는 80,000원이다."
    ref  = "목표주가는 80,000원으로 유지된다."
    print(compute_rouge(pred, ref))
