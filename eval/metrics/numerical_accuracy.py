"""
eval/metrics/numerical_accuracy.py
수치 정확도 — 금융 특화
한국어 단위(억원, 조원, %p, bp, %, 원) 포함 수치 추출 후 일치 여부 확인
"""
import re


# 한국어 금융 수치 패턴
# 예: 1,606억원 / 9조 3,132억원 / +3.3% / -19.7% / 80,000원 / 3.0~3.5개월
NUM_PATTERN = re.compile(
    r'[\+\-]?[\d,]+\.?\d*'
    r'(?:\s*(?:조|억|만)?\s*(?:원|달러|엔|위안))?'
    r'(?:\s*[%％](?:p|P)?)?'
    r'(?:\s*bp)?'
    r'(?:\s*개월)?'
)


def extract_numbers(text: str) -> list[str]:
    """텍스트에서 금융 수치를 추출하여 정규화한 리스트 반환."""
    raw = NUM_PATTERN.findall(text)
    # 공백 제거 + 콤마 제거 후 정규화
    normalized = []
    for n in raw:
        n = n.strip().replace(',', '').replace(' ', '')
        if n and n not in ('.', '-', '+'):
            normalized.append(n)
    return normalized


def compute_numerical_accuracy(prediction: str, reference: str) -> dict:
    """
    정답 수치 중 예측에 포함된 비율을 계산한다.
    Args:
        prediction: 모델 생성 답변
        reference:  정답 레퍼런스
    Returns:
        {'accuracy': float, 'matched': list, 'missed': list, 'total': int}
    """
    ref_nums  = extract_numbers(reference)
    pred_nums = extract_numbers(prediction)

    if not ref_nums:
        return {'accuracy': 1.0, 'matched': [], 'missed': [], 'total': 0}

    matched = [n for n in ref_nums if n in pred_nums]
    missed  = [n for n in ref_nums if n not in pred_nums]

    return {
        'accuracy': round(len(matched) / len(ref_nums), 4),
        'matched':  matched,
        'missed':   missed,
        'total':    len(ref_nums),
    }


if __name__ == "__main__":
    pred = "1분기 영업이익은 1606억원이며 OPM은 7.4%입니다."
    ref  = "1,606억원(-19.7%, OPM 7.4%)"
    print(compute_numerical_accuracy(pred, ref))
