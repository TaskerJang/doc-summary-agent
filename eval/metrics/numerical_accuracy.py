"""
eval/metrics/numerical_accuracy.py
수치 정확도 — 금융 특화
한국어 단위(억원, 조원, %p, bp, %, 원) 포함 수치 추출 후 일치 여부 확인

참고: FinQA / TAT-QA 수치 정규화 방식
- 부호 포함 비교 (부호도 정보의 일부)
- 콤마·공백 제거 후 정규화
- 날짜 패턴 사전 제외
- 중복 수치 set으로 처리
- ~ 범위 표현 통째로 토큰 처리

TODO: 조+억 복합 단위 정규화 (예: 9조 3,132억원 → 단일 토큰)
"""
import re


# 날짜 패턴 — 수치 추출 전 제거
# 예: 2026.03.16 / 2026년 3월 / 25/01
_DATE_PATTERN = re.compile(
    r'\d{4}\.\d{1,2}(?:\.\d{1,2})?'   # 2026.03 / 2026.03.16
    r'|\d{4}년\s*\d{1,2}월'            # 2026년 3월
    r'|\d{2}/\d{2}'                    # 25/01
)

# ~ 범위 표현 — 수치 추출 전 하나의 토큰으로 보호
# 예: 3.0~3.5개월 → 통째로 매칭
_RANGE_PATTERN = re.compile(
    r'[\+\-]?[\d,]+\.?\d*~[\+\-]?[\d,]+\.?\d*'
    r'(?:\s*(?:조|억|만)?\s*(?:원|달러|엔|위안))?'
    r'(?:\s*[%％](?:p|P)?)?'
    r'(?:\s*bp)?'
    r'(?:\s*개월)?'
)

# 일반 금융 수치 패턴
# 예: 1,606억원 / +3.3% / -19.7% / 80,000원
_NUM_PATTERN = re.compile(
    r'[\+\-]?[\d,]+\.?\d*'
    r'(?:\s*(?:조|억|만)?\s*(?:원|달러|엔|위안))?'
    r'(?:\s*[%％](?:p|P)?)?'
    r'(?:\s*bp)?'
    r'(?:\s*개월)?'
)


def _normalize(token: str) -> str:
    """수치 토큰 정규화: 공백·콤마 제거."""
    return token.strip().replace(',', '').replace(' ', '')


def extract_numbers(text: str) -> list[str]:
    """
    텍스트에서 금융 수치를 추출하여 정규화한 리스트 반환.
    - 날짜 패턴 사전 제거
    - ~ 범위 표현 우선 추출
    - 중복 제거 (set)
    """
    # 1. 날짜 제거
    text = _DATE_PATTERN.sub('', text)

    results = []

    # 2. 범위 표현 우선 추출 및 마스킹
    ranges = _RANGE_PATTERN.findall(text)
    for r in ranges:
        n = _normalize(r)
        if n:
            results.append(n)
    text = _RANGE_PATTERN.sub('', text)

    # 3. 일반 수치 추출
    for n in _NUM_PATTERN.findall(text):
        n = _normalize(n)
        if n and n not in ('.', '-', '+'):
            results.append(n)

    # 4. 중복 제거 (순서 유지)
    seen = set()
    deduped = []
    for n in results:
        if n not in seen:
            seen.add(n)
            deduped.append(n)

    return deduped


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
    pred_nums = set(extract_numbers(prediction))

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
    # 콤마 정규화 테스트
    pred = "1분기 영업이익은 1606억원이며 OPM은 7.4%입니다."
    ref  = "1,606억원(-19.7%, OPM 7.4%)"
    print(compute_numerical_accuracy(pred, ref))
    # → matched: ['1606억원', '7.4%'], missed: ['-19.7%']

    # 범위 표현 테스트
    pred2 = "북미 딜러 재고는 3.0~3.5개월 수준이다."
    ref2  = "3.0~3.5개월"
    print(compute_numerical_accuracy(pred2, ref2))
    # → matched: ['3.0~3.5개월'], missed: []

    # 날짜 제외 테스트
    pred3 = "2026년 3월 기준 영업이익은 1606억원이다."
    ref3  = "1,606억원"
    print(compute_numerical_accuracy(pred3, ref3))
    # → matched: ['1606억원'], missed: []