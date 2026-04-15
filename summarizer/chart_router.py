"""
summarizer/chart_router.py
LLM이 반환한 chart_spec(JSON)을 Plotly Figure로 변환한다.

chat2plot 패턴 — 에이전트·코드 실행 없이 LLM JSON spec만 사용.
실패 / 수치 2개 이하 / chart_type=none → None 반환 (graceful fallback)

지원 chart_type:
  - bar       : 항목 간 크기 비교. 양수→파랑, 음수→빨강
  - line      : 시계열 추이
  - pie       : 구성 비율 (donut 스타일)
  - waterfall : 단계별 누적·차감 흐름 (매출→영업이익→순이익 등)

색상 정책 (MIT Sloan 금융 시각화 기준):
  - 양수 / 상승 → POSITIVE_COLOR(파랑)
  - 음수 / 하락 → NEGATIVE_COLOR(빨강)
  - waterfall 합계 bar → TOTAL_COLOR(회색)
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

ChartSpec = dict[str, Any]

# ── 색상 상수 ─────────────────────────────────────────────
POSITIVE_COLOR = "#4C78A8"  # 파랑 — 양수 / 상승
NEGATIVE_COLOR = "#E45756"  # 빨강 — 음수 / 하락
TOTAL_COLOR    = "#72B7B2"  # 청록 — waterfall 합계 bar

# plotly는 선택적 의존성 — 미설치 시 graceful skip
try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    go = None  # type: ignore
    _PLOTLY_AVAILABLE = False
    logger.warning("plotly 미설치 — 시각화 비활성화 (pip install plotly)")


def route(spec: ChartSpec | None) -> "go.Figure | None":
    """
    chart_spec dict를 받아 Plotly Figure를 반환한다.

    반환값이 None인 경우 → UI에서 텍스트 요약만 표시 (graceful fallback).

    Args:
        spec: LLM이 반환한 chart_spec dict.
              None이거나 chart_type="none"이면 None 반환.

    Returns:
        go.Figure 또는 None
    """
    if not _PLOTLY_AVAILABLE or not spec:
        return None

    try:
        chart_type = str(spec.get("chart_type", "none")).lower().strip()
        if chart_type == "none":
            return None

        labels = spec.get("labels") or []
        values = spec.get("values") or []
        title  = str(spec.get("title", ""))
        unit   = str(spec.get("unit", ""))

        # 타입 검증 — labels/values가 리스트인지 확인
        if not isinstance(labels, list) or not isinstance(values, list):
            logger.warning("chart_spec labels/values 타입 오류 — fallback")
            return None

        # 수치 2개 이하 → 차트 불필요
        if len(values) < 3:
            return None

        # labels·values 길이 불일치 → 짧은 쪽 기준으로 trim
        min_len = min(len(labels), len(values))
        labels  = labels[:min_len]
        values  = values[:min_len]

        # 수치형 변환
        try:
            values = [float(v) for v in values]
        except (TypeError, ValueError):
            logger.warning("chart_spec values 수치 변환 실패 — fallback")
            return None

        if chart_type == "line":
            return _make_line(labels, values, title, unit)
        elif chart_type == "bar":
            return _make_bar(labels, values, title, unit)
        elif chart_type == "pie":
            return _make_pie(labels, values, title)
        elif chart_type == "waterfall":
            return _make_waterfall(labels, values, title, unit)
        else:
            logger.warning("알 수 없는 chart_type=%r — fallback", chart_type)
            return None

    except Exception as e:
        logger.warning("차트 생성 실패 (graceful fallback): %s", e)
        return None


# ── 차트 생성 헬퍼 ────────────────────────────────────────

def _bar_colors(values: list[float]) -> list[str]:
    """
    values의 부호에 따라 bar별 색상 리스트를 반환한다.
    양수 → POSITIVE_COLOR(파랑), 음수 → NEGATIVE_COLOR(빨강).
    모든 값이 양수이면 단색 리스트를 반환해 Plotly 최적화를 유지한다.
    """
    if all(v >= 0 for v in values):
        return [POSITIVE_COLOR] * len(values)
    return [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in values]


def _make_line(labels: list, values: list[float], title: str, unit: str) -> "go.Figure":
    fig = go.Figure(
        go.Scatter(
            x=labels,
            y=values,
            mode="lines+markers",
            marker=dict(size=7, color=POSITIVE_COLOR),
            line=dict(width=2, color=POSITIVE_COLOR),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis_title=unit,
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
    )
    return fig


def _make_bar(labels: list, values: list[float], title: str, unit: str) -> "go.Figure":
    colors = _bar_colors(values)
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
        )
    )
    # 음수가 있으면 y=0 기준선을 명시적으로 표시
    has_negative = any(v < 0 for v in values)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis_title=unit,
        yaxis=dict(zeroline=has_negative, zerolinewidth=1.5, zerolinecolor="#888888"),
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
    )
    return fig


def _make_pie(labels: list, values: list[float], title: str) -> "go.Figure":
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.3,  # donut 스타일 — 금융 문서 비율 가독성 향상
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
    )
    return fig


def _make_waterfall(
    labels: list, values: list[float], title: str, unit: str
) -> "go.Figure":
    """
    단계별 누적·차감 흐름 차트 (Zebra BI / pagination.com 금융 보고서 표준).
    마지막 항목을 자동으로 "합계(total)" bar로 처리한다.

    measure 규칙:
      - 마지막 label에 "합계", "순이익", "net", "total" 중 하나가 포함되면 → "total"
      - 나머지는 값의 부호에 따라 "relative" (Plotly waterfall 기본 동작)
    """
    TOTAL_KEYWORDS = {"합계", "순이익", "net profit", "net income", "total", "ebitda"}

    measures = []
    for i, label in enumerate(labels):
        label_lower = str(label).lower()
        is_last = (i == len(labels) - 1)
        if is_last and any(kw in label_lower for kw in TOTAL_KEYWORDS):
            measures.append("total")
        else:
            measures.append("relative")

    # 색상: relative 양수→파랑, 음수→빨강, total→청록
    increasing  = dict(marker_color=POSITIVE_COLOR)
    decreasing  = dict(marker_color=NEGATIVE_COLOR)
    totals      = dict(marker_color=TOTAL_COLOR)

    fig = go.Figure(
        go.Waterfall(
            x=labels,
            y=values,
            measure=measures,
            increasing=increasing,
            decreasing=decreasing,
            totals=totals,
            connector=dict(line=dict(color="#CCCCCC", width=1, dash="dot")),
            textposition="outside",
            text=[f"{v:+.1f}" if m == "relative" else f"{v:.1f}" for v, m in zip(values, measures)],
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis_title=unit,
        yaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor="#888888"),
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
        height=340,  # waterfall은 connector 때문에 bar보다 20px 여유
        showlegend=False,
    )
    return fig
