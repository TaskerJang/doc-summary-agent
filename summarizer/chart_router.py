"""
summarizer/chart_router.py
LLM이 반환한 chart_spec(JSON)을 Plotly Figure로 변환한다.

chat2plot 패턴 — 에이전트·코드 실행 없이 LLM JSON spec만 사용.
실패 / 수치 2개 이하 / chart_type=none → None 반환 (graceful fallback)

지원 chart_type:
  - bar        : 항목 간 크기 비교. 양수→파랑, 음수→빨강
  - line       : 단일 시리즈 시계열 추이
  - multiline  : 복수 시리즈 동시 비교 (지수·펀드 벤치마크 등)
  - pie        : 구성 비율 (donut 스타일)
  - waterfall  : 단계별 누적·차감 흐름 (매출→영업이익→순이익 등)

색상 정책 (MIT Sloan 금융 시각화 기준):
  - 양수 / 상승 → POSITIVE_COLOR(파랑)
  - 음수 / 하락 → NEGATIVE_COLOR(빨강)
  - waterfall 합계 bar → TOTAL_COLOR(청록)
  - multiline 시리즈 → MULTILINE_PALETTE 순환

크기 정책:
  - autosize=False + width/height 명시로 컨테이너 autosize 방지
  - Chainlit 인라인 컨테이너 max-width=600px 기준, width=480으로 여백 확보
  - height는 차트 유형별로 차등 적용
  - 근거: Chainlit 2.11.0 #2861 — Figure height 반영. autosize=True(기본값)시
    컨테이너 너비에 맞게 늘어나 비율이 깨지는 문제를 autosize=False로 방지.

bar 두께 정책:
  - bargap=0.4 — 항목 수가 적을 때 bar가 과도하게 넓어지는 Plotly 기본 동작 방지
  - bargap 범위: 0(간격 없음) ~ 1(bar 없음), 0.4가 금융 보고서 표준 비율
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

ChartSpec = dict[str, Any]

# ── 색상 상수 ─────────────────────────────────────────────
POSITIVE_COLOR   = "#4C78A8"  # 파랑 — 양수 / 상승
NEGATIVE_COLOR   = "#E45756"  # 빨강 — 음수 / 하락
TOTAL_COLOR      = "#72B7B2"  # 청록 — waterfall 합계 bar

# multiline 시리즈 팔레트 — Flourish 금융 시각화 가이드 기반, 최대 5개 시리즈 보장
MULTILINE_PALETTE = [
    "#4C78A8",  # 파랑
    "#F58518",  # 주황
    "#54A24B",  # 초록
    "#B279A2",  # 보라
    "#E45756",  # 빨강
]

# ── 크기 상수 ─────────────────────────────────────────────
CHART_WIDTH      = 480
HEIGHT_DEFAULT   = 260  # bar / line / pie
HEIGHT_WATERFALL = 300  # bar 위 텍스트 공간 필요
HEIGHT_MULTILINE = 320  # 상단 legend 공간 필요

# ── bar 두께 상수 ─────────────────────────────────────────
# 0(간격 없음) ~ 1(bar 없음). 0.4 = bar 너비 60% 수준
BAR_GAP = 0.4

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

    multiline 스키마 (chart_type="multiline"):
      labels  : 공통 x축 레이블 리스트
      series  : [{"name": "코스피", "values": [...]}, {"name": "코스닥", "values": [...]}]
      values  : 단일 시리즈일 때만 사용 (하위 호환)

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
        series = spec.get("series") or []   # multiline 전용
        title  = str(spec.get("title", ""))
        unit   = str(spec.get("unit", ""))

        # multiline은 별도 경로로 분기 (values 대신 series 사용)
        if chart_type == "multiline":
            if not isinstance(labels, list) or not isinstance(series, list) or not series:
                logger.warning("multiline: labels 또는 series 누락 — fallback")
                return None
            return _make_multiline(labels, series, title, unit)

        # 이하 단일 시리즈 공통 검증
        if not isinstance(labels, list) or not isinstance(values, list):
            logger.warning("chart_spec labels/values 타입 오류 — fallback")
            return None

        if len(values) < 3:
            return None

        min_len = min(len(labels), len(values))
        labels  = labels[:min_len]
        values  = values[:min_len]

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
        autosize=False,
        width=CHART_WIDTH,
        height=HEIGHT_DEFAULT,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _make_multiline(
    labels: list, series: list[dict], title: str, unit: str
) -> "go.Figure":
    MAX_SERIES = 5
    traces = []

    for i, s in enumerate(series[:MAX_SERIES]):
        name   = str(s.get("name", f"시리즈{i+1}"))
        vals   = s.get("values") or []
        color  = MULTILINE_PALETTE[i % len(MULTILINE_PALETTE)]

        min_len = min(len(labels), len(vals))
        x = labels[:min_len]
        try:
            y = [float(v) for v in vals[:min_len]]
        except (TypeError, ValueError):
            logger.warning("multiline 시리즈 '%s' values 변환 실패 — 건너뜀", name)
            continue

        if len(y) < 3:
            continue

        traces.append(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                mode="lines+markers",
                marker=dict(size=6, color=color),
                line=dict(width=2, color=color),
            )
        )

    if not traces:
        logger.warning("multiline: 유효한 시리즈 없음 — fallback")
        return None

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis_title=unit,
        template="plotly_white",
        autosize=False,
        width=CHART_WIDTH,
        height=HEIGHT_MULTILINE,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=40, r=20, t=70, b=40),
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
    has_negative = any(v < 0 for v in values)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis_title=unit,
        yaxis=dict(zeroline=has_negative, zerolinewidth=1.5, zerolinecolor="#888888"),
        template="plotly_white",
        autosize=False,
        width=CHART_WIDTH,
        height=HEIGHT_DEFAULT,
        margin=dict(l=40, r=20, t=50, b=40),
        bargap=BAR_GAP,
    )
    return fig


def _make_pie(labels: list, values: list[float], title: str) -> "go.Figure":
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        autosize=False,
        width=CHART_WIDTH,
        height=HEIGHT_DEFAULT,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _make_waterfall(
    labels: list, values: list[float], title: str, unit: str
) -> "go.Figure":
    TOTAL_KEYWORDS = {"합계", "순이익", "net profit", "net income", "total", "ebitda"}

    measures = []
    for i, label in enumerate(labels):
        label_lower = str(label).lower()
        is_last = (i == len(labels) - 1)
        if is_last and any(kw in label_lower for kw in TOTAL_KEYWORDS):
            measures.append("total")
        else:
            measures.append("relative")

    increasing = dict(marker_color=POSITIVE_COLOR)
    decreasing = dict(marker_color=NEGATIVE_COLOR)
    totals     = dict(marker_color=TOTAL_COLOR)

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
        autosize=False,
        width=CHART_WIDTH,
        height=HEIGHT_WATERFALL,
        margin=dict(l=40, r=20, t=50, b=40),
        bargap=BAR_GAP,
        showlegend=False,
    )
    return fig
