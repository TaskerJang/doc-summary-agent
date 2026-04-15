"""
summarizer/chart_router.py
LLM이 반환한 chart_spec(JSON)을 Plotly Figure로 변환한다.

chat2plot 패턴 — 에이전트·코드 실행 없이 LLM JSON spec만 사용.
실패 / 수치 2개 이하 / chart_type=none → None 반환 (graceful fallback)
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

ChartSpec = dict[str, Any]

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
        else:
            logger.warning("알 수 없는 chart_type=%r — fallback", chart_type)
            return None

    except Exception as e:
        logger.warning("차트 생성 실패 (graceful fallback): %s", e)
        return None


# ── 차트 생성 헬퍼 ────────────────────────────────────────

def _make_line(labels: list, values: list[float], title: str, unit: str) -> "go.Figure":
    fig = go.Figure(
        go.Scatter(
            x=labels,
            y=values,
            mode="lines+markers",
            marker=dict(size=7),
            line=dict(width=2),
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
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color="steelblue",
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
