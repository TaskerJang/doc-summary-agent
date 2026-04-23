// public/elements/SourceReference.jsx
//
// Q&A 답변의 출처 리스트 — 호버 미리보기만 제공 (Dialog 모달 제거).
//
// UX:
//   - 근거 텍스트 호버: HoverCard로 원문 앞부분 미리보기
//   - 클릭 시 아무 동작 없음 (모달 제거)
//
// 왜 모달을 제거했나:
//   파서가 PDF 테이블을 파이프(`|`) 구분자 + 개행으로 저장하기 때문에,
//   모달에 청크 원문을 `whitespace-pre-wrap`으로 뿌리면 표가 세로로
//   흩어져 사실상 읽을 수 없었다. 업계 레퍼런스 조사 결과 풀 청크 원문을
//   모달에 띄우는 제품이 없음을 확인:
//
//     - NotebookLM: 인용 클릭 → 원본 문서 위치 하이라이트 (모달 없음)
//     - Perplexity: 호버 프리뷰만 + 클릭 시 외부 이동
//     - Granola/Sana: 호버에서 paraphrase + 짧은 인용
//
//   "호버 프리뷰만 남기고 원문 확인은 PDF 사이드바로" (Perplexity 스타일)가
//   현재 환경(#117 PDF 사이드바 존재)에 가장 맞는다. 새로운 형태의 모달은
//   #120에서 별도 설계.
//
// props:
//   items: Array<{
//     index: int,
//     section: string,
//     snippet: string,
//     fullChunk: string,   // 호버 프리뷰용 (앞 120자만 사용)
//   }>
//   docId: string          // 현재 사용 중. 향후 #120에서 재사용 예정이라 prop은 유지

import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { Pin } from "lucide-react";

/**
 * 청크 본문에 섞여있는 마크다운/HTML artifact 제거.
 *
 * 청크는 파서가 PDF 문서 구조를 인식하면서 다양한 형식의 기호를 남긴 상태로
 * 저장된다:
 *   - 마크다운: ##, **, -, `
 *   - HTML: <br>, <br/>, <table>, <td>, <li> 등
 *
 * 호버 프리뷰용으로는 이런 기호가 보이면 UX 저하이므로 평문으로 정리.
 */
function stripMarkdown(text) {
  if (!text) return "";
  return text
    // ─── HTML 정제 ──────────────────
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/p\s*>/gi, "\n")
    .replace(/<p[^>]*>/gi, "")
    .replace(/<\/(tr|li|div|h[1-6])\s*>/gi, "\n")
    .replace(/<\/?[a-z][a-z0-9]*\b[^>]*>/gi, "")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    // ─── 마크다운 정제 ─────────────────────────────────
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/^[-*+]\s+/gm, "• ")
    .replace(/`([^`]+)`/g, "$1")
    // ─── 공백 정리 ───────────────────────────────────
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]+/g, " ")
    .replace(/[ \t]+\n/g, "\n")
    .trim();
}

/** 호버 팝오버용 짧은 미리보기 (앞 120자, 공백 정리) */
function buildPreview(fullChunk, maxLen = 120) {
  const clean = stripMarkdown(fullChunk).replace(/\s+/g, " ");
  if (clean.length <= maxLen) return clean;
  return clean.slice(0, maxLen).trimEnd() + "…";
}

function SourceItem({ item }) {
  const {
    index = 1,
    section = "",
    snippet = "",
    fullChunk = "",
  } = item || {};
  const hasFullChunk = Boolean(fullChunk);
  const preview = hasFullChunk ? buildPreview(fullChunk) : "";

  // 근거 한 줄 — 번호 배지 + 섹션명 + snippet. 호버 시 팝오버 미리보기.
  // 클릭 동작은 없음 — cursor-default로 시각적 어포던스도 제거.
  const rowContent = (
    <div
      className="
        w-full text-left group
        flex items-start gap-2 py-1.5 px-2 -mx-2 rounded-md
        transition-colors hover:bg-muted/60
        cursor-default
      "
      aria-label={`근거 ${index}: ${section}`}
    >
      <span
        className="
          shrink-0 inline-flex items-center justify-center
          w-5 h-5 mt-0.5 rounded
          bg-primary/10 text-primary
          text-xs font-medium
          group-hover:bg-primary/20
        "
      >
        {index}
      </span>
      <span className="flex-1 min-w-0 text-sm leading-relaxed">
        <span className="font-medium text-foreground">{section}</span>
        {snippet && (
          <>
            <span className="text-muted-foreground"> — </span>
            <span className="text-muted-foreground">{snippet}</span>
          </>
        )}
      </span>
    </div>
  );

  // full_chunk가 없으면 호버 없이 정보만 표시 (엣지케이스 방어).
  if (!hasFullChunk) {
    return <li className="list-none">{rowContent}</li>;
  }

  return (
    <li className="list-none">
      <HoverCard openDelay={250} closeDelay={150}>
        <HoverCardTrigger asChild>{rowContent}</HoverCardTrigger>

        {/*
         * 팝오버 — 호버 시 드는 미리보기 (260px, 컴팩트).
         * 원문 전체 보기 CTA 버튼은 제거됨 — 모달이 없어졌으므로.
         */}
        <HoverCardContent
          side="top"
          align="start"
          className="w-[260px] max-w-[90vw] p-2.5 space-y-1"
        >
          {section && (
            <div className="flex items-center gap-1.5 text-xs font-semibold text-foreground">
              <Pin className="h-3 w-3 text-primary shrink-0" />
              <span className="truncate">{section}</span>
            </div>
          )}
          <div className="text-[11px] leading-relaxed text-muted-foreground whitespace-pre-wrap">
            {preview}
          </div>
        </HoverCardContent>
      </HoverCard>
    </li>
  );
}

export default function SourceReference() {
  const { items = [] } = props || {};

  if (!items.length) return null;

  return (
    <div className="mt-4 pt-4 border-t border-border/60">
      <div className="flex items-center gap-1.5 mb-2 text-sm font-medium text-foreground">
        <Pin className="h-3.5 w-3.5 text-primary" />
        <span>출처</span>
      </div>
      <ul className="space-y-0.5">
        {items.map((it) => (
          <SourceItem key={it.index} item={it} />
        ))}
      </ul>
    </div>
  );
}
