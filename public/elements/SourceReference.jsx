// public/elements/SourceReference.jsx
//
// Q&A 답변의 출처 리스트 — 호버 미리보기 + 클릭 전체 원문 모달.
//
// Perplexity/Granola/Sana 같은 프로덕트들의 "claim-to-source" UX를 따름:
//   - 출처를 답변 아래 번호 뱃지 + 섹션명 + snippet 한 줄로 표시
//   - 호버 시 HoverCard 팝오버로 원문 청크 앞부분 미리보기
//   - 클릭 시 shadcn Dialog 모달로 원문 청크 전체 표시 (ESC/바깥/X로 닫힘)
//
// 단일 엘리먼트 안에서 전체 items를 map으로 렌더하므로 Chainlit의
// msg.elements 순서 미보장 이슈(chainlit#2202)에 영향받지 않는다.
//
// props.items: Array<{
//   index: int,
//   section: string,
//   snippet: string,     // ui/app.py qa_result.sources[i].snippet (요약 섹션 불릿)
//   fullChunk: string,   // reranker top-3 원문 청크 본문
// }>

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { Pin } from "lucide-react";

/**
 * 청크 본문에 섞여있는 마크다운 artifact 제거.
 *
 * 청크는 파서가 문서 구조를 인식하면서 ##, **, -, ` 같은 마크다운 기호를
 * 남긴 상태로 저장된다. 원문 검증 UX엔 이 기호들이 노이즈로 보여 가독성을
 * 해치므로 제거한다. 표/리스트의 구조 손실은 감수 — 구조까지 확인하려면
 * 기존 PDF 사이드 패널을 쓰면 된다.
 */
function stripMarkdown(text) {
  if (!text) return "";
  return text
    .replace(/^#{1,6}\s+/gm, "")      // ## 헤더 → 평문
    .replace(/\*\*([^*]+)\*\*/g, "$1") // **굵은글씨** → 평문
    .replace(/\*([^*]+)\*/g, "$1")    // *기울임* → 평문
    .replace(/^[-*+]\s+/gm, "• ")     // - 리스트 → • 불릿
    .replace(/`([^`]+)`/g, "$1")      // `코드` → 평문
    .replace(/\n{3,}/g, "\n\n")       // 과한 공백 정리
    .trim();
}

/** 호버 팝오버용 짧은 미리보기 (앞 240자, 공백 정리) */
function buildPreview(fullChunk, maxLen = 240) {
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
  const cleanBody = hasFullChunk ? stripMarkdown(fullChunk) : "";

  // 번호 뱃지 + 섹션명 + snippet을 한 줄에 표시하는 클릭 가능한 버튼.
  // 이 자체가 답변의 출처 항목 겸 근거 팝업 트리거.
  const triggerContent = (
    <button
      type="button"
      className="
        w-full text-left group
        flex items-start gap-2 py-1.5 px-2 -mx-2 rounded-md
        transition-colors hover:bg-muted/60 focus-visible:bg-muted/60
        focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring
        disabled:cursor-default disabled:hover:bg-transparent
      "
      disabled={!hasFullChunk}
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
    </button>
  );

  // full_chunk 없으면 호버/클릭 인터랙션 없이 정보만 표시 (엣지케이스 방어).
  // 이 경로는 현재 로직상 도달 어려움 — _make_sources가 relevant_chunks를
  // 인덱스 매칭으로 채우고, overall fallback 주 경로는 sources=[]라 이
  // 엘리먼트 자체가 렌더되지 않음. 장래 변경 대비 방어 코드.
  if (!hasFullChunk) {
    return <li className="list-none">{triggerContent}</li>;
  }

  return (
    <li className="list-none">
      <HoverCard openDelay={250} closeDelay={150}>
        <HoverCardTrigger asChild>
          <Dialog>
            <DialogTrigger asChild>{triggerContent}</DialogTrigger>

            <DialogContent
              className="
                fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2
                w-[90vw] max-w-[640px] max-h-[80vh]
                flex flex-col gap-0
                rounded-lg border bg-background shadow-xl
                p-0
              "
            >
              <DialogHeader className="px-6 pt-6 pb-4 border-b">
                <DialogTitle className="flex items-center gap-2 text-base">
                  <Pin className="h-4 w-4 text-primary" />
                  <span>근거 {index}</span>
                </DialogTitle>
                {section && (
                  <DialogDescription className="text-sm text-muted-foreground mt-1">
                    {section}
                  </DialogDescription>
                )}
              </DialogHeader>

              <div className="overflow-auto px-6 py-4 flex-1">
                <div className="whitespace-pre-wrap text-sm leading-relaxed text-foreground">
                  {cleanBody || "원문 청크 내용이 없습니다."}
                </div>
              </div>
            </DialogContent>
          </Dialog>
        </HoverCardTrigger>

        <HoverCardContent
          side="top"
          align="start"
          className="w-[420px] max-w-[90vw]"
        >
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
              <Pin className="h-3 w-3" />
              <span>근거 {index} · 미리보기</span>
            </div>
            {section && (
              <div className="text-sm font-medium text-foreground">
                {section}
              </div>
            )}
            <div className="text-xs leading-relaxed text-muted-foreground whitespace-pre-wrap">
              {preview}
            </div>
            <div className="pt-1 text-[11px] text-muted-foreground/80">
              클릭하면 원문 전체 보기
            </div>
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
