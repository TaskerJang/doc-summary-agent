// public/elements/SourceReference.jsx
//
// Q&A 답변의 출처 리스트 — 호버 미리보기 + CTA 버튼 클릭으로 전체 원문 모달.
//
// UX 분리 (#111):
//   - 근거 텍스트 항목: 호버 시 팝오버 미리보기만 뜸 (클릭해도 아무 일 없음)
//   - 팝오버 내 "원문 전체 보기" 버튼: 클릭 시 shadcn Dialog 모달 오픈
//   - ESC/바깥 클릭/X로 모달 닫힘 (Dialog 기본 동작)
//
// 이전엔 근거 텍스트 버튼 자체가 호버(팝오버)와 클릭(모달)을 모두 받아
// 텍스트 클릭만 해도 바로 모달이 떴다. 의도와 다르므로 트리거를 분리:
// HoverCardTrigger는 근거 텍스트에, DialogTrigger는 팝오버 내 CTA 버튼에만.
//
// 단일 CustomElement로 전체 items를 map하여 렌더하므로 Chainlit의
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
import { Pin, Maximize2 } from "lucide-react";

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

/** 호버 팝오버용 짧은 미리보기 (앞 180자, 공백 정리) */
function buildPreview(fullChunk, maxLen = 180) {
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

  // 번호 뱃지 + 섹션명 + snippet 한 줄의 근거 항목.
  // <button>이 아닌 <div role="presentation">으로 두어 클릭 의도 제거 —
  // 호버 시에만 팝오버가 뜨고, 모달은 팝오버 내 CTA 버튼을 통해서만 열린다.
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

  // full_chunk 없으면 호버/클릭 인터랙션 없이 정보만 표시 (엣지케이스 방어).
  if (!hasFullChunk) {
    return <li className="list-none">{rowContent}</li>;
  }

  // Dialog는 최상위에 두고, DialogTrigger는 팝오버 안의 "원문 전체 보기"
  // 버튼에 둔다. HoverCard는 근거 텍스트 줄에 걸리고, 두 트리거가 분리되어
  // 의도한 UX:
  //   근거 텍스트 호버 → 팝오버 표시 (클릭해도 모달 안 뜸)
  //   팝오버 내 CTA 버튼 클릭 → 모달 표시
  return (
    <li className="list-none">
      <Dialog>
        <HoverCard openDelay={250} closeDelay={150}>
          <HoverCardTrigger asChild>{rowContent}</HoverCardTrigger>

          <HoverCardContent
            side="top"
            align="start"
            className="w-[320px] max-w-[90vw] p-0 overflow-hidden"
          >
            <div className="p-3 space-y-1.5">
              <div className="flex items-center gap-1.5 text-[11px] font-medium text-muted-foreground">
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
            </div>

            {/*
             * 팝오버 하단 CTA 버튼 — 이것만 DialogTrigger.
             * border-t + bg-primary/5로 본문과 시각 분리, 클릭 가능성을 명확화.
             */}
            <DialogTrigger asChild>
              <button
                type="button"
                className="
                  w-full flex items-center justify-center gap-1.5
                  px-3 py-2 border-t border-border/60
                  bg-primary/5 hover:bg-primary/10
                  text-primary text-xs font-medium
                  transition-colors
                  focus-visible:outline-none focus-visible:bg-primary/10
                "
              >
                <Maximize2 className="h-3 w-3" />
                <span>원문 전체 보기</span>
              </button>
            </DialogTrigger>
          </HoverCardContent>
        </HoverCard>

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
