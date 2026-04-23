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
 * 청크 본문에 섞여있는 마크다운/HTML artifact 제거.
 *
 * 청크는 파서가 PDF 문서 구조를 인식하면서 다양한 형식의 기호를 남긴 상태로
 * 저장된다:
 *   - 마크다운: ##, **, -, `
 *   - HTML (특히 표/리스트): <br>, <br/>, <table>, <td>, <li> 등
 *     (doc_parser가 PDF 테이블 셀 구분자로 <br>을 삽입하는 경우 확인됨)
 *
 * React는 문자열로 들어온 "<br>"를 자동으로 줄바꿈으로 해석하지 않고 그대로
 * 출력하므로, 여기서 선제적으로 평문으로 변환. 표/리스트의 구조 손실은 감수 —
 * 구조까지 확인하려면 사용자가 "📂 원문 보기" 링크로 원본 PDF를 열면 된다.
 *
 * 청크에 HTML 태그가 남는 건 파서 측 이슈로, 별도 개선 대상이지만 현재는
 * 표시 단계에서 안전하게 방어.
 */
function stripMarkdown(text) {
  if (!text) return "";
  return text
    // ─── HTML 정제 (마크다운보다 먼저 처리) ──────────────────────────
    // <br>, <br/>, <BR/>, <br /> 모두 줄바꿈으로 (대소문자/공백 무관)
    .replace(/<br\s*\/?>/gi, "\n")
    // <p>, </p>도 문단 구분이므로 줄바꿈으로 치환
    .replace(/<\/p\s*>/gi, "\n")
    .replace(/<p[^>]*>/gi, "")
    // 표 관련 태그도 줄바꿈으로 — 셀/행 구분 최소 보존
    .replace(/<\/(tr|li|div|h[1-6])\s*>/gi, "\n")
    // 나머지 HTML 태그는 전부 제거 (속성 포함)
    .replace(/<\/?[a-z][a-z0-9]*\b[^>]*>/gi, "")
    // HTML 엔티티 몇 가지 정리
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    // ─── 마크다운 정제 ────────────────────────────────────────────
    .replace(/^#{1,6}\s+/gm, "")      // ## 헤더 → 평문
    .replace(/\*\*([^*]+)\*\*/g, "$1") // **굵은글씨** → 평문
    .replace(/\*([^*]+)\*/g, "$1")    // *기울임* → 평문
    .replace(/^[-*+]\s+/gm, "• ")     // - 리스트 → • 불릿
    .replace(/`([^`]+)`/g, "$1")      // `코드` → 평문
    // ─── 공백 정리 ────────────────────────────────────────────────
    .replace(/\n{3,}/g, "\n\n")       // 과한 줄바꿈 정리
    .replace(/[ \t]+/g, " ")          // 연속 공백/탭 1칸으로
    .replace(/[ \t]+\n/g, "\n")       // 줄 끝 공백 제거
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
