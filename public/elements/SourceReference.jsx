// public/elements/SourceReference.jsx
//
// Q&A 답변의 출처 리스트 — 호버 미리보기 + 클릭 원문 전체 모달.
//
// UX (#111):
//   - 근거 텍스트 호버: HoverCard로 원문 앞부분 미리보기
//   - 근거 텍스트 클릭: Dialog 모달로 원문 청크 전체 보기
//   - ESC/바깥 클릭/X 버튼으로 닫힘 (Dialog 기본)
//
// Radix의 HoverCard와 Dialog는 동일 엘리먼트에 asChild로 중첩 마운트가
// 가능하다. 안쪽 스택(HoverCardTrigger) → 바깥 스택(DialogTrigger) 순서로
// 감싸도 두 Trigger 모두 정상 동작한다.
//
// 레퍼런스
//   - Shape of AI: "dual mode (hover preview / click full source)"
//   - Perplexity / Granola / Sana: claim-to-source UX
//   - thefrontkit: "left border or background shading to distinguish cited material"
//   - Graphlit: "title, page, relevance, excerpt" citation card metadata
//   - thefrontkit: "Keep labels short and consistent. Too much detail can overwhelm."
//
// 단일 CustomElement로 전체 items를 map하므로 Chainlit의
// msg.elements 순서 미보장 이슈(chainlit#2202)에 영향받지 않는다.
//
// props:
//   items: Array<{
//     index: int,
//     section: string,
//     snippet: string,     // ui/app.py qa_result.sources[i].snippet (요약 섹션 불릿)
//     fullChunk: string,   // reranker top-3 원문 청크 본문
//   }>
//   docId: string          // 현재 세션의 원본 문서 파일명 (모달 헤더 메타데이터용)

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { Pin, FileText } from "lucide-react";

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
    // ─── HTML 정제 (마크다운보다 먼저 처리) ──────────────────
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
    // ─── 마크다운 정제 ─────────────────────────────────
    .replace(/^#{1,6}\s+/gm, "")      // ## 헤더 → 평문
    .replace(/\*\*([^*]+)\*\*/g, "$1") // **굵은글씨** → 평문
    .replace(/\*([^*]+)\*/g, "$1")    // *기울임* → 평문
    .replace(/^[-*+]\s+/gm, "• ")     // - 리스트 → • 불릿
    .replace(/`([^`]+)`/g, "$1")      // `코드` → 평문
    // ─── 공백 정리 ───────────────────────────────────
    .replace(/\n{3,}/g, "\n\n")       // 과한 줄바꿈 정리
    .replace(/[ \t]+/g, " ")          // 연속 공백/탭 1칸으로
    .replace(/[ \t]+\n/g, "\n")       // 줄 끝 공백 제거
    .trim();
}

/** 호버 팝오버용 짧은 미리보기 (앞 120자, 공백 정리) */
function buildPreview(fullChunk, maxLen = 120) {
  const clean = stripMarkdown(fullChunk).replace(/\s+/g, " ");
  if (clean.length <= maxLen) return clean;
  return clean.slice(0, maxLen).trimEnd() + "…";
}

function SourceItem({ item, docId }) {
  const {
    index = 1,
    section = "",
    snippet = "",
    fullChunk = "",
  } = item || {};
  const hasFullChunk = Boolean(fullChunk);
  const preview = hasFullChunk ? buildPreview(fullChunk) : "";
  const cleanBody = hasFullChunk ? stripMarkdown(fullChunk) : "";

  // 근거 한 줄 — 번호 배지 + 섹션명 + snippet.
  // 호버(팝오버 미리보기)와 클릭(모달) 둘 다 받는다.
  const rowContent = (
    <div
      className="
        w-full text-left group
        flex items-start gap-2 py-1.5 px-2 -mx-2 rounded-md
        transition-colors hover:bg-muted/60
        cursor-pointer
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
    return (
      <li className="list-none">
        <div className="w-full text-left flex items-start gap-2 py-1.5 px-2 -mx-2 rounded-md cursor-default">
          <span className="shrink-0 inline-flex items-center justify-center w-5 h-5 mt-0.5 rounded bg-primary/10 text-primary text-xs font-medium">
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
      </li>
    );
  }

  // HoverCard와 Dialog의 Trigger를 동일 엘리먼트에 중첩 적용.
  // Radix 문서: "asChild allows you to compose multiple primitives"
  //   → HoverCardTrigger와 DialogTrigger 둘 다 asChild로, HoverCardTrigger가
  //     안에 오도록 중첩하면 둘 다 할 일을 한다 (호버는 외부 Dialog에 영향 안 줌).
  return (
    <li className="list-none">
      <Dialog>
        <HoverCard openDelay={250} closeDelay={150}>
          <DialogTrigger asChild>
            <HoverCardTrigger asChild>{rowContent}</HoverCardTrigger>
          </DialogTrigger>

          {/*
           * 팝오버 — 호버 시 드는 미리보기 (260px, 컴팩트).
           * "원문 전체 보기" CTA 버튼은 제거됨 — 사용자 피드백: 근거 줄
           * 자체를 클릭하는 게 더 직관적이라는 판단. 팝오버는 순수 힌트 역할.
           */}
          <HoverCardContent
            side="top"
            align="start"
            className="w-[260px] max-w-[90vw] p-2.5"
          >
            <div className="space-y-1">
              {section && (
                <div className="flex items-center gap-1.5 text-xs font-semibold text-foreground">
                  <Pin className="h-3 w-3 text-primary shrink-0" />
                  <span className="truncate">{section}</span>
                </div>
              )}
              <div className="text-[11px] leading-relaxed text-muted-foreground whitespace-pre-wrap">
                {preview}
              </div>
              <div className="text-[10px] text-muted-foreground/70 pt-0.5">
                클릭으로 원문 전체 보기
              </div>
            </div>
          </HoverCardContent>
        </HoverCard>

        {/*
         * 모달 — 클릭 시 드는 원문 청크 전체.
         *
         * 헤더 정보 최소화 (레퍼런스 원칙):
         *   - Perplexity/NotebookLM: 헤더는 "어디서 왔는지"만 표시 (파일명)
         *   - thefrontkit: "Keep labels short. Too much detail can overwhelm."
         *   - 섹션명은 본문 첫 줄에 종종 중복되므로 헤더에서 제거
         *     (호버 팝오버와 메시지 출처 리스트에 이미 섹션명 표시됨)
         *   - 헤더: 파일 아이콘 + 파일명 · 근거 N (한 줄로 컴팩트)
         *
         * 본문 영역:
         *   - 좌측 border-l-2 border-primary/40 + pl-4로 원문 인용임을 시각화
         *     (thefrontkit: "background shading or left border")
         *
         * 크기:
         *   - max-w-[560px]: 640px에서 줄임 — 사용자 포커스 존중 (Shape of AI)
         */}
        <DialogContent
          className="
            fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2
            w-[90vw] max-w-[560px] max-h-[80vh]
            flex flex-col gap-0
            rounded-lg border bg-background shadow-xl
            p-0
          "
        >
          <DialogHeader className="px-6 pt-5 pb-4 border-b">
            {/*
             * 헤더를 한 줄로 통합: 파일명(메인) · 근거 N(보조).
             * docId가 없으면 fallback으로 "근거 N"만 표시.
             * DialogTitle은 스크린리더/a11y를 위해 반드시 있어야 하므로
             * 유지하되, 기존의 큰 제목 느낌(text-base) 대신 한 줄 메타데이터
             * 스타일(text-sm + muted 결합)로 바꿈.
             */}
            <DialogTitle className="flex items-center gap-2 text-sm font-medium text-foreground">
              {docId ? (
                <>
                  <FileText className="h-4 w-4 text-primary shrink-0" />
                  <span className="truncate flex-1 min-w-0">{docId}</span>
                  <span className="shrink-0 text-xs text-muted-foreground font-normal">
                    · 근거 {index}
                  </span>
                </>
              ) : (
                <>
                  <Pin className="h-4 w-4 text-primary shrink-0" />
                  <span>근거 {index}</span>
                </>
              )}
            </DialogTitle>
          </DialogHeader>

          <div className="overflow-auto flex-1 px-6 py-4">
            {/*
             * 좌측 primary 보더 + pl-4로 "이건 내가 작성한 게 아니라 원본 인용이다"
             * 를 시각적으로 표시. (thefrontkit: "background shading or left border")
             */}
            <div className="border-l-2 border-primary/40 pl-4">
              <div className="whitespace-pre-wrap text-sm leading-relaxed text-foreground">
                {cleanBody || "원문 청크 내용이 없습니다."}
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </li>
  );
}

export default function SourceReference() {
  const { items = [], docId = "" } = props || {};

  if (!items.length) return null;

  return (
    <div className="mt-4 pt-4 border-t border-border/60">
      <div className="flex items-center gap-1.5 mb-2 text-sm font-medium text-foreground">
        <Pin className="h-3.5 w-3.5 text-primary" />
        <span>출처</span>
      </div>
      <ul className="space-y-0.5">
        {items.map((it) => (
          <SourceItem key={it.index} item={it} docId={docId} />
        ))}
      </ul>
    </div>
  );
}
