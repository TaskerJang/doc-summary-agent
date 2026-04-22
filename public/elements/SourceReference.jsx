// public/elements/SourceReference.jsx
//
// Q&A 답변의 원문 근거 팝업 모달.
// 메시지 인라인에 "📄 근거 N" 버튼으로 렌더되고, 클릭 시 shadcn Dialog로
// 중앙 정렬 카드 형태의 모달이 열린다. ESC/바깥 클릭/우상단 X로 닫힘은
// Dialog 기본 동작. (#111)
//
// Chainlit이 public/elements/*.jsx를 자동 로드하고 shadcn + tailwind
// 환경을 제공한다 (https://docs.chainlit.io/api-reference/elements/custom).
// props는 전역 주입 — 함수 인자로 받지 않는다.
//
// 예상 props:
//   index:     int    — 근거 번호 (1부터)
//   section:   string — 섹션명
//   fullChunk: string — 원문 청크 본문 (파서가 남긴 마크다운 artifact 포함 가능)

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { FileText } from "lucide-react";

/**
 * 청크 본문에 섞여있는 마크다운 artifact 제거.
 *
 * 청크는 파서가 문서 구조를 인식하면서 ##, **, -, ` 같은 마크다운 기호를
 * 남긴 상태로 저장된다. 원문 검증 UX에선 이 기호들이 노이즈로 보여 가독성을
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

export default function SourceReference() {
  const { index = 1, section = "", fullChunk = "" } = props || {};
  const cleanBody = stripMarkdown(fullChunk);

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className="mr-2 mb-2 gap-1.5"
        >
          <FileText className="h-3.5 w-3.5" />
          <span>근거 {index}</span>
        </Button>
      </DialogTrigger>

      {/*
       * 중앙 정렬 카드 스타일을 Chainlit 기본 스타일보다 우선시하기 위해
       * 명시적 픽셀 폭 + top/left 50% + translate 로 덮어쓴다. Chainlit이
       * DialogContent에 top:0 같은 스타일을 주입해 상단 가로 바 형태로
       * 렌더되던 문제를 해결.
       */}
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
            <span>📍</span>
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
  );
}
