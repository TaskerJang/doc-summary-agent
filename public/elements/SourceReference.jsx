// public/elements/SourceReference.jsx
//
// Q&A 원문 근거 팝업 모달 모음.
// 메시지에 엔엘리먼트를 여러 개 붙이는 대신 배열 props(items)를 받은
// 단일 CustomElement에서 .map()으로 버튼들을 렬더한다. 이유는
// chainlit/chainlit#2202 — msg.elements 배열의 순서가 렌더링 순서로
// 보장되지 않는 이슈. 하나의 엔엘리먼트 안에서 map 돌리면
// React가 이 순서를 그대로 지키므로 근거 1/2/3이 항상 순서대로
// 렌더된다.
//
// props.items: Array<{ index: int, section: string, fullChunk: string }>
//
// 각 항목은 shadcn Dialog로 팝업 모달. ESC/바깥 클릭/우상단 X로 닫힘.
// (#111)

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

function SourceDialog({ item }) {
  const { index, section, fullChunk } = item;
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
       * 명시적 픽셀 폭 + top/left 50% + translate 로 덼어쓴다. Chainlit이
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

export default function SourceReference() {
  const { items = [] } = props || {};

  if (!items.length) return null;

  return (
    <div className="flex flex-wrap items-center">
      {items.map((item) => (
        <SourceDialog key={item.index} item={item} />
      ))}
    </div>
  );
}
