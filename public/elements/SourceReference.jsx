// public/elements/SourceReference.jsx
//
// Q&A 답변의 원문 근거 팝업 모달.
// 메시지 인라인에 "🔍 근거 N" 버튼으로 렌더되고, 클릭 시 shadcn Dialog로
// 풀스크린 오버레이 모달이 열린다. ESC/바깥 클릭/우상단 X로 닫힘은
// Dialog 기본 동작. (#111)
//
// Chainlit이 public/elements/*.jsx를 자동 로드하고 shadcn + tailwind
// 환경을 제공한다 (https://docs.chainlit.io/api-reference/elements/custom).
// props는 전역 주입 — 함수 인자로 받지 않는다.
//
// 예상 props:
//   index:     int    — 근거 번호 (1부터)
//   section:   string — 섹션명
//   fullChunk: string — 원문 청크 본문

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Search } from "lucide-react";

export default function SourceReference() {
  const { index = 1, section = "", fullChunk = "" } = props || {};

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className="mr-2 mb-2 gap-1.5"
        >
          <Search className="h-3.5 w-3.5" />
          <span>근거 {index}</span>
        </Button>
      </DialogTrigger>

      <DialogContent className="max-w-2xl max-h-[80vh]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <span>📍</span>
            <span>근거 {index}</span>
            {section && (
              <>
                <span className="text-muted-foreground">·</span>
                <span className="font-normal text-muted-foreground">
                  {section}
                </span>
              </>
            )}
          </DialogTitle>
        </DialogHeader>

        <ScrollArea className="max-h-[60vh] pr-4">
          <div className="whitespace-pre-wrap text-sm leading-relaxed">
            {fullChunk || "원문 청크 내용이 없습니다."}
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
