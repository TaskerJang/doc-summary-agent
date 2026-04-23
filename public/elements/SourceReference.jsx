// public/elements/SourceReference.jsx
//
// Q&A 답변의 출처 리스트 — 인터랙션 없이 정보만 표시.
//
// UX:
//   - 근거 한 줄: 번호 배지 + 섹션명 + snippet
//   - 호버/클릭 등 인터랙션 없음
//   - 원문 확인이 필요하면 "📂 원본 PDF 열기/닫기" 버튼으로 사이드바에서 열기
//
// 왜 이 형태인가:
//   1) 이전엔 Dialog 모달로 청크 원문 전체를 띄웠으나, 파서가 테이블을
//      파이프(`|`) + 개행으로 저장해서 모달에서 표가 완전히 깨져 읽을 수
//      없었다. 모달 자체를 제거.
//   2) HoverCard 프리뷰도 함께 제거. 청크 앞 120자를 미리보기로 띄우는
//      방식이었는데, 표 청크일 때는 미리보기에도 깨진 파이프 문자열이
//      그대로 노출돼 프리뷰의 의미가 없었다. 정보 가치도 낮고 표시 품질도
//      낮은 호버는 빼는 게 낫다.
//   3) 새로운 근거 UX(표 렌더링 + chunk_type 배지 + relevance 순위 + PDF
//      사이드바 CTA)는 #120에서 처음부터 설계.
//
// 레퍼런스:
//   - Perplexity: 호버 프리뷰만 + 클릭 시 외부 이동
//     (우리는 외부 이동 대신 별도 PDF 사이드바 버튼으로 분리)
//   - Shape of AI "Respect the user's focus": 답변이 주인공, 출처는 거들 뿐
//
// props:
//   items: Array<{
//     index: int,
//     section: string,
//     snippet: string,
//     fullChunk: string,   // 현재 미사용. #120 재설계 시 재활용 예정
//   }>
//   docId: string          // 현재 미사용. #120 재설계 시 재활용 예정

import { Pin } from "lucide-react";

function SourceItem({ item }) {
  const {
    index = 1,
    section = "",
    snippet = "",
  } = item || {};

  return (
    <li className="list-none">
      <div
        className="
          w-full text-left
          flex items-start gap-2 py-1.5 px-2 -mx-2 rounded-md
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
