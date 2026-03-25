import sys
from pathlib import Path

import parser as doc_parser


def main():
    if len(sys.argv) < 2:
        print("사용법: python main.py <문서 경로>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if not path.exists():
        print(f"[ERROR] 파일 없음: {path}")
        sys.exit(1)

    print(f"[Step 1] 파싱 시작: {path.name}")
    text = doc_parser.parse(path)
    print(f"[Step 1] 완료: {len(text):,}자 추출")
    print("-" * 60)
    print(text[:500])  # 앞 500자 미리보기


if __name__ == "__main__":
    main()