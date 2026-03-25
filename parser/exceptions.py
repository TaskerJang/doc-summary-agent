class ParseError(Exception):
    """파싱 중 복구 불가능한 오류 기본 클래스"""


class FileNotSupportedError(ParseError):
    """지원하지 않는 포맷 입력"""


class FileCorruptedError(ParseError):
    """손상된 파일 또는 읽기 실패"""


class EmptyDocumentError(ParseError):
    """파일은 정상이나 추출된 텍스트가 비어있음"""