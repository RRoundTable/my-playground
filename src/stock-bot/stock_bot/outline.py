"""Outline 위키 운용 일지 사본 — best-effort.

정본은 로컬 일자별 파일(journal.py)이다. 여기는 사람이 위키에서 보기 위한 사본이라
**실패해도 봇을 멈추지 않는다**: 호출자는 OutlineError를 경고 한 줄로만 다룬다.
문서 본문은 통째로 교체해야 하므로(Outline API 규칙) 받아서 우리 절만 고쳐 돌려보내며,
그 절 밖의 본문은 한 글자도 바꾸지 않는다. 토큰은 메시지·로그 어디에도 넣지 않는다.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

HEADING = "## 봇 자동 기록"
INTRO = ("봇이 매 실행 직후 직접 쓴다(최신이 위). 정본은 컨테이너의 `journal/YYYY-MM-DD.md`이고 "
         "이 절은 사본이다 — 손으로 고치지 마라, 다음 실행이 덮어쓴다.")
DEFAULT_DOC_ID = "81087655-2a9b-4346-afa6-cd480d4720b0"  # 주식봇 운용 일지 (시크릿 아님)
DEFAULT_CONFIG_PATH = "/mnt/outline-cli/config.json"
_UA = "stock-bot-outline/1.0"
_DATED_H2 = re.compile(r"^## \d{4}-\d{2}-\d{2}")


class OutlineError(RuntimeError):
    """Outline 갱신 실패. 메시지에 토큰을 담지 않는다."""


def _unescape(s: str) -> str:
    """Outline이 저장하며 붙이는 마크다운 이스케이프(\\[ \\] 등)를 벗겨 비교 기준으로 쓴다."""
    return re.sub(r"\\([^\w\s])", r"\1", s).strip()


def merge_section(body: str, new_lines: list[str], cap: int = 120) -> str:
    """body의 '봇 자동 기록' 절에 new_lines를 합친다(합집합, 최신이 위, cap건까지).

    절이 없으면 첫 날짜 절(`## YYYY-MM-DD ...`) 앞에 만든다 — 사람이 쓰는 '미결 안건'은
    그대로 맨 위에 남고, 봇 기록이 날짜별 서술 위에 쌓인다. 절 밖은 건드리지 않는다.
    """
    lines = body.split("\n")
    start = next((i for i, l in enumerate(lines) if l.strip() == HEADING), None)
    end = len(lines)
    existing: list[str] = []
    if start is not None:
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
                   len(lines))
        existing = [_unescape(l[2:]) for l in lines[start + 1:end] if l.startswith("- ")]
    merged = sorted({*existing, *(_unescape(l) for l in new_lines if l.strip())}, reverse=True)[:cap]
    block = [HEADING, "", INTRO, ""] + [f"- {l}" for l in merged] + [""]
    if start is not None:
        out = lines[:start] + block + lines[end:]
    else:
        at = next((i for i, l in enumerate(lines) if _DATED_H2.match(l)), None)
        if at is not None:
            out = lines[:at] + block + lines[at:]
        else:
            out = lines + ([""] if lines and lines[-1] != "" else []) + block
    return "\n".join(out)


def load_credentials(path: str) -> tuple[str, str]:
    """(url, token). 읽기 전용 마운트의 config.json에서 읽는다. 값은 호출자만 쥔다."""
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        url, token = str(cfg["url"]).rstrip("/"), str(cfg["token"])
    except (OSError, ValueError, KeyError, TypeError):
        raise OutlineError(f"Outline 설정을 읽지 못함({path}: 없음/형식 오류)") from None
    if not url.startswith(("http://", "https://")) or not token:
        raise OutlineError(f"Outline 설정 값이 비었거나 형식이 틀림({path})")
    return url, token


def _call(url: str, token: str, endpoint: str, payload: dict, timeout: float, opener) -> dict:
    req = urllib.request.Request(
        f"{url}/api/{endpoint}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}",
                 "User-Agent": _UA},
    )
    try:
        with opener(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise OutlineError(f"HTTP {e.code} 인증 실패(API 키 만료 가능)") from None
        if e.code == 404:
            raise OutlineError("HTTP 404 문서를 찾지 못함") from None
        raise OutlineError(f"HTTP {e.code}") from None
    except OutlineError:
        raise
    except Exception as e:  # 타임아웃·DNS·JSON 깨짐. str(e)는 쓰지 않는다.
        raise OutlineError(f"연결/응답 실패({type(e).__name__})") from None
    if not isinstance(data, dict) or data.get("ok") is False:
        raise OutlineError("응답 ok=false")
    return data


def sync(config_path: str, doc_id: str, new_lines: list[str], *, timeout: float = 10.0,
         opener=urllib.request.urlopen, cap: int = 120) -> None:
    """운용 일지 문서의 '봇 자동 기록' 절을 갱신한다. 실패하면 OutlineError."""
    url, token = load_credentials(config_path)
    info = _call(url, token, "documents.info", {"id": doc_id}, timeout, opener)
    body = (info.get("data") or {}).get("text")
    if not isinstance(body, str) or not body.strip():
        # 본문을 못 받았는데 우리 절만 담아 교체하면 문서 전체가 날아간다.
        raise OutlineError("현재 본문을 받지 못해 교체하지 않음")
    new_body = merge_section(body, new_lines, cap)
    if new_body == body:
        return
    _call(url, token, "documents.update", {"id": doc_id, "text": new_body}, timeout, opener)
