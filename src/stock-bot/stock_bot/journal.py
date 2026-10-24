"""일자별 운용 일지 — 정본은 로컬 파일 `<db 디렉터리>/journal/YYYY-MM-DD.md`.

봇이 직접 쓴다(Claude 없이). Outline 사본(outline.py)은 여기서 파생되며 실패해도 되지만,
**로컬 쓰기는 실패하면 JournalError**다 — 호출자가 Discord로 알리고 멈춘다.
같은 날 여러 번 실행되면 같은 파일 뒤에 이어 붙인다. 각 실행 절의 `요약: ` 줄이
위키 사본의 한 줄이다.
"""
from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta

SUMMARY_PREFIX = "요약: "
_DATE_FILE = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")


_WEEKDAYS = "월화수목금토일"


def weekday_ko(d: datetime | date | str) -> str:
    """요일 한 글자. locale에 기대지 않는다(컨테이너 locale은 보장 못 한다)."""
    if isinstance(d, str):
        d = date.fromisoformat(d)
    return _WEEKDAYS[d.weekday()]


def dated(d: datetime | date | str) -> str:
    """`2026-10-24 토` 형태. 문자열 날짜면 그대로 파싱한다."""
    if isinstance(d, str):
        d = date.fromisoformat(d)
    return f"{d.strftime('%Y-%m-%d')} {weekday_ko(d)}"


class JournalError(RuntimeError):
    pass


def journal_dir(db_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(db_path)), "journal")


def preflight(db_path: str) -> None:
    """실행 전에 일지 디렉터리에 실제로 쓸 수 있는지 확인한다(종가·주문을 건드리기 전)."""
    d = journal_dir(db_path)
    probe = os.path.join(d, f".probe-{os.getpid()}")
    try:
        os.makedirs(d, exist_ok=True)
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
    except OSError as e:
        raise JournalError(f"일지 디렉터리에 쓸 수 없음({d}): {e.strerror or type(e).__name__}") from None


def make_line(ticker: str, exit_code: int, message: str, record: dict | None,
              now: datetime) -> str:
    """위키·로컬 공용 한 줄. 날짜 접두라 문자열 정렬이 곧 시간순이다."""
    head = f"{dated(now)} {now.strftime('%H:%M')} UTC · {ticker}"
    if record:
        r = record
        tag = "ATTENTION" if r["notes"] else "OK"
        if r["sent"]:
            order = "전송 " + ", ".join(r["sent"])
        elif r["orders"]:
            gate = f" 차단[{r['blocked_gate']}]" if r["blocked_gate"] else ""
            order = "주문안 " + ", ".join(r["orders"]) + gate
        else:
            order = "주문 없음"
        parts = [
            f"[{tag}]", f"종가일 {dated(r['quote_date'])} 종가 ${r['close']:,.2f}",
            f"풀 ${r['pool']:,.2f}", f"V ${r['V']:,.2f}",
            f"밴드 ${r['band_low']:,.0f}–${r['band_high']:,.0f}",
            f"보유 {r['shares']:.0f}주", order, f"출처 {r['source']}",
        ]
        if r["cycle_started"]:
            parts.append("새 사이클")
        parts.append("이상: " + (" / ".join(r["notes"]) if r["notes"] else "없음"))
        body = " · ".join(parts)
    else:
        first = next((l for l in message.splitlines() if l.strip()), "").strip()
        body = f"{first or '(보고문 없음)'} · exit {exit_code}"
    return f"{head} · {' '.join(body.split())}"


def write(db_path: str, ticker: str, exit_code: int, message: str, line: str,
          now: datetime) -> str:
    """오늘 파일에 한 실행을 덧붙이고 경로를 돌려준다. 실패하면 JournalError."""
    try:
        d = journal_dir(db_path)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, now.strftime("%Y-%m-%d") + ".md")
        new = not os.path.exists(path)
        with open(path, "a", encoding="utf-8") as f:
            if new:
                f.write(f"# {dated(now)} 운용 일지\n")
            f.write(f"\n## {now.strftime('%H:%M:%S')} UTC · {ticker} · exit {exit_code}\n\n")
            f.write(SUMMARY_PREFIX + line + "\n\n")
            f.write(message.rstrip() + "\n")
            f.flush()
            os.fsync(f.fileno())
        return path
    except OSError as e:
        raise JournalError(f"일지 파일 쓰기 실패: {e.strerror or type(e).__name__}") from None


def recent_lines(db_path: str, now: datetime, days: int = 14) -> list[str]:
    """최근 days일 파일의 요약 줄들 — Outline이 며칠 막혔다 풀려도 빠짐없이 따라잡게 한다."""
    d = journal_dir(db_path)
    cutoff = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    out: list[str] = []
    try:
        names = sorted(n for n in os.listdir(d) if _DATE_FILE.match(n) and n[:10] >= cutoff)
    except OSError:
        return out
    for n in names:
        try:
            with open(os.path.join(d, n), encoding="utf-8") as f:
                out += [l[len(SUMMARY_PREFIX):].rstrip("\n") for l in f
                        if l.startswith(SUMMARY_PREFIX)]
        except OSError:
            continue
    return out
