"""시세 어댑터 — 일봉 종가 조회 (의존성 0, urllib).

1순위 Yahoo Finance 차트 API(JSON), 실패 시 stooq CSV 폴백.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from stock_bot import net

_UA = "Mozilla/5.0 (X11; Linux x86_64) stock-bot/0.1"


class QuoteError(RuntimeError):
    pass


def last_close(ticker: str) -> tuple[str, float]:
    """(날짜 'YYYY-MM-DD', 종가) — 가장 최근 일봉.

    ⚠️ **장이 열려 있는 동안에는 진행 중인 바를 돌려준다** — 그 close는 확정
    종가가 아니라 현재가다. 봇 로직에서는 쓰지 말고 `next_unprocessed_close()`를
    써라. 그쪽이 '세션이 끝난 날'만 채택하는 가드를 갖고 있다.
    """
    errors: list[str] = []
    for source in (_yahoo, _stooq):
        try:
            return source(ticker)
        except QuoteError as e:
            errors.append(str(e))
    raise QuoteError(f"모든 시세 소스 실패({ticker}): {' | '.join(errors)}")


def _range_for(after_date: str) -> str:
    """after_date부터 오늘까지를 덮는 최소 yahoo range."""
    try:
        gap = (
            datetime.now(timezone.utc).date()
            - datetime.strptime(after_date, "%Y-%m-%d").date()
        ).days
    except ValueError:
        return "1y"
    for limit, range_ in ((25, "1mo"), (80, "3mo"), (170, "6mo"), (350, "1y")):
        if gap <= limit:
            return range_
    return "5y"


def next_unprocessed_close(
    ticker: str, after_date: str | None, today: str | None = None
) -> tuple[str, float] | None:
    """after_date 다음으로 오래된 **확정** 종가 1건. 없으면 None.

    두 가지를 동시에 지킨다.

    1) **오래된 것부터 하나씩.** 여러 날 밀렸을 때 최신 것만 집으면 그 사이
       영업일이 통째로 사라지고 day_in_cycle은 1만 올라 사이클 위상이 어긋난다
       (2026-08 TSLA 사고). run 1회 = 영업일 1일.

    2) **진행 중인 바를 종가로 착각하지 않는다.** 시세 소스는 장이 열려 있는 동안
       그날 바를 이미 내놓고, 그 close는 확정 종가가 아니라 **현재가**다. 그걸
       집으면 현재가를 종가로 삼아 계산하고 last_quote_date를 그날로 올려
       **실제 종가를 영구히 건너뛴다.** 미국 정규장(13:30~20:00 UTC, 표준시
       14:30~21:00)은 UTC 자정을 넘지 않으므로 **종가일 < 오늘 UTC 날짜**면
       그 세션은 반드시 끝나 있다 — 그것만 채택한다.

    today는 테스트용 주입구다(기본값 = 현재 UTC 날짜).
    """
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not after_date:
        # 부트스트랩: 과거를 재생하지 않고 가장 최근 '확정' 종가부터 시작한다.
        # last_close()는 장중이면 진행 중 바를 주므로 여기서 쓰지 않는다.
        done = [row for row in history(ticker, "1mo") if row[0] < today]
        return done[-1] if done else None
    for day, close in history(ticker, _range_for(after_date)):  # 오름차순
        if after_date < day < today:
            return day, close
    return None


def _yahoo(ticker: str) -> tuple[str, float]:
    try:
        data = net.get_json(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            headers={"User-Agent": _UA},
            params={"range": "5d", "interval": "1d"},
        )
    except (net.HttpError, ValueError) as e:
        raise QuoteError(f"yahoo 요청 실패: {e}") from e

    try:
        result = data["chart"]["result"][0]
        stamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
    except (KeyError, IndexError, TypeError) as e:
        raise QuoteError(f"yahoo 응답 파싱 실패: {e}") from e

    for ts, close in zip(reversed(stamps), reversed(closes)):
        if close is not None:
            day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            return day, float(close)
    raise QuoteError("yahoo: 유효한 종가 없음")


def _stooq(ticker: str) -> tuple[str, float]:
    try:
        text = net.get_text(
            f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d",
            headers={"User-Agent": _UA},
        )
    except net.HttpError as e:
        raise QuoteError(f"stooq 요청 실패: {e}") from e

    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or "Close" not in rows[0]:
        raise QuoteError(f"stooq 응답 파싱 실패: {text[:80]!r}")
    last = rows[-1]
    try:
        return last["Date"], float(last["Close"])
    except (KeyError, ValueError) as e:
        raise QuoteError(f"stooq 종가 파싱 실패: {last}") from e


def history(ticker: str, range_: str = "5y") -> list[tuple[str, float]]:
    """(날짜 'YYYY-MM-DD', 종가) 오름차순 리스트. 백테스트용 전체 구간 조회.

    range_는 yahoo 차트 API 형식(1y/2y/5y/10y/max 등). stooq 폴백은 전체 이력을 반환한다.
    """
    errors: list[str] = []
    for source in (_yahoo_history, _stooq_history):
        try:
            rows = source(ticker, range_)
            if rows:
                return rows
        except QuoteError as e:
            errors.append(str(e))
    raise QuoteError(f"모든 시세 소스 실패({ticker}): {' | '.join(errors)}")


def _yahoo_history(ticker: str, range_: str) -> list[tuple[str, float]]:
    try:
        data = net.get_json(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            headers={"User-Agent": _UA},
            params={"range": range_, "interval": "1d"},
        )
    except (net.HttpError, ValueError) as e:
        raise QuoteError(f"yahoo 요청 실패: {e}") from e

    try:
        result = data["chart"]["result"][0]
        stamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
    except (KeyError, IndexError, TypeError) as e:
        raise QuoteError(f"yahoo 응답 파싱 실패: {e}") from e

    out: list[tuple[str, float]] = []
    for ts, close in zip(stamps, closes):
        if close is not None:
            day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            out.append((day, float(close)))
    if not out:
        raise QuoteError("yahoo: 유효한 종가 없음")
    return out


def _stooq_history(ticker: str, range_: str) -> list[tuple[str, float]]:
    try:
        text = net.get_text(
            f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d",
            headers={"User-Agent": _UA},
        )
    except net.HttpError as e:
        raise QuoteError(f"stooq 요청 실패: {e}") from e

    rows = list(csv.DictReader(io.StringIO(text)))
    out: list[tuple[str, float]] = []
    for r in rows:
        try:
            out.append((r["Date"], float(r["Close"])))
        except (KeyError, ValueError):
            continue
    if not out:
        raise QuoteError(f"stooq 응답 파싱 실패: {text[:80]!r}")
    return out
