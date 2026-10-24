"""Discord 웹훅 알림 — 에이전트 없이 봇이 직접 채널에 보고한다.

**알림이 안 나가면 돈이 나가는 경로를 열지 않는다(fail-closed).** 호출자는 NotifyError를
'중단 사유'로 다뤄야 한다. 웹훅 URL은 시크릿이라 에러 메시지·로그 어디에도 넣지 않는다.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

log = logging.getLogger(__name__)

_CHUNK = 1900  # Discord 메시지 한도는 2000자
_HOSTS = ("discord.com", "discordapp.com")
# Discord는 기본 Python-urllib UA를 403(1010)으로 거른다.
_UA = "stock-bot-notify/1.0"


class NotifyError(RuntimeError):
    """알림 전송 실패. 메시지에 URL을 담지 않는다."""


def valid_webhook_url(url: str) -> bool:
    try:
        u = urlparse(url)
    except ValueError:
        return False
    host = (u.hostname or "").lower()
    return (
        u.scheme == "https"
        and any(host == h or host.endswith("." + h) for h in _HOSTS)
        and u.path.startswith("/api/webhooks/")
    )


def split_message(text: str, limit: int = _CHUNK) -> list[str]:
    """줄 단위로 limit 이내 조각으로 자른다. 한 줄이 limit보다 길면 그 줄만 강제로 자른다."""
    chunks: list[str] = []
    cur = ""
    for line in text.split("\n"):
        while len(line) > limit:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:limit])
            line = line[limit:]
        add = line if not cur else cur + "\n" + line
        if len(add) > limit:
            chunks.append(cur)
            cur = line
        else:
            cur = add
    if cur or not chunks:
        chunks.append(cur)
    return chunks


def _post(url: str, content: str, timeout: float, opener) -> None:
    body = json.dumps({"content": content, "allowed_mentions": {"parse": []}},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", "User-Agent": _UA},
    )
    with opener(req, timeout=timeout) as resp:
        code = getattr(resp, "status", 204)
        if not 200 <= code < 300:
            raise NotifyError(f"Discord 응답 HTTP {code}")


def send(url: str, text: str, *, timeout: float = 10.0, retries: int = 2,
         sleep=time.sleep, opener=urllib.request.urlopen) -> None:
    """text를 채널에 보낸다. 모든 조각이 나가야 성공이다. 실패하면 NotifyError."""
    if not url:
        raise NotifyError("STOCK_DISCORD_WEBHOOK_URL 미설정")
    if not valid_webhook_url(url):
        raise NotifyError("STOCK_DISCORD_WEBHOOK_URL 형식이 Discord 웹훅이 아니다")
    for chunk in split_message(text):
        last = ""
        for attempt in range(retries + 1):
            try:
                _post(url, chunk, timeout, opener)
                break
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code == 429:
                    wait = _retry_after(e)
                elif 400 <= e.code < 500:
                    raise NotifyError(f"Discord 거부: {last}") from None  # 재시도 무의미(URL 폐기 등)
                else:
                    wait = 1.0 * (attempt + 1)
            except NotifyError:
                raise
            except Exception as e:  # 타임아웃·DNS·연결 거부. str(e)에 URL이 섞일 수 있어 클래스만 쓴다.
                last = type(e).__name__
                wait = 1.0 * (attempt + 1)
            if attempt < retries:
                sleep(wait)
        else:
            raise NotifyError(f"Discord 전송 실패({retries + 1}회 시도): {last}")


def _retry_after(e: urllib.error.HTTPError) -> float:
    try:
        return min(float(json.loads(e.read().decode("utf-8")).get("retry_after", 1.0)), 5.0)
    except Exception:
        return 1.0
