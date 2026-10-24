"""의존성 0 HTTP 헬퍼 (urllib 기반)."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request


class HttpError(RuntimeError):
    pass


def _read(method: str, url: str, headers: dict | None, data: bytes | None, timeout: float) -> str:
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raise HttpError(f"HTTP {e.code} {url}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise HttpError(f"요청 실패 {url}: {e}") from e


def get_text(url: str, headers: dict | None = None, params: dict | None = None, timeout: float = 20.0) -> str:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    return _read("GET", url, headers, None, timeout)


def get_json(url: str, headers: dict | None = None, params: dict | None = None, timeout: float = 20.0) -> dict:
    return json.loads(get_text(url, headers, params, timeout))


def post_json(url: str, body: dict, headers: dict | None = None, timeout: float = 15.0) -> dict:
    data = json.dumps(body).encode("utf-8")
    merged = {"content-type": "application/json", **(headers or {})}
    return json.loads(_read("POST", url, merged, data, timeout))
