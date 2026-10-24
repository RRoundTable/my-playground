"""상태 영속화 — sqlite 단일 파일 (단일 상태 행 + 이력)."""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict

from stock_bot.domain.models import AccountState, Order, Side


class Store:
    def __init__(self, path: str, ticker: str | None = None):
        """ticker를 주면 DB에 기록된 종목과 대조한다(처음이면 기록).

        `STOCK_DB_PATH`와 설정 경로가 **각각** 지정되는 구조라 둘이 어긋날 수 있다 —
        A 종목 DB에 B 종목 설정이 먹는 상태다. 그러면 남의 장부에 남의 전략이
        돌아가는데 아무 데서도 에러가 안 난다. 여기서 멈춘다.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS state ("
            "id INTEGER PRIMARY KEY CHECK (id = 1), data TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS history ("
            "ts TEXT NOT NULL, quote_date TEXT, close REAL, snapshot TEXT NOT NULL)"
        )
        # 전송한 주문의 원장. key가 PRIMARY KEY라 같은 주문은 물리적으로 두 번 못 들어간다
        # (멱등성의 최종 방어선 — 게이트를 뚫려도 여기서 막힌다).
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS sent_orders ("
            "key TEXT PRIMARY KEY, ticker TEXT NOT NULL, ord_dt TEXT NOT NULL,"
            "side TEXT NOT NULL, qty INTEGER NOT NULL, limit_price REAL NOT NULL,"
            "excg TEXT, odno TEXT, sent_at TEXT NOT NULL,"
            "status TEXT NOT NULL, filled_qty REAL DEFAULT 0,"
            "reason TEXT DEFAULT '', status_checked_at TEXT)"
        )
        self.conn.commit()
        if ticker:
            self.bind_ticker(ticker)

    # ── 종목 귀속 (설정과 DB가 어긋나는 것을 막는다) ──

    def recorded_ticker(self) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = 'ticker'"
        ).fetchone()
        return row[0] if row else None

    def bind_ticker(self, ticker: str) -> None:
        """DB를 종목에 귀속시킨다. 이미 다른 종목이면 SettingsError."""
        from stock_bot.config import SettingsError

        want = ticker.strip().upper()
        if not want:
            raise SettingsError("설정의 ticker가 비어 있다 — 어느 종목인지 알 수 없다.")
        have = self.recorded_ticker()
        if have is None:
            self.conn.execute(
                "INSERT INTO meta (key, value) VALUES ('ticker', ?)", (want,)
            )
            self.conn.commit()
            return
        if have.strip().upper() != want:
            raise SettingsError(
                f"설정과 DB의 종목이 다르다 — 설정 ticker={want}, DB 기록={have}. "
                "STOCK_DB_PATH와 설정 경로가 어긋났다(다른 종목의 장부에 이 전략이 "
                "돌아가려 하고 있다). STOCK_SETTINGS_PATH를 썼다면 그 값을 먼저 확인하라."
            )

    def load(self) -> tuple[AccountState, list[Order]] | None:
        row = self.conn.execute("SELECT data FROM state WHERE id = 1").fetchone()
        if row is None:
            return None
        blob = json.loads(row[0])
        state = AccountState(**blob["state"])
        pending = [
            Order(
                side=Side(o["side"]),
                limit_price=o["limit_price"],
                qty=o["qty"],
                band=o.get("band", 0.0),
                kind=o.get("kind", "LOC"),
            )
            for o in blob.get("pending", [])
        ]
        return state, pending

    def save(self, state: AccountState, pending: list[Order]) -> None:
        blob = {
            "state": asdict(state),
            "pending": [_order_dict(o) for o in pending],
        }
        self.conn.execute(
            "INSERT INTO state (id, data) VALUES (1, ?) "
            "ON CONFLICT(id) DO UPDATE SET data = excluded.data",
            (json.dumps(blob, ensure_ascii=False),),
        )
        self.conn.commit()

    def append_history(
        self, ts: str, quote_date: str, close: float, snapshot: dict
    ) -> None:
        self.conn.execute(
            "INSERT INTO history (ts, quote_date, close, snapshot) VALUES (?, ?, ?, ?)",
            (ts, quote_date, close, json.dumps(snapshot, ensure_ascii=False)),
        )
        self.conn.commit()

    # ── 전송 주문 원장 (멱등성 + 상태 추적) ──

    def sent_keys(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT key FROM sent_orders")}

    def record_intent(
        self, key: str, ticker: str, ord_dt: str, side: str, qty: int,
        limit_price: float, excg: str, at: str,
    ) -> bool:
        """**전송 전에** 의도를 먼저 커밋한다. 이미 같은 키가 있으면 False.

        2단계 커밋의 1단계다. 이 행이 남고 confirm_sent가 안 불린 채 프로세스가
        죽으면 status='SENDING'으로 남고, 키는 이미 sent_keys에 들어가므로
        다음 run은 IDEMPOTENT 게이트에서 막혀 **재전송하지 않는다.**
        결과는 조회(resolve)로 확인한다.
        """
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO sent_orders "
            "(key, ticker, ord_dt, side, qty, limit_price, excg, odno, sent_at, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, 'SENDING')",
            (key, ticker, ord_dt, side, int(qty), limit_price, excg, at),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def confirm_sent(self, key: str, odno: str, at: str) -> None:
        """2단계 — 전송 성공과 odno를 확정한다."""
        self.conn.execute(
            "UPDATE sent_orders SET status='SENT', odno=?, sent_at=? WHERE key=?",
            (odno, at, key),
        )
        self.conn.commit()

    def mark_send_outcome(self, key: str, status: str, reason: str, at: str) -> None:
        """전송이 실패로 끝났을 때의 기록(불확정 실패 / 접수 거부)."""
        self.conn.execute(
            "UPDATE sent_orders SET status=?, reason=?, status_checked_at=? WHERE key=?",
            (status, reason, at, key),
        )
        self.conn.commit()

    _COLS = ("key", "ticker", "ord_dt", "side", "qty", "limit_price", "excg", "odno", "status")

    def _rows(self, where: str, args: tuple = ()) -> list[dict]:
        rows = self.conn.execute(
            f"SELECT {', '.join(self._COLS)} FROM sent_orders WHERE {where}", args
        ).fetchall()
        return [dict(zip(self._COLS, r)) for r in rows]

    def unresolved_sends(self) -> list[dict]:
        """결과가 확인되지 않은 전송 기록 — 재전송 금지, 조회로 확인할 대상."""
        return self._rows("status IN ('SENDING', 'UNCONFIRMED')")

    def open_sent_orders(self) -> list[dict]:
        """아직 종결되지 않았고 odno가 있는 주문 — 상태 추적 대상."""
        return self._rows(
            "status NOT IN ('FILLED', 'CANCELLED', 'REJECTED', 'SEND_REJECTED') "
            "AND odno != ''"
        )

    def update_sent_status(
        self, key: str, status: str, filled_qty: float, reason: str, checked_at: str
    ) -> None:
        self.conn.execute(
            "UPDATE sent_orders SET status = ?, filled_qty = ?, reason = ?, "
            "status_checked_at = ? WHERE key = ?",
            (status, filled_qty, reason, checked_at, key),
        )
        self.conn.commit()

    def sent_totals_on(self, ord_dt: str) -> tuple[int, float]:
        """그날 전송한 (건수, 총액). 일일 한도 게이트용."""
        row = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(qty * limit_price), 0) "
            "FROM sent_orders WHERE ord_dt = ?",
            (ord_dt,),
        ).fetchone()
        return int(row[0]), float(row[1])

    def closes_by_date(self) -> dict[str, float]:
        """{종가일: 종가} — 취소된 주문이 '체결됐어야 했는지' 판정할 때 쓴다."""
        return {
            d: c
            for d, c in self.conn.execute(
                "SELECT quote_date, close FROM history WHERE quote_date IS NOT NULL"
            )
            if c is not None
        }

    def close(self) -> None:
        self.conn.close()


def _order_dict(o: Order) -> dict:
    return {
        "side": o.side.value,
        "limit_price": o.limit_price,
        "qty": o.qty,
        "band": o.band,
        "kind": o.kind,
    }
