"""계좌 단위 전송 원장 — 두 봇이 공유한다.

QLD와 TSLA는 DB가 갈려 있어 각자 자기 일일 한도만 센다. 그러면 **계좌 기준으로는
상한의 2배까지 나간다.** 계좌는 하나뿐이므로 한도도 계좌 단위로 하나여야 한다.

그래서 봇 DB 바깥에 별도 sqlite 파일을 두고 양쪽이 같은 파일에 기록한다.
두 봇은 09:00 / 09:02로 2분 벌어져 돌지만, 동시 실행을 가정하고 sqlite의
트랜잭션에 맡긴다(키가 PRIMARY KEY라 같은 주문은 두 번 안 들어간다).

집계 기준은 **실제 전송한 달력 날짜**다. 종가일이 아니다 — 계좌 한도는
'하루에 계좌에서 얼마가 나갔나'를 묻는 것이고, 두 봇이 서로 다른 종가일을
처리하는 중일 수도 있기 때문이다.
"""
from __future__ import annotations

import os
import sqlite3


class AccountLedger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=10.0)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS account_sends ("
            "key TEXT PRIMARY KEY, ticker TEXT NOT NULL, sent_date TEXT NOT NULL,"
            "qty INTEGER NOT NULL, limit_price REAL NOT NULL, at TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sent_date ON account_sends (sent_date)"
        )
        self.conn.commit()

    def record(
        self, key: str, ticker: str, sent_date: str, qty: int, limit_price: float, at: str
    ) -> bool:
        """계좌 한도 소진을 기록. 같은 키면 아무것도 쓰지 않고 False.

        **전송 시도 시점에 기록한다**(성공 확인 전). 전송이 실패로 끝나도 한도는
        소진된 것으로 친다 — 한도는 과다 집계 쪽이 안전측이고, 날이 바뀌면 리셋된다.
        """
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO account_sends "
            "(key, ticker, sent_date, qty, limit_price, at) VALUES (?, ?, ?, ?, ?, ?)",
            (key, ticker, sent_date, int(qty), limit_price, at),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def totals_on(self, sent_date: str) -> tuple[int, float]:
        """그날 계좌 전체로 나간 (건수, 총액) — 두 봇 합산."""
        row = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(qty * limit_price), 0) "
            "FROM account_sends WHERE sent_date = ?",
            (sent_date,),
        ).fetchone()
        return int(row[0]), float(row[1])

    def by_ticker_on(self, sent_date: str) -> dict[str, tuple[int, float]]:
        """보고용 — 그날 종목별 (건수, 총액)."""
        rows = self.conn.execute(
            "SELECT ticker, COUNT(*), COALESCE(SUM(qty * limit_price), 0) "
            "FROM account_sends WHERE sent_date = ? GROUP BY ticker",
            (sent_date,),
        ).fetchall()
        return {r[0]: (int(r[1]), float(r[2])) for r in rows}

    def close(self) -> None:
        self.conn.close()
