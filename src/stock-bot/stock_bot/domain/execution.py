"""LOC(종가) 체결 재구성 — KIS 없이 전일 종가만으로 결정적 체결 판정."""
from __future__ import annotations

from dataclasses import dataclass

from stock_bot.domain.models import AccountState, Order, Side


@dataclass
class Fill:
    order: Order
    fill_price: float

    def label(self) -> str:
        return f"{self.order.side.value} {self.order.qty}주 @ ${self.fill_price:.2f} 체결"


def reconcile_loc(state: AccountState, pending: list[Order], close: float) -> list[Fill]:
    """전일 LOC 주문이 종가로 체결됐는지 판정하고 상태를 제자리 갱신한다.

    LOC 규칙: 매수는 종가 ≤ 지정가, 매도는 종가 ≥ 지정가일 때 *종가에* 체결된다.
    체결은 Pool ↔ 주식 이동일 뿐이라 총자산이 보존된다.
    """
    fills: list[Fill] = []
    for o in pending:
        if o.side is Side.BUY and close <= o.limit_price:
            qty = min(o.qty, _affordable(state.pool, close))
            if qty <= 0:
                continue
            new_shares = state.shares + qty
            state.avg_cost = (
                (state.avg_cost * state.shares + close * qty) / new_shares
                if new_shares > 0
                else 0.0
            )
            state.shares = new_shares
            state.pool -= close * qty
            fills.append(Fill(Order(o.side, o.limit_price, qty, o.band), close))
        elif o.side is Side.SELL and close >= o.limit_price:
            qty = min(o.qty, int(state.shares))
            if qty <= 0:
                continue
            state.shares -= qty
            state.pool += close * qty
            fills.append(Fill(Order(o.side, o.limit_price, qty, o.band), close))
    return fills


def _affordable(pool: float, price: float) -> int:
    return int(pool // price) if price > 0 else 0
