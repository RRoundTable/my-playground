"""안전장치 — 회로차단기.

봇은 read-only(주문 전송 코드 없음)이고, 주문 가격/수량은 전략 로직만 산출한다
(외부에서 지정 불가). 그 위에 추가 방어:
  · check_integrity: 검증된 VR 의사코드가 변조되지 않았는지 자가검증
  · check_orders   : 산출된 주문이 불변식을 위반하면 차단
위반 시 GuardError → 그날 실행 중단(잘못된 추천을 내보내느니 멈춘다).
"""
from __future__ import annotations

from stock_bot.domain.models import AccountState, Order, Side, StrategyParams

_EPS = 1e-6


class GuardError(RuntimeError):
    pass


def check_integrity() -> None:
    """검증된 의사코드 벡터(적립식 V9000·Pool1000 → V9350·Pool1250) 재현 확인."""
    from stock_bot.strategies.value_rebalancing import ValueRebalancing

    s = AccountState(shares=0, avg_cost=0, pool=1000, V=9000, cycle_index=1, day_in_cycle=0)
    p = StrategyParams(
        strategy_type="적립식", G=10, cycle_cash_flow=250, max_pool_usage=0.75
    )
    ValueRebalancing().on_cycle_start(s, p)
    if round(s.V, 2) != 9350.0 or round(s.pool, 2) != 1250.0:
        raise GuardError(
            "VR 코어 로직 무결성 검증 실패 — value_rebalancing 변조 의심. 실행 중단."
        )


def check_orders(
    orders: list[Order], state: AccountState, close: float, params: StrategyParams
) -> None:
    """산출된 주문이 안전 불변식을 지키는지 검증. 위반 시 GuardError."""
    if close <= 0:
        raise GuardError(f"종가 비정상: {close}")

    buy_total = 0.0
    for o in orders:
        if o.qty <= 0 or o.qty != int(o.qty):
            raise GuardError(f"주문 수량 비정상: {o.qty} ({o.side.value})")
        # 종가의 0.1~10배를 벗어난 지정가 = 명백한 오류/변조
        if not (close * 0.1 <= o.limit_price <= close * 10):
            raise GuardError(
                f"지정가 ${o.limit_price:.2f}가 종가 ${close:.2f} 대비 비정상 범위"
            )
        if o.side is Side.SELL and o.qty > state.shares + _EPS:
            raise GuardError(f"보유({state.shares:.0f}주) 초과 매도({o.qty}주) 차단")
        if o.side is Side.BUY:
            buy_total += o.qty * o.limit_price

    limit = state.pool * params.max_pool_usage
    if buy_total > limit + _EPS:
        raise GuardError(
            f"매수 총액 ${buy_total:.2f}가 한도(현금 ${state.pool:.2f}×{params.max_pool_usage}=${limit:.2f}) 초과 차단"
        )
