"""라오어 밸류 리밸런싱(VR) — 적립식/거치식/인출식.

사용자가 검증한 의사코드를 그대로 구현한다 (F번 = 변형1).

2층 구조:
  · 층1 on_cycle_start: 사이클(2주)마다 V·Pool 갱신 (검증 의사코드)
  · 층2 daily_orders:   V 기준으로 오늘 밤 걸 LOC 매수/매도 주문 산출
"""
from __future__ import annotations

from stock_bot.domain.models import AccountState, Order, Side, StrategyParams
from stock_bot.domain.strategy import Strategy


class ValueRebalancing(Strategy):
    name = "value_rebalancing"

    def on_cycle_start(self, state: AccountState, p: StrategyParams) -> None:
        # F번 = 변형1: 기울기(Pool/G)는 입금 반영 *전* Pool 사용.
        pool_for_slope = (
            state.pool + p.cycle_cash_flow
            if p.slope_uses_post_cashflow_pool
            else state.pool
        )
        v_next = state.V + (pool_for_slope / p.G) + p.cycle_cash_flow
        # '현금:적립금 5:5' — 입금액은 Pool로(+CCF), 동시에 V도 같은 액수만큼(+CCF) 상향.
        state.pool = max(0.0, state.pool + p.cycle_cash_flow)  # 인출식 음수 가드
        state.V = v_next
        state.cycle_index += 1

    def daily_orders(
        self, state: AccountState, p: StrategyParams, close: float
    ) -> list[Order]:
        if close <= 0:
            return []
        H, V = state.shares, state.V
        evaluation = H * close
        max_band = V * (1 + p.band_rate)
        min_band = V * (1 - p.band_rate)
        orders: list[Order] = []

        if H > 0 and evaluation > max_band:
            # 상한 돌파(과열) → 중심(V)까지 되돌리는 금액을 매도 사다리로 분산.
            # 균등-달러 분할은 시트 공식 확보 전 기본값 (SPEC '가정' 참조).
            sell_amount = evaluation - V
            tranche = sell_amount / len(p.sell_bands)
            for band in p.sell_bands:
                price = round(V * (1 + band) / H, 2)
                qty = int(tranche / price) if price > 0 else 0
                if qty > 0:
                    orders.append(Order(Side.SELL, price, qty, band))
        elif evaluation < min_band:
            # 하한 붕괴(침체) → 가용 현금 한도 내에서 V까지 채우는 LOC 매수.
            required = V - evaluation
            available = state.pool * p.max_pool_usage
            actual = max(0.0, min(required, available))
            # 수량은 실제 주문가(반올림된 지정가) 기준으로 계산해야 한다.
            # close로 나눈 뒤 지정가를 반올림하면(위로 반올림 시) qty*지정가가 available을
            # 넘어서 check_orders의 매수한도 가드가 오작동(정상 주문을 오탐 차단)할 수 있다.
            limit_price = round(close, 2)
            qty = int(actual / limit_price) if limit_price > 0 else 0
            if qty > 0:
                orders.append(Order(Side.BUY, limit_price, qty, 0.0))
        # 밴드 내부면 주문 없음(홀딩).
        return orders
