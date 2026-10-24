"""백테스트 — 과거 종가로 VR 전략을 재생해 성과를 계산한다.

읽기 전용/순수 시뮬레이션: db·KIS를 건드리지 않는다. 실제 일일 루틴(service.run_daily)과
같은 코어(execution.reconcile_loc / strategy.on_cycle_start / daily_orders / guards)를 재사용해
"검증된 로직을 그대로 과거에 재생"하는 것을 보장한다.

성과지표는 DCA(분할입금) 구조라 단순 CAGR이 아니라 자금가중수익률(XIRR)로 연환산하고,
최대낙폭(MDD)도 VR 전략과 벤치마크(같은 현금흐름의 단순 매수후보유) 양쪽에 계산한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from stock_bot.config import params_for
from stock_bot.domain.execution import reconcile_loc
from stock_bot.domain.guards import GuardError, check_integrity, check_orders
from stock_bot.domain.models import AccountState, Order
from stock_bot.strategies.registry import get_strategy


@dataclass
class DayRow:
    date: str
    close: float
    V: float
    pool: float
    shares: float
    avg_cost: float
    equity: float             # pool + shares*close (VR 전략 총자산)
    buy_hold_equity: float    # 같은 현금흐름을 그대로 매수후보유했을 때 총자산
    cycle_started: bool
    orders: list[Order] = field(default_factory=list)


@dataclass
class BacktestResult:
    ticker: str
    vr_mode: str
    rows: list[DayRow]
    total_invested: float
    final_equity: float
    buy_hold_equity: float
    vr_annualized_return: float       # XIRR (자금가중, 연환산)
    vr_max_drawdown: float            # 0.23 == -23%
    buy_hold_annualized_return: float
    buy_hold_max_drawdown: float
    blocked: str | None = None  # GuardError 사유 — 있으면 그 지점에서 재생 중단


def _xirr(cash_flows: list[tuple[float, str]]) -> float:
    """자금가중수익률(연환산). cash_flows=[(금액, 'YYYY-MM-DD'), ...], 유입은 -, 최종평가는 +."""
    if len(cash_flows) < 2:
        return 0.0
    d0 = date.fromisoformat(cash_flows[0][1])
    years = [(date.fromisoformat(d) - d0).days / 365.0 for _, d in cash_flows]
    amounts = [amt for amt, _ in cash_flows]

    def npv(rate: float) -> float:
        return sum(amt / ((1 + rate) ** t) for amt, t in zip(amounts, years))

    lo, hi = -0.99, 10.0
    if npv(lo) * npv(hi) > 0:
        return 0.0  # 부호가 안 바뀌면(전액 손실 등) 수렴 불가 — 안전하게 0 반환
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _max_drawdown(equity_series: list[float]) -> float:
    """최대낙폭(양수 분수, 0.23 == -23%). 시계열 고점 대비 최대 하락폭."""
    peak = float("-inf")
    mdd = 0.0
    for eq in equity_series:
        peak = max(peak, eq)
        if peak > 0:
            mdd = max(mdd, (peak - eq) / peak)
    return mdd


def run_backtest(
    ticker: str,
    closes: list[tuple[str, float]],
    vr_mode: str = "적립식",
    strategy_engine: str = "value_rebalancing",
    cycle_len: int = 10,
    band_rate: float = 0.15,
    sell_bands: tuple[float, ...] = (0.025, 0.05, 0.075, 0.10),
    cycle_cash_flow: float | None = None,
    G: float | None = None,
    max_pool_usage: float | None = None,
    init_v: float = 0.0,
    init_pool: float = 1000.0,
    init_shares: float = 0.0,
    init_avg_cost: float = 0.0,
) -> BacktestResult:
    """과거 종가(오름차순)를 하루씩 재생한다. 벤치마크는 같은 현금흐름의 단순 매수후보유(DCA).

    실전(run_daily)과 달리 KIS 조회가 없으므로 매수/매도는 항상 전일 LOC 가정 체결로 재구성된다.
    G/max_pool_usage는 vr_mode 프리셋값을 파라미터 튜닝용으로 덮어쓸 때만 지정한다.
    """
    if not closes:
        raise ValueError("종가 이력이 비어 있음")

    check_integrity()
    params = params_for(vr_mode, band_rate, sell_bands, cycle_cash_flow, G, max_pool_usage)
    strat = get_strategy(strategy_engine)
    state = AccountState(
        shares=init_shares,
        avg_cost=init_avg_cost,
        pool=init_pool,
        V=init_v,
        cycle_index=0,
        day_in_cycle=cycle_len,  # 첫날 즉시 사이클 시작 (bootstrap_state와 동일 관례)
    )

    pending: list[Order] = []
    rows: list[DayRow] = []
    total_invested = init_pool
    buy_hold_shares = init_pool / closes[0][1] if closes[0][1] > 0 else 0.0
    cash_flows: list[tuple[float, str]] = [(-init_pool, closes[0][0])]
    blocked: str | None = None

    for date_str, close in closes:
        if close is None or close <= 0:
            continue

        reconcile_loc(state, pending, close)
        state.last_close = close
        state.last_quote_date = date_str

        state.day_in_cycle += 1
        cycle_started = False
        if state.day_in_cycle >= cycle_len:
            if params.cycle_cash_flow > 0:
                total_invested += params.cycle_cash_flow
                buy_hold_shares += params.cycle_cash_flow / close
                cash_flows.append((-params.cycle_cash_flow, date_str))
            strat.on_cycle_start(state, params)
            state.day_in_cycle = 0
            cycle_started = True

        equity = state.pool + state.shares * close
        bh_equity = buy_hold_shares * close

        try:
            orders = strat.daily_orders(state, params, close)
            check_orders(orders, state, close, params)
        except GuardError as e:
            blocked = str(e)
            rows.append(
                DayRow(date_str, close, state.V, state.pool, state.shares,
                       state.avg_cost, equity, bh_equity, cycle_started)
            )
            break

        pending = orders
        rows.append(
            DayRow(date_str, close, state.V, state.pool, state.shares,
                   state.avg_cost, equity, bh_equity, cycle_started, orders)
        )

    if not rows:
        raise ValueError("유효한 종가가 없어 재생할 수 없음")

    last = rows[-1]
    vr_flows = cash_flows + [(last.equity, last.date)]
    bh_flows = cash_flows + [(last.buy_hold_equity, last.date)]

    return BacktestResult(
        ticker=ticker,
        vr_mode=vr_mode,
        rows=rows,
        total_invested=total_invested,
        final_equity=last.equity,
        buy_hold_equity=last.buy_hold_equity,
        vr_annualized_return=_xirr(vr_flows),
        vr_max_drawdown=_max_drawdown([row.equity for row in rows]),
        buy_hold_annualized_return=_xirr(bh_flows),
        buy_hold_max_drawdown=_max_drawdown([row.buy_hold_equity for row in rows]),
        blocked=blocked,
    )


def format_report(result: BacktestResult) -> str:
    r = result
    if not r.rows:
        return f"📉 {r.ticker} 백테스트: 유효한 종가 없음"

    first, last = r.rows[0], r.rows[-1]
    n_buy = sum(1 for row in r.rows for o in row.orders if o.side.value == "BUY")
    n_sell = sum(1 for row in r.rows for o in row.orders if o.side.value == "SELL")
    vr_ret = (r.final_equity / r.total_invested - 1) * 100 if r.total_invested > 0 else 0.0
    bh_ret = (r.buy_hold_equity / r.total_invested - 1) * 100 if r.total_invested > 0 else 0.0

    lines = [
        f"📉 **{r.ticker} VR 백테스트 ({r.vr_mode})** — {first.date} ~ {last.date} ({len(r.rows)}거래일)",
        f"• 총 투입원금 ${r.total_invested:,.2f} (시드 + 사이클별 입금)",
        f"• VR 전략: 최종 ${r.final_equity:,.2f} (누적 {vr_ret:+.1f}% · 연환산 {r.vr_annualized_return * 100:+.1f}%/년 · 최대낙폭 -{r.vr_max_drawdown * 100:.1f}%)",
        f"• 매수후보유(같은 현금흐름): 최종 ${r.buy_hold_equity:,.2f} (누적 {bh_ret:+.1f}% · 연환산 {r.buy_hold_annualized_return * 100:+.1f}%/년 · 최대낙폭 -{r.buy_hold_max_drawdown * 100:.1f}%)",
        f"• 매매 {n_buy}회 매수 · {n_sell}회 매도 주문 (LOC 가정 체결)",
        f"• 종료 시점: 보유 {last.shares:.0f}주 · 평단 ${last.avg_cost:.2f} · V=${last.V:,.2f} · 현금풀 ${last.pool:,.2f}",
    ]
    if r.blocked:
        lines.append(f"🛑 안전장치 발동으로 {last.date}에 재생 중단: {r.blocked}")
    return "\n".join(lines)
