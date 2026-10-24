"""일일 루틴(순수 로직) + 보고 메시지 포맷."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from stock_bot.broker.kis_client import BrokerSnapshot, Execution
from stock_bot.config import Config
from stock_bot.domain.execution import Fill, reconcile_loc
from stock_bot.domain.guards import GuardError, check_integrity, check_orders
from stock_bot.domain.models import AccountState, Order, StrategyParams
from stock_bot.domain.strategy import Strategy

log = logging.getLogger("stock-bot")


@dataclass
class Drift:
    """KIS 실제 계좌 vs 봇 가정 모델의 차이."""

    shares: float
    cash: float | None

    def is_significant(self) -> bool:
        return abs(self.shares) >= 1 or (self.cash is not None and abs(self.cash) >= 1)


@dataclass
class DailyResult:
    orders: list[Order]
    cycle_started: bool
    fills: list[Fill] = field(default_factory=list)            # 모델 모드 LOC 추정 체결
    real_executions: list[Execution] = field(default_factory=list)  # KIS 모드 실제 체결
    drift: Drift | None = None
    # 전송 경로 결과 — run_once가 채운다. 종료 코드·요약 줄 판정에 쓰인다.
    sends: list[dict] = field(default_factory=list)
    order_statuses: list = field(default_factory=list)
    resolved: list[dict] = field(default_factory=list)
    # {종가일: 종가} — 취소된 주문이 체결됐어야 했는지 판정하는 데 쓴다.
    closes_by_date: dict = field(default_factory=dict)
    # 일지 한 줄용 요약 — run_once가 채운다(journal.make_line이 읽는다).
    record: dict | None = None


def reconcile_account(state: AccountState, snap: BrokerSnapshot, cash_share: float = 1.0) -> Drift:
    """KIS 실제 잔고를 진실의 원천으로 봇 상태를 보정한다. V·사이클은 전략 소유라 유지.

    cash_share: 한 계좌를 여러 종목 봇이 나눠 쓸 때 이 봇 몫의 비율(기본 1.0=단일 운용).
    보유수량(snap.shares)은 KIS 응답에서 이미 종목코드로 필터링된 값이라 그대로 쓴다 —
    현금(snap.cash)만 계좌 전체 예수금이라 분배 비율을 곱해야 이중계산을 막는다.
    """
    shared_cash = snap.cash * cash_share if snap.cash is not None else None
    drift = Drift(
        shares=snap.shares - state.shares,
        cash=(shared_cash - state.pool) if shared_cash is not None else None,
    )
    state.shares = snap.shares
    state.avg_cost = snap.avg_cost
    if shared_cash is not None:
        state.pool = shared_cash
    return drift


def bootstrap_state(cfg: Config) -> AccountState:
    # day_in_cycle=cycle_len → 첫 run이 즉시 첫 사이클을 계산(V를 공식으로 산출).
    return AccountState(
        shares=cfg.init_shares,
        avg_cost=cfg.init_avg_cost,
        pool=cfg.init_pool,
        V=cfg.init_v,
        cycle_index=0,
        day_in_cycle=cfg.cycle_len,
    )


def make_kis_client(cfg: Config):
    """cfg로 KisClient 구성 (토큰 캐시는 db 디렉터리)."""
    import os

    from stock_bot.broker.kis_client import KisClient, KisConfig

    return KisClient(
        KisConfig(
            env=cfg.kis_env,
            app_key=cfg.kis_app_key,
            app_secret=cfg.kis_app_secret,
            cano=cfg.kis_cano,
            acnt_prdt_cd=cfg.kis_acnt_prdt_cd,
            crcy_cd=cfg.kis_crcy_cd,
            token_path=os.path.join(os.path.dirname(cfg.db_path), "kis_token.json"),
        )
    )


def build_gate_context(
    cfg: Config, store, ledger, client, state, broker, drift, quote_date: str, today: str
) -> "order_gate.GateContext":
    """게이트 판정에 필요한 사실을 모은다. **모르는 값은 None으로 둔다 — 차단된다.**

    거래소 검증은 실계좌 조회가 필요하므로 DRY_RUN이 꺼졌을 때만 수행한다
    (DRY_RUN 게이트가 먼저 막으므로 dry run에서는 그 값이 쓰이지 않는다).
    """
    import os

    from stock_bot.broker import order_gate
    from stock_bot.broker.kis_client import EXCHANGE_BY_TICKER

    sent_count, sent_amount = store.sent_totals_on(quote_date)
    acct_count, acct_amount = ledger.totals_on(today)
    ctx = order_gate.GateContext(
        dry_run=cfg.dry_run,
        kill_switch_path=cfg.kill_switch_path,
        kill_switch_env=os.environ.get("STOCK_KILL_SWITCH"),
        is_backlog=_is_backlog(cfg, quote_date),
        quote_date=quote_date,
        today=today,
        # 장부(대조 전 봇이 알던 값) vs KIS. drift가 없으면 대조 자체가 불가 → 차단.
        book_shares=None if drift is None else broker.shares - drift.shares,
        kis_shares=None if broker is None else broker.shares,
        sent_today_count=sent_count,
        sent_today_amount=sent_amount,
        max_daily_orders=cfg.max_daily_orders,
        max_daily_amount=cfg.max_daily_amount,
        max_order_amount=cfg.max_order_amount,
        account_sent_count=acct_count,
        account_sent_amount=acct_amount,
        max_account_daily_orders=cfg.max_account_daily_orders,
        max_account_daily_amount=cfg.max_account_daily_amount,
        sent_keys=store.sent_keys(),
    )
    if not cfg.dry_run and client is not None:
        excg = EXCHANGE_BY_TICKER.get(cfg.ticker.upper())
        if excg is None:
            ctx.exchange_verified = False
            ctx.exchange_detail = f"{cfg.ticker} 거래소 매핑이 EXCHANGE_BY_TICKER에 없음"
        else:
            ok, detail = client.verify_exchange(cfg.ticker, excg, state.last_close or 0.0)
            ctx.exchange_verified, ctx.exchange_detail = ok, detail
    return ctx


def _is_backlog(cfg: Config, quote_date: str) -> bool:
    """이 run이 밀린 과거 종가를 따라잡는 중인가 — 뒤에 미처리 종가가 더 있으면 True."""
    from stock_bot import market

    try:
        return market.next_unprocessed_close(cfg.ticker, quote_date) is not None
    except Exception as e:  # 시세 조회 실패 시 '백로그일 수도 있다'로 보고 막는다
        log.warning("백로그 판정 실패 — 보수적으로 전송 보류: %s", e)
        return True


def send_orders(
    cfg: Config, store, ledger, client, orders, ctx, quote_date: str, today: str
) -> list[dict]:
    """게이트를 통과한 주문만 전송한다. **의도 선기록 → 전송 → 확정**의 2단계 커밋.

    1단계(record_intent)가 커밋된 뒤 2단계(confirm_sent) 전에 프로세스가 죽으면
    status='SENDING'으로 남는다. 키는 이미 원장에 있으므로 다음 run은 IDEMPOTENT
    게이트에서 막혀 **재전송하지 않고**, resolve_unresolved_sends()가 조회로 결과를
    확인한다. 전송은 했는데 기록을 못 남겨 중복 주문이 나가는 구멍을 이렇게 막는다.

    전송 성공 시 ctx를 제자리 갱신해 **같은 run 안에서도** 한도·멱등성이 누적된다.
    """
    from stock_bot.broker import order_gate, order_status
    from stock_bot.broker.kis_client import EXCHANGE_BY_TICKER, KisError

    out: list[dict] = []
    for o in orders:
        side = o.side.value
        decision = order_gate.evaluate(
            cfg.ticker, quote_date, side, o.qty, o.limit_price, ctx
        )
        rec = {"order": o.label(), "sent": False, "gate": decision.gate,
               "reason": decision.reason, "odno": ""}
        if not decision.allowed:
            out.append(rec)
            continue

        excg = EXCHANGE_BY_TICKER[cfg.ticker.upper()]
        key = order_gate.order_key(cfg.ticker, quote_date, side, o.qty, o.limit_price)
        now = datetime.now().isoformat()
        amount = o.qty * o.limit_price

        # ── 1단계: 전송 '전에' 의도를 커밋한다 ──
        if not store.record_intent(key, cfg.ticker, quote_date, side, o.qty,
                                   o.limit_price, excg, now):
            rec.update(gate="IDEMPOTENT", reason=f"이미 기록된 주문 — {key}")
            out.append(rec)
            continue
        # 계좌 한도도 시도 시점에 소진 처리(과다 집계가 안전측)
        ledger.record(key, cfg.ticker, today, o.qty, o.limit_price, now)
        ctx.sent_keys.add(key)
        ctx.sent_today_count += 1
        ctx.sent_today_amount += amount
        ctx.account_sent_count += 1
        ctx.account_sent_amount += amount

        # ── 전송 ──
        try:
            odno = client.send_order(cfg.ticker, excg, side, o.qty, o.limit_price, decision)
        except KisError as e:
            msg = str(e)
            # 접수 거부(rt_cd≠0)는 주문이 존재하지 않음이 확정이다. 그 외(타임아웃 등)는
            # 나갔는지 알 수 없으므로 UNCONFIRMED로 두고 절대 재전송하지 않는다.
            definite = "주문 거부" in msg
            status = order_status.SEND_REJECTED if definite else order_status.UNCONFIRMED
            store.mark_send_outcome(key, status, msg, datetime.now().isoformat())
            rec.update(gate=status, reason=msg)
            log.warning("주문 전송 실패(%s): %s", status, msg)
            out.append(rec)
            continue

        # ── 2단계: 확정 ──
        store.confirm_sent(key, odno, datetime.now().isoformat())
        rec.update(sent=True, gate="", reason="전송됨", odno=odno)
        out.append(rec)
    return out


def resolve_unresolved_sends(cfg: Config, store, client, quote_date: str) -> list[dict]:
    """결과가 확인되지 않은 전송 기록을 **재전송 없이 조회로** 확인한다.

    status가 SENDING(확정 전 중단) 또는 UNCONFIRMED(불확정 실패)인 건들이다.
    odno가 없으므로 종목·주문일·방향·수량·지정가로 체결내역에서 찾아 맞춘다.
    못 찾으면 그대로 두고 크게 보고한다 — '안 보이니 안 나갔다'로 단정하면
    실제로 살아 있는 주문을 놓친다.
    """
    from stock_bot.broker import order_status
    from stock_bot.broker.kis_client import KisError

    pending = store.unresolved_sends()
    if not pending or client is None:
        return []
    start = min(r["ord_dt"] for r in pending).replace("-", "")
    try:
        rows = client.order_history(start, quote_date.replace("-", ""))
    except KisError as e:
        log.warning("미확인 전송 조회 실패 — 다음 run으로 미룸: %s", e)
        return [{"key": r["key"], "resolved": False, "status": r["status"],
                 "note": f"조회 실패: {e}"} for r in pending]

    now = datetime.now().isoformat()
    out: list[dict] = []
    for row in pending:
        match = order_status.match_unconfirmed(row, rows)
        if match is None:
            store.mark_send_outcome(
                row["key"], order_status.UNCONFIRMED,
                "체결내역에서 찾지 못함 — 사람이 KIS에서 직접 확인할 것", now)
            out.append({"key": row["key"], "resolved": False,
                        "status": order_status.UNCONFIRMED,
                        "note": "체결내역에 없음 — 재전송하지 않음. 사람 확인 필요"})
            continue
        st = order_status.classify(match)
        store.conn.execute(
            "UPDATE sent_orders SET odno=?, status=?, filled_qty=?, reason=?, "
            "status_checked_at=? WHERE key=?",
            (st.odno, st.status, st.filled_qty, st.reason, now, row["key"]),
        )
        store.conn.commit()
        out.append({"key": row["key"], "resolved": True, "status": st.status,
                    "note": f"조회로 확인됨 — {st.label()}"})
    return out


def track_sent_orders(store, client, quote_date: str) -> list:
    """전송했던 주문들의 생사를 확인한다 — 체결/부분/미체결/취소/거부.

    `parse_executions`는 체결수량>0만 담으므로 취소·거부는 그쪽으로 절대 안 잡힌다.
    여기서 CCLD_NCCS_DVSN=00(전체)으로 따로 조회해 메운다.
    """
    from stock_bot.broker import order_status
    from stock_bot.broker.kis_client import KisError

    open_rows = store.open_sent_orders()
    if not open_rows or client is None:
        return []
    start = min(r["ord_dt"] for r in open_rows).replace("-", "")
    end = quote_date.replace("-", "")
    try:
        rows = client.order_history(start, end)
    except KisError as e:
        log.warning("주문 상태 조회 실패 — 상태 갱신 보류: %s", e)
        return []
    statuses = order_status.reconcile(open_rows, rows)
    now = datetime.now().isoformat()
    for row, st in zip(open_rows, statuses):
        store.update_sent_status(row["key"], st.status, st.filled_qty, st.reason, now)
    return statuses


def _broker_snapshot(cfg: Config, quote_date: str) -> BrokerSnapshot | None:
    """KIS 실계좌 스냅샷. 실패 시 None(그날은 가정 모델로 폴백)."""
    from stock_bot.broker.kis_client import KisError

    try:
        client = make_kis_client(cfg)
        end = quote_date.replace("-", "")
        start = (datetime.strptime(quote_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y%m%d")
        return client.snapshot(cfg.ticker, start, end)
    except KisError as e:
        log.warning("KIS 조회 실패 — 가정 모델로 폴백: %s", e)
        return None


def run_once(cfg: Config) -> tuple[str | None, "DailyResult | None"]:
    """하루치 계산+영속화 후 (보고텍스트, 결과)를 반환. 신규 종가 없으면 (None, None).

    CLI-driven 진입점 — nanoclaw가 호출해 결과를 받아 자기 채널로 보고한다.
    """
    from stock_bot import market
    from stock_bot.config import params_for
    from stock_bot.state.store import Store
    from stock_bot.strategies.registry import get_strategy

    params = params_for(
        cfg.vr_mode, cfg.band_rate, cfg.sell_bands, cfg.cycle_cash_flow, cfg.G, cfg.max_pool_usage
    )
    strat = get_strategy(cfg.strategy_engine)
    store = Store(cfg.db_path, cfg.ticker)
    try:
        loaded = store.load()
        if loaded is None:
            state, pending = bootstrap_state(cfg), []
            log.info("부트스트랩: V=%.2f pool=%.2f", state.V, state.pool)
        else:
            state, pending = loaded

        # 밀린 종가는 오래된 것부터 하루에 하나씩. 여러 날 밀렸으면 run을 그만큼 돌린다.
        nxt = market.next_unprocessed_close(cfg.ticker, state.last_quote_date)
        if nxt is None:
            return None, None
        quote_date, close = nxt
        broker = _broker_snapshot(cfg, quote_date) if cfg.kis_enabled else None
        result = run_daily(cfg, params, strat, state, pending, quote_date, close, broker)
        if result is None:
            return None, None

        # 순서가 중요하다: (1) 결과 미확인 전송분을 조회로 먼저 확정하고,
        # (2) 살아 있는 주문의 생사를 갱신한 뒤, (3) 오늘 주문 전송을 시도한다.
        from stock_bot.state.account_ledger import AccountLedger

        client = make_kis_client(cfg) if cfg.kis_enabled else None
        today = datetime.now().strftime("%Y-%m-%d")
        ledger = AccountLedger(cfg.account_ledger_path)
        try:
            resolved = resolve_unresolved_sends(cfg, store, client, quote_date)
            statuses = track_sent_orders(store, client, quote_date)
            ctx = build_gate_context(
                cfg, store, ledger, client, state, broker, result.drift, quote_date, today
            )
            sends = send_orders(
                cfg, store, ledger, client, result.orders, ctx, quote_date, today
            )
        finally:
            ledger.close()

        store.save(state, result.orders)
        store.append_history(
            datetime.now().isoformat(),
            quote_date,
            close,
            {
                "source": "kis" if cfg.kis_enabled else "model",
                "V": state.V,
                "pool": state.pool,
                "shares": state.shares,
                "avg_cost": state.avg_cost,
                "orders": [o.__dict__ for o in result.orders],
                "real_executions": [e.__dict__ for e in result.real_executions],
                "drift": result.drift.__dict__ if result.drift else None,
                "sends": sends,
                "resolved": resolved,
                "order_statuses": [s.__dict__ for s in statuses],
            },
        )
        result.sends, result.order_statuses, result.resolved = sends, statuses, resolved
        # 방금 처리한 종가도 포함해야 한다 — 어제 낸 주문의 판정 근거가 바로 이 값이다.
        result.closes_by_date = {**store.closes_by_date(), quote_date: close}
        result.record = {
            "ticker": cfg.ticker, "quote_date": quote_date, "close": close,
            "pool": state.pool, "V": state.V,
            "band_low": state.V * (1 - params.band_rate),
            "band_high": state.V * (1 + params.band_rate),
            "shares": state.shares, "cycle_started": result.cycle_started,
            "source": "KIS" if cfg.kis_enabled else "모델",
            "orders": [o.label() for o in result.orders],
            "sent": [s_["order"] for s_ in sends if s_.get("sent")],
            "blocked_gate": next((s_["gate"] for s_ in sends if not s_.get("sent")), None),
            "notes": attention_notes(result),
        }
        report = build_report(cfg, params, state, result, close, quote_date)
        body = report + _send_section(cfg, sends, statuses, resolved)
        return summary_line(cfg, state, result, quote_date, close) + "\n" + body, result
    finally:
        store.close()


def run_daily(
    cfg: Config,
    params: StrategyParams,
    strat: Strategy,
    state: AccountState,
    pending: list[Order],
    quote_date: str,
    close: float,
    broker: BrokerSnapshot | None = None,
) -> DailyResult | None:
    """상태를 제자리 갱신하고 결과를 반환. 같은 종가 날짜면 None(휴장/중복).

    broker가 주어지면 KIS 실제 계좌가 진실의 원천(가정 모델 대체).
    """
    if state.last_quote_date == quote_date:
        return None

    check_integrity()  # 코어 로직 변조 차단 (회로차단기)

    fills: list[Fill] = []
    real_executions: list[Execution] = []
    drift: Drift | None = None
    if broker is not None:
        drift = reconcile_account(state, broker, cfg.kis_cash_share)
        real_executions = broker.executions
    else:
        # 유령 체결 차단 — 실주문 모드에서는 가정 모델로 장부를 움직이지 않는다.
        # KIS를 못 읽는 날 묵은 pending을 종가와 대조해 체결 처리하면, 실계좌에 없는
        # 주식이 장부에 생기고 그 장부로 다음 주문을 낸다.
        if not cfg.dry_run:
            raise GuardError(
                "실주문 모드(DRY_RUN=false)에서 KIS 조회 실패 — "
                "가정 모델 폴백(reconcile_loc)은 금지. 오늘은 중단한다."
            )
        fills = reconcile_loc(state, pending, close)

    state.last_close = close
    state.last_quote_date = quote_date

    state.day_in_cycle += 1
    cycle_started = False
    if state.day_in_cycle >= cfg.cycle_len:
        strat.on_cycle_start(state, params)
        state.day_in_cycle = 0
        cycle_started = True

    orders = strat.daily_orders(state, params, close)
    check_orders(orders, state, close, params)  # 주문 불변식 (회로차단기)
    return DailyResult(
        orders=orders,
        cycle_started=cycle_started,
        fills=fills,
        real_executions=real_executions,
        drift=drift,
    )


def build_report(
    cfg: Config,
    params: StrategyParams,
    state: AccountState,
    result: DailyResult,
    close: float,
    quote_date: str,
) -> str:
    lines: list[str] = []
    lines.append(
        f"📊 **{cfg.ticker} VR ({params.strategy_type})** — {quote_date} 종가 ${close:.2f}"
    )
    if result.cycle_started:
        lines.append(
            f"🔄 **새 사이클 #{state.cycle_index} 시작** · 가이드밸류 V=${state.V:,.2f}"
        )
    lines.append("")
    lines.append(
        f"• 보유 {state.shares:.0f}주 · 평단 ${state.avg_cost:.2f} "
        f"· 평가금 ${state.shares * close:,.2f}"
    )
    low, high = state.V * (1 - params.band_rate), state.V * (1 + params.band_rate)
    lines.append(
        f"• 현금풀 ${state.pool:,.2f} · V=${state.V:,.2f} "
        f"· 밴드 [${low:,.0f} ~ ${high:,.0f}]"
    )
    src = "KIS 실계좌" if cfg.kis_enabled else "가정 모델"
    if cfg.kis_enabled and cfg.kis_cash_share < 1.0:
        src += f" (계좌 현금의 {cfg.kis_cash_share * 100:.0f}%만 이 봇 몫)"
    lines.append(
        f"• 사이클 #{state.cycle_index} · {state.day_in_cycle}/{cfg.cycle_len}일차 · 출처: {src}"
    )

    if result.drift and result.drift.is_significant():
        d = result.drift
        cash = f" · 현금 {d.cash:+,.2f}" if d.cash is not None else ""
        lines.append(f"⚠️ 모델 대비 보정: 주식 {d.shares:+.0f}주{cash} (KIS 기준으로 정정됨)")

    if result.real_executions:
        lines.append("")
        lines.append("**실제 체결 (KIS)**")
        lines.extend(f"  ✅ {e.label()}" for e in result.real_executions)
    elif result.fills:
        lines.append("")
        lines.append("**전일 체결 (모델 추정)**")
        lines.extend(f"  ✅ {f.label()}" for f in result.fills)

    lines.append("")
    if result.orders:
        head = "**오늘 밤 걸 LOC 주문**"
        lines.append(head if not cfg.dry_run else head + " (직접 예약하세요)")
        lines.extend(f"  • {o.label()}" for o in result.orders)
    else:
        lines.append("**오늘 밤 주문 없음** — 밴드 내부 홀딩")

    return "\n".join(lines)


def send_anomalies(result: "DailyResult") -> list[str]:
    """**결과를 모른 채 넘어가면 안 되는** 것들. 종료 코드 10의 판정 근거다.

    게이트 차단(DRY_RUN·BACKLOG·한도 등)은 설계된 동작이라 이상이 아니다.
    여기 잡히는 것은 '주문을 냈는데 어떻게 됐는지 모른다' 뿐이다.
    """
    from stock_bot.broker import order_status

    bad = []
    for s in result.sends:
        if s.get("gate") in (order_status.UNCONFIRMED, order_status.SEND_REJECTED):
            bad.append(f"전송 {s['gate']}: {s['order']}")
    for st in result.order_statuses:
        if st.status == order_status.UNKNOWN:
            bad.append(f"상태 UNKNOWN: {st.label()}")
        elif st.status in (order_status.CANCELLED, order_status.REJECTED):
            # LOC이 조건을 못 채워 안 붙은 것은 정상이다. 그걸 매번 이상으로 올리면
            # 알림이 일상이 되고, 일상이 된 알림은 아무도 안 본다.
            # **조건을 채웠는데 0주**인 것만 이상이다(2026-07-15 QLD 건이 그랬다).
            close = result.closes_by_date.get(order_status.quote_date_of(st.ord_dt))
            possible = order_status.fill_was_possible(st, close)
            if possible is True:
                bad.append(
                    f"{st.status} — 종가 ${close:,.2f}로 체결 조건을 충족했는데 "
                    f"체결 0주: {st.label()}")
            elif possible is None:
                bad.append(
                    f"{st.status} — 그날 종가를 몰라 정상 여부 판정 불가: {st.label()}")
    for r in result.resolved:
        if not r.get("resolved"):
            bad.append(f"미확인 전송분 확인 실패: {r.get('key')}")
    return bad


def attention_notes(result: "DailyResult") -> list[str]:
    """사람이 봐야 하는 것 — 이상(send_anomalies)에 더해 '주문이 죽었다'까지 포함.

    취소·거부는 결과가 확정됐으므로 이상은 아니지만, 그냥 지나치면 매일 조용히
    아무것도 못 사는 상태가 정상으로 보인다. 요약 줄에는 반드시 띄운다.
    """
    from stock_bot.broker import order_status

    notes = list(send_anomalies(result))
    seen = " ".join(notes)
    for st in result.order_statuses:
        if st.status in (order_status.CANCELLED, order_status.REJECTED) \
                and st.label() not in seen:
            notes.append(f"주문 {st.status}(정상 범위 — 체결 조건 미충족): {st.label()}")
    return notes


def summary_line(cfg: Config, state, result: "DailyResult", quote_date: str, close: float) -> str:
    """첫 줄 한 줄 요약 — cron이 그대로 보내도 사람이 훑어서 판단할 수 있게."""
    notes = attention_notes(result)
    tag = "ATTENTION" if notes else "OK"
    sent = [s for s in result.sends if s.get("sent")]
    blocked = [s for s in result.sends if not s.get("sent")]
    parts = [f"{cfg.ticker} {quote_date} 종가 ${close:,.2f}"]
    if sent:
        parts.append("전송 " + ", ".join(f"{s['order']}→odno {s['odno']}" for s in sent))
    elif blocked:
        parts.append(f"전송 0건(차단 [{blocked[0]['gate']}])")
    else:
        parts.append("주문 없음")
    parts.append(f"보유 {state.shares:.0f}주 · 풀 ${state.pool:,.2f}")
    if notes:
        parts.append("⚠ " + " / ".join(notes))
    return f"[{tag}] " + " · ".join(parts)


def _send_section(
    cfg: Config, sends: list[dict], statuses: list, resolved: list[dict] | None = None
) -> str:
    """전송 시도 결과와 기존 주문 상태를 보고 끝에 덧붙인다.

    차단도 반드시 보이게 쓴다 — 조용히 안 나가는 것이 가장 위험하다.
    """
    lines: list[str] = []
    for r in resolved or []:
        if not lines:
            lines.extend(["", "**결과 미확인 전송분 확인**"])
        mark = "✅" if r["resolved"] else "❓"
        lines.append(f"  {mark} {r['key']} — {r['note']}")
    if statuses:
        lines.append("")
        lines.append("**전송했던 주문 상태**")
        lines.extend(f"  · {s.label()}" for s in statuses)
    if sends:
        lines.append("")
        mode = "DRY_RUN" if cfg.dry_run else "전송 모드"
        lines.append(f"**주문 전송 ({mode})**")
        for s in sends:
            if s["sent"]:
                lines.append(f"  ✅ 전송됨 {s['order']} · 주문번호 {s['odno']}")
            else:
                lines.append(f"  ⛔ 차단 [{s['gate']}] {s['order']} — {s['reason']}")
    return "\n".join(lines)
