"""주문 전송 게이트 — 전송 직전에 통과해야 하는 관문들.

순수 로직이다(HTTP·DB 없음). 호출자가 사실관계를 모아 `GateContext`로 넘기면
`evaluate()`가 통과/차단을 판정한다. **차단이 기본값이고 통과가 예외다** —
판단에 필요한 사실이 없으면(None) 통과시키지 않는다.

게이트 순서는 '사람의 의사 → 봇의 상태 → 개별 주문' 순이다. 사람이 멈추라고 한
상황에서 다른 계산을 먼저 하지 않는다.

  1. KILL_SWITCH   파일/환경변수로 즉시 차단 (코드 수정·재배포 없이)
  2. DRY_RUN       기본 True. False여야 비로소 '전송 후보'가 된다
  3. EXCHANGE      종목→거래소 매핑이 실계좌로 검증됐는가
  4. BACKLOG       밀린 과거 날짜를 따라잡는 run인가
  5. RECONCILE     봇 장부 보유수량과 KIS 잔고가 일치하는가 (서킷브레이커)
  6. DAILY_LIMIT   이 봇의 1일 전송 건수·금액 상한
  7. ACCOUNT_LIMIT 두 봇 합산 계좌 단위 1일 상한 (공유 원장)
  8. IDEMPOTENT    같은 키가 이미 전송됐는가
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

_EPS = 1e-6


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    gate: str      # 차단한 게이트 이름 (통과 시 "")
    reason: str    # 사람이 읽을 사유

    def label(self) -> str:
        return "전송 허용" if self.allowed else f"[{self.gate}] {self.reason}"


ALLOW = GateDecision(True, "", "모든 게이트 통과")


@dataclass
class GateContext:
    """게이트 판정에 필요한 사실 묶음. 모르는 값은 None으로 둔다(→ 차단된다)."""

    dry_run: bool = True
    kill_switch_path: str | None = None
    kill_switch_env: str | None = None      # 환경변수 '값'(설정돼 있으면 차단)
    # 종목→거래소 매핑이 실계좌 조회로 확인됐는가. None이면 확인 안 한 것 → 차단.
    exchange_verified: bool | None = None
    exchange_detail: str = ""
    # 이 run이 밀린 과거 종가를 따라잡는 중인가
    is_backlog: bool = False
    quote_date: str = ""
    today: str = ""
    # 서킷브레이커 — 봇 장부 vs KIS 잔고
    book_shares: float | None = None
    kis_shares: float | None = None
    # 일일 한도
    sent_today_count: int = 0
    sent_today_amount: float = 0.0
    max_daily_orders: int = 0
    max_daily_amount: float = 0.0
    max_order_amount: float = 0.0
    # 계좌 단위 한도 — 두 봇 합산(공유 원장). 봇별 한도를 통과해도 여기서 다시 막힌다.
    account_sent_count: int = 0
    account_sent_amount: float = 0.0
    max_account_daily_orders: int = 0
    max_account_daily_amount: float = 0.0
    # 멱등성 — 이미 전송된 키 집합(의도만 기록된 SENDING 건도 포함해야 한다)
    sent_keys: set[str] = field(default_factory=set)


def order_key(ticker: str, ord_dt: str, side: str, qty: int, limit_price: float) -> str:
    """멱등성 키. 같은 종목·같은 날·같은 방향·같은 수량·같은 지정가면 같은 주문이다."""
    return f"{ticker.upper()}|{ord_dt}|{side.upper()}|{int(qty)}|{limit_price:.4f}"


def kill_switch_engaged(ctx: GateContext) -> str:
    """걸려 있으면 사유 문자열, 아니면 빈 문자열."""
    if ctx.kill_switch_env is not None and str(ctx.kill_switch_env).strip() != "":
        return "환경변수 STOCK_KILL_SWITCH 가 설정돼 있음"
    if ctx.kill_switch_path and os.path.exists(ctx.kill_switch_path):
        return f"킬스위치 파일 존재: {ctx.kill_switch_path}"
    return ""


def evaluate(
    ticker: str, ord_dt: str, side: str, qty: int, limit_price: float, ctx: GateContext
) -> GateDecision:
    """주문 한 건이 전송 가능한지 판정. 차단이 기본, 통과가 예외."""
    # 1. 킬스위치 — 사람이 멈추라고 한 이상 다른 판단을 하지 않는다
    reason = kill_switch_engaged(ctx)
    if reason:
        return GateDecision(False, "KILL_SWITCH", reason)

    # 2. DRY_RUN
    if ctx.dry_run:
        return GateDecision(False, "DRY_RUN", "DRY_RUN=true — 제안만 생성, 전송하지 않음")

    # 3. 거래소 매핑 검증
    if ctx.exchange_verified is not True:
        detail = ctx.exchange_detail or "검증되지 않음"
        return GateDecision(
            False, "EXCHANGE", f"{ticker} 거래소 매핑 미검증 — {detail}"
        )

    # 4. 백로그 — 과거 날짜 기준 주문을 오늘 시장에 내보내면 안 된다
    if ctx.is_backlog:
        return GateDecision(
            False, "BACKLOG",
            f"밀린 종가({ctx.quote_date}) 처리 중 — 오늘({ctx.today})이 아니므로 전송 보류",
        )

    # 5. 서킷브레이커 — 장부와 실계좌가 어긋나면 그날은 아무것도 내지 않는다
    if ctx.book_shares is None or ctx.kis_shares is None:
        return GateDecision(
            False, "RECONCILE", "장부/잔고 대조 불가(KIS 조회 실패) — 전송 중단"
        )
    if abs(ctx.book_shares - ctx.kis_shares) > _EPS:
        return GateDecision(
            False, "RECONCILE",
            f"장부 {ctx.book_shares:.0f}주 ≠ KIS 잔고 {ctx.kis_shares:.0f}주 — 전송 중단",
        )

    # 6. 일일 한도
    amount = qty * limit_price
    if ctx.max_order_amount <= 0 or amount > ctx.max_order_amount + _EPS:
        return GateDecision(
            False, "DAILY_LIMIT",
            f"주문 금액 ${amount:,.2f} > 1건 상한 ${ctx.max_order_amount:,.2f}",
        )
    if ctx.sent_today_count + 1 > ctx.max_daily_orders:
        return GateDecision(
            False, "DAILY_LIMIT",
            f"오늘 전송 {ctx.sent_today_count}건 — 1일 상한 {ctx.max_daily_orders}건 초과",
        )
    if ctx.sent_today_amount + amount > ctx.max_daily_amount + _EPS:
        return GateDecision(
            False, "DAILY_LIMIT",
            f"오늘 누적 ${ctx.sent_today_amount:,.2f} + ${amount:,.2f} "
            f"> 1일 상한 ${ctx.max_daily_amount:,.2f}",
        )

    # 7. 계좌 단위 한도 — 봇별 한도를 통과해도 계좌는 하나뿐이다
    if ctx.max_account_daily_orders <= 0 or ctx.max_account_daily_amount <= 0:
        return GateDecision(
            False, "ACCOUNT_LIMIT", "계좌 단위 한도가 설정되지 않음 — 전송 보류"
        )
    if ctx.account_sent_count + 1 > ctx.max_account_daily_orders:
        return GateDecision(
            False, "ACCOUNT_LIMIT",
            f"계좌 전체 오늘 {ctx.account_sent_count}건 — "
            f"계좌 1일 상한 {ctx.max_account_daily_orders}건 초과",
        )
    if ctx.account_sent_amount + amount > ctx.max_account_daily_amount + _EPS:
        return GateDecision(
            False, "ACCOUNT_LIMIT",
            f"계좌 전체 오늘 ${ctx.account_sent_amount:,.2f} + ${amount:,.2f} "
            f"> 계좌 1일 상한 ${ctx.max_account_daily_amount:,.2f}",
        )

    # 8. 멱등성 — 의도만 기록되고 결과가 확인 안 된 건(SENDING)도 여기서 막힌다
    key = order_key(ticker, ord_dt, side, qty, limit_price)
    if key in ctx.sent_keys:
        return GateDecision(False, "IDEMPOTENT", f"이미 전송된 주문 — {key}")

    return ALLOW
