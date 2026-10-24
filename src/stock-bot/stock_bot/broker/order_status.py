"""전송한 주문의 최종 상태 판정 — 체결/부분체결/미체결/취소/거부.

`parse_executions`는 **체결수량>0만** 담기 때문에, 주문이 취소·거부·미체결로 죽으면
아무 흔적도 남지 않는다. 2026-07-15 QLD 건(주문 7주·체결 0주)이 정확히 그 경우다.
이 모듈이 그 빈 구멍을 메운다 — `CCLD_NCCS_DVSN=00`(전체)으로 받은 행을 그대로 읽어
odno 기준으로 상태를 확정한다.

순수 파서다. HTTP·DB 없음.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── 체결내역 조회로 판정되는 상태 ──
FILLED = "FILLED"
PARTIAL = "PARTIAL"
UNFILLED = "UNFILLED"
CANCELLED = "CANCELLED"
REJECTED = "REJECTED"
UNKNOWN = "UNKNOWN"

# ── 전송 과정에서 기록되는 상태 (2단계 커밋) ──
SENDING = "SENDING"            # 의도만 기록됨. 전송 결과 미확인 — 여기서 죽으면 이 상태로 남는다
SENT = "SENT"                  # 전송 확정, odno 확보
UNCONFIRMED = "UNCONFIRMED"    # 전송 시도가 불확정 실패(타임아웃 등). 나갔는지 알 수 없다
SEND_REJECTED = "SEND_REJECTED"  # KIS가 접수 자체를 거부(rt_cd≠0). 주문은 존재하지 않는다

# 결과가 확인되지 않은 상태 — 재전송하면 안 되고, 조회로 확인해야 한다
NEEDS_LOOKUP = (SENDING, UNCONFIRMED)

# 종료 상태 — 더 추적할 필요가 없다
TERMINAL = (FILLED, CANCELLED, REJECTED, SEND_REJECTED)

_CANCEL_HINTS = ("취소", "정정취소", "거부", "반대매매")


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class OrderStatus:
    odno: str
    symbol: str
    ord_dt: str
    ord_qty: float
    filled_qty: float
    status: str
    reason: str = ""      # 거부·취소 사유 원문
    side: str = ""        # BUY / SELL
    limit_price: float = 0.0

    def label(self) -> str:
        base = f"{self.symbol} {self.ord_dt} #{self.odno} {self.status} {self.filled_qty:.0f}/{self.ord_qty:.0f}주"
        return f"{base} ({self.reason})" if self.reason else base


def classify(row: dict) -> OrderStatus:
    """inquire-ccnl(CCLD_NCCS_DVSN=00) 행 한 개 → 상태.

    판정 순서가 중요하다. 체결수량이 있으면 그게 사실이고, 0일 때만 거부·취소 사유를
    본다 — 부분체결 후 잔량 취소된 주문을 '취소'로 뭉개면 보유수량이 어긋난다.
    """
    ord_qty = _f(row.get("ft_ord_qty") or row.get("ord_qty"))
    filled = _f(row.get("ft_ccld_qty") or row.get("ccld_qty"))
    reason = (row.get("rjct_rson_name") or row.get("rjct_rson") or "").strip()
    stat = (row.get("prcs_stat_name") or row.get("ord_stat_name") or "").strip()
    st = OrderStatus(
        odno=(row.get("odno") or "").strip(),
        symbol=(row.get("ovrs_pdno") or row.get("pdno") or "").upper(),
        ord_dt=(row.get("ord_dt") or "").strip(),
        ord_qty=ord_qty,
        filled_qty=filled,
        status=UNKNOWN,
        reason=reason or stat,
        side="SELL" if (row.get("sll_buy_dvsn_cd") or "").strip() == "01" else "BUY",
        limit_price=_f(row.get("ft_ord_unpr3") or row.get("ord_unpr")),
    )
    if filled > 0:
        st.status = FILLED if filled >= ord_qty - 1e-9 else PARTIAL
        return st
    # 취소는 rjct_rson_name에 실려 온다(실계좌 실측: "DFD 주문종료 취소").
    # 사유 문구를 먼저 보고 취소/거부를 가른다 — 둘 다 종료 상태지만, 거부는
    # 주문이 안 받아들여진 것이고 취소는 받아들여졌다가 죽은 것이라 원인이 다르다.
    if any(h in reason for h in _CANCEL_HINTS):
        st.status = CANCELLED
        return st
    if reason:
        st.status = REJECTED
        return st
    if any(h in stat for h in _CANCEL_HINTS):
        st.status = CANCELLED
        return st
    st.status = UNFILLED
    return st


def quote_date_of(ord_dt: str) -> str:
    """주문일 '20260715' → 종가일 '2026-07-15'. 형식이 다르면 그대로 돌려준다."""
    d = (ord_dt or "").strip()
    return f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 and d.isdigit() else d


def fill_was_possible(st: OrderStatus, close: float | None) -> bool | None:
    """그날 종가 기준으로 **체결됐어야 하는가**. 판단할 수 없으면 None.

    LOC은 종가로 체결된다 — 매수는 `종가 ≤ 지정가`, 매도는 `종가 ≥ 지정가`.
    조건을 못 채워 안 붙은 것은 **정상**이고, 채웠는데 0주면 **이상**이다.
    2026-07-15 QLD 건이 후자였다(종가 $91.02 ≤ 지정가 $91.19인데 체결 0).

    종가를 모르거나 지정가가 없으면 None — 호출자는 이걸 '정상'으로 읽지 말고
    '확인 불가'로 다뤄야 한다.
    """
    if close is None or st.limit_price <= 0 or st.side not in ("BUY", "SELL"):
        return None
    return close <= st.limit_price if st.side == "BUY" else close >= st.limit_price


def match_unconfirmed(row: dict, ccnl_rows: list[dict]) -> dict | None:
    """odno 없이 남은 전송 기록을 체결내역에서 속성으로 찾아낸다.

    2단계 커밋의 빈틈(전송 성공 → 확정 기록 실패) 복구용이다. odno가 없으므로
    종목·주문일·방향·수량·지정가로 맞춘다. **재전송 대신 이 조회로 확인한다.**
    지정가는 KIS가 소수 8자리로 돌려주므로 센트 단위로 비교한다.
    """
    want_side = "01" if (row.get("side") or "").upper() == "SELL" else "02"
    want_dt = (row.get("ord_dt") or "").replace("-", "")
    want_qty = _f(row.get("qty"))
    want_price = round(_f(row.get("limit_price")), 2)
    for r in ccnl_rows:
        sym = (r.get("ovrs_pdno") or r.get("pdno") or "").upper()
        if sym != (row.get("ticker") or "").upper():
            continue
        if (r.get("ord_dt") or "").strip() != want_dt:
            continue
        if (r.get("sll_buy_dvsn_cd") or "").strip() != want_side:
            continue
        if abs(_f(r.get("ft_ord_qty")) - want_qty) > 1e-9:
            continue
        if abs(round(_f(r.get("ft_ord_unpr3")), 2) - want_price) > 0.005:
            continue
        return r
    return None


def reconcile(open_rows: list[dict], ccnl_rows: list[dict]) -> list[OrderStatus]:
    """DB에 남은 미종결 주문(open_rows) 각각의 현재 상태를 판정한다.

    odno로 매칭한다. 체결내역에서 아예 안 보이는 주문은 UNKNOWN으로 남긴다 —
    '안 보이니까 없던 일'로 처리하면 실제로 살아 있는 주문을 놓친다.
    """
    by_odno: dict[str, dict] = {}
    for r in ccnl_rows:
        odno = (r.get("odno") or "").strip()
        if odno:
            by_odno[odno] = r

    out: list[OrderStatus] = []
    for o in open_rows:
        odno = (o.get("odno") or "").strip()
        row = by_odno.get(odno)
        if row is None:
            out.append(
                OrderStatus(
                    odno=odno,
                    symbol=(o.get("ticker") or "").upper(),
                    ord_dt=o.get("ord_dt") or "",
                    ord_qty=_f(o.get("qty")),
                    filled_qty=0.0,
                    status=UNKNOWN,
                    reason="체결내역에서 조회되지 않음 — 사람이 KIS에서 직접 확인할 것",
                )
            )
            continue
        out.append(classify(row))
    return out
