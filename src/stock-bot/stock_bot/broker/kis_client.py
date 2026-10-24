"""KIS(한국투자증권) 해외주식 **읽기 전용** 클라이언트.

공식 샘플(koreainvestment/open-trading-api) 기준 — 검증된 값:
  · 실전 base  https://openapi.koreainvestment.com:9443   (env=prod)
  · 모의 base  https://openapivts.koreainvestment.com:29443 (env=vps)
  · OAuth      POST /oauth2/tokenP   {grant_type,appkey,appsecret}
  · 잔고       GET  /uapi/overseas-stock/v1/trading/inquire-balance  TR TTTS3012R/VTTS3012R
  · 체결내역   GET  /uapi/overseas-stock/v1/trading/inquire-ccnl     TR TTTS3035R/VTTS3035R

주문 전송 없음 — 잔고/체결 조회만. 실제 계좌가 진실의 원천.
응답 필드명은 실계좌 응답으로 한 번 확인 권장 → 방어적으로 후보 키를 시도한다.

거래소 코드: KIS는 QLD를 AMEX(NYSE Arca), TSLA를 NASD로 분류한다. 잔고 조회에서
NASD가 '미국 전체'인지 '나스닥만'인지는 문서상 모호하므로, 어느 쪽이든 정답이 되도록
**두 거래소를 각각 조회해 종목코드 기준으로 병합**한다(_US_EXCG_CODES). 설정 하나로
거래소를 고르면 고르지 않은 쪽 종목의 보유수량이 0으로 보여 봇이 매수만 반복하게 된다.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from stock_bot import net

_BASE = {
    "prod": "https://openapi.koreainvestment.com:9443",
    "vps": "https://openapivts.koreainvestment.com:29443",
}
_TR = {
    "balance": {"prod": "TTTS3012R", "vps": "VTTS3012R"},
    "ccnl": {"prod": "TTTS3035R", "vps": "VTTS3035R"},
    "present": {"prod": "CTRP6504R", "vps": "VTRP6504R"},
}
# 잔고 조회 대상 거래소. 운용 종목(QLD=AMEX, TSLA=NASD)을 모두 덮는다.
_US_EXCG_CODES = ("NASD", "AMEX")

# ── 주문 (공식 샘플 open-trading-api / examples_user/overseas_stock 기준) ──
# 조회는 거래소를 병합할 수 있지만 **주문은 거래소를 하나로 특정해야 한다.**
# 이 매핑은 기동 시 verify_exchange()가 실계좌로 재확인한다 — 외운 값을 믿지 않는다.
EXCHANGE_BY_TICKER = {"QLD": "AMEX", "TSLA": "NASD"}

_ORDER_PATH = "/uapi/overseas-stock/v1/trading/order"
_PSAMOUNT_PATH = "/uapi/overseas-stock/v1/trading/inquire-psamount"
# 미국(NASD/NYSE/AMEX) 주문 TR. 모의투자는 매수 VTTT1002U / 매도 VTTT1001U.
_ORDER_TR = {
    "BUY": {"prod": "TTTT1002U", "vps": "VTTT1002U"},
    "SELL": {"prod": "TTTT1006U", "vps": "VTTT1001U"},
}
_PSAMOUNT_TR = {"prod": "TTTS3007R", "vps": "VTTS3007R"}
ORD_DVSN_LOC = "34"   # 장마감지정가(LOC). 00=지정가, 32=LOO, 33=MOC. 모의투자는 00만 가능.


class KisError(RuntimeError):
    pass


@dataclass
class Execution:
    date: str
    side: str        # BUY / SELL
    qty: float
    price: float
    symbol: str

    def label(self) -> str:
        return f"{self.side} {self.qty:.0f}주 @ ${self.price:.2f} ({self.symbol}, {self.date})"


@dataclass
class BrokerSnapshot:
    as_of: str
    shares: float
    avg_cost: float
    cash: float | None                       # USD 예수금 (불명확 시 None)
    executions: list[Execution] = field(default_factory=list)


@dataclass
class AccountOverview:
    usd_cash: float | None
    usd_withdrawable: float | None
    tot_asst_krw: float | None
    holdings: list[dict] = field(default_factory=list)  # {symbol,qty,avg,eval}


@dataclass
class KisConfig:
    env: str           # prod / vps
    app_key: str
    app_secret: str
    cano: str
    acnt_prdt_cd: str
    crcy_cd: str
    token_path: str


# ── 순수 파서 (HTTP 없이 테스트 가능) ──

def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def merge_balance_rows(responses: list[dict]) -> list[dict]:
    """여러 거래소의 inquire-balance 응답 → output1 행을 종목코드로 병합(중복 제거).

    NASD가 '미국 전체'라면 AMEX 종목이 양쪽 응답에 모두 잡힌다. 같은 종목이 두 번
    보이더라도 수량을 합치면 안 된다(이중계산) — 먼저 본 행을 채택하고 나머지는 버린다.
    """
    merged: dict[str, dict] = {}
    for data in responses:
        for row in data.get("output1", []) or []:
            sym = (row.get("ovrs_pdno") or "").upper()
            if sym and sym not in merged:
                merged[sym] = row
    return list(merged.values())


def parse_balance(rows: list[dict], ticker: str) -> tuple[float, float]:
    """병합된 잔고 행 → (보유수량, 평단)."""
    for row in rows:
        if (row.get("ovrs_pdno") or "").upper() == ticker.upper():
            return _f(row.get("ovrs_cblc_qty")), _f(row.get("pchs_avg_pric"))
    return 0.0, 0.0


def parse_present_balance(data: dict, crcy: str = "USD") -> dict:
    """inquire-present-balance(CTRP6504R) → 통화별 외화예수금 + 총자산.

    실계좌로 확정한 필드: output2의 crcy_cd로 통화 매칭 → frcr_dncl_amt_2(외화예수금).
    """
    out2 = data.get("output2") or []
    if isinstance(out2, dict):
        out2 = [out2]
    cash = withdrawable = None
    for row in out2:
        if (row.get("crcy_cd") or "").upper() == crcy.upper():
            cash = _f(row.get("frcr_dncl_amt_2"))
            withdrawable = _f(row.get("frcr_drwg_psbl_amt_1"))
            break
    out3 = data.get("output3") or {}
    if isinstance(out3, list):
        out3 = out3[0] if out3 else {}
    return {
        "usd_cash": cash,
        "usd_withdrawable": withdrawable,
        "tot_asst_krw": _f(out3.get("tot_asst_amt")) if out3.get("tot_asst_amt") else None,
    }


def parse_holdings(rows: list[dict]) -> list[dict]:
    """병합된 잔고 행 → 보유종목 [{symbol, qty, avg, eval(USD)}] (수량>0)."""
    out = []
    for r in rows:
        qty = _f(r.get("ovrs_cblc_qty"))
        if qty <= 0:
            continue
        out.append(
            {
                "symbol": (r.get("ovrs_pdno") or "").upper(),
                "qty": qty,
                "avg": _f(r.get("pchs_avg_pric")),
                "eval": _f(r.get("ovrs_stck_evlu_amt")),
            }
        )
    return out


def parse_executions(data: dict, ticker: str) -> list[Execution]:
    """inquire-ccnl 응답 → 체결 리스트 (체결수량>0만)."""
    out: list[Execution] = []
    for row in data.get("output", []) or []:
        sym = (row.get("ovrs_pdno") or row.get("pdno") or "").upper()
        if ticker and sym and sym != ticker.upper():
            continue
        qty = _f(row.get("ft_ccld_qty") or row.get("ccld_qty"))
        if qty <= 0:
            continue
        side = "SELL" if row.get("sll_buy_dvsn_cd") == "01" else "BUY"
        out.append(
            Execution(
                date=row.get("ord_dt") or row.get("ccld_dt") or "",
                side=side,
                qty=qty,
                price=_f(row.get("ft_ccld_unpr3") or row.get("ccld_unpr")),
                symbol=sym,
            )
        )
    return out


class KisClient:
    def __init__(self, cfg: KisConfig):
        self.cfg = cfg
        self.base = _BASE[cfg.env]

    # ── 토큰 (디스크 캐시 — KIS는 발급 1회/분 제한 + 24h 유효) ──
    def _token(self) -> str:
        cached = self._load_token()
        if cached:
            return cached
        try:
            data = net.post_json(
                f"{self.base}/oauth2/tokenP",
                {
                    "grant_type": "client_credentials",
                    "appkey": self.cfg.app_key,
                    "appsecret": self.cfg.app_secret,
                },
                timeout=15.0,
            )
        except (net.HttpError, ValueError) as e:
            raise KisError(f"KIS 토큰 발급 실패: {e}") from e
        token = data.get("access_token")
        if not token:
            raise KisError(f"KIS 토큰 응답 이상: {data}")
        expires = time.time() + int(data.get("expires_in", 86400)) - 600
        self._save_token(token, expires)
        return token

    def _load_token(self) -> str | None:
        try:
            with open(self.cfg.token_path) as f:
                d = json.load(f)
            if d.get("expires", 0) > time.time():
                return d["token"]
        except (OSError, ValueError, KeyError):
            pass
        return None

    def _save_token(self, token: str, expires: float) -> None:
        os.makedirs(os.path.dirname(self.cfg.token_path) or ".", exist_ok=True)
        with open(self.cfg.token_path, "w") as f:
            json.dump({"token": token, "expires": expires}, f)

    def _get(self, path: str, tr_id: str, params: dict) -> dict:
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self._token()}",
            "appkey": self.cfg.app_key,
            "appsecret": self.cfg.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }
        try:
            data = net.get_json(f"{self.base}{path}", headers=headers, params=params, timeout=20.0)
        except (net.HttpError, ValueError) as e:
            raise KisError(f"KIS {tr_id} 요청 실패: {e}") from e
        if str(data.get("rt_cd", "0")) != "0":
            raise KisError(f"KIS {tr_id} 오류: {data.get('msg1')} ({data.get('msg_cd')})")
        return data

    def balance_rows(self) -> list[dict]:
        """_US_EXCG_CODES 전 거래소의 잔고 행을 병합해 반환."""
        responses = [
            self._get(
                "/uapi/overseas-stock/v1/trading/inquire-balance",
                _TR["balance"][self.cfg.env],
                {
                    "CANO": self.cfg.cano,
                    "ACNT_PRDT_CD": self.cfg.acnt_prdt_cd,
                    "OVRS_EXCG_CD": excg,
                    "TR_CRCY_CD": self.cfg.crcy_cd,
                    "CTX_AREA_FK200": "",
                    "CTX_AREA_NK200": "",
                },
            )
            for excg in _US_EXCG_CODES
        ]
        return merge_balance_rows(responses)

    def snapshot(self, ticker: str, start_dt: str, end_dt: str) -> BrokerSnapshot:
        rows = self.balance_rows()
        ccnl = self._get(
            "/uapi/overseas-stock/v1/trading/inquire-ccnl",
            _TR["ccnl"][self.cfg.env],
            {
                "CANO": self.cfg.cano,
                "ACNT_PRDT_CD": self.cfg.acnt_prdt_cd,
                "PDNO": "" if self.cfg.env == "vps" else "%",
                "ORD_STRT_DT": start_dt,
                "ORD_END_DT": end_dt,
                # 이름은 SLL_BUY_DVSN이다 — `_CD`를 붙이면 KIS가 그 파라미터를 무시하고
                # 매매구분을 빈값으로 받아 **결과가 0건**이 된다(rt_cd=0 KIOK0560이라
                # 에러로도 안 보인다). 응답의 ctx_area_fk200 에코에서 6번째 필드가
                # 비어 있는 것으로 확인된다. 2026-09-21 실계좌 A/B로 확정.
                "SLL_BUY_DVSN": "00",
                # 00:전체 / 01:체결 / 02:미체결. 01이면 미체결 주문이 응답에서 아예
                # 빠져 "주문은 냈는데 안 붙었다"를 구분할 수 없다. 전체를 받아온 뒤
                # 체결 여부는 parse_executions가 체결수량으로 판정한다.
                "CCLD_NCCS_DVSN": "00",
                "OVRS_EXCG_CD": "%",
                "SORT_SQN": "DS",
                "ORD_DT": "",
                "ORD_GNO_BRNO": "",
                "ODNO": "",
                "CTX_AREA_FK200": "",
                "CTX_AREA_NK200": "",
            },
        )
        shares, avg = parse_balance(rows, ticker)
        cash = None
        try:  # 예수금은 별도 엔드포인트(present-balance). 실패해도 잔고/체결은 유효.
            cash = self.present_balance().get("usd_cash")
        except KisError:
            cash = None
        return BrokerSnapshot(
            as_of=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            shares=shares,
            avg_cost=avg,
            cash=cash,
            executions=parse_executions(ccnl, ticker),
        )

    def order_history(self, start_dt: str, end_dt: str) -> list[dict]:
        """기간 내 주문 전체(체결+미체결+취소) 원시 행. 상태 추적 전용.

        snapshot()의 체결 조회와 달리 종목 필터·체결수량 필터를 걸지 않는다 —
        죽은 주문을 찾는 게 목적이라 걸러내면 안 된다.
        """
        data = self._get(
            "/uapi/overseas-stock/v1/trading/inquire-ccnl",
            _TR["ccnl"][self.cfg.env],
            {
                "CANO": self.cfg.cano,
                "ACNT_PRDT_CD": self.cfg.acnt_prdt_cd,
                "PDNO": "" if self.cfg.env == "vps" else "%",
                "ORD_STRT_DT": start_dt,
                "ORD_END_DT": end_dt,
                "SLL_BUY_DVSN": "00",  # `_CD` 아님 — 위 snapshot() 주석 참조
                "CCLD_NCCS_DVSN": "00",
                "OVRS_EXCG_CD": "%",
                "SORT_SQN": "DS",
                "ORD_DT": "",
                "ORD_GNO_BRNO": "",
                "ODNO": "",
                "CTX_AREA_FK200": "",
                "CTX_AREA_NK200": "",
            },
        )
        return data.get("output") or []

    def verify_exchange(self, ticker: str, excg: str, price: float) -> tuple[bool, str]:
        """TTTS3007R로 '이 종목이 이 거래소에서 조회되는가'를 실계좌에 물어본다.

        거래소가 틀리면 `APBN0746 상품이 없습니다`로 떨어진다. 주문 전 이걸 통과해야
        EXCHANGE 게이트가 열린다 — 매핑을 코드에 적어두고 믿는 대신 매번 재확인한다.
        조회 전용 TR이라 주문이 나가지 않는다.
        """
        try:
            data = self._get(
                _PSAMOUNT_PATH,
                _PSAMOUNT_TR[self.cfg.env],
                {
                    "CANO": self.cfg.cano,
                    "ACNT_PRDT_CD": self.cfg.acnt_prdt_cd,
                    "OVRS_EXCG_CD": excg,
                    "OVRS_ORD_UNPR": f"{price:.2f}",
                    "ITEM_CD": ticker.upper(),
                },
            )
        except KisError as e:
            return False, f"{ticker}@{excg} 조회 실패: {e}"
        return True, f"{ticker}@{excg} 확인됨 (msg={data.get('msg1', '').strip()})"

    def send_order(
        self, ticker: str, excg: str, side: str, qty: int, limit_price: float, decision
    ) -> str:
        """LOC 주문 전송 → 주문번호(odno). **이 함수가 유일한 주문 송신 지점이다.**

        `decision`(order_gate.GateDecision)이 allowed=True가 아니면 전송하지 않는다.
        게이트 호출을 잊은 채 이 함수에 도달하는 경로를 구조적으로 막기 위한 이중 방어다.
        """
        if decision is None or not getattr(decision, "allowed", False):
            raise KisError(
                f"게이트 미통과 주문의 전송 시도 차단: {getattr(decision, 'label', lambda: decision)()}"
            )
        side = side.upper()
        if side not in _ORDER_TR:
            raise KisError(f"알 수 없는 주문 방향: {side}")
        if qty <= 0:
            raise KisError(f"주문 수량 비정상: {qty}")

        body = {
            "CANO": self.cfg.cano,
            "ACNT_PRDT_CD": self.cfg.acnt_prdt_cd,
            "OVRS_EXCG_CD": excg,
            "PDNO": ticker.upper(),
            "ORD_QTY": str(int(qty)),
            "OVRS_ORD_UNPR": f"{limit_price:.2f}",
            "CTAC_TLNO": "",
            "MGCO_APTM_ODNO": "",
            "SLL_TYPE": "" if side == "BUY" else "00",
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": ORD_DVSN_LOC,
        }
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self._token()}",
            "appkey": self.cfg.app_key,
            "appsecret": self.cfg.app_secret,
            "tr_id": _ORDER_TR[side][self.cfg.env],
            "custtype": "P",
        }
        try:
            data = net.post_json(f"{self.base}{_ORDER_PATH}", body, headers=headers, timeout=20.0)
        except (net.HttpError, ValueError) as e:
            raise KisError(f"KIS 주문 전송 실패: {e}") from e
        if str(data.get("rt_cd", "1")) != "0":
            raise KisError(f"KIS 주문 거부: {data.get('msg1')} ({data.get('msg_cd')})")
        odno = ((data.get("output") or {}).get("ODNO") or "").strip()
        if not odno:
            raise KisError(f"KIS 주문 응답에 ODNO 없음: {data.get('msg1')}")
        return odno

    def present_balance(self) -> dict:
        """체결기준 현재잔고 → {usd_cash, usd_withdrawable, tot_asst_krw}."""
        data = self._get(
            "/uapi/overseas-stock/v1/trading/inquire-present-balance",
            _TR["present"][self.cfg.env],
            {
                "CANO": self.cfg.cano,
                "ACNT_PRDT_CD": self.cfg.acnt_prdt_cd,
                "WCRC_FRCR_DVSN_CD": "02",  # 외화
                "NATN_CD": "840",           # 미국
                "TR_MKET_CD": "00",         # 전체
                "INQR_DVSN_CD": "00",
            },
        )
        return parse_present_balance(data, self.cfg.crcy_cd)

    def account_overview(self) -> AccountOverview:
        """CLI용 계좌 요약 — 예수금 + 보유종목 평가금액 + 총자산."""
        rows = self.balance_rows()
        pb = self.present_balance()
        return AccountOverview(
            usd_cash=pb["usd_cash"],
            usd_withdrawable=pb["usd_withdrawable"],
            tot_asst_krw=pb["tot_asst_krw"],
            holdings=parse_holdings(rows),
        )
