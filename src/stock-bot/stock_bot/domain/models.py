"""도메인 모델 — 주문·전략 파라미터·계좌 상태."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    """하루치 LOC(종가) 지정가 주문 한 건."""

    side: Side
    limit_price: float          # LOC 지정가
    qty: int                    # 정수 주식 수
    band: float = 0.0           # 매도 사다리 밴드(+2.5% 등); 매수는 0.0
    kind: str = "LOC"

    def label(self) -> str:
        tag = f"+{self.band * 100:.1f}%" if self.side is Side.SELL else "종가"
        return f"{self.side.value} {self.qty}주 @ ${self.limit_price:.2f} ({tag})"


@dataclass
class StrategyParams:
    """전략 파라미터 — 라오어 VR 프리셋에서 주입."""

    strategy_type: str
    G: float
    cycle_cash_flow: float
    max_pool_usage: float
    band_rate: float = 0.15
    sell_bands: tuple[float, ...] = (0.025, 0.05, 0.075, 0.10)
    # F번 결정: V 기울기(Pool/G)는 입금 반영 *전* Pool 사용 (변형1, 보수적).
    # 신규 입금의 V 이중 카운팅 방지 — 의도된 설계. 백테스트로 뒤집을 수 있는 config.
    slope_uses_post_cashflow_pool: bool = False


@dataclass
class AccountState:
    """봇이 자체 추적하는 계좌 상태 (KIS 없이 LOC 체결로 재구성)."""

    shares: float                       # 보유 주식수 H
    avg_cost: float                     # 평단 A
    pool: float                         # 현금풀 Pool
    V: float                            # 가이드 밸류
    cycle_index: int                    # 사이클 번호
    day_in_cycle: int                   # 사이클 내 경과 영업일 (0..cycle_len-1)
    last_close: float | None = None
    last_quote_date: str | None = None  # 마지막 처리 종가 날짜 (휴장/중복 가드)
