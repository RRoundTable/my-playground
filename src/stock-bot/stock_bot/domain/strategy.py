"""전략 인터페이스 — 새 전략은 이 ABC를 구현하고 registry에 등록한다."""
from __future__ import annotations

from abc import ABC, abstractmethod

from stock_bot.domain.models import AccountState, Order, StrategyParams


class Strategy(ABC):
    name: str

    @abstractmethod
    def on_cycle_start(self, state: AccountState, p: StrategyParams) -> None:
        """사이클(2주) 시작 시 가이드 밸류·현금풀을 제자리 갱신한다."""

    @abstractmethod
    def daily_orders(
        self, state: AccountState, p: StrategyParams, close: float
    ) -> list[Order]:
        """오늘 종가(close) 기준으로 오늘 밤 걸 LOC 주문 목록을 산출한다."""
