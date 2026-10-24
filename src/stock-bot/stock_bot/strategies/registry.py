"""전략 레지스트리 — 이름으로 전략을 조회한다.

새 전략 추가: Strategy를 구현한 뒤 아래 `_STRATEGIES`에 한 줄 등록.
"""
from __future__ import annotations

from stock_bot.domain.strategy import Strategy
from stock_bot.strategies.value_rebalancing import ValueRebalancing

_STRATEGIES: dict[str, type[Strategy]] = {
    ValueRebalancing.name: ValueRebalancing,
}


def get_strategy(name: str) -> Strategy:
    cls = _STRATEGIES.get(name)
    if cls is None:
        raise RuntimeError(f"알 수 없는 전략 '{name}' (등록됨: {list(_STRATEGIES)})")
    return cls()
