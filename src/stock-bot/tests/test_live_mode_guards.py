"""1단계 — 서비스 층 안전장치: 유령 체결 차단, 백로그 판정, 컨텍스트 조립.

네트워크 없음. 시세·KIS는 전부 대체한다.
"""
import unittest

from stock_bot import service
from stock_bot.config import Config, params_for
from stock_bot.domain.guards import GuardError
from stock_bot.domain.models import AccountState
from stock_bot.strategies.registry import get_strategy


def make_cfg(**over):
    base = dict(
        ticker="QLD", vr_mode="거치식", strategy_engine="value_rebalancing",
        db_path="/tmp/x/stock_bot.db", cycle_len=10, band_rate=0.15,
        sell_bands=(0.025, 0.05, 0.075, 0.10),
        init_v=1000.0, init_pool=1000.0, init_shares=0.0, init_avg_cost=0.0,
        cycle_cash_flow=None, G=None, max_pool_usage=None, kis_cash_share=1.0,
        dry_run=True, kill_switch_path="/tmp/__none__",
        max_daily_orders=2, max_daily_amount=500.0, max_order_amount=500.0,
        account_ledger_path="/tmp/__test_account_ledger.db",
        max_account_daily_orders=2, max_account_daily_amount=500.0,
        kis_env="prod", kis_app_key="", kis_app_secret="", kis_cano="",
        kis_acnt_prdt_cd="01", kis_crcy_cd="USD",
    )
    base.update(over)
    return Config(**base)


def fresh_state():
    return AccountState(shares=0, avg_cost=0, pool=1000.0, V=900.0,
                        cycle_index=1, day_in_cycle=0, last_close=90.0,
                        last_quote_date="2026-09-17")


class TestGhostFillBlockedInLiveMode(unittest.TestCase):
    """KIS를 못 읽은 날, 실주문 모드에서는 가정 모델로 장부를 움직이지 않는다."""

    def _run(self, cfg):
        return service.run_daily(
            cfg, params_for(cfg.vr_mode, cfg.band_rate, cfg.sell_bands),
            get_strategy(cfg.strategy_engine), fresh_state(), [],
            "2026-09-18", 91.0, broker=None,
        )

    def test_실주문_모드에서_broker가_없으면_중단한다(self):
        with self.assertRaises(GuardError) as e:
            self._run(make_cfg(dry_run=False))
        self.assertIn("reconcile_loc", str(e.exception))

    def test_DRY_RUN에서는_기존대로_모델_폴백을_허용한다(self):
        result = self._run(make_cfg(dry_run=True))
        self.assertIsNotNone(result)


class TestIsBacklog(unittest.TestCase):
    def setUp(self):
        # _is_backlog는 `market.next_unprocessed_close`를 부른다. quote 쪽을 패치하면
        # market 패키지가 재수출한 원본이 그대로 호출돼 실제 네트워크를 탄다.
        from stock_bot import market
        self.quote = market
        self._real = market.next_unprocessed_close

    def tearDown(self):
        self.quote.next_unprocessed_close = self._real

    def test_뒤에_미처리_종가가_있으면_백로그다(self):
        self.quote.next_unprocessed_close = lambda t, d: ("2026-09-18", 91.0)
        self.assertTrue(service._is_backlog(make_cfg(), "2026-09-10"))

    def test_최신까지_따라잡았으면_백로그가_아니다(self):
        self.quote.next_unprocessed_close = lambda t, d: None
        self.assertFalse(service._is_backlog(make_cfg(), "2026-09-18"))

    def test_시세_조회가_실패하면_보수적으로_백로그로_본다(self):
        def boom(t, d):
            raise RuntimeError("소스 장애")
        self.quote.next_unprocessed_close = boom
        self.assertTrue(service._is_backlog(make_cfg(), "2026-09-18"),
                        "판정 불가를 '전송 가능'으로 낙관하면 안 된다")


class FakeLedger:
    def totals_on(self, d):
        return 0, 0.0


class FakeStore:
    def sent_totals_on(self, d):
        return 0, 0.0

    def sent_keys(self):
        return set()


class TestBuildGateContext(unittest.TestCase):
    def setUp(self):
        # _is_backlog는 `market.next_unprocessed_close`를 부른다. quote 쪽을 패치하면
        # market 패키지가 재수출한 원본이 그대로 호출돼 실제 네트워크를 탄다.
        from stock_bot import market
        self.quote = market
        self._real = market.next_unprocessed_close
        market.next_unprocessed_close = lambda t, d: None

    def tearDown(self):
        self.quote.next_unprocessed_close = self._real

    def test_drift가_없으면_대조_불가로_차단된다(self):
        ctx = service.build_gate_context(
            make_cfg(), FakeStore(), FakeLedger(), None, fresh_state(), None, None,
            "2026-09-18", "2026-09-18")
        self.assertIsNone(ctx.book_shares)
        self.assertIsNone(ctx.kis_shares)

    def test_drift로부터_대조_전_장부값을_복원한다(self):
        from stock_bot.broker.kis_client import BrokerSnapshot
        from stock_bot.service import Drift

        snap = BrokerSnapshot(as_of="2026-09-18", shares=7.0, avg_cost=90.0, cash=1000.0)
        # KIS 7주 - 장부 0주 = drift +7 → 대조 전 장부는 0주였어야 한다
        ctx = service.build_gate_context(
            make_cfg(), FakeStore(), FakeLedger(), None, fresh_state(), snap, Drift(shares=7.0, cash=0.0),
            "2026-09-18", "2026-09-18")
        self.assertEqual(ctx.kis_shares, 7.0)
        self.assertEqual(ctx.book_shares, 0.0)

    def test_DRY_RUN이면_거래소_검증을_하지_않아_미검증_상태로_남는다(self):
        ctx = service.build_gate_context(
            make_cfg(dry_run=True), FakeStore(), FakeLedger(), None, fresh_state(), None, None,
            "2026-09-18", "2026-09-18")
        self.assertIsNone(ctx.exchange_verified)

    def test_설정된_한도가_그대로_전달된다(self):
        cfg = make_cfg(max_daily_orders=5, max_daily_amount=999.0, max_order_amount=111.0)
        ctx = service.build_gate_context(
            cfg, FakeStore(), FakeLedger(), None, fresh_state(), None, None, "2026-09-18", "2026-09-18")
        self.assertEqual(
            (ctx.max_daily_orders, ctx.max_daily_amount, ctx.max_order_amount),
            (5, 999.0, 111.0))


if __name__ == "__main__":
    unittest.main()
