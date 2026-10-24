"""1단계 안전장치 — 게이트가 '실제로 막는지'를 게이트별로 검증.

전부 순수 로직 테스트다. 네트워크·DB·실계좌 없음.
"""
import os
import tempfile
import unittest

from stock_bot.broker import order_gate
from stock_bot.broker.order_gate import GateContext, evaluate, order_key

TICKER, DATE = "QLD", "2026-09-18"


def open_ctx(**over):
    """모든 게이트를 통과하는 컨텍스트. 테스트는 여기서 하나씩만 망가뜨린다."""
    base = dict(
        dry_run=False,
        kill_switch_path="/tmp/__없는파일__",
        kill_switch_env=None,
        exchange_verified=True,
        is_backlog=False,
        quote_date=DATE,
        today=DATE,
        book_shares=0.0,
        kis_shares=0.0,
        sent_today_count=0,
        sent_today_amount=0.0,
        max_daily_orders=2,
        max_daily_amount=500.0,
        max_order_amount=500.0,
        account_sent_count=0,
        account_sent_amount=0.0,
        max_account_daily_orders=2,
        max_account_daily_amount=500.0,
        sent_keys=set(),
    )
    base.update(over)
    return GateContext(**base)


def decide(ctx, qty=1, price=91.0, side="BUY"):
    return evaluate(TICKER, DATE, side, qty, price, ctx)


class TestBaseline(unittest.TestCase):
    def test_전부_정상이면_통과한다(self):
        d = decide(open_ctx())
        self.assertTrue(d.allowed, d.label())


class Test1KillSwitch(unittest.TestCase):
    def test_파일이_있으면_차단(self):
        with tempfile.NamedTemporaryFile() as f:
            d = decide(open_ctx(kill_switch_path=f.name))
        self.assertFalse(d.allowed)
        self.assertEqual(d.gate, "KILL_SWITCH")

    def test_환경변수가_설정되면_차단(self):
        d = decide(open_ctx(kill_switch_env="1"))
        self.assertEqual(d.gate, "KILL_SWITCH")

    def test_환경변수가_빈문자열이면_차단하지_않는다(self):
        self.assertTrue(decide(open_ctx(kill_switch_env="")).allowed)

    def test_킬스위치는_다른_모든_게이트보다_먼저_판정된다(self):
        # dry_run·백로그·불일치가 전부 걸린 상태에서도 KILL_SWITCH가 보고돼야 한다
        with tempfile.NamedTemporaryFile() as f:
            d = decide(open_ctx(kill_switch_path=f.name, dry_run=True,
                                is_backlog=True, book_shares=5.0, kis_shares=0.0))
        self.assertEqual(d.gate, "KILL_SWITCH")


class Test2DryRun(unittest.TestCase):
    def test_기본값은_차단이다(self):
        d = decide(GateContext())  # 아무것도 설정하지 않은 기본 컨텍스트
        self.assertFalse(d.allowed)
        self.assertEqual(d.gate, "DRY_RUN")

    def test_dry_run이_켜져_있으면_차단(self):
        self.assertEqual(decide(open_ctx(dry_run=True)).gate, "DRY_RUN")

    def test_dry_run을_꺼도_다른_게이트가_걸리면_전송되지_않는다(self):
        d = decide(open_ctx(dry_run=False, is_backlog=True))
        self.assertFalse(d.allowed)
        self.assertEqual(d.gate, "BACKLOG")


class Test3Exchange(unittest.TestCase):
    def test_미검증이면_차단(self):
        self.assertEqual(decide(open_ctx(exchange_verified=None)).gate, "EXCHANGE")

    def test_검증_실패면_차단(self):
        d = decide(open_ctx(exchange_verified=False, exchange_detail="APBN0746 상품이 없습니다"))
        self.assertEqual(d.gate, "EXCHANGE")
        self.assertIn("APBN0746", d.reason)


class Test4Backlog(unittest.TestCase):
    def test_밀린_날짜_처리중이면_차단(self):
        d = decide(open_ctx(is_backlog=True, quote_date="2026-09-10", today="2026-09-21"))
        self.assertEqual(d.gate, "BACKLOG")
        self.assertIn("2026-09-10", d.reason)


class Test5Reconcile(unittest.TestCase):
    def test_보유수량이_어긋나면_차단(self):
        d = decide(open_ctx(book_shares=0.0, kis_shares=7.0))
        self.assertEqual(d.gate, "RECONCILE")

    def test_KIS를_못_읽으면_차단(self):
        self.assertEqual(decide(open_ctx(kis_shares=None)).gate, "RECONCILE")
        self.assertEqual(decide(open_ctx(book_shares=None)).gate, "RECONCILE")

    def test_거래소_사고가_재발해도_2회차에_멈춘다(self):
        # 1회차에 7주가 체결됐는데 장부는 0주로 알고 있는 상황
        self.assertEqual(decide(open_ctx(book_shares=0.0, kis_shares=7.0)).gate, "RECONCILE")


class Test6DailyLimit(unittest.TestCase):
    def test_1건_금액_상한_초과면_차단(self):
        d = decide(open_ctx(), qty=18, price=91.0)   # $1,638 > $500
        self.assertEqual(d.gate, "DAILY_LIMIT")

    def test_1일_건수_상한_초과면_차단(self):
        d = decide(open_ctx(sent_today_count=2, max_daily_orders=2))
        self.assertEqual(d.gate, "DAILY_LIMIT")

    def test_1일_금액_상한_초과면_차단(self):
        d = decide(open_ctx(sent_today_amount=450.0), qty=1, price=91.0)
        self.assertEqual(d.gate, "DAILY_LIMIT")

    def test_상한이_0이면_전부_차단(self):
        self.assertEqual(decide(open_ctx(max_order_amount=0.0)).gate, "DAILY_LIMIT")

    def test_상한과_정확히_같으면_통과한다(self):
        self.assertTrue(decide(open_ctx(max_order_amount=91.0), qty=1, price=91.0).allowed)


class Test7Idempotent(unittest.TestCase):
    def test_이미_전송된_키면_차단(self):
        key = order_key(TICKER, DATE, "BUY", 1, 91.0)
        self.assertEqual(decide(open_ctx(sent_keys={key})).gate, "IDEMPOTENT")

    def test_수량이나_가격이_다르면_다른_주문이다(self):
        key = order_key(TICKER, DATE, "BUY", 1, 91.0)
        self.assertTrue(decide(open_ctx(sent_keys={key}), qty=2).allowed)
        self.assertTrue(decide(open_ctx(sent_keys={key}), price=92.0).allowed)

    def test_매수와_매도는_다른_키다(self):
        key = order_key(TICKER, DATE, "BUY", 1, 91.0)
        self.assertNotEqual(key, order_key(TICKER, DATE, "SELL", 1, 91.0))


class TestKillSwitchHelper(unittest.TestCase):
    def test_존재하지_않는_경로는_걸리지_않는다(self):
        ctx = GateContext(kill_switch_path=os.path.join(tempfile.gettempdir(), "__no_such__"))
        self.assertEqual(order_gate.kill_switch_engaged(ctx), "")


if __name__ == "__main__":
    unittest.main()
