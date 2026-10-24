"""1.5단계 ② — 계좌 단위 일일 한도. 두 봇 합이 계좌 상한을 넘으면 뒤쪽이 막히는지.

QLD·TSLA는 DB가 갈려 있어 각자 자기 한도만 센다. 공유 원장이 없으면 계좌 기준
상한의 2배까지 나간다. 네트워크 없음.
"""
import os
import tempfile
import unittest

from stock_bot.broker.order_gate import GateContext, evaluate
from stock_bot.state.account_ledger import AccountLedger

TODAY = "2026-09-21"


def ctx(**over):
    base = dict(
        dry_run=False, kill_switch_path="/tmp/__none__", exchange_verified=True,
        is_backlog=False, book_shares=0.0, kis_shares=0.0,
        # 봇별 한도는 넉넉하게 — 계좌 한도만 시험한다
        max_daily_orders=99, max_daily_amount=99999.0, max_order_amount=99999.0,
        account_sent_count=0, account_sent_amount=0.0,
        max_account_daily_orders=2, max_account_daily_amount=500.0,
        sent_keys=set(),
    )
    base.update(over)
    return GateContext(**base)


class TestAccountGate(unittest.TestCase):
    def test_계좌_금액_상한을_넘으면_차단(self):
        d = evaluate("TSLA", TODAY, "BUY", 1, 400.0, ctx(account_sent_amount=200.0))
        self.assertFalse(d.allowed)
        self.assertEqual(d.gate, "ACCOUNT_LIMIT")

    def test_계좌_건수_상한을_넘으면_차단(self):
        d = evaluate("TSLA", TODAY, "BUY", 1, 10.0, ctx(account_sent_count=2))
        self.assertEqual(d.gate, "ACCOUNT_LIMIT")

    def test_봇별_한도를_통과해도_계좌에서_막힌다(self):
        # 봇별 한도는 99건/$99,999 — 통과. 계좌는 $500 — 차단.
        d = evaluate("QLD", TODAY, "BUY", 18, 91.0, ctx())
        self.assertEqual(d.gate, "ACCOUNT_LIMIT")

    def test_계좌_한도가_설정되지_않으면_차단(self):
        self.assertEqual(
            evaluate("QLD", TODAY, "BUY", 1, 10.0,
                     ctx(max_account_daily_amount=0.0)).gate, "ACCOUNT_LIMIT")
        self.assertEqual(
            evaluate("QLD", TODAY, "BUY", 1, 10.0,
                     ctx(max_account_daily_orders=0)).gate, "ACCOUNT_LIMIT")

    def test_한도_안이면_통과(self):
        self.assertTrue(evaluate("TSLA", TODAY, "BUY", 1, 400.0, ctx()).allowed)


class TestTwoBotsShareOneLedger(unittest.TestCase):
    """실제 시나리오 — QLD가 먼저 쓰고, TSLA가 그 소진분을 보고 막힌다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "account_ledger.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_QLD가_쓴_금액을_TSLA가_본다(self):
        qld = AccountLedger(self.path)      # QLD 봇 프로세스
        qld.record("QLD|k1", "QLD", TODAY, 4, 91.0, "2026-09-21T09:00")  # $364
        qld.close()

        tsla = AccountLedger(self.path)     # TSLA 봇 프로세스 (2분 뒤, 같은 파일)
        count, amount = tsla.totals_on(TODAY)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(amount, 364.0)

        d = evaluate("TSLA", TODAY, "BUY", 1, 400.0,
                     ctx(account_sent_count=count, account_sent_amount=amount))
        self.assertEqual(d.gate, "ACCOUNT_LIMIT",
                         "$364 + $400 = $764 > 계좌 상한 $500 — 뒤쪽이 막혀야 한다")
        tsla.close()

    def test_공유_원장이_없으면_2배까지_나간다_반례(self):
        # 봇별 한도만 볼 때: 각자 $500까지 → 계좌 기준 $1,000. 이것이 막으려던 상황이다.
        each = evaluate("TSLA", TODAY, "BUY", 1, 400.0,
                        ctx(max_account_daily_amount=99999.0,
                            max_account_daily_orders=99))
        self.assertTrue(each.allowed, "계좌 한도를 풀면 통과한다(=공유 원장이 필요한 이유)")

    def test_같은_키는_두_번_집계되지_않는다(self):
        led = AccountLedger(self.path)
        self.assertTrue(led.record("k", "QLD", TODAY, 1, 91.0, "t"))
        self.assertFalse(led.record("k", "QLD", TODAY, 1, 91.0, "t"))
        self.assertEqual(led.totals_on(TODAY), (1, 91.0))
        led.close()

    def test_날짜가_바뀌면_리셋된다(self):
        led = AccountLedger(self.path)
        led.record("k", "QLD", "2026-09-20", 1, 91.0, "t")
        self.assertEqual(led.totals_on(TODAY), (0, 0.0))
        led.close()

    def test_종목별_집계도_볼_수_있다(self):
        led = AccountLedger(self.path)
        led.record("a", "QLD", TODAY, 2, 91.0, "t")
        led.record("b", "TSLA", TODAY, 1, 400.0, "t")
        self.assertEqual(led.by_ticker_on(TODAY),
                         {"QLD": (1, 182.0), "TSLA": (1, 400.0)})
        led.close()


class TestLedgerPathSharedByBothBots(unittest.TestCase):
    def test_두_봇_설정이_같은_원장_파일을_가리킨다(self):
        from stock_bot.config import load_config

        paths = set()
        for db in ("/workspace/stock-data/stock_bot.db",
                   "/workspace/stock-data-tsla/stock_bot.db"):
            os.environ["STOCK_DB_PATH"] = db
            os.environ.pop("STOCK_ACCOUNT_LEDGER_PATH", None)
            paths.add(load_config().account_ledger_path)
        os.environ.pop("STOCK_DB_PATH", None)
        self.assertEqual(len(paths), 1,
                         f"두 봇이 다른 원장을 보면 계좌 한도가 무의미하다: {paths}")


if __name__ == "__main__":
    unittest.main()
