"""1.5단계 ① — 전송-기록 원자성. "전송 성공 → 기록 실패 → 재전송" 구멍이 막혔는지.

핵심 시나리오: 의도는 커밋됐는데 확정 전에 프로세스가 죽는다. 다음 run은
**재전송하지 않고 조회로 확인**해야 한다. 네트워크·실계좌 없음.
"""
import os
import tempfile
import unittest

from stock_bot.broker.order_gate import GateContext, evaluate, order_key
from stock_bot.broker.order_status import (
    CANCELLED,
    SEND_REJECTED,
    SENDING,
    SENT,
    UNCONFIRMED,
    match_unconfirmed,
)
from stock_bot.state.store import Store

TICKER, DATE = "QLD", "2026-09-18"
KEY = order_key(TICKER, DATE, "BUY", 7, 91.19)

# 실계좌 원시 행 (2026-07-15 QLD, 취소됨)
CCNL_ROW = {
    "odno": "0030853537", "ord_dt": "20260918", "pdno": "QLD",
    "sll_buy_dvsn_cd": "02", "ft_ord_qty": "7", "ft_ccld_qty": "0",
    "ft_ord_unpr3": "91.19000000", "ovrs_excg_cd": "AMEX",
    "rjct_rson_name": "DFD 주문종료 취소", "prcs_stat_name": "완료",
}


class TwoPhaseBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(os.path.join(self.tmp.name, "t.db"))

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def intent(self, key=KEY, qty=7, price=91.19):
        return self.store.record_intent(
            key, TICKER, DATE, "BUY", qty, price, "AMEX", "2026-09-18T09:00")


class TestIntentBeforeSend(TwoPhaseBase):
    def test_의도가_먼저_커밋된다(self):
        self.assertTrue(self.intent())
        row = self.store.unresolved_sends()[0]
        self.assertEqual(row["status"], SENDING)
        self.assertEqual(row["odno"], "", "전송 전이므로 odno는 아직 없다")

    def test_의도만_기록돼도_키는_이미_점유된다(self):
        self.intent()
        self.assertIn(KEY, self.store.sent_keys(),
                      "확정 전이어도 키가 잡혀야 재전송이 막힌다")

    def test_확정되면_odno가_붙고_미확인_목록에서_빠진다(self):
        self.intent()
        self.store.confirm_sent(KEY, "0030853537", "2026-09-18T09:00:01")
        self.assertEqual(self.store.unresolved_sends(), [])
        row = self.store.open_sent_orders()[0]
        self.assertEqual((row["status"], row["odno"]), (SENT, "0030853537"))


class TestCrashWindow(TwoPhaseBase):
    """의도 커밋 직후 프로세스가 죽은 상태 = 확정이 안 된 SENDING 행."""

    def test_다음_run은_재전송하지_않는다(self):
        self.intent()   # 여기서 죽었다고 가정 (confirm_sent 미호출)

        ctx = GateContext(
            dry_run=False, kill_switch_path="/tmp/__none__", exchange_verified=True,
            is_backlog=False, book_shares=0.0, kis_shares=0.0,
            max_daily_orders=5, max_daily_amount=5000.0, max_order_amount=5000.0,
            max_account_daily_orders=5, max_account_daily_amount=5000.0,
            sent_keys=self.store.sent_keys(),
        )
        d = evaluate(TICKER, DATE, "BUY", 7, 91.19, ctx)
        self.assertFalse(d.allowed, "확정 안 된 전송을 다시 내보내면 중복 주문이 된다")
        self.assertEqual(d.gate, "IDEMPOTENT")

    def test_같은_키로_의도를_또_기록할_수_없다(self):
        self.intent()
        self.assertFalse(self.intent(), "DB PRIMARY KEY가 최종 방어선이다")
        self.assertEqual(len(self.store.unresolved_sends()), 1)

    def test_미확인_건은_조회로_확정된다(self):
        self.intent()
        row = self.store.unresolved_sends()[0]
        self.assertIsNotNone(
            match_unconfirmed(row, [CCNL_ROW]),
            "odno가 없어도 종목·주문일·방향·수량·지정가로 찾아야 한다")

    def test_속성이_하나라도_다르면_다른_주문으로_본다(self):
        self.intent()
        row = self.store.unresolved_sends()[0]
        for field, bad in (("ft_ord_qty", "8"), ("ft_ord_unpr3", "92.00000000"),
                           ("sll_buy_dvsn_cd", "01"), ("ord_dt", "20260917"),
                           ("pdno", "TSLA")):
            self.assertIsNone(
                match_unconfirmed(row, [dict(CCNL_ROW, **{field: bad})]),
                f"{field}가 달라도 같은 주문으로 잡으면 남의 주문을 삼킨다")


class TestSendFailureOutcomes(TwoPhaseBase):
    def test_불확정_실패는_재전송_대상이_아니다(self):
        self.intent()
        self.store.mark_send_outcome(KEY, UNCONFIRMED, "타임아웃", "2026-09-18T09:00:20")
        self.assertEqual(self.store.unresolved_sends()[0]["status"], UNCONFIRMED)
        self.assertIn(KEY, self.store.sent_keys())

    def test_접수_거부는_종결로_처리된다(self):
        self.intent()
        self.store.mark_send_outcome(KEY, SEND_REJECTED, "거부", "2026-09-18T09:00:20")
        self.assertEqual(self.store.unresolved_sends(), [],
                         "주문이 존재하지 않음이 확정이므로 더 조회할 것이 없다")
        self.assertEqual(self.store.open_sent_orders(), [])

    def test_거부되어도_키는_남아_그날_재전송을_막는다(self):
        self.intent()
        self.store.mark_send_outcome(KEY, SEND_REJECTED, "거부", "2026-09-18T09:00:20")
        self.assertIn(KEY, self.store.sent_keys())


class TestOpenOrdersRequireOdno(TwoPhaseBase):
    def test_odno_없는_건은_상태추적_대상이_아니다(self):
        self.intent()
        self.assertEqual(self.store.open_sent_orders(), [],
                         "odno가 없으면 조회할 열쇠가 없다 — resolve가 먼저다")


if __name__ == "__main__":
    unittest.main()
