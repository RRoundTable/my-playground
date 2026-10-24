"""1단계 안전장치 — 설정 파싱·원장 멱등성·상태 판정·전송 지점 방어.

실계좌를 호출하지 않는다. KIS 응답은 전부 고정값.
"""
import os
import tempfile
import unittest

from stock_bot.broker.kis_client import (
    EXCHANGE_BY_TICKER,
    ORD_DVSN_LOC,
    KisClient,
    KisConfig,
    KisError,
    _ORDER_TR,
)
from stock_bot.broker.order_gate import ALLOW, GateDecision, order_key
from stock_bot.broker.order_status import (
    CANCELLED,
    FILLED,
    PARTIAL,
    REJECTED,
    UNFILLED,
    UNKNOWN,
    classify,
    reconcile,
)
from stock_bot.config import parse_dry_run
from stock_bot.state.store import Store


class TestDryRunDefaultsSafe(unittest.TestCase):
    """DRY_RUN은 '명시적 거짓'일 때만 꺼진다. 나머지는 전부 켜진 채(차단) 남는다."""

    def test_미설정이면_True(self):
        self.assertTrue(parse_dry_run(None))

    def test_빈값이면_True(self):
        self.assertTrue(parse_dry_run(""))
        self.assertTrue(parse_dry_run("   "))

    def test_오타_해석불가면_True(self):
        for raw in ("ture", "FALSE_", "nope", "2", "-1", "예", "off!"):
            self.assertTrue(parse_dry_run(raw), f"{raw!r}가 전송을 열어선 안 된다")

    def test_명시적_거짓만_False(self):
        for raw in ("false", "False", "FALSE", "0", "no", "off", " false "):
            self.assertFalse(parse_dry_run(raw), f"{raw!r}는 꺼져야 한다")

    def test_true류는_당연히_True(self):
        for raw in ("true", "1", "yes", "on"):
            self.assertTrue(parse_dry_run(raw))


class TestSentOrderLedger(unittest.TestCase):
    """같은 주문은 DB 레벨에서 두 번 들어가지 않는다 — 멱등성 최종 방어선."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(os.path.join(self.tmp.name, "t.db"))
        self.key = order_key("QLD", "2026-09-18", "BUY", 1, 91.0)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def _record(self, odno="0001"):
        """의도 기록 → 전송 확정(2단계 커밋)을 한 번에."""
        ok = self.store.record_intent(
            self.key, "QLD", "2026-09-18", "BUY", 1, 91.0, "AMEX", "2026-09-18T09:00")
        if ok:
            self.store.confirm_sent(self.key, odno, "2026-09-18T09:00:01")
        return ok

    def test_처음_기록은_성공(self):
        self.assertTrue(self._record())
        self.assertIn(self.key, self.store.sent_keys())

    def test_같은_키를_두_번_기록해도_한_건만_남는다(self):
        self._record("0001")
        self.assertFalse(self._record("0002"), "중복 기록은 False를 돌려줘야 한다")
        rows = self.store.open_sent_orders()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["odno"], "0001", "나중 값이 먼저 값을 덮으면 안 된다")

    def test_odno가_반드시_저장된다(self):
        self._record("ODNO-77")
        self.assertEqual(self.store.open_sent_orders()[0]["odno"], "ODNO-77")

    def test_일일_누계가_한도_게이트에_쓸_수_있게_집계된다(self):
        self._record()
        k2 = order_key("QLD", "2026-09-18", "BUY", 2, 50.0)
        self.store.record_intent(k2, "QLD", "2026-09-18", "BUY", 2, 50.0,
                                 "AMEX", "2026-09-18T09:01")
        self.store.confirm_sent(k2, "0002", "2026-09-18T09:01:01")
        count, amount = self.store.sent_totals_on("2026-09-18")
        self.assertEqual(count, 2)
        self.assertAlmostEqual(amount, 91.0 + 100.0)
        self.assertEqual(self.store.sent_totals_on("2026-09-17"), (0, 0.0))

    def test_종결된_주문은_추적_대상에서_빠진다(self):
        self._record()
        self.store.update_sent_status(self.key, FILLED, 1.0, "", "2026-09-19T09:00")
        self.assertEqual(self.store.open_sent_orders(), [])

    def test_미체결은_추적_대상으로_남는다(self):
        self._record()
        self.store.update_sent_status(self.key, UNFILLED, 0.0, "", "2026-09-19T09:00")
        self.assertEqual(len(self.store.open_sent_orders()), 1)


class TestOrderStatusClassify(unittest.TestCase):
    """취소·거부·미체결을 구분한다 — parse_executions가 못 보는 영역."""

    def test_전량체결(self):
        st = classify({"ft_ord_qty": "7", "ft_ccld_qty": "7", "ovrs_pdno": "QLD"})
        self.assertEqual(st.status, FILLED)

    def test_부분체결(self):
        st = classify({"ft_ord_qty": "7", "ft_ccld_qty": "3"})
        self.assertEqual(st.status, PARTIAL)
        self.assertEqual(st.filled_qty, 3.0)

    def test_거부는_사유와_함께_잡힌다(self):
        st = classify({"ft_ord_qty": "7", "ft_ccld_qty": "0",
                       "rjct_rson_name": "증거금 부족"})
        self.assertEqual(st.status, REJECTED)
        self.assertEqual(st.reason, "증거금 부족")

    def test_취소(self):
        st = classify({"ft_ord_qty": "7", "ft_ccld_qty": "0", "prcs_stat_name": "주문취소"})
        self.assertEqual(st.status, CANCELLED)

    def test_07_15_QLD_건처럼_체결0_미체결은_UNFILLED(self):
        st = classify({"ord_dt": "20260715", "ovrs_pdno": "QLD",
                       "ft_ord_qty": "7", "ft_ccld_qty": "0"})
        self.assertEqual(st.status, UNFILLED)
        self.assertEqual(st.ord_qty, 7.0)

    def test_부분체결_후_잔량취소를_취소로_뭉개지_않는다(self):
        st = classify({"ft_ord_qty": "7", "ft_ccld_qty": "3", "prcs_stat_name": "주문취소"})
        self.assertEqual(st.status, PARTIAL, "체결된 3주가 사라지면 보유수량이 어긋난다")


class TestOrderStatusReconcile(unittest.TestCase):
    def test_odno로_매칭한다(self):
        open_rows = [{"key": "k", "ticker": "QLD", "ord_dt": "2026-09-18",
                      "qty": 7, "odno": "A1"}]
        rows = [{"odno": "A1", "ovrs_pdno": "QLD", "ft_ord_qty": "7", "ft_ccld_qty": "0",
                 "rjct_rson_name": "증거금 부족"}]
        self.assertEqual(reconcile(open_rows, rows)[0].status, REJECTED)

    def test_사유에_취소가_들어있으면_취소로_매칭한다(self):
        open_rows = [{"key": "k", "ticker": "QLD", "ord_dt": "2026-09-18",
                      "qty": 7, "odno": "A1"}]
        rows = [{"odno": "A1", "ovrs_pdno": "QLD", "ft_ord_qty": "7", "ft_ccld_qty": "0",
                 "rjct_rson_name": "장중 취소"}]
        self.assertEqual(reconcile(open_rows, rows)[0].status, CANCELLED)

    def test_체결내역에_없는_주문은_UNKNOWN으로_남긴다(self):
        open_rows = [{"key": "k", "ticker": "QLD", "ord_dt": "2026-09-18",
                      "qty": 7, "odno": "MISSING"}]
        st = reconcile(open_rows, [])[0]
        self.assertEqual(st.status, UNKNOWN, "'안 보이니 없던 일'로 처리하면 안 된다")
        self.assertIn("직접 확인", st.reason)


class TestSendOrderIsGuarded(unittest.TestCase):
    """전송 지점 자체가 게이트 결과를 요구한다 — 게이트 호출을 잊어도 나가지 않는다."""

    def setUp(self):
        self.client = KisClient(KisConfig(
            env="prod", app_key="k", app_secret="s", cano="0" * 8,
            acnt_prdt_cd="01", crcy_cd="USD", token_path="/tmp/none.json"))
        self.posted = []
        # 네트워크를 쓰지 않도록 토큰·POST를 대체한다. 호출되면 기록만 남긴다.
        self.client._token = lambda: "tok"
        import stock_bot.broker.kis_client as kc
        self._real_post = kc.net.post_json
        kc.net.post_json = lambda url, body, headers=None, timeout=20.0: (
            self.posted.append((url, body, headers))
            or {"rt_cd": "0", "output": {"ODNO": "ODNO-1"}, "msg1": "ok"})
        self._kc = kc

    def tearDown(self):
        self._kc.net.post_json = self._real_post

    def test_게이트_차단_결정으로는_전송되지_않는다(self):
        blocked = GateDecision(False, "DRY_RUN", "차단")
        with self.assertRaises(KisError):
            self.client.send_order("QLD", "AMEX", "BUY", 1, 91.0, blocked)
        self.assertEqual(self.posted, [], "차단인데 POST가 나갔다")

    def test_decision이_None이면_전송되지_않는다(self):
        with self.assertRaises(KisError):
            self.client.send_order("QLD", "AMEX", "BUY", 1, 91.0, None)
        self.assertEqual(self.posted, [])

    def test_수량0은_전송되지_않는다(self):
        with self.assertRaises(KisError):
            self.client.send_order("QLD", "AMEX", "BUY", 0, 91.0, ALLOW)
        self.assertEqual(self.posted, [])

    def test_통과_결정이면_LOC_지정가로_전송한다(self):
        odno = self.client.send_order("QLD", "AMEX", "BUY", 1, 91.0, ALLOW)
        self.assertEqual(odno, "ODNO-1")
        url, body, headers = self.posted[0]
        self.assertTrue(url.endswith("/uapi/overseas-stock/v1/trading/order"))
        self.assertEqual(headers["tr_id"], "TTTT1002U")
        self.assertEqual(body["ORD_DVSN"], ORD_DVSN_LOC)
        self.assertEqual(body["ORD_DVSN"], "34")
        self.assertEqual(body["OVRS_EXCG_CD"], "AMEX")
        self.assertEqual(body["ORD_QTY"], "1")
        self.assertEqual(body["SLL_TYPE"], "")

    def test_매도는_매도_TR과_SLL_TYPE을_쓴다(self):
        self.client.send_order("TSLA", "NASD", "SELL", 2, 400.0, ALLOW)
        _, body, headers = self.posted[0]
        self.assertEqual(headers["tr_id"], "TTTT1006U")
        self.assertEqual(body["SLL_TYPE"], "00")

    def test_거부응답이면_예외로_올린다(self):
        self._kc.net.post_json = lambda *a, **k: {"rt_cd": "1", "msg1": "거부", "msg_cd": "X"}
        with self.assertRaises(KisError):
            self.client.send_order("QLD", "AMEX", "BUY", 1, 91.0, ALLOW)

    def test_ODNO가_없으면_예외로_올린다(self):
        self._kc.net.post_json = lambda *a, **k: {"rt_cd": "0", "output": {}, "msg1": "?"}
        with self.assertRaises(KisError):
            self.client.send_order("QLD", "AMEX", "BUY", 1, 91.0, ALLOW)


class TestDocumentedConstants(unittest.TestCase):
    """공식 샘플(open-trading-api)에서 확인한 값 — 기억으로 바뀌면 실패한다."""

    def test_미국_주문_TR(self):
        self.assertEqual(_ORDER_TR["BUY"]["prod"], "TTTT1002U")
        self.assertEqual(_ORDER_TR["SELL"]["prod"], "TTTT1006U")
        self.assertEqual(_ORDER_TR["BUY"]["vps"], "VTTT1002U")
        self.assertEqual(_ORDER_TR["SELL"]["vps"], "VTTT1001U")

    def test_LOC_주문구분(self):
        self.assertEqual(ORD_DVSN_LOC, "34")

    def test_종목_거래소_매핑(self):
        self.assertEqual(EXCHANGE_BY_TICKER["QLD"], "AMEX")
        self.assertEqual(EXCHANGE_BY_TICKER["TSLA"], "NASD")


if __name__ == "__main__":
    unittest.main()
