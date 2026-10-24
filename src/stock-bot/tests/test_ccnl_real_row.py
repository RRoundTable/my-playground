"""② 근본 원인 회귀 방지 + 실계좌 원시 행 기반 상태 판정 검증.

REAL_ROW는 2026-09-21에 실계좌 TTTS3035R이 돌려준 2026-07-15 QLD 주문의 **실제 응답**이다
(필드명·값 그대로). 추측한 스키마가 아니라 관측된 스키마로 검증한다. 네트워크 미사용.
"""
import unittest

from stock_bot.broker.kis_client import KisClient, KisConfig
from stock_bot.broker.order_status import CANCELLED, REJECTED, UNFILLED, classify, reconcile
from stock_bot.broker.kis_client import parse_executions

REAL_ROW = {
    "dmst_ord_dt": "20260716",
    "ft_ccld_amt3": "0.00000",
    "ft_ccld_qty": "0",
    "ft_ccld_unpr3": "0.00000000",
    "ft_ord_qty": "7",
    "ft_ord_unpr3": "91.19000000",
    "loan_type_cd": "10",
    "mdia_dvsn_name": "모바일",
    "nccs_qty": "0",
    "odno": "0030853537",
    "ord_dt": "20260715",
    "ord_gno_brno": "01790",
    "ord_tmd": "003425",
    "ovrs_excg_cd": "AMEX",
    "pdno": "QLD",
    "prcs_stat_name": "완료",
    "prdt_name": "PROSHARES QQQ 2X",
    "rjct_rson_name": "DFD 주문종료 취소",
    "rvse_cncl_dvsn": "00",
    "sll_buy_dvsn_cd": "02",
    "sll_buy_dvsn_cd_name": "LOC매수",
    "tr_crcy_cd": "USD",
}


class TestCcnlParamNameRegression(unittest.TestCase):
    """`SLL_BUY_DVSN_CD`로 되돌아가면 실계좌 응답이 조용히 0건이 된다."""

    def setUp(self):
        self.client = KisClient(KisConfig(
            env="prod", app_key="k", app_secret="s", cano="0" * 8,
            acnt_prdt_cd="01", crcy_cd="USD", token_path="/tmp/none.json"))
        self.seen = {}

        def fake_get(path, tr_id, params):
            self.seen[path] = params
            return {"output": [REAL_ROW], "output1": []} if "ccnl" in path else {
                "output1": [], "output2": [], "output3": {}}

        self.client._get = fake_get

    def _check(self, params):
        self.assertIn("SLL_BUY_DVSN", params,
                      "KIS 정식 이름은 SLL_BUY_DVSN이다")
        self.assertNotIn("SLL_BUY_DVSN_CD", params,
                         "`_CD`를 붙이면 KIS가 매매구분을 빈값으로 받아 0건을 돌려준다")
        self.assertEqual(params["SLL_BUY_DVSN"], "00")

    def test_snapshot의_체결조회_파라미터(self):
        self.client.snapshot("QLD", "20260701", "20260731")
        self._check(self.seen["/uapi/overseas-stock/v1/trading/inquire-ccnl"])

    def test_order_history의_파라미터(self):
        self.client.order_history("20260701", "20260731")
        self._check(self.seen["/uapi/overseas-stock/v1/trading/inquire-ccnl"])


class TestClassifyRealRow(unittest.TestCase):
    def test_07_15_QLD_건은_취소로_판정된다(self):
        st = classify(REAL_ROW)
        self.assertEqual(st.status, CANCELLED,
                         "rjct_rson_name에 '취소'가 실려 오므로 거부가 아니라 취소다")
        self.assertEqual(st.odno, "0030853537")
        self.assertEqual(st.symbol, "QLD", "종목코드는 ovrs_pdno가 아니라 pdno로 온다")
        self.assertEqual((st.ord_qty, st.filled_qty), (7.0, 0.0))
        self.assertEqual(st.reason, "DFD 주문종료 취소")

    def test_사유가_취소가_아니면_거부로_남는다(self):
        row = dict(REAL_ROW, rjct_rson_name="증거금 부족")
        self.assertEqual(classify(row).status, REJECTED)

    def test_사유가_없으면_단순_미체결이다(self):
        row = dict(REAL_ROW, rjct_rson_name="", prcs_stat_name="접수")
        self.assertEqual(classify(row).status, UNFILLED)

    def test_원장과_odno로_매칭된다(self):
        open_rows = [{"key": "k", "ticker": "QLD", "ord_dt": "2026-07-15",
                      "qty": 7, "odno": "0030853537"}]
        self.assertEqual(reconcile(open_rows, [REAL_ROW])[0].status, CANCELLED)


class TestExecutionsExcludesCancelled(unittest.TestCase):
    def test_체결0인_취소건은_체결목록에_들어가지_않는다(self):
        self.assertEqual(parse_executions({"output": [REAL_ROW]}, "QLD"), [],
                         "체결수량 0은 체결이 아니다 — 상태 추적은 order_status가 맡는다")


if __name__ == "__main__":
    unittest.main()
