"""과제 ② — 체결내역 조회 파라미터/파싱 검증. KIS 응답 고정, 실계좌 미호출."""
import unittest

from stock_bot.broker.kis_client import KisClient, KisConfig, parse_executions

FILLED = {
    "ovrs_pdno": "QLD",
    "ord_dt": "20260715",
    "sll_buy_dvsn_cd": "02",
    "ft_ord_qty": "7",
    "ft_ccld_qty": "7",
    "ft_ccld_unpr3": "90.50",
    "ovrs_excg_cd": "AMEX",
}
UNFILLED = {  # 2026-07-15 QLD LOC 매수 7주, 체결 0주
    "ovrs_pdno": "QLD",
    "ord_dt": "20260715",
    "sll_buy_dvsn_cd": "02",
    "ft_ord_qty": "7",
    "ft_ccld_qty": "0",
    "ft_ord_unpr3": "90.00",
    "ovrs_excg_cd": "AMEX",
}
SOLD = {
    "ovrs_pdno": "QLD",
    "ord_dt": "20260716",
    "sll_buy_dvsn_cd": "01",
    "ft_ord_qty": "2",
    "ft_ccld_qty": "2",
    "ft_ccld_unpr3": "93.00",
    "ovrs_excg_cd": "AMEX",
}


class TestParseExecutions(unittest.TestCase):
    def test_체결된_건만_담는다(self):
        out = parse_executions({"output": [FILLED, UNFILLED]}, "QLD")
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0].side, out[0].qty, out[0].price), ("BUY", 7.0, 90.5))

    def test_매도구분코드01은_SELL로_읽는다(self):
        out = parse_executions({"output": [SOLD]}, "QLD")
        self.assertEqual(out[0].side, "SELL")

    def test_다른_종목은_걸러낸다(self):
        self.assertEqual(parse_executions({"output": [FILLED]}, "TSLA"), [])

    def test_빈_응답이어도_예외가_아니다(self):
        self.assertEqual(parse_executions({"output": []}, "QLD"), [])
        self.assertEqual(parse_executions({}, "QLD"), [])


class TestCcnlRequestParams(unittest.TestCase):
    """미체결 주문이 응답에서 빠지지 않도록 CCLD_NCCS_DVSN=00(전체)이어야 한다."""

    def setUp(self):
        self.client = KisClient(
            KisConfig(
                env="prod",
                app_key="k",
                app_secret="s",
                cano="0" * 8,
                acnt_prdt_cd="01",
                crcy_cd="USD",
                token_path="/tmp/does-not-matter.json",
            )
        )
        self.ccnl_params = None

        def fake_get(path, tr_id, params):
            if path.endswith("inquire-ccnl"):
                self.ccnl_params = params
                return {"output": [FILLED, UNFILLED]}
            if path.endswith("inquire-balance"):
                return {"output1": []}
            return {"output2": [], "output3": {}}

        self.client._get = fake_get

    def test_체결_미체결_전체를_요청한다(self):
        self.client.snapshot("QLD", "20260708", "20260715")
        self.assertEqual(
            self.ccnl_params["CCLD_NCCS_DVSN"],
            "00",
            "01(체결만)이면 미체결 주문을 영영 볼 수 없다",
        )

    def test_snapshot이_체결분을_담아_돌려준다(self):
        snap = self.client.snapshot("QLD", "20260708", "20260715")
        self.assertEqual([e.qty for e in snap.executions], [7.0])


if __name__ == "__main__":
    unittest.main()
