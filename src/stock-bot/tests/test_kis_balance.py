"""과제 ① — 잔고 조회를 AMEX·NASD 병합으로 바꾼 뒤의 동작 검증.

KIS 응답은 전부 고정값. 실계좌를 호출하지 않는다.
"""
import unittest

from stock_bot.broker.kis_client import (
    KisClient,
    KisConfig,
    merge_balance_rows,
    parse_balance,
    parse_holdings,
)


def _row(sym, qty, avg, excg, eval_amt="0"):
    return {
        "ovrs_pdno": sym,
        "ovrs_cblc_qty": qty,
        "pchs_avg_pric": avg,
        "ovrs_excg_cd": excg,
        "ovrs_stck_evlu_amt": eval_amt,
    }


# NASD='나스닥만'인 경우: QLD는 AMEX 응답에만 나온다.
NASD_ONLY = {"output1": [_row("TSLA", "3", "400.00", "NASD", "1200.00")]}
AMEX_ONLY = {"output1": [_row("QLD", "7", "91.00", "AMEX", "637.00")]}
# NASD='미국 전체'인 경우: QLD가 양쪽에 중복으로 나온다.
NASD_ALL_US = {
    "output1": [
        _row("TSLA", "3", "400.00", "NASD", "1200.00"),
        _row("QLD", "7", "91.00", "AMEX", "637.00"),
    ]
}
EMPTY = {"output1": []}


class TestMergeBalanceRows(unittest.TestCase):
    def test_나스닥만인_경우_AMEX_종목을_찾는다(self):
        rows = merge_balance_rows([NASD_ONLY, AMEX_ONLY])
        self.assertEqual({r["ovrs_pdno"] for r in rows}, {"TSLA", "QLD"})

    def test_미국전체인_경우_중복_종목의_수량을_합치지_않는다(self):
        rows = merge_balance_rows([NASD_ALL_US, AMEX_ONLY])
        qld = [r for r in rows if r["ovrs_pdno"] == "QLD"]
        self.assertEqual(len(qld), 1, "QLD가 두 번 잡히면 보유수량이 이중계산된다")
        self.assertEqual(parse_balance(rows, "QLD"), (7.0, 91.0))

    def test_빈_응답과_섞여도_안전하다(self):
        self.assertEqual(merge_balance_rows([EMPTY, EMPTY]), [])
        rows = merge_balance_rows([EMPTY, AMEX_ONLY])
        self.assertEqual(parse_balance(rows, "QLD"), (7.0, 91.0))

    def test_종목코드_없는_행은_버린다(self):
        rows = merge_balance_rows([{"output1": [_row("", "5", "1.0", "NASD")]}])
        self.assertEqual(rows, [])


class TestParsersOnMergedRows(unittest.TestCase):
    def test_보유하지_않은_종목은_0주(self):
        rows = merge_balance_rows([NASD_ONLY, AMEX_ONLY])
        self.assertEqual(parse_balance(rows, "SOXL"), (0.0, 0.0))

    def test_parse_holdings는_수량0을_제외한다(self):
        rows = merge_balance_rows(
            [{"output1": [_row("QLD", "0", "0", "AMEX"), _row("TSLA", "3", "400", "NASD")]}]
        )
        self.assertEqual([h["symbol"] for h in parse_holdings(rows)], ["TSLA"])


class TestBalanceRowsQueriesBothExchanges(unittest.TestCase):
    """회귀 방지 — 거래소 하나만 조회하던 과거 동작으로 되돌아가면 실패한다."""

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
        self.asked = []
        by_excg = {"NASD": NASD_ONLY, "AMEX": AMEX_ONLY}

        def fake_get(path, tr_id, params):
            self.asked.append(params["OVRS_EXCG_CD"])
            return by_excg[params["OVRS_EXCG_CD"]]

        self.client._get = fake_get

    def test_두_거래소를_모두_조회한다(self):
        rows = self.client.balance_rows()
        self.assertEqual(sorted(self.asked), ["AMEX", "NASD"])
        self.assertEqual(parse_balance(rows, "QLD"), (7.0, 91.0))
        self.assertEqual(parse_balance(rows, "TSLA"), (3.0, 400.0))

    def test_설정으로_거래소를_고르는_필드가_남아있지_않다(self):
        self.assertFalse(
            hasattr(self.client.cfg, "excg_cd"),
            "KisConfig.excg_cd가 부활하면 한쪽 거래소 종목이 0주로 보인다",
        )


if __name__ == "__main__":
    unittest.main()
