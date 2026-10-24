"""장중 '진행 중 바'를 종가로 착각하지 않는지.

시세 소스는 장이 열려 있는 동안 그날 바를 이미 내놓고, 그 close는 확정 종가가
아니라 **현재가**다(개장 중인 DAX로 실측). 그걸 집으면 현재가를 종가로 삼아
계산하고 last_quote_date를 올려 **실제 종가를 영구히 건너뛴다.**

미국 정규장(13:30~20:00 UTC, 표준시 14:30~21:00)은 UTC 자정을 넘지 않으므로
`종가일 < 오늘 UTC 날짜`면 그 세션은 반드시 끝나 있다. 네트워크 미사용.
"""
import unittest

from stock_bot.market import quote

# 09-21(월)이 진행 중인 상황. 마지막 확정 종가는 09-18(금).
SERIES_LIVE = [
    ("2026-09-16", 86.96),
    ("2026-09-17", 89.94),
    ("2026-09-18", 91.00),
    ("2026-09-21", 93.40),   # ← 진행 중 바. close가 아니라 현재가다.
]
SERIES_DONE = SERIES_LIVE[:-1]


class GuardBase(unittest.TestCase):
    def setUp(self):
        self._real = quote.history
        self.series = list(SERIES_LIVE)
        quote.history = lambda ticker, range_="5y": list(self.series)

    def tearDown(self):
        quote.history = self._real


class TestIntradayBarRejected(GuardBase):
    """15:30 UTC — 미국 장 한가운데."""

    NOW = "2026-09-21"

    def test_진행_중인_오늘_바를_채택하지_않는다(self):
        self.assertIsNone(
            quote.next_unprocessed_close("QLD", "2026-09-18", today=self.NOW),
            "현재가를 종가로 삼으면 실제 종가를 영구히 건너뛴다")

    def test_밀린_확정_종가는_정상으로_준다(self):
        self.assertEqual(
            quote.next_unprocessed_close("QLD", "2026-09-16", today=self.NOW),
            ("2026-09-17", 89.94), "오래된 미처리분부터 하나씩은 그대로 유지된다")

    def test_마지막_확정_종가까지는_소진한다(self):
        seen, cursor = [], "2026-09-15"
        while (n := quote.next_unprocessed_close("QLD", cursor, today=self.NOW)):
            seen.append(n[0]); cursor = n[0]
        self.assertEqual(seen, ["2026-09-16", "2026-09-17", "2026-09-18"],
                         "09-21은 아직 진행 중이라 들어오면 안 된다")

    def test_장이_끝난_다음_날에는_채택된다(self):
        self.assertEqual(
            quote.next_unprocessed_close("QLD", "2026-09-18", today="2026-09-22"),
            ("2026-09-21", 93.40), "세션이 끝나면 그때 처리한다")


class TestBackwardCompatible(GuardBase):
    """09:00 KST = 00:00 UTC 스케줄에서는 결과가 이전과 같아야 한다."""

    def test_00시UTC에는_전날_종가를_그대로_집는다(self):
        # 00:00 UTC 09-21 시점: 소스에 09-21 바가 아직 없다(장 개시 전)
        self.series = list(SERIES_DONE)
        self.assertEqual(
            quote.next_unprocessed_close("QLD", "2026-09-17", today="2026-09-21"),
            ("2026-09-18", 91.00))

    def test_이미_처리했으면_None(self):
        self.series = list(SERIES_DONE)
        self.assertIsNone(
            quote.next_unprocessed_close("QLD", "2026-09-18", today="2026-09-21"))

    def test_가드가_없던_때와_같은_순서로_소진된다(self):
        """확정 바만 있는 상황에서는 가드가 아무것도 바꾸지 않는다."""
        self.series = list(SERIES_DONE)
        seen, cursor = [], "2026-09-15"
        while (n := quote.next_unprocessed_close("QLD", cursor, today="2026-09-21")):
            seen.append(n[0]); cursor = n[0]
        self.assertEqual(seen, [d for d, _ in SERIES_DONE])


class TestBootstrapAlsoGuarded(GuardBase):
    """부트스트랩(after_date=None)도 같은 함정에 빠진다 — last_close는 진행 중 바를 준다."""

    def test_부트스트랩도_진행_중_바를_피한다(self):
        self.assertEqual(
            quote.next_unprocessed_close("QLD", None, today="2026-09-21"),
            ("2026-09-18", 91.00), "부트스트랩이 현재가로 V를 잡으면 시작부터 틀어진다")

    def test_확정_종가가_하나도_없으면_None(self):
        self.series = [("2026-09-21", 93.40)]
        self.assertIsNone(quote.next_unprocessed_close("QLD", None, today="2026-09-21"))


class TestTodayInjection(GuardBase):
    def test_today를_안_주면_현재_UTC_날짜를_쓴다(self):
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.series = [("2026-09-18", 91.00), (today, 999.0)]
        got = quote.next_unprocessed_close("QLD", "2026-09-17")
        self.assertEqual(got, ("2026-09-18", 91.00),
                         "주입 없이도 오늘 바는 걸러져야 한다")


if __name__ == "__main__":
    unittest.main()
