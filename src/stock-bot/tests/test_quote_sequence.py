"""과제 ③ — 밀린 종가를 오래된 것부터 하나씩 가져오는지 검증. 네트워크 미사용."""
import unittest

from stock_bot.market import quote

# 2026-09-09 이후 영업일 종가 (오름차순)
SERIES = [
    ("2026-09-08", 90.53),
    ("2026-09-09", 89.98),
    ("2026-09-10", 91.10),
    ("2026-09-11", 92.40),
    ("2026-09-14", 93.05),
    ("2026-09-15", 92.10),
    ("2026-09-16", 91.75),
    ("2026-09-17", 90.60),
    ("2026-09-18", 91.00),
]


class TestNextUnprocessedClose(unittest.TestCase):
    def setUp(self):
        self._history = quote.history
        self._last_close = quote.last_close
        self.range_asked = []

        def fake_history(ticker, range_="5y"):
            self.range_asked.append(range_)
            return list(SERIES)

        quote.history = fake_history
        quote.last_close = lambda ticker: SERIES[-1]

    def tearDown(self):
        quote.history = self._history
        quote.last_close = self._last_close

    def test_가장_오래된_미처리분을_먼저_준다(self):
        self.assertEqual(
            quote.next_unprocessed_close("TSLA", "2026-09-09"), ("2026-09-10", 91.10)
        )

    def test_한_번에_하루씩_순서대로_소진된다(self):
        seen = []
        cursor = "2026-09-09"
        while (nxt := quote.next_unprocessed_close("TSLA", cursor)) is not None:
            seen.append(nxt)
            cursor = nxt[0]
        self.assertEqual([d for d, _ in seen], [d for d, _ in SERIES[2:]])
        self.assertEqual(len(seen), 7, "09-10~09-18은 영업일 7일")

    def test_최신까지_처리했으면_None(self):
        self.assertIsNone(quote.next_unprocessed_close("TSLA", "2026-09-18"))

    def test_미래_날짜여도_None(self):
        self.assertIsNone(quote.next_unprocessed_close("TSLA", "2026-12-31"))

    def test_부트스트랩은_과거를_재생하지_않고_최신부터(self):
        self.assertEqual(quote.next_unprocessed_close("TSLA", None), SERIES[-1])

    def test_밀린_일수에_맞춰_조회_구간을_넓힌다(self):
        quote.next_unprocessed_close("TSLA", "2026-09-18")
        self.assertEqual(self.range_asked[-1], "1mo")
        quote.next_unprocessed_close("TSLA", "2020-01-01")
        self.assertEqual(self.range_asked[-1], "5y")


class TestRangeFor(unittest.TestCase):
    def test_형식이_깨진_날짜는_넉넉한_구간으로_폴백(self):
        self.assertEqual(quote._range_for("어제"), "1y")


if __name__ == "__main__":
    unittest.main()
