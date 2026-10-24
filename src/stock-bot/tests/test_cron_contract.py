"""cron 전환 전제조건 — 종료 코드 / KIS 자격증명 확인 / 요약 줄.

cron은 사람이 아니라 **종료 코드와 stdout 첫 줄**로 판단한다. 그 둘이 상황을
정확히 반영하는지 검증한다. 네트워크·실계좌 없음.
"""
import os
import subprocess
import sys
import tempfile
import unittest

from stock_bot import service
from stock_bot.broker import order_status
from stock_bot.cli import EXIT_GUARD, EXIT_OK, EXIT_SEND_ANOMALY, EXIT_SETTINGS
from stock_bot.config import Config, SettingsError, require_kis
from stock_bot.domain.models import AccountState, Order, Side
from stock_bot.service import DailyResult

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(REPO, "stock_bot", "cli.py")


def make_cfg(**over):
    base = dict(
        ticker="QLD", vr_mode="거치식", strategy_engine="value_rebalancing",
        db_path="/tmp/x/stock_bot.db", cycle_len=10, band_rate=0.15,
        sell_bands=(0.025, 0.05, 0.075, 0.10),
        init_v=1000.0, init_pool=1000.0, init_shares=0.0, init_avg_cost=0.0,
        cycle_cash_flow=None, G=None, max_pool_usage=None, kis_cash_share=1.0,
        dry_run=True, kill_switch_path="/tmp/__none__",
        max_daily_orders=2, max_daily_amount=500.0, max_order_amount=500.0,
        account_ledger_path="/tmp/__t_ledger.db",
        max_account_daily_orders=2, max_account_daily_amount=500.0,
        kis_env="prod", kis_app_key="k", kis_app_secret="s", kis_cano="c",
        kis_acnt_prdt_cd="01", kis_crcy_cd="USD",
        outline_config_path="",  # 테스트가 실제 Outline에 닿지 않게 기본은 사본 끔
    )
    base.update(over)
    return Config(**base)


def st(status, **over):
    base = dict(odno="A1", symbol="QLD", ord_dt="20260921", ord_qty=1.0,
                filled_qty=0.0, status=status, reason="", side="BUY", limit_price=91.19)
    base.update(over)
    return order_status.OrderStatus(**base)


def result_with(**over):
    base = dict(orders=[], cycle_started=False)
    base.update(over)
    return DailyResult(**base)


# ── ① 종료 코드 판정 근거 ──

class TestAnomalyDetection(unittest.TestCase):
    def test_정상이면_이상_없음(self):
        r = result_with(sends=[{"order": "BUY 1주", "sent": True, "gate": "", "odno": "A1"}],
                        order_statuses=[st(order_status.FILLED, filled_qty=1.0)])
        self.assertEqual(service.send_anomalies(r), [])

    def test_게이트_차단은_이상이_아니다(self):
        for gate in ("DRY_RUN", "BACKLOG", "DAILY_LIMIT", "ACCOUNT_LIMIT",
                     "KILL_SWITCH", "RECONCILE", "IDEMPOTENT", "EXCHANGE"):
            r = result_with(sends=[{"order": "BUY 1주", "sent": False, "gate": gate,
                                    "reason": "차단", "odno": ""}])
            self.assertEqual(service.send_anomalies(r), [],
                             f"{gate}은 설계된 동작이라 이상이 아니다")

    def test_UNCONFIRMED_전송은_이상이다(self):
        r = result_with(sends=[{"order": "BUY 1주", "sent": False,
                                "gate": order_status.UNCONFIRMED, "reason": "타임아웃",
                                "odno": ""}])
        self.assertEqual(len(service.send_anomalies(r)), 1)

    def test_SEND_REJECTED도_이상이다(self):
        r = result_with(sends=[{"order": "BUY 1주", "sent": False,
                                "gate": order_status.SEND_REJECTED, "reason": "거부",
                                "odno": ""}])
        self.assertEqual(len(service.send_anomalies(r)), 1)

    def test_상태_UNKNOWN은_이상이다(self):
        r = result_with(order_statuses=[st(order_status.UNKNOWN)])
        self.assertEqual(len(service.send_anomalies(r)), 1)

    def test_미확인분_확인실패는_이상이다(self):
        r = result_with(resolved=[{"key": "k", "resolved": False, "note": "조회 실패"}])
        self.assertEqual(len(service.send_anomalies(r)), 1)

    def _cancelled(self, close, limit=91.19, side="BUY"):
        """2026-07-15 QLD 건과 같은 모양 — 취소됐고 체결 0."""
        return result_with(
            order_statuses=[st(order_status.CANCELLED, ord_dt="20260715",
                               reason="DFD 주문종료 취소", side=side, limit_price=limit)],
            closes_by_date={} if close is None else {"2026-07-15": close})

    def test_체결_조건을_충족했는데_취소면_이상이다(self):
        # 실제 07-15 건: 종가 $91.02 ≤ 지정가 $91.19 인데 체결 0
        bad = service.send_anomalies(self._cancelled(91.02))
        self.assertEqual(len(bad), 1, "됐어야 하는데 안 된 것은 이상이다")
        self.assertIn("체결 조건을 충족", bad[0])

    def test_체결_조건을_못_채워_안_붙은_것은_정상이다(self):
        # 종가 $92.00 > 지정가 $91.19 → LOC 매수는 원래 안 붙는다
        r = self._cancelled(92.00)
        self.assertEqual(service.send_anomalies(r), [],
                         "이걸 매번 이상으로 올리면 알림이 일상이 되어 아무도 안 본다")
        self.assertEqual(len(service.attention_notes(r)), 1, "그래도 요약 줄에는 남긴다")

    def test_매도는_부등호가_반대다(self):
        # LOC 매도는 종가 ≥ 지정가에서 체결된다
        self.assertEqual(len(service.send_anomalies(
            self._cancelled(92.00, side="SELL"))), 1, "종가가 위인데 안 팔렸으면 이상")
        self.assertEqual(service.send_anomalies(
            self._cancelled(91.02, side="SELL")), [], "종가가 아래면 안 붙는 게 정상")

    def test_그날_종가를_모르면_이상으로_본다(self):
        bad = service.send_anomalies(self._cancelled(None))
        self.assertEqual(len(bad), 1, "판정 불가를 '정상'으로 낙관하면 안 된다")
        self.assertIn("판정 불가", bad[0])

    def test_지정가가_없으면_이상으로_본다(self):
        bad = service.send_anomalies(self._cancelled(91.02, limit=0.0))
        self.assertEqual(len(bad), 1)


class TestSummaryLine(unittest.TestCase):
    def state(self, shares=0.0):
        return AccountState(shares=shares, avg_cost=0, pool=4197.57, V=3117.07,
                            cycle_index=4, day_in_cycle=1)

    def line(self, r, shares=0.0):
        return service.summary_line(make_cfg(), self.state(shares), r, "2026-09-21", 91.0)

    def test_정상_전송은_OK와_주문번호(self):
        line = self.line(result_with(
            sends=[{"order": "BUY 1주 @ $91.00", "sent": True, "gate": "", "odno": "A1"}]), 1.0)
        self.assertTrue(line.startswith("[OK]"), line)
        self.assertIn("odno A1", line)

    def test_게이트_차단도_OK이고_차단_게이트명이_보인다(self):
        line = self.line(result_with(
            sends=[{"order": "BUY 23주", "sent": False, "gate": "DRY_RUN",
                    "reason": "차단", "odno": ""}]))
        self.assertTrue(line.startswith("[OK]"), line)
        self.assertIn("DRY_RUN", line)

    def test_취소는_ATTENTION으로_첫줄에_뜬다(self):
        line = self.line(result_with(
            sends=[{"order": "BUY 1주", "sent": False, "gate": "RECONCILE",
                    "reason": "차단", "odno": ""}],
            order_statuses=[st(order_status.CANCELLED, reason="DFD 주문종료 취소")]))
        self.assertTrue(line.startswith("[ATTENTION]"), line)
        self.assertIn("CANCELLED", line)

    def test_UNCONFIRMED도_ATTENTION이다(self):
        line = self.line(result_with(
            sends=[{"order": "BUY 1주", "sent": False, "gate": order_status.UNCONFIRMED,
                    "reason": "타임아웃", "odno": ""}]))
        self.assertTrue(line.startswith("[ATTENTION]"), line)


# ── ② KIS 자격증명 확인 ──

class TestRequireKis(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("STOCK_ALLOW_NO_KIS")
        os.environ.pop("STOCK_ALLOW_NO_KIS", None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("STOCK_ALLOW_NO_KIS", None)
        else:
            os.environ["STOCK_ALLOW_NO_KIS"] = self._saved

    def test_자격증명이_있으면_통과(self):
        require_kis(make_cfg())

    def test_없으면_멈춘다(self):
        with self.assertRaises(SettingsError) as e:
            require_kis(make_cfg(kis_app_key="", kis_app_secret="", kis_cano=""))
        msg = str(e.exception)
        for name in ("KIS_APP_KEY", "KIS_APP_SECRET", "KIS_CANO"):
            self.assertIn(name, msg, "무엇이 비었는지 이름으로 알려줘야 한다")

    def test_하나만_비어도_멈추고_그것만_지목한다(self):
        with self.assertRaises(SettingsError) as e:
            require_kis(make_cfg(kis_cano=""))
        self.assertIn("KIS_CANO", str(e.exception))
        self.assertNotIn("KIS_APP_KEY", str(e.exception))

    def test_시크릿_값은_메시지에_없다(self):
        secret = "SUPERSECRET-APPKEY-0123456789"
        with self.assertRaises(SettingsError) as e:
            require_kis(make_cfg(kis_app_key=secret, kis_app_secret=secret, kis_cano=""))
        self.assertNotIn(secret, str(e.exception),
                         "이름만 알려야 한다 — 값이 로그·알림으로 새면 안 된다")

    def test_더_이상_안_쓰는_변수는_요구하지_않는다(self):
        self.assertNotIn("KIS_OVRS_EXCG_CD", " ".join(
            make_cfg(kis_app_key="").missing_kis_creds() + ["KIS_APP_KEY"]))

    def test_명시적_opt_out이면_통과(self):
        os.environ["STOCK_ALLOW_NO_KIS"] = "1"
        require_kis(make_cfg(kis_app_key="", kis_app_secret="", kis_cano=""))

    def test_opt_out이_거짓값이면_여전히_멈춘다(self):
        for v in ("", "0", "false", "no", "off"):
            os.environ["STOCK_ALLOW_NO_KIS"] = v
            with self.assertRaises(SettingsError):
                require_kis(make_cfg(kis_app_key="", kis_app_secret="", kis_cano=""))


# ── 실제 프로세스 종료 코드 (테스트가 코드를 제대로 읽는지까지 확인) ──

class TestRealExitCodes(unittest.TestCase):
    """subprocess로 실행해 **실제 exit status**를 읽는다.

    오늘 `tail`이 종료코드를 삼킨 것과 같은 계열이라, 테스트 자체가
    코드를 제대로 읽는지를 먼저 증명한다.
    """

    def run_cli(self, *args, env=None):
        e = dict(os.environ)
        e.pop("STOCK_SETTINGS_PATH", None)
        e.update(env or {})
        return subprocess.run([sys.executable, CLI, *args], capture_output=True,
                              text=True, env=e, cwd="/tmp")

    def test_테스트_하니스가_종료코드를_읽는다(self):
        ok = subprocess.run([sys.executable, "-c", "raise SystemExit(0)"])
        bad = subprocess.run([sys.executable, "-c", "raise SystemExit(37)"])
        self.assertEqual((ok.returncode, bad.returncode), (0, 37))

    def test_설정_없으면_EXIT_SETTINGS(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.run_cli("run", env={
                "STOCK_DB_PATH": os.path.join(d, "stock_bot.db"),
                "STOCK_SETTINGS_PATH": os.path.join(d, "__none__.yaml")})
        self.assertEqual(r.returncode, EXIT_SETTINGS, r.stderr[-400:])
        self.assertIn("[SETTINGS]", r.stderr)

    def test_KIS_자격증명_없으면_EXIT_SETTINGS(self):
        with tempfile.TemporaryDirectory() as d:
            s = os.path.join(d, "s.yaml")
            with open(s, "w", encoding="utf-8") as f:
                f.write("ticker: QLD\nvr_mode: 거치식\nkis_cash_share: 1.0\n")
            r = self.run_cli("run", env={
                "STOCK_DB_PATH": os.path.join(d, "stock_bot.db"),
                "STOCK_SETTINGS_PATH": s,
                "KIS_APP_KEY": "", "KIS_APP_SECRET": "", "KIS_CANO": ""})
        self.assertEqual(r.returncode, EXIT_SETTINGS, r.stderr[-400:])
        self.assertIn("KIS_APP_KEY", r.stderr)

    def test_읽기전용_명령은_0(self):
        r = self.run_cli("status", env={
            "STOCK_DB_PATH": "/workspace/stock-data/stock_bot.db"})
        self.assertEqual(r.returncode, EXIT_OK, r.stderr[-400:])


class TestExitCodeConstants(unittest.TestCase):
    def test_구분이_유지된다(self):
        self.assertEqual(
            (EXIT_OK, EXIT_SEND_ANOMALY, EXIT_GUARD, EXIT_SETTINGS), (0, 10, 20, 30),
            "cron이 '10·20은 즉시 알림, 30은 배포 사고'로 라우팅한다")


if __name__ == "__main__":
    unittest.main()
