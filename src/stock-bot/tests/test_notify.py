"""Discord 알림 — fail-closed 계약.

핵심 불변식: **알림 채널이 죽어 있으면 종가도 주문도 건드리지 않는다.**
네트워크 없음(opener 주입), 실계좌 없음.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from datetime import datetime, timezone
from unittest import mock

from stock_bot import cli, journal, notify
from stock_bot.config import SettingsError, require_notify
from tests.test_cron_contract import CLI, make_cfg

SECRET = "123456789/SECRET-TOKEN-abcdef"
URL = f"https://discord.com/api/webhooks/{SECRET}"


class FakeResp:
    def __init__(self, status=204):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code, body=b"{}"):
    return urllib.error.HTTPError(URL, code, "x", {}, io.BytesIO(body))


class TestSplit(unittest.TestCase):
    def test_짧으면_한_조각(self):
        self.assertEqual(notify.split_message("a\nb"), ["a\nb"])

    def test_긴_글은_한도_이내로_나뉘고_내용이_보존된다(self):
        text = "\n".join(f"line {i:04d} " + "x" * 50 for i in range(200))
        parts = notify.split_message(text, 500)
        self.assertTrue(all(len(p) <= 500 for p in parts))
        self.assertEqual("\n".join(parts), text)

    def test_한_줄이_한도보다_길어도_자른다(self):
        parts = notify.split_message("y" * 1250, 500)
        self.assertEqual([len(p) for p in parts], [500, 500, 250])

    def test_빈_문자열도_한_조각(self):
        self.assertEqual(notify.split_message(""), [""])


class TestValidUrl(unittest.TestCase):
    def test_정상(self):
        self.assertTrue(notify.valid_webhook_url(URL))
        self.assertTrue(notify.valid_webhook_url("https://canary.discord.com/api/webhooks/1/a"))

    def test_비정상(self):
        for bad in ("", "http://discord.com/api/webhooks/1/a",
                    "https://evil.example/api/webhooks/1/a",
                    "https://discord.com.evil.example/api/webhooks/1/a",
                    "https://discord.com/other/1/a"):
            self.assertFalse(notify.valid_webhook_url(bad), bad)


class TestSend(unittest.TestCase):
    def test_성공_요청_형태(self):
        seen = []

        def opener(req, timeout):
            seen.append(req)
            return FakeResp(204)

        notify.send(URL, "안녕", opener=opener)
        self.assertEqual(len(seen), 1)
        body = json.loads(seen[0].data.decode("utf-8"))
        self.assertEqual(body["content"], "안녕")
        self.assertEqual(body["allowed_mentions"], {"parse": []}, "@everyone 방지")
        self.assertIn("User-agent", seen[0].headers, "기본 urllib UA는 Discord가 403으로 거른다")

    def test_긴_글은_여러_번_보낸다(self):
        n = []
        notify.send(URL, "z" * 4000, opener=lambda r, timeout: (n.append(1), FakeResp())[1])
        self.assertEqual(len(n), 3)

    def test_5xx는_재시도_후_성공(self):
        calls, sleeps = [], []

        def opener(req, timeout):
            calls.append(1)
            if len(calls) < 3:
                raise http_error(503)
            return FakeResp()

        notify.send(URL, "x", opener=opener, sleep=sleeps.append)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(sleeps), 2)

    def test_계속_실패하면_NotifyError(self):
        def opener(req, timeout):
            raise http_error(500)

        with self.assertRaises(notify.NotifyError):
            notify.send(URL, "x", opener=opener, sleep=lambda s: None)

    def test_4xx는_재시도하지_않는다(self):
        calls = []

        def opener(req, timeout):
            calls.append(1)
            raise http_error(404)

        with self.assertRaises(notify.NotifyError) as e:
            notify.send(URL, "x", opener=opener, sleep=lambda s: None)
        self.assertEqual(len(calls), 1)
        self.assertIn("404", str(e.exception))

    def test_429는_retry_after를_따른다(self):
        calls, sleeps = [], []

        def opener(req, timeout):
            calls.append(1)
            if len(calls) == 1:
                raise http_error(429, b'{"retry_after": 2.5}')
            return FakeResp()

        notify.send(URL, "x", opener=opener, sleep=sleeps.append)
        self.assertEqual(sleeps, [2.5])

    def test_연결_실패도_NotifyError_이고_URL이_새지_않는다(self):
        def opener(req, timeout):
            raise urllib.error.URLError(f"cannot reach {URL}")

        with self.assertRaises(notify.NotifyError) as e:
            notify.send(URL, "x", opener=opener, sleep=lambda s: None)
        self.assertNotIn(SECRET, str(e.exception))

    def test_URL이_없거나_틀리면_네트워크_전에_멈춘다(self):
        def boom(req, timeout):
            raise AssertionError("네트워크를 건드리면 안 된다")

        for bad in ("", "https://evil.example/api/webhooks/1/a"):
            with self.assertRaises(notify.NotifyError):
                notify.send(bad, "x", opener=boom)


class TestRequireNotify(unittest.TestCase):
    def test_정상이면_통과(self):
        require_notify(make_cfg(discord_webhook_url=URL))

    def test_없으면_멈춘다_이름으로_알린다(self):
        with self.assertRaises(SettingsError) as e:
            require_notify(make_cfg())
        self.assertIn("STOCK_DISCORD_WEBHOOK_URL", str(e.exception))

    def test_형식이_틀리면_멈추고_값은_노출하지_않는다(self):
        bad = "https://evil.example/api/webhooks/" + SECRET
        with self.assertRaises(SettingsError) as e:
            require_notify(make_cfg(discord_webhook_url=bad))
        self.assertNotIn(SECRET, str(e.exception))


class TestCmdRunFailClosed(unittest.TestCase):
    """service.run_once와 notify.send를 가짜로 바꿔 호출 순서와 종료 코드를 본다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = make_cfg(db_path=os.path.join(self.tmp.name, "stock_bot.db"),
                            discord_webhook_url=URL)
        self.events = []

    def _run(self, send, run_once):
        with mock.patch("stock_bot.notify.send", side_effect=send), \
                mock.patch("stock_bot.service.run_once", side_effect=run_once), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out, \
                mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            try:
                code = cli.cmd_run(self.cfg, False)
            except Exception as e:  # noqa: BLE001
                code = e
        return code, out.getvalue(), err.getvalue()

    def _journal_text(self):
        d = journal.journal_dir(self.cfg.db_path)
        if not os.path.isdir(d):
            return ""
        out = ""
        for f in os.listdir(d):
            with open(os.path.join(d, f), encoding="utf-8") as fh:
                out += fh.read()
        return out

    def test_사전점검이_실패하면_run_once를_부르지_않는다(self):
        def send(url, text):
            self.events.append("send")
            raise notify.NotifyError("Discord 전송 실패(3회 시도): HTTP 500")

        def run_once(cfg):
            self.events.append("run_once")
            return "x", None

        code, out, _ = self._run(send, run_once)
        self.assertEqual(code, cli.EXIT_GUARD)
        self.assertEqual(self.events, ["send"], "종가·주문 경로는 열리지 않아야 한다")
        self.assertIn("[BLOCKED]", out)
        self.assertIn("알림 채널 불통", self._journal_text(), "멈춘 사실도 일지에 남는다")

    def test_정상이면_시작_알림_다음_실행_다음_보고(self):
        def send(url, text):
            self.events.append(("send", text.split("\n", 1)[0]))

        def run_once(cfg):
            self.events.append("run_once")
            return None, None

        code, out, _ = self._run(send, run_once)
        self.assertEqual(code, cli.EXIT_OK)
        self.assertEqual(self.events[0][0], "send")
        self.assertIn("run 시작", self.events[0][1])
        self.assertEqual(self.events[1], "run_once")
        self.assertEqual(self.events[2][0], "send")
        self.assertIn("[SKIP]", self.events[2][1])
        self.assertIn("[SKIP]", self._journal_text())

    def test_실행_후_보고가_실패하면_종료코드가_20(self):
        n = []

        def send(url, text):
            n.append(1)
            if len(n) == 2:
                raise notify.NotifyError("Discord 거부: HTTP 404")

        code, out, err = self._run(send, lambda cfg: (None, None))
        self.assertEqual(code, cli.EXIT_GUARD)
        self.assertIn("[SKIP]", out, "보고문은 stdout(컨테이너 로그)에 남는다")
        self.assertIn("보고 실패", err)

    def test_실행_중_예외는_알리고_다시_던진다(self):
        sent = []

        def send(url, text):
            sent.append(text)

        def run_once(cfg):
            raise RuntimeError("boom")

        code, _, _ = self._run(send, run_once)
        self.assertIsInstance(code, RuntimeError, "예외가 삼켜지면 exit 1이 사라진다")
        self.assertTrue(any("💥" in t and "RuntimeError" in t for t in sent))
        self.assertIn("RuntimeError", self._journal_text())

    def test_보고문과_알림에_웹훅_URL이_없다(self):
        sent = []
        self._run(lambda url, text: sent.append(text), lambda cfg: (None, None))
        self.assertTrue(all(SECRET not in t for t in sent))
        self.assertNotIn(SECRET, self._journal_text())


class TestRealProcess(unittest.TestCase):
    def test_웹훅_없으면_EXIT_SETTINGS(self):
        with tempfile.TemporaryDirectory() as d:
            s = os.path.join(d, "s.yaml")
            with open(s, "w", encoding="utf-8") as f:
                f.write("ticker: QLD\nvr_mode: 거치식\nkis_cash_share: 1.0\n")
            e = dict(os.environ)
            e.update({"STOCK_DB_PATH": os.path.join(d, "stock_bot.db"),
                      "STOCK_SETTINGS_PATH": s,
                      "KIS_APP_KEY": "k", "KIS_APP_SECRET": "s", "KIS_CANO": "c",
                      "STOCK_DISCORD_WEBHOOK_URL": ""})
            r = subprocess.run([sys.executable, CLI, "run"], capture_output=True,
                               text=True, env=e, cwd="/tmp")
            self.assertEqual(r.returncode, cli.EXIT_SETTINGS, r.stderr[-400:])
            self.assertIn("STOCK_DISCORD_WEBHOOK_URL", r.stderr)
            self.assertFalse(os.path.exists(os.path.join(d, "stock_bot.db")), "DB를 만들지도 않는다")


if __name__ == "__main__":
    unittest.main()
