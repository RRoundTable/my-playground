"""운용 일지 — 정본은 로컬 일자별 파일, Outline은 best-effort 사본.

Outline은 진짜 HTTP 서버(스레드)로 흉내 낸다. 네트워크 밖으로 나가지 않는다.
"""
import io
import json
import os
import stat
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from stock_bot import cli, journal, notify, outline
from tests.test_cron_contract import make_cfg
from tests.test_notify import URL

def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


TOKEN = "ol_api_SUPERSECRETTOKEN123"
DOC = "81087655-2a9b-4346-afa6-cd480d4720b0"
NOW = datetime(2026, 10, 26, 15, 30, 5, tzinfo=timezone.utc)

BODY = """인트로 문단

## 미결 안건 — 사람 승인 대기

1. ~~닫힌 것~~
7. 🔴 기본예탁금

## 2026-09-22 (화) — 첫 주문 거부

### 주문
- 내용 \\[\\[링크\\]\\]

## 2026-09-21 (월) — 퇴역
끝"""

RECORD = dict(ticker="QLD", quote_date="2026-10-23", close=96.27, pool=4197.57, V=3117.07,
              band_low=2649.5, band_high=3584.6, shares=0.0, cycle_started=False,
              source="KIS", orders=["BUY 1주 @ $96.27 (종가)"], sent=[],
              blocked_gate="DRY_RUN", notes=[])


class FakeOutline:
    """documents.info / documents.update 만 아는 가짜 Outline."""

    def __init__(self, text=BODY, info_status=200, update_status=200):
        self.text, self.info_status, self.update_status = text, info_status, update_status
        self.calls, self.auth = [], []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(n))
                outer.calls.append((self.path, payload))
                outer.auth.append(self.headers.get("Authorization"))
                if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                    return self._send(401, {"ok": False, "error": "authentication_required"})
                if self.path == "/api/documents.info":
                    if outer.info_status != 200:
                        return self._send(outer.info_status, {"ok": False})
                    return self._send(200, {"ok": True, "data": {"id": payload["id"], "text": outer.text}})
                if self.path == "/api/documents.update":
                    if outer.update_status != 200:
                        return self._send(outer.update_status, {"ok": False})
                    outer.text = payload["text"]
                    return self._send(200, {"ok": True, "data": {"text": outer.text}})
                self._send(404, {"ok": False})

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


class OutlineBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fake = FakeOutline()
        self.addCleanup(self.fake.close)
        self.conf = os.path.join(self.tmp.name, "config.json")
        self.write_conf(self.fake.url, TOKEN)

    def write_conf(self, url, token):
        with open(self.conf, "w", encoding="utf-8") as f:
            json.dump({"url": url, "token": token, "default_collection": "x"}, f)


class TestMergeSection(unittest.TestCase):
    def test_절이_없으면_첫_날짜절_앞에_만들고_나머지는_그대로(self):
        out = outline.merge_section(BODY, ["2026-10-26 a"])
        head, tail = BODY.split("## 2026-09-22")
        self.assertTrue(out.startswith(head), "미결 안건까지의 앞부분은 한 글자도 안 바뀐다")
        self.assertTrue(out.endswith("## 2026-09-22" + tail), "날짜별 서술도 그대로")
        self.assertIn("## 봇 자동 기록", out)
        self.assertLess(out.index("## 봇 자동 기록"), out.index("## 2026-09-22"))
        self.assertGreater(out.index("## 봇 자동 기록"), out.index("## 미결 안건"))

    def test_멱등이다(self):
        once = outline.merge_section(BODY, ["2026-10-26 a", "2026-10-27 b"])
        self.assertEqual(outline.merge_section(once, ["2026-10-26 a", "2026-10-27 b"]), once)

    def test_최신이_위로_쌓이고_기존_줄은_보존된다(self):
        one = outline.merge_section(BODY, ["2026-10-26 a"])
        two = outline.merge_section(one, ["2026-10-27 b"])
        self.assertLess(two.index("2026-10-27 b"), two.index("2026-10-26 a"))

    def test_Outline이_붙인_이스케이프가_있어도_중복이_안_생긴다(self):
        one = outline.merge_section(BODY, ["x [OK] y"])
        escaped = one.replace("[OK]", "\\[OK\\]")
        two = outline.merge_section(escaped, ["x [OK] y"])
        self.assertEqual(two.count("OK"), 1)

    def test_사람이_쓴_절은_봇_절_뒤에서도_보존된다(self):
        one = outline.merge_section(BODY, ["2026-10-26 a"])
        self.assertIn("7. 🔴 기본예탁금", one)
        self.assertIn("- 내용 \\[\\[링크\\]\\]", one, "다른 절의 이스케이프를 건드리지 않는다")

    def test_상한을_넘으면_오래된_것부터_버린다(self):
        lines = [f"2026-10-{d:02d} x" for d in range(1, 21)]
        out = outline.merge_section(BODY, lines, cap=5)
        self.assertEqual(out.count("\n- 2026-10-"), 5)
        self.assertIn("2026-10-20 x", out)
        self.assertNotIn("2026-10-01 x", out)

    def test_날짜절이_없는_문서는_끝에_붙인다(self):
        out = outline.merge_section("그냥 본문", ["2026-10-26 a"])
        self.assertTrue(out.startswith("그냥 본문"))
        self.assertIn("- 2026-10-26 a", out)


class TestSync(OutlineBase):
    def test_받아서_고쳐_통째로_돌려보낸다(self):
        outline.sync(self.conf, DOC, ["2026-10-26 a"])
        self.assertEqual([c[0] for c in self.fake.calls],
                         ["/api/documents.info", "/api/documents.update"])
        self.assertEqual(self.fake.calls[1][1]["id"], DOC)
        self.assertIn("- 2026-10-26 a", self.fake.text)
        self.assertIn("7. 🔴 기본예탁금", self.fake.text, "본문 전체 교체지만 기존 내용은 살아 있다")
        self.assertTrue(all(a == f"Bearer {TOKEN}" for a in self.fake.auth))

    def test_바뀔_게_없으면_update를_부르지_않는다(self):
        outline.sync(self.conf, DOC, ["2026-10-26 a"])
        n = len(self.fake.calls)
        outline.sync(self.conf, DOC, ["2026-10-26 a"])
        self.assertEqual([c[0] for c in self.fake.calls[n:]], ["/api/documents.info"])

    def test_키_만료_401은_OutlineError_이고_토큰이_새지_않는다(self):
        self.write_conf(self.fake.url, "expired-token-ZZZ")
        with self.assertRaises(outline.OutlineError) as e:
            outline.sync(self.conf, DOC, ["x"])
        self.assertIn("401", str(e.exception))
        self.assertNotIn("expired-token-ZZZ", str(e.exception))
        self.assertEqual(len(self.fake.calls), 1, "인증이 안 되면 update까지 가지 않는다")

    def test_5xx(self):
        self.fake.update_status = 500
        with self.assertRaises(outline.OutlineError) as e:
            outline.sync(self.conf, DOC, ["x"])
        self.assertIn("500", str(e.exception))

    def test_문서가_없으면_404(self):
        self.fake.info_status = 404
        with self.assertRaises(outline.OutlineError):
            outline.sync(self.conf, DOC, ["x"])

    def test_본문을_못_받으면_교체하지_않는다(self):
        self.fake.text = ""
        with self.assertRaises(outline.OutlineError):
            outline.sync(self.conf, DOC, ["x"])
        self.assertEqual([c[0] for c in self.fake.calls], ["/api/documents.info"],
                         "빈 본문 위에 쓰면 문서 전체가 날아간다")

    def test_연결_불가(self):
        self.write_conf("http://127.0.0.1:1", TOKEN)
        with self.assertRaises(outline.OutlineError) as e:
            outline.sync(self.conf, DOC, ["x"], timeout=1)
        self.assertNotIn(TOKEN, str(e.exception))

    def test_설정_파일_문제(self):
        for content in (None, "not json", json.dumps({"url": "http://x"}),
                        json.dumps({"url": "ftp://x", "token": "t"}),
                        json.dumps({"url": "http://x", "token": ""})):
            with self.subTest(content=content):
                if content is None:
                    path = os.path.join(self.tmp.name, "nope.json")
                else:
                    path = os.path.join(self.tmp.name, "bad.json")
                    _write(path, content)
                with self.assertRaises(outline.OutlineError):
                    outline.sync(path, DOC, ["x"])


class TestJournalLocal(unittest.TestCase):
    def test_make_line_정상(self):
        line = journal.make_line("QLD", 0, "", RECORD, NOW)
        for frag in ("2026-10-26 월 15:30 UTC", "QLD", "[OK]", "종가일 2026-10-23 금 종가 $96.27", "풀 $4,197.57",
                     "V $3,117.07", "밴드 $2,650–$3,585", "보유 0주",
                     "주문안 BUY 1주 @ $96.27 (종가) 차단[DRY_RUN]", "출처 KIS", "이상: 없음"):
            self.assertIn(frag, line)
        self.assertNotIn("\n", line)

    def test_요일은_locale과_무관하게_한글_한_글자(self):
        self.assertEqual([journal.weekday_ko(f"2026-10-{d}") for d in range(19, 26)],
                         list("월화수목금토일"))
        self.assertEqual(journal.dated(datetime(2026, 10, 24, tzinfo=timezone.utc)), "2026-10-24 토")

    def test_파일명은_요일_없이_날짜만(self):
        with tempfile.TemporaryDirectory() as d:
            p = journal.write(os.path.join(d, "stock_bot.db"), "QLD", 0, "m", "l", NOW)
            self.assertEqual(os.path.basename(p), "2026-10-26.md")
            self.assertTrue(_read(p).startswith("# 2026-10-26 월 운용 일지\n"))

    def test_make_line_이상징후와_전송(self):
        r = dict(RECORD, notes=["전송 UNCONFIRMED: BUY 1주"], sent=["BUY 1주"], source="모델")
        line = journal.make_line("QLD", 10, "", r, NOW)
        self.assertIn("[ATTENTION]", line)
        self.assertIn("전송 BUY 1주", line)
        self.assertIn("이상: 전송 UNCONFIRMED: BUY 1주", line)
        self.assertIn("출처 모델", line)

    def test_make_line_주문없음(self):
        line = journal.make_line("QLD", 0, "", dict(RECORD, orders=[], blocked_gate=None), NOW)
        self.assertIn("주문 없음", line)

    def test_make_line_기록이_없으면_보고문_첫_줄과_종료코드(self):
        line = journal.make_line("QLD", 20, "\n🛑 [BLOCKED] QLD — 안전장치\n둘째 줄", None, NOW)
        self.assertIn("[BLOCKED] QLD — 안전장치", line)
        self.assertIn("exit 20", line)
        self.assertNotIn("둘째", line)

    def test_날짜별_파일이고_같은_날은_이어_붙인다(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "stock_bot.db")
            t2 = datetime(2026, 10, 26, 16, 0, tzinfo=timezone.utc)
            t3 = datetime(2026, 10, 27, 15, 30, tzinfo=timezone.utc)
            p1 = journal.write(db, "QLD", 0, "첫 보고", "L1", NOW)
            p2 = journal.write(db, "QLD", 20, "둘째 보고", "L2", t2)
            p3 = journal.write(db, "QLD", 0, "다음 날", "L3", t3)
            self.assertEqual(p1, p2)
            self.assertNotEqual(p1, p3)
            self.assertEqual(p1, os.path.join(d, "journal", "2026-10-26.md"))
            text = _read(p1)
            self.assertEqual(text.count("# 2026-10-26 월 운용 일지"), 1)
            for frag in ("첫 보고", "둘째 보고", "요약: L1", "요약: L2", "exit 20"):
                self.assertIn(frag, text)
            self.assertEqual(journal.recent_lines(db, t3), ["L1", "L2", "L3"])

    def test_종목별_DB_디렉터리를_따른다(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "stock-data"), os.path.join(d, "stock-data-soxl")
            os.makedirs(a)
            os.makedirs(b)
            pa = journal.write(os.path.join(a, "stock_bot.db"), "QLD", 0, "m", "l", NOW)
            pb = journal.write(os.path.join(b, "stock_bot.db"), "SOXL", 0, "m", "l", NOW)
            self.assertEqual(os.path.dirname(pa), os.path.join(a, "journal"))
            self.assertEqual(os.path.dirname(pb), os.path.join(b, "journal"))

    def test_recent_lines는_기간_밖_파일을_뺀다(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "stock_bot.db")
            old = datetime(2026, 9, 1, 15, 30, tzinfo=timezone.utc)
            journal.write(db, "QLD", 0, "m", "OLD", old)
            journal.write(db, "QLD", 0, "m", "NEW", NOW)
            self.assertEqual(journal.recent_lines(db, NOW, days=14), ["NEW"])

    def test_쓸_수_없으면_JournalError(self):
        with self.assertRaises(journal.JournalError):
            journal.write("/proc/nope/stock_bot.db", "QLD", 0, "x", "l", NOW)
        with self.assertRaises(journal.JournalError):
            journal.preflight("/proc/nope/stock_bot.db")

    @unittest.skipIf(os.geteuid() == 0, "root는 권한 거부가 안 된다")
    def test_읽기전용_디렉터리(self):
        with tempfile.TemporaryDirectory() as d:
            os.chmod(d, stat.S_IRUSR | stat.S_IXUSR)
            try:
                with self.assertRaises(journal.JournalError):
                    journal.preflight(os.path.join(d, "stock_bot.db"))
            finally:
                os.chmod(d, stat.S_IRWXU)

    def test_preflight는_흔적을_남기지_않는다(self):
        with tempfile.TemporaryDirectory() as d:
            journal.preflight(os.path.join(d, "stock_bot.db"))
            self.assertEqual(os.listdir(os.path.join(d, "journal")), [])


class FakeResult:
    def __init__(self, record):
        self.record = record


class TestCmdRunJournal(OutlineBase):
    """cmd_run 전체 흐름 — run_once·Discord는 가짜, Outline은 가짜 서버."""

    def setUp(self):
        super().setUp()
        self.cfg = make_cfg(db_path=os.path.join(self.tmp.name, "stock_bot.db"),
                            discord_webhook_url=URL, outline_config_path=self.conf,
                            outline_doc_id=DOC)
        self.sent = []

    def run_cmd(self, run_once=None, send=None, cfg=None):
        run_once = run_once or (lambda c: ("[OK] QLD 보고문\n본문", FakeResult(RECORD)))
        send = send or (lambda url, text: self.sent.append(text))
        with mock.patch("stock_bot.notify.send", side_effect=send), \
                mock.patch("stock_bot.service.run_once", side_effect=run_once), \
                mock.patch("stock_bot.service.send_anomalies", return_value=[]), \
                mock.patch("stock_bot.service.attention_notes", return_value=[]), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out, \
                mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            try:
                code = cli.cmd_run(cfg or self.cfg, False)
            except Exception as e:  # noqa: BLE001
                code = e
        return code, out.getvalue(), err.getvalue()

    def jtext(self):
        d = journal.journal_dir(self.cfg.db_path)
        return "".join(_read(os.path.join(d, f)) for f in sorted(os.listdir(d))) \
            if os.path.isdir(d) else ""

    def test_정상이면_로컬_일지와_Outline_사본이_둘_다_남는다(self):
        code, out, err = self.run_cmd()
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("요약: ", self.jtext())
        self.assertIn("종가 $96.27", self.jtext())
        self.assertIn("## 봇 자동 기록", self.fake.text)
        self.assertIn("종가 $96.27", self.fake.text)
        self.assertNotIn("Outline 갱신 실패", self.sent[-1])
        self.assertNotIn("WARN", err)

    def test_Outline_키가_만료돼도_멈추지_않고_Discord에_한_줄_알린다(self):
        self.write_conf(self.fake.url, "expired-ZZZ")
        code, out, err = self.run_cmd()
        self.assertEqual(code, cli.EXIT_OK, "best-effort — 종료코드가 오르지 않는다")
        self.assertIn("종가 $96.27", self.jtext(), "정본은 이미 남았다")
        self.assertIn("⚠️ Outline 갱신 실패", self.sent[-1])
        self.assertIn("[OK] QLD 보고문", self.sent[-1], "보고문은 그대로 간다")
        self.assertIn("401", self.sent[-1])
        for t in (*self.sent, out, err):
            self.assertNotIn("expired-ZZZ", t)
            self.assertNotIn(TOKEN, t)

    def test_Outline_접속정보_파일이_없어도_같다(self):
        cfg = make_cfg(db_path=self.cfg.db_path, discord_webhook_url=URL,
                       outline_config_path=os.path.join(self.tmp.name, "none.json"))
        code, _, _ = self.run_cmd(cfg=cfg)
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("Outline 갱신 실패", self.sent[-1])

    def test_Outline이_5xx여도_멈추지_않는다(self):
        self.fake.update_status = 503
        code, _, _ = self.run_cmd()
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("Outline 갱신 실패", self.sent[-1])

    def test_사본을_끄면_Outline을_부르지_않는다(self):
        cfg = make_cfg(db_path=self.cfg.db_path, discord_webhook_url=URL, outline_config_path="")
        code, _, _ = self.run_cmd(cfg=cfg)
        self.assertEqual(code, cli.EXIT_OK)
        self.assertEqual(self.fake.calls, [])
        self.assertNotIn("Outline", self.sent[-1])

    def test_Outline이_막힌_날의_줄은_다음_성공_때_따라잡는다(self):
        from datetime import timedelta
        past = datetime.now(timezone.utc) - timedelta(days=3)
        journal.write(self.cfg.db_path, "QLD", 0, "m",
                      journal.make_line("QLD", 0, "PAST-LINE-막혔던-날", None, past), past)
        self.assertNotIn("PAST-LINE", self.fake.text)
        code, _, _ = self.run_cmd()
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("PAST-LINE-막혔던-날", self.fake.text)
        self.assertIn("종가 $96.27", self.fake.text)
        self.assertLess(self.fake.text.index("종가 $96.27"), self.fake.text.index("PAST-LINE"),
                        "최신이 위")

    def test_일지를_쓸_수_없으면_실행하지_않는다(self):
        events = []
        with mock.patch("stock_bot.journal.preflight",
                        side_effect=journal.JournalError("일지 디렉터리에 쓸 수 없음")):
            code, out, _ = self.run_cmd(
                run_once=lambda c: (events.append("run_once"), (None, None))[1],
                send=lambda url, text: events.append(("send", text)))
        self.assertEqual(code, cli.EXIT_GUARD)
        self.assertNotIn("run_once", events, "종가·주문 경로는 열리지 않는다")
        self.assertTrue(any(e[0] == "send" and "일지" in e[1] for e in events))
        self.assertEqual(self.fake.calls, [])

    def test_실행_후_로컬_쓰기가_실패하면_결과를_Discord로_알리고_멈춘다(self):
        with mock.patch("stock_bot.journal.write",
                        side_effect=journal.JournalError("일지 파일 쓰기 실패: No space left")):
            code, _, err = self.run_cmd()
        self.assertEqual(code, cli.EXIT_GUARD)
        self.assertEqual(self.fake.calls, [], "정본이 없으면 사본도 만들지 않는다")
        self.assertIn("로컬 일지 쓰기 실패", self.sent[-1])
        self.assertIn("[OK] QLD 보고문", self.sent[-1], "실행 결과가 함께 간다")
        self.assertIn("로컬 일지 쓰기 실패", err)

    def test_로컬_쓰기_실패와_Discord_실패가_겹쳐도_종료코드는_20(self):
        with mock.patch("stock_bot.journal.write", side_effect=journal.JournalError("x")):
            n = []

            def s(url, text):
                n.append(text)
                if len(n) > 1:
                    raise notify.NotifyError("HTTP 500")
            code, _, err = self.run_cmd(send=s)
        self.assertEqual(code, cli.EXIT_GUARD)
        self.assertIn("보고도 실패", err)

    def test_알림_불통이면_Outline도_건드리지_않는다(self):
        def send(url, text):
            raise notify.NotifyError("HTTP 500")
        code, _, _ = self.run_cmd(send=send)
        self.assertEqual(code, cli.EXIT_GUARD)
        self.assertEqual(self.fake.calls, [])
        self.assertIn("알림 채널 불통", self.jtext(), "멈춘 사실은 정본에 남는다")

    def test_Discord_보고가_실패해도_일지와_사본은_이미_남았다(self):
        n = []

        def s(url, text):
            n.append(1)
            if len(n) == 2:
                raise notify.NotifyError("HTTP 404")
        code, _, _ = self.run_cmd(send=s)
        self.assertEqual(code, cli.EXIT_GUARD)
        self.assertIn("종가 $96.27", self.jtext())
        self.assertIn("종가 $96.27", self.fake.text)


class TestRecordFromRealRunOnce(unittest.TestCase):
    """진짜 service.run_once(모델 모드, 임시 DB)가 만든 record로 한 줄이 나온다."""

    def test_record가_채워지고_한_줄이_된다(self):
        from stock_bot import service
        with tempfile.TemporaryDirectory() as d:
            cfg = make_cfg(db_path=os.path.join(d, "stock_bot.db"), kis_app_key="",
                           kis_app_secret="", kis_cano="",
                           account_ledger_path=os.path.join(d, "ledger.db"))
            self.assertFalse(cfg.kis_enabled)
            with mock.patch("stock_bot.market.next_unprocessed_close",
                            return_value=("2026-10-23", 96.27)):
                text, result = service.run_once(cfg)
        r = result.record
        self.assertEqual((r["ticker"], r["quote_date"], r["close"], r["source"]),
                         ("QLD", "2026-10-23", 96.27, "모델"))
        self.assertAlmostEqual(r["band_low"], r["V"] * 0.85)
        self.assertAlmostEqual(r["band_high"], r["V"] * 1.15)
        line = journal.make_line("QLD", 0, text, r, NOW)
        self.assertIn("종가 $96.27", line)
        self.assertIn("출처 모델", line)
        self.assertNotIn("\n", line)


class TestConfigEnv(unittest.TestCase):
    def _cfg(self, env):
        from stock_bot.config import load_config
        with tempfile.TemporaryDirectory() as d:
            s = os.path.join(d, "s.yaml")
            _write(s, "ticker: QLD\nvr_mode: 거치식\nkis_cash_share: 1.0\n")
            e = {"STOCK_DB_PATH": os.path.join(d, "stock_bot.db"), "STOCK_SETTINGS_PATH": s, **env}
            with mock.patch.dict(os.environ, e, clear=False):
                for k in ("STOCK_OUTLINE_CONFIG_PATH", "STOCK_OUTLINE_DOC_ID"):
                    if k not in env:
                        os.environ.pop(k, None)
                return load_config()

    def test_기본값(self):
        c = self._cfg({})
        self.assertEqual(c.outline_config_path, "/mnt/outline-cli/config.json")
        self.assertEqual(c.outline_doc_id, DOC)

    def test_빈_값은_사본_끄기_경로_지정은_그대로(self):
        self.assertEqual(self._cfg({"STOCK_OUTLINE_CONFIG_PATH": ""}).outline_config_path, "")
        self.assertEqual(self._cfg({"STOCK_OUTLINE_CONFIG_PATH": "/x/c.json"}).outline_config_path, "/x/c.json")


if __name__ == "__main__":
    unittest.main()
