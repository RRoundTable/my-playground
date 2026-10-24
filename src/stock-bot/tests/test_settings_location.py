"""설정 경로 유도 + `STOCK_SETTINGS_PATH` 오버라이드 + 종목 일치 검사.

전략 파라미터는 승인 대상이라 저장소 안(버전 관리)에 있어야 하고, 경로는 실행 명령을
바꾸지 않도록 db 디렉터리 이름에서 유도한다. 유도든 오버라이드든 **다른 종목의 설정을
집는 순간 멈춰야** 한다 — 그게 이 파일의 핵심이다. 네트워크 없음.
"""
import os
import tempfile
import unittest

from stock_bot.config import CONFIG_DIR, SettingsError, load_config, settings_path
from stock_bot.state.store import Store


def yaml_for(ticker, **over):
    base = dict(vr_mode="거치식", cycle_len=10, band_rate=0.15,
                sell_bands="0.025,0.05,0.075,0.10", init_v=1000,
                init_pool=1000, init_shares=0, init_avg_cost=0, kis_cash_share=1.0)
    base.update(over)
    return f"ticker: {ticker}\n" + "\n".join(f"{k}: {v}" for k, v in base.items()) + "\n"


class EnvBase(unittest.TestCase):
    """환경변수를 건드리므로 반드시 원복한다."""

    VARS = ("STOCK_DB_PATH", "STOCK_SETTINGS_PATH")

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self.VARS}
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()


class TestPathDerivation(unittest.TestCase):
    def test_db_디렉터리_이름에서_유도한다(self):
        self.assertEqual(
            settings_path("/workspace/stock-data/stock_bot.db"),
            os.path.join(CONFIG_DIR, "stock-data.yaml"))
        self.assertEqual(
            settings_path("/workspace/stock-data-tsla/stock_bot.db"),
            os.path.join(CONFIG_DIR, "stock-data-tsla.yaml"))

    def test_설정은_저장소_안에_있다(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(
            __import__("stock_bot").__file__)))
        self.assertTrue(
            CONFIG_DIR.startswith(repo),
            "설정이 저장소 밖이면 버전 관리가 안 된다 — 이 작업의 목적 자체다")

    def test_종목_수에_무관하다(self):
        """종목이 셋이든 열이든 db 디렉터리 하나당 설정 파일 하나로 1:1 대응한다."""
        dirs = ["stock-data", "stock-data-tsla", "stock-data-soxl",
                "stock-data-schd", "stock-data-vti"]
        paths = [settings_path(f"/workspace/{d}/stock_bot.db") for d in dirs]
        self.assertEqual(len(set(paths)), len(dirs), "경로가 겹치면 설정이 섞인다")
        for d, p in zip(dirs, paths):
            self.assertEqual(os.path.basename(p), f"{d}.yaml")

    def test_상대경로도_절대경로로_풀어_쓴다(self):
        self.assertTrue(os.path.isabs(settings_path("x/stock_bot.db")))


class TestOverride(EnvBase):
    def test_STOCK_SETTINGS_PATH가_유도보다_우선한다(self):
        os.environ["STOCK_SETTINGS_PATH"] = "/tmp/elsewhere.yaml"
        self.assertEqual(settings_path("/workspace/stock-data/stock_bot.db"),
                         "/tmp/elsewhere.yaml")

    def test_빈값이면_유도로_떨어진다(self):
        os.environ["STOCK_SETTINGS_PATH"] = ""
        self.assertEqual(settings_path("/workspace/stock-data/stock_bot.db"),
                         os.path.join(CONFIG_DIR, "stock-data.yaml"))

    def test_오버라이드로_실제_설정을_읽어온다(self):
        p = os.path.join(self.tmp.name, "custom.yaml")
        with open(p, "w", encoding="utf-8") as f:
            f.write(yaml_for("SOXL", band_rate=0.25))
        os.environ["STOCK_SETTINGS_PATH"] = p
        os.environ["STOCK_DB_PATH"] = os.path.join(self.tmp.name, "stock_bot.db")
        cfg = load_config()
        self.assertEqual(cfg.ticker, "SOXL")
        self.assertEqual(cfg.band_rate, 0.25)

    def test_오버라이드_경로가_없으면_멈춘다(self):
        os.environ["STOCK_SETTINGS_PATH"] = "/tmp/__없는파일__.yaml"
        os.environ["STOCK_DB_PATH"] = os.path.join(self.tmp.name, "stock_bot.db")
        with self.assertRaises(SettingsError):
            load_config()


class TestTickerBinding(EnvBase):
    def db(self, name="stock_bot.db"):
        return os.path.join(self.tmp.name, name)

    def test_처음_열면_종목이_기록된다(self):
        s = Store(self.db(), "QLD")
        self.assertEqual(s.recorded_ticker(), "QLD")
        s.close()

    def test_같은_종목으로_다시_열면_통과한다(self):
        Store(self.db(), "QLD").close()
        s = Store(self.db(), "QLD")
        self.assertEqual(s.recorded_ticker(), "QLD")
        s.close()

    def test_다른_종목으로_열면_멈춘다(self):
        Store(self.db(), "TSLA").close()
        with self.assertRaises(SettingsError) as e:
            Store(self.db(), "QLD")
        msg = str(e.exception)
        self.assertIn("QLD", msg)
        self.assertIn("TSLA", msg)

    def test_대소문자는_같은_종목으로_본다(self):
        Store(self.db(), "QLD").close()
        Store(self.db(), "qld").close()   # 예외가 안 나야 한다

    def test_빈_종목은_거부한다(self):
        with self.assertRaises(SettingsError):
            Store(self.db(), "   ")

    def test_종목을_안_주면_검사하지_않는다(self):
        """백테스트 등 종목 귀속이 필요 없는 용도를 막지 않는다."""
        s = Store(self.db())
        self.assertIsNone(s.recorded_ticker())
        s.close()

    def test_세_번째_종목도_각자의_DB에_귀속된다(self):
        for t, f in (("QLD", "a.db"), ("TSLA", "b.db"), ("SOXL", "c.db")):
            s = Store(self.db(f), t)
            self.assertEqual(s.recorded_ticker(), t)
            s.close()
        # 서로 섞으면 전부 멈춰야 한다
        for t, f in (("QLD", "b.db"), ("TSLA", "c.db"), ("SOXL", "a.db")):
            with self.assertRaises(SettingsError, msg=f"{t}가 {f}를 열었는데 안 멈췄다"):
                Store(self.db(f), t)


class TestThirdTickerEndToEnd(EnvBase):
    """가상의 3번째 종목 — 경로 유도부터 종목 귀속까지 한 줄로 통과하는가."""

    def test_새_종목을_추가하는_전체_흐름(self):
        data_dir = os.path.join(self.tmp.name, "stock-data-soxl")
        os.makedirs(data_dir)
        db = os.path.join(data_dir, "stock_bot.db")

        # 설정 경로는 db 디렉터리 이름에서 유도된다
        derived = settings_path(db)
        self.assertEqual(os.path.basename(derived), "stock-data-soxl.yaml")

        # 아직 파일이 없으니 멈춘다 — 조용히 QLD 기본값으로 가지 않는다
        os.environ["STOCK_DB_PATH"] = db
        with self.assertRaises(SettingsError):
            load_config()

        # 설정을 놓아주면(여기선 오버라이드로 대체) 정상 로딩되고 DB에 귀속된다
        p = os.path.join(self.tmp.name, "soxl.yaml")
        with open(p, "w", encoding="utf-8") as f:
            f.write(yaml_for("SOXL"))
        os.environ["STOCK_SETTINGS_PATH"] = p
        cfg = load_config()
        self.assertEqual(cfg.ticker, "SOXL")

        s = Store(cfg.db_path, cfg.ticker)
        self.assertEqual(s.recorded_ticker(), "SOXL")
        s.close()

        # 그 DB를 QLD 설정으로 열려 하면 멈춘다
        with self.assertRaises(SettingsError):
            Store(db, "QLD")


if __name__ == "__main__":
    unittest.main()
