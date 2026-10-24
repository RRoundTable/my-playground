"""전략 설정 파일이 없으면 조용히 기본값으로 가지 않고 멈추는지.

이 구멍은 설정 파일을 옮기는 작업의 전제조건이었다 — 경로가 한 글자 틀리면
TSLA 봇이 QLD·적립식·kis_cash_share=1.0으로 둔갑해 돈이 걸린 계산이 틀어진다.

설정 파일은 `STOCK_SETTINGS_PATH` 오버라이드로 임시 디렉터리에 둔다. 유도 경로를
쓰면 저장소의 `config/`에 테스트 찌꺼기를 남기게 된다. 네트워크 없음.
"""
import os
import tempfile
import unittest

from stock_bot.config import CONFIG_DIR, SettingsError, load_config, settings_path

QLD_YAML = """\
ticker: QLD
vr_mode: 거치식
cycle_len: 10
band_rate: 0.15
sell_bands: 0.025,0.05,0.075,0.10
init_v: 1549
init_pool: 1976
init_shares: 0
init_avg_cost: 0
kis_cash_share: 1.0
"""


class SettingsBase(unittest.TestCase):
    VARS = ("STOCK_DB_PATH", "STOCK_SETTINGS_PATH")

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self.VARS}
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "stock_bot.db")
        self.settings = os.path.join(self.tmp.name, "settings.yaml")
        os.environ["STOCK_DB_PATH"] = self.db
        os.environ["STOCK_SETTINGS_PATH"] = self.settings

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def write(self, text):
        with open(self.settings, "w", encoding="utf-8") as f:
            f.write(text)


class TestMissingSettingsFails(SettingsBase):
    def test_파일이_없으면_SettingsError(self):
        with self.assertRaises(SettingsError) as e:
            load_config()
        self.assertIn(self.settings, str(e.exception),
                      "어느 경로를 찾다 실패했는지 알려줘야 고칠 수 있다")

    def test_오류_메시지가_무엇이_위험한지_알려준다(self):
        with self.assertRaises(SettingsError) as e:
            load_config()
        self.assertIn("kis_cash_share", str(e.exception),
                      "왜 위험한지 모르면 사람이 경로 오타를 그냥 넘긴다")

    def test_디렉터리는_있고_파일만_없어도_멈춘다(self):
        os.makedirs(os.path.dirname(self.db), exist_ok=True)
        with self.assertRaises(SettingsError):
            load_config()

    def test_조용히_QLD_기본값으로_떨어지지_않는다(self):
        try:
            cfg = load_config()
        except SettingsError:
            return
        self.fail(f"설정 없이 로딩됐다 — ticker={cfg.ticker} "
                  f"vr_mode={cfg.vr_mode} kis_cash_share={cfg.kis_cash_share}")


class TestSettingsPresentStillWorks(SettingsBase):
    def test_파일이_있으면_정상_로딩된다(self):
        self.write(QLD_YAML)
        cfg = load_config()
        self.assertEqual(cfg.ticker, "QLD")
        self.assertEqual(cfg.vr_mode, "거치식")
        self.assertEqual(cfg.init_v, 1549.0)
        self.assertEqual(cfg.kis_cash_share, 1.0)

    def test_빈_파일은_허용된다_환경변수로_채울_수_있으므로(self):
        self.write("")
        cfg = load_config()
        self.assertEqual(cfg.ticker, "QLD", "빈 파일은 '없음'과 다르다 — 의도된 상태일 수 있다")

    def test_오버라이드가_없으면_유도_경로를_쓴다(self):
        os.environ.pop("STOCK_SETTINGS_PATH")
        self.assertEqual(settings_path("/a/stock-data-xyz/stock_bot.db"),
                         os.path.join(CONFIG_DIR, "stock-data-xyz.yaml"))


if __name__ == "__main__":
    unittest.main()
