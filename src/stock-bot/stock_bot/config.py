"""설정 — 환경변수 + (선택) kis.json 파일 로딩 + 라오어 VR 프리셋.

CLI-driven: nanoclaw가 `stock run`으로 구동. 봇 자체 스케줄러·웹훅 없음.
KIS creds는 env 우선, 없으면 db 디렉터리의 kis.json에서 로딩
(nanoclaw 컨테이너는 env 시크릿을 못 보므로 파일 경로 지원).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from stock_bot.domain.models import StrategyParams
from stock_bot.outline import DEFAULT_CONFIG_PATH as DEFAULT_OUTLINE_CONFIG_PATH
from stock_bot.outline import DEFAULT_DOC_ID as DEFAULT_OUTLINE_DOC_ID

# 검증된 의사코드의 전략별 매개변수.
STRATEGY_PRESETS: dict[str, dict] = {
    "적립식": dict(G=10.0, cycle_cash_flow=250.0, max_pool_usage=0.75),
    "거치식": dict(G=10.0, cycle_cash_flow=0.0, max_pool_usage=0.50),
    "인출식": dict(G=20.0, cycle_cash_flow=-250.0, max_pool_usage=0.25),
}

_DEFAULT_DB = "/workspace/extra/my-playground/data/stock-bot/stock_bot.db"


@dataclass
class Config:
    ticker: str
    vr_mode: str
    strategy_engine: str
    db_path: str
    cycle_len: int
    band_rate: float
    sell_bands: tuple[float, ...]
    init_v: float
    init_pool: float
    init_shares: float
    init_avg_cost: float
    cycle_cash_flow: float | None  # None이면 vr_mode 프리셋값 사용
    G: float | None  # None이면 vr_mode 프리셋값 사용 (기울기 댐퍼)
    max_pool_usage: float | None  # None이면 vr_mode 프리셋값 사용
    # 한 KIS 계좌를 여러 종목 봇이 공유할 때 실계좌 현금 중 이 봇 몫의 비율.
    # 기본 1.0(단일 종목 운용, 기존 동작 그대로). 두 봇 합이 1.0을 넘으면 안 됨(이중계산 방지).
    kis_cash_share: float
    # ── 주문 전송 안전장치 ──
    # dry_run이 False여야 전송 '후보'가 된다. False라고 바로 나가는 게 아니라
    # order_gate의 게이트를 전부 통과해야 한다. 파싱 실패·미설정은 항상 True(안전측).
    dry_run: bool
    kill_switch_path: str
    max_daily_orders: int
    max_daily_amount: float
    max_order_amount: float
    # 계좌 단위 한도 — 두 봇이 같은 원장 파일을 공유한다. 봇별 한도만으로는
    # 계좌 기준 2배까지 나간다(QLD·TSLA가 각자 자기 한도를 세므로).
    account_ledger_path: str
    max_account_daily_orders: int
    max_account_daily_amount: float
    # KIS 읽기전용 (creds 다 있으면 실제 계좌가 진실의 원천)
    kis_env: str
    kis_app_key: str
    kis_app_secret: str
    kis_cano: str
    kis_acnt_prdt_cd: str
    kis_crcy_cd: str
    # 알림 — 웹훅 URL은 시크릿이다. 환경변수로만 주입하고 어디에도 출력하지 않는다.
    discord_webhook_url: str = ""
    # Outline 사본 — 접속 정보는 읽기 전용 마운트의 config.json. 빈 경로면 사본을 끈다.
    outline_config_path: str = DEFAULT_OUTLINE_CONFIG_PATH
    outline_doc_id: str = DEFAULT_OUTLINE_DOC_ID

    @property
    def kis_enabled(self) -> bool:
        return bool(self.kis_app_key and self.kis_app_secret and self.kis_cano)

    def missing_kis_creds(self) -> list[str]:
        """비어 있는 KIS 자격증명의 **이름만** 돌려준다(값은 절대 노출하지 않는다).

        기본값이 있는 것(ACNT_PRDT_CD, TR_CRCY_CD)과 더 이상 읽지 않는 것
        (OVRS_EXCG_CD — 잔고를 AMEX·NASD 병합 조회로 바꾸며 제거)은 뺀다.
        """
        required = {
            "KIS_APP_KEY": self.kis_app_key,
            "KIS_APP_SECRET": self.kis_app_secret,
            "KIS_CANO": self.kis_cano,
        }
        return [name for name, value in required.items() if not value]


def require_kis(cfg: "Config") -> None:
    """KIS 자격증명이 없으면 멈춘다. **조용히 가정 모델로 떨어지지 않게 하는 것이 목적이다.**

    cron 환경에는 에이전트 프로세스의 환경변수가 없다. 그대로 두면 `kis_enabled`가
    False가 되어 실계좌 대신 가정 모델로 계산하고, 보고는 평소처럼 나간다 —
    '설정 파일이 없으면 남의 종목으로 둔갑'하던 것과 같은 종류의 조용한 오작동이다.

    백테스트·개발용으로 일부러 KIS 없이 돌려야 하면 STOCK_ALLOW_NO_KIS=1을 준다.
    """
    if cfg.kis_enabled:
        return
    if str(os.environ.get("STOCK_ALLOW_NO_KIS", "")).strip() not in ("", "0", "false", "no", "off"):
        return  # 명시적 opt-out
    raise SettingsError(
        "KIS 자격증명이 없다: " + ", ".join(cfg.missing_kis_creds()) + ". "
        "이 상태로 돌면 실계좌 대신 가정 모델로 계산하고 보고는 평소처럼 나간다. "
        "cron이라면 환경변수 주입 경로를 확인하라. "
        "의도한 것이면 STOCK_ALLOW_NO_KIS=1 로 명시하라."
    )


def require_notify(cfg: "Config") -> None:
    """Discord 웹훅이 없거나 형식이 틀리면 멈춘다. **알림 없이 조용히 도는 봇은 없다.**

    KIS와 달리 opt-out이 없다. 보고가 안 나가는 봇이 돈 나가는 경로를 도는 것을 막는
    것이 목적이다. 값은 메시지에 넣지 않는다.
    """
    from stock_bot import notify

    if not cfg.discord_webhook_url:
        raise SettingsError(
            "Discord 알림이 설정되지 않았다: STOCK_DISCORD_WEBHOOK_URL. "
            "알림 없이는 run을 하지 않는다 — 컨테이너 환경변수 주입 경로를 확인하라."
        )
    if not notify.valid_webhook_url(cfg.discord_webhook_url):
        raise SettingsError(
            "STOCK_DISCORD_WEBHOOK_URL 형식이 Discord 웹훅(https://discord.com/api/webhooks/...)이 아니다."
        )


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name, default)
    return v if v is not None else default


def _load_kis_file(db_path: str) -> dict:
    path = os.path.join(os.path.dirname(db_path) or ".", "kis.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _parse_flat_yaml(text: str) -> dict:
    """평면 `key: value` YAML만 파싱 (stdlib, 의존성 0). 중첩/리스트 미지원."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        key, val = line.split(":", 1)
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


class SettingsError(RuntimeError):
    pass


# 전략 설정은 **저장소 안**에 둔다 — 승인이 필요한 파라미터가 버전 관리 밖에 있으면
# '언제 무엇이 어떤 값이었나'를 되짚을 수 없다. 경로는 db 디렉터리 이름에서 유도하므로
# 실행 명령(STOCK_DB_PATH만 지정)은 그대로다.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(os.path.dirname(_PKG_DIR), "config")


def settings_path(db_path: str) -> str:
    """이 db에 대응하는 전략 설정 파일 경로.

    기본 규칙: `{저장소}/config/{db 디렉터리 이름}.yaml`
      /workspace/stock-data/stock_bot.db      → config/stock-data.yaml
      /workspace/stock-data-tsla/stock_bot.db → config/stock-data-tsla.yaml
    종목이 몇 개든 db 디렉터리 하나당 설정 파일 하나로 1:1 대응한다.

    `STOCK_SETTINGS_PATH`가 있으면 그 값을 그대로 쓴다(선택적 오버라이드).
    규칙을 벗어나야 할 때 코드를 고치지 않아도 되지만, 잘못 쓰면 다른 종목의 설정을
    집을 수 있다 — 그건 Store의 종목 일치 검사가 잡는다.
    """
    override = os.environ.get("STOCK_SETTINGS_PATH")
    if override:
        return override
    data_dir = os.path.basename(os.path.dirname(os.path.abspath(db_path)))
    return os.path.join(CONFIG_DIR, f"{data_dir}.yaml")


def _load_settings_file(db_path: str) -> dict:
    """전략 설정을 읽는다. **없으면 조용히 기본값으로 가지 않고 즉시 멈춘다.**

    예전엔 읽기 실패 시 {}를 돌려줬는데, 그러면 코드 기본값(ticker=QLD /
    vr_mode=적립식 / kis_cash_share=1.0 / init_v=9000)으로 조용히 돌아간다.
    TSLA 봇이 QLD로 둔갑하고, kis_cash_share 합이 1.0을 넘어 계좌 현금이
    이중계산돼 V가 부풀려진다 — 아무 경고 없이. 파일 경로가 틀린 것을
    '설정이 없다'로 받아들이면 안 된다.
    """
    path = settings_path(db_path)
    try:
        with open(path, encoding="utf-8") as f:
            return _parse_flat_yaml(f.read())
    except OSError as e:
        raise SettingsError(
            f"전략 설정 파일을 읽을 수 없다: {path} ({e.strerror}). "
            "이 파일 없이 돌면 ticker=QLD·vr_mode=적립식·kis_cash_share=1.0 같은 "
            "코드 기본값으로 조용히 동작한다. STOCK_DB_PATH가 맞는지 확인하라."
        ) from e


def parse_dry_run(raw: str | None) -> bool:
    """DRY_RUN 해석 — **명시적으로 거짓인 값일 때만 False**, 나머지는 전부 True.

    미설정·빈값·오타·해석 불가는 모두 True로 떨어진다. 실수로 전송이 켜지는 것보다
    실수로 전송이 막히는 편이 낫다.
    """
    if raw is None:
        return True
    return str(raw).strip().lower() not in ("false", "0", "no", "off")


def _pos_int(raw: str, default: int) -> int:
    """한도값 파싱 — 해석 불가·음수는 기본값(보수적)으로 되돌린다."""
    try:
        v = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return v if v >= 0 else default


def _pos_float(raw: str, default: float) -> float:
    try:
        v = float(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return v if v >= 0 else default


def _pick(env_key: str, file_dict: dict, file_key: str, default: str) -> str:
    """우선순위: 환경변수 > 설정파일 > 기본값."""
    v = os.environ.get(env_key) or file_dict.get(file_key)
    return v if v not in (None, "") else default


def load_config() -> Config:
    db_path = _env("STOCK_DB_PATH", _DEFAULT_DB)
    kf = _load_kis_file(db_path)  # KIS 시크릿 (env가 비면 파일값)
    st = _load_settings_file(db_path)  # 전략 설정 (env > stock.yaml > 기본값)
    return Config(
        ticker=_pick("STOCK_TICKER", st, "ticker", "QLD"),
        vr_mode=_pick("STOCK_VR_MODE", st, "vr_mode", "적립식"),
        strategy_engine=_pick("STOCK_STRATEGY_ENGINE", st, "strategy_engine", "value_rebalancing"),
        db_path=db_path,
        cycle_len=int(_pick("STOCK_CYCLE_LEN", st, "cycle_len", "10")),
        band_rate=float(_pick("STOCK_BAND_RATE", st, "band_rate", "0.15")),
        sell_bands=tuple(
            float(x)
            for x in _pick("STOCK_SELL_BANDS", st, "sell_bands", "0.025,0.05,0.075,0.10").split(",")
        ),
        init_v=float(_pick("STOCK_INIT_V", st, "init_v", "9000")),
        init_pool=float(_pick("STOCK_INIT_POOL", st, "init_pool", "1000")),
        init_shares=float(_pick("STOCK_INIT_SHARES", st, "init_shares", "0")),
        init_avg_cost=float(_pick("STOCK_INIT_AVG_COST", st, "init_avg_cost", "0")),
        cycle_cash_flow=(
            float(_ccf) if (_ccf := _pick("STOCK_CYCLE_CASH_FLOW", st, "cycle_cash_flow", "")) else None
        ),
        G=(float(_g) if (_g := _pick("STOCK_G", st, "g", "")) else None),
        max_pool_usage=(
            float(_mpu) if (_mpu := _pick("STOCK_MAX_POOL_USAGE", st, "max_pool_usage", "")) else None
        ),
        kis_cash_share=float(_pick("STOCK_KIS_CASH_SHARE", st, "kis_cash_share", "1.0")),
        dry_run=parse_dry_run(os.environ.get("STOCK_DRY_RUN") or st.get("dry_run")),
        kill_switch_path=_pick(
            "STOCK_KILL_SWITCH_PATH", st, "kill_switch_path",
            os.path.join(os.path.dirname(db_path) or ".", "STOP"),
        ),
        # 기본값은 일부러 낮다 — 2단계 섀도 운영(1주)을 넘기려면 사람이 의식적으로 올려야 한다.
        max_daily_orders=_pos_int(_pick("STOCK_MAX_DAILY_ORDERS", st, "max_daily_orders", "2"), 2),
        max_daily_amount=_pos_float(
            _pick("STOCK_MAX_DAILY_AMOUNT", st, "max_daily_amount", "500"), 500.0),
        max_order_amount=_pos_float(
            _pick("STOCK_MAX_ORDER_AMOUNT", st, "max_order_amount", "500"), 500.0),
        # 기본 경로는 봇 데이터 디렉터리의 **부모**다. /workspace/stock-data 와
        # /workspace/stock-data-tsla 는 부모가 같으므로 두 봇이 자동으로 같은 파일을 본다.
        account_ledger_path=_pick(
            "STOCK_ACCOUNT_LEDGER_PATH", st, "account_ledger_path",
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(db_path))),
                "account_ledger.db",
            ),
        ),
        max_account_daily_orders=_pos_int(
            _pick("STOCK_MAX_ACCOUNT_DAILY_ORDERS", st, "max_account_daily_orders", "2"), 2),
        max_account_daily_amount=_pos_float(
            _pick("STOCK_MAX_ACCOUNT_DAILY_AMOUNT", st, "max_account_daily_amount", "500"),
            500.0),
        kis_env=_env("KIS_ENV") or kf.get("env", "vps"),
        kis_app_key=_env("KIS_APP_KEY") or kf.get("app_key", ""),
        kis_app_secret=_env("KIS_APP_SECRET") or kf.get("app_secret", ""),
        kis_cano=_env("KIS_CANO") or kf.get("cano", ""),
        kis_acnt_prdt_cd=_env("KIS_ACNT_PRDT_CD") or kf.get("acnt_prdt_cd", "01"),
        # 거래소 코드 설정은 없앴다 — 잔고는 AMEX·NASD를 모두 조회해 병합한다.
        # (kis_client._US_EXCG_CODES). KIS_OVRS_EXCG_CD 환경변수는 더 이상 읽지 않는다.
        kis_crcy_cd=_env("KIS_TR_CRCY_CD") or kf.get("crcy_cd", "USD"),
        discord_webhook_url=_env("STOCK_DISCORD_WEBHOOK_URL").strip(),
        outline_config_path=(os.environ["STOCK_OUTLINE_CONFIG_PATH"].strip()
                             if "STOCK_OUTLINE_CONFIG_PATH" in os.environ
                             else DEFAULT_OUTLINE_CONFIG_PATH),
        outline_doc_id=_env("STOCK_OUTLINE_DOC_ID").strip()
        or DEFAULT_OUTLINE_DOC_ID,
    )


def params_for(
    vr_mode: str,
    band_rate: float,
    sell_bands: tuple[float, ...],
    cycle_cash_flow: float | None = None,
    G: float | None = None,
    max_pool_usage: float | None = None,
) -> StrategyParams:
    base = STRATEGY_PRESETS.get(vr_mode)
    if base is None:
        raise RuntimeError(
            f"알 수 없는 VR 모드 '{vr_mode}' (가능: {list(STRATEGY_PRESETS)})"
        )
    preset = dict(base)
    if cycle_cash_flow is not None:  # stock.yaml override
        preset["cycle_cash_flow"] = cycle_cash_flow
    if G is not None:
        preset["G"] = G
    if max_pool_usage is not None:
        preset["max_pool_usage"] = max_pool_usage
    return StrategyParams(
        strategy_type=vr_mode,
        band_rate=band_rate,
        sell_bands=tuple(sell_bands),
        **preset,
    )
