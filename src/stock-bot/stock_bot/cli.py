"""stock-bot CLI — nanoclaw가 구동하는 단일 진입점 (의존성 0, stdlib).

  run          오늘치 VR 계산+저장, Discord에 직접 보고, 일지 기록 (매일 1회, cron이 호출)
  notify-test  Discord 웹훅 점검 메시지 한 건 전송
  status       현재 상태 + 오늘 밤 LOC 주문 (읽기 전용)
  history      최근 일별 이력 (읽기 전용)

`run`은 에이전트 없이 혼자 돈다. **Discord에 알릴 수 없으면 시작하지 않는다.**
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_bot.config import load_config  # noqa: E402
from stock_bot.state.store import Store  # noqa: E402


# ── 종료 코드 — cron이 성공/실패를 분기할 수 있어야 한다 ──
# 게이트 차단(DRY_RUN·BACKLOG·한도·킬스위치)은 **설계된 동작**이라 0이다.
# '주문을 냈는데 결과를 모른다'와 '그날 계산을 중단했다'만 0이 아니다.
EXIT_OK = 0
EXIT_SEND_ANOMALY = 10   # UNCONFIRMED / SEND_REJECTED / 상태 UNKNOWN
EXIT_GUARD = 20          # GuardError — 안전장치 발동, 그날 중단
EXIT_SETTINGS = 30       # SettingsError — 설정 없음/종목 불일치/KIS 자격증명 없음
EXIT_ERROR = 1           # 그 외 미처리 예외
# 알림 불통은 EXIT_GUARD(20)로 낸다 — '안전장치 발동, 즉시 사람에게'가 같은 뜻이다.


def _run_and_print(cfg, as_json: bool) -> tuple[int, str, dict | None]:
    """계산·전송을 수행하고 stdout에 찍은 뒤 (종료 코드, 사람이 읽을 보고문, 일지용 record)를 돌려준다."""
    from stock_bot import service
    from stock_bot.domain.guards import GuardError

    try:
        text, result = service.run_once(cfg)
    except GuardError as e:
        line = f"[BLOCKED] {cfg.ticker} — 안전장치 발동: {e}"
        print(json.dumps({"status": "blocked", "needs_attention": True,
                          "reason": str(e), "summary": line},
                         ensure_ascii=False) if as_json else f"🛑 {line}")
        return EXIT_GUARD, f"🛑 {line}", None
    if result is None:
        line = f"[SKIP] {cfg.ticker} — 신규 종가 없음(휴장/중복)"
        print(json.dumps({"status": "skip", "needs_attention": False, "summary": line},
                         ensure_ascii=False) if as_json else line)
        return EXIT_OK, line, None

    anomalies = service.send_anomalies(result)
    notes = service.attention_notes(result)
    if as_json:
        print(json.dumps(
            {
                "status": "attention" if notes else "ok",
                "needs_attention": bool(notes),
                "summary": text.split("\n", 1)[0],
                "attention_notes": notes,
                "anomalies": anomalies,
                "report": text,
                "cycle_started": result.cycle_started,
                "orders": [
                    {"side": o.side.value, "limit_price": o.limit_price,
                     "qty": o.qty, "band": o.band}
                    for o in result.orders
                ],
                "sends": result.sends,
                "order_statuses": [s.__dict__ for s in result.order_statuses],
                "resolved": result.resolved,
            },
            ensure_ascii=False, indent=2))
    else:
        print(text)
    return (EXIT_SEND_ANOMALY if anomalies else EXIT_OK), text, result.record


def cmd_run(cfg, as_json: bool) -> int:
    """사전점검(일지 쓰기·알림) → 실행 → 로컬 일지(정본) → Outline 사본 → 보고.

    순서가 핵심이다. run_once는 그날 종가를 소진하고 주문을 전송한다. 보고 채널이나
    일지가 죽은 채로 그걸 하면 사람은 돈이 나갔는지 모른다. 그래서 **먼저** 둘 다
    두드려 보고, 안 되면 아무것도 하지 않고 멈춘다. Outline 사본만 best-effort다.
    """
    from datetime import datetime, timezone

    from stock_bot import journal, notify, outline
    from stock_bot.config import require_kis, require_notify

    require_kis(cfg)      # SettingsError → main에서 EXIT_SETTINGS
    require_notify(cfg)

    url = cfg.discord_webhook_url
    now = datetime.now(timezone.utc)

    def _blocked(line: str, reason: str) -> int:
        print(json.dumps({"status": "blocked", "needs_attention": True,
                          "reason": reason, "summary": line},
                         ensure_ascii=False) if as_json else f"🛑 {line}")
        return EXIT_GUARD

    def _record(code: int, message: str, record: dict | None) -> str:
        line = journal.make_line(cfg.ticker, code, message, record, now)
        journal.write(cfg.db_path, cfg.ticker, code, message, line, now)
        return line

    try:
        journal.preflight(cfg.db_path)
    except journal.JournalError as e:
        line = f"[BLOCKED] {cfg.ticker} — 일지를 쓸 수 없어 실행하지 않음(종가·주문 손대지 않음): {e}"
        try:
            notify.send(url, f"🛑 {line}")
        except notify.NotifyError as ne:
            print(f"[BLOCKED] 알림도 실패: {ne}", file=sys.stderr)
        return _blocked(line, str(e))

    stamp = now.strftime("%Y-%m-%d %H:%M UTC")
    try:
        notify.send(url, f"▶ {cfg.ticker} run 시작 · {stamp}")
    except notify.NotifyError as e:
        line = f"[BLOCKED] {cfg.ticker} — 알림 채널 불통, 실행하지 않음(종가·주문 손대지 않음): {e}"
        try:
            _record(EXIT_GUARD, f"🛑 {line}", None)
        except journal.JournalError as je:
            print(f"[BLOCKED] 일지도 실패: {je}", file=sys.stderr)
        return _blocked(line, str(e))

    try:
        code, message, record = _run_and_print(cfg, as_json)
    except Exception as e:
        # 죽는 것도 조용하면 안 된다. 알림은 최선을 다하되 예외는 그대로 올려 exit 1을 남긴다.
        crash = f"💥 [ERROR] {cfg.ticker} — run 중 예외 {type(e).__name__}: {e}"
        try:
            _record(EXIT_ERROR, crash, None)
        except journal.JournalError as je:
            print(f"[BLOCKED] 일지도 실패: {je}", file=sys.stderr)
        try:
            notify.send(url, crash)
        except notify.NotifyError as ne:
            print(f"[BLOCKED] 예외 알림도 실패: {ne}", file=sys.stderr)
        raise

    try:
        line = _record(code, message, record)
    except journal.JournalError as e:
        # 계산·전송은 끝났고 정본을 못 남겼다. 실행 결과를 Discord로 알리고 여기서 멈춘다.
        warn = f"🛑 [BLOCKED] {cfg.ticker} — 로컬 일지 쓰기 실패(종료코드 {code}→{EXIT_GUARD}): {e}"
        print(warn, file=sys.stderr)
        try:
            notify.send(url, warn + "\n\n" + message)
        except notify.NotifyError as ne:
            print(f"[BLOCKED] 보고도 실패: {ne}", file=sys.stderr)
        return EXIT_GUARD

    outline_fail = ""
    if cfg.outline_config_path:
        try:
            outline.sync(cfg.outline_config_path, cfg.outline_doc_id,
                         [*journal.recent_lines(cfg.db_path, now), line])
        except outline.OutlineError as e:
            outline_fail = f"⚠️ Outline 갱신 실패: {e}"
            print(f"[WARN] {outline_fail}", file=sys.stderr)

    try:
        notify.send(url, message + (f"\n\n{outline_fail}" if outline_fail else ""))
    except notify.NotifyError as e:
        # 이미 계산·전송이 끝난 뒤라 되돌릴 수 없다. 보고문은 stdout에 남아 있으니
        # 로그로 확인하고, 사람이 알아채도록 종료 코드를 올린다.
        print(f"[BLOCKED] {cfg.ticker} — 실행은 끝났으나 Discord 보고 실패(종료코드 {code}→{EXIT_GUARD}): {e}",
              file=sys.stderr)
        return EXIT_GUARD
    return code


def cmd_notify_test(cfg) -> int:
    from stock_bot import notify
    from stock_bot.config import require_notify

    require_notify(cfg)
    try:
        notify.send(cfg.discord_webhook_url, f"✅ stock-bot 알림 점검 · {cfg.ticker}")
    except notify.NotifyError as e:
        print(f"[BLOCKED] 알림 점검 실패: {e}", file=sys.stderr)
        return EXIT_GUARD
    print("알림 점검 메시지를 보냈다.")
    return EXIT_OK


def cmd_balance(cfg, as_json: bool) -> None:
    """KIS 실계좌 잔액 조회 — USD 예수금 + 보유종목 평가금액 + 총자산."""
    from stock_bot.broker.kis_client import KisError
    from stock_bot.service import make_kis_client

    if not cfg.kis_enabled:
        _emit({"error": "KIS 미설정 (kis.json 필요)"}, as_json)
        return
    try:
        ov = make_kis_client(cfg).account_overview()
    except KisError as e:
        _emit({"error": str(e)}, as_json)
        return
    if as_json:
        _emit(
            {
                "usd_cash": ov.usd_cash,
                "usd_withdrawable": ov.usd_withdrawable,
                "tot_asst_krw": ov.tot_asst_krw,
                "holdings": ov.holdings,
            },
            True,
        )
        return
    cash = f"${ov.usd_cash:,.2f}" if ov.usd_cash is not None else "N/A"
    wd = f"${ov.usd_withdrawable:,.2f}" if ov.usd_withdrawable is not None else "N/A"
    print(f"💰 KIS 계좌 ({cfg.kis_env})")
    print(f"  USD 예수금   : {cash}")
    print(f"  출금가능     : {wd}")
    if ov.tot_asst_krw is not None:
        print(f"  총자산(원화) : {ov.tot_asst_krw:,.0f}원")
    if ov.holdings:
        print("  보유 주식:")
        for h in ov.holdings:
            print(f"    - {h['symbol']}: {h['qty']:.0f}주 · 평단 ${h['avg']:.2f} · 평가 ${h['eval']:,.2f}")
    else:
        print("  보유 주식: 없음")


def cmd_status(store: Store, as_json: bool) -> None:
    loaded = store.load()
    if loaded is None:
        _emit({"error": "상태 없음 — 아직 run을 실행하지 않았습니다"}, as_json)
        return
    state, pending = loaded
    if as_json:
        _emit(
            {
                "state": {
                    "V": round(state.V, 2),
                    "pool": round(state.pool, 2),
                    "shares": state.shares,
                    "avg_cost": round(state.avg_cost, 2),
                    "cycle_index": state.cycle_index,
                    "day_in_cycle": state.day_in_cycle,
                    "last_close": round(state.last_close, 2)
                    if state.last_close is not None
                    else None,
                    "last_quote_date": state.last_quote_date,
                },
                "tonight_orders": [
                    {"side": o.side.value, "limit_price": o.limit_price,
                     "qty": o.qty, "band": o.band}
                    for o in pending
                ],
            },
            True,
        )
        return

    print("📊 stock-bot 상태")
    print(
        f"  V=${state.V:,.2f}  현금풀=${state.pool:,.2f}  "
        f"보유={state.shares:.0f}주  평단=${state.avg_cost:.2f}"
    )
    last_close = f"${state.last_close:.2f}" if state.last_close is not None else "N/A"
    print(
        f"  사이클 #{state.cycle_index} · {state.day_in_cycle}일차 · "
        f"최근종가 {state.last_quote_date} {last_close}"
    )
    if pending:
        print("  오늘 밤 LOC 주문:")
        for o in pending:
            print(f"    • {o.label()}")
    else:
        print("  오늘 밤 주문: 없음(홀딩)")


def cmd_history(store: Store, as_json: bool, limit: int) -> None:
    rows = store.conn.execute(
        "SELECT ts, quote_date, close, snapshot FROM history ORDER BY ts DESC LIMIT ?",
        (limit,),
    ).fetchall()
    items = [
        {"ts": r[0], "quote_date": r[1], "close": r[2], **json.loads(r[3])} for r in rows
    ]
    if as_json:
        _emit(items, True)
        return
    print(f"📜 최근 {len(items)}일")
    for it in items:
        print(
            f"  {it['quote_date']} 종가 ${it['close']:.2f} · "
            f"V=${it.get('V', 0):,.0f} 풀=${it.get('pool', 0):,.0f} "
            f"보유={it.get('shares', 0):.0f}주 · 주문 {len(it.get('orders', []))}건 "
            f"· 출처 {it.get('source', '?')}"
        )


def cmd_backtest(cfg, args) -> None:
    """과거 종가로 VR 전략을 재생 (읽기 전용, db 저장 없음)."""
    from stock_bot import backtest, market

    ticker = args.ticker or cfg.ticker
    try:
        closes = market.history(ticker, args.range)
    except market.QuoteError as e:
        _emit({"error": f"시세 조회 실패: {e}"}, args.json)
        return

    result = backtest.run_backtest(
        ticker,
        closes,
        vr_mode=args.vr_mode or cfg.vr_mode,
        cycle_len=args.cycle_len or cfg.cycle_len,
        band_rate=args.band_rate if args.band_rate is not None else cfg.band_rate,
        sell_bands=(
            tuple(float(x) for x in args.sell_bands.split(","))
            if args.sell_bands
            else cfg.sell_bands
        ),
        cycle_cash_flow=(
            args.cycle_cash_flow if args.cycle_cash_flow is not None else cfg.cycle_cash_flow
        ),
        G=args.g,
        max_pool_usage=args.max_pool_usage,
        init_v=args.init_v,
        init_pool=args.init_pool,
    )
    if args.json:
        _emit(
            {
                "ticker": result.ticker,
                "vr_mode": result.vr_mode,
                "days": len(result.rows),
                "total_invested": round(result.total_invested, 2),
                "final_equity": round(result.final_equity, 2),
                "vr_annualized_return": round(result.vr_annualized_return, 4),
                "vr_max_drawdown": round(result.vr_max_drawdown, 4),
                "buy_hold_equity": round(result.buy_hold_equity, 2),
                "buy_hold_annualized_return": round(result.buy_hold_annualized_return, 4),
                "buy_hold_max_drawdown": round(result.buy_hold_max_drawdown, 4),
                "blocked": result.blocked,
            },
            True,
        )
    else:
        print(backtest.format_report(result))


def _emit(obj: object, as_json: bool) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2) if as_json else obj)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", help="sqlite 경로 (기본: STOCK_DB_PATH 또는 설정 기본값)")
    common.add_argument("--json", action="store_true")
    ap = argparse.ArgumentParser(prog="stock", description="stock-bot CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", parents=[common], help="오늘치 계산+저장, 보고 텍스트 출력")
    sub.add_parser("notify-test", parents=[common], help="Discord 웹훅 점검 메시지 전송")
    sub.add_parser("status", parents=[common], help="현재 상태 + 오늘 밤 LOC 주문")
    sub.add_parser("balance", parents=[common], help="KIS 실계좌 예수금·평가금액 조회")
    h = sub.add_parser("history", parents=[common], help="최근 일별 이력")
    h.add_argument("--limit", type=int, default=10)
    b = sub.add_parser(
        "backtest", parents=[common], help="과거 종가로 VR 전략 시뮬레이션 (읽기 전용, db 저장 없음)"
    )
    b.add_argument("--ticker", help="기본: 설정된 티커")
    b.add_argument("--range", default="5y", help="조회 구간: 1y/2y/5y/10y/max (기본 5y)")
    b.add_argument("--vr-mode", dest="vr_mode", help="적립식/거치식/인출식 (기본: 설정값)")
    b.add_argument("--cycle-len", dest="cycle_len", type=int)
    b.add_argument("--band-rate", dest="band_rate", type=float)
    b.add_argument("--sell-bands", dest="sell_bands", help="쉼표구분, 예: 0.025,0.05,0.075,0.10")
    b.add_argument("--cycle-cash-flow", dest="cycle_cash_flow", type=float)
    b.add_argument("--g", dest="g", type=float, help="기울기 댐퍼 G (기본: vr_mode 프리셋)")
    b.add_argument("--max-pool-usage", dest="max_pool_usage", type=float)
    b.add_argument("--init-v", dest="init_v", type=float, default=0.0)
    b.add_argument("--init-pool", dest="init_pool", type=float, default=1000.0)
    args = ap.parse_args()

    cfg = load_config()
    if args.db:
        cfg.db_path = args.db

    if args.cmd == "run":
        return cmd_run(cfg, args.json)
    if args.cmd == "notify-test":
        return cmd_notify_test(cfg)
    if args.cmd == "balance":
        cmd_balance(cfg, args.json)
        return EXIT_OK
    if args.cmd == "backtest":
        cmd_backtest(cfg, args)
        return EXIT_OK

    if not os.path.exists(cfg.db_path):
        print(json.dumps({"error": f"db 없음: {cfg.db_path}"}, ensure_ascii=False))
        sys.exit(1)
    store = Store(cfg.db_path, cfg.ticker)
    try:
        if args.cmd == "status":
            cmd_status(store, args.json)
        elif args.cmd == "history":
            cmd_history(store, args.json, args.limit)
    finally:
        store.close()
    return EXIT_OK


def cli() -> int:
    """종료 코드를 확정해 돌려준다. SettingsError는 배포·환경 사고라 따로 가른다."""
    from stock_bot.config import SettingsError

    try:
        return main() or EXIT_OK
    except SettingsError as e:
        print(f"[SETTINGS] {e}", file=sys.stderr)
        return EXIT_SETTINGS


if __name__ == "__main__":
    sys.exit(cli())
