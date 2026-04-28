"""レース発走時刻 +15分 に結果を自動取得し、当日バイアス予想を更新する。

使い方:
  python scripts/oi/watch_results.py --date 2026-04-28
  python scripts/oi/watch_results.py --date 2026-04-28 --offset 15  # 発走後N分で取得
  python scripts/oi/watch_results.py --date 2026-04-28 --dry-run    # 時刻のみ表示

Ctrl+C で停止。全レース確定後に自動終了。
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.oi.scraping.race import fetch_race_result


def _load_schedule(date_str: str) -> list[dict]:
    """shutuba から当日レース一覧と発走時刻を返す。"""
    yyyymmdd = date_str.replace("-", "")
    shutuba_dir = ROOT / "data/oi/raw/shutuba"
    schedule = []
    for fp in sorted(shutuba_dir.glob(f"*44{yyyymmdd[4:8]}*.json")):
        d = json.loads(fp.read_text())
        if d.get("date") != yyyymmdd:
            continue
        pt = d.get("post_time")
        if not pt:
            continue
        h, m = map(int, pt.split(":"))
        post_dt = datetime(
            int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10]), h, m
        )
        schedule.append({
            "race_id":  d["race_id"],
            "race_no":  d["race_no"],
            "race_name": d.get("race_name", ""),
            "post_time": post_dt,
        })
    return sorted(schedule, key=lambda x: x["race_no"])


def _is_confirmed(race_id: str, yyyymmdd: str) -> bool:
    cache = ROOT / "data/oi/raw/results" / f"{race_id}.json"
    if not cache.exists():
        return False
    d = json.loads(cache.read_text())
    return d.get("num_runners", 0) > 0 and d.get("date") == yyyymmdd


def _fetch_result(race_id: str) -> bool:
    """結果を取得してキャッシュに保存。成功(確定) なら True。"""
    cache = ROOT / "data/oi/raw/results" / f"{race_id}.json"
    url = f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"
    http_cache = ROOT / "data/oi/cache" / f"{hashlib.md5(url.encode()).hexdigest()}.html"
    if http_cache.exists():
        http_cache.unlink()
    try:
        d = fetch_race_result(race_id)
        if d.get("num_runners", 0) == 0:
            return False
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(d, ensure_ascii=False, indent=2))
        return True
    except Exception as e:
        print(f"  [warn] {race_id}: {e}", file=sys.stderr)
        return False


def _run_update(date_str: str, weight: float) -> None:
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/oi/update_today_bias.py"),
         "--date", date_str, "--weight", str(weight)],
        capture_output=False,
    )
    if result.returncode != 0:
        print("  [warn] update_today_bias.py が失敗しました", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",    required=True, help="YYYY-MM-DD")
    ap.add_argument("--offset",  type=int, default=15, help="発走後何分で取得するか (default: 15)")
    ap.add_argument("--weight",  type=float, default=0.3)
    ap.add_argument("--dry-run", action="store_true", help="スケジュールを表示するだけ")
    args = ap.parse_args()

    yyyymmdd = args.date.replace("-", "")
    schedule = _load_schedule(args.date)
    if not schedule:
        print(f"出馬表が見つかりません: {args.date}", file=sys.stderr)
        sys.exit(1)

    # _no_bias.json が必須
    if not (ROOT / "data/oi/predictions" / f"{args.date}_no_bias.json").exists():
        print(f"_no_bias.json がありません。先に --save-all で生成してください。", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  大井 {args.date} 結果監視  (発走+{args.offset}分で取得)")
    print(f"{'='*60}")
    for r in schedule:
        fetch_at = r["post_time"] + timedelta(minutes=args.offset)
        confirmed = _is_confirmed(r["race_id"], yyyymmdd)
        mark = "✅確定済" if confirmed else f"  {fetch_at.strftime('%H:%M')}取得予定"
        print(f"  {r['race_no']:>2}R {r['race_name'][:16]:<16} 発走{r['post_time'].strftime('%H:%M')}  {mark}")

    if args.dry_run:
        return

    # 既に確定済みのレース数を初期化
    done: set[str] = {r["race_id"] for r in schedule if _is_confirmed(r["race_id"], yyyymmdd)}
    pending = [r for r in schedule if r["race_id"] not in done]

    if not pending:
        print("\n全レース確定済みです。")
        return

    print(f"\n監視開始 (残 {len(pending)}R) — Ctrl+C で停止\n")

    while pending:
        now = datetime.now()
        newly_fetched = []

        for r in pending:
            fetch_at = r["post_time"] + timedelta(minutes=args.offset)
            if now < fetch_at:
                continue

            print(f"  [{now.strftime('%H:%M:%S')}] {r['race_no']}R {r['race_name']} 結果取得中...", end=" ", flush=True)
            ok = _fetch_result(r["race_id"])
            if ok:
                print("✅ 確定")
                newly_fetched.append(r["race_id"])
            else:
                # まだ未確定 → 3分後に再試行するため pending に残す
                next_try = fetch_at + timedelta(minutes=3)
                r["post_time"] = next_try - timedelta(minutes=args.offset)
                print(f"⏳ 未確定 → {next_try.strftime('%H:%M')}に再試行")

        if newly_fetched:
            print(f"  当日バイアス予想を更新中...")
            _run_update(args.date, args.weight)
            done |= set(newly_fetched)
            pending = [r for r in pending if r["race_id"] not in done]
            print(f"  残 {len(pending)}R\n")

        if not pending:
            break

        # 次の取得予定時刻まで待機（最大60秒ごとにチェック）
        next_times = [
            r["post_time"] + timedelta(minutes=args.offset)
            for r in pending
        ]
        nearest = min(next_times)
        sleep_sec = max(30, min(60, (nearest - datetime.now()).total_seconds()))
        time.sleep(sleep_sec)

    print(f"\n全レース確定。監視終了。")


if __name__ == "__main__":
    main()
