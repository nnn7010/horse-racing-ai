"""大井競馬 当日出馬表取得 → 即予測 ワンショットスクリプト。

historicalデータ(results/)がある状態で、当日の出馬表だけ取得して予測まで実行する。

Usage:
  python scripts/oi/fetch_and_predict.py --date 2026-04-30
  python scripts/oi/fetch_and_predict.py --date 2026-04-30 --save-all

前提: data/oi/raw/results/ に過去結果JSON、data/oi/raw/horses/ に馬個体JSONが存在すること。
なければ先に 02_scrape_results.py → 03_fetch_horses.py を実行すること。
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.oi import load_config
from src.oi.scraping.calendar import fetch_race_ids_for_date
from src.oi.scraping.shutuba import fetch_shutuba
from src.oi.scraping.horse import fetch_horse_info
from src.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_today_shutuba(target_date, cfg: dict, root: Path) -> list[Path]:
    """当日の出馬表を取得して shutuba/ に保存。保存済みはスキップ。"""
    raw_dir = root / cfg["paths"]["raw"]
    shutuba_dir = raw_dir / "shutuba"
    shutuba_dir.mkdir(parents=True, exist_ok=True)

    yyyymmdd = target_date.strftime("%Y%m%d")

    # 既存ファイル確認
    existing = {p.stem for p in shutuba_dir.glob(f"*44{yyyymmdd[4:8]}*.json")
                if json.loads(p.read_text()).get("date") == yyyymmdd}

    race_ids = fetch_race_ids_for_date(target_date)
    if not race_ids:
        logger.error(f"{target_date} の大井レースが見つかりません")
        sys.exit(1)

    print(f"  レース数: {len(race_ids)} (取得済: {len(existing)})")
    saved: list[Path] = []
    for rid in race_ids:
        out_path = shutuba_dir / f"{rid}.json"
        if out_path.exists() and rid in existing:
            logger.info(f"スキップ（キャッシュ済）: {rid}")
            saved.append(out_path)
            continue
        try:
            data = fetch_shutuba(rid)
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            logger.info(f"保存: {out_path}")
            saved.append(out_path)
        except Exception as e:
            logger.warning(f"shutuba {rid} 取得失敗: {e}")

    return saved


def fetch_missing_horses(shutuba_files: list[Path], cfg: dict, root: Path) -> None:
    """出馬表に含まれる馬のうち、未取得のものを取得。"""
    raw_dir = root / cfg["paths"]["raw"]
    horses_dir = raw_dir / "horses"
    horses_dir.mkdir(parents=True, exist_ok=True)

    horse_ids: set[str] = set()
    for fp in shutuba_files:
        d = json.loads(fp.read_text())
        for e in d.get("entries", []):
            if e.get("horse_id"):
                horse_ids.add(e["horse_id"])

    missing = [hid for hid in horse_ids if not (horses_dir / f"{hid}.json").exists()]
    print(f"  馬個体: 全{len(horse_ids)}頭 / 未取得{len(missing)}頭")

    for hid in missing:
        try:
            info = fetch_horse_info(hid)
            (horses_dir / f"{hid}.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2)
            )
            logger.info(f"馬保存: {hid}")
        except Exception as e:
            logger.warning(f"馬 {hid} 取得失敗: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--save-all", action="store_true",
                    help="no_bias / prev_bias / today_bias の3バリアントを保存")
    ap.add_argument("--today-weight", type=float, default=0.0,
                    help="当日バイアス重み (0=無効, 0.3推奨)")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="出馬表・馬取得をスキップ（既存データのみで予測）")
    args = ap.parse_args()

    cfg = load_config()

    print(f"\n{'='*60}")
    print(f"  大井 {args.date} 当日予測パイプライン")
    print(f"{'='*60}")

    if not args.skip_fetch:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

        print("\n[1/3] 出馬表取得...")
        saved = fetch_today_shutuba(target_date, cfg, ROOT)
        if not saved:
            print("ERROR: 出馬表が取得できませんでした", file=sys.stderr)
            sys.exit(1)

        print(f"  → {len(saved)}レース取得完了")

        print("\n[2/3] 馬個体データ確認・補完...")
        fetch_missing_horses(saved, cfg, ROOT)
    else:
        print("\n[1-2/3] スキップ（--skip-fetch）")

    print("\n[3/3] 予測実行...")
    # predict_today_quick.py を直接呼ぶ
    from scripts.oi.predict_today_quick import main as predict_main
    import sys as _sys
    _sys.argv = ["predict_today_quick.py", "--date", args.date]
    if args.save_all:
        _sys.argv.append("--save-all")
    if args.today_weight > 0:
        _sys.argv += ["--today-weight", str(args.today_weight)]

    predict_main()


if __name__ == "__main__":
    main()
