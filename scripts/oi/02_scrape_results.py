"""大井レース結果の一括取得。

scripts/oi/01 で生成した race_ids.csv を読み込み、各レースの結果ページを
順番にスクレイプして data/oi/raw/results/{race_id}.json として保存する。

Usage:
  python scripts/oi/02_scrape_results.py [--limit N] [--start-from RACE_ID]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.oi import load_config
from src.oi.scraping.race import fetch_race_result
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="取得レース数の上限（テスト用）")
    parser.add_argument("--start-from", type=str, default=None, help="このrace_idからリジューム")
    parser.add_argument("--ids", type=str, nargs="*", default=None, help="単発取得用race_idリスト")
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = ROOT / cfg["paths"]["raw"]
    results_dir = raw_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # race_id 一覧の取得元
    if args.ids:
        race_ids = args.ids
    else:
        ids_path = raw_dir / "race_ids.csv"
        if not ids_path.exists():
            logger.error(f"{ids_path} が見つかりません。先に 01_fetch_calendar.py を実行してください。")
            sys.exit(1)
        with open(ids_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            race_ids = [row["race_id"] for row in reader]

    # リジューム位置
    if args.start_from:
        try:
            idx = race_ids.index(args.start_from)
            race_ids = race_ids[idx:]
        except ValueError:
            logger.warning(f"{args.start_from} がリスト内にありません。先頭から実行します。")

    # limit
    if args.limit:
        race_ids = race_ids[: args.limit]

    logger.info(f"対象 {len(race_ids)} レースをスクレイプ開始")

    success = 0
    failed: list[str] = []
    for i, rid in enumerate(race_ids, 1):
        out_path = results_dir / f"{rid}.json"
        if out_path.exists():
            success += 1
            continue
        try:
            data = fetch_race_result(rid)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            success += 1
        except Exception as e:
            logger.error(f"[{i}/{len(race_ids)}] {rid} 失敗: {e}")
            failed.append(rid)
            continue
        if i % 50 == 0:
            logger.info(f"進捗 {i}/{len(race_ids)} 成功 {success} 失敗 {len(failed)}")

    logger.info(f"完了: 成功 {success} / 失敗 {len(failed)}")
    if failed:
        fail_path = raw_dir / "failed_race_ids.txt"
        fail_path.write_text("\n".join(failed), encoding="utf-8")
        logger.info(f"失敗リストを保存: {fail_path}")


if __name__ == "__main__":
    main()
