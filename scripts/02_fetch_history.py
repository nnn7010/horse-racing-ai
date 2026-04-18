"""02: 対象コース条件と同条件の過去レース結果を取得する。

最適化: カレンダーから全開催日を一括取得し、
レースIDを収集後、結果を取得してからコース条件でフィルタする。
"""

import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.scraping.race import fetch_race_ids_by_date, fetch_race_result
from src.utils.http import fetch
from src.utils.logger import get_logger

logger = get_logger("02_fetch_history")

EXCLUDE_WORDS = ["新馬", "障害"]


def get_race_dates(start_date: date, end_date: date) -> list[date]:
    """カレンダーから全開催日を取得する。"""
    all_dates = set()
    current = date(start_date.year, start_date.month, 1)
    end_month = date(end_date.year, end_date.month, 1)
    while current <= end_month:
        url = f"https://race.netkeiba.com/top/calendar.html?year={current.year}&month={current.month}"
        html = fetch(url, encoding="utf-8")
        for m in re.finditer(r"kaisai_date=(\d{8})", html):
            dt_str = m.group(1)
            try:
                dt = date(int(dt_str[:4]), int(dt_str[4:6]), int(dt_str[6:8]))
                if start_date <= dt <= end_date:
                    all_dates.add(dt)
            except ValueError:
                pass
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return sorted(all_dates)


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 対象コース条件を読み込み
    courses_file = raw_dir / "target_courses.json"
    if not courses_file.exists():
        logger.error("target_courses.json not found. Run 01_fetch_target.py first.")
        sys.exit(1)

    with open(courses_file, encoding="utf-8") as f:
        courses = json.load(f)

    # 対象場コード
    target_place_codes = set(c["place_code"] for c in courses)
    target_conditions = set(
        (c["place_code"], c["surface"], c["distance"]) for c in courses
    )
    logger.info(f"Target place codes: {target_place_codes}")
    logger.info(f"Target conditions: {len(target_conditions)}")

    start_date = date.fromisoformat(config["train_period"]["start"])
    end_date = date.fromisoformat(config["train_period"]["end"])

    # 1. 全開催日を取得
    race_dates = get_race_dates(start_date, end_date)
    logger.info(f"Found {len(race_dates)} race dates from {start_date} to {end_date}")

    # 2. 各開催日のレースIDを収集（対象場コードのみ）
    all_race_ids = []
    for dt in race_dates:
        race_ids = fetch_race_ids_by_date(dt)
        for rid in race_ids:
            place_code = rid[4:6]
            if place_code in target_place_codes:
                all_race_ids.append(rid)

    all_race_ids = list(dict.fromkeys(all_race_ids))  # 重複除去
    logger.info(f"Total candidate race IDs: {len(all_race_ids)}")

    # 3. 既存結果を読み込み（途中再開用）
    output_file = raw_dir / "historical_results.json"
    existing_results = []
    existing_ids = set()
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            existing_results = json.load(f)
        existing_ids = {r["race_id"] for r in existing_results}
        logger.info(f"Existing results: {len(existing_results)} races")

    remaining = [rid for rid in all_race_ids if rid not in existing_ids]
    logger.info(f"Remaining to fetch: {len(remaining)}")

    # 4. レース結果を取得してフィルタ
    all_results = list(existing_results)
    matched = 0

    for i, race_id in enumerate(remaining):
        try:
            result = fetch_race_result(race_id)
        except Exception as e:
            logger.error(f"Failed to fetch {race_id}: {e}")
            continue

        # 新馬・障害除外
        if result.get("is_debut") or result.get("is_hurdle"):
            continue
        race_name = result.get("race_name", "")
        if any(w in race_name for w in EXCLUDE_WORDS):
            continue

        # コース条件フィルタ: 対象コースの±200m以内
        r_surface = result.get("surface", "")
        r_distance = result.get("distance", 0)
        r_place = race_id[4:6]

        is_match = False
        for pc, sf, dist in target_conditions:
            if pc == r_place and sf == r_surface and abs(dist - r_distance) <= 200:
                is_match = True
                break

        if not is_match:
            continue

        all_results.append(result)
        matched += 1

        # 50レースごとに途中保存
        if matched % 50 == 0:
            logger.info(f"Progress: {i + 1}/{len(remaining)}, matched: {matched}")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    # 最終保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    total_entries = sum(len(r.get("results", [])) for r in all_results)
    logger.info(f"Saved {len(all_results)} races ({total_entries} entries) to {output_file}")


if __name__ == "__main__":
    main()
