"""01: 対象日のレース一覧を取得する。"""

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.scraping.target_day import fetch_race_detail, fetch_race_list
from src.utils.logger import get_logger

logger = get_logger("01_fetch_target")


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_races = []

    for dt_str in config["target_dates"]:
        target_date = date.fromisoformat(dt_str)
        logger.info(f"Fetching race list for {target_date}")

        races = fetch_race_list(target_date)
        logger.info(f"Found {len(races)} races on {target_date}")

        for race in races:
            detail = fetch_race_detail(race["race_id"])

            # 新馬除外
            if detail.get("is_debut"):
                logger.info(f"Skipping {race['race_id']}: debut race")
                continue

            race.update(detail)
            all_races.append(race)

    # 保存
    output_file = raw_dir / "target_races.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_races, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"Saved {len(all_races)} target races to {output_file}")

    # コース条件のログ出力
    courses = set()
    for r in all_races:
        if r.get("surface") and r.get("distance") and r.get("place_code"):
            courses.add((r["place_code"], r["surface"], r["distance"]))

    logger.info(f"Unique course conditions: {len(courses)}")
    for c in sorted(courses):
        logger.info(f"  {c[0]} {c[1]} {c[2]}m")


if __name__ == "__main__":
    main()
