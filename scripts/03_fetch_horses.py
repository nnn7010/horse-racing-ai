"""03: 出走馬の血統情報を取得する。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.scraping.horse import fetch_horse_info
from src.utils.logger import get_logger

logger = get_logger("03_fetch_horses")


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["paths"]["raw"])

    # 全馬IDを収集
    horse_ids = set()

    # 対象レースの出走馬
    target_file = raw_dir / "target_races.json"
    if target_file.exists():
        with open(target_file, encoding="utf-8") as f:
            target_races = json.load(f)
        for race in target_races:
            for entry in race.get("entries", []):
                if entry.get("horse_id"):
                    horse_ids.add(entry["horse_id"])

    # 過去レースの馬
    history_file = raw_dir / "historical_results.json"
    if history_file.exists():
        with open(history_file, encoding="utf-8") as f:
            historical = json.load(f)
        for race in historical:
            for r in race.get("results", []):
                if r.get("horse_id"):
                    horse_ids.add(r["horse_id"])

    logger.info(f"Total unique horses: {len(horse_ids)}")

    # 既にフェッチ済みの馬をスキップ
    horses_file = raw_dir / "horses.json"
    existing = {}
    if horses_file.exists():
        with open(horses_file, encoding="utf-8") as f:
            existing_list = json.load(f)
        existing = {h["horse_id"]: h for h in existing_list}

    remaining = horse_ids - set(existing.keys())
    logger.info(f"Already fetched: {len(existing)}, Remaining: {len(remaining)}")

    horses = list(existing.values())

    for i, horse_id in enumerate(sorted(remaining)):
        try:
            info = fetch_horse_info(horse_id)
            horses.append(info)
        except Exception as e:
            logger.error(f"Failed to fetch horse {horse_id}: {e}")
            continue

        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i + 1}/{len(remaining)}")
            # 途中保存
            with open(horses_file, "w", encoding="utf-8") as f:
                json.dump(horses, f, ensure_ascii=False, indent=2, default=str)

    # 最終保存
    with open(horses_file, "w", encoding="utf-8") as f:
        json.dump(horses, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"Saved {len(horses)} horses to {horses_file}")


if __name__ == "__main__":
    main()
