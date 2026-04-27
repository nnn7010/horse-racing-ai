"""大井の開催日とレースIDを列挙し data/oi/raw/race_ids.csv に保存。

Usage:
  python scripts/oi/01_fetch_calendar.py
"""

import csv
import sys
from datetime import date
from pathlib import Path

# プロジェクトルートをパスに追加
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.oi import load_config
from src.oi.scraping.calendar import collect_all_race_ids
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    start = date.fromisoformat(cfg["fetch_period"]["start"])
    end = date.fromisoformat(cfg["fetch_period"]["end"])

    out_dir = ROOT / cfg["paths"]["raw"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "race_ids.csv"

    pairs = collect_all_race_ids(start, end)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "race_id"])
        for dt, rid in pairs:
            writer.writerow([dt.isoformat(), rid])

    logger.info(f"保存: {out_path} ({len(pairs)}件)")


if __name__ == "__main__":
    main()
