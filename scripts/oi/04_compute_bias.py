"""過去レースから日次トラックバイアスを計算し data/oi/processed/bias_daily.csv に保存。

Usage:
  python scripts/oi/04_compute_bias.py
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.oi import load_config
from src.oi.bias.estimator import estimate_bias
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    raw_dir = ROOT / cfg["paths"]["raw"]
    proc_dir = ROOT / cfg["paths"]["processed"]
    proc_dir.mkdir(parents=True, exist_ok=True)
    min_races = cfg["bias"]["min_races_per_day"]

    # 開催日ごとにレースをグルーピング
    by_date: dict[str, list[dict]] = defaultdict(list)
    for jp in (raw_dir / "results").glob("*.json"):
        with open(jp, encoding="utf-8") as f:
            data = json.load(f)
        d = data.get("date", "")
        if d:
            by_date[d].append(data)

    rows = []
    for d in sorted(by_date.keys()):
        races = by_date[d]
        if len(races) < min_races:
            continue
        est = estimate_bias(races, d)
        rows.append(est.to_dict())

    out = proc_dir / "bias_daily.csv"
    if rows:
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"保存: {out} ({len(rows)}日分)")
    else:
        logger.warning("データ不足のためバイアス推定できませんでした")


if __name__ == "__main__":
    main()
