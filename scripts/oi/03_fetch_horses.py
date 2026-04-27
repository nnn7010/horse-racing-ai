"""出走馬の馬個体ページ（血統・過去成績）を取得する。

scripts/oi/02 で集めた結果JSONから horse_id を集約し、
db.netkeiba.com/horse/{horse_id}/ を取得・保存する。

Usage:
  python scripts/oi/03_fetch_horses.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.oi import load_config
from src.oi.scraping.horse import fetch_horse_info
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    raw_dir = ROOT / cfg["paths"]["raw"]
    results_dir = raw_dir / "results"
    horses_dir = raw_dir / "horses"
    horses_dir.mkdir(parents=True, exist_ok=True)

    # 結果JSONから unique horse_id を集める
    horse_ids: set[str] = set()
    for jp in results_dir.glob("*.json"):
        with open(jp, encoding="utf-8") as f:
            data = json.load(f)
        for r in data.get("results", []):
            hid = r.get("horse_id")
            if hid:
                horse_ids.add(hid)

    logger.info(f"ユニーク馬数: {len(horse_ids)}")

    success = 0
    failed: list[str] = []
    sorted_ids = sorted(horse_ids)
    for i, hid in enumerate(sorted_ids, 1):
        out_path = horses_dir / f"{hid}.json"
        if out_path.exists():
            success += 1
            continue
        try:
            info = fetch_horse_info(hid)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            success += 1
        except Exception as e:
            logger.error(f"[{i}/{len(sorted_ids)}] {hid}: {e}")
            failed.append(hid)
        if i % 100 == 0:
            logger.info(f"進捗 {i}/{len(sorted_ids)} 成功 {success} 失敗 {len(failed)}")

    logger.info(f"完了: 成功 {success} / 失敗 {len(failed)}")
    if failed:
        (raw_dir / "failed_horse_ids.txt").write_text("\n".join(failed), encoding="utf-8")


if __name__ == "__main__":
    main()
