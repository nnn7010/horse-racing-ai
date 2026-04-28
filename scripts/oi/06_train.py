"""勝率モデルと3着内率モデルを学習する。

Usage:
  python scripts/oi/06_train.py
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.oi import load_config
from src.oi.models.trainer import _time_split, train_lightgbm
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["paths"]["processed"]
    models_dir = ROOT / cfg["paths"]["models"]
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(proc / "features.parquet")
    df["date"] = df["date"].astype(str)

    train_idx, valid_idx = _time_split(
        df, cfg["split"]["train_end"], cfg["split"]["valid_start"], cfg["split"]["valid_end"]
    )
    logger.info(f"train {len(train_idx)} / valid {len(valid_idx)}")

    summary: dict = {}
    for target in cfg["model"]["targets"]:
        col = "is_win" if target == "win" else "is_top3"
        model, best_params, best_score = train_lightgbm(
            df, col, train_idx, valid_idx,
            n_trials=cfg["model"]["n_trials"], seed=cfg["model"]["seed"],
        )
        path = models_dir / f"lgbm_{target}.txt"
        model.save_model(str(path))
        summary[target] = {
            "best_params": best_params,
            "best_logloss": best_score,
            "model_path": str(path.relative_to(ROOT)),
        }
        logger.info(f"保存: {path}")

    with open(models_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
