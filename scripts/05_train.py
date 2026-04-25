"""05: モデル学習（LightGBM + Optuna）。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from src.models.train import train_model, train_win_model
from src.utils.logger import get_logger

logger = get_logger("05_train")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=None,
                        help="モデル保存先ディレクトリ（省略時はconfig値）")
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    model_dir = args.model_dir or config["paths"]["models"]
    processed_dir = Path(config["paths"]["processed"])
    features_file = processed_dir / "features.parquet"

    if not features_file.exists():
        logger.error("features.parquet not found. Run 04_build_features.py first.")
        sys.exit(1)

    df = pd.read_parquet(features_file)
    logger.info(f"Loaded features: {df.shape}")
    if args.model_dir:
        logger.info(f"[TEST] モデル出力先: {model_dir}")

    model, feature_cols = train_model(
        df,
        train_end=config["split"]["train_end"],
        valid_start=config["split"]["valid_start"],
        valid_end=config["split"]["valid_end"],
        n_trials=config["model"]["n_trials"],
        seed=config["model"]["seed"],
        model_dir=model_dir,
    )

    logger.info(f"Top3 model training complete. {len(feature_cols)} features used.")

    # 1着予測モデル（単勝・馬単・三連単用）
    logger.info("=== 1着モデル学習開始 ===")
    win_model, _ = train_win_model(
        df,
        train_end=config["split"]["train_end"],
        valid_start=config["split"]["valid_start"],
        valid_end=config["split"]["valid_end"],
        n_trials=20,
        seed=config["model"]["seed"],
        model_dir=model_dir,
    )
    logger.info("Win model training complete.")


if __name__ == "__main__":
    main()
