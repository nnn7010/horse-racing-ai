"""予測モジュール。学習済みモデルで複勝圏内確率を予測する。"""

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.build import get_feature_columns
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_model(model_dir: str = "models") -> tuple[lgb.Booster, list[str], object | None]:
    """学習済みモデルと特徴量列、キャリブレーターをロードする。"""
    model_path = Path(model_dir)
    model = lgb.Booster(model_file=str(model_path / "lgbm_model.txt"))
    with open(model_path / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    calibrator = None
    calib_path = model_path / "calibrator.pkl"
    if calib_path.exists():
        with open(calib_path, "rb") as f:
            calibrator = pickle.load(f)
        logger.info("Calibrator loaded")
    else:
        logger.warning("No calibrator found — using raw probabilities")
    return model, feature_cols, calibrator


def predict_probabilities(
    model: lgb.Booster,
    feature_cols: list[str],
    df: pd.DataFrame,
    calibrator=None,
) -> pd.DataFrame:
    """各馬の複勝圏内確率を予測する。"""
    df = df.copy()

    # 特徴量の準備
    missing_cols = [c for c in feature_cols if c not in df.columns]
    for c in missing_cols:
        df[c] = 0.0

    X = df[feature_cols].astype(float).fillna(-999)
    probs = model.predict(X)

    df["pred_top3_prob_raw"] = probs

    # キャリブレーション適用（検証セットで学習したIsotonic Regression）
    if calibrator is not None:
        probs = calibrator.predict(probs)
        df["pred_top3_prob_raw"] = probs

    # レース内で正規化（合計=3に調整。3頭が3着以内に入るので）
    if "race_id" in df.columns:
        df["pred_top3_prob"] = df.groupby("race_id")["pred_top3_prob_raw"].transform(
            lambda x: x / x.sum() * min(3, len(x))
        )
    else:
        total = probs.sum()
        df["pred_top3_prob"] = probs / total * 3 if total > 0 else probs

    # 0〜80%にクリップし、再正規化
    df["pred_top3_prob"] = df["pred_top3_prob"].clip(0.01, 0.80)
    if "race_id" in df.columns:
        df["pred_top3_prob"] = df.groupby("race_id")["pred_top3_prob"].transform(
            lambda x: x / x.sum() * min(3, len(x))
        )
    df["pred_top3_prob"] = df["pred_top3_prob"].clip(0.01, 0.80)

    logger.info(f"Predicted {len(df)} entries")
    return df
