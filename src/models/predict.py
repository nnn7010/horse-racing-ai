"""予測モジュール。学習済みモデルで複勝圏内確率を予測する。"""

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.build import get_feature_columns
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_model(model_dir: str = "models") -> tuple[lgb.Booster, list[str]]:
    """学習済みモデルと特徴量列をロードする。"""
    model_path = Path(model_dir)
    model = lgb.Booster(model_file=str(model_path / "lgbm_model.txt"))
    with open(model_path / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, feature_cols


def predict_probabilities(
    model: lgb.Booster,
    feature_cols: list[str],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """各馬の複勝圏内確率を予測する。"""
    df = df.copy()

    # 特徴量の準備
    missing_cols = [c for c in feature_cols if c not in df.columns]
    for c in missing_cols:
        df[c] = 0.0

    X = df[feature_cols].astype(float).fillna(-999)
    probs = model.predict(X)

    df["pred_top3_prob"] = probs

    # レース内で正規化（確率の合計調整）
    if "race_id" in df.columns:
        df["pred_top3_prob_norm"] = df.groupby("race_id")["pred_top3_prob"].transform(
            lambda x: x / x.sum() * min(3, len(x))  # 3着以内なので期待値は3
        )
    else:
        df["pred_top3_prob_norm"] = probs

    logger.info(f"Predicted {len(df)} entries")
    return df
