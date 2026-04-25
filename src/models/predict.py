"""予測モジュール。学習済みモデルで複勝圏内確率を予測する。"""

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.build import get_feature_columns
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_model(model_dir: str = "models") -> tuple[lgb.Booster, list[str], object | None, lgb.Booster | None, object | None]:
    """学習済みモデルと特徴量列、キャリブレーターをロードする。

    Returns:
        (top3_model, feature_cols, calibrator, win_model, win_calibrator)
    """
    model_path = Path(model_dir)
    model = lgb.Booster(model_file=str(model_path / "lgbm_model.txt"))
    with open(model_path / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    calibrator = None
    calib_path = model_path / "calibrator.pkl"
    if calib_path.exists():
        with open(calib_path, "rb") as f:
            calibrator = pickle.load(f)
        logger.info("Top3 calibrator loaded")
    else:
        logger.warning("No top3 calibrator found — using raw probabilities")

    win_model = None
    win_calibrator = None
    win_model_path = model_path / "lgbm_win_model.txt"
    if win_model_path.exists():
        win_model = lgb.Booster(model_file=str(win_model_path))
        win_calib_path = model_path / "win_calibrator.pkl"
        if win_calib_path.exists():
            with open(win_calib_path, "rb") as f:
                win_calibrator = pickle.load(f)
        logger.info("Win model loaded")
    else:
        logger.info("No win model found — win prob derived from top3 model")

    return model, feature_cols, calibrator, win_model, win_calibrator


def predict_probabilities(
    model: lgb.Booster,
    feature_cols: list[str],
    df: pd.DataFrame,
    calibrator=None,
    win_model: lgb.Booster | None = None,
    win_calibrator=None,
) -> pd.DataFrame:
    """各馬の複勝圏内確率を予測する。"""
    df = df.copy()

    # 特徴量の準備
    missing_cols = [c for c in feature_cols if c not in df.columns]
    for c in missing_cols:
        df[c] = 0.0

    X = df[feature_cols].astype(float).fillna(-999)
    probs_raw = model.predict(X)

    # Plackett-Luceの強さパラメータ: 生のodds比（log-oddsのexp）
    # p_raw を確率から odds = p/(1-p) に変換 → 相対的な強さとして使用
    # キャリブレーション前の生確率を使う（順位付けにはキャリブレーション不要）
    strength = probs_raw / np.maximum(1.0 - probs_raw, 1e-6)
    df["pred_strength"] = strength

    # キャリブレーション適用（複勝確率の表示用）
    probs = probs_raw.copy()
    if calibrator is not None:
        probs = calibrator.predict(probs_raw)
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

    # 単勝確率: 専用winモデルがあればそれを使用、なければpred_strengthから導出
    if win_model is not None:
        X_win = df[feature_cols].astype(float).fillna(-999)
        win_probs_raw = win_model.predict(X_win)
        if win_calibrator is not None:
            win_probs_cal = win_calibrator.predict(win_probs_raw)
        else:
            win_probs_cal = win_probs_raw

        # Isotonic calibrationのステップ関数による同値化を防ぐため3段階ブレンド:
        # 90% calibrated + 5% win_raw + 5% top3_raw（別モデルでタイブレーク）
        top3_raw = probs_raw  # top3モデルの生スコア（既に計算済み）
        win_probs_blended = 0.90 * win_probs_cal + 0.05 * win_probs_raw + 0.05 * top3_raw
        df["pred_win_prob_raw"] = win_probs_blended

        # レース内で正規化（合計=1: 1頭のみ1着になる）
        if "race_id" in df.columns:
            df["pred_win_prob"] = df.groupby("race_id")["pred_win_prob_raw"].transform(
                lambda x: x / x.sum() if x.sum() > 0 else x
            )
        else:
            total = win_probs_blended.sum()
            df["pred_win_prob"] = win_probs_blended / total if total > 0 else win_probs_blended

        df["pred_win_prob"] = df["pred_win_prob"].clip(0.001, 0.99)

        # 同値が残る場合、馬番の逆順で微小差をつけてタイブレーク
        # (馬番小さい馬をわずかに優遇: 枠順の有利さを反映)
        if "number" in df.columns and "race_id" in df.columns:
            n_horses = df.groupby("race_id")["number"].transform("count")
            rank_in_race = df.groupby("race_id")["pred_win_prob"].rank(method="first", ascending=False)
            tiebreak = (1.0 - (rank_in_race - 1) / n_horses) * 1e-5
            df["pred_win_prob"] = df["pred_win_prob"] + tiebreak
            df["pred_win_prob"] = df.groupby("race_id")["pred_win_prob"].transform(
                lambda x: x / x.sum()
            )
            df["pred_win_prob"] = df["pred_win_prob"].clip(0.001, 0.99)

        # pred_strengthもwinモデルのodds比で更新（PL組み合わせ計算用）
        win_strength = win_probs_raw / np.maximum(1.0 - win_probs_raw, 1e-6)
        df["pred_strength"] = win_strength
    else:
        # winモデルなし: pred_strengthの正規化値を単勝確率として使用
        if "race_id" in df.columns:
            df["pred_win_prob"] = df.groupby("race_id")["pred_strength"].transform(
                lambda x: x / x.sum() if x.sum() > 0 else x
            )
        else:
            total = df["pred_strength"].sum()
            df["pred_win_prob"] = df["pred_strength"] / total if total > 0 else df["pred_strength"]

    logger.info(f"Predicted {len(df)} entries")
    return df
