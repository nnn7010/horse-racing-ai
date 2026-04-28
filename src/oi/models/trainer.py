"""マルチタスクStage1モデル: 勝率(win) + 3着内率(top3)。

実装方針:
  - 1モデル1出力のLightGBM二値分類器を2つ独立に学習する（マルチタスクとは
    厳密には言えないが、特徴量・前処理・分割を共有する）
  - 同じ前処理パイプラインで `is_win` と `is_top3` を別々に学習
  - 時系列分割（学習期間〜train_end, 検証期間 valid_start〜valid_end）
  - Optuna 30 trial でハイパーパラメータ探索
  - レース内で確率合計を1に正規化（同レース他馬との相対評価）
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

CATEGORICAL_COLS = [
    "surface", "track_condition", "weather", "sex_age",
    "sire", "dam_sire", "dam_dam_sire",
]
DROP_COLS = [
    "race_id", "date", "horse_id", "horse_name",
    "finish_position", "is_win", "is_top3",
]


def _prepare(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """LightGBM入力用の特徴量行列・ラベル・カテゴリ列名を返す。"""
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].copy()
    y = df[target].astype(int)

    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    for c in cat_cols:
        X[c] = X[c].fillna("").astype("category")

    # 数値列の欠損を0埋め
    num_cols = [c for c in X.columns if c not in cat_cols]
    X[num_cols] = X[num_cols].fillna(0)
    return X, y, cat_cols


def _time_split(df: pd.DataFrame, train_end: str, valid_start: str, valid_end: str) -> tuple[pd.Index, pd.Index]:
    train_mask = df["date"] <= train_end.replace("-", "")
    valid_mask = (df["date"] >= valid_start.replace("-", "")) & (df["date"] <= valid_end.replace("-", ""))
    return df.index[train_mask], df.index[valid_mask]


def train_lightgbm(
    df: pd.DataFrame,
    target: str,
    train_idx: pd.Index,
    valid_idx: pd.Index,
    n_trials: int = 30,
    seed: int = 42,
):
    """指定ターゲット(target='is_win' or 'is_top3')でLightGBMを学習する。"""
    import lightgbm as lgb
    import optuna

    X, y, cat_cols = _prepare(df, target)
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 5,
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "seed": seed,
        }
        ds_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
        ds_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_cols, reference=ds_train)
        model = lgb.train(
            params, ds_train,
            num_boost_round=500,
            valid_sets=[ds_valid],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        return model.best_score["valid_0"]["binary_logloss"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info(f"[{target}] best params: {best} | best logloss: {study.best_value:.4f}")

    final_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "bagging_freq": 5,
        "seed": seed,
        **best,
    }
    ds_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    ds_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_cols, reference=ds_train)
    model = lgb.train(
        final_params, ds_train,
        num_boost_round=2000,
        valid_sets=[ds_valid],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    return model, best, study.best_value


def normalize_within_race(df_with_proba: pd.DataFrame, proba_col: str, out_col: str) -> pd.DataFrame:
    """各レース内で確率を合計1に正規化する。"""
    df = df_with_proba.copy()
    df[out_col] = df.groupby("race_id")[proba_col].transform(lambda s: s / s.sum() if s.sum() > 0 else s)
    return df
