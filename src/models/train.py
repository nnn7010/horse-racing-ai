"""LightGBMモデルの学習（Stage1: 3着以内の二値分類）。

Optunaによるハイパーパラメータ最適化（30trial）。
時系列分割: 学習〜2026/2/28、検証 2026/3/1〜2026/4/17
"""

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

from src.features.build import get_feature_columns
from src.utils.logger import get_logger

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_model(
    df: pd.DataFrame,
    train_end: str = "2026-02-28",
    valid_start: str = "2026-03-01",
    valid_end: str = "2026-04-17",
    n_trials: int = 30,
    seed: int = 42,
    model_dir: str = "models",
) -> tuple[lgb.Booster, list[str]]:
    """LightGBMモデルを学習する。"""
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    # 日付変換
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

    # ターゲット: 3着以内
    df["target"] = (df["finish_position"].between(1, 3)).astype(int)

    # 無効行除去（着順不明 + 新馬戦は学習から除外）
    df = df[df["finish_position"] > 0].copy()
    if "exclude_from_train" in df.columns:
        before = len(df)
        df = df[~df["exclude_from_train"]].copy()
        logger.info(f"Excluded {before - len(df)} debut race entries from training")

    # 時系列分割
    train_end_dt = pd.Timestamp(train_end)
    valid_start_dt = pd.Timestamp(valid_start)
    valid_end_dt = pd.Timestamp(valid_end)

    train_df = df[df["date"] <= train_end_dt].copy()
    valid_df = df[(df["date"] >= valid_start_dt) & (df["date"] <= valid_end_dt)].copy()

    logger.info(f"Train: {len(train_df)} rows, Valid: {len(valid_df)} rows")

    if train_df.empty or valid_df.empty:
        raise ValueError("Train or validation set is empty")

    # 特徴量列
    feature_cols = get_feature_columns(df)
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")

    X_train = train_df[feature_cols].astype(float).fillna(-999).values
    y_train = train_df["target"].values
    X_valid = valid_df[feature_cols].astype(float).fillna(-999).values
    y_valid = valid_df["target"].values

    # Optuna最適化
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "seed": seed,
            "verbose": -1,
            "n_jobs": 1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        dtrain = lgb.Dataset(X_train.copy(), label=y_train.copy(), free_raw_data=False)
        dvalid = lgb.Dataset(X_valid.copy(), label=y_valid.copy(), reference=dtrain, free_raw_data=False)

        callbacks = [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0),
        ]

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dvalid],
            callbacks=callbacks,
        )

        preds = model.predict(X_valid)
        return log_loss(y_valid, preds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Best trial: {study.best_trial.number}, logloss={study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # 最良パラメータで再学習
    best_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "seed": seed,
        "verbose": -1,
        "n_jobs": 1,
        **study.best_params,
    }

    dtrain = lgb.Dataset(X_train.copy(), label=y_train.copy(), free_raw_data=False)
    dvalid = lgb.Dataset(X_valid.copy(), label=y_valid.copy(), reference=dtrain, free_raw_data=False)

    callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(100),
    ]

    best_model = lgb.train(
        best_params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=callbacks,
    )

    # 検証スコア
    valid_preds = best_model.predict(X_valid)
    auc = roc_auc_score(y_valid, valid_preds)
    logloss = log_loss(y_valid, valid_preds)
    logger.info(f"Final model - AUC: {auc:.4f}, LogLoss: {logloss:.4f}")

    # Isotonic Regression でキャリブレーション
    # 検証セットの生予測値 → 実ラベルでキャリブレーション曲線を学習
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(valid_preds, y_valid)
    calibrated_preds = calibrator.predict(valid_preds)
    calib_logloss = log_loss(y_valid, calibrated_preds)
    logger.info(f"Calibrated LogLoss: {calib_logloss:.4f} (before: {logloss:.4f})")

    # キャリブレーション効果の確認（確率帯別）
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    calib_df = pd.DataFrame({"raw": valid_preds, "calibrated": calibrated_preds, "actual": y_valid})
    calib_df["bin"] = pd.cut(calib_df["calibrated"], bins=bins)
    calib_summary = calib_df.groupby("bin", observed=True).agg(
        n=("actual", "count"),
        mean_calib=("calibrated", "mean"),
        actual_rate=("actual", "mean"),
    )
    logger.info(f"Calibration check:\n{calib_summary.to_string()}")

    # 特徴量重要度
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": best_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    logger.info(f"Top 10 features:\n{importance.head(10).to_string()}")

    # 保存
    best_model.save_model(str(model_path / "lgbm_model.txt"))
    with open(model_path / "feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    with open(model_path / "calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)
    importance.to_csv(model_path / "feature_importance.csv", index=False)

    logger.info(f"Model saved to {model_path}")
    return best_model, feature_cols


def train_win_model(
    df: pd.DataFrame,
    train_end: str = "2026-02-28",
    valid_start: str = "2026-03-01",
    valid_end: str = "2026-04-17",
    n_trials: int = 20,
    seed: int = 42,
    model_dir: str = "models",
) -> tuple[lgb.Booster, list[str]]:
    """1着予測モデルを学習する（単勝・馬単・三連単用）。"""
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

    # ターゲット: 1着のみ
    df["target_win"] = (df["finish_position"] == 1).astype(int)

    df = df[df["finish_position"] > 0].copy()
    if "exclude_from_train" in df.columns:
        df = df[~df["exclude_from_train"]].copy()

    train_end_dt = pd.Timestamp(train_end)
    valid_start_dt = pd.Timestamp(valid_start)
    valid_end_dt = pd.Timestamp(valid_end)

    train_df = df[df["date"] <= train_end_dt].copy()
    valid_df = df[(df["date"] >= valid_start_dt) & (df["date"] <= valid_end_dt)].copy()

    logger.info(f"Win model - Train: {len(train_df)} rows, Valid: {len(valid_df)} rows")
    logger.info(f"Win rate - Train: {train_df['target_win'].mean():.3f}, Valid: {valid_df['target_win'].mean():.3f}")

    # target_win はラベル列なので feature_cols から除外してから取得
    feature_cols = [c for c in get_feature_columns(df) if c != "target_win"]

    X_train = train_df[feature_cols].astype(float).fillna(-999).values
    y_train = train_df["target_win"].values
    X_valid = valid_df[feature_cols].astype(float).fillna(-999).values
    y_valid = valid_df["target_win"].values

    # 1着は不均衡データ: pos_weight で補正
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "seed": seed,
            "verbose": -1,
            "n_jobs": 1,
            "scale_pos_weight": scale_pos_weight,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        dtrain = lgb.Dataset(X_train.copy(), label=y_train.copy(), free_raw_data=False)
        dvalid = lgb.Dataset(X_valid.copy(), label=y_valid.copy(), reference=dtrain, free_raw_data=False)

        callbacks = [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0),
        ]

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dvalid],
            callbacks=callbacks,
        )

        preds = model.predict(X_valid)
        return log_loss(y_valid, preds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Win model best trial: {study.best_trial.number}, logloss={study.best_value:.4f}")

    best_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "seed": seed,
        "verbose": -1,
        "n_jobs": 1,
        "scale_pos_weight": scale_pos_weight,
        **study.best_params,
    }

    dtrain = lgb.Dataset(X_train.copy(), label=y_train.copy(), free_raw_data=False)
    dvalid = lgb.Dataset(X_valid.copy(), label=y_valid.copy(), reference=dtrain, free_raw_data=False)

    callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(100),
    ]

    best_model = lgb.train(
        best_params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=callbacks,
    )

    valid_preds = best_model.predict(X_valid)
    auc = roc_auc_score(y_valid, valid_preds)
    logloss = log_loss(y_valid, valid_preds)
    logger.info(f"Win model - AUC: {auc:.4f}, LogLoss: {logloss:.4f}")

    # キャリブレーション
    win_calibrator = IsotonicRegression(out_of_bounds="clip")
    win_calibrator.fit(valid_preds, y_valid)
    calibrated_preds = win_calibrator.predict(valid_preds)
    calib_logloss = log_loss(y_valid, calibrated_preds)
    logger.info(f"Win model calibrated LogLoss: {calib_logloss:.4f}")

    # 保存
    best_model.save_model(str(model_path / "lgbm_win_model.txt"))
    with open(model_path / "win_calibrator.pkl", "wb") as f:
        pickle.dump(win_calibrator, f)

    logger.info(f"Win model saved to {model_path}")
    return best_model, feature_cols
