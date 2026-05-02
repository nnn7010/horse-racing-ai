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
    n_trials: int = 25,
    seed: int = 42,
    model_dir: str = "models",
) -> tuple[lgb.Booster, list[str]]:
    """1着予測ランキングモデルを学習する（LambdaRank）。

    ラベル: 1着=2, 2-3着=1, それ以外=0（3段階 relevance score）
    目的関数: lambdarank（レース内の順位を直接最適化）
    Optuna 評価指標: rank-1 的中率（最高スコア馬が実際に1着に入る率）
    """
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

    df = df[df["finish_position"] > 0].copy()
    if "exclude_from_train" in df.columns:
        df = df[~df["exclude_from_train"]].copy()

    train_end_dt = pd.Timestamp(train_end)
    valid_start_dt = pd.Timestamp(valid_start)
    valid_end_dt = pd.Timestamp(valid_end)

    # LambdaRank は race_id でソートしてグループを渡す必要がある
    train_df = df[df["date"] <= train_end_dt].sort_values("race_id").reset_index(drop=True)
    valid_df = df[(df["date"] >= valid_start_dt) & (df["date"] <= valid_end_dt)].sort_values("race_id").reset_index(drop=True)

    logger.info(f"Win model (LambdaRank) - Train: {len(train_df)} rows, Valid: {len(valid_df)} rows")

    feature_cols = get_feature_columns(df)

    X_train = train_df[feature_cols].astype(float).fillna(-999).values
    X_valid = valid_df[feature_cols].astype(float).fillna(-999).values

    # 3段階 relevance ラベル
    def _rank_label(pos):
        if pos == 1:
            return 2
        if pos <= 3:
            return 1
        return 0

    y_train = train_df["finish_position"].apply(_rank_label).values
    y_valid = valid_df["finish_position"].apply(_rank_label).values

    train_groups = train_df.groupby("race_id").size().values
    valid_groups = valid_df.groupby("race_id").size().values

    n_valid_races = valid_df["race_id"].nunique()
    logger.info(f"Train groups: {len(train_groups)} races, Valid groups: {len(valid_groups)} races")

    def _rank1_hit_rate(model, X, df_ref):
        scores = model.predict(X)
        tmp = df_ref.copy()
        tmp["_score"] = scores
        hits = tmp.groupby("race_id").apply(
            lambda g: g.nlargest(1, "_score").iloc[0]["finish_position"] == 1
        ).sum()
        return hits / df_ref["race_id"].nunique()

    def objective(trial):
        gain2 = trial.suggest_int("gain2", 2, 10)
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1],
            "label_gain": [0, 1, gain2],
            "boosting_type": "gbdt",
            "seed": seed,
            "verbose": -1,
            "n_jobs": 1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        dtrain = lgb.Dataset(X_train.copy(), label=y_train.copy(), group=train_groups, free_raw_data=False)
        dvalid = lgb.Dataset(X_valid.copy(), label=y_valid.copy(), group=valid_groups, reference=dtrain, free_raw_data=False)

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

        return _rank1_hit_rate(model, X_valid, valid_df)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_rank1 = study.best_value
    logger.info(f"Win model best trial: {study.best_trial.number}, rank1_hit={best_rank1:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # 最良パラメータで再学習
    gain2 = study.best_params.pop("gain2")
    best_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1],
        "label_gain": [0, 1, gain2],
        "boosting_type": "gbdt",
        "seed": seed,
        "verbose": -1,
        "n_jobs": 1,
        **study.best_params,
    }

    dtrain = lgb.Dataset(X_train.copy(), label=y_train.copy(), group=train_groups, free_raw_data=False)
    dvalid = lgb.Dataset(X_valid.copy(), label=y_valid.copy(), group=valid_groups, reference=dtrain, free_raw_data=False)

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

    final_rank1 = _rank1_hit_rate(best_model, X_valid, valid_df)
    logger.info(f"Win model (LambdaRank) - Rank-1 hit rate: {final_rank1:.4f}, Trees: {best_model.num_trees()}")

    # キャリブレーション: exp-softmax スコア → 確率 → IsotonicRegression
    raw_scores = best_model.predict(X_valid)
    # レース内 exp-softmax で確率化
    valid_df_tmp = valid_df.copy()
    valid_df_tmp["_score"] = raw_scores
    valid_df_tmp["_win_prob"] = valid_df_tmp.groupby("race_id")["_score"].transform(
        lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum()
    )
    win_probs = valid_df_tmp["_win_prob"].values
    y_valid_win = (valid_df["finish_position"] == 1).astype(int).values
    win_calibrator = IsotonicRegression(out_of_bounds="clip")
    win_calibrator.fit(win_probs, y_valid_win)
    calibrated = win_calibrator.predict(win_probs)
    calib_logloss = log_loss(y_valid_win, np.clip(calibrated, 1e-7, 1 - 1e-7))
    logger.info(f"Win model calibrated LogLoss: {calib_logloss:.4f}")

    # 保存
    best_model.save_model(str(model_path / "lgbm_win_model.txt"))
    with open(model_path / "win_calibrator.pkl", "wb") as f:
        pickle.dump(win_calibrator, f)

    logger.info(f"Win model saved to {model_path}")
    return best_model, feature_cols
