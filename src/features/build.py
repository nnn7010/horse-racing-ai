"""特徴量構築メインモジュール。

過去レース結果・馬情報・血統から学習用特徴量マトリクスを構築する。
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.features.pedigree import encode_sire_lines
from src.features.pedigree_dict import classify_sire_line
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_features(
    results_df: pd.DataFrame,
    horses_df: pd.DataFrame,
) -> pd.DataFrame:
    """全特徴量を構築して結合する。"""
    df = results_df.copy()

    # 日付をdatetimeに変換
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

    # 日付でソート
    df = df.sort_values(["date", "race_id", "number"]).reset_index(drop=True)

    # 馬情報をマージ
    if not horses_df.empty:
        horse_cols = ["horse_id", "sire", "dam_sire", "dam_dam_sire"]
        available = [c for c in horse_cols if c in horses_df.columns]
        if available:
            df = df.merge(horses_df[available], on="horse_id", how="left")

    # === 馬の特徴量 ===
    df = _add_horse_features(df)

    # === 騎手・調教師の特徴量 ===
    df = _add_jockey_trainer_features(df)

    # === 血統特徴量 ===
    df = _add_pedigree_features(df, horses_df)

    # === レース条件特徴量 ===
    df = _add_race_condition_features(df)

    # 不要な文字列列を除外
    feature_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64, np.float32, np.int32, "float64", "int64", "float32", "int32"]]
    meta_cols = ["race_id", "horse_id", "date", "finish_position", "number", "horse_name", "jockey_name", "trainer_name"]
    keep_cols = list(set(meta_cols) & set(df.columns)) + feature_cols
    keep_cols = list(dict.fromkeys(keep_cols))  # 重複除去、順序保持

    result = df[keep_cols].copy()
    logger.info(f"Built features: {result.shape[0]} rows x {result.shape[1]} cols")
    return result


def _add_horse_features(df: pd.DataFrame) -> pd.DataFrame:
    """馬ごとの過去成績特徴量を追加する。"""
    df = df.copy()

    # 近5走の成績（レースごとにグループ化して計算）
    for col in ["finish_position", "time", "last_3f"]:
        if col not in df.columns:
            continue
        for i in range(1, 6):
            new_col = f"prev{i}_{col}"
            df[new_col] = df.groupby("horse_id")[col].shift(i)

    # 近5走平均着順
    prev_finish_cols = [f"prev{i}_finish_position" for i in range(1, 6) if f"prev{i}_finish_position" in df.columns]
    if prev_finish_cols:
        df["avg_finish_5"] = df[prev_finish_cols].mean(axis=1)

    # 近5走平均上がり3F
    prev_3f_cols = [f"prev{i}_last_3f" for i in range(1, 6) if f"prev{i}_last_3f" in df.columns]
    if prev_3f_cols:
        df["avg_last_3f_5"] = df[prev_3f_cols].mean(axis=1)

    # 当該コース複勝率（同コース・距離・芝ダート）
    if all(c in df.columns for c in ["surface", "distance", "finish_position"]):
        course_key = df["surface"].astype(str) + "_" + df["distance"].astype(str)
        df["_course_key"] = course_key

        course_stats = (
            df.groupby(["horse_id", "_course_key"])
            .apply(
                lambda g: pd.Series({
                    "horse_course_top3_rate": (g["finish_position"].shift(1).rolling(100, min_periods=1).apply(lambda x: (x <= 3).mean())).iloc[-1] if len(g) > 0 else 0.0,
                    "horse_course_runs": len(g) - 1,
                }),
                include_groups=False,
            )
            .reset_index()
        )
        df = df.merge(course_stats, on=["horse_id", "_course_key"], how="left")
        df.drop("_course_key", axis=1, inplace=True)

    # 前走間隔（日数）
    if "date" in df.columns:
        df["prev_race_date"] = df.groupby("horse_id")["date"].shift(1)
        df["days_since_last"] = (df["date"] - df["prev_race_date"]).dt.days
        df.drop("prev_race_date", axis=1, inplace=True)

    # 距離適性（過去の距離と今回距離の差の平均実績）
    if "distance" in df.columns:
        df["prev_distance"] = df.groupby("horse_id")["distance"].shift(1)
        df["distance_change"] = df["distance"] - df["prev_distance"]

    # 馬体重・増減
    if "horse_weight" in df.columns:
        df["prev_weight"] = df.groupby("horse_id")["horse_weight"].shift(1)

    return df


def _add_jockey_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """騎手・調教師の成績特徴量を追加する。"""
    df = df.copy()

    # 騎手の直近1年成績（累積で計算）
    if "jockey_id" in df.columns and "finish_position" in df.columns:
        jockey_stats = (
            df.groupby("jockey_id")
            .apply(
                lambda g: pd.Series({
                    "jockey_win_rate": (g["finish_position"] == 1).expanding().mean().shift(1).iloc[-1] if len(g) > 1 else 0.0,
                    "jockey_top3_rate": (g["finish_position"] <= 3).expanding().mean().shift(1).iloc[-1] if len(g) > 1 else 0.0,
                    "jockey_rides": len(g),
                }),
                include_groups=False,
            )
            .reset_index()
        )
        df = df.merge(jockey_stats, on="jockey_id", how="left")

    # 調教師の直近1年成績
    if "trainer_id" in df.columns and "finish_position" in df.columns:
        trainer_stats = (
            df.groupby("trainer_id")
            .apply(
                lambda g: pd.Series({
                    "trainer_win_rate": (g["finish_position"] == 1).expanding().mean().shift(1).iloc[-1] if len(g) > 1 else 0.0,
                    "trainer_top3_rate": (g["finish_position"] <= 3).expanding().mean().shift(1).iloc[-1] if len(g) > 1 else 0.0,
                }),
                include_groups=False,
            )
            .reset_index()
        )
        df = df.merge(trainer_stats, on="trainer_id", how="left")

    # 騎手×馬コンビ成績
    if all(c in df.columns for c in ["jockey_id", "horse_id", "finish_position"]):
        combo_stats = (
            df.groupby(["jockey_id", "horse_id"])
            .apply(
                lambda g: pd.Series({
                    "combo_top3_rate": (g["finish_position"] <= 3).mean() if len(g) > 0 else 0.0,
                    "combo_rides": len(g),
                }),
                include_groups=False,
            )
            .reset_index()
        )
        df = df.merge(combo_stats, on=["jockey_id", "horse_id"], how="left")

    return df


def _add_pedigree_features(df: pd.DataFrame, horses_df: pd.DataFrame) -> pd.DataFrame:
    """血統特徴量を追加する。"""
    df = df.copy()

    for col in ["sire", "dam_sire", "dam_dam_sire"]:
        if col in df.columns:
            line_col = f"{col}_line"
            df[line_col] = df[col].fillna("").apply(classify_sire_line)

    # 系統をエンコード
    df = encode_sire_lines(df)

    # 父のコース別複勝率
    if all(c in df.columns for c in ["sire", "surface", "distance", "finish_position"]):
        course_key = df["surface"].astype(str) + "_" + df["distance"].astype(str)
        sire_course = (
            df.groupby(["sire", course_key.rename("_ck")])["finish_position"]
            .apply(lambda x: (x <= 3).mean())
            .reset_index(name="sire_course_top3_rate")
        )
        # すでにあれば上書きしない
        if "sire_course_top3_rate" not in df.columns:
            df["_ck"] = course_key
            df = df.merge(sire_course, left_on=["sire", "_ck"], right_on=["sire", "_ck"], how="left")
            df.drop("_ck", axis=1, inplace=True)

    return df


def _add_race_condition_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース条件の特徴量を追加する。"""
    df = df.copy()

    # 競馬場エンコード（place_codeがあれば）
    if "place_code" in df.columns:
        df["place_code_num"] = pd.to_numeric(df["place_code"], errors="coerce").fillna(0).astype(int)

    # 芝ダートエンコード
    if "surface" in df.columns:
        df["is_turf"] = (df["surface"] == "芝").astype(int)

    # 馬場状態エンコード
    condition_map = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
    if "track_condition" in df.columns:
        df["track_condition_num"] = df["track_condition"].map(condition_map).fillna(0).astype(int)

    # クラスエンコード
    class_map = {
        "新馬": 1, "未勝利": 2, "1勝クラス": 3, "2勝クラス": 4,
        "3勝クラス": 5, "オープン": 6, "リステッド": 7, "G3": 8, "G2": 9, "G1": 10,
        "GI": 10, "GII": 9, "GIII": 8,
    }
    if "class" in df.columns:
        df["class_num"] = df["class"].map(class_map).fillna(3).astype(int)

    # 頭数
    if "num_runners" not in df.columns and "race_id" in df.columns:
        df["num_runners"] = df.groupby("race_id")["number"].transform("max")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """学習に使う特徴量列名のリストを返す。"""
    exclude = {
        "race_id", "horse_id", "date", "finish_position", "number",
        "horse_name", "jockey_name", "trainer_name", "jockey_id", "trainer_id",
        "sire", "dam_sire", "dam_dam_sire", "surface", "track_condition",
        "class", "race_name", "time_str", "margin", "passing", "sex_age",
        "place_code", "sire_line", "dam_sire_line", "dam_dam_sire_line",
        "win_odds", "place_odds",
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]
    return feature_cols
