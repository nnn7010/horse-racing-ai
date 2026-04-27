"""特徴量構築メインモジュール。

過去レース結果・馬情報・血統から学習用特徴量マトリクスを構築する。
"""

import pandas as pd
import numpy as np
from pathlib import Path

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

    # === コーナー通過順位特徴量（脚質の本物の指標） ===
    df = _add_corner_features(df)

    # === 馬の特徴量 ===
    df = _add_horse_features(df)

    # === 騎手・調教師の特徴量 ===
    df = _add_jockey_trainer_features(df)

    # === 血統特徴量 ===
    df = _add_pedigree_features(df, horses_df)

    # === レース条件特徴量 ===
    df = _add_race_condition_features(df)

    # === コース適性ギャップ特徴量 ===
    df = _add_course_ability_features(df)

    # 不要な文字列列を除外
    feature_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64, np.float32, np.int32, "float64", "int64", "float32", "int32"]]
    meta_cols = ["race_id", "horse_id", "date", "finish_position", "number", "horse_name", "jockey_name", "trainer_name", "jockey_id", "trainer_id"]
    keep_cols = list(set(meta_cols) & set(df.columns)) + feature_cols
    keep_cols = list(dict.fromkeys(keep_cols))  # 重複除去、順序保持

    result = df[keep_cols].copy()
    logger.info(f"Built features: {result.shape[0]} rows x {result.shape[1]} cols")
    return result


def _parse_passing(passing: str) -> list[int]:
    """通過順位文字列 ("5-5-4-3" など) を整数リストにパースする。

    netkeiba は距離・コース形態によりコーナー数が 2~4 になる。
    要素数を揃えず、取れたぶんだけ返す（呼び出し側でレース内正規化する想定）。
    """
    if not passing or not isinstance(passing, str):
        return []
    parts = []
    for tok in passing.replace(" ", "").split("-"):
        try:
            v = int(tok)
            if 1 <= v <= 30:
                parts.append(v)
        except ValueError:
            continue
    return parts


def _add_corner_features(df: pd.DataFrame) -> pd.DataFrame:
    """コーナー通過順位から脚質関連特徴量を生成する。

    生成する特徴量（過去N走の平均/比率）:
      - corner_first / corner_last: 最初/最後のコーナー通過順位（生値）
      - early_pos_ratio: 序盤位置 / 頭数 （0=最前列、1=最後方）
      - pos_change: 最終 - 最初（負=後ろから差した、正=垂れた）
      - prev{i}_corner_first / prev{i}_early_pos_ratio （i=1..5）
      - avg_early_pos_ratio_5: 過去5走の早期位置比率の平均
      - avg_pos_change_5: 過去5走のコーナー間順位変動の平均
      - early_lead_rate: 過去5走で序盤2位以内だった率
      - closer_rate: 過去5走で序盤後方1/3に居た率
    """
    df = df.copy()
    if "passing" not in df.columns:
        df["passing"] = ""

    # 1走ぶんの基本量を計算
    parsed = df["passing"].apply(_parse_passing)
    df["corner_first"] = parsed.apply(lambda xs: xs[0] if xs else np.nan)
    df["corner_last"] = parsed.apply(lambda xs: xs[-1] if xs else np.nan)
    df["pos_change"] = df["corner_last"] - df["corner_first"]

    # 頭数で正規化（取得済みの num_runners が無い行は number の最大で代用）
    if "num_runners" in df.columns:
        denom = pd.to_numeric(df["num_runners"], errors="coerce")
    else:
        denom = df.groupby("race_id")["number"].transform("max")
    denom = denom.where(denom > 0)
    df["early_pos_ratio"] = df["corner_first"] / denom
    df["last_pos_ratio"] = df["corner_last"] / denom

    # 過去5走シフト
    for i in range(1, 6):
        for col in ["corner_first", "early_pos_ratio", "pos_change"]:
            df[f"prev{i}_{col}"] = df.groupby("horse_id")[col].shift(i)

    prev_early = [f"prev{i}_early_pos_ratio" for i in range(1, 6)]
    prev_pos_change = [f"prev{i}_pos_change" for i in range(1, 6)]
    df["avg_early_pos_ratio_5"] = df[prev_early].mean(axis=1)
    df["avg_pos_change_5"] = df[prev_pos_change].mean(axis=1)

    # 序盤2位以内だった率（リーク防止: 自レースは含めない）
    df["_lead_flag"] = (df["corner_first"] <= 2).astype(float)
    df["_closer_flag"] = (df["early_pos_ratio"] >= 2 / 3).astype(float)
    df["early_lead_rate"] = df.groupby("horse_id")["_lead_flag"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["closer_rate"] = df.groupby("horse_id")["_closer_flag"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df.drop(["_lead_flag", "_closer_flag"], axis=1, inplace=True)

    return df


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

    # 当該コース複勝率（同コース・距離・芝ダート、累積・ベイズ平滑化）
    if all(c in df.columns for c in ["surface", "distance", "finish_position"]):
        course_key = df["surface"].astype(str) + "_" + df["distance"].astype(str)
        df["_course_key"] = course_key
        df["_ht3"] = (df["finish_position"] <= 3).astype(float)
        raw_course = df.groupby(["horse_id", "_course_key"])["_ht3"].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        course_runs = df.groupby(["horse_id", "_course_key"]).cumcount()
        df["horse_course_runs"] = course_runs
        # ベイズ平滑化: k=3
        df["horse_course_top3_rate"] = (raw_course * course_runs + 0.21 * 3) / (course_runs + 3)
        df.drop(["_course_key", "_ht3"], axis=1, inplace=True)

    # タフコース実績（中山=6, 阪神=9, 中京=7, 小倉=10, 福島=3）
    _TOUGH_CODES = {3, 6, 7, 9, 10}
    if all(c in df.columns for c in ["place_code", "finish_position"]):
        df["_is_tough"] = pd.to_numeric(df["place_code"], errors="coerce").isin(_TOUGH_CODES).astype(float)
        df["_tough_top3"] = df["_is_tough"] * (df["finish_position"] <= 3).astype(float)
        tough_runs = df.groupby("horse_id")["_is_tough"].transform(
            lambda x: x.expanding().sum().shift(1)
        )
        raw_tough = df.groupby("horse_id")["_tough_top3"].transform(
            lambda x: x.expanding().sum().shift(1)
        )
        # ベイズ平滑化: prior=0.21, k=3
        df["horse_tough_top3_rate"] = (raw_tough + 0.21 * 3) / (tough_runs + 3)
        df.drop(["_is_tough", "_tough_top3"], axis=1, inplace=True)

    # タイムのかかるレース（消耗戦）での好走歴: スピード指数が低いレース = 遅い展開
    # speed_index = time/distance*1000 → 大きいほど遅いレース
    # race_median_speed: そのレースの全馬中央値 → 全体中央値より遅いレース = タフな消耗戦
    if all(c in df.columns for c in ["time", "distance", "finish_position", "race_id"]):
        df["_race_speed"] = df["time"] / df["distance"].clip(lower=1) * 1000
        race_median = df.groupby("race_id")["_race_speed"].transform("median")
        overall_median = df["_race_speed"].median()
        df["_is_slow_race"] = (race_median > overall_median).astype(float)  # 遅いレース=1
        df["_slow_top3"] = df["_is_slow_race"] * (df["finish_position"] <= 3).astype(float)
        slow_runs = df.groupby("horse_id")["_is_slow_race"].transform(
            lambda x: x.expanding().sum().shift(1)
        )
        raw_slow = df.groupby("horse_id")["_slow_top3"].transform(
            lambda x: x.expanding().sum().shift(1)
        )
        df["horse_slow_race_top3_rate"] = (raw_slow + 0.21 * 3) / (slow_runs + 3)
        df.drop(["_race_speed", "_is_slow_race", "_slow_top3"], axis=1, inplace=True)

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

    # === 追加: 馬の能力特徴量 ===

    # スピード指数: (基準タイム - 走破タイム) / 距離 × 1000
    if "time" in df.columns and "distance" in df.columns:
        # 距離別の平均タイムを基準にする
        df["_speed_raw"] = df["time"] / df["distance"] * 1000  # m当たりのタイム
        df["speed_index"] = df.groupby("horse_id")["_speed_raw"].shift(1)
        # 近3走平均スピード指数
        df["avg_speed_3"] = df.groupby("horse_id")["_speed_raw"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        df.drop("_speed_raw", axis=1, inplace=True)

    # 上がり3Fの相対評価（レース内順位 → 過去の平均）
    if "last_3f" in df.columns:
        # レース内上がり順位（1が最速）
        df["last_3f_rank"] = df.groupby("race_id")["last_3f"].rank(method="min")
        df["prev_last_3f_rank"] = df.groupby("horse_id")["last_3f_rank"].shift(1)
        df["avg_last_3f_rank_3"] = df.groupby("horse_id")["last_3f_rank"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )

    # 馬の累積勝率・複勝率（ベイズ平滑化付き）
    if "finish_position" in df.columns:
        df["_hw"] = (df["finish_position"] == 1).astype(float)
        df["_ht"] = (df["finish_position"] <= 3).astype(float)
        # 生の累積率
        raw_win = df.groupby("horse_id")["_hw"].transform(lambda x: x.expanding().mean().shift(1))
        raw_top3 = df.groupby("horse_id")["_ht"].transform(lambda x: x.expanding().mean().shift(1))
        horse_runs = df.groupby("horse_id").cumcount()
        df["horse_runs"] = horse_runs
        # ベイズ平滑化: (wins + prior*k) / (runs + k)
        # prior = 全体平均（勝率~7%, 複勝率~21%）, k = 5（出走5回分の重み）
        k = 5
        prior_win = 0.07
        prior_top3 = 0.21
        runs_shifted = horse_runs.clip(lower=0)
        df["horse_win_rate"] = (raw_win * runs_shifted + prior_win * k) / (runs_shifted + k)
        df["horse_top3_rate"] = (raw_top3 * runs_shifted + prior_top3 * k) / (runs_shifted + k)
        df.drop(["_hw", "_ht"], axis=1, inplace=True)

    # 過去データの有無フラグ
    df["has_history"] = (df.groupby("horse_id").cumcount() > 0).astype(int)

    # 着順の安定度（近5走の標準偏差）
    if prev_finish_cols:
        df["finish_std_5"] = df[prev_finish_cols].std(axis=1)

    # クラス変動（前走より上のクラスか下か）
    if "class" in df.columns:
        class_order = {
            "新馬": 1, "未勝利": 2, "1勝クラス": 3, "2勝クラス": 4,
            "3勝クラス": 5, "オープン": 6, "リステッド": 7,
            "GIII": 8, "GII": 9, "GI": 10, "G3": 8, "G2": 9, "G1": 10,
        }
        df["class_num"] = df["class"].map(class_order).fillna(3)
        df["prev_class_num"] = df.groupby("horse_id")["class_num"].shift(1)
        df["class_change"] = df["class_num"] - df["prev_class_num"]

    return df


def _add_jockey_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """騎手・調教師の成績特徴量を追加する（過去のみ使用、リーク防止）。"""
    df = df.copy()

    # 騎手の累積成績（各行で自分より前のデータのみ使用）
    if "jockey_id" in df.columns and "finish_position" in df.columns:
        win_flag = (df["finish_position"] == 1).astype(float)
        top3_flag = (df["finish_position"] <= 3).astype(float)
        df["jockey_win_rate"] = df.groupby("jockey_id")[win_flag.name if hasattr(win_flag, 'name') else 0].transform(
            lambda x: x.expanding().mean().shift(1)
        ) if False else 0.0
        # Simpler approach: compute expanding stats with shift
        df["_jw"] = win_flag
        df["_jt"] = top3_flag
        df["jockey_win_rate"] = df.groupby("jockey_id")["_jw"].transform(lambda x: x.expanding().mean().shift(1))
        df["jockey_top3_rate"] = df.groupby("jockey_id")["_jt"].transform(lambda x: x.expanding().mean().shift(1))
        df["jockey_rides"] = df.groupby("jockey_id").cumcount()
        df.drop(["_jw", "_jt"], axis=1, inplace=True)

    # 調教師の累積成績
    if "trainer_id" in df.columns and "finish_position" in df.columns:
        df["_tw"] = (df["finish_position"] == 1).astype(float)
        df["_tt"] = (df["finish_position"] <= 3).astype(float)
        df["trainer_win_rate"] = df.groupby("trainer_id")["_tw"].transform(lambda x: x.expanding().mean().shift(1))
        df["trainer_top3_rate"] = df.groupby("trainer_id")["_tt"].transform(lambda x: x.expanding().mean().shift(1))
        df.drop(["_tw", "_tt"], axis=1, inplace=True)

    # 騎手×馬コンビ成績（累積）
    if all(c in df.columns for c in ["jockey_id", "horse_id", "finish_position"]):
        df["_ct"] = (df["finish_position"] <= 3).astype(float)
        df["combo_top3_rate"] = df.groupby(["jockey_id", "horse_id"])["_ct"].transform(lambda x: x.expanding().mean().shift(1))
        df["combo_rides"] = df.groupby(["jockey_id", "horse_id"]).cumcount()
        df.drop("_ct", axis=1, inplace=True)

    return df


def _add_pedigree_features(df: pd.DataFrame, horses_df: pd.DataFrame) -> pd.DataFrame:
    """血統特徴量を追加する。"""
    df = df.copy()

    # 血統データがなければスキップ
    has_sire = "sire" in df.columns and df["sire"].notna().sum() > 0
    if not has_sire:
        return df

    # 父の全体複勝率・勝率（ベイズ平滑化）
    if all(c in df.columns for c in ["sire", "finish_position"]):
        df["_s1"] = (df["finish_position"] == 1).astype(float)
        df["_s3"] = (df["finish_position"] <= 3).astype(float)
        sire_runs = df.groupby("sire").cumcount()
        raw_sire_win = df.groupby("sire")["_s1"].transform(lambda x: x.expanding().mean().shift(1))
        raw_sire_t3 = df.groupby("sire")["_s3"].transform(lambda x: x.expanding().mean().shift(1))
        df["sire_win_rate"] = (raw_sire_win * sire_runs + 0.08 * 10) / (sire_runs + 10)
        df["sire_top3_rate"] = (raw_sire_t3 * sire_runs + 0.21 * 10) / (sire_runs + 10)
        df.drop(["_s1", "_s3"], axis=1, inplace=True)

    # 母父の全体複勝率・勝率（ベイズ平滑化）
    if all(c in df.columns for c in ["dam_sire", "finish_position"]):
        df["_d1"] = (df["finish_position"] == 1).astype(float)
        df["_d3"] = (df["finish_position"] <= 3).astype(float)
        ds_runs = df.groupby("dam_sire").cumcount()
        raw_ds_win = df.groupby("dam_sire")["_d1"].transform(lambda x: x.expanding().mean().shift(1))
        raw_ds_t3 = df.groupby("dam_sire")["_d3"].transform(lambda x: x.expanding().mean().shift(1))
        df["dam_sire_win_rate"] = (raw_ds_win * ds_runs + 0.08 * 10) / (ds_runs + 10)
        df["dam_sire_top3_rate"] = (raw_ds_t3 * ds_runs + 0.21 * 10) / (ds_runs + 10)
        df.drop(["_d1", "_d3"], axis=1, inplace=True)

    # 父のコース別複勝率（累積・ベイズ平滑化）
    if all(c in df.columns for c in ["sire", "surface", "distance", "finish_position"]):
        course_key = df["surface"].astype(str) + "_" + df["distance"].astype(str)
        df["_ck"] = course_key
        df["_st3"] = (df["finish_position"] <= 3).astype(float)
        raw_sire_course = df.groupby(["sire", "_ck"])["_st3"].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        sire_course_runs = df.groupby(["sire", "_ck"]).cumcount()
        # ベイズ平滑化 k=10
        df["sire_course_top3_rate"] = (raw_sire_course * sire_course_runs + 0.21 * 10) / (sire_course_runs + 10)
        df["sire_course_runs"] = sire_course_runs
        df.drop(["_ck", "_st3"], axis=1, inplace=True)

    # 母父のコース別複勝率（累積・ベイズ平滑化）
    if all(c in df.columns for c in ["dam_sire", "surface", "distance", "finish_position"]):
        course_key = df["surface"].astype(str) + "_" + df["distance"].astype(str)
        df["_ck"] = course_key
        df["_dt3"] = (df["finish_position"] <= 3).astype(float)
        raw_ds_course = df.groupby(["dam_sire", "_ck"])["_dt3"].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        ds_course_runs = df.groupby(["dam_sire", "_ck"]).cumcount()
        df["damsire_course_top3_rate"] = (raw_ds_course * ds_course_runs + 0.21 * 10) / (ds_course_runs + 10)
        df.drop(["_ck", "_dt3"], axis=1, inplace=True)

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


def _add_course_ability_features(df: pd.DataFrame) -> pd.DataFrame:
    """コースが求める能力と馬の能力のギャップを特徴量として追加する。

    course_profiles.json から各コースの能力要求値（5-95pctile基準のスコア 0-100）を取得し、
    馬の各能力軸との差分（surplus: 正=馬が上回る、負=不足）を特徴量にする。
    モデルが「このコースでこの馬の能力は十分か？」を判断できるようにする。
    """
    import json
    profiles_file = Path(__file__).resolve().parents[2] / "data/processed/course_profiles.json"
    if not profiles_file.exists():
        logger.warning("course_profiles.json not found, skipping course ability features")
        return df

    with open(profiles_file) as f:
        profiles = json.load(f)

    df = df.copy()

    # コースキーを構築
    if not all(c in df.columns for c in ["place_code_num", "is_turf", "distance"]):
        # place_code_num/is_turf が未計算の場合
        if "place_code" in df.columns and "place_code_num" not in df.columns:
            df["place_code_num"] = pd.to_numeric(df["place_code"], errors="coerce").fillna(0).astype(int)
        if "surface" in df.columns and "is_turf" not in df.columns:
            df["is_turf"] = (df["surface"] == "芝").astype(int)

    if not all(c in df.columns for c in ["place_code_num", "is_turf", "distance"]):
        return df

    df["_course_key"] = (
        df["place_code_num"].astype(str) + "_"
        + df["is_turf"].astype(str) + "_"
        + df["distance"].astype(str)
    )

    # 各コースの能力要求値をマッピング
    AXES = ["speed", "burst", "power", "course", "form", "stability", "jockey"]
    for axis in AXES:
        df[f"_demand_{axis}"] = df["_course_key"].map(
            {k: v["scores"].get(axis, 50.0) for k, v in profiles.items()}
        ).fillna(50.0)

    # 馬の能力をグローバル基準で0-100に正規化（5-95pctile、コースプロファイルと同じ基準）
    # speed: avg_speed_3 (低い=速い → invert)
    # burst: avg_last_3f_rank_3 (低い=速い → invert)
    # power: 0.5*horse_tough_top3_rate + 0.5*horse_slow_race_top3_rate
    # course: horse_course_top3_rate
    # form: avg_finish_5 (低い=好調 → invert)
    # stability: finish_std_5 (低い=安定 → invert)
    # jockey: jockey_top3_rate

    ability_map = {
        "speed":     ("avg_speed_3",            True),
        "burst":     ("avg_last_3f_rank_3",     True),
        "course":    ("horse_course_top3_rate",  False),
        "form":      ("avg_finish_5",           True),
        "stability": ("finish_std_5",           True),
        "jockey":    ("jockey_top3_rate",       False),
    }

    for axis, (col, invert) in ability_map.items():
        if col not in df.columns:
            df[f"_horse_{axis}"] = 50.0
            continue
        vals = df[col].replace(0, np.nan)
        q05 = vals.quantile(0.05)
        q95 = vals.quantile(0.95)
        if q95 == q05:
            df[f"_horse_{axis}"] = 50.0
        else:
            normed = (vals.fillna(vals.median()) - q05) / (q95 - q05) * 100
            normed = normed.clip(0, 100)
            if invert:
                normed = 100 - normed
            df[f"_horse_{axis}"] = normed

    # パワー: tough + slow の合成
    has_tough = "horse_tough_top3_rate" in df.columns
    has_slow = "horse_slow_race_top3_rate" in df.columns
    if has_tough and has_slow:
        for col in ["horse_tough_top3_rate", "horse_slow_race_top3_rate"]:
            vals = df[col].replace(0, np.nan)
            q05, q95 = vals.quantile(0.05), vals.quantile(0.95)
            if q95 == q05:
                df[f"_normed_{col}"] = 50.0
            else:
                df[f"_normed_{col}"] = ((vals.fillna(vals.median()) - q05) / (q95 - q05) * 100).clip(0, 100)
        df["_horse_power"] = 0.5 * df["_normed_horse_tough_top3_rate"] + 0.5 * df["_normed_horse_slow_race_top3_rate"]
        df.drop(["_normed_horse_tough_top3_rate", "_normed_horse_slow_race_top3_rate"], axis=1, inplace=True)
    elif has_tough:
        vals = df["horse_tough_top3_rate"].replace(0, np.nan)
        q05, q95 = vals.quantile(0.05), vals.quantile(0.95)
        df["_horse_power"] = ((vals.fillna(vals.median()) - q05) / (q95 - q05) * 100).clip(0, 100) if q95 != q05 else 50.0
    else:
        df["_horse_power"] = 50.0

    # ギャップ特徴量: horse_ability - course_demand (正=余裕、負=不足)
    for axis in AXES:
        df[f"course_{axis}_gap"] = df[f"_horse_{axis}"] - df[f"_demand_{axis}"]

    # 総合コース適性スコア: 不足分の平均ペナルティ
    shortfall_cols = []
    for axis in AXES:
        col = f"_shortfall_{axis}"
        df[col] = (df[f"_demand_{axis}"] - df[f"_horse_{axis}"]).clip(lower=0)
        shortfall_cols.append(col)
    df["course_fit_score"] = 100 - df[shortfall_cols].mean(axis=1)
    df["course_fit_score"] = df["course_fit_score"].clip(0, 100)

    # 不要な中間列を削除
    drop_cols = (
        [f"_demand_{a}" for a in AXES]
        + [f"_horse_{a}" for a in AXES]
        + [f"_shortfall_{a}" for a in AXES]
        + ["_course_key"]
    )
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)

    logger.info(f"Added course ability features: {[f'course_{a}_gap' for a in AXES]} + course_fit_score")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """学習に使う特徴量列名のリストを返す。"""
    exclude = {
        "race_id", "horse_id", "date", "finish_position", "number",
        "horse_name", "jockey_name", "trainer_name", "jockey_id", "trainer_id",
        "sire", "dam_sire", "dam_dam_sire", "surface", "track_condition",
        "class", "race_name", "time_str", "margin", "passing", "sex_age",
        "place_code",
        "win_odds", "place_odds",
        "target", "popularity", "time", "last_3f",  # prevent leakage
        "corner_first", "corner_last", "pos_change",
        "early_pos_ratio", "last_pos_ratio",  # 当該レース結果なのでリーク
        "jockey_rides", "last_3f_rank", "prev_class_num", "sire_course_runs",  # intermediate cols
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]
    return feature_cols
