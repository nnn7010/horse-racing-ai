"""血統特徴量の構築。"""

import pandas as pd

from src.features.pedigree_dict import classify_sire_line
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_pedigree_features(
    horses_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """血統系統の特徴量を構築する。

    - 父・母父・母母父の系統分類（one-hot）
    - 各系統のコース別複勝率
    """
    # 系統分類
    horses_df = horses_df.copy()
    horses_df["sire_line"] = horses_df["sire"].apply(classify_sire_line)
    horses_df["dam_sire_line"] = horses_df["dam_sire"].apply(classify_sire_line)
    horses_df["dam_dam_sire_line"] = horses_df["dam_dam_sire"].apply(classify_sire_line)

    # コース別血統成績を計算
    if not results_df.empty and "horse_id" in results_df.columns:
        # resultsにhorse情報をマージ
        merged = results_df.merge(
            horses_df[["horse_id", "sire", "sire_line"]],
            on="horse_id",
            how="left",
        )

        # 父のコース別複勝率
        if "surface" in merged.columns and "distance" in merged.columns:
            sire_course_stats = (
                merged.groupby(["sire", "surface", "distance"])
                .apply(
                    lambda g: pd.Series({
                        "sire_course_top3_rate": (g["finish_position"] <= 3).mean()
                        if len(g) > 0 else 0.0,
                        "sire_course_count": len(g),
                    }),
                    include_groups=False,
                )
                .reset_index()
            )
            horses_df = horses_df.merge(
                sire_course_stats.rename(columns={"sire": "sire"}),
                on="sire",
                how="left",
                suffixes=("", "_sire_stats"),
            )

    return horses_df


def encode_sire_lines(df: pd.DataFrame) -> pd.DataFrame:
    """系統分類をダミー変数化する。"""
    lines = ["サンデーサイレンス系", "ノーザンダンサー系", "ミスプロ系", "ネイティヴダンサー系", "ナスルーラ系", "その他"]

    for col_prefix, col in [("sire_line", "sire_line"), ("damsire_line", "dam_sire_line"), ("damdamsire_line", "dam_dam_sire_line")]:
        if col in df.columns:
            for line in lines:
                df[f"{col_prefix}_{line}"] = (df[col] == line).astype(int)

    return df
