"""馬の能力パラメータ算出モジュール。

全場の過去成績から、馬の能力を抽象的なパラメータに変換する。
LightGBMがコース条件との交互作用を学習する前提。
"""

import re
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_ability(past_results: list[dict], max_races: int = 10) -> dict:
    """過去成績リストから能力パラメータを算出する。

    Args:
        past_results: 馬の過去成績（新しい順）
        max_races: 計算に使う最大レース数

    Returns:
        能力パラメータの辞書
    """
    ability = {
        "ability_speed": 0.0,         # スピード（タイム/距離の正規化値）
        "ability_stamina": 0.0,       # スタミナ（長距離での着順安定度）
        "ability_power": 0.0,         # パワー（ダート・重馬場成績）
        "ability_acceleration": 0.0,  # 瞬発力（上がり3F相対順位）
        "ability_front": 0.0,         # 先行力（通過順位の平均）
        "ability_stability": 0.0,     # 安定性（着順の標準偏差の逆数）
        "ability_growth": 0.0,        # 成長度（近走と全体の差）
        "ability_class": 0.0,         # クラス力（最高クラス）
        "ability_runs": 0,            # 出走回数
        "ability_win_rate": 0.0,      # 通算勝率
        "ability_top3_rate": 0.0,     # 通算複勝率
        "ability_turf_top3": 0.0,     # 芝複勝率
        "ability_dirt_top3": 0.0,     # ダート複勝率
        "ability_heavy_top3": 0.0,    # 道悪複勝率
        "ability_avg_odds": 0.0,      # 平均オッズ（市場評価の目安）
    }

    if not past_results:
        return ability

    # 直近max_races件を使用
    recent = past_results[:max_races]
    all_races = past_results

    # 有効レースのみ（着順0=除外等を除く）
    valid = [r for r in recent if r.get("finish", 0) > 0]
    all_valid = [r for r in all_races if r.get("finish", 0) > 0]

    if not valid:
        return ability

    ability["ability_runs"] = len(all_valid)
    finishes = [r["finish"] for r in valid]
    all_finishes = [r["finish"] for r in all_valid]

    # --- 勝率・複勝率 ---
    ability["ability_win_rate"] = sum(1 for f in all_finishes if f == 1) / len(all_finishes)
    ability["ability_top3_rate"] = sum(1 for f in all_finishes if f <= 3) / len(all_finishes)

    # --- スピード ---
    speeds = []
    for r in valid:
        t = r.get("time", 0)
        d = r.get("distance", 0)
        if t > 0 and d > 0:
            speeds.append(t / d * 1000)  # m当たりのタイム（小さいほど速い）
    if speeds:
        # 反転して正規化（速いほど高スコア）
        ability["ability_speed"] = max(0, 70 - np.mean(speeds))  # 概ね0-10のスケール

    # --- スタミナ ---
    long_races = [r for r in valid if r.get("distance", 0) >= 2000]
    short_races = [r for r in valid if r.get("distance", 0) < 1600]
    if long_races:
        long_avg = np.mean([r["finish"] for r in long_races])
        short_avg = np.mean([r["finish"] for r in short_races]) if short_races else long_avg
        ability["ability_stamina"] = max(0, (short_avg - long_avg) + 5)  # 長距離で着順が良いほど高い

    # --- パワー（ダート + 重馬場） ---
    dirt_races = [r for r in valid if r.get("surface") == "ダート"]
    heavy_races = [r for r in valid if r.get("track_cond") in ("重", "不良")]
    if dirt_races:
        dirt_top3 = sum(1 for r in dirt_races if r["finish"] <= 3) / len(dirt_races)
        ability["ability_power"] = dirt_top3 * 10
        ability["ability_dirt_top3"] = dirt_top3
    if heavy_races:
        heavy_top3 = sum(1 for r in heavy_races if r["finish"] <= 3) / len(heavy_races)
        ability["ability_heavy_top3"] = heavy_top3
        ability["ability_power"] = max(ability["ability_power"], heavy_top3 * 10)

    # --- 芝成績 ---
    turf_races = [r for r in valid if r.get("surface") == "芝"]
    if turf_races:
        ability["ability_turf_top3"] = sum(1 for r in turf_races if r["finish"] <= 3) / len(turf_races)

    # --- 瞬発力（上がり3F）---
    last_3fs = [r.get("last_3f", 0) for r in valid if r.get("last_3f", 0) > 0]
    if last_3fs:
        # 上がり3Fが速いほど高い（33秒台=最速、40秒=遅い）
        ability["ability_acceleration"] = max(0, 40 - np.mean(last_3fs)) * 1.5

    # --- 先行力 ---
    corners = [r.get("first_corner", 0) for r in valid if r.get("first_corner", 0) > 0]
    if corners:
        runners = [r.get("runners", 16) for r in valid if r.get("first_corner", 0) > 0]
        # 通過順位を頭数で正規化（0=最前、1=最後）
        rel_positions = [c / max(n, 1) for c, n in zip(corners, runners)]
        ability["ability_front"] = max(0, (1 - np.mean(rel_positions)) * 10)

    # --- 安定性 ---
    if len(finishes) >= 3:
        std = np.std(finishes)
        ability["ability_stability"] = max(0, 10 - std)  # 標準偏差が小さいほど高い

    # --- 成長度 ---
    if len(finishes) >= 4:
        recent_avg = np.mean(finishes[:3])
        older_avg = np.mean(finishes[3:])
        ability["ability_growth"] = older_avg - recent_avg  # 正なら上昇傾向

    # --- クラス力 ---
    class_map = {
        "新馬": 1, "未勝利": 2, "1勝": 3, "2勝": 4, "3勝": 5,
        "オープン": 6, "OP": 6, "リステッド": 7, "L": 7,
        "G3": 8, "G2": 9, "G1": 10, "GIII": 8, "GII": 9, "GI": 10,
    }
    max_class = 2  # デフォルト未勝利
    for r in all_valid:
        race_name = str(r.get("race_name", ""))
        for key, val in class_map.items():
            if key in race_name:
                if r["finish"] <= 3:  # そのクラスで好走した場合
                    max_class = max(max_class, val)
                else:
                    max_class = max(max_class, val - 1)
    ability["ability_class"] = max_class

    # --- 平均オッズ ---
    odds_list = [r.get("odds", 0) for r in valid if r.get("odds", 0) > 0]
    if odds_list:
        ability["ability_avg_odds"] = np.mean(odds_list)

    return ability


def build_ability_features(horse_results: dict) -> pd.DataFrame:
    """全馬の能力パラメータをDataFrameで返す。

    Args:
        horse_results: {horse_id: [過去成績リスト]} の辞書

    Returns:
        horse_id + 能力パラメータのDataFrame
    """
    rows = []
    for hid, results in horse_results.items():
        ability = compute_ability(results)
        ability["horse_id"] = hid
        rows.append(ability)

    df = pd.DataFrame(rows)
    logger.info(f"Computed ability for {len(df)} horses")

    # 統計情報
    if not df.empty:
        for col in ["ability_speed", "ability_power", "ability_acceleration", "ability_front"]:
            if col in df.columns:
                valid = df[df[col] > 0][col]
                if len(valid) > 0:
                    logger.info(f"  {col}: mean={valid.mean():.2f}, std={valid.std():.2f}")

    return df
