"""学習済みモデルでの予測 + Plackett-Luce 統合。

戦略:
  1. 勝率モデルで pred_win_prob を予測 → レース内正規化(合計=1)
  2. 3着内率モデルで pred_top3_prob を予測 → レース内正規化(合計=3)
  3. PL強度パラメータ = 勝率モデルの odds比 (p/(1-p))
  4. PLで三連複・三連単の組み合わせ確率を計算
  5. 単勝EV = pred_win_prob * win_odds

馬券推奨:
  - 単勝: EV > threshold の馬
  - 3連単フォーメーション: 軸馬(単勝EV最大の馬)を1着固定 →
    pred_top3_prob 上位N頭を相手として2着・3着候補
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.oi.models.trainer import _prepare, CATEGORICAL_COLS, DROP_COLS
from src.probability.plackett_luce import compute_race_probabilities
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_models(models_dir: Path) -> dict[str, lgb.Booster]:
    out = {}
    win_path = models_dir / "lgbm_win.txt"
    top3_path = models_dir / "lgbm_top3.txt"
    if win_path.exists():
        out["win"] = lgb.Booster(model_file=str(win_path))
    if top3_path.exists():
        out["top3"] = lgb.Booster(model_file=str(top3_path))
    if not out:
        raise FileNotFoundError(f"モデルが見つかりません: {models_dir}")
    return out


def predict_race(df_runners: pd.DataFrame, models: dict[str, lgb.Booster]) -> pd.DataFrame:
    """1レース or 複数レースの特徴量行列を受け取り、勝率・3着内率・各種馬券確率を返す。"""
    df = df_runners.copy()
    df["date"] = df["date"].astype(str)

    # _prepareは学習時と同じ前処理を再利用（targetは予測時には使わないのでis_winを仮置き）
    if "is_win" not in df.columns:
        df["is_win"] = 0
    if "is_top3" not in df.columns:
        df["is_top3"] = 0

    X, _, cat_cols = _prepare(df, "is_win")

    if "win" in models:
        df["pred_win_raw"] = models["win"].predict(X)
        df["pred_win_prob"] = df.groupby("race_id")["pred_win_raw"].transform(
            lambda s: s / s.sum() if s.sum() > 0 else s
        )
    if "top3" in models:
        df["pred_top3_raw"] = models["top3"].predict(X)
        df["pred_top3_prob"] = df.groupby("race_id")["pred_top3_raw"].transform(
            lambda s: s / s.sum() * min(3, len(s)) if s.sum() > 0 else s
        )
        df["pred_top3_prob"] = df["pred_top3_prob"].clip(0.01, 0.95)

    # PL強度パラメータ
    if "pred_win_prob" in df.columns:
        df["pred_strength"] = df["pred_win_prob"] / np.maximum(1 - df["pred_win_prob"], 1e-6)
    return df


def race_betting_table(df_race: pd.DataFrame, win_ev_threshold: float, partner_count: int) -> dict:
    """1レース分の馬券推奨を作る。

    Returns:
      {
        'axis': {number, horse_name, win_prob, win_odds, win_ev},
        'partners': [...],  # 3着内率上位N頭
        'trifecta': {...},  # PL確率
        'trio': {...},
      }
    """
    df = df_race.copy().reset_index(drop=True)

    # 単勝EV
    df["win_ev"] = df["pred_win_prob"] * df["win_odds"].fillna(0)

    # PL組み合わせ確率（既存src/probability/を再利用）
    pl = compute_race_probabilities(df)

    # 軸馬: 単勝EV最大
    axis_idx = df["win_ev"].idxmax()
    axis = df.loc[axis_idx, ["number", "horse_name", "pred_win_prob", "win_odds", "win_ev", "pred_top3_prob"]].to_dict()

    # 相手: 3着内率上位N頭（軸馬を除く）
    df_others = df.drop(index=axis_idx).sort_values("pred_top3_prob", ascending=False).head(partner_count)
    partners = df_others[["number", "horse_name", "pred_top3_prob", "win_odds"]].to_dict(orient="records")

    # 軸馬1着固定の3連単（軸→相手→相手 + 軸→相手→他相手）の確率を計算
    axis_num = int(df.loc[axis_idx, "number"])
    partner_nums = set(p["number"] for p in partners)
    trifecta_axis_first = {
        k: v for k, v in pl["trifecta"].items()
        if k[0] == axis_num and k[1] in partner_nums and k[2] in partner_nums
    }

    # 単勝推奨
    win_picks = df[df["win_ev"] > win_ev_threshold][
        ["number", "horse_name", "pred_win_prob", "win_odds", "win_ev"]
    ].sort_values("win_ev", ascending=False).to_dict(orient="records")

    return {
        "axis": axis,
        "partners": partners,
        "win_picks": win_picks,
        "trifecta_axis_first": trifecta_axis_first,
        "all_runners": df[
            ["number", "horse_name", "pred_win_prob", "pred_top3_prob", "win_odds", "win_ev"]
        ].sort_values("pred_win_prob", ascending=False).to_dict(orient="records"),
    }
