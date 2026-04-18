"""バックテストモジュール。

3パターンのバックテストを実行し、結果を集計・グラフ出力する。
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 日本語フォント対応
plt.rcParams["font.family"] = "sans-serif"


def run_backtest(recommendations_df: pd.DataFrame, output_dir: str = "outputs") -> dict:
    """3パターンのバックテストを実行する。

    Args:
        recommendations_df: 推奨馬券のDataFrame
            必須列: race_id, bet_type, numbers, probability, odds,
                    expected_value, hit, payout_per_100, date

    Returns:
        3パターンの結果辞書
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if recommendations_df.empty:
        logger.warning("No recommendations to backtest")
        return {}

    df = recommendations_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    results = {}

    # パターンA: 全該当馬券に100円均一
    results["A"] = _pattern_a(df)

    # パターンB: 1レース上限1,000円、EV上位から配分
    results["B"] = _pattern_b(df)

    # パターンC: 1日予算3,000円、全レースEV降順で合計3,000円以内
    results["C"] = _pattern_c(df)

    # 結果出力
    _print_results(results)
    _save_results(results, out_path)
    _plot_results(df, results, out_path)

    return results


def _pattern_a(df: pd.DataFrame) -> dict:
    """パターンA: 全該当馬券に100円均一。"""
    bet_amount = 100
    total_bets = len(df)
    total_investment = total_bets * bet_amount
    total_return = df["payout_per_100"].sum() if "payout_per_100" in df.columns else 0
    hits = (df["hit"] == 1).sum() if "hit" in df.columns else 0

    return {
        "name": "A: 均一100円",
        "total_bets": total_bets,
        "total_investment": total_investment,
        "total_return": total_return,
        "roi": total_return / total_investment * 100 if total_investment > 0 else 0,
        "hits": hits,
        "hit_rate": hits / total_bets * 100 if total_bets > 0 else 0,
        "by_type": _breakdown_by_type(df, bet_amount),
    }


def _pattern_b(df: pd.DataFrame) -> dict:
    """パターンB: 1レース上限1,000円、EV上位から配分。"""
    max_per_race = 1000
    bet_unit = 100

    total_investment = 0
    total_return = 0
    total_bets = 0
    hits = 0
    details = []

    for race_id, race_df in df.groupby("race_id"):
        race_sorted = race_df.sort_values("expected_value", ascending=False)
        budget = max_per_race
        for _, row in race_sorted.iterrows():
            if budget < bet_unit:
                break
            total_investment += bet_unit
            total_bets += 1
            budget -= bet_unit
            if row.get("hit", 0) == 1:
                total_return += row.get("payout_per_100", 0)
                hits += 1
            details.append(row)

    details_df = pd.DataFrame(details)
    return {
        "name": "B: 1レース上限1000円",
        "total_bets": total_bets,
        "total_investment": total_investment,
        "total_return": total_return,
        "roi": total_return / total_investment * 100 if total_investment > 0 else 0,
        "hits": hits,
        "hit_rate": hits / total_bets * 100 if total_bets > 0 else 0,
        "by_type": _breakdown_by_type(details_df, bet_unit) if not details_df.empty else {},
    }


def _pattern_c(df: pd.DataFrame) -> dict:
    """パターンC: 1レース予算3,000円、EV降順で配分。"""
    race_budget = 3000
    bet_unit = 100

    total_investment = 0
    total_return = 0
    total_bets = 0
    hits = 0
    details = []

    for race_id, race_df in df.groupby("race_id"):
        race_sorted = race_df.sort_values("expected_value", ascending=False)
        budget = race_budget
        for _, row in race_sorted.iterrows():
            if budget < bet_unit:
                break
            total_investment += bet_unit
            total_bets += 1
            budget -= bet_unit
            if row.get("hit", 0) == 1:
                total_return += row.get("payout_per_100", 0)
                hits += 1
            details.append(row)

    details_df = pd.DataFrame(details)
    return {
        "name": "C: 1レース予算3000円",
        "total_bets": total_bets,
        "total_investment": total_investment,
        "total_return": total_return,
        "roi": total_return / total_investment * 100 if total_investment > 0 else 0,
        "hits": hits,
        "hit_rate": hits / total_bets * 100 if total_bets > 0 else 0,
        "by_type": _breakdown_by_type(details_df, bet_unit) if not details_df.empty else {},
    }


def _breakdown_by_type(df: pd.DataFrame, bet_amount: int) -> dict:
    """馬券種別ごとの内訳。"""
    breakdown = {}
    if df.empty or "bet_type" not in df.columns:
        return breakdown
    for bt, bt_df in df.groupby("bet_type"):
        n = len(bt_df)
        h = (bt_df["hit"] == 1).sum() if "hit" in bt_df.columns else 0
        ret = bt_df["payout_per_100"].sum() if "payout_per_100" in bt_df.columns else 0
        inv = n * bet_amount
        breakdown[bt] = {
            "bets": n,
            "hits": h,
            "investment": inv,
            "return": ret,
            "roi": ret / inv * 100 if inv > 0 else 0,
        }
    return breakdown


def _print_results(results: dict):
    """結果を表示する。"""
    for key, r in results.items():
        logger.info(f"\n=== パターン{r['name']} ===")
        logger.info(f"  総賭け数: {r['total_bets']}")
        logger.info(f"  総投資: {r['total_investment']:,}円")
        logger.info(f"  総回収: {r['total_return']:,.0f}円")
        logger.info(f"  回収率: {r['roi']:.1f}%")
        logger.info(f"  的中数: {r['hits']}")
        logger.info(f"  的中率: {r['hit_rate']:.1f}%")
        if r.get("by_type"):
            for bt, stats in r["by_type"].items():
                logger.info(f"    [{bt}] 賭:{stats['bets']} 的中:{stats['hits']} 投資:{stats['investment']:,} 回収:{stats['return']:,.0f} 回収率:{stats['roi']:.1f}%")


def _save_results(results: dict, out_path: Path):
    """結果をCSVに保存する。"""
    rows = []
    for key, r in results.items():
        rows.append({
            "pattern": r["name"],
            "total_bets": r["total_bets"],
            "total_investment": r["total_investment"],
            "total_return": r["total_return"],
            "roi": r["roi"],
            "hits": r["hits"],
            "hit_rate": r["hit_rate"],
        })
    pd.DataFrame(rows).to_csv(out_path / "backtest_results.csv", index=False)


def _plot_results(df: pd.DataFrame, results: dict, out_path: Path):
    """日別回収率のグラフを出力する。"""
    if "date" not in df.columns or df["date"].isna().all():
        logger.warning("No date column for plotting")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (key, r) in enumerate(results.items()):
        ax = axes[idx]

        # 日別集計
        daily = df.copy()
        daily["date_day"] = daily["date"].dt.date
        daily_stats = daily.groupby("date_day").agg(
            bets=("hit", "count"),
            returns=("payout_per_100", "sum"),
        ).reset_index()
        daily_stats["investment"] = daily_stats["bets"] * 100
        daily_stats["roi"] = daily_stats["returns"] / daily_stats["investment"] * 100

        if not daily_stats.empty:
            colors = ["green" if r > 100 else "red" for r in daily_stats["roi"]]
            ax.bar(range(len(daily_stats)), daily_stats["roi"], color=colors, alpha=0.7)
            ax.axhline(y=100, color="black", linestyle="--", linewidth=0.5)
            ax.set_title(f"Pattern {r['name']}")
            ax.set_xlabel("Day")
            ax.set_ylabel("ROI (%)")
            ax.set_xticks(range(len(daily_stats)))
            ax.set_xticklabels([str(d) for d in daily_stats["date_day"]], rotation=45, ha="right", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path / "backtest_daily_roi.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot to {out_path / 'backtest_daily_roi.png'}")
