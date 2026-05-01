"""07: バックテスト - 新しい買い目最適化エンジンで検証する。

検証期間の各レースで:
1. モデルで予測
2. 実際のオッズ（単勝列から取得）で買い目候補を生成
3. optimizer で最適な買い目を選定
4. 実際の着順と照合して回収を計算

モデル比較: 共通モデル vs ハイブリッド（芝専用+ダート専用）を並列評価する。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss, roc_auc_score

from src.models.predict import load_model, predict_probabilities
from src.probability.plackett_luce import compute_race_probabilities
from src.evaluation.backtest import run_backtest
from src.utils.logger import get_logger

logger = get_logger("07_backtest")


def _make_predictions(model_set, valid_df):
    """(model, feature_cols, calibrator, win_model, win_calibrator) のタプルで予測する。"""
    m, fc, cal, wm, wc = model_set
    return predict_probabilities(m, fc, valid_df, calibrator=cal, win_model=wm, win_calibrator=wc)


def _make_hybrid_predictions(surface_models, valid_df):
    """芝/ダートそれぞれ専用モデルで予測し、結合して返す。"""
    parts = []
    for surface, model_set in surface_models.items():
        sub = valid_df[valid_df["surface"] == surface].copy()
        if sub.empty:
            continue
        preds = _make_predictions(model_set, sub)
        parts.append(preds)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).sort_index()


def _accuracy_table(preds_df):
    """surface 別 AUC / LogLoss を計算して DataFrame で返す。"""
    rows = []
    preds_df = preds_df[preds_df["finish_position"] > 0].copy()
    y_top3 = (preds_df["finish_position"] <= 3).astype(int)

    surfaces = {"全体": preds_df.index}
    if "surface" in preds_df.columns:
        for surf in preds_df["surface"].dropna().unique():
            surfaces[surf] = preds_df[preds_df["surface"] == surf].index

    for label, idx in surfaces.items():
        sub = preds_df.loc[idx]
        yt = y_top3.loc[idx]
        if len(sub) == 0 or yt.nunique() < 2:
            continue
        prob_col = "pred_top3_prob_raw" if "pred_top3_prob_raw" in sub.columns else "pred_top3_prob"
        p = sub[prob_col].clip(1e-6, 1 - 1e-6)
        rows.append({
            "surface": label,
            "n_races": sub["race_id"].nunique() if "race_id" in sub.columns else len(sub),
            "n_rows": len(sub),
            "top3_rate": float(yt.mean()),
            "auc": roc_auc_score(yt, p),
            "logloss": log_loss(yt, p),
        })
    return pd.DataFrame(rows).set_index("surface")


def _print_comparison(common_preds, hybrid_preds):
    """共通モデル vs ハイブリッドの精度比較をログ出力する。"""
    common_tbl = _accuracy_table(common_preds)
    hybrid_tbl = _accuracy_table(hybrid_preds)

    logger.info("\n" + "=" * 60)
    logger.info("  モデル精度比較: 共通モデル vs ハイブリッド")
    logger.info("=" * 60)
    header = f"{'surface':<8} {'レース数':>6}  {'共通AUC':>8} {'HybridAUC':>10}  {'共通LL':>8} {'HybridLL':>9}  {'AUC改善':>8}"
    logger.info(header)
    logger.info("-" * 60)

    for surf in common_tbl.index:
        if surf not in hybrid_tbl.index:
            continue
        cr = common_tbl.loc[surf]
        hr = hybrid_tbl.loc[surf]
        auc_diff = hr["auc"] - cr["auc"]
        ll_diff = hr["logloss"] - cr["logloss"]  # 負が改善
        auc_mark = "▲" if auc_diff > 0.001 else ("▼" if auc_diff < -0.001 else " ")
        logger.info(
            f"{surf:<8} {int(cr['n_races']):>6}  "
            f"{cr['auc']:>8.4f} {hr['auc']:>10.4f}  "
            f"{cr['logloss']:>8.4f} {hr['logloss']:>9.4f}  "
            f"{auc_mark}{auc_diff:>+7.4f}"
        )
    logger.info("=" * 60)
    logger.info("▲=ハイブリッド優位  ▼=共通モデル優位  (AUC差 ±0.001 未満は誤差範囲)")



def simulate_race(race_df, race_result, payouts):
    """1レース分の買い目シミュレーション。

    optimizer を使って買い目を選定し、実際の結果と照合する。
    """
    from src.betting.optimizer import build_recommendations

    # 実際のオッズを race_df の win_odds 列から構築
    all_odds = {"win": {}, "place": {}, "quinella": {}, "wide": {}, "exacta": {},
                "trio": {}, "trifecta": {}}

    for _, row in race_df.iterrows():
        num = int(row["number"])
        odds = row.get("win_odds", 0)
        if odds > 0:
            all_odds["win"][str(num).zfill(2)] = odds
            # 複勝オッズ推定: 単勝オッズから概算（単勝の約1/3、最低1.1倍）
            all_odds["place"][str(num).zfill(2)] = max(odds ** 0.5 * 0.6, 1.1)

    # 馬連・ワイド等のオッズは不明なので、単勝から推定（バックテスト用）
    nums = sorted(race_df["number"].astype(int).values)
    for i, j in [(a, b) for a in nums for b in nums if a < b]:
        oi = all_odds["win"].get(str(i).zfill(2), 0)
        oj = all_odds["win"].get(str(j).zfill(2), 0)
        if oi > 0 and oj > 0:
            all_odds["quinella"][f"{i:02d}-{j:02d}"] = max((oi * oj) ** 0.5 * 0.7, 2.0)
            all_odds["wide"][f"{i:02d}-{j:02d}"] = max((oi * oj) ** 0.5 * 0.3, 1.1)
            all_odds["exacta"][f"{i:02d}→{j:02d}"] = max(oi * oj * 0.5, 5.0)
            all_odds["exacta"][f"{j:02d}→{i:02d}"] = max(oj * oi * 0.5, 5.0)

    # 新エンジンで買い目生成
    results = []

    for budget, pattern in [(1000, "B"), (3000, "C")]:
        opt = build_recommendations(race_df, all_odds, budget)
        # ticket_groups から個別買い目を展開
        bets = []
        for group in opt.get("ticket_groups", []):
            bet_type = group["bet_type"]
            n = max(group.get("n_bets", 1), 1)
            amount = max(budget // n // 100 * 100, 100)
            for pick in group.get("picks", []):
                bets.append({
                    "bet_type": bet_type,
                    "numbers": pick["numbers"],
                    "odds": pick.get("odds", 0),
                    "amount": amount,
                })

        for bet in bets:
            # 的中判定
            hit = 0
            payout = 0

            actual_top3 = [
                race_result.get("finish_position", {}).get(1, 0),
                race_result.get("finish_position", {}).get(2, 0),
                race_result.get("finish_position", {}).get(3, 0),
            ]
            actual_1st = actual_top3[0] if actual_top3 else 0
            actual_2nd = actual_top3[1] if len(actual_top3) > 1 else 0
            actual_3rd = actual_top3[2] if len(actual_top3) > 2 else 0

            bt = bet["bet_type"]
            nums_str = bet["numbers"]

            if bt == "単勝":
                num = int(nums_str)
                if num == actual_1st:
                    hit = 1
                    # 実際の単勝払い戻しがあれば使用（100円あたり）
                    if payouts.get("win"):
                        payout = payouts["win"] / 100 * bet["amount"]
                    else:
                        payout = bet["odds"] * bet["amount"]

            elif bt == "複勝":
                num = int(nums_str)
                if num in {actual_1st, actual_2nd, actual_3rd}:
                    hit = 1
                    # 実際の複勝払い戻しから該当馬番の配当を検索
                    place_payout = 0
                    if payouts.get("place_detail"):
                        place_payout = payouts["place_detail"].get(str(num), 0)
                    if place_payout > 0:
                        payout = place_payout / 100 * bet["amount"]
                    else:
                        # フォールバック: 推定オッズ
                        est = all_odds["place"].get(str(num).zfill(2), 0)
                        payout = est * bet["amount"] if est > 0 else 0

            elif bt == "馬連":
                parts = sorted([int(x) for x in nums_str.split("-")])
                if set(parts) == {actual_1st, actual_2nd}:
                    hit = 1
                    key = f"{parts[0]:02d}-{parts[1]:02d}"
                    est = all_odds["quinella"].get(key, 0)
                    payout = est * bet["amount"] if est > 0 else 0

            elif bt == "ワイド":
                parts = sorted([int(x) for x in nums_str.split("-")])
                if len(parts) == 2 and set(parts).issubset({actual_1st, actual_2nd, actual_3rd}):
                    hit = 1
                    key = f"{parts[0]:02d}-{parts[1]:02d}"
                    est = all_odds["wide"].get(key, 0)
                    payout = est * bet["amount"] if est > 0 else 0

            elif bt == "馬単":
                parts = [int(x) for x in nums_str.split("→")]
                if len(parts) == 2 and parts[0] == actual_1st and parts[1] == actual_2nd:
                    hit = 1
                    key = f"{parts[0]:02d}→{parts[1]:02d}"
                    est = all_odds["exacta"].get(key, 0)
                    payout = est * bet["amount"] if est > 0 else 0

            elif bt == "三連複":
                parts = [int(x) for x in nums_str.split("-")]
                if set(parts) == {actual_1st, actual_2nd, actual_3rd}:
                    hit = 1
                    if payouts.get("trio"):
                        payout = payouts["trio"] / 100 * bet["amount"]
                    else:
                        payout = bet["odds"] * bet["amount"]

            elif bt == "三連単":
                parts = [int(x) for x in nums_str.split("→")]
                if (len(parts) == 3 and parts[0] == actual_1st
                        and parts[1] == actual_2nd and parts[2] == actual_3rd):
                    hit = 1
                    if payouts.get("trifecta"):
                        payout = payouts["trifecta"] / 100 * bet["amount"]
                    else:
                        payout = bet["odds"] * bet["amount"]

            results.append({
                "pattern": pattern,
                "bet_type": bt,
                "numbers": nums_str,
                "amount": bet["amount"],
                "odds": bet["odds"],
                "hit": hit,
                "payout": payout,
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=None,
                        help="モデルディレクトリ（省略時はconfig値）")
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    model_dir = args.model_dir or config["paths"]["models"]
    raw_dir = Path(config["paths"]["raw"])
    processed_dir = Path(config["paths"]["processed"])
    output_dir = Path(config["paths"]["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_dir:
        logger.info(f"[TEST] モデル読み込み元: {model_dir}")

    # 共通モデルをロード
    common_model_set = load_model(model_dir)
    model, feature_cols, calibrator, win_model, win_calibrator = common_model_set

    # 芝・ダート専用モデルをロード（存在すればハイブリッド比較を実施）
    surface_models = {}
    for surf, label in [("芝", "turf"), ("ダート", "dirt")]:
        if (Path(model_dir) / f"lgbm_model_{label}.txt").exists():
            surface_models[surf] = load_model(model_dir, surface=surf)
    has_hybrid = len(surface_models) == 2

    df = pd.read_parquet(processed_dir / "features.parquet")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

    valid_start = pd.Timestamp(config["split"]["valid_start"])
    valid_end = pd.Timestamp(config["split"]["valid_end"])
    valid_df = df[(df["date"] >= valid_start) & (df["date"] <= valid_end)].copy()

    logger.info(f"Validation: {valid_start.date()} ~ {valid_end.date()}, {valid_df['race_id'].nunique()} races")

    if valid_df.empty:
        logger.error("No validation data")
        sys.exit(1)

    # パターンA: 共通モデルで全レース予測
    logger.info("=== パターンA: 共通モデル ===")
    common_preds = _make_predictions(common_model_set, valid_df)

    # パターンB: ハイブリッド（芝専用 + ダート専用）
    hybrid_preds = None
    if has_hybrid:
        logger.info("=== パターンB: ハイブリッドモデル ===")
        hybrid_preds = _make_hybrid_predictions(surface_models, valid_df)

        # 精度比較レポート
        _print_comparison(common_preds, hybrid_preds)
    else:
        logger.info("芝/ダート専用モデルが未生成のため、ハイブリッド比較はスキップ")
        logger.info("（05_train.py を実行して芝・ダート別モデルを生成してください）")

    # バックテスト本体は共通モデルとハイブリッドの両方で実施
    pred_sets = [("共通モデル", common_preds)]
    if hybrid_preds is not None and not hybrid_preds.empty:
        pred_sets.append(("ハイブリッド", hybrid_preds))

    def _set_win_prob(preds):
        if "pred_win_prob" in preds.columns:
            preds["win_prob"] = preds["pred_win_prob"]
        else:
            for rid, rdf in preds.groupby("race_id"):
                strengths = rdf["pred_strength"].values if "pred_strength" in rdf.columns else rdf["pred_top3_prob"].values
                total = strengths.sum()
                preds.loc[rdf.index, "win_prob"] = strengths / total if total > 0 else 1.0 / len(rdf)
        return preds

    valid_preds = _set_win_prob(common_preds)

    # レース結果（着順）を取得
    history_file = raw_dir / "historical_results.json"
    race_results = {}
    race_payouts = {}
    if history_file.exists():
        with open(history_file, encoding="utf-8") as f:
            for race in json.load(f):
                rid = race.get("race_id", "")
                fp_map = {}
                for r in race.get("results", []):
                    pos = r.get("finish_position", 0)
                    num = r.get("number", 0)
                    if 1 <= pos <= 3:
                        fp_map[pos] = num
                if fp_map:
                    race_results[rid] = {"finish_position": fp_map}

                # 払い戻し
                payouts = race.get("payouts", {})
                if payouts:
                    p = {}
                    if payouts.get("win"):
                        p["win"] = payouts["win"][0].get("amount", 0)
                    if payouts.get("place"):
                        p["place"] = [x.get("amount", 0) for x in payouts["place"]]
                        # 馬番→配当のマッピング
                        p["place_detail"] = {}
                        for x in payouts["place"]:
                            nums_str = x.get("numbers", "").strip()
                            if nums_str:
                                p["place_detail"][nums_str] = x.get("amount", 0)
                    if payouts.get("trio"):
                        p["trio"] = payouts["trio"][0].get("amount", 0)
                    if payouts.get("trifecta"):
                        p["trifecta"] = payouts["trifecta"][0].get("amount", 0)
                    race_payouts[rid] = p

    # 両パターンでシミュレーション実行
    all_results = {}  # model_label -> bets_df

    for model_label, preds in pred_sets:
        preds = _set_win_prob(preds)
        all_bets = []
        for race_id, race_df in preds.groupby("race_id"):
            if race_id not in race_results:
                continue
            result = race_results[race_id]
            payouts = race_payouts.get(race_id, {})
            date = race_df["date"].iloc[0]
            bets = simulate_race(race_df, result, payouts)
            for b in bets:
                b["race_id"] = race_id
                b["date"] = date
            all_bets.extend(bets)

        if not all_bets:
            logger.warning(f"{model_label}: 買い目なし")
            continue

        bets_df = pd.DataFrame(all_bets)
        all_results[model_label] = bets_df
        fname = "backtest_optimizer.csv" if model_label == "共通モデル" else "backtest_hybrid.csv"
        bets_df.to_csv(output_dir / fname, index=False, encoding="utf-8-sig")

    # 結果サマリー出力
    logger.info("\n" + "=" * 60)
    logger.info("  バックテスト結果比較")
    logger.info("=" * 60)

    for model_label, bets_df in all_results.items():
        logger.info(f"\n【{model_label}】 総賭数: {len(bets_df)}")
        for pattern in ["B", "C"]:
            p_df = bets_df[bets_df["pattern"] == pattern]
            if p_df.empty:
                continue
            total_inv = p_df["amount"].sum()
            total_ret = p_df["payout"].sum()
            hits = (p_df["hit"] == 1).sum()
            roi = total_ret / total_inv * 100 if total_inv > 0 else 0
            logger.info(
                f"  パターン{pattern}: 投資{total_inv:,.0f}円 回収{total_ret:,.0f}円 "
                f"回収率{roi:.1f}% 的中{hits}/{len(p_df)}"
            )
            for bt, bt_df in p_df.groupby("bet_type"):
                inv = bt_df["amount"].sum()
                ret = bt_df["payout"].sum()
                h = (bt_df["hit"] == 1).sum()
                logger.info(
                    f"    [{bt}] 賭:{len(bt_df)} 的中:{h} "
                    f"投資:{inv:,.0f} 回収:{ret:,.0f} 回収率:{ret/inv*100:.1f}%"
                )

    logger.info("\nBacktest complete!")


if __name__ == "__main__":
    main()
