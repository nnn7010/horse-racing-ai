"""予想と実結果を突き合わせて成績評価。

レース後（または開催中の経過チェックでも）実行。
未確定レースはスキップ。

使い方:
  python scripts/oi/evaluate_today.py --date 2026-04-27
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.oi.scraping.race import fetch_race_result


def fetch_or_load_result(race_id: str, force: bool = False) -> dict | None:
    """結果取得。
      - 既走確定済みファイルがあれば即返す（force時のみ再取得）
      - 未確定または強制時は HTTP キャッシュも捨てて再フェッチ
    """
    import hashlib
    cache = ROOT / "data/oi/raw/results" / f"{race_id}.json"
    if cache.exists() and not force:
        d = json.loads(cache.read_text())
        if d.get("num_runners", 0) > 0:
            return d

    # HTTPキャッシュも強制再取得 (force時 or 結果未保存)
    url = f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"
    http_cache = ROOT / "data/oi/cache" / f"{hashlib.md5(url.encode()).hexdigest()}.html"
    if http_cache.exists():
        http_cache.unlink()

    try:
        d = fetch_race_result(race_id)
        if d.get("num_runners", 0) == 0:
            return None  # 未走
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(d, ensure_ascii=False, indent=2))
        return d
    except Exception as e:
        print(f"  [warn] {race_id}: {e}", file=sys.stderr)
        return None


def evaluate_race(pred: dict, result: dict, ev_thresh: float = 1.15) -> dict:
    """1レースの予想 vs 実結果。"""
    # 着順マップ
    actual_finish = {r["number"]: r["finish_position"] for r in result["results"] if r["finish_position"] > 0}
    actual_odds = {r["number"]: r["win_odds"] for r in result["results"] if r["win_odds"] > 0}
    actual_pop = {r["number"]: r["popularity"] for r in result["results"]}

    # 1〜3着馬番
    podium = [r for r in result["results"] if r["finish_position"] in (1, 2, 3)]
    podium_sorted = sorted(podium, key=lambda x: x["finish_position"])
    win_num = next((r["number"] for r in podium_sorted if r["finish_position"] == 1), None)
    place_nums = {r["number"] for r in podium_sorted}

    # 予想ローデータ
    pred_rows = pred["rows"]  # スコア降順
    n = len(pred_rows)

    # 軸馬(スコア1位)
    axis = pred_rows[0]
    axis_actual = actual_finish.get(axis["number"], 99)
    axis_in_top3 = axis["number"] in place_nums

    # 推勝率上位3頭の3着内率
    top3_in_podium = sum(1 for r in pred_rows[:3] if r["number"] in place_nums)

    # スコア順位 vs 実着順 のSpearman簡易版（順位差の総和）
    score_rank = {r["number"]: i + 1 for i, r in enumerate(pred_rows)}
    finish_pairs = [(score_rank[n_], actual_finish[n_]) for n_ in actual_finish if n_ in score_rank]

    # 単勝EV>閾値馬のベット結果
    ev_picks = [r for r in pred_rows if r.get("ev") and r["ev"] > ev_thresh]
    ev_total = len(ev_picks)
    ev_hits = []
    ev_payout = 0  # 100円買って戻った金額の合計
    for r in ev_picks:
        if r["number"] == win_num:
            actual_o = actual_odds.get(r["number"], r["win_odds_est"])
            ev_hits.append({"number": r["number"], "actual_odds": actual_o, "est_odds": r["win_odds_est"]})
            ev_payout += int(actual_o * 100)
    ev_cost = ev_total * 100

    # 三連単フォーメーション(軸→複勝率上位4頭→同) の的中
    axis_n = axis["number"]
    partners_sorted = sorted(pred_rows[1:], key=lambda x: -(x.get("prob_top3") or 0))[:4]
    partners = [r["number"] for r in partners_sorted]
    triples_bought = []
    for a in partners:
        for b in partners:
            if a == b: continue
            triples_bought.append((axis_n, a, b))
    n_combos = len(triples_bought)
    triple_hit = None
    triple_payout = 0
    if win_num and len(podium_sorted) >= 3:
        actual_triple = (podium_sorted[0]["number"], podium_sorted[1]["number"], podium_sorted[2]["number"])
        if actual_triple in triples_bought:
            triple_hit = actual_triple
            # trifecta payout from result
            tri = result.get("payouts", {}).get("trifecta", [])
            if tri:
                triple_payout = tri[0]["amount"]
    triple_cost = n_combos * 100

    return {
        "race_no": pred["race_no"],
        "race_name": pred["race_name"],
        "podium": [(r["finish_position"], r["number"], r["horse_name"], r["popularity"], r["win_odds"]) for r in podium_sorted],
        "finish_all": actual_finish,  # {馬番: 着順} 全馬分
        "axis": {
            "number": axis["number"],
            "name": axis["horse_name"],
            "score": axis["score"],
            "prob_win": axis["prob_win"],
            "actual_finish": axis_actual,
            "in_top3": axis_in_top3,
        },
        "top3_score_in_podium": top3_in_podium,
        "ev_picks": [
            {
                "number": r["number"], "name": r["horse_name"],
                "ev_est": r["ev"], "est_odds": r["win_odds_est"],
                "actual_odds": actual_odds.get(r["number"]),
                "actual_finish": actual_finish.get(r["number"], 99),
                "hit": r["number"] == win_num,
            } for r in ev_picks
        ],
        "ev_summary": {
            "n_bets": ev_total, "n_hits": len(ev_hits),
            "cost": ev_cost, "payout": ev_payout,
            "roi_pct": round(ev_payout / ev_cost * 100, 1) if ev_cost else None,
        },
        "triple_summary": {
            "n_combos": n_combos, "cost": triple_cost,
            "hit": bool(triple_hit), "payout": triple_payout,
            "roi_pct": round(triple_payout / triple_cost * 100, 1) if triple_cost else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--force", action="store_true", help="結果を強制再取得")
    args = ap.parse_args()

    pred_path = ROOT / "data/oi/predictions" / f"{args.date}.json"
    if not pred_path.exists():
        print(f"予想ファイルがありません: {pred_path}", file=sys.stderr); sys.exit(1)
    preds = json.loads(pred_path.read_text())

    evals = []
    skipped = []
    for pred in preds:
        rid = pred["race_id"]
        result = fetch_or_load_result(rid, force=args.force)
        if not result:
            skipped.append(pred["race_no"])
            continue
        evals.append(evaluate_race(pred, result))

    out_dir = ROOT / "data/oi/evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.date}.json"
    out_path.write_text(json.dumps(evals, ensure_ascii=False, indent=2))

    # 表示
    print(f"\n{'='*92}")
    print(f"  大井 {args.date} 評価  確定:{len(evals)}R  未確定:{len(skipped)}R {skipped if skipped else ''}")
    print(f"{'='*92}\n")

    axis_in_top3 = 0
    axis_win = 0
    top3_score_pod_total = 0
    ev_cost_total = ev_payout_total = ev_n_bets_total = ev_n_hits_total = 0
    tri_cost_total = tri_payout_total = tri_hits = 0

    for ev in evals:
        ax = ev["axis"]
        def _fin_str(f): return "除" if f >= 99 else f"{f}着"
        symbol = "◎" if ax["actual_finish"] == 1 else ("○" if ax["in_top3"] else "✕")
        line = f"{ev['race_no']:>2}R {symbol}  軸{ax['number']:>2}({ax['name'][:10]}) →{_fin_str(ax['actual_finish'])}  "
        line += f"スコア上位3が複勝{ev['top3_score_in_podium']}/3  "
        # 三連単
        ts = ev["triple_summary"]
        line += f"3連単({ts['n_combos']}点) " + ("◎ROI" if ts["hit"] else "✕")
        if ts["hit"]:
            line += f"+{ts['payout']:,}円(投{ts['cost']:,} 回収{ts['roi_pct']:.0f}%)"
        # 単勝EV
        es = ev["ev_summary"]
        if es["n_bets"]:
            roi_str = f"回収{es['payout']/es['cost']*100:.0f}%" if es["cost"] else "-"
            line += f"  EV単勝{es['n_bets']}点 命中{es['n_hits']} 払戻{es['payout']:,}({roi_str})"
        print(line)

        # 着順
        pod = ev["podium"]
        pod_str = " - ".join(f"{p[0]}着{p[1]}番({p[2][:8]}, {p[3]}人気,{p[4]:.1f})" for p in pod)
        print(f"      実着: {pod_str}")
        if es["n_bets"]:
            for p in ev["ev_picks"]:
                hit_mark = "◎" if p["hit"] else "─"
                fin = _fin_str(p["actual_finish"])
                print(f"      EV{p['ev_est']:>5.2f} {hit_mark} {p['number']:>2}番{p['name'][:10]:<10} 推オ{p['est_odds']:>5.1f} → 実{p['actual_odds'] if p['actual_odds'] is not None else '-':>5} {fin}")
        print()

        # 集計
        if ax["actual_finish"] == 1: axis_win += 1
        if ax["in_top3"]: axis_in_top3 += 1
        top3_score_pod_total += ev["top3_score_in_podium"]
        ev_cost_total += es["cost"]; ev_payout_total += es["payout"]
        ev_n_bets_total += es["n_bets"]; ev_n_hits_total += es["n_hits"]
        tri_cost_total += ts["cost"]; tri_payout_total += ts["payout"]
        tri_hits += int(ts["hit"])

    n = len(evals)
    if n:
        print(f"━━━ 集計 ({n}R 確定) ━━━")
        print(f"  軸馬 単勝  : {axis_win}/{n} ({axis_win/n*100:.1f}%)")
        print(f"  軸馬 複勝  : {axis_in_top3}/{n} ({axis_in_top3/n*100:.1f}%)")
        print(f"  スコア上位3が複勝に入った数: 平均 {top3_score_pod_total/n:.2f}/3")
        if ev_n_bets_total:
            ev_roi = f"回収率 {ev_payout_total/ev_cost_total*100:.1f}%"
            print(f"  単勝EV>1.15: {ev_n_hits_total}/{ev_n_bets_total} 命中  投{ev_cost_total:,} 払戻{ev_payout_total:,}円  {ev_roi}")
        tri_roi = f"回収率 {tri_payout_total/tri_cost_total*100:.1f}%" if tri_cost_total else "-"
        print(f"  3連単フォーメーション: {tri_hits}/{n} 的中  投{tri_cost_total:,} 払戻{tri_payout_total:,}円  {tri_roi}")

    print(f"\n→ 保存: {out_path}")


if __name__ == "__main__":
    main()
