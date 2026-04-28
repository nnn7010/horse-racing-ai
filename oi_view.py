"""大井 4/27予想・評価ビューア (Streamlit)。

予想と実結果評価をスマホ/PCブラウザで確認するための画面。

実行:
  streamlit run oi_view.py
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="大井予想ビューア",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .main .block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1100px; }
      h1 { margin-top: 0; }
      .small-table td, .small-table th { padding: 4px 6px; font-size: 0.86rem; }
      .axis-row { background: rgba(255,210,0,0.18) !important; }
      .top4-row { background: rgba(80,180,250,0.10) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def list_dates() -> list[str]:
    pred_dir = ROOT / "data/oi/predictions"
    if not pred_dir.exists(): return []
    return sorted([p.stem for p in pred_dir.glob("*.json")], reverse=True)


def load_predictions(date_str: str) -> list[dict]:
    fp = ROOT / "data/oi/predictions" / f"{date_str}.json"
    return json.loads(fp.read_text()) if fp.exists() else []


def load_evaluations(date_str: str) -> list[dict]:
    fp = ROOT / "data/oi/evaluations" / f"{date_str}.json"
    return json.loads(fp.read_text()) if fp.exists() else []


def fmt_pct(x):
    if x is None: return "-"
    return f"{x*100:.1f}%"


def fmt_num(x, suffix=""):
    if x is None: return "-"
    return f"{x:+.2f}{suffix}" if isinstance(x, (int, float)) and (x < 0 or suffix == "") else f"{x:.2f}{suffix}"


# ───────── ヘッダ ─────────
dates = list_dates()
if not dates:
    st.error("予想ファイルがありません。`scripts/oi/predict_today_quick.py --date YYYY-MM-DD` を実行してください。")
    st.stop()

col_h1, col_h2 = st.columns([3, 2])
with col_h1:
    st.title("🏇 大井予想ビューア")
with col_h2:
    sel_date = st.selectbox("日付", dates, index=0)

predictions = load_predictions(sel_date)
evaluations = load_evaluations(sel_date)
eval_by_race = {e["race_no"]: e for e in evaluations}

# ───────── 集計サマリ ─────────
if evaluations:
    n = len(evaluations)
    axis_win = sum(1 for e in evaluations if e["axis"]["actual_finish"] == 1)
    axis_top3 = sum(1 for e in evaluations if e["axis"]["in_top3"])
    ev_cost = sum(e["ev_summary"]["cost"] for e in evaluations)
    ev_pay = sum(e["ev_summary"]["payout"] for e in evaluations)
    tri_cost = sum(e["triple_summary"]["cost"] for e in evaluations)
    tri_pay = sum(e["triple_summary"]["payout"] for e in evaluations)
    tri_hits = sum(1 for e in evaluations if e["triple_summary"]["hit"])
    ev_n_bets = sum(e["ev_summary"]["n_bets"] for e in evaluations)
    ev_n_hits = sum(e["ev_summary"]["n_hits"] for e in evaluations)

    st.subheader(f"📊 集計 ({n}/{len(predictions)} R 確定)")
    cols = st.columns(4)
    cols[0].metric("軸馬 単勝", f"{axis_win}/{n}", f"{axis_win/n*100:.0f}%")
    cols[1].metric("軸馬 複勝", f"{axis_top3}/{n}", f"{axis_top3/n*100:.0f}%")
    if ev_cost:
        cols[2].metric(
            "単勝EV>1.15 ROI",
            f"{(ev_pay-ev_cost)/ev_cost*100:+.1f}%",
            f"{ev_n_hits}/{ev_n_bets}命中 投{ev_cost:,}→{ev_pay:,}",
        )
    cols[3].metric(
        "3連単フォーメ ROI",
        f"{(tri_pay-tri_cost)/tri_cost*100:+.1f}%" if tri_cost else "-",
        f"{tri_hits}/{n}的中 投{tri_cost:,}→{tri_pay:,}",
    )

st.markdown("---")

# ───────── レース選択（全レース or 個別） ─────────
race_no_options = ["全レース"] + [f"{p['race_no']}R {p['race_name']}" for p in predictions]
sel = st.radio("表示", race_no_options, horizontal=True, label_visibility="collapsed")

if sel == "全レース":
    target_preds = predictions
else:
    sel_no = int(sel.split("R")[0])
    target_preds = [p for p in predictions if p["race_no"] == sel_no]

# ───────── 各レース表示 ─────────
for p in target_preds:
    rno = p["race_no"]
    eval_data = eval_by_race.get(rno)

    cap = p.get("course_capability", {})
    cap_str = (
        f"speed={cap.get('speed_focus',0):.2f}  "
        f"finishing={cap.get('finishing_focus',0):.2f}  "
        f"bracket_bias={cap.get('bracket_bias',0):+.2f}"
    )

    header = f"### {rno}R　{p['race_name']}　{p['distance']}m  {p['track']}　({p['n_runners']}頭)"
    if eval_data:
        ax = eval_data["axis"]
        mark = "✅" if ax["actual_finish"] == 1 else ("🟡" if ax["in_top3"] else "❌")
        header += f"　{mark} 軸{ax['number']}番→{ax['actual_finish']}着"

    st.markdown(header)
    st.caption(f"コース性質: {cap_str}")

    # データフレーム化
    rows = []
    podium_set = set()
    actual_pos = {}
    actual_odds_real = {}
    if eval_data:
        for f, n_, name, pop, odds in eval_data["podium"]:
            podium_set.add(n_)
            actual_pos[n_] = f
            actual_odds_real[n_] = odds

    for r in p["rows"]:
        n_ = r["number"]
        rows.append({
            "番": n_,
            "枠": r["bracket"],
            "馬名": r["horse_name"],
            "騎手": r["jockey_name"],
            "スコア": r["score"],
            "勝率(推)": fmt_pct(r["prob_win"]),
            "複勝率(推)": fmt_pct(r.get("prob_top3")),
            "想オ": r["win_odds_est"],
            "EV": r["ev"] if r["ev"] is not None else None,
            "大井n": r["n_oi"],
            "速力": r["speed_skill"],
            "末脚": r["finishing_skill"],
            "地力": r["power"],
            "枠選好": r["bracket_pref"],
            "JRA(%)": (round(r["jra_top3"]*100) if r.get("jra_top3") is not None else None) if r.get("jra_n",0)>=3 else None,
            "実着": actual_pos.get(n_, "-") if eval_data else "",
        })
    df = pd.DataFrame(rows)

    # 強調: 軸馬(スコア1位) + 相手(複勝率上位4頭、軸除く)
    axis_num = p["rows"][0]["number"]
    partners_pool = sorted(p["rows"][1:], key=lambda x: -(x.get("prob_top3") or 0))[:4]
    partner_nums = {r["number"] for r in partners_pool}

    def style_row(s):
        styles = [""] * len(s)
        n_val = s.get("番")
        if n_val == axis_num:
            styles = ["background-color: rgba(255,210,0,0.20)"] * len(s)
        elif n_val in partner_nums:
            styles = ["background-color: rgba(80,180,250,0.12)"] * len(s)
        if eval_data and isinstance(s.get("実着"), int):
            if s["実着"] in (1, 2, 3):
                styles = [st_ + "; font-weight: 800" for st_ in styles]
        return styles

    sty = (
        df.style
          .apply(style_row, axis=1)
          .format({
              "スコア": "{:.1f}",
              "想オ": "{:.1f}",
              "EV": lambda v: f"{v:.2f}" if v is not None else "-",
              "速力": lambda v: f"{v:+.2f}" if v is not None else "-",
              "末脚": lambda v: f"{v:+.2f}" if v is not None else "-",
              "地力": lambda v: f"{v:.2f}" if v is not None else "-",
              "枠選好": lambda v: f"{v:+.2f}" if v is not None else "-",
              "JRA(%)": lambda v: f"{v:.0f}%" if v is not None else "-",
          })
    )
    st.dataframe(sty, use_container_width=True, hide_index=True, height=min(40 * len(df) + 50, 600))

    # 馬券候補: 軸=スコア1位、相手=複勝率上位4頭(軸を除く)
    ev_picks = [r for r in p["rows"] if r["ev"] and r["ev"] > 1.15]
    axis = p["rows"][0]
    partners = [r["number"] for r in partners_pool]

    cols = st.columns(2)
    with cols[0]:
        if ev_picks:
            st.markdown("**🎯 単勝EV>1.15:**")
            for r in ev_picks:
                hit = ""
                if eval_data:
                    for ep in eval_data["ev_picks"]:
                        if ep["number"] == r["number"]:
                            hit = "✅" if ep["hit"] else "❌"
                            break
                st.write(f"{hit} {r['number']}番 {r['horse_name']} EV={r['ev']:.2f} (想{r['win_odds_est']:.1f}倍)")
    with cols[1]:
        st.markdown(f"**🏁 3連単フォーメ:** 軸{axis['number']}番→相手{','.join(map(str,partners))}")
        st.caption(f"({len(partners)*(len(partners)-1)}通り = {len(partners)*(len(partners)-1)*100}円)")
        if eval_data:
            ts = eval_data["triple_summary"]
            if ts["hit"]:
                st.success(f"✅ 的中 払戻 {ts['payout']:,}円 (ROI {ts['roi_pct']:+.0f}%)")
            else:
                st.error(f"❌ 不的中 (-{ts['cost']:,}円)")

    if eval_data:
        st.markdown("**🏆 実着順:**")
        pod_lines = []
        for f, n_, name, pop, odds in eval_data["podium"]:
            pod_lines.append(f"{f}着 {n_}番 {name} ({pop}人気, {odds:.1f}倍)")
        st.write(" / ".join(pod_lines))

    st.markdown("---")

st.caption(f"データ: predictions={ROOT/'data/oi/predictions'/(sel_date+'.json')}  evaluations={ROOT/'data/oi/evaluations'/(sel_date+'.json')}")
