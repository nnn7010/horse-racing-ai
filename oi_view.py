"""大井予想・評価ビューア (Streamlit)。

実行:
  streamlit run oi_view.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

BRACKET_COLORS: dict[int, tuple[str, str]] = {
    1: ("#FFFFFF", "#222222"),
    2: ("#111111", "#FFFFFF"),
    3: ("#E8000D", "#FFFFFF"),
    4: ("#0057B7", "#FFFFFF"),
    5: ("#F5D000", "#222222"),
    6: ("#00A94F", "#FFFFFF"),
    7: ("#F47920", "#FFFFFF"),
    8: ("#EF87C0", "#222222"),
}

# バリアント定義: キー → (ファイルサフィックス, 表示ラベル)
VARIANTS: list[tuple[str, str, str]] = [
    ("today_bias",  "",            "当日バイアス"),
    ("prev_bias",   "_prev_bias",  "前日バイアス"),
    ("no_bias",     "_no_bias",    "バイアスなし"),
]

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
    </style>
    """,
    unsafe_allow_html=True,
)


# ── ユーティリティ ──────────────────────────────────────────────────────────────

def list_dates() -> list[str]:
    pred_dir = ROOT / "data/oi/predictions"
    if not pred_dir.exists():
        return []
    stems = {p.stem.split("_")[0] for p in pred_dir.glob("*.json")}
    return sorted(stems, reverse=True)


def load_variant_map(date_str: str) -> dict[str, dict[int, dict]]:
    """利用可能なバリアントを {key: {race_no: pred}} で返す。"""
    pred_dir = ROOT / "data/oi/predictions"
    result: dict[str, dict[int, dict]] = {}
    for key, suffix, _ in VARIANTS:
        fp = pred_dir / f"{date_str}{suffix}.json"
        if fp.exists():
            preds = json.loads(fp.read_text())
            result[key] = {p["race_no"]: p for p in preds}
    return result


def load_evaluations(date_str: str) -> list[dict]:
    fp = ROOT / "data/oi/evaluations" / f"{date_str}.json"
    return json.loads(fp.read_text()) if fp.exists() else []


def fmt_pct(x):
    return "-" if x is None else f"{x*100:.1f}%"


def variant_label(key: str) -> str:
    return next((lbl for k, _, lbl in VARIANTS if k == key), key)


# ── ヘッダ ─────────────────────────────────────────────────────────────────────

dates = list_dates()
if not dates:
    st.error("予想ファイルがありません。`scripts/oi/predict_today_quick.py --date YYYY-MM-DD` を実行してください。")
    st.stop()

col_h1, col_h2 = st.columns([3, 2])
with col_h1:
    st.title("🏇 大井予想ビューア")
with col_h2:
    sel_date = st.selectbox("日付", dates, index=0)

variant_map = load_variant_map(sel_date)
evaluations = load_evaluations(sel_date)
eval_by_race = {e["race_no"]: e for e in evaluations}

if not variant_map:
    st.error("予想ファイルが見つかりません。")
    st.stop()

# 全レースのrace_noをメインファイル(today_bias優先)から取得
base_key = next((k for k in ["today_bias", "prev_bias", "no_bias"] if k in variant_map), None)
all_race_nos = sorted(variant_map[base_key].keys())
available_keys = list(variant_map.keys())
available_labels = [variant_label(k) for k in available_keys]

# ── 集計サマリ ────────────────────────────────────────────────────────────────

if evaluations:
    n = len(evaluations)
    axis_win  = sum(1 for e in evaluations if e["axis"]["actual_finish"] == 1)
    axis_top3 = sum(1 for e in evaluations if e["axis"]["in_top3"])
    ev_cost = sum(e["ev_summary"]["cost"] for e in evaluations)
    ev_pay  = sum(e["ev_summary"]["payout"] for e in evaluations)
    tri_cost = sum(e["triple_summary"]["cost"] for e in evaluations)
    tri_pay  = sum(e["triple_summary"]["payout"] for e in evaluations)
    tri_hits = sum(1 for e in evaluations if e["triple_summary"]["hit"])
    ev_n_bets = sum(e["ev_summary"]["n_bets"] for e in evaluations)
    ev_n_hits = sum(e["ev_summary"]["n_hits"] for e in evaluations)

    st.subheader(f"📊 集計 ({n}/{len(all_race_nos)} R 確定)")
    cols = st.columns(4)
    cols[0].metric("軸馬 単勝", f"{axis_win}/{n}", f"{axis_win/n*100:.0f}%")
    cols[1].metric("軸馬 複勝", f"{axis_top3}/{n}", f"{axis_top3/n*100:.0f}%")
    if ev_cost:
        cols[2].metric(
            "単勝EV>1.15 回収率",
            f"{ev_pay/ev_cost*100:.1f}%",
            f"{ev_n_hits}/{ev_n_bets}命中 投{ev_cost:,}→{ev_pay:,}",
        )
    cols[3].metric(
        "3連単フォーメ 回収率",
        f"{tri_pay/tri_cost*100:.1f}%" if tri_cost else "-",
        f"{tri_hits}/{n}的中 投{tri_cost:,}→{tri_pay:,}",
    )

st.markdown("---")

# ── レース選択 ────────────────────────────────────────────────────────────────

base_preds = list(variant_map[base_key].values())
base_preds.sort(key=lambda x: x["race_no"])
race_no_options = ["全レース"] + [f"{p['race_no']}R {p['race_name']}" for p in base_preds]
sel = st.radio("表示", race_no_options, horizontal=True, label_visibility="collapsed")

target_race_nos = all_race_nos if sel == "全レース" else [int(sel.split("R")[0])]

# ── 各レース表示 ──────────────────────────────────────────────────────────────

for rno in target_race_nos:
    eval_data = eval_by_race.get(rno)

    # このレースで使えるバリアントを確認
    race_available_keys   = [k for k in available_keys if rno in variant_map[k]]
    race_available_labels = [variant_label(k) for k in race_available_keys]

    # バリアント選択トグル (複数ある場合のみ表示)
    if len(race_available_keys) > 1:
        sk = f"v_{sel_date}_{rno}"
        # デフォルト: セッションにない場合は最初のキー
        if sk not in st.session_state:
            st.session_state[sk] = race_available_labels[0]
        sel_label = st.radio(
            f"{rno}R バージョン",
            race_available_labels,
            key=sk,
            horizontal=True,
            label_visibility="collapsed",
        )
        sel_key = race_available_keys[race_available_labels.index(sel_label)]
    else:
        sel_key   = race_available_keys[0]
        sel_label = race_available_labels[0]

    p = variant_map[sel_key][rno]

    cap = p.get("course_capability", {})
    cap_str = (
        f"speed={cap.get('speed_focus',0):.2f}  "
        f"finishing={cap.get('finishing_focus',0):.2f}  "
        f"bracket_bias={cap.get('bracket_bias',0):+.2f}"
    )

    header = f"### {rno}R　{p['race_name']}　{p['distance']}m {p['track']}　({p['n_runners']}頭)"
    if eval_data:
        current_axis = p["rows"][0]["number"]
        finish_all   = {int(k): v for k, v in eval_data.get("finish_all", {}).items()}
        actual_fin   = finish_all.get(current_axis, 99)
        fin_str      = f"{actual_fin}着" if actual_fin < 99 else "着外"
        mark = "✅" if actual_fin == 1 else ("🟡" if actual_fin <= 3 else "❌")
        header += f"　{mark} 軸{current_axis}番→{fin_str}"
    if len(race_available_keys) > 1:
        header += f"　*{sel_label}*"

    st.markdown(header)
    st.caption(f"コース性質: {cap_str}")

    # データフレーム化
    actual_pos: dict[int, int | str] = {}
    if eval_data:
        finish_all_map = {int(k): v for k, v in eval_data.get("finish_all", {}).items()}
        for num, fin in finish_all_map.items():
            actual_pos[num] = fin if fin < 99 else "除"
        # finish_all がない旧データは podium から補完
        if not finish_all_map:
            for f, n_, name, pop, odds in eval_data["podium"]:
                actual_pos[n_] = f

    axis_num = p["rows"][0]["number"]
    partners_pool = sorted(p["rows"][1:], key=lambda x: -(x.get("prob_top3") or 0))[:4]
    partner_nums  = {r["number"] for r in partners_pool}

    rows = []
    for r in p["rows"]:
        n_ = r["number"]
        rows.append({
            "枠": r["bracket"],
            "番": n_,
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
            "JRA(%)": (round(r["jra_top3"] * 100) if r.get("jra_top3") is not None else None) if r.get("jra_n", 0) >= 3 else None,
            "実着": actual_pos.get(n_, "-") if eval_data else "",
        })
    df = pd.DataFrame(rows)

    def style_row(s):
        n_val    = s.get("番")
        b_val    = s.get("枠")
        cols_idx = list(s.index)

        base = (
            "background-color: rgba(255,210,0,0.20)" if n_val == axis_num else
            "background-color: rgba(80,180,250,0.12)" if n_val in partner_nums else
            ""
        )
        styles = [base] * len(s)

        if b_val in BRACKET_COLORS and "枠" in cols_idx:
            bg, fg = BRACKET_COLORS[b_val]
            styles[cols_idx.index("枠")] = f"background-color:{bg};color:{fg};font-weight:700;text-align:center"

        if eval_data and isinstance(s.get("実着"), int) and s["実着"] in (1, 2, 3):
            styles = [st_ + ";font-weight:800" for st_ in styles]

        return styles

    sty = (
        df.style
          .apply(style_row, axis=1)
          .format({
              "スコア": "{:.1f}",
              "想オ": "{:.1f}",
              "EV":     lambda v: f"{v:.2f}" if v is not None else "-",
              "速力":   lambda v: f"{v:+.2f}" if v is not None else "-",
              "末脚":   lambda v: f"{v:+.2f}" if v is not None else "-",
              "地力":   lambda v: f"{v:.2f}"  if v is not None else "-",
              "枠選好": lambda v: f"{v:+.2f}" if v is not None else "-",
              "JRA(%)": lambda v: f"{v:.0f}%" if v is not None else "-",
          })
    )
    st.dataframe(sty, use_container_width=True, hide_index=True, height=min(40 * len(df) + 50, 600))

    # 馬券候補
    ev_picks = [r for r in p["rows"] if r["ev"] and r["ev"] > 1.15]
    axis     = p["rows"][0]
    partners = [r["number"] for r in partners_pool]

    cols2 = st.columns(2)
    with cols2[0]:
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
    with cols2[1]:
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
        pod_lines = [f"{f}着 {n_}番 {name} ({pop}人気, {odds:.1f}倍)"
                     for f, n_, name, pop, odds in eval_data["podium"]]
        st.write(" / ".join(pod_lines))

    st.markdown("---")

st.caption(f"predictions: {ROOT/'data/oi/predictions'}  |  evaluations: {ROOT/'data/oi/evaluations'}")
