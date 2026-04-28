"""大井競馬予想 Streamlit アプリ。

スマホブラウザから:
  - 当日の全レース予測を確認
  - 各レース終了後に着順を入力 → バイアス再推定 → 後続レース予測を更新
  - 軸馬・相手・3連単フォーメーション候補を確認

実行:
  streamlit run oi_app.py

デプロイ: Streamlit Community Cloud（無料）
状態保存: Supabase（環境変数SUPABASE_URL/KEYあり）or ローカルSQLite
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.oi import load_config
from src.oi.bias.estimator import BiasEstimate, estimate_bias
from src.oi.live import state as live_state

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="大井競馬予想",
    page_icon="🏇",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---- スタイル: スマホ最適化 ----
st.markdown(
    """
    <style>
      .main .block-container { padding-top: 1rem; padding-bottom: 5rem; max-width: 720px; }
      .stButton button { width: 100%; padding: 0.6rem; font-size: 1rem; }
      .stTextInput input, .stNumberInput input { font-size: 1rem; }
      h1, h2, h3 { margin-top: 0.5rem; }
      .race-card { background: #1e1e1e; padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def load_today_predictions(race_date: date) -> dict[int, dict] | None:
    """ローカル outputs/oi/today_{date}.json から当日の初期予測をロードする。

    形式: {"1": {"race_id": "...", "race_name": "...", "runners": [...]}, ...}
    """
    cfg = load_config()
    out_dir = ROOT / cfg["paths"]["outputs"]
    p = out_dir / f"today_{race_date.isoformat()}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _recompute_bias_from_inputs(race_date: date) -> BiasEstimate | None:
    """当日入力済みの結果からバイアスを再推定する。

    注意: 結果入力は着順順の馬番のみ → バラケた特徴量(枠/通過順)はないので、
    厳密なバイアス計算はできない。代わりに当日の prediction_payload に
    枠/前走通過情報があればそれを利用する近似版。
    """
    rows = live_state.get_live_results(race_date)
    if not rows:
        return None

    # 当日の予想ペイロード（出馬表＋特徴量）から枠・脚質を引く
    races: list[dict] = []
    for row in rows:
        pred = live_state.get_latest_prediction(race_date, row["race_no"])
        if not pred:
            continue
        # 結果（着順）と予想ペイロード(all_runners)を突き合わせる
        finish_nums = [int(x) for x in row["finish_order"].split(",") if x.strip()]
        finish_map = {num: pos + 1 for pos, num in enumerate(finish_nums)}
        runners = pred.get("all_runners", [])
        race_results: list[dict] = []
        for r in runners:
            num = int(r["number"])
            race_results.append({
                "finish_position": finish_map.get(num, 0),
                "bracket": int(r.get("bracket", 0) or 0),
                "passing": r.get("passing", ""),
            })
        races.append({"results": race_results, "num_runners": len(race_results)})

    if not races:
        return None
    return estimate_bias(races, race_date.strftime("%Y%m%d"))


# ---- ヘッダ ----

st.title("🏇 大井競馬予想")

cfg = load_config()
default_date = date.today()
race_date = st.date_input("対象日", value=default_date)

predictions = load_today_predictions(race_date)
if predictions is None:
    st.warning(
        f"⚠️ outputs/oi/today_{race_date.isoformat()}.json が見つかりません。\n\n"
        "ローカルで以下を実行してから再アクセスしてください:\n\n"
        "```\npython scripts/oi/08_predict_today.py --date "
        f"{race_date.isoformat()}\n```"
    )

# ---- タブ構成 ----

tab_pred, tab_input, tab_bias = st.tabs(["🎯 予測", "⌨️ 結果入力", "📊 バイアス"])


# ---- 予測タブ ----

with tab_pred:
    if predictions:
        race_no_options = sorted(int(k) for k in predictions.keys())
        race_no = st.selectbox("レース番号", race_no_options, format_func=lambda x: f"{x}R")
        race = predictions[str(race_no)]

        st.subheader(race.get("race_name", f"{race_no}R"))
        st.caption(
            f"{race.get('distance', '?')}m {race.get('surface', '')} | "
            f"頭数 {race.get('num_runners', '?')}"
        )

        # 軸馬
        if "axis" in race:
            ax = race["axis"]
            st.markdown(
                f"### ◎ 軸: {ax['number']}番 {ax['horse_name']}\n\n"
                f"勝率 **{ax['pred_win_prob']*100:.1f}%** / "
                f"3着内率 **{ax.get('pred_top3_prob', 0)*100:.1f}%** / "
                f"単勝オッズ **{ax.get('win_odds', 0):.1f}** / "
                f"単勝EV **{ax.get('win_ev', 0):.2f}**"
            )

        # 相手
        if "partners" in race:
            st.markdown("### 相手候補（3着内率順）")
            df_p = pd.DataFrame(race["partners"])
            if "pred_top3_prob" in df_p:
                df_p["3着内率%"] = (df_p["pred_top3_prob"] * 100).round(1)
            st.dataframe(df_p[["number", "horse_name", "3着内率%", "win_odds"]], hide_index=True)

        # 全頭
        with st.expander("全頭一覧"):
            df_all = pd.DataFrame(race.get("all_runners", []))
            if len(df_all):
                df_all["勝率%"] = (df_all["pred_win_prob"] * 100).round(1)
                df_all["3着内率%"] = (df_all["pred_top3_prob"] * 100).round(1)
                st.dataframe(
                    df_all[["number", "horse_name", "勝率%", "3着内率%", "win_odds", "win_ev"]],
                    hide_index=True,
                )

        # 単勝推奨
        if race.get("win_picks"):
            st.markdown("### 💰 単勝EV推奨")
            for pick in race["win_picks"]:
                st.markdown(
                    f"- **{pick['number']}番 {pick['horse_name']}** | "
                    f"EV={pick['win_ev']:.2f} (勝率{pick['pred_win_prob']*100:.1f}% × オッズ{pick['win_odds']:.1f})"
                )

        # 3連単フォーメーション（軸1着固定）
        if race.get("trifecta_axis_first"):
            st.markdown("### 🎯 3連単（軸1着固定 + 相手2,3着）")
            tri = race["trifecta_axis_first"]
            df_tri = pd.DataFrame(
                [{"組合せ": f"{k[0]}-{k[1]}-{k[2]}", "確率%": round(v * 100, 2)} for k, v in tri.items()]
            ).sort_values("確率%", ascending=False).head(20)
            st.dataframe(df_tri, hide_index=True)
    else:
        st.info("当日予測データを準備すると、ここに各レースの予想が表示されます。")


# ---- 結果入力タブ ----

with tab_input:
    st.markdown("### 終了レースの着順を入力")
    st.caption("3着までの馬番をカンマ区切りで入力（例: 5,3,8）。送信するとバイアスが自動再計算されます。")

    saved = live_state.get_live_results(race_date)
    saved_map = {int(s["race_no"]): s for s in saved}

    if predictions:
        race_no_options = sorted(int(k) for k in predictions.keys())
    else:
        race_no_options = list(range(1, 13))

    cols = st.columns(2)
    for i, rn in enumerate(race_no_options):
        with cols[i % 2]:
            existing = saved_map.get(rn, {}).get("finish_order", "")
            with st.form(key=f"form_r{rn}", clear_on_submit=False):
                v = st.text_input(f"{rn}R 着順", value=existing, placeholder="5,3,8")
                submitted = st.form_submit_button(
                    f"{rn}R 保存" if not existing else f"{rn}R 更新"
                )
                if submitted and v.strip():
                    try:
                        finish = [int(x.strip()) for x in v.split(",") if x.strip()]
                        live_state.upsert_live_result(race_date, rn, finish)
                        st.success(f"{rn}R 保存: {','.join(map(str, finish))}")
                        # バイアス再計算
                        est = _recompute_bias_from_inputs(race_date)
                        if est:
                            live_state.upsert_live_bias(race_date, est.to_dict())
                    except Exception as e:
                        st.error(f"入力エラー: {e}")


# ---- バイアスタブ ----

with tab_bias:
    bias = live_state.get_live_bias(race_date)
    if not bias:
        st.info("結果入力が始まると、ここに当日のバイアスが表示されます。")
    else:
        st.markdown("### 🎚 当日トラックバイアス（事後推定）")
        c1, c2 = st.columns(2)
        c1.metric(
            "内枠⇔外枠（+で内枠有利）",
            f"{bias.get('bias_inner', 0):+.3f}",
            help=f"内枠複勝率 {bias.get('inner_top3_rate', 0)*100:.1f}% vs "
            f"外枠 {bias.get('outer_top3_rate', 0)*100:.1f}%",
        )
        c2.metric(
            "前々⇔差し（+で前有利）",
            f"{bias.get('bias_front', 0):+.3f}",
            help=f"前 {bias.get('front_top3_rate', 0)*100:.1f}% vs "
            f"後 {bias.get('late_top3_rate', 0)*100:.1f}%",
        )
        st.caption(f"サンプル: {bias.get('sample_size', 0)}レース")

        st.markdown("---")
        st.markdown("### 入力済み結果")
        rows = live_state.get_live_results(race_date)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
