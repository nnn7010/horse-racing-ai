#!/usr/bin/env python3
"""Generate static HTML from today_predictions.json for GitHub Pages."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
PRED_FILE = ROOT / "data/raw/today_predictions.json"
TRENDS_FILE = ROOT / "data/raw/today_trends.json"
RESULTS_FILE = ROOT / "data/raw/today_results.json"
OUT_FILE = ROOT / "docs/index.html"




STYLE_BADGE = {
    "逃げ": '<span class="sb sb-hana">逃</span>',
    "先行": '<span class="sb sb-sen">先</span>',
    "差し": '<span class="sb sb-sashi">差</span>',
    "追込": '<span class="sb sb-oikomi">追</span>',
}


def render_race(race, race_num):
    surface_icon = "🌿" if race["surface"] == "芝" else "🟤"
    horses_sorted = sorted(race["horses"], key=lambda h: -h["win_prob"])

    rows = ""
    for i, h in enumerate(horses_sorted):
        win_pct = h["win_prob"] * 100
        place_pct = h["place_prob"] * 100
        odds = h.get("win_odds", 0)
        odds_str = f"{odds:.1f}" if odds > 0 else "-"
        style = h.get("running_style", "")
        badge = STYLE_BADGE.get(style, "")
        comment = h.get("comment", "")
        cls = ' class="top"' if i < 3 else ""
        comment_row = f'<tr class="comment-row"><td></td><td colspan="5" class="comment">{comment}</td></tr>' if comment else ""
        rows += f'<tr{cls}><td>{h["number"]}</td><td>{badge}{h["horse_name"]}</td><td>{h["jockey_name"]}</td><td>{odds_str}</td><td>{win_pct:.1f}%</td><td>{place_pct:.1f}%</td></tr>{comment_row}'

    ability_section = ""

    trio_rows = ""
    for i, t in enumerate(race["trio_top5"], 1):
        combo = "-".join(map(str, t["combo"]))
        prob = t["prob"] * 100
        trio_rows += f'<tr><td>{i}</td><td>{combo}</td><td>{prob:.2f}%</td></tr>'

    trifecta_rows = ""
    for i, t in enumerate(race["trifecta_top5"], 1):
        combo = "-".join(map(str, t["combo"]))
        prob = t["prob"] * 100
        trifecta_rows += f'<tr><td>{i}</td><td>{combo}</td><td>{prob:.3f}%</td></tr>'

    meta = f"{surface_icon}{race['surface']}{race['distance']}m"
    if race.get("track_condition"):
        meta += f" {race['track_condition']}"
    meta += f" {race['n_horses']}頭"
    time_str = race.get("start_time", "")

    # 展開予測
    pp = race.get("pace_prediction", {})
    if pp:
        pace = pp.get("pace", "")
        note = pp.get("note", "")
        front = pp.get("front", 0)
        closers = pp.get("closers", 0)
        pace_color = "#ef5350" if pace == "ハイペース" else "#64b5f6" if pace == "スローペース" else "#ffb74d"
        pace_html = (
            f'<div class="pace">'
            f'<span class="pace-label">展開予測</span>'
            f'<span class="pace-val" style="color:{pace_color}">{pace}</span>'
            f'<span class="pace-note">{note}</span>'
            f'<span class="pace-dist">先行系{front}頭 差追{closers}頭</span>'
            f'</div>'
        )
    else:
        pace_html = ""

    return (
        f'<details><summary>'
        f'<b>{race_num}R</b> {race["race_name"]}'
        f'<small> {meta}{" " + time_str if time_str else ""}</small>'
        f'</summary>'
        f'<div class="d">'
        f'{pace_html}'
        f'<table><thead><tr><th>馬番</th><th>馬名</th><th>騎手</th><th>単勝</th><th>1着%</th><th>3着%</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
        f'{ability_section}'
        f'<div class="bt">'
        f'<table><thead><tr><th colspan="3">三連複 TOP5</th></tr></thead><tbody>{trio_rows}</tbody></table>'
        f'<table><thead><tr><th colspan="3">三連単 TOP5</th></tr></thead><tbody>{trifecta_rows}</tbody></table>'
        f'</div></div></details>'
    )


def render_trends(trends: dict, results: dict) -> str:
    """コース傾向セクションのHTMLを生成する。"""
    if not trends:
        return ""

    cards = ""
    for venue, t in trends.items():
        n = t["completed"]
        if n == 0:
            continue

        pw = {(int(k) if str(k).isdigit() else k): int(v) for k, v in t["pop_wins"].items()}
        fav_color = "#00e676" if t["fav_trust"] == "高" else "#ffeb3b" if t["fav_trust"] == "中" else "#ef5350"
        bracket_color = "#64b5f6" if "内枠" in t["bracket_bias"] else "#ffb74d" if "外枠" in t["bracket_bias"] else "#aaa"

        # 直近結果リスト
        recent = ""
        for r in reversed(t["races"][-5:]):
            pop_str = f"{r['winner_pop']}番人気" if r["winner_pop"] <= 18 else "?"
            recent += (
                f'<tr><td>{r["race_name"]}</td>'
                f'<td>{r["winner_num"]}番 {r["winner_name"]}</td>'
                f'<td>{pop_str}</td>'
                f'<td>{r["winner_odds"]:.1f}倍</td></tr>'
            )

        surface_icon = "🌿" if "芝" in venue else "🟤"
        cards += f"""
<div class="tc">
  <div class="tc-h">{surface_icon} {venue} <span class="tc-n">{n}R完了</span></div>
  <div class="tc-row">
    <div class="tc-item">
      <div class="tc-label">枠順傾向</div>
      <div class="tc-val" style="color:{bracket_color}">{t["bracket_bias"]}</div>
      <div class="tc-sub">内{t["inner_wins"]}勝 外{t["outer_wins"]}勝</div>
    </div>
    <div class="tc-item">
      <div class="tc-label">1番人気信頼度</div>
      <div class="tc-val" style="color:{fav_color}">{t["fav_trust"]}</div>
      <div class="tc-sub">{pw.get(1,0)}勝/{n}R ({pw.get(1,0)*100//n if n else 0}%)</div>
    </div>
    <div class="tc-item">
      <div class="tc-label">平均勝ちオッズ</div>
      <div class="tc-val">{t["avg_winner_odds"]}倍</div>
      <div class="tc-sub">2番人気{pw.get(2,0)}勝 3番人気{pw.get(3,0)}勝</div>
    </div>
  </div>
  <table class="tc-tbl"><thead><tr><th>レース</th><th>勝ち馬</th><th>人気</th><th>オッズ</th></tr></thead>
  <tbody>{recent}</tbody></table>
</div>"""

    return f'<div class="trends"><h2>📊 本日の傾向</h2>{cards}</div>'


def generate():
    with open(PRED_FILE, encoding="utf-8") as f:
        data = json.load(f)

    trends = {}
    if TRENDS_FILE.exists():
        with open(TRENDS_FILE, encoding="utf-8") as f:
            trends = json.load(f)

    results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            results = json.load(f)

    import datetime
    date = datetime.date.today().strftime("%Y/%m/%d")
    races = data["races"]

    venues = {}
    for race in races:
        venues.setdefault(race["place_name"], []).append(race)

    trend_html = render_trends(trends, results)

    sections = ""
    for venue, vraces in venues.items():
        blocks = "".join(render_race(r, i + 1) for i, r in enumerate(vraces))
        sections += f'<h2>🏟 {venue}</h2>{blocks}'

    updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>競馬予想 {date}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#121212;color:#e0e0e0;font-family:-apple-system,sans-serif;font-size:14px;padding:10px}}
h1{{color:#90caf9;font-size:1.1em;margin-bottom:10px;border-bottom:1px solid #333;padding-bottom:6px}}
h2{{color:#ce93d8;font-size:.95em;margin:14px 0 6px}}
details{{background:#1e1e1e;border:1px solid #333;border-radius:6px;margin-bottom:5px;overflow:hidden}}
summary{{cursor:pointer;padding:9px 11px;list-style:none;display:flex;align-items:center;gap:8px}}
summary::-webkit-details-marker{{display:none}}
details[open]>summary{{border-bottom:1px solid #333;background:#252525}}
summary b{{background:#1565c0;color:#fff;border-radius:4px;padding:2px 6px;font-size:.82em;white-space:nowrap}}
summary small{{color:#aaa;font-size:.78em;margin-left:auto}}
.d{{padding:8px 10px}}
table{{width:100%;border-collapse:collapse;margin-bottom:6px;font-size:.82em}}
th{{background:#2a2a2a;color:#aaa;padding:4px 5px;text-align:right;white-space:nowrap}}
th:nth-child(2),th:nth-child(3){{text-align:left}}
td{{padding:4px 5px;border-bottom:1px solid #222;text-align:right;white-space:nowrap}}
td:nth-child(2),td:nth-child(3){{text-align:left}}
tr:last-child td{{border-bottom:none}}
tr.top td{{color:#fff;font-weight:bold}}
tr.top td:nth-child(5){{color:#64b5f6}}
tr.top td:nth-child(6){{color:#81c784}}
td:nth-child(5){{color:#546e7a}}
td:nth-child(6){{color:#4caf50}}
.bt{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
@media(max-width:420px){{.bt{{grid-template-columns:1fr}}}}
.upd{{color:#555;font-size:.72em;text-align:right;margin-top:12px}}
.trends{{margin-bottom:14px}}
.tc{{background:#1a1a2e;border:1px solid #334;border-radius:8px;padding:10px;margin-bottom:8px}}
.tc-h{{color:#90caf9;font-weight:bold;font-size:.9em;margin-bottom:8px}}
.tc-n{{color:#666;font-size:.8em;font-weight:normal;margin-left:6px}}
.tc-row{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:8px}}
.tc-item{{background:#111;border-radius:6px;padding:6px 8px;text-align:center}}
.tc-label{{color:#888;font-size:.72em;margin-bottom:2px}}
.tc-val{{font-size:1em;font-weight:bold}}
.tc-sub{{color:#666;font-size:.7em;margin-top:2px}}
.tc-tbl td,.tc-tbl th{{font-size:.75em;padding:3px 4px}}
.tc-tbl th:nth-child(2){{text-align:left}}
.tc-tbl td:nth-child(2){{text-align:left}}
.sb{{display:inline-block;font-size:.68em;font-weight:bold;border-radius:3px;padding:1px 4px;margin-right:3px;vertical-align:middle}}
.sb-hana{{background:#ef5350;color:#fff}}
.sb-sen{{background:#42a5f5;color:#fff}}
.sb-sashi{{background:#66bb6a;color:#fff}}
.sb-oikomi{{background:#ab47bc;color:#fff}}
.comment-row td{{padding:1px 5px 5px;border-bottom:none}}
.comment{{color:#9e9e9e;font-size:.73em;font-style:italic;text-align:left!important}}
.pace{{background:#1a1a2e;border-radius:5px;padding:5px 10px;margin-bottom:7px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
.pace-label{{color:#666;font-size:.72em}}
.pace-val{{font-weight:bold;font-size:.88em}}
.pace-note{{color:#aaa;font-size:.75em}}
.pace-dist{{color:#666;font-size:.72em;margin-left:auto}}
</style>
</head>
<body>
<h1>🏇 競馬予想 {date}</h1>
{trend_html}
{sections}
<p class="upd">更新: {updated}</p>
</body>
</html>"""

    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Generated: {OUT_FILE} ({len(html):,} bytes)")
    print(f"Races: {len(races)} ({', '.join(f'{v}:{len(r)}R' for v, r in venues.items())})")


if __name__ == "__main__":
    generate()
