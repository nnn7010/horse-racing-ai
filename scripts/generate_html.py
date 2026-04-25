#!/usr/bin/env python3
"""Generate static HTML from today_predictions.json for GitHub Pages."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
PRED_FILE = ROOT / "data/raw/today_predictions.json"
OUT_FILE = ROOT / "docs/index.html"


def render_race(race, race_num):
    surface_icon = "🌿" if race["surface"] == "芝" else "🟤"
    horses_sorted = sorted(race["horses"], key=lambda h: -h["win_prob"])

    rows = ""
    for i, h in enumerate(horses_sorted):
        win_pct = h["win_prob"] * 100
        place_pct = h["place_prob"] * 100
        odds = h.get("win_odds", 0)
        odds_str = f"{odds:.1f}" if odds > 0 else "-"
        cls = ' class="top"' if i < 3 else ""
        rows += f'<tr{cls}><td>{h["number"]}</td><td>{h["horse_name"]}</td><td>{h["jockey_name"]}</td><td>{odds_str}</td><td>{win_pct:.1f}%</td><td>{place_pct:.1f}%</td></tr>'

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

    return (
        f'<details><summary>'
        f'<b>{race_num}R</b> {race["race_name"]}'
        f'<small> {meta}{" " + time_str if time_str else ""}</small>'
        f'</summary>'
        f'<div class="d">'
        f'<table><thead><tr><th>馬番</th><th>馬名</th><th>騎手</th><th>単勝</th><th>1着%</th><th>3着%</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
        f'<div class="bt">'
        f'<table><thead><tr><th colspan="3">三連複 TOP5</th></tr></thead><tbody>{trio_rows}</tbody></table>'
        f'<table><thead><tr><th colspan="3">三連単 TOP5</th></tr></thead><tbody>{trifecta_rows}</tbody></table>'
        f'</div></div></details>'
    )


def generate():
    with open(PRED_FILE, encoding="utf-8") as f:
        data = json.load(f)

    import datetime
    date = datetime.date.today().strftime("%Y/%m/%d")
    races = data["races"]

    venues = {}
    for race in races:
        venues.setdefault(race["place_name"], []).append(race)

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
</style>
</head>
<body>
<h1>🏇 競馬予想 {date}</h1>
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
