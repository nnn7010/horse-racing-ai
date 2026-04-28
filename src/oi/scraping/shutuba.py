"""今週・今開催の出馬表（shutuba）パーサ。

URL: https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}
レース当日の予測対象として使う。確定オッズは後で
update_odds.py のような別スクリプトで更新する想定。
"""

import re

from bs4 import BeautifulSoup

from src.oi.scraping.http import fetch
from src.oi.scraping.race import parse_race_id
from src.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_shutuba(race_id: str) -> dict:
    """指定レースの出馬表を取得する。"""
    url = f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}"
    html = fetch(url, encoding="euc-jp")
    soup = BeautifulSoup(html, "lxml")

    meta = parse_race_id(race_id)
    result: dict = {
        "race_id": race_id,
        "date": meta["date"].strftime("%Y%m%d"),
        "race_no": meta["race_no"],
        "entries": [],
    }

    # レース名・条件（result.html とほぼ同構造）
    race_name_tag = soup.select_one(".RaceName")
    result["race_name"] = race_name_tag.get_text(strip=True) if race_name_tag else ""

    race_data_box = soup.select_one(".RaceData01, .RaceList_Item02")
    if race_data_box:
        info_text = race_data_box.get_text(" ", strip=True)
        result["course_info"] = info_text
        if "芝" in info_text:
            result["surface"] = "芝"
        else:
            result["surface"] = "ダート"
        dist_m = re.search(r"(\d{3,4})m", info_text)
        result["distance"] = int(dist_m.group(1)) if dist_m else 0
        cond_m = re.search(r"馬場\s*[:：]?\s*(良|稍重|重|不良)", info_text)
        result["track_condition"] = cond_m.group(1) if cond_m else ""
        weather_m = re.search(r"天候\s*[:：]?\s*(晴|曇|雨|小雨|雪)", info_text)
        result["weather"] = weather_m.group(1) if weather_m else ""

    # 出走馬テーブル: nar.netkeiba は class="RaceTable01 ShutubaTable" で tbody 無し
    shutuba_table = soup.select_one("table.ShutubaTable, table.Shutuba_Table")
    if shutuba_table:
        rows = shutuba_table.select("tbody tr") or shutuba_table.select("tr")
        for tr in rows:
            if tr.select_one("th"):
                continue  # ヘッダ行
            entry = _parse_shutuba_row(tr, race_id)
            if entry:
                result["entries"].append(entry)

    result["num_runners"] = len(result["entries"])
    return result


def _parse_shutuba_row(tr, race_id: str) -> dict | None:
    """nar.netkeiba 出馬表の1行をパース。

    列: 0:枠 1:馬番 2:印 3:馬名 4:性齢 5:斤量 6:騎手 7:厩舎 8:馬体重(増減) 9:予想オッズ 10:人気
    """
    tds = tr.select("td")
    if len(tds) < 9:
        return None

    def _txt(idx: int) -> str:
        return tds[idx].get_text(strip=True) if idx < len(tds) else ""

    entry: dict = {"race_id": race_id}
    entry["bracket"] = int(_txt(0)) if _txt(0).isdigit() else 0
    entry["number"] = int(_txt(1)) if _txt(1).isdigit() else 0

    horse_link = tr.select_one("a[href*='/horse/']")
    if horse_link:
        entry["horse_name"] = horse_link.get_text(strip=True)
        hid = re.search(r"/horse/(\w+)", horse_link.get("href", ""))
        entry["horse_id"] = hid.group(1) if hid else ""
    else:
        entry["horse_name"] = _txt(3)
        entry["horse_id"] = ""

    entry["sex_age"] = _txt(4)
    try:
        entry["impost"] = float(_txt(5))
    except ValueError:
        entry["impost"] = 0.0

    jockey_link = tr.select_one("a[href*='/jockey/']")
    if jockey_link:
        entry["jockey_name"] = jockey_link.get_text(strip=True)
        jid = re.search(r"/jockey/(?:result/)?(\w+)", jockey_link.get("href", ""))
        entry["jockey_id"] = jid.group(1) if jid else ""
    else:
        entry["jockey_name"] = _txt(6)
        entry["jockey_id"] = ""

    trainer_link = tr.select_one("a[href*='/trainer/']")
    if trainer_link:
        entry["trainer_name"] = trainer_link.get_text(strip=True)
        tid = re.search(r"/trainer/(?:result/)?(\w+)", trainer_link.get("href", ""))
        entry["trainer_id"] = tid.group(1) if tid else ""
    else:
        entry["trainer_name"] = _txt(7)
        entry["trainer_id"] = ""

    weight_text = _txt(8)
    wt_m = re.search(r"(\d{3,4})\s*\(\s*([+\-−]?\d+)\s*\)", weight_text)
    if wt_m:
        entry["horse_weight"] = int(wt_m.group(1))
        entry["weight_change"] = int(wt_m.group(2).replace("−", "-"))
    else:
        entry["horse_weight"] = 0
        entry["weight_change"] = 0

    odds_text = _txt(9)
    try:
        entry["win_odds"] = float(odds_text)
    except ValueError:
        entry["win_odds"] = 0.0

    pop_text = _txt(10)
    entry["popularity"] = int(pop_text) if pop_text.isdigit() else 0

    return entry
