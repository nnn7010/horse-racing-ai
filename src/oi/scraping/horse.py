"""馬個体ページのスクレイピング。

db.netkeiba.com/horse/{horse_id}/ は JRA・地方共通で、過去成績にJRA出走と
地方出走が時系列で混在する。このため `src/scraping/horse.py` のロジックを
そのまま再利用しつつ、OI用のキャッシュディレクトリで取得する。
"""

import re

from bs4 import BeautifulSoup

from src.oi.scraping.http import fetch  # OI用キャッシュを使うfetch
from src.utils.logger import get_logger

# JRA版のパース関数を借りる（DOM構造は同一なため）
from src.scraping.horse import _parse_pedigree, _parse_past_results

logger = get_logger(__name__)


def fetch_horse_info(horse_id: str) -> dict:
    """馬の血統情報・基本情報・過去成績を取得する。

    過去成績にはJRA出走・地方出走が時系列で混在する。
    場（"札幌","大井",...）と距離・コースから、JRA成績/地方成績を後段で分離する。
    """
    url = f"https://db.netkeiba.com/horse/{horse_id}/"
    html = fetch(url, encoding="euc-jp")
    soup = BeautifulSoup(html, "lxml")

    info: dict = {"horse_id": horse_id}

    name_tag = soup.select_one(".horse_title h1, .horse_name h1")
    info["horse_name"] = name_tag.get_text(strip=True) if name_tag else ""

    # プロフィール: 性齢、生年月日、調教師、馬主、生産者など
    prof_table = soup.select_one("table.db_prof_table")
    if prof_table:
        prof: dict = {}
        for tr in prof_table.select("tr"):
            th = tr.select_one("th")
            td = tr.select_one("td")
            if th and td:
                prof[th.get_text(strip=True)] = td.get_text(" ", strip=True)
        info["profile"] = prof
    else:
        info["profile"] = {}

    # 血統
    ped_url = f"https://db.netkeiba.com/horse/ped/{horse_id}/"
    ped_html = fetch(ped_url, encoding="euc-jp")
    ped_soup = BeautifulSoup(ped_html, "lxml")
    info.update(_parse_pedigree(ped_soup))

    # 過去成績（JRA・地方混在）
    info["past_results"] = _parse_past_results(soup)

    return info


# JRA場名（過去成績の "場" 欄に出る漢字略称）
JRA_PLACE_NAMES = {"札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"}

# 南関東4場
NANKAN_PLACE_NAMES = {"大井", "川崎", "船橋", "浦和"}

# その他主要地方競馬場
OTHER_NAR_PLACES = {
    "門別", "盛岡", "水沢", "金沢", "笠松", "名古屋",
    "園田", "姫路", "高知", "佐賀", "帯広",
}


def classify_past_result(row: dict) -> str:
    """過去成績の1行を JRA / 大井 / 南関他 / 地方他 / 海外 に分類する。"""
    place = row.get("place", "")
    if place in JRA_PLACE_NAMES:
        return "jra"
    if place == "大井":
        return "oi"
    if place in NANKAN_PLACE_NAMES:
        return "nankan_other"
    if place in OTHER_NAR_PLACES:
        return "nar_other"
    if place:
        return "overseas_or_unknown"
    return "unknown"


def split_past_results(past_results: list[dict]) -> dict[str, list[dict]]:
    """過去成績を出走場の系列ごとに分離する。"""
    buckets: dict[str, list[dict]] = {
        "jra": [],
        "oi": [],
        "nankan_other": [],
        "nar_other": [],
        "overseas_or_unknown": [],
        "unknown": [],
    }
    for row in past_results:
        buckets[classify_past_result(row)].append(row)
    return buckets
