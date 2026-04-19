"""レース結果の自動取得モジュール。

race.netkeiba.com の結果ページから確定した着順を取得する。
"""

import json
import re

import requests
from bs4 import BeautifulSoup

from src.utils.logger import get_logger

logger = get_logger(__name__)

# キャッシュを使わず直接リクエスト
def fetch_result_live(race_id: str) -> dict | None:
    """レース結果をリアルタイムで取得する（キャッシュなし）。

    Returns:
        {"1st": 馬番, "2nd": 馬番, "3rd": 馬番, "status": "confirmed"} or None
    """
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "horse-racing-ai research bot"},
            timeout=10,
        )
        resp.encoding = "euc-jp"
        html = resp.text
    except Exception as e:
        logger.debug(f"Failed to fetch {race_id}: {e}")
        return None

    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one(".RaceTable01, .ResultTable")
    if not table:
        return None

    top3 = []
    for tr in table.select("tr")[1:]:
        tds = tr.select("td")
        if len(tds) < 4:
            continue
        try:
            pos = int(tds[0].get_text(strip=True))
            num = int(tds[2].get_text(strip=True))
            name = tds[3].get_text(strip=True)
            if pos <= 3:
                top3.append({"pos": pos, "num": num, "name": name})
        except ValueError:
            continue

    if len(top3) >= 3:
        top3.sort(key=lambda x: x["pos"])
        return {
            "1st": top3[0]["num"],
            "2nd": top3[1]["num"],
            "3rd": top3[2]["num"],
            "1st_name": top3[0]["name"],
            "2nd_name": top3[1]["name"],
            "3rd_name": top3[2]["name"],
            "status": "confirmed",
        }

    return None


def check_all_results(race_ids: list[str], known_results: dict) -> dict:
    """未確定のレースだけチェックし、新しい結果があれば返す。

    Args:
        race_ids: チェック対象のレースIDリスト
        known_results: 既に確定済みの結果 {race_id: {...}}

    Returns:
        新しく確定したレースの結果 {race_id: {...}}
    """
    new_results = {}

    for race_id in race_ids:
        if race_id in known_results:
            continue  # 既に確定済み

        result = fetch_result_live(race_id)
        if result:
            new_results[race_id] = result
            logger.info(
                f"Result confirmed: {race_id} "
                f"1着={result['1st']}番{result['1st_name']} "
                f"2着={result['2nd']}番{result['2nd_name']} "
                f"3着={result['3rd']}番{result['3rd_name']}"
            )

    return new_results
