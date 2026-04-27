"""大井競馬の開催日とレースID列挙。

nar.netkeiba の月別カレンダーから大井開催日を抽出し、
各開催日のレース一覧から大井(場コード44)のrace_idを取得する。
"""

import re
from datetime import date, timedelta

from bs4 import BeautifulSoup

from src.oi.scraping.http import fetch
from src.utils.logger import get_logger

logger = get_logger(__name__)

OI_PLACE_CODE = "44"


def _months_between(start: date, end: date):
    """startの月からendの月までを順番にyieldする。"""
    current = date(start.year, start.month, 1)
    end_month = date(end.year, end.month, 1)
    while current <= end_month:
        yield current.year, current.month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def fetch_oi_kaisai_dates(start_date: date, end_date: date) -> list[date]:
    """指定期間内の大井開催日を返す。

    nar.netkeiba の月別カレンダーから大井(リンクテキスト or 場コード)を含む日を抽出。
    """
    kaisai_dates: set[date] = set()

    for year, month in _months_between(start_date, end_date):
        url = f"https://nar.netkeiba.com/top/calendar.html?year={year}&month={month}"
        html = fetch(url, encoding="utf-8")
        soup = BeautifulSoup(html, "lxml")

        # カレンダーセル内の大井リンクを探す。リンクhrefにkaisai_dateが含まれ、
        # セルテキストに「大井」が含まれていれば大井開催日。
        for td in soup.select("td"):
            text = td.get_text()
            if "大井" not in text:
                continue
            link = td.select_one("a[href*='kaisai_date=']")
            if not link:
                continue
            m = re.search(r"kaisai_date=(\d{8})", link.get("href", ""))
            if not m:
                continue
            dt_str = m.group(1)
            try:
                dt = date(int(dt_str[:4]), int(dt_str[4:6]), int(dt_str[6:8]))
            except ValueError:
                continue
            if start_date <= dt <= end_date:
                kaisai_dates.add(dt)

    sorted_dates = sorted(kaisai_dates)
    logger.info(f"大井開催日 {start_date}〜{end_date}: {len(sorted_dates)}日")
    return sorted_dates


def fetch_race_ids_for_date(target_date: date) -> list[str]:
    """指定日の大井全レースIDを返す。

    nar.netkeiba の race_list ページから race_id を抽出し、
    場コード44(大井)のもののみフィルタ。
    """
    dt_str = target_date.strftime("%Y%m%d")
    url = f"https://nar.netkeiba.com/top/race_list.html?kaisai_date={dt_str}"
    html = fetch(url, encoding="utf-8")

    race_ids: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"race_id=(\d{12})", html):
        rid = m.group(1)
        if rid in seen:
            continue
        # 場コードは race_id[4:6]
        if rid[4:6] != OI_PLACE_CODE:
            continue
        seen.add(rid)
        race_ids.append(rid)

    race_ids.sort()
    logger.info(f"{target_date}: {len(race_ids)}レース")
    return race_ids


def collect_all_race_ids(start_date: date, end_date: date) -> list[tuple[date, str]]:
    """期間内の全大井race_idを (開催日, race_id) ペアで返す。"""
    kaisai_dates = fetch_oi_kaisai_dates(start_date, end_date)
    pairs: list[tuple[date, str]] = []
    for dt in kaisai_dates:
        for rid in fetch_race_ids_for_date(dt):
            pairs.append((dt, rid))
    logger.info(f"全 race_id: {len(pairs)}件")
    return pairs
