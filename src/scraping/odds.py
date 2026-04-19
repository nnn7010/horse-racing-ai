"""全券種のオッズをnetkeibaのAPIから取得するモジュール。"""

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

ODDS_TYPES = {
    "win": 1,       # 単勝・複勝
    "quinella": 3,  # 馬連
    "wide": 4,      # ワイド
    "exacta": 5,    # 馬単
    "trio": 6,      # 三連複
    "trifecta": 7,  # 三連単
}


def fetch_all_odds(race_id: str) -> dict:
    """全券種のオッズを取得する。

    Returns:
        {
            "win": {"01": 4.3, "02": 15.2, ...},
            "place": {"01": [1.6, 2.0], ...},
            "quinella": {"01-02": 4.6, "01-03": 13.9, ...},
            "wide": {"01-02": 30.4, ...},
            "exacta": {"01→02": 11.0, ...},
            "trio": {"01-02-03": 54.0, ...},  # 注: 一部のみ
            "trifecta": {"01→02→03": 53.3, ...},  # 注: 一部のみ
            "_updated": "2026-04-18 15:33:06",
        }
    """
    result = {}

    for bet_type, type_num in ODDS_TYPES.items():
        url = (
            f"https://race.netkeiba.com/api/api_get_jra_odds.html"
            f"?race_id={race_id}&type={type_num}&action=update"
        )
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "horse-racing-ai research bot"},
                timeout=10,
            )
            data = resp.json()
        except Exception as e:
            logger.debug(f"Failed to fetch {bet_type} odds: {e}")
            continue

        status = data.get("status", "")
        if status not in ("result", "middle"):
            continue

        odds_data = data.get("data", {}).get("odds", {})
        if not odds_data:
            continue

        if bet_type == "win":
            # type=1: 単勝(key "1")と複勝(key "2")
            win_odds = odds_data.get("1", {})
            place_odds = odds_data.get("2", {})
            result["win"] = {k: float(v[0]) for k, v in win_odds.items() if v[0]}
            result["place"] = {k: [float(v[0]), float(v[1])] for k, v in place_odds.items() if v[0]}
            result["_updated"] = data.get("data", {}).get("official_datetime", "")

        elif bet_type in ("quinella", "wide"):
            # key構造: {"3": {"0102": ["4.6","0.0","1"], ...}} or {"4": {...}}
            inner = odds_data.get(str(type_num), {})
            parsed = {}
            for key, val in inner.items():
                if len(key) == 4:
                    a, b = key[:2], key[2:]
                    label = f"{a}-{b}"
                    try:
                        parsed[label] = float(val[0])
                    except (ValueError, IndexError):
                        pass
            result[bet_type] = parsed

        elif bet_type == "exacta":
            # 馬単: {"5": {"0102": ["11.0","12.0","11"], ...}}
            inner = odds_data.get(str(type_num), {})
            parsed = {}
            for key, val in inner.items():
                if len(key) == 4:
                    a, b = key[:2], key[2:]
                    label = f"{a}→{b}"
                    try:
                        parsed[label] = float(val[0])
                    except (ValueError, IndexError):
                        pass
            result[bet_type] = parsed

        elif bet_type == "trio":
            # 三連複: {"6": {"0102": ["54.0",...], "0103": ...}}
            # キーは最初の2頭、3頭目は暗黙
            inner = odds_data.get(str(type_num), {})
            parsed = {}
            for key, val in inner.items():
                if len(key) == 4:
                    a, b = key[:2], key[2:]
                    # 三連複は残りの1頭が不明なので、このAPIでは2頭軸の人気順オッズ
                    label = f"{a}-{b}"
                    try:
                        parsed[label] = float(val[0])
                    except (ValueError, IndexError):
                        pass
            result[bet_type] = parsed

        elif bet_type == "trifecta":
            # 三連単: {"7": {"010203": ["53.3",...], ...}}
            inner = odds_data.get(str(type_num), {})
            parsed = {}
            for key, val in inner.items():
                if len(key) == 6:
                    a, b, c = key[:2], key[2:4], key[4:]
                    label = f"{a}→{b}→{c}"
                    try:
                        parsed[label] = float(val[0])
                    except (ValueError, IndexError):
                        pass
            result[bet_type] = parsed

    logger.info(
        f"Odds for {race_id}: "
        f"win={len(result.get('win', {}))} "
        f"quinella={len(result.get('quinella', {}))} "
        f"wide={len(result.get('wide', {}))} "
        f"exacta={len(result.get('exacta', {}))} "
        f"trio={len(result.get('trio', {}))} "
        f"trifecta={len(result.get('trifecta', {}))}"
    )
    return result
