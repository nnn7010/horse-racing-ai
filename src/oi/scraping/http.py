"""大井用HTTPフェッチャ。

JRA版(`src/utils/http.py`)とは別キャッシュディレクトリを使う。
nar.netkeiba はBot対策があるためリアルなUAを送る。
"""

import hashlib
import time
from pathlib import Path

import requests

from src.oi import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_cfg = load_config()["scraping"]
INTERVAL = _cfg["interval"]
USER_AGENT = _cfg["user_agent"]
MAX_RETRIES = _cfg["max_retries"]
CACHE_DIR = Path(_cfg["cache_dir"])
TIMEOUT = _cfg["timeout"]

CACHE_DIR.mkdir(parents=True, exist_ok=True)

_last_request_time = 0.0


def _cache_path(url: str) -> Path:
    h = hashlib.md5(url.encode()).hexdigest()
    return CACHE_DIR / f"{h}.html"


def fetch(url: str, encoding: str = "utf-8", force: bool = False) -> str:
    """URLを取得しキャッシュする。

    nar.netkeiba は utf-8、db.netkeiba は euc-jp。
    """
    global _last_request_time

    cached = _cache_path(url)
    if cached.exists() and not force:
        return cached.read_text(encoding="utf-8")

    for attempt in range(1, MAX_RETRIES + 1):
        elapsed = time.time() - _last_request_time
        if elapsed < INTERVAL:
            time.sleep(INTERVAL - elapsed)

        try:
            resp = requests.get(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "ja,en;q=0.9",
                },
                timeout=TIMEOUT,
            )
            _last_request_time = time.time()
            resp.raise_for_status()
            resp.encoding = encoding
            html = resp.text
            cached.write_text(html, encoding="utf-8")
            logger.info(f"Fetched: {url}")
            return html
        except requests.HTTPError as e:
            _last_request_time = time.time()
            if 400 <= resp.status_code < 500:
                raise RuntimeError(f"Client error {resp.status_code} for {url}") from e
            wait = 2 ** attempt
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {url}: {e}. Retry in {wait}s")
            time.sleep(wait)
        except requests.RequestException as e:
            wait = 2 ** attempt
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {url}: {e}. Retry in {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} retries")
