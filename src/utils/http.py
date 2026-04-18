import hashlib
import time
from pathlib import Path

import requests
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

_config_path = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
with open(_config_path) as f:
    _cfg = yaml.safe_load(f)["scraping"]

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


def fetch(url: str, encoding: str = "euc-jp") -> str:
    global _last_request_time

    cached = _cache_path(url)
    if cached.exists():
        return cached.read_text(encoding="utf-8")

    for attempt in range(1, MAX_RETRIES + 1):
        elapsed = time.time() - _last_request_time
        if elapsed < INTERVAL:
            time.sleep(INTERVAL - elapsed)

        try:
            resp = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=TIMEOUT,
            )
            _last_request_time = time.time()
            resp.raise_for_status()
            resp.encoding = encoding
            html = resp.text
            cached.write_text(html, encoding="utf-8")
            logger.info(f"Fetched: {url}")
            return html
        except requests.RequestException as e:
            wait = 2 ** attempt
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {url}: {e}. Retry in {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} retries")
