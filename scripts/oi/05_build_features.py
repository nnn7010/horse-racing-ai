"""学習用特徴量を構築。

Usage:
  python scripts/oi/05_build_features.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.oi import load_config
from src.oi.features.build import build_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    raw = ROOT / cfg["paths"]["raw"]
    proc = ROOT / cfg["paths"]["processed"]
    proc.mkdir(parents=True, exist_ok=True)

    df = build_features(raw, proc)
    out = proc / "features.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"保存: {out}")


if __name__ == "__main__":
    main()
