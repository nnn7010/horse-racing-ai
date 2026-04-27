"""大井競馬予想AI。

JRA向けパッケージ(`src/`配下のscraping/features/models等)とは独立した
大井（地方競馬・南関東）専用パイプライン。

URL方針:
  - メイン: nar.netkeiba (race_id + 結果/出馬表ページ)
  - 補助: nankankeiba (公式情報の裏取り)
  - 馬個体: db.netkeiba.com/horse/{horse_id}/ をJRA版と共有
"""

from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "oi.yaml"


def load_config() -> dict:
    """大井用設定を読み込む。"""
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)
