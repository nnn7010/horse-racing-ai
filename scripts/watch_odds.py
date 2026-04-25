#!/usr/bin/env python3
"""ローカルでオッズを1分ごとに更新するウォッチャー。

レース当日に起動しておくと、本日のオッズを毎分取得して
自動で git push する。Ctrl+Cで停止。

使い方:
    python3 scripts/watch_odds.py                # 60秒間隔
    python3 scripts/watch_odds.py --interval 30  # 30秒間隔
    python3 scripts/watch_odds.py --no-push      # ローカルのみ更新（pushしない）
"""

import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60, help="更新間隔(秒) デフォルト60")
    parser.add_argument("--no-push", action="store_true", help="git pushしない（ローカルのみ）")
    args = parser.parse_args()

    print(f"オッズウォッチャー開始 ({args.interval}秒間隔)")
    print(f"  push: {'なし' if args.no_push else 'あり'}")
    print(f"  Ctrl+C で停止\n")

    n = 0
    while True:
        n += 1
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] #{n} 更新開始...")

        cmd = ["python3", "scripts/update_odds.py"]
        if not args.no_push:
            cmd.append("--push")

        try:
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
            # 最終行を表示
            last_lines = (result.stdout + result.stderr).strip().split("\n")[-3:]
            for line in last_lines:
                print(f"  {line}")
        except subprocess.TimeoutExpired:
            print(f"  [warn] タイムアウト")
        except Exception as e:
            print(f"  [error] {e}")

        # 次の更新まで待機
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n停止")
            sys.exit(0)


if __name__ == "__main__":
    main()
