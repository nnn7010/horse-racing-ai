# 大井競馬予想AI（地方版）

JRA向けパイプラインと並列に動く、大井競馬専用の期待値予想システム。

## 設計の要点

- **ターゲット**: 勝率(P(1着)) と 3着内率(P(top3)) のマルチタスク予測
- **馬券戦略**: 単勝EV > 1.15 の馬を1着固定軸 → 3着内率上位N頭を相手 → 3連単フォーメーション提案
- **データソース**: nar.netkeiba（メイン）+ db.netkeiba.com 馬個体ページ（JRA成績と統合）
- **JRA区別**: 過去成績を場別に分類 → JRA経験フラグ・JRA直近2年成績を独立特徴量化
- **トラックバイアス**: 過去レースから事後推定（日次）+ 当日結果入力で動的更新
- **UI**: Streamlit（Streamlit Cloud デプロイ対応、スマホブラウザ完結）
- **状態保存**: Supabase（無料枠）or ローカルSQLite

## ディレクトリ

```
horse-racing-ai/
├── configs/oi.yaml                # 大井設定
├── src/oi/
│   ├── scraping/
│   │   ├── http.py               # 大井用キャッシュfetch
│   │   ├── calendar.py           # 開催日・race_id列挙
│   │   ├── race.py               # レース結果パーサ
│   │   ├── shutuba.py            # 出馬表パーサ
│   │   └── horse.py              # 馬個体ページ（JRA成績含む）
│   ├── features/build.py         # 特徴量構築
│   ├── bias/estimator.py         # トラックバイアス事後推定
│   ├── models/
│   │   ├── trainer.py            # 勝率/3着内率LightGBM
│   │   └── predictor.py          # 予測+Plackett-Luce統合
│   └── live/state.py             # Supabase/SQLite 状態管理
├── scripts/oi/
│   ├── 01_fetch_calendar.py      # 開催日とrace_id列挙
│   ├── 02_scrape_results.py      # 過去結果一括取得（約3時間/年）
│   ├── 03_fetch_horses.py        # 馬個体取得
│   ├── 04_compute_bias.py        # 過去日次バイアス計算
│   ├── 05_build_features.py      # 特徴量化
│   ├── 06_train.py               # マルチタスク学習
│   ├── 07_predict.py             # 検証・単発予測
│   └── 08_predict_today.py       # 当日shutuba→予測 (Streamlit用入力生成)
├── oi_app.py                     # Streamlitアプリ
├── docs/oi.md                    # 本ファイル
└── docs/oi_supabase.sql          # Supabaseスキーマ
```

## ローカル実行（初回フル）

スクレイピングは外部キーバサイトに直接アクセスするため、**ユーザーのローカル環境で実行**します（Codespaces/Sandbox環境ではブロックされる場合あり）。

### 1. 依存関係インストール

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. データ取得（初回1回、約3〜4時間）

```bash
# 開催日・race_id 列挙
python scripts/oi/01_fetch_calendar.py

# レース結果一括取得（中断・再開可。すでにあるJSONはスキップ）
python scripts/oi/02_scrape_results.py

# 馬個体ページ取得
python scripts/oi/03_fetch_horses.py
```

`configs/oi.yaml` の `fetch_period` で範囲を変更可能。デフォルトは1年分。
2年・3年に拡張する場合は `start` を遡って同じスクリプトを再実行（差分のみ取得）。

### 3. 特徴量・モデル

```bash
python scripts/oi/04_compute_bias.py    # 日次バイアス
python scripts/oi/05_build_features.py  # 特徴量
python scripts/oi/06_train.py           # 勝率モデル + 3着内率モデル学習
```

### 4. バックテスト

```bash
python scripts/oi/07_predict.py --validate
```

検証期間（`configs/oi.yaml` の `valid_start`〜`valid_end`）で単勝EV閾値超え馬の的中率・回収率が表示されます。

## 当日運用

### 朝（開催前）

```bash
python scripts/oi/08_predict_today.py --date 2026-04-29
```

これで `outputs/oi/today_2026-04-29.json` に全レースの予測が保存されます。

### Streamlit起動

#### ローカル
```bash
streamlit run oi_app.py
```
ブラウザで `http://localhost:8501` を開く。同じLAN内のスマホからは `http://<PCのIP>:8501`。

#### Streamlit Cloud（推奨）
1. GitHub に push（このリポジトリ）
2. https://share.streamlit.io で `oi_app.py` をMain fileとして指定
3. **Secrets** に Supabase 認証を設定:
   ```
   SUPABASE_URL = "https://xxx.supabase.co"
   SUPABASE_KEY = "eyJ..."
   ```
4. デプロイ → スマホからURLでアクセス可能

### 当日の使い方

1. **🎯 予測タブ** で各レース番号を選択 → 軸馬・相手・3連単候補を確認
2. レース終了後、**⌨️ 結果入力タブ** で着順を入力（カンマ区切り例: `5,3,8`）
3. 入力するとバックエンドで自動的に:
   - Supabase or SQLite に結果が保存される
   - 当日トラックバイアスが再推定される
4. **📊 バイアスタブ** で当日の傾向を確認
5. （オプション）後続レースの予測を更新したい場合は、PCで再度 `08_predict_today.py` を実行 → JSONを差し替え

## Supabaseセットアップ（クラウド運用時のみ）

1. https://supabase.com で無料プロジェクトを作成
2. Project Settings → API から `URL` と `anon key` をコピー
3. SQL Editor で `docs/oi_supabase.sql` を実行 → テーブル作成
4. ローカル実行時は `.env` または環境変数で:
   ```bash
   export SUPABASE_URL=https://xxx.supabase.co
   export SUPABASE_KEY=eyJ...
   ```
5. Streamlit Cloud では Secrets として登録

環境変数がない場合は自動的に `data/oi/live/state.db` (SQLite) にフォールバック。

## 馬券戦略の数式

```
EV(単勝) = P(1着) × 単勝オッズ
推奨条件: EV(単勝) > 1.15

軸 = argmax_i EV_i(単勝)
相手 = top-N P(top3) where i != 軸

3連単フォーメーション(軸1着固定):
  P(軸→A→B) = Plackett-Luce(軸=1, A=2, B=3)
  軸→相手×相手 の組合せから確率順にN点購入候補
```

## 制約・注意

- nar.netkeiba は Bot 検知あり。`scraping.interval` を 1.5秒以上に保つこと
- 大井のクラス体系（A1, A2, B1...）は中央と異なる。自動エンコーディングはカテゴリ列扱い
- 障害戦は地方には少ないが、新馬戦相当（2歳新馬・サラ系C2新馬）は除外要検討。`exclude` に追加可
- 当日バイアスはサンプル数が少ない序盤レースでは過剰反応するため `bias.smoothing` で平滑化
