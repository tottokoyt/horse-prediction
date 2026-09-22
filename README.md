# シルクホースクラブ データ収集・モデル学習スクリプト

## 実行順序

```bash
# 0. ライブラリインストール
pip install -r requirements.txt

# ── シルク単独データ ──────────────────────────────────────

# 1. 募集データ収集 (公式PDF・ブログからスクレイピング)
python collect_bosyu.py
# -> data/bosyu_2018.csv 〜 data/bosyu_2023.csv, data/bosyu_all.csv

# 2. 実績データ収集 (umadb.comからスクレイピング)
python collect_jisseki.py
# -> data/jisseki_2018.csv 〜 data/jisseki_2023.csv, data/jisseki_all.csv

# 3. 母馬名を取得して募集データと正しくマッチング
python fetch_mother_name.py
# -> data/mother_cache.csv、merged_*.csv を再出力

# 4. 母父（BMS）を取得
python fetch_bms.py
# -> data/bms_cache.csv、merged_*.csv に bms_name 列を追加

# 5. データ結合
python merge.py
# -> data/merged_train.csv  (学習用: 2018〜2021年募集)
# -> data/merged_valid.csv  (検証用: 2022〜2023年募集)
# -> data/merged_all.csv

# ── 他クラブデータ（集計プール拡張用） ──────────────────────

# 6. 他クラブ（carrot/sunday/shadai等10クラブ）の実績データ収集
python collect_jisseki_other.py
# -> data/jisseki_other_all.csv（sire/bms_name 列なし）

# 7. 他クラブ・学習年度(2018〜2021)分の母父を取得
python fetch_bms_other.py
# 8. 他クラブ・検証年度(2022〜2023)分の母父を取得
python fetch_bms_other_remaining.py
# -> data/bms_other_cache.csv（jisseki_other_all.csv 全頭ぶん）

# 9. 他クラブ全頭の父馬（sire）を取得（配合ニック特徴量の集計プール拡張用）
python fetch_sire_other.py
# -> data/sire_other_cache.csv
# umadb.com の robots.txt (Crawl-delay: 30) に従い30秒間隔でアクセスするため
# 対象3,478頭で約29時間かかる。途中で止まっても再実行すれば続きから再開する。

# ── モデル学習 ────────────────────────────────────────────

# 10. モデル学習（最新版）
python train_model_v9.py     # モデルA（測尺あり）: BMS + 配合ニック(父×母父)
python train_model_B_v7.py   # モデルB（測尺なし）: BMS 追加版
# -> models/lgbm_model_A_v9.pkl, models/lgbm_model_B_v7.pkl

# ── Webアプリ ─────────────────────────────────────────────

# 11. FastAPIサーバー起動
uvicorn app.main:app --reload
# app/predictor.py が読み込むモデルを指定（現在: A=v9, B=v7）
# 起動時にSHAP（shap.TreeExplainer）を初期化するため、初回リクエストが
# 返るまで（numbaのJITコンパイルで）15〜20秒ほどかかる場合があります
```

## 出力CSVの主要カラム

| カラム | 説明 |
|--------|------|
| bosyu_year | 募集年度 |
| horse_name | 馬名 |
| sire | 父馬名 |
| bms_name | 母父（BMS）名 |
| sex | 性別 (牡/牝/セ) |
| birth_month | 生月日 |
| total_price_man | 募集総額 (万円) |
| price_per_kuchi | 一口価格 (円) |
| trainer | 預託予定厩舎 |
| height | 体高 (cm) |
| chest | 胸囲 (cm) |
| cannon | 管囲 (cm) |
| weight | 馬体重 (kg) |
| kakutoku_man | 獲得賞金 (万円) ※実績 |
| kaishuu_rate | 回収率 (%) ※実績 |
| races | 出走回数 ※実績 |
| wins | 勝利数 ※実績 |
| farm | 生産牧場 |

## モデルの特徴量（v9 / B_v7時点）

sire・trainer・farm・bms_name（母父）ごとの smooth_mean / smooth_over200 / count に加え、
モデルAには `nick`（父×母父の組み合わせ＝配合ニック）を追加。集計プールはシルク単独では
サンプルが薄く過学習しやすいため、他クラブ（jisseki_other_all.csv + bms_other_cache.csv +
sire_other_cache.csv）の学習年度データを合わせて集計している。

## Webアプリの影響要因（ファクター）表示

`app/predictor.py` は各ファクター（父馬・母父・配合ニック・調教師・牧場・測尺等）の
positive/negative判定を、固定閾値のヒューリスティックではなく **SHAP値**
（`shap.TreeExplainer`、200%超クラスの確率への寄与度）に基づいて行っている。
ファクターはSHAP寄与度の絶対値が大きい順に並ぶ。

## モデル改善実験の評価ルール（重要）

上位25%平均回収率のような指標は検証頭数が少なく（シルク検証156頭、200%超は
8頭のみ）、1頭の外れ値やLightGBMの内部乱数（seed）次第で数値が大きく動く。
実際、他クラブ測尺統合の実験（train_model_v10〜v12.py、下記）では単一seed・
生の回収率で「効果あり」に見えた設定が、複数seed平均＋回収率200%キャップで
再検証すると効果なしと判明した。

**今後モデルの改善案を比較するときは、必ず `eval_utils.py` の
`evaluate_stable()` を使い、複数seedの平均±標準偏差で判断すること。**
v9（現行デプロイ中モデル）の正式ベースラインは
`capped上位25%平均回収率 = 56.6 ± 1.4`、`200%超率 = 10.3% ± 0.0%`（5seed平均）。

## 他クラブ測尺統合の実験（保留中）

`train_model_v10.py` 〜 `v12.py` は、Wayback Machine上に残っていた
キャロットクラブ公式アーカイブ（募集時測尺）を `fetch_carrot_scale.py` /
`fetch_mother_other.py` / `merge_carrot_scale.py` で復元し（242頭中227頭が
マッチ）、モデルAの実学習行をシルク単独634頭から861頭に拡張する実験。
club_name分布差補正・他クラブ行のダウンウェイトなど複数の工夫を試したが、
`sweep_weight_capped.py` での複数seed・capped指標による再検証の結果、
**どの設定もシルク単独ベースラインを上回れなかった**（詳細は各スクリプトの
docstring参照）。効果が確認でき次第 `app/predictor.py` に反映する想定だが、
現時点では保留。再開する場合は他クラブ・他年度（tokyo/normandyのアーカイブ等）
でサンプル数を増やす方向を検討する。

## 注意事項

- PDFのレイアウトによって抽出精度が変わる場合があります
- 実行後 data/ ディレクトリの各CSVを確認し、
  抜け・ズレがあれば手動で修正してください
- umadb.comへのアクセスは適度に間隔を空けています（母父/一覧系は2〜3秒、
  個別ページ大量取得系 fetch_sire_other.py は robots.txt に従い30秒）
- 2018年のブログデータは正規表現でパースするため、
  うまく取れない行があれば collect_bosyu.py の pattern を調整してください
- extract_sire（fetch_bms.py）は umadb の父馬リンクを拾うが、
  国内JBIS登録馬は `/research/sire/...`、海外種牡馬（未登録）は
  `/umalist/?t=...` とリンク形式が異なるため、リンク先を問わず
  該当divの最初の `<a>` を拾うようにしている
