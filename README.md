# 一口馬主 出資検討支援ツール

## このプロジェクトの本来の目的（重要）

**最終的なターゲットはシルクホースクラブではなく、DMMバヌーシー（データが
ほぼ無い若いクラブ）の出資検討。** シルクは単にデータ収集がしやすかった
から学習・検証データとして使っていただけで、「シルクへの予測精度」が
目的ではない。バヌーシー自体のデータが少ないため、他クラブ（シルク含む
11クラブ）のデータをプールして学習データを増やす狙いでこのプロジェクトが
始まった。

この前提を見落として「シルク検証セットでの精度」を物差しにモデル改善を
進めた結果（下記「モデルA/B」およびtrain_model_v10〜v14, train_model_B_v8
などの実験）、他クラブデータを混ぜるとことごとく悪化するという結果が
続いた。しかしこれは「シルク固有のクセを他クラブデータが薄める」ことの
裏返しであり、本当に知りたい「未知のクラブへの汎化性能」には別の検証法が
要ることが `experiment_loco.py` で判明した（詳細は後述「モデルC」節）。

**現状の結論**: 未知クラブへの汎化を狙うなら **モデルC**
（`train_model_kakutoku.py`、獲得賞金の回帰予測＋推論時に募集金額で
回収率換算）を使う。モデルA/Bはシルク固有の予測精度は高いが、
バヌーシーのような未知クラブへの汎化は保証されない。

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
python train_model_v9.py       # モデルA（測尺あり・シルク寄り）: BMS + 配合ニック(父×母父)
python train_model_B_v7.py     # モデルB（測尺なし・シルク寄り）: BMS 追加版
python train_model_kakutoku.py # モデルC（クラブ非依存・未知クラブ向け）: 獲得賞金回帰
# -> models/lgbm_model_A_v9.pkl, models/lgbm_model_B_v7.pkl, models/lgbm_model_kakutoku.pkl

# ── Webアプリ ─────────────────────────────────────────────

# 11. FastAPIサーバー起動
uvicorn app.main:app --reload
# app/predictor.py が読み込むモデルを指定（現在: A=v9, B=v7, C=kakutoku）
# モデルA/Bの予測に加え、常にモデルC（クラブ横断・参考値）の予測も
# 一緒に返す。起動時にSHAP（shap.TreeExplainer）を初期化するため、
# 初回リクエストが返るまで（numbaのJITコンパイルで）15〜20秒ほどかかる
# 場合があります
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

## モデルの特徴量

**モデルA（v9） / モデルB（B_v7）** — シルクの実学習行（Aは634頭、Bは
他クラブ2018-2021分込みで2,893頭）で学習。sire・trainer・farm・bms_name
（母父）ごとの smooth_mean / smooth_over200 / count に加え、モデルAには
`nick`（父×母父の組み合わせ＝配合ニック）を追加。集計プールは他クラブの
学習年度データも合わせて計算している。測尺（体高・胸囲・管囲・体重）は
モデルAのみ使用。3クラス分類（100%未満/100〜200%/200%超）で
回収率（kaishuu_rate = kakutoku_man / 募集総額 × 100、クラブの値付けに
依存する比率）を直接予測する。

**モデルC（kakutoku）** — 全11クラブ（シルク+他10）の学習年度
(2018-2021)分、計2,786頭をプールして学習。特徴量は
性別・生まれ月・募集金額・sire/trainer/farm/bms_nameのsmooth_mean/count
のみ（測尺・配合ニックは使わない。全クラブで揃っている項目に絞ることで
未知クラブへの汎化を優先）。回収率ではなく **獲得賞金（kakutoku_man）を
LightGBM回帰（huber loss）で直接予測**し、推論時に入力された募集金額で
`回収率 = 予測獲得賞金 / 募集金額 × 100` と逆算する。詳細は
`train_model_kakutoku.py` のdocstring参照。

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

**今後モデルA/Bの改善案を比較するときは、必ず `eval_utils.py` の
`evaluate_stable()` を使い、複数seedの平均±標準偏差で判断すること。**
v9（現行デプロイ中モデル）の正式ベースラインは
`capped上位25%平均回収率 = 56.6 ± 1.4`、`200%超率 = 10.3% ± 0.0%`（5seed平均）。

**ただし「シルクへの精度」は、未知クラブ（バヌーシー）への汎化性能の
物差しとしては不適切**（下記「モデルCとLOCO検証」参照）。モデルCや
クラブ横断の一般化に関わる改善案は、必ず `experiment_loco.py` /
`experiment_loco_v2.py` の leave-one-club-out（LOCO）で評価すること。

## モデルCとLOCO検証（未知クラブへの汎化性能）

`experiment_loco.py` は、11クラブ（シルク+他10）を対象に「1クラブを
除いて学習→そのクラブで評価」を全クラブでローテーションする
leave-one-club-out検証。これで「未知のクラブ（バヌーシー）にどれだけ
汎化するか」を今あるデータで疑似的に測れる。この検証で、回収率を
直接予測する分類（①現行方式）より、獲得賞金を回帰予測して募集金額は
推論時に別入力する方式（②）の方が一貫して優れていることが判明した
（全11クラブ平均 capped上位25% 63.8→67.4、クラブ間のブレも±10.6→±8.4）。
これを受けてモデルC（`train_model_kakutoku.py`）を追加した。

**モデルCのLOCO正式ベースライン**: capped上位25%=67.4±8.4、
200%超率=17.2%±4.6%（`experiment_loco_v2.py`のbaseline実行結果）。

`experiment_loco_v2.py` では、シルク基準で「効果なし」と判定された
施策（年度拡張・mother_name・farm_trainer・nick）をLOCO基準でも
再検証したが、**年度拡張(2018-2023)とmother_nameは依然として悪化
（それぞれ capped25=55.6±9.9, 64.5±9.3）、farm_trainer/nickは誤差範囲内
で横ばい**という結果だった。「シルク基準が間違っていた」という発見自体は
モデルCの成功で裏付けられたが、個別の特徴量追加アイデアの当落線までは
覆らなかった。

## ノイズフロア比較ツールとランキング目的関数の検証

バヌーシー実データ検証の教訓を受け、`eval_utils.py` に
`bootstrap_noise_floor()` / `compare_to_noise_floor()` を追加した。
少数サンプルの検証結果は、既知クラブでブートストラップした分布との
パーセンタイル順位で評価する（点推定だけで判断しない）。この検証を
再利用可能にしたのが `validate_banushi.py`（実行するとバヌーシー
53頭の結果を既知クラブのノイズフロアと比較して表示する）。

このツールを使って、モデルCの目的関数をhuber回帰から**ランキング
目的関数（LGBMRanker, lambdarank）**に変えたら改善するか検証した
（`kaishuu_rate`を3段階のrelevance grade: <100%/100-200%/200%超 に
変換）。結果は5seed平均で上位25%差分6.8→7.8pt・spearman0.053→0.069と
平均はわずかに改善したが、クラブ間のブレも±8.4→±10.1、±0.059→±0.084と
同時に増加しており、**このプロジェクトの基準（分散込みで明確に良いと
言えるか）では「改善」と判定できない**。デプロイは見送り、huber回帰の
ままとした。

## バヌーシー実データでの検証（重要な教訓）

LOCO検証は既存11クラブ同士の疑似的な汎化テストに過ぎないため、
本来のターゲットであるDMMバヌーシー自身のデータ（umadbクラブコード
`c123`）を `collect_jisseki_banushi.py` / `fetch_banushi_pedigree.py` で
取得し（89頭、`data/jisseki_banushi.csv` + `data/banushi_pedigree_cache.csv`）、
デプロイ済みモデルCで直接答え合わせした。

**最初の素朴な検証（89頭全部、2018-2023年産混在）は spearman相関 0.02、
上位25%選抜が全体平均を下回るという悪い結果だった。** しかしこれを
そのまま「未知クラブへの汎化失敗」と結論づけるのは早計だった。
`eval_utils.py` 的な複数seed比較だけでなく、**ノイズフロア自体を
ブートストラップで測る**ことで次の2点が判明した:

1. 2022-2023年産（未成熟世代・35頭）を含めていたのが悪化の主因。
   2018-2021年産（学習年度と同じ範囲、53頭、races平均13.2走で
   十分成熟）に絞ると spearman=0.078、上位25%-全体平均差分=+10.0pt
   まで改善する
2. **既知の学習済み11クラブ自体も、n=53で同じ指標を測ると
   平均spearman=0.055・上位25%差分=+5.7ptとかなり低い**（フルサンプルの
   spearmanもcarrot 0.005・silk -0.051・raffian 0.198など軒並み弱い）。
   回収率は個体レベルでは本質的にノイズが大きく、モデルの実力は
   「上位25%選抜」のような集計レベルでしか安定して現れない

バヌーシー(n=53)の実測値を、既知11クラブのn=53ブートストラップ分布と
比べると spearman相関は55.5パーセンタイル、上位25%差分は59.2パーセンタイルで、
**どちらも「既知クラブでの通常のブレの範囲内（やや上振れ）」**という結論に
落ち着いた。単純な点推定だけで「悪化」「改善」を判断すると、この程度の
サンプルサイズでは容易に誤診断する。**今後クラブ単位の少数サンプルで
モデルを評価する際は、同じ指標を既知クラブでブートストラップした
ノイズフロアと比較すること。**

## 他クラブ測尺統合の実験（モデルA向け・保留中）

`train_model_v10.py` 〜 `v12.py` は、Wayback Machine上に残っていた
キャロットクラブ公式アーカイブ（募集時測尺）を `fetch_carrot_scale.py` /
`fetch_mother_other.py` / `merge_carrot_scale.py` で復元し（242頭中227頭が
マッチ）、モデルAの実学習行をシルク単独634頭から861頭に拡張する実験。
club_name分布差補正・他クラブ行のダウンウェイトなど複数の工夫を試したが、
`sweep_weight_capped.py` での複数seed・capped指標による再検証の結果、
**どの設定もシルク単独ベースラインを上回れなかった**（詳細は各スクリプトの
docstring参照）。これはシルク基準での評価であり、モデルA自体が
シルク特化のモデルという位置づけなので、この評価軸自体はモデルAに関しては
妥当。効果が確認でき次第 `app/predictor.py` に反映する想定だが、
現時点では保留。

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
- `shap` はバージョンによって多クラス分類の `shap_values()` の返り値の
  形式が異なる（旧版: クラスごとの配列のlist、新版:
  (n_samples, n_features, n_classes) の3次元配列）。requirements.txtで
  `shap==0.44.1` に固定しているが、アップグレードする場合は
  `app/predictor.py` の `_shap_contributions()` が両形式に対応済みか
  再確認すること（本番Renderでバージョン差により
  "The truth value of an array..." エラーが実際に発生した）
