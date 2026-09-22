"""
train_model_kakutoku.py
モデルC: 獲得賞金（kakutoku_man）を予測する、クラブ非依存の汎用モデル
（huber回帰 + lambdarankランキングモデルのランクアンサンブル）

背景:
  モデルA/Bは回収率（kaishuu_rate = kakutoku_man / 募集総額 × 100）を
  直接予測するが、この指標はクラブの値付け（募集価格の付け方）に強く
  依存する。experiment_loco.py の leave-one-club-out 検証で、
  「未知のクラブへの汎化性能」という観点では、回収率を直接予測するより
  獲得賞金（馬そのものの実力に近い絶対値）を予測して、募集金額は
  推論時に別途入力する方式の方が一貫して優れていることが確認できた。

  さらに、huber回帰単体とlambdarank（ランキング目的関数）単体を比較した
  ところ、lambdarank単体は平均はやや良いが分散が増える微妙な結果だった。
  しかし2つのモデルはクラブごとの得意不得意が異なっていた（例:
  silkではhuberが-2.2ptに対しlambdarankは+18.7pt）ため、両者のランクを
  平均するアンサンブルを試したところ、5seed×11クラブLOCO検証で
  huber単体(diff=6.8±8.4, spearman=0.053±0.059)を
  上位25%差分・spearmanどちらも上回り、かつ分散もhuber・lambdarank
  両方より小さくなった(diff=7.3±7.6, spearman=0.069±0.069)。
  「平均も分散も同時に改善する」数少ないケースだったため採用した。

学習データ:
  シルク+他クラブ全11クラブ+バヌーシー自身、学習年度(2018-2021)分
  すべてをプールして学習に使う。測尺（体高等）は全クラブ揃っていない
  ため特徴量に含めない（sire/trainer/farm/bms_name/price_man/sex/
  birth_month のみ）。

  バヌーシー自身の53頭（本来のターゲットドメインの実データ）を学習に
  含めるかどうかは experiment_add_banushi.py で5-fold CVにより検証した。
  spearman相関は全6シードで一貫して改善（0.166→0.18〜0.31の範囲、
  平均0.274）したが、上位25%差分はn=53を5分割した各foldが10頭程度と
  少なくノイズが大きく方向が定まらなかった（1.9〜26.3pt、平均13.9pt、
  現行の15.2ptとほぼ同水準）。相関の改善が一貫していたため採用した。

推論時の使い方（1頭ずつの予測にランクアンサンブルを適用する方法）:
  huber回帰の予測は万円単位の絶対値として直接解釈できるが、lambdarankの
  予測は学習データ内での相対スコアでしかなく単体では単位を持たない。
  そこで、学習プール全体に対する各モデルの予測分布（reference_huber_preds /
  reference_rank_scores、この.pklに保存済み）内での「パーセンタイル順位」に
  変換し、2つのパーセンタイル順位を平均してから、huber分布の
  同パーセンタイル値に逆変換することで、万円単位の1つの予測値に戻す。
    huber_pct = パーセンタイル順位(huber予測, reference_huber_preds)
    rank_pct  = パーセンタイル順位(lambdarank予測, reference_rank_scores)
    ensemble_pct = (huber_pct + rank_pct) / 2
    predicted_kakutoku_man = reference_huber_predsのensemble_pct分位点
  predicted_kaishuu_rate = predicted_kakutoku_man / 募集総額(万円) × 100

出力:
  models/lgbm_model_kakutoku.pkl
  models/feature_importance_kakutoku.csv
  data/valid_predictions_kakutoku.csv
"""

import re
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TRAIN_YEARS = [2018, 2019, 2020, 2021]
HOLDOUT_FRAC = 0.15  # 最終モデルの健全性チェック用ランダムホールドアウト（クラブ問わず）


# ──────────────────────────────────────────
# データ読み込み・統合
# ──────────────────────────────────────────

def load_combined():
    silk = pd.read_csv(DATA_DIR / "merged_train.csv", encoding="utf-8-sig")
    silk = silk.copy()
    silk["club_name"] = "silk"
    silk["price_man"] = pd.to_numeric(silk["total_price_man"], errors="coerce")
    if "farm" not in silk.columns:
        silk["farm"] = silk["farm_x"] if "farm_x" in silk.columns else silk.get("farm_y")

    other = pd.read_csv(DATA_DIR / "jisseki_other_all.csv", encoding="utf-8-sig")
    other = other[other["bosyu_year"].isin(TRAIN_YEARS)].copy()
    other["price_man"] = pd.to_numeric(other["price_man"], errors="coerce")
    if "birth_month" not in other.columns:
        other["birth_month"] = np.nan

    bms = pd.read_csv(DATA_DIR / "bms_other_cache.csv", encoding="utf-8-sig")[["horse_id", "bms_name"]]
    other = pd.merge(other, bms, on="horse_id", how="left")
    sire = pd.read_csv(DATA_DIR / "sire_other_cache.csv", encoding="utf-8-sig")[["horse_id", "sire"]]
    other = pd.merge(other, sire, on="horse_id", how="left")

    cols = ["club_name", "horse_name", "sire", "trainer", "farm", "bms_name", "sex", "birth_month",
            "price_man", "kaishuu_rate", "kakutoku_man"]
    dfs = [silk[cols], other[cols]]

    # バヌーシー自身の実データ（本来のターゲットドメイン、53頭・
    # experiment_add_banushi.pyの5-fold CVでspearman相関の一貫した改善を
    # 確認済み）も学習プールに含める
    banushi_path = DATA_DIR / "jisseki_banushi.csv"
    banushi_pedigree_path = DATA_DIR / "banushi_pedigree_cache.csv"
    if banushi_path.exists() and banushi_pedigree_path.exists():
        banushi = pd.read_csv(banushi_path, encoding="utf-8-sig")
        banushi = banushi[banushi["bosyu_year"].isin(TRAIN_YEARS)].copy()
        banushi["price_man"] = pd.to_numeric(banushi["price_man"], errors="coerce")
        banushi["club_name"] = "banushi"
        banushi_pedigree = pd.read_csv(banushi_pedigree_path, encoding="utf-8-sig")[
            ["horse_id", "sire", "bms_name"]
        ]
        banushi = pd.merge(banushi, banushi_pedigree, on="horse_id", how="left")
        for c in cols:
            if c not in banushi.columns:
                banushi[c] = np.nan
        dfs.append(banushi[cols])

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=["kaishuu_rate", "kakutoku_man", "price_man"]).reset_index(drop=True)
    return combined


# ──────────────────────────────────────────
# 特徴量エンジニアリング
# ──────────────────────────────────────────

def extract_birth_month(v):
    if pd.isna(v):
        return -1
    m = re.search(r"(\d+)月", str(v))
    if m:
        return int(m.group(1))
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


def smooth_mean(df, key_col, value_col, global_mean, min_samples=5):
    sub = df.dropna(subset=[key_col, value_col])
    if len(sub) == 0:
        return pd.DataFrame(columns=[key_col, f"{key_col}_smooth_mean", f"{key_col}_count"])
    g = sub.groupby(key_col)[value_col]
    stats = pd.DataFrame({"count": g.count(), "mean": g.mean()})
    stats[f"{key_col}_smooth_mean"] = (
        (stats["count"] * stats["mean"] + min_samples * global_mean) / (stats["count"] + min_samples)
    )
    stats[f"{key_col}_count"] = stats["count"]
    return stats[[f"{key_col}_smooth_mean", f"{key_col}_count"]].reset_index()


def build_aggs(train_df):
    # 集計はkakutoku_man（円）ではなくkaishuu_rate（比率）で行う。
    # kakutoku_manは牧場・厩舎の規模や馬の頭数構成によって桁が大きく振れやすく、
    # smooth_meanが特定の大口生産者に強く引っ張られるのを避けるため。
    aggs = {}
    for col in ["sire", "trainer", "farm", "bms_name"]:
        gm = train_df["kaishuu_rate"].mean()
        result = smooth_mean(train_df, col, "kaishuu_rate", gm)
        if len(result) > 0:
            aggs[col] = result
    return aggs


def add_features(df, aggs):
    df = df.copy()
    df["birth_month"] = df["birth_month"].apply(extract_birth_month)
    df["sex_num"] = df["sex"].map({"牡": 0, "牝": 1, "セ": 2}).fillna(-1).astype(int)

    for col, stats_df in aggs.items():
        sc = stats_df.copy()
        df[col] = df[col].astype(object)
        sc[col] = sc[col].astype(object)
        df = pd.merge(df, sc, on=col, how="left")

    for col in df.columns:
        if col.endswith("_smooth_mean"):
            df[col] = df[col].fillna(df[col].median())
        elif col.endswith("_count"):
            df[col] = df[col].fillna(0)

    return df.reset_index(drop=True)


FEATURE_BASE = ["sex_num", "birth_month", "price_man"]


def get_feature_cols(df):
    agg = [c for c in df.columns if c.endswith(("_smooth_mean", "_count"))]
    return [c for c in FEATURE_BASE + agg if c in df.columns]


def make_relevance(kaishuu_rate):
    """lambdarank用の3段階relevance grade（<100%/100-200%/200%超）"""
    bins = [-1, 100, 200, 1e9]
    return pd.cut(kaishuu_rate, bins=bins, labels=[0, 1, 2]).astype(int)


# ──────────────────────────────────────────
# モデル学習（huber回帰 + lambdarank のランクアンサンブル）
# ──────────────────────────────────────────

def train_huber(X_train, y_train, random_state_override=None):
    model = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=15,
        min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0,
        random_state=42 if random_state_override is None else random_state_override,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_lambdarank(X_train, relevance, random_state_override=None):
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=15,
        min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0,
        random_state=42 if random_state_override is None else random_state_override,
        verbose=-1,
    )
    model.fit(X_train, relevance, group=[len(X_train)])
    return model


def percentile_of(value, reference):
    return float((np.asarray(reference) < value).mean())


def ensemble_predict_one(huber_pred, rank_pred, ref_huber_preds, ref_rank_scores):
    """1頭ぶんのhuber予測・lambdarankスコアを、学習プール内でのパーセンタイル
    順位に変換して平均し、huber分布の同パーセンタイル値に逆変換して
    万円単位の1つの予測値に戻す"""
    huber_pct = percentile_of(huber_pred, ref_huber_preds)
    rank_pct  = percentile_of(rank_pred, ref_rank_scores)
    ensemble_pct = (huber_pct + rank_pct) / 2
    return float(np.quantile(ref_huber_preds, ensemble_pct))


def ensemble_score_batch(huber_preds, rank_preds):
    """複数頭を一度に評価するとき用: 2つのモデルのパーセンタイル順位の平均を
    そのままランキングスコアとして使う（バッチ内で完結するので逆変換不要）"""
    from scipy.stats import rankdata
    return rankdata(huber_preds) / len(huber_preds) + rankdata(rank_preds) / len(rank_preds)


def evaluate(huber_model, rank_model, X_val, df_val):
    huber_pred = huber_model.predict(X_val)
    rank_pred  = rank_model.predict(X_val)
    ens_score  = ensemble_score_batch(huber_pred, rank_pred)

    df_eval = df_val[["horse_name", "club_name", "kaishuu_rate", "kakutoku_man", "price_man"]].copy().reset_index(drop=True)
    df_eval["pred_kakutoku_man"] = huber_pred
    df_eval["ensemble_score"]    = ens_score

    thr      = df_eval["ensemble_score"].quantile(0.75)
    top25    = df_eval[df_eval["ensemble_score"] >= thr]
    all_avg  = df_eval["kaishuu_rate"].mean()
    t25_avg  = top25["kaishuu_rate"].mean()
    t25_o200 = (top25["kaishuu_rate"] >= 200).mean()

    print(f"全頭平均回収率    : {all_avg:.1f}%")
    print(f"上位25%平均回収率 : {t25_avg:.1f}%  (差分 {t25_avg - all_avg:+.1f}%)")
    print(f"上位25%の200%超率 : {t25_o200:.1%}")

    return df_eval, t25_avg, t25_o200


def save_importance(model, feature_cols, suffix=""):
    fi = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(MODEL_DIR / f"feature_importance_kakutoku{suffix}.csv", index=False, encoding="utf-8-sig")
    print(f"\n--- 特徴量重要度 TOP10{suffix} ---")
    print(fi.head(10).to_string(index=False))
    return fi


# ──────────────────────────────────────────
# メイン
# ──────────────────────────────────────────

def main():
    print("=== データ読み込み・統合（全11クラブ） ===")
    combined = load_combined()
    print(f"合計 {len(combined)} 頭")
    print(combined.groupby("club_name").size().sort_values(ascending=False))

    # ランダムホールドアウト（クラブを問わず15%）で最終モデルの健全性を確認
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(combined))
    n_holdout = int(len(combined) * HOLDOUT_FRAC)
    holdout_idx, train_idx = idx[:n_holdout], idx[n_holdout:]
    train_df   = combined.iloc[train_idx].reset_index(drop=True)
    holdout_df = combined.iloc[holdout_idx].reset_index(drop=True)
    print(f"\n学習: {len(train_df)} 頭 / ホールドアウト検証: {len(holdout_df)} 頭（クラブ横断ランダム分割）")

    aggs = build_aggs(train_df)
    for col, stats_df in aggs.items():
        print(f"  {col}: {len(stats_df)} ユニーク")

    tr = add_features(train_df, aggs)
    ho = add_features(holdout_df, aggs)
    feature_cols = get_feature_cols(tr)
    print(f"\n特徴量 ({len(feature_cols)}個): {feature_cols}")

    X_tr = tr[feature_cols].fillna(-1)
    y_tr = tr["kakutoku_man"]
    X_ho = ho[feature_cols].fillna(-1)

    y_relevance = make_relevance(tr["kaishuu_rate"])

    print("\n=== 全学習データで最終モデルを学習（huber回帰 + lambdarank） ===")
    huber_model = train_huber(X_tr, y_tr)
    rank_model  = train_lambdarank(X_tr, y_relevance)

    # 推論時に1頭ずつパーセンタイル変換するための参照分布（学習プール全体での予測）
    ref_huber_preds = huber_model.predict(X_tr)
    ref_rank_scores = rank_model.predict(X_tr)

    print("\n=== ホールドアウト検証（クラブ横断ランダム15%・アンサンブル） ===")
    df_eval, t25_avg, t25_o200 = evaluate(huber_model, rank_model, X_ho, ho)

    print("\n=== 安定性チェック（5seed平均・huber単体 vs アンサンブル） ===")
    huber_diffs, ens_diffs = [], []
    for seed in range(5):
        hm = train_huber(X_tr, y_tr, random_state_override=seed)
        rm = train_lambdarank(X_tr, y_relevance, random_state_override=seed)
        hp, rp = hm.predict(X_ho), rm.predict(X_ho)
        ho_capped = ho["kaishuu_rate"].clip(upper=200)
        for score, bucket in [(hp, huber_diffs), (ensemble_score_batch(hp, rp), ens_diffs)]:
            thr = np.quantile(score, 0.75)
            top25 = ho_capped[score >= thr]
            bucket.append(top25.mean() - ho_capped.mean())
    print(f"  huber単体   : 上位25%差分={np.mean(huber_diffs):.1f}±{np.std(huber_diffs):.1f}")
    print(f"  アンサンブル: 上位25%差分={np.mean(ens_diffs):.1f}±{np.std(ens_diffs):.1f}")
    print("  (参考: experiment_loco.pyのLOCO検証 huber diff=6.8±8.4, アンサンブル diff=7.3±7.6)")

    save_importance(huber_model, feature_cols, suffix="_huber")

    joblib.dump({
        "huber_model":     huber_model,
        "rank_model":      rank_model,
        "ref_huber_preds": ref_huber_preds,
        "ref_rank_scores": ref_rank_scores,
        "feature_cols":    feature_cols,
        "aggs":            aggs,
        "model_type":      "kakutoku_ensemble",
    }, MODEL_DIR / "lgbm_model_kakutoku.pkl")
    print("\nモデル保存: models/lgbm_model_kakutoku.pkl")

    df_eval.to_csv(DATA_DIR / "valid_predictions_kakutoku.csv", index=False, encoding="utf-8-sig")
    print("予測結果保存: data/valid_predictions_kakutoku.csv")


if __name__ == "__main__":
    main()
