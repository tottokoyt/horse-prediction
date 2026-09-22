"""
train_model_v13.py
モデルAを3クラス分類から回収率の直接回帰に変更した実験版

背景:
  v9（3クラス分類: 100%未満/100〜200%/200%超）は、検証セット156頭のうち
  200%超がわずか8頭しかなく、この少数派クラスの学習・評価が本質的に
  不安定。分類ではなく回収率そのものを連続値として回帰予測すれば、
  「200%超か否か」の境界だけでなく「どれくらい良さそうか」の
  ランキング情報をフルに使える。

  回収率には600%超などの極端な外れ値が混ざるため、二乗誤差だと
  外れ値1頭に学習が引っ張られる。LightGBMの objective="huber" で
  外れ値に頑健な損失関数を使う。

学習・検証データ、特徴量エンジニアリング（sire/trainer/farm/bms/nick
のsmooth_mean集計、他クラブは集計プールのみ利用）はv9と同一。
実学習行もシルク単独634頭のまま（v10〜v12の教訓通り、他クラブ実データ
混入は現時点で効果が確認できていないため見送り）。

評価は eval_utils.evaluate_stable() を回帰モデル用スコア関数
（regression_score_fn = 予測回収率そのものでランキング）で実行し、
v9の正式ベースライン（capped=56.6±1.4, 200%超率=10.3%±0.0%）と比較する。

出力:
  models/lgbm_model_A_v13.pkl
  models/feature_importance_A_v13.csv
  data/valid_predictions_v13.csv
"""

import re
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

from eval_utils import evaluate_stable, regression_score_fn

warnings.filterwarnings("ignore")

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TRAIN_YEARS = [2018, 2019, 2020, 2021]


# ──────────────────────────────────────────
# データ読み込み
# ──────────────────────────────────────────

def load_silk():
    train = pd.read_csv(DATA_DIR / "merged_train.csv", encoding="utf-8-sig")
    valid = pd.read_csv(DATA_DIR / "merged_valid.csv", encoding="utf-8-sig")
    for df in [train, valid]:
        if "farm" not in df.columns:
            df["farm"] = df["farm_x"] if "farm_x" in df.columns else df.get("farm_y")
        df["price_man"] = pd.to_numeric(df["total_price_man"], errors="coerce")
    return train, valid


def load_other():
    df = pd.read_csv(DATA_DIR / "jisseki_other_all.csv", encoding="utf-8-sig")
    df["price_man"] = pd.to_numeric(df["price_man"], errors="coerce")
    for col in ["birth_month", "sire"]:
        if col not in df.columns:
            df[col] = np.nan

    bms_path = DATA_DIR / "bms_other_cache.csv"
    if bms_path.exists():
        bms = pd.read_csv(bms_path, encoding="utf-8-sig")[["horse_id", "bms_name"]]
        df = pd.merge(df, bms, on="horse_id", how="left")
    else:
        df["bms_name"] = np.nan

    sire_path = DATA_DIR / "sire_other_cache.csv"
    if sire_path.exists():
        sire = pd.read_csv(sire_path, encoding="utf-8-sig")[["horse_id", "sire"]]
        df = df.drop(columns=["sire"]).merge(sire, on="horse_id", how="left")
    return df


def add_target(df):
    df = df.copy()
    df = df.dropna(subset=["kaishuu_rate"]).reset_index(drop=True)
    df["target"] = df["kaishuu_rate"].astype(float)
    return df


def add_nick(df):
    df = df.copy()
    has_both = df["sire"].notna() & df["bms_name"].notna()
    df["nick"] = np.where(has_both, df["sire"].astype(str) + "×" + df["bms_name"].astype(str), np.nan)
    return df


# ──────────────────────────────────────────
# 特徴量エンジニアリング（v9と同一）
# ──────────────────────────────────────────

def extract_birth_month(v):
    if pd.isna(v):
        return -1
    m = re.search(r"(\d+)月", str(v))
    return int(m.group(1)) if m else -1


def smooth_mean(df, key_col, value_col, global_mean, min_samples=5):
    sub = df.dropna(subset=[key_col, value_col])
    if len(sub) == 0:
        return pd.DataFrame(columns=[key_col,
                                      f"{key_col}_smooth_mean",
                                      f"{key_col}_smooth_over200",
                                      f"{key_col}_count"])
    g = sub.groupby(key_col)[value_col]
    stats = pd.DataFrame({"count": g.count(), "mean": g.mean()})
    stats[f"{key_col}_smooth_mean"] = (
        (stats["count"] * stats["mean"] + min_samples * global_mean)
        / (stats["count"] + min_samples)
    )
    stats[f"{key_col}_count"] = stats["count"]
    global_over200 = (sub[value_col] >= 200).mean()
    over200 = (sub[value_col] >= 200).groupby(sub[key_col]).mean()
    stats[f"{key_col}_smooth_over200"] = (
        (stats["count"] * over200 + min_samples * global_over200)
        / (stats["count"] + min_samples)
    )
    return stats[[
        f"{key_col}_smooth_mean",
        f"{key_col}_smooth_over200",
        f"{key_col}_count",
    ]].reset_index()


def build_aggs(pool, bms_pool, nick_pool):
    global_mean = pool["kaishuu_rate"].mean()
    aggs = {}
    for col in ["sire", "trainer", "farm"]:
        if col not in pool.columns:
            continue
        result = smooth_mean(pool, col, "kaishuu_rate", global_mean)
        if len(result) > 0:
            n_used = pool[col].notna().sum()
            aggs[col] = result
            print(f"  {col}: {len(result)} ユニーク / {n_used} 頭")

    bms_global_mean = bms_pool["kaishuu_rate"].mean()
    result = smooth_mean(bms_pool, "bms_name", "kaishuu_rate", bms_global_mean)
    if len(result) > 0:
        n_used = bms_pool["bms_name"].notna().sum()
        aggs["bms_name"] = result
        print(f"  bms_name: {len(result)} ユニーク / {n_used} 頭")

    nick_global_mean = nick_pool["kaishuu_rate"].mean()
    result = smooth_mean(nick_pool, "nick", "kaishuu_rate", nick_global_mean)
    if len(result) > 0:
        n_used = nick_pool["nick"].notna().sum()
        aggs["nick"] = result
        print(f"  nick: {len(result)} ユニーク / {n_used} 頭")

    return aggs


def add_features(df, aggs):
    df = df.copy()
    df["birth_month"] = df["birth_month"].apply(extract_birth_month)
    df["sex_num"] = df["sex"].map({"牡": 0, "牝": 1, "セ": 2}).fillna(-1).astype(int)

    for col, stats_df in aggs.items():
        if col not in df.columns:
            continue
        sc = stats_df.copy()
        df[col] = df[col].astype(object)
        sc[col] = sc[col].astype(object)
        df = pd.merge(df, sc, on=col, how="left")

    for col in df.columns:
        if col.endswith(("_smooth_mean", "_smooth_over200")):
            df[col] = df[col].fillna(df[col].median())
        elif col.endswith("_count"):
            df[col] = df[col].fillna(0)

    return df.reset_index(drop=True)


def get_feature_cols(df):
    base = ["sex_num", "birth_month", "price_man",
            "height", "chest", "cannon", "weight"]
    agg  = [c for c in df.columns
            if c.endswith(("_smooth_mean", "_smooth_over200", "_count"))]
    return [c for c in base + agg if c in df.columns]


# ──────────────────────────────────────────
# モデル学習（回帰・huber loss）
# ──────────────────────────────────────────

def train_lgbm_reg(X_train, y_train, random_state_override=None):
    model = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42 if random_state_override is None else random_state_override,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


# ──────────────────────────────────────────
# 評価
# ──────────────────────────────────────────

def evaluate(model, X_val, df_val):
    pred = model.predict(X_val)

    df_eval = df_val[["horse_name", "bosyu_year", "kaishuu_rate"]].copy().reset_index(drop=True)
    df_eval["pred_kaishuu_rate"] = pred

    thr      = df_eval["pred_kaishuu_rate"].quantile(0.75)
    top25    = df_eval[df_eval["pred_kaishuu_rate"] >= thr]
    all_avg  = df_eval["kaishuu_rate"].mean()
    t25_avg  = top25["kaishuu_rate"].mean()
    t25_o200 = (top25["kaishuu_rate"] >= 200).mean()

    print(f"全頭平均回収率    : {all_avg:.1f}%")
    print(f"上位25%平均回収率 : {t25_avg:.1f}%  (差分 {t25_avg - all_avg:+.1f}%)")
    print(f"上位25%の200%超率 : {t25_o200:.1%}")

    return df_eval, t25_avg, t25_o200


def save_importance(model, feature_cols):
    fi = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(MODEL_DIR / "feature_importance_A_v13.csv", index=False, encoding="utf-8-sig")
    print("\n--- 特徴量重要度 TOP10 ---")
    print(fi.head(10).to_string(index=False))
    return fi


# ──────────────────────────────────────────
# メイン
# ──────────────────────────────────────────

def main():
    print("=== データ読み込み ===")
    silk_train, silk_valid = load_silk()
    other_all = load_other()
    other_train = other_all[other_all["bosyu_year"].isin(TRAIN_YEARS)].copy()

    silk_train = add_target(silk_train)
    silk_valid = add_target(silk_valid)
    silk_train = add_nick(silk_train)
    silk_valid = add_nick(silk_valid)
    other_train = add_nick(other_train)

    print(f"シルク 学習:{len(silk_train)} / 検証:{len(silk_valid)}")
    print(f"他クラブ(集計プールのみ): {len(other_train)} 頭")

    pool = pd.concat([
        silk_train[["sire", "trainer", "farm", "kaishuu_rate"]],
        other_train[["sire", "trainer", "farm", "kaishuu_rate"]],
    ], ignore_index=True)
    bms_pool = pd.concat([
        silk_train[["bms_name", "kaishuu_rate"]],
        other_train[["bms_name", "kaishuu_rate"]],
    ], ignore_index=True)
    nick_pool = pd.concat([
        silk_train[["nick", "kaishuu_rate"]],
        other_train[["nick", "kaishuu_rate"]],
    ], ignore_index=True)
    print(f"\n=== 集計プール: {len(pool)} 頭 ===")
    aggs = build_aggs(pool, bms_pool, nick_pool)

    print("\n=== 特徴量追加 ===")
    tr = add_features(silk_train, aggs)
    vl = add_features(silk_valid, aggs)

    feature_cols = get_feature_cols(tr)
    print(f"特徴量 ({len(feature_cols)}個): {feature_cols}")

    X_tr = tr[feature_cols].fillna(-1)
    y_tr = tr["target"]
    X_vl = vl[feature_cols].fillna(-1)

    print(f"\n目的変数（回収率）分布: mean={y_tr.mean():.1f} median={y_tr.median():.1f} max={y_tr.max():.1f}")

    print("\n=== 全データで学習（huber loss） ===")
    model = train_lgbm_reg(X_tr, y_tr)

    print("\n=== 検証セット評価（シルク） ===")
    df_eval, t25_avg, t25_o200 = evaluate(model, X_vl, vl)

    print("\n=== 安定性チェック（5seed平均・回収率200%キャップ） ===")
    stable = evaluate_stable(
        lambda seed: train_lgbm_reg(X_tr, y_tr, random_state_override=seed),
        X_tr, y_tr, X_vl, vl, label="v13(回帰)", score_fn=regression_score_fn,
    )
    print("  " + stable["summary"])
    print("  v9(分類・正式ベースライン) raw=86.7±1.4  capped=56.6±1.4  200%超率=10.3%±0.0%")

    save_importance(model, feature_cols)

    joblib.dump({
        "model":        model,
        "feature_cols": feature_cols,
        "aggs":         aggs,
        "model_type":   "regression",
    }, MODEL_DIR / "lgbm_model_A_v13.pkl")
    print("\nモデル保存: models/lgbm_model_A_v13.pkl")

    df_eval.to_csv(DATA_DIR / "valid_predictions_v13.csv", index=False, encoding="utf-8-sig")
    print("予測結果保存: data/valid_predictions_v13.csv")


if __name__ == "__main__":
    main()
