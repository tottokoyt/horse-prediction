"""
train_model_v11.py
v10（他クラブ測尺227頭を実学習行に追加）にクラブ間の分布差補正を加えた版

v10で判明した問題:
  実学習を634頭→861頭に増やしたのに、シルク検証セットでの性能は悪化した
  （上位25%平均回収率 87.0%→73.9%）。クラブごとに牧場構成・価格帯・
  測尺と回収率の関係が異質なまま混ぜたことが原因と推測される。

v10からの変更点:
  - club_name（silk/carrot等）を sire/trainer/farm と同じ要領で
    smooth_mean特徴量化し、モデルに「このクラブは平均的にこのくらいの
    回収率」というオフセットを学習させる（分布差補正）

出力:
  models/lgbm_model_A_v11.pkl
  models/feature_importance_A_v11.csv
  data/valid_predictions_v11.csv
"""

import re
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BINS        = [-1, 100, 200, 99999]
LABELS      = [0, 1, 2]
LABEL_NAMES = {0: "100%未満", 1: "100〜200%", 2: "200%超"}
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
        df["club_name"] = "silk"
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

    scale_path = DATA_DIR / "other_scale_cache.csv"
    if scale_path.exists():
        scale = pd.read_csv(scale_path, encoding="utf-8-sig")[
            ["horse_id", "height", "chest", "cannon", "weight"]
        ]
        df = pd.merge(df, scale, on="horse_id", how="left")
    else:
        for col in ["height", "chest", "cannon", "weight"]:
            df[col] = np.nan
    return df


def add_target(df):
    df = df.copy()
    df = df.dropna(subset=["kaishuu_rate"]).reset_index(drop=True)
    df["target"] = pd.cut(df["kaishuu_rate"], bins=BINS, labels=LABELS).astype(int)
    return df


def add_nick(df):
    df = df.copy()
    has_both = df["sire"].notna() & df["bms_name"].notna()
    df["nick"] = np.where(has_both, df["sire"].astype(str) + "×" + df["bms_name"].astype(str), np.nan)
    return df


# ──────────────────────────────────────────
# 特徴量エンジニアリング
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


def build_aggs(pool, bms_pool, nick_pool, club_pool):
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

    # bms（母父）はシルク＋他クラブ学習データで集計
    bms_global_mean = bms_pool["kaishuu_rate"].mean()
    result = smooth_mean(bms_pool, "bms_name", "kaishuu_rate", bms_global_mean)
    if len(result) > 0:
        n_used = bms_pool["bms_name"].notna().sum()
        aggs["bms_name"] = result
        print(f"  bms_name: {len(result)} ユニーク / {n_used} 頭")

    # nick（父×母父）もシルク＋他クラブ学習データで集計（v9で拡張）
    nick_global_mean = nick_pool["kaishuu_rate"].mean()
    result = smooth_mean(nick_pool, "nick", "kaishuu_rate", nick_global_mean)
    if len(result) > 0:
        n_used = nick_pool["nick"].notna().sum()
        aggs["nick"] = result
        print(f"  nick: {len(result)} ユニーク / {n_used} 頭")

    # club_name（分布差補正・v11で追加）はシルク+他クラブ学習データで集計
    # min_samples を小さくし、クラブごとの平均差をしっかり学習させる
    club_global_mean = club_pool["kaishuu_rate"].mean()
    result = smooth_mean(club_pool, "club_name", "kaishuu_rate", club_global_mean, min_samples=2)
    if len(result) > 0:
        n_used = club_pool["club_name"].notna().sum()
        aggs["club_name"] = result
        print(f"  club_name: {len(result)} ユニーク / {n_used} 頭")

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
# モデル学習
# ──────────────────────────────────────────

def get_class_weight(y):
    counts = y.value_counts()
    total  = len(y)
    return {cls: total / (len(counts) * cnt) for cls, cnt in counts.items()}


def train_lgbm(X_train, y_train, cw):
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        class_weight=cw,
        random_state=42,
        verbose=-1,
        objective="multiclass",
        num_class=3,
    )
    model.fit(X_train, y_train, sample_weight=y_train.map(cw).values)
    return model


def cross_validate_model(X, y, cw, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        m  = train_lgbm(X_tr, y_tr, cw)
        pp = m.predict_proba(X_val)
        try:
            auc = roc_auc_score(y_val, pp, multi_class="ovr", average="macro")
        except Exception:
            auc = float("nan")
        aucs.append(auc)
        print(f"  Fold {fold+1}: AUC={auc:.3f}")
    mean_auc = np.nanmean(aucs)
    print(f"  CV平均AUC: {mean_auc:.3f} ± {np.nanstd(aucs):.3f}")
    return mean_auc


# ──────────────────────────────────────────
# 評価
# ──────────────────────────────────────────

def evaluate(model, X_val, df_val):
    pred  = model.predict(X_val)
    proba = model.predict_proba(X_val)
    y_val = df_val["target"]

    print("\n--- 分類レポート ---")
    print(classification_report(
        y_val, pred,
        target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
        zero_division=0,
    ))

    df_eval = df_val[["horse_name", "bosyu_year", "kaishuu_rate"]].copy().reset_index(drop=True)
    df_eval["pred_class"]  = pred
    df_eval["prob_class2"] = proba[:, 2]

    thr      = df_eval["prob_class2"].quantile(0.75)
    top25    = df_eval[df_eval["prob_class2"] >= thr]
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
    fi.to_csv(MODEL_DIR / "feature_importance_A_v11.csv", index=False, encoding="utf-8-sig")
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
    other_train = add_target(other_train)
    silk_train = add_nick(silk_train)
    silk_valid = add_nick(silk_valid)
    other_train = add_nick(other_train)

    print(f"シルク 学習:{len(silk_train)} / 検証:{len(silk_valid)}")
    print(f"他クラブ(学習年度のみ): {len(other_train)} 頭 (sire有: {other_train['sire'].notna().sum()})")

    # ── 集計プール ──
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
    club_pool = pd.concat([
        silk_train[["club_name", "kaishuu_rate"]],
        other_train[["club_name", "kaishuu_rate"]],
    ], ignore_index=True)
    print(f"\n=== 集計プール: {len(pool)} 頭 (silk {len(silk_train)} + other {len(other_train)}) ===")
    print(f"=== bms集計プール: {len(bms_pool)} 頭 (bms_name有: {bms_pool['bms_name'].notna().sum()}) ===")
    print(f"=== nick集計プール: {len(nick_pool)} 頭 (nick有: {nick_pool['nick'].notna().sum()}, ユニーク: {nick_pool['nick'].nunique()}) ===")
    print(f"=== club集計プール: {len(club_pool)} 頭 (ユニーククラブ: {club_pool['club_name'].nunique()}) ===")
    aggs = build_aggs(pool, bms_pool, nick_pool, club_pool)

    # ── 実学習行: シルク + 測尺が復元できた他クラブ馬（v10の変更点） ──
    other_train_scaled = other_train.dropna(subset=["height", "chest", "cannon", "weight"])
    print(f"\n=== 実学習行に追加する他クラブ馬（測尺あり）: {len(other_train_scaled)} 頭 ===")

    print("\n=== 特徴量追加 ===")
    tr_silk  = add_features(silk_train, aggs)
    tr_other = add_features(other_train_scaled, aggs)
    tr = pd.concat([tr_silk, tr_other], ignore_index=True)
    vl = add_features(silk_valid, aggs)
    print(f"実学習頭数: {len(tr)} 頭 (silk {len(tr_silk)} + other {len(tr_other)})")

    feature_cols = get_feature_cols(tr)
    print(f"特徴量 ({len(feature_cols)}個): {feature_cols}")

    X_tr = tr[feature_cols].fillna(-1)
    y_tr = tr["target"]
    X_vl = vl[feature_cols].fillna(-1)

    cw = get_class_weight(y_tr)
    print(f"\nクラス重み: { {k: round(v,2) for k,v in sorted(cw.items())} }")

    print("\n=== クロスバリデーション (5fold) ===")
    cv_auc = cross_validate_model(X_tr, y_tr, cw)

    print("\n=== 全データで学習 ===")
    model = train_lgbm(X_tr, y_tr, cw)

    print("\n=== 検証セット評価（シルク） ===")
    df_eval, t25_avg, t25_o200 = evaluate(model, X_vl, vl)

    save_importance(model, feature_cols)

    joblib.dump({
        "model":        model,
        "feature_cols": feature_cols,
        "aggs":         aggs,
        "bins":         BINS,
        "labels":       LABELS,
        "label_names":  LABEL_NAMES,
    }, MODEL_DIR / "lgbm_model_A_v11.pkl")
    print("\nモデル保存: models/lgbm_model_A_v11.pkl")

    df_eval.to_csv(DATA_DIR / "valid_predictions_v11.csv", index=False, encoding="utf-8-sig")
    print("予測結果保存: data/valid_predictions_v11.csv")

    print("\n" + "="*60)
    print("=== v10(club補正なし) vs v11(club_name分布差補正あり) 比較 ===")
    print("="*60)
    print(f"{'指標':<28} {'v11':>10}")
    print("-"*60)
    print(f"{'CV AUC':<28} {cv_auc:>10.3f}")
    print(f"{'上位25%平均回収率 (%)':<28} {t25_avg:>10.1f}")
    print(f"{'上位25%の200%超率':<28} {t25_o200:>10.1%}")
    print("="*60)


if __name__ == "__main__":
    main()
