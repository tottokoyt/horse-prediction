"""
train_model_v6.py
v5モデルAの改良版

v5モデルAからの変更点:
  - sire/trainer/farm の smooth_mean を
    merged_train.csv（シルク）+ jisseki_other_all.csv（他10クラブ）
    の結合データから計算（サンプル数増でスムージング精度向上）
  - 学習・検証はシルクのみ（変更なし）
  - 測尺特徴量（height/chest/cannon/weight）はそのまま使う（変更なし）

出力:
  models/lgbm_model_A_v6.pkl
  models/feature_importance_A_v6.csv
  data/valid_predictions_v6.csv
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

# v5モデルA比較値
V5A_CV_AUC     = 0.978
V5A_TOP25_AVG  = 84.4
V5A_TOP25_O200 = 0.051


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
    return df


def add_target(df):
    df = df.copy()
    df = df.dropna(subset=["kaishuu_rate"]).reset_index(drop=True)
    df["target"] = pd.cut(df["kaishuu_rate"], bins=BINS, labels=LABELS).astype(int)
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


def build_aggs(pool):
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
    fi.to_csv(MODEL_DIR / "feature_importance_A_v6.csv", index=False, encoding="utf-8-sig")
    print("\n--- 特徴量重要度 TOP10 ---")
    print(fi.head(10).to_string(index=False))


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

    print(f"シルク 学習:{len(silk_train)} / 検証:{len(silk_valid)}")
    print(f"他クラブ(学習年度のみ): {len(other_train)} 頭")

    # ── 集計プール: silk + other（学習年度分） ──
    pool = pd.concat([
        silk_train[["sire", "trainer", "farm", "kaishuu_rate"]],
        other_train[["sire", "trainer", "farm", "kaishuu_rate"]],
    ], ignore_index=True)
    print(f"\n=== 集計プール: {len(pool)} 頭 (silk {len(silk_train)} + other {len(other_train)}) ===")
    aggs = build_aggs(pool)

    # ── 特徴量追加（学習・検証ともシルクのみ） ──
    print("\n=== 特徴量追加 ===")
    tr = add_features(silk_train, aggs)
    vl = add_features(silk_valid, aggs)

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

    # ── 保存 ──
    joblib.dump({
        "model":        model,
        "feature_cols": feature_cols,
        "aggs":         aggs,
        "bins":         BINS,
        "labels":       LABELS,
        "label_names":  LABEL_NAMES,
    }, MODEL_DIR / "lgbm_model_A_v6.pkl")
    print("\nモデル保存: models/lgbm_model_A_v6.pkl")

    df_eval.to_csv(DATA_DIR / "valid_predictions_v6.csv", index=False, encoding="utf-8-sig")
    print("予測結果保存: data/valid_predictions_v6.csv")

    # ── v5モデルA vs v6 比較 ──
    print("\n" + "="*60)
    print("=== v5モデルA vs v6 比較 ===")
    print("="*60)
    print(f"{'指標':<28} {'v5モデルA':>10} {'v6':>10} {'差分':>10}")
    print("-"*60)
    print(f"{'CV AUC':<28} {V5A_CV_AUC:>10.3f} {cv_auc:>10.3f} {cv_auc - V5A_CV_AUC:>+10.3f}")
    print(f"{'上位25%平均回収率 (%)':<28} {V5A_TOP25_AVG:>10.1f} {t25_avg:>10.1f} {t25_avg - V5A_TOP25_AVG:>+10.1f}")
    print(f"{'上位25%の200%超率':<28} {V5A_TOP25_O200:>10.1%} {t25_o200:>10.1%} {t25_o200 - V5A_TOP25_O200:>+10.1%}")
    print("="*60)


if __name__ == "__main__":
    main()
