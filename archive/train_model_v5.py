"""
train_model_v5.py
モデルA（測尺あり・silk専用）とモデルB（測尺なし・汎用）の2本立て

モデルA:
  学習: merged_train.csv (silk 2018-2021)
  特徴量: sex_num, birth_month, price_man, sire/trainer/farm smooth_mean,
          height, chest, cannon, weight
  検証: merged_valid.csv (silk 2022-2023)

モデルB:
  学習: merged_train.csv + jisseki_other_all.csv (2018-2021産)
  特徴量: sex_num, birth_month, price_man, sire/trainer/farm smooth_mean
  検証: merged_valid.csv (silk) および jisseki_other_all.csv (2022-2023産) で別々評価

出力:
  models/lgbm_model_A.pkl
  models/lgbm_model_B.pkl
  data/valid_predictions_v5.csv
"""

import re
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BINS        = [-1, 100, 200, 99999]
LABELS      = [0, 1, 2]
LABEL_NAMES = {0: "100%未満", 1: "100〜200%", 2: "200%超"}
TRAIN_YEARS = [2018, 2019, 2020, 2021]
VALID_YEARS = [2022, 2023]


# ──────────────────────────────────────────
# データ読み込み
# ──────────────────────────────────────────

def load_silk():
    train = pd.read_csv(DATA_DIR / "merged_train.csv", encoding="utf-8-sig")
    valid = pd.read_csv(DATA_DIR / "merged_valid.csv", encoding="utf-8-sig")
    for df in [train, valid]:
        # farm 列を統一
        if "farm" not in df.columns:
            df["farm"] = df["farm_x"] if "farm_x" in df.columns else df.get("farm_y")
        # price_man に統一（bosyu の総額を使用）
        df["price_man"] = pd.to_numeric(df["total_price_man"], errors="coerce")
    return train, valid


def load_other():
    df = pd.read_csv(DATA_DIR / "jisseki_other_all.csv", encoding="utf-8-sig")
    df["price_man"] = pd.to_numeric(df["price_man"], errors="coerce")
    # silk にない列を NaN で補完
    for col in ["birth_month", "sire", "height", "chest", "cannon", "weight"]:
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


def build_aggs(pool, label=""):
    global_mean = pool["kaishuu_rate"].mean()
    aggs = {}
    for col in ["sire", "trainer", "farm"]:
        if col not in pool.columns:
            continue
        result = smooth_mean(pool, col, "kaishuu_rate", global_mean)
        if len(result) > 0:
            aggs[col] = result
            sub_n = pool[col].notna().sum()
            print(f"  {col}: {len(result)} ユニーク / {sub_n} 頭使用 {label}")
    return aggs


def add_features(df, aggs):
    df = df.copy()
    df["birth_month"] = df["birth_month"].apply(extract_birth_month)
    df["sex_num"] = df["sex"].map({"牡": 0, "牝": 1, "セ": 2}).fillna(-1).astype(int)

    for col, stats_df in aggs.items():
        if col not in df.columns:
            continue
        sc = stats_df.copy()
        df[col]   = df[col].astype(object)
        sc[col]   = sc[col].astype(object)
        df = pd.merge(df, sc, on=col, how="left")

    for col in df.columns:
        if col.endswith(("_smooth_mean", "_smooth_over200")):
            df[col] = df[col].fillna(df[col].median())
        elif col.endswith("_count"):
            df[col] = df[col].fillna(0)

    return df.reset_index(drop=True)


def feat_cols_base(df):
    agg = [c for c in df.columns
           if c.endswith(("_smooth_mean", "_smooth_over200", "_count"))]
    base = ["sex_num", "birth_month", "price_man"]
    return [c for c in base + agg if c in df.columns]


def feat_cols_with_scale(df):
    agg = [c for c in df.columns
           if c.endswith(("_smooth_mean", "_smooth_over200", "_count"))]
    base = ["sex_num", "birth_month", "price_man",
            "height", "chest", "cannon", "weight"]
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
    sw = y_train.map(cw).values
    model.fit(X_train, y_train, sample_weight=sw)
    return model


def cross_validate_model(X, y, cw, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        m = train_lgbm(X_tr, y_tr, cw)
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

def evaluate_model(model, X_val, df_val, label=""):
    pred   = model.predict(X_val)
    proba  = model.predict_proba(X_val)
    y_val  = df_val["target"]

    print(f"\n--- {label} 分類レポート ---")
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


def save_importance(model, feature_cols, filename):
    fi = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(MODEL_DIR / filename, index=False, encoding="utf-8-sig")
    print(f"\n特徴量重要度TOP10 ({filename}):")
    print(fi.head(10).to_string(index=False))


# ──────────────────────────────────────────
# メイン
# ──────────────────────────────────────────

def main():
    # ── データ準備 ──
    print("=== データ読み込み ===")
    silk_train_raw, silk_valid_raw = load_silk()
    other_all = load_other()

    other_train_raw = other_all[other_all["bosyu_year"].isin(TRAIN_YEARS)].copy()
    other_valid_raw = other_all[other_all["bosyu_year"].isin(VALID_YEARS)].copy()

    silk_train_raw = add_target(silk_train_raw)
    silk_valid_raw = add_target(silk_valid_raw)
    other_train_raw = add_target(other_train_raw)
    other_valid_raw = add_target(other_valid_raw)

    print(f"シルク 学習:{len(silk_train_raw)} / 検証:{len(silk_valid_raw)}")
    print(f"他クラブ 学習:{len(other_train_raw)} / 検証:{len(other_valid_raw)}")

    # ══════════════════════════════════════
    # ■ モデルB（測尺なし・汎用版）
    # ══════════════════════════════════════
    print("\n" + "="*55)
    print("■ モデルB: 測尺なし・汎用版 (silk + other学習)")
    print("="*55)

    # 集計プール: silk + other の学習データ
    pool_B = pd.concat([
        silk_train_raw[["sire", "trainer", "farm", "kaishuu_rate"]],
        other_train_raw[["sire", "trainer", "farm", "kaishuu_rate"]],
    ], ignore_index=True)
    print(f"\n集計プール: {len(pool_B)} 頭")
    aggs_B = build_aggs(pool_B, "(silk+other)")

    # 特徴量追加
    silk_tr_B  = add_features(silk_train_raw,  aggs_B)
    other_tr_B = add_features(other_train_raw, aggs_B)
    silk_vl_B  = add_features(silk_valid_raw,  aggs_B)
    other_vl_B = add_features(other_valid_raw, aggs_B)

    # 結合して学習
    train_B = pd.concat([silk_tr_B, other_tr_B], ignore_index=True)
    fcols_B = feat_cols_base(train_B)
    print(f"\nモデルB 特徴量 ({len(fcols_B)}個): {fcols_B}")

    X_tr_B = train_B[fcols_B].fillna(-1)
    y_tr_B = train_B["target"]
    cw_B   = get_class_weight(y_tr_B)

    print(f"\nクラス重み: { {k: round(v,2) for k,v in sorted(cw_B.items())} }")
    print("\n--- モデルB CV (5fold) ---")
    cv_auc_B = cross_validate_model(X_tr_B, y_tr_B, cw_B)

    print("\n--- モデルB 全データ学習 ---")
    model_B = train_lgbm(X_tr_B, y_tr_B, cw_B)

    print("\n--- モデルB 検証: シルク ---")
    X_svl_B = silk_vl_B[fcols_B].fillna(-1)
    df_evl_silk_B, t25_silk_B, o200_silk_B = evaluate_model(
        model_B, X_svl_B, silk_vl_B, "モデルB × シルク"
    )

    print("\n--- モデルB 検証: 他クラブ ---")
    X_ovl_B = other_vl_B[fcols_B].fillna(-1)
    df_evl_other_B, t25_other_B, o200_other_B = evaluate_model(
        model_B, X_ovl_B, other_vl_B, "モデルB × 他クラブ"
    )

    save_importance(model_B, fcols_B, "feature_importance_B.csv")
    joblib.dump({
        "model": model_B, "feature_cols": fcols_B, "aggs": aggs_B,
        "bins": BINS, "labels": LABELS, "label_names": LABEL_NAMES,
    }, MODEL_DIR / "lgbm_model_B.pkl")
    print("\nモデルB保存: models/lgbm_model_B.pkl")

    # ══════════════════════════════════════
    # ■ モデルA（測尺あり・精密版）
    # ══════════════════════════════════════
    print("\n" + "="*55)
    print("■ モデルA: 測尺あり・精密版 (silkのみ学習)")
    print("="*55)

    # 集計プール: silk のみ
    pool_A = silk_train_raw[["sire", "trainer", "farm", "kaishuu_rate"]].copy()
    print(f"\n集計プール: {len(pool_A)} 頭")
    aggs_A = build_aggs(pool_A, "(silkのみ)")

    silk_tr_A = add_features(silk_train_raw, aggs_A)
    silk_vl_A = add_features(silk_valid_raw, aggs_A)

    fcols_A = feat_cols_with_scale(silk_tr_A)
    print(f"\nモデルA 特徴量 ({len(fcols_A)}個): {fcols_A}")

    X_tr_A = silk_tr_A[fcols_A].fillna(-1)
    y_tr_A = silk_tr_A["target"]
    cw_A   = get_class_weight(y_tr_A)

    print(f"\nクラス重み: { {k: round(v,2) for k,v in sorted(cw_A.items())} }")
    print("\n--- モデルA CV (5fold) ---")
    cv_auc_A = cross_validate_model(X_tr_A, y_tr_A, cw_A)

    print("\n--- モデルA 全データ学習 ---")
    model_A = train_lgbm(X_tr_A, y_tr_A, cw_A)

    print("\n--- モデルA 検証: シルク ---")
    X_svl_A = silk_vl_A[fcols_A].fillna(-1)
    df_evl_silk_A, t25_silk_A, o200_silk_A = evaluate_model(
        model_A, X_svl_A, silk_vl_A, "モデルA × シルク"
    )

    save_importance(model_A, fcols_A, "feature_importance_A.csv")
    joblib.dump({
        "model": model_A, "feature_cols": fcols_A, "aggs": aggs_A,
        "bins": BINS, "labels": LABELS, "label_names": LABEL_NAMES,
    }, MODEL_DIR / "lgbm_model_A.pkl")
    print("\nモデルA保存: models/lgbm_model_A.pkl")

    # ══════════════════════════════════════
    # ■ 結果まとめ
    # ══════════════════════════════════════
    print("\n" + "="*62)
    print("■ 結果まとめ")
    print("="*62)
    hdr = f"{'モデル / 検証セット':<24} {'CV AUC':>8} {'上位25%回収率':>14} {'200%超率':>10}"
    print(hdr)
    print("-"*62)
    print(f"{'モデルA (silk検証)':<24} {cv_auc_A:>8.3f} {t25_silk_A:>13.1f}% {o200_silk_A:>9.1%}")
    print(f"{'モデルB (silk検証)':<24} {cv_auc_B:>8.3f} {t25_silk_B:>13.1f}% {o200_silk_B:>9.1%}")
    print(f"{'モデルB (他クラブ検証)':<24} {cv_auc_B:>8.3f} {t25_other_B:>13.1f}% {o200_other_B:>9.1%}")
    print("="*62)

    # CSV 保存
    df_evl_silk_A["model"]  = "A_silk"
    df_evl_silk_B["model"]  = "B_silk"
    df_evl_other_B["model"] = "B_other"
    out = pd.concat([df_evl_silk_A, df_evl_silk_B, df_evl_other_B], ignore_index=True)
    out.to_csv(DATA_DIR / "valid_predictions_v5.csv", index=False, encoding="utf-8-sig")
    print("\n予測結果保存: data/valid_predictions_v5.csv")


if __name__ == "__main__":
    main()
