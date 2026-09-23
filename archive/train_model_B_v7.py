"""
train_model_B_v7.py
母父（BMS）特徴量を追加したモデルB（測尺なし・汎用版）

train_model_v5.py のモデルB部分がベース。変更点:
  - bms_name（母父）ごとの smooth_mean / smooth_over200 / count を
    sire/trainer/farm と同じスムージング方式で追加
  - bms は jisseki_other_all.csv に列がないため、
    umadb個別ページから取得した data/bms_other_cache.csv
    （fetch_bms_other.py / fetch_bms_other_remaining.py）を使い、
    silk + other の学習データで集計プールを作成
    （評価用の他クラブ検証データにも bms_name をマージして特徴量を計算）

学習: merged_train.csv (silk) + jisseki_other_all.csv (2018-2021産)
検証: merged_valid.csv (silk) および jisseki_other_all.csv (2022-2023産) で別々評価

出力:
  models/lgbm_model_B_v7.pkl
  models/feature_importance_B_v7.csv
  data/valid_predictions_B_v7.csv
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
VALID_YEARS = [2022, 2023]


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
    bms = pd.read_csv(bms_path, encoding="utf-8-sig")[["horse_id", "bms_name"]]
    df = pd.merge(df, bms, on="horse_id", how="left")
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
    for col in ["sire", "trainer", "farm", "bms_name"]:
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
    base = ["sex_num", "birth_month", "price_man"]
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

def evaluate(model, X_val, df_val, label=""):
    pred  = model.predict(X_val)
    proba = model.predict_proba(X_val)
    y_val = df_val["target"]

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


def save_importance(model, feature_cols):
    fi = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(MODEL_DIR / "feature_importance_B_v7.csv", index=False, encoding="utf-8-sig")
    print("\n--- 特徴量重要度 TOP10 ---")
    print(fi.head(10).to_string(index=False))


# ──────────────────────────────────────────
# メイン
# ──────────────────────────────────────────

def main():
    print("=== データ読み込み ===")
    silk_train_raw, silk_valid_raw = load_silk()
    other_all = load_other()

    other_train_raw = other_all[other_all["bosyu_year"].isin(TRAIN_YEARS)].copy()
    other_valid_raw = other_all[other_all["bosyu_year"].isin(VALID_YEARS)].copy()

    silk_train_raw  = add_target(silk_train_raw)
    silk_valid_raw  = add_target(silk_valid_raw)
    other_train_raw = add_target(other_train_raw)
    other_valid_raw = add_target(other_valid_raw)

    print(f"シルク 学習:{len(silk_train_raw)} / 検証:{len(silk_valid_raw)}")
    print(f"他クラブ 学習:{len(other_train_raw)} / 検証:{len(other_valid_raw)}")

    # 集計プール: silk + other の学習データ（sire/trainer/farm/bms すべて）
    pool = pd.concat([
        silk_train_raw[["sire", "trainer", "farm", "bms_name", "kaishuu_rate"]],
        other_train_raw[["sire", "trainer", "farm", "bms_name", "kaishuu_rate"]],
    ], ignore_index=True)
    print(f"\n集計プール: {len(pool)} 頭 (bms_name有: {pool['bms_name'].notna().sum()})")
    aggs = build_aggs(pool, "(silk+other)")

    # 特徴量追加
    silk_tr = add_features(silk_train_raw,  aggs)
    other_tr = add_features(other_train_raw, aggs)
    silk_vl = add_features(silk_valid_raw,  aggs)
    other_vl = add_features(other_valid_raw, aggs)

    train = pd.concat([silk_tr, other_tr], ignore_index=True)
    feature_cols = get_feature_cols(train)
    print(f"\n特徴量 ({len(feature_cols)}個): {feature_cols}")

    X_tr = train[feature_cols].fillna(-1)
    y_tr = train["target"]
    cw   = get_class_weight(y_tr)

    print(f"\nクラス重み: { {k: round(v,2) for k,v in sorted(cw.items())} }")
    print("\n=== クロスバリデーション (5fold) ===")
    cv_auc = cross_validate_model(X_tr, y_tr, cw)

    print("\n=== 全データで学習 ===")
    model = train_lgbm(X_tr, y_tr, cw)

    print("\n=== 検証: シルク ===")
    X_svl = silk_vl[feature_cols].fillna(-1)
    df_evl_silk, t25_silk, o200_silk = evaluate(model, X_svl, silk_vl, "シルク")

    print("\n=== 検証: 他クラブ ===")
    X_ovl = other_vl[feature_cols].fillna(-1)
    df_evl_other, t25_other, o200_other = evaluate(model, X_ovl, other_vl, "他クラブ")

    save_importance(model, feature_cols)

    joblib.dump({
        "model":        model,
        "feature_cols": feature_cols,
        "aggs":         aggs,
        "bins":         BINS,
        "labels":       LABELS,
        "label_names":  LABEL_NAMES,
    }, MODEL_DIR / "lgbm_model_B_v7.pkl")
    print("\nモデル保存: models/lgbm_model_B_v7.pkl")

    df_evl_silk["eval_set"]  = "silk"
    df_evl_other["eval_set"] = "other"
    out = pd.concat([df_evl_silk, df_evl_other], ignore_index=True)
    out.to_csv(DATA_DIR / "valid_predictions_B_v7.csv", index=False, encoding="utf-8-sig")
    print("予測結果保存: data/valid_predictions_B_v7.csv")

    print("\n" + "="*62)
    print("=== モデルB v5(旧) vs v7（母父追加）比較 ===")
    print("="*62)
    hdr = f"{'検証セット':<20} {'CV AUC':>8} {'上位25%回収率':>14} {'200%超率':>10}"
    print(hdr)
    print("-"*62)
    print(f"{'silk':<20} {cv_auc:>8.3f} {t25_silk:>13.1f}% {o200_silk:>9.1%}")
    print(f"{'other':<20} {cv_auc:>8.3f} {t25_other:>13.1f}% {o200_other:>9.1%}")
    print("="*62)


if __name__ == "__main__":
    main()
