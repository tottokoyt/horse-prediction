"""
train_model_B_v8.py
モデルB（測尺なし）の実学習行に、他クラブの検証年度(2022-2023)分も
追加してデータ量を増やした版

背景:
  モデルBはB_v7の時点で既に他クラブ学習年度(2018-2021)分を実学習行に
  使っており（Aと違い測尺が不要なため）、634頭ではなく2,893頭で
  学習できていた。一方で他クラブの検証年度(2022-2023)分
  （jisseki_other_all.csvに約1,200頭）は「other_valid」として
  評価専用に取り置かれ、学習には使われていなかった。
  この約1,200頭を学習に回せば、追加のデータ収集なしで実学習行を
  増やせる（bms_other_cache.csv は元々2018-2023年産すべてを
  カバーしている）。

v7からの変更点:
  - other_train を TRAIN_YEARS(2018-2021) から全年度(jisseki_other_all.csv
    の2018-2023すべて)に拡張し、実学習行として利用
  - 評価は引き続きシルク検証156頭（merged_valid.csv）のみで行う
    （他クラブ検証セットは学習に取り込んだため廃止、silk評価は
    B_v7と同一条件で比較可能）
  - eval_utils.evaluate_stable() で複数seed平均・capped指標を使い、
    B_v7自身の安定ベースラインとフェアに比較する

出力:
  models/lgbm_model_B_v8.pkl
  models/feature_importance_B_v8.csv
  data/valid_predictions_B_v8.csv
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

from eval_utils import evaluate_stable

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


def train_lgbm(X_train, y_train, cw, random_state_override=None):
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
        random_state=42 if random_state_override is None else random_state_override,
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
    fi.to_csv(MODEL_DIR / "feature_importance_B_v8.csv", index=False, encoding="utf-8-sig")
    print("\n--- 特徴量重要度 TOP10 ---")
    print(fi.head(10).to_string(index=False))


# ──────────────────────────────────────────
# メイン
# ──────────────────────────────────────────

def build_variant(silk_train_raw, silk_valid_raw, other_all, other_years, label):
    """other_years に含まれる他クラブ年度を実学習行に使ってモデルを構築・評価する"""
    other_train_raw = other_all[other_all["bosyu_year"].isin(other_years)].copy()
    other_train_raw = add_target(other_train_raw)

    print(f"\n--- {label}: 他クラブ実学習行 {len(other_train_raw)} 頭（年度{sorted(other_years)}） ---")

    pool = pd.concat([
        silk_train_raw[["sire", "trainer", "farm", "bms_name", "kaishuu_rate"]],
        other_train_raw[["sire", "trainer", "farm", "bms_name", "kaishuu_rate"]],
    ], ignore_index=True)
    aggs = build_aggs(pool, f"({label})")

    silk_tr  = add_features(silk_train_raw,  aggs)
    other_tr = add_features(other_train_raw, aggs)
    silk_vl  = add_features(silk_valid_raw,  aggs)

    train = pd.concat([silk_tr, other_tr], ignore_index=True)
    feature_cols = get_feature_cols(train)

    X_tr = train[feature_cols].fillna(-1)
    y_tr = train["target"]
    X_vl = silk_vl[feature_cols].fillna(-1)
    cw   = get_class_weight(y_tr)

    stable = evaluate_stable(
        lambda seed: train_lgbm(X_tr, y_tr, cw, random_state_override=seed),
        X_tr, y_tr, X_vl, silk_vl, label=label,
    )
    print("  " + stable["summary"])

    model = train_lgbm(X_tr, y_tr, cw)
    df_eval, t25_avg, t25_o200 = evaluate(model, X_vl, silk_vl, label)

    return {
        "model": model, "feature_cols": feature_cols, "aggs": aggs,
        "df_eval": df_eval, "stable": stable,
    }


def main():
    print("=== データ読み込み ===")
    silk_train_raw, silk_valid_raw = load_silk()
    other_all = load_other()
    silk_train_raw = add_target(silk_train_raw)
    silk_valid_raw = add_target(silk_valid_raw)
    print(f"シルク 学習:{len(silk_train_raw)} / 検証:{len(silk_valid_raw)}")

    print("\n" + "="*62)
    print("=== B_v7相当（他クラブ学習年度2018-2021のみ）のベースライン ===")
    print("="*62)
    v7 = build_variant(silk_train_raw, silk_valid_raw, other_all, TRAIN_YEARS, "B_v7相当")

    print("\n" + "="*62)
    print("=== B_v8（他クラブ全年度2018-2023を実学習行に追加） ===")
    print("="*62)
    v8 = build_variant(silk_train_raw, silk_valid_raw, other_all, TRAIN_YEARS + VALID_YEARS, "B_v8")

    print("\n" + "="*62)
    print("=== 比較（複数seed平均・capped指標） ===")
    print("="*62)
    print("  " + v7["stable"]["summary"])
    print("  " + v8["stable"]["summary"])

    if v8["stable"]["capped_mean"] > v7["stable"]["capped_mean"]:
        print("\n=> B_v8がベースラインを上回ったためデプロイ用に保存")
        chosen = v8
    else:
        print("\n=> B_v8はベースラインを上回れなかったため保存のみ（デプロイはしない）")
        chosen = v8

    save_importance(chosen["model"], chosen["feature_cols"])

    joblib.dump({
        "model":        chosen["model"],
        "feature_cols": chosen["feature_cols"],
        "aggs":         chosen["aggs"],
        "bins":         BINS,
        "labels":       LABELS,
        "label_names":  LABEL_NAMES,
    }, MODEL_DIR / "lgbm_model_B_v8.pkl")
    print("モデル保存: models/lgbm_model_B_v8.pkl")

    chosen["df_eval"].to_csv(DATA_DIR / "valid_predictions_B_v8.csv", index=False, encoding="utf-8-sig")
    print("予測結果保存: data/valid_predictions_B_v8.csv")


if __name__ == "__main__":
    main()
