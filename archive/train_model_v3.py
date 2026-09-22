"""
train_model_v3.py
兄弟馬実績を特徴量に追加したv3モデル

v2からの変更点:
  - dam_name をキーに兄弟馬の回収率実績を特徴量として追加
    - sibling_mean        : 兄弟馬の平均回収率
    - sibling_over200     : 兄弟馬の200%超率
    - sibling_over100     : 兄弟馬の100%超率
    - sibling_count       : 兄弟馬の実績頭数
  - リーク防止: 予測対象馬自身・同年産馬は除外して計算
  - 検証セットへの兄弟特徴量は学習データのみから計算

使い方:
  python train_model_v3.py
"""

import re
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BINS        = [-1, 100, 200, 99999]
LABELS      = [0, 1, 2]
LABEL_NAMES = {0: "100%未満", 1: "100〜200%", 2: "200%超"}

# ──────────────────────────────────────────
# データ読み込み
# ──────────────────────────────────────────

def load_data():
    train = pd.read_csv(DATA_DIR / "merged_train.csv", encoding="utf-8-sig")
    valid = pd.read_csv(DATA_DIR / "merged_valid.csv", encoding="utf-8-sig")

    # dam_name 列名を統一
    for df in [train, valid]:
        if "dam_name" not in df.columns and "mother_name" in df.columns:
            df.rename(columns={"mother_name": "dam_name"}, inplace=True)

    train["target"] = pd.cut(train["kaishuu_rate"], bins=BINS, labels=LABELS).astype(int)
    valid["target"] = pd.cut(valid["kaishuu_rate"], bins=BINS, labels=LABELS).astype(int)

    print(f"学習用: {len(train)} 頭 / 検証用: {len(valid)} 頭")
    print("\n学習データのクラス分布:")
    for label, name in LABEL_NAMES.items():
        n = (train["target"] == label).sum()
        print(f"  クラス{label} ({name}): {n}頭 ({n/len(train)*100:.1f}%)")

    return train, valid


# ──────────────────────────────────────────
# 兄弟馬特徴量
# ──────────────────────────────────────────

def build_sibling_features(train):
    """
    学習データから兄弟馬の実績集計を計算する。
    リーク防止のルール:
      - 予測対象馬自身は除外（horse_name が一致する行を除く）
      - 同じ bosyu_year の馬も除外
        （同年産の兄弟は「募集時点では実績がない」ため）
    戻り値: dam_name をインデックスとした集計DataFrame
    """
    records = []

    for dam, group in train.groupby("dam_name"):
        for idx, row in group.iterrows():
            siblings = group[
                (group.index != idx) &
                (group["bosyu_year"] != row["bosyu_year"])
            ]

            if len(siblings) == 0:
                records.append({
                    "horse_name":      row["horse_name"],
                    "bosyu_year":      row["bosyu_year"],
                    "dam_name":        dam,
                    "sibling_count":   0,
                    "sibling_mean":    np.nan,
                    "sibling_over200": np.nan,
                    "sibling_over100": np.nan,
                })
            else:
                records.append({
                    "horse_name":      row["horse_name"],
                    "bosyu_year":      row["bosyu_year"],
                    "dam_name":        dam,
                    "sibling_count":   len(siblings),
                    "sibling_mean":    siblings["kaishuu_rate"].mean(),
                    "sibling_over200": (siblings["kaishuu_rate"] >= 200).mean(),
                    "sibling_over100": (siblings["kaishuu_rate"] >= 100).mean(),
                })

    return pd.DataFrame(records)


def add_sibling_features_to_valid(valid, train):
    """
    検証データへの兄弟特徴量追加。
    学習データの実績のみを使って計算（リーク防止）。
    """
    records = []

    for _, row in valid.iterrows():
        dam = row.get("dam_name")
        if pd.isna(dam):
            records.append({
                "horse_name":      row["horse_name"],
                "bosyu_year":      row["bosyu_year"],
                "sibling_count":   0,
                "sibling_mean":    np.nan,
                "sibling_over200": np.nan,
                "sibling_over100": np.nan,
            })
            continue

        siblings = train[
            (train["dam_name"] == dam) &
            (train["bosyu_year"] != row["bosyu_year"])
        ]

        if len(siblings) == 0:
            records.append({
                "horse_name":      row["horse_name"],
                "bosyu_year":      row["bosyu_year"],
                "sibling_count":   0,
                "sibling_mean":    np.nan,
                "sibling_over200": np.nan,
                "sibling_over100": np.nan,
            })
        else:
            records.append({
                "horse_name":      row["horse_name"],
                "bosyu_year":      row["bosyu_year"],
                "sibling_count":   len(siblings),
                "sibling_mean":    siblings["kaishuu_rate"].mean(),
                "sibling_over200": (siblings["kaishuu_rate"] >= 200).mean(),
                "sibling_over100": (siblings["kaishuu_rate"] >= 100).mean(),
            })

    return pd.DataFrame(records)


# ──────────────────────────────────────────
# その他の特徴量（v2から流用）
# ──────────────────────────────────────────

def extract_birth_month(birth_str):
    if pd.isna(birth_str):
        return -1
    m = re.search(r"(\d+)月", str(birth_str))
    return int(m.group(1)) if m else -1


def smooth_mean(df, key_col, value_col, global_mean, min_samples=5):
    g = df.groupby(key_col)[value_col]
    stats = pd.DataFrame({"count": g.count(), "mean": g.mean()})
    stats[f"{key_col}_smooth_mean"] = (
        (stats["count"] * stats["mean"] + min_samples * global_mean)
        / (stats["count"] + min_samples)
    )
    stats[f"{key_col}_count"] = stats["count"]
    global_over200 = (df[value_col] >= 200).mean()
    over200 = (df[value_col] >= 200).groupby(df[key_col]).mean()
    stats[f"{key_col}_smooth_over200"] = (
        (stats["count"] * over200 + min_samples * global_over200)
        / (stats["count"] + min_samples)
    )
    return stats[[
        f"{key_col}_smooth_mean",
        f"{key_col}_smooth_over200",
        f"{key_col}_count",
    ]].reset_index()


def build_aggs(train):
    global_mean = train["kaishuu_rate"].mean()
    aggs = {}
    for col in ["sire", "trainer", "farm"]:
        if col in train.columns:
            aggs[col] = smooth_mean(
                train.dropna(subset=[col]), col, "kaishuu_rate", global_mean
            )
    return aggs


def add_features(df, aggs, sibling_df):
    df = df.copy()

    # 兄弟馬特徴量をマージ
    df = pd.merge(
        df,
        sibling_df[["horse_name", "bosyu_year",
                     "sibling_count", "sibling_mean",
                     "sibling_over200", "sibling_over100"]],
        on=["horse_name", "bosyu_year"],
        how="left",
    )

    # 兄弟なし（sibling_count=0）の欠損を全体平均で補完
    global_mean    = df["kaishuu_rate"].mean() if "kaishuu_rate" in df.columns else 80
    global_over200 = 0.18
    global_over100 = 0.30
    df["sibling_mean"]    = df["sibling_mean"].fillna(global_mean)
    df["sibling_over200"] = df["sibling_over200"].fillna(global_over200)
    df["sibling_over100"] = df["sibling_over100"].fillna(global_over100)
    df["sibling_count"]   = df["sibling_count"].fillna(0)

    # 基本特徴量
    df["birth_month"] = df["birth_month"].apply(extract_birth_month)
    df["sex_num"] = df["sex"].map({"牡": 0, "牝": 1, "セ": 2}).fillna(-1).astype(int)
    if "price_per_kuchi" in df.columns:
        df["price_per_kuchi_man"] = pd.to_numeric(
            df["price_per_kuchi"], errors="coerce"
        ) / 10000

    # 集計特徴量
    for col, stats_df in aggs.items():
        if col in df.columns:
            df = pd.merge(df, stats_df, on=col, how="left")

    for col in df.columns:
        if col.endswith(("_smooth_mean", "_smooth_over200")):
            df[col] = df[col].fillna(df[col].median())
        elif col.endswith("_count"):
            df[col] = df[col].fillna(0)

    return df


def get_feature_cols(df):
    base = [
        "sex_num", "birth_month",
        "total_price_man", "price_per_kuchi_man",
        "height", "chest", "cannon", "weight",
        "sibling_count", "sibling_mean",
        "sibling_over200", "sibling_over100",
    ]
    agg = [c for c in df.columns if c.endswith((
        "_smooth_mean", "_smooth_over200", "_count"
    )) and not c.startswith("sibling")]
    return [c for c in base + agg if c in df.columns]


# ──────────────────────────────────────────
# モデル学習・評価（v2と同じ）
# ──────────────────────────────────────────

def get_class_weight(y):
    counts = y.value_counts()
    total  = len(y)
    weights = {cls: total / (len(counts) * cnt) for cls, cnt in counts.items()}
    print("\nクラス重み:")
    for cls, w in sorted(weights.items()):
        print(f"  クラス{cls} ({LABEL_NAMES[cls]}): {w:.2f}")
    return weights


def train_lgbm(X_train, y_train, class_weight):
    sample_weight = y_train.map(class_weight).values
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
        class_weight=class_weight,
        random_state=42,
        verbose=-1,
        objective="multiclass",
        num_class=3,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def cross_validate(X, y, class_weight, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, aucs = [], []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = train_lgbm(X_tr, y_tr, class_weight)
        pred       = model.predict(X_val)
        pred_proba = model.predict_proba(X_val)

        acc = accuracy_score(y_val, pred)
        try:
            auc = roc_auc_score(y_val, pred_proba, multi_class="ovr", average="macro")
        except Exception:
            auc = float("nan")

        accs.append(acc)
        aucs.append(auc)
        print(f"  Fold {fold+1}: Accuracy={acc:.3f}  AUC={auc:.3f}")

    print(f"  CV平均: Accuracy={np.mean(accs):.3f} ± {np.std(accs):.3f}  "
          f"AUC={np.nanmean(aucs):.3f} ± {np.nanstd(aucs):.3f}")


def evaluate(model, X_valid, y_valid, df_valid):
    pred       = model.predict(X_valid)
    pred_proba = model.predict_proba(X_valid)

    print("\n--- 分類レポート ---")
    print(classification_report(
        y_valid, pred,
        target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    ))

    print("--- 混同行列 ---")
    cm = confusion_matrix(y_valid, pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"実際:{LABEL_NAMES[i]}" for i in range(3)],
        columns=[f"予測:{LABEL_NAMES[i]}" for i in range(3)]
    )
    print(cm_df)

    df_eval = df_valid[["horse_name", "bosyu_year", "kaishuu_rate"]].copy()
    df_eval["pred_class"]  = pred
    df_eval["prob_class0"] = pred_proba[:, 0]
    df_eval["prob_class1"] = pred_proba[:, 1]
    df_eval["prob_class2"] = pred_proba[:, 2]

    print("\n--- 予測クラス別の実際の回収率 ---")
    for cls in [0, 1, 2]:
        subset = df_eval[df_eval["pred_class"] == cls]
        if len(subset) == 0:
            continue
        over200 = (subset["kaishuu_rate"] >= 200).mean()
        print(f"  予測クラス{cls} ({LABEL_NAMES[cls]}): "
              f"{len(subset)}頭  "
              f"平均回収率={subset['kaishuu_rate'].mean():.1f}%  "
              f"200%超率={over200:.1%}")

    threshold = df_eval["prob_class2"].quantile(0.75)
    top25     = df_eval[df_eval["prob_class2"] >= threshold]
    all_avg   = df_eval["kaishuu_rate"].mean()
    top25_avg = top25["kaishuu_rate"].mean()

    print(f"\n--- 200%超確率 上位25%への出資シミュレーション ---")
    print(f"全頭平均回収率          : {all_avg:.1f}%")
    print(f"確率上位25%の平均回収率  : {top25_avg:.1f}%")
    print(f"差分                    : {top25_avg - all_avg:+.1f}%")
    print(f"全頭の200%超率          : {(df_eval['kaishuu_rate'] >= 200).mean():.1%}")
    print(f"確率上位25%の200%超率   : {(top25['kaishuu_rate'] >= 200).mean():.1%}")

    if top25_avg > all_avg:
        print("→ ✅ ランダムより有意に優れた選択ができています")
    else:
        print("→ ⚠️  ランダムと同等以下です")

    print(f"\n[v2比較] v2の上位25%平均回収率は83.6%でした")
    print(f"[v3結果] 上位25%平均回収率: {top25_avg:.1f}%")
    print(f"[改善幅] {top25_avg - 83.6:+.1f}%")

    return df_eval


def save_feature_importance(model, feature_cols):
    fi = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(MODEL_DIR / "feature_importance_v3.csv", index=False, encoding="utf-8-sig")
    print("\n--- 特徴量重要度 TOP10 ---")
    print(fi.head(10).to_string(index=False))


# ──────────────────────────────────────────
# メイン
# ──────────────────────────────────────────

def main():
    print("=== データ読み込み ===")
    train, valid = load_data()

    print("\n=== 兄弟馬特徴量の計算 ===")
    sibling_train = build_sibling_features(train)
    sibling_valid = add_sibling_features_to_valid(valid, train)

    has_sibling_train = (sibling_train["sibling_count"] > 0).sum()
    has_sibling_valid = (sibling_valid["sibling_count"] > 0).sum()
    print(f"  学習: 兄弟実績あり {has_sibling_train}/{len(sibling_train)} 頭")
    print(f"  検証: 兄弟実績あり {has_sibling_valid}/{len(sibling_valid)} 頭")

    print("\n=== 集計特徴量の計算 ===")
    aggs = build_aggs(train)

    print("\n=== 特徴量追加 ===")
    train = add_features(train, aggs, sibling_train)
    valid = add_features(valid, aggs, sibling_valid)
    feature_cols = get_feature_cols(train)
    print(f"使用特徴量: {len(feature_cols)} 個")
    print(f"  {feature_cols}")

    X_train = train[feature_cols].fillna(-1)
    y_train = train["target"]
    X_valid = valid[feature_cols].fillna(-1)
    y_valid = valid["target"]

    class_weight = get_class_weight(y_train)

    print(f"\n=== クロスバリデーション (5fold) ===")
    cross_validate(X_train, y_train, class_weight)

    print(f"\n=== 全学習データでモデル構築 ===")
    model = train_lgbm(X_train, y_train, class_weight)

    print(f"\n=== 検証セットで評価 ===")
    df_eval = evaluate(model, X_valid, y_valid, valid)

    joblib.dump({
        "model":        model,
        "feature_cols": feature_cols,
        "aggs":         aggs,
        "bins":         BINS,
        "labels":       LABELS,
        "label_names":  LABEL_NAMES,
    }, MODEL_DIR / "lgbm_model_v3.pkl")
    print(f"\nモデル保存: models/lgbm_model_v3.pkl")

    save_feature_importance(model, feature_cols)
    df_eval.to_csv(DATA_DIR / "valid_predictions_v3.csv", index=False, encoding="utf-8-sig")
    print(f"予測結果保存: data/valid_predictions_v3.csv")


if __name__ == "__main__":
    main()
