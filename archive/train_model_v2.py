"""
train_model_v2.py
シルクホースクラブ 回収率3分類予測モデル

クラス定義:
  0: 回収率100%未満  (元本割れ)
  1: 回収率100〜200% (回収できた)
  2: 回収率200%超    (大きく稼げた)

改善点 (v1からの変更):
  - 回帰 → 3値分類に変更
  - モデルを単純化（過学習対策）
  - クラス不均衡補正 (class_weight)
  - 集計特徴量にスムージング適用（サンプル少ない父馬・厩舎の補正）
  - 評価指標を分類向けに変更

出力:
  models/lgbm_model_v2.pkl      学習済みモデル
  models/feature_importance_v2.csv
  data/valid_predictions_v2.csv  検証予測結果（各クラスの確率付き）

使い方:
  python train_model_v2.py
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

# クラス定義
BINS   = [-1, 100, 200, 99999]
LABELS = [0, 1, 2]
LABEL_NAMES = {0: "100%未満", 1: "100〜200%", 2: "200%超"}

# ──────────────────────────────────────────
# データ読み込み
# ──────────────────────────────────────────

def load_data():
    train = pd.read_csv(DATA_DIR / "merged_train.csv", encoding="utf-8-sig")
    valid = pd.read_csv(DATA_DIR / "merged_valid.csv", encoding="utf-8-sig")

    # ターゲット変数を3分類に変換
    train["target"] = pd.cut(train["kaishuu_rate"], bins=BINS, labels=LABELS).astype(int)
    valid["target"] = pd.cut(valid["kaishuu_rate"], bins=BINS, labels=LABELS).astype(int)

    print(f"学習用: {len(train)} 頭 / 検証用: {len(valid)} 頭")
    print("\n学習データのクラス分布:")
    for label, name in LABEL_NAMES.items():
        n = (train["target"] == label).sum()
        print(f"  クラス{label} ({name}): {n}頭 ({n/len(train)*100:.1f}%)")

    return train, valid


# ──────────────────────────────────────────
# 特徴量エンジニアリング
# ──────────────────────────────────────────

def extract_birth_month(birth_str):
    if pd.isna(birth_str):
        return -1
    m = re.search(r"(\d+)月", str(birth_str))
    return int(m.group(1)) if m else -1


def smooth_mean(df, key_col, value_col, global_mean, min_samples=5):
    """
    ベイズ平滑化 (スムージング)
    サンプル数が少ないグループは全体平均に引き寄せる
    smooth = (n * group_mean + min_samples * global_mean) / (n + min_samples)
    """
    g = df.groupby(key_col)[value_col]
    stats = pd.DataFrame({
        "count": g.count(),
        "mean":  g.mean(),
    })
    stats[f"{key_col}_smooth_mean"] = (
        (stats["count"] * stats["mean"] + min_samples * global_mean)
        / (stats["count"] + min_samples)
    )
    stats[f"{key_col}_count"] = stats["count"]
    stats[f"{key_col}_over200_rate"] = (
        (df[value_col] >= 200).groupby(df[key_col]).sum()
        / stats["count"]
    )

    # 200%超率もスムージング
    global_over200 = (df[value_col] >= 200).mean()
    stats[f"{key_col}_smooth_over200"] = (
        (stats["count"] * stats[f"{key_col}_over200_rate"] + min_samples * global_over200)
        / (stats["count"] + min_samples)
    )

    return stats[[
        f"{key_col}_smooth_mean",
        f"{key_col}_smooth_over200",
        f"{key_col}_count",
    ]].reset_index()


def build_aggs(train):
    """学習データのみから集計特徴量を計算（リーク防止）"""
    global_mean = train["kaishuu_rate"].mean()
    aggs = {}
    for col in ["sire", "trainer", "farm"]:
        if col in train.columns:
            aggs[col] = smooth_mean(
                train.dropna(subset=[col]), col, "kaishuu_rate", global_mean
            )
    return aggs


def add_features(df, aggs):
    df = df.copy()

    # 生月
    df["birth_month"] = df["birth_month"].apply(extract_birth_month)

    # 性別
    df["sex_num"] = df["sex"].map({"牡": 0, "牝": 1, "セ": 2}).fillna(-1).astype(int)

    # 一口価格を万円に
    if "price_per_kuchi" in df.columns:
        df["price_per_kuchi_man"] = pd.to_numeric(df["price_per_kuchi"], errors="coerce") / 10000

    # 集計特徴量をマージ
    for col, stats_df in aggs.items():
        if col in df.columns:
            df = pd.merge(df, stats_df, on=col, how="left")

    # 欠損補完（全体平均・中央値）
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
    ]
    agg = [c for c in df.columns if c.endswith((
        "_smooth_mean", "_smooth_over200", "_count"
    ))]
    return [c for c in base + agg if c in df.columns]


# ──────────────────────────────────────────
# モデル学習
# ──────────────────────────────────────────

def get_class_weight(y):
    """クラス不均衡補正用の重み計算"""
    counts = y.value_counts()
    total = len(y)
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
        num_class=3,
        objective="multiclass",
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


# ──────────────────────────────────────────
# クロスバリデーション
# ──────────────────────────────────────────

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


# ──────────────────────────────────────────
# 検証・評価
# ──────────────────────────────────────────

def evaluate(model, X_valid, y_valid, df_valid):
    pred       = model.predict(X_valid)
    pred_proba = model.predict_proba(X_valid)  # shape: (n, 3)

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

    # ── 実用評価: 予測クラス2（200%超）に絞った場合の実績 ──
    df_eval = df_valid[["horse_name", "bosyu_year", "kaishuu_rate"]].copy()
    df_eval["pred_class"]  = pred
    df_eval["prob_class0"] = pred_proba[:, 0]
    df_eval["prob_class1"] = pred_proba[:, 1]
    df_eval["prob_class2"] = pred_proba[:, 2]  # 200%超の確率

    print("\n--- 実用評価: 予測クラス別の実際の回収率 ---")
    for cls in [0, 1, 2]:
        subset = df_eval[df_eval["pred_class"] == cls]
        if len(subset) == 0:
            continue
        actual_over200 = (subset["kaishuu_rate"] >= 200).mean()
        print(f"  予測クラス{cls} ({LABEL_NAMES[cls]}): "
              f"{len(subset)}頭  "
              f"実際の平均回収率={subset['kaishuu_rate'].mean():.1f}%  "
              f"実際の200%超率={actual_over200:.1%}")

    # ── prob_class2 上位25%に出資した場合 ──
    threshold = df_eval["prob_class2"].quantile(0.75)
    top25 = df_eval[df_eval["prob_class2"] >= threshold]
    all_avg   = df_eval["kaishuu_rate"].mean()
    top25_avg = top25["kaishuu_rate"].mean()

    print(f"\n--- 200%超確率 上位25%への出資シミュレーション ---")
    print(f"全頭平均回収率          : {all_avg:.1f}%")
    print(f"確率上位25%の平均回収率  : {top25_avg:.1f}%")
    print(f"差分                    : {top25_avg - all_avg:+.1f}%")

    over200_all = (df_eval["kaishuu_rate"] >= 200).mean()
    over200_top = (top25["kaishuu_rate"] >= 200).mean()
    print(f"全頭の200%超率          : {over200_all:.1%}")
    print(f"確率上位25%の200%超率   : {over200_top:.1%}")

    if top25_avg > all_avg:
        print("→ ✅ ランダムより有意に優れた選択ができています")
    else:
        print("→ ⚠️  ランダムと同等以下です")

    return df_eval


# ──────────────────────────────────────────
# 特徴量重要度
# ──────────────────────────────────────────

def save_feature_importance(model, feature_cols):
    fi = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(MODEL_DIR / "feature_importance_v2.csv", index=False, encoding="utf-8-sig")
    print("\n--- 特徴量重要度 TOP10 ---")
    print(fi.head(10).to_string(index=False))


# ──────────────────────────────────────────
# メイン
# ──────────────────────────────────────────

def main():
    print("=== データ読み込み ===")
    train, valid = load_data()

    print("\n=== 集計特徴量の計算 ===")
    aggs = build_aggs(train)

    print("\n=== 特徴量追加 ===")
    train = add_features(train, aggs)
    valid = add_features(valid, aggs)
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

    # 保存
    joblib.dump({
        "model":        model,
        "feature_cols": feature_cols,
        "aggs":         aggs,
        "bins":         BINS,
        "labels":       LABELS,
        "label_names":  LABEL_NAMES,
    }, MODEL_DIR / "lgbm_model_v2.pkl")
    print(f"\nモデル保存: models/lgbm_model_v2.pkl")

    save_feature_importance(model, feature_cols)
    df_eval.to_csv(DATA_DIR / "valid_predictions_v2.csv", index=False, encoding="utf-8-sig")
    print(f"予測結果保存: data/valid_predictions_v2.csv")


if __name__ == "__main__":
    main()
