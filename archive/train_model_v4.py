"""
train_model_v4.py
他クラブデータで集計特徴量を強化したv4モデル

v2からの変更点:
  - sire / trainer / farm の smooth_mean を
    merged_train.csv（シルク学習用）+ jisseki_other_all.csv（他10クラブ）
    の結合データから計算（サンプル数が増えスムージング精度向上）
    ※ sire は jisseki_other_all に列がないためシルクのみ
  - club_name をカテゴリ特徴量として追加（"silk" 固定）
  - 測尺特徴量（height / chest / cannon / weight）を除外

学習・検証ターゲット: シルクのみ（v2と同じ）
モデルパラメータ: v2と同じ

出力:
  models/lgbm_model_v4.pkl
  models/feature_importance_v4.csv
  data/valid_predictions_v4.csv
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

# v2比較用ハードコード値
V2_CV_AUC      = 0.979
V2_TOP25_AVG   = 83.6
V2_TOP25_OVER200 = 0.051

# ──────────────────────────────────────────
# データ読み込み
# ──────────────────────────────────────────

def load_data():
    train = pd.read_csv(DATA_DIR / "merged_train.csv", encoding="utf-8-sig")
    valid = pd.read_csv(DATA_DIR / "merged_valid.csv", encoding="utf-8-sig")
    other = pd.read_csv(DATA_DIR / "jisseki_other_all.csv", encoding="utf-8-sig")

    # farm列を統一（farm_x を farm として使う）
    for df in [train, valid]:
        if "farm" not in df.columns:
            if "farm_x" in df.columns:
                df["farm"] = df["farm_x"]
            elif "farm_y" in df.columns:
                df["farm"] = df["farm_y"]

    # silk の club_name
    train["club_name"] = "silk"
    valid["club_name"] = "silk"

    train["target"] = pd.cut(train["kaishuu_rate"], bins=BINS, labels=LABELS).astype(int)
    valid["target"] = pd.cut(valid["kaishuu_rate"], bins=BINS, labels=LABELS).astype(int)

    print(f"学習用(silk): {len(train)} 頭 / 検証用(silk): {len(valid)} 頭")
    print(f"他クラブ計  : {len(other)} 頭")
    print("\n学習データのクラス分布:")
    for label, name in LABEL_NAMES.items():
        n = (train["target"] == label).sum()
        print(f"  クラス{label} ({name}): {n}頭 ({n/len(train)*100:.1f}%)")

    return train, valid, other


# ──────────────────────────────────────────
# 集計プール（silk + other）
# ──────────────────────────────────────────

def build_pool(train: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """
    smooth_mean 計算用のプールデータを作る。
    - sire  : silk のみ（other に列なし）
    - trainer / farm : silk + other
    kaishuu_rate は全クラブ共通で使える。
    """
    silk_cols = pd.DataFrame({
        "sire":         train["sire"] if "sire" in train.columns else np.nan,
        "trainer":      train["trainer"],
        "farm":         train["farm"],
        "kaishuu_rate": train["kaishuu_rate"],
    })

    other_cols = pd.DataFrame({
        "sire":         np.nan,
        "trainer":      other["trainer"],
        "farm":         other["farm"],
        "kaishuu_rate": other["kaishuu_rate"],
    })

    pool = pd.concat([silk_cols, other_cols], ignore_index=True)
    print(f"\n集計プール: {len(pool)} 頭 (silk {len(silk_cols)} + other {len(other_cols)})")
    return pool


# ──────────────────────────────────────────
# 特徴量エンジニアリング（v2ベース）
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


def build_aggs(pool: pd.DataFrame) -> dict:
    """
    プールデータから集計特徴量を計算。
    sire は NaN 行が多いので dropna して silk のみ実質集計。
    """
    global_mean = pool["kaishuu_rate"].mean()
    aggs = {}
    for col in ["sire", "trainer", "farm"]:
        sub = pool.dropna(subset=[col, "kaishuu_rate"])
        if len(sub) > 0:
            aggs[col] = smooth_mean(sub, col, "kaishuu_rate", global_mean)
            print(f"  {col}: {len(aggs[col])} ユニーク (プール {len(sub)} 頭)")
    return aggs


def encode_club(df: pd.DataFrame, club_encoder: dict) -> pd.DataFrame:
    """club_name を整数にエンコード"""
    df = df.copy()
    df["club_enc"] = df["club_name"].map(club_encoder).fillna(-1).astype(int)
    return df


def add_features(df: pd.DataFrame, aggs: dict, club_encoder: dict) -> pd.DataFrame:
    df = df.copy()

    # 生月
    df["birth_month"] = df["birth_month"].apply(extract_birth_month)

    # 性別
    df["sex_num"] = df["sex"].map({"牡": 0, "牝": 1, "セ": 2}).fillna(-1).astype(int)

    # 一口価格を万円に
    if "price_per_kuchi" in df.columns:
        df["price_per_kuchi_man"] = pd.to_numeric(
            df["price_per_kuchi"], errors="coerce"
        ) / 10000

    # club_name エンコード
    df = encode_club(df, club_encoder)

    # 集計特徴量をマージ（型不一致を防ぐため両側を object に統一）
    for col, stats_df in aggs.items():
        if col in df.columns:
            stats_copy = stats_df.copy()
            df[col] = df[col].astype(object)
            stats_copy[col] = stats_copy[col].astype(object)
            df = pd.merge(df, stats_copy, on=col, how="left")

    # 欠損補完
    for col in df.columns:
        if col.endswith(("_smooth_mean", "_smooth_over200")):
            df[col] = df[col].fillna(df[col].median())
        elif col.endswith("_count"):
            df[col] = df[col].fillna(0)

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    base = [
        "sex_num", "birth_month",
        "total_price_man", "price_per_kuchi_man",
        "club_enc",
        # height / chest / cannon / weight は除外
    ]
    agg = [c for c in df.columns if c.endswith((
        "_smooth_mean", "_smooth_over200", "_count"
    ))]
    return [c for c in base + agg if c in df.columns]


# ──────────────────────────────────────────
# モデル学習・評価（v2と同じパラメータ）
# ──────────────────────────────────────────

def get_class_weight(y):
    counts = y.value_counts()
    total  = len(y)
    weights = {cls: total / (len(counts) * cnt) for cls, cnt in counts.items()}
    print("\nクラス重み:")
    for cls, w in sorted(weights.items()):
        print(f"  クラス{cls} ({LABEL_NAMES[cls]}): {w:.2f}")
    return weights


def train_lgbm(X_train, y_train, class_weight, cat_features=None):
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
    fit_params = {"sample_weight": sample_weight}
    if cat_features:
        fit_params["categorical_feature"] = cat_features
    model.fit(X_train, y_train, **fit_params)
    return model


def cross_validate(X, y, class_weight, cat_features=None, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, aucs = [], []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = train_lgbm(X_tr, y_tr, class_weight, cat_features)
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

    mean_auc = np.nanmean(aucs)
    print(f"  CV平均: Accuracy={np.mean(accs):.3f} ± {np.std(accs):.3f}  "
          f"AUC={mean_auc:.3f} ± {np.nanstd(aucs):.3f}")
    return mean_auc


def evaluate(model, X_valid, y_valid, df_valid, cat_features=None):
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
    top25_over200 = (top25["kaishuu_rate"] >= 200).mean()

    print(f"\n--- 200%超確率 上位25%への出資シミュレーション ---")
    print(f"全頭平均回収率          : {all_avg:.1f}%")
    print(f"確率上位25%の平均回収率  : {top25_avg:.1f}%")
    print(f"差分                    : {top25_avg - all_avg:+.1f}%")
    print(f"全頭の200%超率          : {(df_eval['kaishuu_rate'] >= 200).mean():.1%}")
    print(f"確率上位25%の200%超率   : {top25_over200:.1%}")

    if top25_avg > all_avg:
        print("→ ✅ ランダムより有意に優れた選択ができています")
    else:
        print("→ ⚠️  ランダムと同等以下です")

    return df_eval, top25_avg, top25_over200


def save_feature_importance(model, feature_cols):
    fi = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(MODEL_DIR / "feature_importance_v4.csv", index=False, encoding="utf-8-sig")
    print("\n--- 特徴量重要度 TOP10 ---")
    print(fi.head(10).to_string(index=False))


# ──────────────────────────────────────────
# メイン
# ──────────────────────────────────────────

def main():
    print("=== データ読み込み ===")
    train, valid, other = load_data()

    print("\n=== 集計プール構築 ===")
    pool = build_pool(train, other)

    print("\n=== 集計特徴量の計算（プールデータから） ===")
    aggs = build_aggs(pool)

    # club_name エンコーダ（学習データに "silk" のみだが将来拡張可）
    all_clubs = sorted(["silk"] + other["club_name"].unique().tolist())
    club_encoder = {name: i for i, name in enumerate(all_clubs)}
    print(f"\nclub_name エンコード: {club_encoder}")

    print("\n=== 特徴量追加 ===")
    train = add_features(train, aggs, club_encoder)
    valid = add_features(valid, aggs, club_encoder)
    feature_cols = get_feature_cols(train)
    print(f"使用特徴量: {len(feature_cols)} 個")
    print(f"  {feature_cols}")

    X_train = train[feature_cols].fillna(-1)
    y_train = train["target"]
    X_valid = valid[feature_cols].fillna(-1)
    y_valid = valid["target"]

    # club_enc をカテゴリ列として指定
    cat_features = ["club_enc"] if "club_enc" in feature_cols else None

    class_weight = get_class_weight(y_train)

    print(f"\n=== クロスバリデーション (5fold) ===")
    cv_auc = cross_validate(X_train, y_train, class_weight, cat_features)

    print(f"\n=== 全学習データでモデル構築 ===")
    model = train_lgbm(X_train, y_train, class_weight, cat_features)

    print(f"\n=== 検証セットで評価 ===")
    df_eval, top25_avg, top25_over200 = evaluate(model, X_valid, y_valid, valid, cat_features)

    # ── v2 vs v4 比較 ──
    print("\n" + "="*50)
    print("=== v2 vs v4 比較 ===")
    print("="*50)
    print(f"{'指標':<30} {'v2':>10} {'v4':>10} {'差分':>10}")
    print("-" * 50)
    print(f"{'CV AUC':<30} {V2_CV_AUC:>10.3f} {cv_auc:>10.3f} {cv_auc - V2_CV_AUC:>+10.3f}")
    print(f"{'上位25%平均回収率 (%)':<30} {V2_TOP25_AVG:>10.1f} {top25_avg:>10.1f} {top25_avg - V2_TOP25_AVG:>+10.1f}")
    print(f"{'上位25%の200%超率':<30} {V2_TOP25_OVER200:>10.1%} {top25_over200:>10.1%} {top25_over200 - V2_TOP25_OVER200:>+10.1%}")
    print("="*50)

    # 保存
    joblib.dump({
        "model":        model,
        "feature_cols": feature_cols,
        "aggs":         aggs,
        "club_encoder": club_encoder,
        "cat_features": cat_features,
        "bins":         BINS,
        "labels":       LABELS,
        "label_names":  LABEL_NAMES,
    }, MODEL_DIR / "lgbm_model_v4.pkl")
    print(f"\nモデル保存: models/lgbm_model_v4.pkl")

    save_feature_importance(model, feature_cols)
    df_eval.to_csv(DATA_DIR / "valid_predictions_v4.csv", index=False, encoding="utf-8-sig")
    print(f"予測結果保存: data/valid_predictions_v4.csv")


if __name__ == "__main__":
    main()
