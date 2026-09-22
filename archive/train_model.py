"""
train_model.py
シルクホースクラブ 回収率予測モデル

特徴量:
  - 父馬の過去回収率平均・中央値・200%超率
  - 厩舎の過去回収率平均・200%超率
  - 生産牧場の過去回収率平均
  - 馬体数値 (体高・胸囲・管囲・体重)
  - 性別
  - 生月 (早生まれ有利仮説)
  - 募集総額 (割安感の指標)

モデル:
  LightGBM による回帰 (回収率を直接予測)

出力:
  models/lgbm_model.pkl        学習済みモデル
  models/feature_importance.csv 特徴量重要度
  data/valid_predictions.csv   検証用予測結果

使い方:
  pip install lightgbm scikit-learn joblib
  python train_model.py
"""

import re
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────
# データ読み込み
# ──────────────────────────────────────────

def load_data():
    train = pd.read_csv(DATA_DIR / "merged_train.csv", encoding="utf-8-sig")
    valid = pd.read_csv(DATA_DIR / "merged_valid.csv", encoding="utf-8-sig")
    print(f"学習用: {len(train)} 頭 / 検証用: {len(valid)} 頭")
    return train, valid


# ──────────────────────────────────────────
# 特徴量エンジニアリング
# ──────────────────────────────────────────

def extract_birth_month(birth_str):
    """'1月9日' -> 1"""
    if pd.isna(birth_str):
        return None
    m = re.search(r"(\d+)月", str(birth_str))
    return int(m.group(1)) if m else None


def build_aggregation_features(train: pd.DataFrame) -> dict:
    """
    学習データから集計特徴量を計算する
    (リークを防ぐため、学習データのみから計算)
    戻り値: {feature_name: DataFrame(key -> stat)}
    """
    aggs = {}

    def calc_stats(df, key_col, value_col="kaishuu_rate"):
        g = df.groupby(key_col)[value_col]
        result = pd.DataFrame({
            f"{key_col}_mean":    g.mean(),
            f"{key_col}_median":  g.median(),
            f"{key_col}_std":     g.std().fillna(0),
            f"{key_col}_over200": (df[value_col] >= 200).groupby(df[key_col]).mean(),
            f"{key_col}_over100": (df[value_col] >= 100).groupby(df[key_col]).mean(),
            f"{key_col}_count":   g.count(),
        }).reset_index()
        return result

    # 父馬集計
    if "sire" in train.columns:
        aggs["sire"] = calc_stats(train.dropna(subset=["sire"]), "sire")

    # 厩舎集計
    if "trainer" in train.columns:
        aggs["trainer"] = calc_stats(train.dropna(subset=["trainer"]), "trainer")

    # 生産牧場集計
    if "farm" in train.columns:
        aggs["farm"] = calc_stats(train.dropna(subset=["farm"]), "farm")

    return aggs


def add_features(df: pd.DataFrame, aggs: dict, is_train: bool = True) -> pd.DataFrame:
    df = df.copy()

    # 生月
    df["birth_month"] = df["birth_month"].apply(extract_birth_month)

    # 性別を数値化
    sex_map = {"牡": 0, "牝": 1, "セ": 2}
    df["sex_num"] = df["sex"].map(sex_map).fillna(-1).astype(int)

    # 一口価格を万円に統一
    if "price_per_kuchi" in df.columns:
        df["price_per_kuchi_man"] = df["price_per_kuchi"] / 10000

    # 集計特徴量をマージ
    for key_col, stats_df in aggs.items():
        if key_col in df.columns:
            df = pd.merge(df, stats_df, on=key_col, how="left")

    # 集計値の欠損を全体平均で補完
    for col in df.columns:
        if col.endswith(("_mean", "_median", "_over200", "_over100")):
            df[col] = df[col].fillna(df[col].median())
        elif col.endswith("_count"):
            df[col] = df[col].fillna(0)

    return df


def get_feature_cols(df: pd.DataFrame):
    """予測に使う特徴量列を返す"""
    base_cols = [
        "sex_num",
        "birth_month",
        "total_price_man",
        "price_per_kuchi_man",
        "height",
        "chest",
        "cannon",
        "weight",
    ]
    agg_cols = [c for c in df.columns if c.endswith((
        "_mean", "_median", "_std", "_over200", "_over100", "_count"
    ))]
    all_cols = base_cols + agg_cols
    # 実際にdfに存在するもののみ
    return [c for c in all_cols if c in df.columns]


# ──────────────────────────────────────────
# モデル学習
# ──────────────────────────────────────────

def train_lgbm(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
        eval_set=[(X_train, y_train)],
    )
    return model


def cross_validate(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """KFold CVでモデルの安定性を確認"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes, r2s = [], []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model = train_lgbm(X_tr, y_tr)
        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        r2  = r2_score(y_val, pred)
        maes.append(mae)
        r2s.append(r2)
        print(f"  Fold {fold+1}: MAE={mae:.1f}  R2={r2:.3f}")
    print(f"  CV平均: MAE={np.mean(maes):.1f} ± {np.std(maes):.1f}  "
          f"R2={np.mean(r2s):.3f} ± {np.std(r2s):.3f}")


# ──────────────────────────────────────────
# 検証・評価
# ──────────────────────────────────────────

def evaluate(model, X_valid: pd.DataFrame, y_valid: pd.Series,
             df_valid: pd.DataFrame) -> pd.DataFrame:
    pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    r2  = r2_score(y_valid, pred)
    print(f"\n--- 検証セット評価 ---")
    print(f"MAE (平均絶対誤差): {mae:.1f}%")
    print(f"R2スコア          : {r2:.3f}")

    # ランダム出資との比較
    df_eval = df_valid[["horse_name", "bosyu_year", "kaishuu_rate"]].copy()
    df_eval["pred_kaishuu"] = pred

    threshold_25 = df_eval["pred_kaishuu"].quantile(0.75)
    top25 = df_eval[df_eval["pred_kaishuu"] >= threshold_25]
    all_avg   = df_eval["kaishuu_rate"].mean()
    top25_avg = top25["kaishuu_rate"].mean()

    print(f"\n--- ランダム出資との比較 ---")
    print(f"全頭平均回収率        : {all_avg:.1f}%")
    print(f"予測上位25%の平均回収率: {top25_avg:.1f}%")
    print(f"差分                  : {top25_avg - all_avg:+.1f}%")
    if top25_avg > all_avg:
        print("→ ✅ ランダムより有意に良い予測ができています")
    else:
        print("→ ⚠️  ランダムと同等以下です（特徴量・モデルの見直しが必要）")

    # カテゴリ別の予測精度
    print(f"\n--- 予測スコア四分位 × 実際の回収率 ---")
    df_eval["pred_quartile"] = pd.qcut(
        df_eval["pred_kaishuu"], q=4,
        labels=["予測Q1(低)", "予測Q2", "予測Q3", "予測Q4(高)"]
    )
    print(df_eval.groupby("pred_quartile")["kaishuu_rate"].agg(["mean", "median", "count"]))

    return df_eval


# ──────────────────────────────────────────
# 特徴量重要度
# ──────────────────────────────────────────

def save_feature_importance(model, feature_cols):
    fi = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(MODEL_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    print(f"\n--- 特徴量重要度 TOP10 ---")
    print(fi.head(10).to_string(index=False))


# ──────────────────────────────────────────
# メイン
# ──────────────────────────────────────────

def main():
    print("=== データ読み込み ===")
    train, valid = load_data()

    print("\n=== 集計特徴量の計算 (学習データのみ) ===")
    aggs = build_aggregation_features(train)
    for key, df_agg in aggs.items():
        print(f"  {key}: {len(df_agg)} 件")

    print("\n=== 特徴量追加 ===")
    train = add_features(train, aggs, is_train=True)
    valid = add_features(valid, aggs, is_train=False)
    feature_cols = get_feature_cols(train)
    print(f"  使用特徴量: {len(feature_cols)} 個")
    print(f"  {feature_cols}")

    X_train = train[feature_cols].fillna(-1)
    y_train = train["kaishuu_rate"]
    X_valid = valid[feature_cols].fillna(-1)
    y_valid = valid["kaishuu_rate"]

    print(f"\n=== クロスバリデーション (5fold) ===")
    cross_validate(X_train, y_train)

    print(f"\n=== 全学習データでモデル構築 ===")
    model = train_lgbm(X_train, y_train)

    print(f"\n=== 検証セットで評価 ===")
    df_eval = evaluate(model, X_valid, y_valid, valid)

    # 保存
    joblib.dump({
        "model":        model,
        "feature_cols": feature_cols,
        "aggs":         aggs,
    }, MODEL_DIR / "lgbm_model.pkl")
    print(f"\nモデル保存: models/lgbm_model.pkl")

    save_feature_importance(model, feature_cols)
    df_eval.to_csv(DATA_DIR / "valid_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"予測結果保存: data/valid_predictions.csv")


if __name__ == "__main__":
    main()
