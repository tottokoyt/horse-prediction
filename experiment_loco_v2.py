"""
experiment_loco_v2.py
モデルC（獲得賞金回帰）向けに、シルク検証で過去に「効果なし」と
判定した施策（年度拡張・mother_name・farm_trainer・nick）を
leave-one-club-out（LOCO = 未知クラブへの汎化性能の疑似検証）基準で
再検証する。

背景:
  train_model_v10〜v12, v14, train_model_B_v8 はすべて「シルク単独への
  予測精度」を物差しに他クラブデータ拡張・特徴量追加を評価しており、
  ことごとく悪化と判定されていた。しかしこのプロジェクトの本来の目的は
  シルクではなく未知のクラブ（DMMバヌーシー）への汎化なので、
  その物差し自体が違っていた可能性がある（experiment_loco.py参照）。
  ここでは同じ施策をLOCO基準で測り直す。

使い方:
  python experiment_loco_v2.py
"""

import re
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
TRAIN_YEARS = [2018, 2019, 2020, 2021]
ALL_YEARS   = [2018, 2019, 2020, 2021, 2022, 2023]
SEEDS = 3
CAP = 200.0


# ──────────────────────────────────────────
# データ読み込み・統合
# ──────────────────────────────────────────

def load_combined(years):
    silk = pd.read_csv(DATA_DIR / "merged_all.csv", encoding="utf-8-sig")
    silk = silk[silk["bosyu_year"].isin(years)].copy()
    silk["club_name"] = "silk"
    silk["price_man"] = pd.to_numeric(silk["total_price_man"], errors="coerce")
    if "farm" not in silk.columns:
        silk["farm"] = silk["farm_x"] if "farm_x" in silk.columns else silk.get("farm_y")

    other = pd.read_csv(DATA_DIR / "jisseki_other_all.csv", encoding="utf-8-sig")
    other = other[other["bosyu_year"].isin(years)].copy()
    other["price_man"] = pd.to_numeric(other["price_man"], errors="coerce")
    if "birth_month" not in other.columns:
        other["birth_month"] = np.nan

    bms = pd.read_csv(DATA_DIR / "bms_other_cache.csv", encoding="utf-8-sig")[["horse_id", "bms_name"]]
    other = pd.merge(other, bms, on="horse_id", how="left")
    sire = pd.read_csv(DATA_DIR / "sire_other_cache.csv", encoding="utf-8-sig")[["horse_id", "sire"]]
    other = pd.merge(other, sire, on="horse_id", how="left")

    mother_path = DATA_DIR / "mother_other_cache.csv"
    if mother_path.exists():
        mother = pd.read_csv(mother_path, encoding="utf-8-sig")[["horse_id", "mother_name"]]
        other = pd.merge(other, mother, on="horse_id", how="left")
    else:
        other["mother_name"] = np.nan
    if "mother_name" not in silk.columns:
        silk["mother_name"] = np.nan

    cols = ["club_name", "sire", "trainer", "farm", "bms_name", "mother_name", "sex", "birth_month",
            "price_man", "kaishuu_rate", "kakutoku_man"]
    combined = pd.concat([silk[cols], other[cols]], ignore_index=True)
    combined = combined.dropna(subset=["kaishuu_rate", "kakutoku_man", "price_man"]).reset_index(drop=True)
    return combined


# ──────────────────────────────────────────
# 特徴量エンジニアリング
# ──────────────────────────────────────────

def extract_birth_month(v):
    if pd.isna(v):
        return -1
    m = re.search(r"(\d+)月", str(v))
    if m:
        return int(m.group(1))
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


def add_nick(df):
    df = df.copy()
    has_both = df["sire"].notna() & df["bms_name"].notna()
    df["nick"] = np.where(has_both, df["sire"].astype(str) + "×" + df["bms_name"].astype(str), np.nan)
    return df


def add_farm_trainer(df):
    df = df.copy()
    has_both = df["farm"].notna() & df["trainer"].notna()
    df["farm_trainer"] = np.where(has_both, df["farm"].astype(str) + "×" + df["trainer"].astype(str), np.nan)
    return df


def smooth_mean(df, key_col, value_col, global_mean, min_samples=5):
    sub = df.dropna(subset=[key_col, value_col])
    if len(sub) == 0:
        return pd.DataFrame(columns=[key_col, f"{key_col}_smooth_mean", f"{key_col}_count"])
    g = sub.groupby(key_col)[value_col]
    stats = pd.DataFrame({"count": g.count(), "mean": g.mean()})
    stats[f"{key_col}_smooth_mean"] = (
        (stats["count"] * stats["mean"] + min_samples * global_mean) / (stats["count"] + min_samples)
    )
    stats[f"{key_col}_count"] = stats["count"]
    return stats[[f"{key_col}_smooth_mean", f"{key_col}_count"]].reset_index()


def build_aggs(train_df, extra_cols):
    aggs = {}
    base_cols = ["sire", "trainer", "farm", "bms_name"] + extra_cols
    for col in base_cols:
        gm = train_df["kaishuu_rate"].mean()
        result = smooth_mean(train_df, col, "kaishuu_rate", gm)
        if len(result) > 0:
            aggs[col] = result
    return aggs


def add_features(df, aggs):
    df = df.copy()
    df["birth_month"] = df["birth_month"].apply(extract_birth_month)
    df["sex_num"] = df["sex"].map({"牡": 0, "牝": 1, "セ": 2}).fillna(-1).astype(int)

    for col, stats_df in aggs.items():
        sc = stats_df.copy()
        df[col] = df[col].astype(object)
        sc[col] = sc[col].astype(object)
        df = pd.merge(df, sc, on=col, how="left")

    for col in df.columns:
        if col.endswith("_smooth_mean"):
            df[col] = df[col].fillna(df[col].median())
        elif col.endswith("_count"):
            df[col] = df[col].fillna(0)

    return df.reset_index(drop=True)


FEATURE_BASE = ["sex_num", "birth_month", "price_man"]


def get_feature_cols(df):
    agg = [c for c in df.columns if c.endswith(("_smooth_mean", "_count"))]
    return [c for c in FEATURE_BASE + agg if c in df.columns]


def eval_top25(test_df, score):
    d = test_df[["kaishuu_rate"]].copy()
    d["score"] = score
    d["kaishuu_rate_capped"] = d["kaishuu_rate"].clip(upper=CAP)
    thr = d["score"].quantile(0.75)
    top25 = d[d["score"] >= thr]
    return top25["kaishuu_rate_capped"].mean(), (top25["kaishuu_rate"] >= 200).mean()


def run_kakutoku_regressor(X_tr, y_tr, X_te, test_df, seed):
    model = lgb.LGBMRegressor(
        objective="huber", n_estimators=300, learning_rate=0.05, max_depth=4,
        num_leaves=15, min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0, random_state=seed, verbose=-1,
    )
    model.fit(X_tr, y_tr)
    score = model.predict(X_te)
    return eval_top25(test_df, score)


# ──────────────────────────────────────────
# 1つの設定でLOCOを実行
# ──────────────────────────────────────────

def run_loco(label, years, extra_cols, prep_fn=None):
    combined = load_combined(years)
    if prep_fn:
        combined = prep_fn(combined)
    clubs = sorted(combined["club_name"].unique())

    rows = []
    for club in clubs:
        train_df = combined[combined["club_name"] != club].copy()
        test_df  = combined[combined["club_name"] == club].copy()
        if len(test_df) < 20:
            continue

        aggs = build_aggs(train_df, extra_cols)
        tr = add_features(train_df, aggs)
        te = add_features(test_df, aggs)
        feature_cols = get_feature_cols(tr)

        X_tr = tr[feature_cols].fillna(-1)
        y_tr = tr["kakutoku_man"]
        X_te = te[feature_cols].fillna(-1)

        for seed in range(SEEDS):
            c, o = run_kakutoku_regressor(X_tr, y_tr, X_te, te, seed)
            rows.append((club, c, o))

    df = pd.DataFrame(rows, columns=["club", "capped25", "o200"])
    overall = df.groupby("club").agg({"capped25": "mean", "o200": "mean"})
    mean_c, std_c = overall["capped25"].mean(), overall["capped25"].std()
    mean_o, std_o = overall["o200"].mean(), overall["o200"].std()
    print(f"[{label}] capped25={mean_c:.1f}±{std_c:.1f}  200%超率={mean_o:.1%}±{std_o:.1%}  (対象クラブ{len(overall)})")
    return overall


def main():
    print("=== LOCO再検証（モデルC=kakutoku回帰） ===\n")

    run_loco("baseline(2018-2021)", TRAIN_YEARS, [])
    run_loco("年度拡張(2018-2023)", ALL_YEARS, [])
    run_loco("+farm_trainer(2018-2021)", TRAIN_YEARS, ["farm_trainer"], add_farm_trainer)
    run_loco("+nick(2018-2021)", TRAIN_YEARS, ["nick"], add_nick)

    # mother_nameは2018-2021のみ収集済み（silk側は元々mother_name列あり）
    def only_train_years_mother(df):
        return df
    run_loco("+mother_name(2018-2021)", TRAIN_YEARS, ["mother_name"])


if __name__ == "__main__":
    main()
