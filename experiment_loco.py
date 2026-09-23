"""
experiment_loco.py
leave-one-club-out（LOCO）検証実験

背景:
  このプロジェクトはもともとDMMバヌーシー（データがほぼ無い若いクラブ）の
  出資検討用に、他クラブのデータも集めて学習データを増やす狙いで作られた。
  シルクは単にデータが取りやすかったから検証データに使っていただけで、
  「シルクへの予測精度」が最終目的ではない。

  これまでの実験（train_model_v10〜v12, v14, train_model_B_v8）は
  すべて「シルク検証セットでの精度」を物差しにしており、他クラブデータを
  混ぜるとことごとく悪化するという結果だった。しかしこれは
  「シルク固有のクセを他クラブデータが薄める」ことの裏返しである可能性があり、
  本当に知りたい「未知のクラブ（バヌーシー）にどれだけ汎化するか」には
  別の検証方法が要る。

  本スクリプトでは、11クラブ（シルク+他10）を対象に leave-one-club-out
  （1クラブを除いて学習→そのクラブで評価、を全クラブでローテーション）で
  検証し、「未知のクラブへの汎化性能」を今あるデータで疑似的に測る。

  あわせて、目的変数を回収率（kaishuu_rate, クラブの値付けに依存する比率）
  ではなく、獲得賞金（kakutoku_man, 馬そのものの実力に近い絶対値）や
  勝ち上がり（wins>=1）にした場合の方が、クラブ間でより汎化するかも比較する。
  募集金額（price_man）は特徴量としてモデルに渡しているので、
  「獲得賞金の予測値／募集金額」で疑似的な回収率も算出できる。

データ範囲:
  全11クラブとも学習年度(2018-2021)のみ使用（年度拡張は既存実験で
  悪化することが分かっているため踏襲しない）。測尺は使わない
  （全クラブで揃っているのが sire/trainer/farm/bms_name/price_man/sex/
  birth_month のみのため、モデルB相当の特徴量セットで統一する）。

使い方:
  python experiment_loco.py
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
SEEDS = 3
CAP = 200.0


# ──────────────────────────────────────────
# データ読み込み・統合
# ──────────────────────────────────────────

def load_combined():
    silk = pd.read_csv(DATA_DIR / "merged_train.csv", encoding="utf-8-sig")
    silk = silk.copy()
    silk["club_name"] = "silk"
    silk["price_man"] = pd.to_numeric(silk["total_price_man"], errors="coerce")
    if "farm" not in silk.columns:
        silk["farm"] = silk["farm_x"] if "farm_x" in silk.columns else silk.get("farm_y")

    other = pd.read_csv(DATA_DIR / "jisseki_other_all.csv", encoding="utf-8-sig")
    other = other[other["bosyu_year"].isin(TRAIN_YEARS)].copy()
    other["price_man"] = pd.to_numeric(other["price_man"], errors="coerce")
    if "birth_month" not in other.columns:
        other["birth_month"] = np.nan

    bms = pd.read_csv(DATA_DIR / "bms_other_cache.csv", encoding="utf-8-sig")[["horse_id", "bms_name"]]
    other = pd.merge(other, bms, on="horse_id", how="left")

    sire = pd.read_csv(DATA_DIR / "sire_other_cache.csv", encoding="utf-8-sig")[["horse_id", "sire"]]
    other = pd.merge(other, sire, on="horse_id", how="left")

    cols = ["club_name", "sire", "trainer", "farm", "bms_name", "sex", "birth_month",
            "price_man", "kaishuu_rate", "kakutoku_man", "wins", "races"]
    silk_sub  = silk[cols].copy()
    other_sub = other[cols].copy()
    combined = pd.concat([silk_sub, other_sub], ignore_index=True)
    combined = combined.dropna(subset=["kaishuu_rate"]).reset_index(drop=True)
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


def build_aggs(train_df):
    """held-outクラブを含まないtrain_dfだけから集計統計を作る（リーク防止）"""
    aggs = {}
    for col, value_col in [("sire", "kaishuu_rate"), ("trainer", "kaishuu_rate"),
                            ("farm", "kaishuu_rate"), ("bms_name", "kaishuu_rate")]:
        gm = train_df[value_col].mean()
        result = smooth_mean(train_df, col, value_col, gm)
        if len(result) > 0:
            aggs[col] = result
    return aggs


def add_features(df, aggs, fallback_medians=None):
    """
    fallback_medians を渡さない場合（デフォルト）は従来通り、処理中の
    dfそのものの中央値で欠損を埋める。df が学習セット全体（数千頭）なら
    これで問題ないが、少数の検証フォールド（10頭程度）に対して呼ぶと
    「そのフォールドだけの数頭の中央値」という非常にノイズの大きい値に
    なる（2026-09-23未明に発見）。検証用に呼ぶ場合は、学習セットで
    計算した中央値を fallback_medians として明示的に渡すこと。
    """
    df = df.copy()
    df["birth_month"] = df["birth_month"].apply(extract_birth_month)
    df["sex_num"] = df["sex"].map({"牡": 0, "牝": 1, "セ": 2}).fillna(-1).astype(int)

    for col, stats_df in aggs.items():
        sc = stats_df.copy()
        df[col] = df[col].astype(object)
        sc[col] = sc[col].astype(object)
        df = pd.merge(df, sc, on=col, how="left")

    fallback_medians = fallback_medians or {}
    for col in df.columns:
        if col.endswith("_smooth_mean"):
            fill_value = fallback_medians.get(col, df[col].median())
            df[col] = df[col].fillna(fill_value)
        elif col.endswith("_count"):
            df[col] = df[col].fillna(0)

    return df.reset_index(drop=True)


def compute_fallback_medians(tr):
    """学習セット(add_features適用後のtr)から、検証時に使うべき中央値を取り出す"""
    return {c: float(tr[c].median()) for c in tr.columns if c.endswith("_smooth_mean")}


FEATURE_BASE = ["sex_num", "birth_month", "price_man"]


def get_feature_cols(df):
    agg = [c for c in df.columns if c.endswith(("_smooth_mean", "_count"))]
    return [c for c in FEATURE_BASE + agg if c in df.columns]


# ──────────────────────────────────────────
# 3種類のモデル
# ──────────────────────────────────────────

def eval_top25(test_df, score):
    d = test_df[["kaishuu_rate"]].copy()
    d["score"] = score
    d["kaishuu_rate_capped"] = d["kaishuu_rate"].clip(upper=CAP)
    thr = d["score"].quantile(0.75)
    top25 = d[d["score"] >= thr]
    return top25["kaishuu_rate_capped"].mean(), (top25["kaishuu_rate"] >= 200).mean()


def run_kaishuu_classifier(X_tr, y_kaishuu_tr, X_te, test_df, seed):
    target = pd.cut(y_kaishuu_tr, bins=[-1, 100, 200, 99999], labels=[0, 1, 2]).astype(int)
    counts = target.value_counts()
    cw = {c: len(target) / (len(counts) * n) for c, n in counts.items()}
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=15,
        min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0, class_weight=cw, random_state=seed,
        verbose=-1, objective="multiclass", num_class=3,
    )
    model.fit(X_tr, target, sample_weight=target.map(cw).values)
    score = model.predict_proba(X_te)[:, 2]
    return eval_top25(test_df, score)


def run_kakutoku_regressor(X_tr, y_kakutoku_tr, X_te, test_df, seed):
    model = lgb.LGBMRegressor(
        objective="huber", n_estimators=300, learning_rate=0.05, max_depth=4,
        num_leaves=15, min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0, random_state=seed, verbose=-1,
    )
    model.fit(X_tr, y_kakutoku_tr)
    score = model.predict(X_te)
    return eval_top25(test_df, score)


def run_win_classifier(X_tr, y_win_tr, X_te, test_df, seed):
    counts = y_win_tr.value_counts()
    cw = {c: len(y_win_tr) / (len(counts) * n) for c, n in counts.items()}
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=15,
        min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0, class_weight=cw, random_state=seed,
        verbose=-1, objective="binary",
    )
    model.fit(X_tr, y_win_tr, sample_weight=y_win_tr.map(cw).values)
    score = model.predict_proba(X_te)[:, 1]
    return eval_top25(test_df, score)


# ──────────────────────────────────────────
# メイン: leave-one-club-out
# ──────────────────────────────────────────

def main():
    print("=== データ読み込み・統合 ===")
    combined = load_combined()
    clubs = sorted(combined["club_name"].unique())
    print(f"対象クラブ ({len(clubs)}): {clubs}")
    print(combined.groupby("club_name").size().sort_values(ascending=False))

    results = {"kaishuu": [], "kakutoku": [], "win": []}

    for club in clubs:
        train_df = combined[combined["club_name"] != club].copy()
        test_df  = combined[combined["club_name"] == club].copy()
        if len(test_df) < 20:
            print(f"[スキップ] {club}: 対象頭数 {len(test_df)} が少なすぎる")
            continue

        aggs = build_aggs(train_df)
        tr = add_features(train_df, aggs)
        te = add_features(test_df, aggs)
        feature_cols = get_feature_cols(tr)

        X_tr = tr[feature_cols].fillna(-1)
        X_te = te[feature_cols].fillna(-1)
        y_kaishuu = tr["kaishuu_rate"]
        y_kakutoku = tr["kakutoku_man"]
        y_win = (tr["wins"] >= 1).astype(int)

        for seed in range(SEEDS):
            c1, o1 = run_kaishuu_classifier(X_tr, y_kaishuu, X_te, te, seed)
            c2, o2 = run_kakutoku_regressor(X_tr, y_kakutoku, X_te, te, seed)
            c3, o3 = run_win_classifier(X_tr, y_win, X_te, te, seed)
            results["kaishuu"].append((club, c1, o1))
            results["kakutoku"].append((club, c2, o2))
            results["win"].append((club, c3, o3))

        print(f"  {club} (test={len(test_df)}, train={len(train_df)}) 完了")

    print("\n" + "="*70)
    print("=== leave-one-club-out 結果（全クラブ平均、seed=%d） ===" % SEEDS)
    print("="*70)
    for name, label in [("kaishuu", "①kaishuu_rate 3クラス分類（現行方式）"),
                         ("kakutoku", "②kakutoku_man 回帰（huber）"),
                         ("win", "③勝ち上がり(wins>=1) 2値分類")]:
        df = pd.DataFrame(results[name], columns=["club", "capped25", "o200"])
        overall = df.groupby("club").agg({"capped25": "mean", "o200": "mean"})
        print(f"\n--- {label} ---")
        print(overall.to_string(float_format=lambda x: f"{x:.1f}"))
        print(f"全クラブ平均: capped上位25%={overall['capped25'].mean():.1f}±{overall['capped25'].std():.1f}  "
              f"200%超率={overall['o200'].mean():.1%}±{overall['o200'].std():.1%}")


if __name__ == "__main__":
    main()
