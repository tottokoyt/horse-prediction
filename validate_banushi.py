"""
validate_banushi.py
デプロイ済みモデルC（またはexperiment_loco.py系の再学習モデル）を、
DMMバヌーシーの実データで検証し、eval_utils.compare_to_noise_floor()で
既知クラブのノイズフロアと比較する。

前提データ:
  data/jisseki_banushi.csv          (collect_jisseki_banushi.py)
  data/banushi_pedigree_cache.csv   (fetch_banushi_pedigree.py)

使い方:
  python validate_banushi.py
"""

import warnings
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, "app")
import predictor  # noqa: E402
import experiment_loco as el  # noqa: E402
from eval_utils import compare_to_noise_floor  # noqa: E402

DATA_DIR = Path("data")
TRAIN_YEARS = [2018, 2019, 2020, 2021]


def load_banushi(years=TRAIN_YEARS):
    jisseki = pd.read_csv(DATA_DIR / "jisseki_banushi.csv", encoding="utf-8-sig")
    pedigree = pd.read_csv(DATA_DIR / "banushi_pedigree_cache.csv", encoding="utf-8-sig")[
        ["horse_id", "sire", "bms_name"]
    ]
    df = pd.merge(jisseki, pedigree, on="horse_id", how="left")
    df = df.dropna(subset=["kaishuu_rate", "kakutoku_man", "price_man"])
    return df[df["bosyu_year"].isin(years)].reset_index(drop=True)


def predict_with_deployed_model(df):
    """デプロイ済み models/lgbm_model_kakutoku.pkl を使って予測する"""
    predictor.load_models()
    saved = predictor._models["C"]
    preds = []
    for _, row in df.iterrows():
        class Req:
            pass
        r = Req()
        r.sire, r.trainer, r.farm, r.bms = row["sire"], row["trainer"], row["farm"], row["bms_name"]
        r.sex, r.birth_month, r.price = row["sex"], None, row["price_man"]
        r.height = r.chest = r.cannon = r.weight = None
        X = predictor.build_feature_row(r, saved)
        preds.append(saved["model"].predict(X)[0])
    return np.array(preds)


def build_known_club_results(seed=42):
    """11クラブでleave-one-club-outし、known club_resultsを作る（ノイズフロア用）
    experiment_loco.py の load_combined() は内部で TRAIN_YEARS(2018-2021)に
    固定フィルタしているため、ここでも常に同じ年度範囲になる"""
    combined = el.load_combined()
    clubs = sorted(combined["club_name"].unique())
    results = {}
    for club in clubs:
        train_df = combined[combined["club_name"] != club].copy()
        test_df  = combined[combined["club_name"] == club].copy()
        if len(test_df) < 20:
            continue
        aggs = el.build_aggs(train_df)
        tr = el.add_features(train_df, aggs)
        te = el.add_features(test_df, aggs)
        fc = el.get_feature_cols(tr)
        X_tr, y_tr = tr[fc].fillna(-1), tr["kakutoku_man"]
        X_te = te[fc].fillna(-1)
        model = lgb.LGBMRegressor(
            objective="huber", n_estimators=300, learning_rate=0.05, max_depth=4,
            num_leaves=15, min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=1.0, random_state=seed, verbose=-1,
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        results[club] = (pred, te["kaishuu_rate"].values)
    return results


def main():
    print("=== バヌーシー実データ検証（vs 既知クラブのノイズフロア） ===\n")

    df = load_banushi()
    print(f"対象: バヌーシー {len(df)} 頭（募集年度 {TRAIN_YEARS}）")

    pred = predict_with_deployed_model(df)
    print("既知11クラブのLOCO予測を計算中（ノイズフロア用）...")
    club_results = build_known_club_results()

    result = compare_to_noise_floor("banushi", pred, df["kaishuu_rate"].values, club_results)
    print("\n" + result["summary"])

    print("""
判定の目安:
  パーセンタイル順位が概ね30〜70%なら「既知クラブと同程度のブレ・平均的」
  10%未満なら「既知クラブより明確に悪い」、90%超なら「明確に良い」
""")


if __name__ == "__main__":
    main()
