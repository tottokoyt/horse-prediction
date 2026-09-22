"""
validate_new_club.py
バヌーシー以外の新興・小規模クラブ（ワラウカド c122, インゼルTC c125等）
の実データで、デプロイ済みモデルCを検証する。

validate_banushi.py はバヌーシー自身が学習データに含まれるようになった
ため使えなくなった（README参照）。このスクリプトはモデルCの学習に
含まれていない、正真正銘の未知クラブでの検証に使う。

前提データ（club名ごとに用意）:
  data/jisseki_{club}.csv          (collect_jisseki_other.py の
                                     collect_club_year を再利用して収集)
  data/{club}_pedigree_cache.csv   (fetch_bms.py の extract_sire/extract_bms
                                     を再利用して収集)

使い方:
  python validate_new_club.py waraukado
  python validate_new_club.py insel
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, "app")
import predictor  # noqa: E402
from eval_utils import compare_to_noise_floor  # noqa: E402
from validate_banushi import build_known_club_results  # noqa: E402

DATA_DIR = Path("data")
TRAIN_YEARS = [2018, 2019, 2020, 2021]


def load_club(club: str, years=TRAIN_YEARS):
    jisseki = pd.read_csv(DATA_DIR / f"jisseki_{club}.csv", encoding="utf-8-sig")
    pedigree = pd.read_csv(DATA_DIR / f"{club}_pedigree_cache.csv", encoding="utf-8-sig")[
        ["horse_id", "sire", "bms_name"]
    ]
    df = pd.merge(jisseki, pedigree, on="horse_id", how="left")
    df = df.dropna(subset=["kaishuu_rate", "kakutoku_man", "price_man"])
    return df[df["bosyu_year"].isin(years)].reset_index(drop=True)


def predict_with_deployed_model(df):
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
        preds.append(predictor._ensemble_kakutoku(saved, X))
    return np.array(preds)


def main():
    if len(sys.argv) < 2:
        print("使い方: python validate_new_club.py <club名>  (例: waraukado, insel)")
        return
    club = sys.argv[1]

    print(f"=== {club} 実データ検証（vs 既知クラブのノイズフロア） ===\n")
    df = load_club(club)
    print(f"対象: {club} {len(df)} 頭（募集年度 {TRAIN_YEARS}, races平均={df['races'].mean():.1f}）")

    pred = predict_with_deployed_model(df)
    print("既知クラブのLOCO予測を計算中（ノイズフロア用）...")
    club_results = build_known_club_results()

    result = compare_to_noise_floor(club, pred, df["kaishuu_rate"].values, club_results)
    print("\n" + result["summary"])


if __name__ == "__main__":
    main()
