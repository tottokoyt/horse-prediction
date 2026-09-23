"""
experiment_scale_2022.py
学習年度を2018-2022に広げたモデルCについて、normandy・unionの2022年分の測尺
（2026-09-23夜に追加取得、92頭）を学習に入れた場合と入れない場合を比較する

背景:
  2022年追加（experiment_add_2022.py）の時点では、2022年分の測尺はシルクだけで、
  normandy・unionの2022年の馬は測尺が欠損扱いだった。
  fetch_normandy_scale.py / fetch_union_scale.py を2022年まで広げ、
  足りなかった母馬名（archive/fetch_mother_other.py の関数で100頭分）も取得して補った。

比較:
  2022測尺なし : normandy・unionの2022年の馬の測尺を欠損に戻したもの（直前の本番と同じ状態）
  2022測尺あり : 取得した値をそのまま使うもの
  評価は全クラブLOCO、各クラブの2018-2021年の馬（成績がほぼ確定）だけ、10 seed。

使い方:
  python experiment_scale_2022.py
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import train_model_kakutoku as tmk
from eval_utils import _top25_diff
from experiment_add_2022 import fit_and_score, BASE_YEARS

warnings.filterwarnings("ignore")

SEEDS = tuple(range(1, 11))
VARIANTS = ["2022測尺なし", "2022測尺あり"]
MEASURED_CLUBS = ["silk", "carrot", "normandy", "union"]


def main():
    print(f"[{datetime.now():%H:%M:%S}] 開始", flush=True)
    # 本番用の data/other_scale_cache.csv は2018-2021年分のまま（この実験で不採用と判断したため）。
    # 2022年分の候補は data/other_scale_cache_with2022.csv にあるので、そこから補う
    without_scale = tmk.load_combined()
    mask = without_scale["club_name"].isin(["normandy", "union"]) & (without_scale["bosyu_year"] == 2022)
    without_scale.loc[mask, tmk.SCALE_COLS] = np.nan
    cand = pd.read_csv(tmk.DATA_DIR / "other_scale_cache_with2022.csv", encoding="utf-8-sig")
    cand = cand[cand["bosyu_year"] == 2022][["horse_name", "bosyu_year"] + tmk.SCALE_COLS]
    with_scale = without_scale.copy()
    filled = with_scale.loc[mask, ["horse_name", "bosyu_year"]].merge(cand, on=["horse_name", "bosyu_year"], how="left")
    with_scale.loc[mask, tmk.SCALE_COLS] = filled[tmk.SCALE_COLS].values
    print(f"normandy・unionの2022年の馬 {int(mask.sum())} 頭のうち測尺あり {int(with_scale.loc[mask, 'height'].notna().sum())} 頭")
    print("測尺つきの頭数（学習プール全体）:",
          {"なし": int(without_scale["height"].notna().sum()), "あり": int(with_scale["height"].notna().sum())})

    pools = {"2022測尺なし": without_scale, "2022測尺あり": with_scale}
    clubs = [c for c, n in with_scale[with_scale["bosyu_year"].isin(BASE_YEARS)].groupby("club_name").size().items()
             if n >= 20]
    rows = []
    for club in clubs:
        for variant, pool in pools.items():
            test_df = pool[(pool["club_name"] == club) & pool["bosyu_year"].isin(BASE_YEARS)].reset_index(drop=True)
            train_df = pool[pool["club_name"] != club].reset_index(drop=True)
            actual = test_df["kaishuu_rate"].values
            for seed in SEEDS:
                score = fit_and_score(train_df, test_df, seed)
                rows.append({"club": club, "variant": variant, "seed": seed,
                             "spearman": spearmanr(score, actual)[0], "diff": _top25_diff(score, actual)})
        print(f"[{datetime.now():%H:%M:%S}] held-out={club} 完了", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(tmk.DATA_DIR / "experiment_scale_2022.csv", index=False, encoding="utf-8-sig")
    for metric in ["spearman", "diff"]:
        print(f"\n=== クラブ別 {metric}（{len(SEEDS)} seed 平均） ===")
        print(res.groupby(["club", "variant"])[metric].mean().unstack()[VARIANTS].round(3).to_string())
        for label, subset in [("測尺4クラブ", MEASURED_CLUBS), ("全クラブ", clubs)]:
            per_run = res[res["club"].isin(subset)].groupby(["seed", "variant"])[metric].mean().unstack()
            d = per_run["2022測尺あり"] - per_run["2022測尺なし"]
            print(f"  {label}: なし {per_run['2022測尺なし'].mean():.3f} / あり {per_run['2022測尺あり'].mean():.3f}"
                  f"  差 {d.mean():+.3f} ± {d.sem():.3f}(SE), 上回ったseed {int((d > 0).sum())}/{len(SEEDS)}")
    print(f"\n[{datetime.now():%H:%M:%S}] 完了", flush=True)


if __name__ == "__main__":
    main()
