"""
experiment_add_2022.py
モデルCの学習データに2022年募集の世代を加えると、未知クラブへの予測が良くなるかを検証する

背景:
  学習年度は2018-2021。2022年募集（2021年産）の馬は2026年9月時点で5歳で、
  まだ走っている途中（平均出走数が成熟世代の約11戦に対し8〜9戦程度）。
  獲得賞金が「稼ぎ途中」のまま学習に入れると、世代全体を実力より低く学ぶおそれがある。

比較する3パターン:
  2018-2021       : 現状（本番と同じ）
  +2022           : 2022年を成績そのままで追加
  +2022(世代補正) : 稼ぎ途中の世代（2021・2022年）の獲得賞金・回収率に、
                    成熟世代（2018-2020）と log1p(獲得賞金) の平均がそろう倍率を掛けてから追加
                    （倍率は学習データ側だけから計算し、評価クラブの情報は使わない）

評価:
  全クラブを1つずつ未知扱い（LOCO）し、その**2018-2021年の馬**（成績がほぼ確定）だけで評価する。
  3パターンとも評価に使う馬は同じで、違うのは学習データだけ。SEEDS 回繰り返す。

使い方:
  python experiment_add_2022.py
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import train_model_kakutoku as tmk
from eval_utils import _top25_diff

warnings.filterwarnings("ignore")

SEEDS = tuple(range(1, 11))
BASE_YEARS = [2018, 2019, 2020, 2021]
MATURE_YEARS = [2018, 2019, 2020]
IMMATURE_YEARS = [2021, 2022]
VARIANTS = ["2018-2021", "+2022", "+2022(世代補正)"]


def cohort_adjust(train_df):
    """稼ぎ途中の世代の獲得賞金・回収率を、成熟世代と log1p 平均がそろうように底上げする"""
    df = train_df.copy()
    log_k = np.log1p(df["kakutoku_man"])
    ref = log_k[df["bosyu_year"].isin(MATURE_YEARS)].mean()
    factors = {}
    for y in IMMATURE_YEARS:
        m = df["bosyu_year"] == y
        if m.sum() == 0:
            continue
        f = float(np.exp(ref - log_k[m].mean()))
        factors[y] = f
        df.loc[m, "kakutoku_man"] = df.loc[m, "kakutoku_man"] * f
        df.loc[m, "kaishuu_rate"] = df.loc[m, "kaishuu_rate"] * f
    return df, factors


def fit_and_score(train_df, test_df, seed):
    aggs = tmk.build_aggs(train_df)
    tr = tmk.add_features(train_df, aggs)
    te = tmk.add_features(test_df, aggs)
    fc = tmk.get_feature_cols(tr)
    X_tr, X_te = tr[fc].fillna(-1), te[fc].fillna(-1)
    huber = tmk.train_huber(X_tr, np.log1p(tr["kakutoku_man"]), random_state_override=seed)
    rank = tmk.train_lambdarank(X_tr, tmk.make_relevance(tr["kaishuu_rate"]), random_state_override=seed)
    return tmk.ensemble_score_batch(huber.predict(X_te), rank.predict(X_te))


def main():
    print(f"[{datetime.now():%H:%M:%S}] 開始", flush=True)
    pool = tmk.load_combined(BASE_YEARS + [2022])
    print("世代別の頭数・平均出走数の代わりに平均log1p(獲得賞金):")
    print(pool.groupby("bosyu_year").agg(n=("kakutoku_man", "size"),
                                         mean_log_kakutoku=("kakutoku_man", lambda s: np.log1p(s).mean())).round(2))

    clubs = [c for c, n in pool[pool["bosyu_year"].isin(BASE_YEARS)].groupby("club_name").size().items() if n >= 20]
    rows, factor_log = [], []
    for club in clubs:
        test_df = pool[(pool["club_name"] == club) & pool["bosyu_year"].isin(BASE_YEARS)].reset_index(drop=True)
        others = pool[pool["club_name"] != club].reset_index(drop=True)
        actual = test_df["kaishuu_rate"].values
        adjusted, factors = cohort_adjust(others)
        factor_log.append(factors)
        trains = {
            "2018-2021": others[others["bosyu_year"].isin(BASE_YEARS)].reset_index(drop=True),
            "+2022": others,
            "+2022(世代補正)": adjusted,
        }
        for variant, train_df in trains.items():
            for seed in SEEDS:
                score = fit_and_score(train_df, test_df, seed)
                rows.append({"club": club, "n": len(test_df), "variant": variant, "seed": seed,
                             "spearman": spearmanr(score, actual)[0], "diff": _top25_diff(score, actual)})
        print(f"[{datetime.now():%H:%M:%S}] held-out={club} (n={len(test_df)}) 完了", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(tmk.DATA_DIR / "experiment_add_2022.csv", index=False, encoding="utf-8-sig")

    f = pd.DataFrame(factor_log)
    print(f"\n世代補正の倍率（クラブを除いた学習データごと、平均）: " +
          ", ".join(f"{y}年 ×{f[y].mean():.2f}" for y in f.columns))

    for metric in ["spearman", "diff"]:
        print(f"\n=== クラブ別 {metric}（{len(SEEDS)} seed 平均） ===")
        t = res.groupby(["club", "variant"])[metric].mean().unstack()[VARIANTS]
        t["n"] = res.groupby("club")["n"].first()
        print(t.round(3).to_string())

        per_run = res.groupby(["seed", "variant"])[metric].mean().unstack()[VARIANTS]
        print(f"\n=== {metric}: 全クラブ平均 ===")
        for v in VARIANTS:
            print(f"  {v}: {per_run[v].mean():.3f} ± {per_run[v].sem():.3f}(SE)")
        club_mean = res.groupby(["club", "variant"])[metric].mean().unstack()
        for v in VARIANTS[1:]:
            d = per_run[v] - per_run["2018-2021"]
            wins = int((club_mean[v] > club_mean["2018-2021"]).sum())
            print(f"  {v} − 2018-2021（同じseedで対応あり）: {d.mean():+.3f} ± {d.sem():.3f}(SE), "
                  f"上回ったseed {int((d > 0).sum())}/{len(SEEDS)}, 上回ったクラブ {wins}/{len(club_mean)}")

    print(f"\n[{datetime.now():%H:%M:%S}] 完了", flush=True)


if __name__ == "__main__":
    main()
