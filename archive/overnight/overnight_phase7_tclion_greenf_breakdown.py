"""
overnight_phase7_tclion_greenf_breakdown.py
フェーズ6-2で「tclion+greenfを一緒に追加すると6クラブへの汎化が改善する」
という、フェーズ2とは逆転した結果が出た。これがseed運なのか、
tclion単体/greenf単体でも同じ傾向が出るのかを確認する。

比較対象（すべて同じクリーン評価手法、10 seed）:
  A) なし（11クラブのみ）
  B) tclion のみ追加
  C) greenf のみ追加
  D) tclion + greenf 両方追加

評価は残り6クラブ（yushun/turffight/hiroo/ygg/waraukado/insel）。

使い方:
  python overnight_phase7_tclion_greenf_breakdown.py
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr, rankdata

warnings.filterwarnings("ignore")

sys.path.insert(0, "app")
import experiment_loco as el  # noqa: E402
from eval_utils import _top25_diff  # noqa: E402
from train_model_kakutoku import make_relevance  # noqa: E402

DATA_DIR = Path("data")
REPORT_PATH = Path("overnight_report.md")
TRAIN_YEARS = [2018, 2019, 2020, 2021]

HOLDOUT_CLUBS = ["yushun", "turffight", "hiroo", "ygg", "waraukado", "insel"]
SEEDS = tuple(range(1, 11))

COLS = ["club_name", "sire", "trainer", "farm", "bms_name", "sex", "birth_month",
        "price_man", "kaishuu_rate", "kakutoku_man"]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def report(md):
    print(md, flush=True)
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(md + "\n")


def load_club(name):
    jisseki = pd.read_csv(DATA_DIR / f"jisseki_{name}.csv", encoding="utf-8-sig")
    pedigree = pd.read_csv(DATA_DIR / f"{name}_pedigree_cache.csv", encoding="utf-8-sig")[
        ["horse_id", "sire", "bms_name"]
    ]
    df = pd.merge(jisseki, pedigree, on="horse_id", how="left")
    df = df.dropna(subset=["kaishuu_rate", "kakutoku_man", "price_man"])
    df = df[df["bosyu_year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    if "birth_month" not in df.columns:
        df["birth_month"] = np.nan
    df["club_name"] = name
    return df[COLS]


def train_ens(train_df, seed):
    aggs = el.build_aggs(train_df)
    tr = el.add_features(train_df, aggs)
    fc = el.get_feature_cols(tr)
    fallback_medians = el.compute_fallback_medians(tr)
    X_tr = tr[fc].fillna(-1)
    y_tr = tr["kakutoku_man"]
    y_rel = make_relevance(tr["kaishuu_rate"])
    huber = lgb.LGBMRegressor(
        objective="huber", n_estimators=300, learning_rate=0.05, max_depth=4,
        num_leaves=15, min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0, random_state=seed, verbose=-1,
    )
    huber.fit(X_tr, y_tr)
    rank = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=300, learning_rate=0.05, max_depth=4,
        num_leaves=15, min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0, random_state=seed, verbose=-1,
    )
    rank.fit(X_tr, y_rel, group=[len(X_tr)])
    return {"huber": huber, "rank": rank, "aggs": aggs, "fc": fc, "fallback_medians": fallback_medians}


def predict_clean(m, df):
    te = el.add_features(df, m["aggs"], m["fallback_medians"])
    X_te = te[m["fc"]].fillna(-1)
    return m["huber"].predict(X_te), m["rank"].predict(X_te)


def blended_score(hp, rp):
    return rankdata(hp) / len(hp) + rankdata(rp) / len(rp)


def eval_pool(pool, holdout, seed):
    m = train_ens(pool, seed)
    sum_c, sum_d = 0.0, 0.0
    for name, df in holdout.items():
        hp, rp = predict_clean(m, df)
        score = blended_score(hp, rp)
        actual = df["kaishuu_rate"].values
        sum_c += spearmanr(score, actual)[0]
        sum_d += _top25_diff(score, actual)
    return sum_c, sum_d


def main():
    report("\n\n# フェーズ7: tclion/greenf単体・組み合わせの内訳確認（10 seed）\n")
    log("=== フェーズ7開始 ===")

    pool_base = el.load_combined()
    for c in COLS:
        if c not in pool_base.columns:
            pool_base[c] = np.nan
    pool_base = pool_base[COLS]

    tclion_df = load_club("tclion")
    greenf_df = load_club("greenf")
    holdout = {name: load_club(name) for name in HOLDOUT_CLUBS}

    pools = {
        "A_なし": pool_base,
        "B_tclionのみ": pd.concat([pool_base, tclion_df], ignore_index=True),
        "C_greenfのみ": pd.concat([pool_base, greenf_df], ignore_index=True),
        "D_両方": pd.concat([pool_base, tclion_df, greenf_df], ignore_index=True),
    }

    results = {k: {"c": [], "d": []} for k in pools}
    for seed in SEEDS:
        log(f"seed={seed} 学習中...")
        for key, pool in pools.items():
            c, d = eval_pool(pool, holdout, seed)
            results[key]["c"].append(c)
            results[key]["d"].append(d)

    report("| パターン | spearman合計(平均±std) | 上位25%diff合計(平均±std) |")
    report("|---|---|---|")
    for key in pools:
        cs = results[key]["c"]
        ds = results[key]["d"]
        report(f"| {key} | {np.mean(cs):.3f}±{np.std(cs):.3f} | {np.mean(ds):.1f}±{np.std(ds):.1f} |")

    base_c = results["A_なし"]["c"]
    base_d = results["A_なし"]["d"]
    for key in ["B_tclionのみ", "C_greenfのみ", "D_両方"]:
        win_c = sum(1 for a, b in zip(base_c, results[key]["c"]) if b > a)
        win_d = sum(1 for a, b in zip(base_d, results[key]["d"]) if b > a)
        report(f"\n{key} が なし を上回った回数（10 seed中）: spearman {win_c}/10, 上位25%diff {win_d}/10")

    report(f"\n完了: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log("=== フェーズ7完了 ===")


if __name__ == "__main__":
    main()
