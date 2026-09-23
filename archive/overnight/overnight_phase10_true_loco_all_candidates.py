"""
overnight_phase10_true_loco_all_candidates.py
フェーズ9ではtclion追加のみ、11クラブ全部を未知として扱う真のLOCOで
検証した。同じ厳密さでgreenf単体・tclion+greenf同時追加も検証し、
3候補を公平に比較する（フェーズ7では6クラブ・10seedの簡易版でしか
比較していなかったため、tclionだけ証拠の質が高い状態だった）。

対象:
  A) なし（11クラブのみ、比較基準）
  B) tclionのみ追加
  C) greenfのみ追加
  D) tclion+greenf両方追加

各11クラブをleave-one-club-outで未知扱いし、5 seedで評価。

使い方:
  python overnight_phase10_true_loco_all_candidates.py
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
SEEDS = tuple(range(1, 6))

COLS = ["club_name", "sire", "trainer", "farm", "bms_name", "sex", "birth_month",
        "price_man", "kaishuu_rate", "kakutoku_man"]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def report(md):
    print(md, flush=True)
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(md + "\n")


def load_candidate(name):
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


def main():
    report("\n\n# フェーズ10: greenf・両方追加も含めた真のLOCO比較（公平な3候補比較）\n")
    report("フェーズ9はtclionのみ真のLOCOで検証した。ここではgreenf単体・"
           "tclion+greenf同時追加も同じ厳密さ（11クラブ全部を1つずつ未知扱い、"
           "5 seed）で検証し、3候補を公平に比較する。\n")
    log("=== フェーズ10開始 ===")

    pool_all = el.load_combined()
    for c in COLS:
        if c not in pool_all.columns:
            pool_all[c] = np.nan
    pool_all = pool_all[COLS]
    clubs = sorted(pool_all["club_name"].unique().tolist())

    tclion_df = load_candidate("tclion")
    greenf_df = load_candidate("greenf")

    extras = {
        "tclionのみ": tclion_df,
        "greenfのみ": greenf_df,
        "両方": pd.concat([tclion_df, greenf_df], ignore_index=True),
    }

    agg = {k: {"c": [], "d": []} for k in ["なし"] + list(extras.keys())}
    win_counts = {k: {"c": 0, "d": 0, "n": 0} for k in extras}

    report("| クラブ | n | spearman(なし) | spearman(tclion) | spearman(greenf) | spearman(両方) | "
           "diff(なし) | diff(tclion) | diff(greenf) | diff(両方) |")
    report("|---|---|---|---|---|---|---|---|---|---|")

    for club in clubs:
        held = pool_all[pool_all["club_name"] == club].reset_index(drop=True)
        rest = pool_all[pool_all["club_name"] != club].reset_index(drop=True)
        if len(held) < 10:
            continue
        log(f"held-out={club} (n={len(held)}) を検証中...")

        cs = {k: [] for k in ["なし"] + list(extras.keys())}
        ds = {k: [] for k in ["なし"] + list(extras.keys())}
        for seed in SEEDS:
            pools = {"なし": rest}
            for k, extra_df in extras.items():
                pools[k] = pd.concat([rest, extra_df], ignore_index=True)
            for k, pool in pools.items():
                m = train_ens(pool, seed)
                hp, rp = predict_clean(m, held)
                score = blended_score(hp, rp)
                actual = held["kaishuu_rate"].values
                cs[k].append(spearmanr(score, actual)[0])
                ds[k].append(_top25_diff(score, actual))

        means_c = {k: np.mean(v) for k, v in cs.items()}
        means_d = {k: np.mean(v) for k, v in ds.items()}
        report(f"| {club} | {len(held)} | {means_c['なし']:.3f} | {means_c['tclionのみ']:.3f} | "
               f"{means_c['greenfのみ']:.3f} | {means_c['両方']:.3f} | "
               f"{means_d['なし']:.1f} | {means_d['tclionのみ']:.1f} | "
               f"{means_d['greenfのみ']:.1f} | {means_d['両方']:.1f} |")

        for k in ["なし"] + list(extras.keys()):
            agg[k]["c"].append(means_c[k])
            agg[k]["d"].append(means_d[k])
        for k in extras:
            win_counts[k]["n"] += 1
            if means_c[k] > means_c["なし"]:
                win_counts[k]["c"] += 1
            if means_d[k] > means_d["なし"]:
                win_counts[k]["d"] += 1

    report("\n**11クラブ集計**:")
    for k in ["なし"] + list(extras.keys()):
        cs_all, ds_all = agg[k]["c"], agg[k]["d"]
        report(f"- {k}: spearman={np.mean(cs_all):.3f}±{np.std(cs_all):.3f}, "
               f"上位25%diff={np.mean(ds_all):.1f}±{np.std(ds_all):.1f}")

    report("\n**なしを上回ったクラブ数（11クラブ中）**:")
    for k in extras:
        wc, wd, n = win_counts[k]["c"], win_counts[k]["d"], win_counts[k]["n"]
        report(f"- {k}: spearman {wc}/{n}, 上位25%diff {wd}/{n}")

    report(f"\n完了: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log("=== フェーズ10完了 ===")


if __name__ == "__main__":
    main()
