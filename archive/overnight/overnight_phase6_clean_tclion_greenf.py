"""
overnight_phase6_clean_tclion_greenf.py
フェーズ2で「tclion/greenfを学習に追加する」候補が有望に見えたが、
その評価は(1) predictor.pyのGLOBAL_MEANバグ、(2) experiment_loco.pyの
小バッチ中央値ノイズ、の両方の影響を受けていた可能性がある。

フェーズ6で確立した「クリーンな」評価手法（学習セット由来の
fallback_medians を検証データにも固定で使う）を使って、
tclion/greenf追加判断を再検証する。

比較対象は6クラブ（tclion/greenfを除いた残り）:
  yushun, turffight, hiroo, ygg, waraukado, insel

使い方:
  python overnight_phase6_clean_tclion_greenf.py
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
CANDIDATE_CLUBS = ["tclion", "greenf"]
SEEDS = (1, 2, 3, 4, 5)


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
    return df


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
    report("\n\n# フェーズ6-2: 両バグ修正後のクリーン再検証（tclion/greenf追加判断）\n")
    log("=== フェーズ6-2開始 ===")

    pool_without = el.load_combined()
    cand_dfs = {name: load_club(name) for name in CANDIDATE_CLUBS}
    cols = ["club_name", "sire", "trainer", "farm", "bms_name", "sex", "birth_month",
            "price_man", "kaishuu_rate", "kakutoku_man"]
    for c in cols:
        if c not in pool_without.columns:
            pool_without[c] = np.nan
    extra = []
    for name, df in cand_dfs.items():
        d = df.copy()
        for c in cols:
            if c not in d.columns:
                d[c] = np.nan
        d["club_name"] = name
        extra.append(d[cols])
    pool_with = pd.concat([pool_without[cols]] + extra, ignore_index=True)

    holdout = {name: load_club(name) for name in HOLDOUT_CLUBS}

    report("| seed | spearman合計(なし) | spearman合計(あり) | 上位25%diff合計(なし) | 上位25%diff合計(あり) |")
    report("|---|---|---|---|---|")

    wo_c, w_c, wo_d, w_d = [], [], [], []
    for seed in SEEDS:
        log(f"seed={seed} 学習中...")
        m_without = train_ens(pool_without, seed)
        m_with = train_ens(pool_with, seed)
        sum_c_wo = sum_c_w = sum_d_wo = sum_d_w = 0.0
        for name, df in holdout.items():
            hp_wo, rp_wo = predict_clean(m_without, df)
            hp_w, rp_w = predict_clean(m_with, df)
            score_wo = blended_score(hp_wo, rp_wo)
            score_w = blended_score(hp_w, rp_w)
            actual = df["kaishuu_rate"].values
            sum_c_wo += spearmanr(score_wo, actual)[0]
            sum_c_w += spearmanr(score_w, actual)[0]
            sum_d_wo += _top25_diff(score_wo, actual)
            sum_d_w += _top25_diff(score_w, actual)
        wo_c.append(sum_c_wo); w_c.append(sum_c_w)
        wo_d.append(sum_d_wo); w_d.append(sum_d_w)
        report(f"| {seed} | {sum_c_wo:.3f} | {sum_c_w:.3f} | {sum_d_wo:.1f} | {sum_d_w:.1f} |")

    report(f"\n**5 seed平均**: spearman合計 なし={np.mean(wo_c):.3f}±{np.std(wo_c):.3f} "
           f"あり={np.mean(w_c):.3f}±{np.std(w_c):.3f}　"
           f"／ 上位25%diff合計 なし={np.mean(wo_d):.1f}±{np.std(wo_d):.1f} "
           f"あり={np.mean(w_d):.1f}±{np.std(w_d):.1f}")
    win_c = sum(1 for a, b in zip(wo_c, w_c) if a > b)
    win_d = sum(1 for a, b in zip(wo_d, w_d) if a > b)
    report(f"\nなしが優勢だった回数: spearman {win_c}/{len(wo_c)}, 上位25%diff {win_d}/{len(wo_d)}")
    report(f"\n完了: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log("=== フェーズ6-2完了 ===")


if __name__ == "__main__":
    main()
