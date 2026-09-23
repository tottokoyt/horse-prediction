"""
overnight_phase9_true_loco_tclion.py
フェーズ6〜7では「同じ6クラブ（yushun/turffight/hiroo/ygg/waraukado/insel）」
を繰り返し評価に使ってtclion追加の効果を見てきた。同じ検証セットを
何度も見て判断する（多重比較的な選択バイアス）リスクがあるため、
本当のleave-one-club-out（LOCO）で、既知の11クラブ（silk+other10）
それぞれを1つずつ完全に未知として扱い、"tclionを学習に足すと
未知クラブへの汎化が改善するか"をより広く検証する。

各held-outクラブについて:
  A) 残り10クラブのみで学習 → held-outクラブで評価
  B) 残り10クラブ + tclion で学習 → held-outクラブで評価
を複数seedで行い、A/Bどちらが良いかを11クラブ全体で集計する。

使い方:
  python overnight_phase9_true_loco_tclion.py
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


def load_tclion():
    jisseki = pd.read_csv(DATA_DIR / "jisseki_tclion.csv", encoding="utf-8-sig")
    pedigree = pd.read_csv(DATA_DIR / "tclion_pedigree_cache.csv", encoding="utf-8-sig")[
        ["horse_id", "sire", "bms_name"]
    ]
    df = pd.merge(jisseki, pedigree, on="horse_id", how="left")
    df = df.dropna(subset=["kaishuu_rate", "kakutoku_man", "price_man"])
    df = df[df["bosyu_year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    if "birth_month" not in df.columns:
        df["birth_month"] = np.nan
    df["club_name"] = "tclion"
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
    report("\n\n# フェーズ9: tclion追加判断の真のLOCO検証（11クラブ全部を1つずつ未知扱い）\n")
    report("フェーズ6〜7で使った6クラブだけでなく、既知の11クラブ"
           "（silk含む）それぞれをleave-one-club-outで未知クラブとして扱い、"
           "選択バイアスを排除してtclion追加の効果を再確認する。\n")
    log("=== フェーズ9開始 ===")

    pool_all = el.load_combined()
    for c in COLS:
        if c not in pool_all.columns:
            pool_all[c] = np.nan
    pool_all = pool_all[COLS]
    clubs = sorted(pool_all["club_name"].unique().tolist())
    log(f"対象クラブ（{len(clubs)}）: {clubs}")

    tclion_df = load_tclion()

    report("| クラブ | n | spearman(なし) | spearman(tclion追加) | 上位25%diff(なし) | 上位25%diff(tclion追加) |")
    report("|---|---|---|---|---|---|")

    win_c_total, win_d_total, n_evals = 0, 0, 0
    agg_wo_c, agg_w_c, agg_wo_d, agg_w_d = [], [], [], []

    for club in clubs:
        held = pool_all[pool_all["club_name"] == club].reset_index(drop=True)
        rest = pool_all[pool_all["club_name"] != club].reset_index(drop=True)
        if len(held) < 10:
            continue
        log(f"held-out={club} (n={len(held)}) を検証中...")

        cs_wo, cs_w, ds_wo, ds_w = [], [], [], []
        for seed in SEEDS:
            m_wo = train_ens(rest, seed)
            m_w = train_ens(pd.concat([rest, tclion_df], ignore_index=True), seed)
            hp_wo, rp_wo = predict_clean(m_wo, held)
            hp_w, rp_w = predict_clean(m_w, held)
            score_wo = blended_score(hp_wo, rp_wo)
            score_w = blended_score(hp_w, rp_w)
            actual = held["kaishuu_rate"].values
            cs_wo.append(spearmanr(score_wo, actual)[0])
            cs_w.append(spearmanr(score_w, actual)[0])
            ds_wo.append(_top25_diff(score_wo, actual))
            ds_w.append(_top25_diff(score_w, actual))

        m_cs_wo, m_cs_w = np.mean(cs_wo), np.mean(cs_w)
        m_ds_wo, m_ds_w = np.mean(ds_wo), np.mean(ds_w)
        report(f"| {club} | {len(held)} | {m_cs_wo:.3f} | {m_cs_w:.3f} | {m_ds_wo:.1f} | {m_ds_w:.1f} |")

        agg_wo_c.append(m_cs_wo); agg_w_c.append(m_cs_w)
        agg_wo_d.append(m_ds_wo); agg_w_d.append(m_ds_w)
        if m_cs_w > m_cs_wo:
            win_c_total += 1
        if m_ds_w > m_ds_wo:
            win_d_total += 1
        n_evals += 1

    report(f"\n**{n_evals}クラブ集計（各クラブseed平均のさらに平均）**: "
           f"spearman なし={np.mean(agg_wo_c):.3f}±{np.std(agg_wo_c):.3f} "
           f"tclion追加={np.mean(agg_w_c):.3f}±{np.std(agg_w_c):.3f}　"
           f"／ 上位25%diff なし={np.mean(agg_wo_d):.1f}±{np.std(agg_wo_d):.1f} "
           f"tclion追加={np.mean(agg_w_d):.1f}±{np.std(agg_w_d):.1f}")
    report(f"\ntclion追加がなしを上回ったクラブ数: spearman {win_c_total}/{n_evals}, "
           f"上位25%diff {win_d_total}/{n_evals}")

    report(f"\n完了: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log("=== フェーズ9完了 ===")


if __name__ == "__main__":
    main()
