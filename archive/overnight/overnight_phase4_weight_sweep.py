"""
overnight_phase4_weight_sweep.py
フェーズ3で「バヌーシーを学習に全部混ぜる」と他クラブへの汎化が
犠牲になることが分かった。全部混ぜる/全く混ぜないの二択ではなく、
バヌーシー行のsample_weightを下げた「部分的な混ぜ方」で
①バヌーシー自身への予測改善と②他クラブへの汎化維持を両立できないか
weightを振って探索する。

評価は複数seedで、
  (a) バヌーシー自身の5-fold CV（本来のCVの目的=自分自身への予測改善）
  (b) 他クラブ8つでの汎化性能（フェーズ3と同じ8クラブ集合）
の両方を見る。

使い方:
  python overnight_phase4_weight_sweep.py
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

sys.path.insert(0, "app")
import experiment_loco as el  # noqa: E402
from eval_utils import _top25_diff  # noqa: E402
from train_model_kakutoku import make_relevance  # noqa: E402
import predictor  # noqa: E402

DATA_DIR = Path("data")
REPORT_PATH = Path("overnight_report.md")
TRAIN_YEARS = [2018, 2019, 2020, 2021]

OTHER_CHECK_CLUBS = ["yushun", "turffight", "hiroo", "ygg", "waraukado", "insel", "tclion", "greenf"]
WEIGHTS = [0.0, 0.3, 0.5, 0.7, 1.0]
SEEDS = (1, 2, 3, 4, 5)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def report(md):
    print(md, flush=True)
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(md + "\n")


def load_club_common(name, years=TRAIN_YEARS):
    jisseki = pd.read_csv(DATA_DIR / f"jisseki_{name}.csv", encoding="utf-8-sig")
    pedigree = pd.read_csv(DATA_DIR / f"{name}_pedigree_cache.csv", encoding="utf-8-sig")[
        ["horse_id", "sire", "bms_name"]
    ]
    df = pd.merge(jisseki, pedigree, on="horse_id", how="left")
    df = df.dropna(subset=["kaishuu_rate", "kakutoku_man", "price_man"])
    df = df[df["bosyu_year"].isin(years)].reset_index(drop=True)
    df["club_name"] = "banushi"
    return df


def load_banushi_common():
    jisseki = pd.read_csv(DATA_DIR / "jisseki_banushi.csv", encoding="utf-8-sig")
    pedigree = pd.read_csv(DATA_DIR / "banushi_pedigree_cache.csv", encoding="utf-8-sig")[
        ["horse_id", "sire", "bms_name"]
    ]
    df = pd.merge(jisseki, pedigree, on="horse_id", how="left")
    df = df.dropna(subset=["kaishuu_rate", "kakutoku_man", "price_man"])
    df = df[df["bosyu_year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    df["club_name"] = "banushi"
    return df


def train_weighted_ensemble(pool_11, banushi_train, weight, seed):
    """11クラブ(weight=1.0) + バヌーシー(指定weight)で学習"""
    cols = ["club_name", "sire", "trainer", "farm", "bms_name", "sex", "birth_month",
            "price_man", "kaishuu_rate", "kakutoku_man"]
    for c in cols:
        if c not in banushi_train.columns:
            banushi_train[c] = np.nan
    train_df = pd.concat([pool_11, banushi_train[cols]], ignore_index=True)
    train_df = train_df.dropna(subset=["kaishuu_rate", "kakutoku_man", "price_man"]).reset_index(drop=True)

    aggs = el.build_aggs(train_df)
    tr = el.add_features(train_df, aggs)
    fc = el.get_feature_cols(tr)
    X_tr = tr[fc].fillna(-1)
    y_tr = tr["kakutoku_man"]
    y_rel = make_relevance(tr["kaishuu_rate"])
    sw = np.where(tr["club_name"] == "banushi", weight, 1.0)

    huber = lgb.LGBMRegressor(
        objective="huber", n_estimators=300, learning_rate=0.05, max_depth=4,
        num_leaves=15, min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0, random_state=seed, verbose=-1,
    )
    huber.fit(X_tr, y_tr, sample_weight=sw)
    rank = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=300, learning_rate=0.05, max_depth=4,
        num_leaves=15, min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0, random_state=seed, verbose=-1,
    )
    rank.fit(X_tr, y_rel, group=[len(X_tr)], sample_weight=sw)
    return {
        "huber_model": huber, "rank_model": rank,
        "ref_huber_preds": huber.predict(X_tr), "ref_rank_scores": rank.predict(X_tr),
        "feature_cols": fc, "aggs": aggs,
    }


def predict_with_model(saved, df):
    preds = []
    for _, row in df.iterrows():
        class Req:
            pass
        r = Req()
        r.sire, r.trainer, r.farm, r.bms = row["sire"], row["trainer"], row["farm"], row["bms_name"]
        r.sex, r.birth_month, r.price = row["sex"], None, row["price_man"]
        r.height = r.chest = r.cannon = r.weight = None
        X = predictor.build_feature_row(r, saved)
        hp = float(saved["huber_model"].predict(X)[0])
        rp = float(saved["rank_model"].predict(X)[0])
        hpct = float((saved["ref_huber_preds"] < hp).mean())
        rpct = float((saved["ref_rank_scores"] < rp).mean())
        preds.append(float(np.quantile(saved["ref_huber_preds"], (hpct + rpct) / 2)))
    return np.array(preds)


def main():
    report("\n\n# フェーズ4: バヌーシーのsample_weightを振った探索\n")
    log("=== フェーズ4開始 ===")

    pool_11 = el.load_combined()
    banushi_all = load_banushi_common()
    other_clubs = {}
    for name in OTHER_CHECK_CLUBS:
        jisseki = pd.read_csv(DATA_DIR / f"jisseki_{name}.csv", encoding="utf-8-sig")
        pedigree = pd.read_csv(DATA_DIR / f"{name}_pedigree_cache.csv", encoding="utf-8-sig")[
            ["horse_id", "sire", "bms_name"]
        ]
        df = pd.merge(jisseki, pedigree, on="horse_id", how="left")
        df = df.dropna(subset=["kaishuu_rate", "kakutoku_man", "price_man"])
        other_clubs[name] = df[df["bosyu_year"].isin(TRAIN_YEARS)].reset_index(drop=True)

    report("| weight | バヌーシーCV spearman | 他8クラブ spearman合計 | 他8クラブ diff合計 |")
    report("|---|---|---|---|")

    for weight in WEIGHTS:
        log(f"weight={weight} を検証中...")
        banushi_cv_corrs = []
        other_corrs_sum, other_diffs_sum = [], []

        for seed in SEEDS:
            # (a) バヌーシー自身の5-fold CV
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            idx = np.arange(len(banushi_all))
            cv_pred, cv_actual = [], []
            for train_idx, test_idx in kf.split(idx):
                b_tr, b_te = banushi_all.iloc[train_idx], banushi_all.iloc[test_idx]
                saved = train_weighted_ensemble(pool_11, b_tr, weight, seed)
                p = predict_with_model(saved, b_te)
                cv_pred.extend(p.tolist())
                cv_actual.extend(b_te["kaishuu_rate"].tolist())
            banushi_cv_corrs.append(spearmanr(cv_pred, cv_actual)[0])

            # (b) 他8クラブへの汎化（バヌーシー全部を指定weightで学習に使う）
            saved_full = train_weighted_ensemble(pool_11, banushi_all, weight, seed)
            sum_c, sum_d = 0.0, 0.0
            for name, df in other_clubs.items():
                if len(df) < 10:
                    continue
                p = predict_with_model(saved_full, df)
                actual = df["kaishuu_rate"].values
                sum_c += spearmanr(p, actual)[0]
                sum_d += _top25_diff(p, actual)
            other_corrs_sum.append(sum_c)
            other_diffs_sum.append(sum_d)

        report(f"| {weight} | {np.mean(banushi_cv_corrs):.3f}±{np.std(banushi_cv_corrs):.3f} | "
               f"{np.mean(other_corrs_sum):.3f}±{np.std(other_corrs_sum):.3f} | "
               f"{np.mean(other_diffs_sum):.1f}±{np.std(other_diffs_sum):.1f} |")

    report(f"\n完了: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report("\n参考: weight=0.0が「バヌーシーなし」相当、weight=1.0が「現行本番」相当。"
           "中間のweightで、バヌーシーCV相関を大きく落とさずに他クラブへの汎化を"
           "取り戻せる点があれば、それが両立点の候補。")
    log("=== フェーズ4完了 ===")


if __name__ == "__main__":
    main()
