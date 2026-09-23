"""
overnight_phase3_recheck_banushi.py
phase2で「tclion/greenfは自分自身の保留フォールドでは良く見えても、
他クラブへの汎化はむしろ悪化する」ことが分かった。同じ基準で、
本番採用済みの「バヌーシーを学習に追加する」判断も、より広い未知クラブ
集合（今夜取得した8クラブ）で再チェックする。

バヌーシーは本来のターゲットドメインなので、他クラブへの汎化が
多少犠牲になっても含める価値がある可能性はあるが、実際どの程度の
トレードオフなのかを定量的に見ておく。

使い方:
  python overnight_phase3_recheck_banushi.py
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
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

CHECK_CLUBS = ["yushun", "turffight", "hiroo", "ygg", "waraukado", "insel", "tclion", "greenf"]


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
    return df[df["bosyu_year"].isin(years)].reset_index(drop=True)


def train_ensemble(train_df, seed=42):
    aggs = el.build_aggs(train_df)
    tr = el.add_features(train_df, aggs)
    fc = el.get_feature_cols(tr)
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
        huber_pred = float(saved["huber_model"].predict(X)[0])
        rank_pred = float(saved["rank_model"].predict(X)[0])
        huber_pct = float((saved["ref_huber_preds"] < huber_pred).mean())
        rank_pct = float((saved["ref_rank_scores"] < rank_pred).mean())
        ens_pct = (huber_pct + rank_pct) / 2
        preds.append(float(np.quantile(saved["ref_huber_preds"], ens_pct)))
    return np.array(preds)


def main():
    report("\n\n# フェーズ3: バヌーシー追加判断の再チェック（広い未知クラブ集合で）\n")
    log("=== フェーズ3開始 ===")

    log("バヌーシーなし(11クラブ)モデルを学習中...")
    pool_without_banushi = el.load_combined()
    model_without = train_ensemble(pool_without_banushi)

    log("バヌーシーあり(12クラブ、現行本番相当)モデルを学習中...")
    predictor.load_models()
    model_with = predictor._models["C"]  # 現行デプロイ済みモデル（12クラブ、アンサンブル込み）

    report("## 「バヌーシーなし」vs「バヌーシーあり(現行本番)」を8クラブで比較\n")
    report("| クラブ | n | なしspearman | ありspearman | なし上位25%差分 | あり上位25%差分 |")
    report("|---|---|---|---|---|---|")

    without_corrs, with_corrs = [], []
    without_diffs, with_diffs = [], []

    for name in CHECK_CLUBS:
        df = load_club_common(name)
        if len(df) < 10:
            continue
        pred_without = predict_with_model(model_without, df)
        # 現行本番モデルの予測にはpredictor._ensemble_kakutokuを使う（学習プール参照分布込み）
        preds_with = []
        for _, row in df.iterrows():
            class Req:
                pass
            r = Req()
            r.sire, r.trainer, r.farm, r.bms = row["sire"], row["trainer"], row["farm"], row["bms_name"]
            r.sex, r.birth_month, r.price = row["sex"], None, row["price_man"]
            r.height = r.chest = r.cannon = r.weight = None
            X = predictor.build_feature_row(r, model_with)
            preds_with.append(predictor._ensemble_kakutoku(model_with, X))
        preds_with = np.array(preds_with)

        actual = df["kaishuu_rate"].values
        c_without, _ = spearmanr(pred_without, actual)
        c_with, _ = spearmanr(preds_with, actual)
        d_without = _top25_diff(pred_without, actual)
        d_with = _top25_diff(preds_with, actual)

        without_corrs.append(c_without); with_corrs.append(c_with)
        without_diffs.append(d_without); with_diffs.append(d_with)

        report(f"| {name} | {len(df)} | {c_without:.3f} | {c_with:.3f} | {d_without:.1f} | {d_with:.1f} |")

    report(f"\n**集計**: spearman合計 なし={np.sum(without_corrs):.3f} vs あり={np.sum(with_corrs):.3f}　"
           f"／ 上位25%差分合計 なし={np.sum(without_diffs):.1f} vs あり={np.sum(with_diffs):.1f}")
    report(f"\n完了: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log("=== フェーズ3完了 ===")


if __name__ == "__main__":
    main()
