"""
overnight_phase2_candidate.py
phase1（overnight_expand_clubs.py）で「追加を検討」と判定された5クラブ
（laurel, kyoto, taiju, tclion, greenf）を、バヌーシーと同じ6seedの
CVで再検証し、確度の高いものだけを学習プールに加えた「候補モデル」を
作って検証する。本番モデルは上書きしない（models/lgbm_model_kakutoku.pkl
は変更せず、別名で保存する）。

候補モデルの検証は、追加しなかった残りのクラブ（yushun, turffight,
hiroo, ygg, waraukado, insel）＝正真正銘の未知クラブへの当てはまりで行う。

使い方:
  python overnight_phase2_candidate.py
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

sys.path.insert(0, "app")
import experiment_loco as el  # noqa: E402
from eval_utils import _top25_diff, compare_to_noise_floor  # noqa: E402
from train_model_kakutoku import make_relevance, load_combined as load_combined_c  # noqa: E402
import predictor  # noqa: E402

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_PATH = Path("overnight_report.md")
TRAIN_YEARS = [2018, 2019, 2020, 2021]

CANDIDATE_CLUBS = ["laurel", "kyoto", "taiju", "tclion", "greenf"]  # phase1で「追加を検討」
HOLDOUT_CLUBS   = ["yushun", "turffight", "hiroo", "ygg", "waraukado", "insel"]  # 未追加=真の検証用

RECHECK_SEEDS = (1, 2, 3, 4, 5, 6)  # バヌーシーと同じ6seed


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


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
    df["club_name"] = name
    cols = ["club_name", "sire", "trainer", "farm", "bms_name", "sex", "birth_month",
            "price_man", "kaishuu_rate", "kakutoku_man"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


def cv_recheck(name, club_common, known_pool, seeds=RECHECK_SEEDS):
    corrs, diffs = [], []
    for seed in seeds:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        idx = np.arange(len(club_common))
        all_pred, all_actual = [], []
        for train_idx, test_idx in kf.split(idx):
            c_tr, c_te = club_common.iloc[train_idx], club_common.iloc[test_idx]
            train_df = pd.concat([known_pool, c_tr], ignore_index=True).dropna(
                subset=["kaishuu_rate", "kakutoku_man", "price_man"]
            )
            aggs = el.build_aggs(train_df)
            tr = el.add_features(train_df, aggs)
            te = el.add_features(c_te, aggs)
            fc = el.get_feature_cols(tr)
            X_tr, X_te = tr[fc].fillna(-1), te[fc].fillna(-1)

            huber = lgb.LGBMRegressor(
                objective="huber", n_estimators=300, learning_rate=0.05, max_depth=4,
                num_leaves=15, min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=1.0, random_state=seed, verbose=-1,
            )
            huber.fit(X_tr, tr["kakutoku_man"])
            rank = lgb.LGBMRanker(
                objective="lambdarank", n_estimators=300, learning_rate=0.05, max_depth=4,
                num_leaves=15, min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=1.0, random_state=seed, verbose=-1,
            )
            rank.fit(X_tr, make_relevance(tr["kaishuu_rate"]), group=[len(X_tr)])

            hp, rp = huber.predict(X_te), rank.predict(X_te)
            ref_hp, ref_rp = huber.predict(X_tr), rank.predict(X_tr)
            ens = [(float((ref_hp < h).mean()) + float((ref_rp < r).mean())) / 2 for h, r in zip(hp, rp)]
            all_pred.extend(ens)
            all_actual.extend(c_te["kaishuu_rate"].tolist())

        all_pred, all_actual = np.array(all_pred), np.array(all_actual)
        corrs.append(spearmanr(all_pred, all_actual)[0])
        diffs.append(_top25_diff(all_pred, all_actual))
    return corrs, diffs


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
    ref_huber_preds = huber.predict(X_tr)
    ref_rank_scores = rank.predict(X_tr)
    return {
        "huber_model": huber, "rank_model": rank,
        "ref_huber_preds": ref_huber_preds, "ref_rank_scores": ref_rank_scores,
        "feature_cols": fc, "aggs": aggs, "model_type": "kakutoku_ensemble_candidate_v2",
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
    report("\n\n# フェーズ2: 候補モデル検証\n")
    log("=== フェーズ2開始 ===")

    known_pool_current = load_combined_c()  # 現行本番と同じ12クラブ(バヌーシー込み)

    club_data = {name: load_club_common(name) for name in CANDIDATE_CLUBS + HOLDOUT_CLUBS}

    report("## 5クラブの6seed再検証\n")
    report("| クラブ | n | spearman平均±std | 全seedで正か | 上位25%差分平均±std |")
    report("|---|---|---|---|---|")

    accepted = []
    for name in CANDIDATE_CLUBS:
        log(f"再検証中: {name} ({len(club_data[name])}頭, 6seed)...")
        corrs, diffs = cv_recheck(name, club_data[name], known_pool_current)
        all_positive = min(corrs) > 0
        report(f"| {name} | {len(club_data[name])} | {np.mean(corrs):.3f}±{np.std(corrs):.3f} | "
               f"{'○' if all_positive else '×'} | {np.mean(diffs):.1f}±{np.std(diffs):.1f} |")
        if all_positive:
            accepted.append(name)

    report(f"\n**6seed再検証後の採用クラブ**: {accepted if accepted else 'なし'}\n")
    log(f"採用クラブ: {accepted}")

    if not accepted:
        report("採用できるクラブが無かったため、候補モデルの作成をスキップします。")
        log("=== フェーズ2終了（候補なし） ===")
        return

    # 候補モデル学習
    log("候補モデルを学習中...")
    extra = pd.concat([club_data[name] for name in accepted], ignore_index=True)
    candidate_pool = pd.concat([known_pool_current, extra], ignore_index=True).dropna(
        subset=["kaishuu_rate", "kakutoku_man", "price_man"]
    )
    candidate_saved = train_ensemble(candidate_pool)
    joblib.dump(candidate_saved, MODEL_DIR / "lgbm_model_kakutoku_candidate_v2.pkl")
    report(f"候補モデル保存: models/lgbm_model_kakutoku_candidate_v2.pkl "
           f"（学習プール {len(candidate_pool)}頭 = 現行12クラブ + {accepted}）")

    # 現行本番モデルもロード（比較用）
    predictor.load_models()
    prod_saved = predictor._models["C"]

    report("\n## 候補モデル vs 現行本番モデル（正真正銘の未知クラブで比較）\n")
    report("| クラブ | n | 現行spearman | 候補spearman | 現行上位25%差分 | 候補上位25%差分 |")
    report("|---|---|---|---|---|---|")

    for name in HOLDOUT_CLUBS:
        df = club_data[name]
        if len(df) < 10:
            continue
        prod_pred = predict_with_model(prod_saved, df)
        cand_pred = predict_with_model(candidate_saved, df)
        actual = df["kaishuu_rate"].values
        prod_corr, _ = spearmanr(prod_pred, actual)
        cand_corr, _ = spearmanr(cand_pred, actual)
        prod_diff = _top25_diff(prod_pred, actual)
        cand_diff = _top25_diff(cand_pred, actual)
        report(f"| {name} | {len(df)} | {prod_corr:.3f} | {cand_corr:.3f} | {prod_diff:.1f} | {cand_diff:.1f} |")

    report(f"\n完了: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report("\n**この候補モデルは本番にデプロイされていません。** "
           "上記の比較結果を見て、良さそうであれば "
           "`cp models/lgbm_model_kakutoku_candidate_v2.pkl models/lgbm_model_kakutoku.pkl` "
           "で本番に反映しgit push・Renderデプロイしてください。")
    log("=== フェーズ2完了 ===")


if __name__ == "__main__":
    main()
