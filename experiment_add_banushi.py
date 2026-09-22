"""
experiment_add_banushi.py
バヌーシー自身の実データ（53頭、2018-2021年産・成熟済み）を、11クラブの
学習プールに追加した場合に効果があるか検証する。

方法:
  バヌーシー53頭を5-foldに分割し、各foldで
    学習 = 11クラブ + バヌーシーの残り4/5
    検証 = バヌーシーの残り1/5
  としてhuber+lambdarankアンサンブルを学習・評価する（全foldを通じて
  53頭全部が一度は検証に回る＝out-of-sample予測を53頭ぶん集められる）。

  これを「バヌーシーなしで学習した現行デプロイモデル」の
  バヌーシー53頭への予測（validate_banushi.pyで得られる）と比較する。

使い方:
  python experiment_add_banushi.py
"""

import warnings
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

sys.path.insert(0, "app")
import experiment_loco as el  # noqa: E402
from eval_utils import _top25_diff, compare_to_noise_floor  # noqa: E402
from train_model_kakutoku import make_relevance  # noqa: E402
from validate_banushi import load_banushi, predict_with_deployed_model, build_known_club_results  # noqa: E402

DATA_DIR = Path("data")
N_FOLDS = 5
SEED = 42


def load_banushi_full():
    df = load_banushi()  # TRAIN_YEARS(2018-2021)の53頭
    df = df.rename(columns={"bms_name": "bms_name"})
    df["club_name"] = "banushi"
    return df


def to_common_cols(df):
    cols = ["club_name", "sire", "trainer", "farm", "bms_name", "sex", "birth_month",
            "price_man", "kaishuu_rate", "kakutoku_man"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


def main():
    print("=== バヌーシー実データを学習に追加する効果を検証 ===\n")

    banushi = load_banushi_full()
    print(f"バヌーシー対象: {len(banushi)} 頭")

    known_combined = el.load_combined()  # 11クラブ、TRAIN_YEARS
    banushi_common = to_common_cols(banushi)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    idx = np.arange(len(banushi_common))

    all_pred, all_actual = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(idx)):
        banushi_train = banushi_common.iloc[train_idx]
        banushi_test  = banushi_common.iloc[test_idx]

        train_df = pd.concat([known_combined, banushi_train], ignore_index=True)
        train_df = train_df.dropna(subset=["kaishuu_rate", "kakutoku_man", "price_man"])

        aggs = el.build_aggs(train_df)
        tr = el.add_features(train_df, aggs)
        te = el.add_features(banushi_test, aggs)
        fc = el.get_feature_cols(tr)
        X_tr, X_te = tr[fc].fillna(-1), te[fc].fillna(-1)

        huber = lgb.LGBMRegressor(
            objective="huber", n_estimators=300, learning_rate=0.05, max_depth=4,
            num_leaves=15, min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=1.0, random_state=SEED, verbose=-1,
        )
        huber.fit(X_tr, tr["kakutoku_man"])
        rank = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=300, learning_rate=0.05, max_depth=4,
            num_leaves=15, min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=1.0, random_state=SEED, verbose=-1,
        )
        rank.fit(X_tr, make_relevance(tr["kaishuu_rate"]), group=[len(X_tr)])

        hp, rp = huber.predict(X_te), rank.predict(X_te)
        # foldごとのテスト頭数が10頭前後と少なく、fold内バッチランキングは
        # 不安定（別foldの順位と比較不能）。デプロイ済みモデルと同じ方式
        # （学習プール全体を参照分布にしたパーセンタイル順位の平均）で
        # 1頭ずつスコアを出し、fold間で比較可能にする
        ref_hp, ref_rp = huber.predict(X_tr), rank.predict(X_tr)
        ens_score = [
            (float((ref_hp < h).mean()) + float((ref_rp < r).mean())) / 2
            for h, r in zip(hp, rp)
        ]

        all_pred.extend(ens_score)
        all_actual.extend(banushi_test["kaishuu_rate"].tolist())
        print(f"  fold {fold+1}/{N_FOLDS}: train={len(train_df)}頭 (バヌーシー{len(banushi_train)}頭込み), test={len(banushi_test)}頭")

    all_pred = np.array(all_pred)
    all_actual = np.array(all_actual)

    diff = _top25_diff(all_pred, all_actual)
    corr, _ = spearmanr(all_pred, all_actual)
    print(f"\n[バヌーシー追加あり] n={len(all_actual)}  spearman={corr:.3f}  上位25%差分={diff:.1f}pt")

    print("\n=== 比較: 現行デプロイモデル（バヌーシーなしで学習）を同じ53頭で評価 ===")
    baseline_df = load_banushi()
    baseline_pred = predict_with_deployed_model(baseline_df)
    baseline_diff = _top25_diff(baseline_pred, baseline_df["kaishuu_rate"].values)
    baseline_corr, _ = spearmanr(baseline_pred, baseline_df["kaishuu_rate"].values)
    print(f"[バヌーシーなし(現行)] n={len(baseline_df)}  spearman={baseline_corr:.3f}  上位25%差分={baseline_diff:.1f}pt")

    print("\n=== ノイズフロアとの比較 ===")
    club_results = build_known_club_results()
    result_with = compare_to_noise_floor("追加あり", all_pred, all_actual, club_results)
    print("  " + result_with["summary"])
    result_without = compare_to_noise_floor("追加なし(現行)", baseline_pred, baseline_df["kaishuu_rate"].values, club_results)
    print("  " + result_without["summary"])


if __name__ == "__main__":
    main()
