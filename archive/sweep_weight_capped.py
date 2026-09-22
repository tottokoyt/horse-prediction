"""
sweep_weight_capped.py
train_model_v12.py の他クラブ行ダウンウェイト(OTHER_CLUB_WEIGHT)を
複数の値で振り、raw指標とcapped指標（回収率を200%で上限クリップ）の
両方で比較する一時的な分析スクリプト。

背景:
  上位25%平均回収率はサンプル数が少ない上に、200%超クラスに極端な
  外れ値（600%超など）が混ざるため、たった1頭の入れ替わりで数値が
  大きく動く。回収率を200%でキャップした「capped版」の指標も併記し、
  外れ値に引っ張られていないかを確認する。

使い方:
  python sweep_weight_capped.py
"""

import warnings
import numpy as np
import pandas as pd

import train_model_v12 as m

warnings.filterwarnings("ignore")

CAP = 200.0
WEIGHTS = [0.15, 0.3, 0.5, 0.7, 1.0]
N_SEEDS = 5  # 検証split自体の揺れを見るため複数シードでCVを回す


def evaluate_capped(model, X_val, df_val):
    pred  = model.predict(X_val)
    proba = model.predict_proba(X_val)

    df_eval = df_val[["horse_name", "bosyu_year", "kaishuu_rate"]].copy().reset_index(drop=True)
    df_eval["prob_class2"] = proba[:, 2]
    df_eval["kaishuu_rate_capped"] = df_eval["kaishuu_rate"].clip(upper=CAP)

    thr   = df_eval["prob_class2"].quantile(0.75)
    top25 = df_eval[df_eval["prob_class2"] >= thr]

    raw_avg    = top25["kaishuu_rate"].mean()
    capped_avg = top25["kaishuu_rate_capped"].mean()
    o200_rate  = (top25["kaishuu_rate"] >= 200).mean()
    return raw_avg, capped_avg, o200_rate, len(top25)


def main():
    print("=== データ準備（v12と同じロジック） ===")
    silk_train, silk_valid = m.load_silk()
    other_all = m.load_other()
    other_train = other_all[other_all["bosyu_year"].isin(m.TRAIN_YEARS)].copy()

    silk_train  = m.add_target(silk_train)
    silk_valid  = m.add_target(silk_valid)
    other_train = m.add_target(other_train)
    silk_train  = m.add_nick(silk_train)
    silk_valid  = m.add_nick(silk_valid)
    other_train = m.add_nick(other_train)

    pool = pd.concat([
        silk_train[["sire", "trainer", "farm", "kaishuu_rate"]],
        other_train[["sire", "trainer", "farm", "kaishuu_rate"]],
    ], ignore_index=True)
    bms_pool = pd.concat([
        silk_train[["bms_name", "kaishuu_rate"]],
        other_train[["bms_name", "kaishuu_rate"]],
    ], ignore_index=True)
    nick_pool = pd.concat([
        silk_train[["nick", "kaishuu_rate"]],
        other_train[["nick", "kaishuu_rate"]],
    ], ignore_index=True)
    club_pool = pd.concat([
        silk_train[["club_name", "kaishuu_rate"]],
        other_train[["club_name", "kaishuu_rate"]],
    ], ignore_index=True)
    aggs = m.build_aggs(pool, bms_pool, nick_pool, club_pool)

    other_train_scaled = other_train.dropna(subset=["height", "chest", "cannon", "weight"])
    tr_silk  = m.add_features(silk_train, aggs)
    tr_other = m.add_features(other_train_scaled, aggs)
    tr = pd.concat([tr_silk, tr_other], ignore_index=True)
    vl = m.add_features(silk_valid, aggs)

    feature_cols = m.get_feature_cols(tr)
    X_tr = tr[feature_cols].fillna(-1)
    y_tr = tr["target"]
    X_vl = vl[feature_cols].fillna(-1)

    print(f"実学習頭数: {len(tr)} (silk {len(tr_silk)} + other {len(tr_other)})")
    print(f"検証頭数: {len(vl)}\n")

    # v9相当（シルクのみ）のベースラインも比較用に計算
    cw_silk = m.get_class_weight(tr_silk["target"])
    baseline_scores = []
    for seed in range(N_SEEDS):
        model = m.lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=15,
            min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=1.0, class_weight=cw_silk,
            random_state=seed, verbose=-1, objective="multiclass", num_class=3,
        )
        X_tr_silk = tr_silk[feature_cols].fillna(-1)
        y_tr_silk = tr_silk["target"]
        model.fit(X_tr_silk, y_tr_silk, sample_weight=y_tr_silk.map(cw_silk).values)
        raw_avg, capped_avg, o200, n = evaluate_capped(model, X_vl, vl)
        baseline_scores.append((raw_avg, capped_avg, o200))
    b = np.array(baseline_scores)
    print(f"[v9相当 シルクのみ] raw={b[:,0].mean():.1f}±{b[:,0].std():.1f}  "
          f"capped={b[:,1].mean():.1f}±{b[:,1].std():.1f}  200%超率={b[:,2].mean():.1%}±{b[:,2].std():.1%}")

    print()
    for w in WEIGHTS:
        source_weight = np.where(tr["club_name"] == "silk", 1.0, w).astype(float)
        cw = m.get_class_weight(y_tr)
        scores = []
        for seed in range(N_SEEDS):
            model = m.lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=15,
                min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=1.0, class_weight=cw,
                random_state=seed, verbose=-1, objective="multiclass", num_class=3,
            )
            weight = y_tr.map(cw).values * source_weight
            model.fit(X_tr, y_tr, sample_weight=weight)
            raw_avg, capped_avg, o200, n = evaluate_capped(model, X_vl, vl)
            scores.append((raw_avg, capped_avg, o200))
        s = np.array(scores)
        print(f"[weight={w:<4}] raw={s[:,0].mean():6.1f}±{s[:,0].std():4.1f}  "
              f"capped={s[:,1].mean():6.1f}±{s[:,1].std():4.1f}  "
              f"200%超率={s[:,2].mean():.1%}±{s[:,2].std():.1%}")


if __name__ == "__main__":
    main()
