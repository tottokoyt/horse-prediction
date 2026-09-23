"""
experiment_banushi_oos.py
現行のモデルC（log1p huber + lambdarank、測尺入り、train_model_kakutoku.py と同じ学習方法）で、
DMMバヌーシー53頭への予測精度を「学習に使っていない馬だけ」で測り直す

背景:
  validate_banushi.py はデプロイ済みモデルでバヌーシーを評価するが、
  バヌーシー53頭は学習プールに含まれているので in-sample になり、
  spearman 0.542・上位25%差分 +37.9pt と実力より高く出る。
  過去の正しい評価（experiment_add_banushi.py の5-fold CV）は、
  log1p修正・測尺追加の前のモデルで測ったもの。

2つの評価:
  (A) 5-fold CV: バヌーシーを5分割し、1/5を学習から外して予測（他クラブ + 残りのバヌーシーで学習）。
      本番の使い方（既存のバヌーシー馬から学び、新しい馬を予測）に近い
  (B) LOCO: バヌーシーを丸ごと学習から外して予測。「バヌーシーのデータ無しでも使えるか」

どちらも SEEDS 回（seedとfold分割を変えて）繰り返す。53頭では1回の数字が大きくブレるので、
他クラブのLOCO予測から53頭を無作為抽出したときの分布（ノイズフロア、eval_utils）と比べる。

使い方:
  python experiment_banushi_oos.py
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

import train_model_kakutoku as tmk
from eval_utils import _top25_diff, bootstrap_noise_floor, percentile_of

warnings.filterwarnings("ignore")

SEEDS = tuple(range(1, 11))
N_FOLDS = 5
TARGET = "banushi"


def fit_and_score(train_df, test_df, seed):
    aggs = tmk.build_aggs(train_df)
    tr = tmk.add_features(train_df, aggs)
    te = tmk.add_features(test_df, aggs)
    fc = tmk.get_feature_cols(tr)
    X_tr, X_te = tr[fc].fillna(-1), te[fc].fillna(-1)
    huber = tmk.train_huber(X_tr, np.log1p(tr["kakutoku_man"]), random_state_override=seed)
    rank = tmk.train_lambdarank(X_tr, tmk.make_relevance(tr["kaishuu_rate"]), random_state_override=seed)
    # 5-foldで集めた予測を1つにまとめて順位付けするため、バッチ内順位ではなく
    # huber予測（log1p空間）とlambdarankスコアを学習プール内のパーセンタイルに変換して平均する
    ref_h, ref_r = huber.predict(X_tr), rank.predict(X_tr)
    hp, rp = huber.predict(X_te), rank.predict(X_te)
    return np.array([(ref_h < h).mean() + (ref_r < r).mean() for h, r in zip(hp, rp)])


def metrics(score, actual):
    return spearmanr(score, actual)[0], _top25_diff(score, actual)


def main():
    print(f"[{datetime.now():%H:%M:%S}] 開始", flush=True)
    pool = tmk.load_combined()
    target = pool[pool["club_name"] == TARGET].reset_index(drop=True)
    rest = pool[pool["club_name"] != TARGET].reset_index(drop=True)
    actual = target["kaishuu_rate"].values
    print(f"バヌーシー {len(target)} 頭 / その他 {len(rest)} 頭")

    rows = []
    for seed in SEEDS:
        # (A) 5-fold CV
        cv_score = np.zeros(len(target))
        for tr_idx, te_idx in KFold(N_FOLDS, shuffle=True, random_state=seed).split(target):
            train_df = pd.concat([rest, target.iloc[tr_idx]], ignore_index=True)
            cv_score[te_idx] = fit_and_score(train_df, target.iloc[te_idx].reset_index(drop=True), seed)
        # (B) LOCO
        loco_score = fit_and_score(rest, target, seed)
        for name, score in [("5-fold CV", cv_score), ("LOCO", loco_score)]:
            c, d = metrics(score, actual)
            rows.append({"seed": seed, "eval": name, "spearman": c, "diff": d})
        print(f"[{datetime.now():%H:%M:%S}] seed {seed} 完了", flush=True)

    # ノイズフロア: 他クラブのLOCO予測（seed=1）から53頭を抽出したときの分布
    print(f"[{datetime.now():%H:%M:%S}] ノイズフロア計算中（他クラブのLOCO）", flush=True)
    club_results = {}
    for club in sorted(rest["club_name"].unique()):
        held = rest[rest["club_name"] == club].reset_index(drop=True)
        if len(held) < len(target):
            continue
        others = pool[pool["club_name"] != club].reset_index(drop=True)
        club_results[club] = (fit_and_score(others, held, 1), held["kaishuu_rate"].values)
    floor = bootstrap_noise_floor(club_results, len(target))

    res = pd.DataFrame(rows)
    res.to_csv(tmk.DATA_DIR / "experiment_banushi_oos.csv", index=False, encoding="utf-8-sig")

    print(f"\n=== バヌーシー {len(target)} 頭・学習に使っていない馬での評価（{len(SEEDS)} seed） ===")
    for name in ["5-fold CV", "LOCO"]:
        r = res[res["eval"] == name]
        c_mean, d_mean = r["spearman"].mean(), r["diff"].mean()
        print(f"  {name:9s}: spearman={c_mean:.3f}（seed間 {r['spearman'].min():.3f}〜{r['spearman'].max():.3f}、"
              f"ノイズフロア{percentile_of(c_mean, floor['corrs']):.0f}%点）  "
              f"上位25%差分={d_mean:+.1f}pt（{r['diff'].min():+.1f}〜{r['diff'].max():+.1f}、"
              f"ノイズフロア{percentile_of(d_mean, floor['diffs']):.0f}%点）")
    print(f"\n  ノイズフロア（他{len(club_results)}クラブのLOCO予測から53頭を無作為抽出）: "
          f"spearman={floor['corrs'].mean():.3f}±{floor['corrs'].std():.3f}, "
          f"上位25%差分={floor['diffs'].mean():+.1f}±{floor['diffs'].std():.1f}pt")
    print("  参考（in-sample、validate_banushi.py）: spearman=0.542, 上位25%差分=+37.9pt")
    print(f"\n[{datetime.now():%H:%M:%S}] 完了", flush=True)


if __name__ == "__main__":
    main()
