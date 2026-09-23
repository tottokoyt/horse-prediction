"""
experiment_measurement_placebo.py
experiment_measurement_loco.py の追試: 測尺のある4クラブ（silk/carrot/normandy/union）に
絞って試行回数を増やし、本物の測尺（標準化）の効果がプラセボ（クラブ×年度内シャッフル）の
分布と区別できるかを確かめる

背景:
  5 seed の LOCO では、測尺の全く無いクラブ（g1/sunday/tclion）でも
  特徴量を足しただけで上位25%diffが ±5pt 程度動き、carrot ではプラセボが
  本物を上回った。5 seed ではノイズと区別できないため、
  N_RUNS 回（毎回 seed とシャッフルを変える）の分布で比較する。

判定:
  各試行 k で、4クラブそれぞれを held-out にして spearman / 上位25%diff を出し、
  4クラブ平均を1つの値とする。なし・標準化・シャッフルの N_RUNS 個の値を比べ、
  「標準化の平均がシャッフル分布の何パーセンタイルか」「なしとの差の平均±標準誤差」を出す。
  生値（推論時の標準化基準が不要な方式、PLANの選択肢(c)）も同じ試行で比較する。

使い方:
  python experiment_measurement_placebo.py
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import experiment_measurement_loco as eml
from eval_utils import _top25_diff

warnings.filterwarnings("ignore")

N_RUNS = 20
VARIANTS = ["なし", "生値", "標準化", "シャッフル"]


def main():
    print(f"[{datetime.now():%H:%M:%S}] 開始（{N_RUNS} 試行 × 4クラブ × {len(VARIANTS)} パターン）", flush=True)
    pool = eml.add_standardized(eml.load_pool_with_scale())

    rows = []
    for k in range(1, N_RUNS + 1):
        shuffled = eml.add_shuffled(pool, k)
        for club in eml.MEASURED_CLUBS:
            for variant in VARIANTS:
                src = shuffled if variant == "シャッフル" else pool
                held = src[src["club_name"] == club].reset_index(drop=True)
                rest = src[src["club_name"] != club].reset_index(drop=True)
                actual = held["kaishuu_rate"].values
                score, _, _ = eml.train_and_score(rest, held, variant, seed=k)
                rows.append({"run": k, "club": club, "variant": variant,
                             "spearman": spearmanr(score, actual)[0],
                             "diff": _top25_diff(score, actual)})
        print(f"[{datetime.now():%H:%M:%S}] 試行 {k}/{N_RUNS} 完了", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(eml.DATA_DIR / "experiment_measurement_placebo.csv", index=False, encoding="utf-8-sig")

    for metric in ["spearman", "diff"]:
        print(f"\n=== {metric}: クラブ別（{N_RUNS}試行の平均±標準誤差） ===")
        g = res.groupby(["club", "variant"])[metric]
        t = (g.mean().round(3).astype(str) + "±" + g.sem().round(3).astype(str)).unstack()[VARIANTS]
        print(t.to_string())

        per_run = res.groupby(["run", "variant"])[metric].mean().unstack()[VARIANTS]
        print(f"\n=== {metric}: 4クラブ平均（{N_RUNS}試行） ===")
        for v in VARIANTS:
            print(f"  {v}: {per_run[v].mean():.3f} ± {per_run[v].sem():.3f}(SE)")
        for v in ["生値", "標準化", "シャッフル"]:
            d = per_run[v] - per_run["なし"]
            print(f"  {v} − なし（同じseedで対応あり）: {d.mean():+.3f} ± {d.sem():.3f}(SE), "
                  f"なしを上回った試行 {int((d > 0).sum())}/{N_RUNS}")
        for v in ["生値", "標準化"]:
            d = per_run[v] - per_run["シャッフル"]
            pct = (per_run["シャッフル"] < per_run[v].mean()).mean()
            print(f"  {v} − シャッフル: {d.mean():+.3f} ± {d.sem():.3f}(SE), "
                  f"{v}の平均はシャッフル分布の {pct:.0%} 点")
        d = per_run["標準化"] - per_run["生値"]
        print(f"  標準化 − 生値（対応あり）: {d.mean():+.3f} ± {d.sem():.3f}(SE), "
              f"標準化が上回った試行 {int((d > 0).sum())}/{N_RUNS}")

    print(f"\n[{datetime.now():%H:%M:%S}] 完了", flush=True)


if __name__ == "__main__":
    main()
