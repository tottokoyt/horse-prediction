"""
eval_utils.py
train_model_v*.py 共通の「安定した」評価ユーティリティ

背景:
  上位25%平均回収率は検証頭数が少ない（シルク検証156頭、200%超は
  たった8頭）ため、1頭の外れ値やLightGBMの内部乱数（seed）次第で
  数値が大きく動く。実際、他クラブ測尺統合の実験（v10〜v12）では
  単一seed・生の回収率で「効果あり」に見えた設定が、複数seed平均＋
  回収率200%キャップで再検証すると効果なしと判明した。

  今後モデルの改善案を比較するときは、必ずこのモジュールの
  evaluate_stable() を使い、複数seedの平均±標準偏差で判断すること。
  「1回の実行でたまたま良い数字が出た」を改善と誤認しないための保険。

使い方:
  from eval_utils import evaluate_stable
  result = evaluate_stable(build_model_fn, X_tr, y_tr, X_vl, df_vl)
  print(result["summary"])
"""

import numpy as np
import pandas as pd

CAP = 200.0
DEFAULT_SEEDS = 5


def _evaluate_once(model, X_vl, df_vl, cap=CAP):
    proba = model.predict_proba(X_vl)

    df_eval = df_vl[["kaishuu_rate"]].copy().reset_index(drop=True)
    df_eval["prob_class2"] = proba[:, 2]
    df_eval["kaishuu_rate_capped"] = df_eval["kaishuu_rate"].clip(upper=cap)

    thr   = df_eval["prob_class2"].quantile(0.75)
    top25 = df_eval[df_eval["prob_class2"] >= thr]

    raw_avg    = top25["kaishuu_rate"].mean()
    capped_avg = top25["kaishuu_rate_capped"].mean()
    o200_rate  = (top25["kaishuu_rate"] >= 200).mean()
    return raw_avg, capped_avg, o200_rate


def evaluate_stable(build_model_fn, X_tr, y_tr, X_vl, df_vl, cap=CAP, seeds=DEFAULT_SEEDS, label=""):
    """
    build_model_fn(seed) -> 学習済みモデル を複数seed分呼び出し、
    raw / capped 上位25%平均回収率と200%超率の平均±標準偏差を返す。

    「生の回収率・単一seed」だけを見て改善判定しないための共通経路。
    """
    scores = []
    for seed in range(seeds):
        model = build_model_fn(seed)
        scores.append(_evaluate_once(model, X_vl, df_vl, cap))
    s = np.array(scores)

    result = {
        "raw_mean":    s[:, 0].mean(), "raw_std":    s[:, 0].std(),
        "capped_mean": s[:, 1].mean(), "capped_std": s[:, 1].std(),
        "o200_mean":   s[:, 2].mean(), "o200_std":   s[:, 2].std(),
        "n_seeds": seeds,
    }
    result["summary"] = (
        f"[{label}] raw={result['raw_mean']:.1f}±{result['raw_std']:.1f}  "
        f"capped={result['capped_mean']:.1f}±{result['capped_std']:.1f}  "
        f"200%超率={result['o200_mean']:.1%}±{result['o200_std']:.1%}  (seeds={seeds})"
    )
    return result
