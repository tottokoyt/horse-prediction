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

  さらに、DMMバヌーシー実データでの検証（README「バヌーシー実データでの
  検証」参照）で、n=53程度の少数サンプルでは「既知の学習済みクラブ」で
  さえ同じ指標が大きくブレる（spearman相関が-0.05〜0.2の間でばらつく等）
  ことが判明した。少数サンプルの検証結果を見るときは、単一の点推定を
  見るのではなく、必ず compare_to_noise_floor() で既知データの
  ブートストラップ分布と比較し、パーセンタイル順位で評価すること。

使い方:
  from eval_utils import evaluate_stable, compare_to_noise_floor
  result = evaluate_stable(build_model_fn, X_tr, y_tr, X_vl, df_vl)
  print(result["summary"])

  # 少数サンプル（新クラブ等）の検証結果を、既知クラブのノイズフロアと比較
  compare_to_noise_floor("banushi", pred, actual, known_club_results)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CAP = 200.0
DEFAULT_SEEDS = 5
DEFAULT_BOOT = 80


def _default_score_fn(model, X_vl):
    """分類モデル（LGBMClassifier, num_class=3）用: 200%超クラスの確率でランキング"""
    return model.predict_proba(X_vl)[:, 2]


def regression_score_fn(model, X_vl):
    """回帰モデル（LGBMRegressor）用: 予測回収率そのものでランキング"""
    return model.predict(X_vl)


def _evaluate_once(model, X_vl, df_vl, cap=CAP, score_fn=_default_score_fn):
    score = score_fn(model, X_vl)

    df_eval = df_vl[["kaishuu_rate"]].copy().reset_index(drop=True)
    df_eval["score"] = score
    df_eval["kaishuu_rate_capped"] = df_eval["kaishuu_rate"].clip(upper=cap)

    thr   = df_eval["score"].quantile(0.75)
    top25 = df_eval[df_eval["score"] >= thr]

    raw_avg    = top25["kaishuu_rate"].mean()
    capped_avg = top25["kaishuu_rate_capped"].mean()
    o200_rate  = (top25["kaishuu_rate"] >= 200).mean()
    return raw_avg, capped_avg, o200_rate


def evaluate_stable(build_model_fn, X_tr, y_tr, X_vl, df_vl, cap=CAP, seeds=DEFAULT_SEEDS, label="", score_fn=_default_score_fn):
    """
    build_model_fn(seed) -> 学習済みモデル を複数seed分呼び出し、
    raw / capped 上位25%平均回収率と200%超率の平均±標準偏差を返す。

    「生の回収率・単一seed」だけを見て改善判定しないための共通経路。
    score_fn(model, X_vl) -> array でランキングに使うスコアを取り出す
    （分類モデルなら200%超確率、回帰モデルなら予測値そのもの）。
    """
    scores = []
    for seed in range(seeds):
        model = build_model_fn(seed)
        scores.append(_evaluate_once(model, X_vl, df_vl, cap, score_fn))
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


# ──────────────────────────────────────────
# ノイズフロア比較（少数サンプルでの誤診断防止）
# ──────────────────────────────────────────

def _top25_diff(pred, actual, cap=CAP):
    """予測スコアでソートした上位25%の（capped）回収率 − 全体平均"""
    actual_capped = np.clip(actual, None, cap)
    thr = np.quantile(pred, 0.75)
    top25 = actual_capped[pred >= thr]
    return top25.mean() - actual_capped.mean()


def bootstrap_noise_floor(club_results, n_sample, n_boot=DEFAULT_BOOT, cap=CAP, seed=0):
    """
    club_results: {club_name: (pred配列, 実際のkaishuu_rate配列)} の辞書
      （leave-one-club-out等で得た、既知クラブへのout-of-sample予測）
    n_sample: 比較したい少数サンプル（新クラブ等）の頭数

    n_sample以上の頭数を持つ既知クラブから毎回n_sample頭をランダム抽出し、
    上位25%差分とspearman相関を n_boot 回ぶん計算して分布を返す。
    「この頭数ならこの指標はどれくらいブレて当然か」を測るためのもの。
    """
    rng = np.random.RandomState(seed)
    diffs, corrs = [], []
    for pred, actual in club_results.values():
        pred = np.asarray(pred)
        actual = np.asarray(actual)
        if len(actual) < n_sample:
            continue
        for _ in range(n_boot):
            idx = rng.choice(len(actual), n_sample, replace=False)
            p, a = pred[idx], actual[idx]
            diffs.append(_top25_diff(p, a, cap))
            c, _ = spearmanr(p, a)
            corrs.append(c)
    return {"diffs": np.array(diffs), "corrs": np.array(corrs)}


def percentile_of(value, distribution):
    """distribution中でvalue未満の割合（%）。50%前後なら「平均的」、
    極端に低い/高ければ本当に外れている可能性が高い"""
    return float((np.asarray(distribution) < value).mean() * 100)


def compare_to_noise_floor(label, pred, actual, club_results, cap=CAP, n_boot=DEFAULT_BOOT, seed=0):
    """
    少数サンプルの検証結果（pred, actual）を、既知クラブのノイズフロアと
    比較してレポートする。パーセンタイル順位が40〜60%あたりなら
    「既知データと同程度のブレ・平均的な結果」、大きく外れていれば
    本当に良い/悪いモデルの可能性が高い。
    """
    pred = np.asarray(pred)
    actual = np.asarray(actual)
    n = len(actual)

    diff = _top25_diff(pred, actual, cap)
    corr, _ = spearmanr(pred, actual)
    floor = bootstrap_noise_floor(club_results, n, n_boot=n_boot, seed=seed)

    diff_pct = percentile_of(diff, floor["diffs"]) if len(floor["diffs"]) else None
    corr_pct = percentile_of(corr, floor["corrs"]) if len(floor["corrs"]) else None

    result = {
        "n": n, "diff": diff, "corr": corr,
        "diff_percentile": diff_pct, "corr_percentile": corr_pct,
        "floor_diff_mean": floor["diffs"].mean() if len(floor["diffs"]) else None,
        "floor_diff_std":  floor["diffs"].std()  if len(floor["diffs"]) else None,
        "floor_corr_mean": floor["corrs"].mean() if len(floor["corrs"]) else None,
        "floor_corr_std":  floor["corrs"].std()  if len(floor["corrs"]) else None,
    }
    result["summary"] = (
        f"[{label}] n={n}  "
        f"spearman={corr:.3f}(ノイズフロア{corr_pct:.0f}%tile, 既知平均{result['floor_corr_mean']:.3f}±{result['floor_corr_std']:.3f})  "
        f"上位25%差分={diff:.1f}pt(ノイズフロア{diff_pct:.0f}%tile, 既知平均{result['floor_diff_mean']:.1f}±{result['floor_diff_std']:.1f})"
    )
    return result
