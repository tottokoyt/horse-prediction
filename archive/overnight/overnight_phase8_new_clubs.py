"""
overnight_phase8_new_clubs.py
umadb.comのクラブ一覧を確認したところ、これまで未収集の2クラブが
見つかった:
  - エプソム愛馬会 (c117, 145頭)
  - ゴールドホースクラブ (c119, 95頭)
  (c126 Blooming Horse Clubも見つかったが、全馬2歳未出走のため
   学習年度(2018-2021)には該当データが無くスキップ)

これらは今夜の検証・チューニングに一切使っていない「真に未知の
クラブ」なので、これまでの判断（バヌーシー追加、tclion追加候補など）
がどれだけ汎化するかを検証する最良のテストケースになる。

手順:
  1. collect_jisseki_other.pyのcollect_club_year()で実績収集
  2. fetch_bms.pyのextract_sire/extract_bmsで血統取得
  3. 現行本番モデル(C, バヌーシー込み12クラブ)と、
     tclion追加候補モデル（フェーズ7で最良だったtclion単体追加）を
     クリーン評価手法で比較し、この2つの真に未知のクラブでどちらが
     良いか最終確認する

使い方:
  python overnight_phase8_new_clubs.py
"""

import sys
import time
import traceback
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr, rankdata

warnings.filterwarnings("ignore")

sys.path.insert(0, "app")
from collect_jisseki_other import collect_club_year, YEAR_MAP  # noqa: E402
from fetch_bms import fetch_html, extract_sire, extract_bms  # noqa: E402
import experiment_loco as el  # noqa: E402
from eval_utils import _top25_diff  # noqa: E402
from train_model_kakutoku import make_relevance, load_combined as load_combined_c  # noqa: E402

DATA_DIR = Path("data")
REPORT_PATH = Path("overnight_report.md")
TRAIN_YEARS = [2018, 2019, 2020, 2021]
SEEDS = tuple(range(1, 6))

NEW_CLUBS = {
    "epsom": "c117",   # エプソム愛馬会
    "gold":  "c119",   # ゴールドホースクラブ
}

COLS = ["club_name", "sire", "trainer", "farm", "bms_name", "sex", "birth_month",
        "price_man", "kaishuu_rate", "kakutoku_man"]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


def report(md):
    print(md, flush=True)
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(md + "\n")


def collect_club(name, code):
    dfs = []
    for birth_year in sorted(YEAR_MAP.keys()):
        try:
            df = collect_club_year(name, code, birth_year)
            dfs.append(df)
        except Exception as e:
            log(f"  [ERROR] {name} {birth_year}年産 収集失敗: {e}")
        time.sleep(2)
    all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    all_df.to_csv(DATA_DIR / f"jisseki_{name}.csv", index=False, encoding="utf-8-sig")
    return all_df


def fetch_pedigree(name, jisseki_df):
    rows = []
    df = jisseki_df.dropna(subset=["horse_url", "horse_id"])
    for _, row in df.iterrows():
        try:
            soup = fetch_html(row["horse_url"])
            rows.append({
                "horse_id": row["horse_id"],
                "sire": extract_sire(soup),
                "bms_name": extract_bms(soup),
            })
        except Exception:
            rows.append({"horse_id": row["horse_id"], "sire": None, "bms_name": None})
        time.sleep(2)
    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / f"{name}_pedigree_cache.csv", index=False, encoding="utf-8-sig")
    return out


def load_club_common(name, years=TRAIN_YEARS):
    jisseki = pd.read_csv(DATA_DIR / f"jisseki_{name}.csv", encoding="utf-8-sig")
    pedigree = pd.read_csv(DATA_DIR / f"{name}_pedigree_cache.csv", encoding="utf-8-sig")[
        ["horse_id", "sire", "bms_name"]
    ]
    df = pd.merge(jisseki, pedigree, on="horse_id", how="left")
    df = df.dropna(subset=["kaishuu_rate", "kakutoku_man", "price_man"])
    df = df[df["bosyu_year"].isin(years)].reset_index(drop=True)
    if "birth_month" not in df.columns:
        df["birth_month"] = np.nan
    df["club_name"] = name
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


def eval_pool(pool, holdout_dfs, seed):
    m = train_ens(pool, seed)
    sum_c, sum_d = 0.0, 0.0
    per_club = {}
    for name, df in holdout_dfs.items():
        hp, rp = predict_clean(m, df)
        score = blended_score(hp, rp)
        actual = df["kaishuu_rate"].values
        c = spearmanr(score, actual)[0]
        d = _top25_diff(score, actual)
        sum_c += c
        sum_d += d
        per_club[name] = (c, d)
    return sum_c, sum_d, per_club


def main():
    report("\n\n# フェーズ8: 新規発見クラブ（真に未知）での最終確認\n")
    report("umadb.comのクラブ一覧を確認し、今夜これまで一切触れていない"
           "「エプソム愛馬会(c117)」「ゴールドホースクラブ(c119)」を新たに"
           "収集した。この2クラブは検証にも学習判断にも使っておらず、"
           "真に未知のクラブへの汎化を測る最良のテストケースになる。\n")
    log("=== フェーズ8開始 ===")

    for name, code in NEW_CLUBS.items():
        try:
            log(f"=== {name} ({code}) データ収集開始 ===")
            jisseki = collect_club(name, code)
            log(f"  実績データ: {len(jisseki)} 頭")
            log(f"  血統取得開始（{len(jisseki)}頭）...")
            fetch_pedigree(name, jisseki)
        except Exception as e:
            log(f"  [ERROR] {name} 収集失敗: {e}")
            report(f"- {name}: 収集エラー: {e}\n```\n{traceback.format_exc()}\n```")

    holdout = {}
    for name in NEW_CLUBS:
        try:
            df = load_club_common(name)
            if len(df) >= 10:
                holdout[name] = df
                report(f"- {name}: 学習年度(2018-2021)対象 {len(df)}頭 を収集完了")
            else:
                report(f"- {name}: サンプル不足（n={len(df)}）のため検証対象から除外")
        except Exception as e:
            report(f"- {name}: 読み込みエラー: {e}")

    if not holdout:
        report("\n有効なデータが無かったため検証をスキップします。")
        log("=== フェーズ8完了（データなし）===")
        return

    pool_current = load_combined_c()  # 現行本番相当（バヌーシー込み12クラブ）
    for c in COLS:
        if c not in pool_current.columns:
            pool_current[c] = np.nan
    pool_current = pool_current[COLS]

    pool_11 = el.load_combined()
    for c in COLS:
        if c not in pool_11.columns:
            pool_11[c] = np.nan
    pool_11 = pool_11[COLS]

    tclion_df = load_club_common("tclion")
    pool_tclion_candidate = pd.concat([pool_current, tclion_df], ignore_index=True)

    pools = {
        "現行本番(banushi込み12クラブ)": pool_current,
        "banushi+tclion候補(13クラブ)": pool_tclion_candidate,
    }

    report(f"\n## 真に未知の{len(holdout)}クラブ({', '.join(holdout.keys())})での比較（{len(SEEDS)} seed, クリーン評価）\n")
    report("| パターン | spearman合計 | 上位25%diff合計 |")
    report("|---|---|---|")

    results = {k: {"c": [], "d": []} for k in pools}
    for seed in SEEDS:
        log(f"seed={seed} 学習・評価中...")
        for key, pool in pools.items():
            c, d, _ = eval_pool(pool, holdout, seed)
            results[key]["c"].append(c)
            results[key]["d"].append(d)

    for key in pools:
        cs, ds = results[key]["c"], results[key]["d"]
        report(f"| {key} | {np.mean(cs):.3f}±{np.std(cs):.3f} | {np.mean(ds):.1f}±{np.std(ds):.1f} |")

    base_c, base_d = results["現行本番(banushi込み12クラブ)"]["c"], results["現行本番(banushi込み12クラブ)"]["d"]
    cand_c, cand_d = results["banushi+tclion候補(13クラブ)"]["c"], results["banushi+tclion候補(13クラブ)"]["d"]
    win_c = sum(1 for a, b in zip(base_c, cand_c) if b > a)
    win_d = sum(1 for a, b in zip(base_d, cand_d) if b > a)
    report(f"\ntclion追加候補が現行本番を上回った回数（{len(SEEDS)} seed中）: "
           f"spearman {win_c}/{len(SEEDS)}, 上位25%diff {win_d}/{len(SEEDS)}")

    report(f"\n完了: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log("=== フェーズ8完了 ===")


if __name__ == "__main__":
    main()
