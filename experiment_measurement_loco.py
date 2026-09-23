"""
experiment_measurement_loco.py
モデルCに測尺（体高・胸囲・管囲・体重）を「あれば使う」追加特徴量として
組み込んだ場合の効果を、真のLOCO（全クラブを1つずつ未知扱い、5 seed）で検証する
（PLAN_measurement_expansion.md フェーズ4）

測尺のあるクラブ（学習年度）:
  silk     : merged_train.csv（4項目）
  carrot   : other_scale_cache.csv（4項目、2020年度なし）
  normandy : other_scale_cache.csv（4項目）
  union    : other_scale_cache.csv（体高・胸囲・管囲のみ、体重なし、6月計測）
  それ以外のクラブは NaN（モデルCの慣例どおり fillna(-1)）

比較する4パターン:
  なし   : 現行モデルC（測尺なし）
  生値   : 測尺4列をそのまま追加
  標準化 : クラブ×募集年度内で z-score 化した測尺4列を追加
           （union の6月計測と carrot/normandy の秋計測の時期差や、
             クラブごとの計測・水準の違いを打ち消し、「同じ世代・同じクラブ内で
             大きいか小さいか」だけを使う）
  シャッフル : 標準化と同じ列だが、値をクラブ×募集年度内で馬の間でランダムに入れ替えた
             プラセボ対照（欠損パターン・分布・クラブ構成は同じで、馬と値の対応だけ壊す）。
             初回実行で測尺の無いクラブ（g1/sunday等）まで改善したため、
             「特徴量が増えて木構造が変わっただけの揺らぎ」と区別するために追加した。
             標準化 > シャッフル でなければ測尺の情報による改善とは言えない

学習・目的変数は現行の train_model_kakutoku.py と同じ（log1p(kakutoku_man) の
huber回帰 + lambdarank のランクアンサンブル）。

評価は測尺のある4クラブを未知扱いにしたときの結果が本命（他クラブは
測尺が全頭NaNなので、差はモデル学習の揺らぎ程度しか出ないはず）。

使い方:
  python experiment_measurement_loco.py
"""

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import train_model_kakutoku as tmk
from eval_utils import _top25_diff

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
SEEDS = tuple(range(1, 6))
SCALE_COLS = ["height", "chest", "cannon", "weight"]
MEASURED_CLUBS = ["silk", "carrot", "normandy", "union"]
VARIANTS = ["なし", "生値", "標準化", "シャッフル"]
EXPERIMENT_YEARS = [2018, 2019, 2020, 2021]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_pool_with_scale():
    """load_combined() と同じ学習プールに、測尺4列と bosyu_year を付与する"""
    # load_combined は2026-09-23から測尺・bosyu_year を含むようになったが、
    # この実験は silk の同名馬除外など独自の突き合わせで作った列を使うので落としておく
    # 学習年度は2026-09-23に2022年まで広げたが、この実験は2018-2021年で行ったので固定する
    pool = tmk.load_combined(EXPERIMENT_YEARS).drop(columns=SCALE_COLS + ["bosyu_year"], errors="ignore")

    # load_combined は horse_id を落とすので、同じ順序で読み直して測尺を横付けする
    silk = pd.read_csv(DATA_DIR / "merged_train.csv", encoding="utf-8-sig")
    silk_scale = silk[["horse_name"] + SCALE_COLS].copy()
    silk_scale["club_name"] = "silk"
    silk_scale["bosyu_year"] = silk["bosyu_year"] if "bosyu_year" in silk.columns else np.nan

    other = pd.read_csv(DATA_DIR / "jisseki_other_all.csv", encoding="utf-8-sig")
    other = other[other["bosyu_year"].isin(EXPERIMENT_YEARS)]
    scale = pd.read_csv(DATA_DIR / "other_scale_cache.csv", encoding="utf-8-sig")[["horse_id"] + SCALE_COLS]
    other_scale = pd.merge(other[["horse_id", "horse_name", "club_name", "bosyu_year"]], scale,
                           on="horse_id", how="left").drop(columns="horse_id")

    lookup = pd.concat([silk_scale, other_scale], ignore_index=True)
    dup = lookup.duplicated(subset=["club_name", "horse_name"], keep=False)
    if dup.any():
        # 同一クラブ内の同名馬は突き合わせが曖昧になるので測尺を使わない
        log(f"同一クラブ内の同名馬 {dup.sum()} 行は測尺を付与しない")
        lookup = lookup[~dup]

    pool = pd.merge(pool, lookup, on=["club_name", "horse_name"], how="left")
    return pool


def add_standardized(df):
    """クラブ×募集年度内で z-score 化（集団が小さすぎる場合はクラブ内で代替）"""
    df = df.copy()
    for col in SCALE_COLS:
        g = df.groupby(["club_name", "bosyu_year"])[col]
        mean, std, cnt = g.transform("mean"), g.transform("std"), g.transform("count")
        club_mean = df.groupby("club_name")[col].transform("mean")
        club_std = df.groupby("club_name")[col].transform("std")
        use_club = (cnt < 10) | std.isna() | (std == 0)
        mean = mean.where(~use_club, club_mean)
        std = std.where(~use_club, club_std)
        df[f"{col}_z"] = (df[col] - mean) / std
    return df


def feature_cols_for(variant, base_cols):
    if variant == "なし":
        return base_cols
    if variant == "生値":
        return base_cols + SCALE_COLS
    if variant == "シャッフル":
        return base_cols + [f"{c}_zshuf" for c in SCALE_COLS]
    return base_cols + [f"{c}_z" for c in SCALE_COLS]


def add_shuffled(df, seed):
    """z-score 列の各行を、クラブ×募集年度内でランダムに並べ替えたプラセボ列を作る"""
    df = df.copy()
    rng = np.random.RandomState(1000 + seed)
    z_cols = [f"{c}_z" for c in SCALE_COLS]
    shuffled = df[z_cols].copy()
    # 欠損の位置は保ったまま、測尺のある馬どうしの間でだけ入れ替える
    measured = df[df["height_z"].notna()]
    for _, idx in measured.groupby(["club_name", "bosyu_year"], dropna=False).groups.items():
        idx = np.asarray(idx)
        shuffled.loc[idx, z_cols] = df.loc[rng.permutation(idx), z_cols].values
    for c in SCALE_COLS:
        df[f"{c}_zshuf"] = shuffled[f"{c}_z"]
    return df


def train_and_score(train_df, test_df, variant, seed):
    aggs = tmk.build_aggs(train_df)
    tr = tmk.add_features(train_df, aggs)
    te = tmk.add_features(test_df, aggs)
    # get_feature_cols は本番採用後は測尺の生値も返すので、比較のため一旦除く
    base_cols = [c for c in tmk.get_feature_cols(tr) if c not in SCALE_COLS]
    fc = feature_cols_for(variant, base_cols)

    X_tr, X_te = tr[fc].fillna(-1), te[fc].fillna(-1)
    huber = tmk.train_huber(X_tr, np.log1p(tr["kakutoku_man"]), random_state_override=seed)
    rank = tmk.train_lambdarank(X_tr, tmk.make_relevance(tr["kaishuu_rate"]), random_state_override=seed)
    score = tmk.ensemble_score_batch(huber.predict(X_te), rank.predict(X_te))
    return score, huber, fc


def main():
    log("=== 測尺追加のLOCO検証 開始 ===")
    pool = add_standardized(load_pool_with_scale())
    shuffled_pools = {seed: add_shuffled(pool, seed) for seed in SEEDS}

    print("\n学習プールの測尺カバレッジ（体高が入っている頭数 / 全頭数）:")
    cov = pool.groupby("club_name").agg(n=("horse_name", "size"), with_scale=("height", "count"))
    print(cov.sort_values("n", ascending=False).to_string())

    clubs = [c for c in cov.index if cov.loc[c, "n"] >= 10]
    rows = []
    importances = {v: [] for v in VARIANTS[1:]}
    for club in clubs:
        log(f"held-out={club} (n={(pool['club_name'] == club).sum()})")
        for variant in VARIANTS:
            cs, ds = [], []
            for seed in SEEDS:
                src = shuffled_pools[seed] if variant == "シャッフル" else pool
                held = src[src["club_name"] == club].reset_index(drop=True)
                rest = src[src["club_name"] != club].reset_index(drop=True)
                actual = held["kaishuu_rate"].values
                score, huber, fc = train_and_score(rest, held, variant, seed)
                cs.append(spearmanr(score, actual)[0])
                ds.append(_top25_diff(score, actual))
                if variant != "なし" and club in MEASURED_CLUBS:
                    imp = pd.Series(huber.feature_importances_, index=fc)
                    importances[variant].append(imp / imp.sum())
            rows.append({"club": club, "n": len(held), "variant": variant,
                         "spearman": np.mean(cs), "diff": np.mean(ds),
                         "spearman_by_seed": cs, "diff_by_seed": ds})

    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR / "experiment_measurement_loco.csv", index=False, encoding="utf-8-sig")

    for metric in ["spearman", "diff"]:
        print(f"\n=== クラブ別 {metric}（5 seed 平均、上位25%diffは回収率200%キャップ） ===")
        t = res.pivot(index="club", columns="variant", values=metric)[VARIANTS]
        t["n"] = res.groupby("club")["n"].first()
        print(t.round(3).to_string())

    for label, subset in [("測尺のある4クラブ", MEASURED_CLUBS), ("全クラブ", clubs)]:
        print(f"\n=== 集計: {label} ===")
        sub = res[res["club"].isin(subset)]
        base = sub[sub["variant"] == "なし"].set_index("club")
        for variant in VARIANTS:
            v = sub[sub["variant"] == variant].set_index("club")
            line = (f"  {variant}: spearman={v['spearman'].mean():.3f}±{v['spearman'].std(ddof=0):.3f}, "
                    f"上位25%diff={v['diff'].mean():.1f}±{v['diff'].std(ddof=0):.1f}")
            if variant != "なし":
                wc = (v["spearman"] > base["spearman"]).sum()
                wd = (v["diff"] > base["diff"]).sum()
                line += f"  / なしを上回ったクラブ: spearman {wc}/{len(v)}, diff {wd}/{len(v)}"
            print(line)

    for variant, imps in importances.items():
        if imps:
            mean_imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
            scale_share = mean_imp[[c for c in mean_imp.index if c.split("_z")[0] in SCALE_COLS]].sum()
            print(f"\n[{variant}] huber特徴量重要度（測尺クラブ held-out 時の平均・割合）: 測尺合計 {scale_share:.1%}")
            print(mean_imp.head(8).round(3).to_string())

    log("=== 完了 ===")


if __name__ == "__main__":
    main()
