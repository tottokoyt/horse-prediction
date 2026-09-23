"""
overnight_expand_clubs.py
一晩で、umadb掲載の残り9クラブ（未収集）についてデータ収集→検証→
「学習に追加すべきか」のCV検証まで一括で行い、レポートを出す。

対象クラブ（umadb /data/evaluate/ 一覧より、未収集の9クラブ）:
  ローレルC(c116) / 友駿HC(c108) / 京都TC(c124) / 大樹RC(c113) /
  ターファイトC(c109) / 広尾TC(c115) / TCライオン(c111) /
  グリーンF(c114) / YGGOC(c110)

各クラブについて:
  1. collect_jisseki_other.py の collect_club_year() で実績データ収集
  2. fetch_bms.py の extract_sire/extract_bms で個別ページから血統取得
  3. validate_new_club.py 相当の検証（現行デプロイモデルへの当てはまり、
     既知クラブのノイズフロアと比較）
  4. experiment_add_banushi.py 相当の5-fold CV
     （「学習プールに追加したら良くなるか」を複数シードで検証）

本番モデルの再学習・デプロイ・git push はここでは行わない
（安全のため、朝起きてから内容を確認してもらってから判断する）。
結果はすべて overnight_report.md に追記していく。

使い方:
  python overnight_expand_clubs.py
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
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

sys.path.insert(0, "app")
from collect_jisseki_other import collect_club_year, YEAR_MAP  # noqa: E402
from fetch_bms import fetch_html, extract_sire, extract_bms  # noqa: E402
import experiment_loco as el  # noqa: E402
from eval_utils import _top25_diff, compare_to_noise_floor  # noqa: E402
from train_model_kakutoku import make_relevance, load_combined as load_combined_c  # noqa: E402
from validate_banushi import build_known_club_results  # noqa: E402
import predictor  # noqa: E402

DATA_DIR = Path("data")
REPORT_PATH = Path("overnight_report.md")
TRAIN_YEARS = [2018, 2019, 2020, 2021]

TARGET_CLUBS = {
    "laurel":     "c116",  # ローレルC
    "yushun":     "c108",  # 友駿HC
    "kyoto":      "c124",  # 京都TC
    "taiju":      "c113",  # 大樹RC
    "turffight":  "c109",  # ターファイトC
    "hiroo":      "c115",  # 広尾TC
    "tclion":     "c111",  # TCライオン
    "greenf":     "c114",  # グリーンF
    "ygg":        "c110",  # YGGOC
}


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
        except Exception as e:
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
    df["club_name"] = name
    return df


def predict_with_deployed(df):
    predictor.load_models()
    saved = predictor._models["C"]
    preds = []
    for _, row in df.iterrows():
        class Req:
            pass
        r = Req()
        r.sire, r.trainer, r.farm, r.bms = row["sire"], row["trainer"], row["farm"], row["bms_name"]
        r.sex, r.birth_month, r.price = row["sex"], None, row["price_man"]
        r.height = r.chest = r.cannon = r.weight = None
        X = predictor.build_feature_row(r, saved)
        preds.append(predictor._ensemble_kakutoku(saved, X))
    return np.array(preds)


def cv_check_add_to_training(name, club_df, known_pool, seeds=(1, 2, 3)):
    """バヌーシーのときと同じ5-fold CVで「学習に追加したら良くなるか」を検証"""
    cols = ["club_name", "sire", "trainer", "farm", "bms_name", "sex", "birth_month",
            "price_man", "kaishuu_rate", "kakutoku_man"]
    for c in cols:
        if c not in club_df.columns:
            club_df[c] = np.nan
    club_common = club_df[cols]

    if len(club_common) < 15:
        return None  # サンプル数が少なすぎてCVが意味をなさない

    results = {}
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
        diff = _top25_diff(all_pred, all_actual)
        corr, _ = spearmanr(all_pred, all_actual)
        results[seed] = (corr, diff)

    return results


def main():
    REPORT_PATH.write_text(
        f"# 夜間クラブ拡張レポート ({datetime.now().strftime('%Y-%m-%d %H:%M')}開始)\n\n"
        "本番モデルの再学習・デプロイ・git pushはここでは行っていません。\n"
        "内容を確認してから判断してください。\n\n",
        encoding="utf-8",
    )

    log(f"対象クラブ: {list(TARGET_CLUBS.keys())}")

    known_pool_11 = el.load_combined()          # 11クラブLOCO用ノイズフロア基準
    known_pool_current = load_combined_c()       # 現行デプロイモデルと同じ12クラブ(バヌーシー込み)
    log("既知クラブのLOCOノイズフロアを計算中...")
    try:
        club_noise_floor = build_known_club_results()
    except Exception as e:
        log(f"  [ERROR] ノイズフロア計算失敗: {e}")
        club_noise_floor = {}

    summary_rows = []

    for name, code in TARGET_CLUBS.items():
        report(f"\n## {name} ({code})\n")
        try:
            log(f"=== {name} ({code}) データ収集開始 ===")
            jisseki = collect_club(name, code)
            log(f"  実績データ: {len(jisseki)} 頭")

            log(f"  血統取得開始（{len(jisseki)}頭）...")
            fetch_pedigree(name, jisseki)

            club_df = load_club_common(name)
            n = len(club_df)
            report(f"- 学習年度(2018-2021)対象: {n}頭, races平均={club_df['races'].mean() if n else float('nan'):.1f}")

            if n < 10:
                report(f"- サンプル数が少なすぎるため検証スキップ（n={n}）")
                summary_rows.append((name, n, None, None, None))
                continue

            # ① 現行デプロイモデルへの当てはまり
            pred = predict_with_deployed(club_df)
            if club_noise_floor:
                result = compare_to_noise_floor(name, pred, club_df["kaishuu_rate"].values, club_noise_floor)
                report(f"- 現行モデルでの検証: {result['summary']}")
            else:
                result = None
                report("- 現行モデルでの検証: ノイズフロア計算に失敗したためスキップ")

            # ② 学習に追加すべきかのCV検証
            log(f"  CV検証中（学習に追加する効果、3seed）...")
            cv = cv_check_add_to_training(name, club_df.copy(), known_pool_current)
            if cv:
                corrs = [v[0] for v in cv.values()]
                diffs = [v[1] for v in cv.values()]
                report(f"- 追加した場合のCV結果（3seed）: spearman平均={np.mean(corrs):.3f}±{np.std(corrs):.3f}, "
                       f"上位25%差分平均={np.mean(diffs):.1f}±{np.std(diffs):.1f}pt")
                base_corr = result["corr"] if result else None
                # バヌーシーのときの採用基準を踏襲: CV相関が「正の値」かつ
                # 「現行より一貫して（全seedで）高い」の両方を満たす場合のみ
                # 「追加を検討」とする（単に現行を上回るだけでは不十分。
                # 現行自体がマイナスのケースもあるため）
                if base_corr is not None and min(corrs) > base_corr and min(corrs) > 0:
                    verdict = "追加を検討（CV相関が全seedで正・現行超え）"
                elif base_corr is not None and min(corrs) > base_corr:
                    verdict = "参考程度（現行より改善はしたが、CV相関自体はまだ弱い/マイナス）"
                else:
                    verdict = "現状追加の明確な根拠なし"
                report(f"- 判定: {verdict}")
                summary_rows.append((name, n, result["corr"] if result else None, np.mean(corrs), np.mean(diffs)))
            else:
                report("- CV検証: サンプル不足のためスキップ")
                summary_rows.append((name, n, result["corr"] if result else None, None, None))

        except Exception as e:
            log(f"  [ERROR] {name} 処理中に例外: {e}")
            report(f"- エラーが発生しました: {e}\n```\n{traceback.format_exc()}\n```")
            continue

    # 総括
    report("\n## 総括\n")
    report("| クラブ | n | 現行モデルspearman | 追加時CV spearman平均 | 追加時CV 上位25%差分平均 |")
    report("|---|---|---|---|---|")
    for name, n, base_corr, cv_corr, cv_diff in summary_rows:
        bc = f"{base_corr:.3f}" if base_corr is not None else "-"
        cc = f"{cv_corr:.3f}" if cv_corr is not None else "-"
        cd = f"{cv_diff:.1f}" if cv_diff is not None else "-"
        report(f"| {name} | {n} | {bc} | {cc} | {cd} |")

    report(f"\n完了: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log("=== 全クラブ処理完了 ===")


if __name__ == "__main__":
    main()
