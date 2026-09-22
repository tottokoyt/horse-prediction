"""
fetch_mother_other.py
他クラブ（jisseki_other_all.csv）の学習年度（2018〜2021）分について
umadb個別ページから母馬名を取得する

目的:
  クラブ公式サイト（Wayback Machineのアーカイブ含む）の募集時測尺データは
  「母馬名＋産年」の仮名（例: レインデートの2017）で馬が識別されているため、
  デビュー後の競走馬名（jisseki_other_all.csv の horse_name）と直接は
  突き合わせられない。母馬名を取得し「母馬名＋産年下2桁」の仮名を作ることで、
  measurement アーカイブと horse_id を紐付けられるようにする。

使い方:
  python fetch_mother_other.py

注意: 対象は bosyu_year が 2018〜2021 の他クラブ馬のみ（約2,259頭）
     2秒間隔で約75分かかります
     途中で止まっても data/mother_other_cache.csv に進捗が保存されるので
     再実行すれば続きから再開します
"""

import time
import pandas as pd
from pathlib import Path

from fetch_bms import fetch_html, extract_mother

DATA_DIR = Path("data")
CACHE_FILE = DATA_DIR / "mother_other_cache.csv"
TRAIN_YEARS = [2018, 2019, 2020, 2021]


def load_cache() -> pd.DataFrame:
    if CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE, encoding="utf-8-sig")
        print(f"キャッシュ読み込み: {len(df)} 件")
        return df
    return pd.DataFrame(columns=["horse_id", "horse_url", "mother_name"])


def save_cache(df: pd.DataFrame):
    df.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")


def fetch_all_mother_other(df_targets: pd.DataFrame) -> pd.DataFrame:
    cache = load_cache()
    cached_ids = set(cache["horse_id"].tolist())

    new_rows = []
    targets = df_targets[~df_targets["horse_id"].isin(cached_ids)].copy()
    total = len(targets)

    print(f"取得対象: {total} 頭（キャッシュ済み: {len(cached_ids)} 頭）")

    for i, (_, row) in enumerate(targets.iterrows()):
        horse_id = row["horse_id"]
        url = row["horse_url"]

        try:
            soup = fetch_html(url)
            mother_name = extract_mother(soup)
            new_rows.append({
                "horse_id": horse_id,
                "horse_url": url,
                "mother_name": mother_name,
            })

            if (i + 1) % 20 == 0 or i == total - 1:
                cache = pd.concat(
                    [cache, pd.DataFrame(new_rows)],
                    ignore_index=True
                ).drop_duplicates(subset=["horse_id"])
                save_cache(cache)
                new_rows = []
                print(f"  進捗: {i+1}/{total} 完了")

        except Exception as e:
            print(f"  [ERROR] {horse_id}: {e}")
            new_rows.append({
                "horse_id": horse_id,
                "horse_url": url,
                "mother_name": None,
            })

        time.sleep(2)

    if new_rows:
        cache = pd.concat(
            [cache, pd.DataFrame(new_rows)],
            ignore_index=True
        ).drop_duplicates(subset=["horse_id"])
        save_cache(cache)

    return cache


def main():
    print("=== 他クラブ 母馬名取得（学習年度のみ） ===\n")
    df_other = pd.read_csv(DATA_DIR / "jisseki_other_all.csv", encoding="utf-8-sig")
    df_other = df_other.dropna(subset=["horse_url", "horse_id"])
    df_targets = df_other[df_other["bosyu_year"].isin(TRAIN_YEARS)].copy()
    print(f"対象（{TRAIN_YEARS}）: {len(df_targets)} 頭")

    df_mother = fetch_all_mother_other(df_targets)
    print(f"\n母馬名取得完了: {df_mother['mother_name'].notna().sum()}/{len(df_mother)} 件")


if __name__ == "__main__":
    main()
