"""
fetch_bms_other.py
他クラブ（jisseki_other_all.csv）の学習年度（2018〜2021）分について
umadb個別ページから母父（BMS）を取得する

目的:
  シルク単独（634頭）では母父の集計サンプルが少なく、
  train_model_v7.py で過学習の兆候が見られた。
  sire/trainer/farm と同様に、他クラブの学習年度データを合わせて
  bms（母父）の集計プールを拡大する。

使い方:
  python fetch_bms_other.py

注意: 対象は bosyu_year が 2018〜2021 の他クラブ馬のみ（約2,259頭）
     2秒間隔で約75分かかります
     途中で止まっても data/bms_other_cache.csv に進捗が保存されるので
     再実行すれば続きから再開します
"""

import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

from fetch_bms import fetch_html, extract_bms

DATA_DIR = Path("data")
CACHE_FILE = DATA_DIR / "bms_other_cache.csv"
TRAIN_YEARS = [2018, 2019, 2020, 2021]


def load_cache() -> pd.DataFrame:
    if CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE, encoding="utf-8-sig")
        print(f"キャッシュ読み込み: {len(df)} 件")
        return df
    return pd.DataFrame(columns=["horse_id", "horse_url", "bms_name"])


def save_cache(df: pd.DataFrame):
    df.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")


def fetch_all_bms_other(df_targets: pd.DataFrame) -> pd.DataFrame:
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
            bms_name = extract_bms(soup)
            new_rows.append({
                "horse_id": horse_id,
                "horse_url": url,
                "bms_name": bms_name,
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
                "bms_name": None,
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
    print("=== 他クラブ 母父（BMS）取得（学習年度のみ） ===\n")
    df_other = pd.read_csv(DATA_DIR / "jisseki_other_all.csv", encoding="utf-8-sig")
    df_other = df_other.dropna(subset=["horse_url", "horse_id"])
    df_targets = df_other[df_other["bosyu_year"].isin(TRAIN_YEARS)].copy()
    print(f"対象（{TRAIN_YEARS}）: {len(df_targets)} 頭")

    df_bms = fetch_all_bms_other(df_targets)
    print(f"\n母父取得完了: {df_bms['bms_name'].notna().sum()}/{len(df_bms)} 件")


if __name__ == "__main__":
    main()
