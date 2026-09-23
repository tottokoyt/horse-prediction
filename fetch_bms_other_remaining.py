"""
fetch_bms_other_remaining.py
jisseki_other_all.csv のうち、まだ bms_other_cache.csv に無い馬
（主に検証年度 2022〜2023 産）の母父を取得する

train_model_v5.py のモデルBは silk+other の学習データに加え、
other の検証データ（2022〜2023）でも別途評価するため、
その評価用の母父データも必要になる。

使い方:
  python fetch_bms_other_remaining.py
"""

import time
import pandas as pd
from pathlib import Path

from fetch_bms import fetch_html, extract_bms

DATA_DIR = Path("data")
CACHE_FILE = DATA_DIR / "bms_other_cache.csv"


def load_cache() -> pd.DataFrame:
    if CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE, encoding="utf-8-sig")
        print(f"キャッシュ読み込み: {len(df)} 件")
        return df
    return pd.DataFrame(columns=["horse_id", "horse_url", "bms_name"])


def save_cache(df: pd.DataFrame):
    df.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")


def main():
    print("=== 他クラブ 母父（BMS）取得（残り分） ===\n")
    df_other = pd.read_csv(DATA_DIR / "jisseki_other_all.csv", encoding="utf-8-sig")
    df_other = df_other.dropna(subset=["horse_url", "horse_id"])

    cache = load_cache()
    cached_ids = set(cache["horse_id"].tolist())
    targets = df_other[~df_other["horse_id"].isin(cached_ids)].copy()
    total = len(targets)
    print(f"取得対象: {total} 頭（キャッシュ済み: {len(cached_ids)} 頭）")

    new_rows = []
    for i, (_, row) in enumerate(targets.iterrows()):
        horse_id = row["horse_id"]
        url = row["horse_url"]
        try:
            soup = fetch_html(url)
            bms_name = extract_bms(soup)
            new_rows.append({"horse_id": horse_id, "horse_url": url, "bms_name": bms_name})
            if (i + 1) % 20 == 0 or i == total - 1:
                cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True).drop_duplicates(subset=["horse_id"])
                save_cache(cache)
                new_rows = []
                print(f"  進捗: {i+1}/{total} 完了")
        except Exception as e:
            print(f"  [ERROR] {horse_id}: {e}")
            new_rows.append({"horse_id": horse_id, "horse_url": url, "bms_name": None})
        time.sleep(2)

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True).drop_duplicates(subset=["horse_id"])
        save_cache(cache)

    print(f"\n完了: 合計 {len(cache)} 件（bms取得済み: {cache['bms_name'].notna().sum()} 件）")


if __name__ == "__main__":
    main()
