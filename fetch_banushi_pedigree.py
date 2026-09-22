"""
fetch_banushi_pedigree.py
DMMバヌーシー89頭ぶんの父馬（sire）・母父（bms_name）を
umadb個別ページから取得する

使い方:
  python fetch_banushi_pedigree.py
"""

import time
import pandas as pd
from pathlib import Path

from fetch_bms import fetch_html, extract_bms, extract_sire

DATA_DIR = Path("data")
CACHE_FILE = DATA_DIR / "banushi_pedigree_cache.csv"


def load_cache() -> pd.DataFrame:
    if CACHE_FILE.exists():
        return pd.read_csv(CACHE_FILE, encoding="utf-8-sig")
    return pd.DataFrame(columns=["horse_id", "horse_url", "sire", "bms_name"])


def save_cache(df: pd.DataFrame):
    df.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")


def main():
    print("=== DMMバヌーシー 父馬・母父取得 ===\n")
    df = pd.read_csv(DATA_DIR / "jisseki_banushi.csv", encoding="utf-8-sig")
    df = df.dropna(subset=["horse_url", "horse_id"])

    cache = load_cache()
    cached_ids = set(cache["horse_id"].tolist())
    targets = df[~df["horse_id"].isin(cached_ids)]
    print(f"取得対象: {len(targets)} 頭（キャッシュ済み: {len(cached_ids)} 頭）")

    new_rows = []
    for i, (_, row) in enumerate(targets.iterrows()):
        horse_id, url = row["horse_id"], row["horse_url"]
        try:
            soup = fetch_html(url)
            new_rows.append({
                "horse_id": horse_id, "horse_url": url,
                "sire": extract_sire(soup), "bms_name": extract_bms(soup),
            })
        except Exception as e:
            print(f"  [ERROR] {horse_id}: {e}")
            new_rows.append({"horse_id": horse_id, "horse_url": url, "sire": None, "bms_name": None})
        if (i + 1) % 10 == 0:
            print(f"  進捗: {i+1}/{len(targets)}")
        time.sleep(2)

    cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True).drop_duplicates(subset=["horse_id"])
    save_cache(cache)
    print(f"\n完了: {len(cache)} 件（sire取得済み: {cache['sire'].notna().sum()}, bms取得済み: {cache['bms_name'].notna().sum()}）")


if __name__ == "__main__":
    main()
