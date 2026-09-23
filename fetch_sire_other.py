"""
fetch_sire_other.py
他クラブ（jisseki_other_all.csv）の既存3,478頭について
umadb個別ページから父馬（sire）を取得する

目的:
  jisseki_other_all.csv には sire 列がなく、これまでは
  sire×bms_name の「配合ニック」特徴量がシルク単独（601頭）でしか
  計算できなかった。個別ページには母父と同時に父馬も載っているため、
  既存の bms_other_cache.csv 取得対象と同じ馬について sire も取得し、
  他クラブを含めた大きなプールでニック特徴量を再検証できるようにする。

注意:
  umadb.com の robots.txt は ClaudeBot に Crawl-delay: 30 を指定しているため、
  このスクリプトは30秒間隔でアクセスする（対象3,478頭で約29時間）。
  途中で止まっても data/sire_other_cache.csv に進捗が保存されるので
  再実行すれば続きから再開する。

使い方:
  python fetch_sire_other.py
"""

import time
import pandas as pd
from pathlib import Path

from fetch_bms import fetch_html, extract_sire

DATA_DIR = Path("data")
CACHE_FILE = DATA_DIR / "sire_other_cache.csv"
SLEEP_SECONDS = 30  # umadb.com robots.txt: User-agent ClaudeBot Crawl-delay: 30


def load_cache() -> pd.DataFrame:
    if CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE, encoding="utf-8-sig")
        print(f"キャッシュ読み込み: {len(df)} 件")
        return df
    return pd.DataFrame(columns=["horse_id", "horse_url", "sire"])


def save_cache(df: pd.DataFrame):
    df.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")


def main():
    print("=== 他クラブ 父馬（sire）再取得 ===\n")
    df_other = pd.read_csv(DATA_DIR / "jisseki_other_all.csv", encoding="utf-8-sig")
    df_other = df_other.dropna(subset=["horse_url", "horse_id"])
    df_other = df_other.drop_duplicates(subset=["horse_id"])

    cache = load_cache()
    cached_ids = set(cache["horse_id"].tolist())
    targets = df_other[~df_other["horse_id"].isin(cached_ids)].copy()
    total = len(targets)
    print(f"取得対象: {total} 頭（キャッシュ済み: {len(cached_ids)} 頭）")
    print(f"間隔: {SLEEP_SECONDS}秒 → 想定所要時間: 約{total * SLEEP_SECONDS / 3600:.1f}時間\n")

    new_rows = []
    for i, (_, row) in enumerate(targets.iterrows()):
        horse_id = row["horse_id"]
        url = row["horse_url"]
        try:
            soup = fetch_html(url)
            sire = extract_sire(soup)
            new_rows.append({"horse_id": horse_id, "horse_url": url, "sire": sire})
            if (i + 1) % 10 == 0 or i == total - 1:
                cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True).drop_duplicates(subset=["horse_id"])
                save_cache(cache)
                new_rows = []
                print(f"  進捗: {i+1}/{total} 完了")
        except Exception as e:
            print(f"  [ERROR] {horse_id}: {e}")
            new_rows.append({"horse_id": horse_id, "horse_url": url, "sire": None})

        if i < total - 1:
            time.sleep(SLEEP_SECONDS)

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True).drop_duplicates(subset=["horse_id"])
        save_cache(cache)

    print(f"\n完了: 合計 {len(cache)} 件（sire取得済み: {cache['sire'].notna().sum()} 件）")


if __name__ == "__main__":
    main()
