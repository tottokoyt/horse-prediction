"""
fetch_bms.py
umadb個別ページから母父（BMS）を取得し、
merged_all.csv / merged_train.csv / merged_valid.csv に bms_name 列を追加する

処理の流れ:
  1. jisseki_all.csv の horse_url を1件ずつ取得
  2. 個別ページから母父（BMS）を抽出
     （ページ内に <div>母父 <span class="black">馬名</span></div> の形で存在）
  3. horse_id をキーに merged_*.csv へマージ

使い方:
  python fetch_bms.py

注意: 505頭分のページアクセスが発生します（2秒間隔で約17分）
     途中で止まっても data/bms_cache.csv に進捗が保存されるので
     再実行すれば続きから再開します
"""

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

DATA_DIR = Path("data")
CACHE_FILE = DATA_DIR / "bms_cache.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_html(url: str, retries: int = 3) -> BeautifulSoup:
    for i in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            print(f"  retry {i+1}/{retries}: {e}")
            time.sleep(3)
    raise Exception(f"取得失敗: {url}")


def extract_bms(soup: BeautifulSoup):
    """
    umadb個別ページから母父（BMS）を抽出する
    ページ内に <div>母父 <span class="black">馬名</span></div> がある
    """
    for div in soup.find_all("div"):
        text = div.get_text(strip=True)
        if text.startswith("母父"):
            span = div.find("span")
            if span:
                return span.get_text(strip=True)
            m = re.match(r"母父[：:\s]*(.+)", text)
            if m:
                return m.group(1).strip()
    return None


def extract_sire(soup: BeautifulSoup):
    """
    umadb個別ページから父馬（sire）を抽出する
    ページ内に <div>　父 <a href="/research/sire/...">馬名</a></div> がある
    （「母父」は別途「母父」始まりのdivなので、「父」始まり単独のdivのみ拾う）

    国内JBIS登録馬は /research/sire/... 、海外種牡馬（未登録）は
    /umalist/?t=... というリンク形式になるため、リンク先は問わず
    その div 内の最初の <a> を拾う。
    """
    for div in soup.find_all("div"):
        text = div.get_text(strip=True)
        if text.startswith("父") and not text.startswith("父父") and not text.startswith("父母"):
            a = div.find("a")
            if a:
                return a.get_text(strip=True)
    return None


def load_cache() -> pd.DataFrame:
    """既存キャッシュを読み込む"""
    if CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE, encoding="utf-8-sig")
        print(f"キャッシュ読み込み: {len(df)} 件")
        return df
    return pd.DataFrame(columns=["horse_id", "horse_url", "bms_name"])


def save_cache(df: pd.DataFrame):
    df.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")


def fetch_all_bms(df_jisseki: pd.DataFrame) -> pd.DataFrame:
    """全馬の母父（BMS）を取得（キャッシュ活用）"""
    cache = load_cache()
    cached_ids = set(cache["horse_id"].tolist())

    new_rows = []
    targets = df_jisseki[~df_jisseki["horse_id"].isin(cached_ids)].copy()
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

            if (i + 1) % 10 == 0 or i == total - 1:
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


def merge_bms_into_merged():
    """horse_id をキーに merged_*.csv へ bms_name をマージ"""
    df_bms = pd.read_csv(CACHE_FILE, encoding="utf-8-sig")[["horse_id", "bms_name"]]

    for fname in ["merged_all.csv", "merged_train.csv", "merged_valid.csv"]:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  [スキップ] {fname} が見つかりません")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "bms_name" in df.columns:
            df = df.drop(columns=["bms_name"])
        df = pd.merge(df, df_bms, on="horse_id", how="left")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        n_matched = df["bms_name"].notna().sum()
        print(f"  {fname}: bms_name付与 {n_matched}/{len(df)} 頭")


def main():
    print("=== 母父（BMS）取得 ===\n")
    df_jisseki = pd.read_csv(DATA_DIR / "jisseki_all.csv", encoding="utf-8-sig")

    df_bms = fetch_all_bms(df_jisseki)
    print(f"\n母父取得完了: {df_bms['bms_name'].notna().sum()}/{len(df_bms)} 件\n")

    print("=== merged_*.csv へのマージ ===")
    merge_bms_into_merged()


if __name__ == "__main__":
    main()
