"""
fetch_carrot_scale.py
キャロットクラブの募集時測尺（体高・胸囲・管囲・馬体重）を
Wayback Machine（web.archive.org）のアーカイブページから取得する

背景:
  キャロットクラブ公式サイトは当該年度の募集ページしか公開しておらず、
  過去年度（2018〜2021年募集）の測尺ページは現在リンクが無い。
  しかし Wayback Machine には当時クロールされたページがそのまま残っている。

  測尺ページの馬名は「母馬名＋産年」の募集時仮名（例: レインデートの2017）
  になっており、jisseki_other_all.csv の horse_name（デビュー後の競走馬名）
  とは直接一致しない。fetch_mother_other.py で取得した母馬名を使い
  「母馬名＋産年」でマッチングする（merge_carrot_scale.py で実施）。

出力:
  data/carrot_scale_raw.csv
    columns: bosyu_year, bosyu_name(募集時仮名), height, chest, cannon, weight, farm

使い方:
  python fetch_carrot_scale.py
"""

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

DATA_DIR = Path("data")
OUT_FILE = DATA_DIR / "carrot_scale_raw.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# bosyu_year -> (archived timestamp, path)
# jisseki_other_all.csv がカバーする学習年度(2018-2021)のうち、
# Wayback Machineで測尺ページが見つかった年度のみ
PAGES = {
    2018: ("20220120022528", "https://carrotclub.net/horse/2018-bosyu-size.asp"),
    2019: ("20190923172637", "https://carrotclub.net/horse/201908-measure.asp"),
    2021: ("20210918065024", "https://carrotclub.net/horse/2021_size_farm.asp"),
    # 2020は現時点でWayback上に測尺ページが見つかっていない
}


def fetch_wayback(timestamp: str, url: str) -> BeautifulSoup:
    wb_url = f"https://web.archive.org/web/{timestamp}/{url}"
    resp = requests.get(wb_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = "cp932"
    return BeautifulSoup(resp.text, "html.parser")


def parse_table(soup: BeautifulSoup, bosyu_year: int) -> pd.DataFrame:
    table = soup.find("table")
    rows = table.find_all("tr")
    records = []
    for tr in rows[1:]:  # 先頭行はヘッダー
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) < 6:
            continue
        _, bosyu_name, height, chest, cannon, weight = cells[:6]
        farm = cells[6] if len(cells) > 6 else None
        records.append({
            "bosyu_year": bosyu_year,
            "bosyu_name": bosyu_name,
            "height": _to_float(height),
            "chest": _to_float(chest),
            "cannon": _to_float(cannon),
            "weight": _to_float(weight),
            "farm": farm,
        })
    return pd.DataFrame(records)


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    print("=== キャロットクラブ 測尺データ取得（Wayback Machine） ===\n")
    all_dfs = []
    for bosyu_year, (timestamp, url) in PAGES.items():
        print(f"取得中: {bosyu_year}年度募集 ({url})")
        soup = fetch_wayback(timestamp, url)
        df = parse_table(soup, bosyu_year)
        print(f"  -> {len(df)} 頭")
        all_dfs.append(df)
        time.sleep(2)

    result = pd.concat(all_dfs, ignore_index=True)
    result.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n合計 {len(result)} 頭ぶん保存: {OUT_FILE}")


if __name__ == "__main__":
    main()
