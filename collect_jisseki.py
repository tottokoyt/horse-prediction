"""
collect_jisseki.py
一口馬主DB (umadb.com) からシルクの実績データを収集するスクリプト

収集対象: 2017〜2022年産 (募集年度 2018〜2023 に対応)
収集項目: 馬名・性別・募集価格・獲得賞金・回収率・戦績・厩舎・生産牧場・年度

出力: data/jisseki_YYYY.csv (年度ごと)
     data/jisseki_all.csv  (全年度結合)

使い方:
  pip install requests beautifulsoup4 pandas
  python collect_jisseki.py
"""

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}
BASE_URL = "https://www.umadb.com"

# 産年 -> 募集年度の対応
# umadbは「産年」で管理 (2017年産 = 2018年募集)
YEAR_MAP = {
    2017: 2018,
    2018: 2019,
    2019: 2020,
    2020: 2021,
    2021: 2022,
    2022: 2023,
}


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


def parse_price(text: str):
    """
    募集価格テキストを万円intに変換
    例: "4,000万" -> 4000 / "1億2,000万" -> 12000
    """
    if not text:
        return None
    text = str(text).replace(",", "").replace(" ", "")
    m = re.search(r"(\d+)億(\d+)万?", text)
    if m:
        return int(m.group(1)) * 10000 + int(m.group(2))
    m = re.search(r"(\d+)億", text)
    if m:
        return int(m.group(1)) * 10000
    m = re.search(r"(\d+)万", text)
    if m:
        return int(m.group(1))
    return None


def parse_kakutoku(text: str):
    """
    獲得賞金テキストを万円intに変換
    例: "22億1,482万" -> 221482 / "9,993万" -> 9993
    """
    if not text:
        return None
    text = str(text).replace(",", "").replace(" ", "")
    m = re.search(r"(\d+)億(\d+)万?", text)
    if m:
        return int(m.group(1)) * 10000 + int(m.group(2))
    m = re.search(r"(\d+)億", text)
    if m:
        return int(m.group(1)) * 10000
    m = re.search(r"(\d+)万", text)
    if m:
        return int(m.group(1))
    return None


def parse_senreki(text: str):
    """
    戦績テキストをパース
    例: "10戦8勝" -> (10, 8)
    """
    if not text:
        return None, None
    m = re.search(r"(\d+)戦(\d+)勝", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def collect_year(birth_year: int) -> pd.DataFrame:
    """指定産年のシルク馬データを全ページ収集"""
    bosyu_year = YEAR_MAP[birth_year]
    print(f"=== {birth_year}年産 (募集{bosyu_year}年) 収集開始 ===")

    base_page_url = f"{BASE_URL}/umalist/c104/y{birth_year}/"
    rows = []
    page_n = 0  # nパラメータ (100刻み)

    while True:
        url = base_page_url if page_n == 0 else f"{BASE_URL}/umalist/c104/y{birth_year}/?n={page_n}"
        print(f"  page: {url}")
        soup = fetch_html(url)

        # 馬データテーブルを特定（/uma/ リンクを含む最初のテーブル）
        table = None
        for t in soup.find_all("table"):
            if t.find("a", href=re.compile(r"/uma/")):
                table = t
                break
        if not table:
            break

        trs = table.find_all("tr")
        found_in_page = 0

        for tr in trs:
            tds = tr.find_all("td")
            if len(tds) < 8:
                continue

            try:
                # 回収率列 (1列目)
                kaishuu_text = tds[1].get_text(strip=True)
                kaishuu = int(re.search(r"(\d+)%?", kaishuu_text).group(1)) if re.search(r"\d+", kaishuu_text) else None

                # 馬名・リンク (3列目)
                horse_link = tds[3].find("a")
                if not horse_link:
                    continue
                horse_name = horse_link.get_text(strip=True)
                horse_url = BASE_URL + horse_link["href"]
                horse_id = re.search(r"/uma/(\w+)/", horse_link["href"])
                horse_id = horse_id.group(1) if horse_id else ""

                # 馬齢/性 (4列目)
                age_sex = tds[4].get_text(strip=True)
                sex_m = re.search(r"(牡|牝|セ)", age_sex)
                sex = sex_m.group(1) if sex_m else ""

                # 募集価格 (5列目)
                price_text = tds[5].get_text(strip=True)
                price_man = parse_price(price_text)

                # 獲得賞金 (6列目)
                kakutoku_text = tds[6].get_text(strip=True)
                kakutoku_man = parse_kakutoku(kakutoku_text)

                # 戦績 (7列目)
                senreki_text = tds[7].get_text(strip=True)
                races, wins = parse_senreki(senreki_text)

                # クラス (8列目)
                class_text = tds[8].get_text(strip=True) if len(tds) > 8 else ""

                # 厩舎 (9列目)
                trainer_text = tds[9].get_text(strip=True) if len(tds) > 9 else ""
                trainer = re.sub(r"^\[.\]", "", trainer_text).strip()

                # 生産牧場 (10列目)
                farm_text = tds[10].get_text(strip=True) if len(tds) > 10 else ""

                rows.append({
                    "birth_year": birth_year,
                    "bosyu_year": bosyu_year,
                    "horse_id": horse_id,
                    "horse_name": horse_name,
                    "sex": sex,
                    "price_man": price_man,
                    "kakutoku_man": kakutoku_man,
                    "kaishuu_rate": kaishuu,
                    "races": races,
                    "wins": wins,
                    "class": class_text,
                    "trainer": trainer,
                    "farm": farm_text,
                    "horse_url": horse_url,
                })
                found_in_page += 1

            except Exception as e:
                continue

        print(f"  -> {found_in_page} 頭取得")

        # 次ページ確認
        next_link = soup.find("a", string=re.compile(r"次の\d+頭"))
        if not next_link:
            break
        page_n += 100
        time.sleep(2)

    df = pd.DataFrame(rows)
    print(f"  合計: {len(df)} 頭")
    return df


def main():
    all_dfs = []

    for birth_year in sorted(YEAR_MAP.keys()):
        try:
            df = collect_year(birth_year)
            bosyu_year = YEAR_MAP[birth_year]
            df.to_csv(OUTPUT_DIR / f"jisseki_{bosyu_year}.csv", index=False, encoding="utf-8-sig")
            print(f"  -> data/jisseki_{bosyu_year}.csv 保存完了\n")
            all_dfs.append(df)
        except Exception as e:
            print(f"  [ERROR] {birth_year}年産: {e}\n")
        time.sleep(3)

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all.to_csv(OUTPUT_DIR / "jisseki_all.csv", index=False, encoding="utf-8-sig")
        print(f"全年度結合: data/jisseki_all.csv ({len(df_all)} 頭)")


if __name__ == "__main__":
    main()
