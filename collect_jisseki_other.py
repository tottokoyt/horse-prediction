"""
collect_jisseki_other.py
一口馬主DB (umadb.com) から10クラブの実績データを収集するスクリプト

収集クラブ:
  carrot / sunday / shadai / g1 / normandy /
  tokyo / raffian / union / win / road

収集産年: 2017〜2022年産（募集年度 2018〜2023）
出力:     data/jisseki_other_all.csv（club_name 列付き）

使い方:
  python collect_jisseki_other.py
"""

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

HEADERS  = {"User-Agent": "Mozilla/5.0"}
BASE_URL = "https://www.umadb.com"

CLUBS = {
    "carrot":   "c105",
    "sunday":   "c102",
    "shadai":   "c101",
    "g1":       "c120",
    "normandy": "c121",
    "tokyo":    "c118",
    "raffian":  "c103",
    "union":    "c106",
    "win":      "c107",
    "road":     "c112",
}

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
    if not text:
        return None, None
    m = re.search(r"(\d+)戦(\d+)勝", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def collect_club_year(club_name: str, club_code: str, birth_year: int) -> pd.DataFrame:
    """指定クラブ・産年のデータを全ページ収集"""
    bosyu_year = YEAR_MAP[birth_year]
    print(f"  [{club_name}] {birth_year}年産 (募集{bosyu_year}年) 収集開始")

    base_page_url = f"{BASE_URL}/umalist/{club_code}/y{birth_year}/"
    rows = []
    page_n = 0

    while True:
        url = base_page_url if page_n == 0 else f"{BASE_URL}/umalist/{club_code}/y{birth_year}/?n={page_n}"
        print(f"    page: {url}")
        try:
            soup = fetch_html(url)
        except Exception as e:
            print(f"    [ERROR] ページ取得失敗: {e}")
            break

        # 馬データテーブルを特定（/uma/ リンクを含む最初のテーブル）
        table = None
        for t in soup.find_all("table"):
            if t.find("a", href=re.compile(r"/uma/")):
                table = t
                break
        if not table:
            print(f"    テーブルなし → 終了")
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
                horse_url  = BASE_URL + horse_link["href"]
                horse_id_m = re.search(r"/uma/(\w+)/", horse_link["href"])
                horse_id   = horse_id_m.group(1) if horse_id_m else ""

                # 馬齢/性 (4列目)
                age_sex = tds[4].get_text(strip=True)
                sex_m   = re.search(r"(牡|牝|セ)", age_sex)
                sex     = sex_m.group(1) if sex_m else ""

                # 募集価格 (5列目)
                price_man = parse_price(tds[5].get_text(strip=True))

                # 獲得賞金 (6列目)
                kakutoku_man = parse_kakutoku(tds[6].get_text(strip=True))

                # 戦績 (7列目)
                races, wins = parse_senreki(tds[7].get_text(strip=True))

                # クラス (8列目)
                class_text = tds[8].get_text(strip=True) if len(tds) > 8 else ""

                # 厩舎 (9列目)
                trainer_text = tds[9].get_text(strip=True) if len(tds) > 9 else ""
                trainer = re.sub(r"^\[.\]", "", trainer_text).strip()

                # 生産牧場 (10列目)
                farm_text = tds[10].get_text(strip=True) if len(tds) > 10 else ""

                rows.append({
                    "club_name":    club_name,
                    "birth_year":   birth_year,
                    "bosyu_year":   bosyu_year,
                    "horse_id":     horse_id,
                    "horse_name":   horse_name,
                    "sex":          sex,
                    "price_man":    price_man,
                    "kakutoku_man": kakutoku_man,
                    "kaishuu_rate": kaishuu,
                    "races":        races,
                    "wins":         wins,
                    "class":        class_text,
                    "trainer":      trainer,
                    "farm":         farm_text,
                    "horse_url":    horse_url,
                })
                found_in_page += 1

            except Exception:
                continue

        print(f"    -> {found_in_page} 頭取得")

        # 次ページ確認
        next_link = soup.find("a", string=re.compile(r"次の\d+頭"))
        if not next_link:
            break
        page_n += 100
        time.sleep(2)

    return pd.DataFrame(rows)


def main():
    all_dfs = []
    total_clubs = len(CLUBS)

    for club_idx, (club_name, club_code) in enumerate(CLUBS.items(), 1):
        print(f"\n=== [{club_idx}/{total_clubs}] {club_name} ({club_code}) ===")
        club_dfs = []

        for birth_year in sorted(YEAR_MAP.keys()):
            try:
                df = collect_club_year(club_name, club_code, birth_year)
                club_dfs.append(df)
                print(f"    {birth_year}年産: {len(df)} 頭")
            except Exception as e:
                print(f"    [ERROR] {club_name} {birth_year}年産: {e}")
            time.sleep(2)

        if club_dfs:
            df_club = pd.concat(club_dfs, ignore_index=True)
            all_dfs.append(df_club)
            print(f"  {club_name} 合計: {len(df_club)} 頭")
        else:
            print(f"  {club_name}: データなし")

        time.sleep(3)

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        out_path = OUTPUT_DIR / "jisseki_other_all.csv"
        df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n=== 完了 ===")
        print(f"総頭数: {len(df_all)} 頭")
        print(f"出力: {out_path}")
        print("\nクラブ別頭数:")
        print(df_all.groupby("club_name")["horse_id"].count().sort_values(ascending=False).to_string())
    else:
        print("データが取得できませんでした")


if __name__ == "__main__":
    main()
