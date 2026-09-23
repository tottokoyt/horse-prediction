"""
fetch_mother_name.py
jisseki_all.csv の各馬の個別ページから母馬名を取得し、
募集データ (bosyu_all.csv) と正しくマッチングするスクリプト

処理の流れ:
  1. jisseki_all.csv の horse_url を1件ずつ取得
  2. 個別ページから母馬名・産年を抽出
  3. 「母馬名+産年2桁」の形式で bosyu_all.csv とマッチング
  4. 結合済みの merged_all.csv / train / valid を再出力

使い方:
  python fetch_mother_name.py

注意: 505頭分のページアクセスが発生します（2秒間隔で約17分）
     途中で止まっても mother_cache.csv に進捗が保存されるので
     再実行すれば続きから再開します
"""

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

DATA_DIR = Path("data")
CACHE_FILE = DATA_DIR / "mother_cache.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}

TRAIN_YEARS = [2018, 2019, 2020, 2021]
VALID_YEARS = [2022, 2023]


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


def extract_mother_info(soup: BeautifulSoup) -> dict:
    """
    umadb個別ページから母馬名・産年を抽出する
    ページ内に「母: ○○○○」「産年: 20XX年」などの情報がある
    """
    result = {"mother_name": None, "birth_year": None, "bosyu_name": None}

    text = soup.get_text()

    # 母馬名の抽出（「母」ラベルの近くにある馬名）
    m = re.search(r"母[：:\s]+([^\n\r\t/（(]+)", text)
    if m:
        result["mother_name"] = m.group(1).strip()

    # 産年の抽出
    m = re.search(r"(\d{4})年産", text)
    if m:
        result["birth_year"] = int(m.group(1))

    # テーブルから取得を試みる
    for th in soup.find_all("th"):
        if "母" in th.get_text():
            td = th.find_next_sibling("td")
            if td:
                mother = td.get_text(strip=True)
                # リンクテキストのみ取得
                a = td.find("a")
                if a:
                    mother = a.get_text(strip=True)
                result["mother_name"] = mother
                break

    # 産年をテーブルから取得
    for th in soup.find_all("th"):
        th_text = th.get_text(strip=True)
        if "生年" in th_text or "産年" in th_text:
            td = th.find_next_sibling("td")
            if td:
                m = re.search(r"(\d{4})", td.get_text())
                if m:
                    result["birth_year"] = int(m.group(1))
                break

    # 募集時の名前（母馬名+産年下2桁）を生成
    if result["mother_name"] and result["birth_year"]:
        year_2digit = str(result["birth_year"])[-2:]
        result["bosyu_name"] = f"{result['mother_name']}の{year_2digit}"

    return result


def load_cache() -> pd.DataFrame:
    """既存キャッシュを読み込む"""
    if CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE, encoding="utf-8-sig")
        print(f"キャッシュ読み込み: {len(df)} 件")
        return df
    return pd.DataFrame(columns=["horse_id", "horse_url", "mother_name", "birth_year", "bosyu_name"])


def save_cache(df: pd.DataFrame):
    df.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")


def fetch_all_mother_names(df_jisseki: pd.DataFrame) -> pd.DataFrame:
    """全馬の母馬名を取得（キャッシュ活用）"""
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
            info = extract_mother_info(soup)
            new_rows.append({
                "horse_id": horse_id,
                "horse_url": url,
                **info,
            })

            if (i + 1) % 10 == 0 or i == total - 1:
                # 10件ごとにキャッシュ保存
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
                "birth_year": None,
                "bosyu_name": None,
            })

        time.sleep(2)

    # 最終保存
    if new_rows:
        cache = pd.concat(
            [cache, pd.DataFrame(new_rows)],
            ignore_index=True
        ).drop_duplicates(subset=["horse_id"])
        save_cache(cache)

    return cache


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.strip()
    name = re.sub(r"[\s　]+", "", name)
    return name


def rematch(df_bosyu: pd.DataFrame, df_jisseki: pd.DataFrame, df_mother: pd.DataFrame) -> pd.DataFrame:
    """
    母馬名情報を使って bosyu と jisseki を再マッチング
    """
    # jisseki に母馬名情報をくっつける
    df_j = pd.merge(
        df_jisseki,
        df_mother[["horse_id", "mother_name", "birth_year", "bosyu_name"]],
        on="horse_id",
        how="left",
    )

    # bosyu_name を正規化
    df_j["bosyu_name_norm"] = df_j["bosyu_name"].apply(normalize_name)
    df_bosyu["horse_name_norm"] = df_bosyu["horse_name"].apply(normalize_name)

    # マージ
    jisseki_cols = [
        "bosyu_name_norm", "bosyu_year",
        "horse_name", "horse_id", "horse_url",
        "kakutoku_man", "kaishuu_rate", "races", "wins", "class", "farm",
        "mother_name",
    ]
    df_j_sub = df_j[[c for c in jisseki_cols if c in df_j.columns]].copy()
    df_j_sub = df_j_sub.rename(columns={
        "bosyu_name_norm": "horse_name_norm",
        "horse_name": "race_horse_name",  # 競走馬名（イクイノックスなど）
    })

    df = pd.merge(
        df_bosyu,
        df_j_sub,
        on=["horse_name_norm", "bosyu_year"],
        how="left",
    )

    # 回収率が取れなかった馬は0埋め
    df["kaishuu_rate"] = df["kaishuu_rate"].fillna(0).astype(int)
    df["kakutoku_man"] = df["kakutoku_man"].fillna(0)
    df["races"]        = df["races"].fillna(0).astype(int)
    df["wins"]         = df["wins"].fillna(0).astype(int)

    # 回収率カテゴリ
    df["kaishuu_category"] = pd.cut(
        df["kaishuu_rate"],
        bins=[-1, 0, 50, 100, 200, 99999],
        labels=["未出走/0%", "50%未満", "50〜100%", "100〜200%", "200%超"],
    )

    matched = df["horse_id"].notna().sum()
    print(f"マッチ: {matched}/{len(df)} 頭 ({matched/len(df)*100:.1f}%)")

    return df


def main():
    print("=== 母馬名取得・再マッチング ===\n")

    df_bosyu   = pd.read_csv(DATA_DIR / "bosyu_all.csv",   encoding="utf-8-sig")
    df_jisseki = pd.read_csv(DATA_DIR / "jisseki_all.csv", encoding="utf-8-sig")

    df_bosyu["bosyu_year"] = df_bosyu["year"] if "year" in df_bosyu.columns else df_bosyu["bosyu_year"]

    # 母馬名を取得
    df_mother = fetch_all_mother_names(df_jisseki)
    print(f"\n母馬名取得完了: {df_mother['bosyu_name'].notna().sum()}/{len(df_mother)} 件\n")

    # 再マッチング
    print("=== 再マッチング ===")
    df_all = rematch(df_bosyu, df_jisseki, df_mother)

    # 分割・保存
    df_train = df_all[df_all["bosyu_year"].isin(TRAIN_YEARS)].copy()
    df_valid = df_all[df_all["bosyu_year"].isin(VALID_YEARS)].copy()

    df_all.to_csv(  DATA_DIR / "merged_all.csv",   index=False, encoding="utf-8-sig")
    df_train.to_csv(DATA_DIR / "merged_train.csv", index=False, encoding="utf-8-sig")
    df_valid.to_csv(DATA_DIR / "merged_valid.csv", index=False, encoding="utf-8-sig")

    print(f"\n--- 出力完了 ---")
    print(f"merged_all.csv   : {len(df_all)} 頭")
    print(f"merged_train.csv : {len(df_train)} 頭 (学習用)")
    print(f"merged_valid.csv : {len(df_valid)} 頭 (検証用)")

    print("\n--- 回収率カテゴリ分布 (学習用) ---")
    print(df_train["kaishuu_category"].value_counts())


if __name__ == "__main__":
    main()
