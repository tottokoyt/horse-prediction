"""
collect_bosyu.py
シルクホースクラブ 募集データ収集スクリプト

収集対象：
  2018年: 個人ブログ (テキスト)
  2019年: 公式PDF x2 (募集一覧 + 測尺)
  2020年: 公式サイト一覧ページ + 測尺PDF
  2021年: 公式PDF x2 (募集確定 + 測尺)
  2022年: 公式PDF x3 (美浦 + 栗東 + 測尺)
  2023年: 公式PDF x2 (募集一覧 + 測尺)

出力: data/bosyu_YYYY.csv (年度ごと)
     data/bosyu_all.csv  (全年度結合)

使い方:
  pip install requests beautifulsoup4 pdfplumber
  python collect_bosyu.py
"""

import re
import time
import requests
import pdfplumber
import pandas as pd
from bs4 import BeautifulSoup
from io import BytesIO
from pathlib import Path

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}


# ──────────────────────────────────────────
# 共通ユーティリティ
# ──────────────────────────────────────────

def fetch_pdf(url: str) -> BytesIO:
    """URLからPDFをダウンロードしてBytesIOで返す"""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BytesIO(resp.content)


def fetch_html(url: str) -> BeautifulSoup:
    """URLからHTMLを取得してBeautifulSoupで返す"""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def clean_price(text: str) -> int:
    """
    価格文字列を万円単位のintに変換
    例: "80,000円" -> 8  (一口価格は円表記なので万円に変換)
        "4,000万円" -> 4000
        "1億2,000万" -> 12000
    """
    if not text:
        return None
    text = str(text).replace(",", "").replace(" ", "").replace("\u3000", "")
    # 億円
    m = re.search(r"(\d+)億(\d+)万", text)
    if m:
        return int(m.group(1)) * 10000 + int(m.group(2))
    m = re.search(r"(\d+)億", text)
    if m:
        return int(m.group(1)) * 10000
    # 万円
    m = re.search(r"(\d+)万", text)
    if m:
        return int(m.group(1))
    # 円（一口価格）→ 万円換算
    m = re.search(r"(\d+)円?$", text)
    if m:
        val = int(m.group(1))
        return round(val / 10000, 2)
    return None


# ──────────────────────────────────────────
# 2018年: 個人ブログからスクレイピング
# ──────────────────────────────────────────

def collect_2018() -> pd.DataFrame:
    print("=== 2018年 収集開始 ===")
    url = "http://uma1kuti.blog106.fc2.com/blog-entry-5575.html"
    soup = fetch_html(url)

    rows = []
    # ブログ本文のテキストから表形式データを抽出
    # 形式: No. 馬名 父馬 性別 体高 胸囲 管囲 体重 生月日 総額(万円) 一口価格(円) 厩舎
    text = soup.get_text()
    lines = text.split("\n")

    pattern = re.compile(
        r"(\d+)\s+"                          # No
        r"(.+?の17)\s+"                       # 馬名
        r"(\S+(?:\s+\S+)?)\s+"               # 父馬
        r"(牡|牝|めす|メス|セ)\s+"            # 性別
        r"([\d.]+)\s+"                        # 体高
        r"([\d.]+)\s+"                        # 胸囲
        r"([\d.]+)\s+"                        # 管囲
        r"(\d+)\s+"                           # 体重
        r"(\d+月\d+日)\s+"                    # 生月日
        r"([\d,]+)\s+"                        # 総額
        r"([\d,]+)円\s+"                      # 一口価格
        r"(.+)"                               # 厩舎
    )

    for line in lines:
        line = line.strip()
        m = pattern.match(line)
        if m:
            rows.append({
                "year": 2018,
                "no": int(m.group(1)),
                "horse_name": m.group(2).strip(),
                "sire": m.group(3).strip(),
                "sex": normalize_sex(m.group(4)),
                "height": float(m.group(5)),
                "chest": float(m.group(6)),
                "cannon": float(m.group(7)),
                "weight": int(m.group(8)),
                "birth_month": m.group(9),
                "total_price_man": int(m.group(10).replace(",", "")),
                "price_per_kuchi": int(m.group(11).replace(",", "")),
                "trainer": m.group(12).strip(),
            })

    df = pd.DataFrame(rows)
    print(f"  {len(df)} 頭取得")
    return df


def normalize_sex(s: str) -> str:
    mapping = {"牡": "牡", "牝": "牝", "めす": "牝", "メス": "牝", "セ": "セ"}
    return mapping.get(s, s)


# ──────────────────────────────────────────
# PDFから測尺データを抽出する共通関数
# ──────────────────────────────────────────

def extract_scale_from_pdf(pdf_io: BytesIO, year: int) -> pd.DataFrame:
    """
    測尺PDFから体高・胸囲・管囲・体重を抽出
    extract_tables()でテーブルを取得し、ヘッダから列インデックスを動的に特定する
    """
    rows = []

    with pdfplumber.open(pdf_io) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                header = table[0]
                header_text = " ".join(str(h or "") for h in header)
                if "体高" not in header_text and "No" not in header_text:
                    continue

                col = {}
                for i, h in enumerate(header):
                    h_str = str(h or "").replace("\n", " ").strip()
                    if re.match(r"^No\.?$", h_str):
                        col["no"] = i
                    elif re.search(r"^募集馬$", h_str):
                        col["horse_name"] = i
                    elif "性別" in h_str:
                        col["sex"] = i
                    elif "体高" in h_str:
                        col["height"] = i
                    elif "胸囲" in h_str:
                        col["chest"] = i
                    elif "管囲" in h_str:
                        col["cannon"] = i
                    elif re.search(r"体重|馬体重", h_str):
                        col["weight"] = i

                if "horse_name" not in col or "height" not in col:
                    continue

                for row in table[1:]:
                    if not row:
                        continue
                    row = [str(c or "").replace("﻿", "").strip() for c in row]

                    hn_idx = col.get("horse_name")
                    horse_name = row[hn_idx] if hn_idx is not None and hn_idx < len(row) else ""
                    if not horse_name or horse_name in ("募集馬", ""):
                        continue

                    no_idx = col.get("no")
                    no_str = row[no_idx] if no_idx is not None and no_idx < len(row) else ""
                    if not re.match(r"^\d+$", no_str):
                        continue
                    no = int(no_str)

                    def gcol(key):
                        idx = col.get(key)
                        return row[idx] if idx is not None and idx < len(row) else ""

                    try:
                        rows.append({
                            "no": no,
                            "horse_name": horse_name.strip(),
                            "sex": normalize_sex(gcol("sex")),
                            "height": float(gcol("height")),
                            "chest": float(gcol("chest")),
                            "cannon": float(gcol("cannon")),
                            "weight": int(float(gcol("weight"))),
                        })
                    except Exception:
                        continue

    return pd.DataFrame(rows).drop_duplicates(subset=["no"]) if rows else pd.DataFrame()


# ──────────────────────────────────────────
# PDFから募集一覧データを抽出する共通関数
# ──────────────────────────────────────────

def safe_parse_kuchi(x):
    """一口価格テキストから数値のみ抽出（円）"""
    m = re.search(r"(\d[\d,]*)", str(x))
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


def extract_bosyu_from_pdf(pdf_io: BytesIO, year: int) -> pd.DataFrame:
    """
    募集一覧PDFから馬名・父馬・性別・毛色・生月日・募集総額・一口価格・厩舎・牧場を抽出
    ヘッダ行から列インデックスを動的に特定する（年度ごとの列順差異に対応）
    """
    rows = []
    auto_no = 0

    with pdfplumber.open(pdf_io) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue

                # データテーブルかどうか確認（ヘッダ行に No. と 馬 が含まれる）
                header = table[0]
                header_text = " ".join(str(h or "") for h in header)
                if "No" not in header_text or "馬" not in header_text:
                    continue

                # ヘッダから列インデックスを特定
                col = {}
                for i, h in enumerate(header):
                    h_str = str(h or "").replace("\n", " ").strip()
                    if re.match(r"^No\.?$", h_str):
                        col["no"] = i
                    elif re.search(r"(父馬|種牡馬)", h_str):
                        col["sire"] = i
                    elif re.search(r"(募集馬|馬名)", h_str):
                        col["horse_name"] = i
                    elif "性別" in h_str:
                        col["sex"] = i
                    elif "毛色" in h_str:
                        col["color"] = i
                    elif "生月日" in h_str or "生年月日" in h_str:
                        col["birth_month"] = i
                    elif "総額" in h_str:
                        col["total_price"] = i
                    elif "一口" in h_str:
                        col["price_per_kuchi"] = i
                    elif "牧場" in h_str:
                        col["farm"] = i
                    elif "cid" in h_str or "舎" in h_str or "預" in h_str:
                        col["trainer"] = i

                if "horse_name" not in col:
                    continue

                # データ行を抽出
                for row in table[1:]:
                    if not row:
                        continue
                    row = [str(c or "").strip() for c in row]

                    hn_idx = col.get("horse_name")
                    horse_name = row[hn_idx] if hn_idx is not None and hn_idx < len(row) else ""
                    if not horse_name or horse_name in ("募集馬", "募集馬名", ""):
                        continue

                    no_idx = col.get("no")
                    no_str = row[no_idx] if no_idx is not None and no_idx < len(row) else ""
                    if re.match(r"^\d+$", no_str):
                        no = int(no_str)
                    else:
                        auto_no += 1
                        no = auto_no

                    def gcol(key, default=""):
                        idx = col.get(key)
                        return row[idx] if idx is not None and idx < len(row) else default

                    try:
                        rows.append({
                            "no": no,
                            "horse_name": horse_name,
                            "sire": gcol("sire"),
                            "sex": normalize_sex(gcol("sex")),
                            "color": gcol("color"),
                            "birth_month": gcol("birth_month"),
                            "total_price_man": clean_price(gcol("total_price")),
                            "price_per_kuchi": safe_parse_kuchi(gcol("price_per_kuchi")),
                            "trainer": gcol("trainer"),
                            "farm": gcol("farm"),
                        })
                    except Exception:
                        continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["no"])


# ──────────────────────────────────────────
# 2019年
# ──────────────────────────────────────────

def collect_2019() -> pd.DataFrame:
    print("=== 2019年 収集開始 ===")
    bosyu_url = "https://www2.silkhorseclub.jp/wp/wp-content/uploads/2019/06/2019_bosyu061517084.pdf"
    scale_url = "https://www2.silkhorseclub.jp/wp/wp-content/uploads/2019/07/2019sokusyaku.pdf"

    df_bosyu = extract_bosyu_from_pdf(fetch_pdf(bosyu_url), 2019)
    time.sleep(1)
    df_scale = extract_scale_from_pdf(fetch_pdf(scale_url), 2019)

    df = merge_bosyu_scale(df_bosyu, df_scale, 2019)
    print(f"  {len(df)} 頭取得")
    return df


# ──────────────────────────────────────────
# 2020年
# ──────────────────────────────────────────

def collect_2020() -> pd.DataFrame:
    print("=== 2020年 収集開始 ===")
    list_url = "https://www2.silkhorseclub.jp/wp/2020rec"
    scale_url = "https://www2.silkhorseclub.jp/wp/wp-content/uploads/2020/07/2020sokusyaku.pdf"

    soup = fetch_html(list_url)
    rows = []

    # 各馬のブロックを抽出
    for item in soup.select("a[href*='/horse_bosyu/2020/']"):
        parent = item.find_parent()
        if not parent:
            continue
        try:
            no_text = item.text.strip()
            m = re.match(r"(\d+)\s+(.+)", no_text)
            if not m:
                continue
            no = int(m.group(1))
            horse_name = m.group(2).strip()

            # 隣接テーブルから情報取得
            table = parent.find_next("table")
            if not table:
                continue
            tds = [td.get_text(strip=True) for td in table.find_all("td")]

            # テーブル構造: [性別, 毛色, 生月日, 厩舎, 父馬, (空), 総額, 一口価格(円), (空)]
            if len(tds) >= 8:
                row = {
                    "year": 2020,
                    "no": no,
                    "horse_name": horse_name,
                    "sex": normalize_sex(tds[0]),
                    "color": tds[1],
                    "birth_month": tds[2],
                    "trainer": re.sub(r"\s+", "", tds[3]),
                    "sire": tds[4],
                    "total_price_man": clean_price(tds[6]),
                    "price_per_kuchi": safe_parse_kuchi(tds[7]),
                }
            else:
                row = {"year": 2020, "no": no, "horse_name": horse_name}
            rows.append(row)
        except Exception as e:
            print(f"    skip: {e}")
            continue

    df_bosyu = pd.DataFrame(rows)
    # 2020年は公式サイトのHTML一覧ページから直接スクレイピングしており、
    # 同じ馬へのリンク（サムネイル・馬名テキスト・詳細ボタン等）が
    # ページ内に複数存在すると同じ行が重複してしまう構造的リスクがある
    # （2026-09-23、実際に全馬が正確に7重複しているバグを発見・修正した）。
    # 募集no（1頭1no）をキーに必ず重複除去する
    n_before = len(df_bosyu)
    df_bosyu = df_bosyu.drop_duplicates(subset=["no"]).reset_index(drop=True)
    if len(df_bosyu) < n_before:
        print(f"  [重複除去] {n_before}行 → {len(df_bosyu)}行")
    time.sleep(1)

    df_scale = extract_scale_from_pdf(fetch_pdf(scale_url), 2020)
    df = merge_bosyu_scale(df_bosyu, df_scale, 2020)
    print(f"  {len(df)} 頭取得")
    return df


# ──────────────────────────────────────────
# 2021年
# ──────────────────────────────────────────

def collect_2021() -> pd.DataFrame:
    print("=== 2021年 収集開始 ===")
    bosyu_url = "https://www2.silkhorseclub.jp/wp/wp-content/uploads/2021/06/2021Confirm_page1.pdf"
    scale_url = "https://www2.silkhorseclub.jp/wp/wp-content/uploads/2021/07/2021.pdf"

    df_bosyu = extract_bosyu_from_pdf(fetch_pdf(bosyu_url), 2021)
    time.sleep(1)
    df_scale = extract_scale_from_pdf(fetch_pdf(scale_url), 2021)

    df = merge_bosyu_scale(df_bosyu, df_scale, 2021)
    print(f"  {len(df)} 頭取得")
    return df


# ──────────────────────────────────────────
# 2022年
# ──────────────────────────────────────────

def collect_2022() -> pd.DataFrame:
    print("=== 2022年 収集開始 ===")
    miho_url  = "https://www2.silkhorseclub.jp/wp/wp-content/uploads/2022/06/Recruitment2022_Miho.pdf"
    ritto_url = "https://www2.silkhorseclub.jp/wp/wp-content/uploads/2022/06/Recruitment2022_Ritto.pdf"
    scale_url = "https://www2.silkhorseclub.jp/wp/pdf/2022scale.pdf"

    df_miho  = extract_bosyu_from_pdf(fetch_pdf(miho_url),  2022)
    time.sleep(1)
    df_ritto = extract_bosyu_from_pdf(fetch_pdf(ritto_url), 2022)
    time.sleep(1)
    df_bosyu = pd.concat([df_miho, df_ritto], ignore_index=True).drop_duplicates(subset=["no"])

    df_scale = extract_scale_from_pdf(fetch_pdf(scale_url), 2022)
    df = merge_bosyu_scale(df_bosyu, df_scale, 2022)
    print(f"  {len(df)} 頭取得")
    return df


# ──────────────────────────────────────────
# 2023年
# ──────────────────────────────────────────

def collect_2023() -> pd.DataFrame:
    print("=== 2023年 収集開始 ===")
    bosyu_url = "https://www.silkhorseclub.jp/assets_static/pdf/boshu_list2023.pdf"
    scale_url = "https://www.silkhorseclub.jp/assets_static/pdf/scale_list2023.pdf"

    df_bosyu = extract_bosyu_from_pdf(fetch_pdf(bosyu_url), 2023)
    time.sleep(1)
    df_scale = extract_scale_from_pdf(fetch_pdf(scale_url), 2023)

    df = merge_bosyu_scale(df_bosyu, df_scale, 2023)
    print(f"  {len(df)} 頭取得")
    return df


# ──────────────────────────────────────────
# 募集データと測尺データの結合
# ──────────────────────────────────────────

def merge_bosyu_scale(df_bosyu: pd.DataFrame, df_scale: pd.DataFrame, year: int) -> pd.DataFrame:
    """No をキーに募集情報と測尺を結合し、year列を付与"""
    if df_bosyu.empty or "no" not in df_bosyu.columns:
        print(f"  [WARN] {year}年: 募集データが空です")
        return pd.DataFrame()

    if df_scale.empty or "no" not in df_scale.columns:
        print(f"  [WARN] {year}年: 測尺データが空、募集データのみで返します")
        df = df_bosyu.copy()
        df["year"] = year
        return df

    scale_cols = ["no"] + [c for c in ["height", "chest", "cannon", "weight"] if c in df_scale.columns]
    df = pd.merge(df_bosyu, df_scale[scale_cols], on="no", how="left", suffixes=("", "_scale"))

    for col in ["height", "chest", "cannon", "weight"]:
        if f"{col}_scale" in df.columns:
            df[col] = df[f"{col}_scale"].combine_first(df.get(col, pd.Series(dtype=float)))
            df.drop(columns=[f"{col}_scale"], inplace=True)

    df["year"] = year
    return df


# ──────────────────────────────────────────
# メイン
# ──────────────────────────────────────────

def main():
    all_dfs = []

    collectors = [
        (2018, collect_2018),
        (2019, collect_2019),
        (2020, collect_2020),
        (2021, collect_2021),
        (2022, collect_2022),
        (2023, collect_2023),
    ]

    for year, func in collectors:
        try:
            df = func()
            df["year"] = year
            df.to_csv(OUTPUT_DIR / f"bosyu_{year}.csv", index=False, encoding="utf-8-sig")
            print(f"  -> data/bosyu_{year}.csv 保存完了")
            all_dfs.append(df)
        except Exception as e:
            print(f"  [ERROR] {year}年: {e}")
        time.sleep(2)

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all.to_csv(OUTPUT_DIR / "bosyu_all.csv", index=False, encoding="utf-8-sig")
        print(f"\n全年度結合: data/bosyu_all.csv ({len(df_all)} 頭)")


if __name__ == "__main__":
    main()
