"""
fetch_union_scale.py
ユニオン・オーナーズ・クラブの募集時測尺（体高・胸囲・管囲。この時期は体重の掲載なし）を
Wayback Machine（web.archive.org）の個別馬ページ（/id/NNNN）アーカイブから取得し、
jisseki_other_all.csv の horse_id と突き合わせて other_scale_cache.csv に追記する

背景:
  学習年度(2018-2021募集 = 2017-2020年産)の馬は現行サイトでは既にページが無い
  （引退馬はリダイレクト）。Wayback Machine に残る個別馬ページには
  「測尺(cm)」欄に計測履歴が複数時点ぶん載っている、例:
    体高：155 胸囲：181 管囲：20.0（2018年10月現在）
    体高：155 胸囲：174 管囲：20.0（2018年6月現在）
  募集時に最も近い「最も早い計測時点」の値を採用する。

  個別ページの ID と世代の対応はおおむね 3200番台=2017年産 〜 3500番台=2020年産。
  ID範囲は広めに取り、生年月日で学習年度に絞り込む。

  マッチングは (募集年度, 母馬名) を優先し、取れなければ (募集年度, 馬名) で補完する
  （デビュー後のキャプチャには競走馬名も載っているため）。

出力:
  data/union_scale_raw.csv
  data/other_scale_cache.csv に union 分を追記（weight は NaN）

使い方:
  python fetch_union_scale.py
"""

import re
import time
import unicodedata
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "_wayback_cache" / "union"
RAW_FILE = DATA_DIR / "union_scale_raw.csv"
SCALE_CACHE = DATA_DIR / "other_scale_cache.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}

TRAIN_YEARS = [2018, 2019, 2020, 2021]
ID_MIN, ID_MAX = 3150, 3700

# 年度によって「体高：151」「体高：151cm」の両方の表記がある
MEASURE_RE = re.compile(
    r"体高：\s*([\d.]+)(?:cm)?\s*胸囲：\s*([\d.]+)(?:cm)?\s*管囲：\s*([\d.]+)(?:cm)?"
    r"(?:\s*体重：\s*([\d.]+)(?:kg)?)?\s*（(\d{4})年(\d{1,2})月"
)


def list_captures():
    resp = requests.get(
        "https://web.archive.org/cdx/search/cdx",
        params={
            "url": "union-oc.co.jp/id/*", "output": "txt",
            "fl": "timestamp,original", "filter": "statuscode:200",
            "collapse": "urlkey", "limit": 5000,
        },
        timeout=120,
    )
    resp.raise_for_status()
    captures = []
    for line in resp.text.splitlines():
        ts, url = line.split()
        m = re.search(r"/id/(\d+)", url)
        if m and ID_MIN <= int(m.group(1)) <= ID_MAX:
            captures.append((int(m.group(1)), ts, url))
    return captures


def fetch_page(page_id: int, ts: str, url: str) -> str:
    path = CACHE_DIR / f"{page_id}.html"
    if path.exists() and path.stat().st_size > 0:
        return path.read_text(encoding="utf-8", errors="ignore")
    resp = requests.get(f"https://web.archive.org/web/{ts}id_/{url}", headers=HEADERS, timeout=60)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    path.write_text(resp.text, encoding="utf-8")
    time.sleep(3)
    return resp.text


def parse_page(html: str, page_id: int):
    soup = BeautifulSoup(html, "html.parser")
    fields = {}
    for th in soup.select("table.umadetailtbl th"):
        td = th.find_next_sibling("td")
        if td is not None:
            fields[th.get_text(strip=True)] = td.get_text("\n", strip=True)

    birth = re.match(r"(\d{4})/", fields.get("生年月日", ""))
    measures = MEASURE_RE.findall(fields.get("測尺(cm)", "").replace("\n", " "))
    if not (birth and measures):
        return None

    # 最も早い計測時点（募集時に最も近い値）
    h, c, k, w, y, mo = min(measures, key=lambda m: (int(m[4]), int(m[5])))
    return {
        "page_id": page_id,
        "bosyu_year": int(birth.group(1)) + 1,
        "horse_name_page": _page_horse_name(fields),
        "mother_name": unicodedata.normalize("NFKC", fields.get("母", "")).lstrip("*"),
        "sex": fields.get("性別"),
        "height": float(h), "chest": float(c), "cannon": float(k),
        "weight": float(w) if w else None,
        "measured_ym": f"{y}-{int(mo):02d}",
        "n_measurements": len(measures),
    }


def _page_horse_name(fields):
    # 「馬名意味・由来」は「クイーンズバラッド…女王の物語詩」の形式
    v = fields.get("馬名\n意味・由来") or fields.get("馬名意味・由来") or ""
    return v.split("…")[0].strip() or None


def match_to_jisseki(scale: pd.DataFrame) -> pd.DataFrame:
    other = pd.read_csv(DATA_DIR / "jisseki_other_all.csv", encoding="utf-8-sig")
    target = other[(other["club_name"] == "union") & other["bosyu_year"].isin(TRAIN_YEARS)].copy()
    mother = pd.read_csv(DATA_DIR / "mother_other_cache.csv", encoding="utf-8-sig")[["horse_id", "mother_name"]]
    target = pd.merge(target, mother, on="horse_id", how="left")
    target["mother_name"] = target["mother_name"].map(
        lambda x: unicodedata.normalize("NFKC", x).lstrip("*") if isinstance(x, str) else x)

    scale_cols = ["height", "chest", "cannon", "weight"]
    by_mother = pd.merge(
        target[["horse_id", "horse_name", "bosyu_year", "mother_name"]],
        scale[["bosyu_year", "mother_name"] + scale_cols],
        on=["bosyu_year", "mother_name"], how="inner",
    )
    rest = target[~target["horse_id"].isin(by_mother["horse_id"])]
    by_name = pd.merge(
        rest[["horse_id", "horse_name", "bosyu_year"]],
        scale.dropna(subset=["horse_name_page"])[["bosyu_year", "horse_name_page"] + scale_cols],
        left_on=["bosyu_year", "horse_name"], right_on=["bosyu_year", "horse_name_page"], how="inner",
    )
    merged = pd.concat([by_mother, by_name], ignore_index=True).drop_duplicates(subset=["horse_id"])

    summary = pd.DataFrame({
        "total": target.groupby("bosyu_year").size(),
        "matched": merged.groupby("bosyu_year").size(),
    }).fillna(0).astype(int)
    print("\n募集年度別マッチ状況:")
    print(summary.to_string())
    print(f"合計: {len(merged)} / {len(target)} 頭（母馬名 {len(by_mother)}, 馬名で補完 {len(merged) - len(by_mother)}）")
    return merged


def main():
    print("=== ユニオン 測尺データ取得（Wayback Machine） ===\n")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    captures = list_captures()
    print(f"対象ページ: {len(captures)} 件（ID {ID_MIN}〜{ID_MAX}）")
    for page_id, ts, url in captures:
        try:
            rec = parse_page(fetch_page(page_id, ts, url), page_id)
        except requests.RequestException as e:
            print(f"  取得失敗 id={page_id}: {e}")
            continue
        if rec:
            records.append(rec)

    raw = pd.DataFrame(records).sort_values("page_id")
    raw.to_csv(RAW_FILE, index=False, encoding="utf-8-sig")
    print(f"保存: {RAW_FILE}（{len(raw)} 頭、学習年度内 {raw['bosyu_year'].isin(TRAIN_YEARS).sum()} 頭）")

    merged = match_to_jisseki(raw)
    merged["farm"] = None
    out_cols = ["horse_id", "horse_name", "bosyu_year", "height", "chest", "cannon", "weight", "farm"]
    if SCALE_CACHE.exists():
        existing = pd.read_csv(SCALE_CACHE, encoding="utf-8-sig")
        combined = pd.concat([existing, merged[out_cols]], ignore_index=True).drop_duplicates(subset=["horse_id"])
    else:
        combined = merged[out_cols]
    combined.to_csv(SCALE_CACHE, index=False, encoding="utf-8-sig")
    print(f"保存: {SCALE_CACHE}（合計 {len(combined)} 頭）")


if __name__ == "__main__":
    main()
