"""
fetch_normandy_scale.py
ノルマンディーオーナーズクラブの募集時測尺（体高・胸囲・管囲・馬体重）を
Wayback Machine（web.archive.org）のアーカイブページから取得し、
jisseki_other_all.csv の horse_id と突き合わせて other_scale_cache.csv に追記する

背景:
  公式サイトの測尺ページ（/collect/height.aspx）は「その時点で募集中の世代」
  しか表示せず、満口になった馬は一覧から消える。現行サイトの6歳以上タブは
  現役馬数頭の「現在の」測尺のみで、募集時の値ではないため使えない。
  そこで Wayback Machine 上の複数時点のキャプチャを合算し、
  各馬について「最も早いキャプチャ時点の値」＝募集時に最も近い測尺を採用する
  （再計測で値が更新される馬がいるため、後のキャプチャは使わない）。

  馬名は「母馬名の産年(全角2桁)」の募集時仮名（例: フィアレスの１８）なので、
  NFKC正規化した上で mother_other_cache.csv の母馬名 + 募集年度で突き合わせる
  （募集年度 = 産年 + 1）。

出力:
  data/normandy_scale_raw.csv
    columns: bosyu_year, bosyu_name, mother_name, sex, height, chest, cannon, weight, capture_ts
  data/other_scale_cache.csv に normandy 分を追記

使い方:
  python fetch_normandy_scale.py
"""

import re
import time
import unicodedata
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "_wayback_cache" / "normandy"
RAW_FILE = DATA_DIR / "normandy_scale_raw.csv"
SCALE_CACHE = DATA_DIR / "other_scale_cache.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}

TARGET_URL = "http://www.normandyoc.com/collect/height.aspx"
TRAIN_YEARS = [2018, 2019, 2020, 2021]
# 学習年度の募集期間をカバーするキャプチャ範囲
CAPTURE_FROM, CAPTURE_TO = "2018", "2022"


def list_captures():
    resp = requests.get(
        "https://web.archive.org/cdx/search/cdx",
        params={
            "url": "normandyoc.com/collect/height.aspx",
            "output": "txt", "fl": "timestamp", "filter": "statuscode:200",
            "from": CAPTURE_FROM, "to": CAPTURE_TO,
            "collapse": "timestamp:6",  # 1か月に1キャプチャ
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.text.split()


def fetch_capture(ts: str) -> str:
    path = CACHE_DIR / f"{ts}.html"
    if path.exists() and path.stat().st_size > 0:
        return path.read_text(encoding="utf-8", errors="ignore")
    resp = requests.get(f"https://web.archive.org/web/{ts}id_/{TARGET_URL}", headers=HEADERS, timeout=60)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    path.write_text(resp.text, encoding="utf-8")
    time.sleep(3)
    return resp.text


def parse_capture(html: str, ts: str) -> list:
    table = BeautifulSoup(html, "html.parser").find("table", class_="body")
    if table is None:
        return []
    records = []
    for tr in table.find_all("tr")[1:]:
        cells = [c.get_text(strip=True) for c in tr.find_all("td")]
        if len(cells) < 11:
            continue
        name = unicodedata.normalize("NFKC", cells[1])
        m = re.match(r"^(.+?)の(\d{2})$", name)
        if not m:
            continue
        records.append({
            "bosyu_year": 2000 + int(m.group(2)) + 1,
            "bosyu_name": name,
            "mother_name": m.group(1),
            "sex": cells[4],
            "height": _to_float(cells[7]),
            "chest": _to_float(cells[8]),
            "cannon": _to_float(cells[9]),
            "weight": _to_float(cells[10]),
            "capture_ts": ts,
        })
    return records


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def match_to_jisseki(scale: pd.DataFrame) -> pd.DataFrame:
    other = pd.read_csv(DATA_DIR / "jisseki_other_all.csv", encoding="utf-8-sig")
    target = other[(other["club_name"] == "normandy") & other["bosyu_year"].isin(TRAIN_YEARS)].copy()
    mother = pd.read_csv(DATA_DIR / "mother_other_cache.csv", encoding="utf-8-sig")[["horse_id", "mother_name"]]
    target = pd.merge(target, mother, on="horse_id", how="left")
    target["mother_name"] = target["mother_name"].map(
        lambda x: unicodedata.normalize("NFKC", x) if isinstance(x, str) else x)

    merged = pd.merge(
        target[["horse_id", "horse_name", "bosyu_year", "mother_name"]],
        scale[["bosyu_year", "mother_name", "height", "chest", "cannon", "weight"]],
        on=["bosyu_year", "mother_name"], how="inner",
    ).drop_duplicates(subset=["horse_id"])

    summary = pd.DataFrame({
        "total": target.groupby("bosyu_year").size(),
        "matched": merged.groupby("bosyu_year").size(),
    }).fillna(0).astype(int)
    print("\n募集年度別マッチ状況:")
    print(summary.to_string())
    print(f"合計: {len(merged)} / {len(target)} 頭")
    return merged


def main():
    print("=== ノルマンディー 測尺データ取得（Wayback Machine） ===\n")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for ts in list_captures():
        rows = parse_capture(fetch_capture(ts), ts)
        print(f"  {ts}: {len(rows)} 頭")
        records.extend(rows)

    raw = pd.DataFrame(records)
    # 計測前で測尺欄が空のキャプチャがある（例: 2018/10時点の2017年産）ので除外してから、
    # 同じ馬が複数キャプチャに現れる場合は最も早い（募集時に近い）値を採用
    raw = raw.dropna(subset=["height", "chest", "cannon", "weight"], how="all")
    raw = raw.sort_values("capture_ts").drop_duplicates(subset=["bosyu_year", "mother_name"], keep="first")
    raw.to_csv(RAW_FILE, index=False, encoding="utf-8-sig")
    print(f"\n保存: {RAW_FILE}（{len(raw)} 頭）")

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
