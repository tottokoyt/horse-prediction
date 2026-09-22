"""
merge_carrot_scale.py
carrot_scale_raw.csv（Wayback Machineから取得した募集時測尺、母馬名+産年の仮名）を
jisseki_other_all.csv の horse_id と突き合わせ、data/other_scale_cache.csv に
（他クラブ共通フォーマットで）追記する。

マッチング方法:
  1. carrot_scale_raw.csv の bosyu_name（例:「レインデートの2017」）から
     正規表現で末尾の数字（産年）を除いた母馬名を取り出す
  2. jisseki_other_all.csv（club_name=carrot・該当bosyu_year）に
     mother_other_cache.csv の母馬名をマージ
  3. 同じ bosyu_year 内で「母馬名」が完全一致するものを対応付ける
     （同一年度に同じ母馬から複数の募集馬が出ることは基本的に無い前提）

使い方:
  python merge_carrot_scale.py
"""

import re
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")


def extract_mother_from_bosyu_name(bosyu_name: str):
    m = re.match(r"^(.+?)の\d+$", str(bosyu_name))
    return m.group(1) if m else None


def main():
    print("=== キャロット測尺データ マッチング ===\n")

    scale = pd.read_csv(DATA_DIR / "carrot_scale_raw.csv", encoding="utf-8-sig")
    scale["mother_name"] = scale["bosyu_name"].apply(extract_mother_from_bosyu_name)
    scale = scale.dropna(subset=["mother_name"])
    print(f"測尺データ: {len(scale)} 頭（母馬名抽出済み）")

    other = pd.read_csv(DATA_DIR / "jisseki_other_all.csv", encoding="utf-8-sig")
    other_carrot = other[
        (other["club_name"] == "carrot") & (other["bosyu_year"].isin(scale["bosyu_year"].unique()))
    ].copy()

    mother_cache_path = DATA_DIR / "mother_other_cache.csv"
    if not mother_cache_path.exists():
        raise FileNotFoundError("data/mother_other_cache.csv が無い。先に fetch_mother_other.py を実行してください")
    mother = pd.read_csv(mother_cache_path, encoding="utf-8-sig")[["horse_id", "mother_name"]]

    other_carrot = pd.merge(other_carrot, mother, on="horse_id", how="left")
    print(f"他クラブ側（carrot・対象年度）: {len(other_carrot)} 頭（母馬名取得済み: {other_carrot['mother_name'].notna().sum()}）")

    merged = pd.merge(
        other_carrot[["horse_id", "horse_name", "bosyu_year", "mother_name"]],
        scale[["bosyu_year", "mother_name", "height", "chest", "cannon", "weight", "farm"]],
        on=["bosyu_year", "mother_name"],
        how="inner",
    )
    merged = merged.drop_duplicates(subset=["horse_id"])

    print(f"\nマッチ成功: {len(merged)} / {len(scale)} 頭（測尺データ側）")
    print(f"マッチ率（jisseki_other_all側 carrot対象年度 {len(other_carrot)}頭に対して）: {len(merged)/len(other_carrot)*100:.1f}%")

    out_path = DATA_DIR / "other_scale_cache.csv"
    out_cols = ["horse_id", "horse_name", "bosyu_year", "height", "chest", "cannon", "weight", "farm"]
    if out_path.exists():
        existing = pd.read_csv(out_path, encoding="utf-8-sig")
        combined = pd.concat([existing, merged[out_cols]], ignore_index=True).drop_duplicates(subset=["horse_id"])
    else:
        combined = merged[out_cols]
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"保存: {out_path}（合計 {len(combined)} 頭）")


if __name__ == "__main__":
    main()
