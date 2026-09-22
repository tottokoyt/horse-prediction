"""
collect_jisseki_banushi.py
一口馬主DB (umadb.com) から DMMバヌーシー（本来のターゲットクラブ）の
実績データを収集する。

このプロジェクトの本来の目的はDMMバヌーシーの出資検討支援であり
（README「このプロジェクトの本来の目的」参照）、モデルC（クラブ横断
汎用モデル）の性能をLOCO（疑似検証）だけでなく実データで答え合わせ
できるよう、バヌーシー自身のデータを取得する。

umadbクラブコード: c123 (バヌーシー自体は新しいクラブのため、
collect_jisseki_other.py の10クラブとは別に単独で収集する)

使い方:
  python collect_jisseki_banushi.py
"""

import pandas as pd
from pathlib import Path

from collect_jisseki_other import collect_club_year, YEAR_MAP

OUTPUT_DIR = Path("data")
CLUB_NAME = "banushi"
CLUB_CODE = "c123"


def main():
    print(f"=== DMMバヌーシー ({CLUB_CODE}) 実績データ収集 ===\n")
    dfs = []
    for birth_year in sorted(YEAR_MAP.keys()):
        try:
            df = collect_club_year(CLUB_NAME, CLUB_CODE, birth_year)
            dfs.append(df)
            print(f"  {birth_year}年産: {len(df)} 頭")
        except Exception as e:
            print(f"  [ERROR] {birth_year}年産: {e}")

    df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    out_path = OUTPUT_DIR / "jisseki_banushi.csv"
    df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n=== 完了: 合計 {len(df_all)} 頭 ===")
    print(f"出力: {out_path}")
    if len(df_all) > 0:
        print(df_all.groupby("bosyu_year").size())


if __name__ == "__main__":
    main()
