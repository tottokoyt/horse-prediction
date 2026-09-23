"""
merge.py
募集データ (bosyu_all.csv) と実績データ (jisseki_all.csv) を結合するスクリプト

マッチングキー: horse_name (馬名) x year (募集年度)
出力:
  data/merged_train.csv  - 学習用 (2018〜2021年募集)
  data/merged_valid.csv  - 検証用 (2022〜2023年募集)
  data/merged_all.csv    - 全件

使い方:
  python merge.py

【注意】成績の更新のためにこのスクリプトを単独で再実行しないこと（2026-09-23）。
  merged_*.csv には、このあと fetch_bms.py（母父）と fetch_mother_name.py
  （母馬名による募集馬⇔競走馬の再マッチング、race_horse_name/mother_name列）が
  手を加えており、merge.py だけを再実行すると母父の列が消え、340頭中324頭の
  対応付けが不正確な「馬名×年度」の結合に戻ってしまう。成績だけ更新したい場合は、
  既存の merged_*.csv を残し、新しい jisseki_all.csv から horse_id で
  kakutoku_man / kaishuu_rate / races / wins / class を上書きすること。
"""

import re
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

TRAIN_YEARS = [2018, 2019, 2020, 2021]
VALID_YEARS = [2022, 2023]


def normalize_name(name: str) -> str:
    """
    馬名の表記ゆれを吸収する
    例: "エディスワートンの20" -> "エディスワートンの20"
    全角スペース除去・末尾の空白除去など
    """
    if not isinstance(name, str):
        return ""
    name = name.strip()
    name = name.replace("\u3000", "")  # 全角スペース
    name = re.sub(r"\s+", "", name)    # 半角スペース
    return name


def load_bosyu() -> pd.DataFrame:
    path = DATA_DIR / "bosyu_all.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["horse_name_norm"] = df["horse_name"].apply(normalize_name)
    df = df.rename(columns={"year": "bosyu_year"})
    print(f"募集データ: {len(df)} 頭")
    return df


def load_jisseki() -> pd.DataFrame:
    path = DATA_DIR / "jisseki_all.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["horse_name_norm"] = df["horse_name"].apply(normalize_name)
    print(f"実績データ: {len(df)} 頭")
    return df


def merge(df_bosyu: pd.DataFrame, df_jisseki: pd.DataFrame) -> pd.DataFrame:
    """
    馬名 + 募集年度でマージ
    マッチしなかった馬は実績NaNで保持（出走0や未デビューも含む）
    """
    df = pd.merge(
        df_bosyu,
        df_jisseki[[
            "horse_name_norm", "bosyu_year",
            "kakutoku_man", "kaishuu_rate", "races", "wins", "class",
            "farm", "horse_url", "horse_id"
        ]],
        on=["horse_name_norm", "bosyu_year"],
        how="left",
    )

    # 回収率が取れなかった馬は0として扱う（未デビュー・出走なしと同義）
    df["kaishuu_rate"] = df["kaishuu_rate"].fillna(0).astype(int)
    df["kakutoku_man"] = df["kakutoku_man"].fillna(0)
    df["races"] = df["races"].fillna(0).astype(int)
    df["wins"] = df["wins"].fillna(0).astype(int)

    # マッチ率を確認
    matched = df["horse_id"].notna().sum()
    print(f"マッチ: {matched}/{len(df)} 頭 ({matched/len(df)*100:.1f}%)")

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    予測に使いやすい特徴量を追加
    """
    # 回収率カテゴリ (ターゲット変数の参考用)
    df["kaishuu_category"] = pd.cut(
        df["kaishuu_rate"],
        bins=[-1, 0, 50, 100, 200, 99999],
        labels=["未出走", "50%未満", "50〜100%", "100〜200%", "200%超"],
    )

    # 一口価格から総額を推定（総額が取れていない場合の補完）
    if "price_per_kuchi" in df.columns and "total_price_man" in df.columns:
        df["total_price_man"] = df["total_price_man"].fillna(
            df["price_per_kuchi"] * 500 / 10000  # 500口想定・円→万円
        )

    return df


def main():
    df_bosyu   = load_bosyu()
    df_jisseki = load_jisseki()

    df_all = merge(df_bosyu, df_jisseki)
    df_all = add_features(df_all)

    # 学習用 / 検証用に分割
    df_train = df_all[df_all["bosyu_year"].isin(TRAIN_YEARS)].copy()
    df_valid = df_all[df_all["bosyu_year"].isin(VALID_YEARS)].copy()

    # 保存
    df_all.to_csv(DATA_DIR / "merged_all.csv",   index=False, encoding="utf-8-sig")
    df_train.to_csv(DATA_DIR / "merged_train.csv", index=False, encoding="utf-8-sig")
    df_valid.to_csv(DATA_DIR / "merged_valid.csv", index=False, encoding="utf-8-sig")

    print(f"\n--- 出力完了 ---")
    print(f"merged_all.csv   : {len(df_all)} 頭")
    print(f"merged_train.csv : {len(df_train)} 頭 (学習用: {TRAIN_YEARS})")
    print(f"merged_valid.csv : {len(df_valid)} 頭 (検証用: {VALID_YEARS})")

    # 簡易サマリ
    print("\n--- 回収率カテゴリ分布 (学習用) ---")
    print(df_train["kaishuu_category"].value_counts())


if __name__ == "__main__":
    main()
