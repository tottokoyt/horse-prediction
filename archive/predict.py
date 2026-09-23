"""
predict.py
募集馬の回収率予測CLIツール

使い方:
  # 測尺ありの場合（モデルA）
  python predict.py \
    --sire "キタサンブラック" \
    --trainer "国枝栄" \
    --farm "ノーザンファーム" \
    --bms "シンボリクリスエス" \
    --sex 牡 \
    --birth_month 3 \
    --price 5000 \
    --height 157 \
    --chest 175 \
    --cannon 20.5 \
    --weight 460

  # 測尺なしの場合（モデルB）
  python predict.py \
    --sire "キタサンブラック" \
    --trainer "国枝栄" \
    --farm "ノーザンファーム" \
    --sex 牡 \
    --birth_month 3 \
    --price 5000
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("models")
DATA_DIR  = Path("data")


def load_model(use_scale: bool):
    key = "A_v7" if use_scale else "B"
    path = MODEL_DIR / f"lgbm_model_{key}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {path}")
    return joblib.load(path)


def apply_smooth_mean(value, stats_df, key_col, global_mean, global_over200):
    """1件分の集計特徴量を計算"""
    if stats_df is None or value is None:
        return global_mean, global_over200, 0
    row = stats_df[stats_df[key_col] == value]
    if len(row) == 0:
        return global_mean, global_over200, 0
    return (
        row[f"{key_col}_smooth_mean"].values[0],
        row[f"{key_col}_smooth_over200"].values[0],
        row[f"{key_col}_count"].values[0],
    )


def build_features(args, saved) -> pd.DataFrame:
    """引数から特徴量DataFrameを1行作る"""
    aggs         = saved["aggs"]
    feature_cols = saved["feature_cols"]

    # 全体平均・over200率（デフォルト）
    global_mean    = 80.0
    global_over200 = 0.18

    sex_map = {"牡": 0, "牝": 1, "セ": 2}
    sex_num = sex_map.get(args.sex, -1)

    row = {
        "sex_num":     sex_num,
        "birth_month": args.birth_month if args.birth_month else -1,
        "price_man":   args.price if args.price else -1,
    }

    # 測尺
    if args.height:
        row["height"] = args.height
        row["chest"]  = args.chest
        row["cannon"] = args.cannon
        row["weight"] = args.weight

    # 集計特徴量（bms_name はモデルAのみ・引数名は --bms）
    arg_name_map = {"bms_name": "bms"}
    for col in aggs.keys():
        arg_name = arg_name_map.get(col, col)
        val      = getattr(args, arg_name, None)
        stats_df = aggs.get(col)
        m, o, c  = apply_smooth_mean(val, stats_df, col, global_mean, global_over200)
        row[f"{col}_smooth_mean"]    = m
        row[f"{col}_smooth_over200"] = o
        row[f"{col}_count"]          = c

    df = pd.DataFrame([row])

    # feature_cols に合わせて列を揃える
    for col in feature_cols:
        if col not in df.columns:
            df[col] = -1

    df = df[feature_cols].fillna(-1)
    return df


def predict(args):
    use_scale = all([args.height, args.chest, args.cannon, args.weight])
    saved     = load_model(use_scale)
    model     = saved["model"]
    label_names = saved.get("label_names", {0: "100%未満", 1: "100〜200%", 2: "200%超"})

    X     = build_features(args, saved)
    proba = model.predict_proba(X)[0]  # shape: (3,)
    pred_class = int(np.argmax(proba))

    # 出力
    model_type = "A（測尺あり・精密版）" if use_scale else "B（測尺なし・汎用版）"
    print("\n" + "="*50)
    print(f"  使用モデル: モデル{model_type}")
    print("="*50)

    print(f"\n  入力情報:")
    print(f"    父馬     : {args.sire or '不明'}")
    print(f"    厩舎     : {args.trainer or '不明'}")
    print(f"    牧場     : {args.farm or '不明'}")
    if use_scale:
        print(f"    母父     : {args.bms or '不明'}")
    print(f"    性別     : {args.sex or '不明'}")
    print(f"    生月     : {args.birth_month or '不明'}月")
    print(f"    募集総額 : {args.price or '不明'}万円")
    if use_scale:
        print(f"    体高     : {args.height}cm")
        print(f"    胸囲     : {args.chest}cm")
        print(f"    管囲     : {args.cannon}cm")
        print(f"    体重     : {args.weight}kg")

    print(f"\n  予測結果:")
    print(f"    予測クラス: クラス{pred_class}（{label_names[pred_class]}）")

    print(f"\n  各クラスの確率:")
    for i, (prob, name) in enumerate(zip(proba, label_names.values())):
        bar    = "█" * int(prob * 30)
        marker = " ← 予測" if i == pred_class else ""
        print(f"    {name:12s}: {prob*100:5.1f}% {bar}{marker}")

    print(f"\n  200%超の確率: {proba[2]*100:.1f}%")

    print(f"\n  判定:")
    if proba[2] >= 0.35:
        print("    ✅ 有望候補（200%超の可能性が高い）")
    elif proba[2] >= 0.20:
        print("    🔶 検討候補（200%超の可能性あり）")
    elif pred_class == 0 and proba[0] >= 0.70:
        print("    ❌ 見送り推奨（元本割れリスクが高い）")
    else:
        print("    ➖ 中程度（慎重に検討）")

    print("="*50 + "\n")
    return proba


def main():
    parser = argparse.ArgumentParser(
        description="競走馬の回収率予測ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sire",        type=str,   help="父馬名")
    parser.add_argument("--trainer",     type=str,   help="預託予定厩舎")
    parser.add_argument("--farm",        type=str,   help="生産牧場")
    parser.add_argument("--bms",         type=str,   help="母父（モデルAのみ有効）")
    parser.add_argument("--sex",         type=str,   help="性別（牡/牝/セ）")
    parser.add_argument("--birth_month", type=int,   help="生まれ月（1〜5）")
    parser.add_argument("--price",       type=float, help="募集総額（万円）")
    parser.add_argument("--height",      type=float, help="体高（cm）")
    parser.add_argument("--chest",       type=float, help="胸囲（cm）")
    parser.add_argument("--cannon",      type=float, help="管囲（cm）")
    parser.add_argument("--weight",      type=float, help="体重（kg）")

    args = parser.parse_args()

    scale_inputs = [args.height, args.chest, args.cannon, args.weight]
    if any(scale_inputs) and not all(scale_inputs):
        parser.error("測尺は height/chest/cannon/weight の4つ全て入力してください")

    predict(args)


if __name__ == "__main__":
    main()
