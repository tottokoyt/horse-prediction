"""
predictor.py - 予測ロジック（モデル読み込み・特徴量生成・ファクター判定）
"""

from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import pandas as pd

# ── 定数（ここを変更してロジック調整） ────────────────────────
GLOBAL_MEAN          = 80.0   # 全体平均回収率のデフォルト値（%）
GLOBAL_OVER200       = 0.18   # 全体の200%超率のデフォルト値
IMPACT_POS_THRESHOLD = 15.0   # smooth_mean が GLOBAL_MEAN + この値以上 → positive
IMPACT_NEG_THRESHOLD = -15.0  # smooth_mean が GLOBAL_MEAN + この値以下 → negative

# 管囲しきい値
CANNON_GOOD  = 21.0   # この値以上 → positive
CANNON_POOR  = 19.5   # この値以下 → negative

# 体重しきい値
WEIGHT_LARGE = 460    # この値以上 → positive
WEIGHT_SMALL = 430    # この値以下 → negative

# 判定しきい値
VERDICT_PROB_POSITIVE = 0.35  # prob_class2 >= → 有望候補
VERDICT_PROB_CONSIDER = 0.20  # prob_class2 >= → 検討候補
VERDICT_PROB_NEGATIVE = 0.70  # pred_class==0 かつ prob_class0 >= → 見送り推奨

SEX_MAP     = {"牡": 0, "牝": 1, "セ": 2}
LABEL_NAMES = {0: "100%未満", 1: "100〜200%", 2: "200%超"}

MODEL_DIR = Path(__file__).parent.parent / "models"

# ── モデル管理 ─────────────────────────────────────────────────
_models: dict = {}


def load_models():
    for key, fname in [("A", "lgbm_model_A_v9.pkl"), ("B", "lgbm_model_B.pkl")]:
        path = MODEL_DIR / fname
        if path.exists():
            _models[key] = joblib.load(path)
            print(f"[predictor] モデル{key} 読み込み完了: {path}")
        else:
            print(f"[predictor] 警告: モデル{key} が見つかりません: {path}")


def _get_model(use_scale: bool):
    key = "A" if use_scale else "B"
    if key not in _models:
        raise RuntimeError(f"モデル{key}が読み込まれていません")
    return key, _models[key]


# ── 集計特徴量の取得 ───────────────────────────────────────────
def _lookup(value, stats_df, key_col):
    """stats_df から 1件の smooth_mean / smooth_over200 / count を返す"""
    if stats_df is None or not value:
        return GLOBAL_MEAN, GLOBAL_OVER200, 0
    row = stats_df[stats_df[key_col].astype(str) == str(value)]
    if len(row) == 0:
        return GLOBAL_MEAN, GLOBAL_OVER200, 0
    return (
        float(row[f"{key_col}_smooth_mean"].values[0]),
        float(row[f"{key_col}_smooth_over200"].values[0]),
        int(row[f"{key_col}_count"].values[0]),
    )


# ── リクエスト属性 → 集計キー値の解決 ─────────────────────────
# bms_name（母父）はモデルAのみ・リクエスト属性名は bms
# nick（配合ニック）はリクエストに直接の属性がなく、sire×bms から組み立てる
_ARG_NAME_MAP = {"bms_name": "bms"}


def _resolve_value(req, col: str):
    if col == "nick":
        sire = getattr(req, "sire", None)
        bms  = getattr(req, "bms", None)
        if sire and bms:
            return f"{sire}×{bms}"
        return None
    arg_name = _ARG_NAME_MAP.get(col, col)
    return getattr(req, arg_name, None)


# ── 特徴量 DataFrame 生成 ──────────────────────────────────────
def build_feature_row(req, saved: dict) -> pd.DataFrame:
    aggs         = saved["aggs"]
    feature_cols = saved["feature_cols"]

    row = {
        "sex_num":     SEX_MAP.get(req.sex or "", -1),
        "birth_month": req.birth_month if req.birth_month else -1,
        "price_man":   req.price if req.price else -1,
    }

    if req.height is not None:
        row.update({
            "height": req.height,
            "chest":  req.chest,
            "cannon": req.cannon,
            "weight": req.weight,
        })

    for col in aggs.keys():
        val = _resolve_value(req, col)
        m, o, c = _lookup(val, aggs.get(col), col)
        row[f"{col}_smooth_mean"]    = m
        row[f"{col}_smooth_over200"] = o
        row[f"{col}_count"]          = c

    df = pd.DataFrame([row])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = -1
    return df[feature_cols].fillna(-1)


# ── ファクター生成 ─────────────────────────────────────────────
def _impact(smooth_mean: float) -> str:
    diff = smooth_mean - GLOBAL_MEAN
    if diff >= IMPACT_POS_THRESHOLD:
        return "positive"
    if diff <= IMPACT_NEG_THRESHOLD:
        return "negative"
    return "neutral"


def build_factors(req, saved: dict) -> list:
    aggs    = saved["aggs"]
    factors = []

    # 厩舎 / 牧場 / 父馬 / 母父 / 配合ニック（父×母父）
    nick_label = f"配合ニック（{req.sire or '不明'}×{req.bms or '不明'}）"
    col_labels = [
        ("trainer",  f"調教師（{req.trainer or '不明'}）"),
        ("farm",     f"牧場（{req.farm or '不明'}）"),
        ("sire",     f"父馬（{req.sire or '不明'}）"),
        ("bms_name", f"母父（{req.bms or '不明'}）"),
        ("nick",     nick_label),
    ]
    for col, label in col_labels:
        if col not in aggs:
            continue
        val = _resolve_value(req, col)
        m, o, c = _lookup(val, aggs.get(col), col)
        if c == 0:
            factors.append({
                "name":   label,
                "value":  "データなし（学習データ未登録）",
                "impact": "neutral",
            })
        else:
            factors.append({
                "name":   label,
                "value":  f"平均回収率 {m:.0f}%・200%超率 {o*100:.0f}%（{c}頭実績）",
                "impact": _impact(m),
            })

    # 測尺
    if req.cannon is not None:
        if req.cannon >= CANNON_GOOD:
            imp = "positive"
            tag = "太め・骨量あり"
        elif req.cannon <= CANNON_POOR:
            imp = "negative"
            tag = "細め・骨量懸念"
        else:
            imp = "neutral"
            tag = "標準"
        factors.append({
            "name":   "管囲",
            "value":  f"{req.cannon}cm（{tag}）",
            "impact": imp,
        })

    if req.weight is not None:
        if req.weight >= WEIGHT_LARGE:
            imp = "positive"
            tag = "大型馬"
        elif req.weight <= WEIGHT_SMALL:
            imp = "negative"
            tag = "小型・軽量"
        else:
            imp = "neutral"
            tag = "標準"
        factors.append({
            "name":   "体重",
            "value":  f"{req.weight}kg（{tag}）",
            "impact": imp,
        })

    # 生月
    if req.birth_month:
        if req.birth_month <= 2:
            factors.append({"name": "生月", "value": f"{req.birth_month}月生まれ（早生まれ有利）",   "impact": "positive"})
        elif req.birth_month >= 5:
            factors.append({"name": "生月", "value": f"{req.birth_month}月生まれ（遅生まれ注意）", "impact": "negative"})
        else:
            factors.append({"name": "生月", "value": f"{req.birth_month}月生まれ",                "impact": "neutral"})

    return factors


# ── 判定文字列 ─────────────────────────────────────────────────
def get_verdict(pred_class: int, proba: list) -> str:
    p2 = proba[2]
    if p2 >= VERDICT_PROB_POSITIVE:
        return "有望候補"
    if p2 >= VERDICT_PROB_CONSIDER:
        return "検討候補"
    if pred_class == 0 and proba[0] >= VERDICT_PROB_NEGATIVE:
        return "見送り推奨"
    return "中程度"


# ── メイン予測 ─────────────────────────────────────────────────
def run_predict(req) -> dict:
    use_scale = all(
        v is not None for v in [req.height, req.chest, req.cannon, req.weight]
    )
    model_key, saved = _get_model(use_scale)
    proba      = saved["model"].predict_proba(build_feature_row(req, saved))[0].tolist()
    pred_class = int(np.argmax(proba))

    return {
        "model_used": model_key,
        "pred_class": pred_class,
        "prob":       proba,
        "verdict":    get_verdict(pred_class, proba),
        "factors":    build_factors(req, saved),
    }
