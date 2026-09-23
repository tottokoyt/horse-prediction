"""
predictor.py - 予測ロジック（モデル読み込み・特徴量生成・ファクター判定）

使うのはモデルC（train_model_kakutoku.py、クラブ横断の獲得賞金予測）のみ。
シルク特化のモデルA（測尺あり分類）・モデルB（測尺なし分類）は2026-09-23に廃止した
（PLAN_measurement_expansion.md）。モデルCが全クラブを学習に使い、測尺も
「あれば使う」特徴量として取り込んだため、A/Bを別に持つ意味が無くなった。
"""

from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import shap

# ── 定数（ここを変更してロジック調整） ────────────────────────
GLOBAL_MEAN = 80.0   # 未知値フォールバックの最終手段（通常は fallback_medians を使う）

# huber回帰（log1p(kakutoku_man)スケール）のSHAP値がこれ未満ならneutral扱い。
# 実際の学習済みモデルで60頭サンプルのSHAP値を計測したところ、
# trainer/farm/sire/bms_nameの|SHAP|中央値は0.2〜0.7、10パーセンタイルでも
# 0.05〜0.2程度だったため、それより十分小さい0.05を「ほぼ寄与なし」の閾値とした。
SHAP_EPSILON = 0.05

SEX_MAP = {"牡": 0, "牝": 1, "セ": 2}
SCALE_FIELDS = [("height", "体高", "cm"), ("chest", "胸囲", "cm"),
                ("cannon", "管囲", "cm"), ("weight", "体重", "kg")]

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_FILE = "lgbm_model_kakutoku.pkl"

# ── モデル管理 ─────────────────────────────────────────────────
# validate_banushi.py / validate_new_club.py が _models["C"] を直接参照している
_models: dict = {}
_explainers: dict = {}


def load_models():
    path = MODEL_DIR / MODEL_FILE
    if not path.exists():
        print(f"[predictor] 警告: モデルが見つかりません: {path}")
        return
    saved = joblib.load(path)
    _models["C"] = saved
    # huber回帰 + lambdarank のランクアンサンブル。
    # SHAPによるファクター説明は解釈しやすいhuber_modelを使う
    _explainers["C"] = shap.TreeExplainer(saved["huber_model"])
    print(f"[predictor] モデル読み込み完了: {path}")


def _shap_contributions(X: pd.DataFrame) -> dict:
    """1件分の特徴量行について、huber予測値へのSHAP寄与度を特徴量名 -> 値 の辞書で返す"""
    row = np.asarray(_explainers["C"].shap_values(X))[0]
    return dict(zip(X.columns, row))


# 集計特徴量（sire/trainer/farm/bms_name）は {col}_smooth_mean / {col}_count の
# 2特徴量に分かれているため、SHAP寄与度は合算して「そのファクター全体の寄与」とする
def _grouped_shap(col: str, shap_map: dict) -> float:
    return sum(shap_map.get(f"{col}_{suffix}", 0.0) for suffix in ("smooth_mean", "count"))


# ── 集計特徴量の取得 ───────────────────────────────────────────
def _lookup(value, stats_df, key_col, fallback_medians=None):
    """
    stats_df から 1件の smooth_mean / count を返す。

    値が学習データに無い場合（未知の父馬・調教師等）のデフォルト値は、
    学習時に実際使われたfillna値（fallback_medians、モデルpklに保存済み）
    を優先的に使う。無い場合はstats_dfから中央値を近似計算し、それも
    無ければ最終手段としてGLOBAL_MEAN定数を使う。

    【背景】以前はハードコードされた GLOBAL_MEAN=80.0 を常に使っていたが、
    実際の学習時中央値（sire/trainer/farm/bms_nameいずれも96〜98%程度）と
    大きくズレており、未知の値を含む予測（特に学習データの薄い
    小規模クラブ）が系統的に歪んでいた（2026-09-23未明、夜間のクラブ
    拡張検証中に発見）。
    """
    mean_col = f"{key_col}_smooth_mean"
    fallback_medians = fallback_medians or {}
    if mean_col in fallback_medians:
        default_mean = fallback_medians[mean_col]
    elif stats_df is not None and mean_col in stats_df.columns:
        default_mean = float(stats_df[mean_col].median())
    else:
        default_mean = GLOBAL_MEAN
    if stats_df is None or not value:
        return default_mean, 0
    row = stats_df[stats_df[key_col].astype(str) == str(value)]
    if len(row) == 0:
        return default_mean, 0
    return float(row[mean_col].values[0]), int(row[f"{key_col}_count"].values[0])


# 母父の集計キーは bms_name・リクエスト属性名は bms
_ARG_NAME_MAP = {"bms_name": "bms"}


def _resolve_value(req, col: str):
    return getattr(req, _ARG_NAME_MAP.get(col, col), None)


# ── 特徴量 DataFrame 生成 ──────────────────────────────────────
def build_feature_row(req, saved: dict) -> pd.DataFrame:
    aggs         = saved["aggs"]
    feature_cols = saved["feature_cols"]

    row = {
        "sex_num":     SEX_MAP.get(req.sex or "", -1),
        "birth_month": req.birth_month if req.birth_month else -1,
        "price_man":   req.price if req.price else -1,
    }

    # 測尺は1項目ずつ扱う（学習時も一部項目だけの馬がいる。例: unionは体重なし）。
    # 未入力の項目は下の fillna(-1) で学習時と同じ欠損扱いになる
    for col, _, _ in SCALE_FIELDS:
        v = getattr(req, col, None)
        if v is not None:
            row[col] = v

    for col in aggs.keys():
        m, c = _lookup(_resolve_value(req, col), aggs.get(col), col, saved.get("fallback_medians"))
        row[f"{col}_smooth_mean"] = m
        row[f"{col}_count"]       = c

    df = pd.DataFrame([row])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = -1
    return df[feature_cols].fillna(-1)


# ── 判定・ファクター ───────────────────────────────────────────
def _impact_from_shap(shap_value: float) -> str:
    if shap_value >= SHAP_EPSILON:
        return "positive"
    if shap_value <= -SHAP_EPSILON:
        return "negative"
    return "neutral"


def get_verdict(pred_kaishuu_rate: Optional[float]) -> Optional[str]:
    if pred_kaishuu_rate is None:
        return None
    if pred_kaishuu_rate >= 200:
        return "有望候補"
    if pred_kaishuu_rate >= 100:
        return "検討候補"
    return "慎重に検討"


def build_factors(req, saved: dict, shap_map: dict) -> list:
    aggs    = saved["aggs"]
    factors = []

    col_labels = [
        ("trainer",  f"調教師（{req.trainer or '不明'}）"),
        ("farm",     f"牧場（{req.farm or '不明'}）"),
        ("sire",     f"父馬（{req.sire or '不明'}）"),
        ("bms_name", f"母父（{req.bms or '不明'}）"),
    ]
    for col, label in col_labels:
        if col not in aggs:
            continue
        m, c = _lookup(_resolve_value(req, col), aggs.get(col), col, saved.get("fallback_medians"))
        shap_val = _grouped_shap(col, shap_map)
        value = ("データなし（学習データ未登録）" if c == 0
                 else f"平均回収率 {m:.0f}%（{c}頭実績・全クラブ横断）")
        factors.append({"name": label, "value": value,
                        "impact": _impact_from_shap(shap_val), "shap": round(shap_val, 4)})

    if req.birth_month:
        shap_val = shap_map.get("birth_month", 0.0)
        factors.append({"name": "生月", "value": f"{req.birth_month}月生まれ",
                        "impact": _impact_from_shap(shap_val), "shap": round(shap_val, 4)})

    # 測尺（学習済みモデルが測尺を使っている場合のみ。古いpklとの互換のため確認する）
    for col, label, unit in SCALE_FIELDS:
        v = getattr(req, col, None)
        if v is None or col not in saved["feature_cols"]:
            continue
        shap_val = shap_map.get(col, 0.0)
        factors.append({"name": label, "value": f"{v}{unit}",
                        "impact": _impact_from_shap(shap_val), "shap": round(shap_val, 4)})

    # 寄与度の大きい順に並べ替え（|SHAP|降順）
    factors.sort(key=lambda f: abs(f["shap"]), reverse=True)
    return factors


def _percentile_of(value: float, reference: np.ndarray) -> float:
    return float((reference < value).mean())


def _ensemble_kakutoku(saved: dict, X: pd.DataFrame) -> float:
    """
    huber回帰の予測（log1p(万円)空間、2026-09-23に生の万円から変更。
    kakutoku_manは極端な右裾分布のため生の値のままだと少数の超高額馬に
    予測が支配されてほぼ一定値になってしまうバグがあった）とlambdarankの
    予測（学習データ内での相対スコアのみで単位を持たない）を、学習プール
    全体でのパーセンタイル順位に変換してから平均し、huber分布の同
    パーセンタイル値に逆変換したのち、expm1で万円単位の実スケールに戻す
    （train_model_kakutoku.pyのdocstring参照）。
    """
    huber_pred = float(saved["huber_model"].predict(X)[0])
    rank_pred  = float(saved["rank_model"].predict(X)[0])
    huber_pct = _percentile_of(huber_pred, saved["ref_huber_preds"])
    rank_pct  = _percentile_of(rank_pred, saved["ref_rank_scores"])
    ensemble_pct = (huber_pct + rank_pct) / 2
    log_result = np.quantile(saved["ref_huber_preds"], ensemble_pct)
    return float(np.expm1(log_result))


# ── メイン予測 ─────────────────────────────────────────────────
def run_predict(req) -> dict:
    """
    獲得賞金を予測し（huber回帰+lambdarankのランクアンサンブル）、
    募集金額（price）が入力されていれば回収率換算値も返す。クラブを問わず
    学習しているため、未知のクラブ（例: DMMバヌーシー）の馬にも同じロジックで使える。
    """
    if "C" not in _models:
        raise RuntimeError("モデルが読み込まれていません")

    saved = _models["C"]
    X = build_feature_row(req, saved)
    pred_kakutoku_man = _ensemble_kakutoku(saved, X)
    pred_kaishuu_rate = pred_kakutoku_man / req.price * 100 if req.price else None
    shap_map = _shap_contributions(X)

    return {
        "pred_kakutoku_man": round(pred_kakutoku_man, 1),
        "pred_kaishuu_rate": round(pred_kaishuu_rate, 1) if pred_kaishuu_rate is not None else None,
        "verdict":           get_verdict(pred_kaishuu_rate),
        "factors":           build_factors(req, saved, shap_map),
    }
