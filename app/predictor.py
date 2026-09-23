"""
predictor.py - 予測ロジック（モデル読み込み・特徴量生成・ファクター判定）
"""

from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import shap

# ── 定数（ここを変更してロジック調整） ────────────────────────
GLOBAL_MEAN          = 80.0   # 全体平均回収率のデフォルト値（%）
GLOBAL_OVER200       = 0.18   # 全体の200%超率のデフォルト値

TARGET_CLASS  = 2      # SHAPで説明する対象クラス（2 = 200%超）
SHAP_EPSILON  = 0.02   # モデルA/B（分類確率0〜1スケール）用: |SHAP合計| がこれ未満ならneutral扱い
# モデルC（huber回帰、2026-09-23よりlog1p(kakutoku_man)スケール）用。
# A/Bとは出力スケールが全く異なるため別定数にした。実際の学習済みモデルで
# 60頭サンプルのSHAP値を計測したところ、trainer/farm/sire/bms_nameの
# |SHAP|中央値は0.2〜0.7、10パーセンタイルでも0.05〜0.2程度だったため、
# それより十分小さい0.05を「ほぼ寄与なし」の閾値とした。
SHAP_EPSILON_C = 0.05

# 測尺（cannon/weight）は表示用のタグ付けのみに使用（判定自体はSHAP由来）
CANNON_GOOD  = 21.0   # この値以上 → 「太め・骨量あり」タグ
CANNON_POOR  = 19.5   # この値以下 → 「細め・骨量懸念」タグ
WEIGHT_LARGE = 460    # この値以上 → 「大型馬」タグ
WEIGHT_SMALL = 430    # この値以下 → 「小型・軽量」タグ

# 判定しきい値
VERDICT_PROB_POSITIVE = 0.35  # prob_class2 >= → 有望候補
VERDICT_PROB_CONSIDER = 0.20  # prob_class2 >= → 検討候補
VERDICT_PROB_NEGATIVE = 0.70  # pred_class==0 かつ prob_class0 >= → 見送り推奨

SEX_MAP     = {"牡": 0, "牝": 1, "セ": 2}
LABEL_NAMES = {0: "100%未満", 1: "100〜200%", 2: "200%超"}

MODEL_DIR = Path(__file__).parent.parent / "models"

# ── モデル管理 ─────────────────────────────────────────────────
_models: dict = {}
_explainers: dict = {}


def load_models():
    for key, fname in [("A", "lgbm_model_A_v9.pkl"), ("B", "lgbm_model_B_v7.pkl"),
                        ("C", "lgbm_model_kakutoku.pkl")]:
        path = MODEL_DIR / fname
        if path.exists():
            saved = joblib.load(path)
            _models[key] = saved
            # モデルC（kakutoku）は huber回帰 + lambdarank のランクアンサンブル。
            # SHAPによるファクター説明は解釈しやすいhuber_modelを使う
            explain_target = saved["huber_model"] if key == "C" else saved["model"]
            _explainers[key] = shap.TreeExplainer(explain_target)
            print(f"[predictor] モデル{key} 読み込み完了: {path}")
        else:
            print(f"[predictor] 警告: モデル{key} が見つかりません: {path}")


def _get_model(use_scale: bool):
    key = "A" if use_scale else "B"
    if key not in _models:
        raise RuntimeError(f"モデル{key}が読み込まれていません")
    return key, _models[key]


def _shap_contributions(model_key: str, X: pd.DataFrame) -> dict:
    """
    1件分の特徴量行について、SHAP寄与度を特徴量名 -> 値 の辞書で返す。
    分類モデル（A/B）は TARGET_CLASS（200%超）確率への寄与、
    回帰モデル（C, kakutoku_man）は予測値そのものへの寄与。

    shapライブラリはバージョンによって多クラス分類のshap_values()の
    返り値の形が異なる（古い版: クラスごとの配列のlist、新しい版:
    (n_samples, n_features, n_classes) の3次元配列）ため、両方に対応する。
    """
    explainer = _explainers[model_key]
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        # 旧shap: [class0の(n_samples, n_features), class1の(...), ...]
        row = np.asarray(shap_values[TARGET_CLASS])[0]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            # 新shap: (n_samples, n_features, n_classes)
            row = arr[0, :, TARGET_CLASS]
        else:
            # 回帰・2値分類: (n_samples, n_features)
            row = arr[0]
    return dict(zip(X.columns, row))


# 集計特徴量（sire/trainer/farm/bms_name/nick/club_name）は
# {col}_smooth_mean / {col}_smooth_over200 / {col}_count の3特徴量に
# 分かれているため、SHAP寄与度は合算して「そのファクター全体の寄与」とする
def _grouped_shap(col: str, shap_map: dict) -> float:
    return sum(
        shap_map.get(f"{col}_{suffix}", 0.0)
        for suffix in ("smooth_mean", "smooth_over200", "count")
    )


# ── 集計特徴量の取得 ───────────────────────────────────────────
def _lookup(value, stats_df, key_col, fallback_medians=None):
    """
    stats_df から 1件の smooth_mean / smooth_over200 / count を返す。

    値が学習データに無い場合（未知の父馬・調教師等）のデフォルト値は、
    学習時に実際使われたfillna値（fallback_medians、モデルpklに保存済み）
    を優先的に使う。無い場合はstats_dfから中央値を近似計算し、それも
    無ければ最終手段としてGLOBAL_MEAN定数を使う。

    【背景】以前はハードコードされた GLOBAL_MEAN=80.0 を常に使っていたが、
    実際の学習時中央値（sire/trainer/farm/bms_nameいずれも96〜98%程度）と
    大きくズレており、未知の値を含む予測（特に学習データの薄い
    小規模クラブ）が系統的に歪んでいた（2026-09-23未明、夜間のクラブ
    拡張検証中に発見）。

    モデルC（kakutoku）の集計テーブルには smooth_over200 列が無いため、
    その場合はデフォルト値で補う。
    """
    mean_col = f"{key_col}_smooth_mean"
    over200_col = f"{key_col}_smooth_over200"
    fallback_medians = fallback_medians or {}
    if mean_col in fallback_medians:
        default_mean = fallback_medians[mean_col]
    elif stats_df is not None and mean_col in stats_df.columns:
        default_mean = float(stats_df[mean_col].median())
    else:
        default_mean = GLOBAL_MEAN
    if over200_col in fallback_medians:
        default_over200 = fallback_medians[over200_col]
    elif stats_df is not None and over200_col in stats_df.columns:
        default_over200 = float(stats_df[over200_col].median())
    else:
        default_over200 = GLOBAL_OVER200
    if stats_df is None or not value:
        return default_mean, default_over200, 0
    row = stats_df[stats_df[key_col].astype(str) == str(value)]
    if len(row) == 0:
        return default_mean, default_over200, 0
    return (
        float(row[f"{key_col}_smooth_mean"].values[0]),
        float(row[over200_col].values[0]) if over200_col in row.columns else default_over200,
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
        m, o, c = _lookup(val, aggs.get(col), col, saved.get("fallback_medians"))
        row[f"{col}_smooth_mean"]    = m
        row[f"{col}_smooth_over200"] = o
        row[f"{col}_count"]          = c

    df = pd.DataFrame([row])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = -1
    return df[feature_cols].fillna(-1)


# ── ファクター生成 ─────────────────────────────────────────────
def _impact_from_shap(shap_value: float, epsilon: float = SHAP_EPSILON) -> str:
    if shap_value >= epsilon:
        return "positive"
    if shap_value <= -epsilon:
        return "negative"
    return "neutral"


def build_factors(req, saved: dict, shap_map: dict) -> list:
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
        m, o, c = _lookup(val, aggs.get(col), col, saved.get("fallback_medians"))
        shap_val = _grouped_shap(col, shap_map)
        if c == 0:
            factors.append({
                "name":   label,
                "value":  "データなし（学習データ未登録）",
                "impact": _impact_from_shap(shap_val),
                "shap":   round(shap_val, 4),
            })
        else:
            factors.append({
                "name":   label,
                "value":  f"平均回収率 {m:.0f}%・200%超率 {o*100:.0f}%（{c}頭実績）",
                "impact": _impact_from_shap(shap_val),
                "shap":   round(shap_val, 4),
            })

    # 測尺（判定はSHAP寄与度に基づき、cm/kgのタグは参考表示のみ）
    if req.cannon is not None:
        tag = "太め・骨量あり" if req.cannon >= CANNON_GOOD else \
              "細め・骨量懸念" if req.cannon <= CANNON_POOR else "標準"
        shap_val = shap_map.get("cannon", 0.0)
        factors.append({
            "name":   "管囲",
            "value":  f"{req.cannon}cm（{tag}）",
            "impact": _impact_from_shap(shap_val),
            "shap":   round(shap_val, 4),
        })

    if req.weight is not None:
        tag = "大型馬" if req.weight >= WEIGHT_LARGE else \
              "小型・軽量" if req.weight <= WEIGHT_SMALL else "標準"
        shap_val = shap_map.get("weight", 0.0)
        factors.append({
            "name":   "体重",
            "value":  f"{req.weight}kg（{tag}）",
            "impact": _impact_from_shap(shap_val),
            "shap":   round(shap_val, 4),
        })

    # 生月
    if req.birth_month:
        shap_val = shap_map.get("birth_month", 0.0)
        factors.append({
            "name":   "生月",
            "value":  f"{req.birth_month}月生まれ",
            "impact": _impact_from_shap(shap_val),
            "shap":   round(shap_val, 4),
        })

    # 寄与度の大きい順に並べ替え（|SHAP|降順）
    factors.sort(key=lambda f: abs(f["shap"]), reverse=True)
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


def get_general_verdict(pred_kaishuu_rate: Optional[float]) -> Optional[str]:
    if pred_kaishuu_rate is None:
        return None
    if pred_kaishuu_rate >= 200:
        return "有望候補"
    if pred_kaishuu_rate >= 100:
        return "検討候補"
    return "慎重に検討"


# ── モデルC（獲得賞金・クラブ非依存汎用モデル）のファクター ──────
def build_general_factors(req, saved: dict, shap_map: dict) -> list:
    """
    モデルCは全クラブ共通の sire/trainer/farm/bms_name/生月/募集価格のみを
    使う（測尺・配合ニックは使わない）ため、モデルA/Bとは別の一覧を作る。
    """
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
        val = _resolve_value(req, col)
        m, o, c = _lookup(val, aggs.get(col), col, saved.get("fallback_medians"))
        shap_val = _grouped_shap(col, shap_map)
        if c == 0:
            factors.append({
                "name":   label,
                "value":  "データなし（学習データ未登録）",
                "impact": _impact_from_shap(shap_val, SHAP_EPSILON_C),
                "shap":   round(shap_val, 4),
            })
        else:
            factors.append({
                "name":   label,
                "value":  f"平均回収率 {m:.0f}%・200%超率 {o*100:.0f}%（{c}頭実績・全クラブ横断）",
                "impact": _impact_from_shap(shap_val, SHAP_EPSILON_C),
                "shap":   round(shap_val, 4),
            })

    if req.birth_month:
        shap_val = shap_map.get("birth_month", 0.0)
        factors.append({
            "name":   "生月",
            "value":  f"{req.birth_month}月生まれ",
            "impact": _impact_from_shap(shap_val, SHAP_EPSILON_C),
            "shap":   round(shap_val, 4),
        })

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


def run_general_predict(req) -> Optional[dict]:
    """
    モデルC: 獲得賞金を予測し（huber回帰+lambdarankのランクアンサンブル）、
    募集金額（price）が入力されていれば回収率換算値も返す。クラブを問わず
    学習しているため、シルク以外の未知のクラブ（例: DMMバヌーシー）の馬にも
    同じロジックで使える。
    """
    if "C" not in _models:
        return None

    saved = _models["C"]
    X = build_feature_row(req, saved)
    pred_kakutoku_man = _ensemble_kakutoku(saved, X)
    pred_kaishuu_rate = (
        pred_kakutoku_man / req.price * 100 if req.price else None
    )
    shap_map = _shap_contributions("C", X)

    return {
        "pred_kakutoku_man": round(pred_kakutoku_man, 1),
        "pred_kaishuu_rate": round(pred_kaishuu_rate, 1) if pred_kaishuu_rate is not None else None,
        "verdict":           get_general_verdict(pred_kaishuu_rate),
        "factors":           build_general_factors(req, saved, shap_map),
    }


# ── メイン予測 ─────────────────────────────────────────────────
def run_predict(req) -> dict:
    use_scale = all(
        v is not None for v in [req.height, req.chest, req.cannon, req.weight]
    )
    model_key, saved = _get_model(use_scale)
    X          = build_feature_row(req, saved)
    proba      = saved["model"].predict_proba(X)[0].tolist()
    pred_class = int(np.argmax(proba))
    shap_map   = _shap_contributions(model_key, X)

    return {
        "model_used": model_key,
        "pred_class": pred_class,
        "prob":       proba,
        "verdict":    get_verdict(pred_class, proba),
        "factors":    build_factors(req, saved, shap_map),
        "general":    run_general_predict(req),
    }
