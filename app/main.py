"""
main.py - FastAPI バックエンド
起動: uvicorn app.main:app --reload
"""

from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import sys

# app/ ディレクトリを import パスに追加
sys.path.insert(0, str(Path(__file__).parent))
import predictor

app = FastAPI(title="競走馬回収率予測API", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"


# ── リクエスト/レスポンス スキーマ ────────────────────────────
class PredictRequest(BaseModel):
    sire:        Optional[str]   = None
    trainer:     Optional[str]   = None
    farm:        Optional[str]   = None
    sex:         Optional[str]   = None
    birth_month: Optional[int]   = None
    price:       Optional[float] = None
    height:      Optional[float] = None
    chest:       Optional[float] = None
    cannon:      Optional[float] = None
    weight:      Optional[float] = None


# ── 起動時にモデルをロード ─────────────────────────────────────
@app.on_event("startup")
async def startup():
    predictor.load_models()


# ── エンドポイント ─────────────────────────────────────────────
@app.get("/")
async def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html が見つかりません")
    return FileResponse(html_path)


@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        result = predictor.run_predict(req)
        return JSONResponse(content=result)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"予測エラー: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}
