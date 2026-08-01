"""FastAPI service. Synchronous inference (2-5s/image is fine for a single-user demo;
under real load I'd move this behind a Celery/Redis queue -- omitted by design).

    uvicorn api.main:app --reload      # from the repo root
"""
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

from api.predict import predict_latex
from api.render import render_latex_png

app = FastAPI(title="scribe2tex", description="Handwritten & printed math -> LaTeX")


class PredictionResponse(BaseModel):
    latex: str
    confidence: float


class RenderRequest(BaseModel):
    latex: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="could not read image file")
    return predict_latex(image)


@app.post("/render")
def render(req: RenderRequest):
    try:
        png = render_latex_png(req.latex)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not render latex: {exc}")
    return Response(content=png, media_type="image/png")
