"""FastAPI app: web UI at / and JSON API at /api/predict.

    uvicorn app:app --reload          # development
    docker compose up --build         # production-style

The API also accepts requests from a separately hosted frontend (for example on
Vercel). Allowed origins come from the ALLOWED_ORIGINS environment variable
(comma separated), or from the default list below.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import config
from src.predict import Predictor

logger = logging.getLogger("next-word")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Websites allowed to call this API from a browser. Replace the Vercel address with
# your real one after the first deploy, or set ALLOWED_ORIGINS on the server instead.
DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000,https://YOUR-PROJECT.vercel.app"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()]


class PredictRequest(BaseModel):
    text: str = Field("", max_length=config.MAX_INPUT_CHARS, description="What the user has typed so far")
    k: int = Field(config.TOP_K, ge=1, le=10, description="How many suggestions to return")


class Suggestion(BaseModel):
    word: str
    probability: float


class PredictResponse(BaseModel):
    suggestions: list[Suggestion]
    mode: str = Field(description="'next' = suggest next word, 'complete' = finish the current word")
    latency_ms: float


def create_app(model_path: Path | str | None = None, vocab_path: Path | str | None = None) -> FastAPI:
    state: dict = {"predictor": None, "error": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            state["predictor"] = Predictor.load(model_path, vocab_path)
            state["predictor"].predict_next("warm up ", 1)  # build the graph once, before real traffic
            logger.info("Model loaded")
        except Exception as exc:  # keep the process alive so /health can explain the problem
            state["error"] = str(exc)
            logger.error("Model failed to load: %s", exc)
        yield

    app = FastAPI(title="Next-Word Prediction", version="1.0.0", lifespan=lifespan)

    # Lets the separately hosted frontend (Vercel) call this API from the browser
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    base = Path(__file__).resolve().parent
    app.mount("/static", StaticFiles(directory=base / "static"), name="static")
    templates = Jinja2Templates(directory=base / "templates")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index(request: Request):
        predictor = state["predictor"]
        info = {
            "model": (Path(model_path).stem if model_path else config.PRODUCTION_MODEL).upper(),
            "vocab_size": len(predictor.vocab) if predictor else 0,
            "ready": predictor is not None,
            "max_chars": config.MAX_INPUT_CHARS,
        }
        return templates.TemplateResponse(request, "index.html", {"info": info})

    @app.get("/health")
    def health():
        if state["predictor"] is None:
            return {"status": "degraded", "detail": state["error"]}
        return {"status": "ok", "vocab_size": len(state["predictor"].vocab)}

    @app.post("/api/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        predictor = state["predictor"]
        if predictor is None:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {state['error']}")
        started = time.perf_counter()
        suggestions = predictor.predict_next(req.text, req.k)
        completing = bool(req.text) and not req.text[-1].isspace() and bool(req.text.split())
        return PredictResponse(
            suggestions=suggestions,
            mode="complete" if completing else "next",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    return app


app = create_app()