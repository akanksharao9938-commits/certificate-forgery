from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes.prediction_routes import router


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(r"D:\certificate-forgery-backend(2)")

RESULTS_DIR = BASE_DIR / "results"
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI-Based Certificate Forgery Detection API",
    description=(
        "Backend for academic certificate authenticity analysis "
        "using ResNet50, YOLO11, OCR, LOF and pixel analysis."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# RESULTS FOLDER
# ============================================================

app.mount(
    "/results",
    StaticFiles(directory=str(RESULTS_DIR)),
    name="results"
)


# ============================================================
# API ROUTES
# ============================================================

app.include_router(router)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# FRONTEND
# ============================================================

if FRONTEND_DIST.exists():

    app.mount(
        "/",
        StaticFiles(
            directory=str(FRONTEND_DIST),
            html=True
        ),
        name="frontend"
    )