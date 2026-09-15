from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes.prediction_routes import router


app = FastAPI(
    title="CNN-Based Certificate Forgery Detection API",
    description="Backend for detecting forged academic certificates using a CNN with residual connections.",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount(
    "/results",
    StaticFiles(
        directory=r"D:\certificate-forgery-backend(2)\results"
    ),
    name="results"
)


app.include_router(router)


@app.get("/")
def home():
    return {
        "success": True,
        "message": "Certificate Forgery Detection API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }