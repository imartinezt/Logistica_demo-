from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import time
import uvicorn
from datetime import datetime

from config.settings import settings
from controllers.fee_controller import router as fee_router
from utils.logger import setup_logging, logger

# Setup logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title=f"🎯 {settings.APP_NAME}",
    version=settings.VERSION,
    description="Sistema inteligente de predicción FEE con LightGBM + Gemini 2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include router
app.include_router(fee_router)

# Root endpoint
@app.get("/")
async def root():
    """🏠 Liverpool FEE Predictor"""
    return {
        "sistema": "Liverpool FEE Predictor",
        "version": settings.VERSION,
        "motor": "LightGBM + Gemini 2.0",
        "estado": "✅ Operativo",
        "timestamp": datetime.now(),
        "endpoint": "/api/v1/fee/predict"
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info(
        "🚀 Liverpool FEE Predictor iniciado",
        version=settings.VERSION,
        motor="LightGBM + Gemini 2.0"
    )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )