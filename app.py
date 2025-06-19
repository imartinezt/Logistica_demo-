import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from controllers.fee_controller import router as fee_router
from utils.logger import setup_logging, logger

setup_logging()

@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "🚀Autor: Iván Martínez Trejo",
        extra={"version": settings.VERSION, "motor": "LightGBM + Gemini 2.0"},
    )
    yield
application = FastAPI(
    title=f"🎯 {settings.APP_NAME}",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

application.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@application.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time() - start:.4f}"
    return response

application.include_router(fee_router)

if __name__ == "__main__":
    uvicorn.run(
        "app:application",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
