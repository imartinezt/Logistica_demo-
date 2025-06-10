# app.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import uvicorn
from datetime import datetime

from config.settings import settings
from controllers.fee_controller import router as fee_router
from controllers.insights_controller import router as insights_router
from utils.logger import setup_logging, logger

# Setup logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title=f"🎯 {settings.APP_NAME}",
    version=settings.VERSION,
    description="""
    # 🚀 Sistema Inteligente Liverpool FEE

    ## 🧠 **Motor Híbrido: LangGraph + Gemini 2.0 Flash**

    ### **Arquitectura:**
    - 🔄 **LangGraph**: Orquesta flujo de razonamiento explicable
    - 🧠 **Gemini 2.0**: Toma decisiones inteligentes en cada nodo
    - 📊 **Datos Reales**: Usa TODOS los campos CSV dinámicamente
    - 🕒 **Auto-Detection**: Detecta automáticamente Navidad, Buen Fin, clima, tráfico

    ### **Flujo de Razonamiento:**

    ```
    INPUT → LangGraph Workflow → Gemini Decisions → OUTPUT
    ```

    **Pasos del LangGraph:**
    1. 🔍 **Gemini valida producto** → Detecta riesgos y características
    2. 🏠 **Gemini analiza zona** → Determina seguridad y flota requerida
    3. 🌤️ **Auto-detecta factores + Gemini evalúa** → Criticidad de Navidad, lluvia, etc.
    4. 📦 **Verifica stock OH** → Ubicaciones con inventario disponible
    5. 📍 **Calcula distancias reales** → Nodos más cercanos con coordenadas
    6. 🚚 **Evalúa rutas factibles** → Filtra por zona roja y capacidad
    7. 🎯 **Gemini optimiza selección** → Decisión inteligente multiobjetivo
    8. 📊 **Gemini genera análisis final** → Recomendaciones y confianza

    ### **Características Avanzadas:**
    - ✅ **Explicabilidad Total**: Cada decisión con reasoning de Gemini
    - ✅ **Detección Automática Eventos**: Navidad 2024 → factor_demanda 2.8x
    - ✅ **Zona Roja Inteligente**: Gemini evalúa seguridad por contexto
    - ✅ **Factores Temporales**: Clima y tráfico por fecha/hora
    - ✅ **Endpoints Insights**: Para frontend y dashboards
    - ✅ **Cálculos Geográficos**: Distancias reales con coordenadas

    ### **Ejemplo Navidad 2024:**
    ```json
    {
        "codigo_postal": "06700",
        "sku_id": "LIV001", 
        "cantidad": 2,
        "fecha_compra": "2024-12-24T10:00:00"
    }
    ```

    **→ Gemini detecta:** Nochebuena + factor 3.0x + zona segura + flota interna + FEE optimizada
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "🎯 Liverpool FEE System",
            "description": "Predicción inteligente con LangGraph + Gemini"
        },
        {
            "name": "📊 Insights & Analytics",
            "description": "Endpoints para dashboards y análisis"
        }
    ]
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Engine"] = "LangGraph+Gemini"
    return response


# Include routers
app.include_router(fee_router)
app.include_router(insights_router)


# Root endpoint
@app.get("/", tags=["🏠 Health"])
async def root():
    """🏠 Sistema Liverpool FEE - LangGraph + Gemini 2.0"""
    return {
        "🎯": "Liverpool FEE Predictor",
        "version": settings.VERSION,
        "🧠": "LangGraph + Gemini 2.0 Flash",
        "estado": "✅ Operativo",
        "timestamp": datetime.now(),
        "🚀": "Predicción inteligente FEE",
        "características": [
            "🔄 LangGraph workflow explicable",
            "🧠 Gemini 2.0 reasoning en cada paso",
            "🕒 Auto-detección Navidad/Buen Fin",
            "📍 Cálculos geográficos reales",
            "🚨 Detección inteligente zonas rojas",
            "📊 Insights para frontend",
            "⚡ Respuesta < 3 segundos"
        ],
        "endpoints": {
            "predicción": "/api/v1/fee/predict",
            "insights": "/api/v1/insights/*",
            "datos": "/api/v1/data/*",
            "análisis": "/api/v1/analysis/*"
        }
    }


@app.get("/stats", tags=["📊 System Stats"])
async def system_stats():
    """📊 Estadísticas del sistema"""
    try:
        from controllers.fee_controller import get_repositories
        repos = get_repositories()

        return {
            "sistema": "Liverpool FEE Stats",
            "datos_cargados": {
                "productos": len(repos['product'].load_data()),
                "nodos": len(repos['node'].load_data()),
                "inventarios": len(repos['inventory'].load_data()),
                "rutas": len(repos['route'].load_data()),
                "codigos_postales": len(repos['postal_code'].load_data()),
                "factores_externos": len(repos['external_factors'].load_data())
            },
            "motor": {
                "langgraph": "✅ Activo",
                "gemini_model": settings.MODEL_NAME,
                "project_id": settings.PROJECT_ID
            },
            "configuracion": {
                "peso_tiempo": settings.PESO_TIEMPO,
                "peso_costo": settings.PESO_COSTO,
                "peso_probabilidad": settings.PESO_PROBABILIDAD,
                "horario_corte_flash": f"{settings.HORARIO_CORTE_FLASH}:00"
            }
        }
    except Exception as e:
        return {"error": f"Error obteniendo stats: {str(e)}"}


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Liverpool FEE Predictor iniciado",
                version=settings.VERSION,
                engine="LangGraph + Gemini 2.0 Flash",
                data_dir=str(settings.DATA_DIR),
                model=settings.MODEL_NAME)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("💥 Error global no manejado",
                 error=str(exc),
                 path=request.url.path,
                 method=request.method)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Error interno del servidor",
            "path": request.url.path,
            "engine": "LangGraph + Gemini",
            "support": "Revisar logs del sistema"
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )