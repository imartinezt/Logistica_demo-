# controllers/fee_controller.py
from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict

from models.schemas import PredictionRequest, PredictionResponse
from services.fee_prediction import FEEPredictionService
from services.data.repositories import (
    ProductRepository, NodeRepository, InventoryRepository,
    RouteRepository, PostalCodeRepository, ExternalFactorsRepository
)
from config.settings import settings
from utils.logger import logger

router = APIRouter(prefix="/api/v1", tags=["🎯 Liverpool FEE System"])


def get_repositories() -> Dict:
    """🔧 Dependency injection para repositorios"""
    return {
        'product': ProductRepository(settings.DATA_DIR),
        'node': NodeRepository(settings.DATA_DIR),
        'inventory': InventoryRepository(settings.DATA_DIR),
        'route': RouteRepository(settings.DATA_DIR),
        'postal_code': PostalCodeRepository(settings.DATA_DIR),
        'external_factors': ExternalFactorsRepository(settings.DATA_DIR)
    }


def get_fee_service(repos=Depends(get_repositories)) -> FEEPredictionService:
    """🎯 Dependency injection para FEE service"""
    return FEEPredictionService(repos)


@router.post("/fee/predict", response_model=PredictionResponse)
async def predict_delivery_fee(
        request: PredictionRequest,
        service: FEEPredictionService = Depends(get_fee_service)
):
    """
    🎯 **PREDICCIÓN INTELIGENTE FEE - LANGGRAPH + GEMINI**

    **Motor Híbrido:**
    - 🔄 **LangGraph**: Orquesta el flujo de razonamiento
    - 🧠 **Gemini 2.0 Flash**: Toma decisiones inteligentes en cada paso
    - 📊 **Datos Reales**: Usa TODOS los CSV dinámicamente
    - 🕒 **Auto-Detection**: Detecta Navidad, Buen Fin, etc. por fecha

    **Flujo LangGraph:**
    1. Gemini valida producto y detecta riesgos
    2. Gemini analiza zona y determina seguridad
    3. Auto-detecta factores + Gemini evalúa criticidad
    4. Verifica stock OH disponible
    5. Calcula distancias reales a nodos
    6. Evalúa rutas factibles
    7. Gemini optimiza selección final
    8. Gemini genera análisis completo

    **Salida:** FEE + explicabilidad completa paso a paso
    """
    try:
        logger.info("🚀 Iniciando predicción FEE LangGraph+Gemini",
                    codigo_postal=request.codigo_postal,
                    sku_id=request.sku_id,
                    cantidad=request.cantidad)

        # 🧠 Ejecutar predicción con LangGraph + Gemini
        resultado = await service.predict_fee(request)

        logger.info("✅ Predicción FEE completada",
                    fee=resultado.fecha_entrega_estimada,
                    tipo=resultado.tipo_entrega,
                    carrier=resultado.carrier_asignado,
                    costo=resultado.costo_envio_mxn,
                    confianza=resultado.explicabilidad.flujo_decision[-1].score)

        return resultado

    except ValueError as e:
        logger.error("❌ Error de validación", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("💥 Error interno predicción", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {str(e)}"
        )


@router.get("/fee/test-gemini")
async def test_gemini_connection():
    """🧠 Test de conexión con Gemini"""
    try:
        from services.ai.gemini_service import GeminiReasoningService

        gemini_service = GeminiReasoningService()

        # Test simple
        test_producto = {
            "sku_id": "TEST001",
            "nombre_producto": "Producto Test",
            "categoria": "Test",
            "peso_kg": 1.0,
            "es_fragil": False
        }

        test_request = {
            "codigo_postal": "06700",
            "cantidad": 1
        }

        result = await gemini_service.validate_product_decision(test_producto, test_request)

        return {
            "status": "✅ Gemini conectado",
            "model": settings.MODEL_NAME,
            "test_result": result
        }

    except Exception as e:
        logger.error("❌ Error conexión Gemini", error=str(e))
        raise HTTPException(500, f"Error conectando Gemini: {str(e)}")


@router.get("/fee/health")
async def health_check():
    """🏥 Health check del sistema completo"""
    try:
        repos = get_repositories()

        # Test básico de cada repositorio
        health_status = {
            "sistema": "Liverpool FEE Predictor",
            "version": settings.VERSION,
            "status": "✅ Healthy",
            "components": {
                "productos": len(repos['product'].load_data()),
                "nodos": len(repos['node'].load_data()),
                "inventarios": len(repos['inventory'].load_data()),
                "rutas": len(repos['route'].load_data()),
                "codigos_postales": len(repos['postal_code'].load_data()),
                "factores_externos": len(repos['external_factors'].load_data())
            },
            "engine": "LangGraph + Gemini 2.0 Flash",
            "timestamp": "2024-12-09"
        }

        return health_status

    except Exception as e:
        raise HTTPException(500, f"Sistema no saludable: {str(e)}")


