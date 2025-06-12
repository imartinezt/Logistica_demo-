import time

from fastapi import APIRouter, HTTPException, Depends, status

from config.settings import settings
from models.schemas import PredictionRequest, PredictionResponse
from services.data.repositories import OptimizedRepositories  # ✅ CORREGIDO
from services.fee_prediction_service import FEEPredictionService  # ✅ CORREGIDO
from utils.logger import logger

router = APIRouter(prefix="/api/v1", tags=["🎯 Liverpool FEE System"])
_fee_service: FEEPredictionService = None


def get_fee_service() -> FEEPredictionService:
    """🔧 Dependency injection para FEE service optimizado"""
    global _fee_service
    if _fee_service is None:
        # ✅ CORREGIDO: Inicializar repositorios optimizados primero
        logger.info("🚀 Inicializando repositorios optimizados...")
        repositories = OptimizedRepositories(settings.DATA_DIR)

        logger.info("🎯 Inicializando servicio FEE optimizado...")
        _fee_service = FEEPredictionService(repositories)

        logger.info("✅ Servicio FEE optimizado listo!")
    return _fee_service


@router.post("/fee/predict", response_model=PredictionResponse)
async def predict_delivery_fee(
        request: PredictionRequest,
        service: FEEPredictionService = Depends(get_fee_service)
):
    """
    🎯 PREDICCIÓN INTELIGENTE FEE - MOTOR HÍBRIDO LightGBM + Gemini
    """

    start_time = time.time()

    try:
        logger.info(
            "🚀 Iniciando predicción FEE",
            codigo_postal=request.codigo_postal,
            sku_id=request.sku_id,
            cantidad=request.cantidad
        )

        resultado = await service.predict_fee(request)
        processing_time = (time.time() - start_time) * 1000

        logger.info(
            "✅ Predicción completada",
            processing_time_ms=processing_time,
            fecha_entrega=resultado.fecha_entrega_estimada.isoformat(),
            tipo_entrega=resultado.tipo_entrega.value,
            costo=resultado.costo_envio_mxn,
            probabilidad=resultado.probabilidad_cumplimiento
        )

        return resultado

    except ValueError as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error("❌ Error de validación", error=str(e), processing_time_ms=processing_time)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error("💥 Error interno", error=str(e), processing_time_ms=processing_time)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {str(e)}"
        )