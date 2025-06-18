import time
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends, status

from config.settings import settings
from models.schemas import PredictionRequest
from services.data.repositories import OptimizedRepositories
from services.fee_prediction_service import FEEPredictionService
from utils.logger import logger

router = APIRouter(prefix="/api/v1", tags=["🎯 Liverpool FEE System"])
_fee_service: FEEPredictionService = None


def get_fee_service() -> FEEPredictionService:
    """🔧 Dependency injection para FEE service optimizado"""
    global _fee_service
    if _fee_service is None:
        logger.info("🚀 Inicializando repositorios")
        repositories = OptimizedRepositories(settings.DATA_DIR)

        logger.info("🎯 Inicializando servicio FEE...")
        _fee_service = FEEPredictionService(repositories)

        logger.info("✅ Servicio FEE listo!")
    return _fee_service


@router.post("/fee/predict", response_model=Dict[str, Any])
async def predict_delivery_fee(
        request: PredictionRequest,
        service: FEEPredictionService = Depends(get_fee_service)
):
    """ PREDICCIÓN FEE"""

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
            fecha_entrega=resultado['resultado_final']['fecha_entrega_estimada'],
            tipo_entrega=resultado['resultado_final']['tipo_entrega'],
            costo=resultado['resultado_final']['costo_mxn'],
            probabilidad=resultado['resultado_final']['probabilidad_exito']
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