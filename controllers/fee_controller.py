import time

from fastapi import APIRouter, HTTPException, Depends, status

from config.settings import settings
from models.schemas import PredictionRequest, PredictionResponse
from services.fee_prediction_service import FEEPredictionService
from utils.logger import logger

router = APIRouter(prefix="/api/v1", tags=["🎯 Liverpool FEE System"])

# Global service instance
_fee_service: FEEPredictionService = None


def get_fee_service() -> FEEPredictionService:
    """🔧 Dependency injection para FEE service"""
    global _fee_service
    if _fee_service is None:
        _fee_service = FEEPredictionService(settings.DATA_DIR)
    return _fee_service


@router.post("/fee/predict", response_model=PredictionResponse)
async def predict_delivery_fee(
        request: PredictionRequest,
        service: FEEPredictionService = Depends(get_fee_service)
):
    """
    🎯 **PREDICCIÓN INTELIGENTE FEE - MOTOR HÍBRIDO LightGBM + Gemini**

    **Arquitectura:**
    - 🤖 **LightGBM**: Optimización multiobjetivo de rutas candidatas
    - 🧠 **Gemini 2.0**: Decisión final inteligente con explicabilidad
    - 📊 **Polars**: Procesamiento ultra-rápido de datos
    - 🌍 **PyProj**: Cálculos geoespaciales precisos México

    **Casos Resueltos:**
    - ✅ Split de inventario entre múltiples ubicaciones
    - ✅ Rutas híbridas flota interna + externa  
    - ✅ Auto-detección eventos: Navidad, Buen Fin, clima
    - ✅ Zonas rojas con restricciones automáticas
    - ✅ Optimización tiempo ↔ costo ↔ confiabilidad
    """

    start_time = time.time()

    try:
        logger.info(
            "🚀 Iniciando predicción FEE",
            codigo_postal=request.codigo_postal,
            sku_id=request.sku_id,
            cantidad=request.cantidad
        )

        # 🎯 Ejecutar predicción con arquitectura híbrida
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
        logger.error("❌ Error de validación", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("💥 Error interno", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {str(e)}"
        )