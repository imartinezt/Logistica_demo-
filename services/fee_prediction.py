# services/fee_prediction.py
from typing import Dict
from datetime import datetime, timedelta

from models.schemas import (
    PredictionRequest, PredictionResponse, TipoEntregaEnum,
    ExplicabilidadCompleta, FactoresExternosDetectados,
    AnalisisStock, AnalisisRuta
)
from services.ai.langgraph_engine import LangGraphReasoningEngine, ReasoningState
from config.settings import settings
from utils.logger import logger


class FEEPredictionService:
    """
    🎯 SERVICIO PRINCIPAL FEE

    Orquesta LangGraph + Gemini para predicciones inteligentes
    """

    def __init__(self, repositories: Dict):
        self.repos = repositories
        self.reasoning_engine = LangGraphReasoningEngine(repositories)

    async def predict_fee(self, request: PredictionRequest) -> PredictionResponse:
        """🚀 Predicción principal usando LangGraph + Gemini"""
        try:
            logger.info("🎯 Iniciando predicción FEE con LangGraph + Gemini",
                        codigo_postal=request.codigo_postal,
                        sku_id=request.sku_id)

            # 🧠 Ejecutar razonamiento LangGraph + Gemini
            state = await self.reasoning_engine.reason(request)

            # 🏗️ Construir respuesta final
            response = self._build_final_response(state)

            logger.info("✅ Predicción FEE completada",
                        fee=response.fecha_entrega_estimada,
                        carrier=response.carrier_asignado,
                        confianza=response.explicabilidad.flujo_decision[-1].score)

            return response

        except Exception as e:
            logger.error("💥 Error en predicción FEE", error=str(e))
            raise

    def _build_final_response(self, state: ReasoningState) -> PredictionResponse:
        """🏗️ Construye respuesta final desde estado LangGraph"""
        mejor_ruta = state['mejor_ruta']
        request = state['request']
        gemini_decisions = state['gemini_decisions']

        # 📅 Determinar tipo de entrega
        tipo_entrega = self._determine_delivery_type(
            mejor_ruta['tiempo_ajustado'],
            request.fecha_compra.hour
        )

        # 🗓️ Calcular fecha de entrega
        fecha_entrega = self._calculate_delivery_date(
            request.fecha_compra,
            tipo_entrega,
            mejor_ruta['tiempo_ajustado']
        )

        # 🚚 Determinar carrier
        carrier = self._determine_carrier(mejor_ruta['tipo_flota'])

        # 📊 Construir explicabilidad completa
        explicabilidad = self._build_explicabilidad_completa(state)

        return PredictionResponse(
            fecha_entrega_estimada=fecha_entrega,
            codigo_postal=request.codigo_postal,
            tipo_entrega=tipo_entrega,
            costo_envio_mxn=float(mejor_ruta['costo_total_mxn']),
            es_flota_externa=mejor_ruta['tipo_flota'] in ['FE', 'FI_FE'],
            carrier_asignado=carrier,
            tiempo_estimado_horas=float(mejor_ruta['tiempo_ajustado']),
            probabilidad_cumplimiento=float(mejor_ruta['probabilidad_cumplimiento']),
            explicabilidad=explicabilidad
        )

    def _determine_delivery_type(self, tiempo_horas: float, hora_compra: int) -> TipoEntregaEnum:
        """📦 Determina tipo de entrega basado en tiempo y hora"""
        if tiempo_horas <= 24 and hora_compra <= settings.HORARIO_CORTE_FLASH:
            return TipoEntregaEnum.FLASH
        elif tiempo_horas <= 48:
            return TipoEntregaEnum.EXPRESS
        elif tiempo_horas <= 72:
            return TipoEntregaEnum.STANDARD
        else:
            return TipoEntregaEnum.PROGRAMADA

    def _calculate_delivery_date(self, fecha_compra: datetime, tipo_entrega: TipoEntregaEnum,
                                 tiempo_ajustado: float) -> datetime:
        """🗓️ Calcula fecha de entrega evitando fines de semana"""
        # Usar tiempo ajustado real en lugar de estándar
        fecha_entrega = fecha_compra + timedelta(hours=tiempo_ajustado)

        # Ajustar por días hábiles
        while fecha_entrega.weekday() >= 5:  # Evitar fines de semana
            fecha_entrega += timedelta(days=1)

        # Ajustar hora a horario laboral (9-18)
        if fecha_entrega.hour < 9:
            fecha_entrega = fecha_entrega.replace(hour=9)
        elif fecha_entrega.hour > 18:
            fecha_entrega = fecha_entrega.replace(hour=18)

        return fecha_entrega

    def _determine_carrier(self, tipo_flota: str) -> str:
        """🚚 Determina carrier basado en tipo de flota"""
        if tipo_flota in ['FE', 'FI_FE']:
            return "DHL"
        else:
            return "Liverpool"

    def _build_explicabilidad_completa(self, state: ReasoningState) -> ExplicabilidadCompleta:
        """📊 Construye explicabilidad completa con decisiones de Gemini"""
        gemini_decisions = state['gemini_decisions']

        # Análisis de stock
        stock_analisis = AnalisisStock(
            ubicaciones_disponibles=len(state['stock_disponible']),
            stock_total=sum(s['stock_oh'] for s in state['stock_disponible']),
            ubicacion_optima=str(state['mejor_ruta']['nodo_stock']),
            stock_ubicacion_optima=next(
                s['stock_oh'] for s in state['stock_disponible']
                if s['nodo_id'] == state['mejor_ruta']['nodo_stock']
            ),
            cobertura_demanda_dias=30.0,  # Placeholder - calcular real
            necesita_reabastecimiento=False
        )

        # Análisis de ruta
        ruta_analisis = AnalisisRuta(
            ruta_id=state['mejor_ruta']['ruta_id'],
            eslabones_secuencia=state['mejor_ruta'].get('eslabones_secuencia', ''),
            nodo_origen=state['mejor_ruta']['nodo_origen'],
            tiempo_base_horas=state['mejor_ruta']['tiempo_total_horas'],
            tiempo_ajustado_horas=state['mejor_ruta']['tiempo_ajustado'],
            costo_total_mxn=state['mejor_ruta']['costo_total_mxn'],
            probabilidad_cumplimiento=state['mejor_ruta']['probabilidad_cumplimiento'],
            tipo_flota=state['mejor_ruta']['tipo_flota'],
            carrier=self._determine_carrier(state['mejor_ruta']['tipo_flota']),
            es_factible=True,
            razon_seleccion=gemini_decisions.get('optimizacion', {}).get('razon_seleccion', 'Optimizada por Gemini')
        )

        # Factores externos detectados
        factores_externos = FactoresExternosDetectados(
            eventos_detectados=state['factores_externos'].get('eventos_detectados', []),
            factor_demanda=state['factores_externos'].get('factor_demanda', 1.0),
            condicion_clima=state['factores_externos'].get('condicion_clima', 'Templado'),
            probabilidad_lluvia=state['factores_externos'].get('probabilidad_lluvia', 30),
            temperatura=state['factores_externos'].get('temperatura', 22),
            trafico_nivel=state['factores_externos'].get('trafico_nivel', 'Moderado'),
            es_temporada_alta=state['factores_externos'].get('es_temporada_alta', False),
            impacto_tiempo_extra=state['factores_externos'].get('impacto_tiempo_extra', 0)
        )

        return ExplicabilidadCompleta(
            flujo_decision=state['explicabilidad'],
            producto_info=state['producto'],
            zona_info=state['zona_info'],
            stock_analisis=stock_analisis,
            factores_externos_detectados=factores_externos,
            rutas_evaluadas=state['rutas_evaluadas'][:5],  # Top 5
            ruta_seleccionada=ruta_analisis,
            tiempo_breakdown={
                'tiempo_base': state['mejor_ruta']['tiempo_total_horas'],
                'factor_demanda': state['factores_externos'].get('factor_demanda', 1.0),
                'tiempo_extra': state['factores_externos'].get('impacto_tiempo_extra_horas', 0),
                'tiempo_final': state['mejor_ruta']['tiempo_ajustado']
            },
            costo_breakdown={
                'costo_base': state['mejor_ruta']['costo_total_mxn'],
                'costo_extra': 0,
                'costo_total': state['mejor_ruta']['costo_total_mxn']
            },
            warnings=state['warnings'],
            tiempo_total_procesamiento_ms=(state['tiempo_inicio'] and
                                           (state['pasos_completados'] * 100))  # Estimación
        )