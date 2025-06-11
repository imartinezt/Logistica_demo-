from pydantic import BaseModel, Field, validator
from datetime import datetime, time
from typing import Optional, List, Dict, Any, Union
from enum import Enum



class TipoEntregaEnum(str, Enum):
    FLASH = "FLASH"
    EXPRESS = "EXPRESS"
    STANDARD = "STANDARD"
    PROGRAMADA = "PROGRAMADA"


class TipoFlotaEnum(str, Enum):
    FI = "FI"  # Flota Interna Liverpool
    FE = "FE"  # Flota Externa
    FI_FE = "FI_FE"  # Híbrida


class EstadoRutaEnum(str, Enum):
    FACTIBLE = "FACTIBLE"
    NO_FACTIBLE = "NO_FACTIBLE"
    CONDICIONAL = "CONDICIONAL"



class PredictionRequest(BaseModel):
    """📥 Request para predicción FEE"""
    codigo_postal: str = Field(..., min_length=5, max_length=5,
                               description="Código postal destino")
    sku_id: str = Field(..., description="ID del producto")
    cantidad: int = Field(..., ge=1, le=100,
                          description="Cantidad a entregar")
    fecha_compra: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="Fecha y hora exacta de compra"
    )

    @validator('codigo_postal')
    def validate_cp(cls, v):
        if not v.isdigit():
            raise ValueError('Código postal debe ser numérico')
        return v


class UbicacionStock(BaseModel):
    """📦 Stock en una ubicación específica"""
    ubicacion_id: str = Field(..., description="ID de tienda o CEDIS")
    ubicacion_tipo: str = Field(..., description="TIENDA | CEDIS")
    nombre_ubicacion: str = Field(..., description="Nombre legible")
    stock_disponible: int = Field(..., ge=0, description="Stock OH disponible")
    stock_reservado: int = Field(default=0, ge=0, description="Stock reservado")
    coordenadas: Dict[str, float] = Field(..., description="lat, lon")
    horario_operacion: str = Field(..., description="Horario de la ubicación")
    tiempo_preparacion_horas: float = Field(
        default=1.5, description="Tiempo picking/packing"
    )


class SplitInventory(BaseModel):
    """🔄 Split de inventario entre múltiples ubicaciones"""
    ubicaciones: List[UbicacionStock] = Field(
        ..., description="Ubicaciones con stock"
    )
    cantidad_total_requerida: int = Field(..., ge=1)
    cantidad_total_disponible: int = Field(..., ge=0)
    es_split_factible: bool = Field(..., description="Si el split es posible")
    razon_split: str = Field(..., description="Por qué se hace el split")


class Segmento(BaseModel):
    """🚚 Segmento individual de una ruta"""
    segmento_id: str = Field(..., description="ID único del segmento")
    origen_id: str = Field(..., description="ID ubicación origen")
    destino_id: str = Field(..., description="ID ubicación destino")
    origen_nombre: str = Field(..., description="Nombre origen")
    destino_nombre: str = Field(..., description="Nombre destino")
    distancia_km: float = Field(..., ge=0, description="Distancia real")
    tiempo_viaje_horas: float = Field(..., ge=0, description="Tiempo de viaje")
    tipo_flota: TipoFlotaEnum = Field(..., description="Tipo de flota")
    carrier: str = Field(..., description="Carrier responsable")
    costo_segmento_mxn: float = Field(..., ge=0, description="Costo del segmento")
    factores_aplicados: List[str] = Field(
        default_factory=list, description="Factores que afectaron el segmento"
    )


class RutaCompleta(BaseModel):
    """🗺️ Ruta completa multi-segmento"""
    ruta_id: str = Field(..., description="ID único de la ruta")
    segmentos: List[Segmento] = Field(..., description="Segmentos de la ruta")
    split_inventory: Optional[SplitInventory] = Field(
        None, description="Info de split si aplica"
    )

    # Métricas
    tiempo_total_horas: float = Field(..., ge=0, description="Tiempo total")
    costo_total_mxn: float = Field(..., ge=0, description="Costo total")
    distancia_total_km: float = Field(..., ge=0, description="Distancia total")

    # Scores
    score_tiempo: float = Field(..., ge=0, le=1, description="Score tiempo normalizado")
    score_costo: float = Field(..., ge=0, le=1, description="Score costo normalizado")
    score_confiabilidad: float = Field(..., ge=0, le=1, description="Score confiabilidad")
    score_lightgbm: Optional[float] = Field(None, description="Score LightGBM")

    # Estado y viabilidad
    estado: EstadoRutaEnum = Field(..., description="Estado de la ruta")
    probabilidad_cumplimiento: float = Field(..., ge=0, le=1)
    factores_riesgo: List[str] = Field(default_factory=list)



class FactoresExternos(BaseModel):
    """🌤️ Factores externos detectados y calculados"""
    fecha_analisis: datetime = Field(..., description="Fecha del análisis")

    # Eventos temporales
    eventos_detectados: List[str] = Field(default_factory=list)
    factor_demanda: float = Field(default=1.0, ge=0.5, le=5.0)
    es_temporada_alta: bool = Field(default=False)

    # Clima
    condicion_clima: str = Field(default="Templado")
    temperatura_celsius: int = Field(default=22, ge=-10, le=50)
    probabilidad_lluvia: int = Field(default=30, ge=0, le=100)
    viento_kmh: Optional[int] = Field(None, ge=0, le=200)

    # Tráfico y logística
    trafico_nivel: str = Field(default="Moderado")
    impacto_tiempo_extra_horas: float = Field(default=0.0, ge=0)
    impacto_costo_extra_pct: float = Field(default=0.0, ge=0)

    # Factores de zona
    zona_seguridad: str = Field(default="Media")
    restricciones_vehiculares: List[str] = Field(default_factory=list)

    # Criticidad general
    criticidad_logistica: str = Field(
        default="Normal",
        description="Baja|Normal|Media|Alta|Crítica"
    )


class CandidatoRuta(BaseModel):
    """🎯 Candidato generado por LightGBM"""
    ruta: RutaCompleta = Field(..., description="Ruta completa")
    score_lightgbm: float = Field(..., ge=0, le=1, description="Score ML")
    ranking_position: int = Field(..., ge=1, description="Posición en ranking")
    features_utilizadas: Dict[str, float] = Field(
        ..., description="Features que usó LightGBM"
    )
    trade_offs: Dict[str, str] = Field(
        default_factory=dict, description="Trade-offs identificados"
    )


class DecisionGemini(BaseModel):
    """🧠 Decisión final de Gemini"""
    candidato_seleccionado: CandidatoRuta = Field(..., description="Ganador")
    razonamiento: str = Field(..., description="Por qué Gemini lo eligió")
    candidatos_evaluados: List[CandidatoRuta] = Field(
        ..., description="Todos los candidatos evaluados"
    )
    factores_decisivos: List[str] = Field(..., description="Factores clave")
    confianza_decision: float = Field(..., ge=0, le=1)
    alertas_gemini: List[str] = Field(default_factory=list)



class FEECalculation(BaseModel):
    """🗓️ Cálculo final de FEE"""
    fecha_entrega_estimada: datetime = Field(..., description="FEE calculada")
    rango_horario_entrega: Dict[str, time] = Field(
        ..., description="{'inicio': time, 'fin': time}"
    )
    tipo_entrega: TipoEntregaEnum = Field(..., description="Tipo de entrega")
    tiempo_total_horas: float = Field(..., ge=0, description="Tiempo total real")

    # Desglose de tiempos
    tiempo_preparacion: float = Field(..., description="Tiempo picking/packing")
    tiempo_transito: float = Field(..., description="Tiempo en tránsito")
    tiempo_contingencia: float = Field(..., description="Tiempo buffer")


class ExplicabilidadCompleta(BaseModel):
    """📊 Explicabilidad completa del proceso"""


    request_procesado: PredictionRequest = Field(..., description="Request original")
    factores_externos: FactoresExternos = Field(..., description="Factores detectados")
    split_inventory: Optional[SplitInventory] = Field(None)

    candidatos_lightgbm: List[CandidatoRuta] = Field(
        ..., description="Candidatos generados por ML"
    )
    decision_gemini: DecisionGemini = Field(
        ..., description="Decisión final de Gemini"
    )
    fee_calculation: FEECalculation = Field(..., description="Cálculo FEE")
    tiempo_procesamiento_ms: float = Field(..., ge=0)
    warnings: List[str] = Field(default_factory=list)
    debug_info: Dict[str, Any] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    """📤 Respuesta final del sistema"""

    fecha_entrega_estimada: datetime = Field(..., description="FEE final")
    rango_horario: Dict[str, str] = Field(
        ..., description="Rango de horario de entrega"
    )


    ruta_seleccionada: RutaCompleta = Field(..., description="Ruta ganadora")
    tipo_entrega: TipoEntregaEnum = Field(..., description="Tipo de entrega")
    carrier_principal: str = Field(..., description="Carrier responsable")


    costo_envio_mxn: float = Field(..., ge=0, description="Costo total")
    probabilidad_cumplimiento: float = Field(..., ge=0, le=1)
    confianza_prediccion: float = Field(..., ge=0, le=1)


    explicabilidad: ExplicabilidadCompleta = Field(
        ..., description="Explicabilidad completa"
    )

    # Metadatos
    timestamp_response: datetime = Field(default_factory=datetime.now)
    version_sistema: str = Field(default="3.0.0")