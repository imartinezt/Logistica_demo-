# models/schemas.py
from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum


# =====================================
# ENUMS
# =====================================

class TipoEntregaEnum(str, Enum):
    FLASH = "FLASH"
    EXPRESS = "EXPRESS"
    STANDARD = "STANDARD"
    PROGRAMADA = "PROGRAMADA"


class TipoFlotaEnum(str, Enum):
    FI = "FI"  # Flota Interna Liverpool
    FE = "FE"  # Flota Externa
    FI_FE = "FI_FE"  # Híbrida


# =====================================
# REQUEST/RESPONSE SCHEMAS
# =====================================

class PredictionRequest(BaseModel):
    """📥 Request para predicción FEE"""
    codigo_postal: str = Field(..., min_length=5, max_length=5, description="Código postal destino")
    sku_id: str = Field(..., description="ID del producto")
    cantidad: int = Field(..., ge=1, le=50, description="Cantidad a entregar")
    fecha_compra: Optional[datetime] = Field(default_factory=datetime.now, description="Fecha y hora de compra")

    @validator('codigo_postal')
    def validate_cp(cls, v):
        if not v.isdigit():
            raise ValueError('Código postal debe ser numérico')
        return v


class Razonamiento(BaseModel):
    """🧠 Paso individual del razonamiento LangGraph + Gemini"""
    paso: str = Field(..., description="Identificador del paso")
    decision: str = Field(..., description="Decisión tomada en este paso")
    factores: List[str] = Field(default_factory=list, description="Factores considerados")
    score: float = Field(..., ge=0, le=1, description="Score de confianza")
    alternativas: List[Dict] = Field(default_factory=list, description="Alternativas evaluadas")
    tiempo_procesamiento_ms: Optional[float] = Field(None, description="Tiempo de procesamiento en ms")


class FactoresExternosDetectados(BaseModel):
    """🌤️ Factores externos detectados automáticamente"""
    eventos_detectados: List[str] = Field(default_factory=list)
    factor_demanda: float = Field(default=1.0, ge=0.5, le=5.0)
    condicion_clima: str = Field(default="Templado")
    probabilidad_lluvia: int = Field(default=30, ge=0, le=100)
    temperatura: int = Field(default=22)
    trafico_nivel: str = Field(default="Moderado")
    es_temporada_alta: bool = Field(default=False)
    impacto_tiempo_extra: int = Field(default=0, description="Horas extra por factores")


class AnalisisStock(BaseModel):
    """📦 Análisis de inventario disponible"""
    ubicaciones_disponibles: int = Field(..., ge=0)
    stock_total: int = Field(..., ge=0)
    ubicacion_optima: str = Field(...)
    stock_ubicacion_optima: int = Field(..., ge=0)
    cobertura_demanda_dias: float = Field(..., ge=0)
    necesita_reabastecimiento: bool = Field(...)


class AnalisisRuta(BaseModel):
    """🚚 Análisis de ruta seleccionada"""
    ruta_id: str = Field(...)
    eslabones_secuencia: str = Field(...)
    nodo_origen: int = Field(...)
    tiempo_base_horas: float = Field(..., ge=0)
    tiempo_ajustado_horas: float = Field(..., ge=0)
    costo_total_mxn: float = Field(..., ge=0)
    probabilidad_cumplimiento: float = Field(..., ge=0, le=1)
    tipo_flota: str = Field(...)
    carrier: str = Field(...)
    es_factible: bool = Field(...)
    razon_seleccion: str = Field(...)


class ExplicabilidadCompleta(BaseModel):
    """📊 Explicabilidad completa del proceso de decisión"""
    flujo_decision: List[Razonamiento] = Field(..., description="Flujo completo de razonamiento")
    producto_info: Dict = Field(..., description="Información del producto")
    zona_info: Dict = Field(..., description="Información de la zona")
    stock_analisis: AnalisisStock = Field(..., description="Análisis de stock")
    factores_externos_detectados: FactoresExternosDetectados = Field(..., description="Factores externos")
    rutas_evaluadas: List[Dict] = Field(..., description="Top rutas evaluadas")
    ruta_seleccionada: AnalisisRuta = Field(..., description="Ruta final seleccionada")
    tiempo_breakdown: Dict = Field(..., description="Desglose de tiempos")
    costo_breakdown: Dict = Field(..., description="Desglose de costos")
    warnings: List[str] = Field(default_factory=list, description="Advertencias del proceso")
    tiempo_total_procesamiento_ms: Optional[float] = Field(None, description="Tiempo total de procesamiento")


class PredictionResponse(BaseModel):
    """📤 Respuesta completa de predicción FEE"""
    fecha_entrega_estimada: datetime = Field(..., description="FEE calculada")
    codigo_postal: str = Field(..., description="CP destino")
    tipo_entrega: TipoEntregaEnum = Field(..., description="Tipo de entrega")
    costo_envio_mxn: float = Field(..., ge=0, description="Costo de envío")
    es_flota_externa: bool = Field(..., description="Si usa flota externa")
    carrier_asignado: str = Field(..., description="Carrier responsable")
    tiempo_estimado_horas: float = Field(..., ge=0, description="Tiempo estimado total")
    probabilidad_cumplimiento: float = Field(..., ge=0, le=1, description="Probabilidad de cumplir FEE")
    explicabilidad: ExplicabilidadCompleta = Field(..., description="Explicabilidad completa")


# =====================================
# INSIGHTS SCHEMAS
# =====================================

class InsightProductos(BaseModel):
    """📊 Insights del catálogo de productos"""
    total_productos: int = Field(..., ge=0)
    por_categoria: Dict[str, int] = Field(default_factory=dict)
    por_nivel_demanda: Dict[str, int] = Field(default_factory=dict)
    productos_fragiles: int = Field(..., ge=0)
    peso_promedio: float = Field(..., ge=0)
    precio_promedio: float = Field(..., ge=0)
    productos_temporada_actual: List[Dict] = Field(default_factory=list)
    top_productos_caros: List[Dict] = Field(default_factory=list)


class InsightInventarios(BaseModel):
    """📦 Insights de inventarios"""
    total_skus: int = Field(..., ge=0)
    stock_total: int = Field(..., ge=0)
    stock_disponible: int = Field(..., ge=0)
    utilizacion_por_nodo: Dict[str, float] = Field(default_factory=dict)
    productos_bajo_stock: List[Dict] = Field(default_factory=list)
    rotacion_promedio: float = Field(..., ge=0)
    nodos_criticos: List[str] = Field(default_factory=list)


class InsightRutas(BaseModel):
    """🚚 Insights de rutas"""
    total_rutas: int = Field(..., ge=0)
    rutas_activas: int = Field(..., ge=0)
    por_tipo_flota: Dict[str, int] = Field(default_factory=dict)
    tiempo_promedio: float = Field(..., ge=0)
    costo_promedio: float = Field(..., ge=0)
    probabilidad_promedio: float = Field(..., ge=0, le=1)
    rutas_mas_eficientes: List[Dict] = Field(default_factory=list)
    rutas_problematicas: List[Dict] = Field(default_factory=list)


# =====================================
# ANALYSIS SCHEMAS
# =====================================

class AnalisisZona(BaseModel):
    """🏠 Análisis completo de zona"""
    codigo_postal: str = Field(...)
    zona_info: Dict = Field(...)
    es_zona_roja: bool = Field(...)
    nivel_riesgo: str = Field(...)
    flota_recomendada: str = Field(...)
    factores_riesgo: List[str] = Field(default_factory=list)
    recomendaciones: List[str] = Field(default_factory=list)
    tiempo_extra_estimado: str = Field(...)


class AnalisisProducto(BaseModel):
    """📦 Análisis completo de producto"""
    sku_id: str = Field(...)
    producto_info: Dict = Field(...)
    stock_ubicaciones: List[Dict] = Field(default_factory=list)
    analisis_demanda: Dict = Field(...)
    consideraciones_especiales: List[str] = Field(default_factory=list)
    recomendaciones: List[str] = Field(default_factory=list)


class DashboardResumen(BaseModel):
    """📊 Resumen para dashboard ejecutivo"""
    kpis_principales: Dict[str, int] = Field(...)
    alertas_operativas: List[str] = Field(default_factory=list)
    rendimiento_sistema: Dict[str, str] = Field(...)
    timestamp: str = Field(...)


# =====================================
# VALIDATION HELPERS
# =====================================

class ResponseStatus(BaseModel):
    """✅ Status response genérico"""
    status: str = Field(...)
    message: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Optional[Dict] = Field(None)


class ErrorResponse(BaseModel):
    """❌ Error response"""
    error: str = Field(...)
    detail: str = Field(...)
    path: Optional[str] = Field(None)
    timestamp: datetime = Field(default_factory=datetime.now)