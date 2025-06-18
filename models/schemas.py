from datetime import datetime, time
from enum import Enum
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, field_validator


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
    """ Request FEE"""
    codigo_postal: str = Field(
        ...,
        pattern=r"^\d{5}$",  # ← solo 5 dígitos
        description="Código postal destino (5 dígitos)"
    )
    sku_id: str = Field(
        ...,
        description="ID del producto"
    )
    cantidad: int = Field(
        ...,
        ge=1, le=100,
        description="Cantidad a entregar"
    )
    fecha_compra: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="Fecha y hora exacta de compra"
    )
    @field_validator("codigo_postal")
    def solo_digitos(cls, v: str) -> str:
        if not v.isdigit():
            raise ValueError("Código postal debe contener solo dígitos")
        return v

class UbicacionStock(BaseModel):
    """ Stock en una ubicación específica"""
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

    tiempo_preparacion: float = Field(..., description="Tiempo picking/packing")
    tiempo_transito: float = Field(..., description="Tiempo en tránsito")
    tiempo_contingencia: float = Field(..., description="Tiempo buffer")