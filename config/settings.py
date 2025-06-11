from pathlib import Path
from typing import List, Dict, Any
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """🎯 Configuración avanzada del sistema Liverpool FEE"""

    # ------------------------------------------------------------------
    # App metadata
    # ------------------------------------------------------------------
    APP_NAME: str = "Liverpool FEE Predictor Advanced"
    VERSION: str = "3.0.0"
    DEBUG: bool = False

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"

    # ------------------------------------------------------------------
    # Gemini configuration
    # ------------------------------------------------------------------
    PROJECT_ID: str = "crp-dev-dig-mlcatalog"
    REGION: str = "us-central1"
    MODEL_NAME: str = "gemini-2.0-flash-001"
    GOOGLE_CREDENTIALS_PATH: str = "keys.json"

    # ------------------------------------------------------------------
    # Business rules
    # ------------------------------------------------------------------
    HORARIO_CORTE_FLASH: int = 12   # 12h - same day delivery
    HORARIO_CORTE_EXPRESS: int = 21 # 21h - next day delivery
    TIEMPO_PICKING_PACKING: float = 1.5  # horas por tienda
    TIEMPO_PREPARACION_CEDIS: float = 2.0  # horas en CEDIS

    # ------------------------------------------------------------------
    # Multi-objective optimization weights
    # ------------------------------------------------------------------
    PESO_TIEMPO: float = 0.4
    PESO_COSTO: float = 0.3
    PESO_PROBABILIDAD: float = 0.2
    PESO_DISTANCIA: float = 0.1

    # ------------------------------------------------------------------
    # LightGBM configuration
    # ------------------------------------------------------------------
    LIGHTGBM_PARAMS: Dict[str, Any] = {
        'objective': 'ranking',
        'metric': 'ndcg',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'max_depth': 6,
        'min_data_in_leaf': 20
    }

    # ------------------------------------------------------------------
    # Route optimization thresholds
    # ------------------------------------------------------------------
    MAX_CANDIDATOS_LIGHTGBM: int = 50  # Máximo candidatos para LightGBM
    TOP_CANDIDATOS_GEMINI: int = 3     # Top candidatos para Gemini
    MAX_SPLIT_LOCATIONS: int = 3       # Máximo ubicaciones para split inventory
    MIN_STOCK_THRESHOLD: int = 1       # Stock mínimo requerido

    # ------------------------------------------------------------------
    # Geospatial configuration
    # ------------------------------------------------------------------
    EARTH_RADIUS_KM: float = 6371.0
    MAX_DISTANCE_KM: float = 2000.0    # Distancia máxima entre ubicaciones
    SPEED_FLOTA_INTERNA_KMH: float = 25.0  # Velocidad promedio flota interna
    SPEED_FLOTA_EXTERNA_KMH: float = 35.0  # Velocidad promedio flota externa

    # ------------------------------------------------------------------
    # External factors multipliers
    # ------------------------------------------------------------------
    FACTOR_MULTIPLIERS: Dict[str, Dict[str, float]] = {
        'temporada_alta': {
            'tiempo': 1.8,
            'costo': 1.3,
            'probabilidad': 0.8
        },
        'clima_adverso': {
            'tiempo': 1.4,
            'costo': 1.1,
            'probabilidad': 0.9
        },
        'trafico_alto': {
            'tiempo': 1.6,
            'costo': 1.05,
            'probabilidad': 0.85
        },
        'zona_roja': {
            'tiempo': 1.3,
            'costo': 1.2,
            'probabilidad': 0.7
        }
    }

    # ------------------------------------------------------------------
    # CSV mappings to new files
    # ------------------------------------------------------------------
    CSV_FILES: Dict[str, str] = {
        'productos': 'productos_liverpool_50.csv',
        'tiendas': 'liverpool_tiendas_completo.csv',
        'cedis': 'cedis_liverpool_completo.csv',
        'stock': 'stock_tienda_sku.csv',
        'codigos_postales': 'codigos_postales_rangos_mexico.csv',
        'clima': 'clima_por_rango_cp.csv',
        'factores_externos': 'factores_externos_mexico_completo.csv',
        'flota_externa': 'flota_externa_costos_reales.csv'
    }

    # ------------------------------------------------------------------
    # Seasonal events detection
    # ------------------------------------------------------------------
    EVENTOS_TEMPORADA: Dict[int, List[Dict[str, Any]]] = {
        12: [
            {'evento': 'Navidad', 'factor_demanda': 2.8, 'dias': [20, 25]},
            {'evento': 'Fin_Año', 'factor_demanda': 2.2, 'dias': [26, 31]}
        ],
        11: [
            {'evento': 'Buen_Fin', 'factor_demanda': 3.2, 'dias': [15, 17]},
            {'evento': 'Post_Buen_Fin', 'factor_demanda': 2.0, 'dias': [18, 25]}
        ],
        2: [
            {'evento': 'San_Valentin', 'factor_demanda': 2.2, 'dias': [14, 14]}
        ],
        5: [
            {'evento': 'Dia_Madres', 'factor_demanda': 2.5, 'dias': [8, 12]}
        ]
    }

    # ------------------------------------------------------------------
    # Delivery type rules
    # ------------------------------------------------------------------
    DELIVERY_RULES: Dict[str, Dict[str, Any]] = {
        'FLASH': {
            'max_horas': 24,
            'requiere_flota_interna': True,
            'horario_corte': 12,
            'descripcion': 'Entrega mismo día'
        },
        'EXPRESS': {
            'max_horas': 48,
            'requiere_flota_interna': False,
            'horario_corte': 21,
            'descripcion': 'Entrega siguiente día'
        },
        'STANDARD': {
            'max_horas': 72,
            'requiere_flota_interna': False,
            'horario_corte': 23,
            'descripcion': 'Entrega 2-3 días'
        },
        'PROGRAMADA': {
            'max_horas': 168,  # 7 días
            'requiere_flota_interna': False,
            'horario_corte': 23,
            'descripcion': 'Entrega programada'
        }
    }

    # ------------------------------------------------------------------
    # Performance thresholds
    # ------------------------------------------------------------------
    PERFORMANCE_THRESHOLDS: Dict[str, float] = {
        'max_processing_time_seconds': 5.0,
        'min_confidence_score': 0.7,
        'max_memory_usage_mb': 512,
        'cache_ttl_minutes': 30
    }

    class Config:
        env_file = ".env"


settings = Settings()