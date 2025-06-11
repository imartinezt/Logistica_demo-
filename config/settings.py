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
    # Business rules (más permisivos)
    # ------------------------------------------------------------------
    HORARIO_CORTE_FLASH: int = 14   # Era 12, ahora 14h - más permisivo
    HORARIO_CORTE_EXPRESS: int = 22 # Era 21, ahora 22h - más permisivo
    TIEMPO_PICKING_PACKING: float = 1.2  # Era 1.5, ahora más optimista
    TIEMPO_PREPARACION_CEDIS: float = 1.8  # Era 2.0, ahora más optimista

    # ------------------------------------------------------------------
    # Multi-objective optimization weights (rebalanceados)
    # ------------------------------------------------------------------
    PESO_TIEMPO: float = 0.35      # Era 0.4, reducido para ser menos restrictivo
    PESO_COSTO: float = 0.25       # Era 0.3, reducido
    PESO_PROBABILIDAD: float = 0.30 # Era 0.2, aumentado (más importante)
    PESO_DISTANCIA: float = 0.10   # Igual, menos importante

    # ------------------------------------------------------------------
    # LightGBM configuration (más permisivo)
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
    # Route optimization thresholds (más permisivos)
    # ------------------------------------------------------------------
    MAX_CANDIDATOS_LIGHTGBM: int = 50  # Igual
    TOP_CANDIDATOS_GEMINI: int = 5     # Era 3, ahora 5 para más opciones
    MAX_SPLIT_LOCATIONS: int = 4       # Era 3, ahora 4 para más flexibilidad
    MIN_STOCK_THRESHOLD: int = 1       # Igual

    # ------------------------------------------------------------------
    # Geospatial configuration (más permisivo)
    # ------------------------------------------------------------------
    EARTH_RADIUS_KM: float = 6371.0
    MAX_DISTANCE_KM: float = 2500.0    # Era 2000.0, ahora más permisivo
    SPEED_FLOTA_INTERNA_KMH: float = 28.0  # Era 25.0, más optimista
    SPEED_FLOTA_EXTERNA_KMH: float = 38.0  # Era 35.0, más optimista

    # ------------------------------------------------------------------
    # External factors multipliers (más conservadores)
    # ------------------------------------------------------------------
    FACTOR_MULTIPLIERS: Dict[str, Dict[str, float]] = {
        'temporada_alta': {
            'tiempo': 1.6,      # Era 1.8, más conservador
            'costo': 1.2,       # Era 1.3, más conservador
            'probabilidad': 0.85 # Era 0.8, menos penalización
        },
        'clima_adverso': {
            'tiempo': 1.3,      # Era 1.4, más conservador
            'costo': 1.08,      # Era 1.1, más conservador
            'probabilidad': 0.92 # Era 0.9, menos penalización
        },
        'trafico_alto': {
            'tiempo': 1.4,      # Era 1.6, más conservador
            'costo': 1.03,      # Era 1.05, más conservador
            'probabilidad': 0.88 # Era 0.85, menos penalización
        },
        'zona_roja': {
            'tiempo': 1.25,     # Era 1.3, más conservador
            'costo': 1.15,      # Era 1.2, más conservador
            'probabilidad': 0.75 # Era 0.7, menos penalización
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
    # Seasonal events detection (factores reducidos)
    # ------------------------------------------------------------------
    EVENTOS_TEMPORADA: Dict[int, List[Dict[str, Any]]] = {
        12: [
            {'evento': 'Navidad', 'factor_demanda': 2.4, 'dias': [20, 25]},      # Era 2.8
            {'evento': 'Fin_Año', 'factor_demanda': 1.9, 'dias': [26, 31]}      # Era 2.2
        ],
        11: [
            {'evento': 'Buen_Fin', 'factor_demanda': 2.8, 'dias': [15, 17]},     # Era 3.2
            {'evento': 'Post_Buen_Fin', 'factor_demanda': 1.8, 'dias': [18, 25]} # Era 2.0
        ],
        2: [
            {'evento': 'San_Valentin', 'factor_demanda': 2.0, 'dias': [14, 14]}  # Era 2.2
        ],
        5: [
            {'evento': 'Dia_Madres', 'factor_demanda': 2.2, 'dias': [8, 12]}     # Era 2.5
        ]
    }

    # ------------------------------------------------------------------
    # Delivery type rules (más flexibles)
    # ------------------------------------------------------------------
    DELIVERY_RULES: Dict[str, Dict[str, Any]] = {
        'FLASH': {
            'max_horas': 24,
            'requiere_flota_interna': False,  # Era True, ahora más flexible
            'horario_corte': 14,              # Era 12, ahora más permisivo
            'descripcion': 'Entrega mismo día'
        },
        'EXPRESS': {
            'max_horas': 48,
            'requiere_flota_interna': False,
            'horario_corte': 22,              # Era 21, más permisivo
            'descripcion': 'Entrega siguiente día'
        },
        'STANDARD': {
            'max_horas': 96,                  # Era 72, ahora 4 días
            'requiere_flota_interna': False,
            'horario_corte': 23,
            'descripcion': 'Entrega 2-4 días'
        },
        'PROGRAMADA': {
            'max_horas': 192,                 # Era 168 (7 días), ahora 8 días
            'requiere_flota_interna': False,
            'horario_corte': 23,
            'descripcion': 'Entrega programada'
        }
    }

    # ------------------------------------------------------------------
    # Performance thresholds (más permisivos)
    # ------------------------------------------------------------------
    PERFORMANCE_THRESHOLDS: Dict[str, float] = {
        'max_processing_time_seconds': 8.0,     # Era 5.0, más tiempo
        'min_confidence_score': 0.6,            # Era 0.7, más permisivo
        'max_memory_usage_mb': 768,             # Era 512, más memoria
        'cache_ttl_minutes': 45                 # Era 30, cache más largo
    }

    # ------------------------------------------------------------------
    # Validation thresholds (más permisivos)
    # ------------------------------------------------------------------
    VALIDATION_THRESHOLDS: Dict[str, float] = {
        'min_probability_threshold': 0.5,       # Probabilidad mínima aceptable
        'max_time_hours': 168.0,                # 7 días máximo (era implícito 120)
        'max_cost_mxn': 1000.0,                 # Costo máximo base
        'max_distance_per_segment_km': 500.0,   # Distancia máxima por segmento
        'min_efficiency_score': 0.3             # Score de eficiencia mínimo
    }

    # ------------------------------------------------------------------
    # Error handling configuration
    # ------------------------------------------------------------------
    ERROR_HANDLING: Dict[str, Any] = {
        'max_retries': 3,
        'timeout_seconds': 10.0,
        'fallback_enabled': True,
        'graceful_degradation': True,
        'log_all_errors': True
    }

    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------
    FEATURE_FLAGS: Dict[str, bool] = {
        'enable_gemini_analysis': True,
        'enable_split_inventory': True,
        'enable_hybrid_routes': True,
        'enable_external_factors': True,
        'enable_weather_analysis': True,
        'enable_traffic_analysis': True,
        'enable_seasonal_detection': True,
        'enable_performance_monitoring': True,
        'enable_graceful_fallbacks': True,      # Nuevo
        'enable_permissive_validation': True    # Nuevo
    }

    # ------------------------------------------------------------------
    # Operational limits (más generosos)
    # ------------------------------------------------------------------
    OPERATIONAL_LIMITS: Dict[str, int] = {
        'max_concurrent_requests': 50,          # Era implícito menor
        'max_candidates_generated': 100,        # Era implícito 50
        'max_locations_per_split': 5,           # Era 3
        'max_segments_per_route': 8,            # Era implícito 5
        'max_processing_time_ms': 15000,        # 15 segundos
        'max_gemini_calls_per_request': 5       # Límite de llamadas a Gemini
    }

    class Config:
        env_file = ".env"


settings = Settings()