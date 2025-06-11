from pathlib import Path
from typing import List, Dict, Any
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """🎯 Configuración sistema Liverpool FEE"""

    APP_NAME: str = "Liverpool FEE Predictor Advanced"
    VERSION: str = "3.0.0"
    DEBUG: bool = False
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
    # Business rules REALISTAS para México
    # ------------------------------------------------------------------
    HORARIO_CORTE_FLASH: int = 12  # FLASH: compra antes de 12h para entrega mismo día
    HORARIO_CORTE_EXPRESS: int = 20  # EXPRESS: compra antes de 8pm para entrega siguiente día
    TIEMPO_PICKING_PACKING: float = 1.0  # 1 hora para picking/packing
    TIEMPO_PREPARACION_CEDIS: float = 2.0  # 2 horas para CEDIS (incluye cross-dock)

    # ------------------------------------------------------------------
    # Multi-objective weights
    # ------------------------------------------------------------------
    PESO_TIEMPO: float = 0.40  # Prioridad alta al tiempo
    PESO_COSTO: float = 0.20  # Menor peso al costo (no filtrar por límites)
    PESO_PROBABILIDAD: float = 0.35  # Alta prioridad a confiabilidad
    PESO_DISTANCIA: float = 0.05  # Menor importancia a distancia pura
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
    # Route thresholds
    # ------------------------------------------------------------------
    MAX_CANDIDATOS_LIGHTGBM: int = 20  # Menos candidatos, más inteligentes
    TOP_CANDIDATOS_GEMINI: int = 5  # Top 5 para Gemini
    MAX_SPLIT_LOCATIONS: int = 3  # Máximo 3 ubicaciones por split
    MIN_STOCK_THRESHOLD: int = 1  # Mínimo 1 unidad para considerar

    # ------------------------------------------------------------------
    # Geospatial
    # ------------------------------------------------------------------
    EARTH_RADIUS_KM: float = 6371.0
    MAX_DISTANCE_KM: float = 1500.0  # México: máximo 1500km (realista)
    SPEED_FLOTA_INTERNA_KMH: float = 25.0  # Velocidad promedio realista FI
    SPEED_FLOTA_EXTERNA_KMH: float = 35.0  # Velocidad promedio realista FE

    # ------------------------------------------------------------------
    # Factores externos
    # ------------------------------------------------------------------
    FACTOR_MULTIPLIERS: Dict[str, Dict[str, float]] = {
        'temporada_alta': {
            'tiempo': 1.3,  # 30% más tiempo
            'costo': 1.15,  # 15% más costo
            'probabilidad': 0.90  # 90% de la probabilidad base
        },
        'clima_adverso': {
            'tiempo': 1.2,  # 20% más tiempo
            'costo': 1.05,  # 5% más costo
            'probabilidad': 0.95  # 95% de la probabilidad base
        },
        'trafico_alto': {
            'tiempo': 1.25,  # 25% más tiempo
            'costo': 1.02,  # 2% más costo (gasolina)
            'probabilidad': 0.92  # 92% de la probabilidad base
        },
        'zona_roja': {
            'tiempo': 1.15,  # 15% más tiempo
            'costo': 1.10,  # 10% más costo (seguridad)
            'probabilidad': 0.85  # 85% de la probabilidad base
        },
        'temporada_critica': {
            'tiempo': 1.5,  # 50% más tiempo en Navidad
            'costo': 1.25,  # 25% más costo
            'probabilidad': 0.80  # 80% de la probabilidad base
        }
    }

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

    EVENTOS_TEMPORADA: Dict[int, List[Dict[str, Any]]] = {
        12: [
            {'evento': 'Pre_Navidad', 'factor_demanda': 2.0, 'dias': [15, 19]},
            {'evento': 'Navidad_Pico', 'factor_demanda': 3.2, 'dias': [20, 24]},
            {'evento': 'Navidad', 'factor_demanda': 0.3, 'dias': [25, 25]},
            {'evento': 'Post_Navidad', 'factor_demanda': 1.8, 'dias': [26, 31]}
        ],
        11: [
            {'evento': 'Buen_Fin', 'factor_demanda': 3.5, 'dias': [15, 17]},
            {'evento': 'Post_Buen_Fin', 'factor_demanda': 2.0, 'dias': [18, 25]}
        ],
        2: [
            {'evento': 'San_Valentin', 'factor_demanda': 2.2, 'dias': [13, 14]}
        ],
        5: [
            {'evento': 'Dia_Madres', 'factor_demanda': 2.8, 'dias': [8, 12]}
        ],
        1: [
            {'evento': 'Dia_Reyes', 'factor_demanda': 2.4, 'dias': [5, 6]},
            {'evento': 'Enero_Cuesta_Arriba', 'factor_demanda': 0.7, 'dias': [7, 31]}
        ]
    }

    DELIVERY_RULES: Dict[str, Dict[str, Any]] = {
        'FLASH': {
            'max_horas': 8,  # 8 horas máximo (mismo día)
            'requiere_flota_interna': False,
            'horario_corte': 12,  # Compra antes de 12h
            'descripcion': 'Entrega mismo día',
            'solo_area_metropolitana': True
        },
        'EXPRESS': {
            'max_horas': 24,  # 24 horas (siguiente día)
            'requiere_flota_interna': False,
            'horario_corte': 20,  # Compra antes de 8pm
            'descripcion': 'Entrega siguiente día',
            'cobertura_nacional': True
        },
        'STANDARD': {
            'max_horas': 72,  # 3 días
            'requiere_flota_interna': False,
            'horario_corte': 23,
            'descripcion': 'Entrega 2-3 días',
            'cobertura_nacional': True
        },
        'PROGRAMADA': {
            'max_horas': 168,  # 7 días máximo
            'requiere_flota_interna': False,
            'horario_corte': 23,
            'descripcion': 'Entrega programada',
            'cobertura_nacional': True
        }
    }

    PERFORMANCE_THRESHOLDS: Dict[str, float] = {
        'max_processing_time_seconds': 15.0,  # 15 segundos máximo
        'min_confidence_score': 0.65,  # 65% confianza mínima
        'max_memory_usage_mb': 1024,  # 1GB máximo
        'cache_ttl_minutes': 30  # Cache 30 minutos
    }

    VALIDATION_THRESHOLDS: Dict[str, float] = {
        'min_probability_threshold': 0.60,  # 60% probabilidad mínima
        'max_time_hours': 168.0,  # 7 días máximo absoluto
        'max_distance_per_segment_km': 800.0,  # 800km por segmento (realista México)
        'min_efficiency_score': 0.4,  # Score eficiencia mínimo

        # ELIMINADOS: límites de costo arbitrarios
        # El sistema debe evaluar todas las opciones y elegir la mejor
        # sin rechazar por costo alto
    }

    ERROR_HANDLING: Dict[str, Any] = {
        'max_retries': 3,
        'timeout_seconds': 12.0,  # Timeout más alto para Gemini
        'fallback_enabled': True,
        'graceful_degradation': True,
        'log_all_errors': True
    }

    FEATURE_FLAGS: Dict[str, bool] = {
        'enable_gemini_analysis': True,
        'enable_split_inventory': True,
        'enable_hybrid_routes': True,
        'enable_external_factors': True,
        'enable_weather_analysis': True,
        'enable_traffic_analysis': True,
        'enable_seasonal_detection': True,
        'enable_performance_monitoring': True,
        'enable_graceful_fallbacks': True,
        'enable_intelligent_routing': True,  # Nueva funcionalidad
        'enable_cost_optimization': True,  # Optimización sin límites
        'enable_real_csv_factors': True  # Usar factores reales del CSV
    }

    OPERATIONAL_LIMITS: Dict[str, int] = {
        'max_concurrent_requests': 25,  # Límite realista
        'max_candidates_generated': 20,  # Menos candidatos, más inteligentes
        'max_locations_per_split': 3,  # Máximo 3 ubicaciones por split
        'max_segments_per_route': 5,  # Máximo 5 segmentos por ruta
        'max_processing_time_ms': 20000,  # 20 segundos máximo
        'max_gemini_calls_per_request': 3,  # Máximo 3 llamadas a Gemini
        'max_distance_direct_route_km': 300,  # Distancia máxima ruta directa
        'min_stock_for_consideration': 1  # Mínimo stock para considerar ubicación
    }

    # ------------------------------------------------------------------
    # Logica de negocio lIVER
    # ------------------------------------------------------------------
    BUSINESS_CONSTANTS: Dict[str, Any] = {
        # Horarios operativos
        'horario_inicio_operaciones': 9,  # 9 AM
        'horario_fin_operaciones': 18,  # 6 PM
        'dias_operativos': [0, 1, 2, 3, 4, 5],  # Lunes a Sábado

        # Zonas geográficas
        'zona_metropolitana_cdmx_radius_km': 80,
        'zona_nacional_max_distance_km': 1500,

        # Capacidades operativas
        'capacidad_maxima_flota_interna': 150,  # Envíos por día
        'capacidad_maxima_por_tienda': 50,  # Envíos por tienda por día

        # Costos base (para referencia)
        'costo_base_km_flota_interna': 8.5,
        'costo_base_km_flota_externa': 12.0,
        'costo_minimo_envio': 35.0,

        # Tiempos operativos
        'tiempo_minimo_preparacion_minutos': 30,
        'tiempo_maximo_preparacion_horas': 3,
        'tiempo_contingencia_porcentaje': 0.1,  # 10% buffer

        # Factores de calidad
        'probabilidad_base_flota_interna': 0.90,
        'probabilidad_base_flota_externa': 0.82,
        'probabilidad_minima_aceptable': 0.60
    }

    # ------------------------------------------------------------------
    # FEE
    # ------------------------------------------------------------------
    FEE_FORMULAS: Dict[str, Dict[str, Any]] = {
        'FLASH': {
            'formula': 'FC + tiempo_preparacion + tiempo_transito',
            'condiciones': [
                'hora_compra <= 12',
                'distancia <= 80km',
                'zona_metropolitana == True',
                'stock_disponible_local == True'
            ],
            'tiempo_maximo_horas': 8,
            'ejemplo': 'Compra 11 AM → Entrega 4-8 PM mismo día'
        },
        'EXPRESS': {
            'formula': 'FC + 1 día + tiempo_transito',
            'condiciones': [
                'hora_compra <= 20',
                'distancia <= 300km',
                'stock_disponible == True'
            ],
            'tiempo_maximo_horas': 30,
            'ejemplo': 'Compra hoy → Entrega mañana 10 AM - 6 PM'
        },
        'STANDARD': {
            'formula': 'FC + 2-3 días + tiempo_transito + factores_externos',
            'condiciones': [
                'cobertura_nacional == True',
                'cualquier_hora_compra'
            ],
            'tiempo_maximo_horas': 72,
            'ejemplo': 'Compra hoy → Entrega en 2-3 días'
        },
        'PROGRAMADA': {
            'formula': 'FC + días_programados + tiempo_transito + factores_externos',
            'condiciones': [
                'distancia > 300km OR',
                'temporada_critica == True OR',
                'stock_split_complejo == True'
            ],
            'tiempo_maximo_horas': 168,
            'ejemplo': 'Compra hoy → Entrega en 4-7 días'
        }
    }

    class Config:
        env_file = ".env"


settings = Settings()