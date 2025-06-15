import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import polars as pl
from config.settings import settings
from utils.logger import logger


class DataManager:
    """🚀 Gestor de datos optimizado - CSVs en memoria, sin cache de respuestas"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._data: Dict[str, pl.DataFrame] = {}
        self._load_time = None
        self._load_all_csvs()

    @staticmethod
    def _fix_lat(lat_val):
        """Corrige latitudes corruptas 194.326 → 19.4326"""
        if lat_val > 90:
            lat_str = str(abs(lat_val))
            return float(lat_str[:2] + '.' + lat_str[2:].replace('.', ''))
        return lat_val

    @staticmethod
    def _fix_lon(lon_val):
        """Corrige longitudes corruptas 991.332 → -99.1332"""
        if lon_val > 180:
            lon_str = str(abs(lon_val))
            return -float(lon_str[:2] + '.' + lon_str[2:].replace('.', ''))
        return lon_val

    def _load_all_csvs(self):
        """📂 Carga Y PRE-PROCESA todos los CSVs optimizados"""
        start_time = time.time()

        # CSV con nombres exactos que mencionaste
        csv_files = {
            'tiendas': 'liverpool_tiendas_completo.csv',
            'productos': 'productos_liverpool_50.csv',
            'stock': 'stock_tienda_sku.csv',
            'cedis': 'cedis_liverpool_completo.csv',
            'codigos_postales': 'codigos_postales_rangos_mexico.csv',
            'factores_externos': 'factores_externos_mexico_completo.csv',
            'flota_externa': 'flota_externa_costos_reales.csv',
            'clima': 'clima_por_rango_cp.csv'
        }

        for key, filename in csv_files.items():
            df = pl.read_csv(self.data_dir / filename)

            # PRE-PROCESAR según tus especificaciones
            if key == 'tiendas':
                # Corregir coordenadas corruptas AUTOMÁTICAMENTE
                df = df.with_columns([
                    pl.col('latitud').map_elements(self._fix_lat).alias('latitud'),
                    pl.col('longitud').map_elements(self._fix_lon).alias('longitud')
                ])

            elif key == 'cedis':
                # Corregir coordenadas CEDIS también
                df = df.with_columns([
                    pl.col('latitud').map_elements(self._fix_lat).alias('latitud'),
                    pl.col('longitud').map_elements(self._fix_lon).alias('longitud')
                ])

            elif key == 'productos':
                # Pre-procesar tiendas_disponibles como lista
                df = df.with_columns([
                    pl.col('tiendas_disponibles').str.split(',').alias('tiendas_list')
                ])

            self._data[key] = df

    def get_data(self, key: str) -> pl.DataFrame:
        """📊 Obtiene DataFrame desde memoria"""
        return self._data.get(key, pl.DataFrame())


class OptimizedProductRepository:
    """📦 Repositorio de productos optimizado"""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def get_product_by_sku(self, sku_id: str) -> Optional[Dict[str, Any]]:
        """🔍 Busca producto por SKU"""
        df = self.data_manager.get_data('productos')
        result = df.filter(pl.col('sku_id') == sku_id)

        if result.height == 0:
            logger.warning(f"❌ Producto no encontrado: {sku_id}")
            return None

        product = result.to_dicts()[0]
        logger.info(f"📦 Producto encontrado: {sku_id} - {product.get('nombre_producto', 'N/A')}")
        return product


class OptimizedStoreRepository:
    """🏪 Repositorio de tiendas optimizado"""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def find_stores_by_postal_range(self, codigo_postal: str) -> List[Dict[str, Any]]:
        """📍 Encuentra tiendas por rango de código postal"""
        # 1. Primero buscar en qué rango está el CP
        cp_info = self._get_postal_info(codigo_postal)
        if not cp_info:
            logger.warning(f"❌ CP no encontrado: {codigo_postal}")
            return []

        # 2. Buscar tiendas en el mismo estado/alcaldía
        target_state = cp_info['estado_alcaldia']
        tiendas_df = self.data_manager.get_data('tiendas')

        # Filtrar por estado/alcaldía similar
        state_matches = tiendas_df.filter(
            pl.col('estado').str.contains(target_state.split()[0]) |
            pl.col('alcaldia_municipio').str.contains(target_state.split()[0])
        )

        if state_matches.height == 0:
            # Fallback: buscar en CDMX o Estado de México
            fallback_states = ['Ciudad de México', 'Estado de México']
            for state in fallback_states:
                state_matches = tiendas_df.filter(pl.col('estado').str.contains(state))
                if state_matches.height > 0:
                    break

        stores = state_matches.to_dicts()

        # 3. Calcular distancias reales
        target_lat = cp_info['latitud_centro']
        target_lon = cp_info['longitud_centro']

        stores_with_distance = []
        for store in stores:
            try:
                from utils.geo_calculator import GeoCalculator
                distance = GeoCalculator.calculate_distance_km(
                    target_lat, target_lon,
                    float(store['latitud']), float(store['longitud'])
                )
                store['distancia_km'] = distance
                stores_with_distance.append(store)
            except Exception as e:
                logger.warning(f"⚠️ Error calculando distancia para {store.get('tienda_id')}: {e}")

        # Ordenar por distancia
        stores_with_distance.sort(key=lambda x: x['distancia_km'])

        logger.info(f"📍 Tiendas encontradas para {codigo_postal}: {len(stores_with_distance)}")
        for i, store in enumerate(stores_with_distance[:5]):
            logger.info(f"  {i + 1}. {store['tienda_id']} - {store['nombre_tienda']} ({store['distancia_km']:.1f}km)")

        return stores_with_distance

    def _get_postal_info(self, codigo_postal: str) -> Optional[Dict[str, Any]]:
        """🔍 Obtiene información del código postal"""
        df = self.data_manager.get_data('codigos_postales')
        cp_int = int(codigo_postal)

        for row in df.to_dicts():
            rango_cp = row.get('rango_cp', '')
            if '-' in rango_cp:
                try:
                    start_cp, end_cp = map(int, rango_cp.split('-'))
                    if start_cp <= cp_int <= end_cp:
                        return row
                except ValueError:
                    continue

        # Fallback: usar CDMX
        return {
            'rango_cp': codigo_postal,
            'estado_alcaldia': 'Ciudad de México',
            'zona_seguridad': 'Amarilla',
            'latitud_centro': 19.4326,
            'longitud_centro': -99.1332,
            'cobertura_liverpool': True,
            'tiempo_entrega_base_horas': '2-4'
        }


class OptimizedStockRepository:
    """📦 Repositorio de stock optimizado"""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def get_stock_for_stores_and_sku(self, sku_id: str, store_ids: List[str],
                                     cantidad_requerida: int) -> List[Dict[str, Any]]:
        """📊 Obtiene stock específico para tiendas y SKU"""
        stock_df = self.data_manager.get_data('stock')

        # Filtrar por SKU y tiendas
        filtered_stock = stock_df.filter(
            (pl.col('sku_id') == sku_id) &
            (pl.col('tienda_id').is_in(store_ids)) &
            (pl.col('stock_disponible') > 0)
        ).sort('stock_disponible', descending=True)

        stock_locations = filtered_stock.to_dicts()

        logger.info(f"📦 Stock encontrado para {sku_id} en {len(store_ids)} tiendas:")
        total_available = 0
        for stock in stock_locations:
            total_available += stock['stock_disponible']
            logger.info(f"  📍 {stock['tienda_id']}: {stock['stock_disponible']} unidades")

        logger.info(f"📊 Stock total disponible: {total_available} | Requerido: {cantidad_requerida}")

        return stock_locations

    def calculate_optimal_allocation(self, stock_locations: List[Dict[str, Any]],
                                     cantidad_requerida: int,
                                     stores_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """🧠 Calcula asignación óptima de stock"""
        if not stock_locations:
            return {
                'factible': False,
                'razon': 'Sin stock disponible',
                'plan': []
            }

        # Crear mapa de distancias
        distance_map = {store['tienda_id']: store['distancia_km'] for store in stores_info}

        # Ordenar por preferencia: stock alto + distancia corta
        def preference_score(stock):
            stock_score = stock['stock_disponible'] / 10.0  # Normalizar
            distance_penalty = distance_map.get(stock['tienda_id'], 999) / 100.0
            return stock_score - distance_penalty

        stock_locations.sort(key=preference_score, reverse=True)

        # Asignar stock de manera óptima
        plan = []
        cantidad_cubierta = 0

        for stock in stock_locations:
            if cantidad_cubierta >= cantidad_requerida:
                break

            cantidad_a_tomar = min(
                stock['stock_disponible'],
                cantidad_requerida - cantidad_cubierta
            )

            plan.append({
                'tienda_id': stock['tienda_id'],
                'cantidad': cantidad_a_tomar,
                'stock_disponible': stock['stock_disponible'],
                'distancia_km': distance_map.get(stock['tienda_id'], 0),
                'precio_tienda': stock.get('precio_tienda', 0)
            })

            cantidad_cubierta += cantidad_a_tomar

        logger.info(f"📋 Plan de asignación:")
        for item in plan:
            logger.info(f"  🏪 {item['tienda_id']}: {item['cantidad']} unidades (${item['precio_tienda']})")

        return {
            'factible': cantidad_cubierta >= cantidad_requerida,
            'plan': plan,
            'cantidad_cubierta': cantidad_cubierta,
            'cantidad_faltante': max(0, cantidad_requerida - cantidad_cubierta),
            'razon': f'Plan con {len(plan)} tiendas' if plan else 'Sin stock suficiente'
        }


class OptimizedExternalFactorsRepository:
    """🌤️ Repositorio de factores externos optimizado"""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def get_factors_for_date_and_cp(self, fecha: datetime, codigo_postal: str) -> Dict[str, Any]:
        """📅 Obtiene factores externos por fecha y CP"""
        df = self.data_manager.get_data('factores_externos')
        fecha_str = fecha.date().isoformat()

        # Buscar por fecha exacta
        exact_match = df.filter(pl.col('fecha') == fecha_str)

        if exact_match.height > 0:
            factors = exact_match.to_dicts()[0]
            logger.info(f"📅 Factores REALES encontrados para {fecha_str}")
            return self._process_factors(factors, fecha, codigo_postal)

        # Buscar fechas cercanas
        for delta in range(1, 4):
            for direction in [-1, 1]:
                check_date = fecha + timedelta(days=delta * direction)
                check_str = check_date.date().isoformat()

                nearby_match = df.filter(pl.col('fecha') == check_str)
                if nearby_match.height > 0:
                    factors = nearby_match.to_dicts()[0]
                    logger.info(f"📅 Usando factores de fecha cercana: {check_str}")
                    return self._process_factors(factors, fecha, codigo_postal)

        # Generar factores automáticos
        logger.info(f"🤖 Generando factores automáticos para {fecha_str}")
        return self._generate_factors(fecha, codigo_postal)

    def _process_factors(self, factors: Dict[str, Any], fecha: datetime, cp: str) -> Dict[str, Any]:
        """🔄 Procesa factores del CSV"""

        # Extraer factor de demanda
        factor_demanda_raw = factors.get('factor_demanda', '1.0')
        if isinstance(factor_demanda_raw, str) and '/' in factor_demanda_raw:
            try:
                num, den = map(float, factor_demanda_raw.split('/'))
                factor_demanda = num / den
            except:
                factor_demanda = 1.0
        else:
            factor_demanda = float(factor_demanda_raw)

        # Calcular impactos
        impacto_tiempo = self._calculate_time_impact(factor_demanda, factors)
        impacto_costo = self._calculate_cost_impact(factor_demanda, factors)

        result = {
            'evento_detectado': factors.get('evento_detectado', 'Normal'),
            'factor_demanda': factor_demanda,
            'condicion_clima': factors.get('condicion_clima', 'Templado'),
            'trafico_nivel': factors.get('trafico_nivel', 'Moderado'),
            'criticidad_logistica': factors.get('criticidad_logistica', 'Normal'),
            'impacto_tiempo_extra_horas': impacto_tiempo,
            'impacto_costo_extra_pct': impacto_costo,
            'es_temporada_alta': factor_demanda > 1.8,
            'es_temporada_critica': factor_demanda > 2.5,
            'fuente_datos': 'CSV_real'
        }

        logger.info(
            f"🌤️ Factores procesados: demanda={factor_demanda:.2f}, tiempo_extra={impacto_tiempo}h, costo_extra={impacto_costo}%")
        return result

    def _calculate_time_impact(self, factor_demanda: float, factors: Dict[str, Any]) -> float:
        """⏱️ Calcula impacto en tiempo"""
        base_impact = max(0, (factor_demanda - 1.0) * 2.0)

        # Eventos críticos
        evento = factors.get('evento_detectado', '')
        if any(critical in evento for critical in ['Navidad', 'Nochebuena', 'Buen_Fin']):
            base_impact += 3.0

        return round(min(8.0, base_impact), 1)

    def _calculate_cost_impact(self, factor_demanda: float, factors: Dict[str, Any]) -> float:
        """💰 Calcula impacto en costo"""
        base_impact = max(0, (factor_demanda - 1.0) * 15)

        # Eventos premium
        evento = factors.get('evento_detectado', '')
        if any(premium in evento for premium in ['Navidad', 'Nochebuena', 'Buen_Fin']):
            base_impact += 20.0

        return round(min(50.0, base_impact), 1)

    def _generate_factors(self, fecha: datetime, cp: str) -> Dict[str, Any]:
        """🤖 Genera factores automáticos"""
        from utils.temporal_detector import TemporalFactorDetector
        return TemporalFactorDetector.detect_comprehensive_factors(fecha, cp)


class OptimizedFleetRepository:
    """🚚 Repositorio de flota externa optimizado"""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def get_best_carriers_for_cp(self, codigo_postal: str, peso_kg: float) -> List[Dict[str, Any]]:
        """🚛 Obtiene mejores carriers para CP y peso"""
        df = self.data_manager.get_data('flota_externa')
        cp_int = int(codigo_postal)

        # Filtrar carriers disponibles
        available = df.filter(
            (pl.col('activo') == True) &
            (pl.col('zona_cp_inicio') <= cp_int) &
            (pl.col('zona_cp_fin') >= cp_int) &
            (pl.col('peso_min_kg') <= peso_kg) &
            (pl.col('peso_max_kg') >= peso_kg)
        ).sort('costo_base_mxn')

        carriers = available.to_dicts()

        logger.info(f"🚚 Carriers disponibles para {codigo_postal} ({peso_kg}kg): {len(carriers)}")
        for carrier in carriers:
            logger.info(
                f"  📦 {carrier['carrier']}: ${carrier['costo_base_mxn']} base ({carrier['tiempo_entrega_dias_habiles']} días)")

        return carriers


# Repositorio unificado
class OptimizedRepositories:
    """🎯 Repositorios optimizados unificados"""

    def __init__(self, data_dir: Path):
        self.data_manager = DataManager(data_dir)
        self.product = OptimizedProductRepository(self.data_manager)
        self.store = OptimizedStoreRepository(self.data_manager)
        self.stock = OptimizedStockRepository(self.data_manager)
        self.external_factors = OptimizedExternalFactorsRepository(self.data_manager)
        self.fleet = OptimizedFleetRepository(self.data_manager)

        logger.info("🚀 Repositorios optimizados inicializados")