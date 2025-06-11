import re
import time
from abc import ABC
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import polars as pl

from config.settings import settings
from utils.logger import logger


class BaseRepository(ABC):
    """📊 Repositorio base con Polars y cache inteligente"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._cache: Dict[str, pl.DataFrame] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self.cache_ttl = settings.PERFORMANCE_THRESHOLDS['cache_ttl_minutes'] * 60

    def _load_with_cache(self, filename: str, force_reload: bool = False) -> pl.DataFrame:
        """🔄 Carga CSV con cache automático"""
        current_time = time.time()

        if (force_reload or
                filename not in self._cache or
                current_time - self._cache_timestamps.get(filename, 0) > self.cache_ttl):

            file_path = self.data_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"CSV no encontrado: {file_path}")

            logger.info(f"📂 Cargando CSV: {filename}")
            df = pl.read_csv(file_path)

            self._cache[filename] = df
            self._cache_timestamps[filename] = current_time

            logger.info(f"✅ CSV cargado: {filename} ({df.height} filas)")

        return self._cache[filename]

    def _clean_coordinate_value(self, value: Any) -> float:
        """🧹 Limpia valores de coordenadas problemáticos"""
        if value is None:
            return 0.0

        str_value = str(value).strip()
        cleaned = re.sub(r'[^\d\.\-]', '', str_value)

        try:
            coord_float = float(cleaned)
            return coord_float
        except (ValueError, TypeError):
            logger.warning(f"⚠️ Coordenada inválida: {value} -> usando 0.0")
            return 0.0

    def _clean_id_value(self, value: Any) -> str:
        """🧹 Limpia valores de ID problemáticos"""
        if value is None:
            return ""

        str_value = str(value).strip()
        cleaned = re.sub(r'^\([^)]*\)', '', str_value).strip()
        cleaned = re.sub(r'[^\w\-]', '', cleaned)

        return cleaned if len(cleaned) >= 2 else ""


class StoreRepository(BaseRepository):
    """🏪 Repositorio OPTIMIZADO de tiendas Liverpool"""

    def load_data(self) -> pl.DataFrame:
        return self._load_with_cache(settings.CSV_FILES['tiendas'])

    def get_store_by_id(self, tienda_id: str) -> Optional[Dict[str, Any]]:
        """🔍 Busca tienda por ID con limpieza robusta"""
        df = self.load_data()
        clean_search_id = self._clean_id_value(tienda_id)

        for store in df.to_dicts():
            try:
                store_id_raw = store.get('tienda_id', '')
                store_id_clean = self._clean_id_value(store_id_raw)

                if store_id_clean == clean_search_id and store_id_clean:
                    store['latitud'] = self._clean_coordinate_value(store.get('latitud'))
                    store['longitud'] = self._clean_coordinate_value(store.get('longitud'))
                    store['tienda_id'] = store_id_clean

                    if not (14.0 <= store['latitud'] <= 33.0 and -118.0 <= store['longitud'] <= -86.0):
                        logger.warning(f"⚠️ Coordenadas fuera de México para {store_id_clean}")
                        continue

                    return store

            except Exception as e:
                logger.warning(f"❌ Error procesando tienda {store_id_raw}: {e}")
                continue

        logger.warning(f"❌ Tienda no encontrada: {tienda_id}")
        return None

    def find_stores_by_postal_code_range(self, codigo_postal: str,
                                         max_distance_km: float = 100.0) -> List[Dict[str, Any]]:
        """📍 OPTIMIZADO: Encuentra tiendas cercanas por código postal"""
        from utils.geo_calculator import GeoCalculator

        # Obtener coordenadas del CP destino
        postal_repo = PostalCodeRepository(self.data_dir)
        cp_info = postal_repo.get_postal_code_info(codigo_postal)

        if not cp_info:
            logger.warning(f"❌ CP no encontrado: {codigo_postal}")
            return []

        cp_lat = cp_info.get('latitud_centro', 19.4326)
        cp_lon = cp_info.get('longitud_centro', -99.1332)

        # OPTIMIZACIÓN: Pre-filtrar por región si es posible
        stores_list = self._get_stores_by_region_first(codigo_postal[:2])

        if not stores_list:
            # Si no hay filtro por región, cargar todas
            df = self.load_data()
            stores_list = df.to_dicts()

        nearby_stores = []
        processed_count = 0

        for store in stores_list:
            try:
                processed_count += 1

                tienda_id_raw = store.get('tienda_id', '')
                tienda_id_clean = self._clean_id_value(tienda_id_raw)

                if not tienda_id_clean:
                    continue

                store_lat = self._clean_coordinate_value(store.get('latitud'))
                store_lon = self._clean_coordinate_value(store.get('longitud'))

                if not (14.0 <= store_lat <= 33.0 and -118.0 <= store_lon <= -86.0):
                    continue

                distance = GeoCalculator.calculate_distance_km(
                    cp_lat, cp_lon, store_lat, store_lon
                )

                if distance <= max_distance_km:
                    store_clean = store.copy()
                    store_clean['tienda_id'] = tienda_id_clean
                    store_clean['latitud'] = store_lat
                    store_clean['longitud'] = store_lon
                    store_clean['distancia_km'] = distance
                    nearby_stores.append(store_clean)

            except Exception as e:
                logger.warning(f"❌ Error procesando tienda {tienda_id_raw}: {e}")
                continue

        # Ordenar por distancia
        nearby_stores.sort(key=lambda x: x['distancia_km'])

        logger.info(f"📍 Encontradas {len(nearby_stores)} tiendas cercanas a {codigo_postal} "
                    f"(procesadas {processed_count} tiendas)")
        return nearby_stores

    def _get_stores_by_region_first(self, cp_prefix: str) -> List[Dict[str, Any]]:
        """🗺️ Pre-filtro por región para optimizar búsqueda"""

        # Mapeo de prefijos CP a regiones conocidas
        region_filters = {
            '01': 'CDMX',
            '02': 'CDMX',
            '03': 'CDMX',
            '04': 'CDMX',
            '05': 'CDMX',
            '06': 'CDMX',
            '07': 'CDMX',
            '08': 'CDMX',
            '09': 'CDMX',
            '10': 'Estado_Mexico',
            '11': 'Estado_Mexico',
            '12': 'Estado_Mexico',
            '44': 'Jalisco',
            '45': 'Jalisco',
            '64': 'Nuevo_Leon',
            '80': 'Sinaloa'
        }

        target_region = region_filters.get(cp_prefix)
        if not target_region:
            return []  # Si no conocemos la región, procesar todas

        try:
            df = self.load_data()

            # Si tenemos columna de región/estado, filtrar
            if 'estado' in df.columns:
                filtered_df = df.filter(
                    pl.col('estado').str.contains(target_region)
                )
                if filtered_df.height > 0:
                    logger.info(f"📍 Pre-filtrado por región {target_region}: {filtered_df.height} tiendas")
                    return filtered_df.to_dicts()

            # Si no hay columna de región, no filtrar
            return []

        except Exception as e:
            logger.warning(f"❌ Error en pre-filtro por región: {e}")
            return []


class StockRepository(BaseRepository):
    """📦 Repositorio OPTIMIZADO de inventarios con split inteligente"""

    def load_data(self) -> pl.DataFrame:
        return self._load_with_cache(settings.CSV_FILES['stock'])

    def get_stock_locations_optimized(self, sku_id: str, cantidad_requerida: int,
                                      nearby_store_ids: List[str] = None) -> List[Dict[str, Any]]:
        """📋 OPTIMIZADO: Obtiene stock SOLO en tiendas cercanas"""

        df = self.load_data()

        # OPTIMIZACIÓN: Si tenemos IDs de tiendas cercanas, filtrar primero
        if nearby_store_ids:
            # Crear lista de IDs limpios para comparación
            clean_nearby_ids = [self._clean_id_value(store_id) for store_id in nearby_store_ids]

            # Filtrar por SKU y tiendas cercanas
            stock_df = df.filter(pl.col('sku_id') == sku_id)

            nearby_stock = []
            for stock_row in stock_df.to_dicts():
                clean_tienda_id = self._clean_id_value(stock_row.get('tienda_id', ''))
                if (clean_tienda_id in clean_nearby_ids and
                        stock_row.get('stock_disponible', 0) >= settings.MIN_STOCK_THRESHOLD):
                    stock_row['tienda_id'] = clean_tienda_id
                    nearby_stock.append(stock_row)

            # Ordenar por stock disponible (descendente)
            nearby_stock.sort(key=lambda x: x.get('stock_disponible', 0), reverse=True)

            logger.info(f"📦 Stock en tiendas cercanas para {sku_id}: {len(nearby_stock)} ubicaciones")
            return nearby_stock

        # Si no hay filtro de tiendas cercanas, buscar en todas (método original)
        return self.get_stock_locations(sku_id, cantidad_requerida)

    def get_stock_locations(self, sku_id: str, cantidad_requerida: int) -> List[Dict[str, Any]]:
        """📋 Obtiene ubicaciones con stock para un SKU (método original)"""
        df = self.load_data()

        stock_df = df.filter(
            (pl.col('sku_id') == sku_id) &
            (pl.col('stock_disponible') > 0)
        ).sort('stock_disponible', descending=True)

        stock_locations = []
        for location in stock_df.to_dicts():
            tienda_id_raw = location.get('tienda_id', '')
            tienda_id_clean = self._clean_id_value(tienda_id_raw)

            if tienda_id_clean and location['stock_disponible'] >= settings.MIN_STOCK_THRESHOLD:
                location['tienda_id'] = tienda_id_clean
                stock_locations.append(location)

        logger.info(f"📦 Stock disponible para {sku_id}: {len(stock_locations)} ubicaciones")
        return stock_locations

    def calculate_smart_split_inventory(self, sku_id: str, cantidad_requerida: int,
                                        nearby_stores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """🧠 SPLIT INTELIGENTE: evalúa opciones reales optimizado"""

        # Obtener stock SOLO en tiendas cercanas
        nearby_store_ids = [store['tienda_id'] for store in nearby_stores]
        stock_locations = self.get_stock_locations_optimized(
            sku_id, cantidad_requerida, nearby_store_ids
        )

        if not stock_locations:
            # Crear un split_inventory vacío para mantener consistencia
            from models.schemas import SplitInventory

            empty_split_inventory = SplitInventory(
                ubicaciones=[],
                cantidad_total_requerida=cantidad_requerida,
                cantidad_total_disponible=0,
                es_split_factible=False,
                razon_split='Sin stock disponible en tiendas cercanas'
            )

            return {
                'es_factible': False,
                'razon': 'Sin stock disponible en tiendas cercanas',
                'split_inventory': empty_split_inventory,  # ✅ Consistente
                'split_plan': [],
                'cantidad_cubierta': 0,
                'cantidad_faltante': cantidad_requerida,
                'opciones_evaluadas': 0
            }

        # Crear diccionario de distancias para acceso rápido
        distance_map = {store['tienda_id']: store['distancia_km'] for store in nearby_stores}

        # OPCIÓN 1: Tienda única con suficiente stock
        single_store_options = []
        for stock_loc in stock_locations:
            if stock_loc['stock_disponible'] >= cantidad_requerida:
                distance = distance_map.get(stock_loc['tienda_id'], 999)
                single_store_options.append({
                    'tipo': 'tienda_unica',
                    'tienda_id': stock_loc['tienda_id'],
                    'cantidad': cantidad_requerida,
                    'stock_disponible': stock_loc['stock_disponible'],
                    'distancia_km': distance,
                    'complejidad': 1,
                    'eficiencia_score': self._calculate_efficiency_score(cantidad_requerida, distance, 1)
                })

        # OPCIÓN 2: Split entre múltiples tiendas (máximo 3)
        multi_store_options = []
        if len(stock_locations) > 1:
            split_combinations = self._generate_split_combinations(
                stock_locations, cantidad_requerida, distance_map, max_stores=3
            )
            multi_store_options.extend(split_combinations)

        # Evaluar todas las opciones
        all_options = single_store_options + multi_store_options

        if not all_options:
            # Crear un split_inventory vacío para mantener consistencia
            from models.schemas import SplitInventory

            empty_split_inventory = SplitInventory(
                ubicaciones=[],
                cantidad_total_requerida=cantidad_requerida,
                cantidad_total_disponible=sum(loc["stock_disponible"] for loc in stock_locations),
                es_split_factible=False,
                razon_split=f'Stock insuficiente. Disponible: {sum(loc["stock_disponible"] for loc in stock_locations)}, Requerido: {cantidad_requerida}'
            )

            return {
                'es_factible': False,
                'razon': f'Stock insuficiente. Disponible: {sum(loc["stock_disponible"] for loc in stock_locations)}, Requerido: {cantidad_requerida}',
                'split_inventory': empty_split_inventory,  # ✅ Consistente
                'split_plan': [],
                'cantidad_cubierta': 0,
                'cantidad_faltante': cantidad_requerida,
                'opciones_evaluadas': 0
            }

        # Elegir la MEJOR opción por eficiencia
        mejor_opcion = max(all_options, key=lambda x: x['eficiencia_score'])

        # Construir split plan
        if mejor_opcion['tipo'] == 'tienda_unica':
            split_plan = [{
                'tienda_id': mejor_opcion['tienda_id'],
                'cantidad': mejor_opcion['cantidad'],
                'stock_disponible': mejor_opcion['stock_disponible'],
                'distancia_km': mejor_opcion['distancia_km'],
                'prioridad': 1
            }]
        else:
            split_plan = mejor_opcion['split_plan']

        # CREAR SPLIT_INVENTORY OBJECT
        split_inventory_obj = self._build_split_inventory_object(split_plan, cantidad_requerida, nearby_stores)

        return {
            'es_factible': True,
            'cantidad_cubierta': cantidad_requerida,
            'cantidad_faltante': 0,
            'split_inventory': split_inventory_obj,  # ✅ Agregado el objeto requerido
            'split_plan': split_plan,
            'razon': f'Opción {mejor_opcion["tipo"]}: {len(split_plan)} ubicaciones (eficiencia: {mejor_opcion["eficiencia_score"]:.3f})',
            'opciones_evaluadas': len(all_options),
            'mejor_opcion': mejor_opcion
        }

    def _generate_split_combinations(self, stock_locations: List[Dict[str, Any]],
                                     cantidad_requerida: int,
                                     distance_map: Dict[str, float],
                                     max_stores: int = 3) -> List[Dict[str, Any]]:
        """🔄 Genera combinaciones inteligentes de split"""

        combinations = []

        # Combinaciones de 2 tiendas
        for i in range(len(stock_locations)):
            for j in range(i + 1, len(stock_locations)):
                combo = self._evaluate_two_store_combo(
                    stock_locations[i], stock_locations[j],
                    cantidad_requerida, distance_map
                )
                if combo and combo['cantidad_cubierta'] >= cantidad_requerida:
                    combinations.append(combo)

        # Combinaciones de 3 tiendas (solo si es necesario)
        if max_stores >= 3 and not combinations:
            for i in range(len(stock_locations)):
                for j in range(i + 1, len(stock_locations)):
                    for k in range(j + 1, len(stock_locations)):
                        combo = self._evaluate_three_store_combo(
                            stock_locations[i], stock_locations[j], stock_locations[k],
                            cantidad_requerida, distance_map
                        )
                        if combo and combo['cantidad_cubierta'] >= cantidad_requerida:
                            combinations.append(combo)

        return combinations

    def _evaluate_two_store_combo(self, store1: Dict[str, Any], store2: Dict[str, Any],
                                  cantidad_requerida: int, distance_map: Dict[str, float]) -> Dict[str, Any]:
        """📊 Evalúa combinación de 2 tiendas"""

        stock1 = store1['stock_disponible']
        stock2 = store2['stock_disponible']

        if stock1 + stock2 < cantidad_requerida:
            return None

        # Optimizar distribución
        cantidad1 = min(stock1, cantidad_requerida)
        cantidad2 = max(0, cantidad_requerida - cantidad1)

        if cantidad2 > stock2:
            return None

        dist1 = distance_map.get(store1['tienda_id'], 999)
        dist2 = distance_map.get(store2['tienda_id'], 999)

        split_plan = [
            {
                'tienda_id': store1['tienda_id'],
                'cantidad': cantidad1,
                'stock_disponible': stock1,
                'distancia_km': dist1,
                'prioridad': 1
            },
            {
                'tienda_id': store2['tienda_id'],
                'cantidad': cantidad2,
                'stock_disponible': stock2,
                'distancia_km': dist2,
                'prioridad': 2
            }
        ]

        total_distance = dist1 + dist2
        eficiencia_score = self._calculate_efficiency_score(cantidad_requerida, total_distance, 2)

        return {
            'tipo': 'dos_tiendas',
            'split_plan': split_plan,
            'cantidad_cubierta': cantidad1 + cantidad2,
            'distancia_total': total_distance,
            'complejidad': 2,
            'eficiencia_score': eficiencia_score
        }

    def _evaluate_three_store_combo(self, store1: Dict[str, Any], store2: Dict[str, Any],
                                    store3: Dict[str, Any], cantidad_requerida: int,
                                    distance_map: Dict[str, float]) -> Dict[str, Any]:
        """📊 Evalúa combinación de 3 tiendas"""

        stock1 = store1['stock_disponible']
        stock2 = store2['stock_disponible']
        stock3 = store3['stock_disponible']

        if stock1 + stock2 + stock3 < cantidad_requerida:
            return None

        # Distribución optimizada (greedy por stock disponible)
        stores = [
            (store1, stock1, distance_map.get(store1['tienda_id'], 999)),
            (store2, stock2, distance_map.get(store2['tienda_id'], 999)),
            (store3, stock3, distance_map.get(store3['tienda_id'], 999))
        ]

        # Ordenar por distancia para optimizar
        stores.sort(key=lambda x: x[2])

        split_plan = []
        cantidad_pendiente = cantidad_requerida
        total_distance = 0

        for i, (store, stock, distance) in enumerate(stores):
            if cantidad_pendiente <= 0:
                break

            cantidad_a_tomar = min(stock, cantidad_pendiente)

            split_plan.append({
                'tienda_id': store['tienda_id'],
                'cantidad': cantidad_a_tomar,
                'stock_disponible': stock,
                'distancia_km': distance,
                'prioridad': i + 1
            })

            cantidad_pendiente -= cantidad_a_tomar
            total_distance += distance

        if cantidad_pendiente > 0:
            return None

        eficiencia_score = self._calculate_efficiency_score(cantidad_requerida, total_distance, 3)

        return {
            'tipo': 'tres_tiendas',
            'split_plan': split_plan,
            'cantidad_cubierta': cantidad_requerida,
            'distancia_total': total_distance,
            'complejidad': 3,
            'eficiencia_score': eficiencia_score
        }

    def _calculate_efficiency_score(self, cantidad: int, distancia_total: float, complejidad: int) -> float:
        """📊 Calcula score de eficiencia para comparar opciones"""

        # Penalizar distancia y complejidad
        distance_penalty = min(1.0, distancia_total / 100.0)  # Normalizar a 100km
        complexity_penalty = (complejidad - 1) * 0.2  # 20% de penalización por tienda adicional

        # Bonus por cantidad (economías de escala)
        quantity_bonus = min(0.3, cantidad / 10.0)  # Hasta 30% bonus por cantidad

        # Score final (0-1, donde 1 es mejor)
        score = 1.0 - distance_penalty - complexity_penalty + quantity_bonus

        return max(0.1, min(1.0, score))

    def _build_split_inventory_object(self, split_plan: List[Dict[str, Any]],
                                      cantidad_requerida: int,
                                      nearby_stores: List[Dict[str, Any]]):
        """🏗️ Construye objeto SplitInventory requerido por el sistema"""
        from models.schemas import SplitInventory, UbicacionStock
        from config.settings import settings

        # Crear diccionario de tiendas para acceso rápido
        stores_map = {store['tienda_id']: store for store in nearby_stores}

        ubicaciones_split = []
        for location_plan in split_plan:
            tienda_id = location_plan['tienda_id']
            store_info = stores_map.get(tienda_id)

            if store_info:
                ubicacion_split = UbicacionStock(
                    ubicacion_id=tienda_id,
                    ubicacion_tipo='TIENDA',
                    nombre_ubicacion=store_info.get('nombre_tienda', f"Tienda {tienda_id}"),
                    stock_disponible=location_plan['cantidad'],
                    stock_reservado=0,
                    coordenadas={
                        'lat': store_info['latitud'],
                        'lon': store_info['longitud']
                    },
                    horario_operacion=store_info.get('horario_operacion', '09:00-21:00'),
                    tiempo_preparacion_horas=settings.TIEMPO_PICKING_PACKING
                )
                ubicaciones_split.append(ubicacion_split)

        cantidad_total_disponible = sum(loc['cantidad'] for loc in split_plan)

        split_inventory = SplitInventory(
            ubicaciones=ubicaciones_split,
            cantidad_total_requerida=cantidad_requerida,
            cantidad_total_disponible=cantidad_total_disponible,
            es_split_factible=True,
            razon_split=f"Split optimizado: {len(split_plan)} ubicaciones"
        )

        return split_inventory


class ExternalFactorsRepository(BaseRepository):
    """🌐 Repositorio MEJORADO de factores externos"""

    def load_data(self) -> pl.DataFrame:
        return self._load_with_cache(settings.CSV_FILES['factores_externos'])

    def get_factors_by_date_and_region(self, fecha: datetime,
                                       codigo_postal: str) -> Dict[str, Any]:
        """📅 MEJORADO: Busca factores REALES en CSV primero"""

        df = self.load_data()
        fecha_str = fecha.date().isoformat()

        # Buscar por fecha exacta
        exact_match = df.filter(pl.col('fecha') == fecha_str)

        if exact_match.height > 0:
            # Si hay múltiples registros, filtrar por CP
            if 'rango_cp_afectado' in df.columns:
                cp_prefix = codigo_postal[:2]
                cp_filtered = exact_match.filter(
                    pl.col('rango_cp_afectado').str.contains(cp_prefix)
                )
                if cp_filtered.height > 0:
                    factors = cp_filtered.to_dicts()[0]
                else:
                    factors = exact_match.to_dicts()[0]
            else:
                factors = exact_match.to_dicts()[0]

            logger.info(f"📅 Factores REALES encontrados en CSV para {fecha_str}")
            return self._process_real_factors(factors, fecha, codigo_postal)

        # Buscar fechas cercanas (±3 días)
        for delta in range(1, 4):
            for direction in [-1, 1]:
                check_date = fecha + timedelta(days=delta * direction)
                check_str = check_date.date().isoformat()

                nearby_match = df.filter(pl.col('fecha') == check_str)
                if nearby_match.height > 0:
                    factors = nearby_match.to_dicts()[0]
                    logger.info(f"📅 Usando factores de fecha cercana: {check_str}")
                    return self._process_real_factors(factors, fecha, codigo_postal)

        # Si no hay datos reales, generar automáticamente
        logger.info(f"🤖 Generando factores automáticos para {fecha_str}")
        return self._generate_auto_factors(fecha, codigo_postal)

    def _process_real_factors(self, factors: Dict[str, Any], fecha: datetime, codigo_postal: str) -> Dict[str, Any]:
        """🔄 Procesa factores REALES del CSV"""

        return {
            'fecha': fecha.date().isoformat(),
            'evento_detectado': factors.get('evento_detectado', 'Normal'),
            'factor_demanda': float(factors.get('factor_demanda', 1.0)),
            'condicion_clima': factors.get('condicion_clima', 'Templado'),
            'trafico_nivel': factors.get('trafico_nivel', 'Moderado'),
            'rango_cp_afectado': factors.get('rango_cp_afectado', f"{codigo_postal[:2]}000-{codigo_postal[:2]}999"),
            'impacto_tiempo_extra_horas': float(factors.get('impacto_tiempo_extra_horas', 0)),
            'impacto_costo_extra_pct': float(factors.get('impacto_costo_extra_pct', 0)),
            'criticidad_logistica': factors.get('criticidad_logistica', 'Normal'),
            'fuente_datos': 'CSV_real'
        }

    def _generate_auto_factors(self, fecha: datetime, codigo_postal: str) -> Dict[str, Any]:
        """🤖 Genera factores automáticamente con lógica mejorada"""
        from utils.temporal_detector import TemporalFactorDetector

        # Usar el detector temporal para generar factores inteligentes
        detected_factors = TemporalFactorDetector.detect_comprehensive_factors(fecha, codigo_postal)

        return {
            'fecha': fecha.date().isoformat(),
            'evento_detectado': detected_factors['eventos_detectados'][0] if detected_factors[
                'eventos_detectados'] else 'Normal',
            'factor_demanda': detected_factors['factor_demanda'],
            'condicion_clima': detected_factors['condicion_clima'],
            'trafico_nivel': detected_factors['trafico_nivel'],
            'rango_cp_afectado': f"{codigo_postal[:2]}000-{codigo_postal[:2]}999",
            'impacto_tiempo_extra_horas': detected_factors['impacto_tiempo_extra_horas'],
            'impacto_costo_extra_pct': detected_factors['impacto_costo_extra_pct'],
            'criticidad_logistica': detected_factors['criticidad_logistica'],
            'fuente_datos': 'generado_automatico'
        }


# Los demás repositorios mantienen su implementación original
class ProductRepository(BaseRepository):
    """📦 Repositorio de productos Liverpool"""

    def load_data(self) -> pl.DataFrame:
        return self._load_with_cache(settings.CSV_FILES['productos'])

    def get_product_by_sku(self, sku_id: str) -> Optional[Dict[str, Any]]:
        """🔍 Busca producto por SKU con validación avanzada"""
        df = self.load_data()
        result = df.filter(pl.col('sku_id') == sku_id)

        if result.height == 0:
            logger.warning(f"❌ Producto no encontrado: {sku_id}")
            return None

        product_dict = result.to_dicts()[0]

        if product_dict.get('peso_kg', 0) > 30:
            logger.warning(f"⚠️ Producto pesado detectado: {product_dict['peso_kg']}kg")

        if product_dict.get('es_fragil', False):
            logger.info(f"🔸 Producto frágil detectado: {sku_id}")

        return product_dict

    def check_seasonal_availability(self, sku_id: str, fecha: datetime) -> bool:
        """🗓️ Verifica disponibilidad estacional"""
        product = self.get_product_by_sku(sku_id)
        if not product:
            return False

        temporada_producto = product.get('temporada', 'Todo_Año')
        mes_actual = fecha.month

        if temporada_producto == 'Todo_Año':
            return True
        elif temporada_producto == 'Invierno' and mes_actual in [12, 1, 2]:
            return True
        elif temporada_producto == 'Verano' and mes_actual in [6, 7, 8]:
            return True

        return temporada_producto == 'Todo_Año'


class CEDISRepository(BaseRepository):
    """🏭 Repositorio de CEDIS"""

    def load_data(self) -> pl.DataFrame:
        return self._load_with_cache(settings.CSV_FILES['cedis'])

    def get_cedis_by_id(self, cedis_id: str) -> Optional[Dict[str, Any]]:
        """🔍 Busca CEDIS por ID con limpieza robusta"""
        df = self.load_data()
        clean_search_id = self._clean_id_value(cedis_id)

        for cedis in df.to_dicts():
            try:
                cedis_id_raw = cedis.get('cedis_id', '')
                cedis_id_clean = self._clean_id_value(cedis_id_raw)

                if cedis_id_clean == clean_search_id and cedis_id_clean:
                    cedis['latitud'] = self._clean_coordinate_value(cedis.get('latitud'))
                    cedis['longitud'] = self._clean_coordinate_value(cedis.get('longitud'))
                    cedis['cedis_id'] = cedis_id_clean

                    if 14.0 <= cedis['latitud'] <= 33.0 and -118.0 <= cedis['longitud'] <= -86.0:
                        return cedis

            except Exception as e:
                logger.warning(f"❌ Error procesando CEDIS {cedis_id_raw}: {e}")
                continue

        logger.warning(f"❌ CEDIS no encontrado: {cedis_id}")
        return None


class PostalCodeRepository(BaseRepository):
    """📮 Repositorio de códigos postales y rangos"""

    def load_data(self) -> pl.DataFrame:
        return self._load_with_cache(settings.CSV_FILES['codigos_postales'])

    def get_postal_code_info(self, codigo_postal: str) -> Optional[Dict[str, Any]]:
        """🔍 Obtiene información de código postal mejorado"""
        df = self.load_data()

        try:
            cp_int = int(codigo_postal)
        except ValueError:
            logger.warning(f"❌ CP inválido: {codigo_postal}")
            return None

        for row in df.to_dicts():
            try:
                rango_cp = str(row.get('rango_cp', '')).strip()
                rango_cp_clean = self._clean_id_value(rango_cp)

                if '-' in rango_cp_clean:
                    parts = rango_cp_clean.split('-')
                    if len(parts) == 2:
                        try:
                            start_cp = int(parts[0].strip())
                            end_cp = int(parts[1].strip())

                            if start_cp <= cp_int <= end_cp:
                                return self._process_postal_info(row)
                        except ValueError:
                            continue
                elif rango_cp_clean.isdigit():
                    if int(rango_cp_clean) == cp_int:
                        return self._process_postal_info(row)

            except Exception as e:
                logger.warning(f"❌ Error procesando rango CP {rango_cp}: {e}")
                continue

        logger.warning(f"❌ CP no encontrado: {codigo_postal}, usando default CDMX")
        return {
            'rango_cp': codigo_postal,
            'estado_alcaldia': 'Ciudad de México',
            'zona_seguridad': 'Media',
            'latitud_centro': 19.4326,
            'longitud_centro': -99.1332,
            'cobertura_liverpool': True,
            'tiempo_entrega_base_horas': '24-48',
            'observaciones': 'Default CDMX'
        }

    def _process_postal_info(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """🔧 Procesa información de CP limpiando coordenadas"""
        processed = row.copy()

        processed['latitud_centro'] = self._clean_coordinate_value(row.get('latitud_centro'))
        processed['longitud_centro'] = self._clean_coordinate_value(row.get('longitud_centro'))

        if not (14.0 <= processed['latitud_centro'] <= 33.0 and
                -118.0 <= processed['longitud_centro'] <= -86.0):
            logger.warning(f"⚠️ Coordenadas fuera de México, usando CDMX")
            processed['latitud_centro'] = 19.4326
            processed['longitud_centro'] = -99.1332

        return processed

    def is_zona_roja(self, codigo_postal: str) -> bool:
        """🚨 Detecta zona roja"""
        cp_info = self.get_postal_code_info(codigo_postal)
        if not cp_info:
            return False

        zona_seguridad = str(cp_info.get('zona_seguridad', '')).lower()
        return zona_seguridad in ['roja', 'alta', 'crítica']


class ClimateRepository(BaseRepository):
    """🌤️ Repositorio de datos climáticos"""

    def load_data(self) -> pl.DataFrame:
        return self._load_with_cache(settings.CSV_FILES['clima'])

    def get_climate_by_postal_code(self, codigo_postal: str, fecha: datetime) -> Dict[str, Any]:
        """🌡️ Obtiene datos climáticos por CP y fecha"""
        # Implementación original mantenida
        return self._get_default_climate(fecha)

    def _get_default_climate(self, fecha: datetime) -> Dict[str, Any]:
        """🌡️ Clima por defecto"""
        return {
            'clima_actual': 'Templado',
            'temperatura': 22,
            'precipitacion_mm': 30,
            'factor_clima': 1.0
        }


class ExternalFleetRepository(BaseRepository):
    """🚚 Repositorio de flota externa"""

    def load_data(self) -> pl.DataFrame:
        return self._load_with_cache(settings.CSV_FILES['flota_externa'])

    def get_available_carriers(self, codigo_postal: str, peso_kg: float) -> List[Dict[str, Any]]:
        """📋 Obtiene carriers disponibles para CP y peso"""
        df = self.load_data()

        try:
            cp_int = int(codigo_postal)
        except ValueError:
            return []

        carriers = df.filter(
            (pl.col('activo') == True) &
            (cp_int >= pl.col('zona_cp_inicio')) &
            (cp_int <= pl.col('zona_cp_fin')) &
            (peso_kg >= pl.col('peso_min_kg')) &
            (peso_kg <= pl.col('peso_max_kg'))
        ).sort('costo_base_mxn')

        carriers_list = carriers.to_dicts()
        logger.info(f"🚚 Carriers disponibles para {codigo_postal}: {len(carriers_list)}")
        return carriers_list

    def calculate_external_cost(self, carrier_info: Dict[str, Any],
                                peso_kg: float, distancia_km: float) -> float:
        """💰 Calcula costo de flota externa"""
        costo_base = carrier_info['costo_base_mxn']
        peso_extra = max(0, peso_kg - carrier_info['peso_min_kg'])
        costo_peso_extra = peso_extra * carrier_info['costo_por_kg_adicional']

        # Factor por distancia (simplificado)
        factor_distancia = 1.0 + (distancia_km / 1000) * 0.1

        costo_total = (costo_base + costo_peso_extra) * factor_distancia

        return round(costo_total, 2)

    def get_delivery_time_for_carrier(self, carrier_info: Dict[str, Any]) -> tuple:
        """⏱️ Obtiene tiempo de entrega del carrier"""
        tiempo_str = carrier_info.get('tiempo_entrega_dias_habiles', '3-5')

        try:
            if '-' in tiempo_str:
                min_days, max_days = map(int, tiempo_str.split('-'))
            else:
                min_days = max_days = int(tiempo_str)

            # Convertir días a horas (asumiendo días hábiles)
            min_hours = min_days * 24
            max_hours = max_days * 24

            return min_hours, max_hours
        except (ValueError, AttributeError):
            # Fallback: 3-5 días
            return 72.0, 120.0