import polars as pl
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from abc import ABC, abstractmethod
from datetime import datetime
import time

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

        # Validaciones adicionales
        if product_dict.get('peso_kg', 0) > 30:
            logger.warning(f"⚠️ Producto pesado detectado: {product_dict['peso_kg']}kg")

        if product_dict.get('es_fragil', False):
            logger.info(f"🔸 Producto frágil detectado: {sku_id}")

        return product_dict

    def get_products_by_category(self, categoria: str) -> List[Dict[str, Any]]:
        """📑 Obtiene productos por categoría"""
        df = self.load_data()
        return df.filter(pl.col('categoria') == categoria).to_dicts()

    def check_seasonal_availability(self, sku_id: str, fecha: datetime) -> bool:
        """🗓️ Verifica disponibilidad estacional"""
        product = self.get_product_by_sku(sku_id)
        if not product:
            return False

        # Lógica de temporada basada en el campo 'temporada'
        temporada_producto = product.get('temporada', 'Todo_Año')
        mes_actual = fecha.month

        if temporada_producto == 'Todo_Año':
            return True
        elif temporada_producto == 'Invierno' and mes_actual in [12, 1, 2]:
            return True
        elif temporada_producto == 'Verano' and mes_actual in [6, 7, 8]:
            return True
        # Agregar más lógica según necesidades

        return temporada_producto == 'Todo_Año'


class StoreRepository(BaseRepository):
    """🏪 Repositorio de tiendas Liverpool"""

    def load_data(self) -> pl.DataFrame:
        return self._load_with_cache(settings.CSV_FILES['tiendas'])

    def get_store_by_id(self, tienda_id: str) -> Optional[Dict[str, Any]]:
        """🔍 Busca tienda por ID"""
        df = self.load_data()
        result = df.filter(pl.col('tienda_id') == tienda_id)
        return result.to_dicts()[0] if result.height > 0 else None

    def find_stores_by_postal_code_range(self, codigo_postal: str,
                                         max_distance_km: float = 50.0) -> List[Dict[str, Any]]:
        """📍 Encuentra tiendas cercanas por código postal"""
        from utils.geo_calculator import GeoCalculator

        # Obtener coordenadas del CP destino
        postal_repo = PostalCodeRepository(self.data_dir)
        cp_info = postal_repo.get_postal_code_info(codigo_postal)

        if not cp_info:
            logger.warning(f"❌ CP no encontrado: {codigo_postal}")
            return []

        cp_lat = cp_info.get('latitud_centro', cp_info.get('latitud', 19.4326))
        cp_lon = cp_info.get('longitud_centro', cp_info.get('longitud', -99.1332))

        # Calcular distancias a todas las tiendas
        df = self.load_data()
        stores_list = []

        for store in df.to_dicts():
            try:
                # Verificar que tienda_id sea válido
                tienda_id = str(store.get('tienda_id', ''))
                if not tienda_id or tienda_id.startswith('(') or len(tienda_id) < 2:
                    logger.warning(f"⚠️ Tienda ID inválido: {tienda_id}")
                    continue

                # Limpiar coordenadas de la tienda
                store_lat = store.get('latitud', 0)
                store_lon = store.get('longitud', 0)

                # Si las coordenadas están como string con formato problemático
                if isinstance(store_lat, str):
                    store_lat_clean = store_lat.replace('(', '').replace(')', '').strip()
                    if store_lat_clean and store_lat_clean.replace('.', '').replace('-', '').isdigit():
                        store_lat = float(store_lat_clean)
                    else:
                        continue

                if isinstance(store_lon, str):
                    store_lon_clean = store_lon.replace('(', '').replace(')', '').strip()
                    if store_lon_clean and store_lon_clean.replace('.', '').replace('-', '').isdigit():
                        store_lon = float(store_lon_clean)
                    else:
                        continue

                if store_lat == 0 and store_lon == 0:
                    continue  # Skip tiendas sin coordenadas válidas

                # Validar rangos de coordenadas para México
                if not (14.0 <= store_lat <= 33.0 and -118.0 <= store_lon <= -86.0):
                    logger.warning(f"⚠️ Coordenadas fuera de México: lat={store_lat}, lon={store_lon}")
                    continue

                distance = GeoCalculator.calculate_distance_km(
                    cp_lat, cp_lon, store_lat, store_lon
                )

                if distance <= max_distance_km:
                    store['distancia_km'] = distance
                    store['latitud'] = store_lat  # Coordenadas limpias
                    store['longitud'] = store_lon
                    stores_list.append(store)

            except (ValueError, TypeError) as e:
                logger.warning(f"❌ Error procesando tienda {store.get('tienda_id', 'unknown')}: {e}")
                continue

        # Ordenar por distancia
        stores_list.sort(key=lambda x: x['distancia_km'])

        logger.info(f"📍 Encontradas {len(stores_list)} tiendas cercanas a {codigo_postal}")
        return stores_list


class CEDISRepository(BaseRepository):
    """🏭 Repositorio de CEDIS"""

    def load_data(self) -> pl.DataFrame:
        return self._load_with_cache(settings.CSV_FILES['cedis'])

    def get_cedis_by_id(self, cedis_id: str) -> Optional[Dict[str, Any]]:
        """🔍 Busca CEDIS por ID"""
        df = self.load_data()

        for cedis in df.to_dicts():
            try:
                current_id = str(cedis.get('cedis_id', ''))
                if current_id == cedis_id:
                    # Validar y limpiar coordenadas
                    lat = cedis.get('latitud', 0)
                    lon = cedis.get('longitud', 0)

                    if isinstance(lat, str):
                        lat = float(lat.replace('(', '').replace(')', '').strip())
                    if isinstance(lon, str):
                        lon = float(lon.replace('(', '').replace(')', '').strip())

                    # Validar rangos para México
                    if 14.0 <= lat <= 33.0 and -118.0 <= lon <= -86.0:
                        cedis['latitud'] = lat
                        cedis['longitud'] = lon
                        return cedis

            except (ValueError, TypeError):
                continue

        return None

    def find_cedis_for_coverage(self, estado_destino: str) -> List[Dict[str, Any]]:
        """🗺️ Encuentra CEDIS que cubren un estado"""
        df = self.load_data()

        # Filtrar CEDIS que cubren el estado o todos los estados
        cedis_list = []

        for cedis in df.to_dicts():
            try:
                # Verificar que cedis_id sea válido
                cedis_id = str(cedis.get('cedis_id', ''))
                if not cedis_id or cedis_id.startswith('(') or len(cedis_id) < 2:
                    continue

                cobertura = str(cedis.get('cobertura_estados', ''))

                if (estado_destino.lower() in cobertura.lower() or
                        'nacional' in cobertura.lower() or
                        'todos' in cobertura.lower()):

                    # Validar coordenadas
                    lat = cedis.get('latitud', 0)
                    lon = cedis.get('longitud', 0)

                    # Limpiar coordenadas si están mal formateadas
                    if isinstance(lat, str):
                        lat_clean = lat.replace('(', '').replace(')', '').strip()
                        if lat_clean and lat_clean.replace('.', '').replace('-', '').isdigit():
                            lat = float(lat_clean)
                        else:
                            continue

                    if isinstance(lon, str):
                        lon_clean = lon.replace('(', '').replace(')', '').strip()
                        if lon_clean and lon_clean.replace('.', '').replace('-', '').isdigit():
                            lon = float(lon_clean)
                        else:
                            continue

                    # Validar rangos para México
                    if 14.0 <= lat <= 33.0 and -118.0 <= lon <= -86.0:
                        cedis['latitud'] = lat
                        cedis['longitud'] = lon
                        cedis_list.append(cedis)

            except (ValueError, TypeError) as e:
                logger.warning(f"❌ Error procesando CEDIS {cedis.get('cedis_id', 'unknown')}: {e}")
                continue

        return cedis_list


class StockRepository(BaseRepository):
    """📦 Repositorio de inventarios con split avanzado"""

    def load_data(self) -> pl.DataFrame:
        return self._load_with_cache(settings.CSV_FILES['stock'])

    def get_stock_locations(self, sku_id: str, cantidad_requerida: int) -> List[Dict[str, Any]]:
        """📋 Obtiene ubicaciones con stock para un SKU"""
        df = self.load_data()

        # Filtrar por SKU y stock disponible
        stock_df = df.filter(
            (pl.col('sku_id') == sku_id) &
            (pl.col('stock_disponible') > 0)
        ).sort('stock_disponible', descending=True)

        stock_locations = []
        for location in stock_df.to_dicts():
            if location['stock_disponible'] >= settings.MIN_STOCK_THRESHOLD:
                stock_locations.append(location)

        logger.info(f"📦 Stock disponible para {sku_id}: {len(stock_locations)} ubicaciones")
        return stock_locations

    def calculate_split_inventory(self, sku_id: str, cantidad_requerida: int,
                                  ubicaciones_cercanas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """🔄 Calcula split óptimo de inventario"""
        stock_locations = self.get_stock_locations(sku_id, cantidad_requerida)

        if not stock_locations:
            return {
                'es_factible': False,
                'razon': 'Sin stock disponible',
                'split_plan': []
            }

        # Ordenar ubicaciones por cercanía al destino
        ubicaciones_dict = {loc['tienda_id']: loc for loc in ubicaciones_cercanas}

        split_plan = []
        cantidad_pendiente = cantidad_requerida

        for stock_loc in stock_locations:
            if cantidad_pendiente <= 0:
                break

            tienda_id = stock_loc['tienda_id']
            if tienda_id not in ubicaciones_dict:
                continue  # Skip ubicaciones muy lejanas

            cantidad_a_tomar = min(stock_loc['stock_disponible'], cantidad_pendiente)

            split_plan.append({
                'tienda_id': tienda_id,
                'cantidad': cantidad_a_tomar,
                'stock_disponible': stock_loc['stock_disponible'],
                'distancia_km': ubicaciones_dict[tienda_id]['distancia_km'],
                'prioridad': len(split_plan) + 1
            })

            cantidad_pendiente -= cantidad_a_tomar

            # Limitar a máximo ubicaciones por configuración
            if len(split_plan) >= settings.MAX_SPLIT_LOCATIONS:
                break

        es_factible = cantidad_pendiente <= 0

        logger.info(f"🔄 Split calculado para {sku_id}: "
                    f"{len(split_plan)} ubicaciones, factible: {es_factible}")

        return {
            'es_factible': es_factible,
            'cantidad_cubierta': cantidad_requerida - cantidad_pendiente,
            'cantidad_faltante': cantidad_pendiente,
            'split_plan': split_plan,
            'razon': 'Split exitoso' if es_factible else 'Stock insuficiente'
        }


class PostalCodeRepository(BaseRepository):
    """📮 Repositorio de códigos postales y rangos"""

    def load_data(self) -> pl.DataFrame:
        return self._load_with_cache(settings.CSV_FILES['codigos_postales'])

    def get_postal_code_info(self, codigo_postal: str) -> Optional[Dict[str, Any]]:
        """🔍 Obtiene información de código postal"""
        df = self.load_data()

        try:
            cp_int = int(codigo_postal)

            # El CSV tiene "rango_cp", no "codigo_postal"
            # Buscar en rango_cp que contenga el código postal
            result = None

            # Intentar buscar por rango_cp que contenga el CP
            for row in df.to_dicts():
                rango_cp = str(row.get('rango_cp', ''))

                # Limpiar el rango_cp de caracteres problemáticos
                rango_cp_clean = rango_cp.replace('(', '').replace(')', '').strip()

                # Verificar si el CP está en el rango
                if '-' in rango_cp_clean:
                    try:
                        parts = rango_cp_clean.split('-')
                        if len(parts) == 2:
                            start_part = parts[0].strip()
                            end_part = parts[1].strip()

                            # Validar que las partes sean números válidos
                            if (start_part.isdigit() and end_part.isdigit() and
                                    len(start_part) <= 5 and len(end_part) <= 5):

                                start_cp = int(start_part)
                                end_cp = int(end_part)

                                if start_cp <= cp_int <= end_cp:
                                    result = row
                                    break
                    except (ValueError, IndexError):
                        continue
                elif rango_cp_clean.isdigit() and len(rango_cp_clean) <= 5:
                    if int(rango_cp_clean) == cp_int:
                        result = row
                        break

            if result:
                # Limpiar coordenadas si tienen formato problemático
                for coord_field in ['latitud', 'longitud', 'latitud_centro', 'longitud_centro']:
                    if coord_field in result and result[coord_field]:
                        coord_value = str(result[coord_field])
                        # Limpiar paréntesis, espacios y caracteres extraños
                        coord_clean = coord_value.replace('(', '').replace(')', '').replace(' ', '').strip()
                        try:
                            if coord_clean and coord_clean.replace('.', '').replace('-', '').isdigit():
                                coord_float = float(coord_clean)
                                # Validar rango de coordenadas para México
                                if coord_field.startswith('latitud') and 14.0 <= coord_float <= 33.0:
                                    result[coord_field] = coord_float
                                elif coord_field.startswith('longitud') and -118.0 <= coord_float <= -86.0:
                                    result[coord_field] = coord_float
                                else:
                                    raise ValueError("Coordenada fuera de rango México")
                            else:
                                raise ValueError("Formato de coordenada inválido")
                        except (ValueError, TypeError):
                            logger.warning(f"⚠️ Coordenada inválida en {coord_field}: {coord_value}")
                            # Usar coordenadas por defecto para CDMX si falla
                            if 'latitud' in coord_field:
                                result[coord_field] = 19.4326
                            else:
                                result[coord_field] = -99.1332

                return result

        except ValueError as e:
            logger.warning(f"❌ Error procesando CP {codigo_postal}: {e}")

        logger.warning(f"❌ Información de CP no encontrada: {codigo_postal}")
        return None

    def is_zona_roja(self, codigo_postal: str) -> bool:
        """🚨 Detecta zona roja"""
        cp_info = self.get_postal_code_info(codigo_postal)
        if not cp_info:
            return True  # Precaución si no conocemos el CP

        zona_seguridad = cp_info.get('zona_seguridad', '').lower()
        return zona_seguridad in ['roja', 'alta', 'crítica']


class ClimateRepository(BaseRepository):
    """🌤️ Repositorio de datos climáticos"""

    def load_data(self) -> pl.DataFrame:
        return self._load_with_cache(settings.CSV_FILES['clima'])

    def get_climate_by_postal_code(self, codigo_postal: str,
                                   fecha: datetime) -> Dict[str, Any]:
        """🌡️ Obtiene datos climáticos por CP y fecha"""
        df = self.load_data()
        cp_int = int(codigo_postal)

        # Buscar en rangos de CP
        climate_data = df.filter(
            (cp_int >= pl.col('rango_cp_inicio')) &
            (cp_int <= pl.col('rango_cp_fin'))
        )

        if climate_data.height == 0:
            # Datos por defecto si no se encuentra
            return {
                'clima_actual': 'Templado',
                'temperatura': 22,
                'precipitacion_mm': 30,
                'factor_clima': 1.0
            }

        clima_dict = climate_data.to_dicts()[0]

        # Determinar temporada
        mes = fecha.month
        if mes in [12, 1, 2]:
            temporada = 'invierno'
        elif mes in [3, 4, 5]:
            temporada = 'primavera'
        elif mes in [6, 7, 8]:
            temporada = 'verano'
        else:
            temporada = 'otoño'

        clima_temporada = clima_dict.get(f'clima_{temporada}', 'Templado')

        return {
            'clima_actual': clima_temporada,
            'temperatura': clima_dict.get(f'temperatura_min_{temporada}', 22),
            'precipitacion_mm': clima_dict.get('precipitacion_anual_mm', 800) // 12,
            'factor_clima': 1.2 if 'Lluvioso' in clima_temporada else 1.0,
            'factores_especiales': clima_dict.get('factores_especiales', '')
        }


class ExternalFactorsRepository(BaseRepository):
    """🌐 Repositorio de factores externos"""

    def load_data(self) -> pl.DataFrame:
        return self._load_with_cache(settings.CSV_FILES['factores_externos'])

    def get_factors_by_date_and_region(self, fecha: datetime,
                                       codigo_postal: str) -> Dict[str, Any]:
        """📅 Obtiene factores externos por fecha y región"""
        df = self.load_data()
        fecha_str = fecha.date().isoformat()

        # Buscar por fecha exacta
        exact_match = df.filter(pl.col('fecha') == fecha_str)

        if exact_match.height > 0:
            factors = exact_match.to_dicts()[0]
            logger.info(f"📅 Factores encontrados para {fecha_str}")
            return factors

        # Si no hay datos exactos, generar automáticamente
        return self._generate_auto_factors(fecha, codigo_postal)

    def _generate_auto_factors(self, fecha: datetime,
                               codigo_postal: str) -> Dict[str, Any]:
        """🤖 Genera factores automáticamente si no hay datos"""
        from utils.temporal_detector import TemporalFactorDetector

        detected_factors = TemporalFactorDetector.detect_seasonal_factors(fecha)

        logger.info(f"🤖 Factores auto-generados para {fecha.date()}")

        return {
            'fecha': fecha.date().isoformat(),
            'evento_detectado': detected_factors['eventos_detectados'][0]
            if detected_factors['eventos_detectados'] else 'Normal',
            'factor_demanda': detected_factors['factor_demanda'],
            'condicion_clima': detected_factors['condicion_clima'],
            'trafico_nivel': detected_factors['trafico_nivel'],
            'rango_cp_afectado': codigo_postal[:2] + '000-' + codigo_postal[:2] + '999',
            'impacto_tiempo_extra_horas': detected_factors['impacto_tiempo_extra'],
            'criticidad_logistica': detected_factors['criticidad_logistica']
        }


class ExternalFleetRepository(BaseRepository):
    """🚚 Repositorio de flota externa"""

    def load_data(self) -> pl.DataFrame:
        return self._load_with_cache(settings.CSV_FILES['flota_externa'])

    def get_available_carriers(self, codigo_postal: str,
                               peso_kg: float) -> List[Dict[str, Any]]:
        """📋 Obtiene carriers disponibles para CP y peso"""
        df = self.load_data()
        cp_int = int(codigo_postal)

        # Filtrar carriers activos que cubren la zona y peso
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