import re
import time
from abc import ABC
from datetime import datetime
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

        # Convertir a string para limpiar
        str_value = str(value).strip()

        # Remover paréntesis, espacios y caracteres especiales
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

        # Remover paréntesis al inicio y caracteres especiales
        cleaned = re.sub(r'^\([^)]*\)', '', str_value).strip()
        cleaned = re.sub(r'[^\w\-]', '', cleaned)

        return cleaned if len(cleaned) >= 2 else ""


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
        """🔍 Busca tienda por ID con limpieza robusta"""
        df = self.load_data()

        # Limpiar el ID buscado
        clean_search_id = self._clean_id_value(tienda_id)

        for store in df.to_dicts():
            try:
                # Limpiar ID de la tienda en el CSV
                store_id_raw = store.get('tienda_id', '')
                store_id_clean = self._clean_id_value(store_id_raw)

                if store_id_clean == clean_search_id and store_id_clean:
                    # Limpiar coordenadas
                    store['latitud'] = self._clean_coordinate_value(store.get('latitud'))
                    store['longitud'] = self._clean_coordinate_value(store.get('longitud'))
                    store['tienda_id'] = store_id_clean  # Usar ID limpio

                    # Validar coordenadas para México
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
                                         max_distance_km: float = 50.0) -> List[Dict[str, Any]]:
        """📍 Encuentra tiendas cercanas por código postal"""
        from utils.geo_calculator import GeoCalculator

        # Obtener coordenadas del CP destino
        postal_repo = PostalCodeRepository(self.data_dir)
        cp_info = postal_repo.get_postal_code_info(codigo_postal)

        if not cp_info:
            logger.warning(f"❌ CP no encontrado: {codigo_postal}")
            return []

        cp_lat = cp_info.get('latitud_centro', 19.4326)  # Default CDMX
        cp_lon = cp_info.get('longitud_centro', -99.1332)

        # Calcular distancias a todas las tiendas
        df = self.load_data()
        stores_list = []

        for store in df.to_dicts():
            try:
                # Limpiar tienda_id
                tienda_id_raw = store.get('tienda_id', '')
                tienda_id_clean = self._clean_id_value(tienda_id_raw)

                if not tienda_id_clean:
                    continue

                # Limpiar coordenadas
                store_lat = self._clean_coordinate_value(store.get('latitud'))
                store_lon = self._clean_coordinate_value(store.get('longitud'))

                # Validar coordenadas
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
                    stores_list.append(store_clean)

            except Exception as e:
                logger.warning(f"❌ Error procesando tienda {tienda_id_raw}: {e}")
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
        """🔍 Busca CEDIS por ID con limpieza robusta"""
        df = self.load_data()

        # Limpiar el ID buscado
        clean_search_id = self._clean_id_value(cedis_id)

        for cedis in df.to_dicts():
            try:
                # Limpiar ID del CEDIS
                cedis_id_raw = cedis.get('cedis_id', '')
                cedis_id_clean = self._clean_id_value(cedis_id_raw)

                if cedis_id_clean == clean_search_id and cedis_id_clean:
                    # Limpiar coordenadas
                    cedis['latitud'] = self._clean_coordinate_value(cedis.get('latitud'))
                    cedis['longitud'] = self._clean_coordinate_value(cedis.get('longitud'))
                    cedis['cedis_id'] = cedis_id_clean

                    # Validar coordenadas para México
                    if 14.0 <= cedis['latitud'] <= 33.0 and -118.0 <= cedis['longitud'] <= -86.0:
                        return cedis

            except Exception as e:
                logger.warning(f"❌ Error procesando CEDIS {cedis_id_raw}: {e}")
                continue

        logger.warning(f"❌ CEDIS no encontrado: {cedis_id}")
        return None

    def find_cedis_for_coverage(self, estado_destino: str) -> List[Dict[str, Any]]:
        """🗺️ Encuentra CEDIS que cubren un estado"""
        df = self.load_data()
        cedis_list = []

        for cedis in df.to_dicts():
            try:
                # Limpiar cedis_id
                cedis_id_raw = cedis.get('cedis_id', '')
                cedis_id_clean = self._clean_id_value(cedis_id_raw)

                if not cedis_id_clean:
                    continue

                cobertura = str(cedis.get('cobertura_estados', ''))

                if (estado_destino.lower() in cobertura.lower() or
                        'nacional' in cobertura.lower() or
                        'todos' in cobertura.lower()):

                    # Limpiar coordenadas
                    lat = self._clean_coordinate_value(cedis.get('latitud'))
                    lon = self._clean_coordinate_value(cedis.get('longitud'))

                    # Validar coordenadas para México
                    if 14.0 <= lat <= 33.0 and -118.0 <= lon <= -86.0:
                        cedis_clean = cedis.copy()
                        cedis_clean['cedis_id'] = cedis_id_clean
                        cedis_clean['latitud'] = lat
                        cedis_clean['longitud'] = lon
                        cedis_list.append(cedis_clean)

            except Exception as e:
                logger.warning(f"❌ Error procesando CEDIS {cedis_id_raw}: {e}")
                continue

        logger.info(f"🏭 Encontrados {len(cedis_list)} CEDIS para {estado_destino}")
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
            # Limpiar tienda_id
            tienda_id_raw = location.get('tienda_id', '')
            tienda_id_clean = self._clean_id_value(tienda_id_raw)

            if tienda_id_clean and location['stock_disponible'] >= settings.MIN_STOCK_THRESHOLD:
                location['tienda_id'] = tienda_id_clean
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
                'split_plan': [],
                'cantidad_cubierta': 0,
                'cantidad_faltante': cantidad_requerida
            }

        # Crear diccionario de ubicaciones cercanas por ID limpio
        ubicaciones_dict = {}
        for loc in ubicaciones_cercanas:
            clean_id = self._clean_id_value(loc.get('tienda_id', ''))
            if clean_id:
                ubicaciones_dict[clean_id] = loc

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
        cantidad_cubierta = cantidad_requerida - cantidad_pendiente

        logger.info(f"🔄 Split calculado para {sku_id}: "
                    f"{len(split_plan)} ubicaciones, factible: {es_factible}")

        return {
            'es_factible': es_factible,
            'cantidad_cubierta': cantidad_cubierta,
            'cantidad_faltante': cantidad_pendiente,
            'split_plan': split_plan,
            'razon': 'Split exitoso' if es_factible else f'Stock insuficiente (faltan {cantidad_pendiente})'
        }


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

        # Buscar en rangos
        for row in df.to_dicts():
            try:
                rango_cp = str(row.get('rango_cp', '')).strip()

                # Limpiar el rango_cp de caracteres problemáticos
                rango_cp_clean = self._clean_id_value(rango_cp)

                # Buscar por rango o CP exacto
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

        # Si no se encuentra, crear entrada por defecto para CDMX
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

        # Limpiar coordenadas
        processed['latitud_centro'] = self._clean_coordinate_value(row.get('latitud_centro'))
        processed['longitud_centro'] = self._clean_coordinate_value(row.get('longitud_centro'))

        # Si las coordenadas están fuera de rango, usar CDMX como fallback
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
            return False  # Si no conocemos el CP, asumimos zona segura

        zona_seguridad = str(cp_info.get('zona_seguridad', '')).lower()
        return zona_seguridad in ['roja', 'alta', 'crítica']


class ClimateRepository(BaseRepository):
    """🌤️ Repositorio de datos climáticos"""

    def load_data(self) -> pl.DataFrame:
        return self._load_with_cache(settings.CSV_FILES['clima'])

    def get_climate_by_postal_code(self, codigo_postal: str,
                                   fecha: datetime) -> Dict[str, Any]:
        """🌡️ Obtiene datos climáticos por CP y fecha"""
        df = self.load_data()

        try:
            cp_int = int(codigo_postal)
        except ValueError:
            return self._get_default_climate(fecha)

        # Buscar en rangos de CP
        climate_data = df.filter(
            (cp_int >= pl.col('rango_cp_inicio')) &
            (cp_int <= pl.col('rango_cp_fin'))
        )

        if climate_data.height == 0:
            return self._get_default_climate(fecha)

        clima_dict = climate_data.to_dicts()[0]
        return self._process_climate_data(clima_dict, fecha)

    def _get_default_climate(self, fecha: datetime) -> Dict[str, Any]:
        """🌡️ Clima por defecto"""
        return {
            'clima_actual': 'Templado',
            'temperatura': 22,
            'precipitacion_mm': 30,
            'factor_clima': 1.0
        }

    def _process_climate_data(self, clima_dict: Dict[str, Any], fecha: datetime) -> Dict[str, Any]:
        """🌡️ Procesa datos climáticos"""
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

        detected_factors = TemporalFactorDetector.detect_comprehensive_factors(fecha, codigo_postal)

        logger.info(f"🤖 Factores auto-generados para {fecha.date()}")

        return {
            'fecha': fecha.date().isoformat(),
            'evento_detectado': detected_factors['eventos_detectados'][0]
            if detected_factors['eventos_detectados'] else 'Normal',
            'factor_demanda': detected_factors['factor_demanda'],
            'condicion_clima': detected_factors['condicion_clima'],
            'trafico_nivel': detected_factors['trafico_nivel'],
            'rango_cp_afectado': codigo_postal[:2] + '000-' + codigo_postal[:2] + '999',
            'impacto_tiempo_extra_horas': detected_factors['impacto_tiempo_extra_horas'],
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

        try:
            cp_int = int(codigo_postal)
        except ValueError:
            return []

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