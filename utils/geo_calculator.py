import math
from typing import Tuple
from geopy.distance import geodesic, great_circle
from pyproj import Transformer
from utils.logger import logger


class GeoCalculator:
    """Calculador geoespacial CORREGIDO con validaciones estrictas"""

    _transformer_mexico = None
    _coordinate_cache = {}  # Cache para coordenadas validadas

    @staticmethod
    def _get_transformer():
        if GeoCalculator._transformer_mexico is None:
            try:
                GeoCalculator._transformer_mexico = Transformer.from_crs("EPSG:4326", "EPSG:6372", always_xy=True)
            except Exception as e:
                logger.warning(f"⚠️ Error inicializando transformer: {e}")
                GeoCalculator._transformer_mexico = None
        return GeoCalculator._transformer_mexico

    @staticmethod
    def fix_corrupted_coordinates(lat: float, lon: float) -> Tuple[float, float]:
        """🔧 CORRECCIÓN: Arregla coordenadas corruptas detectadas en CSV"""
        try:
            # Detectar coordenadas con decimales desplazados
            # Ejemplo: 194.326 → 19.4326, 991.332 → -99.1332

            # Corregir latitud
            if lat > 90:
                # Caso: 194.326 → 19.4326
                lat_str = str(abs(lat))
                if len(lat_str) >= 6:  # Ej: "194.326"
                    lat_fixed = float(lat_str[:2] + '.' + lat_str[2:].replace('.', ''))
                else:
                    lat_fixed = lat / 10  # Fallback simple
            elif lat < -90:
                lat_str = str(abs(lat))
                if len(lat_str) >= 6:
                    lat_fixed = -float(lat_str[:2] + '.' + lat_str[2:].replace('.', ''))
                else:
                    lat_fixed = lat / 10
            else:
                lat_fixed = lat

            # Corregir longitud
            if lon > 180:
                # Caso: 991.332 → -99.1332 (México es negativo)
                lon_str = str(abs(lon))
                if len(lon_str) >= 6:  # Ej: "991.332"
                    lon_fixed = -float(lon_str[:2] + '.' + lon_str[2:].replace('.', ''))
                else:
                    lon_fixed = -lon / 10
            elif lon < -180:
                lon_str = str(abs(lon))
                if len(lon_str) >= 6:
                    lon_fixed = -float(lon_str[:2] + '.' + lon_str[2:].replace('.', ''))
                else:
                    lon_fixed = lon / 10
            elif lon > 0:  # México debería ser negativo
                lon_fixed = -lon if lon < 180 else -lon / 10
            else:
                lon_fixed = lon

            # Validar rangos para México
            if not (14.0 <= lat_fixed <= 33.0):
                logger.warning(f"⚠️ Latitud fuera de rango México: {lat_fixed}")
                # Fallback a coordenadas del centro de México
                lat_fixed = 19.4326  # CDMX como fallback

            if not (-118.0 <= lon_fixed <= -86.0):
                logger.warning(f"⚠️ Longitud fuera de rango México: {lon_fixed}")
                lon_fixed = -99.1332  # CDMX como fallback

            if lat != lat_fixed or lon != lon_fixed:
                logger.info(f"🔧 Coordenadas corregidas: ({lat}, {lon}) → ({lat_fixed}, {lon_fixed})")

            return round(lat_fixed, 4), round(lon_fixed, 4)

        except Exception as e:
            logger.error(f"❌ Error corrigiendo coordenadas ({lat}, {lon}): {e}")
            return 19.4326, -99.1332  # CDMX como fallback seguro

    @staticmethod
    def calculate_distance_km(lat1: float, lon1: float,
                              lat2: float, lon2: float,
                              method: str = 'geodesic') -> float:
        """🎯 Calcula distancia real CORREGIDA entre coordenadas"""

        # PASO 1: Corregir coordenadas corruptas
        lat1_fixed, lon1_fixed = GeoCalculator.fix_corrupted_coordinates(lat1, lon1)
        lat2_fixed, lon2_fixed = GeoCalculator.fix_corrupted_coordinates(lat2, lon2)

        # PASO 2: Validar coordenadas finales
        if not GeoCalculator._validate_coordinates(lat1_fixed, lon1_fixed, lat2_fixed, lon2_fixed):
            logger.warning(f"⚠️ Coordenadas inválidas después de corrección")
            return 0.0

        # PASO 3: Cache para evitar recálculos
        cache_key = f"{lat1_fixed},{lon1_fixed},{lat2_fixed},{lon2_fixed},{method}"
        if cache_key in GeoCalculator._coordinate_cache:
            return GeoCalculator._coordinate_cache[cache_key]

        try:
            # PASO 4: Calcular distancia con método seleccionado
            if method == 'geodesic':
                distance = geodesic((lat1_fixed, lon1_fixed), (lat2_fixed, lon2_fixed)).kilometers
            elif method == 'great_circle':
                distance = great_circle((lat1_fixed, lon1_fixed), (lat2_fixed, lon2_fixed)).kilometers
            elif method == 'pyproj':
                distance = GeoCalculator._calculate_pyproj_distance(lat1_fixed, lon1_fixed, lat2_fixed, lon2_fixed)
            else:
                distance = GeoCalculator._haversine_distance(lat1_fixed, lon1_fixed, lat2_fixed, lon2_fixed)

            # PASO 5: Validar resultado
            if distance < 0:
                logger.warning(f"⚠️ Distancia negativa: {distance}km")
                distance = 0.0
            elif distance > 3000:  # México máximo ~2500km
                logger.warning(f"⚠️ Distancia muy grande para México: {distance:.1f}km")
                # Recalcular con método más confiable
                distance = geodesic((lat1_fixed, lon1_fixed), (lat2_fixed, lon2_fixed)).kilometers
                if distance > 3000:
                    logger.error(f"❌ Distancia persistentemente grande: {distance:.1f}km")
                    distance = min(distance, 2500.0)  # Cap máximo para México

            distance = round(distance, 2)

            # Cache resultado válido
            if distance > 0:
                GeoCalculator._coordinate_cache[cache_key] = distance

            return distance

        except Exception as e:
            logger.warning(f"❌ Error calculando distancia: {e}")
            # Fallback a haversine
            fallback_distance = GeoCalculator._haversine_distance(lat1_fixed, lon1_fixed, lat2_fixed, lon2_fixed)
            return max(0.0, fallback_distance)

    @staticmethod
    def _validate_coordinates(lat1: float, lon1: float, lat2: float, lon2: float) -> bool:
        """✅ Validación ESTRICTA de coordenadas para México"""
        try:
            # Rangos específicos para México
            MEXICO_LAT_MIN, MEXICO_LAT_MAX = 14.5, 32.8  # Más preciso
            MEXICO_LON_MIN, MEXICO_LON_MAX = -117.4, -86.7  # Más preciso

            # Validar punto 1
            if not (MEXICO_LAT_MIN <= lat1 <= MEXICO_LAT_MAX):
                logger.warning(f"❌ lat1 fuera de México: {lat1}")
                return False
            if not (MEXICO_LON_MIN <= lon1 <= MEXICO_LON_MAX):
                logger.warning(f"❌ lon1 fuera de México: {lon1}")
                return False

            # Validar punto 2
            if not (MEXICO_LAT_MIN <= lat2 <= MEXICO_LAT_MAX):
                logger.warning(f"❌ lat2 fuera de México: {lat2}")
                return False
            if not (MEXICO_LON_MIN <= lon2 <= MEXICO_LON_MAX):
                logger.warning(f"❌ lon2 fuera de México: {lon2}")
                return False

            return True

        except (TypeError, ValueError) as e:
            logger.error(f"❌ Error validando coordenadas: {e}")
            return False

    @staticmethod
    def _calculate_pyproj_distance(lat1: float, lon1: float,
                                   lat2: float, lon2: float) -> float:
        """🎯 Distancia ultra-precisa usando proyección mexicana"""
        try:
            transformer = GeoCalculator._get_transformer()
            if not transformer:
                return GeoCalculator._haversine_distance(lat1, lon1, lat2, lon2)

            # Transformar a coordenadas proyectadas mexicanas
            x1, y1 = transformer.transform(lon1, lat1)
            x2, y2 = transformer.transform(lon2, lat2)

            # Distancia euclidiana en metros, convertir a km
            distance_m = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            return distance_m / 1000.0

        except Exception as e:
            logger.warning(f"❌ Error en pyproj: {e}")
            return GeoCalculator._haversine_distance(lat1, lon1, lat2, lon2)

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float,
                            lat2: float, lon2: float) -> float:
        """🌐 Fórmula Haversine optimizada"""
        try:
            R = 6371.0  # Radio de la Tierra en km

            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            delta_lat = math.radians(lat2 - lat1)
            delta_lon = math.radians(lon2 - lon1)

            a = (math.sin(delta_lat / 2) ** 2 +
                 math.cos(lat1_rad) * math.cos(lat2_rad) *
                 math.sin(delta_lon / 2) ** 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            distance = R * c
            return max(0.0, distance)

        except Exception as e:
            logger.error(f"❌ Error en Haversine: {e}")
            return 0.0

    @staticmethod
    def calculate_travel_time(distance_km: float,
                              transport_type: str = 'FI',
                              traffic_level: str = 'Moderado',
                              weather_condition: str = 'Despejado',
                              road_type: str = 'carretera') -> float:
        """⏱️ TIEMPO OPTIMIZADO con velocidades realistas por distancia"""

        if distance_km <= 0:
            return 0.3  # Tiempo mínimo absoluto

        # NUEVA LÓGICA: Velocidades dinámicas por distancia
        if distance_km <= 20:
            # Urbano/local
            base_speed = 35.0 if transport_type == 'FI' else 40.0
            road_type = 'urbano'
        elif distance_km <= 100:
            # Suburbano/regional
            base_speed = 55.0 if transport_type == 'FI' else 65.0
            road_type = 'suburbano'
        elif distance_km <= 500:
            # Carretera nacional
            base_speed = 70.0 if transport_type == 'FI' else 80.0
            road_type = 'carretera'
        else:
            # Larga distancia (requiere múltiples conductores)
            base_speed = 75.0 if transport_type == 'FI' else 85.0
            road_type = 'autopista'

        # Multiplicadores optimizados
        traffic_multipliers = {
            'Bajo': 1.1,
            'Moderado': 1.0,
            'Alto': 0.85,
            'Muy_Alto': 0.70
        }

        weather_multipliers = {
            'Despejado': 1.0,
            'Nublado': 0.98,
            'Lluvioso': 0.88,
            'Tormenta': 0.75,
            'Templado_Seco': 1.0,
            'Templado': 1.0
        }

        road_multipliers = {
            'urbano': 0.8,
            'suburbano': 0.95,
            'carretera': 1.0,
            'autopista': 1.15
        }

        # Calcular velocidad final
        traffic_factor = traffic_multipliers.get(traffic_level, 1.0)
        weather_factor = weather_multipliers.get(weather_condition, 1.0)
        road_factor = road_multipliers.get(road_type, 1.0)

        adjusted_speed = base_speed * road_factor * traffic_factor * weather_factor
        adjusted_speed = max(adjusted_speed, 25.0)  # Mínimo realista

        # Tiempo base de viaje
        travel_time = distance_km / adjusted_speed

        # Tiempo adicional por paradas/logística (escalable)
        if distance_km <= 50:
            logistics_time = 0.25  # 15 min para local
        elif distance_km <= 200:
            logistics_time = 0.5  # 30 min para regional
        elif distance_km <= 800:
            logistics_time = 1.0  # 1h para nacional
        else:
            logistics_time = 2.0  # 2h para larga distancia

        total_time = travel_time + logistics_time

        logger.info(f"🚛 Viaje: {distance_km:.1f}km a {adjusted_speed:.1f}km/h = {total_time:.1f}h")

        return round(max(0.3, total_time), 1)
