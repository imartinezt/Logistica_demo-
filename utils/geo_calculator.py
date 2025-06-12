import math
from typing import Tuple, List, Dict, Any

from geopy.distance import geodesic, great_circle
from pyproj import Transformer

from config.settings import settings
from utils.logger import logger


class GeoCalculator:
    """Calculador geoespacial """

    _transformer_mexico = None

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
    def calculate_route_efficiency(distance_km: float,
                                   tiempo_estimado_horas: float,
                                   costo_mxn: float) -> Dict[str, Any]:
        """Calcula métricas de eficiencia de ruta"""

        if tiempo_estimado_horas <= 0 or distance_km <= 0 or costo_mxn <= 0:
            return {
                "velocidad_promedio_kmh": 0,
                "costo_por_km": float('inf') if distance_km <= 0 else costo_mxn / max(0.1, distance_km),
                "costo_por_hora": float('inf') if tiempo_estimado_horas <= 0 else costo_mxn / max(0.1,
                                                                                                  tiempo_estimado_horas),
                "efficiency_score": 0,
                "clasificacion": "Deficiente"
            }

        velocidad_promedio = distance_km / tiempo_estimado_horas
        costo_por_km = costo_mxn / distance_km
        costo_por_hora = costo_mxn / tiempo_estimado_horas
        efficiency_score = min(1.0, (velocidad_promedio / 40) * (1 / max(1, costo_por_km / 12)))

        return {
            "velocidad_promedio_kmh": round(velocidad_promedio, 2),
            "costo_por_km": round(costo_por_km, 2),
            "costo_por_hora": round(costo_por_hora, 2),
            "efficiency_score": round(efficiency_score, 3),
            "clasificacion": (
                "Excelente" if efficiency_score >= 0.8 else
                "Buena" if efficiency_score >= 0.6 else
                "Regular" if efficiency_score >= 0.4 else
                "Deficiente"
            )
        }

    @staticmethod
    def calculate_distance_km(lat1: float, lon1: float,
                              lat2: float, lon2: float,
                              method: str = 'geodesic') -> float:
        """Calcula distancia real entre coordenadas """

        if not GeoCalculator._validate_coordinates(lat1, lon1, lat2, lon2):
            logger.warning(f"⚠️ Coordenadas inválidas: ({lat1},{lon1}) -> ({lat2},{lon2})")
            return 0.0

        try:
            if method == 'geodesic':
                # Método más preciso usando geodesic
                distance = geodesic((lat1, lon1), (lat2, lon2)).kilometers
            elif method == 'great_circle':
                # Método rápido para distancias largas
                distance = great_circle((lat1, lon1), (lat2, lon2)).kilometers
            elif method == 'pyproj':
                # Método ultra-preciso usando proyección mexicana
                distance = GeoCalculator._calculate_pyproj_distance(lat1, lon1, lat2, lon2)
            else:
                # Fallback a haversine
                distance = GeoCalculator._haversine_distance(lat1, lon1, lat2, lon2)

            # Validar resultado
            if distance < 0 or distance > settings.MAX_DISTANCE_KM:
                logger.warning(f"⚠️ Distancia fuera de rango: {distance}km")
                return min(distance, settings.MAX_DISTANCE_KM) if distance > 0 else 0.0

            return round(distance, 2)

        except Exception as e:
            logger.warning(f"❌ Error calculando distancia: {e}")
            return GeoCalculator._haversine_distance(lat1, lon1, lat2, lon2)

    @staticmethod
    def _validate_coordinates(lat1: float, lon1: float, lat2: float, lon2: float) -> bool:
        """✅ Valida que las coordenadas sean válidas para México"""
        try:
            # Validar rangos generales
            if not (-90 <= lat1 <= 90 and -180 <= lon1 <= 180):
                return False
            if not (-90 <= lat2 <= 90 and -180 <= lon2 <= 180):
                return False

            # Validar rangos específicos para México (más permisivo)
            mexico_lat_range = (12.0, 35.0)  # Más amplio
            mexico_lon_range = (-120.0, -84.0)  # Más amplio

            if not (mexico_lat_range[0] <= lat1 <= mexico_lat_range[1] and
                    mexico_lon_range[0] <= lon1 <= mexico_lon_range[1]):
                return False
            if not (mexico_lat_range[0] <= lat2 <= mexico_lat_range[1] and
                    mexico_lon_range[0] <= lon2 <= mexico_lon_range[1]):
                return False

            return True
        except (TypeError, ValueError):
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
        """🌐 Fórmula Haversine como fallback"""
        try:
            R = settings.EARTH_RADIUS_KM

            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            delta_lat = math.radians(lat2 - lat1)
            delta_lon = math.radians(lon2 - lon1)

            a = (math.sin(delta_lat / 2) ** 2 +
                 math.cos(lat1_rad) * math.cos(lat2_rad) *
                 math.sin(delta_lon / 2) ** 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            distance = R * c
            return max(0.0, distance)  # Asegurar que no sea negativo
        except Exception as e:
            logger.error(f"❌ Error en Haversine: {e}")
            return 0.0

    @staticmethod
    def calculate_bearing(lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
        """🧭 Calcula el bearing (dirección) entre dos puntos"""
        try:
            if not GeoCalculator._validate_coordinates(lat1, lon1, lat2, lon2):
                return 0.0

            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            delta_lon = math.radians(lon2 - lon1)

            y = math.sin(delta_lon) * math.cos(lat2_rad)
            x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
                 math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))

            bearing = math.atan2(y, x)
            bearing_degrees = math.degrees(bearing)

            # Normalizar a 0-360 grados
            return (bearing_degrees + 360) % 360
        except Exception as e:
            logger.warning(f"❌ Error calculando bearing: {e}")
            return 0.0

    @staticmethod
    def find_closest_locations(target_lat: float, target_lon: float,
                               locations: List[Dict[str, Any]],
                               max_results: int = 10,
                               max_distance_km: float = None) -> List[Dict[str, Any]]:
        """📍 Encuentra las ubicaciones más cercanas usando cálculos optimizados"""
        if not locations:
            return []

        max_distance_km = max_distance_km or settings.MAX_DISTANCE_KM

        # Validar coordenadas objetivo
        if not GeoCalculator._validate_coordinates(target_lat, target_lon, target_lat, target_lon):
            logger.warning(f"⚠️ Coordenadas objetivo inválidas: ({target_lat}, {target_lon})")
            return []

        # Calcular distancias para todas las ubicaciones
        locations_with_distance = []

        for location in locations:
            try:
                lat = location.get('latitud', 0)
                lon = location.get('longitud', 0)

                # Validar coordenadas de la ubicación
                if not GeoCalculator._validate_coordinates(lat, lon, lat, lon):
                    continue

                distance = GeoCalculator.calculate_distance_km(
                    target_lat, target_lon, lat, lon, method='geodesic'
                )

                if 0 < distance <= max_distance_km:
                    location_copy = location.copy()
                    location_copy['distancia_km'] = distance
                    location_copy['bearing'] = GeoCalculator.calculate_bearing(
                        target_lat, target_lon, lat, lon
                    )
                    locations_with_distance.append(location_copy)

            except Exception as e:
                logger.warning(f"❌ Error procesando ubicación {location.get('id', 'unknown')}: {e}")
                continue

        # Ordenar por distancia y limitar resultados
        locations_with_distance.sort(key=lambda x: x['distancia_km'])
        result = locations_with_distance[:max_results]

        logger.info(f"📍 Encontradas {len(result)} ubicaciones cercanas")
        return result

    @staticmethod
    def calculate_route_geometry(waypoints: List[Tuple[float, float]]) -> Dict[str, Any]:
        """🗺️ Calcula geometría de ruta con múltiples waypoints"""
        if len(waypoints) < 2:
            return {'total_distance': 0, 'segments': [], 'bbox': None}

        segments = []
        total_distance = 0

        # Calcular segmentos
        for i in range(len(waypoints) - 1):
            lat1, lon1 = waypoints[i]
            lat2, lon2 = waypoints[i + 1]

            # Validar waypoints
            if not GeoCalculator._validate_coordinates(lat1, lon1, lat2, lon2):
                logger.warning(f"⚠️ Waypoint inválido en segmento {i}")
                continue

            distance = GeoCalculator.calculate_distance_km(lat1, lon1, lat2, lon2)
            bearing = GeoCalculator.calculate_bearing(lat1, lon1, lat2, lon2)

            segments.append({
                'segment_id': i + 1,
                'start': {'lat': lat1, 'lon': lon1},
                'end': {'lat': lat2, 'lon': lon2},
                'distance_km': distance,
                'bearing': bearing
            })

            total_distance += distance

        # Calcular bounding box
        if waypoints:
            lats = [wp[0] for wp in waypoints]
            lons = [wp[1] for wp in waypoints]

            bbox = {
                'min_lat': min(lats),
                'max_lat': max(lats),
                'min_lon': min(lons),
                'max_lon': max(lons)
            }
        else:
            bbox = None

        return {
            'total_distance_km': round(total_distance, 2),
            'segments': segments,
            'bbox': bbox,
            'waypoints_count': len(waypoints)
        }

    @staticmethod
    def calculate_travel_time(distance_km: float,
                              transport_type: str = 'FI',
                              traffic_level: str = 'Moderado',
                              weather_condition: str = 'Despejado',
                              road_type: str = 'carretera') -> float:
        """⏱️ Tiempo de viaje CORREGIDO con velocidades realistas"""

        if distance_km <= 0:
            return 0.5  # Tiempo mínimo base

        # CORRECCIÓN: Velocidades base realistas
        if transport_type == 'FI':
            base_speed = 55.0  # 55 km/h promedio para flota interna
        elif transport_type == 'FE':
            base_speed = 65.0  # 65 km/h promedio para flota externa
        else:
            base_speed = 60.0

        # CORRECCIÓN: Multiplicadores más realistas
        traffic_multipliers = {
            'Bajo': 1.0,
            'Moderado': 0.90,  # Reducción menor
            'Alto': 0.75,  # Más realista
            'Muy_Alto': 0.60  # Más realista
        }

        weather_multipliers = {
            'Despejado': 1.0,
            'Nublado': 0.98,
            'Lluvioso': 0.85,  # Más realista
            'Tormenta': 0.70,  # Más realista
            'Templado': 1.0
        }

        # Para distancias largas, asumir carretera
        if distance_km > 100:
            road_type = 'carretera'

        road_multipliers = {
            'urbano': 0.60,  # Ciudad es más lenta
            'suburbano': 0.80,  # Suburbios intermedios
            'carretera': 1.10,  # Carretera es más rápida
            'autopista': 1.25  # Autopista más rápida
        }

        # Calcular velocidad ajustada
        traffic_factor = traffic_multipliers.get(traffic_level, 0.9)
        weather_factor = weather_multipliers.get(weather_condition, 1.0)
        road_factor = road_multipliers.get(road_type, 1.0)

        adjusted_speed = base_speed * road_factor * traffic_factor * weather_factor
        adjusted_speed = max(adjusted_speed, 35.0)  # Mínimo 35 km/h

        # Tiempo en horas
        travel_time = distance_km / adjusted_speed

        # CORRECCIÓN: Tiempo base para paradas/maniobras más realista
        stops_time = 0.5 + (distance_km / 200) * 0.5  # Más paradas en distancias largas

        total_time = travel_time + stops_time

        logger.info(f"🚛 Viaje: {distance_km:.1f}km a {adjusted_speed:.1f}km/h = {total_time:.1f}h")

        return round(max(0.5, total_time), 1)

    @staticmethod
    def calculate_optimal_route_sequence(locations: List[Dict[str, Any]],
                                         start_location: Dict[str, Any]) -> List[Dict[str, Any]]:
        """🎯 Calcula secuencia óptima de ubicaciones (TSP simplificado)"""
        if not locations:
            return []

        if len(locations) == 1:
            return locations

        if len(locations) == 2:
            # Para 2 ubicaciones, calcular cuál es más cercana primero
            loc1, loc2 = locations[0], locations[1]

            try:
                dist_to_1 = GeoCalculator.calculate_distance_km(
                    start_location['latitud'], start_location['longitud'],
                    loc1['latitud'], loc1['longitud']
                )

                dist_to_2 = GeoCalculator.calculate_distance_km(
                    start_location['latitud'], start_location['longitud'],
                    loc2['latitud'], loc2['longitud']
                )

                if dist_to_1 <= dist_to_2:
                    return [loc1, loc2]
                else:
                    return [loc2, loc1]
            except Exception as e:
                logger.warning(f"❌ Error ordenando 2 ubicaciones: {e}")
                return locations

        # Para 3+ ubicaciones, usar heurística greedy
        return GeoCalculator._greedy_tsp(start_location, locations)

    @staticmethod
    def _greedy_tsp(start: Dict[str, Any], locations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """🔄 Algoritmo greedy para TSP simplificado"""
        try:
            unvisited = locations.copy()
            route = []
            current_location = start

            while unvisited:
                # Encontrar la ubicación más cercana no visitada
                closest_location = None
                min_distance = float('inf')

                for location in unvisited:
                    try:
                        distance = GeoCalculator.calculate_distance_km(
                            current_location['latitud'], current_location['longitud'],
                            location['latitud'], location['longitud']
                        )

                        if distance < min_distance:
                            min_distance = distance
                            closest_location = location
                    except Exception as e:
                        logger.warning(f"❌ Error calculando distancia en TSP: {e}")
                        continue

                if closest_location:
                    route.append(closest_location)
                    unvisited.remove(closest_location)
                    current_location = closest_location
                else:
                    # Si no se puede calcular, agregar el resto en orden
                    route.extend(unvisited)
                    break

            logger.info(f"🎯 Secuencia óptima calculada para {len(route)} ubicaciones")
            return route

        except Exception as e:
            logger.error(f"❌ Error en TSP greedy: {e}")
            return locations  # Fallback al orden original

    @staticmethod
    def calculate_delivery_zone_metrics(centro_lat: float, centro_lon: float,
                                        locations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """📊 Calcula métricas de zona de entrega"""
        if not locations:
            return {'coverage_radius': 0, 'density': 0, 'avg_distance': 0}

        # Validar coordenadas del centro
        if not GeoCalculator._validate_coordinates(centro_lat, centro_lon, centro_lat, centro_lon):
            return {'coverage_radius': 0, 'density': 0, 'avg_distance': 0}

        distances = []
        for location in locations:
            try:
                lat = location.get('latitud', 0)
                lon = location.get('longitud', 0)

                if GeoCalculator._validate_coordinates(lat, lon, lat, lon):
                    distance = GeoCalculator.calculate_distance_km(
                        centro_lat, centro_lon, lat, lon
                    )
                    if distance > 0:
                        distances.append(distance)
            except Exception as e:
                logger.warning(f"❌ Error calculando distancia en métricas: {e}")
                continue

        if not distances:
            return {'coverage_radius': 0, 'density': 0, 'avg_distance': 0}

        # Calcular métricas
        avg_distance = sum(distances) / len(distances)
        max_distance = max(distances)
        min_distance = min(distances)

        # Área aproximada del círculo que cubre todas las ubicaciones
        coverage_area = math.pi * (max_distance ** 2)
        density = len(locations) / coverage_area if coverage_area > 0 else 0

        return {
            'coverage_radius_km': round(max_distance, 2),
            'avg_distance_km': round(avg_distance, 2),
            'min_distance_km': round(min_distance, 2),
            'density_per_km2': round(density, 4),
            'total_locations': len(locations),
            'coverage_area_km2': round(coverage_area, 2)
        }

    @staticmethod
    def is_within_delivery_polygon(lat: float, lon: float,
                                   polygon_points: List[Tuple[float, float]]) -> bool:
        """🔍 Verifica si un punto está dentro de un polígono de entrega"""
        if len(polygon_points) < 3:
            return True

        if not GeoCalculator._validate_coordinates(lat, lon, lat, lon):
            return False

        try:
            x, y = lon, lat
            n = len(polygon_points)
            inside = False

            j = n - 1
            for i in range(n):
                if ((polygon_points[i][0] > y) != (polygon_points[j][0] > y)) and \
                        (x < (polygon_points[j][1] - polygon_points[i][1]) *
                         (y - polygon_points[i][0]) / (polygon_points[j][0] - polygon_points[i][0]) +
                         polygon_points[i][1]):
                    inside = not inside
                j = i

            return inside

        except Exception as e:
            logger.warning(f"❌ Error en verificación de polígono: {e}")
            return True