# utils/distance_calculator.py
import math
from typing import Tuple
from geopy.distance import geodesic


class DistanceCalculator:
    """📏 Calculadora de distancias y tiempos de viaje"""

    @staticmethod
    def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """🌍 Calcula distancia real usando coordenadas geodésicas"""
        try:
            return geodesic((lat1, lon1), (lat2, lon2)).kilometers
        except Exception:
            # Fallback a fórmula haversine si geopy falla
            return DistanceCalculator._haversine_distance(lat1, lon1, lat2, lon2)

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """🌐 Fórmula Haversine para distancia entre coordenadas"""
        R = 6371  # Radio de la Tierra en km

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    @staticmethod
    def calculate_travel_time(
            distance_km: float,
            trafico_nivel: str = "Moderado",
            condicion_clima: str = "Despejado"
    ) -> float:
        """⏱️ Calcula tiempo de viaje considerando tráfico y clima"""
        base_speed = 25  # km/h velocidad base en ciudad

        # 🚦 Ajuste por nivel de tráfico
        traffic_multiplier = {
            "Bajo": 1.0,
            "Moderado": 1.3,
            "Alto": 1.7,
            "Muy_Alto": 2.2
        }.get(trafico_nivel, 1.3)

        # 🌤️ Ajuste por condiciones climáticas
        weather_multiplier = {
            "Despejado": 1.0,
            "Nublado": 1.1,
            "Lluvioso": 1.4,
            "Tormenta": 1.8,
            "Frio": 1.2,
            "Templado": 1.0
        }.get(condicion_clima, 1.0)

        # Velocidad ajustada
        adjusted_speed = base_speed / (traffic_multiplier * weather_multiplier)

        # Tiempo en horas
        return distance_km / adjusted_speed

    @staticmethod
    def calculate_route_efficiency(
            distance_km: float,
            tiempo_estimado_horas: float,
            costo_mxn: float
    ) -> dict:
        """📊 Calcula métricas de eficiencia de ruta"""
        velocidad_promedio = distance_km / tiempo_estimado_horas if tiempo_estimado_horas > 0 else 0
        costo_por_km = costo_mxn / distance_km if distance_km > 0 else 0
        costo_por_hora = costo_mxn / tiempo_estimado_horas if tiempo_estimado_horas > 0 else 0

        # Score de eficiencia (0-1)
        # Mejor = mayor velocidad + menor costo
        efficiency_score = min(1.0, (velocidad_promedio / 50) * (1 / max(1, costo_por_km / 10)))

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