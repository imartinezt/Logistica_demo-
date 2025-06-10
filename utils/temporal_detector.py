# utils/temporal_detector.py
from datetime import datetime
from typing import Dict, List, Any

from config.settings import settings


class TemporalFactorDetector:
    """🕒 Detector automático de factores temporales y eventos especiales"""

    @staticmethod
    def detect_seasonal_factors(fecha: datetime) -> Dict[str, Any]:
        """🎯 Detecta factores estacionales y eventos especiales automáticamente"""
        mes = fecha.month
        dia = fecha.day
        hora = fecha.hour

        # 🎄 Detectar eventos especiales
        eventos = []
        factor_demanda = 1.0

        # Eventos por mes (configurados en settings)
        if mes in settings.EVENTOS_TEMPORADA:
            eventos.extend(settings.EVENTOS_TEMPORADA[mes])

        # 🎯 Eventos específicos por fecha con factor de demanda
        if mes == 12:  # Diciembre
            if 20 <= dia <= 25:
                eventos.append("Temporada_Navidad_Pico")
                factor_demanda = 2.8  # Pico máximo Navidad
            elif 26 <= dia <= 31:
                eventos.append("Fin_Ano_Preparacion")
                factor_demanda = 2.2
            elif 15 <= dia <= 19:
                eventos.append("Pre_Navidad")
                factor_demanda = 2.0
            else:
                factor_demanda = 1.5  # Diciembre en general

        elif mes == 11:  # Noviembre
            if 15 <= dia <= 17:  # Buen Fin México
                eventos.append("Buen_Fin_Pico")
                factor_demanda = 3.2  # Pico máximo del año
            elif 18 <= dia <= 25:
                eventos.append("Post_Buen_Fin")
                factor_demanda = 2.0
            else:
                factor_demanda = 1.4

        elif mes == 2 and dia == 14:  # San Valentín
            eventos.append("San_Valentin_Pico")
            factor_demanda = 2.2

        elif mes == 5 and 8 <= dia <= 12:  # Día de las Madres
            eventos.append("Dia_Madres_Periodo")
            factor_demanda = 2.5

        elif mes == 9 and 13 <= dia <= 16:  # Fiestas Patrias
            eventos.append("Fiestas_Patrias_Periodo")
            factor_demanda = 1.8

        elif mes == 10 and 28 <= dia <= 31:  # Halloween/Día de Muertos
            eventos.append("Halloween_Muertos")
            factor_demanda = 1.6

        elif mes == 1 and 1 <= dia <= 7:  # Día de Reyes
            eventos.append("Dia_Reyes_Periodo")
            factor_demanda = 2.0

        # 🌤️ Detectar clima esperado por temporada
        clima_info = settings.CLIMA_TEMPORADA.get(mes, {
            "condicion": "Templado",
            "lluvia_prob": 30,
            "temp": 22
        })

        # 🚦 Detectar nivel de tráfico por hora
        trafico = TemporalFactorDetector._get_traffic_level(hora)

        # 📈 Detectar si es temporada alta
        es_temporada_alta = factor_demanda > 1.5

        # ⏰ Impacto adicional en tiempo por eventos
        impacto_tiempo_extra = (
            3 if factor_demanda > 2.5 else  # Buen Fin
            2 if factor_demanda > 2.0 else  # Navidad/Madres
            1 if factor_demanda > 1.5 else  # Temporada media
            0  # Normal
        )

        return {
            "eventos_detectados": eventos,
            "factor_demanda": factor_demanda,
            "condicion_clima": clima_info["condicion"],
            "probabilidad_lluvia": clima_info["lluvia_prob"],
            "temperatura": clima_info["temp"],
            "trafico_nivel": trafico,
            "es_temporada_alta": es_temporada_alta,
            "impacto_tiempo_extra": impacto_tiempo_extra,
            "criticidad_logistica": (
                "Crítica" if factor_demanda > 2.5 else
                "Alta" if factor_demanda > 2.0 else
                "Media" if factor_demanda > 1.5 else
                "Normal"
            )
        }

    @staticmethod
    def _get_traffic_level(hora: int) -> str:
        """🚦 Determina nivel de tráfico por hora del día"""
        if 7 <= hora <= 10:  # Rush matutino
            return "Muy_Alto"
        elif 17 <= hora <= 20:  # Rush vespertino
            return "Muy_Alto"
        elif 11 <= hora <= 16:  # Horas comerciales
            return "Alto"
        elif 21 <= hora <= 23 or 6 <= hora <= 7:  # Transición
            return "Moderado"
        else:  # Madrugada/noche
            return "Bajo"

    @staticmethod
    def get_seasonal_recommendations(factores: Dict[str, Any]) -> List[str]:
        """💡 Genera recomendaciones basadas en factores detectados"""
        recomendaciones = []

        if factores["es_temporada_alta"]:
            recomendaciones.append("🎯 Temporada alta: preparar inventario adicional")
            recomendaciones.append("📦 Considerar fulfillment distribuido")

        if factores["factor_demanda"] > 2.5:
            recomendaciones.append("🚨 Pico crítico: activar rutas express adicionales")
            recomendaciones.append("👥 Reforzar personal de preparación")

        if factores["probabilidad_lluvia"] > 60:
            recomendaciones.append("🌧️ Alta probabilidad lluvia: ajustar tiempos")
            recomendaciones.append("🚚 Priorizar flota cubierta")

        if factores["trafico_nivel"] in ["Alto", "Muy_Alto"]:
            recomendaciones.append("🚦 Tráfico intenso: considerar ventanas horarias")

        # Recomendaciones específicas por evento
        eventos = factores["eventos_detectados"]
        if "Buen_Fin_Pico" in eventos:
            recomendaciones.append("🛍️ Buen Fin: maximizar capacidad FE")
        elif "Temporada_Navidad_Pico" in eventos:
            recomendaciones.append("🎄 Navidad: garantizar cumplimiento promesas")
        elif "Dia_Madres_Periodo" in eventos:
            recomendaciones.append("💐 Día Madres: priorizar productos especiales")

        return recomendaciones

    @staticmethod
    def predict_delivery_impact(factores: Dict[str, Any]) -> Dict[str, Any]:
        """📊 Predice impacto en entregas basado en factores"""
        factor_demanda = factores["factor_demanda"]
        lluvia_prob = factores["probabilidad_lluvia"]
        trafico = factores["trafico_nivel"]

        # Calcular multiplicador de tiempo total
        tiempo_multiplier = factor_demanda

        if lluvia_prob > 60:
            tiempo_multiplier *= 1.3
        elif lluvia_prob > 30:
            tiempo_multiplier *= 1.1

        if trafico == "Muy_Alto":
            tiempo_multiplier *= 1.4
        elif trafico == "Alto":
            tiempo_multiplier *= 1.2

        # Calcular impacto en costo
        costo_multiplier = 1.0
        if factor_demanda > 2.0:
            costo_multiplier = 1.2  # Costos premium en temporada alta

        # Calcular impacto en probabilidad de cumplimiento
        prob_impact = max(0.6, 1.0 - (factor_demanda - 1.0) * 0.15)

        return {
            "tiempo_multiplier": round(tiempo_multiplier, 2),
            "costo_multiplier": round(costo_multiplier, 2),
            "probabilidad_cumplimiento_ajustada": round(prob_impact, 2),
            "nivel_complejidad": (
                "Muy Alta" if tiempo_multiplier > 2.5 else
                "Alta" if tiempo_multiplier > 2.0 else
                "Media" if tiempo_multiplier > 1.5 else
                "Normal"
            )
        }