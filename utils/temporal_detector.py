from datetime import datetime, timedelta
from typing import Dict, List, Any
import polars as pl
from pathlib import Path

from config.settings import settings
from utils.logger import logger


class TemporalFactorDetector:
    """🕒 Detector avanzado de factores temporales que USA REALMENTE los CSV"""

    @staticmethod
    def detect_comprehensive_factors(fecha: datetime,
                                     codigo_postal: str = None,
                                     data_dir: Path = None) -> Dict[str, Any]:
        """🎯 Detección REAL usando CSV de factores externos"""

        real_factors = TemporalFactorDetector._load_real_external_factors(fecha, codigo_postal, data_dir)

        if real_factors:
            logger.info(f"📅 Factores REALES encontrados en CSV para {fecha.date()}")
            return real_factors

        logger.info(f"🤖 Generando factores automáticos para {fecha.date()}")
        return TemporalFactorDetector._generate_intelligent_factors(fecha, codigo_postal)

    @staticmethod
    def _load_real_external_factors(fecha: datetime, codigo_postal: str, data_dir: Path) -> Dict[str, Any]:
        """📂 Carga factores REALES del CSV"""

        if not data_dir:
            data_dir = settings.DATA_DIR

        csv_path = data_dir / settings.CSV_FILES['factores_externos']

        try:
            df = pl.read_csv(csv_path)
            fecha_str = fecha.date().isoformat()
            exact_match = df.filter(pl.col('fecha') == fecha_str)

            if exact_match.height > 0:
                if codigo_postal and 'rango_cp_afectado' in df.columns:
                    cp_prefix = codigo_postal[:2]
                    cp_matches = exact_match.filter(
                        pl.col('rango_cp_afectado').str.contains(cp_prefix)
                    )
                    if cp_matches.height > 0:
                        factor_data = cp_matches.to_dicts()[0]
                    else:
                        factor_data = exact_match.to_dicts()[0]
                else:
                    factor_data = exact_match.to_dicts()[0]

                return TemporalFactorDetector._process_csv_factors(factor_data, fecha, codigo_postal)

            for delta in range(1, 4):
                for direction in [-1, 1]:
                    check_date = fecha + timedelta(days=delta * direction)
                    check_str = check_date.date().isoformat()

                    nearby_match = df.filter(pl.col('fecha') == check_str)
                    if nearby_match.height > 0:
                        factor_data = nearby_match.to_dicts()[0]
                        logger.info(f"📅 Usando factores de fecha cercana: {check_str}")
                        return TemporalFactorDetector._process_csv_factors(factor_data, fecha, codigo_postal)

            return None

        except Exception as e:
            logger.warning(f"❌ Error cargando factores externos del CSV: {e}")
            return None

    @staticmethod
    def _process_csv_factors(factor_data: Dict[str, Any], fecha: datetime, codigo_postal: str) -> Dict[str, Any]:
        """🔄 Procesa factores del CSV y los estructura correctamente"""

        # Extraer eventos detectados
        eventos_detectados = []
        evento_principal = factor_data.get('evento_detectado', 'Normal')
        if evento_principal and evento_principal != 'Normal':
            eventos_detectados.append(evento_principal)

        # También detectar eventos por fecha (backup)
        eventos_fecha = TemporalFactorDetector._detect_events_by_date(fecha)
        eventos_detectados.extend(eventos_fecha)
        eventos_detectados = list(set(eventos_detectados))  # Eliminar duplicados

        # Factor de demanda del CSV o calculado
        factor_demanda = float(factor_data.get('factor_demanda', 1.0))

        # Si el factor del CSV es muy bajo, usar cálculo inteligente
        if factor_demanda < 1.5 and eventos_detectados:
            factor_demanda = TemporalFactorDetector._calculate_demand_factor_smart(fecha, eventos_detectados)

        # Clima del CSV o detectado
        condicion_clima = factor_data.get('condicion_clima', 'Templado')
        if not condicion_clima or condicion_clima == 'N/A':
            condicion_clima = TemporalFactorDetector._predict_weather_by_date(fecha)

        # Tráfico del CSV o calculado
        trafico_nivel = factor_data.get('trafico_nivel', 'Moderado')
        if not trafico_nivel or trafico_nivel == 'N/A':
            trafico_nivel = TemporalFactorDetector._calculate_traffic_by_date(fecha)

        # Impactos del CSV o calculados
        impacto_tiempo = float(factor_data.get('impacto_tiempo_extra_horas', 0))
        if impacto_tiempo == 0:
            impacto_tiempo = TemporalFactorDetector._calculate_time_impact_smart(factor_demanda, eventos_detectados)

        impacto_costo = float(factor_data.get('impacto_costo_extra_pct', 0))
        if impacto_costo == 0:
            impacto_costo = TemporalFactorDetector._calculate_cost_impact_smart(factor_demanda, eventos_detectados)

        # Criticidad del CSV o calculada
        criticidad = factor_data.get('criticidad_logistica', 'Normal')
        if not criticidad or criticidad == 'N/A':
            criticidad = TemporalFactorDetector._assess_criticality_smart(factor_demanda, impacto_tiempo,
                                                                          eventos_detectados)

        return {
            'eventos_detectados': eventos_detectados,
            'es_temporada_alta': factor_demanda > 1.8,
            'es_temporada_critica': factor_demanda > 2.5,
            'factor_demanda': factor_demanda,
            'categoria_temporada': TemporalFactorDetector._categorize_season(factor_demanda),

            'condicion_clima': condicion_clima,
            'temperatura_celsius': TemporalFactorDetector._get_temp_by_season(fecha),
            'probabilidad_lluvia': TemporalFactorDetector._get_rain_by_season(fecha),
            'viento_esperado': 15,

            'trafico_nivel': trafico_nivel,
            'impacto_tiempo_extra_horas': impacto_tiempo,
            'impacto_costo_extra_pct': impacto_costo,
            'zona_seguridad': 'Media',
            'restricciones_vehiculares': [],
            'criticidad_logistica': criticidad,

            'fecha_analisis': fecha.isoformat(),
            'fuente_datos': 'CSV_real' if factor_data.get('fecha') else 'calculado',
            'confianza_prediccion': 0.95 if factor_data.get('fecha') else 0.80,
            'ultima_actualizacion': datetime.now().isoformat()
        }

    @staticmethod
    def _detect_events_by_date(fecha: datetime) -> List[str]:
        """🎄 Detecta eventos específicos por fecha (mejorado)"""

        mes = fecha.month
        dia = fecha.day
        eventos = []

        # Navidad - FECHAS EXACTAS
        if mes == 12:
            if dia == 24:
                eventos.extend(['Nochebuena', 'Navidad_Pico', 'Emergencia_Regalos'])
            elif dia == 25:
                eventos.extend(['Navidad', 'Dia_Navidad'])
            elif 20 <= dia <= 23:
                eventos.extend(['Pre_Navidad_Intenso', 'Compras_Ultimo_Momento'])
            elif 15 <= dia <= 19:
                eventos.extend(['Pre_Navidad', 'Preparacion_Navidad'])
            elif dia == 31:
                eventos.extend(['Fin_Ano', 'Nochevieja'])
            elif 26 <= dia <= 30:
                eventos.extend(['Post_Navidad', 'Entre_Navidad_Ano_Nuevo'])

        # Año Nuevo
        elif mes == 1:
            if dia == 1:
                eventos.extend(['Ano_Nuevo', 'Primer_Dia_Ano'])
            elif dia == 6:
                eventos.extend(['Dia_Reyes', 'Reyes_Magos'])
            elif 2 <= dia <= 5:
                eventos.extend(['Post_Ano_Nuevo', 'Preparacion_Reyes'])

        # Buen Fin (Noviembre)
        elif mes == 11:
            # Buen Fin es siempre el fin de semana antes de Thanksgiving (4to jueves)
            # Aproximadamente días 15-17
            if 15 <= dia <= 17:
                eventos.extend(['Buen_Fin', 'Black_Friday_MX', 'Descuentos_Masivos'])
            elif 18 <= dia <= 20:
                eventos.extend(['Post_Buen_Fin', 'Cyber_Monday'])

        # San Valentín
        elif mes == 2 and dia == 14:
            eventos.extend(['San_Valentin', 'Dia_Amor'])

        # Día de las Madres (segundo domingo de mayo)
        elif mes == 5:
            segundo_domingo = TemporalFactorDetector._get_second_sunday(fecha.year, 5)
            if dia == segundo_domingo:
                eventos.extend(['Dia_Madres', 'Dia_Madre'])
            elif segundo_domingo and segundo_domingo - 2 <= dia <= segundo_domingo - 1:
                eventos.extend(['Pre_Dia_Madres'])

        return eventos

    @staticmethod
    def _calculate_demand_factor_smart(fecha: datetime, eventos: List[str]) -> float:
        """📈 Calcula factor de demanda inteligente basado en eventos REALES"""

        # Factor base por día de semana
        weekday = fecha.weekday()
        base_factor = {
            0: 1.0,  # Lunes
            1: 1.05,  # Martes
            2: 1.1,  # Miércoles
            3: 1.15,  # Jueves
            4: 1.25,  # Viernes
            5: 1.2,  # Sábado
            6: 0.9  # Domingo
        }[weekday]

        # Multiplicadores por evento ESPECÍFICO
        event_multipliers = {
            'Navidad_Pico': 3.2,
            'Nochebuena': 3.5,
            'Navidad': 0.3,  # Todo cerrado
            'Emergencia_Regalos': 3.0,
            'Buen_Fin': 3.5,
            'Black_Friday_MX': 3.2,
            'Dia_Madres': 2.8,
            'San_Valentin': 2.2,
            'Pre_Navidad_Intenso': 2.5,
            'Post_Buen_Fin': 2.0,
            'Dia_Reyes': 2.4
        }

        # Aplicar multiplicador del evento más importante
        max_multiplier = 1.0
        for evento in eventos:
            if evento in event_multipliers:
                max_multiplier = max(max_multiplier, event_multipliers[evento])

        final_factor = base_factor * max_multiplier

        # Límites realistas
        return max(0.3, min(4.0, final_factor))

    @staticmethod
    def _calculate_time_impact_smart(factor_demanda: float, eventos: List[str]) -> float:
        """⏱️ Calcula impacto en tiempo de manera inteligente"""

        # Impacto base por demanda
        base_impact = max(0, (factor_demanda - 1.0) * 1.5)

        # Impacto adicional por eventos críticos
        critical_events = ['Navidad_Pico', 'Nochebuena', 'Buen_Fin', 'Emergencia_Regalos']
        if any(event in eventos for event in critical_events):
            base_impact += 2.0

        return round(min(6.0, base_impact), 1)  # Máximo 6 horas extra

    @staticmethod
    def _calculate_cost_impact_smart(factor_demanda: float, eventos: List[str]) -> float:
        """💰 Calcula impacto en costo de manera inteligente"""

        # Impacto base por demanda
        base_impact = max(0, (factor_demanda - 1.0) * 15)

        # Impacto adicional por eventos premium
        premium_events = ['Navidad_Pico', 'Nochebuena', 'Buen_Fin']
        if any(event in eventos for event in premium_events):
            base_impact += 20.0

        return round(min(50.0, base_impact), 1)  # Máximo 50% extra

    @staticmethod
    def _assess_criticality_smart(factor_demanda: float, tiempo_impacto: float, eventos: List[str]) -> str:
        """🎯 Evalúa criticidad de manera inteligente"""

        score = 0

        if factor_demanda > 3.0:
            score += 4
        elif factor_demanda > 2.5:
            score += 3
        elif factor_demanda > 2.0:
            score += 2
        elif factor_demanda > 1.5:
            score += 1

        if tiempo_impacto > 3.0:
            score += 3
        elif tiempo_impacto > 2.0:
            score += 2
        elif tiempo_impacto > 1.0:
            score += 1

        critical_events = ['Navidad_Pico', 'Nochebuena', 'Emergencia_Regalos']
        if any(event in eventos for event in critical_events):
            score += 2

        if score >= 7:
            return 'Crítica'
        elif score >= 5:
            return 'Alta'
        elif score >= 3:
            return 'Media'
        elif score >= 1:
            return 'Baja'
        else:
            return 'Normal'

    @staticmethod
    def _generate_intelligent_factors(fecha: datetime, codigo_postal: str) -> Dict[str, Any]:
        """🧠 Genera factores inteligentes cuando no hay datos en CSV"""

        eventos = TemporalFactorDetector._detect_events_by_date(fecha)
        factor_demanda = TemporalFactorDetector._calculate_demand_factor_smart(fecha, eventos)
        clima = TemporalFactorDetector._predict_weather_by_date(fecha)
        trafico = TemporalFactorDetector._calculate_traffic_by_date(fecha)

        impacto_tiempo = TemporalFactorDetector._calculate_time_impact_smart(factor_demanda, eventos)
        impacto_costo = TemporalFactorDetector._calculate_cost_impact_smart(factor_demanda, eventos)
        criticidad = TemporalFactorDetector._assess_criticality_smart(factor_demanda, impacto_tiempo, eventos)

        return {
            'eventos_detectados': eventos,
            'es_temporada_alta': factor_demanda > 1.8,
            'es_temporada_critica': factor_demanda > 2.5,
            'factor_demanda': factor_demanda,
            'categoria_temporada': TemporalFactorDetector._categorize_season(factor_demanda),

            'condicion_clima': clima,
            'temperatura_celsius': TemporalFactorDetector._get_temp_by_season(fecha),
            'probabilidad_lluvia': TemporalFactorDetector._get_rain_by_season(fecha),
            'viento_esperado': 15,

            'trafico_nivel': trafico,
            'impacto_tiempo_extra_horas': impacto_tiempo,
            'impacto_costo_extra_pct': impacto_costo,
            'zona_seguridad': 'Media',
            'restricciones_vehiculares': [],
            'criticidad_logistica': criticidad,

            'fecha_analisis': fecha.isoformat(),
            'fuente_datos': 'generado_inteligente',
            'confianza_prediccion': 0.85,
            'ultima_actualizacion': datetime.now().isoformat()
        }

    # Métodos auxiliares
    @staticmethod
    def _get_second_sunday(year: int, month: int) -> int:
        """📅 Obtiene el segundo domingo del mes"""
        try:
            first_day = datetime(year, month, 1)
            first_sunday = 1 + (6 - first_day.weekday()) % 7
            second_sunday = first_sunday + 7

            # Verificar que esté dentro del mes
            try:
                datetime(year, month, second_sunday)
                return second_sunday
            except ValueError:
                return None
        except:
            return None

    @staticmethod
    def _predict_weather_by_date(fecha: datetime) -> str:
        """🌤️ Predice clima por fecha"""
        mes = fecha.month

        climate_map = {
            1: 'Frio_Seco', 2: 'Templado_Seco', 3: 'Templado',
            4: 'Calido_Seco', 5: 'Calido', 6: 'Lluvioso_Inicio',
            7: 'Lluvioso_Intenso', 8: 'Lluvioso_Intenso', 9: 'Lluvioso',
            10: 'Templado_Humedo', 11: 'Templado_Seco', 12: 'Frio_Seco'
        }

        return climate_map.get(mes, 'Templado')

    @staticmethod
    def _calculate_traffic_by_date(fecha: datetime) -> str:
        """🚦 Calcula tráfico por fecha"""
        weekday = fecha.weekday()
        hora = fecha.hour

        if weekday < 5:  # Días laborales
            if 7 <= hora <= 10 or 17 <= hora <= 20:
                return 'Alto'
            elif 11 <= hora <= 16:
                return 'Moderado'
            else:
                return 'Bajo'
        elif weekday == 5:  # Sábado
            if 10 <= hora <= 18:
                return 'Moderado'
            else:
                return 'Bajo'
        else:  # Domingo
            return 'Bajo'

    @staticmethod
    def _get_temp_by_season(fecha: datetime) -> int:
        """🌡️ Temperatura por temporada"""
        mes = fecha.month
        temp_map = {
            1: 16, 2: 18, 3: 22, 4: 25, 5: 27, 6: 25,
            7: 24, 8: 24, 9: 23, 10: 21, 11: 19, 12: 17
        }
        return temp_map.get(mes, 22)

    @staticmethod
    def _get_rain_by_season(fecha: datetime) -> int:
        """🌧️ Probabilidad de lluvia por temporada"""
        mes = fecha.month
        rain_map = {
            1: 10, 2: 15, 3: 20, 4: 25, 5: 40, 6: 70,
            7: 85, 8: 80, 9: 75, 10: 45, 11: 25, 12: 15
        }
        return rain_map.get(mes, 30)

    @staticmethod
    def _categorize_season(demand_factor: float) -> str:
        """📊 Categoriza temporada"""
        if demand_factor >= 3.0:
            return 'Hiper_Critica'
        elif demand_factor >= 2.5:
            return 'Critica'
        elif demand_factor >= 2.0:
            return 'Alta'
        elif demand_factor >= 1.5:
            return 'Elevada'
        elif demand_factor >= 1.2:
            return 'Media_Alta'
        else:
            return 'Normal'