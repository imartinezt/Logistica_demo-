from datetime import datetime, timedelta
from typing import Dict, List, Any

from config.settings import settings
from utils.logger import logger


class TemporalFactorDetector:
    """🕒 Detector avanzado de factores temporales con IA de patrones estacionales"""

    @staticmethod
    def detect_comprehensive_factors(fecha: datetime,
                                     codigo_postal: str = None) -> Dict[str, Any]:
        """🎯 Detección completa de factores temporales, estacionales y logísticos"""

        # Componentes base
        eventos = TemporalFactorDetector._detect_events(fecha)
        demanda = TemporalFactorDetector._calculate_demand_factor(fecha, eventos)
        clima = TemporalFactorDetector._predict_weather_impact(fecha, codigo_postal)
        trafico = TemporalFactorDetector._calculate_traffic_patterns(fecha)

        # Factores operativos
        capacidad = TemporalFactorDetector._assess_operational_capacity(fecha, demanda)
        riesgos = TemporalFactorDetector._identify_logistical_risks(fecha, eventos)

        # Impactos calculados
        tiempo_impacto = TemporalFactorDetector._calculate_time_impact(
            fecha, demanda, clima, trafico, capacidad
        )

        costo_impacto = TemporalFactorDetector._calculate_cost_impact(
            fecha, demanda, eventos, capacidad
        )

        # Predicciones avanzadas
        predicciones = TemporalFactorDetector._generate_predictions(
            fecha, demanda, eventos, clima
        )

        result = {
            # Eventos y temporadas
            'eventos_detectados': eventos['eventos'],
            'es_temporada_alta': demanda > 1.8,
            'es_temporada_critica': demanda > 2.5,
            'factor_demanda': demanda,
            'categoria_temporada': TemporalFactorDetector._categorize_season(demanda),

            # Clima y ambiente
            'condicion_clima': clima['condicion'],
            'temperatura_celsius': clima['temperatura'],
            'probabilidad_lluvia': clima['probabilidad_lluvia'],
            'indice_calor': clima.get('indice_calor', 'Normal'),
            'viento_esperado': clima.get('viento_kmh', 15),

            # Tráfico y movilidad
            'trafico_nivel': trafico['nivel'],
            'congestion_score': trafico['score'],
            'zonas_criticas': trafico['zonas_criticas'],
            'horas_pico_evitar': trafico['horas_evitar'],

            # Impactos operativos
            'impacto_tiempo_extra_horas': tiempo_impacto,
            'impacto_costo_extra_pct': costo_impacto,
            'capacidad_operativa_pct': capacidad['disponible_pct'],
            'saturacion_red': capacidad['saturacion'],

            # Riesgos y alertas
            'riesgos_logisticos': riesgos,
            'criticidad_logistica': TemporalFactorDetector._assess_criticality(
                demanda, tiempo_impacto, costo_impacto, riesgos
            ),
            'alertas_operativas': TemporalFactorDetector._generate_alerts(
                demanda, clima, trafico, capacidad
            ),

            # Predicciones
            'predicciones_proximos_dias': predicciones,
            'recomendaciones_estrategicas': TemporalFactorDetector._generate_recommendations(
                fecha, demanda, eventos, clima, capacidad
            ),

            # Metadatos
            'fecha_analisis': fecha.isoformat(),
            'confianza_prediccion': TemporalFactorDetector._calculate_prediction_confidence(
                fecha, eventos, clima
            ),
            'ultima_actualizacion': datetime.now().isoformat()
        }

        logger.info(f"🕒 Factores temporales detectados para {fecha.date()}: "
                    f"demanda={demanda:.1f}, criticidad={result['criticidad_logistica']}")

        return result

    @staticmethod
    def detect_seasonal_factors(fecha: datetime) -> Dict[str, Any]:
        """🎄 Método simplificado para compatibilidad con external_factors_repository"""

        eventos = TemporalFactorDetector._detect_events(fecha)
        demanda = TemporalFactorDetector._calculate_demand_factor(fecha, eventos)
        clima = TemporalFactorDetector._predict_weather_impact(fecha)
        trafico = TemporalFactorDetector._calculate_traffic_patterns(fecha)

        return {
            'eventos_detectados': eventos['eventos'],
            'factor_demanda': demanda,
            'condicion_clima': clima['condicion'],
            'trafico_nivel': trafico['nivel'],
            'impacto_tiempo_extra': min(2.0, demanda * 0.5),  # Máximo 2 horas extra
            'criticidad_logistica': TemporalFactorDetector._assess_criticality(
                demanda, demanda * 0.5, demanda * 5, eventos['eventos']
            )
        }

    @staticmethod
    def _detect_events(fecha: datetime) -> Dict[str, Any]:
        """🎄 Detección avanzada de eventos con sub-períodos"""

        mes = fecha.month
        dia = fecha.day
        eventos_detectados = []
        intensidad_maxima = 1.0

        # Navidad y Fin de Año (Diciembre) - MEJORADO
        if mes == 12:
            if 1 <= dia <= 15:
                eventos_detectados.extend(['Pre_Navidad', 'Preparacion_Diciembre'])
                intensidad_maxima = 1.6  # Era 1.8, reducido
            elif 16 <= dia <= 19:
                eventos_detectados.extend(['Pre_Navidad_Intenso', 'Compras_Ultimo_Momento'])
                intensidad_maxima = 2.0  # Era 2.3, reducido
            elif 20 <= dia <= 24:
                eventos_detectados.extend(['Navidad_Pico', 'Emergencia_Regalos'])
                intensidad_maxima = 2.5  # Era 3.0, reducido
            elif dia == 25:
                eventos_detectados.extend(['Navidad', 'Dia_Navidad'])
                intensidad_maxima = 0.3  # Todo cerrado
            elif 26 <= dia <= 30:
                eventos_detectados.extend(['Post_Navidad', 'Preparacion_Fin_Ano'])
                intensidad_maxima = 1.8  # Era 2.0, reducido
            elif dia == 31:
                eventos_detectados.extend(['Fin_Ano', 'Nochevieja'])
                intensidad_maxima = 0.4

        # Buen Fin y Black Friday (Noviembre) - MEJORADO
        elif mes == 11:
            if 10 <= dia <= 14:
                eventos_detectados.extend(['Pre_Buen_Fin', 'Preparacion_BF'])
                intensidad_maxima = 1.8  # Era 2.0, reducido
            elif 15 <= dia <= 17:  # Buen Fin México
                eventos_detectados.extend(['Buen_Fin_Pico', 'Black_Friday_MX'])
                intensidad_maxima = 2.8  # Era 3.2, reducido
            elif 18 <= dia <= 25:
                eventos_detectados.extend(['Post_Buen_Fin', 'Cyber_Monday_Extended'])
                intensidad_maxima = 2.0  # Era 2.2, reducido
            elif 26 <= dia <= 30:
                eventos_detectados.extend(['Pre_Diciembre', 'Preparacion_Navidad'])
                intensidad_maxima = 1.4  # Era 1.6, reducido

        # San Valentín (Febrero) - MEJORADO
        elif mes == 2:
            if dia == 14:
                eventos_detectados.extend(['San_Valentin', 'Dia_Amor'])
                intensidad_maxima = 2.0  # Era 2.4, reducido
            elif 12 <= dia <= 13:
                eventos_detectados.extend(['Pre_San_Valentin', 'Compras_Ultimo_Momento'])
                intensidad_maxima = 1.8  # Era 2.0, reducido
            elif 10 <= dia <= 11:
                eventos_detectados.extend(['Preparacion_San_Valentin'])
                intensidad_maxima = 1.3  # Era 1.5, reducido

        # Día de las Madres (Mayo) - MEJORADO
        elif mes == 5:
            segundo_domingo = TemporalFactorDetector._get_nth_weekday(fecha.year, 5, 6, 2)

            if segundo_domingo and dia == segundo_domingo:
                eventos_detectados.extend(['Dia_Madres', 'Dia_Madre_Pico'])
                intensidad_maxima = 2.4  # Era 2.8, reducido
            elif segundo_domingo and segundo_domingo - 3 <= dia <= segundo_domingo - 1:
                eventos_detectados.extend(['Pre_Dia_Madres', 'Compras_Emergencia'])
                intensidad_maxima = 2.2  # Era 2.5, reducido
            elif segundo_domingo and segundo_domingo - 7 <= dia <= segundo_domingo - 4:
                eventos_detectados.extend(['Preparacion_Dia_Madres'])
                intensidad_maxima = 1.6  # Era 1.8, reducido

        # Fiestas Patrias (Septiembre)
        elif mes == 9:
            if 15 <= dia <= 16:
                eventos_detectados.extend(['Fiestas_Patrias', 'Independencia_Mexico'])
                intensidad_maxima = 1.5  # Era 1.7, reducido
            elif 13 <= dia <= 14:
                eventos_detectados.extend(['Pre_Fiestas_Patrias'])
                intensidad_maxima = 1.2  # Era 1.4, reducido

        # Halloween y Día de Muertos (Octubre-Noviembre)
        elif mes == 10:
            if 28 <= dia <= 31:
                eventos_detectados.extend(['Halloween', 'Preparacion_Dia_Muertos'])
                intensidad_maxima = 1.4  # Era 1.6, reducido
        elif mes == 11 and 1 <= dia <= 2:
            eventos_detectados.extend(['Dia_Muertos', 'Tradicion_Mexico'])
            intensidad_maxima = 1.3  # Era 1.5, reducido

        # Día de Reyes (Enero)
        elif mes == 1:
            if dia == 1:
                eventos_detectados.extend(['Ano_Nuevo', 'Nuevo_Ano'])
                intensidad_maxima = 0.4  # Todo cerrado
            elif dia == 6:
                eventos_detectados.extend(['Dia_Reyes', 'Reyes_Magos'])
                intensidad_maxima = 2.0  # Era 2.2, reducido
            elif 2 <= dia <= 5:
                eventos_detectados.extend(['Post_Ano_Nuevo', 'Preparacion_Reyes'])
                intensidad_maxima = 1.6  # Era 1.8, reducido
            elif 7 <= dia <= 15:
                eventos_detectados.extend(['Post_Reyes', 'Enero_Cuesta_Arriba'])
                intensidad_maxima = 0.7  # Baja después de gastos

        # Regreso a clases (Agosto)
        elif mes == 8:
            if 15 <= dia <= 31:
                eventos_detectados.extend(['Regreso_Clases', 'Back_to_School'])
                intensidad_maxima = 1.6  # Era 1.8, reducido

        # Agregar eventos por configuración de settings (con factor reducido)
        if mes in settings.EVENTOS_TEMPORADA:
            for evento_config in settings.EVENTOS_TEMPORADA[mes]:
                evento_dias = evento_config['dias']
                if evento_dias[0] <= dia <= evento_dias[1]:
                    eventos_detectados.append(evento_config['evento'])
                    # Aplicar factor de reducción del 15%
                    factor_reducido = evento_config['factor_demanda'] * 0.85
                    intensidad_maxima = max(intensidad_maxima, factor_reducido)

        return {
            'eventos': eventos_detectados,
            'intensidad_maxima': intensidad_maxima,
            'es_evento_mayor': intensidad_maxima >= 2.0,
            'categoria_evento': TemporalFactorDetector._categorize_event(intensidad_maxima)
        }

    @staticmethod
    def _calculate_demand_factor(fecha: datetime, eventos: Dict[str, Any]) -> float:
        """📈 Calcula factor de demanda con algoritmos más conservadores"""

        base_factor = eventos['intensidad_maxima']

        # Ajustes por día de la semana (más conservadores)
        weekday = fecha.weekday()  # 0=Lunes, 6=Domingo

        weekday_multipliers = {
            0: 0.98,  # Lunes - Era 0.95, menos penalización
            1: 1.02,  # Martes - Era 1.05, más conservador
            2: 1.05,  # Miércoles - Era 1.10, más conservador
            3: 1.08,  # Jueves - Era 1.15, más conservador
            4: 1.15,  # Viernes - Era 1.25, más conservador
            5: 1.12,  # Sábado - Era 1.20, más conservador
            6: 0.90  # Domingo - Era 0.85, menos penalización
        }

        weekday_factor = weekday_multipliers[weekday]

        # Ajustes por proximidad a quincena (más conservadores)
        day_of_month = fecha.day
        quincena_factor = 1.0

        if 1 <= day_of_month <= 3:  # Inicio de mes
            quincena_factor = 1.08  # Era 1.15, más conservador
        elif 14 <= day_of_month <= 16:  # Quincena
            quincena_factor = 1.15  # Era 1.25, más conservador
        elif 28 <= day_of_month <= 31:  # Fin de mes
            quincena_factor = 1.05  # Era 1.10, más conservador

        # Ajustes estacionales macro (más conservadores)
        mes = fecha.month
        seasonal_multipliers = {
            1: 0.90,  # Enero - Era 0.85, menos penalización
            2: 0.95,  # Febrero - Era 0.90, menos penalización
            3: 1.02,  # Marzo - Era 1.05, más conservador
            4: 1.05,  # Abril - Era 1.10, más conservador
            5: 1.12,  # Mayo - Era 1.20, más conservador
            6: 1.00,  # Junio - igual
            7: 0.98,  # Julio - Era 0.95, menos penalización
            8: 1.08,  # Agosto - Era 1.15, más conservador
            9: 1.02,  # Septiembre - Era 1.05, más conservador
            10: 1.05,  # Octubre - Era 1.10, más conservador
            11: 1.20,  # Noviembre - Era 1.30, más conservador
            12: 1.25  # Diciembre - Era 1.35, más conservador
        }

        seasonal_factor = seasonal_multipliers[mes]

        # Factor de año (crecimiento anual más conservador)
        year_factor = 1.0 + (fecha.year - 2024) * 0.02  # Era 0.03, ahora 2%

        # Cálculo final
        final_factor = (base_factor * weekday_factor * quincena_factor *
                        seasonal_factor * year_factor)

        # Límites de seguridad más conservadores
        final_factor = max(0.4, min(3.5, final_factor))  # Era 0.3-4.0, ahora 0.4-3.5

        return round(final_factor, 2)

    @staticmethod
    def _predict_weather_impact(fecha: datetime, codigo_postal: str = None) -> Dict[str, Any]:
        """🌤️ Predicción avanzada de impacto climático"""

        mes = fecha.month
        dia = fecha.day

        # Patrones climáticos México por mes
        climate_patterns = {
            1: {'temp': 16, 'lluvia': 10, 'condicion': 'Frio_Seco'},
            2: {'temp': 18, 'lluvia': 15, 'condicion': 'Templado_Seco'},
            3: {'temp': 22, 'lluvia': 20, 'condicion': 'Templado'},
            4: {'temp': 25, 'lluvia': 25, 'condicion': 'Calido_Seco'},
            5: {'temp': 27, 'lluvia': 40, 'condicion': 'Calido'},
            6: {'temp': 25, 'lluvia': 70, 'condicion': 'Lluvioso_Inicio'},
            7: {'temp': 24, 'lluvia': 85, 'condicion': 'Lluvioso_Intenso'},
            8: {'temp': 24, 'lluvia': 80, 'condicion': 'Lluvioso_Intenso'},
            9: {'temp': 23, 'lluvia': 75, 'condicion': 'Lluvioso'},
            10: {'temp': 21, 'lluvia': 45, 'condicion': 'Templado_Humedo'},
            11: {'temp': 19, 'lluvia': 25, 'condicion': 'Templado_Seco'},
            12: {'temp': 17, 'lluvia': 15, 'condicion': 'Frio_Seco'}
        }

        base_climate = climate_patterns[mes]

        # Variaciones por región (aproximación por CP)
        if codigo_postal:
            cp_region = TemporalFactorDetector._identify_region_by_cp(codigo_postal)
            base_climate = TemporalFactorDetector._adjust_climate_by_region(
                base_climate, cp_region
            )

        # Variaciones aleatorias realistas día a día (más estables)
        import random
        random.seed(fecha.year * 1000 + fecha.timetuple().tm_yday)  # Seed determinístico

        temp_variation = random.randint(-2, 2)  # Era -3,3, ahora más estable
        lluvia_variation = random.randint(-10, 10)  # Era -15,15, ahora más estable

        final_temp = base_climate['temp'] + temp_variation
        final_lluvia = max(0, min(100, base_climate['lluvia'] + lluvia_variation))

        # Determinar condición específica
        condicion = base_climate['condicion']
        if final_lluvia > 70:
            condicion = 'Lluvioso_Intenso'
        elif final_lluvia > 50:
            condicion = 'Lluvioso'
        elif final_temp < 10:
            condicion = 'Frio_Extremo'
        elif final_temp > 35:
            condicion = 'Calor_Extremo'

        # Índice de calor
        indice_calor = 'Normal'
        if final_temp > 32:
            indice_calor = 'Alto'
        elif final_temp > 38:
            indice_calor = 'Extremo'
        elif final_temp < 5:
            indice_calor = 'Frio_Peligroso'

        return {
            'condicion': condicion,
            'temperatura': final_temp,
            'probabilidad_lluvia': final_lluvia,
            'indice_calor': indice_calor,
            'viento_kmh': random.randint(10, 20),  # Era 10-25, más estable
            'impacto_logistico': TemporalFactorDetector._assess_weather_logistics_impact(
                condicion, final_temp, final_lluvia
            )
        }

    @staticmethod
    def _calculate_traffic_patterns(fecha: datetime) -> Dict[str, Any]:
        """🚦 Patrones de tráfico avanzados México (más conservadores)"""

        hora = fecha.hour
        weekday = fecha.weekday()
        mes = fecha.month

        # Patrones base por hora (más conservadores)
        hourly_traffic = {
            0: 0.2, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.2, 5: 0.3,
            6: 0.4, 7: 0.7, 8: 0.8, 9: 0.6, 10: 0.5, 11: 0.6,  # Era 0.5,0.8,0.9,0.7
            12: 0.7, 13: 0.8, 14: 0.7, 15: 0.6, 16: 0.5, 17: 0.7,
            18: 0.8, 19: 0.8, 20: 0.6, 21: 0.4, 22: 0.3, 23: 0.3  # Era 0.8,0.9,0.9,0.7
        }

        base_congestion = hourly_traffic[hora]

        # Ajustes por día de semana (más conservadores)
        if weekday < 5:  # Lunes a Viernes
            if 7 <= hora <= 10 or 17 <= hora <= 20:
                base_congestion = min(1.0, base_congestion * 1.2)  # Era 1.3, más conservador
        elif weekday == 5:  # Sábado
            if 10 <= hora <= 18:
                base_congestion = min(1.0, base_congestion * 1.1)  # Era 1.2, más conservador
        else:  # Domingo
            base_congestion *= 0.8  # Era 0.7, menos penalización

        # Ajustes por temporadas especiales (más conservadores)
        if mes == 12:  # Diciembre
            base_congestion = min(1.0, base_congestion * 1.2)  # Era 1.4, más conservador
        elif mes == 11:  # Noviembre (Buen Fin)
            base_congestion = min(1.0, base_congestion * 1.15)  # Era 1.3, más conservador

        # Determinar nivel categórico (umbrales ajustados)
        if base_congestion >= 0.85:  # Era 0.9, más permisivo
            nivel = 'Muy_Alto'
        elif base_congestion >= 0.65:  # Era 0.7, más permisivo
            nivel = 'Alto'
        elif base_congestion >= 0.45:  # Era 0.5, más permisivo
            nivel = 'Moderado'
        elif base_congestion >= 0.25:  # Era 0.3, más permisivo
            nivel = 'Bajo'
        else:
            nivel = 'Muy_Bajo'

        # Zonas críticas México (solo si muy alto)
        zonas_criticas = []
        if base_congestion >= 0.8:  # Era 0.7, más restrictivo
            zonas_criticas = [
                'CDMX_Centro', 'Periferico_Sur', 'Insurgentes',
                'Circuito_Interior', 'Viaducto', 'Ejes_Viales'
            ]

        # Horas recomendadas para evitar
        horas_evitar = []
        if weekday < 5:  # Días laborales
            horas_evitar = ['07:00-10:00', '17:00-20:00']

        return {
            'nivel': nivel,
            'score': round(base_congestion, 2),
            'zonas_criticas': zonas_criticas,
            'horas_evitar': horas_evitar,
            'factor_velocidad': round(1.0 - (base_congestion * 0.3), 2)  # Era 0.4, más conservador
        }

    @staticmethod
    def _assess_operational_capacity(fecha: datetime,
                                     demand_factor: float) -> Dict[str, Any]:
        """🏭 Evaluación de capacidad operativa (más conservadora)"""

        # Capacidad base (100% en condiciones normales)
        base_capacity = 100.0

        # Ajustes por día de semana
        weekday = fecha.weekday()
        if weekday < 5:  # Lunes a Viernes
            weekday_capacity = 100.0
        elif weekday == 5:  # Sábado
            weekday_capacity = 90.0  # Era 85.0, más optimista
        else:  # Domingo
            weekday_capacity = 70.0  # Era 60.0, más optimista

        # Ajustes por horario
        hora = fecha.hour
        if 9 <= hora <= 18:  # Horario normal
            hour_capacity = 100.0
        elif 6 <= hora <= 8 or 19 <= hora <= 21:  # Horario extendido
            hour_capacity = 80.0  # Era 75.0, más optimista
        else:  # Horario nocturno/madrugada
            hour_capacity = 35.0  # Era 25.0, más optimista

        # Impacto de la demanda en la capacidad (más conservador)
        if demand_factor > 2.8:  # Era 2.5, umbral más alto
            demand_impact = 0.7  # Era 0.6, menos penalización
        elif demand_factor > 2.3:  # Era 2.0, umbral más alto
            demand_impact = 0.85  # Era 0.8, menos penalización
        elif demand_factor > 1.8:  # Era 1.5, umbral más alto
            demand_impact = 0.95  # Era 0.9, menos penalización
        else:  # Demanda normal
            demand_impact = 1.0

        # Cálculo final
        effective_capacity = (weekday_capacity * hour_capacity / 100.0 *
                              demand_impact)

        # Saturación de red (más conservadora)
        utilization = min(100.0, (demand_factor * 40))  # Era 50%, más conservador
        saturation = 'Baja'

        if utilization >= 85:  # Era 90, más estricto
            saturation = 'Crítica'
        elif utilization >= 70:  # Era 75, más estricto
            saturation = 'Alta'
        elif utilization >= 55:  # Era 60, más estricto
            saturation = 'Media'

        return {
            'disponible_pct': round(effective_capacity, 1),
            'utilizacion_pct': round(utilization, 1),
            'saturacion': saturation,
            'holgura_operativa': round(max(0, 100 - utilization), 1)
        }

    # Métodos auxiliares (sin cambios significativos)
    @staticmethod
    def _get_nth_weekday(year: int, month: int, weekday: int, n: int) -> int:
        """📅 Obtiene el día del n-ésimo día de la semana en un mes"""
        try:
            first_day = datetime(year, month, 1)
            first_weekday = first_day.weekday()

            days_ahead = weekday - first_weekday
            if days_ahead <= 0:
                days_ahead += 7

            target_date = first_day + timedelta(days=days_ahead + (n - 1) * 7)

            if target_date.month != month:
                return None

            return target_date.day
        except Exception:
            return None

    @staticmethod
    def _identify_region_by_cp(codigo_postal: str) -> str:
        """🗺️ Identifica región por código postal"""
        if not codigo_postal or len(codigo_postal) < 2:
            return 'centro'

        cp_prefix = codigo_postal[:2]

        # Mapeo simplificado por prefijos CP México
        region_map = {
            '01': 'cdmx_norte', '02': 'cdmx_centro', '03': 'cdmx_centro',
            '04': 'cdmx_sur', '05': 'cdmx_poniente', '06': 'cdmx_centro',
            '07': 'cdmx_norte', '08': 'cdmx_oriente', '09': 'cdmx_sur',
            '10': 'estado_mexico', '11': 'estado_mexico', '12': 'estado_mexico',
            '20': 'oaxaca', '21': 'puebla', '22': 'puebla',
            '44': 'jalisco', '45': 'jalisco', '64': 'nuevo_leon',
            '80': 'sinaloa', '81': 'sinaloa'
        }

        return region_map.get(cp_prefix, 'centro')

    @staticmethod
    def _adjust_climate_by_region(base_climate: Dict[str, Any],
                                  region: str) -> Dict[str, Any]:
        """🌡️ Ajusta clima por región"""

        # Ajustes regionales
        regional_adjustments = {
            'cdmx_centro': {'temp': 0, 'lluvia': 0},
            'cdmx_norte': {'temp': -1, 'lluvia': -5},
            'cdmx_sur': {'temp': -2, 'lluvia': 10},
            'nuevo_leon': {'temp': 3, 'lluvia': -20},
            'jalisco': {'temp': 2, 'lluvia': -10},
            'sinaloa': {'temp': 5, 'lluvia': -30},
            'oaxaca': {'temp': 4, 'lluvia': 15},
            'puebla': {'temp': -3, 'lluvia': 5}
        }

        adjustment = regional_adjustments.get(region, {'temp': 0, 'lluvia': 0})

        adjusted_climate = base_climate.copy()
        adjusted_climate['temp'] += adjustment['temp']
        adjusted_climate['lluvia'] = max(0, min(100,
                                                adjusted_climate['lluvia'] + adjustment['lluvia']))

        return adjusted_climate

    @staticmethod
    def _assess_weather_logistics_impact(condicion: str, temp: int,
                                         lluvia: int) -> Dict[str, float]:
        """🌦️ Evalúa impacto logístico del clima (más conservador)"""

        # Multiplicadores de impacto (más conservadores)
        time_impact = 1.0
        cost_impact = 1.0
        reliability_impact = 1.0

        # Impacto por lluvia (más conservador)
        if lluvia > 80:
            time_impact *= 1.4  # Era 1.6, más conservador
            reliability_impact *= 0.8  # Era 0.7, menos penalización
        elif lluvia > 60:
            time_impact *= 1.2  # Era 1.3, más conservador
            reliability_impact *= 0.9  # Era 0.85, menos penalización
        elif lluvia > 40:
            time_impact *= 1.1  # Era 1.15, más conservador
            reliability_impact *= 0.97  # Era 0.95, menos penalización

        # Impacto por temperatura (más conservador)
        if temp > 38 or temp < 5:
            time_impact *= 1.2  # Era 1.3, más conservador
            cost_impact *= 1.15  # Era 1.2, más conservador
            reliability_impact *= 0.85  # Era 0.8, menos penalización
        elif temp > 35 or temp < 10:
            time_impact *= 1.1  # Era 1.15, más conservador
            cost_impact *= 1.05  # Era 1.1, más conservador
            reliability_impact *= 0.95  # Era 0.9, menos penalización

        return {
            'factor_tiempo': round(time_impact, 2),
            'factor_costo': round(cost_impact, 2),
            'factor_confiabilidad': round(reliability_impact, 2)
        }

    @staticmethod
    def _categorize_season(demand_factor: float) -> str:
        """📊 Categoriza temporada por factor de demanda (umbrales ajustados)"""
        if demand_factor >= 2.8:  # Era 3.0, más accesible
            return 'Hiper_Critica'
        elif demand_factor >= 2.3:  # Era 2.5, más accesible
            return 'Critica'
        elif demand_factor >= 1.8:  # Era 2.0, más accesible
            return 'Alta'
        elif demand_factor >= 1.4:  # Era 1.5, más accesible
            return 'Elevada'
        elif demand_factor >= 1.1:  # Era 1.2, más accesible
            return 'Media_Alta'
        else:
            return 'Normal'

    @staticmethod
    def _categorize_event(intensidad: float) -> str:
        """🎯 Categoriza evento por intensidad (umbrales ajustados)"""
        if intensidad >= 2.8:  # Era 3.0, más accesible
            return 'Mega_Evento'
        elif intensidad >= 2.3:  # Era 2.5, más accesible
            return 'Evento_Mayor'
        elif intensidad >= 1.8:  # Era 2.0, más accesible
            return 'Evento_Significativo'
        elif intensidad >= 1.3:  # Era 1.5, más accesible
            return 'Evento_Menor'
        else:
            return 'Dia_Normal'

    @staticmethod
    def _identify_logistical_risks(fecha: datetime,
                                   eventos: Dict[str, Any]) -> List[str]:
        """⚠️ Identifica riesgos logísticos específicos (más conservador)"""

        riesgos = []

        # Riesgos por temporada alta (umbral más alto)
        if eventos['intensidad_maxima'] > 2.3:  # Era 2.5, umbral más bajo
            riesgos.extend([
                'saturacion_red_distribucion',
                'agotamiento_inventario',
                'demoras_picking_packing'
            ])

            # Solo agregar riesgo crítico si realmente es muy alto
            if eventos['intensidad_maxima'] > 2.8:
                riesgos.append('congestion_centros_distribucion')

        # Riesgos por día de semana
        if fecha.weekday() == 4:  # Viernes
            riesgos.append('pico_demanda_fin_semana')
        elif fecha.weekday() == 6:  # Domingo
            riesgos.append('capacidad_operativa_limitada')

        # Riesgos por mes (más selectivos)
        mes = fecha.month
        if mes in [7, 8, 9]:  # Solo meses de lluvia intensa
            riesgos.extend([
                'retrasos_clima_adverso',
                'dificultades_ultima_milla'
            ])

            # Solo en pico de lluvia
            if mes in [7, 8]:
                riesgos.append('inundaciones_vias_acceso')

        if mes == 12 and fecha.day > 20:  # Solo en pico navideño
            riesgos.extend([
                'personal_reducido_vacaciones',
                'congestion_extrema_ciudades'
            ])

        return riesgos

    @staticmethod
    def _calculate_time_impact(fecha: datetime, demanda: float,
                               clima: Dict[str, Any], trafico: Dict[str, Any],
                               capacidad: Dict[str, Any]) -> float:
        """⏱️ Calcula impacto total en tiempo (más conservador)"""

        base_impact = 0.0

        # Impacto por demanda (umbrales más altos)
        if demanda > 2.8:  # Era 2.5, umbral más alto
            base_impact += 2.0  # Era 3.0, más conservador
        elif demanda > 2.3:  # Era 2.0, umbral más alto
            base_impact += 1.5  # Era 2.0, más conservador
        elif demanda > 1.8:  # Era 1.5, umbral más alto
            base_impact += 0.8  # Era 1.0, más conservador

        # Impacto por clima (más conservador)
        weather_impact = clima.get('impacto_logistico', {})
        time_factor = weather_impact.get('factor_tiempo', 1.0)
        base_impact += (time_factor - 1.0) * 1.5  # Era 2.0, más conservador

        # Impacto por tráfico (más conservador)
        traffic_score = trafico['score']
        if traffic_score > 0.85:  # Era 0.8, umbral más alto
            base_impact += 1.5  # Era 2.0, más conservador
        elif traffic_score > 0.65:  # Era 0.6, umbral más alto
            base_impact += 0.8  # Era 1.0, más conservador

        # Impacto por capacidad limitada (más conservador)
        if capacidad['disponible_pct'] < 65:  # Era 70, umbral más bajo
            base_impact += 1.5  # Era 2.0, más conservador
        elif capacidad['disponible_pct'] < 80:  # Era 85, umbral más bajo
            base_impact += 0.8  # Era 1.0, más conservador

        return round(max(0.0, base_impact), 1)

    @staticmethod
    def _calculate_cost_impact(fecha: datetime, demanda: float,
                               eventos: Dict[str, Any],
                               capacidad: Dict[str, Any]) -> float:
        """💰 Calcula impacto en costos (más conservador)"""

        base_impact = 0.0

        # Costos premium por alta demanda (más conservador)
        if demanda > 2.8:  # Era 2.5, umbral más alto
            base_impact += 18.0  # Era 25.0, más conservador
        elif demanda > 2.3:  # Era 2.0, umbral más alto
            base_impact += 12.0  # Era 15.0, más conservador
        elif demanda > 1.8:  # Era 1.5, umbral más alto
            base_impact += 6.0  # Era 8.0, más conservador

        # Costos por eventos especiales (más conservador)
        if eventos['es_evento_mayor']:
            base_impact += 8.0  # Era 10.0, más conservador

        # Costos por capacidad limitada (más conservador)
        if capacidad['saturacion'] == 'Crítica':
            base_impact += 15.0  # Era 20.0, más conservador
        elif capacidad['saturacion'] == 'Alta':
            base_impact += 9.0  # Era 12.0, más conservador

        # Costos adicionales fin de semana (más conservador)
        if fecha.weekday() >= 5:
            base_impact += 3.0  # Era 5.0, más conservador

        return round(base_impact, 1)

    @staticmethod
    def _assess_criticality(demanda: float, tiempo_impacto: float,
                            costo_impacto: float, riesgos: List[str]) -> str:
        """🎯 Evalúa criticidad general (umbrales ajustados)"""

        score = 0

        # Score por demanda (umbrales más altos)
        if demanda > 2.8:  # Era 2.8, igual
            score += 4
        elif demanda > 2.3:  # Era 2.3, igual
            score += 3
        elif demanda > 1.8:  # Era 1.8, igual
            score += 2
        elif demanda > 1.3:  # Era 1.3, igual
            score += 1

        # Score por impacto tiempo (umbrales más altos)
        if tiempo_impacto > 2.5:  # Era 3.0, umbral más bajo
            score += 3
        elif tiempo_impacto > 1.5:  # Era 2.0, umbral más bajo
            score += 2
        elif tiempo_impacto > 0.8:  # Era 1.0, umbral más bajo
            score += 1

        # Score por número de riesgos (más conservador)
        score += min(2, len(riesgos) // 3)  # Era //2, ahora más conservador

        # Determinar criticidad (umbrales ajustados)
        if score >= 7:  # Era 8, umbral más bajo
            return 'Crítica'
        elif score >= 5:  # Era 6, umbral más bajo
            return 'Alta'
        elif score >= 3:  # Era 4, umbral más bajo
            return 'Media'
        elif score >= 1:  # Era 2, umbral más bajo
            return 'Baja'
        else:
            return 'Normal'

    @staticmethod
    def _generate_alerts(demanda: float, clima: Dict[str, Any],
                         trafico: Dict[str, Any],
                         capacidad: Dict[str, Any]) -> List[str]:
        """🚨 Genera alertas operativas (umbrales ajustados)"""

        alertas = []

        if demanda > 2.8:  # Era 2.8, igual
            alertas.append('🔴 DEMANDA CRÍTICA: Activar protocolos de emergencia')
        elif demanda > 2.3:  # Era 2.3, igual
            alertas.append('🟡 DEMANDA ALTA: Monitorear capacidad de cerca')

        if clima['probabilidad_lluvia'] > 75:  # Era 70, umbral más alto
            alertas.append('🌧️ CLIMA ADVERSO: Preparar rutas alternativas')

        if trafico['score'] > 0.85:  # Era 0.8, umbral más alto
            alertas.append('🚦 TRÁFICO CRÍTICO: Evitar horas pico')

        if capacidad['saturacion'] == 'Crítica':
            alertas.append('⚠️ SATURACIÓN RED: Capacidad al límite')

        return alertas

    @staticmethod
    def _generate_predictions(fecha: datetime, demanda: float,
                              eventos: Dict[str, Any],
                              clima: Dict[str, Any]) -> List[Dict[str, Any]]:
        """🔮 Genera predicciones para próximos días"""

        predicciones = []

        for i in range(1, 4):  # Próximos 3 días
            fecha_pred = fecha + timedelta(days=i)

            # Predicción simplificada (más conservadora)
            pred_demanda = demanda * 0.95  # Era 0.9, decremento más lento

            predicciones.append({
                'fecha': fecha_pred.date().isoformat(),
                'demanda_predicha': round(pred_demanda, 2),
                'tendencia': 'descendente' if pred_demanda < demanda else 'estable',
                'confianza': 0.85 - (i * 0.05)  # Era 0.8-(i*0.1), más conservador
            })

        return predicciones

    @staticmethod
    def _generate_recommendations(fecha: datetime, demanda: float,
                                  eventos: Dict[str, Any], clima: Dict[str, Any],
                                  capacidad: Dict[str, Any]) -> List[str]:
        """💡 Genera recomendaciones estratégicas (más selectivas)"""

        recomendaciones = []

        if demanda > 2.8:  # Era 2.5, umbral más alto
            recomendaciones.extend([
                '📈 Activar capacidad adicional de fulfillment',
                '🚚 Contratar flota externa adicional',
                '⏰ Extender horarios de operación'
            ])

            # Solo si es muy crítico
            if demanda > 3.0:
                recomendaciones.append('📦 Priorizar productos alta rotación')

        if clima['probabilidad_lluvia'] > 75:  # Era 60, umbral más alto
            recomendaciones.extend([
                '🌧️ Usar vehículos cubiertos',
                '🛣️ Activar rutas alternativas'
            ])

            # Solo si es muy probable
            if clima['probabilidad_lluvia'] > 85:
                recomendaciones.append('📱 Comunicar delays proactivamente')

        if capacidad['saturacion'] == 'Crítica':
            recomendaciones.extend([
                '🔄 Redistribuir carga entre nodos',
                '⚡ Implementar fulfillment distribuido'
            ])

        return recomendaciones

    @staticmethod
    def _calculate_prediction_confidence(fecha: datetime,
                                         eventos: Dict[str, Any],
                                         clima: Dict[str, Any]) -> float:
        """📊 Calcula confianza de las predicciones (más conservadora)"""

        base_confidence = 0.88  # Era 0.85, ligeramente más optimista

        # Reducir confianza en eventos impredecibles (más conservador)
        if eventos['intensidad_maxima'] > 2.8:  # Era 2.5, umbral más alto
            base_confidence -= 0.12  # Era 0.15, menos penalización

        # Reducir confianza en clima adverso (más conservador)
        if clima['probabilidad_lluvia'] > 75:  # Era 70, umbral más alto
            base_confidence -= 0.08  # Era 0.1, menos penalización

        # Reducir confianza en fines de semana (más conservador)
        if fecha.weekday() >= 5:
            base_confidence -= 0.03  # Era 0.05, menos penalización

        return round(max(0.6, base_confidence), 2)  # Era 0.5, mínimo más alto