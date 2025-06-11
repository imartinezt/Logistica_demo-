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
    def _detect_events(fecha: datetime) -> Dict[str, Any]:
        """🎄 Detección avanzada de eventos con sub-períodos"""

        mes = fecha.month
        dia = fecha.day
        eventos_detectados = []
        intensidad_maxima = 1.0

        # Navidad y Fin de Año (Diciembre)
        if mes == 12:
            if 1 <= dia <= 15:
                eventos_detectados.extend(['Pre_Navidad', 'Preparacion_Diciembre'])
                intensidad_maxima = 1.8
            elif 16 <= dia <= 19:
                eventos_detectados.extend(['Pre_Navidad_Intenso', 'Compras_Ultimo_Momento'])
                intensidad_maxima = 2.3
            elif 20 <= dia <= 24:
                eventos_detectados.extend(['Navidad_Pico', 'Emergencia_Regalos'])
                intensidad_maxima = 3.0  # Pico máximo del año
            elif dia == 25:
                eventos_detectados.extend(['Navidad', 'Dia_Navidad'])
                intensidad_maxima = 0.3  # Todo cerrado
            elif 26 <= dia <= 30:
                eventos_detectados.extend(['Post_Navidad', 'Preparacion_Fin_Ano'])
                intensidad_maxima = 2.0
            elif dia == 31:
                eventos_detectados.extend(['Fin_Ano', 'Nochevieja'])
                intensidad_maxima = 0.4

        # Buen Fin y Black Friday (Noviembre)
        elif mes == 11:
            if 10 <= dia <= 14:
                eventos_detectados.extend(['Pre_Buen_Fin', 'Preparacion_BF'])
                intensidad_maxima = 2.0
            elif 15 <= dia <= 17:  # Buen Fin México
                eventos_detectados.extend(['Buen_Fin_Pico', 'Black_Friday_MX'])
                intensidad_maxima = 3.2  # Segundo pico más alto
            elif 18 <= dia <= 25:
                eventos_detectados.extend(['Post_Buen_Fin', 'Cyber_Monday_Extended'])
                intensidad_maxima = 2.2
            elif 26 <= dia <= 30:
                eventos_detectados.extend(['Pre_Diciembre', 'Preparacion_Navidad'])
                intensidad_maxima = 1.6

        # San Valentín (Febrero)
        elif mes == 2:
            if dia == 14:
                eventos_detectados.extend(['San_Valentin', 'Dia_Amor'])
                intensidad_maxima = 2.4
            elif 12 <= dia <= 13:
                eventos_detectados.extend(['Pre_San_Valentin', 'Compras_Ultimo_Momento'])
                intensidad_maxima = 2.0
            elif 10 <= dia <= 11:
                eventos_detectados.extend(['Preparacion_San_Valentin'])
                intensidad_maxima = 1.5

        # Día de las Madres (Mayo)
        elif mes == 5:
            segundo_domingo = TemporalFactorDetector._get_nth_weekday(2025, 5, 6, 2)  # 2do domingo

            if dia == segundo_domingo:
                eventos_detectados.extend(['Dia_Madres', 'Dia_Madre_Pico'])
                intensidad_maxima = 2.8
            elif segundo_domingo - 3 <= dia <= segundo_domingo - 1:
                eventos_detectados.extend(['Pre_Dia_Madres', 'Compras_Emergencia'])
                intensidad_maxima = 2.5
            elif segundo_domingo - 7 <= dia <= segundo_domingo - 4:
                eventos_detectados.extend(['Preparacion_Dia_Madres'])
                intensidad_maxima = 1.8

        # Fiestas Patrias (Septiembre)
        elif mes == 9:
            if 15 <= dia <= 16:
                eventos_detectados.extend(['Fiestas_Patrias', 'Independencia_Mexico'])
                intensidad_maxima = 1.7
            elif 13 <= dia <= 14:
                eventos_detectados.extend(['Pre_Fiestas_Patrias'])
                intensidad_maxima = 1.4

        # Halloween y Día de Muertos (Octubre-Noviembre)
        elif mes == 10:
            if 28 <= dia <= 31:
                eventos_detectados.extend(['Halloween', 'Preparacion_Dia_Muertos'])
                intensidad_maxima = 1.6
        elif mes == 11 and 1 <= dia <= 2:
            eventos_detectados.extend(['Dia_Muertos', 'Tradicion_Mexico'])
            intensidad_maxima = 1.5

        # Día de Reyes (Enero)
        elif mes == 1:
            if dia == 1:
                eventos_detectados.extend(['Ano_Nuevo', 'Nuevo_Ano'])
                intensidad_maxima = 0.4  # Todo cerrado
            elif dia == 6:
                eventos_detectados.extend(['Dia_Reyes', 'Reyes_Magos'])
                intensidad_maxima = 2.2
            elif 2 <= dia <= 5:
                eventos_detectados.extend(['Post_Ano_Nuevo', 'Preparacion_Reyes'])
                intensidad_maxima = 1.8
            elif 7 <= dia <= 15:
                eventos_detectados.extend(['Post_Reyes', 'Enero_Cuesta_Arriba'])
                intensidad_maxima = 0.7  # Baja después de gastos

        # Regreso a clases (Agosto)
        elif mes == 8:
            if 15 <= dia <= 31:
                eventos_detectados.extend(['Regreso_Clases', 'Back_to_School'])
                intensidad_maxima = 1.8

        # Agregar eventos por configuración de settings
        if mes in settings.EVENTOS_TEMPORADA:
            for evento_config in settings.EVENTOS_TEMPORADA[mes]:
                evento_dias = evento_config['dias']
                if evento_dias[0] <= dia <= evento_dias[1]:
                    eventos_detectados.append(evento_config['evento'])
                    intensidad_maxima = max(intensidad_maxima, evento_config['factor_demanda'])

        return {
            'eventos': eventos_detectados,
            'intensidad_maxima': intensidad_maxima,
            'es_evento_mayor': intensidad_maxima >= 2.0,
            'categoria_evento': TemporalFactorDetector._categorize_event(intensidad_maxima)
        }

    @staticmethod
    def _calculate_demand_factor(fecha: datetime, eventos: Dict[str, Any]) -> float:
        """📈 Calcula factor de demanda con algoritmos avanzados"""

        base_factor = eventos['intensidad_maxima']

        # Ajustes por día de la semana
        weekday = fecha.weekday()  # 0=Lunes, 6=Domingo

        weekday_multipliers = {
            0: 0.95,  # Lunes - arranque lento
            1: 1.05,  # Martes - día productivo
            2: 1.10,  # Miércoles - pico mid-week
            3: 1.15,  # Jueves - preparación fin de semana
            4: 1.25,  # Viernes - máximo día de compras
            5: 1.20,  # Sábado - compras familiares
            6: 0.85  # Domingo - día de descanso
        }

        weekday_factor = weekday_multipliers[weekday]

        # Ajustes por proximidad a quincena
        day_of_month = fecha.day
        quincena_factor = 1.0

        if 1 <= day_of_month <= 3:  # Inicio de mes
            quincena_factor = 1.15
        elif 14 <= day_of_month <= 16:  # Quincena
            quincena_factor = 1.25
        elif 28 <= day_of_month <= 31:  # Fin de mes
            quincena_factor = 1.10

        # Ajustes estacionales macro
        mes = fecha.month
        seasonal_multipliers = {
            1: 0.85,  # Enero - post navidad
            2: 0.90,  # Febrero - San Valentín
            3: 1.05,  # Marzo - primavera
            4: 1.10,  # Abril - temporada fuerte
            5: 1.20,  # Mayo - Día Madres
            6: 1.00,  # Junio - normal
            7: 0.95,  # Julio - vacaciones
            8: 1.15,  # Agosto - regreso a clases
            9: 1.05,  # Septiembre - fiestas patrias
            10: 1.10,  # Octubre - Halloween
            11: 1.30,  # Noviembre - Buen Fin
            12: 1.35  # Diciembre - Navidad
        }

        seasonal_factor = seasonal_multipliers[mes]

        # Factor de año (crecimiento anual estimado)
        year_factor = 1.0 + (fecha.year - 2024) * 0.03  # 3% crecimiento anual

        # Cálculo final
        final_factor = (base_factor * weekday_factor * quincena_factor *
                        seasonal_factor * year_factor)

        # Límites de seguridad
        final_factor = max(0.3, min(4.0, final_factor))

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

        # Variaciones aleatorias realistas día a día
        import random
        random.seed(fecha.year * 1000 + fecha.timetuple().tm_yday)  # Seed determinístico

        temp_variation = random.randint(-3, 3)
        lluvia_variation = random.randint(-15, 15)

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
            'viento_kmh': random.randint(10, 25),
            'impacto_logistico': TemporalFactorDetector._assess_weather_logistics_impact(
                condicion, final_temp, final_lluvia
            )
        }

    @staticmethod
    def _calculate_traffic_patterns(fecha: datetime) -> Dict[str, Any]:
        """🚦 Patrones de tráfico avanzados México"""

        hora = fecha.hour
        weekday = fecha.weekday()
        mes = fecha.month

        # Patrones base por hora
        hourly_traffic = {
            0: 0.2, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.2, 5: 0.3,
            6: 0.5, 7: 0.8, 8: 0.9, 9: 0.7, 10: 0.6, 11: 0.7,
            12: 0.8, 13: 0.9, 14: 0.8, 15: 0.7, 16: 0.6, 17: 0.8,
            18: 0.9, 19: 0.9, 20: 0.7, 21: 0.5, 22: 0.4, 23: 0.3
        }

        base_congestion = hourly_traffic[hora]

        # Ajustes por día de semana
        if weekday < 5:  # Lunes a Viernes
            if 7 <= hora <= 10 or 17 <= hora <= 20:
                base_congestion = min(1.0, base_congestion * 1.3)  # Rush hours
        elif weekday == 5:  # Sábado
            if 10 <= hora <= 18:
                base_congestion = min(1.0, base_congestion * 1.2)  # Shopping hours
        else:  # Domingo
            base_congestion *= 0.7  # Día más tranquilo

        # Ajustes por temporadas especiales
        if mes == 12:  # Diciembre
            base_congestion = min(1.0, base_congestion * 1.4)
        elif mes == 11:  # Noviembre (Buen Fin)
            base_congestion = min(1.0, base_congestion * 1.3)

        # Determinar nivel categórico
        if base_congestion >= 0.9:
            nivel = 'Muy_Alto'
        elif base_congestion >= 0.7:
            nivel = 'Alto'
        elif base_congestion >= 0.5:
            nivel = 'Moderado'
        elif base_congestion >= 0.3:
            nivel = 'Bajo'
        else:
            nivel = 'Muy_Bajo'

        # Zonas críticas México
        zonas_criticas = []
        if base_congestion >= 0.7:
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
            'factor_velocidad': round(1.0 - (base_congestion * 0.4), 2)
        }

    @staticmethod
    def _assess_operational_capacity(fecha: datetime,
                                     demand_factor: float) -> Dict[str, Any]:
        """🏭 Evaluación de capacidad operativa"""

        # Capacidad base (100% en condiciones normales)
        base_capacity = 100.0

        # Ajustes por día de semana
        weekday = fecha.weekday()
        if weekday < 5:  # Lunes a Viernes
            weekday_capacity = 100.0
        elif weekday == 5:  # Sábado
            weekday_capacity = 85.0  # Menos personal
        else:  # Domingo
            weekday_capacity = 60.0  # Capacidad limitada

        # Ajustes por horario
        hora = fecha.hour
        if 9 <= hora <= 18:  # Horario normal
            hour_capacity = 100.0
        elif 6 <= hora <= 8 or 19 <= hora <= 21:  # Horario extendido
            hour_capacity = 75.0
        else:  # Horario nocturno/madrugada
            hour_capacity = 25.0

        # Impacto de la demanda en la capacidad
        if demand_factor > 2.5:  # Demanda crítica
            demand_impact = 0.6  # Capacidad reducida por saturación
        elif demand_factor > 2.0:  # Demanda alta
            demand_impact = 0.8
        elif demand_factor > 1.5:  # Demanda elevada
            demand_impact = 0.9
        else:  # Demanda normal
            demand_impact = 1.0

        # Cálculo final
        effective_capacity = (weekday_capacity * hour_capacity / 100.0 *
                              demand_impact)

        # Saturación de red
        utilization = min(100.0, (demand_factor * 50))  # 50% base utilization
        saturation = 'Baja'

        if utilization >= 90:
            saturation = 'Crítica'
        elif utilization >= 75:
            saturation = 'Alta'
        elif utilization >= 60:
            saturation = 'Media'

        return {
            'disponible_pct': round(effective_capacity, 1),
            'utilizacion_pct': round(utilization, 1),
            'saturacion': saturation,
            'holgura_operativa': round(max(0, 100 - utilization), 1)
        }

    # Métodos auxiliares
    @staticmethod
    def _get_nth_weekday(year: int, month: int, weekday: int, n: int) -> int:
        """📅 Obtiene el día del n-ésimo día de la semana en un mes"""
        first_day = datetime(year, month, 1)
        first_weekday = first_day.weekday()

        days_ahead = weekday - first_weekday
        if days_ahead <= 0:
            days_ahead += 7

        target_date = first_day + timedelta(days=days_ahead + (n - 1) * 7)

        if target_date.month != month:
            return None

        return target_date.day

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
        """🌦️ Evalúa impacto logístico del clima"""

        # Multiplicadores de impacto
        time_impact = 1.0
        cost_impact = 1.0
        reliability_impact = 1.0

        # Impacto por lluvia
        if lluvia > 80:
            time_impact *= 1.6
            reliability_impact *= 0.7
        elif lluvia > 60:
            time_impact *= 1.3
            reliability_impact *= 0.85
        elif lluvia > 40:
            time_impact *= 1.15
            reliability_impact *= 0.95

        # Impacto por temperatura
        if temp > 38 or temp < 5:
            time_impact *= 1.3
            cost_impact *= 1.2
            reliability_impact *= 0.8
        elif temp > 35 or temp < 10:
            time_impact *= 1.15
            cost_impact *= 1.1
            reliability_impact *= 0.9

        return {
            'factor_tiempo': round(time_impact, 2),
            'factor_costo': round(cost_impact, 2),
            'factor_confiabilidad': round(reliability_impact, 2)
        }

    @staticmethod
    def _categorize_season(demand_factor: float) -> str:
        """📊 Categoriza temporada por factor de demanda"""
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

    @staticmethod
    def _categorize_event(intensidad: float) -> str:
        """🎯 Categoriza evento por intensidad"""
        if intensidad >= 3.0:
            return 'Mega_Evento'
        elif intensidad >= 2.5:
            return 'Evento_Mayor'
        elif intensidad >= 2.0:
            return 'Evento_Significativo'
        elif intensidad >= 1.5:
            return 'Evento_Menor'
        else:
            return 'Dia_Normal'

    @staticmethod
    def _identify_logistical_risks(fecha: datetime,
                                   eventos: Dict[str, Any]) -> List[str]:
        """⚠️ Identifica riesgos logísticos específicos"""

        riesgos = []

        # Riesgos por temporada alta
        if eventos['intensidad_maxima'] > 2.5:
            riesgos.extend([
                'saturacion_red_distribucion',
                'agotamiento_inventario',
                'demoras_picking_packing',
                'congestion_centros_distribucion'
            ])

        # Riesgos por día de semana
        if fecha.weekday() == 4:  # Viernes
            riesgos.append('pico_demanda_fin_semana')
        elif fecha.weekday() == 6:  # Domingo
            riesgos.append('capacidad_operativa_limitada')

        # Riesgos por mes
        mes = fecha.month
        if mes in [6, 7, 8, 9]:  # Temporada lluvia
            riesgos.extend([
                'retrasos_clima_adverso',
                'inundaciones_vias_acceso',
                'dificultades_ultima_milla'
            ])

        if mes == 12:
            riesgos.extend([
                'personal_reducido_vacaciones',
                'cierre_proveedores_terceros',
                'congestion_extrema_ciudades'
            ])

        return riesgos

    @staticmethod
    def _calculate_time_impact(fecha: datetime, demanda: float,
                               clima: Dict[str, Any], trafico: Dict[str, Any],
                               capacidad: Dict[str, Any]) -> float:
        """⏱️ Calcula impacto total en tiempo"""

        base_impact = 0.0

        # Impacto por demanda
        if demanda > 2.5:
            base_impact += 3.0
        elif demanda > 2.0:
            base_impact += 2.0
        elif demanda > 1.5:
            base_impact += 1.0

        # Impacto por clima
        weather_impact = clima.get('impacto_logistico', {})
        time_factor = weather_impact.get('factor_tiempo', 1.0)
        base_impact += (time_factor - 1.0) * 2.0

        # Impacto por tráfico
        traffic_score = trafico['score']
        if traffic_score > 0.8:
            base_impact += 2.0
        elif traffic_score > 0.6:
            base_impact += 1.0

        # Impacto por capacidad limitada
        if capacidad['disponible_pct'] < 70:
            base_impact += 2.0
        elif capacidad['disponible_pct'] < 85:
            base_impact += 1.0

        return round(max(0.0, base_impact), 1)

    @staticmethod
    def _calculate_cost_impact(fecha: datetime, demanda: float,
                               eventos: Dict[str, Any],
                               capacidad: Dict[str, Any]) -> float:
        """💰 Calcula impacto en costos"""

        base_impact = 0.0

        # Costos premium por alta demanda
        if demanda > 2.5:
            base_impact += 25.0  # 25% extra
        elif demanda > 2.0:
            base_impact += 15.0
        elif demanda > 1.5:
            base_impact += 8.0

        # Costos por eventos especiales
        if eventos['es_evento_mayor']:
            base_impact += 10.0

        # Costos por capacidad limitada (overtime, etc)
        if capacidad['saturacion'] == 'Crítica':
            base_impact += 20.0
        elif capacidad['saturacion'] == 'Alta':
            base_impact += 12.0

        # Costos adicionales fin de semana
        if fecha.weekday() >= 5:
            base_impact += 5.0

        return round(base_impact, 1)

    @staticmethod
    def _assess_criticality(demanda: float, tiempo_impacto: float,
                            costo_impacto: float, riesgos: List[str]) -> str:
        """🎯 Evalúa criticidad general"""

        score = 0

        # Score por demanda
        if demanda > 2.8:
            score += 4
        elif demanda > 2.3:
            score += 3
        elif demanda > 1.8:
            score += 2
        elif demanda > 1.3:
            score += 1

        # Score por impacto tiempo
        if tiempo_impacto > 3.0:
            score += 3
        elif tiempo_impacto > 2.0:
            score += 2
        elif tiempo_impacto > 1.0:
            score += 1

        # Score por número de riesgos
        score += min(3, len(riesgos) // 2)

        # Determinar criticidad
        if score >= 8:
            return 'Crítica'
        elif score >= 6:
            return 'Alta'
        elif score >= 4:
            return 'Media'
        elif score >= 2:
            return 'Baja'
        else:
            return 'Normal'

    @staticmethod
    def _generate_alerts(demanda: float, clima: Dict[str, Any],
                         trafico: Dict[str, Any],
                         capacidad: Dict[str, Any]) -> List[str]:
        """🚨 Genera alertas operativas"""

        alertas = []

        if demanda > 2.8:
            alertas.append('🔴 DEMANDA CRÍTICA: Activar protocolos de emergencia')
        elif demanda > 2.3:
            alertas.append('🟡 DEMANDA ALTA: Monitorear capacidad de cerca')

        if clima['probabilidad_lluvia'] > 70:
            alertas.append('🌧️ CLIMA ADVERSO: Preparar rutas alternativas')

        if trafico['score'] > 0.8:
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

            # Predicción simplificada
            pred_demanda = demanda * 0.9  # Decremento gradual

            predicciones.append({
                'fecha': fecha_pred.date().isoformat(),
                'demanda_predicha': round(pred_demanda, 2),
                'tendencia': 'descendente' if pred_demanda < demanda else 'estable',
                'confianza': 0.8 - (i * 0.1)  # Menos confianza a futuro
            })

        return predicciones

    @staticmethod
    def _generate_recommendations(fecha: datetime, demanda: float,
                                  eventos: Dict[str, Any], clima: Dict[str, Any],
                                  capacidad: Dict[str, Any]) -> List[str]:
        """💡 Genera recomendaciones estratégicas"""

        recomendaciones = []

        if demanda > 2.5:
            recomendaciones.extend([
                '📈 Activar capacidad adicional de fulfillment',
                '🚚 Contratar flota externa adicional',
                '⏰ Extender horarios de operación',
                '📦 Priorizar productos alta rotación'
            ])

        if clima['probabilidad_lluvia'] > 60:
            recomendaciones.extend([
                '🌧️ Usar vehículos cubiertos',
                '🛣️ Activar rutas alternativas',
                '📱 Comunicar delays proactivamente'
            ])

        if capacidad['saturacion'] == 'Crítica':
            recomendaciones.extend([
                '🔄 Redistribuir carga entre nodos',
                '⚡ Implementar fulfillment distribuido',
                '📊 Activar análisis en tiempo real'
            ])

        return recomendaciones

    @staticmethod
    def _calculate_prediction_confidence(fecha: datetime,
                                         eventos: Dict[str, Any],
                                         clima: Dict[str, Any]) -> float:
        """📊 Calcula confianza de las predicciones"""

        base_confidence = 0.85

        # Reducir confianza en eventos impredecibles
        if eventos['intensidad_maxima'] > 2.5:
            base_confidence -= 0.15

        # Reducir confianza en clima adverso
        if clima['probabilidad_lluvia'] > 70:
            base_confidence -= 0.1

        # Reducir confianza en fines de semana
        if fecha.weekday() >= 5:
            base_confidence -= 0.05

        return round(max(0.5, base_confidence), 2)