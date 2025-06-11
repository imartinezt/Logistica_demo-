from datetime import datetime
from typing import List, Dict, Any, Tuple

import joblib
import lightgbm as lgb
import numpy as np

from config.settings import settings
from utils.geo_calculator import GeoCalculator
from utils.logger import logger


class RouteOptimizer:
    """🎯 Optimizador de rutas usando LightGBM para ranking multiobjetivo"""

    def __init__(self):
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.model_path = settings.MODELS_DIR / "route_optimizer_lgb.pkl"

        # Crear directorio de modelos si no existe
        settings.MODELS_DIR.mkdir(exist_ok=True)

        # Pesos para la función objetivo combinada
        self.weights = {
            'tiempo': settings.PESO_TIEMPO,
            'costo': settings.PESO_COSTO,
            'probabilidad': settings.PESO_PROBABILIDAD,
            'distancia': settings.PESO_DISTANCIA
        }

    def generate_route_candidates(self,
                                  split_inventory: Dict[str, Any],
                                  target_coordinates: Tuple[float, float],
                                  factores_externos: Dict[str, Any],
                                  repositories: Dict[str, Any]) -> List[Dict[str, Any]]:
        """🔄 Genera candidatos de rutas con todas las combinaciones posibles"""

        candidates = []
        target_lat, target_lon = target_coordinates

        if not split_inventory['es_factible']:
            logger.warning("❌ Split de inventario no factible")
            return []

        split_plan = split_inventory['split_plan']

        # Generar candidatos basados en diferentes estrategias

        # Estrategia 1: Ruta directa desde cada ubicación
        for location_split in split_plan:
            candidate = self._create_direct_route_candidate(
                location_split, target_coordinates, factores_externos, repositories
            )
            if candidate:
                candidates.append(candidate)

        # Estrategia 2: Ruta consolidada (recoger de múltiples ubicaciones)
        if len(split_plan) > 1:
            consolidated_candidate = self._create_consolidated_route_candidate(
                split_plan, target_coordinates, factores_externos, repositories
            )
            if consolidated_candidate:
                candidates.append(consolidated_candidate)

        # Estrategia 3: Rutas híbridas (FI + FE)
        hybrid_candidates = self._create_hybrid_route_candidates(
            split_plan, target_coordinates, factores_externos, repositories
        )
        candidates.extend(hybrid_candidates)

        # Estrategia 4: Rutas vía CEDIS
        cedis_candidates = self._create_cedis_route_candidates(
            split_plan, target_coordinates, factores_externos, repositories
        )
        candidates.extend(cedis_candidates)

        logger.info(f"🔄 Generados {len(candidates)} candidatos de rutas")
        return candidates

    def _create_direct_route_candidate(self,
                                       location_split: Dict[str, Any],
                                       target_coordinates: Tuple[float, float],
                                       factores_externos: Dict[str, Any],
                                       repositories: Dict[str, Any]) -> Dict[str, Any]:
        """📍 Crea candidato de ruta directa"""

        tienda_id = location_split['tienda_id']
        store_info = repositories['store'].get_store_by_id(tienda_id)

        if not store_info:
            logger.warning(f"❌ Tienda no encontrada: {tienda_id}")
            return None

        # Calcular distancia y tiempo
        distance_km = GeoCalculator.calculate_distance_km(
            store_info['latitud'], store_info['longitud'],
            target_coordinates[0], target_coordinates[1]
        )

        # Determinar tipo de flota basado en distancia y factores
        # Código postal aproximado para zona roja
        cp_approx = f"{int(target_coordinates[0] * 100):05d}"[:5]
        is_zona_roja = repositories['postal_code'].is_zona_roja(cp_approx)

        tipo_flota = 'FE' if (distance_km > 100 or is_zona_roja) else 'FI'

        travel_time = GeoCalculator.calculate_travel_time(
            distance_km,
            tipo_flota,
            factores_externos.get('trafico_nivel', 'Moderado'),
            factores_externos.get('condicion_clima', 'Templado')
        )

        # Tiempo total incluyendo preparación
        tiempo_preparacion = settings.TIEMPO_PICKING_PACKING
        tiempo_total = travel_time + tiempo_preparacion

        # Calcular costo
        costo_base = self._calculate_route_cost(
            distance_km, tipo_flota, location_split['cantidad']
        )

        # Aplicar factores externos al costo
        factor_demanda = factores_externos.get('factor_demanda', 1.0)
        costo_ajustado = costo_base * min(factor_demanda, 2.0)  # Cap factor

        # Calcular probabilidad de cumplimiento (mejorada)
        probabilidad = self._calculate_success_probability(
            distance_km, tiempo_total, factores_externos, tipo_flota
        )

        return {
            'ruta_id': f"direct_{tienda_id}",
            'tipo_ruta': 'directa',
            'origen_principal': tienda_id,
            'segmentos': [
                {
                    'origen': tienda_id,
                    'destino': 'cliente',
                    'distancia_km': distance_km,
                    'tiempo_horas': travel_time,
                    'tipo_flota': tipo_flota
                }
            ],
            'tiempo_total_horas': tiempo_total,
            'costo_total_mxn': costo_ajustado,
            'distancia_total_km': distance_km,
            'probabilidad_cumplimiento': probabilidad,
            'cantidad_cubierta': location_split['cantidad'],
            'factores_aplicados': [
                f"factor_demanda_{factor_demanda}",
                f"trafico_{factores_externos.get('trafico_nivel', 'Moderado')}",
                f"flota_{tipo_flota}"
            ]
        }

    def _create_consolidated_route_candidate(self,
                                             split_plan: List[Dict[str, Any]],
                                             target_coordinates: Tuple[float, float],
                                             factores_externos: Dict[str, Any],
                                             repositories: Dict[str, Any]) -> Dict[str, Any]:
        """🔄 Crea candidato de ruta consolidada"""

        if len(split_plan) < 2:
            return None

        # Obtener información de todas las tiendas
        stores_info = []
        for location_split in split_plan:
            store = repositories['store'].get_store_by_id(location_split['tienda_id'])
            if store:
                store['cantidad'] = location_split['cantidad']
                stores_info.append(store)

        if not stores_info:
            return None

        # Encontrar secuencia óptima de recolección
        start_location = stores_info[0]  # Empezar por la más cercana
        optimal_sequence = GeoCalculator.calculate_optimal_route_sequence(
            stores_info[1:], start_location
        )
        optimal_sequence.insert(0, start_location)

        # Calcular segmentos de la ruta
        segmentos = []
        tiempo_total = 0
        distancia_total = 0
        costo_total = 0
        cantidad_total = 0

        # Tiempo de preparación en cada tienda
        for i, store in enumerate(optimal_sequence):
            tiempo_total += settings.TIEMPO_PICKING_PACKING
            cantidad_total += store['cantidad']

            # Segmento hacia siguiente ubicación
            if i < len(optimal_sequence) - 1:
                next_store = optimal_sequence[i + 1]
                distance = GeoCalculator.calculate_distance_km(
                    store['latitud'], store['longitud'],
                    next_store['latitud'], next_store['longitud']
                )
                travel_time = GeoCalculator.calculate_travel_time(
                    distance, 'FI',  # Flota interna para consolidación
                    factores_externos.get('trafico_nivel', 'Moderado'),
                    factores_externos.get('condicion_clima', 'Templado')
                )

                segmentos.append({
                    'origen': store['tienda_id'],
                    'destino': next_store['tienda_id'],
                    'distancia_km': distance,
                    'tiempo_horas': travel_time,
                    'tipo_flota': 'FI'
                })

                tiempo_total += travel_time
                distancia_total += distance
                costo_total += self._calculate_route_cost(distance, 'FI', 1)

        # Segmento final hacia cliente
        last_store = optimal_sequence[-1]
        final_distance = GeoCalculator.calculate_distance_km(
            last_store['latitud'], last_store['longitud'],
            target_coordinates[0], target_coordinates[1]
        )

        final_travel_time = GeoCalculator.calculate_travel_time(
            final_distance, 'FI',
            factores_externos.get('trafico_nivel', 'Moderado'),
            factores_externos.get('condicion_clima', 'Templado')
        )

        segmentos.append({
            'origen': last_store['tienda_id'],
            'destino': 'cliente',
            'distancia_km': final_distance,
            'tiempo_horas': final_travel_time,
            'tipo_flota': 'FI'
        })

        tiempo_total += final_travel_time
        distancia_total += final_distance
        costo_total += self._calculate_route_cost(final_distance, 'FI', cantidad_total)

        # Aplicar factores externos
        factor_demanda = factores_externos.get('factor_demanda', 1.0)
        costo_total *= min(factor_demanda, 2.0)

        # Calcular probabilidad (penalizar rutas complejas pero menos)
        base_probability = self._calculate_success_probability(
            distancia_total, tiempo_total, factores_externos, 'FI'
        )
        complexity_penalty = 0.95 ** len(segmentos)  # Penalización menor
        probabilidad = base_probability * complexity_penalty

        return {
            'ruta_id': f"consolidated_{'_'.join([s['tienda_id'] for s in stores_info])}",
            'tipo_ruta': 'consolidada',
            'origen_principal': stores_info[0]['tienda_id'],
            'segmentos': segmentos,
            'tiempo_total_horas': tiempo_total,
            'costo_total_mxn': costo_total,
            'distancia_total_km': distancia_total,
            'probabilidad_cumplimiento': probabilidad,
            'cantidad_cubierta': cantidad_total,
            'secuencia_optima': [s['tienda_id'] for s in optimal_sequence],
            'factores_aplicados': [
                f"consolidation_{len(segmentos)}_segments",
                f"factor_demanda_{factor_demanda}",
                'flota_interna_consolidacion'
            ]
        }

    def _create_hybrid_route_candidates(self,
                                        split_plan: List[Dict[str, Any]],
                                        target_coordinates: Tuple[float, float],
                                        factores_externos: Dict[str, Any],
                                        repositories: Dict[str, Any]) -> List[Dict[str, Any]]:
        """🔄 Crea candidatos híbridos FI + FE"""

        candidates = []

        # Para cada ubicación con stock, evaluar híbrido
        for location_split in split_plan:
            # Estrategia: FI hasta CEDIS más cercano, luego FE hasta cliente

            store_info = repositories['store'].get_store_by_id(location_split['tienda_id'])
            if not store_info:
                continue

            # Encontrar CEDIS más cercano
            cedis_list = repositories['cedis'].load_data().to_dicts()
            cedis_limpio = []

            for cedis_raw in cedis_list:
                # Limpiar cedis_id y coordenadas
                cedis_id_clean = repositories['cedis']._clean_id_value(cedis_raw.get('cedis_id', ''))
                if not cedis_id_clean:
                    continue

                lat_clean = repositories['cedis']._clean_coordinate_value(cedis_raw.get('latitud'))
                lon_clean = repositories['cedis']._clean_coordinate_value(cedis_raw.get('longitud'))

                if 14.0 <= lat_clean <= 33.0 and -118.0 <= lon_clean <= -86.0:
                    cedis_clean = cedis_raw.copy()
                    cedis_clean['cedis_id'] = cedis_id_clean
                    cedis_clean['latitud'] = lat_clean
                    cedis_clean['longitud'] = lon_clean
                    cedis_limpio.append(cedis_clean)

            closest_cedis = GeoCalculator.find_closest_locations(
                store_info['latitud'], store_info['longitud'],
                cedis_limpio, max_results=1
            )

            if not closest_cedis:
                continue

            cedis = closest_cedis[0]

            # Segmento 1: Tienda -> CEDIS (FI)
            dist_to_cedis = GeoCalculator.calculate_distance_km(
                store_info['latitud'], store_info['longitud'],
                cedis['latitud'], cedis['longitud']
            )

            time_to_cedis = GeoCalculator.calculate_travel_time(
                dist_to_cedis, 'FI',
                factores_externos.get('trafico_nivel', 'Moderado'),
                factores_externos.get('condicion_clima', 'Templado')
            )

            # Segmento 2: CEDIS -> Cliente (FE)
            dist_to_client = GeoCalculator.calculate_distance_km(
                cedis['latitud'], cedis['longitud'],
                target_coordinates[0], target_coordinates[1]
            )

            time_to_client = GeoCalculator.calculate_travel_time(
                dist_to_client, 'FE',
                factores_externos.get('trafico_nivel', 'Moderado'),
                factores_externos.get('condicion_clima', 'Templado')
            )

            # Tiempo total
            tiempo_prep_tienda = settings.TIEMPO_PICKING_PACKING
            tiempo_prep_cedis = settings.TIEMPO_PREPARACION_CEDIS
            tiempo_total = tiempo_prep_tienda + time_to_cedis + tiempo_prep_cedis + time_to_client

            # Costo total
            costo_fi = self._calculate_route_cost(dist_to_cedis, 'FI', location_split['cantidad'])
            costo_fe = self._calculate_route_cost(dist_to_client, 'FE', location_split['cantidad'])
            costo_total = costo_fi + costo_fe

            # Aplicar factores
            factor_demanda = factores_externos.get('factor_demanda', 1.0)
            costo_total *= min(factor_demanda, 2.0)

            # Probabilidad (híbridos son generalmente más confiables)
            probabilidad = self._calculate_success_probability(
                dist_to_cedis + dist_to_client, tiempo_total, factores_externos, 'FI_FE'
            )
            probabilidad *= 1.05  # Bonus menor por usar híbrido
            probabilidad = min(probabilidad, 0.98)  # Cap máximo

            candidate = {
                'ruta_id': f"hybrid_{location_split['tienda_id']}_{cedis['cedis_id']}",
                'tipo_ruta': 'hibrida',
                'origen_principal': location_split['tienda_id'],
                'cedis_intermedio': cedis['cedis_id'],
                'segmentos': [
                    {
                        'origen': location_split['tienda_id'],
                        'destino': cedis['cedis_id'],
                        'distancia_km': dist_to_cedis,
                        'tiempo_horas': time_to_cedis,
                        'tipo_flota': 'FI'
                    },
                    {
                        'origen': cedis['cedis_id'],
                        'destino': 'cliente',
                        'distancia_km': dist_to_client,
                        'tiempo_horas': time_to_client,
                        'tipo_flota': 'FE'
                    }
                ],
                'tiempo_total_horas': tiempo_total,
                'costo_total_mxn': costo_total,
                'distancia_total_km': dist_to_cedis + dist_to_client,
                'probabilidad_cumplimiento': probabilidad,
                'cantidad_cubierta': location_split['cantidad'],
                'factores_aplicados': [
                    'ruta_hibrida_fi_fe',
                    f"cedis_{cedis['cedis_id']}",
                    f"factor_demanda_{factor_demanda}"
                ]
            }

            candidates.append(candidate)

        logger.info(f"🔄 Generados {len(candidates)} candidatos híbridos")
        return candidates

    def _create_cedis_route_candidates(self,
                                       split_plan: List[Dict[str, Any]],
                                       target_coordinates: Tuple[float, float],
                                       factores_externos: Dict[str, Any],
                                       repositories: Dict[str, Any]) -> List[Dict[str, Any]]:
        """🏭 Crea candidatos que van directo desde CEDIS"""

        candidates = []

        # Obtener todos los CEDIS disponibles (limpios)
        cedis_raw_list = repositories['cedis'].load_data().to_dicts()
        cedis_list = []

        for cedis_raw in cedis_raw_list:
            cedis_id_clean = repositories['cedis']._clean_id_value(cedis_raw.get('cedis_id', ''))
            if not cedis_id_clean:
                continue

            lat_clean = repositories['cedis']._clean_coordinate_value(cedis_raw.get('latitud'))
            lon_clean = repositories['cedis']._clean_coordinate_value(cedis_raw.get('longitud'))

            if 14.0 <= lat_clean <= 33.0 and -118.0 <= lon_clean <= -86.0:
                cedis_clean = cedis_raw.copy()
                cedis_clean['cedis_id'] = cedis_id_clean
                cedis_clean['latitud'] = lat_clean
                cedis_clean['longitud'] = lon_clean
                cedis_list.append(cedis_clean)

        # Para cada CEDIS, evaluar si puede cubrir la demanda
        for cedis in cedis_list:
            # Asumimos que CEDIS siempre tiene stock (simplificación)
            can_supply_cedis = True
            total_cantidad_disponible = sum(item['cantidad'] for item in split_plan)

            if not can_supply_cedis:
                continue

            # Crear candidato CEDIS -> Cliente
            dist_cedis_client = GeoCalculator.calculate_distance_km(
                cedis['latitud'], cedis['longitud'],
                target_coordinates[0], target_coordinates[1]
            )

            time_cedis_client = GeoCalculator.calculate_travel_time(
                dist_cedis_client, 'FE',  # CEDIS generalmente usa flota externa
                factores_externos.get('trafico_nivel', 'Moderado'),
                factores_externos.get('condicion_clima', 'Templado')
            )

            # Tiempo de preparación en CEDIS
            tiempo_prep_cedis = settings.TIEMPO_PREPARACION_CEDIS
            tiempo_total = tiempo_prep_cedis + time_cedis_client

            # Costo
            costo_total = self._calculate_route_cost(
                dist_cedis_client, 'FE', total_cantidad_disponible
            )

            # Aplicar factores
            factor_demanda = factores_externos.get('factor_demanda', 1.0)
            costo_total *= min(factor_demanda, 2.0)

            # Probabilidad (CEDIS son más confiables)
            probabilidad = self._calculate_success_probability(
                dist_cedis_client, tiempo_total, factores_externos, 'FE'
            )
            probabilidad *= 1.1  # Bonus por usar CEDIS
            probabilidad = min(probabilidad, 0.98)

            candidate = {
                'ruta_id': f"cedis_{cedis['cedis_id']}",
                'tipo_ruta': 'cedis_directo',
                'origen_principal': cedis['cedis_id'],
                'segmentos': [
                    {
                        'origen': cedis['cedis_id'],
                        'destino': 'cliente',
                        'distancia_km': dist_cedis_client,
                        'tiempo_horas': time_cedis_client,
                        'tipo_flota': 'FE'
                    }
                ],
                'tiempo_total_horas': tiempo_total,
                'costo_total_mxn': costo_total,
                'distancia_total_km': dist_cedis_client,
                'probabilidad_cumplimiento': probabilidad,
                'cantidad_cubierta': total_cantidad_disponible,
                'factores_aplicados': [
                    'ruta_cedis_directo',
                    f"cedis_{cedis['cedis_id']}",
                    'flota_externa',
                    f"factor_demanda_{factor_demanda}"
                ]
            }

            candidates.append(candidate)

        logger.info(f"🏭 Generados {len(candidates)} candidatos desde CEDIS")
        return candidates

    def _calculate_route_cost(self, distance_km: float,
                              fleet_type: str, cantidad: int) -> float:
        """💰 Calcula costo de ruta basado en distancia y tipo de flota"""

        # Costos base por kilómetro según tipo de flota
        cost_per_km = {
            'FI': 8.5,  # Flota interna más barata
            'FE': 12.0,  # Flota externa más cara
            'FI_FE': 10.0  # Híbrido intermedio
        }

        base_cost = distance_km * cost_per_km.get(fleet_type, 10.0)

        # Factor por cantidad (economías de escala)
        quantity_factor = 1.0 + (cantidad - 1) * 0.05  # 5% extra por unidad adicional
        quantity_factor = min(quantity_factor, 1.5)  # Cap máximo 1.5x

        # Costo fijo mínimo
        minimum_cost = 25.0

        total_cost = max(base_cost * quantity_factor, minimum_cost)

        return round(total_cost, 2)

    def _calculate_success_probability(self, distance_km: float,
                                       tiempo_total: float,
                                       factores_externos: Dict[str, Any],
                                       fleet_type: str) -> float:
        """📊 Calcula probabilidad de cumplimiento exitoso (MEJORADA)"""

        # Probabilidad base por tipo de flota (MEJORADA)
        base_probability = {
            'FI': 0.88,  # Era 0.92, ahora más realista
            'FE': 0.85,  # Era 0.87, ahora más realista
            'FI_FE': 0.87  # Era 0.90, ahora más realista
        }.get(fleet_type, 0.83)

        # Penalización por distancia (REDUCIDA)
        distance_penalty = min(0.1, distance_km / 2000)  # Era /1000, ahora más tolerante

        # Penalización por tiempo (REDUCIDA)
        time_penalty = min(0.08, max(0, (tiempo_total - 4) / 50))  # Era 2h y /20, ahora más tolerante

        # Factor por condiciones externas (MEJORADO)
        external_factor = 1.0

        condicion_clima = factores_externos.get('condicion_clima', 'Templado')
        if condicion_clima in ['Lluvioso_Intenso', 'Tormenta']:
            external_factor *= 0.95  # Era 0.9, menos penalización
        elif condicion_clima in ['Lluvioso']:
            external_factor *= 0.98  # Nueva categoría intermedia

        trafico_nivel = factores_externos.get('trafico_nivel', 'Moderado')
        if trafico_nivel in ['Muy_Alto']:
            external_factor *= 0.92  # Era 0.85, menos penalización
        elif trafico_nivel in ['Alto']:
            external_factor *= 0.96  # Era implícito en 0.85, ahora específico

        factor_demanda = factores_externos.get('factor_demanda', 1.0)
        if factor_demanda > 3.0:  # Solo para demanda muy alta
            external_factor *= 0.95  # Era 0.92 desde 2.0
        elif factor_demanda > 2.5:
            external_factor *= 0.98  # Nueva categoría

        # Cálculo final con mínimo más alto
        final_probability = (base_probability - distance_penalty - time_penalty) * external_factor

        # Mínimo más alto y máximo ajustado
        final_probability = max(0.65, min(0.98, final_probability))  # Era 0.5 min, ahora 0.65

        return round(final_probability, 3)

    def rank_candidates_with_lightgbm(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """🏆 Rankea candidatos usando LightGBM"""

        if not candidates:
            return []

        # Si no hay modelo entrenado, usar scoring simple
        if not self.is_trained:
            return self._simple_multiobj_ranking(candidates)

        try:
            # Preparar features para LightGBM
            features_matrix = self._prepare_features_matrix(candidates)

            # Predecir scores
            scores = self.model.predict(features_matrix)

            # Agregar scores a candidatos
            for i, candidate in enumerate(candidates):
                candidate['score_lightgbm'] = float(scores[i])
                candidate['ranking_position'] = i + 1

            # Ordenar por score descendente
            ranked_candidates = sorted(candidates, key=lambda x: x['score_lightgbm'], reverse=True)

            # Actualizar posiciones de ranking
            for i, candidate in enumerate(ranked_candidates):
                candidate['ranking_position'] = i + 1

            logger.info(f"🏆 {len(candidates)} candidatos rankeados con LightGBM")
            return ranked_candidates

        except Exception as e:
            logger.warning(f"❌ Error en LightGBM ranking: {e}")
            return self._simple_multiobj_ranking(candidates)

    def _simple_multiobj_ranking(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """📊 Ranking simple multiobjetivo sin ML"""

        if not candidates:
            return []

        # Normalizar métricas
        tiempos = [c['tiempo_total_horas'] for c in candidates]
        costos = [c['costo_total_mxn'] for c in candidates]
        distancias = [c['distancia_total_km'] for c in candidates]
        probabilidades = [c['probabilidad_cumplimiento'] for c in candidates]

        min_tiempo, max_tiempo = min(tiempos), max(tiempos)
        min_costo, max_costo = min(costos), max(costos)
        min_distancia, max_distancia = min(distancias), max(distancias)

        # Calcular scores combinados
        for candidate in candidates:
            # Normalizar (mejor = 1, peor = 0)
            if max_tiempo > min_tiempo:
                score_tiempo = 1 - (candidate['tiempo_total_horas'] - min_tiempo) / (max_tiempo - min_tiempo)
            else:
                score_tiempo = 1.0

            if max_costo > min_costo:
                score_costo = 1 - (candidate['costo_total_mxn'] - min_costo) / (max_costo - min_costo)
            else:
                score_costo = 1.0

            if max_distancia > min_distancia:
                score_distancia = 1 - (candidate['distancia_total_km'] - min_distancia) / (
                        max_distancia - min_distancia)
            else:
                score_distancia = 1.0

            score_probabilidad = candidate['probabilidad_cumplimiento']

            # Score combinado ponderado
            score_combinado = (
                    self.weights['tiempo'] * score_tiempo +
                    self.weights['costo'] * score_costo +
                    self.weights['probabilidad'] * score_probabilidad +
                    self.weights['distancia'] * score_distancia
            )

            candidate['score_lightgbm'] = round(score_combinado, 4)
            candidate['score_breakdown'] = {
                'tiempo': score_tiempo,
                'costo': score_costo,
                'distancia': score_distancia,
                'probabilidad': score_probabilidad
            }

        # Ordenar por score
        ranked_candidates = sorted(candidates, key=lambda x: x['score_lightgbm'], reverse=True)

        # Asignar posiciones
        for i, candidate in enumerate(ranked_candidates):
            candidate['ranking_position'] = i + 1

        logger.info(f"📊 {len(candidates)} candidatos rankeados con scoring simple")
        return ranked_candidates

    def _prepare_features_matrix(self, candidates: List[Dict[str, Any]]) -> np.ndarray:
        """🔧 Prepara matriz de features para LightGBM"""

        features_list = []

        for candidate in candidates:
            features = [
                candidate['tiempo_total_horas'],
                candidate['costo_total_mxn'],
                candidate['distancia_total_km'],
                candidate['probabilidad_cumplimiento'],
                candidate['cantidad_cubierta'],
                len(candidate['segmentos']),  # Complejidad de ruta
                1 if candidate['tipo_ruta'] == 'directa' else 0,
                1 if candidate['tipo_ruta'] == 'consolidada' else 0,
                1 if candidate['tipo_ruta'] == 'hibrida' else 0,
                1 if candidate['tipo_ruta'] == 'cedis_directo' else 0,
            ]

            features_list.append(features)

        return np.array(features_list)

    def get_top_candidates(self, ranked_candidates: List[Dict[str, Any]],
                           max_candidates: int = None) -> List[Dict[str, Any]]:
        """🏆 Obtiene top candidatos para Gemini"""

        max_candidates = max_candidates or settings.TOP_CANDIDATOS_GEMINI

        if not ranked_candidates:
            return []

        # Filtrar candidatos con score mínimo (REDUCIDO)
        min_score = 0.2  # Era 0.3, ahora más permisivo
        viable_candidates = [c for c in ranked_candidates if c.get('score_lightgbm', 0) >= min_score]

        # Si no hay candidatos viables, tomar los mejores disponibles
        if not viable_candidates and ranked_candidates:
            logger.warning("⚠️ No hay candidatos con score mínimo, tomando los mejores disponibles")
            viable_candidates = ranked_candidates[:max_candidates]

        # Asegurar diversidad en los candidatos top
        diverse_candidates = self._ensure_candidate_diversity(viable_candidates, max_candidates)

        # Tomar top candidatos
        top_candidates = diverse_candidates[:max_candidates]

        logger.info(f"🏆 Seleccionados {len(top_candidates)} candidatos top para Gemini")
        return top_candidates

    def _ensure_candidate_diversity(self, candidates: List[Dict[str, Any]],
                                    max_candidates: int) -> List[Dict[str, Any]]:
        """🎯 Asegura diversidad en tipos de rutas para mejor decisión de Gemini"""

        if len(candidates) <= max_candidates:
            return candidates

        # Agrupar por tipo de ruta
        by_type = {}
        for candidate in candidates:
            route_type = candidate['tipo_ruta']
            if route_type not in by_type:
                by_type[route_type] = []
            by_type[route_type].append(candidate)

        # Seleccionar al menos uno de cada tipo (si existe)
        diverse_candidates = []

        # Prioridad de tipos de ruta
        type_priority = ['directa', 'cedis_directo', 'hibrida', 'consolidada']

        for route_type in type_priority:
            if route_type in by_type and len(diverse_candidates) < max_candidates:
                # Tomar el mejor de este tipo
                best_of_type = sorted(by_type[route_type],
                                      key=lambda x: x['score_lightgbm'], reverse=True)[0]
                diverse_candidates.append(best_of_type)

        # Llenar con los mejores restantes
        remaining_candidates = [c for c in candidates if c not in diverse_candidates]
        remaining_slots = max_candidates - len(diverse_candidates)

        if remaining_slots > 0:
            top_remaining = sorted(remaining_candidates,
                                   key=lambda x: x['score_lightgbm'], reverse=True)
            diverse_candidates.extend(top_remaining[:remaining_slots])

        return diverse_candidates

    def train_model(self, training_data: List[Dict[str, Any]],
                    labels: List[float]) -> bool:
        """🎓 Entrena modelo LightGBM con datos históricos"""

        try:
            if len(training_data) < 10:
                logger.warning("❌ Datos insuficientes para entrenar LightGBM")
                return False

            # Preparar datos
            X = self._prepare_features_matrix(training_data)
            y = np.array(labels)

            # Crear dataset LightGBM
            train_data = lgb.Dataset(X, label=y)

            # Parámetros del modelo
            params = settings.LIGHTGBM_PARAMS.copy()

            # Entrenar modelo
            logger.info("🎓 Entrenando modelo LightGBM...")
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )

            self.is_trained = True

            # Guardar modelo
            self._save_model()

            logger.info("✅ Modelo LightGBM entrenado exitosamente")
            return True

        except Exception as e:
            logger.error(f"❌ Error entrenando LightGBM: {e}")
            return False

    def _save_model(self):
        """💾 Guarda modelo entrenado"""
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'weights': self.weights,
                'trained_at': datetime.now().isoformat()
            }

            joblib.dump(model_data, self.model_path)
            logger.info(f"💾 Modelo guardado en {self.model_path}")

        except Exception as e:
            logger.warning(f"❌ Error guardando modelo: {e}")

    def load_model(self) -> bool:
        """📂 Carga modelo pre-entrenado"""
        try:
            if not self.model_path.exists():
                logger.info("📂 No hay modelo pre-entrenado disponible")
                return False

            model_data = joblib.load(self.model_path)

            self.model = model_data['model']
            self.feature_names = model_data.get('feature_names', [])
            self.weights = model_data.get('weights', self.weights)
            self.is_trained = True

            logger.info("✅ Modelo LightGBM cargado exitosamente")
            return True

        except Exception as e:
            logger.warning(f"❌ Error cargando modelo: {e}")
            return False

    def generate_training_features(self, candidate: Dict[str, Any]) -> List[float]:
        """🔧 Genera features para entrenamiento desde un candidato"""

        return [
            candidate.get('tiempo_total_horas', 0),
            candidate.get('costo_total_mxn', 0),
            candidate.get('distancia_total_km', 0),
            candidate.get('probabilidad_cumplimiento', 0),
            candidate.get('cantidad_cubierta', 0),
            len(candidate.get('segmentos', [])),
            1 if candidate.get('tipo_ruta') == 'directa' else 0,
            1 if candidate.get('tipo_ruta') == 'consolidada' else 0,
            1 if candidate.get('tipo_ruta') == 'hibrida' else 0,
            1 if candidate.get('tipo_ruta') == 'cedis_directo' else 0,
        ]

    def explain_ranking(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """📊 Explica por qué un candidato recibió su ranking"""

        explanation = {
            'score_total': candidate.get('score_lightgbm', 0),
            'factores_positivos': [],
            'factores_negativos': [],
            'recomendaciones': []
        }

        # Analizar factores
        tiempo = candidate.get('tiempo_total_horas', 0)
        costo = candidate.get('costo_total_mxn', 0)
        probabilidad = candidate.get('probabilidad_cumplimiento', 0)
        distancia = candidate.get('distancia_total_km', 0)

        # Factores positivos
        if tiempo <= 24:
            explanation['factores_positivos'].append("⚡ Entrega rápida (≤24h)")
        if costo <= 100:
            explanation['factores_positivos'].append("💰 Costo económico (≤$100)")
        if probabilidad >= 0.85:  # Era 0.9, ahora más realista
            explanation['factores_positivos'].append("🎯 Alta confiabilidad (≥85%)")
        if distancia <= 50:
            explanation['factores_positivos'].append("📍 Distancia corta (≤50km)")

        # Factores negativos
        if tiempo > 72:
            explanation['factores_negativos'].append("⏰ Tiempo excesivo (>72h)")
        if costo > 300:
            explanation['factores_negativos'].append("💸 Costo elevado (>$300)")
        if probabilidad < 0.65:  # Era 0.7, ahora más realista
            explanation['factores_negativos'].append("⚠️ Baja confiabilidad (<65%)")
        if distancia > 200:
            explanation['factores_negativos'].append("🛣️ Distancia muy larga (>200km)")

        # Recomendaciones basadas en tipo de ruta
        tipo_ruta = candidate.get('tipo_ruta', '')
        if tipo_ruta == 'consolidada':
            explanation['recomendaciones'].append("🔄 Ruta compleja: verificar tiempos de preparación")
        elif tipo_ruta == 'hibrida':
            explanation['recomendaciones'].append("🔀 Ruta híbrida: monitorear transferencia en CEDIS")
        elif tipo_ruta == 'directa':
            explanation['recomendaciones'].append("➡️ Ruta directa: optimal para urgencia")

        return explanation