from typing import List, Dict, Any, Tuple

import joblib

from config.settings import settings
from utils.geo_calculator import GeoCalculator
from utils.logger import logger


class RouteOptimizer:
    """🎯 Optimizador CORREGIDO que NO genera CEDIS directos"""

    def __init__(self):
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.model_path = settings.MODELS_DIR / "route_optimizer_lgb.pkl"

        settings.MODELS_DIR.mkdir(exist_ok=True)
        self.weights = {
            'tiempo': 0.4,
            'costo': 0.2,
            'probabilidad': 0.35,
            'distancia': 0.05
        }

    def generate_route_candidates(self,
                                  split_inventory: Dict[str, Any],
                                  target_coordinates: Tuple[float, float],
                                  factores_externos: Dict[str, Any],
                                  repositories: Dict[str, Any]) -> List[Dict[str, Any]]:
        """🗺️ Generación CORREGIDA: NO incluye CEDIS directos"""

        candidates = []
        target_lat, target_lon = target_coordinates

        if not split_inventory['es_factible']:
            logger.warning("❌ Split de inventario no factible")
            return []

        split_plan = split_inventory.get('split_plan', [])
        if not split_plan:
            split_obj = split_inventory.get('split_inventory')
            if split_obj and hasattr(split_obj, 'ubicaciones'):
                split_plan = []
                for ubicacion in split_obj.ubicaciones:
                    split_plan.append({
                        'tienda_id': ubicacion.ubicacion_id,
                        'cantidad': ubicacion.stock_disponible,
                        'distancia_km': 0
                    })

        if not split_plan:
            logger.warning("❌ No se pudo extraer split_plan")
            return []

        logger.info(f"📊 Generando candidatos para {len(split_plan)} ubicaciones")

        # 1. SOLO rutas DIRECTAS desde tiendas
        for location in split_plan:
            direct_candidate = self._create_direct_store_route_only(
                location, target_coordinates, factores_externos, repositories
            )
            if direct_candidate:
                candidates.append(direct_candidate)

        # 2. Rutas CONSOLIDADAS si hay múltiples tiendas
        if len(split_plan) > 1:
            consolidated_candidate = self._create_smart_consolidated_route(
                split_plan, target_coordinates, factores_externos, repositories
            )
            if consolidated_candidate:
                candidates.append(consolidated_candidate)

        # 3. ELIMINADO: Ya no generamos rutas CEDIS directas
        # Los CEDIS solo se usan como intermediarios en el servicio principal

        logger.info(f"🔄 Generados {len(candidates)} candidatos SIN CEDIS directos")
        return candidates

    def _create_direct_store_route_only(self,
                                        location_split: Dict[str, Any],
                                        target_coordinates: Tuple[float, float],
                                        factores_externos: Dict[str, Any],
                                        repositories: Dict[str, Any]) -> Dict[str, Any]:
        """📍 Crea SOLO rutas directas desde tienda (sin CEDIS)"""

        tienda_id = location_split['tienda_id']
        store_info = repositories['store'].get_store_by_id(tienda_id)

        if not store_info:
            return None

        distance_km = GeoCalculator.calculate_distance_km(
            store_info['latitud'], store_info['longitud'],
            target_coordinates[0], target_coordinates[1]
        )

        distance_km = max(distance_km, 5.0)

        # Determinar flota usando reglas simples
        if distance_km <= 100:
            tipo_flota = 'FI'
        else:
            tipo_flota = 'FE'

        # Tiempo de viaje básico
        base_travel_time = GeoCalculator.calculate_travel_time(
            distance_km,
            tipo_flota,
            factores_externos.get('trafico_nivel', 'Moderado'),
            factores_externos.get('condicion_clima', 'Templado')
        )

        # Tiempo total con preparación
        tiempo_preparacion = settings.TIEMPO_PICKING_PACKING
        tiempo_total = tiempo_preparacion + base_travel_time

        # Aplicar factores externos
        factor_tiempo_extra = factores_externos.get('impacto_tiempo_extra_horas', 0)
        tiempo_total += factor_tiempo_extra

        # Costo básico
        costo_base = self._calculate_simple_cost(
            distance_km, tipo_flota, location_split['cantidad'], factores_externos
        )

        # Probabilidad básica
        probabilidad = self._calculate_simple_probability(
            distance_km, tiempo_total, factores_externos, tipo_flota
        )

        return {
            'ruta_id': f"direct_store_{tienda_id}",
            'tipo_ruta': 'directa_tienda',
            'origen_principal': tienda_id,
            'segmentos': [
                {
                    'origen': tienda_id,
                    'destino': 'cliente',
                    'distancia_km': distance_km,
                    'tiempo_horas': base_travel_time,
                    'tipo_flota': tipo_flota,
                    'costo_segmento': costo_base
                }
            ],
            'tiempo_total_horas': tiempo_total,
            'costo_total_mxn': costo_base,
            'distancia_total_km': distance_km,
            'probabilidad_cumplimiento': probabilidad,
            'cantidad_cubierta': location_split['cantidad'],
            'factores_aplicados': [
                f"factor_demanda_{factores_externos.get('factor_demanda', 1.0)}",
                f"trafico_{factores_externos.get('trafico_nivel', 'Moderado')}",
                f"flota_{tipo_flota}",
                'ruta_directa_optimizada'
            ],
            'tiempo_preparacion_total': tiempo_preparacion,
            'tiempo_viaje_base': base_travel_time,
            'impacto_factores_externos': factor_tiempo_extra
        }

    def _create_smart_consolidated_route(self,
                                         split_plan: List[Dict[str, Any]],
                                         target_coordinates: Tuple[float, float],
                                         factores_externos: Dict[str, Any],
                                         repositories: Dict[str, Any]) -> Dict[str, Any]:
        """🔄 Consolidación INTELIGENTE mejorada"""

        if len(split_plan) < 2:
            return None

        # Encontrar tienda más cercana al cliente como HUB
        store_distances = []
        for location_split in split_plan:
            store = repositories['store'].get_store_by_id(location_split['tienda_id'])
            if store:
                distance = GeoCalculator.calculate_distance_km(
                    store['latitud'], store['longitud'],
                    target_coordinates[0], target_coordinates[1]
                )
                store_distances.append({
                    'location': location_split,
                    'store': store,
                    'distance_to_client': distance
                })

        if not store_distances:
            return None

        # Ordenar por distancia al cliente
        store_distances.sort(key=lambda x: x['distance_to_client'])
        consolidation_hub = store_distances[0]
        other_stores = store_distances[1:]

        # Evaluar si vale la pena consolidar
        consolidation_cost = self._calculate_consolidation_cost_simple(
            consolidation_hub, other_stores, factores_externos
        )

        direct_costs = sum([
            self._calculate_simple_cost(
                store_dist['distance_to_client'], 'FI',
                store_dist['location']['cantidad'], factores_externos
            )
            for store_dist in store_distances
        ])

        # Si consolidar es más caro, no hacerlo
        if consolidation_cost > direct_costs * 1.2:
            return None

        # Construir ruta consolidada
        return self._build_consolidation_route_simple(
            consolidation_hub, other_stores, target_coordinates, factores_externos
        )

    def _calculate_simple_cost(self, distance_km: float, fleet_type: str,
                               cantidad: int, factores_externos: Dict[str, Any]) -> float:
        """💰 Cálculo de costo simplificado pero realista"""

        distance_km = max(distance_km, 5.0)

        # Costos base
        cost_per_km = {
            'FI': 12.0,
            'FE': 18.0
        }.get(fleet_type, 12.0)

        base_cost = distance_km * cost_per_km

        # Factor por cantidad
        if cantidad >= 5:
            quantity_factor = 0.85
        elif cantidad >= 3:
            quantity_factor = 0.92
        else:
            quantity_factor = 1.0

        # Factor de demanda
        demand_factor = factores_externos.get('factor_demanda', 1.0)
        if demand_factor > 3.0:
            cost_multiplier = 1.8
        elif demand_factor > 2.5:
            cost_multiplier = 1.5
        elif demand_factor > 2.0:
            cost_multiplier = 1.0 + (demand_factor - 2.0) * 0.4
        else:
            cost_multiplier = 1.0

        # Aplicar impacto de costo extra
        costo_extra_pct = factores_externos.get('impacto_costo_extra_pct', 0)
        if costo_extra_pct > 0:
            cost_multiplier = max(cost_multiplier, 1.0 + costo_extra_pct / 100)

        minimum_cost = 50.0
        final_cost = max(base_cost * quantity_factor * cost_multiplier, minimum_cost)

        return round(final_cost, 2)

    def _calculate_simple_probability(self, distance_km: float, tiempo_total: float,
                                      factores_externos: Dict[str, Any], fleet_type: str) -> float:
        """📊 Cálculo de probabilidad simplificado"""

        base_probability = {
            'FI': 0.90,
            'FE': 0.82
        }.get(fleet_type, 0.85)

        # Penalizaciones suaves
        distance_penalty = min(0.15, distance_km / 1000)
        time_penalty = min(0.10, max(0, (tiempo_total - 6) / 100))

        # Factor por condiciones externas
        external_factor = 1.0

        criticidad = factores_externos.get('criticidad_logistica', 'Normal')
        if criticidad == 'Crítica':
            external_factor *= 0.85
        elif criticidad == 'Alta':
            external_factor *= 0.90
        elif criticidad == 'Media':
            external_factor *= 0.95

        if factores_externos.get('trafico_nivel') == 'Alto':
            external_factor *= 0.95
        elif factores_externos.get('trafico_nivel') == 'Muy_Alto':
            external_factor *= 0.90

        final_probability = (base_probability - distance_penalty - time_penalty) * external_factor
        return round(max(0.60, min(0.98, final_probability)), 3)

    def _calculate_consolidation_cost_simple(self, hub_info: Dict[str, Any],
                                             other_stores: List[Dict[str, Any]],
                                             factores_externos: Dict[str, Any]) -> float:
        """💰 Costo de consolidación simplificado"""

        total_cost = 0

        # Costo de recolección
        for store_info in other_stores:
            distance = GeoCalculator.calculate_distance_km(
                store_info['store']['latitud'], store_info['store']['longitud'],
                hub_info['store']['latitud'], hub_info['store']['longitud']
            )
            total_cost += distance * 8.0

        # Costo desde hub al cliente
        total_cost += hub_info['distance_to_client'] * 12.0

        # Aplicar factor de demanda
        demand_factor = factores_externos.get('factor_demanda', 1.0)
        total_cost *= min(2.0, demand_factor)

        return total_cost

    def _build_consolidation_route_simple(self, hub_info: Dict[str, Any],
                                          other_stores: List[Dict[str, Any]],
                                          target_coordinates: Tuple[float, float],
                                          factores_externos: Dict[str, Any]) -> Dict[str, Any]:
        """🏗️ Construye ruta consolidada simplificada"""

        segmentos = []
        tiempo_total = settings.TIEMPO_PICKING_PACKING  # Preparación inicial
        costo_total = 0
        cantidad_total = hub_info['location']['cantidad']

        # Recolección desde otras tiendas
        current_location = hub_info['store']

        for store_info in other_stores:
            distance = GeoCalculator.calculate_distance_km(
                current_location['latitud'], current_location['longitud'],
                store_info['store']['latitud'], store_info['store']['longitud']
            )

            travel_time = GeoCalculator.calculate_travel_time(
                distance, 'FI',
                factores_externos.get('trafico_nivel', 'Moderado'),
                factores_externos.get('condicion_clima', 'Templado')
            )

            segmentos.append({
                'origen': current_location['tienda_id'],
                'destino': store_info['store']['tienda_id'],
                'distancia_km': distance,
                'tiempo_horas': travel_time,
                'tipo_flota': 'FI'
            })

            tiempo_total += travel_time + settings.TIEMPO_PICKING_PACKING
            costo_total += distance * 8.0
            cantidad_total += store_info['location']['cantidad']
            current_location = store_info['store']

        # Segmento final al cliente
        final_distance = GeoCalculator.calculate_distance_km(
            current_location['latitud'], current_location['longitud'],
            target_coordinates[0], target_coordinates[1]
        )

        final_travel_time = GeoCalculator.calculate_travel_time(
            final_distance, 'FI',
            factores_externos.get('trafico_nivel', 'Moderado'),
            factores_externos.get('condicion_clima', 'Templado')
        )

        segmentos.append({
            'origen': current_location['tienda_id'],
            'destino': 'cliente',
            'distancia_km': final_distance,
            'tiempo_horas': final_travel_time,
            'tipo_flota': 'FI'
        })

        tiempo_total += final_travel_time
        costo_total += self._calculate_simple_cost(
            final_distance, 'FI', cantidad_total, factores_externos
        )

        # Aplicar factores externos
        tiempo_total += factores_externos.get('impacto_tiempo_extra_horas', 0)

        distancia_total = sum(seg['distancia_km'] for seg in segmentos)
        probabilidad = self._calculate_simple_probability(
            distancia_total, tiempo_total, factores_externos, 'FI'
        )

        return {
            'ruta_id': f"consolidated_{hub_info['store']['tienda_id']}_{len(other_stores)}",
            'tipo_ruta': 'consolidada_optimizada',
            'origen_principal': hub_info['store']['tienda_id'],
            'segmentos': segmentos,
            'tiempo_total_horas': tiempo_total,
            'costo_total_mxn': costo_total,
            'distancia_total_km': distancia_total,
            'probabilidad_cumplimiento': probabilidad,
            'cantidad_cubierta': cantidad_total,
            'factores_aplicados': [
                'consolidacion_inteligente',
                f"hub_{hub_info['store']['tienda_id']}",
                f"recolecciones_{len(other_stores)}",
                'sin_cedis_directo'
            ],
            'tiempo_preparacion_total': settings.TIEMPO_PICKING_PACKING * (len(other_stores) + 1)
        }

    # Métodos de ranking (mantener los existentes pero mejorados)
    def rank_candidates_with_lightgbm(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """🏆 Rankea candidatos con criterios mejorados"""

        if not candidates:
            return []

        return self._intelligent_multiobj_ranking_enhanced(candidates)

    def _intelligent_multiobj_ranking_enhanced(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """📊 Ranking MEJORADO con penalización para rutas complejas"""

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

        for candidate in candidates:
            # Scores normalizados
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

            # Score combinado
            score_combinado = (
                    self.weights['tiempo'] * score_tiempo +
                    self.weights['costo'] * score_costo +
                    self.weights['probabilidad'] * score_probabilidad +
                    self.weights['distancia'] * score_distancia
            )

            # Bonus por simplicidad de ruta
            if candidate['tipo_ruta'] == 'directa_tienda':
                score_combinado *= 1.1  # Bonus para rutas directas
            elif candidate['tipo_ruta'] == 'consolidada_optimizada':
                num_segmentos = len(candidate.get('segmentos', []))
                if num_segmentos <= 3:
                    score_combinado *= 1.02  # Bonus menor para consolidación simple

            candidate['score_lightgbm'] = round(score_combinado, 4)
            candidate['score_breakdown'] = {
                'tiempo': round(score_tiempo, 3),
                'costo': round(score_costo, 3),
                'distancia': round(score_distancia, 3),
                'probabilidad': round(score_probabilidad, 3)
            }

        # Ordenar por score
        ranked_candidates = sorted(candidates, key=lambda x: x['score_lightgbm'], reverse=True)

        # Asignar posiciones
        for i, candidate in enumerate(ranked_candidates):
            candidate['ranking_position'] = i + 1

        logger.info(f"📊 {len(candidates)} candidatos rankeados (sin CEDIS directos)")
        return ranked_candidates

    def get_top_candidates(self, ranked_candidates: List[Dict[str, Any]],
                           max_candidates: int = None) -> List[Dict[str, Any]]:
        """🏆 Obtiene top candidatos con diversidad"""

        max_candidates = max_candidates or settings.TOP_CANDIDATOS_GEMINI

        if not ranked_candidates:
            return []

        top_candidates = ranked_candidates[:max_candidates]
        diverse_candidates = self._ensure_route_diversity(top_candidates, max_candidates)

        logger.info(f"🏆 Seleccionados {len(diverse_candidates)} candidatos diversos")
        return diverse_candidates

    def _ensure_route_diversity(self, candidates: List[Dict[str, Any]],
                                max_candidates: int) -> List[Dict[str, Any]]:
        """🎯 Asegura diversidad de tipos de ruta"""

        if len(candidates) <= max_candidates:
            return candidates

        # Agrupar por tipo
        by_type = {}
        for candidate in candidates:
            route_type = candidate['tipo_ruta']
            if route_type not in by_type:
                by_type[route_type] = []
            by_type[route_type].append(candidate)

        # Priorizar rutas directas
        type_priority = [
            'directa_tienda',
            'consolidada_optimizada'
        ]

        diverse_candidates = []

        # Tomar al menos uno de cada tipo
        for route_type in type_priority:
            if route_type in by_type and len(diverse_candidates) < max_candidates:
                best_of_type = max(by_type[route_type], key=lambda x: x['score_lightgbm'])
                diverse_candidates.append(best_of_type)

        # Llenar espacios restantes con mejores scores
        remaining_candidates = [c for c in candidates if c not in diverse_candidates]
        remaining_slots = max_candidates - len(diverse_candidates)

        if remaining_slots > 0:
            top_remaining = sorted(remaining_candidates,
                                   key=lambda x: x['score_lightgbm'], reverse=True)
            diverse_candidates.extend(top_remaining[:remaining_slots])

        return diverse_candidates

    # Mantener métodos de carga de modelo
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