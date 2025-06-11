from typing import List, Dict, Any, Tuple

import joblib

from config.settings import settings
from utils.geo_calculator import GeoCalculator
from utils.logger import logger


class RouteOptimizer:
    """ Optimizador"""

    def __init__(self):
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.model_path = settings.MODELS_DIR / "route_optimizer_lgb.pkl"

        settings.MODELS_DIR.mkdir(exist_ok=True)
        self.weights = {
            'tiempo': 0.4,  # Más peso al tiempo
            'costo': 0.2,  # Menos peso al costo
            'probabilidad': 0.35,  # Más peso a confiabilidad
            'distancia': 0.05  # Menos peso a distancia
        }

    def generate_route_candidates(self,
                                  split_inventory: Dict[str, Any],
                                  target_coordinates: Tuple[float, float],
                                  factores_externos: Dict[str, Any],
                                  repositories: Dict[str, Any]) -> List[Dict[str, Any]]:

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
                        'distancia_km': 0  # TODO esto lo voy a calcular despues [  RECORDATORIO ]
                    })
                logger.info(f"📊 Extraído split_plan desde objeto SplitInventory: {len(split_plan)} ubicaciones")

            if not split_plan:
                logger.warning("❌ No se pudo extraer split_plan")
                return []

        logger.info(f"📊 Split plan obtenido: {len(split_plan)} ubicaciones")

        if len(split_plan) == 1:
            primary_candidate = self._create_optimized_direct_route(
                split_plan[0], target_coordinates, factores_externos, repositories
            )
            if primary_candidate:
                candidates.append(primary_candidate)

        elif len(split_plan) > 1:
            consolidated_candidate = self._create_intelligent_consolidated_route(
                split_plan, target_coordinates, factores_externos, repositories
            )
            if consolidated_candidate:
                candidates.append(consolidated_candidate)

            for location in split_plan:
                full_stock_candidate = self._check_full_stock_alternative(
                    location, split_inventory, target_coordinates, factores_externos, repositories
                )
                if full_stock_candidate:
                    candidates.append(full_stock_candidate)

        distance_to_closest = min([
            GeoCalculator.calculate_distance_km(
                target_lat, target_lon,
                repositories['store'].get_store_by_id(loc['tienda_id'])['latitud'],
                repositories['store'].get_store_by_id(loc['tienda_id'])['longitud']
            ) for loc in split_plan
            if repositories['store'].get_store_by_id(loc['tienda_id'])
        ])

        if distance_to_closest > 100:
            hybrid_candidates = self._create_strategic_hybrid_routes(
                split_plan, target_coordinates, factores_externos, repositories
            )
            candidates.extend(hybrid_candidates)

        cedis_candidates = self._create_selective_cedis_routes(
            split_plan, target_coordinates, factores_externos, repositories, distance_to_closest
        )
        candidates.extend(cedis_candidates)

        logger.info(f"🔄 Generados {len(candidates)} candidatos INTELIGENTES")
        return candidates

    def _create_optimized_direct_route(self,
                                       location_split: Dict[str, Any],
                                       target_coordinates: Tuple[float, float],
                                       factores_externos: Dict[str, Any],
                                       repositories: Dict[str, Any]) -> Dict[str, Any]:
        """📍 Crea ruta directa OPTIMIZADA"""

        tienda_id = location_split['tienda_id']
        store_info = repositories['store'].get_store_by_id(tienda_id)

        if not store_info:
            return None

        distance_km = GeoCalculator.calculate_distance_km(
            store_info['latitud'], store_info['longitud'],
            target_coordinates[0], target_coordinates[1]
        )

        distance_km = max(distance_km, 5.0)

        tipo_flota = self._determine_optimal_fleet_type(
            distance_km, factores_externos, repositories, target_coordinates
        )

        base_travel_time = GeoCalculator.calculate_travel_time(
            distance_km,
            tipo_flota,
            factores_externos.get('trafico_nivel', 'Moderado'),
            factores_externos.get('condicion_clima', 'Templado')
        )

        impacto_tiempo_extra = factores_externos.get('impacto_tiempo_extra_horas', 0)
        tiempo_preparacion = settings.TIEMPO_PICKING_PACKING
        tiempo_total = tiempo_preparacion + base_travel_time + impacto_tiempo_extra

        logger.info(
            f"⏱️ Tiempo calculado: prep={tiempo_preparacion}h + viaje={base_travel_time}h + extra={impacto_tiempo_extra}h = {tiempo_total}h")

        costo_base = self._calculate_realistic_cost(
            distance_km, tipo_flota, location_split['cantidad'], factores_externos
        )

        probabilidad = self._calculate_realistic_probability(
            distance_km, tiempo_total, factores_externos, tipo_flota
        )

        return {
            'ruta_id': f"direct_optimal_{tienda_id}",
            'tipo_ruta': 'directa_optimizada',
            'origen_principal': tienda_id,
            'segmentos': [
                {
                    'origen': tienda_id,
                    'destino': 'cliente',
                    'distancia_km': distance_km,
                    'tiempo_horas': base_travel_time,  # Solo tiempo de viaje
                    'tipo_flota': tipo_flota,
                    'costo_segmento': costo_base
                }
            ],
            'tiempo_total_horas': tiempo_total,  # Tiempo total con factores
            'costo_total_mxn': costo_base,
            'distancia_total_km': distance_km,
            'probabilidad_cumplimiento': probabilidad,
            'cantidad_cubierta': location_split['cantidad'],
            'factores_aplicados': [
                f"factor_demanda_{factores_externos.get('factor_demanda', 1.0)}",
                f"trafico_{factores_externos.get('trafico_nivel', 'Moderado')}",
                f"flota_{tipo_flota}",
                f"impacto_tiempo_+{impacto_tiempo_extra}h",
                'optimizacion_inteligente'
            ],
            'tiempo_preparacion_total': tiempo_preparacion,
            'tiempo_viaje_base': base_travel_time,
            'impacto_factores_externos': impacto_tiempo_extra
        }

    def _create_intelligent_consolidated_route(self,
                                               split_plan: List[Dict[str, Any]],
                                               target_coordinates: Tuple[float, float],
                                               factores_externos: Dict[str, Any],
                                               repositories: Dict[str, Any]) -> Dict[str, Any]:
        """🔄 Consolidación INTELIGENTE: evalúa si vale la pena"""

        if len(split_plan) < 2:
            return None

        # Encontrar la ubicación MÁS CERCANA al cliente
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
        consolidation_hub = store_distances[0]  # La más cercana al cliente
        other_stores = store_distances[1:]

        # DECISIÓN INTELIGENTE: ¿Vale la pena consolidar?
        consolidation_cost = self._calculate_consolidation_cost(consolidation_hub, other_stores)
        direct_costs = sum([
            self._calculate_direct_route_cost(store_dist, target_coordinates, factores_externos)
            for store_dist in store_distances
        ])

        # Si consolidar es más caro que envíos directos, NO consolidar
        if consolidation_cost > direct_costs * 1.3:  # 30% de tolerancia
            logger.info("📊 Consolidación no eficiente, preferir rutas directas")
            return None

        # Crear ruta consolidada
        return self._build_consolidation_route(
            consolidation_hub, other_stores, target_coordinates, factores_externos
        )

    def _check_full_stock_alternative(self,
                                      location: Dict[str, Any],
                                      split_inventory: Dict[str, Any],
                                      target_coordinates: Tuple[float, float],
                                      factores_externos: Dict[str, Any],
                                      repositories: Dict[str, Any]) -> Dict[str, Any]:
        """🔍 Verifica si una ubicación puede cubrir TODA la demanda"""

        tienda_id = location['tienda_id']

        # Buscar stock real de esta tienda para el SKU
        stock_locations = repositories['stock'].get_stock_locations(
            split_inventory.get('sku_id', ''),
            split_inventory.get('cantidad_total_requerida', 0)
        )

        tienda_stock = None
        for stock_loc in stock_locations:
            if stock_loc['tienda_id'] == tienda_id:
                tienda_stock = stock_loc
                break

        if not tienda_stock:
            return None

        # Si esta tienda puede cubrir TODA la demanda
        cantidad_requerida = split_inventory.get('cantidad_total_requerida', 0)
        if tienda_stock['stock_disponible'] >= cantidad_requerida:
            # Crear ruta alternativa usando TODA la demanda desde esta tienda
            alternative_location = {
                'tienda_id': tienda_id,
                'cantidad': cantidad_requerida
            }

            return self._create_optimized_direct_route(
                alternative_location, target_coordinates, factores_externos, repositories
            )

        return None

    def _create_strategic_hybrid_routes(self,
                                        split_plan: List[Dict[str, Any]],
                                        target_coordinates: Tuple[float, float],
                                        factores_externos: Dict[str, Any],
                                        repositories: Dict[str, Any]) -> List[Dict[str, Any]]:
        """🔀 Rutas híbridas ESTRATÉGICAS (solo cuando vale la pena)"""

        candidates = []

        # Solo crear híbridas para el 50% de ubicaciones más prometedoras
        promising_locations = split_plan[:max(1, len(split_plan) // 2)]

        for location_split in promising_locations:
            store_info = repositories['store'].get_store_by_id(location_split['tienda_id'])
            if not store_info:
                continue

            # Encontrar CEDIS más estratégico (no solo el más cercano)
            strategic_cedis = self._find_strategic_cedis(
                store_info, target_coordinates, repositories
            )

            if strategic_cedis:
                hybrid_candidate = self._build_strategic_hybrid_route(
                    location_split, strategic_cedis, target_coordinates, factores_externos, repositories
                )
                if hybrid_candidate:
                    candidates.append(hybrid_candidate)

        return candidates

    def _create_selective_cedis_routes(self,
                                       split_plan: List[Dict[str, Any]],
                                       target_coordinates: Tuple[float, float],
                                       factores_externos: Dict[str, Any],
                                       repositories: Dict[str, Any],
                                       distance_to_closest: float) -> List[Dict[str, Any]]:
        """🏭 Rutas CEDIS SELECTIVAS (solo casos específicos)"""

        candidates = []

        # SOLO crear rutas CEDIS si:
        # 1. La distancia es muy larga (>200km)
        # 2. Es temporada crítica
        # 3. Hay problemas de stock en tiendas cercanas

        should_use_cedis = (
                distance_to_closest > 200 or
                factores_externos.get('es_temporada_critica', False) or
                len(split_plan) > 2  # Muchas ubicaciones = problema de stock
        )

        if not should_use_cedis:
            return candidates

        # Obtener solo CEDIS estratégicos (no todos)
        strategic_cedis_list = self._get_strategic_cedis_only(target_coordinates, repositories)

        # Máximo 3 CEDIS para no saturar
        for cedis in strategic_cedis_list[:3]:
            cedis_candidate = self._create_efficient_cedis_route(
                cedis, split_plan, target_coordinates, factores_externos, repositories
            )
            if cedis_candidate:
                candidates.append(cedis_candidate)

        return candidates

    def _determine_optimal_fleet_type(self, distance_km: float,
                                      factores_externos: Dict[str, Any],
                                      repositories: Dict[str, Any],
                                      target_coordinates: Tuple[float, float]) -> str:
        """🚛 Determina tipo de flota ÓPTIMO"""

        # Verificar zona roja
        cp_approx = f"{int(target_coordinates[0] * 100):05d}"[:5]
        is_zona_roja = repositories['postal_code'].is_zona_roja(cp_approx)

        # Factores de decisión
        es_temporada_critica = factores_externos.get('es_temporada_critica', False)
        trafico_alto = factores_externos.get('trafico_nivel') in ['Alto', 'Muy_Alto']

        # Lógica optimizada
        if distance_km <= 50 and not is_zona_roja:
            return 'FI'  # Flota interna para distancias cortas y zonas seguras
        elif distance_km > 150 or is_zona_roja:
            return 'FE'  # Flota externa para distancias largas o zonas rojas
        elif es_temporada_critica or trafico_alto:
            return 'FE'  # Flota externa en condiciones críticas
        else:
            return 'FI'  # Default: flota interna

    def _calculate_realistic_cost(self, distance_km: float, fleet_type: str,
                                  cantidad: int, factores_externos: Dict[str, Any]) -> float:
        """💰 Calcula costo REALISTA con factores externos aplicados"""

        # Asegurar distancia mínima para cálculo realista
        distance_km = max(distance_km, 5.0)

        # Costos base más realistas
        cost_per_km = {
            'FI': 12.0,  # Liverpool flota interna (aumentado)
            'FE': 18.0,  # Flota externa (aumentado)
            'FI_FE': 15.0  # Híbrido
        }

        base_cost = distance_km * cost_per_km.get(fleet_type, 12.0)

        # Factor por cantidad (economías de escala reales)
        if cantidad >= 5:
            quantity_factor = 0.85  # 15% descuento por volumen
        elif cantidad >= 3:
            quantity_factor = 0.92  # 8% descuento por volumen
        else:
            quantity_factor = 1.0

        # Factor por demanda REAL (del CSV o calculado)
        demand_factor = factores_externos.get('factor_demanda', 1.0)

        # Aplicar factor de demanda de manera realista
        if demand_factor > 3.0:
            cost_multiplier = 1.8  # 80% incremento para Navidad
        elif demand_factor > 2.5:
            cost_multiplier = 1.5  # 50% incremento temporada crítica
        elif demand_factor > 2.0:
            cost_multiplier = 1.0 + (demand_factor - 2.0) * 0.4  # Escalado gradual
        else:
            cost_multiplier = 1.0

        # Aplicar impacto de costo extra (del CSV)
        costo_extra_pct = factores_externos.get('impacto_costo_extra_pct', 0)
        if costo_extra_pct > 0:
            cost_multiplier = max(cost_multiplier, 1.0 + costo_extra_pct / 100)

        # Costo mínimo realista
        minimum_cost = 50.0  # Aumentado para ser más realista

        final_cost = max(base_cost * quantity_factor * cost_multiplier, minimum_cost)

        logger.info(
            f"💰 Costo calculado: base=${base_cost:.1f} × qty_factor={quantity_factor} × demand_factor={cost_multiplier:.2f} = ${final_cost:.1f}")

        return round(final_cost, 2)

    def _calculate_realistic_probability(self, distance_km: float, tiempo_total: float,
                                         factores_externos: Dict[str, Any], fleet_type: str) -> float:
        """📊 Calcula probabilidad REALISTA de cumplimiento"""

        # Probabilidad base por tipo de flota
        base_probability = {
            'FI': 0.90,  # Liverpool tiene buen control
            'FE': 0.82,  # Externos menos control
            'FI_FE': 0.86  # Híbrido intermedio
        }.get(fleet_type, 0.85)

        # Penalización por distancia (más suave)
        distance_penalty = min(0.15, distance_km / 1000)  # Máximo 15% de penalización

        # Penalización por tiempo (más suave)
        time_penalty = min(0.10, max(0, (tiempo_total - 6) / 100))  # Más tolerante

        # Factor por condiciones externas
        external_factor = 1.0

        if factores_externos.get('es_temporada_critica', False):
            external_factor *= 0.90  # 10% reducción en temporada crítica
        elif factores_externos.get('es_temporada_alta', False):
            external_factor *= 0.95  # 5% reducción en temporada alta

        if factores_externos.get('trafico_nivel') == 'Alto':
            external_factor *= 0.95
        elif factores_externos.get('trafico_nivel') == 'Muy_Alto':
            external_factor *= 0.90

        # Cálculo final
        final_probability = (base_probability - distance_penalty - time_penalty) * external_factor

        # Rango realista: 60%-98%
        return round(max(0.60, min(0.98, final_probability)), 3)

    def _calculate_consolidation_cost(self, hub_info: Dict[str, Any],
                                      other_stores: List[Dict[str, Any]]) -> float:
        """💰 Calcula costo de consolidación"""

        total_cost = 0

        # Costo de recolección desde otras tiendas al hub
        for store_info in other_stores:
            distance = GeoCalculator.calculate_distance_km(
                store_info['store']['latitud'], store_info['store']['longitud'],
                hub_info['store']['latitud'], hub_info['store']['longitud']
            )
            total_cost += distance * 8.0  # Costo interno de recolección

        # Costo desde hub al cliente
        total_cost += hub_info['distance_to_client'] * 12.0

        return total_cost

    def _calculate_direct_route_cost(self, store_info: Dict[str, Any],
                                     target_coordinates: Tuple[float, float],
                                     factores_externos: Dict[str, Any]) -> float:
        """💰 Calcula costo de ruta directa para comparación"""

        distance = store_info['distance_to_client']
        return self._calculate_realistic_cost(distance, 'FI', 1, factores_externos)

    def _find_strategic_cedis(self, store_info: Dict[str, Any],
                              target_coordinates: Tuple[float, float],
                              repositories: Dict[str, Any]) -> Dict[str, Any]:
        """🎯 Encuentra CEDIS estratégico (no solo el más cercano)"""

        cedis_raw_list = repositories['cedis'].load_data().to_dicts()
        cedis_candidates = []

        for cedis_raw in cedis_raw_list:
            cedis_id_clean = repositories['cedis']._clean_id_value(cedis_raw.get('cedis_id', ''))
            if not cedis_id_clean:
                continue

            lat_clean = repositories['cedis']._clean_coordinate_value(cedis_raw.get('latitud'))
            lon_clean = repositories['cedis']._clean_coordinate_value(cedis_raw.get('longitud'))

            if 14.0 <= lat_clean <= 33.0 and -118.0 <= lon_clean <= -86.0:
                # Calcular eficiencia estratégica
                dist_store_cedis = GeoCalculator.calculate_distance_km(
                    store_info['latitud'], store_info['longitud'], lat_clean, lon_clean
                )
                dist_cedis_client = GeoCalculator.calculate_distance_km(
                    lat_clean, lon_clean, target_coordinates[0], target_coordinates[1]
                )

                # Score estratégico: balancear distancias
                strategic_score = 1.0 / (1.0 + dist_store_cedis * 0.01 + dist_cedis_client * 0.01)

                cedis_candidates.append({
                    'cedis_id': cedis_id_clean,
                    'latitud': lat_clean,
                    'longitud': lon_clean,
                    'strategic_score': strategic_score,
                    'total_distance': dist_store_cedis + dist_cedis_client
                })

        if cedis_candidates:
            # Ordenar por score estratégico
            cedis_candidates.sort(key=lambda x: x['strategic_score'], reverse=True)
            return cedis_candidates[0]

        return None

    def _get_strategic_cedis_only(self, target_coordinates: Tuple[float, float],
                                  repositories: Dict[str, Any]) -> List[Dict[str, Any]]:
        """🏭 Obtiene solo CEDIS estratégicos"""

        cedis_raw_list = repositories['cedis'].load_data().to_dicts()
        strategic_cedis = []

        for cedis_raw in cedis_raw_list:
            cedis_id_clean = repositories['cedis']._clean_id_value(cedis_raw.get('cedis_id', ''))
            if not cedis_id_clean:
                continue

            lat_clean = repositories['cedis']._clean_coordinate_value(cedis_raw.get('latitud'))
            lon_clean = repositories['cedis']._clean_coordinate_value(cedis_raw.get('longitud'))

            if 14.0 <= lat_clean <= 33.0 and -118.0 <= lon_clean <= -86.0:
                distance_to_target = GeoCalculator.calculate_distance_km(
                    lat_clean, lon_clean, target_coordinates[0], target_coordinates[1]
                )

                # Solo CEDIS relativamente cercanos (dentro de 300km)
                if distance_to_target <= 300:
                    strategic_cedis.append({
                        'cedis_id': cedis_id_clean,
                        'latitud': lat_clean,
                        'longitud': lon_clean,
                        'distance_to_target': distance_to_target
                    })

        # Ordenar por distancia y tomar los mejores
        strategic_cedis.sort(key=lambda x: x['distance_to_target'])
        return strategic_cedis

    def _build_consolidation_route(self, hub_info: Dict[str, Any],
                                   other_stores: List[Dict[str, Any]],
                                   target_coordinates: Tuple[float, float],
                                   factores_externos: Dict[str, Any]) -> Dict[str, Any]:
        """🏗️ Construye ruta de consolidación"""

        segmentos = []
        tiempo_total = 0
        costo_total = 0
        cantidad_total = hub_info['location']['cantidad']

        # Tiempo de preparación en hub
        tiempo_total += settings.TIEMPO_PICKING_PACKING

        # Segmentos de recolección
        for i, store_info in enumerate(other_stores):
            if i == 0:
                # Primer segmento: hub -> otra tienda
                origen = hub_info['store']
                destino = store_info['store']
            else:
                # Segmentos siguientes: tienda anterior -> siguiente tienda
                origen = other_stores[i - 1]['store']
                destino = store_info['store']

            distance = GeoCalculator.calculate_distance_km(
                origen['latitud'], origen['longitud'],
                destino['latitud'], destino['longitud']
            )

            travel_time = GeoCalculator.calculate_travel_time(
                distance, 'FI',
                factores_externos.get('trafico_nivel', 'Moderado'),
                factores_externos.get('condicion_clima', 'Templado')
            )

            segmentos.append({
                'origen': origen['tienda_id'],
                'destino': destino['tienda_id'],
                'distancia_km': distance,
                'tiempo_horas': travel_time,
                'tipo_flota': 'FI'
            })

            tiempo_total += travel_time + settings.TIEMPO_PICKING_PACKING  # Tiempo de recogida
            costo_total += distance * 8.0  # Costo interno
            cantidad_total += store_info['location']['cantidad']

        # Segmento final: última tienda -> cliente
        last_store = other_stores[-1]['store'] if other_stores else hub_info['store']
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
        costo_total += self._calculate_realistic_cost(final_distance, 'FI', cantidad_total, factores_externos)

        distancia_total = sum(seg['distancia_km'] for seg in segmentos)
        probabilidad = self._calculate_realistic_probability(
            distancia_total, tiempo_total, factores_externos, 'FI'
        )

        return {
            'ruta_id': f"consolidated_{hub_info['store']['tienda_id']}_{len(other_stores)}",
            'tipo_ruta': 'consolidada_inteligente',
            'origen_principal': hub_info['store']['tienda_id'],
            'segmentos': segmentos,
            'tiempo_total_horas': tiempo_total,
            'costo_total_mxn': costo_total,
            'distancia_total_km': distancia_total,
            'probabilidad_cumplimiento': probabilidad,
            'cantidad_cubierta': cantidad_total,
            'factores_aplicados': [
                'consolidacion_optimizada',
                f"hub_{hub_info['store']['tienda_id']}",
                f"recolecciones_{len(other_stores)}"
            ],
            'tiempo_preparacion_total': settings.TIEMPO_PICKING_PACKING * (len(other_stores) + 1)
        }

    def _build_strategic_hybrid_route(self, location_split: Dict[str, Any],
                                      strategic_cedis: Dict[str, Any],
                                      target_coordinates: Tuple[float, float],
                                      factores_externos: Dict[str, Any],
                                      repositories: Dict[str, Any]) -> Dict[str, Any]:
        """🔀 Construye ruta híbrida estratégica"""

        store_info = repositories['store'].get_store_by_id(location_split['tienda_id'])
        if not store_info:
            return None

        # Segmento 1: Tienda -> CEDIS (FI)
        dist_to_cedis = GeoCalculator.calculate_distance_km(
            store_info['latitud'], store_info['longitud'],
            strategic_cedis['latitud'], strategic_cedis['longitud']
        )

        time_to_cedis = GeoCalculator.calculate_travel_time(
            dist_to_cedis, 'FI',
            factores_externos.get('trafico_nivel', 'Moderado'),
            factores_externos.get('condicion_clima', 'Templado')
        )

        # Segmento 2: CEDIS -> Cliente (FE)
        dist_to_client = GeoCalculator.calculate_distance_km(
            strategic_cedis['latitud'], strategic_cedis['longitud'],
            target_coordinates[0], target_coordinates[1]
        )

        time_to_client = GeoCalculator.calculate_travel_time(
            dist_to_client, 'FE',
            factores_externos.get('trafico_nivel', 'Moderado'),
            factores_externos.get('condicion_clima', 'Templado')
        )

        # Tiempos totales
        tiempo_prep_tienda = settings.TIEMPO_PICKING_PACKING
        tiempo_prep_cedis = settings.TIEMPO_PREPARACION_CEDIS
        tiempo_total = tiempo_prep_tienda + time_to_cedis + tiempo_prep_cedis + time_to_client

        # Costos
        costo_fi = self._calculate_realistic_cost(dist_to_cedis, 'FI', location_split['cantidad'], factores_externos)
        costo_fe = self._calculate_realistic_cost(dist_to_client, 'FE', location_split['cantidad'], factores_externos)
        costo_total = costo_fi + costo_fe

        # Probabilidad (híbridos suelen ser más confiables)
        probabilidad = self._calculate_realistic_probability(
            dist_to_cedis + dist_to_client, tiempo_total, factores_externos, 'FI_FE'
        )
        probabilidad = min(0.95, probabilidad * 1.05)  # Bonus por redundancia

        return {
            'ruta_id': f"hybrid_strategic_{location_split['tienda_id']}_{strategic_cedis['cedis_id']}",
            'tipo_ruta': 'hibrida_estrategica',
            'origen_principal': location_split['tienda_id'],
            'cedis_intermedio': strategic_cedis['cedis_id'],
            'segmentos': [
                {
                    'origen': location_split['tienda_id'],
                    'destino': strategic_cedis['cedis_id'],
                    'distancia_km': dist_to_cedis,
                    'tiempo_horas': time_to_cedis,
                    'tipo_flota': 'FI'
                },
                {
                    'origen': strategic_cedis['cedis_id'],
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
                'hibrida_estrategica',
                f"cedis_{strategic_cedis['cedis_id']}",
                'optimizacion_distancia_tiempo'
            ],
            'tiempo_preparacion_total': tiempo_prep_tienda + tiempo_prep_cedis
        }

    def _create_efficient_cedis_route(self, cedis_info: Dict[str, Any],
                                      split_plan: List[Dict[str, Any]],
                                      target_coordinates: Tuple[float, float],
                                      factores_externos: Dict[str, Any],
                                      repositories: Dict[str, Any]) -> Dict[str, Any]:
        """🏭 Crea ruta CEDIS eficiente"""

        total_cantidad = sum(item['cantidad'] for item in split_plan)

        # Calcular distancia y tiempo CEDIS -> Cliente
        dist_cedis_client = GeoCalculator.calculate_distance_km(
            cedis_info['latitud'], cedis_info['longitud'],
            target_coordinates[0], target_coordinates[1]
        )

        time_cedis_client = GeoCalculator.calculate_travel_time(
            dist_cedis_client, 'FE',  # CEDIS usa flota externa
            factores_externos.get('trafico_nivel', 'Moderado'),
            factores_externos.get('condicion_clima', 'Templado')
        )

        # Tiempo de preparación en CEDIS
        tiempo_prep_cedis = settings.TIEMPO_PREPARACION_CEDIS
        tiempo_total = tiempo_prep_cedis + time_cedis_client

        # Costo
        costo_total = self._calculate_realistic_cost(
            dist_cedis_client, 'FE', total_cantidad, factores_externos
        )

        # Probabilidad (CEDIS son más confiables para volúmenes grandes)
        probabilidad = self._calculate_realistic_probability(
            dist_cedis_client, tiempo_total, factores_externos, 'FE'
        )
        if total_cantidad >= 5:
            probabilidad = min(0.96, probabilidad * 1.08)  # Bonus por volumen

        return {
            'ruta_id': f"cedis_efficient_{cedis_info['cedis_id']}",
            'tipo_ruta': 'cedis_eficiente',
            'origen_principal': cedis_info['cedis_id'],
            'segmentos': [
                {
                    'origen': cedis_info['cedis_id'],
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
            'cantidad_cubierta': total_cantidad,
            'factores_aplicados': [
                'cedis_eficiente',
                f"volumen_{total_cantidad}",
                'flota_externa_especializada'
            ],
            'tiempo_preparacion_total': tiempo_prep_cedis
        }

    # Métodos de ranking MEJORADOS (mantener estructura original pero sin filtros restrictivos)
    def rank_candidates_with_lightgbm(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """🏆 Rankea candidatos SIN FILTROS RESTRICTIVOS"""

        if not candidates:
            return []

        # CAMBIO CLAVE: No filtrar por score mínimo, rankear TODOS los candidatos
        return self._intelligent_multiobj_ranking(candidates)

    def _intelligent_multiobj_ranking(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """📊 Ranking INTELIGENTE multiobjetivo"""

        if not candidates:
            return []

        # Normalizar métricas para comparación justa
        tiempos = [c['tiempo_total_horas'] for c in candidates]
        costos = [c['costo_total_mxn'] for c in candidates]
        distancias = [c['distancia_total_km'] for c in candidates]
        probabilidades = [c['probabilidad_cumplimiento'] for c in candidates]

        min_tiempo, max_tiempo = min(tiempos), max(tiempos)
        min_costo, max_costo = min(costos), max(costos)
        min_distancia, max_distancia = min(distancias), max(distancias)

        # Calcular scores inteligentes
        for candidate in candidates:
            # Score tiempo (invertido: menos tiempo = mejor score)
            if max_tiempo > min_tiempo:
                score_tiempo = 1 - (candidate['tiempo_total_horas'] - min_tiempo) / (max_tiempo - min_tiempo)
            else:
                score_tiempo = 1.0

            # Score costo (invertido: menos costo = mejor score)
            if max_costo > min_costo:
                score_costo = 1 - (candidate['costo_total_mxn'] - min_costo) / (max_costo - min_costo)
            else:
                score_costo = 1.0

            # Score distancia (invertido: menos distancia = mejor score)
            if max_distancia > min_distancia:
                score_distancia = 1 - (candidate['distancia_total_km'] - min_distancia) / (
                        max_distancia - min_distancia)
            else:
                score_distancia = 1.0

            # Score probabilidad (directo: más probabilidad = mejor score)
            score_probabilidad = candidate['probabilidad_cumplimiento']

            # Score combinado con pesos AJUSTADOS
            score_combinado = (
                    self.weights['tiempo'] * score_tiempo +
                    self.weights['costo'] * score_costo +
                    self.weights['probabilidad'] * score_probabilidad +
                    self.weights['distancia'] * score_distancia
            )

            # Bonus por tipo de ruta eficiente
            if candidate['tipo_ruta'] in ['directa_optimizada', 'cedis_eficiente']:
                score_combinado *= 1.05
            elif candidate['tipo_ruta'] == 'consolidada_inteligente':
                score_combinado *= 1.02

            candidate['score_lightgbm'] = round(score_combinado, 4)
            candidate['score_breakdown'] = {
                'tiempo': round(score_tiempo, 3),
                'costo': round(score_costo, 3),
                'distancia': round(score_distancia, 3),
                'probabilidad': round(score_probabilidad, 3)
            }

        # Ordenar por score combinado
        ranked_candidates = sorted(candidates, key=lambda x: x['score_lightgbm'], reverse=True)

        # Asignar posiciones
        for i, candidate in enumerate(ranked_candidates):
            candidate['ranking_position'] = i + 1

        logger.info(f"📊 {len(candidates)} candidatos rankeados inteligentemente")
        return ranked_candidates

    def get_top_candidates(self, ranked_candidates: List[Dict[str, Any]],
                           max_candidates: int = None) -> List[Dict[str, Any]]:
        """🏆 Obtiene top candidatos SIN FILTROS RESTRICTIVOS"""

        max_candidates = max_candidates or settings.TOP_CANDIDATOS_GEMINI

        if not ranked_candidates:
            return []

        # CAMBIO CLAVE: No filtrar por score mínimo, tomar los mejores disponibles
        top_candidates = ranked_candidates[:max_candidates]

        # Asegurar diversidad
        diverse_candidates = self._ensure_intelligent_diversity(top_candidates, max_candidates)

        logger.info(f"🏆 Seleccionados {len(diverse_candidates)} candidatos top para Gemini")
        return diverse_candidates

    def _ensure_intelligent_diversity(self, candidates: List[Dict[str, Any]],
                                      max_candidates: int) -> List[Dict[str, Any]]:
        """🎯 Asegura diversidad INTELIGENTE"""

        if len(candidates) <= max_candidates:
            return candidates

        # Agrupar por tipo de ruta
        by_type = {}
        for candidate in candidates:
            route_type = candidate['tipo_ruta']
            if route_type not in by_type:
                by_type[route_type] = []
            by_type[route_type].append(candidate)

        # Prioridad INTELIGENTE de tipos
        type_priority = [
            'directa_optimizada',
            'cedis_eficiente',
            'hibrida_estrategica',
            'consolidada_inteligente'
        ]

        diverse_candidates = []

        # Tomar al menos uno de cada tipo (si existe)
        for route_type in type_priority:
            if route_type in by_type and len(diverse_candidates) < max_candidates:
                best_of_type = max(by_type[route_type], key=lambda x: x['score_lightgbm'])
                diverse_candidates.append(best_of_type)

        # Llenar espacios restantes con los mejores scores
        remaining_candidates = [c for c in candidates if c not in diverse_candidates]
        remaining_slots = max_candidates - len(diverse_candidates)

        if remaining_slots > 0:
            top_remaining = sorted(remaining_candidates,
                                   key=lambda x: x['score_lightgbm'], reverse=True)
            diverse_candidates.extend(top_remaining[:remaining_slots])

        return diverse_candidates

    # Mantener métodos de entrenamiento y carga (sin cambios significativos)
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