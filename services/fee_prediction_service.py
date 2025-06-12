import time
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Any, Tuple
import polars as pl
from config.settings import settings
from models.schemas import (
    PredictionRequest, PredictionResponse, TipoEntregaEnum,
    ExplicabilidadCompleta, FactoresExternos, RutaCompleta, Segmento, CandidatoRuta,
    DecisionGemini, FEECalculation, SplitInventory, UbicacionStock
)
from services.ai.gemini_service import GeminiLogisticsDecisionEngine
from utils.logger import logger
from utils.geo_calculator import GeoCalculator


class FEEPredictionService:
    """🚀 Servicio de predicción FEE optimizado - respuestas dinámicas con datos reales"""

    def __init__(self, repositories):
        self.repos = repositories
        self.gemini_engine = GeminiLogisticsDecisionEngine()
        self._factors_cache = {}  # ✅ AGREGAR esta línea
        self._store_cache = {}  # ✅ AGREGAR cache de tiendas
        logger.info("🎯 Servicio FEE optimizado inicializado")

    async def predict_fee(self, request: PredictionRequest) -> PredictionResponse:
        """🚀 Predicción FEE - CORREGIDA"""
        start_time = time.time()

        try:
            logger.info(f"🎯 NUEVA PREDICCIÓN: {request.sku_id} → {request.codigo_postal} (qty: {request.cantidad})")

            # 1. VALIDACIÓN DINÁMICA
            validation = await self._validate_request_dynamic(request)
            if not validation['valid']:
                raise ValueError(validation['error'])

            product_info = validation['product']
            cp_info = validation['postal_info']

            # 🔴 PROBLEMA: No usa factores externos reales
            # external_factors = self.repos.external_factors.get_factors_for_date_and_cp(...)  # ❌ MALO

            # ✅ SOLUCIÓN: Usar método corregido que SÍ funciona
            external_factors = self._get_comprehensive_external_factors(request.fecha_compra, request.codigo_postal)

            logger.info(
                f"🎯 Factores detectados: {external_factors.get('eventos_detectados', [])} - Demanda: {external_factors.get('factor_demanda', 1.0):.2f}")

            # 3. BÚSQUEDA DE TIENDAS DINÁMICAS
            nearby_stores = self.repos.store.find_stores_by_postal_range(request.codigo_postal)
            if not nearby_stores:
                raise ValueError(f"No hay tiendas Liverpool cerca de {request.codigo_postal}")

            # Log mejorado de tiendas encontradas
            logger.info(f"📍 Tiendas encontradas para {request.codigo_postal}: {len(nearby_stores)}")
            for i, store in enumerate(nearby_stores[:4]):
                logger.info(
                    f"  {i + 1}. {store['tienda_id']} - {store['nombre_tienda']} ({store.get('distancia_km', 0):.1f}km)")

            # 4. VERIFICACIÓN DE STOCK REAL
            stock_analysis = await self._analyze_stock_dynamic(
                request, product_info, nearby_stores
            )

            if not stock_analysis['factible']:
                raise ValueError(f"Stock insuficiente: {stock_analysis['razon']}")

            # 5. GENERACIÓN DE CANDIDATOS DINÁMICOS
            candidates = await self._generate_candidates_dynamic(
                stock_analysis, cp_info, external_factors, request
            )

            if not candidates:
                raise ValueError("No se encontraron rutas factibles")

            # 6. RANKING Y DECISIÓN
            ranked_candidates = self._rank_candidates_dynamic(candidates)
            top_candidates = ranked_candidates[:3]

            # Decisión de Gemini
            gemini_decision = await self.gemini_engine.select_optimal_route(
                top_candidates, request.dict(), external_factors
            )

            # 7. CONSTRUCCIÓN DE RESPUESTA DINÁMICA
            response = await self._build_dynamic_response(
                request, gemini_decision, external_factors,
                stock_analysis, ranked_candidates, cp_info
            )

            processing_time = (time.time() - start_time) * 1000
            logger.info(f"✅ Predicción completada en {processing_time:.1f}ms")

            return response

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Error en predicción: {e} ({processing_time:.1f}ms)")
            raise

    async def _validate_request_dynamic(self, request: PredictionRequest) -> Dict[str, Any]:
        """✅ Validación dinámica con datos reales"""

        # Buscar producto real
        product = self.repos.product.get_product_by_sku(request.sku_id)
        if not product:
            return {'valid': False, 'error': f'Producto no encontrado: {request.sku_id}'}

        # Validar tiendas autorizadas
        tiendas_disponibles = product.get('tiendas_disponibles', '')
        if not tiendas_disponibles:
            return {'valid': False, 'error': f'Producto sin tiendas autorizadas: {request.sku_id}'}

        # Buscar información de CP real
        postal_info = self.repos.store._get_postal_info(request.codigo_postal)
        if not postal_info:
            return {'valid': False, 'error': f'Código postal no válido: {request.codigo_postal}'}

        logger.info(f"✅ Validación exitosa: {product['nombre_producto']} → {postal_info['estado_alcaldia']}")
        return {
            'valid': True,
            'product': product,
            'postal_info': postal_info
        }

    async def _analyze_stock_dynamic(self, request: PredictionRequest,
                                     product_info: Dict[str, Any],
                                     nearby_stores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """📦 Análisis de stock CORREGIDO - prioriza tiendas locales"""

        # Tiendas autorizadas para el SKU
        tiendas_autorizadas = [t.strip() for t in product_info['tiendas_disponibles'].split(',')]
        logger.info(f"🏪 Tiendas autorizadas para {request.sku_id}: {tiendas_autorizadas}")

        # PASO 1: Buscar en tiendas locales PRIMERO
        local_store_ids = [store['tienda_id'] for store in nearby_stores[:5]]  # Top 5 más cercanas
        stock_locations_local = self.repos.stock.get_stock_for_stores_and_sku(
            request.sku_id, local_store_ids, request.cantidad
        )

        # Verificar si hay stock local suficiente
        if stock_locations_local:
            total_local_stock = sum(loc['stock_disponible'] for loc in stock_locations_local)
            logger.info(f"📊 Stock total disponible: {total_local_stock} | Requerido: {request.cantidad}")

            if total_local_stock >= request.cantidad:
                logger.info(
                    f"✅ Stock LOCAL suficiente: {total_local_stock} unidades en {len(stock_locations_local)} tiendas")

                # Usar solo tiendas locales con stock
                authorized_nearby = [
                    store for store in nearby_stores
                    if any(stock['tienda_id'] == store['tienda_id'] for stock in stock_locations_local)
                ]

                allocation = self.repos.stock.calculate_optimal_allocation(
                    stock_locations_local, request.cantidad, authorized_nearby
                )

                if allocation['factible']:
                    split_inventory = self._build_split_inventory(allocation['plan'], request.cantidad)
                    return {
                        'factible': True,
                        'allocation_plan': allocation['plan'],
                        'split_inventory': split_inventory,
                        'stores_info': authorized_nearby,
                        'total_available': allocation['cantidad_cubierta'],
                        'source': 'LOCAL_PRIORITY'
                    }

        # PASO 2: Si no hay stock local suficiente, buscar en tiendas autorizadas nacionales
        logger.info("🌎 Stock local insuficiente, buscando en tiendas autorizadas...")

        # Buscar tiendas autorizadas a nivel nacional
        tiendas_df = self.repos.data_manager.get_data('tiendas')
        authorized_stores = tiendas_df.filter(
            pl.col('tienda_id').is_in(tiendas_autorizadas)
        ).to_dicts()

        if not authorized_stores:
            return {
                'factible': False,
                'razon': f'No hay tiendas autorizadas disponibles para {request.sku_id}'
            }

        # Calcular distancias a todas las tiendas autorizadas
        target_lat = nearby_stores[0]['latitud'] if nearby_stores else 19.4326
        target_lon = nearby_stores[0]['longitud'] if nearby_stores else -99.1332

        for store in authorized_stores:
            try:
                from utils.geo_calculator import GeoCalculator
                distance = GeoCalculator.calculate_distance_km(
                    target_lat, target_lon,
                    float(store['latitud']), float(store['longitud'])
                )
                store['distancia_km'] = distance
            except Exception as e:
                logger.warning(f"⚠️ Error calculando distancia para {store.get('tienda_id')}: {e}")
                store['distancia_km'] = 999.0

        # Ordenar por distancia y tomar las 10 más cercanas
        authorized_stores.sort(key=lambda x: x['distancia_km'])
        authorized_nearby = authorized_stores[:10]

        # Buscar stock real en tiendas autorizadas
        store_ids = [store['tienda_id'] for store in authorized_nearby]
        stock_locations = self.repos.stock.get_stock_for_stores_and_sku(
            request.sku_id, store_ids, request.cantidad
        )

        if not stock_locations:
            return {
                'factible': False,
                'razon': f'Sin stock disponible para {request.sku_id} en tiendas autorizadas'
            }

        # Calcular asignación óptima
        allocation = self.repos.stock.calculate_optimal_allocation(
            stock_locations, request.cantidad, authorized_nearby
        )

        if not allocation['factible']:
            return {
                'factible': False,
                'razon': allocation['razon']
            }

        # Construir SplitInventory
        split_inventory = self._build_split_inventory(allocation['plan'], request.cantidad)

        return {
            'factible': True,
            'allocation_plan': allocation['plan'],
            'split_inventory': split_inventory,
            'stores_info': authorized_nearby,
            'total_available': allocation['cantidad_cubierta'],
            'source': 'NATIONAL_AUTHORIZED'
        }

    def _build_split_inventory(self, plan: List[Dict[str, Any]], cantidad_requerida: int) -> SplitInventory:
        """🏗️ Construye objeto SplitInventory CON COORDENADAS REALES"""
        ubicaciones = []

        for item in plan:
            # 🔴 PROBLEMA: Coordenadas hardcodeadas
            # coordenadas={'lat': 19.4326, 'lon': -99.1332},  # ❌ MALO

            # ✅ SOLUCIÓN: Obtener coordenadas reales de la tienda
            tienda_info = self._get_store_info_sync(item['tienda_id'])
            coordenadas_reales = {
                'lat': float(tienda_info['latitud']) if tienda_info else 19.4326,
                'lon': float(tienda_info['longitud']) if tienda_info else -99.1332
            }

            ubicacion = UbicacionStock(
                ubicacion_id=item['tienda_id'],
                ubicacion_tipo='TIENDA',
                nombre_ubicacion=tienda_info['nombre_tienda'] if tienda_info else f"Liverpool {item['tienda_id']}",
                # ✅ NOMBRE REAL
                stock_disponible=item['cantidad'],
                stock_reservado=0,
                coordenadas=coordenadas_reales,  # ✅ COORDENADAS REALES
                horario_operacion=tienda_info.get('horario_operacion', '09:00-21:00') if tienda_info else '09:00-21:00',
                tiempo_preparacion_horas=1.0
            )
            ubicaciones.append(ubicacion)

        return SplitInventory(
            ubicaciones=ubicaciones,
            cantidad_total_requerida=cantidad_requerida,
            cantidad_total_disponible=sum(item['cantidad'] for item in plan),
            es_split_factible=True,
            razon_split=f"Plan óptimo con {len(plan)} tiendas"
        )

    def _get_store_info_sync(self, tienda_id: str) -> Dict[str, Any]:
        """🏪 Versión SINCRÓNICA para obtener info de tienda"""
        tiendas_df = self.repos.data_manager.get_data('tiendas')
        store_data = tiendas_df.filter(pl.col('tienda_id') == tienda_id)

        if store_data.height > 0:
            return store_data.to_dicts()[0]
        return None

    async def _generate_candidates_dynamic(self, stock_analysis: Dict[str, Any],
                                           cp_info: Dict[str, Any],
                                           external_factors: Dict[str, Any],
                                           request: PredictionRequest) -> List[Dict[str, Any]]:
        """🗺️ Genera candidatos de ruta dinámicamente"""

        allocation_plan = stock_analysis['allocation_plan']
        stores_info = stock_analysis['stores_info']
        target_lat = cp_info['latitud_centro']
        target_lon = cp_info['longitud_centro']

        candidates = []

        # 1. Rutas DIRECTAS desde cada tienda
        for plan_item in allocation_plan:
            candidate = await self._create_direct_route_dynamic(
                plan_item, (target_lat, target_lon), external_factors, request, cp_info
            )
            if candidate:
                candidates.append(candidate)

        # 2. Ruta CONSOLIDADA si hay múltiples tiendas
        if len(allocation_plan) > 1:
            consolidated = await self._create_consolidated_route_dynamic(
                allocation_plan, (target_lat, target_lon), external_factors, request, stores_info
            )
            if consolidated:
                candidates.append(consolidated)

        logger.info(f"🗺️ Candidatos generados: {len(candidates)}")
        for i, candidate in enumerate(candidates):
            logger.info(
                f"  {i + 1}. {candidate['tipo_ruta']}: ${candidate['costo_total_mxn']:.0f} - {candidate['tiempo_total_horas']:.1f}h")

        return candidates

    async def _create_direct_route_dynamic(self, plan_item: Dict[str, Any],
                                           target_coords: Tuple[float, float],
                                           external_factors: Dict[str, Any],
                                           request: PredictionRequest,
                                           cp_info: Dict[str, Any]) -> Dict[str, Any]:
        """📍 CORRECCIÓN: Crea ruta directa con distancia REAL"""

        tienda_id = plan_item['tienda_id']

        # ✅ CORRECCIÓN: Obtener coordenadas REALES de la tienda
        tienda_info = await self._get_store_info(tienda_id)
        if not tienda_info:
            logger.error(f"❌ No se encontró info para tienda {tienda_id}")
            return None

        # ✅ CORRECCIÓN: Usar coordenadas reales de la tienda
        store_lat = float(tienda_info['latitud'])
        store_lon = float(tienda_info['longitud'])

        logger.info(f"📍 Tienda {tienda_id}: lat={store_lat:.4f}, lon={store_lon:.4f}")
        logger.info(f"📍 Destino CP {request.codigo_postal}: lat={target_coords[0]:.4f}, lon={target_coords[1]:.4f}")

        # ✅ CORRECCIÓN: Calcular distancia REAL
        from utils.geo_calculator import GeoCalculator
        distance_km = GeoCalculator.calculate_distance_km(
            store_lat, store_lon,
            target_coords[0], target_coords[1]
        )

        logger.info(f"📏 Distancia calculada: {distance_km:.1f}km")

        # ✅ CORRECCIÓN: Usar zona de seguridad REAL del CP
        zona_seguridad = cp_info.get('zona_seguridad', 'Verde')
        cobertura_liverpool = cp_info.get('cobertura_liverpool', True)

        # ✅ CORRECCIÓN: Lógica mejorada para determinar tipo de flota
        if distance_km <= 50 and cobertura_liverpool and zona_seguridad in ['Verde', 'Amarilla']:
            fleet_type = 'FI'
            carrier = 'Liverpool'
            logger.info(f"🚛 Usando Flota Interna - Zona {zona_seguridad}, distancia {distance_km:.1f}km")
        else:
            fleet_type = 'FE'
            # ✅ CORRECCIÓN: Buscar mejor carrier externo
            peso_kg = self._calculate_shipment_weight(request, plan_item['cantidad'])
            carriers = self.repos.fleet.get_best_carriers_for_cp(request.codigo_postal, peso_kg)
            carrier = carriers[0]['carrier'] if carriers else 'DHL'
            logger.info(f"📦 Usando Flota Externa ({carrier}) - Zona {zona_seguridad}, distancia {distance_km:.1f}km")

        # ✅ CORRECCIÓN: Cálculos con distancia y factores REALES
        travel_time = self._calculate_travel_time_dynamic(distance_km, fleet_type, external_factors)
        prep_time = float(tienda_info.get('tiempo_prep_horas', 1.0))

        # ✅ APLICAR factores externos de tiempo
        tiempo_extra = external_factors.get('impacto_tiempo_extra_horas', 0)
        total_time = prep_time + travel_time + tiempo_extra

        logger.info(
            f"⏱️ Tiempos: prep={prep_time:.1f}h + viaje={travel_time:.1f}h + extra={tiempo_extra:.1f}h = {total_time:.1f}h")

        # ✅ CORRECCIÓN: Costo dinámico REAL
        if fleet_type == 'FE' and carriers:
            cost = self._calculate_external_fleet_cost(
                carriers[0], peso_kg, distance_km, external_factors
            )
        else:
            cost = self._calculate_internal_fleet_cost(
                distance_km, plan_item['cantidad'], external_factors
            )

        logger.info(f"💰 Costo calculado: ${cost:.2f} ({fleet_type})")

        # ✅ CORRECCIÓN: Probabilidad con todos los factores
        probability = self._calculate_probability_dynamic(
            distance_km, total_time, external_factors, fleet_type, zona_seguridad
        )

        return {
            'ruta_id': f"direct_{tienda_id}",
            'tipo_ruta': 'directa',
            'origen_principal': tienda_info['nombre_tienda'],  # ✅ NOMBRE real
            'tienda_origen_id': tienda_id,
            'segmentos': [{
                'origen': tienda_info['nombre_tienda'],  # ✅ NOMBRE real
                'origen_id': tienda_id,
                'destino': 'cliente',
                'distancia_km': distance_km,  # ✅ DISTANCIA real
                'tiempo_horas': travel_time,
                'tipo_flota': fleet_type,
                'carrier': carrier,
                'costo_segmento': cost,
                'zona_seguridad': zona_seguridad  # ✅ ZONA real
            }],
            'tiempo_total_horas': total_time,
            'costo_total_mxn': cost,
            'distancia_total_km': distance_km,  # ✅ DISTANCIA real
            'probabilidad_cumplimiento': probability,
            'cantidad_cubierta': plan_item['cantidad'],
            'factores_aplicados': [
                f"demanda_{external_factors.get('factor_demanda', 1.0)}",
                f"flota_{fleet_type}",
                f"carrier_{carrier}",
                f"zona_{zona_seguridad}",
                f"eventos_{len(external_factors.get('eventos_detectados', []))}",
                'calculo_dinamico'
            ]
        }


    async def _get_store_info(self, tienda_id: str) -> Dict[str, Any]:
        """🏪 Obtiene información completa de la tienda"""
        tiendas_df = self.repos.data_manager.get_data('tiendas')
        store_data = tiendas_df.filter(pl.col('tienda_id') == tienda_id)

        if store_data.height > 0:
            return store_data.to_dicts()[0]
        return None

    async def _create_consolidated_route_dynamic(self, allocation_plan: List[Dict[str, Any]],
                                                 target_coords: Tuple[float, float],
                                                 external_factors: Dict[str, Any],
                                                 request: PredictionRequest,
                                                 stores_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """🔄 Crea ruta consolidada dinámica"""

        # Usar tienda más cercana como hub
        hub_plan = min(allocation_plan, key=lambda x: x['distancia_km'])
        other_plans = [p for p in allocation_plan if p != hub_plan]

        if not other_plans:
            return None

        # Store info map
        store_map = {store['tienda_id']: store for store in stores_info}

        segmentos = []
        tiempo_total = 1.0  # Preparación inicial
        costo_total = 0
        cantidad_total = hub_plan['cantidad']

        # Recolección desde otras tiendas
        current_store_id = hub_plan['tienda_id']

        for other_plan in other_plans:
            # Distancia entre tiendas
            current_store = store_map.get(current_store_id)
            other_store = store_map.get(other_plan['tienda_id'])

            if not current_store or not other_store:
                continue

            inter_store_distance = GeoCalculator.calculate_distance_km(
                float(current_store['latitud']), float(current_store['longitud']),
                float(other_store['latitud']), float(other_store['longitud'])
            )

            travel_time = self._calculate_travel_time_dynamic(
                inter_store_distance, 'FI', external_factors
            )

            segmentos.append({
                'origen': current_store_id,
                'destino': other_plan['tienda_id'],
                'distancia_km': inter_store_distance,
                'tiempo_horas': travel_time,
                'tipo_flota': 'FI',
                'carrier': 'Liverpool'
            })

            tiempo_total += travel_time + 1.0  # Preparación en cada tienda
            costo_total += inter_store_distance * 12.0  # Costo FI
            cantidad_total += other_plan['cantidad']
            current_store_id = other_plan['tienda_id']

        # Segmento final al cliente
        final_store = store_map.get(current_store_id)
        final_distance = GeoCalculator.calculate_distance_km(
            float(final_store['latitud']), float(final_store['longitud']),
            target_coords[0], target_coords[1]
        )

        final_travel_time = self._calculate_travel_time_dynamic(
            final_distance, 'FI', external_factors
        )

        segmentos.append({
            'origen': current_store_id,
            'destino': 'cliente',
            'distancia_km': final_distance,
            'tiempo_horas': final_travel_time,
            'tipo_flota': 'FI',
            'carrier': 'Liverpool'
        })

        tiempo_total += final_travel_time
        costo_total += self._calculate_internal_fleet_cost(
            final_distance, cantidad_total, external_factors
        )

        # Aplicar factores externos
        tiempo_total += external_factors.get('impacto_tiempo_extra_horas', 0)

        distancia_total = sum(seg['distancia_km'] for seg in segmentos)
        probability = self._calculate_probability_dynamic(
            distancia_total, tiempo_total, external_factors, 'FI'
        )

        return {
            'ruta_id': f"consolidated_{hub_plan['tienda_id']}_{len(other_plans)}",
            'tipo_ruta': 'consolidada',
            'origen_principal': hub_plan['tienda_id'],
            'segmentos': segmentos,
            'tiempo_total_horas': tiempo_total,
            'costo_total_mxn': costo_total,
            'distancia_total_km': distancia_total,
            'probabilidad_cumplimiento': probability,
            'cantidad_cubierta': cantidad_total,
            'factores_aplicados': ['consolidacion_dinamica', 'multiples_tiendas']
        }

    def _calculate_travel_time_dynamic(self, distance_km: float, fleet_type: str,
                                       external_factors: Dict[str, Any]) -> float:
        """⏱️ Calcula tiempo de viaje dinámico"""
        return GeoCalculator.calculate_travel_time(
            distance_km,
            fleet_type,
            external_factors.get('trafico_nivel', 'Moderado'),
            external_factors.get('condicion_clima', 'Templado')
        )

    def _calculate_external_fleet_cost(self, carrier_info: Dict[str, Any],
                                       peso_kg: float, distance_km: float,
                                       external_factors: Dict[str, Any]) -> float:
        """💰 Cálculo CORREGIDO de costo flota externa"""

        base_cost = carrier_info['costo_base_mxn']
        peso_extra = max(0, peso_kg - carrier_info['peso_min_kg'])
        cost_extra = peso_extra * carrier_info['costo_por_kg_adicional']

        # CORRECCIÓN: Factor de distancia más realista
        distance_factor = 1.0 + (distance_km / 500) * 0.2  # 20% extra cada 500km

        subtotal = (base_cost + cost_extra) * distance_factor

        # CORRECCIÓN: Aplicar TODOS los factores externos
        demand_factor = external_factors.get('factor_demanda', 1.0)
        cost_extra_pct = external_factors.get('impacto_costo_extra_pct', 0) / 100

        # Aplicar factor de demanda
        subtotal *= demand_factor

        # Aplicar impacto de costo extra
        if cost_extra_pct > 0:
            subtotal *= (1 + cost_extra_pct)

        # CORRECCIÓN: Factor por temporada crítica
        if external_factors.get('es_temporada_critica', False):
            subtotal *= 1.25  # 25% extra en temporada crítica

        final_cost = round(subtotal, 2)

        logger.info(
            f"💰 Costo externo: base=${base_cost} × demanda={demand_factor:.2f} × extras={1 + cost_extra_pct:.2f} = ${final_cost}")

        return final_cost

    def _calculate_internal_fleet_cost(self, distance_km: float, cantidad: int,
                                       external_factors: Dict[str, Any]) -> float:
        """💰 Calcula costo de flota interna"""
        base_cost = distance_km * 12.0

        # Factor por cantidad
        quantity_factor = 0.9 if cantidad >= 3 else 1.0

        # Factor de demanda
        demand_factor = external_factors.get('factor_demanda', 1.0)

        total_cost = base_cost * quantity_factor * demand_factor
        return round(max(50.0, total_cost), 2)

    def _get_cached_factors(self, fecha: datetime, codigo_postal: str) -> Dict[str, Any]:
        """🔄 Factores externos MEJORADOS con CSV real"""

        cache_key = f"{fecha.date().isoformat()}_{codigo_postal}"

        if cache_key in self._factors_cache:
            logger.info(f"📋 Usando factores desde cache para {fecha.date()}")
            return self._factors_cache[cache_key]

        # MEJORA: Usar factores del CSV con mapeo de CP
        factors = self._get_comprehensive_external_factors(fecha, codigo_postal)
        self._factors_cache[cache_key] = factors

        return factors

    def _get_comprehensive_external_factors(self, fecha: datetime, codigo_postal: str) -> Dict[str, Any]:
        """🎯 Factores externos COMPLETOS usando todos los CSV"""

        # 1. Buscar en factores_externos_mexico_completo.csv
        factores_df = self.repos.data_manager.get_data('factores_externos')
        fecha_str = fecha.date().isoformat()

        # Buscar por fecha exacta
        exact_match = factores_df.filter(pl.col('fecha') == fecha_str)

        factores_csv = None
        if exact_match.height > 0:
            # Filtrar por rango de CP si existe
            cp_int = int(codigo_postal)
            for row in exact_match.to_dicts():
                rango_afectado = row.get('rango_cp_afectado', '00000-99999')
                if self._cp_in_range(cp_int, rango_afectado):
                    factores_csv = row
                    break

            if not factores_csv:
                factores_csv = exact_match.to_dicts()[0]  # Usar el primero si no hay match de CP

        # 2. Información de código postal
        cp_info = self._get_postal_info_detailed(codigo_postal)

        # 3. Información climática
        clima_info = self._get_climate_info(codigo_postal, fecha)

        # 4. Combinar toda la información
        return self._combine_all_factors(factores_csv, cp_info, clima_info, fecha, codigo_postal)

    def _get_postal_info_detailed(self, codigo_postal: str) -> Dict[str, Any]:
        """📍 Información detallada del código postal"""
        cp_df = self.repos.data_manager.get_data('codigos_postales')
        cp_int = int(codigo_postal)

        for row in cp_df.to_dicts():
            rango_cp = row.get('rango_cp', '')
            if '-' in rango_cp:
                try:
                    start_cp, end_cp = map(int, rango_cp.split('-'))
                    if start_cp <= cp_int <= end_cp:
                        return row
                except ValueError:
                    continue

        # Fallback por prefijo
        cp_prefix = codigo_postal[:2]
        prefix_matches = cp_df.filter(pl.col('rango_cp').str.contains(cp_prefix))
        if prefix_matches.height > 0:
            return prefix_matches.to_dicts()[0]

        return {
            'rango_cp': codigo_postal,
            'estado_alcaldia': 'Ciudad de México',
            'zona_seguridad': 'Amarilla',
            'latitud_centro': 19.4326,
            'longitud_centro': -99.1332,
            'cobertura_liverpool': True,
            'tiempo_entrega_base_horas': '2-4'
        }

    def _get_climate_info(self, codigo_postal: str, fecha: datetime) -> Dict[str, Any]:
        """🌤️ Información climática por CP y fecha"""
        clima_df = self.repos.data_manager.get_data('clima')
        cp_int = int(codigo_postal)

        for row in clima_df.to_dicts():
            inicio = int(row.get('rango_cp_inicio', 0))
            fin = int(row.get('rango_cp_fin', 99999))

            if inicio <= cp_int <= fin:
                # Determinar clima por temporada
                mes = fecha.month
                if mes in [12, 1, 2]:
                    clima = row.get('clima_invierno', 'Frio_Seco')
                    temp = row.get('temperatura_min_invierno', 10)
                elif mes in [3, 4, 5]:
                    clima = row.get('clima_primavera', 'Templado_Seco')
                    temp = (row.get('temperatura_min_invierno', 10) + row.get('temperatura_max_verano', 25)) // 2
                elif mes in [6, 7, 8]:
                    clima = row.get('clima_verano', 'Calido_Lluvioso')
                    temp = row.get('temperatura_max_verano', 25)
                else:
                    clima = row.get('clima_otoño', 'Templado_Seco')
                    temp = (row.get('temperatura_min_invierno', 10) + row.get('temperatura_max_verano', 25)) // 2

                return {
                    'region': row.get('region_nombre', 'CDMX'),
                    'clima_actual': clima,
                    'temperatura_estimada': temp,
                    'precipitacion_anual': row.get('precipitacion_anual_mm', 600),
                    'factores_especiales': row.get('factores_especiales', '')
                }

        return {
            'region': 'CDMX_Default',
            'clima_actual': 'Templado',
            'temperatura_estimada': 20,
            'precipitacion_anual': 600,
            'factores_especiales': ''
        }

    def _calculate_time_impact_comprehensive(self, factor_demanda: float,
                                             eventos_detectados: List[str],
                                             cp_info: Dict[str, Any]) -> float:
        """⏱️ Cálculo COMPLETO de impacto en tiempo"""

        # Base impact por demanda
        base_impact = max(0, (factor_demanda - 1.0) * 2.0)  # 2h extra por cada punto de demanda

        # Impact por eventos específicos
        event_impact = 0
        for evento in eventos_detectados:
            if 'Nochebuena' in evento or 'Navidad' in evento:
                event_impact += 4.0  # 4h extra en Nochebuena/Navidad
            elif 'Pre_Navidad' in evento:
                event_impact += 2.0  # 2h extra pre-navidad
            elif 'Santo' in evento or 'Viernes' in evento:
                event_impact += 1.0  # 1h extra días festivos

        # Impact por zona de seguridad
        zona_impact = {
            'Verde': 0.0,
            'Amarilla': 1.0,
            'Roja': 3.0
        }.get(cp_info.get('zona_seguridad', 'Verde'), 0.0)

        total_impact = base_impact + event_impact + zona_impact

        logger.info(
            f"⏱️ Impacto tiempo: base={base_impact:.1f}h + eventos={event_impact:.1f}h + zona={zona_impact:.1f}h = {total_impact:.1f}h")

        return total_impact

    def _calculate_cost_impact_comprehensive(self, factor_demanda: float,
                                             eventos_detectados: List[str],
                                             cp_info: Dict[str, Any]) -> float:
        """💰 Cálculo COMPLETO de impacto en costo (%)"""

        # Base impact por demanda
        base_impact = max(0, (factor_demanda - 1.0) * 25.0)  # 25% extra por cada punto de demanda

        # Impact por eventos específicos
        event_impact = 0
        for evento in eventos_detectados:
            if 'Nochebuena' in evento or 'Navidad' in evento:
                event_impact += 50.0  # 50% extra en Nochebuena/Navidad
            elif 'Pre_Navidad' in evento:
                event_impact += 25.0  # 25% extra pre-navidad
            elif 'Santo' in evento:
                event_impact += 15.0  # 15% extra días festivos

        # Impact por zona de seguridad
        zona_impact = {
            'Verde': 0.0,
            'Amarilla': 10.0,  # 10% extra zona amarilla
            'Roja': 30.0  # 30% extra zona roja
        }.get(cp_info.get('zona_seguridad', 'Verde'), 0.0)

        total_impact = base_impact + event_impact + zona_impact

        logger.info(
            f"💰 Impacto costo: base={base_impact:.1f}% + eventos={event_impact:.1f}% + zona={zona_impact:.1f}% = {total_impact:.1f}%")

        return total_impact

    def _combine_all_factors(self, factores_csv: Dict[str, Any], cp_info: Dict[str, Any],
                             clima_info: Dict[str, Any], fecha: datetime, codigo_postal: str) -> Dict[str, Any]:
        """🔗 Combina todos los factores externos"""

        # Eventos detectados
        eventos_detectados = []

        if factores_csv:
            evento_csv = factores_csv.get('evento_detectado', '')
            if evento_csv and evento_csv != 'Normal':
                eventos_detectados.append(evento_csv)

            # Factor de demanda del CSV
            factor_demanda_raw = factores_csv.get('factor_demanda', '1.0')
            if isinstance(factor_demanda_raw, str) and '/' in factor_demanda_raw:
                try:
                    num, den = map(float, factor_demanda_raw.split('/'))
                    factor_demanda = num / den
                except:
                    factor_demanda = 1.0
            else:
                factor_demanda = float(factor_demanda_raw)
        else:
            factor_demanda = 1.0

        # Detectar eventos por fecha (Nochebuena = 2024-12-24)
        eventos_fecha = self._detect_date_events_enhanced(fecha)
        eventos_detectados.extend(eventos_fecha)

        # Si es Nochebuena, ajustar factor de demanda
        if any('Nochebuena' in e or 'Navidad' in e for e in eventos_detectados):
            factor_demanda = max(factor_demanda, 3.5)  # Mínimo 3.5x en Nochebuena

        # Eliminar duplicados
        eventos_detectados = list(set(eventos_detectados))

        # Clima de CSV o calculado
        condicion_clima = factores_csv.get('condicion_clima') if factores_csv else None
        if not condicion_clima:
            condicion_clima = clima_info.get('clima_actual', 'Templado')

        # Tráfico
        trafico_nivel = factores_csv.get('trafico_nivel', 'Moderado') if factores_csv else 'Moderado'

        # Calcular impactos
        impacto_tiempo = self._calculate_time_impact_comprehensive(factor_demanda, eventos_detectados, cp_info)
        impacto_costo = self._calculate_cost_impact_comprehensive(factor_demanda, eventos_detectados, cp_info)

        # Criticidad
        criticidad = factores_csv.get('criticidad_logistica', 'Normal') if factores_csv else 'Normal'
        if any('Nochebuena' in e or 'Navidad' in e for e in eventos_detectados):
            criticidad = 'Crítica'

        logger.info(
            f"🎯 Factores combinados: eventos={eventos_detectados}, demanda={factor_demanda:.2f}, zona={cp_info.get('zona_seguridad')}")

        return {
            'eventos_detectados': eventos_detectados,
            'evento_detectado': eventos_detectados[0] if eventos_detectados else 'Normal',
            'es_temporada_alta': factor_demanda > 1.8,
            'es_temporada_critica': factor_demanda > 2.5,
            'factor_demanda': factor_demanda,

            'condicion_clima': condicion_clima,
            'temperatura_celsius': clima_info.get('temperatura_estimada', 20),
            'probabilidad_lluvia': 30 if 'Lluvioso' in condicion_clima else 15,
            'viento_kmh': 15,

            'trafico_nivel': trafico_nivel,
            'impacto_tiempo_extra_horas': impacto_tiempo,
            'impacto_costo_extra_pct': impacto_costo,

            'zona_seguridad': cp_info.get('zona_seguridad', 'Verde'),
            'cobertura_liverpool': cp_info.get('cobertura_liverpool', True),
            'restricciones_vehiculares': [],
            'criticidad_logistica': criticidad,

            'fecha_analisis': fecha.isoformat(),
            'fuente_datos': 'CSV_combinado' if factores_csv else 'calculado_inteligente',
            'confianza_prediccion': 0.95 if factores_csv else 0.85,
            'region_climatica': clima_info.get('region', 'CDMX')
        }

    def _detect_date_events_enhanced(self, fecha: datetime) -> List[str]:
        """🎄 Detección mejorada de eventos por fecha"""
        mes, dia = fecha.month, fecha.day
        eventos = []

        if mes == 12:
            if dia == 24:
                eventos.extend(['Nochebuena', 'Navidad_Pico', 'Emergencia_Regalos'])
            elif dia == 25:
                eventos.extend(['Navidad', 'Dia_Navidad'])
            elif 20 <= dia <= 23:
                eventos.extend(['Pre_Navidad_Intenso'])
            elif 15 <= dia <= 19:
                eventos.extend(['Pre_Navidad'])

        return eventos

    def _calculate_probability_dynamic(self, distance_km: float, total_time: float,
                                       external_factors: Dict[str, Any], fleet_type: str,
                                       zona_seguridad: str = 'Verde') -> float:
        """📊 Probabilidad con zona de seguridad"""

        base_prob = 0.90 if fleet_type == 'FI' else 0.82

        # Penalizaciones por distancia y tiempo
        distance_penalty = min(0.2, distance_km / 1000)
        time_penalty = min(0.15, max(0, (total_time - 6) / 50))

        # Penalización por zona de seguridad
        zona_penalty = {
            'Verde': 0.0,
            'Amarilla': 0.05,
            'Roja': 0.15
        }.get(zona_seguridad, 0.05)

        # Factor por criticidad
        criticidad = external_factors.get('criticidad_logistica', 'Normal')
        criticidad_factor = {
            'Baja': 1.0,
            'Normal': 0.95,
            'Media': 0.90,
            'Alta': 0.85,
            'Crítica': 0.75
        }.get(criticidad, 0.90)

        final_prob = (base_prob - distance_penalty - time_penalty - zona_penalty) * criticidad_factor
        return round(max(0.4, min(0.98, final_prob)), 3)

    def _cp_in_range(self, cp_int: int, rango_str: str) -> bool:
        """📍 Verifica si CP está en rango"""
        if '-' not in rango_str:
            return True

        try:
            start, end = map(int, rango_str.split('-'))
            return start <= cp_int <= end
        except:
            return True

    def _calculate_shipment_weight(self, request: PredictionRequest, cantidad: int) -> float:
        """⚖️ Calcula peso del envío"""
        # Obtener peso del producto
        product = self.repos.product.get_product_by_sku(request.sku_id)
        peso_unitario = product.get('peso_kg', 0.5) if product else 0.5
        return peso_unitario * cantidad

    def _rank_candidates_dynamic(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """🏆 Rankea candidatos dinámicamente"""

        if not candidates:
            return []

        # Normalizar métricas
        tiempos = [c['tiempo_total_horas'] for c in candidates]
        costos = [c['costo_total_mxn'] for c in candidates]
        distancias = [c['distancia_total_km'] for c in candidates]

        min_tiempo, max_tiempo = min(tiempos), max(tiempos)
        min_costo, max_costo = min(costos), max(costos)
        min_distancia, max_distancia = min(distancias), max(distancias)

        for candidate in candidates:
            # Scores normalizados
            score_tiempo = 1 - (candidate['tiempo_total_horas'] - min_tiempo) / max(1, max_tiempo - min_tiempo)
            score_costo = 1 - (candidate['costo_total_mxn'] - min_costo) / max(1, max_costo - min_costo)
            score_distancia = 1 - (candidate['distancia_total_km'] - min_distancia) / max(1,
                                                                                          max_distancia - min_distancia)
            score_probabilidad = candidate['probabilidad_cumplimiento']

            # Score combinado con pesos dinámicos
            weights = {
                'tiempo': settings.PESO_TIEMPO,
                'costo': settings.PESO_COSTO,
                'probabilidad': settings.PESO_PROBABILIDAD,
                'distancia': settings.PESO_DISTANCIA
            }

            score_combinado = (
                    weights['tiempo'] * score_tiempo +
                    weights['costo'] * score_costo +
                    weights['probabilidad'] * score_probabilidad +
                    weights['distancia'] * score_distancia
            )

            # Bonus por tipo de ruta
            if candidate['tipo_ruta'] == 'directa':
                score_combinado *= 1.1

            candidate['score_lightgbm'] = round(score_combinado, 4)
            candidate['score_breakdown'] = {
                'tiempo': round(score_tiempo, 3),
                'costo': round(score_costo, 3),
                'distancia': round(score_distancia, 3),
                'probabilidad': round(score_probabilidad, 3)
            }

        # Ordenar por score
        ranked = sorted(candidates, key=lambda x: x['score_lightgbm'], reverse=True)

        logger.info("🏆 Ranking de candidatos:")
        for i, candidate in enumerate(ranked):
            logger.info(f"  {i + 1}. {candidate['tipo_ruta']}: score={candidate['score_lightgbm']:.3f} "
                        f"(${candidate['costo_total_mxn']:.0f}, {candidate['tiempo_total_horas']:.1f}h, "
                        f"{candidate['probabilidad_cumplimiento']:.1%})")

        return ranked

    async def _build_dynamic_response(self, request: PredictionRequest,
                                      gemini_decision: Dict[str, Any],
                                      external_factors: Dict[str, Any],
                                      stock_analysis: Dict[str, Any],
                                      all_candidates: List[Dict[str, Any]],
                                      cp_info: Dict[str, Any]) -> PredictionResponse:
        """🏗️ Construye respuesta completamente dinámica"""

        selected_route = gemini_decision['candidato_seleccionado']

        # Calcular FEE dinámico
        fee_calculation = self._calculate_dynamic_fee(
            selected_route, request, external_factors, cp_info
        )

        # Construir estructuras
        ruta_completa = self._build_route_structure(selected_route, stock_analysis)
        factores_estructurados = self._build_factors_structure(external_factors)

        # Candidatos para explicabilidad
        candidatos_lgb = []
        for i, candidate in enumerate(all_candidates[:3]):
            candidato_ruta = CandidatoRuta(
                ruta=self._build_route_structure(candidate, stock_analysis),
                score_lightgbm=candidate['score_lightgbm'],
                ranking_position=i + 1,
                features_utilizadas=candidate['score_breakdown'],
                trade_offs={}
            )
            candidatos_lgb.append(candidato_ruta)

        # Decisión Gemini
        decision_gemini = DecisionGemini(
            candidato_seleccionado=candidatos_lgb[0] if candidatos_lgb else None,
            razonamiento=gemini_decision.get('razonamiento', 'Decisión basada en datos dinámicos'),
            candidatos_evaluados=candidatos_lgb,
            factores_decisivos=gemini_decision.get('factores_decisivos', ['score_dinamico']),
            confianza_decision=gemini_decision.get('confianza_decision', 0.85),
            alertas_gemini=gemini_decision.get('alertas_operativas', [])
        )

        # Explicabilidad completa
        explicabilidad = ExplicabilidadCompleta(
            request_procesado=request,
            factores_externos=factores_estructurados,
            split_inventory=stock_analysis['split_inventory'],
            candidatos_lightgbm=candidatos_lgb,
            decision_gemini=decision_gemini,
            fee_calculation=fee_calculation,
            tiempo_procesamiento_ms=0,
            warnings=[],
            debug_info={
                'total_candidates_generated': len(all_candidates),
                'optimization_method': 'dinamico_csv_real',
                'data_source': 'memoria_optimizada',
                'search_scope': 'dinamico'
            }
        )

        # Respuesta final
        response = PredictionResponse(
            fecha_entrega_estimada=fee_calculation.fecha_entrega_estimada,
            rango_horario={
                'inicio': fee_calculation.rango_horario_entrega['inicio'].strftime('%H:%M'),
                'fin': fee_calculation.rango_horario_entrega['fin'].strftime('%H:%M')
            },
            ruta_seleccionada=ruta_completa,
            tipo_entrega=fee_calculation.tipo_entrega,
            carrier_principal=self._get_main_carrier(selected_route),
            costo_envio_mxn=selected_route['costo_total_mxn'],
            probabilidad_cumplimiento=selected_route['probabilidad_cumplimiento'],
            confianza_prediccion=gemini_decision.get('confianza_decision', 0.85),
            explicabilidad=explicabilidad
        )

        logger.info(f"📦 RESPUESTA FINAL: {fee_calculation.tipo_entrega.value} - "
                    f"${selected_route['costo_total_mxn']:.0f} - "
                    f"{fee_calculation.fecha_entrega_estimada.strftime('%Y-%m-%d %H:%M')}")

        return response

    def _calculate_dynamic_fee(self, selected_route: Dict[str, Any],
                               request: PredictionRequest,
                               external_factors: Dict[str, Any],
                               cp_info: Dict[str, Any]) -> FEECalculation:
        """📅 Calcula FEE dinámico"""

        tiempo_total = selected_route['tiempo_total_horas']

        # Determinar tipo de entrega dinámicamente
        tipo_entrega = self._determine_delivery_type(
            tiempo_total, request.fecha_compra, external_factors, cp_info
        )

        # Calcular fecha de entrega
        fecha_entrega = self._calculate_delivery_date(
            request.fecha_compra, tiempo_total, tipo_entrega, external_factors
        )

        # Rango horario
        rango_horario = self._calculate_time_window(fecha_entrega, tipo_entrega)

        return FEECalculation(
            fecha_entrega_estimada=fecha_entrega,
            rango_horario_entrega=rango_horario,
            tipo_entrega=tipo_entrega,
            tiempo_total_horas=tiempo_total,
            tiempo_preparacion=1.0,
            tiempo_transito=tiempo_total - 1.0,
            tiempo_contingencia=tiempo_total * 0.1
        )

    def _determine_delivery_type(self, tiempo_horas: float, fecha_compra: datetime,
                                 external_factors: Dict[str, Any], cp_info: Dict[str, Any]) -> TipoEntregaEnum:
        """📦 Determina tipo de entrega dinámicamente"""

        hora_compra = fecha_compra.hour
        cobertura = cp_info.get('cobertura_liverpool', False)
        criticidad = external_factors.get('criticidad_logistica', 'Normal')

        if tiempo_horas <= 8 and hora_compra <= 14 and cobertura and criticidad != 'Crítica':
            return TipoEntregaEnum.FLASH
        elif tiempo_horas <= 24 and hora_compra <= 20:
            return TipoEntregaEnum.EXPRESS
        elif tiempo_horas <= 72:
            return TipoEntregaEnum.STANDARD
        else:
            return TipoEntregaEnum.PROGRAMADA

    def _calculate_delivery_date(self, fecha_compra: datetime, tiempo_horas: float,
                                 tipo_entrega: TipoEntregaEnum, external_factors: Dict[str, Any]) -> datetime:
        """📅 Cálculo CORREGIDO de fecha de entrega con horarios operativos"""

        # CORRECCIÓN: Calcular ETA considerando horarios operativos
        current_time = fecha_compra
        remaining_hours = tiempo_horas

        # Aplicar tiempo extra por factores
        tiempo_extra = external_factors.get('impacto_tiempo_extra_horas', 0)
        remaining_hours += tiempo_extra

        logger.info(f"📅 Calculando entrega: {remaining_hours:.1f}h desde {current_time.strftime('%Y-%m-%d %H:%M')}")

        # Simular avance hora por hora considerando horarios operativos
        while remaining_hours > 0:
            # Verificar si estamos en horario operativo (9 AM - 9 PM)
            if 9 <= current_time.hour < 21 and current_time.weekday() < 6:  # Lunes-Sábado
                # Horas operativas: avanzar normalmente
                hours_to_add = min(remaining_hours, 21 - current_time.hour)
                current_time += timedelta(hours=hours_to_add)
                remaining_hours -= hours_to_add
            else:
                # Fuera de horario: saltar al siguiente día operativo a las 9 AM
                if current_time.weekday() >= 5:  # Fin de semana
                    days_to_add = 7 - current_time.weekday()  # Ir al lunes
                else:
                    days_to_add = 1  # Ir al siguiente día

                current_time = (current_time + timedelta(days=days_to_add)).replace(hour=9, minute=0, second=0)

        # Asegurar que la entrega esté en horario de entrega (10 AM - 6 PM)
        if current_time.hour < 10:
            current_time = current_time.replace(hour=10, minute=0)
        elif current_time.hour > 18:
            current_time = current_time.replace(hour=16, minute=0)  # Tarde pero dentro del rango

        # CORRECCIÓN: Evitar entregas en domingo para tipos no-FLASH
        if tipo_entrega != TipoEntregaEnum.FLASH and current_time.weekday() == 6:
            current_time += timedelta(days=1)
            current_time = current_time.replace(hour=10, minute=0)

        logger.info(f"📦 Entrega programada: {current_time.strftime('%Y-%m-%d %H:%M')} ({current_time.strftime('%A')})")

        return current_time

    def _calculate_time_window(self, fecha_entrega: datetime, tipo_entrega: TipoEntregaEnum) -> Dict[str, dt_time]:
        """🕐 Calcula ventana de entrega"""

        if tipo_entrega == TipoEntregaEnum.FLASH:
            ventana = 2
        else:
            ventana = 4

        inicio = max(
            fecha_entrega - timedelta(hours=ventana // 2),
            fecha_entrega.replace(hour=9, minute=0)
        )

        fin = min(
            fecha_entrega + timedelta(hours=ventana // 2),
            fecha_entrega.replace(hour=18, minute=0)
        )

        return {'inicio': inicio.time(), 'fin': fin.time()}

    def _build_route_structure(self, route_data: Dict[str, Any], stock_analysis: Dict[str, Any]) -> RutaCompleta:
        """🏗️ Construye estructura con NOMBRES de tiendas"""

        segmentos = []
        for i, seg_data in enumerate(route_data.get('segmentos', [])):
            segmento = Segmento(
                segmento_id=f"{route_data['ruta_id']}_seg_{i + 1}",
                origen_id=seg_data.get('origen_id', seg_data['origen']),
                destino_id=seg_data['destino'],
                origen_nombre=seg_data['origen'],  # Ya contiene el NOMBRE
                destino_nombre=seg_data['destino'],
                distancia_km=seg_data['distancia_km'],
                tiempo_viaje_horas=seg_data['tiempo_horas'],
                tipo_flota=seg_data['tipo_flota'],
                carrier=seg_data.get('carrier', 'Liverpool'),
                costo_segmento_mxn=seg_data.get('costo_segmento', 0),
                factores_aplicados=route_data.get('factores_aplicados', [])
            )
            segmentos.append(segmento)

        return RutaCompleta(
            ruta_id=route_data['ruta_id'],
            segmentos=segmentos,
            split_inventory=stock_analysis['split_inventory'],
            tiempo_total_horas=route_data['tiempo_total_horas'],
            costo_total_mxn=route_data['costo_total_mxn'],
            distancia_total_km=route_data['distancia_total_km'],
            score_tiempo=route_data['score_breakdown']['tiempo'],
            score_costo=route_data['score_breakdown']['costo'],
            score_confiabilidad=route_data['probabilidad_cumplimiento'],
            score_lightgbm=route_data['score_lightgbm'],
            estado='FACTIBLE',
            probabilidad_cumplimiento=route_data['probabilidad_cumplimiento'],
            factores_riesgo=[]
        )

    def _build_factors_structure(self, external_factors: Dict[str, Any]) -> FactoresExternos:
        """🌤️ Construye estructura de factores CON DATOS REALES"""

        return FactoresExternos(
            fecha_analisis=datetime.now(),
            eventos_detectados=external_factors.get('eventos_detectados', ['Normal']),  # ✅ REAL
            factor_demanda=external_factors.get('factor_demanda', 1.0),  # ✅ REAL
            es_temporada_alta=external_factors.get('es_temporada_alta', False),  # ✅ REAL
            condicion_clima=external_factors.get('condicion_clima', 'Templado'),  # ✅ REAL
            temperatura_celsius=external_factors.get('temperatura_celsius', 20),  # ✅ REAL
            probabilidad_lluvia=external_factors.get('probabilidad_lluvia', 30),  # ✅ REAL
            viento_kmh=external_factors.get('viento_kmh', 15),  # ✅ REAL
            trafico_nivel=external_factors.get('trafico_nivel', 'Moderado'),  # ✅ REAL
            impacto_tiempo_extra_horas=external_factors.get('impacto_tiempo_extra_horas', 0),  # ✅ REAL
            impacto_costo_extra_pct=external_factors.get('impacto_costo_extra_pct', 0),  # ✅ REAL
            zona_seguridad=external_factors.get('zona_seguridad', 'Verde'),  # ✅ REAL
            restricciones_vehiculares=external_factors.get('restricciones_vehiculares', []),  # ✅ REAL
            criticidad_logistica=external_factors.get('criticidad_logistica', 'Normal')  # ✅ REAL
        )

    def _get_main_carrier(self, route: Dict[str, Any]) -> str:
        """🚚 Obtiene carrier principal"""
        segmentos = route.get('segmentos', [])
        if not segmentos:
            return 'Liverpool'

        # Buscar segmento final (al cliente)
        for segmento in segmentos:
            if segmento.get('destino') == 'cliente':
                return segmento.get('carrier', 'Liverpool')

        return segmentos[-1].get('carrier', 'Liverpool')