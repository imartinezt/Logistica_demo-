import time
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Any

from config.settings import settings
from models.schemas import (
    PredictionRequest, PredictionResponse, TipoEntregaEnum,
    ExplicabilidadCompleta, FactoresExternos, UbicacionStock,
    SplitInventory, RutaCompleta, Segmento, CandidatoRuta,
    DecisionGemini, FEECalculation
)
from services.ai.gemini_service import GeminiLogisticsDecisionEngine
from services.data.repositories import (
    ProductRepository, StoreRepository, CEDISRepository,
    StockRepository, PostalCodeRepository, ClimateRepository,
    ExternalFactorsRepository, ExternalFleetRepository
)
from services.ml.lightgbm_optimizer import RouteOptimizer
from utils.geo_calculator import GeoCalculator
from utils.logger import logger
from utils.temporal_detector import TemporalFactorDetector


class FEEPredictionService:
    """🎯 Servicio CORREGIDO de predicción FEE con lógica realista"""

    def __init__(self, data_dir):
        # Inicializar repositorios
        self.repositories = {
            'product': ProductRepository(data_dir),
            'store': StoreRepository(data_dir),
            'cedis': CEDISRepository(data_dir),
            'stock': StockRepository(data_dir),
            'postal_code': PostalCodeRepository(data_dir),
            'climate': ClimateRepository(data_dir),
            'external_factors': ExternalFactorsRepository(data_dir),
            'external_fleet': ExternalFleetRepository(data_dir)
        }

        self.data_dir = data_dir
        self.route_optimizer = RouteOptimizer()
        self.gemini_engine = GeminiLogisticsDecisionEngine()
        self.route_optimizer.load_model()

    async def predict_fee(self, request: PredictionRequest) -> PredictionResponse:
        """🚀 Predicción CORREGIDA con lógica realista"""

        start_time = time.time()

        try:
            logger.info(f"🎯 Iniciando predicción FEE para {request.sku_id} -> {request.codigo_postal}")

            # 1️⃣ Validación básica
            validation_result = await self._validate_request(request)
            if not validation_result['is_valid']:
                raise ValueError(f"Validación fallida: {'; '.join(validation_result['errors'])}")

            product_info = validation_result['product_info']
            postal_info = validation_result['postal_info']

            # 2️⃣ Detectar factores externos REALES del CSV
            logger.info("🌤️ Detectando factores externos REALES...")
            external_factors = TemporalFactorDetector.detect_comprehensive_factors(
                request.fecha_compra, request.codigo_postal, self.data_dir
            )

            # 3️⃣ Análisis OPTIMIZADO de inventario (por proximidad primero)
            logger.info("📦 Analizando inventario OPTIMIZADO...")
            try:
                inventory_analysis = await self._analyze_inventory_optimized(
                    request, product_info, postal_info
                )
                logger.info(f"📦 Resultado análisis inventario: {inventory_analysis.get('inventory_available', 'N/A')}")

            except Exception as e:
                logger.error(f"❌ Error en análisis de inventario: {e}")
                raise ValueError(f"Error en análisis de inventario: {str(e)}")

            if not inventory_analysis.get('inventory_available', False):
                raise ValueError(f"Inventario no disponible: {inventory_analysis.get('reason', 'Sin stock')}")

            # 4️⃣ Generación de candidatos REALISTAS
            logger.info("🗺️ Generando candidatos REALISTAS...")
            try:
                route_candidates = await self._generate_realistic_candidates(
                    inventory_analysis, postal_info, external_factors, request
                )
                logger.info(f"🗺️ Generados {len(route_candidates)} candidatos")

            except Exception as e:
                logger.error(f"❌ Error en generación de candidatos: {e}")
                raise ValueError(f"Error en generación de candidatos: {str(e)}")

            if not route_candidates:
                raise ValueError("No se encontraron rutas factibles")

            # 5️⃣ Ranking por TIEMPO+COSTO (no por límites absolutos)
            logger.info("🎯 Ranking por eficiencia TIEMPO+COSTO...")
            ranked_candidates = self.route_optimizer.rank_candidates_with_lightgbm(
                route_candidates
            )

            # 6️⃣ Decisión final con Gemini
            logger.info("🧠 Decisión final con Gemini...")
            top_candidates = self.route_optimizer.get_top_candidates(ranked_candidates, max_candidates=5)

            gemini_decision = await self.gemini_engine.select_optimal_route(
                top_candidates,
                request.dict(),
                external_factors
            )

            # 7️⃣ Cálculo CORREGIDO de FEE
            logger.info("📅 Calculando FEE CORREGIDO...")
            final_response = await self._build_corrected_response(
                request, gemini_decision, external_factors,
                inventory_analysis, ranked_candidates
            )

            processing_time = (time.time() - start_time) * 1000
            logger.info(f"✅ Predicción FEE completada en {processing_time:.1f}ms")

            return final_response

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Error en predicción FEE: {e}")
            raise

    async def _validate_request(self, request: PredictionRequest) -> Dict[str, Any]:
        """✅ Validación básica mejorada"""

        errors = []
        warnings = []

        # Validar producto
        product_info = self.repositories['product'].get_product_by_sku(request.sku_id)
        if not product_info:
            errors.append(f"Producto no encontrado: {request.sku_id}")

        # Validar código postal
        postal_info = self.repositories['postal_code'].get_postal_code_info(request.codigo_postal)
        if not postal_info:
            errors.append(f"Código postal no válido: {request.codigo_postal}")

        # Validar fecha (permitir fechas del pasado para simulaciones)
        # Solo validar que sea una fecha válida
        if request.fecha_compra.year < 2020 or request.fecha_compra.year > 2030:
            warnings.append(f"Fecha de compra fuera del rango esperado: {request.fecha_compra.year}")

        # Información para simulaciones
        if request.fecha_compra.date() < datetime.now().date():
            logger.info(f"📊 Ejecutando SIMULACIÓN para fecha histórica: {request.fecha_compra.date()}")

        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'product_info': product_info,
            'postal_info': postal_info
        }

    async def _analyze_inventory_optimized(self,
                                           request: PredictionRequest,
                                           product_info: Dict[str, Any],
                                           postal_info: Dict[str, Any]) -> Dict[str, Any]:
        """📦 Análisis OPTIMIZADO: busca por proximidad primero"""

        target_lat = postal_info['latitud_centro']
        target_lon = postal_info['longitud_centro']

        # OPTIMIZACIÓN: Buscar tiendas cercanas PRIMERO (radio más grande)
        nearby_stores = self.repositories['store'].find_stores_by_postal_code_range(
            request.codigo_postal, max_distance_km=200.0  # Radio más amplio
        )

        if not nearby_stores:
            logger.warning(f"❌ No hay tiendas cercanas a {request.codigo_postal}")
            return {
                'inventory_available': False,
                'reason': 'Sin tiendas cercanas',
                'split_inventory': None
            }

        # OPTIMIZACIÓN: Filtrar solo tiendas cercanas que tengan stock
        stock_locations = self.repositories['stock'].get_stock_locations(
            request.sku_id, request.cantidad
        )

        # Cruzar tiendas cercanas con stock disponible
        nearby_with_stock = []
        nearby_store_ids = {store['tienda_id'] for store in nearby_stores}

        for stock_loc in stock_locations:
            if stock_loc['tienda_id'] in nearby_store_ids:
                # Agregar distancia de la tienda cercana
                for store in nearby_stores:
                    if store['tienda_id'] == stock_loc['tienda_id']:
                        stock_loc['distancia_km'] = store['distancia_km']
                        break
                nearby_with_stock.append(stock_loc)

        if not nearby_with_stock:
            logger.warning(f"❌ No hay stock disponible en tiendas cercanas")
            return {
                'inventory_available': False,
                'reason': 'Sin stock en tiendas cercanas',
                'split_inventory': None
            }

        # Ordenar por distancia
        nearby_with_stock.sort(key=lambda x: x.get('distancia_km', 999))

        # Calcular split INTELIGENTE
        try:
            split_analysis = self.repositories['stock'].calculate_smart_split_inventory(
                request.sku_id, request.cantidad, nearby_stores
            )
            logger.info(f"📊 Split analysis result keys: {list(split_analysis.keys())}")
            logger.info(f"📊 Split factible: {split_analysis.get('es_factible', 'N/A')}")

        except Exception as e:
            logger.error(f"❌ Error en split de inventario: {e}")
            return {
                'inventory_available': False,
                'reason': f'Error en análisis de split: {str(e)}',
                'split_inventory': None
            }

        # Validar estructura de split_analysis
        if not isinstance(split_analysis, dict):
            logger.error(f"❌ split_analysis no es un diccionario: {type(split_analysis)}")
            return {
                'inventory_available': False,
                'reason': 'Error en estructura de split_analysis',
                'split_inventory': None
            }

        return {
            'inventory_available': split_analysis['es_factible'],
            'split_inventory': split_analysis.get('split_inventory'),  # ✅ Usar get() para seguridad
            'stock_locations': nearby_with_stock,
            'nearby_stores': nearby_stores,
            'split_analysis': split_analysis,
            'reason': split_analysis.get('razon', 'Análisis completado')  # ✅ Incluir razón
        }

    async def _generate_realistic_candidates(self,
                                             inventory_analysis: Dict[str, Any],
                                             postal_info: Dict[str, Any],
                                             external_factors: Dict[str, Any],
                                             request: PredictionRequest) -> List[Dict[str, Any]]:
        """🗺️ Genera candidatos REALISTAS sin filtros de costo estrictos"""

        split_inventory = inventory_analysis['split_analysis']
        target_coordinates = (
            postal_info['latitud_centro'],
            postal_info['longitud_centro']
        )

        # Generar candidatos con el optimizador
        raw_candidates = self.route_optimizer.generate_route_candidates(
            split_inventory, target_coordinates, external_factors, self.repositories
        )

        # Enriquecer candidatos SIN rechazar por costo
        enriched_candidates = []

        for candidate in raw_candidates:
            enhanced_candidate = await self._enhance_candidate_realistic(
                candidate, external_factors, request
            )

            # Validación BÁSICA (solo factibilidad técnica)
            if self._is_technically_feasible(enhanced_candidate, request):
                enriched_candidates.append(enhanced_candidate)
            else:
                logger.warning(f"❌ Ruta técnicamente no factible: {candidate['ruta_id']}")

        logger.info(f"🗺️ Generados {len(enriched_candidates)} candidatos realistas")
        return enriched_candidates

    async def _enhance_candidate_realistic(self,
                                           candidate: Dict[str, Any],
                                           external_factors: Dict[str, Any],
                                           request: PredictionRequest) -> Dict[str, Any]:
        """⚡ Mejora candidato con cálculos REALISTAS"""

        enhanced = candidate.copy()

        # Aplicar factores externos REALES
        factor_demanda = external_factors.get('factor_demanda', 1.0)
        impacto_tiempo = external_factors.get('impacto_tiempo_extra_horas', 0)
        impacto_costo = external_factors.get('impacto_costo_extra_pct', 0)

        # Tiempo total REALISTA
        tiempo_base = candidate['tiempo_total_horas']
        tiempo_ajustado = tiempo_base + impacto_tiempo

        # Costo total REALISTA
        costo_base = candidate['costo_total_mxn']
        costo_ajustado = costo_base * (1 + impacto_costo / 100)

        # Probabilidad ajustada por factores externos
        prob_base = candidate['probabilidad_cumplimiento']
        # Reducir probabilidad en temporadas críticas
        if external_factors.get('es_temporada_critica', False):
            prob_ajustada = prob_base * 0.85
        elif external_factors.get('es_temporada_alta', False):
            prob_ajustada = prob_base * 0.9
        else:
            prob_ajustada = prob_base

        enhanced.update({
            'tiempo_ajustado_horas': round(tiempo_ajustado, 2),
            'costo_ajustado_mxn': round(costo_ajustado, 2),
            'probabilidad_ajustada': round(max(0.4, prob_ajustada), 3),
            'external_factors_applied': {
                'factor_demanda': factor_demanda,
                'impacto_tiempo': impacto_tiempo,
                'impacto_costo': impacto_costo
            }
        })

        return enhanced

    def _is_technically_feasible(self, candidate: Dict[str, Any], request: PredictionRequest) -> bool:
        """✅ Validación técnica BÁSICA (sin límites de costo arbitrarios)"""

        # Solo verificar factibilidad técnica real
        tiempo_max = 7 * 24  # 7 días máximo absoluto
        if candidate['tiempo_ajustado_horas'] > tiempo_max:
            return False

        # Probabilidad mínima técnica
        if candidate['probabilidad_ajustada'] < 0.3:
            return False

        # Verificar que tenga segmentos válidos
        if not candidate.get('segmentos') or len(candidate['segmentos']) == 0:
            return False

        return True

    def _calculate_realistic_fee(self, selected_route: Dict[str, Any],
                                 request: PredictionRequest,
                                 external_factors: Dict[str, Any]) -> FEECalculation:
        """📅 Cálculo REALISTA de FEE con horarios correctos"""

        tiempo_total = selected_route['tiempo_ajustado_horas']
        fecha_compra = request.fecha_compra

        # Determinar tipo de entrega REALISTA
        tipo_entrega = self._determine_realistic_delivery_type(
            tiempo_total, fecha_compra, external_factors
        )

        # Calcular fecha de entrega REALISTA
        fecha_entrega = self._calculate_realistic_delivery_date(
            fecha_compra, tiempo_total, tipo_entrega, external_factors
        )

        # Calcular rango horario REALISTA
        rango_horario = self._calculate_realistic_time_range(fecha_entrega, tipo_entrega)

        # Desglose de tiempos
        tiempo_preparacion = selected_route.get('tiempo_preparacion_total',
                                                settings.TIEMPO_PICKING_PACKING)
        tiempo_transito = tiempo_total - tiempo_preparacion
        tiempo_contingencia = tiempo_total * 0.1

        return FEECalculation(
            fecha_entrega_estimada=fecha_entrega,
            rango_horario_entrega=rango_horario,
            tipo_entrega=tipo_entrega,
            tiempo_total_horas=tiempo_total,
            tiempo_preparacion=tiempo_preparacion,
            tiempo_transito=tiempo_transito,
            tiempo_contingencia=tiempo_contingencia
        )

    def _determine_realistic_delivery_type(self, tiempo_horas: float,
                                           fecha_compra: datetime,
                                           external_factors: Dict[str, Any]) -> TipoEntregaEnum:
        """📦 Determina tipo de entrega REALISTA"""

        hora_compra = fecha_compra.hour
        es_temporada_critica = external_factors.get('es_temporada_critica', False)

        # FLASH: Mismo día (solo si compra antes del corte y no es temporada crítica)
        if (tiempo_horas <= 8 and
                hora_compra <= settings.HORARIO_CORTE_FLASH and
                not es_temporada_critica):
            return TipoEntregaEnum.FLASH

        # EXPRESS: Siguiente día
        elif (tiempo_horas <= 24 and
              hora_compra <= settings.HORARIO_CORTE_EXPRESS):
            return TipoEntregaEnum.EXPRESS

        # STANDARD: 2-3 días
        elif tiempo_horas <= 72:
            return TipoEntregaEnum.STANDARD

        # PROGRAMADA: Más de 3 días
        else:
            return TipoEntregaEnum.PROGRAMADA

    def _calculate_realistic_delivery_date(self, fecha_compra: datetime,
                                           tiempo_horas: float,
                                           tipo_entrega: TipoEntregaEnum,
                                           external_factors: Dict[str, Any]) -> datetime:
        """📅 Calcula fecha REALISTA (optimizado para simulaciones)"""

        # Para simulaciones: calcular desde la fecha de compra proporcionada
        fecha_base = fecha_compra + timedelta(hours=tiempo_horas)

        # Asegurar que es DESPUÉS de la compra (mínimo según tipo de entrega)
        if tipo_entrega == TipoEntregaEnum.FLASH:
            fecha_minima = fecha_compra + timedelta(hours=2)  # FLASH: mínimo 2 horas
        elif tipo_entrega == TipoEntregaEnum.EXPRESS:
            fecha_minima = fecha_compra + timedelta(hours=6)  # EXPRESS: mínimo 6 horas
        else:
            fecha_minima = fecha_compra + timedelta(hours=12)  # STANDARD+: mínimo 12 horas

        fecha_entrega = max(fecha_base, fecha_minima)

        # Ajustar por horario laboral (9 AM - 6 PM) solo si es día laboral
        if fecha_entrega.hour < 9:
            fecha_entrega = fecha_entrega.replace(hour=10, minute=0, second=0)
        elif fecha_entrega.hour > 18:
            # Si es muy tarde, mover al siguiente día hábil
            fecha_entrega = fecha_entrega.replace(hour=10, minute=0, second=0) + timedelta(days=1)

        # Evitar fines de semana para entregas normales (excepto FLASH en área metropolitana)
        if tipo_entrega != TipoEntregaEnum.FLASH:
            while fecha_entrega.weekday() >= 5:  # Sábado=5, Domingo=6
                fecha_entrega += timedelta(days=1)
                fecha_entrega = fecha_entrega.replace(hour=10, minute=0, second=0)

        # Ajuste por factores externos críticos
        if external_factors.get('es_temporada_critica', False):
            # En temporada crítica, agregar días extra según evento
            eventos = external_factors.get('eventos_detectados', [])
            if 'Navidad' in str(eventos):
                extra_days = 2  # Navidad: +2 días
            elif 'Buen_Fin' in str(eventos):
                extra_days = 1  # Buen Fin: +1 día
            else:
                extra_days = 1  # Otros eventos críticos: +1 día

            fecha_entrega += timedelta(days=extra_days)

        return fecha_entrega

    def _calculate_realistic_time_range(self, fecha_entrega: datetime,
                                        tipo_entrega: TipoEntregaEnum) -> Dict[str, dt_time]:
        """🕐 Calcula rango horario REALISTA"""

        # Ventana de entrega según tipo
        if tipo_entrega == TipoEntregaEnum.FLASH:
            # FLASH: ventana de 4 horas
            inicio = max(fecha_entrega - timedelta(hours=2),
                         fecha_entrega.replace(hour=9, minute=0, second=0))
            fin = min(fecha_entrega + timedelta(hours=2),
                      fecha_entrega.replace(hour=18, minute=0, second=0))
        else:
            # Otros: ventana de 6 horas
            inicio = max(fecha_entrega - timedelta(hours=3),
                         fecha_entrega.replace(hour=9, minute=0, second=0))
            fin = min(fecha_entrega + timedelta(hours=3),
                      fecha_entrega.replace(hour=18, minute=0, second=0))

        return {
            'inicio': inicio.time(),
            'fin': fin.time()
        }

    async def _build_corrected_response(self,
                                        request: PredictionRequest,
                                        gemini_decision: Dict[str, Any],
                                        external_factors: Dict[str, Any],
                                        inventory_analysis: Dict[str, Any],
                                        all_candidates: List[Dict[str, Any]]) -> PredictionResponse:
        """🏗️ Construye respuesta CORREGIDA"""

        selected_route = gemini_decision['candidato_seleccionado']

        # Calcular FEE REALISTA
        fee_calculation = self._calculate_realistic_fee(selected_route, request, external_factors)

        # Construir resto de la respuesta (similar al original pero con datos corregidos)
        ruta_completa = self._build_route_structure(selected_route, inventory_analysis)
        factores_estructurados = self._build_external_factors_structure(external_factors)

        # Explicación mejorada
        explicacion_final = f"""
        🎯 RUTA SELECCIONADA: {selected_route['tipo_ruta']} desde {selected_route.get('origen_principal', 'N/A')}

        ⏱️ TIEMPO: {selected_route['tiempo_ajustado_horas']:.1f}h (incluye {external_factors.get('impacto_tiempo_extra_horas', 0)}h por factores externos)
        💰 COSTO: ${selected_route['costo_ajustado_mxn']:.0f} (factor demanda: {external_factors.get('factor_demanda', 1.0):.1f}x)
        🎯 CONFIABILIDAD: {selected_route['probabilidad_ajustada']:.1%}

        📊 FACTORES DECISIVOS:
        • Eventos detectados: {', '.join(external_factors.get('eventos_detectados', ['Ninguno']))}
        • Criticidad logística: {external_factors.get('criticidad_logistica', 'Normal')}
        • Fuente de datos: {external_factors.get('fuente_datos', 'calculado')}
        """

        # Construir candidatos para explicabilidad
        candidatos_lgb = []
        for i, candidate in enumerate(all_candidates[:5]):
            candidato_ruta = CandidatoRuta(
                ruta=self._build_route_structure(candidate, inventory_analysis),
                score_lightgbm=candidate.get('score_lightgbm', 0),
                ranking_position=i + 1,
                features_utilizadas=candidate.get('score_breakdown', {}),
                trade_offs={}
            )
            candidatos_lgb.append(candidato_ruta)

        # Decisión Gemini
        decision_gemini = DecisionGemini(
            candidato_seleccionado=candidatos_lgb[0] if candidatos_lgb else None,
            razonamiento=gemini_decision.get('razonamiento', explicacion_final),
            candidatos_evaluados=candidatos_lgb[:3],
            factores_decisivos=gemini_decision.get('factores_decisivos', ['tiempo_realista', 'costo_optimizado']),
            confianza_decision=gemini_decision.get('confianza_decision', 0.85),
            alertas_gemini=gemini_decision.get('alertas_operativas', [])
        )

        # Explicabilidad completa
        explicabilidad = ExplicabilidadCompleta(
            request_procesado=request,
            factores_externos=factores_estructurados,
            split_inventory=inventory_analysis.get('split_inventory'),
            candidatos_lightgbm=candidatos_lgb,
            decision_gemini=decision_gemini,
            fee_calculation=fee_calculation,
            tiempo_procesamiento_ms=0,
            warnings=[],
            debug_info={
                'total_candidates_generated': len(all_candidates),
                'optimization_method': 'tiempo_costo_optimizado',
                'factors_source': external_factors.get('fuente_datos', 'calculado'),
                'delivery_type_logic': 'realista_corregida'
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
            carrier_principal=self._determine_main_carrier(selected_route),
            costo_envio_mxn=selected_route['costo_ajustado_mxn'],
            probabilidad_cumplimiento=selected_route['probabilidad_ajustada'],
            confianza_prediccion=gemini_decision.get('confianza_decision', 0.85),
            explicabilidad=explicabilidad
        )

        return response

    # Métodos auxiliares (mantener los originales con pequeños ajustes)
    def _build_route_structure(self, route_data: Dict[str, Any],
                               inventory_analysis: Dict[str, Any]) -> RutaCompleta:
        """🏗️ Construye estructura de ruta"""

        segmentos = []
        for i, seg_data in enumerate(route_data.get('segmentos', [])):
            segmento = Segmento(
                segmento_id=f"{route_data['ruta_id']}_seg_{i + 1}",
                origen_id=seg_data['origen'],
                destino_id=seg_data['destino'],
                origen_nombre=seg_data.get('origen_nombre', seg_data['origen']),
                destino_nombre=seg_data.get('destino_nombre', seg_data['destino']),
                distancia_km=seg_data['distancia_km'],
                tiempo_viaje_horas=seg_data['tiempo_horas'],
                tipo_flota=seg_data['tipo_flota'],
                carrier=self._get_carrier_for_fleet(seg_data['tipo_flota']),
                costo_segmento_mxn=route_data['costo_ajustado_mxn'] / len(route_data.get('segmentos', [1])),
                factores_aplicados=route_data.get('factores_aplicados', [])
            )
            segmentos.append(segmento)

        return RutaCompleta(
            ruta_id=route_data['ruta_id'],
            segmentos=segmentos,
            split_inventory=inventory_analysis.get('split_inventory'),
            tiempo_total_horas=route_data['tiempo_ajustado_horas'],
            costo_total_mxn=route_data['costo_ajustado_mxn'],
            distancia_total_km=route_data['distancia_total_km'],
            score_tiempo=route_data.get('score_breakdown', {}).get('tiempo', 0.8),
            score_costo=route_data.get('score_breakdown', {}).get('costo', 0.8),
            score_confiabilidad=route_data['probabilidad_ajustada'],
            score_lightgbm=route_data.get('score_lightgbm', 0),
            estado='FACTIBLE',
            probabilidad_cumplimiento=route_data['probabilidad_ajustada'],
            factores_riesgo=route_data.get('external_factors_applied', {}).get('factores_criticos', [])
        )

    def _build_external_factors_structure(self, external_factors: Dict[str, Any]) -> FactoresExternos:
        """🌤️ Construye estructura de factores externos"""

        return FactoresExternos(
            fecha_analisis=datetime.fromisoformat(external_factors['fecha_analisis']),
            eventos_detectados=external_factors['eventos_detectados'],
            factor_demanda=external_factors['factor_demanda'],
            es_temporada_alta=external_factors['es_temporada_alta'],
            condicion_clima=external_factors['condicion_clima'],
            temperatura_celsius=external_factors['temperatura_celsius'],
            probabilidad_lluvia=external_factors['probabilidad_lluvia'],
            viento_kmh=external_factors.get('viento_esperado', 15),
            trafico_nivel=external_factors['trafico_nivel'],
            impacto_tiempo_extra_horas=external_factors['impacto_tiempo_extra_horas'],
            impacto_costo_extra_pct=external_factors['impacto_costo_extra_pct'],
            zona_seguridad=external_factors.get('zona_seguridad', 'Media'),
            restricciones_vehiculares=external_factors.get('restricciones_vehiculares', []),
            criticidad_logistica=external_factors['criticidad_logistica']
        )

    def _determine_main_carrier(self, route: Dict[str, Any]) -> str:
        """🚚 Determina carrier principal"""
        segmentos = route.get('segmentos', [])
        if not segmentos:
            return 'Liverpool'

        # Buscar el segmento de última milla (al cliente)
        for segmento in segmentos:
            if segmento.get('destino') == 'cliente':
                fleet_type = segmento.get('tipo_flota', 'FI')
                return self._get_carrier_for_fleet(fleet_type)

        # Si no hay segmento al cliente, usar el último
        last_segment = segmentos[-1]
        fleet_type = last_segment.get('tipo_flota', 'FI')
        return self._get_carrier_for_fleet(fleet_type)

    def _get_carrier_for_fleet(self, fleet_type: str) -> str:
        """🏷️ Obtiene carrier por tipo de flota"""
        carrier_map = {
            'FI': 'Liverpool',
            'FE': 'DHL',
            'FI_FE': 'Liverpool + DHL'
        }
        return carrier_map.get(fleet_type, 'Liverpool')