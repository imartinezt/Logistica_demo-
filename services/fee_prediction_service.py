import time
from datetime import datetime, timedelta
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
    """🎯 Servicio principal de predicción FEE con arquitectura híbrida avanzada"""

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

        # Inicializar motores de decisión
        self.route_optimizer = RouteOptimizer()
        self.gemini_engine = GeminiLogisticsDecisionEngine()

        # Cargar modelo LightGBM si existe
        self.route_optimizer.load_model()

        # Métricas de performance
        self.performance_metrics = {
            'total_predictions': 0,
            'avg_processing_time_ms': 0,
            'success_rate': 0,
            'cache_hits': 0
        }

    async def predict_fee(self, request: PredictionRequest) -> PredictionResponse:
        """🚀 Predicción principal con arquitectura híbrida completa"""

        start_time = time.time()

        try:
            logger.info(f"🎯 Iniciando predicción FEE para {request.sku_id} -> {request.codigo_postal}")

            # 1️⃣ Validación y preparación de datos
            validation_result = await self._validate_and_prepare_request(request)
            if not validation_result['is_valid']:
                raise ValueError(f"Validación fallida: {'; '.join(validation_result['errors'])}")

            product_info = validation_result['product_info']
            postal_info = validation_result['postal_info']

            # 2️⃣ Detección de factores externos avanzados
            logger.info("🌤️ Detectando factores externos...")
            external_factors = TemporalFactorDetector.detect_comprehensive_factors(
                request.fecha_compra, request.codigo_postal
            )

            # 3️⃣ Análisis de inventario y split inteligente
            logger.info("📦 Analizando inventario disponible...")
            inventory_analysis = await self._analyze_inventory_availability(
                request, product_info, postal_info
            )

            # 4️⃣ Generación de candidatos de rutas
            logger.info("🗺️ Generando candidatos de rutas...")
            route_candidates = await self._generate_comprehensive_route_candidates(
                inventory_analysis, postal_info, external_factors, request
            )

            if not route_candidates:
                raise ValueError("No se encontraron rutas factibles")

            # 5️⃣ Optimización con LightGBM
            logger.info("🎯 Optimizando rutas con LightGBM...")
            ranked_candidates = self.route_optimizer.rank_candidates_with_lightgbm(
                route_candidates
            )

            # 6️⃣ Selección final con Gemini
            logger.info("🧠 Decisión final con Gemini...")
            top_candidates = self.route_optimizer.get_top_candidates(ranked_candidates)

            gemini_decision = await self.gemini_engine.select_optimal_route(
                top_candidates,
                request.dict(),
                external_factors
            )

            # 7️⃣ Cálculo de FEE y construcción de respuesta
            logger.info("📅 Calculando FEE final...")
            final_response = await self._build_comprehensive_response(
                request, gemini_decision, external_factors,
                inventory_analysis, ranked_candidates
            )

            # 8️⃣ Métricas de performance
            processing_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(processing_time, success=True)

            logger.info(f"✅ Predicción FEE completada en {processing_time:.1f}ms")
            return final_response

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(processing_time, success=False)

            logger.error(f"❌ Error en predicción FEE: {e}")
            raise

    async def _validate_and_prepare_request(self, request: PredictionRequest) -> Dict[str, Any]:
        """✅ Validación avanzada y preparación de datos"""

        errors = []
        warnings = []

        # Validar producto
        product_info = self.repositories['product'].get_product_by_sku(request.sku_id)
        if not product_info:
            errors.append(f"Producto no encontrado: {request.sku_id}")
        else:
            # Validaciones adicionales del producto
            if product_info.get('peso_kg', 0) > 30:
                warnings.append("Producto de peso elevado detectado")

            # Verificar disponibilidad estacional
            if not self.repositories['product'].check_seasonal_availability(
                    request.sku_id, request.fecha_compra
            ):
                warnings.append("Producto fuera de temporada")

        # Validar código postal
        postal_info = self.repositories['postal_code'].get_postal_code_info(request.codigo_postal)
        if not postal_info:
            errors.append(f"Código postal no válido: {request.codigo_postal}")

        # Validar cantidad
        if request.cantidad > 50:
            warnings.append("Cantidad elevada solicitada")

        # Validar fecha (no muy en el pasado)
        if request.fecha_compra < datetime.now() - timedelta(hours=1):
            warnings.append("Fecha de compra en el pasado")

        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'product_info': product_info,
            'postal_info': postal_info
        }

    async def _analyze_inventory_availability(self,
                                              request: PredictionRequest,
                                              product_info: Dict[str, Any],
                                              postal_info: Dict[str, Any]) -> Dict[str, Any]:
        """📦 Análisis avanzado de inventario con split inteligente"""

        # Obtener ubicaciones cercanas al CP destino
        target_lat = postal_info['latitud_centro']
        target_lon = postal_info['longitud_centro']

        nearby_stores = self.repositories['store'].find_stores_by_postal_code_range(
            request.codigo_postal, max_distance_km=150.0  # Aumentado de 100 a 150km
        )

        # Obtener stock disponible
        stock_locations = self.repositories['stock'].get_stock_locations(
            request.sku_id, request.cantidad
        )

        if not stock_locations:
            return {
                'inventory_available': False,
                'reason': 'Sin stock disponible',
                'split_inventory': None,
                'recommendations': ['Verificar reabastecimiento', 'Contactar proveedores']
            }

        # Calcular split óptimo
        split_analysis = self.repositories['stock'].calculate_split_inventory(
            request.sku_id, request.cantidad, nearby_stores
        )

        # Validar split con Gemini si es complejo (pero no bloquear por esto)
        if len(split_analysis.get('split_plan', [])) > 1:
            try:
                gemini_validation = await self.gemini_engine.validate_inventory_split(
                    split_analysis, product_info, request.dict()
                )
                split_analysis['gemini_recommendation'] = gemini_validation
            except Exception as e:
                logger.warning(f"⚠️ Error en validación Gemini del split: {e}")
                # Continuar sin la validación de Gemini

        # Construir ubicaciones de stock
        stock_ubicaciones = []
        for split_item in split_analysis.get('split_plan', []):
            store_info = self.repositories['store'].get_store_by_id(split_item['tienda_id'])
            if store_info:
                ubicacion = UbicacionStock(
                    ubicacion_id=split_item['tienda_id'],
                    ubicacion_tipo='TIENDA',
                    nombre_ubicacion=store_info.get('nombre_tienda', f"Tienda {split_item['tienda_id']}"),
                    stock_disponible=split_item['cantidad'],
                    stock_reservado=0,
                    coordenadas={
                        'lat': store_info['latitud'],
                        'lon': store_info['longitud']
                    },
                    horario_operacion=store_info.get('horario_operacion', '09:00-21:00'),
                    tiempo_preparacion_horas=settings.TIEMPO_PICKING_PACKING
                )
                stock_ubicaciones.append(ubicacion)

        # Construir split inventory completo
        split_inventory = None
        if split_analysis['es_factible']:
            split_inventory = SplitInventory(
                ubicaciones=stock_ubicaciones,
                cantidad_total_requerida=request.cantidad,
                cantidad_total_disponible=split_analysis['cantidad_cubierta'],
                es_split_factible=True,
                razon_split=split_analysis['razon']
            )

        return {
            'inventory_available': split_analysis['es_factible'],
            'split_inventory': split_inventory,
            'stock_locations': stock_locations,
            'nearby_stores': nearby_stores,
            'split_analysis': split_analysis,
            'total_locations_with_stock': len(stock_ubicaciones)
        }

    async def _generate_comprehensive_route_candidates(self,
                                                       inventory_analysis: Dict[str, Any],
                                                       postal_info: Dict[str, Any],
                                                       external_factors: Dict[str, Any],
                                                       request: PredictionRequest) -> List[Dict[str, Any]]:
        """🗺️ Generación completa de candidatos de rutas"""

        split_inventory = inventory_analysis['split_analysis']
        target_coordinates = (
            postal_info['latitud_centro'],
            postal_info['longitud_centro']
        )

        # Generar candidatos usando el optimizador
        raw_candidates = self.route_optimizer.generate_route_candidates(
            split_inventory, target_coordinates, external_factors, self.repositories
        )

        # Enriquecer candidatos con información adicional
        enriched_candidates = []

        for candidate in raw_candidates:
            # Calcular métricas adicionales
            enhanced_candidate = await self._enhance_route_candidate(
                candidate, external_factors, request
            )

            # Validar factibilidad con umbrales más permisivos
            feasibility = self._validate_route_feasibility(
                enhanced_candidate, external_factors, request
            )

            if feasibility['is_feasible']:
                enhanced_candidate.update(feasibility)
                enriched_candidates.append(enhanced_candidate)
            else:
                logger.warning(f"❌ Ruta no factible: {candidate['ruta_id']} - {feasibility['reason']}")

        logger.info(f"🗺️ Generados {len(enriched_candidates)} candidatos factibles")
        return enriched_candidates

    async def _enhance_route_candidate(self,
                                       candidate: Dict[str, Any],
                                       external_factors: Dict[str, Any],
                                       request: PredictionRequest) -> Dict[str, Any]:
        """⚡ Mejora candidato con información adicional"""

        enhanced = candidate.copy()

        # Calcular métricas de eficiencia
        efficiency_metrics = GeoCalculator.calculate_route_efficiency(
            candidate['distancia_total_km'],
            candidate['tiempo_total_horas'],
            candidate['costo_total_mxn']
        )

        enhanced['efficiency_metrics'] = efficiency_metrics

        # Aplicar factores externos (con try-catch para no bloquear)
        try:
            external_impact = await self.gemini_engine.analyze_external_factors_impact(
                external_factors, request.codigo_postal, request.fecha_compra
            )
        except Exception as e:
            logger.warning(f"⚠️ Error analizando factores externos: {e}")
            # Usar valores por defecto
            external_impact = {
                'impacto_tiempo_horas': 0.5,
                'impacto_costo_pct': 5.0,
                'probabilidad_retraso': 0.1,
                'factores_criticos': []
            }

        # Ajustar métricas por factores externos
        tiempo_ajustado = candidate['tiempo_total_horas'] * (1 + external_impact.get('impacto_tiempo_horas', 0) / 24)
        costo_ajustado = candidate['costo_total_mxn'] * (1 + external_impact.get('impacto_costo_pct', 0) / 100)
        probabilidad_ajustada = candidate['probabilidad_cumplimiento'] * (
                    1 - external_impact.get('probabilidad_retraso', 0))

        enhanced.update({
            'tiempo_ajustado_horas': round(tiempo_ajustado, 2),
            'costo_ajustado_mxn': round(costo_ajustado, 2),
            'probabilidad_ajustada': round(max(0.5, probabilidad_ajustada), 3),
            'external_impact': external_impact,
            'factor_urgencia': self._calculate_urgency_factor(request),
            'factor_complejidad': len(candidate.get('segmentos', [])) / 5.0
        })

        # Score normalizado final
        enhanced['score_normalizado'] = self._calculate_normalized_score(enhanced)

        return enhanced

    def _validate_route_feasibility(self,
                                    candidate: Dict[str, Any],
                                    external_factors: Dict[str, Any],
                                    request: PredictionRequest) -> Dict[str, Any]:
        """✅ Validación de factibilidad de ruta (MEJORADA)"""

        feasibility_checks = []
        is_feasible = True
        reason = ""

        # 1. Verificar tiempo máximo aceptable (MÁS PERMISIVO)
        max_time = self._get_max_acceptable_time(request)
        if candidate['tiempo_ajustado_horas'] > max_time:
            is_feasible = False
            reason = f"Tiempo excede máximo aceptable ({max_time}h)"
            feasibility_checks.append(f"❌ Tiempo: {candidate['tiempo_ajustado_horas']:.1f}h > {max_time}h")
        else:
            feasibility_checks.append(f"✅ Tiempo: {candidate['tiempo_ajustado_horas']:.1f}h")

        # 2. Verificar costo máximo aceptable (MÁS PERMISIVO)
        max_cost = self._get_max_acceptable_cost(request)
        if candidate['costo_ajustado_mxn'] > max_cost:
            is_feasible = False
            reason = f"Costo excede máximo aceptable (${max_cost})"
            feasibility_checks.append(f"❌ Costo: ${candidate['costo_ajustado_mxn']} > ${max_cost}")
        else:
            feasibility_checks.append(f"✅ Costo: ${candidate['costo_ajustado_mxn']}")

        # 3. Verificar probabilidad mínima (MÁS PERMISIVO)
        min_probability = 0.5  # Era 0.6, ahora 0.5
        if candidate['probabilidad_ajustada'] < min_probability:
            is_feasible = False
            reason = f"Probabilidad muy baja ({candidate['probabilidad_ajustada']:.1%})"
            feasibility_checks.append(f"❌ Probabilidad: {candidate['probabilidad_ajustada']:.1%}")
        else:
            feasibility_checks.append(f"✅ Probabilidad: {candidate['probabilidad_ajustada']:.1%}")

        # 4. Verificar restricciones de zona (MÁS FLEXIBLE)
        if self.repositories['postal_code'].is_zona_roja(request.codigo_postal):
            has_only_fi = all(seg.get('tipo_flota') == 'FI' for seg in candidate.get('segmentos', []))

            if has_only_fi and candidate.get('tipo_ruta') not in ['hibrida', 'cedis_directo']:
                # Solo advertencia, no bloquear completamente
                feasibility_checks.append("⚠️ Flota: Solo FI en zona roja (riesgo)")
                logger.warning(f"⚠️ Ruta {candidate['ruta_id']} usa solo FI en zona roja")
                # No marcar como no factible, solo reducir probabilidad
                candidate['probabilidad_ajustada'] *= 0.9
            else:
                feasibility_checks.append("✅ Flota: Adecuada para zona roja")
        else:
            feasibility_checks.append("✅ Zona: Sin restricciones")

        # 5. Verificar horarios de operación (MÁS FLEXIBLE)
        hora_compra = request.fecha_compra.hour
        if self._check_delivery_time_feasibility(candidate, hora_compra):
            feasibility_checks.append("✅ Horarios: Compatibles")
        else:
            # Solo advertencia para horarios, no bloquear
            feasibility_checks.append("⚠️ Horarios: Entrega fuera de horario ideal")
            logger.warning(f"⚠️ Ruta {candidate['ruta_id']} fuera de horario ideal")

        return {
            'is_feasible': is_feasible,
            'reason': reason if not is_feasible else 'Ruta factible',
            'feasibility_checks': feasibility_checks,
            'feasibility_score': sum(1 for check in feasibility_checks if check.startswith('✅')) / len(
                feasibility_checks)
        }

    async def _build_comprehensive_response(self,
                                            request: PredictionRequest,
                                            gemini_decision: Dict[str, Any],
                                            external_factors: Dict[str, Any],
                                            inventory_analysis: Dict[str, Any],
                                            all_candidates: List[Dict[str, Any]]) -> PredictionResponse:
        """🏗️ Construcción de respuesta comprehensiva"""

        selected_route = gemini_decision['candidato_seleccionado']

        # Calcular FEE final
        fee_calculation = self._calculate_final_fee(selected_route, request)

        # Construir ruta completa estructurada
        ruta_completa = self._build_route_structure(selected_route, inventory_analysis)

        # Construir factores externos estructurados
        factores_estructurados = self._build_external_factors_structure(external_factors)

        # Generar explicación final con Gemini (con try-catch)
        try:
            final_explanation = await self.gemini_engine.generate_final_explanation(
                selected_route, {
                    'request': request.dict(),
                    'external_factors': external_factors,
                    'all_candidates': len(all_candidates),
                    'decision_process': gemini_decision
                }
            )
        except Exception as e:
            logger.warning(f"⚠️ Error generando explicación final: {e}")
            final_explanation = {
                'resumen_ejecutivo': 'Ruta optimizada seleccionada automáticamente',
                'nivel_confianza': 'Medio'
            }

        # Construir candidatos para explicabilidad
        candidatos_lgb = []
        for candidate in all_candidates[:10]:  # Top 10 para explicabilidad
            candidato_ruta = CandidatoRuta(
                ruta=self._build_route_structure(candidate, inventory_analysis),
                score_lightgbm=candidate.get('score_lightgbm', 0),
                ranking_position=candidate.get('ranking_position', 0),
                features_utilizadas=candidate.get('score_breakdown', {}),
                trade_offs=candidate.get('trade_offs', {})
            )
            candidatos_lgb.append(candidato_ruta)

        # Construir decisión Gemini estructurada
        decision_gemini = DecisionGemini(
            candidato_seleccionado=candidatos_lgb[0] if candidatos_lgb else None,
            razonamiento=gemini_decision.get('razonamiento', ''),
            candidatos_evaluados=candidatos_lgb[:3],
            factores_decisivos=gemini_decision.get('factores_decisivos', []),
            confianza_decision=gemini_decision.get('confianza_decision', 0.8),
            alertas_gemini=gemini_decision.get('alertas_operativas', [])
        )

        # Construir explicabilidad completa
        explicabilidad = ExplicabilidadCompleta(
            request_procesado=request,
            factores_externos=factores_estructurados,
            split_inventory=inventory_analysis.get('split_inventory'),
            candidatos_lightgbm=candidatos_lgb,
            decision_gemini=decision_gemini,
            fee_calculation=fee_calculation,
            tiempo_procesamiento_ms=0,  # Se actualizará después
            warnings=gemini_decision.get('alertas_operativas', []),
            debug_info={
                'total_candidates_generated': len(all_candidates),
                'inventory_locations': len(inventory_analysis.get('stock_locations', [])),
                'external_factors_detected': len(external_factors.get('eventos_detectados', [])),
                'model_version': 'LightGBM + Gemini 2.0',
                'optimization_method': 'hybrid_ml_ai'
            }
        )

        # Determinar carrier principal
        carrier_principal = self._determine_main_carrier(selected_route)

        # Construir respuesta final
        response = PredictionResponse(
            fecha_entrega_estimada=fee_calculation.fecha_entrega_estimada,
            rango_horario={
                'inicio': fee_calculation.rango_horario_entrega['inicio'].strftime('%H:%M'),
                'fin': fee_calculation.rango_horario_entrega['fin'].strftime('%H:%M')
            },
            ruta_seleccionada=ruta_completa,
            tipo_entrega=fee_calculation.tipo_entrega,
            carrier_principal=carrier_principal,
            costo_envio_mxn=selected_route['costo_ajustado_mxn'],
            probabilidad_cumplimiento=selected_route['probabilidad_ajustada'],
            confianza_prediccion=gemini_decision.get('confianza_decision', 0.8),
            explicabilidad=explicabilidad
        )

        return response

    def _calculate_final_fee(self, selected_route: Dict[str, Any],
                             request: PredictionRequest) -> FEECalculation:
        """📅 Cálculo final de FEE con rangos horarios"""

        tiempo_total = selected_route['tiempo_ajustado_horas']
        fecha_compra = request.fecha_compra

        # Determinar tipo de entrega
        tipo_entrega = self._determine_delivery_type(tiempo_total, fecha_compra.hour)

        # Calcular fecha de entrega base
        fecha_entrega_base = fecha_compra + timedelta(hours=tiempo_total)

        # Ajustar a horario laboral
        fecha_entrega_final = self._adjust_to_business_hours(fecha_entrega_base, tipo_entrega)

        # Calcular rango horario (±2 horas)
        inicio_ventana = max(
            fecha_entrega_final.replace(hour=9, minute=0),
            fecha_entrega_final - timedelta(hours=2)
        )
        fin_ventana = min(
            fecha_entrega_final.replace(hour=18, minute=0),
            fecha_entrega_final + timedelta(hours=2)
        )

        # Desglose de tiempos
        tiempo_preparacion = sum(
            settings.TIEMPO_PICKING_PACKING
            for _ in selected_route.get('segmentos', [])
        )

        tiempo_transito = tiempo_total - tiempo_preparacion
        tiempo_contingencia = tiempo_total * 0.1  # 10% buffer

        return FEECalculation(
            fecha_entrega_estimada=fecha_entrega_final,
            rango_horario_entrega={
                'inicio': inicio_ventana.time(),
                'fin': fin_ventana.time()
            },
            tipo_entrega=tipo_entrega,
            tiempo_total_horas=tiempo_total,
            tiempo_preparacion=tiempo_preparacion,
            tiempo_transito=tiempo_transito,
            tiempo_contingencia=tiempo_contingencia
        )

    # Métodos auxiliares (MEJORADOS)
    def _get_max_acceptable_time(self, request: PredictionRequest) -> float:
        """⏰ Tiempo máximo aceptable (MÁS PERMISIVO)"""
        # Tiempo base más permisivo
        base_time = 168.0  # 7 días máximo (era 120.0 = 5 días)

        # Ajustar por cantidad
        if request.cantidad > 10:
            base_time *= 1.2  # 20% más tiempo para cantidades grandes

        return base_time

    def _get_max_acceptable_cost(self, request: PredictionRequest) -> float:
        """💰 Costo máximo aceptable (MÁS PERMISIVO)"""
        base_cost = 800.0  # Era 500.0, ahora más permisivo

        if request.cantidad > 10:
            base_cost *= 1.8  # Era 1.5, ahora más permisivo para cantidades grandes

        return base_cost

    def _calculate_urgency_factor(self, request: PredictionRequest) -> float:
        """⚡ Factor de urgencia basado en hora de compra"""
        hora = request.fecha_compra.hour
        if hora <= settings.HORARIO_CORTE_FLASH:
            return 1.5  # Alta urgencia para FLASH
        elif hora <= settings.HORARIO_CORTE_EXPRESS:
            return 1.2  # Media urgencia para EXPRESS
        else:
            return 1.0  # Sin urgencia

    def _calculate_normalized_score(self, candidate: Dict[str, Any]) -> float:
        """📊 Score normalizado multiobjetivo"""

        # Normalizar métricas (invertir para que menor sea mejor en tiempo/costo)
        tiempo_norm = max(0, 1 - (candidate['tiempo_ajustado_horas'] - 2) / 168)  # 0-170h range
        costo_norm = max(0, 1 - (candidate['costo_ajustado_mxn'] - 50) / 800)  # $50-850 range (era 550)
        prob_norm = candidate['probabilidad_ajustada']
        dist_norm = max(0, 1 - candidate['distancia_total_km'] / 2000)  # 0-2000km range

        # Score ponderado
        score = (
                settings.PESO_TIEMPO * tiempo_norm +
                settings.PESO_COSTO * costo_norm +
                settings.PESO_PROBABILIDAD * prob_norm +
                settings.PESO_DISTANCIA * dist_norm
        )

        return round(score, 4)

    def _determine_delivery_type(self, tiempo_horas: float, hora_compra: int) -> TipoEntregaEnum:
        """📦 Determina tipo de entrega"""
        if tiempo_horas <= 24 and hora_compra <= settings.HORARIO_CORTE_FLASH:
            return TipoEntregaEnum.FLASH
        elif tiempo_horas <= 48 and hora_compra <= settings.HORARIO_CORTE_EXPRESS:
            return TipoEntregaEnum.EXPRESS
        elif tiempo_horas <= 72:
            return TipoEntregaEnum.STANDARD
        else:
            return TipoEntregaEnum.PROGRAMADA

    def _adjust_to_business_hours(self, fecha: datetime,
                                  tipo_entrega: TipoEntregaEnum) -> datetime:
        """🕘 Ajusta fecha a horario laboral"""

        # Evitar fines de semana
        while fecha.weekday() >= 5:
            fecha += timedelta(days=1)

        # Ajustar hora
        if fecha.hour < 9:
            fecha = fecha.replace(hour=10, minute=0)
        elif fecha.hour > 18:
            fecha = fecha.replace(hour=17, minute=0)

        return fecha

    def _check_delivery_time_feasibility(self, candidate: Dict[str, Any],
                                         hora_compra: int) -> bool:
        """⏰ Verifica factibilidad de horarios (MÁS PERMISIVO)"""

        tiempo_total = candidate.get('tiempo_ajustado_horas', 0)

        # Reglas de cut-off más permisivas
        if tiempo_total <= 24:  # FLASH
            return hora_compra <= settings.HORARIO_CORTE_FLASH + 2  # +2 horas de tolerancia
        elif tiempo_total <= 48:  # EXPRESS
            return hora_compra <= settings.HORARIO_CORTE_EXPRESS + 1  # +1 hora de tolerancia
        else:
            return True  # STANDARD/PROGRAMADA sin restricción

    def _build_route_structure(self, route_data: Dict[str, Any],
                               inventory_analysis: Dict[str, Any]) -> RutaCompleta:
        """🏗️ Construye estructura de ruta completa"""

        # Construir segmentos
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
            factores_riesgo=route_data.get('external_impact', {}).get('factores_criticos', [])
        )

    def _build_external_factors_structure(self,
                                          external_factors: Dict[str, Any]) -> FactoresExternos:
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
            restricciones_vehiculares=external_factors.get('zonas_criticas', []),
            criticidad_logistica=external_factors['criticidad_logistica']
        )

    def _determine_main_carrier(self, route: Dict[str, Any]) -> str:
        """🚚 Determina carrier principal"""

        segmentos = route.get('segmentos', [])
        if not segmentos:
            return 'Liverpool'

        # Buscar el segmento más largo (última milla)
        longest_segment = max(segmentos, key=lambda x: x.get('distancia_km', 0))
        fleet_type = longest_segment.get('tipo_flota', 'FI')

        return self._get_carrier_for_fleet(fleet_type)

    def _get_carrier_for_fleet(self, fleet_type: str) -> str:
        """🏷️ Obtiene carrier por tipo de flota"""

        carrier_map = {
            'FI': 'Liverpool',
            'FE': 'DHL',
            'FI_FE': 'Liverpool + DHL'
        }

        return carrier_map.get(fleet_type, 'Liverpool')

    def _update_performance_metrics(self, processing_time_ms: float, success: bool):
        """📊 Actualiza métricas de performance"""

        self.performance_metrics['total_predictions'] += 1

        # Promedio móvil del tiempo de procesamiento
        current_avg = self.performance_metrics['avg_processing_time_ms']
        total_predictions = self.performance_metrics['total_predictions']

        new_avg = ((current_avg * (total_predictions - 1)) + processing_time_ms) / total_predictions
        self.performance_metrics['avg_processing_time_ms'] = round(new_avg, 2)

        # Tasa de éxito
        if success:
            success_count = self.performance_metrics['success_rate'] * (total_predictions - 1) + 1
        else:
            success_count = self.performance_metrics['success_rate'] * (total_predictions - 1)

        self.performance_metrics['success_rate'] = round(success_count / total_predictions, 3)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """📈 Obtiene métricas de performance"""
        return self.performance_metrics.copy()

    async def health_check(self) -> Dict[str, Any]:
        """🏥 Health check del sistema"""

        try:
            # Verificar repositorios
            repo_status = {}
            for name, repo in self.repositories.items():
                try:
                    data = repo.load_data()
                    repo_status[name] = {
                        'status': 'healthy',
                        'rows': data.height if hasattr(data, 'height') else len(data)
                    }
                except Exception as e:
                    repo_status[name] = {'status': 'error', 'error': str(e)}

            # Verificar LightGBM
            lgb_status = 'trained' if self.route_optimizer.is_trained else 'not_trained'

            return {
                'system_status': 'healthy',
                'repositories': repo_status,
                'lightgbm_model': lgb_status,
                'performance_metrics': self.performance_metrics,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'system_status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }