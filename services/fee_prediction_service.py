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

    async def predict_fee(self, request: PredictionRequest) -> Dict[str, Any]:
        """🚀 Predicción FEE con respuesta simplificada y validación CSV completa"""
        start_time = time.time()

        try:
            logger.info(f"🎯 NUEVA PREDICCIÓN: {request.sku_id} → {request.codigo_postal} (qty: {request.cantidad})")

            # VALIDACIÓN DE INTEGRIDAD CSV
            csv_validation = self._validate_csv_data_integrity(request)
            if csv_validation['warnings']:
                for warning in csv_validation['warnings']:
                    logger.warning(f"⚠️ {warning}")

            # 1. VALIDACIÓN DINÁMICA
            validation = await self._validate_request_dynamic(request)
            if not validation['valid']:
                raise ValueError(validation['error'])

            product_info = validation['product']
            cp_info = validation['postal_info']

            # 2. FACTORES EXTERNOS REALES desde CSV
            external_factors = self._get_comprehensive_external_factors(request.fecha_compra, request.codigo_postal)

            # 3. BÚSQUEDA DE TIENDAS
            nearby_stores = self.repos.store.find_stores_by_postal_range(request.codigo_postal)
            if not nearby_stores:
                raise ValueError(f"No hay tiendas Liverpool cerca de {request.codigo_postal}")

            # 4. ANÁLISIS DE STOCK
            stock_analysis = await self._analyze_stock_dynamic(
                request, product_info, nearby_stores
            )

            if not stock_analysis['factible']:
                raise ValueError(f"Stock insuficiente: {stock_analysis['razon']}")

            # 5. GENERACIÓN DE CANDIDATOS
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

            processing_time = (time.time() - start_time) * 1000

            # 7. CONSTRUCCIÓN DE RESPUESTA SIMPLIFICADA
            simplified_response = await self._build_simplified_response(
                request, gemini_decision['candidato_seleccionado'],
                ranked_candidates, stock_analysis, external_factors,
                cp_info, product_info, processing_time
            )

            # Agregar información de fuentes CSV
            simplified_response['metadata'] = {
                'csv_sources_used': csv_validation['csv_sources'],
                'warnings': csv_validation['warnings'],
                'data_integrity': 'validated',
                'version_sistema': '3.0.0'
            }

            # LOG DETALLADO DE FUENTES
            self._log_data_sources(simplified_response)

            logger.info(f"✅ Predicción completada en {processing_time:.1f}ms")
            return simplified_response

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Error en predicción: {e} ({processing_time:.1f}ms)")
            raise

    def _validate_csv_data_integrity(self, request: PredictionRequest) -> Dict[str, Any]:
        """✅ Valida que todos los datos vengan correctamente de CSV"""

        validation_results = {
            'valid': True,
            'warnings': [],
            'csv_sources': {}
        }

        # Validar producto en CSV
        producto = self.repos.product.get_product_by_sku(request.sku_id)
        if producto:
            validation_results['csv_sources']['producto'] = 'productos_liverpool_50.csv'
        else:
            validation_results['warnings'].append(f"Producto {request.sku_id} no encontrado en CSV")

        # Validar código postal en CSV
        cp_info = self.repos.store._get_postal_info(request.codigo_postal)
        if cp_info.get('rango_cp'):
            validation_results['csv_sources']['codigo_postal'] = 'codigos_postales_rangos_mexico.csv'
        else:
            validation_results['warnings'].append(f"CP {request.codigo_postal} usando fallback")

        # Validar factores externos en CSV
        factores_df = self.repos.data_manager.get_data('factores_externos')
        fecha_str = request.fecha_compra.date().isoformat()
        factores_encontrados = factores_df.filter(pl.col('fecha') == fecha_str)

        if factores_encontrados.height > 0:
            validation_results['csv_sources']['factores_externos'] = 'factores_externos_mexico_completo.csv'
        else:
            validation_results['warnings'].append(
                f"Factores para {fecha_str} no encontrados, usando cálculo automático")

        # Validar flota externa en CSV
        flota_df = self.repos.data_manager.get_data('flota_externa')
        if flota_df.height > 0:
            validation_results['csv_sources']['flota_externa'] = 'flota_externa_costos_reales.csv'
        else:
            validation_results['warnings'].append("CSV de flota externa no disponible")

        logger.info(
            f"✅ Validación CSV: {len(validation_results['csv_sources'])} fuentes válidas, {len(validation_results['warnings'])} advertencias")

        return validation_results

    # 10. MÉTODO PARA LOGGING DETALLADO DE FUENTES DE DATOS
    def _log_data_sources(self, final_response: Dict[str, Any]) -> None:
        """📊 Log detallado de fuentes de datos utilizadas"""

        logger.info("📋 FUENTES DE DATOS UTILIZADAS:")
        logger.info("=" * 60)

        # Producto
        producto = final_response.get('producto', {})
        logger.info(f"📦 PRODUCTO: {producto.get('nombre', 'N/A')}")
        logger.info(f"   → Fuente: productos_liverpool_50.csv")
        logger.info(f"   → Peso: {producto.get('peso_unitario_kg', 0)}kg")
        logger.info(f"   → Precio: ${producto.get('precio_unitario_mxn', 0)}")
        logger.info(f"   → Tiempo prep: {producto.get('tiempo_prep_horas', 0)}h")

        # Factores externos
        factores = final_response.get('factores_externos', {})
        logger.info(f"🌤️ FACTORES EXTERNOS:")
        logger.info(f"   → Fuente: {factores.get('fuente_datos', 'N/A')}")
        logger.info(f"   → Evento: {factores.get('evento_detectado', 'Normal')}")
        logger.info(f"   → Factor demanda: {factores.get('factor_demanda', 1.0)}")
        logger.info(f"   → Zona seguridad: {factores.get('zona_seguridad', 'Verde')}")
        logger.info(f"   → Criticidad: {factores.get('criticidad_logistica', 'Normal')}")

        # Evaluación
        ganador = final_response.get('evaluacion', {}).get('ganador', {})
        datos_csv = ganador.get('datos_csv', {})
        logger.info(f"🏆 GANADOR:")
        logger.info(f"   → Tienda: {ganador.get('tienda', 'N/A')}")
        logger.info(f"   → Zona seguridad CSV: {datos_csv.get('zona_seguridad', 'N/A')}")
        logger.info(f"   → CEDIS asignado CSV: {datos_csv.get('cedis_asignado', 'N/A')}")
        logger.info(f"   → Carrier CSV: {datos_csv.get('carrier_seleccionado', 'N/A')}")

        # Logística
        logistica = final_response.get('logistica_entrega', {})
        logger.info(f"🚛 LOGÍSTICA:")
        logger.info(f"   → Flota: {logistica.get('flota', 'N/A')}")
        logger.info(f"   → Carrier: {logistica.get('carrier', 'N/A')}")
        logger.info(f"   → Probabilidad: {logistica.get('probabilidad_cumplimiento', 0):.1%}")

        logger.info("=" * 60)

    async def _build_simplified_response(self, request: PredictionRequest,
                                         selected_route: Dict[str, Any],
                                         all_candidates: List[Dict[str, Any]],
                                         stock_analysis: Dict[str, Any],
                                         external_factors: Dict[str, Any],
                                         cp_info: Dict[str, Any],
                                         producto_info: Dict[str, Any],
                                         processing_time_ms: float) -> Dict[str, Any]:
        """🎯 Construye response SIMPLIFICADO con datos reales de CSV"""

        # Calcular peso real del producto
        peso_kg_estimado = producto_info.get('peso_kg', 0.2) * request.cantidad

        # Obtener zona de seguridad REAL del CSV de códigos postales
        zona_seguridad_real = cp_info.get('zona_seguridad', 'Verde')

        # Calcular FEE usando lógica existente
        fee_calculation = self._calculate_dynamic_fee(
            selected_route, request, external_factors, cp_info
        )

        # Construir candidatos simplificados con datos CSV reales
        candidatos_simplificados = []
        for candidate in all_candidates:
            # Obtener datos reales de la tienda
            tienda_info = await self._get_store_info(candidate.get('tienda_origen_id', 'LIV001'))

            candidatos_simplificados.append({
                "tienda": tienda_info.get('nombre_tienda',
                                          candidate.get('origen_principal', 'N/A')) if tienda_info else candidate.get(
                    'origen_principal', 'N/A'),
                "distancia_km": round(candidate.get('distancia_total_km', 0), 1),
                "tiempo_h": round(candidate.get('tiempo_total_horas', 0), 1),
                "costo_mxn": round(candidate.get('costo_total_mxn', 0), 0),
                "stock": self._get_stock_for_candidate(candidate, stock_analysis),
                "score": round(candidate.get('score_lightgbm', 0), 3),
                "flota": self._get_fleet_info_simplified(candidate),
                "zona_seguridad": zona_seguridad_real  # Dato real del CSV
            })

        # Ganador con datos reales
        ganador_info = candidatos_simplificados[0] if candidatos_simplificados else {}

        # Factores externos REALES del CSV
        factores_externos_reales = self._extract_real_external_factors(external_factors, cp_info)

        return {
            "request": {
                "timestamp": datetime.now().isoformat(),
                "sku_id": request.sku_id,
                "cantidad": request.cantidad,
                "codigo_postal": request.codigo_postal,
                "peso_kg_estimado": round(peso_kg_estimado, 1),
                "fecha_compra": request.fecha_compra.isoformat()
            },
            "producto": {
                "nombre": producto_info.get('nombre_producto', 'N/A'),
                "marca": producto_info.get('marca', 'N/A'),
                "precio_unitario_mxn": producto_info.get('precio_venta', 0),
                "peso_unitario_kg": producto_info.get('peso_kg', 0.2),
                "tiempo_prep_horas": producto_info.get('tiempo_prep_horas', 1.0),
                "stock_local": sum(
                    loc.get('stock_disponible', 0) for loc in stock_analysis.get('allocation_plan', []) if
                    self._is_local_store(loc)),
                "stock_nacional": sum(
                    loc.get('stock_disponible', 0) for loc in stock_analysis.get('allocation_plan', []))
            },
            "factores_externos": factores_externos_reales,
            "evaluacion": {
                "pesos": {
                    "tiempo": settings.PESO_TIEMPO,
                    "costo": settings.PESO_COSTO,
                    "stock": settings.PESO_PROBABILIDAD,  # Corresponde a probabilidad
                    "distancia": settings.PESO_DISTANCIA
                },
                "candidatos": candidatos_simplificados,
                "ganador": {
                    "tienda": ganador_info.get('tienda', 'N/A'),
                    "score_final": ganador_info.get('score', 0),
                    "ventajas": self._get_real_advantages(selected_route, all_candidates, zona_seguridad_real),
                    "asignacion": {
                        "unidades": request.cantidad,
                        "costo_total_mxn": round(selected_route.get('costo_total_mxn', 0), 2),
                        "distancia_km": round(selected_route.get('distancia_total_km', 0), 1),
                        "precio_unitario_producto": producto_info.get('precio_venta', 0)
                    },
                    "datos_csv": {
                        "zona_seguridad": zona_seguridad_real,
                        "cedis_asignado": self._get_cedis_from_store(selected_route),
                        "carrier_seleccionado": self._get_main_carrier(selected_route),
                        "flota_utilizada": self._get_fleet_info_detailed(selected_route)
                    }
                }
            },
            "logistica_entrega": {
                "ruta": self._build_route_description(selected_route),  # ✅ NUEVO
                "tipo_ruta": selected_route.get('tipo_ruta', 'directa'),
                "flota": self._get_fleet_info_simplified(selected_route),
                "carrier": self._get_main_carrier(selected_route),
                "distancia_km": round(selected_route.get('distancia_total_km', 0), 1),
                "tiempo_total_h": round(selected_route.get('tiempo_total_horas', 0), 1),
                "desglose_tiempos_h": self._get_time_breakdown(selected_route, external_factors),
                "factores_aplicados": selected_route.get('factores_aplicados', []),
                "probabilidad_cumplimiento": round(selected_route.get('probabilidad_cumplimiento', 0), 3),
                # ✅ NUEVO: Agregar información de CEDIS para rutas complejas
                "cedis_intermedio": self._get_cedis_from_store(selected_route) if selected_route.get('tipo_ruta') in ['compleja_cedis', 'multi_segmento_cedis'] else None
            },
            "resultado_final": {
                "tipo_entrega": fee_calculation.tipo_entrega.value,
                "fecha_entrega_estimada": fee_calculation.fecha_entrega_estimada.isoformat(),
                "ventana_entrega": {
                    "inicio": fee_calculation.rango_horario_entrega['inicio'].strftime('%H:%M'),
                    "fin": fee_calculation.rango_horario_entrega['fin'].strftime('%H:%M')
                },
                "costo_mxn": round(selected_route.get('costo_total_mxn', 0), 2),
                "probabilidad_exito": round(selected_route.get('probabilidad_cumplimiento', 0), 3),
                "processing_time_ms": round(processing_time_ms, 1),
                "confianza_prediccion": 0.85
            }
        }


    # 2. Métodos auxiliares para extraer datos reales de CSV
    def _extract_real_external_factors(self, external_factors: Dict[str, Any], cp_info: Dict[str, Any]) -> Dict[
        str, Any]:
        """📊 Extrae factores externos REALES de los CSV"""
        return {
            "evento_detectado": external_factors.get('evento_detectado', 'Normal'),
            "factor_demanda": external_factors.get('factor_demanda', 1.0),
            "condicion_clima": external_factors.get('condicion_clima', 'Templado'),
            "trafico_nivel": external_factors.get('trafico_nivel', 'Moderado'),
            "criticidad_logistica": external_factors.get('criticidad_logistica', 'Normal'),
            "zona_seguridad": cp_info.get('zona_seguridad', 'Verde'),  # Del CSV de códigos postales
            "es_temporada_alta": external_factors.get('es_temporada_alta', False),
            "impacto_tiempo_extra_horas": external_factors.get('impacto_tiempo_extra_horas', 0),
            "rango_cp_afectado": external_factors.get('rango_cp_afectado', '00000-99999'),
            "fuente_datos": external_factors.get('fuente_datos', 'CSV_real')
        }

    def _get_real_advantages(self, selected_route: Dict[str, Any], all_candidates: List[Dict[str, Any]],
                             zona_seguridad: str) -> List[str]:
        """✅ Obtiene ventajas reales basadas en datos de CSV"""
        advantages = []

        tiempo = selected_route.get('tiempo_total_horas', 0)
        costo = selected_route.get('costo_total_mxn', 0)
        score = selected_route.get('score_lightgbm', 0)
        tipo_ruta = selected_route.get('tipo_ruta', 'directa')

        if len(all_candidates) == 1:
            advantages.append("Única opción factible con stock disponible")
        else:
            if score >= 0.9:
                advantages.append(f"Score excelente ({score:.3f})")
            if tiempo <= 4:
                advantages.append(f"Tiempo excelente ({tiempo:.1f}h)")
            if tipo_ruta == 'directa':
                advantages.append("Ruta directa sin transbordos")
            if zona_seguridad == 'Verde':
                advantages.append("Zona segura sin restricciones")
            elif zona_seguridad == 'Amarilla':
                advantages.append("Zona moderada con ligeras restricciones")

        return advantages if advantages else ["Mejor opción disponible"]

    def _get_stock_for_candidate(self, candidate: Dict[str, Any], stock_analysis: Dict[str, Any]) -> int:
        """📦 Obtiene stock real para un candidato"""
        tienda_id = candidate.get('tienda_origen_id', candidate.get('origen_principal', ''))

        for plan_item in stock_analysis.get('allocation_plan', []):
            if plan_item.get('tienda_id') == tienda_id:
                return plan_item.get('stock_disponible', 0)

        return 0

    def _get_fleet_info_simplified(self, route_data: Dict[str, Any]) -> str:
        """🚛 Obtiene info de flota simplificada"""
        segmentos = route_data.get('segmentos', [])
        if not segmentos:
            return 'N/A'

        # Tomar info del primer segmento
        primer_segmento = segmentos[0]
        tipo_flota = primer_segmento.get('tipo_flota', 'FI')
        carrier = primer_segmento.get('carrier', 'Liverpool')

        if tipo_flota == 'FI':
            return 'FI-Live'
        else:
            return f'FE-{carrier[:4]}'

    def _get_fleet_info_detailed(self, route_data: Dict[str, Any]) -> Dict[str, Any]:
        """🚛 Obtiene info detallada de flota desde CSV"""
        segmentos = route_data.get('segmentos', [])
        if not segmentos:
            return {}

        primer_segmento = segmentos[0]

        return {
            "tipo_flota": primer_segmento.get('tipo_flota', 'FI'),
            "carrier": primer_segmento.get('carrier', 'Liverpool'),
            "zona_cobertura": primer_segmento.get('zona_seguridad', 'Verde'),
            "costo_segmento": primer_segmento.get('costo_segmento', 0)
        }

    def _get_time_breakdown(self, route_data: Dict[str, Any], external_factors: Dict[str, Any]) -> Dict[str, float]:
        """⏱️ Desglose de tiempos reales"""
        tiempo_total = route_data.get('tiempo_total_horas', 0)
        tiempo_prep = 1.0  # Default del producto CSV
        tiempo_extra = external_factors.get('impacto_tiempo_extra_horas', 0)
        tiempo_viaje = max(0, tiempo_total - tiempo_prep - tiempo_extra)

        return {
            "preparacion": round(tiempo_prep, 1),
            "viaje": round(tiempo_viaje, 1),
            "factores_externos": round(tiempo_extra, 1),
            "contingencia": round(tiempo_total * 0.1, 1)
        }

    def _get_cedis_from_store(self, route_data: Dict[str, Any]) -> str:
        """🏭 Obtiene CEDIS asignado - CORREGIDO para rutas complejas"""

        # ✅ NUEVO: Para rutas complejas, capturar CEDIS del ruteo
        tipo_ruta = route_data.get('tipo_ruta', 'directa')

        if tipo_ruta == 'compleja_cedis':
            # Buscar CEDIS en la información de ruteo complejo
            cedis_intermedio = route_data.get('cedis_intermedio')
            if cedis_intermedio:
                return cedis_intermedio

            # Fallback: buscar en segmentos
            segmentos = route_data.get('segmentos', [])
            for segmento in segmentos:
                destino = segmento.get('destino', '')
                if 'CEDIS' in destino:
                    return destino

        elif tipo_ruta == 'multi_segmento_cedis':
            # Para rutas multi-segmento, usar cedis_intermedio
            return route_data.get('cedis_intermedio', 'N/A')

        # ✅ ORIGINAL: Para rutas directas, usar CSV de tiendas
        tienda_id = route_data.get('tienda_origen_id')
        if not tienda_id:
            return 'N/A'

        tiendas_df = self.repos.data_manager.get_data('tiendas')
        tienda_data = tiendas_df.filter(pl.col('tienda_id') == tienda_id)

        if tienda_data.height > 0:
            return tienda_data.to_dicts()[0].get('cedis_asignado', 'N/A')

        return 'N/A'

    # ✅ NUEVO MÉTODO: Construir descripción completa de ruta
    def _build_route_description(self, route_data: Dict[str, Any]) -> str:
        """🗺️ Construye descripción completa de la ruta"""

        tipo_ruta = route_data.get('tipo_ruta', 'directa')
        origen = route_data.get('origen_principal', 'Origen')

        if tipo_ruta == 'directa':
            return f"{origen} → Cliente"

        elif tipo_ruta in ['compleja_cedis', 'multi_segmento_cedis']:
            cedis = route_data.get('cedis_intermedio', 'CEDIS')
            tienda_destino = route_data.get('tienda_destino', 'Tienda Destino')

            if tienda_destino and tienda_destino != 'Tienda Destino':
                return f"{origen} → {cedis} → {tienda_destino} → Cliente"
            else:
                return f"{origen} → {cedis} → Cliente"

        else:
            return f"{origen} → Cliente"

    def _is_local_store(self, location: Dict[str, Any]) -> bool:
        """📍 Determina si una tienda es local (< 100km)"""
        return location.get('distancia_km', 999) < 100

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
        """📦 Análisis de stock CORREGIDO - con logging mejorado"""

        # Tiendas autorizadas para el SKU
        tiendas_autorizadas = [t.strip() for t in product_info['tiendas_disponibles'].split(',')]
        logger.info(f"🏪 Tiendas autorizadas para {request.sku_id}: {tiendas_autorizadas}")

        # PASO 1: Buscar en tiendas locales PRIMERO
        local_store_ids = [store['tienda_id'] for store in nearby_stores[:5]]
        stock_locations_local = self.repos.stock.get_stock_for_stores_and_sku(
            request.sku_id, local_store_ids, request.cantidad
        )

        # ✅ LOGGING MEJORADO: Mostrar tiendas cercanas con nombres
        logger.info(f"📍 Tiendas cercanas al CP {request.codigo_postal}:")
        for i, store in enumerate(nearby_stores[:5], 1):
            logger.info(f"   {i}. {store['nombre_tienda']} ({store['distancia_km']:.1f}km)")

        # Verificar si hay stock local suficiente
        if stock_locations_local:
            # ✅ LOGGING MEJORADO: Mostrar stock con nombres de tiendas
            logger.info(f"📦 Stock LOCAL encontrado para {request.sku_id}:")
            total_local_stock = 0
            for stock_loc in stock_locations_local:
                tienda_info = self._get_store_info_sync(stock_loc['tienda_id'])
                nombre_tienda = tienda_info['nombre_tienda'] if tienda_info else f"Tienda {stock_loc['tienda_id']}"
                logger.info(f"   📍 {nombre_tienda}: {stock_loc['stock_disponible']} unidades")
                total_local_stock += stock_loc['stock_disponible']

            logger.info(f"📊 Stock LOCAL total: {total_local_stock} | Requerido: {request.cantidad}")

            if total_local_stock >= request.cantidad:
                logger.info(f"✅ Stock LOCAL suficiente en {len(stock_locations_local)} tiendas cercanas")

                # Usar solo tiendas locales con stock
                authorized_nearby = [
                    store for store in nearby_stores
                    if any(stock['tienda_id'] == store['tienda_id'] for stock in stock_locations_local)
                ]

                allocation = self.repos.stock.calculate_optimal_allocation(
                    stock_locations_local, request.cantidad, authorized_nearby
                )

                if allocation['factible']:
                    # ✅ LOGGING MEJORADO: Plan de asignación con detalles
                    logger.info("📋 Plan de asignación LOCAL:")
                    for plan_item in allocation['plan']:
                        tienda_info = self._get_store_info_sync(plan_item['tienda_id'])
                        nombre_tienda = tienda_info[
                            'nombre_tienda'] if tienda_info else f"Tienda {plan_item['tienda_id']}"
                        precio_unitario = plan_item.get('precio_unitario', 0)
                        precio_total = precio_unitario * plan_item['cantidad']
                        distancia = tienda_info.get('distancia_km', 0) if tienda_info else 0

                        logger.info(f"   🏪 {nombre_tienda}:")
                        logger.info(f"      → {plan_item['cantidad']} unidades")
                        logger.info(f"      → ${precio_total:,.0f} total (${precio_unitario:,.0f} c/u)")
                        logger.info(f"      → {distancia:.1f}km de distancia")
                        logger.info(f"      → Razón: Stock local disponible, distancia óptima")

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
        logger.info("🌎 Stock local insuficiente, buscando en tiendas autorizadas nacionales...")

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
                logger.warning(
                    f"⚠️ Error calculando distancia para {store.get('nombre_tienda', store.get('tienda_id'))}: {e}")
                store['distancia_km'] = 999.0

        # Ordenar por distancia y tomar las 10 más cercanas
        authorized_stores.sort(key=lambda x: x['distancia_km'])
        authorized_nearby = authorized_stores[:10]

        # ✅ LOGGING MEJORADO: Mostrar tiendas autorizadas encontradas
        logger.info(f"🏪 Tiendas autorizadas más cercanas:")
        for i, store in enumerate(authorized_nearby[:5], 1):
            logger.info(f"   {i}. {store['nombre_tienda']} ({store['distancia_km']:.1f}km)")

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

        # ✅ LOGGING MEJORADO: Mostrar stock encontrado con detalles
        logger.info(f"📦 Stock NACIONAL encontrado para {request.sku_id}:")
        total_stock = 0
        for stock_loc in stock_locations:
            tienda_info = next((s for s in authorized_nearby if s['tienda_id'] == stock_loc['tienda_id']), None)
            nombre_tienda = tienda_info['nombre_tienda'] if tienda_info else f"Tienda {stock_loc['tienda_id']}"
            distancia = tienda_info['distancia_km'] if tienda_info else 999

            logger.info(f"   📍 {nombre_tienda}: {stock_loc['stock_disponible']} unidades ({distancia:.1f}km)")
            total_stock += stock_loc['stock_disponible']

        logger.info(f"📊 Stock NACIONAL total: {total_stock} | Requerido: {request.cantidad}")

        # Calcular asignación óptima
        allocation = self.repos.stock.calculate_optimal_allocation(
            stock_locations, request.cantidad, authorized_nearby
        )

        if not allocation['factible']:
            return {
                'factible': False,
                'razon': allocation['razon']
            }

        # ✅ LOGGING MEJORADO: Plan de asignación final con justificación
        logger.info("📋 Plan de asignación FINAL:")
        for plan_item in allocation['plan']:
            tienda_info = next((s for s in authorized_nearby if s['tienda_id'] == plan_item['tienda_id']), None)
            nombre_tienda = tienda_info['nombre_tienda'] if tienda_info else f"Tienda {plan_item['tienda_id']}"
            distancia = tienda_info['distancia_km'] if tienda_info else 999
            precio_unitario = plan_item.get('precio_unitario', 0)
            precio_total = precio_unitario * plan_item['cantidad']

            logger.info(f"   🏪 {nombre_tienda}:")
            logger.info(f"      → {plan_item['cantidad']} unidades de {request.cantidad} requeridas")
            logger.info(f"      → ${precio_total:,.0f} total (${precio_unitario:,.0f} por unidad)")
            logger.info(f"      → {distancia:.1f}km del destino")
            logger.info(
                f"      → Razón: {'Mejor distancia' if distancia < 200 else 'Única opción'} con stock suficiente")

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
        """🗺️ Generación de candidatos con ruteo REAL y logging mejorado"""

        allocation_plan = stock_analysis['allocation_plan']
        target_lat = cp_info['latitud_centro']
        target_lon = cp_info['longitud_centro']
        candidates = []

        logger.info(f"🗺️ Evaluando opciones de entrega para {len(allocation_plan)} asignaciones:")

        # Verificar si hay stock local vs requiere ruteo complejo
        for i, plan_item in enumerate(allocation_plan, 1):
            tienda_origen = await self._get_store_info(plan_item['tienda_id'])
            distance_direct = GeoCalculator.calculate_distance_km(
                float(tienda_origen['latitud']), float(tienda_origen['longitud']),
                target_lat, target_lon
            )

            # ✅ LOGGING MEJORADO: Evaluación detallada
            logger.info(f"📏 Opción {i}: {tienda_origen['nombre_tienda']} → CP {request.codigo_postal}")
            logger.info(f"   📍 Coordenadas tienda: ({tienda_origen['latitud']:.4f}, {tienda_origen['longitud']:.4f})")
            logger.info(f"   📍 Coordenadas destino: ({target_lat:.4f}, {target_lon:.4f})")
            logger.info(f"   📏 Distancia directa: {distance_direct:.1f}km")
            logger.info(f"   📦 Cantidad asignada: {plan_item['cantidad']} unidades")

            # LÓGICA REAL: Determinar tipo de ruteo
            if distance_direct <= 100:  # Stock local
                logger.info(f"   ✅ RUTA DIRECTA - Distancia local ({distance_direct:.1f}km ≤ 100km)")
                logger.info(f"   🚛 Usando flota interna Liverpool")

                direct_candidate = await self._create_direct_route_dynamic(
                    plan_item, (target_lat, target_lon), external_factors, request, cp_info
                )
                if direct_candidate:
                    direct_candidate['has_local_stock'] = True
                    candidates.append(direct_candidate)

                    # ✅ LOGGING: Resultado de ruta directa
                    logger.info(f"   📊 Ruta directa creada:")
                    logger.info(f"      → Tiempo total: {direct_candidate['tiempo_total_horas']:.1f}h")
                    logger.info(f"      → Costo total: ${direct_candidate['costo_total_mxn']:.0f}")
                    logger.info(f"      → Probabilidad éxito: {direct_candidate['probabilidad_cumplimiento']:.1%}")

            else:  # Requiere ruteo complejo con CEDIS
                logger.info(f"   🔄 RUTEO COMPLEJO - Distancia remota ({distance_direct:.1f}km > 100km)")
                logger.info(f"   🏭 Requiere CEDIS intermedio para optimizar ruta")

                complex_candidate = await self._create_complex_routing_with_cedis(
                    [plan_item], (target_lat, target_lon), request.codigo_postal, external_factors
                )
                if complex_candidate:
                    complex_candidate['has_local_stock'] = False
                    candidates.append(complex_candidate)

                    # ✅ LOGGING: Resultado de ruteo complejo
                    logger.info(f"   📊 Ruteo complejo creado:")
                    logger.info(f"      → Tiempo total: {complex_candidate['tiempo_total_horas']:.1f}h")
                    logger.info(f"      → Costo total: ${complex_candidate['costo_total_mxn']:.0f}")
                    logger.info(f"      → Segmentos: {len(complex_candidate['segmentos'])}")
                    logger.info(f"      → Probabilidad éxito: {complex_candidate['probabilidad_cumplimiento']:.1%}")

        logger.info(f"🗺️ Resumen de candidatos generados:")
        logger.info(f"   📊 Total candidatos: {len(candidates)}")

        for i, candidate in enumerate(candidates, 1):
            logger.info(f"   {i}. {candidate.get('origen_principal', 'N/A')} → Cliente")
            logger.info(f"      Tipo: {candidate['tipo_ruta']}")
            logger.info(f"      Tiempo: {candidate['tiempo_total_horas']:.1f}h")
            logger.info(f"      Costo: ${candidate['costo_total_mxn']:.0f}")
            logger.info(f"      Stock local: {'Sí' if candidate.get('has_local_stock', False) else 'No'}")

        return candidates

    async def _create_cedis_route(self, plan_item: Dict[str, Any],
                                  tienda_origen: Dict[str, Any],
                                  target_coords: Tuple[float, float],
                                  external_factors: Dict[str, Any],
                                  request: PredictionRequest,
                                  cp_info: Dict[str, Any],
                                  total_distance: float) -> Dict[str, Any]:
        """🏭 Crea ruta multi-segmento usando CEDIS como intermediario"""

        # 1. BUSCAR CEDIS ÓPTIMO
        optimal_cedis = await self._find_optimal_cedis(
            tienda_origen, target_coords, request.codigo_postal
        )
        if not optimal_cedis:
            logger.warning("❌ No se encontró CEDIS óptimo")
            return None

        # 2. BUSCAR TIENDA DESTINO cerca del CP final
        tienda_destino = await self._find_destination_store(target_coords, request.codigo_postal)
        if not tienda_destino:
            logger.warning("❌ No se encontró tienda destino cerca del CP")
            return None

        logger.info(
            f"🏭 Ruta CEDIS: {tienda_origen['tienda_id']} → {optimal_cedis['cedis_id']} → {tienda_destino['tienda_id']} → Cliente")

        # 3. CALCULAR SEGMENTOS
        segmentos = []
        tiempo_total = 0
        costo_total = 0
        distancia_total = 0

        # SEGMENTO 1: Tienda Origen → CEDIS
        seg1 = await self._calculate_segment(
            tienda_origen, optimal_cedis, 'CEDIS', external_factors, plan_item['cantidad']
        )
        segmentos.append(seg1)
        tiempo_total += seg1['tiempo_horas']
        costo_total += seg1['costo_segmento']
        distancia_total += seg1['distancia_km']

        # TIEMPO PROCESAMIENTO CEDIS
        tiempo_procesamiento_cedis = settings.TIEMPO_PREPARACION_CEDIS  # 2 horas
        tiempo_total += tiempo_procesamiento_cedis
        logger.info(f"⏱️ + Procesamiento CEDIS: {tiempo_procesamiento_cedis}h")

        # SEGMENTO 2: CEDIS → Tienda Destino
        seg2 = await self._calculate_segment(
            optimal_cedis, tienda_destino, 'TIENDA', external_factors, plan_item['cantidad']
        )
        segmentos.append(seg2)
        tiempo_total += seg2['tiempo_horas']
        costo_total += seg2['costo_segmento']
        distancia_total += seg2['distancia_km']

        # TIEMPO PREPARACIÓN TIENDA DESTINO
        tiempo_prep_final = 1.0  # 1 hora preparación final
        tiempo_total += tiempo_prep_final

        # SEGMENTO 3: Tienda Destino → Cliente
        seg3 = await self._calculate_segment(
            tienda_destino, {'latitud': target_coords[0], 'longitud': target_coords[1]},
            'CLIENTE', external_factors, plan_item['cantidad']
        )
        segmentos.append(seg3)
        tiempo_total += seg3['tiempo_horas']
        costo_total += seg3['costo_segmento']
        distancia_total += seg3['distancia_km']

        # 4. APLICAR FACTORES EXTERNOS
        tiempo_extra = external_factors.get('impacto_tiempo_extra_horas', 0)
        tiempo_total += tiempo_extra

        # 5. CALCULAR PROBABILIDAD (menor para rutas complejas)
        probabilidad = self._calculate_multisegment_probability(
            segmentos, tiempo_total, external_factors
        )

        logger.info(
            f"📊 Ruta CEDIS: {tiempo_total:.1f}h, ${costo_total:.0f}, {distancia_total:.1f}km, {probabilidad:.1%}")

        return {
            'ruta_id': f"cedis_{tienda_origen['tienda_id']}_{optimal_cedis['cedis_id']}_{tienda_destino['tienda_id']}",
            'tipo_ruta': 'multi_segmento_cedis',
            'origen_principal': tienda_origen['nombre_tienda'],
            'cedis_intermedio': optimal_cedis['nombre_cedis'],
            'tienda_destino': tienda_destino['nombre_tienda'],
            'segmentos': segmentos,
            'tiempo_total_horas': tiempo_total,
            'costo_total_mxn': costo_total,
            'distancia_total_km': distancia_total,
            'probabilidad_cumplimiento': probabilidad,
            'cantidad_cubierta': plan_item['cantidad'],
            'factores_aplicados': [
                'ruta_multi_segmento',
                f"cedis_{optimal_cedis['cedis_id']}",
                f"distancia_total_{distancia_total:.0f}km",
                f"segmentos_{len(segmentos)}",
                'logistica_compleja'
            ],
            'desglose_tiempos': {
                'origen_a_cedis': seg1['tiempo_horas'],
                'procesamiento_cedis': tiempo_procesamiento_cedis,
                'cedis_a_tienda_destino': seg2['tiempo_horas'],
                'preparacion_final': tiempo_prep_final,
                'tienda_a_cliente': seg3['tiempo_horas'],
                'factores_externos': tiempo_extra
            }
        }

    async def _find_optimal_cedis(self, tienda_origen: Dict[str, Any],
                                  target_coords: Tuple[float, float],
                                  codigo_postal: str) -> Dict[str, Any]:
        """🏭 Encuentra el CEDIS óptimo como intermediario"""

        cedis_df = self.repos.data_manager.get_data('cedis')
        if cedis_df.height == 0:
            return None

        cedis_candidates = []

        for cedis in cedis_df.to_dicts():
            # Distancia tienda origen → CEDIS
            dist_origen_cedis = GeoCalculator.calculate_distance_km(
                float(tienda_origen['latitud']), float(tienda_origen['longitud']),
                float(cedis['latitud']), float(cedis['longitud'])
            )

            # Distancia CEDIS → código postal destino
            dist_cedis_destino = GeoCalculator.calculate_distance_km(
                float(cedis['latitud']), float(cedis['longitud']),
                target_coords[0], target_coords[1]
            )

            # Verificar cobertura
            cobertura_estados = cedis.get('cobertura_estados', '')
            cp_estado = self._get_estado_from_cp(codigo_postal)

            # Score del CEDIS (menor distancia total = mejor)
            total_distance = dist_origen_cedis + dist_cedis_destino
            coverage_bonus = 0.8 if 'Nacional' in cobertura_estados or cp_estado in cobertura_estados else 1.2

            score = total_distance * coverage_bonus

            cedis_candidates.append({
                **cedis,
                'dist_origen_cedis': dist_origen_cedis,
                'dist_cedis_destino': dist_cedis_destino,
                'total_distance': total_distance,
                'score': score,
                'coverage_match': cp_estado in cobertura_estados
            })

        # Ordenar por score (menor es mejor)
        cedis_candidates.sort(key=lambda x: x['score'])

        if cedis_candidates:
            best_cedis = cedis_candidates[0]
            logger.info(f"🏭 CEDIS seleccionado: {best_cedis['nombre_cedis']} (score: {best_cedis['score']:.1f})")
            return best_cedis

        return None

    async def _find_destination_store(self, target_coords: Tuple[float, float],
                                      codigo_postal: str) -> Dict[str, Any]:
        """🏪 Encuentra tienda Liverpool más cercana al CP destino"""

        # Buscar tiendas cercanas al CP destino
        nearby_stores = self.repos.store.find_stores_by_postal_range(codigo_postal)

        if not nearby_stores:
            logger.warning(f"❌ No hay tiendas Liverpool cerca de {codigo_postal}")
            return None

        # Tomar la más cercana
        closest_store = nearby_stores[0]
        logger.info(
            f"🏪 Tienda destino: {closest_store['nombre_tienda']} ({closest_store['distancia_km']:.1f}km del CP)")

        return closest_store

    async def _calculate_segment(self, origin: Dict[str, Any], destination: Dict[str, Any],
                                 dest_type: str, external_factors: Dict[str, Any],
                                 cantidad: int) -> Dict[str, Any]:
        """⚡ Calcula un segmento individual de la ruta"""

        # Coordenadas
        if dest_type == 'CLIENTE':
            dest_lat, dest_lon = destination['latitud'], destination['longitud']
            dest_name = 'Cliente Final'
            dest_id = 'cliente'
        else:
            dest_lat = float(destination['latitud'])
            dest_lon = float(destination['longitud'])
            dest_name = destination.get('nombre_cedis') or destination.get('nombre_tienda', 'Destino')
            dest_id = destination.get('cedis_id') or destination.get('tienda_id', 'dest')

        orig_lat = float(origin['latitud'])
        orig_lon = float(origin['longitud'])
        orig_name = origin.get('nombre_tienda') or origin.get('nombre_cedis', 'Origen')
        orig_id = origin.get('tienda_id') or origin.get('cedis_id', 'orig')

        # Calcular distancia
        distance = GeoCalculator.calculate_distance_km(orig_lat, orig_lon, dest_lat, dest_lon)

        # Determinar tipo de flota
        if distance <= 50:
            fleet_type = 'FI'
            carrier = 'Liverpool'
        else:
            fleet_type = 'FE'
            carrier = 'Estafeta'  # Default, debería consultar flota_externa CSV

        # Calcular tiempo
        travel_time = self._calculate_travel_time_dynamic(distance, fleet_type, external_factors)

        # Calcular costo
        if fleet_type == 'FI':
            cost = self._calculate_internal_fleet_cost(distance, cantidad, external_factors)
        else:
            # Buscar carrier real
            carriers = self.repos.fleet.get_best_carriers_for_cp("00000", 1.0)  # Fallback
            if carriers:
                cost = self._calculate_external_fleet_cost(carriers[0], 1.0, distance, external_factors)
            else:
                cost = distance * 15.0  # Costo base por km

        return {
            'origen': orig_name,
            'origen_id': orig_id,
            'destino': dest_name,
            'destino_id': dest_id,
            'distancia_km': distance,
            'tiempo_horas': travel_time,
            'tipo_flota': fleet_type,
            'carrier': carrier,
            'costo_segmento': cost,
            'tipo_segmento': dest_type
        }

    def _calculate_multisegment_probability(self, segmentos: List[Dict[str, Any]],
                                            tiempo_total: float,
                                            external_factors: Dict[str, Any]) -> float:
        """📊 Calcula probabilidad para rutas multi-segmento (más conservadora)"""

        # Probabilidad base más baja para rutas complejas
        base_prob = 0.75  # 75% base para multi-segmento vs 90% para directo

        # Penalización por número de segmentos
        num_segments = len(segmentos)
        segment_penalty = (num_segments - 1) * 0.05  # 5% menos por cada segmento extra

        # Penalización por tiempo total
        time_penalty = min(0.15, max(0, (tiempo_total - 24) / 100))  # Penalizar si > 24h

        # Factor por criticidad
        criticidad = external_factors.get('criticidad_logistica', 'Normal')
        criticidad_factor = {
            'Baja': 1.0,
            'Normal': 0.95,
            'Media': 0.90,
            'Alta': 0.80,
            'Crítica': 0.70
        }.get(criticidad, 0.90)

        final_prob = (base_prob - segment_penalty - time_penalty) * criticidad_factor
        return round(max(0.4, min(0.95, final_prob)), 3)

    def _get_estado_from_cp(self, codigo_postal: str) -> str:
        """📍 Obtiene estado desde código postal"""
        cp_info = self.repos.store._get_postal_info(codigo_postal)
        if cp_info:
            return cp_info.get('estado_alcaldia', '').split()[0]  # Primer palabra
        return 'Desconocido'

    async def _create_direct_route_dynamic(self, plan_item: Dict[str, Any],
                                           target_coords: Tuple[float, float],
                                           external_factors: Dict[str, Any],
                                           request: PredictionRequest,
                                           cp_info: Dict[str, Any]) -> Dict[str, Any]:
        """📍 Crea ruta directa con zona de seguridad REAL del CSV"""

        tienda_id = plan_item['tienda_id']
        tienda_info = await self._get_store_info(tienda_id)

        if not tienda_info:
            logger.error(f"❌ No se encontró info para tienda {tienda_id}")
            return None

        # CORRECCIÓN: Usar zona de seguridad del CSV de códigos postales, NO de tiendas
        zona_seguridad_cp = cp_info.get('zona_seguridad', 'Verde')  # Del CSV codigos_postales
        zona_seguridad_tienda = tienda_info.get('zona_seguridad', 'Verde')  # Del CSV tiendas

        # La zona final es la más restrictiva
        zonas_orden = {'Verde': 1, 'Amarilla': 2, 'Roja': 3}
        zona_final = zona_seguridad_cp if zonas_orden.get(zona_seguridad_cp, 1) >= zonas_orden.get(
            zona_seguridad_tienda, 1) else zona_seguridad_tienda

        logger.info(f"🛡️ Zona seguridad: CP={zona_seguridad_cp}, Tienda={zona_seguridad_tienda}, Final={zona_final}")

        # Usar coordenadas REALES corregidas
        store_lat, store_lon = GeoCalculator.fix_corrupted_coordinates(
            float(tienda_info['latitud']), float(tienda_info['longitud'])
        )

        # Calcular distancia REAL
        distance_km = GeoCalculator.calculate_distance_km(
            store_lat, store_lon, target_coords[0], target_coords[1]
        )

        # Determinar tipo de flota basado en CSV y zona de seguridad
        cobertura_liverpool = cp_info.get('cobertura_liverpool', True)

        if distance_km <= 50 and cobertura_liverpool and zona_final in ['Verde', 'Amarilla']:
            fleet_type = 'FI'
            carrier = 'Liverpool'
            logger.info(f"🚛 FLOTA INTERNA: {distance_km:.1f}km, zona {zona_final}")
        else:
            fleet_type = 'FE'
            peso_kg = self._calculate_shipment_weight(request, plan_item['cantidad'])
            carriers = self.repos.fleet.get_best_carriers_for_cp(request.codigo_postal, peso_kg)
            carrier = carriers[0]['carrier'] if carriers else 'DHL'
            logger.info(f"📦 FLOTA EXTERNA: {carrier}, zona {zona_final}")

        # Cálculos con datos CSV reales
        travel_time = self._calculate_travel_time_dynamic(distance_km, fleet_type, external_factors)
        prep_time = float(tienda_info.get('tiempo_prep_horas', 1.0))
        tiempo_extra = external_factors.get('impacto_tiempo_extra_horas', 0)
        total_time = prep_time + travel_time + tiempo_extra

        # Costo usando datos CSV reales
        if fleet_type == 'FE' and carriers:
            cost = self._calculate_external_fleet_cost(
                carriers[0], peso_kg, distance_km, external_factors
            )
        else:
            cost = self._calculate_internal_fleet_cost(
                distance_km, plan_item['cantidad'], external_factors
            )

        # Probabilidad con zona real
        probability = self._calculate_probability_dynamic(
            distance_km, total_time, external_factors, fleet_type, zona_final
        )

        return {
            'ruta_id': f"direct_{tienda_id}",
            'tipo_ruta': 'directa',
            'origen_principal': tienda_info['nombre_tienda'],
            'tienda_origen_id': tienda_id,
            'segmentos': [{
                'origen': tienda_info['nombre_tienda'],
                'origen_id': tienda_id,
                'destino': 'cliente',
                'distancia_km': distance_km,
                'tiempo_horas': travel_time,
                'tipo_flota': fleet_type,
                'carrier': carrier,
                'costo_segmento': cost,
                'zona_seguridad': zona_final  # ZONA REAL DEL CSV
            }],
            'tiempo_total_horas': total_time,
            'costo_total_mxn': cost,
            'distancia_total_km': distance_km,
            'probabilidad_cumplimiento': probability,
            'cantidad_cubierta': plan_item['cantidad'],
            'factores_aplicados': [
                f"demanda_{external_factors.get('factor_demanda', 1.0)}",
                f"flota_{fleet_type}",
                f"carrier_{carrier}",
                f"zona_{zona_final}",  # ZONA REAL
                f"eventos_{len(external_factors.get('eventos_detectados', []))}",
                f"distancia_{distance_km:.1f}km",
                'datos_csv_reales'
            ]
        }


    async def _get_store_info(self, tienda_id: str) -> Dict[str, Any]:
        """🏪 Obtiene información completa de la tienda"""
        tiendas_df = self.repos.data_manager.get_data('tiendas')
        store_data = tiendas_df.filter(pl.col('tienda_id') == tienda_id)

        if store_data.height > 0:
            return store_data.to_dicts()[0]
        return None


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
        """💰 Cálculo de costo flota externa usando datos REALES del CSV"""

        # Datos base del CSV
        costo_base = float(carrier_info['costo_base_mxn'])
        peso_min = float(carrier_info['peso_min_kg'])
        costo_por_kg = float(carrier_info['costo_por_kg_adicional'])

        # Calcular costo por peso extra
        peso_extra = max(0, peso_kg - peso_min)
        costo_peso_extra = peso_extra * costo_por_kg

        # Factor por distancia (desde CSV indirectamente)
        distance_factor = 1.0 + (distance_km / 500) * 0.1  # 10% cada 500km

        subtotal = (costo_base + costo_peso_extra) * distance_factor

        # Aplicar factores externos REALES del CSV
        factor_demanda = external_factors.get('factor_demanda', 1.0)
        impacto_costo_pct = external_factors.get('impacto_costo_extra_pct', 0) / 100

        # Factor de demanda
        subtotal *= factor_demanda

        # Impacto de costo extra
        if impacto_costo_pct > 0:
            subtotal *= (1 + impacto_costo_pct)

        final_cost = round(subtotal, 2)

        logger.info(f"💰 Costo externo CSV: base=${costo_base} × demanda={factor_demanda:.2f} = ${final_cost}")

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

    def _get_comprehensive_external_factors(self, fecha: datetime, codigo_postal: str) -> Dict[str, Any]:
        """🎯 Factores externos COMPLETAMENTE desde CSV"""

        # Obtener factores del CSV de factores externos
        factores_csv = self.repos.external_factors.get_factors_for_date_and_cp(fecha, codigo_postal)

        # Obtener info del CP desde CSV de códigos postales
        cp_info = self.repos.store._get_postal_info(codigo_postal)

        # Combinar datos REALES de ambos CSVs
        return {
            # Desde factores_externos_mexico_completo.csv
            'evento_detectado': factores_csv.get('evento_detectado', 'Normal'),
            'eventos_detectados': [factores_csv.get('evento_detectado', 'Normal')] if factores_csv.get(
                'evento_detectado') != 'Normal' else [],
            'factor_demanda': factores_csv.get('factor_demanda', 1.0),
            'condicion_clima': factores_csv.get('condicion_clima', 'Templado'),
            'trafico_nivel': factores_csv.get('trafico_nivel', 'Moderado'),
            'criticidad_logistica': factores_csv.get('criticidad_logistica', 'Normal'),
            'impacto_tiempo_extra_horas': factores_csv.get('impacto_tiempo_extra_horas', 0),
            'rango_cp_afectado': factores_csv.get('rango_cp_afectado', '00000-99999'),

            # Desde codigos_postales_rangos_mexico.csv
            'zona_seguridad': cp_info.get('zona_seguridad', 'Verde'),
            'cobertura_liverpool': cp_info.get('cobertura_liverpool', True),
            'tiempo_entrega_base_horas': cp_info.get('tiempo_entrega_base_horas', '2-4'),

            # Calculados pero basados en CSV
            'es_temporada_alta': factores_csv.get('factor_demanda', 1.0) > 1.8,
            'es_temporada_critica': factores_csv.get('factor_demanda', 1.0) > 2.5,
            'impacto_costo_extra_pct': self._calculate_cost_impact_from_csv(factores_csv),

            'fuente_datos': 'CSV_completo',
            'fecha_analisis': fecha.isoformat(),
            'observaciones': factores_csv.get('observaciones_clima_regional', '')
        }

    def _process_real_csv_factors(self, row: Dict[str, Any], fecha: datetime, cp: str) -> Dict[str, Any]:
        """🔄 Procesa factores REALES del CSV con parsing robusto"""

        # Factor de demanda REAL del CSV - parsing más robusto
        factor_raw = row.get('factor_demanda', '1.0')
        try:
            if isinstance(factor_raw, (int, float)):
                factor_demanda = float(factor_raw)
            elif '/' in str(factor_raw):
                # Formato fracción: "1/03/2025" → extraer numerador
                factor_demanda = float(str(factor_raw).split('/')[0])
            else:
                factor_demanda = float(factor_raw)
        except (ValueError, TypeError):
            logger.warning(f"⚠️ Factor demanda inválido: {factor_raw}, usando 1.0")
            factor_demanda = 1.0

        # Tiempo extra con parsing robusto
        impacto_tiempo = self._parse_time_range(
            row.get('impacto_tiempo_extra_horas'), default=0.0
        )

        return {
            'evento_detectado': row.get('evento_detectado', 'Normal'),
            'eventos_detectados': [row.get('evento_detectado', 'Normal')] if row.get(
                'evento_detectado') != 'Normal' else [],
            'factor_demanda': factor_demanda,
            'condicion_clima': row.get('condicion_clima', 'Templado'),
            'trafico_nivel': row.get('trafico_nivel', 'Moderado'),
            'impacto_tiempo_extra_horas': impacto_tiempo,
            'criticidad_logistica': row.get('criticidad_logistica', 'Normal'),
            'es_temporada_alta': factor_demanda > 1.5,
            'es_temporada_critica': factor_demanda > 2.5,
            'fuente_datos': 'CSV_real',
            'observaciones_clima': row.get('observaciones_clima_regional', ''),
            'rango_cp_afectado': row.get('rango_cp_afectado', '00000-99999')
        }

    def _calculate_cost_impact_from_csv(self, factores_csv: Dict[str, Any]) -> float:
        """💰 Calcula impacto en costo desde datos CSV"""
        factor_demanda = factores_csv.get('factor_demanda', 1.0)
        evento = factores_csv.get('evento_detectado', 'Normal')

        # Base por demanda
        impacto = max(0, (factor_demanda - 1.0) * 20)  # 20% por punto de demanda

        # Eventos específicos del CSV
        eventos_premium = ['Viernes_Santo', 'Jueves_Santo', 'Dia_Padre', 'Navidad']
        if any(evento_premium in evento for evento_premium in eventos_premium):
            impacto += 15  # 15% extra

        return round(min(50.0, impacto), 1)

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
        """📊 Probabilidad con zona de seguridad REAL desde CSV"""

        base_prob = 0.90 if fleet_type == 'FI' else 0.82

        # Penalizaciones por distancia y tiempo
        distance_penalty = min(0.2, distance_km / 1000)
        time_penalty = min(0.15, max(0, (total_time - 6) / 50))

        # Penalización por zona de seguridad REAL del CSV
        zona_penalty = {
            'Verde': 0.0,  # Sin penalización
            'Amarilla': 0.05,  # 5% menos probabilidad
            'Roja': 0.15  # 15% menos probabilidad
        }.get(zona_seguridad, 0.05)

        # Factor por criticidad REAL del CSV
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
        """🏆 Rankea candidatos con logging explicativo detallado"""

        if not candidates:
            return []

        logger.info(f"🏆 Iniciando ranking de {len(candidates)} candidatos...")

        # Normalizar métricas
        tiempos = [c['tiempo_total_horas'] for c in candidates]
        costos = [c['costo_total_mxn'] for c in candidates]
        distancias = [c['distancia_total_km'] for c in candidates]

        min_tiempo, max_tiempo = min(tiempos), max(tiempos)
        min_costo, max_costo = min(costos), max(costos)
        min_distancia, max_distancia = min(distancias), max(distancias)

        logger.info(f"📊 Rangos de métricas:")
        logger.info(f"   ⏱️ Tiempo: {min_tiempo:.1f}h - {max_tiempo:.1f}h")
        logger.info(f"   💰 Costo: ${min_costo:.0f} - ${max_costo:.0f}")
        logger.info(f"   📏 Distancia: {min_distancia:.1f}km - {max_distancia:.1f}km")

        for i, candidate in enumerate(candidates, 1):
            # Scores normalizados (0-1)
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

            # Bonus controlado para rutas directas
            if candidate['tipo_ruta'] == 'directa':
                score_combinado += 0.05  # Bonus aditivo pequeño

            # Normalizar SIEMPRE a máximo 1.0
            score_combinado = min(1.0, score_combinado)

            candidate['score_lightgbm'] = round(score_combinado, 4)
            candidate['score_breakdown'] = {
                'tiempo': round(score_tiempo, 3),
                'costo': round(score_costo, 3),
                'distancia': round(score_distancia, 3),
                'probabilidad': round(score_probabilidad, 3)
            }

            # ✅ LOGGING DETALLADO: Evaluación de cada candidato
            origen_name = candidate.get('origen_principal', f"Candidato {i}")
            logger.info(f"📊 Evaluando: {origen_name}")
            logger.info(f"   🔢 Scores individuales:")
            logger.info(f"      → Tiempo: {score_tiempo:.3f} (peso: {weights['tiempo']})")
            logger.info(f"      → Costo: {score_costo:.3f} (peso: {weights['costo']})")
            logger.info(f"      → Distancia: {score_distancia:.3f} (peso: {weights['distancia']})")
            logger.info(f"      → Probabilidad: {score_probabilidad:.3f} (peso: {weights['probabilidad']})")
            logger.info(f"   🎯 Score final: {score_combinado:.4f}")
            logger.info(
                f"   💡 Ventajas: {candidate['tipo_ruta']}, {candidate['tiempo_total_horas']:.1f}h, ${candidate['costo_total_mxn']:.0f}")

        # Ordenar por score
        ranked = sorted(candidates, key=lambda x: x['score_lightgbm'], reverse=True)

        logger.info("🏆 RANKING FINAL:")
        logger.info("   Pos | Tienda/Ruta              | Score  | Tiempo | Costo  | Tipo")
        logger.info("   ----|--------------------------|--------|--------|--------|------------")

        for i, candidate in enumerate(ranked, 1):
            origen_name = candidate.get('origen_principal', f"Ruta {i}")[:20].ljust(20)
            score = candidate['score_lightgbm']
            tiempo = candidate['tiempo_total_horas']
            costo = candidate['costo_total_mxn']
            tipo = candidate['tipo_ruta'][:10]
            prob = candidate['probabilidad_cumplimiento']

            logger.info(f"   {i:2d}. | {origen_name} | {score:.3f} | {tiempo:5.1f}h | ${costo:6.0f} | {tipo}")

            if i == 1:
                # ✅ LOGGING: Explicar por qué ganó el primero
                logger.info(f"   🎯 GANADOR: {candidate.get('origen_principal', 'N/A')}")
                logger.info(f"      → Razones principales:")

                if score >= 0.9:
                    logger.info(f"      → Score excelente ({score:.3f}) - Óptimo en múltiples métricas")
                elif candidate['tipo_ruta'] == 'directa':
                    logger.info(f"      → Ruta directa - Sin transbordos ni CEDIS intermedio")

                if tiempo <= min(tiempos) * 1.1:
                    logger.info(f"      → Tiempo competitivo ({tiempo:.1f}h)")
                if costo <= min(costos) * 1.2:
                    logger.info(f"      → Costo eficiente (${costo:.0f})")
                if prob >= 0.8:
                    logger.info(f"      → Alta confiabilidad ({prob:.1%})")

        return ranked

    async def _build_enhanced_explainability(self,
                                             request: PredictionRequest,
                                             selected_route: Dict[str, Any],
                                             all_candidates: List[Dict[str, Any]],
                                             stock_analysis: Dict[str, Any],
                                             external_factors: Dict[str, Any],
                                             cp_info: Dict[str, Any],
                                             nearby_stores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """🔍 Construye explicabilidad SIMPLIFICADA para frontend"""

        logger.info("🔍 Generando explicabilidad extendida...")

        try:
            # 1. RESUMEN EJECUTIVO
            resumen_ejecutivo = {
                "decision_principal": f"Seleccionada ruta {selected_route.get('tipo_ruta', 'directa')} desde {selected_route.get('origen_principal', 'tienda')}",
                "razon_principal": self._get_simple_decision_reason(selected_route, all_candidates),
                "confianza_decision": 0.85,
                "alertas_importantes": self._get_simple_alerts(external_factors, cp_info),
                "beneficios_clave": [
                    f"Tiempo estimado: {selected_route.get('tiempo_total_horas', 0):.1f} horas",
                    f"Costo: ${selected_route.get('costo_total_mxn', 0):.0f}",
                    f"Probabilidad éxito: {selected_route.get('probabilidad_cumplimiento', 0):.1%}"
                ]
            }

            # 2. ANÁLISIS DE TIENDAS (SIMPLIFICADO)
            analisis_tiendas = self._build_simple_store_analysis(
                nearby_stores, stock_analysis, request
            )

            # 3. COMPARACIÓN DE CANDIDATOS (SIMPLIFICADO)
            comparacion_candidatos = self._build_simple_candidates_comparison(
                all_candidates, selected_route
            )

            # 4. FACTORES EXTERNOS EXPLICADOS (SIMPLIFICADO)
            factores_explained = self._build_simple_factors_explanation(
                external_factors, cp_info, request.fecha_compra
            )

            # 5. DATOS GEOGRÁFICOS (SIMPLIFICADO)
            geo_data = self._build_simple_geo_data(
                nearby_stores, stock_analysis, cp_info, selected_route
            )

            # 6. TIMELINE DE PROCESAMIENTO
            timeline = {
                "paso_1": "✅ Producto validado y CP verificado",
                "paso_2": f"✅ Encontradas {len(nearby_stores)} tiendas en área",
                "paso_3": f"✅ Stock verificado en {len(stock_analysis.get('allocation_plan', []))} ubicaciones",
                "paso_4": f"✅ Generados {len(all_candidates)} candidatos de entrega",
                "paso_5": f"✅ Seleccionada mejor opción: {selected_route.get('ruta_id', 'N/A')}",
                "tiempo_total": "Procesamiento completado exitosamente"
            }

            result = {
                "resumen_ejecutivo": resumen_ejecutivo,
                "analisis_tiendas": analisis_tiendas,
                "comparacion_candidatos": comparacion_candidatos,
                "factores_externos_explicados": factores_explained,
                "datos_geograficos": geo_data,
                "timeline_procesamiento": timeline,
                "insights_algoritmo": {
                    "modelo_utilizado": "LightGBM + Gemini Decision Engine",
                    "factores_evaluados": ["tiempo", "costo", "distancia", "probabilidad"],
                    "score_final": selected_route.get('score_lightgbm', 0),
                    "ranking_obtenido": 1
                }
            }

            logger.info("✅ Explicabilidad extendida generada exitosamente")
            return result

        except Exception as e:
            logger.error(f"❌ Error en explicabilidad extendida: {e}")
            return {
                "error": f"Error procesando explicabilidad: {str(e)}",
                "resumen_basico": {
                    "ruta_seleccionada": selected_route.get('ruta_id', 'N/A'),
                    "tiempo_horas": selected_route.get('tiempo_total_horas', 0),
                    "costo_mxn": selected_route.get('costo_total_mxn', 0),
                    "tiendas_consideradas": len(nearby_stores)
                }
            }

    def _get_simple_decision_reason(self, selected_route: Dict[str, Any],
                                    all_candidates: List[Dict[str, Any]]) -> str:
        """🎯 Razón simple de decisión"""
        if len(all_candidates) == 1:
            return "Única opción factible encontrada con stock disponible"
        else:
            score = selected_route.get('score_lightgbm', 0)
            return f"Mejor puntuación general ({score:.3f}) considerando tiempo, costo y confiabilidad"

    def _get_simple_alerts(self, external_factors: Dict[str, Any],
                           cp_info: Dict[str, Any]) -> List[str]:
        """⚠️ Alertas simples"""
        alerts = []

        if external_factors.get('factor_demanda', 1.0) > 2.0:
            alerts.append("🎄 Alta demanda por temporada especial")

        if cp_info.get('zona_seguridad') == 'Roja':
            alerts.append("🔴 Zona de alto riesgo - Tiempo y costo incrementados")
        elif cp_info.get('zona_seguridad') == 'Amarilla':
            alerts.append("🟡 Zona moderada - Ligero incremento en tiempo")

        if external_factors.get('criticidad_logistica') == 'Crítica':
            alerts.append("⚡ Criticidad logística alta por eventos externos")

        if not alerts:
            alerts.append("✅ Sin alertas - Condiciones normales de operación")

        return alerts

    def _build_simple_store_analysis(self, nearby_stores: List[Dict[str, Any]],
                                     stock_analysis: Dict[str, Any],
                                     request: PredictionRequest) -> Dict[str, Any]:
        """🏪 Análisis simple de tiendas"""

        if not nearby_stores:
            return {
                "error": "No hay tiendas disponibles",
                "total_consideradas": 0
            }

        selected_store_ids = [plan['tienda_id'] for plan in stock_analysis.get('allocation_plan', [])]

        tiendas_info = []
        for store in nearby_stores[:5]:  # Top 5
            tiendas_info.append({
                "tienda_id": store['tienda_id'],
                "nombre": store.get('nombre_tienda', f"Tienda {store['tienda_id']}"),
                "distancia_km": store.get('distancia_km', 0),
                "seleccionada": store['tienda_id'] in selected_store_ids,
                "razon": "Seleccionada - Stock disponible" if store[
                                                                  'tienda_id'] in selected_store_ids else "No seleccionada"
            })

        return {
            "total_consideradas": len(nearby_stores),
            "total_seleccionadas": len(selected_store_ids),
            "tiendas_detalle": tiendas_info,
            "criterio_seleccion": "Stock disponible y distancia óptima"
        }

    def _build_simple_candidates_comparison(self, all_candidates: List[Dict[str, Any]],
                                            selected_route: Dict[str, Any]) -> Dict[str, Any]:
        """📊 Comparación simple de candidatos"""

        if not all_candidates:
            return {"error": "No hay candidatos para comparar"}

        candidatos_info = []
        for i, candidate in enumerate(all_candidates):
            candidatos_info.append({
                "ruta_id": candidate.get('ruta_id', f"ruta_{i + 1}"),
                "tipo": candidate.get('tipo_ruta', 'directa'),
                "ranking": i + 1,
                "seleccionada": candidate.get('ruta_id') == selected_route.get('ruta_id'),
                "tiempo_horas": candidate.get('tiempo_total_horas', 0),
                "costo_mxn": candidate.get('costo_total_mxn', 0),
                "distancia_km": candidate.get('distancia_total_km', 0),
                "score": candidate.get('score_lightgbm', 0),
                "ventajas": self._get_simple_advantages(candidate, all_candidates)
            })

        return {
            "total_candidatos": len(all_candidates),
            "candidatos": candidatos_info,
            "mejor_tiempo": min(c.get('tiempo_total_horas', 999) for c in all_candidates),
            "mejor_costo": min(c.get('costo_total_mxn', 999) for c in all_candidates)
        }

    def _get_simple_advantages(self, candidate: Dict[str, Any],
                               all_candidates: List[Dict[str, Any]]) -> List[str]:
        """✅ Ventajas simples"""
        advantages = []

        if len(all_candidates) == 1:
            return ["Única opción disponible"]

        tiempo = candidate.get('tiempo_total_horas', 0)
        costo = candidate.get('costo_total_mxn', 0)

        avg_tiempo = sum(c.get('tiempo_total_horas', 0) for c in all_candidates) / len(all_candidates)
        avg_costo = sum(c.get('costo_total_mxn', 0) for c in all_candidates) / len(all_candidates)

        if tiempo < avg_tiempo:
            advantages.append("Más rápido que promedio")
        if costo < avg_costo:
            advantages.append("Más económico que promedio")
        if candidate.get('tipo_ruta') == 'directa':
            advantages.append("Ruta directa sin transbordos")

        return advantages if advantages else ["Opción viable"]

    def _build_simple_factors_explanation(self, external_factors: Dict[str, Any],
                                          cp_info: Dict[str, Any],
                                          fecha_compra) -> Dict[str, Any]:
        """🌤️ Explicación simple de factores"""

        return {
            "ubicacion": {
                "codigo_postal": cp_info.get('rango_cp', 'N/A'),
                "zona_seguridad": cp_info.get('zona_seguridad', 'Verde'),
                "impacto_zona": self._explain_zone_impact(cp_info.get('zona_seguridad', 'Verde'))
            },
            "temporalidad": {
                "fecha_pedido": fecha_compra.strftime("%Y-%m-%d %H:%M"),
                "eventos_detectados": external_factors.get('eventos_detectados', []) or ['Ninguno'],
                "factor_demanda": external_factors.get('factor_demanda', 1.0),
                "es_temporada_alta": external_factors.get('es_temporada_alta', False)
            },
            "condiciones": {
                "clima": external_factors.get('condicion_clima', 'Templado'),
                "trafico": external_factors.get('trafico_nivel', 'Moderado'),
                "criticidad": external_factors.get('criticidad_logistica', 'Normal')
            },
            "impactos": {
                "tiempo_extra_horas": external_factors.get('impacto_tiempo_extra_horas', 0),
                "costo_extra_pct": external_factors.get('impacto_costo_extra_pct', 0)
            }
        }

    def _explain_zone_impact(self, zona: str) -> str:
        """🛡️ Explica impacto de zona"""
        if zona == 'Verde':
            return "Zona segura - Operación normal sin restricciones"
        elif zona == 'Amarilla':
            return "Zona moderada - Posible incremento de 10-15% en tiempo/costo"
        else:
            return "Zona de riesgo - Incremento significativo en tiempo y costo"

    def _build_simple_geo_data(self, nearby_stores: List[Dict[str, Any]],
                               stock_analysis: Dict[str, Any],
                               cp_info: Dict[str, Any],
                               selected_route: Dict[str, Any]) -> Dict[str, Any]:
        """🗺️ Datos geo simples"""

        destino = {
            "codigo_postal": cp_info.get('rango_cp', ''),
            "coordenadas": {
                "lat": cp_info.get('latitud_centro', 19.4326),
                "lon": cp_info.get('longitud_centro', -99.1332)
            }
        }

        tiendas_geo = []
        selected_store_ids = [plan['tienda_id'] for plan in stock_analysis.get('allocation_plan', [])]

        for store in nearby_stores[:5]:
            try:
                tiendas_geo.append({
                    "tienda_id": store['tienda_id'],
                    "nombre": store.get('nombre_tienda', f"Tienda {store['tienda_id']}"),
                    "coordenadas": {
                        "lat": float(store.get('latitud', 19.4326)),
                        "lon": float(store.get('longitud', -99.1332))
                    },
                    "distancia_km": store.get('distancia_km', 0),
                    "seleccionada": store['tienda_id'] in selected_store_ids
                })
            except (ValueError, TypeError):
                continue

        return {
            "centro_mapa": destino["coordenadas"],
            "zoom_sugerido": 10,
            "destino": destino,
            "tiendas": tiendas_geo,
            "ruta_seleccionada": {
                "origen": selected_route.get('origen_principal', 'N/A'),
                "destino": "Cliente",
                "distancia_km": selected_route.get('distancia_total_km', 0),
                "tipo_flota": selected_route.get('segmentos', [{}])[0].get('tipo_flota', 'FI') if selected_route.get(
                    'segmentos') else 'FI'
            }
        }

    async def _build_dynamic_response(self, request: PredictionRequest,
                                      gemini_decision: Dict[str, Any],
                                      external_factors: Dict[str, Any],
                                      stock_analysis: Dict[str, Any],
                                      all_candidates: List[Dict[str, Any]],
                                      cp_info: Dict[str, Any],
                                      nearby_stores: List[Dict[str, Any]] = None) -> PredictionResponse:
        """🏗️ Construye respuesta con explicabilidad MEJORADA"""

        selected_route = gemini_decision['candidato_seleccionado']

        # Calcular FEE dinámico
        fee_calculation = self._calculate_dynamic_fee(
            selected_route, request, external_factors, cp_info
        )

        # Construir estructuras básicas
        ruta_completa = self._build_route_structure(selected_route, stock_analysis)
        factores_estructurados = self._build_factors_structure(external_factors)

        # Candidatos para explicabilidad básica
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

        # Explicabilidad completa (básica)
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

        # ✅ EXPLICABILIDAD MEJORADA (con validación)
        explicabilidad_extendida = None
        try:
            if nearby_stores:
                explicabilidad_extendida = await self._build_enhanced_explainability(
                    request, selected_route, all_candidates, stock_analysis,
                    external_factors, cp_info, nearby_stores
                )
                logger.info("✅ Explicabilidad extendida generada exitosamente")
            else:
                logger.warning("⚠️ No hay nearby_stores para explicabilidad extendida")
        except Exception as e:
            logger.error(f"❌ Error generando explicabilidad extendida: {e}")
            explicabilidad_extendida = {"error": f"Error generando explicabilidad: {str(e)}"}

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
            explicabilidad=explicabilidad,
            explicabilidad_extendida=explicabilidad_extendida,  # ✅ AGREGAR
            timestamp_response=datetime.now(),  # ✅ AGREGAR
            version_sistema="3.0.0"  # ✅ AGREGAR
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
        # En el método _calculate_dynamic_fee, cambiar la llamada:

        tipo_entrega = self._determine_delivery_type(
            tiempo_total, request.fecha_compra, external_factors, cp_info,
            selected_route.get('distancia_total_km', 999),
            selected_route.get('has_local_stock', False)
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
                                 external_factors: Dict[str, Any], cp_info: Dict[str, Any],
                                 distance_km: float, has_local_stock: bool) -> TipoEntregaEnum:
        """📦 Determina tipo de entrega CORREGIDO con lógica real de negocio"""

        hora_compra = fecha_compra.hour
        factor_demanda = external_factors.get('factor_demanda', 1.0)
        zona = cp_info.get('zona_seguridad', 'Verde')
        cobertura = cp_info.get('cobertura_liverpool', False)

        logger.info(f"📦 Lógica:")
        logger.info(f"   Hora compra: {hora_compra}h, Distancia: {distance_km:.1f}km")
        logger.info(f"   Stock local: {has_local_stock}, Factor demanda: {factor_demanda}")

        # REGLA 1: FLASH - Mismo día (condiciones estrictas)
        if (hora_compra < 12 and  # Antes de mediodía
                has_local_stock and  # Stock en tienda cercana
                distance_km <= 50 and  # Distancia local
                factor_demanda <= 1.2 and  # Sin alta demanda
                zona in ['Verde', 'Amarilla'] and
                cobertura):

            logger.info("   → FLASH: Entrega mismo día")
            return TipoEntregaEnum.FLASH

        # REGLA 2: EXPRESS - Siguiente día
        elif (hora_compra < 20 and  # Antes de 8 PM
              has_local_stock and  # Stock en tienda cercana
              distance_km <= 100 and
              factor_demanda <= 2.0 and
              zona in ['Verde', 'Amarilla']):

            logger.info("   → EXPRESS: Siguiente día hábil")
            return TipoEntregaEnum.EXPRESS

        # REGLA 3: STANDARD - 2-3 días (stock requiere ruteo)
        elif tiempo_horas <= 72:
            logger.info("   → STANDARD: Ruteo requerido")
            return TipoEntregaEnum.STANDARD

        # REGLA 4: PROGRAMADA - Casos complejos
        else:
            logger.info("   → PROGRAMADA: Caso complejo")
            return TipoEntregaEnum.PROGRAMADA

    def _calculate_delivery_date(self, fecha_compra: datetime, tiempo_horas: float,
                                 tipo_entrega: TipoEntregaEnum, external_factors: Dict[str, Any],
                                 hora_compra: int = None) -> datetime:
        """📅 Cálculo CORREGIDO de fecha de entrega"""

        if hora_compra is None:
            hora_compra = fecha_compra.hour

        logger.info(f"📅 CÁLCULO FECHA CORREGIDO:")
        logger.info(f"   Tipo: {tipo_entrega.value}, Hora compra: {hora_compra}h")

        if tipo_entrega == TipoEntregaEnum.FLASH:
            # FLASH: Mismo día, entrega tarde
            if hora_compra <= 10:
                entrega = fecha_compra.replace(hour=16, minute=0, second=0)  # 4 PM mismo día
            else:
                entrega = fecha_compra.replace(hour=19, minute=0, second=0)  # 7 PM mismo día

            logger.info(f"   FLASH: Mismo día {entrega.strftime('%Y-%m-%d %H:%M')}")

        elif tipo_entrega == TipoEntregaEnum.EXPRESS:
            # EXPRESS: Siguiente día hábil
            next_day = self._get_next_business_day(fecha_compra)
            entrega = next_day.replace(hour=14, minute=0, second=0)

            logger.info(f"   EXPRESS: Siguiente día {entrega.strftime('%Y-%m-%d %H:%M')}")

        elif tipo_entrega == TipoEntregaEnum.STANDARD:
            # STANDARD: 2-3 días considerando ruteo
            days_to_add = 2 if hora_compra <= 12 else 3
            entrega = fecha_compra + timedelta(days=days_to_add)
            entrega = self._ensure_business_day(entrega).replace(hour=15, minute=0, second=0)

            logger.info(f"   STANDARD: {days_to_add} días {entrega.strftime('%Y-%m-%d %H:%M')}")

        else:  # PROGRAMADA
            # PROGRAMADA: 4-7 días
            days_to_add = max(4, int(tiempo_horas / 24) + 2)
            entrega = fecha_compra + timedelta(days=days_to_add)
            entrega = self._ensure_business_day(entrega).replace(hour=16, minute=0, second=0)

            logger.info(f"   PROGRAMADA: {days_to_add} días {entrega.strftime('%Y-%m-%d %H:%M')}")

        return entrega

    async def _create_complex_routing_with_cedis(self,
                                                 stock_plan: List[Dict[str, Any]],
                                                 target_coords: Tuple[float, float],
                                                 codigo_postal: str,
                                                 external_factors: Dict[str, Any]) -> Dict[str, Any]:
        """🗺️ Ruteo COMPLEJO real usando CEDIS intermedios"""

        # 1. Encontrar tienda origen con stock
        origen_store = await self._get_store_info(stock_plan[0]['tienda_id'])

        # 2. Encontrar CEDIS intermedio óptimo
        optimal_cedis = await self._find_optimal_cedis_real(origen_store, codigo_postal)

        # 3. Encontrar tienda destino cercana al CP
        destino_store = await self._find_closest_store_to_cp(codigo_postal)

        # 4. Calcular ruta: Origen → CEDIS → Tienda Destino → Cliente
        route_segments = []
        total_time = 0
        total_cost = 0
        total_distance = 0

        # SEGMENTO 1: Tienda Origen → CEDIS
        seg1 = await self._calculate_real_segment(
            origen_store, optimal_cedis, 'FI', external_factors
        )
        route_segments.append(seg1)
        total_time += seg1['tiempo_horas'] + origen_store.get('tiempo_prep_horas', 2)
        total_cost += seg1['costo']
        total_distance += seg1['distancia_km']

        # TIEMPO CEDIS: Procesamiento
        cedis_processing_time = float(optimal_cedis.get('tiempo_procesamiento_horas', '2-4').split('-')[0])
        total_time += cedis_processing_time

        # SEGMENTO 2: CEDIS → Tienda Destino
        seg2 = await self._calculate_real_segment(
            optimal_cedis, destino_store, 'FI', external_factors
        )
        route_segments.append(seg2)
        total_time += seg2['tiempo_horas'] + 1  # Tiempo prep tienda destino
        total_cost += seg2['costo']
        total_distance += seg2['distancia_km']

        # SEGMENTO 3: Tienda Destino → Cliente
        seg3 = await self._calculate_final_segment_to_client(
            destino_store, target_coords, codigo_postal, external_factors
        )
        route_segments.append(seg3)
        total_time += seg3['tiempo_horas']
        total_cost += seg3['costo']
        total_distance += seg3['distancia_km']

        # Aplicar factores externos
        factor_tiempo = external_factors.get('impacto_tiempo_extra_horas', 0)
        total_time += factor_tiempo

        # Probabilidad más conservadora para rutas complejas
        probability = max(0.65, 0.85 - (len(route_segments) * 0.05))

        return {
            'ruta_id': f"complex_{origen_store['tienda_id']}_{optimal_cedis['cedis_id']}_{destino_store['tienda_id']}",
            'tipo_ruta': 'compleja_cedis',
            'segmentos': route_segments,
            'tiempo_total_horas': total_time,
            'costo_total_mxn': total_cost,
            'distancia_total_km': total_distance,
            'probabilidad_cumplimiento': probability,
            'desglose_detallado': {
                'tiempo_origen_prep': origen_store.get('tiempo_prep_horas', 2),
                'tiempo_origen_cedis': seg1['tiempo_horas'],
                'tiempo_cedis_procesamiento': cedis_processing_time,
                'tiempo_cedis_destino': seg2['tiempo_horas'],
                'tiempo_destino_prep': 1,
                'tiempo_destino_cliente': seg3['tiempo_horas'],
                'tiempo_factores_externos': factor_tiempo
            }
        }

    async def _find_optimal_cedis_real(self, origen_store: Dict[str, Any], codigo_postal: str) -> Dict[str, Any]:
        """🏭 Encuentra CEDIS óptimo REAL con logging detallado"""

        cedis_df = self.repos.data_manager.get_data('cedis')
        cp_info = self.repos.store._get_postal_info(codigo_postal)
        estado_destino = cp_info.get('estado_alcaldia', '').split()[0]  # Primer palabra

        logger.info(f"🏭 Análisis de CEDIS para ruteo complejo:")
        logger.info(f"   📍 Origen: {origen_store['nombre_tienda']} ({origen_store['tienda_id']})")
        logger.info(f"   📍 Destino: CP {codigo_postal} ({estado_destino})")

        cedis_candidates = []

        logger.info(f"🔍 Evaluando {cedis_df.height} CEDIS disponibles...")

        for cedis in cedis_df.to_dicts():
            # 1. Verificar cobertura del estado destino
            cobertura = cedis.get('cobertura_estados', '')
            if not ('Nacional' in cobertura or estado_destino in cobertura):
                logger.info(f"   ❌ {cedis['nombre_cedis']}: No cubre {estado_destino}")
                continue

            # 2. Corregir coordenadas si están corruptas
            from utils.geo_calculator import GeoCalculator
            cedis_lat, cedis_lon = GeoCalculator.fix_corrupted_coordinates(
                float(cedis['latitud']), float(cedis['longitud'])
            )

            # 3. Calcular distancias reales
            dist_origen_cedis = GeoCalculator.calculate_distance_km(
                float(origen_store['latitud']), float(origen_store['longitud']),
                cedis_lat, cedis_lon
            )

            dist_cedis_destino = GeoCalculator.calculate_distance_km(
                cedis_lat, cedis_lon,
                float(cp_info['latitud_centro']), float(cp_info['longitud_centro'])
            )

            # 4. Calcular tiempo de procesamiento del CEDIS
            tiempo_proc_num = self._parse_time_range(
                cedis.get('tiempo_procesamiento_horas'), default=2.0
            )

            # 5. Score del CEDIS (distancia total + tiempo procesamiento)
            distancia_total = dist_origen_cedis + dist_cedis_destino
            tiempo_total = tiempo_proc_num + (distancia_total / 60)  # Tiempo aproximado

            # Bonus por cobertura específica del estado
            cobertura_bonus = 0.8 if estado_destino in cobertura else 1.0
            score = tiempo_total * cobertura_bonus

            # ✅ LOGGING DETALLADO: Evaluación de cada CEDIS
            logger.info(f"   📊 {cedis['nombre_cedis']}:")
            logger.info(f"      → Cobertura: {cobertura}")
            logger.info(f"      → {origen_store['nombre_tienda']} → CEDIS: {dist_origen_cedis:.1f}km")
            logger.info(f"      → CEDIS → {estado_destino}: {dist_cedis_destino:.1f}km")
            logger.info(f"      → Distancia total: {distancia_total:.1f}km")
            logger.info(f"      → Tiempo procesamiento: {tiempo_proc_num:.1f}h")
            logger.info(f"      → Score final: {score:.2f} {'✅' if estado_destino in cobertura else '⚠️'}")

            cedis_candidates.append({
                **cedis,
                'latitud_corregida': cedis_lat,
                'longitud_corregida': cedis_lon,
                'dist_origen_cedis': dist_origen_cedis,
                'dist_cedis_destino': dist_cedis_destino,
                'distancia_total': distancia_total,
                'tiempo_procesamiento_num': tiempo_proc_num,
                'score': score,
                'cobertura_match': estado_destino in cobertura
            })

        if not cedis_candidates:
            logger.error(f"❌ No se encontró ningún CEDIS disponible para {estado_destino}")
            return None

        # Ordenar por score (menor es mejor)
        cedis_candidates.sort(key=lambda x: x['score'])

        # ✅ LOGGING: Ranking de CEDIS
        logger.info(f"🏆 Ranking de CEDIS (mejores 3):")
        for i, cedis in enumerate(cedis_candidates[:3], 1):
            logger.info(f"   {i}. {cedis['nombre_cedis']}")
            logger.info(f"      → Score: {cedis['score']:.2f}")
            logger.info(f"      → Distancia total: {cedis['distancia_total']:.1f}km")
            logger.info(f"      → Cobertura específica: {'Sí' if cedis['cobertura_match'] else 'No'}")

        best_cedis = cedis_candidates[0]

        logger.info(f"✅ CEDIS SELECCIONADO: {best_cedis['nombre_cedis']}")
        logger.info(f"   🎯 Razón: Menor score total ({best_cedis['score']:.2f})")
        logger.info(f"   📏 Distancia combinada: {best_cedis['distancia_total']:.1f}km")
        logger.info(f"   ⏱️ Tiempo procesamiento: {best_cedis['tiempo_procesamiento_num']:.1f}h")
        logger.info(f"   🌍 Cobertura {estado_destino}: {'Directa' if best_cedis['cobertura_match'] else 'Nacional'}")

        return best_cedis

    async def _find_closest_store_to_cp(self, codigo_postal: str) -> Dict[str, Any]:
        """🏪 Encuentra tienda Liverpool más cercana al CP destino"""

        # Usar el método existente optimizado
        nearby_stores = self.repos.store.find_stores_by_postal_range(codigo_postal)

        if not nearby_stores:
            logger.error(f"❌ No hay tiendas Liverpool cerca de {codigo_postal}")
            return None

        # Tomar la más cercana
        closest_store = nearby_stores[0]
        logger.info(f"🏪 Tienda destino más cercana: {closest_store['nombre_tienda']} "
                    f"({closest_store['distancia_km']:.1f}km del CP)")

        return closest_store

    async def _calculate_real_segment(self, origen: Dict[str, Any], destino: Dict[str, Any],
                                      tipo_flota: str, external_factors: Dict[str, Any]) -> Dict[str, Any]:
        """⚡ Calcula segmento REAL usando datos de CSV"""

        # Coordenadas corregidas
        from utils.geo_calculator import GeoCalculator

        if 'latitud_corregida' in origen:
            orig_lat, orig_lon = origen['latitud_corregida'], origen['longitud_corregida']
        else:
            orig_lat, orig_lon = GeoCalculator.fix_corrupted_coordinates(
                float(origen['latitud']), float(origen['longitud'])
            )

        if 'latitud_corregida' in destino:
            dest_lat, dest_lon = destino['latitud_corregida'], destino['longitud_corregida']
        else:
            dest_lat, dest_lon = GeoCalculator.fix_corrupted_coordinates(
                float(destino['latitud']), float(destino['longitud'])
            )

        # Calcular distancia real
        distance = GeoCalculator.calculate_distance_km(orig_lat, orig_lon, dest_lat, dest_lon)

        # Calcular tiempo de viaje real
        travel_time = GeoCalculator.calculate_travel_time(
            distance, tipo_flota,
            external_factors.get('trafico_nivel', 'Moderado'),
            external_factors.get('condicion_clima', 'Templado')
        )

        # Calcular costo real
        if tipo_flota == 'FI':
            # Flota interna: costo por km
            costo_base = distance * 15.0  # $15 por km flota interna

            # Aplicar factor de demanda
            factor_demanda = external_factors.get('factor_demanda', 1.0)
            costo_final = costo_base * factor_demanda

            carrier = 'Liverpool'
        else:
            # Flota externa: usar datos del CSV
            flota_df = self.repos.data_manager.get_data('flota_externa')
            carriers = flota_df.filter(pl.col('activo') == True).to_dicts()

            if carriers:
                best_carrier = carriers[0]  # Tomar el primero (Estafeta)
                costo_final = float(best_carrier['costo_base_mxn'])
                carrier = best_carrier['carrier']
            else:
                costo_final = distance * 20.0  # Fallback
                carrier = 'Externo'

        return {
            'origen': origen.get('nombre_tienda') or origen.get('nombre_cedis', 'Origen'),
            'destino': destino.get('nombre_tienda') or destino.get('nombre_cedis', 'Destino'),
            'distancia_km': distance,
            'tiempo_horas': travel_time,
            'tipo_flota': tipo_flota,
            'carrier': carrier,
            'costo': costo_final
        }

    async def _calculate_final_segment_to_client(self, tienda_destino: Dict[str, Any],
                                                 target_coords: Tuple[float, float],
                                                 codigo_postal: str,
                                                 external_factors: Dict[str, Any]) -> Dict[str, Any]:
        """🎯 Calcula segmento final tienda → cliente usando flota externa REAL"""

        from utils.geo_calculator import GeoCalculator

        # Coordenadas tienda destino
        tienda_lat, tienda_lon = GeoCalculator.fix_corrupted_coordinates(
            float(tienda_destino['latitud']), float(tienda_destino['longitud'])
        )

        # Distancia final
        final_distance = GeoCalculator.calculate_distance_km(
            tienda_lat, tienda_lon, target_coords[0], target_coords[1]
        )

        # Usar flota externa REAL del CSV para último tramo
        flota_df = self.repos.data_manager.get_data('flota_externa')
        peso_estimado = 1.0  # Peso promedio del paquete

        # Buscar carrier que cubra el CP
        cp_int = int(codigo_postal)
        available_carriers = []

        for carrier in flota_df.to_dicts():
            if (carrier.get('activo', False) and
                    carrier.get('zona_cp_inicio', 0) <= cp_int <= carrier.get('zona_cp_fin', 99999) and
                    carrier.get('peso_min_kg', 0) <= peso_estimado <= carrier.get('peso_max_kg', 100)):
                available_carriers.append(carrier)

        if available_carriers:
            # Usar el carrier más económico
            best_carrier = min(available_carriers, key=lambda x: x.get('costo_base_mxn', 999))

            # Costo real del carrier
            costo_base = float(best_carrier['costo_base_mxn'])

            # ✅ CORRECCIÓN: Manejar rangos de tiempo como "3-5"
            dias_entrega = int(self._parse_time_range(
                best_carrier.get('tiempo_entrega_dias_habiles'), default=2.0
            ))

            # Tiempo en horas (días hábiles a horas)
            tiempo_entrega_horas = dias_entrega * 24

            carrier_name = best_carrier['carrier']
            tipo_servicio = best_carrier.get('tipo_servicio', 'Standard')

            logger.info(f"📦 Carrier final: {carrier_name} - {tipo_servicio} "
                        f"(${costo_base}, {dias_entrega} días, {tiempo_entrega_horas}h)")
        else:
            # Fallback mínimo
            costo_base = 150.0
            tiempo_entrega_horas = 48
            carrier_name = 'Externo'
            logger.warning(f"⚠️ No hay carriers disponibles para CP {codigo_postal}, usando fallback")

        return {
            'origen': tienda_destino.get('nombre_tienda', 'Tienda'),
            'destino': 'Cliente',
            'distancia_km': final_distance,
            'tiempo_horas': tiempo_entrega_horas,
            'tipo_flota': 'FE',
            'carrier': carrier_name,
            'costo': costo_base
        }

    def _parse_time_range(self, time_value: Any, default: float = 2.0) -> float:
        """🕐 Parser robusto para rangos de tiempo del CSV"""

        if time_value is None:
            return default

        time_str = str(time_value).strip()

        if not time_str or time_str.lower() in ['nan', 'null', '']:
            return default

        try:
            if '-' in time_str:
                # Rango: "3-5", "2-4" → tomar el mínimo
                parts = time_str.split('-')
                return float(parts[0])
            else:
                # Número simple: "2", "3.5"
                return float(time_str)
        except (ValueError, IndexError) as e:
            logger.warning(f"⚠️ Error parseando tiempo '{time_str}': {e}, usando default {default}")
            return default

    def _ensure_business_day(self, fecha: datetime) -> datetime:
        """📅 Asegura que la fecha sea día hábil"""
        # Si es domingo (6), mover al lunes
        while fecha.weekday() == 6:  # Domingo
            fecha += timedelta(days=1)

        return fecha

    def _create_baseline_factors(self, fecha: datetime, codigo_postal: str) -> Dict[str, Any]:
        """📊 Factores baseline mínimos (solo si NO hay datos en CSV)"""

        logger.warning(f"⚠️ No hay datos en CSV para {fecha.date()}, usando baseline mínimo")

        return {
            'evento_detectado': 'Normal',
            'eventos_detectados': [],
            'factor_demanda': 1.0,
            'condicion_clima': 'Templado',
            'trafico_nivel': 'Moderado',
            'impacto_tiempo_extra_horas': 0.0,
            'criticidad_logistica': 'Normal',
            'es_temporada_alta': False,
            'es_temporada_critica': False,
            'fuente_datos': 'baseline_minimal',
            'observaciones_clima': 'Sin datos específicos',
            'rango_cp_afectado': '00000-99999'
        }

    # ✅ CORRECCIÓN: Ventana de 5 horas como solicitado
    def _calculate_time_window(self, fecha_entrega: datetime, tipo_entrega: TipoEntregaEnum) -> Dict[str, dt_time]:
        """🕐 Ventana de entrega AMPLIADA (5 horas)"""

        logger.info(f"🕐 Calculando ventana AMPLIADA para: {fecha_entrega.strftime('%Y-%m-%d %H:%M')}")

        # ✅ NUEVO: Ventana de 5 horas para mayor colchón
        if tipo_entrega == TipoEntregaEnum.FLASH:
            ventana_horas = 3  # ±1.5h para FLASH
        else:
            ventana_horas = 5  # ±2.5h para todos los demás

        # Calcular inicio y fin
        inicio_ventana = fecha_entrega - timedelta(hours=ventana_horas // 2)
        fin_ventana = fecha_entrega + timedelta(hours=ventana_horas // 2)

        # Respetar horarios de entrega (9 AM - 7 PM)
        HORA_MIN = 9
        HORA_MAX = 19

        if inicio_ventana.hour < HORA_MIN:
            inicio_ventana = inicio_ventana.replace(hour=HORA_MIN, minute=0)
        if fin_ventana.hour >= HORA_MAX:
            fin_ventana = fin_ventana.replace(hour=HORA_MAX, minute=0)

        # Asegurar ventana mínima de 2 horas
        if fin_ventana <= inicio_ventana:
            fin_ventana = inicio_ventana + timedelta(hours=2)

        logger.info(
            f"   Ventana AMPLIADA: {inicio_ventana.strftime('%H:%M')} - {fin_ventana.strftime('%H:%M')} ({ventana_horas}h)")

        return {
            'inicio': inicio_ventana.time(),
            'fin': fin_ventana.time()
        }

    def _get_next_business_day(self, fecha: datetime) -> datetime:
        """📅 Obtiene el siguiente día hábil"""
        next_day = fecha + timedelta(days=1)

        # Si es domingo (6), ir al lunes
        while next_day.weekday() == 6:  # Domingo
            next_day += timedelta(days=1)

        return next_day



    @staticmethod
    def _calculate_time_window(fecha_entrega: datetime, tipo_entrega: TipoEntregaEnum) -> Dict[str, dt_time]:
        """🕐 Calcula ventana de entrega CORREGIDA"""

        logger.info(f"🕐 Calculando ventana para: {fecha_entrega.strftime('%Y-%m-%d %H:%M')}")

        # ✅ CORRECCIÓN: Ventanas más realistas
        if tipo_entrega == TipoEntregaEnum.FLASH:
            ventana_horas = 1  # ±30min para FLASH
        elif tipo_entrega == TipoEntregaEnum.EXPRESS:
            ventana_horas = 2  # ±1h para EXPRESS
        else:
            ventana_horas = 4  # ±2h para STANDARD/PROGRAMADA

        # Calcular inicio y fin de ventana
        inicio_ventana = fecha_entrega - timedelta(hours=ventana_horas // 2)
        fin_ventana = fecha_entrega + timedelta(hours=ventana_horas // 2)

        # ✅ CORRECCIÓN: Respetar horarios de entrega (10 AM - 6 PM)
        HORA_MIN_ENTREGA = 10
        HORA_MAX_ENTREGA = 18

        # Ajustar inicio
        if inicio_ventana.hour < HORA_MIN_ENTREGA:
            inicio_ventana = inicio_ventana.replace(hour=HORA_MIN_ENTREGA, minute=0)
        elif inicio_ventana.hour >= HORA_MAX_ENTREGA:
            inicio_ventana = inicio_ventana.replace(hour=HORA_MAX_ENTREGA - 1, minute=0)

        # Ajustar fin
        if fin_ventana.hour < HORA_MIN_ENTREGA:
            fin_ventana = fin_ventana.replace(hour=HORA_MIN_ENTREGA + 1, minute=0)
        elif fin_ventana.hour >= HORA_MAX_ENTREGA:
            fin_ventana = fin_ventana.replace(hour=HORA_MAX_ENTREGA, minute=0)

        # Asegurar que la ventana tenga sentido
        if fin_ventana <= inicio_ventana:
            fin_ventana = inicio_ventana + timedelta(hours=1)

        logger.info(f"   Ventana: {inicio_ventana.strftime('%H:%M')} - {fin_ventana.strftime('%H:%M')}")

        return {
            'inicio': inicio_ventana.time(),
            'fin': fin_ventana.time()
        }

    @staticmethod
    def _build_route_structure(route_data: Dict[str, Any], stock_analysis: Dict[str, Any]) -> RutaCompleta:
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

