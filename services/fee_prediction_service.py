import time
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Any, Tuple

import polars as pl

from config.settings import settings
from models.schemas import (
    PredictionRequest, TipoEntregaEnum,
    FEECalculation, SplitInventory, UbicacionStock
)
from services.ai.gemini_service import GeminiLogisticsDecisionEngine
from utils.geo_calculator import GeoCalculator
from utils.logger import logger


class FEEPredictionService:
    """🚀 Servicio de predicción FEE """

    def __init__(self, repositories):
        self.repos = repositories
        self.gemini_engine = GeminiLogisticsDecisionEngine()
        self._factors_cache = {}
        self._store_cache = {}
        logger.info("Servicio FEE")

    async def predict_fee(self, request: PredictionRequest) -> Dict[str, Any]:
        """🚀 Predicción FEE con respuesta simplificada y validación CSV completa"""
        start_time = time.time()

        try:
            logger.info(f"🎯 NUEVA PREDICCIÓN: {request.sku_id} → {request.codigo_postal} (qty: {request.cantidad})")

            csv_validation = self._validate_csv_data_integrity(request)
            if csv_validation['warnings']:
                for warning in csv_validation['warnings']:
                    logger.warning(f"⚠️ {warning}")

            validation = await self._validate_request_dynamic(request)
            if not validation['valid']:
                raise ValueError(validation['error'])

            product_info = validation['product']
            cp_info = validation['postal_info']

            external_factors = self._get_comprehensive_external_factors(request.fecha_compra, request.codigo_postal)
            nearby_stores = self.repos.store.find_stores_by_postal_range(request.codigo_postal)
            if not nearby_stores:
                raise ValueError(f"No hay tiendas Liverpool cerca de {request.codigo_postal}")

            stock_analysis = await self._analyze_stock_dynamic(
                request, product_info, nearby_stores
            )

            if not stock_analysis['factible']:
                raise ValueError(f"Stock insuficiente: {stock_analysis['razon']}")

            candidates = await self._generate_candidates_dynamic(
                stock_analysis, cp_info, external_factors, request
            )

            if not candidates:
                raise ValueError("No se encontraron rutas factibles")

            ranked_candidates = self._rank_candidates_dynamic(candidates)
            top_candidates = ranked_candidates[:3]

            # Decisión de Gemini
            gemini_decision = await self.gemini_engine.select_optimal_route(
                top_candidates, request.dict(), external_factors
            )

            processing_time = (time.time() - start_time) * 1000
            simplified_response = await self._build_simplified_response(
                request, gemini_decision['candidato_seleccionado'],
                ranked_candidates, stock_analysis, external_factors,
                cp_info, product_info, processing_time
            )

            simplified_response['metadata'] = {
                'csv_sources_used': csv_validation['csv_sources'],
                'warnings': csv_validation['warnings'],
                'data_integrity': 'validated',
                'version_sistema': '3.0.0'
            }
            self._log_data_sources(simplified_response)

            logger.info(f"✅ Predicción completada en {processing_time:.1f}ms")
            return simplified_response

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Error en predicción: {e} ({processing_time:.1f}ms)")
            raise

    def _validate_csv_data_integrity(self, request: PredictionRequest) -> Dict[str, Any]:
        """Validcion para datos reales"""

        validation_results = {
            'valid': True,
            'warnings': [],
            'csv_sources': {}
        }
        producto = self.repos.product.get_product_by_sku(request.sku_id)
        if producto:
            validation_results['csv_sources']['producto'] = 'productos_liverpool_50.csv'
        else:
            validation_results['warnings'].append(f"Producto {request.sku_id} no encontrado en CSV")

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

    @staticmethod
    def _log_data_sources(final_response: Dict[str, Any]) -> None:
        """Log detallado de fuentes de datos utilizadas
        TODO -> NEXT UPDATE Esta madre hay que moverla al utils -> PENDIENTE
        """

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
        """ response DETALLADO con toda la información de logs en JSON"""

        peso_kg_estimado = producto_info.get('peso_kg', 0.2) * request.cantidad
        zona_seguridad_real = cp_info.get('zona_seguridad', 'Verde')
        fee_calculation = self._calculate_dynamic_fee(
            selected_route, request, external_factors, cp_info,
            stock_analysis=stock_analysis
        )
        candidatos_simplificados = []
        for candidate in all_candidates:
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
                "zona_seguridad": zona_seguridad_real
            })

        ganador_info = candidatos_simplificados[0] if candidatos_simplificados else {}
        factores_externos_reales = self._extract_real_external_factors(external_factors, cp_info)
        evaluacion_detallada = {
            'pesos_configurados': {
                'tiempo': settings.PESO_TIEMPO,
                'costo': settings.PESO_COSTO,
                'stock': settings.PESO_PROBABILIDAD,
                'distancia': settings.PESO_DISTANCIA
            },
            'rutas_evaluadas': None,
            'stock_analysis': None,
            'cedis_analysis': None
        }

        #  información de rutas evaluadas
        if all_candidates and hasattr(all_candidates[0], 'rutas_evaluadas_detalle'):
            evaluacion_detallada['rutas_evaluadas'] = all_candidates[0].rutas_evaluadas_detalle

        # análisis de stock
        if 'analysis_details' in stock_analysis:
            evaluacion_detallada['stock_analysis'] = stock_analysis['analysis_details']

        # CEDIS (si es ruta compleja)
        if selected_route.get('tipo_ruta') in ['compleja_cedis', 'multi_segmento_cedis']:
            if 'cedis_analysis' in selected_route:
                evaluacion_detallada['cedis_analysis'] = selected_route['cedis_analysis']

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
                    "stock": settings.PESO_PROBABILIDAD,
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
            "evaluacion_detallada": evaluacion_detallada,
            "logistica_entrega": {
                "ruta": self._build_route_description(selected_route),
                "tipo_ruta": selected_route.get('tipo_ruta', 'directa'),
                "flota": self._get_fleet_info_simplified(selected_route),
                "carrier": self._get_main_carrier(selected_route),
                "distancia_km": round(selected_route.get('distancia_total_km', 0), 1),
                "tiempo_total_h": round(selected_route.get('tiempo_total_horas', 0), 1),
                "desglose_tiempos_h": self._get_time_breakdown(selected_route, external_factors),
                "factores_aplicados": selected_route.get('factores_aplicados', []),
                "probabilidad_cumplimiento": round(selected_route.get('probabilidad_cumplimiento', 0), 3),
                "cedis_intermedio": self._get_cedis_from_store(selected_route) if selected_route.get('tipo_ruta') in [
                    'compleja_cedis', 'multi_segmento_cedis'] else None
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
            },
            "metadata": {
                "csv_sources_used": {
                    "producto": "productos_liverpool_50.csv",
                    "codigo_postal": "codigos_postales_rangos_mexico.csv",
                    "flota_externa": "flota_externa_costos_reales.csv"
                },
                "warnings": [],
                "data_integrity": "validated",
                "version_sistema": "3.0.0"
            }
        }


    @staticmethod
    def _extract_real_external_factors(external_factors: Dict[str, Any], cp_info: Dict[str, Any]) -> Dict[
        str, Any]:
        """📊 Extrae factores externos REALES de los CSV"""
        return {
            "evento_detectado": external_factors.get('evento_detectado', 'Normal'),
            "factor_demanda": external_factors.get('factor_demanda', 1.0),
            "condicion_clima": external_factors.get('condicion_clima', 'Templado'),
            "trafico_nivel": external_factors.get('trafico_nivel', 'Moderado'),
            "criticidad_logistica": external_factors.get('criticidad_logistica', 'Normal'),
            "zona_seguridad": cp_info.get('zona_seguridad', 'Verde'),
            "es_temporada_alta": external_factors.get('es_temporada_alta', False),
            "impacto_tiempo_extra_horas": external_factors.get('impacto_tiempo_extra_horas', 0),
            "rango_cp_afectado": external_factors.get('rango_cp_afectado', '00000-99999'),
            "fuente_datos": external_factors.get('fuente_datos', 'CSV_real')
        }

    @staticmethod
    def _get_real_advantages(selected_route: Dict[str, Any], all_candidates: List[Dict[str, Any]],
                             zona_seguridad: str) -> List[str]:
        """ventajas reales basadas en datos de CSV"""
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

    @staticmethod
    def _get_stock_for_candidate(candidate: Dict[str, Any], stock_analysis: Dict[str, Any]) -> int:
        """Obtiene stock real para un candidato"""
        tienda_id = candidate.get('tienda_origen_id', candidate.get('origen_principal', ''))

        for plan_item in stock_analysis.get('allocation_plan', []):
            if plan_item.get('tienda_id') == tienda_id:
                return plan_item.get('stock_disponible', 0)

        return 0

    @staticmethod
    def _get_fleet_info_simplified(route_data: Dict[str, Any]) -> str:
        """ Obtiene info de flota """
        segmentos = route_data.get('segmentos', [])
        if not segmentos:
            return 'N/A'

        primer_segmento = segmentos[0]
        tipo_flota = primer_segmento.get('tipo_flota', 'FI')
        carrier = primer_segmento.get('carrier', 'Liverpool')

        if tipo_flota == 'FI':
            return 'FI-Live'
        else:
            return f'FE-{carrier[:4]}'

    @staticmethod
    def _get_fleet_info_detailed(route_data: Dict[str, Any]) -> Dict[str, Any]:
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

    @staticmethod
    def _get_time_breakdown(route_data: Dict[str, Any], external_factors: Dict[str, Any]) -> Dict[str, float]:
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

        tipo_ruta = route_data.get('tipo_ruta', 'directa')

        if tipo_ruta == 'compleja_cedis':
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
            return route_data.get('cedis_intermedio', 'N/A')

        tienda_id = route_data.get('tienda_origen_id')
        if not tienda_id:
            return 'N/A'

        tiendas_df = self.repos.data_manager.get_data('tiendas')
        tienda_data = tiendas_df.filter(pl.col('tienda_id') == tienda_id)

        if tienda_data.height > 0:
            return tienda_data.to_dicts()[0].get('cedis_asignado', 'N/A')

        return 'N/A'

    @staticmethod
    def _build_route_description(route_data: Dict[str, Any]) -> str:
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

    @staticmethod
    def _is_local_store(location: Dict[str, Any]) -> bool:
        """📍 Determina si una tienda es local (< 100km)"""
        return location.get('distancia_km', 999) < 100

    async def _validate_request_dynamic(self, request: PredictionRequest) -> Dict[str, Any]:
        """Validación dinámica con datos reales"""

        product = self.repos.product.get_product_by_sku(request.sku_id)
        if not product:
            return {'valid': False, 'error': f'Producto no encontrado: {request.sku_id}'}

        tiendas_disponibles = product.get('tiendas_disponibles', '')
        if not tiendas_disponibles:
            return {'valid': False, 'error': f'Producto sin tiendas autorizadas: {request.sku_id}'}

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
        """Análisis de stock MEJORADO - captura información detallada para response"""

        analysis_details = {
            'tiendas_cercanas': [],
            'tiendas_autorizadas': [],
            'stock_encontrado': [],
            'asignacion_detallada': {},
            'resumen_stock': {}
        }

        tiendas_autorizadas = [t.strip() for t in product_info['tiendas_disponibles'].split(',')]
        logger.info(f"🏪 Tiendas autorizadas para {request.sku_id}: {tiendas_autorizadas}")

        # PASO 1: Buscar en tiendas locales PRIMERO
        local_store_ids = [store['tienda_id'] for store in nearby_stores[:5]]
        stock_locations_local = self.repos.stock.get_stock_for_stores_and_sku(
            request.sku_id, local_store_ids, request.cantidad
        )

        #  CAPTURAR: Tiendas cercanas
        logger.info(f"📍 Tiendas cercanas al CP {request.codigo_postal}:")
        for i, store in enumerate(nearby_stores[:5], 1):
            logger.info(f"   {i}. {store['nombre_tienda']} ({store['distancia_km']:.1f}km)")

            analysis_details['tiendas_cercanas'].append({
                'posicion': i,
                'tienda_id': store['tienda_id'],
                'nombre': store['nombre_tienda'],
                'distancia_km': round(store['distancia_km'], 1),
                'coordenadas': {
                    'lat': float(store['latitud']),
                    'lon': float(store['longitud'])
                },
                'zona_seguridad': store.get('zona_seguridad', 'Verde'),
                'estado': store.get('estado', 'N/A'),
                'alcaldia_municipio': store.get('alcaldia_municipio', 'N/A')
            })

        if stock_locations_local:
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

                authorized_nearby = [
                    store for store in nearby_stores
                    if any(stock['tienda_id'] == store['tienda_id'] for stock in stock_locations_local)
                ]

                allocation = self.repos.stock.calculate_optimal_allocation(
                    stock_locations_local, request.cantidad, authorized_nearby
                )

                if allocation['factible']:
                    for stock_loc in stock_locations_local:
                        tienda_info = self._get_store_info_sync(stock_loc['tienda_id'])
                        analysis_details['stock_encontrado'].append({
                            'tienda_id': stock_loc['tienda_id'],
                            'nombre_tienda': tienda_info[
                                'nombre_tienda'] if tienda_info else f"Tienda {stock_loc['tienda_id']}",
                            'stock_disponible': stock_loc['stock_disponible'],
                            'distancia_km': round(tienda_info.get('distancia_km', 0) if tienda_info else 0, 1),
                            'precio_tienda': stock_loc.get('precio_tienda', 0),
                            'es_local': True
                        })

                    analysis_details['resumen_stock'] = {
                        'total_disponible': total_local_stock,
                        'requerido': request.cantidad,
                        'suficiente': True,
                        'tiendas_con_stock': len(stock_locations_local),
                        'tipo_stock': 'LOCAL'
                    }

                    self._capture_allocation_details(allocation, authorized_nearby, analysis_details, request.cantidad)

                    split_inventory = self._build_split_inventory(allocation['plan'], request.cantidad)
                    return {
                        'factible': True,
                        'allocation_plan': allocation['plan'],
                        'split_inventory': split_inventory,
                        'stores_info': authorized_nearby,
                        'total_available': allocation['cantidad_cubierta'],
                        'source': 'LOCAL_PRIORITY',
                        'analysis_details': analysis_details
                    }

        logger.info("🌎 Stock local insuficiente, buscando en tiendas autorizadas nacionales...")
        tiendas_df = self.repos.data_manager.get_data('tiendas')
        authorized_stores = tiendas_df.filter(
            pl.col('tienda_id').is_in(tiendas_autorizadas)
        ).to_dicts()

        if not authorized_stores:
            return {
                'factible': False,
                'razon': f'No hay tiendas autorizadas disponibles para {request.sku_id}',
                'analysis_details': analysis_details
            }

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

        authorized_stores.sort(key=lambda x: x['distancia_km'])
        authorized_nearby = authorized_stores[:10]

        logger.info(f"🏪 Tiendas autorizadas más cercanas:")
        for i, store in enumerate(authorized_nearby[:10], 1):
            if i <= 5:  # Solo log las primeras 5
                logger.info(f"   {i}. {store['nombre_tienda']} ({store['distancia_km']:.1f}km)")

            analysis_details['tiendas_autorizadas'].append({
                'posicion': i,
                'tienda_id': store['tienda_id'],
                'nombre': store['nombre_tienda'],
                'distancia_km': round(store['distancia_km'], 1),
                'estado': store.get('estado', 'N/A'),
                'zona_seguridad': store.get('zona_seguridad', 'Verde'),
                'alcaldia_municipio': store.get('alcaldia_municipio', 'N/A')
            })

        store_ids = [store['tienda_id'] for store in authorized_nearby]
        stock_locations = self.repos.stock.get_stock_for_stores_and_sku(
            request.sku_id, store_ids, request.cantidad
        )

        if not stock_locations:
            return {
                'factible': False,
                'razon': f'Sin stock disponible para {request.sku_id} en tiendas autorizadas',
                'analysis_details': analysis_details
            }

        logger.info(f"📦 Stock NACIONAL encontrado para {request.sku_id}:")
        total_stock = 0
        for stock_loc in stock_locations:
            tienda_info = next((s for s in authorized_nearby if s['tienda_id'] == stock_loc['tienda_id']), None)
            nombre_tienda = tienda_info['nombre_tienda'] if tienda_info else f"Tienda {stock_loc['tienda_id']}"
            distancia = tienda_info['distancia_km'] if tienda_info else 999

            logger.info(f"   📍 {nombre_tienda}: {stock_loc['stock_disponible']} unidades ({distancia:.1f}km)")

            analysis_details['stock_encontrado'].append({
                'tienda_id': stock_loc['tienda_id'],
                'nombre_tienda': nombre_tienda,
                'stock_disponible': stock_loc['stock_disponible'],
                'distancia_km': round(distancia, 1),
                'precio_tienda': stock_loc.get('precio_tienda', 0),
                'precio_total': stock_loc.get('precio_tienda', 0) * request.cantidad,
                'es_local': False
            })

            total_stock += stock_loc['stock_disponible']

        logger.info(f"📊 Stock NACIONAL total: {total_stock} | Requerido: {request.cantidad}")

        analysis_details['resumen_stock'] = {
            'total_disponible': total_stock,
            'requerido': request.cantidad,
            'suficiente': total_stock >= request.cantidad,
            'tiendas_con_stock': len(stock_locations),
            'tipo_stock': 'NACIONAL'
        }

        allocation = self.repos.stock.calculate_optimal_allocation(
            stock_locations, request.cantidad, authorized_nearby
        )

        if not allocation['factible']:
            return {
                'factible': False,
                'razon': allocation['razon'],
                'analysis_details': analysis_details
            }

        self._capture_allocation_details(allocation, authorized_nearby, analysis_details, request.cantidad)

        #  SplitInventory
        split_inventory = self._build_split_inventory(allocation['plan'], request.cantidad)

        return {
            'factible': True,
            'allocation_plan': allocation['plan'],
            'split_inventory': split_inventory,
            'stores_info': authorized_nearby,
            'total_available': allocation['cantidad_cubierta'],
            'source': 'NATIONAL_AUTHORIZED',
            'analysis_details': analysis_details  # ✅ NUEVO
        }

    @staticmethod
    def _capture_allocation_details(allocation: Dict[str, Any],
                                    authorized_nearby: List[Dict[str, Any]],
                                    analysis_details: Dict[str, Any],
                                    cantidad_requerida: int):
        """📋 Captura detalles de la asignación para el JSON response"""

        logger.info("📋 Plan de asignación FINAL:")

        analysis_details['asignacion_detallada'] = {
            'metodo_asignacion': allocation.get('selection_method', 'tiempo_costo_optimizado'),
            'pesos_utilizados': {
                'tiempo': 35,
                'costo': 35,
                'stock': 20,
                'distancia': 10
            },
            'plan_asignacion': []
        }

        for plan_item in allocation['plan']:
            tienda_info = next((s for s in authorized_nearby if s['tienda_id'] == plan_item['tienda_id']), None)
            nombre_tienda = tienda_info['nombre_tienda'] if tienda_info else f"Tienda {plan_item['tienda_id']}"
            distancia = tienda_info['distancia_km'] if tienda_info else 999
            precio_unitario = plan_item.get('precio_tienda', 0)
            precio_total = precio_unitario * plan_item['cantidad']

            logger.info(f"   🏪 {nombre_tienda}:")
            logger.info(f"      → {plan_item['cantidad']} unidades de {cantidad_requerida} requeridas")
            logger.info(f"      → ${precio_total:,.0f} total (${precio_unitario:,.0f} por unidad)")
            logger.info(f"      → {distancia:.1f}km del destino")
            logger.info(f"      → Razón: {plan_item.get('razon_seleccion', 'Stock disponible, distancia óptima')}")

            plan_detail = {
                'tienda_id': plan_item['tienda_id'],
                'nombre_tienda': nombre_tienda,
                'cantidad_asignada': plan_item['cantidad'],
                'stock_disponible': plan_item['stock_disponible'],
                'distancia_km': round(distancia, 1),
                'tiempo_total_h': round(plan_item.get('tiempo_total_horas', 0), 1),
                'costo_total_mxn': round(plan_item.get('costo_total', 0), 2),
                'score_total': round(plan_item.get('score_total', 0), 3),
                'fleet_type': plan_item.get('fleet_type', 'FE'),
                'carrier': plan_item.get('carrier', 'Externo'),
                'precio_unitario': precio_unitario,
                'precio_total': precio_total,
                'razon_seleccion': plan_item.get('razon_seleccion', 'Stock disponible, distancia óptima')
            }
            analysis_details['asignacion_detallada']['plan_asignacion'].append(plan_detail)

    def _build_split_inventory(self, plan: List[Dict[str, Any]], cantidad_requerida: int) -> SplitInventory:
        """🏗️ Construye objeto SplitInventory CON COORDENADAS REALES"""
        ubicaciones = []

        for item in plan:
            tienda_info = self._get_store_info_sync(item['tienda_id'])
            coordenadas_reales = {
                'lat': float(tienda_info['latitud']) if tienda_info else 19.4326,
                'lon': float(tienda_info['longitud']) if tienda_info else -99.1332
            }

            ubicacion = UbicacionStock(
                ubicacion_id=item['tienda_id'],
                ubicacion_tipo='TIENDA',
                nombre_ubicacion=tienda_info['nombre_tienda'] if tienda_info else f"Liverpool {item['tienda_id']}",
                stock_disponible=item['cantidad'],
                stock_reservado=0,
                coordenadas=coordenadas_reales,
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
        """🗺️ Generación de candidatos con CONSOLIDACIÓN INTELIGENTE"""

        allocation_plan = stock_analysis['allocation_plan']
        target_lat = cp_info['latitud_centro']
        target_lon = cp_info['longitud_centro']
        candidates = []

        logger.info(f"🗺️ ANÁLISIS DE ESTRATEGIAS DE ENTREGA")
        logger.info(f"{'=' * 60}")

        local_stores = []
        remote_stores = []

        for plan_item in allocation_plan:
            tienda_origen = await self._get_store_info(plan_item['tienda_id'])
            distance_direct = GeoCalculator.calculate_distance_km(
                float(tienda_origen['latitud']), float(tienda_origen['longitud']),
                target_lat, target_lon
            )

            plan_item['store_info'] = tienda_origen
            plan_item['distance_to_customer'] = distance_direct

            if distance_direct <= 50:  # Local CDMX
                local_stores.append(plan_item)
            else:  # Remota
                remote_stores.append(plan_item)

        logger.info(f"📍 Tiendas locales (<50km): {len(local_stores)}")
        logger.info(f"📍 Tiendas remotas (>50km): {len(remote_stores)}")

        logger.info(f"\n📦 ESTRATEGIA 1: Múltiples paquetes directos")
        logger.info(f"{'-' * 50}")

        total_cost_direct = 0
        total_time_direct = 0
        paquetes_info = []

        for plan_item in allocation_plan:
            tienda_origen = plan_item['store_info']
            distance = plan_item['distance_to_customer']

            # Crear candidato directo
            if distance <= 100:  # Local
                direct_candidate = await self._create_direct_route_dynamic(
                    plan_item, (target_lat, target_lon), external_factors, request, cp_info
                )
                if direct_candidate:
                    candidates.append(direct_candidate)
                    total_cost_direct += direct_candidate['costo_total_mxn']
                    total_time_direct = max(total_time_direct, direct_candidate['tiempo_total_horas'])

                    paquetes_info.append({
                        'origen': tienda_origen['nombre_tienda'],
                        'cantidad': plan_item['cantidad'],
                        'costo': direct_candidate['costo_total_mxn'],
                        'tiempo': direct_candidate['tiempo_total_horas']
                    })
            else:  # Remota - usar carrier externo
                peso_envio = self._calculate_shipment_weight(request, plan_item['cantidad'])
                carriers = self.repos.fleet.get_best_carriers_for_cp(request.codigo_postal, peso_envio)

                if carriers:
                    best_carrier = carriers[0]
                    costo_directo = self._calculate_external_fleet_cost(
                        best_carrier, peso_envio, distance, external_factors
                    )
                    tiempo_directo = 72.0  # 3 días típico carrier

                    total_cost_direct += costo_directo
                    total_time_direct = max(total_time_direct, tiempo_directo)

                    paquetes_info.append({
                        'origen': tienda_origen['nombre_tienda'],
                        'cantidad': plan_item['cantidad'],
                        'costo': costo_directo,
                        'tiempo': tiempo_directo,
                        'carrier': best_carrier['carrier']
                    })

        logger.info(f"   Paquetes separados: {len(paquetes_info)}")
        for i, paq in enumerate(paquetes_info, 1):
            logger.info(f"   {i}. {paq['origen']}: {paq['cantidad']} unidades")
            logger.info(f"      → Costo: ${paq['costo']:.0f}, Tiempo: {paq['tiempo']:.1f}h")
            if 'carrier' in paq:
                logger.info(f"      → Carrier: {paq['carrier']}")
        logger.info(f"   📊 TOTALES: ${total_cost_direct:.0f}, {total_time_direct:.1f}h máximo")

        # ESTRATEGIA 2: CONSOLIDACIÓN EN TIENDA CDMX
        if remote_stores and local_stores:
            logger.info(f"\n📦 ESTRATEGIA 2: Consolidación en tienda CDMX")
            logger.info(f"{'-' * 50}")

            best_consolidator = max(local_stores, key=lambda x: x['stock_disponible'])
            consolidator_info = best_consolidator['store_info']

            logger.info(f"   Hub consolidador: {consolidator_info['nombre_tienda']}")

            total_cost_consolidated = 0
            total_time_consolidated = 0
            consolidation_steps = []

            # Envíos remotos → consolidador
            for remote in remote_stores:
                origen = remote['store_info']

                # tienda remota → tienda consolidadora
                distance_to_hub = GeoCalculator.calculate_distance_km(
                    float(origen['latitud']), float(origen['longitud']),
                    float(consolidator_info['latitud']), float(consolidator_info['longitud'])
                )
                tiempo_transferencia = self._calculate_travel_time_dynamic(
                    distance_to_hub, 'FI', external_factors
                )
                costo_transferencia = distance_to_hub * 8.0

                total_cost_consolidated += costo_transferencia
                total_time_consolidated = max(total_time_consolidated, tiempo_transferencia + 4)

                consolidation_steps.append({
                    'tipo': 'transferencia',
                    'origen': origen['nombre_tienda'],
                    'destino': consolidator_info['nombre_tienda'],
                    'cantidad': remote['cantidad'],
                    'distancia': distance_to_hub,
                    'tiempo': tiempo_transferencia,
                    'costo': costo_transferencia
                })

                logger.info(f"   → {origen['nombre_tienda']} → {consolidator_info['nombre_tienda']}")
                logger.info(f"     {remote['cantidad']} unidades, {distance_to_hub:.0f}km")
                logger.info(f"     Costo: ${costo_transferencia:.0f}, Tiempo: {tiempo_transferencia:.1f}h")

            # Entrega consolidada → cliente
            cantidad_total = sum(item['cantidad'] for item in allocation_plan)
            peso_consolidado = self._calculate_shipment_weight(request, cantidad_total)
            distance_final = best_consolidator['distance_to_customer']

            if distance_final <= 20:  # Última milla
                costo_final = 150  # Tarifa plana última milla
                tiempo_final = 4  # Mismo día
                carrier_final = "Liverpool Express"
            else:
                carriers = self.repos.fleet.get_best_carriers_for_cp(request.codigo_postal, peso_consolidado)
                if carriers:
                    best_carrier = carriers[0]
                    costo_final = best_carrier['costo_base_mxn']
                    tiempo_final = 24  # Siguiente día
                    carrier_final = best_carrier['carrier']
                else:
                    costo_final = 200
                    tiempo_final = 24
                    carrier_final = "Externo"

            total_cost_consolidated += costo_final
            total_time_consolidated += tiempo_final

            logger.info(f"   → {consolidator_info['nombre_tienda']} → Cliente")
            logger.info(f"     {cantidad_total} unidades consolidadas")
            logger.info(f"     Costo: ${costo_final:.0f}, Tiempo: {tiempo_final:.1f}h")
            logger.info(f"     Carrier: {carrier_final}")
            logger.info(f"   📊 TOTALES: ${total_cost_consolidated:.0f}, {total_time_consolidated:.1f}h total")

            if total_cost_consolidated < total_cost_direct * 0.8:
                consolidated_candidate = {
                    'ruta_id': f"consolidated_{consolidator_info['tienda_id']}",
                    'tipo_ruta': 'consolidada',
                    'origen_principal': f"Múltiples → {consolidator_info['nombre_tienda']} → Cliente",
                    'estrategia': 'consolidacion_hub',
                    'hub_consolidador': consolidator_info['nombre_tienda'],
                    'segmentos': consolidation_steps,
                    'tiempo_total_horas': total_time_consolidated,
                    'costo_total_mxn': total_cost_consolidated,
                    'distancia_total_km': sum(s['distancia'] for s in consolidation_steps) + distance_final,
                    'probabilidad_cumplimiento': 0.75,
                    'cantidad_cubierta': cantidad_total,
                    'paquetes': 1,
                    'ahorro_vs_directo': total_cost_direct - total_cost_consolidated,
                    'factores_aplicados': ['consolidacion',
                                           f'ahorro_${total_cost_direct - total_cost_consolidated:.0f}']
                }
                candidates.append(consolidated_candidate)

                logger.info(f"\n   ✅ CONSOLIDACIÓN VIABLE: Ahorro ${total_cost_direct - total_cost_consolidated:.0f}")

        # ESTRATEGIA 3: HÍBRIDA (locales directo, remotas por carrier)
        if remote_stores and local_stores:
            logger.info(f"\n📦 ESTRATEGIA 3: Híbrida (locales FI, remotas carrier)")
            logger.info(f"{'-' * 50}")

            total_cost_hybrid = 0
            total_time_hybrid = 0
            hybrid_info = []

            for local in local_stores:
                direct_local = await self._create_direct_route_dynamic(
                    local, (target_lat, target_lon), external_factors, request, cp_info
                )
                if direct_local:
                    total_cost_hybrid += direct_local['costo_total_mxn']
                    total_time_hybrid = max(total_time_hybrid, direct_local['tiempo_total_horas'])
                    hybrid_info.append({
                        'tipo': 'local_FI',
                        'origen': local['store_info']['nombre_tienda'],
                        'cantidad': local['cantidad'],
                        'costo': direct_local['costo_total_mxn']
                    })

            for remote in remote_stores:
                peso_envio = self._calculate_shipment_weight(request, remote['cantidad'])
                # Buscar carrier express (no el más barato)
                carriers = self.repos.fleet.get_best_carriers_for_cp(request.codigo_postal, peso_envio)

                express_carrier = None
                for carrier in carriers:
                    if 'Express' in carrier.get('tipo_servicio', '') or carrier.get('tiempo_entrega_dias_habiles',
                                                                                    '5') <= '2':
                        express_carrier = carrier
                        break

                if express_carrier:
                    costo_express = float(express_carrier['costo_base_mxn']) * 1.3  # Premium
                    tiempo_express = 48  # 2 días

                    total_cost_hybrid += costo_express
                    total_time_hybrid = max(total_time_hybrid, tiempo_express)

                    hybrid_info.append({
                        'tipo': 'remota_express',
                        'origen': remote['store_info']['nombre_tienda'],
                        'cantidad': remote['cantidad'],
                        'costo': costo_express,
                        'carrier': express_carrier['carrier']
                    })

            logger.info(f"   Envíos locales FI: {len([h for h in hybrid_info if h['tipo'] == 'local_FI'])}")
            logger.info(f"   Envíos express: {len([h for h in hybrid_info if h['tipo'] == 'remota_express'])}")
            logger.info(f"   📊 TOTALES: ${total_cost_hybrid:.0f}, {total_time_hybrid:.1f}h máximo")

        logger.info(f"\n🎯 RESUMEN DE ESTRATEGIAS:")
        logger.info(f"{'=' * 60}")
        logger.info(f"1. MÚLTIPLES PAQUETES: ${total_cost_direct:.0f} / {total_time_direct:.1f}h")
        if 'total_cost_consolidated' in locals():
            logger.info(f"2. CONSOLIDACIÓN HUB: ${total_cost_consolidated:.0f} / {total_time_consolidated:.1f}h")
            logger.info(f"   → Ahorro: ${total_cost_direct - total_cost_consolidated:.0f}")
        if 'total_cost_hybrid' in locals():
            logger.info(f"3. HÍBRIDA: ${total_cost_hybrid:.0f} / {total_time_hybrid:.1f}h")

        return candidates


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

        zona_seguridad_cp = cp_info.get('zona_seguridad', 'Verde')
        zona_seguridad_tienda = tienda_info.get('zona_seguridad', 'Verde')
        zonas_orden = {'Verde': 1, 'Amarilla': 2, 'Roja': 3}
        zona_final = zona_seguridad_cp if zonas_orden.get(zona_seguridad_cp, 1) >= zonas_orden.get(
            zona_seguridad_tienda, 1) else zona_seguridad_tienda

        logger.info(f"🛡️ Zona seguridad: CP={zona_seguridad_cp}, Tienda={zona_seguridad_tienda}, Final={zona_final}")

        store_lat, store_lon = GeoCalculator.fix_corrupted_coordinates(
            float(tienda_info['latitud']), float(tienda_info['longitud'])
        )

        distance_km = GeoCalculator.calculate_distance_km(
            store_lat, store_lon, target_coords[0], target_coords[1]
        )

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

        travel_time = self._calculate_travel_time_dynamic(distance_km, fleet_type, external_factors)
        prep_time = float(tienda_info.get('tiempo_prep_horas', 1.0))
        tiempo_extra = external_factors.get('impacto_tiempo_extra_horas', 0)
        total_time = prep_time + travel_time + tiempo_extra

        if fleet_type == 'FE' and carriers:
            cost = self._calculate_external_fleet_cost(
                carriers[0], peso_kg, distance_km, external_factors
            )
        else:
            cost = self._calculate_internal_fleet_cost(
                distance_km, plan_item['cantidad'], external_factors
            )

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
                'zona_seguridad': zona_final
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
                f"zona_{zona_final}",
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


    @staticmethod
    def _calculate_travel_time_dynamic(distance_km: float, fleet_type: str,
                                       external_factors: Dict[str, Any]) -> float:
        """ Calcula tiempo de viaje dinámico"""
        return GeoCalculator.calculate_travel_time(
            distance_km,
            fleet_type,
            external_factors.get('trafico_nivel', 'Moderado'),
            external_factors.get('condicion_clima', 'Templado')
        )

    @staticmethod
    def _calculate_external_fleet_cost(carrier_info: Dict[str, Any],
                                       peso_kg: float, distance_km: float,
                                       external_factors: Dict[str, Any]) -> float:
        """💰 Cálculo de costo flota externa usando datos REALES del CSV"""

        costo_base = float(carrier_info['costo_base_mxn'])
        peso_min = float(carrier_info['peso_min_kg'])
        costo_por_kg = float(carrier_info['costo_por_kg_adicional'])
        peso_extra = max(0, peso_kg - peso_min)
        costo_peso_extra = peso_extra * costo_por_kg
        distance_factor = 1.0 + (distance_km / 500) * 0.1  # 10% cada 500km

        subtotal = (costo_base + costo_peso_extra) * distance_factor

        factor_demanda = external_factors.get('factor_demanda', 1.0)
        impacto_costo_pct = external_factors.get('impacto_costo_extra_pct', 0) / 100
        subtotal *= factor_demanda

        if impacto_costo_pct > 0:
            subtotal *= (1 + impacto_costo_pct)

        final_cost = round(subtotal, 2)

        logger.info(f"💰 Costo externo CSV: base=${costo_base} × demanda={factor_demanda:.2f} = ${final_cost}")

        return final_cost

    @staticmethod
    def _calculate_internal_fleet_cost(distance_km: float, cantidad: int,
                                       external_factors: Dict[str, Any]) -> float:
        """Calcula costo de flota interna
        Es decir los factores
        """
        base_cost = distance_km * 12.0
        quantity_factor = 0.9 if cantidad >= 3 else 1.0
        demand_factor = external_factors.get('factor_demanda', 1.0)
        total_cost = base_cost * quantity_factor * demand_factor
        return round(max(50.0, total_cost), 2)

    def _get_comprehensive_external_factors(self, fecha: datetime, codigo_postal: str) -> Dict[str, Any]:
        """🎯 Factores externos """

        factores_csv = self.repos.external_factors.get_factors_for_date_and_cp(fecha, codigo_postal)
        cp_info = self.repos.store._get_postal_info(codigo_postal)

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



    @staticmethod
    def _calculate_cost_impact_from_csv(factores_csv: Dict[str, Any]) -> float:
        """💰 Calcula impacto en costo desde datos CSV"""
        factor_demanda = factores_csv.get('factor_demanda', 1.0)
        evento = factores_csv.get('evento_detectado', 'Normal')
        impacto = max(0, (factor_demanda - 1.0) * 20)  # 20% por punto de demanda

        eventos_premium = ['Viernes_Santo', 'Jueves_Santo', 'Dia_Padre', 'Navidad']
        if any(evento_premium in evento for evento_premium in eventos_premium):
            impacto += 15  # 15% extra

        return round(min(50.0, impacto), 1)


    @staticmethod
    def _calculate_probability_dynamic(distance_km: float, total_time: float,
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

    def _calculate_shipment_weight(self, request: PredictionRequest, cantidad: int) -> float:
        """⚖️ Calcula peso del envío"""
        product = self.repos.product.get_product_by_sku(request.sku_id)
        peso_unitario = product.get('peso_kg', 0.5) if product else 0.5
        return peso_unitario * cantidad

    @staticmethod
    def _rank_candidates_dynamic(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """🏆 Rankea candidatos y captura información detallada para JSON response"""

        if not candidates:
            return []

        logger.info(f"🏆 Iniciando ranking de {len(candidates)} candidatos...")

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

        # TODO -> AQUI ESTAN LOS PESOS que vienen  desde el settings
        peso_tiempo = settings.PESO_TIEMPO
        peso_costo = settings.PESO_COSTO
        peso_probabilidad = settings.PESO_PROBABILIDAD
        peso_distancia = settings.PESO_DISTANCIA

        rutas_evaluadas = {
            'pesos_utilizados': {
                'tiempo': peso_tiempo,
                'costo': peso_costo,
                'stock': peso_probabilidad,  # Nota: probabilidad se mapea a "stock" en la respuesta
                'distancia': peso_distancia
            },
            'rangos_metricas': {
                'tiempo_min_h': round(min_tiempo, 1),
                'tiempo_max_h': round(max_tiempo, 1),
                'costo_min_mxn': round(min_costo, 0),
                'costo_max_mxn': round(max_costo, 0),
                'distancia_min_km': round(min_distancia, 1),
                'distancia_max_km': round(max_distancia, 1)
            },
            'candidatos_evaluados': [],
            'ranking_final': []
        }

        for i, candidate in enumerate(candidates, 1):
            # Scores normalizados (0-1)
            score_tiempo = 1 - (candidate['tiempo_total_horas'] - min_tiempo) / max(1, max_tiempo - min_tiempo)
            score_costo = 1 - (candidate['costo_total_mxn'] - min_costo) / max(1, max_costo - min_costo)
            score_distancia = 1 - (candidate['distancia_total_km'] - min_distancia) / max(1,
                                                                                          max_distancia - min_distancia)
            score_probabilidad = candidate['probabilidad_cumplimiento']

            score_combinado = (
                    peso_tiempo * score_tiempo +
                    peso_costo * score_costo +
                    peso_probabilidad * score_probabilidad +
                    peso_distancia * score_distancia
            )

            if candidate['tipo_ruta'] == 'directa':
                score_combinado += 0.05
            score_combinado = min(1.0, score_combinado)

            candidate['score_lightgbm'] = round(score_combinado, 4)
            candidate['score_breakdown'] = {
                'tiempo': round(score_tiempo, 3),
                'costo': round(score_costo, 3),
                'distancia': round(score_distancia, 3),
                'probabilidad': round(score_probabilidad, 3)
            }
            candidato_info = {
                'posicion_evaluacion': i,
                'nombre_origen': candidate.get('origen_principal', f"Candidato {i}"),
                'tipo_ruta': candidate['tipo_ruta'],
                'distancia_km': round(candidate['distancia_total_km'], 1),
                'tiempo_h': round(candidate['tiempo_total_horas'], 1),
                'costo_mxn': round(candidate['costo_total_mxn'], 0),
                'probabilidad': round(candidate['probabilidad_cumplimiento'], 3),
                'scores_individuales': candidate['score_breakdown'],
                'score_final': candidate['score_lightgbm'],
                'fleet_type': candidate.get('segmentos', [{}])[0].get('tipo_flota', 'FI') if candidate.get(
                    'segmentos') else 'FI',
                'carrier': candidate.get('segmentos', [{}])[0].get('carrier', 'Liverpool') if candidate.get(
                    'segmentos') else 'Liverpool',
                'stock_local': candidate.get('has_local_stock', False),
                'factores_aplicados': candidate.get('factores_aplicados', [])
            }

            rutas_evaluadas['candidatos_evaluados'].append(candidato_info)
            origen_name = candidate.get('origen_principal', f"Candidato {i}")
            logger.info(f"📊 Evaluando: {origen_name}")
            logger.info(f"   🔢 Scores individuales:")
            logger.info(f"      → Tiempo: {score_tiempo:.3f} (peso: {peso_tiempo})")
            logger.info(f"      → Costo: {score_costo:.3f} (peso: {peso_costo})")
            logger.info(f"      → Distancia: {score_distancia:.3f} (peso: {peso_distancia})")
            logger.info(f"      → Probabilidad: {score_probabilidad:.3f} (peso: {peso_probabilidad})")
            logger.info(f"   🎯 Score final: {score_combinado:.4f}")
            logger.info(
                f"   💡 Ventajas: {candidate['tipo_ruta']}, {candidate['tiempo_total_horas']:.1f}h, ${candidate['costo_total_mxn']:.0f}")
        ranked = sorted(candidates, key=lambda x: x['score_lightgbm'], reverse=True)

        for i, candidate in enumerate(ranked, 1):
            ranking_info = {
                'posicion_final': i,
                'ruta_id': candidate['ruta_id'],
                'nombre_origen': candidate.get('origen_principal', f"Ruta {i}"),
                'score_final': candidate['score_lightgbm'],
                'tiempo_h': round(candidate['tiempo_total_horas'], 1),
                'costo_mxn': round(candidate['costo_total_mxn'], 0),
                'tipo_ruta': candidate['tipo_ruta'],
                'probabilidad': round(candidate['probabilidad_cumplimiento'], 3),
                'es_ganador': i == 1
            }

            if i == 1:
                ranking_info['razones_victoria'] = []
                if candidate['score_lightgbm'] >= 0.9:
                    ranking_info['razones_victoria'].append(f"Score excelente ({candidate['score_lightgbm']:.3f})")
                if candidate['tipo_ruta'] == 'directa':
                    ranking_info['razones_victoria'].append("Ruta directa sin transbordos")
                if candidate['tiempo_total_horas'] <= min_tiempo * 1.1:
                    ranking_info['razones_victoria'].append(
                        f"Tiempo competitivo ({candidate['tiempo_total_horas']:.1f}h)")
                if candidate['probabilidad_cumplimiento'] >= 0.8:
                    ranking_info['razones_victoria'].append(
                        f"Alta confiabilidad ({candidate['probabilidad_cumplimiento']:.1%})")
                if candidate['costo_total_mxn'] <= min_costo * 1.2:
                    ranking_info['razones_victoria'].append(f"Costo eficiente (${candidate['costo_total_mxn']:.0f})")

                if not ranking_info['razones_victoria']:
                    ranking_info['razones_victoria'].append("Mejor opción disponible")

            rutas_evaluadas['ranking_final'].append(ranking_info)

        logger.info("🏆 RANKING FINAL:")
        logger.info("   Pos | Tienda/Ruta              | Score  | Tiempo | Costo  | Tipo")
        logger.info("   ----|--------------------------|--------|--------|--------|------------")

        for i, candidate in enumerate(ranked, 1):
            origen_name = candidate.get('origen_principal', f"Ruta {i}")[:20].ljust(20)
            score = candidate['score_lightgbm']
            tiempo = candidate['tiempo_total_horas']
            costo = candidate['costo_total_mxn']
            tipo = candidate['tipo_ruta'][:10]

            logger.info(f"   {i:2d}. | {origen_name} | {score:.3f} | {tiempo:5.1f}h | ${costo:6.0f} | {tipo}")

            if i == 1:
                logger.info(f"   🎯 GANADOR: {candidate.get('origen_principal', 'N/A')}")
                logger.info(f"      → Razones principales:")

                if score >= 0.9:
                    logger.info(f"      → Score excelente ({score:.3f}) - Óptimo en múltiples métricas")
                elif candidate['tipo_ruta'] == 'directa':
                    logger.info(f"      → Ruta directa - Sin transbordos ni CEDIS intermedio")

                if tiempo <= min_tiempo * 1.1:
                    logger.info(f"      → Tiempo competitivo ({tiempo:.1f}h)")
                if costo <= min_costo * 1.2:
                    logger.info(f"      → Costo eficiente (${costo:.0f})")
                if candidate['probabilidad_cumplimiento'] >= 0.8:
                    logger.info(f"      → Alta confiabilidad ({candidate['probabilidad_cumplimiento']:.1%})")

        for candidate in ranked:
            candidate['rutas_evaluadas_detalle'] = rutas_evaluadas

        return ranked


    def _calculate_dynamic_fee(self, selected_route: Dict[str, Any],
                               request: PredictionRequest,
                               external_factors: Dict[str, Any],
                               cp_info: Dict[str, Any],
                               # NUEVO: Agregar stock_analysis
                               stock_analysis: Dict[str, Any] = None) -> FEECalculation:
        """📅 Calcula FEE dinámico"""

        tiempo_total = selected_route['tiempo_total_horas']


        tipo_entrega = self._determine_delivery_type(
            tiempo_total, request.fecha_compra, external_factors, cp_info,
            selected_route.get('distancia_total_km', 999),
            selected_route.get('has_local_stock', False),
            allocation_plan=stock_analysis.get('allocation_plan', []) if stock_analysis else [],
            selected_route=selected_route
        )

        fecha_entrega = self._calculate_delivery_date(
            request.fecha_compra, tiempo_total, tipo_entrega, external_factors
        )

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

    @staticmethod
    def _determine_delivery_type(tiempo_horas: float, fecha_compra: datetime,
                                 external_factors: Dict[str, Any], cp_info: Dict[str, Any],
                                 distance_km: float, has_local_stock: bool,
                                 allocation_plan: List[Dict[str, Any]] = None,
                                 selected_route: Dict[str, Any] = None) -> TipoEntregaEnum:
        """📦 Determina tipo de entrega CORREGIDO con lógica real de negocio"""

        hora_compra = fecha_compra.hour
        factor_demanda = external_factors.get('factor_demanda', 1.0)
        zona = cp_info.get('zona_seguridad', 'Verde')
        cobertura = cp_info.get('cobertura_liverpool', False)

        is_split_inventory = len(allocation_plan) > 1 if allocation_plan else False
        max_distance_in_plan = 0
        total_cantidad = 0

        if allocation_plan:
            for plan in allocation_plan:
                max_distance_in_plan = max(max_distance_in_plan, plan.get('distancia_km', 0))
                total_cantidad += plan.get('cantidad', 0)

        is_complex_route = selected_route and selected_route.get('tipo_ruta') in ['compleja_cedis',
                                                                                  'multi_segmento_cedis']

        logger.info(f"📦 Lógica mejorada:")
        logger.info(f"   Hora compra: {hora_compra}h, Distancia max: {max_distance_in_plan:.1f}km")
        logger.info(f"   Stock local: {has_local_stock}, Split inventory: {is_split_inventory}")
        logger.info(
            f"   Cantidad total: {total_cantidad}, Tiendas involucradas: {len(allocation_plan) if allocation_plan else 0}")
        logger.info(f"   Ruta compleja: {is_complex_route}")

        # REGLA 1: FLASH
        if (hora_compra < 12 and
                has_local_stock and
                not is_split_inventory and  # NUEVO: NO split inventory
                max_distance_in_plan <= 50 and  # NUEVO: Usar distancia máxima real
                total_cantidad <= 10 and  # NUEVO: Cantidad manejable
                factor_demanda <= 1.2 and
                zona in ['Verde', 'Amarilla'] and
                cobertura and
                not is_complex_route):  # NUEVO: NO rutas complejas

            logger.info("   → FLASH: Entrega mismo día (condiciones ideales)")
            return TipoEntregaEnum.FLASH

        # REGLA 2: EXPRESS
        elif (hora_compra < 20 and
              has_local_stock and
              max_distance_in_plan <= 100 and  # NUEVO: Todas las tiendas cercanas
              total_cantidad <= 50 and  # NUEVO: Cantidad moderada
              len(allocation_plan) <= 2 if allocation_plan else True and  # NUEVO: Máximo 2 tiendas
                                                                factor_demanda <= 2.0 and
                                                                zona in ['Verde', 'Amarilla'] and
                                                                not is_complex_route):

            logger.info("   → EXPRESS: Siguiente día hábil")
            return TipoEntregaEnum.EXPRESS

        # REGLA 3: STANDARD
        elif (tiempo_horas <= 72 and
              not is_complex_route and
              (not allocation_plan or len(allocation_plan) <= 3)):  # NUEVO: Máximo 3 tiendas

            logger.info("   → STANDARD: 2-3 días (ruteo moderado)")
            return TipoEntregaEnum.STANDARD

        # REGLA 4: PROGRAMADA - Casos muy complejos
        else:
            razones = []
            if is_split_inventory and len(allocation_plan) > 3:
                razones.append(f"split desde {len(allocation_plan)} tiendas")
            if max_distance_in_plan > 500:
                razones.append(f"distancia máxima {max_distance_in_plan:.0f}km")
            if is_complex_route:
                razones.append("ruteo con CEDIS")
            if total_cantidad > 50:
                razones.append(f"cantidad alta ({total_cantidad} unidades)")

            logger.info(f"   → PROGRAMADA: Caso complejo - {', '.join(razones)}")
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

        else:
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

        origen_store = await self._get_store_info(stock_plan[0]['tienda_id'])
        optimal_cedis = await self._find_optimal_cedis_real(origen_store, codigo_postal)
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

        #  factores externos
        factor_tiempo = external_factors.get('impacto_tiempo_extra_horas', 0)
        total_time += factor_tiempo

        # Probabilidad para rutas complejas
        probability = max(0.65, 0.85 - (len(route_segments) * 0.05))

        result = {
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

        if 'cedis_analysis' in optimal_cedis:
            result['cedis_analysis'] = optimal_cedis['cedis_analysis']

        return result

    async def _find_optimal_cedis_real(self, origen_store: Dict[str, Any], codigo_postal: str) -> Dict[str, Any]:
        """🏭 Encuentra CEDIS óptimo REAL y captura información detallada para JSON"""

        cedis_df = self.repos.data_manager.get_data('cedis')
        cp_info = self.repos.store._get_postal_info(codigo_postal)
        estado_destino = cp_info.get('estado_alcaldia', '').split()[0]

        logger.info(f"🏭 Análisis de CEDIS para ruteo complejo:")
        logger.info(f"   📍 Origen: {origen_store['nombre_tienda']} ({origen_store['tienda_id']})")
        logger.info(f"   📍 Destino: CP {codigo_postal} ({estado_destino})")

        cedis_analysis = {
            'origen_tienda': {
                'id': origen_store['tienda_id'],
                'nombre': origen_store['nombre_tienda'],
                'coordenadas': {
                    'lat': float(origen_store['latitud']),
                    'lon': float(origen_store['longitud'])
                }
            },
            'destino_info': {
                'codigo_postal': codigo_postal,
                'estado_destino': estado_destino,
                'coordenadas': {
                    'lat': float(cp_info['latitud_centro']),
                    'lon': float(cp_info['longitud_centro'])
                }
            },
            'cedis_evaluados': [],
            'cedis_seleccionado': None,
            'total_cedis_disponibles': cedis_df.height,
            'cedis_descartados': []
        }

        cedis_candidates = []

        logger.info(f"🔍 Evaluando {cedis_df.height} CEDIS disponibles...")

        for cedis in cedis_df.to_dicts():
            cobertura = cedis.get('cobertura_estados', '')

            cedis_eval = {
                'cedis_id': cedis['cedis_id'],
                'nombre': cedis['nombre_cedis'],
                'cobertura_estados': cobertura,
                'cubre_destino': 'Nacional' in cobertura or estado_destino in cobertura,
                'evaluado': False
            }

            if not cedis_eval['cubre_destino']:
                logger.info(f"   ❌ {cedis['nombre_cedis']}: No cubre {estado_destino}")
                cedis_eval['razon_descarte'] = f"No cubre {estado_destino}"
                cedis_analysis['cedis_descartados'].append(cedis_eval)
                continue

            # 2. Corregir coordenadas si están corruptas
            from utils.geo_calculator import GeoCalculator
            cedis_lat, cedis_lon = GeoCalculator.fix_corrupted_coordinates(
                float(cedis['latitud']), float(cedis['longitud'])
            )

            dist_origen_cedis = GeoCalculator.calculate_distance_km(
                float(origen_store['latitud']), float(origen_store['longitud']),
                cedis_lat, cedis_lon
            )

            dist_cedis_destino = GeoCalculator.calculate_distance_km(
                cedis_lat, cedis_lon,
                float(cp_info['latitud_centro']), float(cp_info['longitud_centro'])
            )

            tiempo_proc_num = self._parse_time_range(
                cedis.get('tiempo_procesamiento_horas'), default=2.0
            )

            distancia_total = dist_origen_cedis + dist_cedis_destino
            tiempo_total = tiempo_proc_num + (distancia_total / 60)  # Tiempo aproximado
            cobertura_bonus = 0.8 if estado_destino in cobertura else 1.0
            score = tiempo_total * cobertura_bonus

            cedis_eval.update({
                'evaluado': True,
                'coordenadas': {
                    'lat': cedis_lat,
                    'lon': cedis_lon
                },
                'distancia_origen_cedis_km': round(dist_origen_cedis, 1),
                'distancia_cedis_destino_km': round(dist_cedis_destino, 1),
                'distancia_total_km': round(distancia_total, 1),
                'tiempo_procesamiento_h': tiempo_proc_num,
                'score': round(score, 2),
                'cobertura_especifica': estado_destino in cobertura
            })

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

            cedis_analysis['cedis_evaluados'].append(cedis_eval)

        if not cedis_candidates:
            logger.error(f"❌ No se encontró ningún CEDIS disponible para {estado_destino}")
            return None

        cedis_candidates.sort(key=lambda x: x['score'])
        cedis_analysis['ranking_cedis'] = []
        logger.info(f"🏆 Ranking de CEDIS (mejores 3):")

        for i, cedis in enumerate(cedis_candidates[:5], 1):
            logger.info(f"   {i}. {cedis['nombre_cedis']}")
            logger.info(f"      → Score: {cedis['score']:.2f}")
            logger.info(f"      → Distancia total: {cedis['distancia_total']:.1f}km")
            logger.info(f"      → Cobertura específica: {'Sí' if cedis['cobertura_match'] else 'No'}")

            cedis_analysis['ranking_cedis'].append({
                'posicion': i,
                'cedis_id': cedis['cedis_id'],
                'nombre': cedis['nombre_cedis'],
                'score': round(cedis['score'], 2),
                'distancia_total_km': round(cedis['distancia_total'], 1),
                'cobertura_especifica': cedis['cobertura_match'],
                'seleccionado': i == 1
            })

        best_cedis = cedis_candidates[0]
        cedis_analysis['cedis_seleccionado'] = {
            'cedis_id': best_cedis['cedis_id'],
            'nombre': best_cedis['nombre_cedis'],
            'score': round(best_cedis['score'], 2),
            'distancia_total_km': round(best_cedis['distancia_total'], 1),
            'tiempo_procesamiento_h': best_cedis['tiempo_procesamiento_num'],
            'cobertura_especifica': best_cedis['cobertura_match'],
            'razon_seleccion': f"Menor score total ({best_cedis['score']:.2f})",
            'coordenadas': {
                'lat': best_cedis['latitud_corregida'],
                'lon': best_cedis['longitud_corregida']
            }
        }

        logger.info(f"✅ CEDIS SELECCIONADO: {best_cedis['nombre_cedis']}")
        logger.info(f"   🎯 Razón: Menor score total ({best_cedis['score']:.2f})")
        logger.info(f"   📏 Distancia combinada: {best_cedis['distancia_total']:.1f}km")
        logger.info(f"   ⏱️ Tiempo procesamiento: {best_cedis['tiempo_procesamiento_num']:.1f}h")
        logger.info(f"   🌍 Cobertura {estado_destino}: {'Directa' if best_cedis['cobertura_match'] else 'Nacional'}")

        best_cedis['cedis_analysis'] = cedis_analysis

        return best_cedis

    async def _find_closest_store_to_cp(self, codigo_postal: str) -> Dict[str, Any]:
        """🏪 Encuentra tienda Liverpool más cercana al CP destino"""

        nearby_stores = self.repos.store.find_stores_by_postal_range(codigo_postal)

        if not nearby_stores:
            logger.error(f"❌ No hay tiendas Liverpool cerca de {codigo_postal}")
            return None

        closest_store = nearby_stores[0]
        logger.info(f"🏪 Tienda destino más cercana: {closest_store['nombre_tienda']} "
                    f"({closest_store['distancia_km']:.1f}km del CP)")

        return closest_store

    async def _calculate_real_segment(self, origen: Dict[str, Any], destino: Dict[str, Any],
                                      tipo_flota: str, external_factors: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula segmento REAL usando datos de CSV"""

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

        distance = GeoCalculator.calculate_distance_km(orig_lat, orig_lon, dest_lat, dest_lon)
        travel_time = GeoCalculator.calculate_travel_time(
            distance, tipo_flota,
            external_factors.get('trafico_nivel', 'Moderado'),
            external_factors.get('condicion_clima', 'Templado')
        )

        if tipo_flota == 'FI':
            costo_base = distance * 15.0  # $15 por km flota interna

            factor_demanda = external_factors.get('factor_demanda', 1.0)
            costo_final = costo_base * factor_demanda

            carrier = 'Liverpool'
        else:
            flota_df = self.repos.data_manager.get_data('flota_externa')
            carriers = flota_df.filter(pl.col('activo') == True).to_dicts()

            if carriers:
                best_carrier = carriers[0]
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

        tienda_lat, tienda_lon = GeoCalculator.fix_corrupted_coordinates(
            float(tienda_destino['latitud']), float(tienda_destino['longitud'])
        )

        final_distance = GeoCalculator.calculate_distance_km(
            tienda_lat, tienda_lon, target_coords[0], target_coords[1]
        )

        flota_df = self.repos.data_manager.get_data('flota_externa')
        peso_estimado = 1.0  # Peso promedio del paquete

        cp_int = int(codigo_postal)
        available_carriers = []

        for carrier in flota_df.to_dicts():
            if (carrier.get('activo', False) and
                    carrier.get('zona_cp_inicio', 0) <= cp_int <= carrier.get('zona_cp_fin', 99999) and
                    carrier.get('peso_min_kg', 0) <= peso_estimado <= carrier.get('peso_max_kg', 100)):
                available_carriers.append(carrier)

        if available_carriers:
            best_carrier = min(available_carriers, key=lambda x: x.get('costo_base_mxn', 999))

            costo_base = float(best_carrier['costo_base_mxn'])

            # TODO-> CORRECCIÓN:  rangos de tiempo como "3-5"
            dias_entrega = int(self._parse_time_range(
                best_carrier.get('tiempo_entrega_dias_habiles'), default=2.0
            ))

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

    @staticmethod
    def _parse_time_range(time_value: Any, default: float = 2.0) -> float:
        """🕐 Parser robusto para rangos de tiempo del CSV"""

        if time_value is None:
            return default

        time_str = str(time_value).strip()

        if not time_str or time_str.lower() in ['nan', 'null', '']:
            return default

        try:
            if '-' in time_str:
                parts = time_str.split('-')
                return float(parts[0])
            else:
                return float(time_str)
        except (ValueError, IndexError) as e:
            logger.warning(f"⚠️ Error parseando tiempo '{time_str}': {e}, usando default {default}")
            return default

    @staticmethod
    def _ensure_business_day(fecha: datetime) -> datetime:
        """📅 Asegura que la fecha sea día hábil"""
        while fecha.weekday() == 6:  # Domingo
            fecha += timedelta(days=1)

        return fecha

    # ✅ CORRECCIÓN: Ventana de 5 horas como solicitado
    @staticmethod
    def _calculate_time_window(fecha_entrega: datetime, tipo_entrega: TipoEntregaEnum) -> Dict[str, dt_time]:
        """🕐 Ventana de entrega AMPLIADA (5 horas)"""

        logger.info(f"🕐 Calculando ventana AMPLIADA para: {fecha_entrega.strftime('%Y-%m-%d %H:%M')}")

        if tipo_entrega == TipoEntregaEnum.FLASH:
            ventana_horas = 3  # ±1.5h para FLASH
        else:
            ventana_horas = 5  # ±2.5h para todos los demás

        inicio_ventana = fecha_entrega - timedelta(hours=ventana_horas // 2)
        fin_ventana = fecha_entrega + timedelta(hours=ventana_horas // 2)

        #  horarios de entrega
        HORA_MIN = 9
        HORA_MAX = 19

        if inicio_ventana.hour < HORA_MIN:
            inicio_ventana = inicio_ventana.replace(hour=HORA_MIN, minute=0)
        if fin_ventana.hour >= HORA_MAX:
            fin_ventana = fin_ventana.replace(hour=HORA_MAX, minute=0)

        if fin_ventana <= inicio_ventana:
            fin_ventana = inicio_ventana + timedelta(hours=2)

        logger.info(
            f"   Ventana AMPLIADA: {inicio_ventana.strftime('%H:%M')} - {fin_ventana.strftime('%H:%M')} ({ventana_horas}h)")

        return {
            'inicio': inicio_ventana.time(),
            'fin': fin_ventana.time()
        }

    @staticmethod
    def _get_next_business_day(fecha: datetime) -> datetime:
        """📅 Obtiene el siguiente día hábil"""
        next_day = fecha + timedelta(days=1)

        while next_day.weekday() == 6:
            next_day += timedelta(days=1)

        return next_day


    @staticmethod
    def _calculate_time_window(fecha_entrega: datetime, tipo_entrega: TipoEntregaEnum) -> Dict[str, dt_time]:
        """🕐 Calcula ventana de entrega CORREGIDA"""

        logger.info(f"🕐 Calculando ventana para: {fecha_entrega.strftime('%Y-%m-%d %H:%M')}")

        if tipo_entrega == TipoEntregaEnum.FLASH:
            ventana_horas = 1  # ±30min para FLASH
        elif tipo_entrega == TipoEntregaEnum.EXPRESS:
            ventana_horas = 2  # ±1h para EXPRESS
        else:
            ventana_horas = 4  # ±2h para STANDARD/PROGRAMADA

        inicio_ventana = fecha_entrega - timedelta(hours=ventana_horas // 2)
        fin_ventana = fecha_entrega + timedelta(hours=ventana_horas // 2)

        # ✅ CORRECCIÓN: Respetar horarios de entrega (10 AM - 6 PM)
        HORA_MIN_ENTREGA = 10
        HORA_MAX_ENTREGA = 18

        if inicio_ventana.hour < HORA_MIN_ENTREGA:
            inicio_ventana = inicio_ventana.replace(hour=HORA_MIN_ENTREGA, minute=0)
        elif inicio_ventana.hour >= HORA_MAX_ENTREGA:
            inicio_ventana = inicio_ventana.replace(hour=HORA_MAX_ENTREGA - 1, minute=0)

        if fin_ventana.hour < HORA_MIN_ENTREGA:
            fin_ventana = fin_ventana.replace(hour=HORA_MIN_ENTREGA + 1, minute=0)
        elif fin_ventana.hour >= HORA_MAX_ENTREGA:
            fin_ventana = fin_ventana.replace(hour=HORA_MAX_ENTREGA, minute=0)

        if fin_ventana <= inicio_ventana:
            fin_ventana = inicio_ventana + timedelta(hours=1)

        logger.info(f"   Ventana: {inicio_ventana.strftime('%H:%M')} - {fin_ventana.strftime('%H:%M')}")

        return {
            'inicio': inicio_ventana.time(),
            'fin': fin_ventana.time()
        }


    @staticmethod
    def _get_main_carrier(route: Dict[str, Any]) -> str:
        """🚚 Obtiene carrier principal"""
        segmentos = route.get('segmentos', [])
        if not segmentos:
            return 'Liverpool'

        for segmento in segmentos:
            if segmento.get('destino') == 'cliente':
                return segmento.get('carrier', 'Liverpool')

        return segmentos[-1].get('carrier', 'Liverpool')