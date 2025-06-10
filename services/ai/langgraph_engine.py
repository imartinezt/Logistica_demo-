# services/ai/langgraph_engine.py
from langgraph.graph import Graph, END
from typing import TypedDict, Dict, List, Optional
import time
from datetime import datetime

from models.schemas import PredictionRequest, Razonamiento
from services.ai.gemini_service import GeminiReasoningService
from utils.logger import logger


class ReasoningState(TypedDict):
    """🧠 Estado compartido del razonamiento LangGraph + Gemini"""
    request: PredictionRequest

    # Datos de contexto
    producto: Dict
    zona_info: Dict
    factores_externos: Dict
    stock_disponible: List[Dict]
    nodos_cercanos: List[Dict]
    rutas_evaluadas: List[Dict]

    # Decisiones de Gemini
    gemini_decisions: Dict
    mejor_ruta: Optional[Dict]

    # Explicabilidad
    explicabilidad: List[Razonamiento]
    warnings: List[str]

    # Metadata
    es_zona_roja: bool
    tiempo_inicio: float
    pasos_completados: int


class LangGraphReasoningEngine:
    """
    🔄 ORQUESTADOR LANGGRAPH + GEMINI

    LangGraph maneja el flujo, Gemini hace el razonamiento inteligente
    """

    def __init__(self, repositories: Dict):
        self.repos = repositories
        self.gemini = GeminiReasoningService()
        self.graph = self._build_reasoning_graph()

    def _build_reasoning_graph(self) -> Graph:
        """🔧 Construye el grafo de razonamiento"""
        workflow = Graph()

        # 🎯 Nodos del workflow (cada uno usa Gemini)
        workflow.add_node("validate_product", self._validate_product_with_gemini)
        workflow.add_node("analyze_zone", self._analyze_zone_with_gemini)
        workflow.add_node("detect_external_factors", self._detect_external_factors)
        workflow.add_node("check_inventory", self._check_inventory_availability)
        workflow.add_node("find_closest_nodes", self._find_closest_distribution_nodes)
        workflow.add_node("evaluate_routes", self._evaluate_feasible_routes)
        workflow.add_node("optimize_selection", self._optimize_route_with_gemini)
        workflow.add_node("finalize_prediction", self._finalize_with_gemini_analysis)

        # 🔄 Flujo del grafo
        workflow.set_entry_point("validate_product")
        workflow.add_edge("validate_product", "analyze_zone")
        workflow.add_edge("analyze_zone", "detect_external_factors")
        workflow.add_edge("detect_external_factors", "check_inventory")
        workflow.add_edge("check_inventory", "find_closest_nodes")
        workflow.add_edge("find_closest_nodes", "evaluate_routes")
        workflow.add_edge("evaluate_routes", "optimize_selection")
        workflow.add_edge("optimize_selection", "finalize_prediction")
        workflow.add_edge("finalize_prediction", END)

        return workflow.compile()

    async def reason(self, request: PredictionRequest) -> ReasoningState:
        """🚀 Ejecuta flujo completo LangGraph + Gemini"""
        initial_state = ReasoningState(
            request=request,
            producto={},
            zona_info={},
            factores_externos={},
            stock_disponible=[],
            nodos_cercanos=[],
            rutas_evaluadas=[],
            gemini_decisions={},
            mejor_ruta=None,
            explicabilidad=[],
            warnings=[],
            es_zona_roja=False,
            tiempo_inicio=time.time(),
            pasos_completados=0
        )

        logger.info("🧠 Iniciando razonamiento LangGraph + Gemini")
        final_state = await self.graph.ainvoke(initial_state)

        tiempo_total = time.time() - final_state['tiempo_inicio']
        logger.info(f"✅ Razonamiento completado en {tiempo_total:.2f}s")

        return final_state

    async def _validate_product_with_gemini(self, state: ReasoningState) -> ReasoningState:
        """🔍 Paso 1: Validación inteligente con Gemini"""
        producto = self.repos['product'].get_product_by_sku(state['request'].sku_id)

        if not producto:
            raise ValueError(f"Producto {state['request'].sku_id} no encontrado")

        # 🧠 Gemini analiza el producto
        gemini_analysis = await self.gemini.validate_product_decision(
            producto,
            state['request'].dict()
        )

        state['producto'] = producto
        state['gemini_decisions']['producto'] = gemini_analysis
        state['pasos_completados'] += 1

        # Construir explicabilidad con insights de Gemini
        state['explicabilidad'].append(Razonamiento(
            paso="1_validacion_producto_gemini",
            decision=gemini_analysis.get('decision', 'Producto validado'),
            factores=gemini_analysis.get('factores_clave', []),
            score=gemini_analysis.get('score_producto', 1.0),
            alternativas=[],  # Lista vacía de diccionarios
            tiempo_procesamiento_ms=(time.time() - state['tiempo_inicio']) * 1000
        ))

        # Agregar warnings si Gemini detecta riesgos
        if 'riesgos' in gemini_analysis:
            state['warnings'].extend(gemini_analysis['riesgos'])

        return state

    async def _analyze_zone_with_gemini(self, state: ReasoningState) -> ReasoningState:
        """🏠 Paso 2: Análisis de zona con Gemini"""
        cp = state['request'].codigo_postal
        zona_info = self.repos['postal_code'].get_cp_info(cp)

        if not zona_info:
            zona_info = {
                'codigo_postal': int(cp),
                'zona': 'Desconocida',
                'nivel_seguridad': 'Bajo',
                'tipo_zona': 'Urbana'
            }
            state['warnings'].append(f"CP {cp} no encontrado")

        # 🧠 Gemini analiza la zona
        gemini_zone_analysis = await self.gemini.analyze_zone_safety(zona_info, cp)

        state['zona_info'] = zona_info
        state['es_zona_roja'] = gemini_zone_analysis.get('es_zona_roja', False)
        state['gemini_decisions']['zona'] = gemini_zone_analysis
        state['pasos_completados'] += 1

        if state['es_zona_roja']:
            state['warnings'].append("⚠️ Zona roja detectada por Gemini")

        state['explicabilidad'].append(Razonamiento(
            paso="2_analisis_zona_gemini",
            decision=gemini_zone_analysis.get('decision', 'Zona analizada'),
            factores=gemini_zone_analysis.get('factores', []),
            score=gemini_zone_analysis.get('score_zona', 0.8),
            alternativas=[],  # Lista vacía de diccionarios
            tiempo_procesamiento_ms=(time.time() - state['tiempo_inicio']) * 1000
        ))

        return state

    async def _detect_external_factors(self, state: ReasoningState) -> ReasoningState:
        """🌤️ Paso 3: Detección + Análisis Gemini de factores externos"""
        from utils.temporal_detector import TemporalFactorDetector

        fecha = state['request'].fecha_compra
        zona = state['zona_info']['zona']

        # Detectar factores automáticamente
        factores_detectados = TemporalFactorDetector.detect_seasonal_factors(fecha)

        # Buscar en CSV o usar detectados
        factores = self.repos['external_factors'].get_factors_or_predict(fecha, zona)
        factores.update(factores_detectados)

        # 🧠 Gemini evalúa criticidad de factores
        gemini_factors_analysis = await self.gemini.evaluate_external_factors(
            factores,
            fecha.isoformat()
        )

        state['factores_externos'] = factores
        state['gemini_decisions']['factores'] = gemini_factors_analysis
        state['pasos_completados'] += 1

        # Warnings basados en análisis de Gemini
        if gemini_factors_analysis.get('criticidad') in ['Alta', 'Crítica']:
            state['warnings'].append(f"🌤️ Factores críticos detectados: {gemini_factors_analysis.get('criticidad')}")

        state['explicabilidad'].append(Razonamiento(
            paso="3_factores_externos_gemini",
            decision=f"Criticidad: {gemini_factors_analysis.get('criticidad', 'Media')}",
            factores=gemini_factors_analysis.get('factores_criticos', []),
            score=gemini_factors_analysis.get('score_factores', 0.7),
            alternativas=[],  # Lista vacía de diccionarios
            tiempo_procesamiento_ms=(time.time() - state['tiempo_inicio']) * 1000
        ))

        return state

    async def _check_inventory_availability(self, state: ReasoningState) -> ReasoningState:
        """📦 Paso 4: Verificación de stock (sin Gemini - lógica directa)"""
        sku_id = state['request'].sku_id
        cantidad = state['request'].cantidad

        stock_df = self.repos['inventory'].get_available_stock(sku_id, cantidad)
        stock_disponible = stock_df.to_dicts()

        if not stock_disponible:
            raise ValueError(f"Sin stock para {cantidad} unidades de {sku_id}")

        state['stock_disponible'] = stock_disponible
        state['pasos_completados'] += 1

        # Convertir las alternativas a formato diccionario
        alternativas_dict = []
        for s in stock_disponible[:3]:
            alternativas_dict.append({
                "nodo_id": s.get('nodo_id', ''),
                "stock_oh": s.get('stock_oh', 0),
                "nombre": s.get('nombre_ubicacion', f"Nodo {s.get('nodo_id', '')}")
            })

        state['explicabilidad'].append(Razonamiento(
            paso="4_verificacion_stock",
            decision=f"Stock en {len(stock_disponible)} ubicaciones",
            factores=[f"Ubicaciones: {[s['nodo_id'] for s in stock_disponible]}"],
            score=min(1.0, len(stock_disponible) / 3.0),
            alternativas=alternativas_dict,
            tiempo_procesamiento_ms=(time.time() - state['tiempo_inicio']) * 1000
        ))

        return state

    async def _find_closest_distribution_nodes(self, state: ReasoningState) -> ReasoningState:
        """📍 Paso 5: Encontrar nodos cercanos (sin Gemini - geográfico)"""
        cp = state['request'].codigo_postal
        postal_df = self.repos['postal_code'].load_data()

        nodos_df = self.repos['node'].find_closest_nodes_to_cp(cp, postal_df)
        nodos_cercanos = nodos_df.to_dicts()

        state['nodos_cercanos'] = nodos_cercanos
        state['pasos_completados'] += 1

        # Convertir las alternativas a formato diccionario
        alternativas_dict = []
        for n in nodos_cercanos[:3]:
            alternativas_dict.append({
                "nodo_id": n.get('nodo_id', ''),
                "nombre": n.get('nombre_ubicacion', ''),
                "distancia_km": round(n.get('distancia_km', 0), 1)
            })

        state['explicabilidad'].append(Razonamiento(
            paso="5_nodos_cercanos",
            decision=f"Nodos por distancia a {cp}",
            factores=[f"Más cercano: {nodos_cercanos[0]['nombre_ubicacion']}"],
            score=1.0,
            alternativas=alternativas_dict,
            tiempo_procesamiento_ms=(time.time() - state['tiempo_inicio']) * 1000
        ))

        return state

    async def _evaluate_feasible_routes(self, state: ReasoningState) -> ReasoningState:
        """🚚 Paso 6: Evaluar rutas factibles (cálculos + filtros)"""
        rutas_evaluadas = []
        stock_nodos = [s['nodo_id'] for s in state['stock_disponible']]

        for nodo_id in stock_nodos:
            rutas_df = self.repos['route'].get_feasible_routes(nodo_id, state['request'].codigo_postal)

            for ruta in rutas_df.to_dicts():
                # Filtrar por zona roja
                if state['es_zona_roja'] and ruta['tipo_flota'] == 'FI':
                    continue

                # Aplicar factores externos
                factor_demanda = state['factores_externos'].get('factor_demanda', 1.0)
                tiempo_ajustado = ruta['tiempo_total_horas'] * factor_demanda
                tiempo_ajustado += state['gemini_decisions'].get('factores', {}).get('impacto_tiempo_horas', 0)

                ruta_evaluada = {
                    **ruta,
                    'nodo_stock': nodo_id,
                    'tiempo_ajustado': tiempo_ajustado,
                    'factible': True
                }
                rutas_evaluadas.append(ruta_evaluada)

        if not rutas_evaluadas:
            raise ValueError("Sin rutas factibles")

        state['rutas_evaluadas'] = rutas_evaluadas
        state['pasos_completados'] += 1

        # Convertir las alternativas a formato diccionario
        alternativas_dict = []
        for r in rutas_evaluadas[:5]:
            alternativas_dict.append({
                "ruta_id": r.get('ruta_id', ''),
                "tiempo_horas": round(r.get('tiempo_ajustado', 0), 1),
                "costo_mxn": round(r.get('costo_total_mxn', 0), 2),
                "tipo_flota": r.get('tipo_flota', '')
            })

        state['explicabilidad'].append(Razonamiento(
            paso="6_evaluacion_rutas",
            decision=f"{len(rutas_evaluadas)} rutas factibles",
            factores=[
                f"Tiempo promedio: {sum(r['tiempo_ajustado'] for r in rutas_evaluadas) / len(rutas_evaluadas):.1f}h"],
            score=min(1.0, len(rutas_evaluadas) / 10.0),
            alternativas=alternativas_dict,
            tiempo_procesamiento_ms=(time.time() - state['tiempo_inicio']) * 1000
        ))

        return state

    async def _optimize_route_with_gemini(self, state: ReasoningState) -> ReasoningState:
        """🎯 Paso 7: Optimización inteligente con Gemini"""
        contexto = {
            "producto": state['producto'],
            "zona_info": state['zona_info'],
            "factores_externos": state['factores_externos'],
            "es_zona_roja": state['es_zona_roja'],
            "decisions_previas": state['gemini_decisions']
        }

        # 🧠 Gemini optimiza la selección
        gemini_optimization = await self.gemini.optimize_route_selection(
            state['rutas_evaluadas'],
            contexto
        )

        # Encontrar la ruta recomendada por Gemini
        ruta_optima_id = gemini_optimization.get('ruta_optima_id')
        mejor_ruta = next(
            (r for r in state['rutas_evaluadas'] if r['ruta_id'] == ruta_optima_id),
            state['rutas_evaluadas'][0]  # Fallback
        )

        state['mejor_ruta'] = mejor_ruta
        state['gemini_decisions']['optimizacion'] = gemini_optimization
        state['pasos_completados'] += 1

        # Convertir las alternativas a formato diccionario
        alternativas_dict = []
        if 'alternativas' in gemini_optimization and isinstance(gemini_optimization['alternativas'], list):
            for alt in gemini_optimization['alternativas'][:2]:
                if isinstance(alt, str):
                    alternativas_dict.append({"descripcion": alt})
                else:
                    alternativas_dict.append(alt)

        state['explicabilidad'].append(Razonamiento(
            paso="7_optimizacion_gemini",
            decision=gemini_optimization.get('razon_seleccion', 'Ruta optimizada'),
            factores=gemini_optimization.get('factores_decisivos', []),
            score=gemini_optimization.get('score_final', 0.8),
            alternativas=alternativas_dict,
            tiempo_procesamiento_ms=(time.time() - state['tiempo_inicio']) * 1000
        ))

        return state

    async def _finalize_with_gemini_analysis(self, state: ReasoningState) -> ReasoningState:
        """🏁 Paso 8: Análisis final con Gemini"""
        estado_completo = {
            "request": state['request'].dict(),
            "mejor_ruta": state['mejor_ruta'],
            "todas_las_decisiones": state['gemini_decisions'],
            "warnings": state['warnings'],
            "pasos_completados": state['pasos_completados']
        }

        # 🧠 Gemini genera análisis final
        gemini_final_analysis = await self.gemini.finalize_prediction_analysis(estado_completo)

        state['gemini_decisions']['analisis_final'] = gemini_final_analysis
        state['pasos_completados'] += 1

        # Agregar alertas finales de Gemini
        if 'alertas' in gemini_final_analysis:
            state['warnings'].extend(gemini_final_analysis['alertas'])

        state['explicabilidad'].append(Razonamiento(
            paso="8_analisis_final_gemini",
            decision=gemini_final_analysis.get('resumen_ejecutivo', 'Predicción finalizada'),
            factores=gemini_final_analysis.get('recomendaciones', []),
            score=gemini_final_analysis.get('confianza_prediccion', 0.85),
            alternativas=[],  # Lista vacía de diccionarios
            tiempo_procesamiento_ms=(time.time() - state['tiempo_inicio']) * 1000
        ))

        return state