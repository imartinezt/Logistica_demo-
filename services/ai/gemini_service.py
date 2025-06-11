import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession

from config.settings import settings
from utils.logger import logger


class VertexAIModelSingleton:
    """🧠 Singleton para Gemini optimizado para decisiones logísticas"""
    _model = None
    _chat_session = None

    @classmethod
    def get_model(cls) -> GenerativeModel:
        if cls._model is None:
            logger.info("🧠 Inicializando Gemini 2.0 Flash para decisiones logísticas")
            try:
                vertexai.init(project=settings.PROJECT_ID, location=settings.REGION)
                cls._model = GenerativeModel(settings.MODEL_NAME)
            except Exception as e:
                logger.error(f"❌ Error inicializando Gemini: {e}")
                cls._model = None
        return cls._model

    @classmethod
    def get_chat_session(cls) -> ChatSession:
        if cls._chat_session is None and cls.get_model():
            cls._chat_session = cls.get_model().start_chat()
        return cls._chat_session


class GeminiLogisticsDecisionEngine:
    """🎯 Motor de decisión logística con Gemini especializado en optimización de rutas"""

    def __init__(self):
        self.model = VertexAIModelSingleton.get_model()
        self.decision_context = {}

    def _serialize_for_json(self, obj: Any) -> Any:
        """🔧 Serialización robusta para JSON"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._serialize_for_json(obj.__dict__)
        else:
            return obj

    async def select_optimal_route(self,
                                   top_candidates: List[Dict[str, Any]],
                                   request_context: Dict[str, Any],
                                   external_factors: Dict[str, Any]) -> Dict[str, Any]:
        """🏆 Selección final de ruta óptima por Gemini"""

        if not top_candidates:
            raise ValueError("No hay candidatos para evaluar")

        if len(top_candidates) == 1:
            return {
                'candidato_seleccionado': top_candidates[0],
                'razonamiento': 'Único candidato disponible tras optimización',
                'confianza_decision': 0.85,
                'factores_decisivos': ['unica_opcion_factible'],
                'candidatos_evaluados': top_candidates,
                'timestamp_decision': datetime.now().isoformat(),
                'alertas_operativas': []
            }

        # Verificar si Gemini está disponible
        if not self.model:
            logger.warning("⚠️ Gemini no disponible, usando selección automática")
            return self._fallback_decision(top_candidates)

        try:
            # Preparar contexto completo para Gemini
            context_data = self._serialize_for_json({
                'request': request_context,
                'factores_externos': external_factors,
                'candidatos': top_candidates,
                'business_rules': settings.DELIVERY_RULES,
                'weights': {
                    'tiempo': settings.PESO_TIEMPO,
                    'costo': settings.PESO_COSTO,
                    'probabilidad': settings.PESO_PROBABILIDAD,
                    'distancia': settings.PESO_DISTANCIA
                }
            })

            prompt = self._build_route_selection_prompt(context_data)

            # Timeout más corto para evitar bloqueos
            response = await asyncio.wait_for(
                self.model.generate_content_async(prompt),
                timeout=10.0  # 10 segundos máximo
            )

            decision = self._parse_json_response(response.text)

            # Validar que la decisión sea válida
            selected_id = decision.get('candidato_seleccionado_id')
            selected_candidate = None

            for candidate in top_candidates:
                if candidate['ruta_id'] == selected_id:
                    selected_candidate = candidate
                    break

            if not selected_candidate:
                logger.warning(f"❌ Gemini seleccionó ID inválido: {selected_id}")
                selected_candidate = top_candidates[0]  # Fallback al mejor por score
                decision['razonamiento'] = f"Fallback al mejor candidato (ID inválido: {selected_id})"

            decision['candidato_seleccionado'] = selected_candidate
            decision['candidatos_evaluados'] = top_candidates
            decision['timestamp_decision'] = datetime.now().isoformat()

            logger.info(f"🧠 Gemini seleccionó ruta: {selected_candidate['ruta_id']}")
            return decision

        except asyncio.TimeoutError:
            logger.warning("⏰ Timeout en Gemini, usando fallback")
            return self._fallback_decision(top_candidates)
        except Exception as e:
            logger.error(f"❌ Error en decisión Gemini: {e}")
            return self._fallback_decision(top_candidates)

    def _build_route_selection_prompt(self, context: Dict[str, Any]) -> str:
        """🔧 Construye prompt especializado para selección de rutas"""

        # Prompt más compacto para evitar timeouts
        prompt = f"""
# SISTEMA EXPERTO LOGÍSTICO LIVERPOOL

Eres un experto en logística mexicana. Selecciona la MEJOR ruta de entrega.

## CANDIDATOS:
{json.dumps(context['candidatos'][:3], indent=1)}

## FACTORES EXTERNOS:
- Demanda: {context['factores_externos'].get('factor_demanda', 1.0)}
- Clima: {context['factores_externos'].get('condicion_clima', 'Normal')}
- Tráfico: {context['factores_externos'].get('trafico_nivel', 'Moderado')}

## PESOS ESTRATÉGICOS:
- Tiempo: {context['weights']['tiempo']}
- Costo: {context['weights']['costo']}
- Probabilidad: {context['weights']['probabilidad']}

## RESPUESTA REQUERIDA (JSON):
```json
{{
    "candidato_seleccionado_id": "ID_DEL_MEJOR",
    "razonamiento": "Por qué es el mejor en 1-2 oraciones",
    "factores_decisivos": ["factor1", "factor2"],
    "confianza_decision": 0.XX,
    "alertas_operativas": ["alerta1"]
}}
```

IMPORTANTE: Prioriza PROBABILIDAD y TIEMPO sobre costo. Selecciona el candidato más CONFIABLE.
"""

        return prompt

    async def validate_inventory_split(self,
                                       split_plan: Dict[str, Any],
                                       product_info: Dict[str, Any],
                                       request_context: Dict[str, Any]) -> Dict[str, Any]:
        """📦 Valida y optimiza plan de split de inventario"""

        # Si Gemini no está disponible, validación simple
        if not self.model:
            return {
                'split_recomendado': split_plan.get('es_factible', False),
                'justificacion': 'Validación automática (Gemini no disponible)',
                'score_viabilidad': 0.8,
                'optimizaciones': ['revision_manual_recomendada'],
                'riesgos_identificados': []
            }

        try:
            context_data = self._serialize_for_json({
                'split_plan': split_plan,
                'producto': product_info,
                'request': request_context
            })

            prompt = f"""
# VALIDADOR DE SPLIT LIVERPOOL

Evalúa este plan de división de inventario:

## SPLIT PLAN:
```json
{json.dumps(context_data['split_plan'], indent=1)}
```

## PRODUCTO:
- Peso: {context_data['producto'].get('peso_kg', 'N/A')}kg
- Frágil: {context_data['producto'].get('es_fragil', False)}

## RESPUESTA JSON:
```json
{{
    "split_recomendado": true/false,
    "justificacion": "Razón principal",
    "score_viabilidad": 0.XX,
    "optimizaciones": ["opt1"],
    "riesgos_identificados": ["riesgo1"]
}}
```
            """

            response = await asyncio.wait_for(
                self.model.generate_content_async(prompt),
                timeout=8.0
            )
            return self._parse_json_response(response.text)

        except Exception as e:
            logger.warning(f"❌ Error validando split con Gemini: {e}")
            return {
                'split_recomendado': split_plan.get('es_factible', False),
                'justificacion': 'Validación automática por error en IA',
                'score_viabilidad': 0.75,
                'optimizaciones': ['verificar_manualmente'],
                'riesgos_identificados': ['validacion_ia_fallida']
            }

    async def analyze_external_factors_impact(self,
                                              external_factors: Dict[str, Any],
                                              target_postal_code: str,
                                              delivery_date: datetime) -> Dict[str, Any]:
        """🌤️ Analiza impacto de factores externos en la entrega"""

        # Si Gemini no está disponible, usar análisis simple
        if not self.model:
            factor_demanda = external_factors.get('factor_demanda', 1.0)
            return {
                'impacto_tiempo_horas': min(2.0, (factor_demanda - 1.0) * 2),
                'impacto_costo_pct': min(15.0, (factor_demanda - 1.0) * 10),
                'probabilidad_retraso': min(0.2, (factor_demanda - 1.0) * 0.15),
                'criticidad': 'Alta' if factor_demanda > 2.5 else 'Media',
                'factores_criticos': external_factors.get('eventos_detectados', []),
                'estrategias_mitigacion': ['monitoreo_activo'],
                'alertas_especiales': [],
                'recomendacion_flota': 'FE' if factor_demanda > 2.0 else 'FI'
            }

        try:
            context_data = self._serialize_for_json({
                'factores': external_factors,
                'codigo_postal': target_postal_code,
                'fecha_entrega': delivery_date,
                'fecha_actual': datetime.now()
            })

            prompt = f"""
# ANÁLISIS DE FACTORES EXTERNOS MÉXICO

Analiza el impacto logístico:

## FACTORES:
- Demanda: {context_data['factores'].get('factor_demanda', 1.0)}
- Clima: {context_data['factores'].get('condicion_clima', 'Normal')}
- Eventos: {context_data['factores'].get('eventos_detectados', [])}
- CP: {context_data['codigo_postal']}

## RESPUESTA JSON:
```json
{{
    "impacto_tiempo_horas": X.X,
    "impacto_costo_pct": X.X,
    "probabilidad_retraso": 0.XX,
    "criticidad": "Alta|Media|Baja",
    "factores_criticos": ["factor1"],
    "recomendacion_flota": "FI|FE|FI_FE"
}}
```
            """

            response = await asyncio.wait_for(
                self.model.generate_content_async(prompt),
                timeout=8.0
            )
            return self._parse_json_response(response.text)

        except Exception as e:
            logger.warning(f"❌ Error analizando factores con Gemini: {e}")
            # Fallback a análisis simple
            factor_demanda = external_factors.get('factor_demanda', 1.0)
            return {
                'impacto_tiempo_horas': min(1.5, (factor_demanda - 1.0) * 1.5),
                'impacto_costo_pct': min(12.0, (factor_demanda - 1.0) * 8),
                'probabilidad_retraso': min(0.15, (factor_demanda - 1.0) * 0.1),
                'criticidad': 'Media',
                'factores_criticos': ['analisis_automatico'],
                'estrategias_mitigacion': ['monitoreo_estandar'],
                'alertas_especiales': [],
                'recomendacion_flota': 'FE' if factor_demanda > 2.2 else 'FI'
            }

    async def generate_final_explanation(self,
                                         selected_route: Dict[str, Any],
                                         all_context: Dict[str, Any]) -> Dict[str, Any]:
        """📊 Genera explicación ejecutiva completa"""

        # Si Gemini no está disponible, generar explicación simple
        if not self.model:
            return {
                'resumen_ejecutivo': f"Ruta {selected_route.get('tipo_ruta', 'optimizada')} seleccionada automáticamente",
                'valor_cliente': 'Entrega eficiente y confiable',
                'eficiencia_operativa': 'Optimización de recursos disponibles',
                'metricas_clave': {
                    'tiempo_entrega': f"{selected_route.get('tiempo_total_horas', 0):.1f} horas",
                    'costo_total': f"${selected_route.get('costo_total_mxn', 0):.0f} MXN",
                    'confiabilidad': f"{selected_route.get('probabilidad_cumplimiento', 0.8) * 100:.0f}%"
                },
                'factores_determinantes': ['optimizacion_automatica', 'mejor_score_disponible'],
                'acciones_operativas': ['ejecutar_ruta_seleccionada', 'monitorear_progreso'],
                'kpis_monitoreo': ['tiempo_real_entrega', 'satisfaccion_cliente'],
                'nivel_confianza': 'Medio',
                'proxima_revision': 'Al completar entrega'
            }

        try:
            context_data = self._serialize_for_json({
                'ruta_seleccionada': selected_route,
                'contexto_completo': all_context
            })

            prompt = f"""
# EXPLICACIÓN EJECUTIVA LIVERPOOL

Genera resumen ejecutivo para esta decisión logística:

## RUTA SELECCIONADA:
- Tipo: {selected_route.get('tipo_ruta', 'N/A')}
- Tiempo: {selected_route.get('tiempo_total_horas', 0):.1f}h
- Costo: ${selected_route.get('costo_total_mxn', 0):.0f}
- Probabilidad: {selected_route.get('probabilidad_cumplimiento', 0) * 100:.0f}%

## RESPUESTA JSON:
```json
{{
    "resumen_ejecutivo": "Decisión principal en una línea",
    "valor_cliente": "Beneficio para el cliente",
    "metricas_clave": {{
        "tiempo_entrega": "X horas",
        "costo_total": "$X MXN",
        "confiabilidad": "XX%"
    }},
    "factores_determinantes": ["factor1", "factor2"],
    "nivel_confianza": "Alto|Medio|Bajo"
}}
```
            """

            response = await asyncio.wait_for(
                self.model.generate_content_async(prompt),
                timeout=8.0
            )
            return self._parse_json_response(response.text)

        except Exception as e:
            logger.warning(f"❌ Error generando explicación con Gemini: {e}")
            return {
                'resumen_ejecutivo': 'Ruta optimizada seleccionada por criterios de eficiencia',
                'valor_cliente': 'Entrega confiable en tiempo óptimo',
                'eficiencia_operativa': 'Balance óptimo tiempo-costo-calidad',
                'metricas_clave': {
                    'tiempo_entrega': f"{selected_route.get('tiempo_total_horas', 0):.1f} horas",
                    'costo_total': f"${selected_route.get('costo_total_mxn', 0):.0f} MXN",
                    'confiabilidad': f"{selected_route.get('probabilidad_cumplimiento', 0.8) * 100:.0f}%"
                },
                'factores_determinantes': ['optimizacion_ml', 'reglas_negocio'],
                'acciones_operativas': ['ejecutar_ruta', 'monitorear_kpis'],
                'kpis_monitoreo': ['tiempo_entrega', 'costo_real'],
                'nivel_confianza': 'Medio',
                'proxima_revision': 'Post-entrega'
            }

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """🔧 Parser robusto mejorado para respuestas JSON de Gemini"""

        try:
            # Limpiar markdown
            clean_text = response_text.strip()

            # Remover markdown blocks
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0]
            elif "```" in clean_text:
                # Manejar cases donde no hay 'json' especificado
                parts = clean_text.split("```")
                for part in parts:
                    if "{" in part and "}" in part:
                        clean_text = part
                        break

            # Encontrar el JSON válido más largo
            start_pos = clean_text.find("{")
            if start_pos == -1:
                raise json.JSONDecodeError("No JSON found", clean_text, 0)

            # Buscar el } que cierra correctamente
            brace_count = 0
            end_pos = start_pos

            for i, char in enumerate(clean_text[start_pos:], start_pos):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break

            json_text = clean_text[start_pos:end_pos]

            # Intentar parsear
            parsed = json.loads(json_text)
            logger.debug("✅ JSON parseado correctamente de Gemini")
            return parsed

        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            logger.error(f"❌ Error parsing JSON de Gemini: {e}")
            logger.error(f"Texto recibido: {response_text[:200]}...")

            # Fallback response más robusto con todos los campos esperados
            return {
                "error": "JSON parsing failed",
                "candidato_seleccionado_id": "fallback",
                "razonamiento": "Error en parsing - selección automática por score",
                "confianza_decision": 0.75,
                "factores_decisivos": ["error_parsing", "fallback_automatico"],
                "alertas_operativas": ["revision_manual_recomendada"],
                "split_recomendado": True,
                "justificacion": "Fallback por error de parsing",
                "score_viabilidad": 0.7,
                "optimizaciones": ["revision_manual"],
                "riesgos_identificados": ["parsing_fallido"],
                "impacto_tiempo_horas": 1.0,
                "impacto_costo_pct": 8.0,
                "probabilidad_retraso": 0.1,
                "criticidad": "Media",
                "factores_criticos": ["error_gemini"],
                "estrategias_mitigacion": ["monitoreo_manual"],
                "alertas_especiales": ["ia_no_disponible"],
                "recomendacion_flota": "FE",
                "resumen_ejecutivo": "Decisión automática por error en IA",
                "valor_cliente": "Entrega estándar garantizada",
                "eficiencia_operativa": "Proceso automatizado de respaldo",
                "metricas_clave": {
                    "tiempo_entrega": "Estimado automáticamente",
                    "costo_total": "Cálculo estándar",
                    "confiabilidad": "Promedio histórico"
                },
                "factores_determinantes": ["sistema_respaldo"],
                "acciones_operativas": ["ejecutar_plan_automatico"],
                "kpis_monitoreo": ["seguimiento_basico"],
                "nivel_confianza": "Medio",
                "proxima_revision": "Inmediata post-entrega"
            }

    def _fallback_decision(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """🔄 Decisión fallback cuando Gemini falla (MEJORADA)"""

        if not candidates:
            raise ValueError("No hay candidatos para fallback")

        # Seleccionar el mejor por score LightGBM
        best_candidate = max(candidates, key=lambda x: x.get('score_lightgbm', 0))

        # Análisis simple de factores decisivos
        factores_decisivos = ['score_lightgbm_alto']

        if best_candidate.get('probabilidad_cumplimiento', 0) > 0.8:
            factores_decisivos.append('alta_confiabilidad')
        if best_candidate.get('tiempo_total_horas', 48) < 24:
            factores_decisivos.append('entrega_rapida')
        if best_candidate.get('tipo_ruta') == 'directa':
            factores_decisivos.append('ruta_simple')

        # Generar razonamiento automático
        razonamiento = f"Selección automática: {best_candidate.get('tipo_ruta', 'ruta')} con score {best_candidate.get('score_lightgbm', 0):.3f}"

        if best_candidate.get('probabilidad_cumplimiento', 0) > 0.85:
            razonamiento += ", alta confiabilidad"
        if best_candidate.get('tiempo_total_horas', 48) < 24:
            razonamiento += ", entrega rápida"

        return {
            'candidato_seleccionado': best_candidate,
            'candidatos_evaluados': candidates,
            'razonamiento': razonamiento,
            'confianza_decision': 0.78,  # Ligeramente más alta que antes
            'factores_decisivos': factores_decisivos,
            'trade_offs_identificados': {
                'ventajas': ['mejor_score_ml', 'optimizacion_automatica'],
                'desventajas': ['sin_analisis_contextual_ia']
            },
            'alertas_operativas': ['decision_automatica', 'gemini_no_disponible'],
            'timestamp_decision': datetime.now().isoformat()
        }