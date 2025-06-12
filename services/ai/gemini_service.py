# ARCHIVO CORREGIDO: gemini_service.py
# Mantiene compatibilidad con código existente + nueva clase optimizada

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List

import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession

from config.settings import settings
from utils.logger import logger


class VertexAIModelSingleton:
    """Singleton para Gemini - MANTENER PARA COMPATIBILIDAD"""
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
    """Motor de decisión logística con Gemini - CLASE ORIGINAL MANTENIDA"""

    def __init__(self):
        self.model = VertexAIModelSingleton.get_model()
        self.decision_context = {}

    def _serialize_for_json(self, obj: Any) -> Any:
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
        """Selección final de ruta óptima por Gemini"""

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

        if not self.model:
            logger.warning("⚠️ Gemini no disponible, usando selección automática")
            return self._fallback_decision(top_candidates)

        try:
            context_data = self._serialize_for_json({
                'request': request_context,
                'factores_externos': external_factors,
                'candidatos': top_candidates,
                'business_rules': getattr(settings, 'DELIVERY_RULES', {}),
                'weights': {
                    'tiempo': getattr(settings, 'PESO_TIEMPO', 0.4),
                    'costo': getattr(settings, 'PESO_COSTO', 0.2),
                    'probabilidad': getattr(settings, 'PESO_PROBABILIDAD', 0.35),
                    'distancia': getattr(settings, 'PESO_DISTANCIA', 0.05)
                }
            })

            prompt = self._build_route_selection_prompt(context_data)
            response = await asyncio.wait_for(
                self.model.generate_content_async(prompt),
                timeout=10.0
            )

            decision = self._parse_json_response(response.text)
            selected_id = decision.get('candidato_seleccionado_id')
            selected_candidate = None

            for candidate in top_candidates:
                if candidate['ruta_id'] == selected_id:
                    selected_candidate = candidate
                    break

            if not selected_candidate:
                logger.warning(f"❌ Gemini seleccionó ID inválido: {selected_id}")
                selected_candidate = top_candidates[0]
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

    @staticmethod
    def _build_route_selection_prompt(context: Dict[str, Any]) -> str:
        """Construye prompt especializado para selección de rutas"""

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

    @staticmethod
    def _parse_json_response(response_text: str) -> Dict[str, Any]:
        """Parser robusto mejorado para respuestas JSON de Gemini"""

        try:
            clean_text = response_text.strip()
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0]
            elif "```" in clean_text:
                parts = clean_text.split("```")
                for part in parts:
                    if "{" in part and "}" in part:
                        clean_text = part
                        break

            start_pos = clean_text.find("{")
            if start_pos == -1:
                raise json.JSONDecodeError("No JSON found", clean_text, 0)

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
            parsed = json.loads(json_text)
            logger.debug("✅ JSON parseado correctamente de Gemini")
            return parsed

        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            logger.error(f"❌ Error parsing JSON de Gemini: {e}")
            logger.error(f"Texto recibido: {response_text[:200]}...")

            return {
                "error": "JSON parsing failed",
                "candidato_seleccionado_id": "fallback",
                "razonamiento": "Error en parsing - selección automática por score",
                "confianza_decision": 0.75,
                "factores_decisivos": ["error_parsing", "fallback_automatico"],
                "alertas_operativas": ["revision_manual_recomendada"]
            }

    @staticmethod
    def _fallback_decision(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Decisión fallback cuando Gemini falla"""

        if not candidates:
            raise ValueError("No hay candidatos para fallback")

        best_candidate = max(candidates, key=lambda x: x.get('score_lightgbm', 0))
        factores_decisivos = ['score_lightgbm_alto']

        if best_candidate.get('probabilidad_cumplimiento', 0) > 0.8:
            factores_decisivos.append('alta_confiabilidad')
        if best_candidate.get('tiempo_total_horas', 48) < 24:
            factores_decisivos.append('entrega_rapida')
        if best_candidate.get('tipo_ruta') == 'directa':
            factores_decisivos.append('ruta_simple')

        razonamiento = f"Selección automática: {best_candidate.get('tipo_ruta', 'ruta')} con score {best_candidate.get('score_lightgbm', 0):.3f}"

        if best_candidate.get('probabilidad_cumplimiento', 0) > 0.85:
            razonamiento += ", alta confiabilidad"
        if best_candidate.get('tiempo_total_horas', 48) < 24:
            razonamiento += ", entrega rápida"

        return {
            'candidato_seleccionado': best_candidate,
            'candidatos_evaluados': candidates,
            'razonamiento': razonamiento,
            'confianza_decision': 0.78,
            'factores_decisivos': factores_decisivos,
            'trade_offs_identificados': {
                'ventajas': ['mejor_score_ml', 'optimizacion_automatica'],
                'desventajas': ['sin_analisis_contextual_ia']
            },
            'alertas_operativas': ['decision_automatica', 'gemini_no_disponible'],
            'timestamp_decision': datetime.now().isoformat()
        }


# NUEVA CLASE OPTIMIZADA - ALIAS PARA USAR EN SERVICIOS OPTIMIZADOS
class OptimizedGeminiEngine(GeminiLogisticsDecisionEngine):
    """🚀 Versión optimizada del motor Gemini con mejores prompts"""

    def __init__(self):
        super().__init__()
        logger.info("🚀 Motor Gemini optimizado inicializado")

    async def select_optimal_route(self,
                                   candidates: List[Dict[str, Any]],
                                   request_context: Dict[str, Any],
                                   external_factors: Dict[str, Any]) -> Dict[str, Any]:
        """🎯 Selección optimizada con prompts mejorados"""

        if not candidates:
            raise ValueError("No hay candidatos para evaluar")

        if len(candidates) == 1:
            return {
                'candidato_seleccionado': candidates[0],
                'razonamiento': 'Único candidato disponible tras optimización dinámica',
                'confianza_decision': 0.85,
                'factores_decisivos': ['unica_opcion_dinamica'],
                'alertas_operativas': []
            }

        if not self.model:
            logger.warning("⚠️ Gemini no disponible, usando selección automática")
            return self._optimized_fallback(candidates)

        try:
            prompt = self._build_enhanced_prompt(candidates, request_context, external_factors)

            response = await asyncio.wait_for(
                self.model.generate_content_async(prompt),
                timeout=10.0
            )

            decision = self._parse_enhanced_response(response.text)
            selected_candidate = self._find_candidate_by_id(decision, candidates)

            if not selected_candidate:
                logger.warning(f"❌ ID inválido seleccionado, usando fallback")
                return self._optimized_fallback(candidates)

            decision['candidato_seleccionado'] = selected_candidate
            decision['timestamp_decision'] = datetime.now().isoformat()

            logger.info(f"🧠 Gemini optimizado seleccionó: {selected_candidate['ruta_id']} "
                        f"(score: {selected_candidate.get('score_lightgbm', 0):.3f})")

            return decision

        except asyncio.TimeoutError:
            logger.warning("⏰ Timeout en Gemini optimizado, usando fallback")
            return self._optimized_fallback(candidates)
        except Exception as e:
            logger.error(f"❌ Error en Gemini optimizado: {e}")
            return self._optimized_fallback(candidates)

    def _build_enhanced_prompt(self, candidates: List[Dict[str, Any]],
                               request_context: Dict[str, Any],
                               external_factors: Dict[str, Any]) -> str:
        """🔧 Prompt optimizado para mejores decisiones"""

        # Extraer información clave
        sku_id = request_context.get('sku_id', 'N/A')
        codigo_postal = request_context.get('codigo_postal', 'N/A')
        cantidad = request_context.get('cantidad', 0)

        factor_demanda = external_factors.get('factor_demanda', 1.0)
        criticidad = external_factors.get('criticidad_logistica', 'Normal')
        evento = external_factors.get('evento_detectado', 'Normal')

        # Top 3 candidatos
        top_candidates = candidates[:3]

        prompt = f"""# DECISIÓN LOGÍSTICA LIVERPOOL - ANÁLISIS EXPERTO

Eres un experto en logística mexicana. Analiza estos candidatos de ruta y selecciona el ÓPTIMO.

## CONTEXTO DEL PEDIDO:
- SKU: {sku_id}
- Destino: CP {codigo_postal}
- Cantidad: {cantidad} unidades
- Evento actual: {evento}
- Factor demanda: {factor_demanda:.2f}x
- Criticidad: {criticidad}

## CANDIDATOS A EVALUAR:
"""

        for i, candidate in enumerate(top_candidates, 1):
            prompt += f"""
### CANDIDATO {i}: {candidate.get('tipo_ruta', 'N/A').upper()}
- ID: {candidate['ruta_id']}
- Tiempo total: {candidate['tiempo_total_horas']:.1f} horas
- Costo total: ${candidate['costo_total_mxn']:.0f} MXN
- Distancia: {candidate['distancia_total_km']:.1f} km
- Probabilidad éxito: {candidate['probabilidad_cumplimiento']:.1%}
- Score ML: {candidate.get('score_lightgbm', 0):.3f}
- Origen principal: {candidate.get('origen_principal', 'N/A')}
"""

        prompt += f"""

## CRITERIOS DE DECISIÓN:
1. **PRIORIDAD ALTA**: Probabilidad de cumplimiento y tiempo de entrega
2. **PRIORIDAD MEDIA**: Costo competitivo y simplicidad operativa
3. **PRIORIDAD BAJA**: Distancia total

## FACTORES ESPECIALES:
- Demanda {factor_demanda:.1f}x ➜ {"Temporada crítica" if factor_demanda > 2.5 else "Demanda normal"}
- Criticidad {criticidad} ➜ {"Requiere alta confiabilidad" if criticidad in ["Alta", "Crítica"] else "Operación estándar"}

## RESPUESTA REQUERIDA (JSON ESTRICTO):
```json
{{
    "candidato_seleccionado_id": "ruta_id_del_mejor",
    "razonamiento": "Razón principal en 1-2 oraciones concisas",
    "factores_decisivos": ["factor1", "factor2", "factor3"],
    "confianza_decision": 0.XX,
    "alertas_operativas": ["alerta1", "alerta2"]
}}
```

**REGLAS CRÍTICAS:**
- Selecciona SIEMPRE el candidato más CONFIABLE y RÁPIDO
- En temporada crítica (demanda >2.5x), prioriza PROBABILIDAD sobre costo
- El ID debe coincidir EXACTAMENTE con uno de los candidatos

Analiza y decide:"""

        return prompt

    def _parse_enhanced_response(self, response_text: str) -> Dict[str, Any]:
        """🔧 Parser mejorado para respuestas optimizadas"""
        try:
            clean_text = response_text.strip()

            if "```json" in clean_text:
                start = clean_text.find("```json") + 7
                end = clean_text.find("```", start)
                json_text = clean_text[start:end].strip()
            else:
                start = clean_text.find("{")
                end = clean_text.rfind("}") + 1
                json_text = clean_text[start:end]

            decision = json.loads(json_text)

            # Validar y completar campos
            decision['confianza_decision'] = float(decision.get('confianza_decision', 0.8))
            decision['factores_decisivos'] = list(decision.get('factores_decisivos', ['decision_automatica']))
            decision['alertas_operativas'] = list(decision.get('alertas_operativas', []))

            logger.info(f"✅ Decisión optimizada parseada: {decision['candidato_seleccionado_id']}")
            return decision

        except Exception as e:
            logger.error(f"❌ Error parsing optimizado: {e}")
            return {
                "candidato_seleccionado_id": "fallback",
                "razonamiento": "Error en parsing - selección automática por score ML",
                "factores_decisivos": ["error_parsing", "fallback_ml_score"],
                "confianza_decision": 0.75,
                "alertas_operativas": ["revision_manual_recomendada"]
            }

    def _find_candidate_by_id(self, decision: Dict[str, Any],
                              candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """🔍 Encuentra candidato por ID con búsqueda optimizada"""
        selected_id = decision.get('candidato_seleccionado_id', '')

        # Búsqueda exacta
        for candidate in candidates:
            if candidate['ruta_id'] == selected_id:
                return candidate

        # Búsqueda parcial
        for candidate in candidates:
            if selected_id in candidate['ruta_id'] or candidate['ruta_id'] in selected_id:
                logger.warning(f"⚠️ ID parcial: {candidate['ruta_id']} ≈ {selected_id}")
                return candidate

        return None

    def _optimized_fallback(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """🔄 Fallback optimizado con mejor lógica"""
        best_candidate = max(candidates, key=lambda x: x.get('score_lightgbm', 0))

        factores = ['score_ml_optimizado']
        if best_candidate.get('probabilidad_cumplimiento', 0) > 0.8:
            factores.append('alta_confiabilidad')
        if best_candidate.get('tiempo_total_horas', 48) < 24:
            factores.append('entrega_rapida')

        return {
            'candidato_seleccionado': best_candidate,
            'razonamiento': f"Selección optimizada automática (score: {best_candidate.get('score_lightgbm', 0):.3f})",
            'confianza_decision': 0.80,
            'factores_decisivos': factores,
            'alertas_operativas': ['decision_automatica_optimizada'],
            'timestamp_decision': datetime.now().isoformat()
        }