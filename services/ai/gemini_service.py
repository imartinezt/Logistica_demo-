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
            vertexai.init(project=settings.PROJECT_ID, location=settings.REGION)
            cls._model = GenerativeModel(settings.MODEL_NAME)
        return cls._model

    @classmethod
    def get_chat_session(cls) -> ChatSession:
        if cls._chat_session is None:
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
                'razonamiento': 'Único candidato disponible',
                'confianza_decision': 0.8,
                'factores_decisivos': ['unica_opcion_disponible']
            }

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

        try:
            response = await self.model.generate_content_async(prompt)
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

        except Exception as e:
            logger.error(f"❌ Error en decisión Gemini: {e}")
            return self._fallback_decision(top_candidates)

    def _build_route_selection_prompt(self, context: Dict[str, Any]) -> str:
        """🔧 Construye prompt especializado para selección de rutas"""

        prompt = f"""
# SISTEMA EXPERTO LOGÍSTICO LIVERPOOL

Eres el sistema de decisión más avanzado de Liverpool para optimización de rutas de entrega. 
Tienes expertise en logística mexicana, factores climáticos, temporadas altas y comportamiento del consumidor.

## CONTEXTO DE LA DECISIÓN

**Request del Cliente:**
```json
{json.dumps(context['request'], indent=2)}
```

**Factores Externos Detectados:**
```json
{json.dumps(context['factores_externos'], indent=2)}
```

**Candidatos Top (Preseleccionados por LightGBM):**
```json
{json.dumps(context['candidatos'], indent=2)}
```

**Pesos Estratégicos Liverpool:**
- Tiempo: {context['weights']['tiempo']} (prioridad en satisfacción cliente)
- Costo: {context['weights']['costo']} (impacto en margen)
- Probabilidad: {context['weights']['probabilidad']} (confiabilidad promesa)
- Distancia: {context['weights']['distancia']} (eficiencia operativa)

## TU MISIÓN

Selecciona EL MEJOR candidato considerando:

1. **Experiencia del Cliente**: ¿Cuál cumple mejor la promesa de entrega?
2. **Eficiencia Operativa**: ¿Cuál optimiza recursos Liverpool?
3. **Gestión de Riesgo**: ¿Cuál minimiza probabilidad de falla?
4. **Contexto Temporal**: ¿Cómo afectan los factores externos detectados?

## REGLAS DE NEGOCIO CRÍTICAS

- Si compra antes 12:00 → priorizar FLASH (mismo día)
- Si es temporada alta (factor > 2.0) → priorizar confiabilidad sobre costo
- Si es zona roja → NUNCA flota interna sola
- Si producto frágil → priorizar rutas con menos transferencias
- Si lluvia > 60% → penalizar rutas largas

## RESPUESTA REQUERIDA

Responde ÚNICAMENTE en JSON válido:

```json
{{
    "candidato_seleccionado_id": "ID_DEL_CANDIDATO_GANADOR",
    "razonamiento": "Explicación detallada de 2-3 oraciones de por qué este candidato es superior, citando métricas específicas",
    "factores_decisivos": ["factor1", "factor2", "factor3"],
    "confianza_decision": 0.XX,
    "trade_offs_identificados": {{
        "ventajas": ["ventaja1", "ventaja2"],
        "desventajas": ["desventaja1", "desventaja2"]
    }},
    "alertas_operativas": ["alerta1", "alerta2"],
    "recomendaciones_monitoreo": ["recomendacion1", "recomendacion2"]
}}
```

## CRITERIOS DE DECISIÓN AVANZADOS

**Para Temporada Normal (factor ≤ 1.5):**
- Optimizar tiempo-costo
- Priorizar rutas directas
- Minimizar complejidad

**Para Temporada Alta (factor > 2.0):**
- Priorizar confiabilidad (probabilidad)
- Aceptar costos premium por garantía
- Evitar rutas con múltiples transferencias

**Para Zona Roja:**
- OBLIGATORIO flota externa o híbrida
- Verificar cobertura carrier externo
- Aumentar buffer de tiempo

**Para Clima Adverso:**
- Penalizar distancias > 100km
- Priorizar rutas urbanas
- Considerar delays adicionales

ANALIZA profundamente y decide con la expertise de 20 años en logística México.
"""

        return prompt

    async def validate_inventory_split(self,
                                       split_plan: Dict[str, Any],
                                       product_info: Dict[str, Any],
                                       request_context: Dict[str, Any]) -> Dict[str, Any]:
        """📦 Valida y optimiza plan de split de inventario"""

        context_data = self._serialize_for_json({
            'split_plan': split_plan,
            'producto': product_info,
            'request': request_context
        })

        prompt = f"""
# VALIDADOR DE SPLIT DE INVENTARIO LIVERPOOL

Analiza este plan de división de inventario como experto en fulfillment:

## DATOS DEL SPLIT
```json
{json.dumps(context_data, indent=2)}
```

## EVALÚA

1. **Viabilidad Operativa**: ¿Es ejecutable en la práctica?
2. **Eficiencia de Costos**: ¿Justifica la complejidad adicional?
3. **Experiencia Cliente**: ¿Afecta la promesa de entrega?
4. **Riesgo Operativo**: ¿Qué puede fallar?

## ALTERNATIVAS A CONSIDERAR

- ¿Consolidar todo desde una ubicación es mejor?
- ¿El split agrega valor real al cliente?
- ¿Los tiempos de preparación son realistas?

Responde en JSON:

```json
{{
    "split_recomendado": true/false,
    "justificacion": "Razones específicas",
    "optimizaciones": ["optimizacion1", "optimizacion2"],
    "riesgos_identificados": ["riesgo1", "riesgo2"],
    "alternativa_sugerida": "Descripción si split no es óptimo",
    "score_viabilidad": 0.XX
}}
```
        """

        try:
            response = await self.model.generate_content_async(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            logger.error(f"❌ Error validando split: {e}")
            return {
                'split_recomendado': split_plan.get('es_factible', False),
                'justificacion': 'Validación automática por error en Gemini',
                'score_viabilidad': 0.7
            }

    async def analyze_external_factors_impact(self,
                                              external_factors: Dict[str, Any],
                                              target_postal_code: str,
                                              delivery_date: datetime) -> Dict[str, Any]:
        """🌤️ Analiza impacto de factores externos en la entrega"""

        context_data = self._serialize_for_json({
            'factores': external_factors,
            'codigo_postal': target_postal_code,
            'fecha_entrega': delivery_date,
            'fecha_actual': datetime.now()
        })

        prompt = f"""
# ANÁLISIS DE FACTORES EXTERNOS MÉXICO

Como experto en logística mexicana, analiza el impacto de estos factores:

```json
{json.dumps(context_data, indent=2)}
```

## CONTEXTO MÉXICO

- Temporadas: Buen Fin (Nov), Navidad (Dic), Día Madres (May)
- Clima: Temporada lluvia Jun-Sep
- Tráfico: CDMX crítico 7-10am, 6-9pm
- Zonas: Norte más seguro, Sur más complicado

## ANALIZA

1. **Impacto Temporal**: ¿Cómo afectan los tiempos?
2. **Impacto Económico**: ¿Aumentan los costos?
3. **Riesgo Operativo**: ¿Qué probabilidad de falla?
4. **Mitigación**: ¿Qué acciones tomar?

Responde en JSON:

```json
{{
    "impacto_tiempo_horas": X.X,
    "impacto_costo_pct": X.X,
    "probabilidad_retraso": 0.XX,
    "criticidad": "Baja|Media|Alta|Crítica",
    "factores_criticos": ["factor1", "factor2"],
    "estrategias_mitigacion": ["estrategia1", "estrategia2"],
    "alertas_especiales": ["alerta1", "alerta2"],
    "recomendacion_flota": "FI|FE|FI_FE"
}}
```
        """

        try:
            response = await self.model.generate_content_async(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            logger.error(f"❌ Error analizando factores: {e}")
            return {
                'impacto_tiempo_horas': 0.5,
                'impacto_costo_pct': 5.0,
                'criticidad': 'Media',
                'factores_criticos': ['error_gemini']
            }

    async def generate_final_explanation(self,
                                         selected_route: Dict[str, Any],
                                         all_context: Dict[str, Any]) -> Dict[str, Any]:
        """📊 Genera explicación ejecutiva completa"""

        context_data = self._serialize_for_json({
            'ruta_seleccionada': selected_route,
            'contexto_completo': all_context
        })

        prompt = f"""
# EXPLICACIÓN EJECUTIVA LIVERPOOL FEE

Genera un resumen ejecutivo de esta decisión logística para stakeholders:

```json
{json.dumps(context_data, indent=2)}
```

## AUDIENCIA
- Gerentes de operaciones
- Customer service
- Equipos de fulfillment

## INCLUYE

1. **Resumen de 1 línea**: La decisión principal
2. **Justificación**: Por qué es la mejor opción
3. **Métricas clave**: Tiempo, costo, probabilidad
4. **Factores considerados**: Qué influyó en la decisión
5. **Acciones requeridas**: Qué debe hacer el equipo operativo
6. **Monitoreo**: Qué vigilar durante la ejecución

Responde en JSON:

```json
{{
    "resumen_ejecutivo": "Una línea describiendo la decisión",
    "valor_cliente": "Cómo beneficia al cliente",
    "eficiencia_operativa": "Impacto en operaciones",
    "metricas_clave": {{
        "tiempo_entrega": "X horas",
        "costo_total": "$X MXN",
        "confiabilidad": "XX%"
    }},
    "factores_determinantes": ["factor1", "factor2"],
    "acciones_operativas": ["accion1", "accion2"],
    "kpis_monitoreo": ["kpi1", "kpi2"],
    "nivel_confianza": "Alto|Medio|Bajo",
    "proxima_revision": "Cuándo revisar la predicción"
}}
```
        """

        try:
            response = await self.model.generate_content_async(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            logger.error(f"❌ Error generando explicación: {e}")
            return {
                'resumen_ejecutivo': 'Ruta optimizada seleccionada automáticamente',
                'nivel_confianza': 'Medio'
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
            logger.error(f"Texto recibido: {response_text[:300]}...")

            # Fallback response más robusto
            return {
                "error": "JSON parsing failed",
                "candidato_seleccionado_id": "fallback",
                "razonamiento": "Error en parsing - selección automática",
                "confianza_decision": 0.5,
                "factores_decisivos": ["error_parsing"],
                "split_recomendado": True,
                "justificacion": "Fallback por error de parsing",
                "score_viabilidad": 0.6,
                "impacto_tiempo_horas": 1.0,
                "impacto_costo_pct": 10.0,
                "criticidad": "Media",
                "resumen_ejecutivo": "Decisión automática por error en IA"
            }

    def _fallback_decision(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """🔄 Decisión fallback cuando Gemini falla"""

        if not candidates:
            raise ValueError("No hay candidatos para fallback")

        # Seleccionar el mejor por score LightGBM
        best_candidate = max(candidates, key=lambda x: x.get('score_lightgbm', 0))

        return {
            'candidato_seleccionado': best_candidate,
            'candidatos_evaluados': candidates,
            'razonamiento': 'Selección automática por score LightGBM (fallback)',
            'confianza_decision': 0.75,
            'factores_decisivos': ['score_lightgbm', 'fallback_system'],
            'trade_offs_identificados': {
                'ventajas': ['mejor_score_ml'],
                'desventajas': ['sin_analisis_gemini']
            },
            'alertas_operativas': ['decision_fallback'],
            'timestamp_decision': datetime.now().isoformat()
        }