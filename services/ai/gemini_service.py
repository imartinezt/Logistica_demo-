import json
from datetime import datetime
from typing import Dict, Any, List

import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession

from config.settings import settings
from utils.logger import logger


class VertexAIModelSingleton:
    """🧠 Singleton para Gemini - Un solo modelo para toda la app"""
    _model = None
    _chat_session = None

    @classmethod
    def get_model(cls) -> GenerativeModel:
        if cls._model is None:
            logger.info("🧠 Inicializando Gemini 2.0 Flash")
            vertexai.init(project=settings.PROJECT_ID, location=settings.REGION)
            cls._model = GenerativeModel(settings.MODEL_NAME)
        return cls._model

    @classmethod
    def get_chat_session(cls) -> ChatSession:
        if cls._chat_session is None:
            cls._chat_session = cls.get_model().start_chat()
        return cls._chat_session


class GeminiReasoningService:
    """🎯 Servicio que maneja todos los prompts y razonamiento con Gemini"""

    def __init__(self):
        self.model = VertexAIModelSingleton.get_model()

    def _serialize_for_json(self, obj: Any) -> Any:
        """🔧 Serializa objetos para JSON, convirtiendo datetime a string"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        else:
            return obj

    async def validate_product_decision(self, producto: Dict, request_data: Dict) -> Dict[str, Any]:
        """🔍 Paso 1: Gemini valida y analiza el producto"""
        # Serializar datos para evitar problemas con datetime
        producto_serialized = self._serialize_for_json(producto)
        request_serialized = self._serialize_for_json(request_data)

        prompt = f"""
        Eres un experto en logística de Liverpool. Analiza este producto y request:

        PRODUCTO: {json.dumps(producto_serialized, indent=2)}
        REQUEST: {json.dumps(request_serialized, indent=2)}

        Evalúa:
        1. ¿Es factible la entrega? (peso, fragilidad, cantidad)
        2. ¿Qué consideraciones especiales tiene?
        3. ¿Qué riesgos potenciales detectas?

        Responde en JSON:
        {{
            "factible": boolean,
            "decision": "string explicando la decisión",
            "factores_clave": ["factor1", "factor2", ...],
            "score_producto": float (0-1),
            "consideraciones_especiales": ["consideracion1", ...],
            "riesgos": ["riesgo1", ...]
        }}
        """

        response = await self.model.generate_content_async(prompt)
        return self._parse_json_response(response.text)

    async def analyze_zone_safety(self, zona_info: Dict, codigo_postal: str) -> Dict[str, Any]:
        """🏠 Paso 2: Gemini analiza la zona y seguridad"""
        zona_serialized = self._serialize_for_json(zona_info)

        prompt = f"""
        Analiza esta zona de entrega como experto en logística México:

        ZONA: {json.dumps(zona_serialized, indent=2)}
        CODIGO_POSTAL: {codigo_postal}

        Evalúa:
        1. ¿Es zona roja? (nivel_seguridad, tipo_zona)
        2. ¿Qué tipo de flota se requiere?
        3. ¿Qué precauciones tomar?

        Responde en JSON:
        {{
            "es_zona_roja": boolean,
            "tipo_flota_requerida": "FI" | "FE" | "FI_FE",
            "nivel_riesgo": "Bajo" | "Medio" | "Alto",
            "decision": "string",
            "factores": ["factor1", ...],
            "precauciones": ["precaucion1", ...],
            "score_zona": float (0-1)
        }}
        """

        response = await self.model.generate_content_async(prompt)
        return self._parse_json_response(response.text)

    async def evaluate_external_factors(self, factores_detectados: Dict, fecha: str) -> Dict[str, Any]:
        """🌤️ Paso 3: Gemini evalúa factores externos detectados"""
        factores_serialized = self._serialize_for_json(factores_detectados)

        prompt = f"""
        Eres experto en logística México. Evalúa estos factores externos para {fecha}:

        FACTORES: {json.dumps(factores_serialized, indent=2)}

        Analiza:
        1. ¿Qué tan críticos son estos factores?
        2. ¿Cómo afectan los tiempos de entrega?
        3. ¿Qué estrategias recomiendas?

        Responde en JSON:
        {{
            "criticidad": "Baja" | "Media" | "Alta" | "Crítica",
            "impacto_tiempo_horas": int,
            "impacto_costo_pct": float,
            "decision": "string",
            "factores_criticos": ["factor1", ...],
            "estrategias": ["estrategia1", ...],
            "score_factores": float (0-1)
        }}
        """

        response = await self.model.generate_content_async(prompt)
        return self._parse_json_response(response.text)

    async def optimize_route_selection(self, rutas_candidatas: List[Dict], contexto: Dict) -> Dict[str, Any]:
        """🚚 Paso 6: Gemini optimiza selección de rutas"""
        rutas_serialized = self._serialize_for_json(rutas_candidatas[:5])
        contexto_serialized = self._serialize_for_json(contexto)

        prompt = f"""
        Optimiza la selección de ruta como experto Liverpool:

        RUTAS_CANDIDATAS: {json.dumps(rutas_serialized, indent=2)}
        CONTEXTO: {json.dumps(contexto_serialized, indent=2)}

        Considera:
        1. Tiempo, costo, probabilidad
        2. Factores externos
        3. Zona de entrega
        4. Características del producto

        Responde en JSON:
        {{
            "ruta_optima_id": "string",
            "razon_seleccion": "string detallada",
            "score_final": float (0-1),
            "trade_offs": ["trade_off1", ...],
            "alternativas": ["alt1", "alt2"],
            "confianza": float (0-1),
            "factores_decisivos": ["factor1", ...]
        }}
        """

        response = await self.model.generate_content_async(prompt)
        return self._parse_json_response(response.text)

    async def finalize_prediction_analysis(self, estado_completo: Dict) -> Dict[str, Any]:
        """🎯 Paso Final: Gemini genera análisis final y recomendaciones"""
        estado_serialized = self._serialize_for_json(estado_completo)

        prompt = f"""
        Como experto Liverpool, genera el análisis final de esta predicción FEE:

        ESTADO_COMPLETO: {json.dumps(estado_serialized, indent=2)}

        Proporciona:
        1. Resumen ejecutivo de la predicción
        2. Nivel de confianza
        3. Factores de riesgo principales
        4. Recomendaciones operativas

        Responde en JSON:
        {{
            "resumen_ejecutivo": "string",
            "confianza_prediccion": float (0-1),
            "factores_riesgo": ["riesgo1", ...],
            "recomendaciones": ["rec1", ...],
            "alertas": ["alerta1", ...],
            "siguiente_revision": "string fecha"
        }}
        """

        response = await self.model.generate_content_async(prompt)
        return self._parse_json_response(response.text)

    # Mejorar el método _parse_json_response en services/ai/gemini_service.py

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """🔧 Parser robusto para respuestas JSON de Gemini"""
        try:
            # Limpiar markdown si existe
            clean_text = response_text.strip()

            # Remover ```json y ```
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.startswith("```"):
                clean_text = clean_text[3:]

            # Buscar el final del JSON
            if "```" in clean_text:
                clean_text = clean_text.split("```")[0]

            # Encontrar el último } válido
            # Esto maneja casos donde Gemini agrega texto después del JSON
            brace_count = 0
            last_valid_pos = 0

            for i, char in enumerate(clean_text):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_valid_pos = i + 1
                        break

            if last_valid_pos > 0:
                clean_text = clean_text[:last_valid_pos]

            # Intentar parsear
            parsed = json.loads(clean_text.strip())
            logger.debug("✅ JSON parseado correctamente")
            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parsing Gemini JSON: {e}")
            logger.error(f"Texto recibido: {response_text[:500]}...")  # Log primeros 500 chars

            # Fallback response mejorado
            return {
                "error": "JSON parsing failed",
                "raw_response": response_text[:200],  # Solo primeros 200 chars
                "factible": True,
                "decision": "Análisis automático por error de parsing",
                "score": 0.5,
                "es_zona_roja": False,
                "tipo_flota_requerida": "FI",
                "nivel_riesgo": "Medio",
                "factores": ["Error en parsing - usando valores default"],
                "precauciones": [],
                "criticidad": "Media",
                "impacto_tiempo_horas": 0,
                "impacto_costo_pct": 0.0,
                "factores_criticos": [],
                "estrategias": [],
                "score_factores": 0.5
            }