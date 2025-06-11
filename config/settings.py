from pathlib import Path
from typing import List, Dict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración global del sistema FEE"""

    # ------------------------------------------------------------------
    # App metadata
    # ------------------------------------------------------------------
    APP_NAME: str = "Liverpool FEE Predictor"
    VERSION: str = "2.0.0"
    DEBUG: bool = False

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"

    # ------------------------------------------------------------------
    # Gemini configuration
    # ------------------------------------------------------------------
    PROJECT_ID: str = "crp-dev-dig-mlcatalog"
    REGION: str = "us-central1"
    MODEL_NAME: str = "gemini-2.0-flash-001"
    GOOGLE_CREDENTIALS_PATH: str = "keys.json"

    # ------------------------------------------------------------------
    # Business rules
    # ------------------------------------------------------------------
    HORARIO_CORTE_FLASH: int = 12   # 12 h – same‑day
    HORARIO_CORTE_EXPRESS: int = 21 # 21 h – 48 h
    NIVEL_SEGURIDAD_MINIMO: str = "Medio"

    # ------------------------------------------------------------------
    # Optimisation weights
    # ------------------------------------------------------------------
    PESO_TIEMPO: float = 0.4
    PESO_COSTO: float = 0.3
    PESO_PROBABILIDAD: float = 0.3
    PESO_DISTANCIA: float = 0.1   #  ➕ nueva penalización por kilómetro

    # ------------------------------------------------------------------
    # Calendario de eventos estacionales
    # ------------------------------------------------------------------
    EVENTOS_TEMPORADA: Dict[int, List[str]] = {
        12: ["Navidad", "Nochebuena", "Fin_Ano"],
        11: ["Buen_Fin", "Black_Friday"],
        2:  ["San_Valentin", "Dia_Amor"],
        5:  ["Dia_Madres", "Mayo"],
        9:  ["Fiestas_Patrias", "Independencia"],
        10: ["Halloween", "Dia_Muertos"],
        1:  ["Dia_Reyes", "Ano_Nuevo"],
    }

    # ------------------------------------------------------------------
    # Clima por temporada
    # ------------------------------------------------------------------
    CLIMA_TEMPORADA: Dict[int, Dict] = {
        12: {"condicion": "Frio",      "lluvia_prob": 15, "temp": 18},
        1:  {"condicion": "Frio",      "lluvia_prob": 10, "temp": 16},
        2:  {"condicion": "Templado",  "lluvia_prob": 20, "temp": 20},
        6:  {"condicion": "Lluvioso",  "lluvia_prob": 70, "temp": 25},
        7:  {"condicion": "Lluvioso",  "lluvia_prob": 80, "temp": 24},
        8:  {"condicion": "Lluvioso",  "lluvia_prob": 75, "temp": 23},
    }

    class Config:
        env_file = ".env"


settings = Settings()