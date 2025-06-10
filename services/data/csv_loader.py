# services/data/csv_loader.py
import polars as pl
from pathlib import Path
from typing import Dict, Optional
import time

from utils.logger import logger


class CSVLoader:
    """📂 Cargador y cache de archivos CSV"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._cache: Dict[str, pl.DataFrame] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self.cache_ttl = 3600  # 1 hora TTL

    def load_csv(self, filename: str, force_reload: bool = False) -> pl.DataFrame:
        """📋 Carga CSV con cache automático"""
        current_time = time.time()

        # Check if we need to reload
        if (force_reload or
                filename not in self._cache or
                current_time - self._cache_timestamps.get(filename, 0) > self.cache_ttl):

            try:
                file_path = self.data_dir / filename
                if not file_path.exists():
                    raise FileNotFoundError(f"CSV no encontrado: {file_path}")

                logger.info(f"📂 Cargando CSV: {filename}")
                df = pl.read_csv(file_path)

                self._cache[filename] = df
                self._cache_timestamps[filename] = current_time

                logger.info(f"✅ CSV cargado: {filename} ({len(df)} filas)")

            except Exception as e:
                logger.error(f"❌ Error cargando CSV {filename}: {e}")
                raise

        return self._cache[filename]

    def get_cache_info(self) -> Dict[str, Dict]:
        """ℹ️ Información del cache"""
        current_time = time.time()
        info = {}

        for filename, df in self._cache.items():
            timestamp = self._cache_timestamps.get(filename, 0)
            age_minutes = (current_time - timestamp) / 60

            info[filename] = {
                "filas": len(df),
                "columnas": len(df.columns),
                "edad_minutos": round(age_minutes, 1),
                "expires_in_minutes": round((self.cache_ttl - (current_time - timestamp)) / 60, 1)
            }

        return info

    def clear_cache(self, filename: Optional[str] = None):
        """🗑️ Limpiar cache"""
        if filename:
            self._cache.pop(filename, None)
            self._cache_timestamps.pop(filename, None)
            logger.info(f"🗑️ Cache limpiado para: {filename}")
        else:
            self._cache.clear()
            self._cache_timestamps.clear()
            logger.info("🗑️ Todo el cache limpiado")

    def preload_all_csvs(self):
        """🚀 Pre-carga todos los CSVs principales"""
        csv_files = [
            "productos_softline.csv",
            "nodos_ubicaciones.csv",
            "inventarios_oh.csv",
            "rutas_combinadas.csv",
            "codigos_postales_destino.csv",
            "factores_externos.csv"
        ]

        logger.info("🚀 Pre-cargando todos los CSVs...")

        for csv_file in csv_files:
            try:
                self.load_csv(csv_file)
            except FileNotFoundError:
                logger.warning(f"⚠️ CSV no encontrado (opcional): {csv_file}")
            except Exception as e:
                logger.error(f"❌ Error pre-cargando {csv_file}: {e}")

        logger.info("✅ Pre-carga de CSVs completada")