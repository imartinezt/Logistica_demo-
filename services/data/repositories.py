# services/data/repositories.py
import polars as pl
from pathlib import Path
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
from datetime import datetime

from models.schemas import InsightProductos, InsightInventarios, InsightRutas
from utils.logger import logger


class BaseRepository(ABC):
    """📊 Repositorio base para manejo de CSV con cache"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._cache = {}

    def clear_cache(self):
        self._cache.clear()

    def load_data(self) -> pl.DataFrame:  # pragma: no cover – sobrescrito
        raise NotImplementedError



class ProductRepository(BaseRepository):
    """📦 Repositorio de productos softline"""

    def load_data(self) -> pl.DataFrame:
        if 'products' not in self._cache:
            self._cache['products'] = pl.read_csv(
                self.data_dir / "productos_softline.csv"
            )
        return self._cache['products']

    def get_product_by_sku(self, sku_id: str) -> Optional[Dict]:
        """🔍 Busca producto por SKU"""
        df = self.load_data()
        result = df.filter(pl.col('sku_id') == sku_id)
        return result.to_dicts()[0] if len(result) > 0 else None

    def get_insights(self) -> InsightProductos:
        """📊 Genera insights del catálogo"""
        df = self.load_data()

        # Productos por categoría
        por_categoria = df.group_by('categoria').agg(pl.count().alias('count')).to_dicts()
        categoria_dict = {row['categoria']: row['count'] for row in por_categoria}

        # Productos por nivel demanda
        por_demanda = df.group_by('nivel_demanda').agg(pl.count().alias('count')).to_dicts()
        demanda_dict = {row['nivel_demanda']: row['count'] for row in por_demanda}

        # Productos temporada actual (ejemplo: invierno)
        mes_actual = datetime.now().month
        temporada_actual = "Invierno" if mes_actual in [12, 1, 2] else "Verano" if mes_actual in [6, 7,
                                                                                                  8] else "Primavera"
        productos_temporada = df.filter(pl.col('temporada') == temporada_actual).to_dicts()

        # Top productos caros
        top_caros = df.sort('precio_venta', descending=True).head(10).to_dicts()

        return InsightProductos(
            total_productos=len(df),
            por_categoria=categoria_dict,
            por_nivel_demanda=demanda_dict,
            productos_fragiles=len(df.filter(pl.col('es_fragil') == True)),
            peso_promedio=float(df['peso_kg'].mean()),
            precio_promedio=float(df['precio_venta'].mean()),
            productos_temporada_actual=productos_temporada[:5],
            top_productos_caros=top_caros
        )


class NodeRepository(BaseRepository):
    """📍 Repositorio de nodos/ubicaciones"""

    def load_data(self) -> pl.DataFrame:
        if 'nodes' not in self._cache:
            self._cache['nodes'] = pl.read_csv(
                self.data_dir / "nodos_ubicaciones.csv"
            )
        return self._cache['nodes']

    def find_closest_nodes_to_cp(self, codigo_postal: str, cp_df: pl.DataFrame) -> pl.DataFrame:
        """📏 Encuentra nodos más cercanos usando coordenadas reales"""
        from utils.distance_calculator import DistanceCalculator

        # Obtener coordenadas del CP destino
        cp_info = cp_df.filter(pl.col('codigo_postal') == int(codigo_postal))
        if len(cp_info) == 0:
            return self.load_data()  # Return all if CP not found

        cp_lat, cp_lon = cp_info[0, 'latitud'], cp_info[0, 'longitud']

        # Calcular distancias reales
        nodes_df = self.load_data()
        distances = []

        for row in nodes_df.iter_rows(named=True):
            dist = DistanceCalculator.calculate_distance_km(
                cp_lat, cp_lon, row['latitud'], row['longitud']
            )
            distances.append(dist)

        # Agregar columna de distancia y ordenar
        nodes_with_dist = nodes_df.with_columns(
            pl.Series("distancia_km", distances)
        ).sort("distancia_km")

        return nodes_with_dist


class InventoryRepository(BaseRepository):
    """📦 Repositorio de inventarios OH"""

    def load_data(self) -> pl.DataFrame:
        if 'inventory' not in self._cache:
            self._cache['inventory'] = pl.read_csv(
                self.data_dir / "inventarios_oh.csv"
            )
        return self._cache['inventory']

    def get_available_stock(self, sku_id: str, cantidad_requerida: int) -> pl.DataFrame:
        """📋 Retorna ubicaciones con stock OH suficiente"""
        df = self.load_data()
        return df.filter(
            (pl.col('sku_id') == sku_id) &
            (pl.col('stock_oh') >= cantidad_requerida)
        ).sort('stock_oh', descending=True)

    def get_insights(self) -> InsightInventarios:
        """📊 Genera insights de inventarios"""
        df = self.load_data()

        # Stock por nodo
        stock_por_nodo = df.group_by('nodo_id').agg([
            pl.col('stock_oh').sum().alias('total_stock')
        ]).to_dicts()

        utilizacion = {str(row['nodo_id']): row['total_stock'] for row in stock_por_nodo}

        # Productos bajo stock
        bajo_stock = df.filter(
            pl.col('stock_oh') < (pl.col('safety_factor') * 2)
        ).select(['sku_id', 'nodo_id', 'stock_oh', 'safety_factor']).to_dicts()

        # Rotación promedio (estimada)
        rotacion_promedio = df['demanda_proyectada'].sum() / df['stock_oh'].sum() if df['stock_oh'].sum() > 0 else 0

        # Nodos críticos (bajo stock)
        nodos_criticos = [str(row['nodo_id']) for row in stock_por_nodo if row['total_stock'] < 100]

        return InsightInventarios(
            total_skus=len(df['sku_id'].unique()),
            stock_total=int(df['stock_fisico'].sum()),
            stock_disponible=int(df['stock_oh'].sum()),
            utilizacion_por_nodo=utilizacion,
            productos_bajo_stock=bajo_stock[:10],  # Top 10
            rotacion_promedio=float(rotacion_promedio),
            nodos_criticos=nodos_criticos
        )


class RouteRepository(BaseRepository):
    """🚚 Repositorio de rutas combinadas"""

    def load_data(self) -> pl.DataFrame:
        if "routes" not in self._cache:
            self._cache["routes"] = pl.read_csv(self.data_dir / "rutas_combinadas.csv")
        return self._cache["routes"]

    def get_feasible_routes(self, nodo_origen: int, codigo_postal_destino: str) -> pl.DataFrame:
        df = self.load_data()
        cp_dest = int(codigo_postal_destino)
        return (
            df.filter(
                (pl.col("nodo_origen") == nodo_origen)
                & (pl.col("codigo_postal_destino") == cp_dest)
                & (pl.col("activa") == True)
            )
            .sort("probabilidad_cumplimiento", descending=True)
        )

    def get_insights(self) -> InsightRutas:
        """📊 Genera insights de rutas"""
        df = self.load_data()

        # Rutas por tipo flota
        por_flota = df.group_by('tipo_flota').agg(pl.count().alias('count')).to_dicts()
        flota_dict = {row['tipo_flota']: row['count'] for row in por_flota}

        # Rutas más eficientes
        eficientes = df.sort(['tiempo_total_horas', 'costo_total_mxn']).head(5).to_dicts()

        # Rutas problemáticas (baja probabilidad)
        problematicas = df.filter(pl.col('probabilidad_cumplimiento') < 0.7).to_dicts()

        return InsightRutas(
            total_rutas=len(df),
            rutas_activas=len(df.filter(pl.col('activa') == True)),
            por_tipo_flota=flota_dict,
            tiempo_promedio=float(df['tiempo_total_horas'].mean()),
            costo_promedio=float(df['costo_total_mxn'].mean()),
            probabilidad_promedio=float(df['probabilidad_cumplimiento'].mean()),
            rutas_mas_eficientes=eficientes,
            rutas_problematicas=problematicas[:5]
        )


class PostalCodeRepository(BaseRepository):
    """📮 Repositorio de códigos postales"""

    def load_data(self) -> pl.DataFrame:
        if 'postal_codes' not in self._cache:
            self._cache['postal_codes'] = pl.read_csv(
                self.data_dir / "codigos_postales_destino.csv"
            )
        return self._cache['postal_codes']

    def get_cp_info(self, codigo_postal: str) -> Optional[Dict]:
        """🔍 Obtiene información de un código postal"""
        df = self.load_data()
        result = df.filter(pl.col('codigo_postal') == int(codigo_postal))
        return result.to_dicts()[0] if len(result) > 0 else None

    def is_zona_roja(self, codigo_postal: str) -> bool:
        """🚨 Detecta zona roja basado en nivel_seguridad"""
        cp_info = self.get_cp_info(codigo_postal)
        if not cp_info:
            return True  # Si no conocemos el CP, asumimos riesgo
        return cp_info['nivel_seguridad'] == 'Bajo'


class ExternalFactorsRepository(BaseRepository):
    """🌤️ Repositorio de factores externos"""

    def load_data(self) -> pl.DataFrame:
        if 'factors' not in self._cache:
            self._cache['factors'] = pl.read_csv(
                self.data_dir / "factores_externos.csv"
            )
        return self._cache['factors']

    def get_factors_or_predict(self, fecha: datetime, zona: str) -> Dict:
        """🔮 Obtiene factores del CSV o los predice automáticamente"""
        fecha_str = fecha.date().isoformat()
        df = self.load_data()

        # Buscar en CSV primero
        result = df.filter(
            (pl.col('fecha') == fecha_str) &
            (pl.col('zona') == zona)
        )

        if len(result) > 0:
            return result.to_dicts()[0]

        # Si no existe, generar automáticamente
        from utils.temporal_detector import TemporalFactorDetector
        detected_factors = TemporalFactorDetector.detect_seasonal_factors(fecha)

        return {
            'fecha': fecha_str,
            'zona': zona,
            'condicion_clima': detected_factors['condicion_clima'],
            'probabilidad_lluvia': detected_factors['probabilidad_lluvia'],
            'temperatura': detected_factors['temperatura'],
            'trafico_nivel': detected_factors['trafico_nivel'],
            'evento_liverpool': detected_factors['eventos_detectados'][0] if detected_factors[
                'eventos_detectados'] else 'Normal',
            'factor_demanda': detected_factors['factor_demanda'],
            'impacto_tiempo_extra_horas': detected_factors['impacto_tiempo_extra'],
            'indice_seguridad': 'Alto'
        }