import time
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional, List, Dict, Any

import polars as pl

from utils.logger import logger


class DataManager:
    """🚀 Gestor de datos optimizado - CSVs en memoria, sin cache de respuestas"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._data: Dict[str, pl.DataFrame] = {}
        self._load_time = None
        self._load_all_csvs()

    @staticmethod
    def _fix_lat(lat_val):
        """Corrige latitudes corruptas 194.326 → 19.4326"""
        if lat_val > 90:
            lat_str = str(abs(lat_val))
            return float(lat_str[:2] + '.' + lat_str[2:].replace('.', ''))
        return lat_val

    @staticmethod
    def _fix_lon(lon_val):
        """Corrige longitudes corruptas 991.332 → -99.1332"""
        if lon_val > 180:
            lon_str = str(abs(lon_val))
            return -float(lon_str[:2] + '.' + lon_str[2:].replace('.', ''))
        return lon_val

    def _load_all_csvs(self):
        """📂 Carga Y PRE-PROCESA todos los CSVs optimizados"""
        start_time = time.time()

        # CSV con nombres exactos que mencionaste
        csv_files = {
            'tiendas': 'liverpool_tiendas_completo.csv',
            'productos': 'productos_liverpool_50.csv',
            'stock': 'stock_tienda_sku.csv',
            'cedis': 'cedis_liverpool_completo.csv',
            'codigos_postales': 'codigos_postales_rangos_mexico.csv',
            'factores_externos': 'factores_externos_mexico_completo.csv',
            'flota_externa': 'flota_externa_costos_reales.csv',
            'clima': 'clima_por_rango_cp.csv'
        }

        for key, filename in csv_files.items():
            df = pl.read_csv(self.data_dir / filename)

            # PRE-PROCESAR según tus especificaciones
            if key == 'tiendas':
                # Corregir coordenadas corruptas AUTOMÁTICAMENTE
                df = df.with_columns([
                    pl.col('latitud').map_elements(self._fix_lat, return_dtype=pl.Float64),
                    pl.col('longitud').map_elements(self._fix_lon).alias('longitud')
                ])

            elif key == 'cedis':
                # Corregir coordenadas CEDIS también
                df = df.with_columns([
                    pl.col('latitud').map_elements(self._fix_lat, return_dtype=pl.Float64),
                    pl.col('longitud').map_elements(self._fix_lon).alias('longitud')
                ])

            elif key == 'productos':
                # Pre-procesar tiendas_disponibles como lista
                df = df.with_columns([
                    pl.col('tiendas_disponibles').str.split(',').alias('tiendas_list')
                ])

            self._data[key] = df

    def get_data(self, key: str) -> pl.DataFrame:
        """📊 Obtiene DataFrame desde memoria"""
        return self._data.get(key, pl.DataFrame())


class OptimizedProductRepository:
    """📦 Repositorio de productos optimizado"""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def get_product_by_sku(self, sku_id: str) -> Optional[Dict[str, Any]]:
        """🔍 Busca producto por SKU"""
        df = self.data_manager.get_data('productos')
        result = df.filter(pl.col('sku_id') == sku_id)

        if result.height == 0:
            logger.warning(f"❌ Producto no encontrado: {sku_id}")
            return None

        product = result.to_dicts()[0]
        logger.info(f"📦 Producto encontrado: {sku_id} - {product.get('nombre_producto', 'N/A')}")
        return product


class OptimizedStoreRepository:
    """🏪 Repositorio de tiendas optimizado"""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def find_stores_by_postal_range(self, codigo_postal: str) -> List[Dict[str, Any]]:
        """📍 Encuentra tiendas por rango de código postal"""
        # 1. Primero buscar en qué rango está el CP
        cp_info = self._get_postal_info(codigo_postal)
        if not cp_info:
            logger.warning(f"❌ CP no encontrado: {codigo_postal}")
            return []

        # 2. Buscar tiendas en el mismo estado/alcaldía
        target_state = cp_info['estado_alcaldia']
        tiendas_df = self.data_manager.get_data('tiendas')

        # Filtrar por estado/alcaldía similar
        state_matches = tiendas_df.filter(
            pl.col('estado').str.contains(target_state.split()[0]) |
            pl.col('alcaldia_municipio').str.contains(target_state.split()[0])
        )

        if state_matches.height == 0:
            # Fallback: buscar en CDMX o Estado de México
            fallback_states = ['Ciudad de México', 'Estado de México']
            for state in fallback_states:
                state_matches = tiendas_df.filter(pl.col('estado').str.contains(state))
                if state_matches.height > 0:
                    break

        stores = state_matches.to_dicts()

        # 3. Calcular distancias reales
        target_lat = cp_info['latitud_centro']
        target_lon = cp_info['longitud_centro']

        stores_with_distance = []
        for store in stores:
            try:
                from utils.geo_calculator import GeoCalculator
                distance = GeoCalculator.calculate_distance_km(
                    target_lat, target_lon,
                    float(store['latitud']), float(store['longitud'])
                )
                store['distancia_km'] = distance
                stores_with_distance.append(store)
            except Exception as e:
                logger.warning(f"⚠️ Error calculando distancia para {store.get('tienda_id')}: {e}")

        # Ordenar por distancia
        stores_with_distance.sort(key=lambda x: x['distancia_km'])

        logger.info(f"📍 Tiendas encontradas para {codigo_postal}: {len(stores_with_distance)}")
        for i, store in enumerate(stores_with_distance[:5]):
            logger.info(f"  {i + 1}. {store['tienda_id']} - {store['nombre_tienda']} ({store['distancia_km']:.1f}km)")

        return stores_with_distance

    def _get_postal_info(self, codigo_postal: str) -> Optional[Dict[str, Any]]:
        """🔍 Obtiene información del código postal"""
        df = self.data_manager.get_data('codigos_postales')
        cp_int = int(codigo_postal)

        for row in df.to_dicts():
            rango_cp = row.get('rango_cp', '')
            if '-' in rango_cp:
                try:
                    start_cp, end_cp = map(int, rango_cp.split('-'))
                    if start_cp <= cp_int <= end_cp:
                        return row
                except ValueError:
                    continue

        # Fallback: usar CDMX
        return {
            'rango_cp': codigo_postal,
            'estado_alcaldia': 'Ciudad de México',
            'zona_seguridad': 'Amarilla',
            'latitud_centro': 19.4326,
            'longitud_centro': -99.1332,
            'cobertura_liverpool': True,
            'tiempo_entrega_base_horas': '2-4'
        }


class OptimizedStockRepository:
    """📦 Repositorio de stock optimizado"""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def get_stock_for_stores_and_sku(self, sku_id: str, store_ids: List[str],
                                     cantidad_requerida: int) -> List[Dict[str, Any]]:
        """📊 Obtiene stock específico para tiendas y SKU"""
        stock_df = self.data_manager.get_data('stock')

        # Filtrar por SKU y tiendas
        filtered_stock = stock_df.filter(
            (pl.col('sku_id') == sku_id) &
            (pl.col('tienda_id').is_in(store_ids)) &
            (pl.col('stock_disponible') > 0)
        ).sort('stock_disponible', descending=True)

        stock_locations = filtered_stock.to_dicts()

        logger.info(f"📦 Stock encontrado para {sku_id} en {len(store_ids)} tiendas:")
        total_available = 0
        for stock in stock_locations:
            total_available += stock['stock_disponible']
            logger.info(f"  📍 {stock['tienda_id']}: {stock['stock_disponible']} unidades")

        logger.info(f"📊 Stock total disponible: {total_available} | Requerido: {cantidad_requerida}")

        return stock_locations

    def calculate_optimal_allocation(self, stock_locations: List[Dict[str, Any]],
                                     cantidad_requerida: int,
                                     stores_info: List[Dict[str, Any]],
                                     sku_id: str = None,
                                     codigo_postal: str = None,
                                     fecha_entrega: str = None) -> Dict[str, Any]:
        """🧠 SISTEMA DE ASIGNACIÓN OPTIMIZADO con logs profesionales y datos reales"""

        if not stock_locations:
            return {'factible': False, 'razon': 'Sin stock disponible', 'plan': []}

        # ✅ 1. CONTEXTO DEL DESTINO
        contexto = self._get_destination_context(codigo_postal, fecha_entrega) if codigo_postal else {}

        logger.info(f"")
        logger.info(f"🎯 ANÁLISIS DE ASIGNACIÓN OPTIMIZADA")
        logger.info(f"{'=' * 60}")
        logger.info(f"📦 SKU: {sku_id} | Cantidad: {cantidad_requerida} unidades")

        if codigo_postal:
            logger.info(f"📍 Destino: CP {codigo_postal}")

            if contexto.get('cp_info'):
                cp_data = contexto['cp_info']
                logger.info(f"   🏙️ Zona: {cp_data.get('estado_alcaldia', 'N/A')}")
                logger.info(f"   🛡️ Seguridad: {cp_data.get('zona_seguridad', 'N/A')}")
                logger.info(f"   ⏰ Tiempo base: {cp_data.get('tiempo_entrega_base_horas', 'N/A')} horas")

            if contexto.get('factores_externos'):
                factores = contexto['factores_externos']
                logger.info(f"   📅 Evento: {factores.get('evento_detectado', 'Normal')}")
                logger.info(f"   📈 Factor demanda: {factores.get('factor_demanda', '1.0')}x")
                logger.info(f"   🌤️ Clima: {factores.get('condicion_clima', 'Normal')}")
                logger.info(f"   🚦 Tráfico: {factores.get('trafico_nivel', 'Normal')}")
                logger.info(f"   ⚠️ Criticidad: {factores.get('criticidad_logistica', 'Baja')}")

        # ✅ 2. EVALUACIÓN DE CANDIDATOS CON DATOS REALES
        logger.info(f"")
        logger.info(f"📊 EVALUACIÓN DE CANDIDATOS")
        logger.info(f"{'-' * 60}")

        # Crear mapas de datos
        distance_map = {store['tienda_id']: store['distancia_km'] for store in stores_info}

        # ✅ CORRECCIÓN DEL BUG: Obtener nombres reales de tiendas
        name_map = {}
        tiendas_df = self.data_manager.get_data('tiendas')
        for store in stores_info:
            tienda_real = tiendas_df.filter(pl.col('tienda_id') == store['tienda_id'])
            if tienda_real.height > 0:
                name_map[store['tienda_id']] = tienda_real.to_dicts()[0]['nombre_tienda']
            else:
                name_map[store['tienda_id']] = f"Tienda {store['tienda_id']}"

        # Calcular métricas reales para cada candidato
        candidates_data = []

        for stock in stock_locations:
            tienda_id = stock['tienda_id']
            distancia = distance_map.get(tienda_id, 999)
            nombre_tienda = name_map.get(tienda_id, f"Tienda {tienda_id}")

            # ✅ CALCULAR TIEMPO Y COSTO REALES
            metrics = self._calculate_real_time_and_cost(
                tienda_id, sku_id, cantidad_requerida, distancia
            ) if sku_id else self._calculate_fallback_metrics(distancia, cantidad_requerida)

            candidates_data.append({
                'tienda_id': tienda_id,
                'nombre_tienda': nombre_tienda,
                'stock_disponible': stock['stock_disponible'],
                'distancia_km': distancia,
                'precio_tienda': stock.get('precio_tienda', 0),
                **metrics,
                'stock_data': stock
            })

        if not candidates_data:
            return {'factible': False, 'razon': 'No hay candidatos válidos', 'plan': []}

        # ✅ 3. NORMALIZACIÓN DE SCORES (0-1)
        tiempos = [c['tiempo_total_horas'] for c in candidates_data]
        costos = [c['costo_total'] for c in candidates_data]
        distancias = [c['distancia_km'] for c in candidates_data]
        stocks = [c['stock_disponible'] for c in candidates_data]

        min_tiempo, max_tiempo = min(tiempos), max(tiempos)
        min_costo, max_costo = min(costos), max(costos)
        min_distancia, max_distancia = min(distancias), max(distancias)
        min_stock, max_stock = min(stocks), max(stocks)

        logger.info(f"📈 PESOS UTILIZADOS:")
        logger.info(f"   ⏱️ Tiempo: 35% | 💰 Costo: 35% | 📦 Stock: 20% | 📏 Distancia: 10%")
        logger.info(f"")

        # Calcular scores normalizados
        for candidate in candidates_data:
            # Scores normalizados (0-1, donde 1 = mejor)
            score_tiempo = 1 - (candidate['tiempo_total_horas'] - min_tiempo) / max(max_tiempo - min_tiempo, 0.1)
            score_costo = 1 - (candidate['costo_total'] - min_costo) / max(max_costo - min_costo, 0.1)
            score_distancia = 1 - (candidate['distancia_km'] - min_distancia) / max(max_distancia - min_distancia, 0.1)
            score_stock = 1.0 if candidate['stock_disponible'] >= cantidad_requerida else 0.6

            # Score total ponderado
            PESO_TIEMPO, PESO_COSTO, PESO_STOCK, PESO_DISTANCIA = 0.35, 0.35, 0.20, 0.10
            total_score = (
                    score_tiempo * PESO_TIEMPO +
                    score_costo * PESO_COSTO +
                    score_stock * PESO_STOCK +
                    score_distancia * PESO_DISTANCIA
            )

            candidate['scores'] = {
                'tiempo': score_tiempo,
                'costo': score_costo,
                'distancia': score_distancia,
                'stock': score_stock,
                'total': total_score
            }

        # ✅ 4. RANKING FINAL CON TABLA PROFESIONAL
        candidates_data.sort(key=lambda x: x['scores']['total'], reverse=True)

        logger.info(f"📊 RUTAS EVALUADAS")
        logger.info(
            f"{'Tienda':<20} │ {'Dist':<8} │ {'Tiempo':<8} │ {'Costo':<10} │ {'Stock':<8} │ {'Score':<8} │ {'Flota'}")
        logger.info(f"{'─' * 20}┼{'─' * 9}┼{'─' * 9}┼{'─' * 11}┼{'─' * 9}┼{'─' * 9}┼{'─' * 10}")

        for i, candidate in enumerate(candidates_data[:6]):  # Mostrar top 6
            nombre = candidate['nombre_tienda'][:19].ljust(19)
            dist = f"{candidate['distancia_km']:.0f}km"
            tiempo = f"{candidate['tiempo_total_horas']:.1f}h"
            costo = f"${candidate['costo_total']:.0f}"
            stock = f"{candidate['stock_disponible']}"
            score = f"{candidate['scores']['total']:.3f}"
            flota = f"{candidate['fleet_type']}-{candidate['carrier'][:4]}"

            # Destacar el ganador
            marker = "🏆" if i == 0 else f"{i + 1:2d}."

            logger.info(f"{marker} {nombre} │ {dist:>7} │ {tiempo:>7} │ {costo:>9} │ {stock:>7} │ {score:>7} │ {flota}")

        # ✅ 5. GANADOR Y EXPLICACIÓN DETALLADA
        if candidates_data:
            ganador = candidates_data[0]
            logger.info(f"")
            logger.info(f"🏆 GANADOR: {ganador['nombre_tienda']}")
            logger.info(f"{'─' * 50}")

            scores = ganador['scores']
            explicaciones = []

            if scores['tiempo'] >= 0.8:
                explicaciones.append(f"⚡ Tiempo excelente ({ganador['tiempo_total_horas']}h)")
            elif scores['tiempo'] >= 0.6:
                explicaciones.append(f"⏱️ Tiempo competitivo ({ganador['tiempo_total_horas']}h)")

            if scores['costo'] >= 0.8:
                explicaciones.append(f"💰 Costo óptimo (${ganador['costo_total']:.0f})")
            elif scores['costo'] >= 0.6:
                explicaciones.append(f"💵 Costo aceptable (${ganador['costo_total']:.0f})")

            if scores['stock'] == 1.0:
                explicaciones.append("📦 Stock completo disponible")

            if ganador['fleet_type'] == 'FI':
                explicaciones.append("🚛 Flota interna (más rápido)")
            else:
                explicaciones.append(f"🚚 Flota externa ({ganador['carrier']})")

            logger.info(f"📋 Ventajas clave:")
            for exp in explicaciones:
                logger.info(f"   {exp}")

            logger.info(f"")
            logger.info(f"📈 Desglose del score (Total: {scores['total']:.3f}):")
            logger.info(f"   ⏱️ Tiempo: {scores['tiempo']:.3f} × 35% = {scores['tiempo'] * 0.35:.3f}")
            logger.info(f"   💰 Costo:  {scores['costo']:.3f} × 35% = {scores['costo'] * 0.35:.3f}")
            logger.info(f"   📦 Stock:  {scores['stock']:.3f} × 20% = {scores['stock'] * 0.20:.3f}")
            logger.info(f"   📏 Dist:   {scores['distancia']:.3f} × 10% = {scores['distancia'] * 0.10:.3f}")

        # ✅ 6. PLAN DE ASIGNACIÓN
        plan = []
        cantidad_cubierta = 0

        for candidate in candidates_data:
            if cantidad_cubierta >= cantidad_requerida:
                break

            cantidad_a_tomar = min(
                candidate['stock_disponible'],
                cantidad_requerida - cantidad_cubierta
            )

            if cantidad_a_tomar > 0:
                plan.append({
                    'tienda_id': candidate['tienda_id'],
                    'nombre_tienda': candidate['nombre_tienda'],
                    'cantidad': cantidad_a_tomar,
                    'stock_disponible': candidate['stock_disponible'],
                    'distancia_km': candidate['distancia_km'],
                    'precio_tienda': candidate['precio_tienda'],
                    'tiempo_total_horas': candidate['tiempo_total_horas'],
                    'costo_total': candidate['costo_total'],
                    'fleet_type': candidate['fleet_type'],
                    'carrier': candidate['carrier'],
                    'score_total': candidate['scores']['total'],
                    'zona_seguridad': candidate.get('zona_seguridad', 'N/A'),
                    'razon_seleccion': self._get_detailed_selection_reason(candidate, candidates_data)
                })

                cantidad_cubierta += cantidad_a_tomar

        logger.info(f"")
        logger.info(f"✅ ASIGNACIÓN COMPLETADA: {cantidad_cubierta}/{cantidad_requerida} unidades")
        logger.info(f"{'=' * 60}")

        return {
            'factible': cantidad_cubierta >= cantidad_requerida,
            'plan': plan,
            'cantidad_cubierta': cantidad_cubierta,
            'cantidad_faltante': max(0, cantidad_requerida - cantidad_cubierta),
            'razon': f'Optimización tiempo-costo con {len(plan)} tienda(s)',
            'contexto_destino': contexto,
            'candidates_evaluated': len(candidates_data),
            'selection_method': 'tiempo_costo_optimizado_v2'
        }

    def _get_destination_context(self, codigo_postal: str, fecha_entrega: str) -> Dict[str, Any]:
        """🎯 Obtiene contexto COMPLETO del destino usando los CSVs"""

        if not codigo_postal:
            return {}

        # Obtener información del CP
        cp_info = None
        cp_df = self.data_manager.get_data('codigos_postales')
        cp_int = int(codigo_postal)

        for row in cp_df.to_dicts():
            rango_cp = row.get('rango_cp', '')
            if '-' in rango_cp:
                try:
                    inicio, fin = map(int, rango_cp.split('-'))
                    if inicio <= cp_int <= fin:
                        cp_info = row
                        break
                except ValueError:
                    continue

        # Obtener clima
        clima_info = None
        clima_df = self.data_manager.get_data('clima')
        for row in clima_df.to_dicts():
            if row['rango_cp_inicio'] <= cp_int <= row['rango_cp_fin']:
                clima_info = row
                break

        # Obtener factores externos para la fecha
        factores_externos = {}
        if fecha_entrega:
            from datetime import datetime
            try:
                fecha_dt = datetime.strptime(fecha_entrega, '%Y-%m-%d')
                factores_externos = self.data_manager.external_factors.get_factors_for_date_and_cp(fecha_dt,
                                                                                                   codigo_postal)
            except:
                factores_externos = {}

        return {
            'cp_info': cp_info or {},
            'clima_info': clima_info or {},
            'factores_externos': factores_externos,
            'codigo_postal': codigo_postal
        }

    def _get_selection_reason(self, candidate: Dict[str, Any], all_candidates: List[Dict[str, Any]]) -> str:
        """📝 Genera razón de selección para una tienda"""

        scores = candidate['scores']

        reasons = []

        # Razón principal por score
        if candidate == all_candidates[0]:
            reasons.append("Mayor score total")

        # Razones específicas
        if scores['distancia'] >= 8.0:
            reasons.append("Muy cerca")
        elif scores['distancia'] >= 6.0:
            reasons.append("Distancia aceptable")

        if scores['disponibilidad'] == 10.0:
            reasons.append("Cubre demanda completa")

        if scores['stock'] >= 8.0:
            reasons.append("Alto stock")

        if scores['precio'] >= 8.0:
            reasons.append("Precio competitivo")

        return ", ".join(reasons) if reasons else "Mejor opción disponible"

    def _calculate_real_time_and_cost(self, tienda_id: str, sku_id: str,
                                      cantidad: int, distancia_km: float) -> Dict[str, Any]:
        """⏱️💰 Calcula tiempo y costo REALES usando datos CSV"""

        # Obtener info de producto
        productos_df = self.data_manager.get_data('productos')
        producto_data = productos_df.filter(pl.col('sku_id') == sku_id)

        if producto_data.height > 0:
            producto = producto_data.to_dicts()[0]
            tiempo_prep = producto['tiempo_prep_horas']
            peso_total = producto['peso_kg'] * cantidad
        else:
            tiempo_prep = 1.0
            peso_total = 0.5 * cantidad  # Fallback

        # Obtener info de tienda
        tiendas_df = self.data_manager.get_data('tiendas')
        tienda_data = tiendas_df.filter(pl.col('tienda_id') == tienda_id)

        if tienda_data.height > 0:
            tienda = tienda_data.to_dicts()[0]
            zona_seguridad = tienda['zona_seguridad']
        else:
            zona_seguridad = 'Amarilla'

        # ✅ LÓGICA DE FLOTA CORREGIDA
        if distancia_km <= 50:
            # Flota interna (FI)
            fleet_type = 'FI'
            carrier = 'Liverpool'
            velocidad_promedio = 45.0  # km/h en ciudad
            tiempo_viaje = distancia_km / velocidad_promedio

            # Costo flota interna
            costo_base = distancia_km * 12.0  # $12/km
            factor_cantidad = 0.9 if cantidad >= 3 else 1.0
            costo_total = costo_base * factor_cantidad

        else:
            # Flota externa (FE)
            fleet_type = 'FE'
            velocidad_promedio = 65.0  # km/h carretera
            tiempo_viaje = distancia_km / velocidad_promedio

            # Obtener mejor carrier externo usando CSV real
            flota_df = self.data_manager.get_data('flota_externa')
            carriers_disponibles = flota_df.filter(
                (pl.col('activo') == True) &
                (pl.col('peso_min_kg') <= peso_total) &
                (pl.col('peso_max_kg') >= peso_total)
            )

            if carriers_disponibles.height > 0:
                # Calcular costo para cada carrier y tomar el más barato
                carriers_list = carriers_disponibles.to_dicts()
                mejor_carrier = min(carriers_list,
                                    key=lambda x: x['costo_base_mxn'] + (peso_total * x['costo_por_kg_adicional']))

                carrier = mejor_carrier['carrier']
                costo_total = mejor_carrier['costo_base_mxn'] + (peso_total * mejor_carrier['costo_por_kg_adicional'])
            else:
                # Fallback
                carrier = 'Estafeta'
                costo_total = 150.0 + (distancia_km * 8.0)

        # Factores de tiempo adicional
        factor_clima = 0.2 if zona_seguridad == 'Roja' else 0.1
        factor_distancia = 0.3 if distancia_km > 100 else 0.1

        tiempo_total = tiempo_prep + tiempo_viaje + factor_clima + factor_distancia

        return {
            'tiempo_total_horas': round(tiempo_total, 1),
            'costo_total': round(costo_total, 2),
            'fleet_type': fleet_type,
            'carrier': carrier,
            'tiempo_prep': tiempo_prep,
            'tiempo_viaje': round(tiempo_viaje, 1),
            'peso_total_kg': peso_total,
            'zona_seguridad': zona_seguridad
        }

    def _calculate_fallback_metrics(self, distancia_km: float, cantidad: int) -> Dict[str, Any]:
        """🔄 Métricas de fallback cuando no hay datos completos"""

        if distancia_km <= 50:
            fleet_type = 'FI'
            carrier = 'Liverpool'
            tiempo_total = 1.0 + (distancia_km / 45.0) + 0.2
            costo_total = distancia_km * 12.0 * (0.9 if cantidad >= 3 else 1.0)
        else:
            fleet_type = 'FE'
            carrier = 'Estafeta'
            tiempo_total = 1.0 + (distancia_km / 65.0) + 0.3
            costo_total = 150.0 + (distancia_km * 8.0)

        return {
            'tiempo_total_horas': round(tiempo_total, 1),
            'costo_total': round(costo_total, 2),
            'fleet_type': fleet_type,
            'carrier': carrier,
            'tiempo_prep': 1.0,
            'tiempo_viaje': round(tiempo_total - 1.2, 1),
            'peso_total_kg': 0.5 * cantidad,
            'zona_seguridad': 'Amarilla'
        }

    def _get_detailed_selection_reason(self, candidate: Dict[str, Any], all_candidates: List[Dict[str, Any]]) -> str:
        """📝 Genera razón DETALLADA de selección"""

        scores = candidate['scores']

        reasons = []

        # Razón principal
        if candidate == all_candidates[0]:
            reasons.append("Mejor score tiempo-costo")

        # Razones específicas por performance
        if scores['tiempo'] >= 0.8:
            reasons.append(f"Tiempo excelente ({candidate['tiempo_total_horas']}h)")
        elif scores['tiempo'] >= 0.6:
            reasons.append("Tiempo competitivo")

        if scores['costo'] >= 0.8:
            reasons.append(f"Costo eficiente (${candidate['costo_total']:.0f})")
        elif scores['costo'] >= 0.6:
            reasons.append("Costo aceptable")

        if scores['stock'] == 1.0:
            reasons.append("Stock completo")

        if candidate['fleet_type'] == 'FI':
            reasons.append("Flota interna")

        return ", ".join(reasons) if reasons else "Mejor opción disponible"


# REEMPLAZAR SOLO LA CLASE OptimizedExternalFactorsRepository en tu repositories.py

class OptimizedExternalFactorsRepository:
    """🌤️ Repositorio de factores externos optimizado - MEJORADO EVENTOS"""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def get_factors_for_date_and_cp(self, fecha: datetime, codigo_postal: str) -> Dict[str, Any]:
        """🎯 Obtiene factores externos para fecha y CP específicos - LÓGICA MEJORADA"""

        fecha_str = fecha.date().isoformat()
        df = self.data_manager.get_data('factores_externos')

        # 1. Buscar fecha exacta primero
        exact_match = df.filter(pl.col('fecha') == fecha_str)

        if exact_match.height > 0:
            result = exact_match.to_dicts()[0]
            logger.info(f"📅 Factores encontrados para fecha exacta: {fecha_str}")
            return self._process_factors(result, fecha, codigo_postal)

        # 2. Buscar fechas cercanas con LÓGICA INTELIGENTE
        return self._get_intelligent_factors(fecha, codigo_postal, df)

    def _get_intelligent_factors(self, fecha: datetime, codigo_postal: str, df: pl.DataFrame) -> Dict[str, Any]:
        """🧠 Obtiene factores con lógica inteligente para eventos"""

        fecha_target = fecha.date()

        # Convertir todas las fechas del CSV para comparación
        eventos_disponibles = []
        for row in df.to_dicts():
            try:
                fecha_csv = datetime.strptime(row['fecha'], '%Y-%m-%d').date()
                dias_diferencia = (fecha_target - fecha_csv).days

                eventos_disponibles.append({
                    'fecha_csv': fecha_csv,
                    'dias_diferencia': dias_diferencia,
                    'data': row
                })
            except:
                continue

        if not eventos_disponibles:
            logger.warning(f"⚠️ No se encontraron eventos válidos, usando factores estacionales")
            return self._calculate_seasonal_factors(fecha, codigo_postal)

        # Ordenar por cercanía
        eventos_disponibles.sort(key=lambda x: abs(x['dias_diferencia']))

        # NUEVA LÓGICA: Solo usar eventos si son válidos temporalmente
        for evento in eventos_disponibles:
            dias_diff = evento['dias_diferencia']
            evento_data = evento['data']
            evento_nombre = evento_data.get('evento_detectado', 'Normal')

            # ✅ REGLAS INTELIGENTES POR TIPO DE EVENTO
            if self._is_event_still_valid(evento_nombre, dias_diff, fecha_target):
                fecha_csv = evento['fecha_csv']
                logger.info(f"📅 Usando evento válido: {evento_nombre} de {fecha_csv} (diferencia: {dias_diff} días)")
                return self._process_factors(evento_data, fecha, codigo_postal)

        # Si no encuentra eventos válidos, usar factores estacionales
        logger.info(f"📅 No hay eventos válidos cerca de {fecha_target}, calculando factores estacionales")
        return self._calculate_seasonal_factors(fecha, codigo_postal)

    def _is_event_still_valid(self, evento_nombre: str, dias_diferencia: int, fecha_target: date) -> bool:
        """✅ Determina si un evento sigue siendo válido según su tipo y tiempo transcurrido"""

        # Eventos FUTUROS - solo válidos si son eventos de preparación/anticipación
        if dias_diferencia < 0:
            # Eventos que requieren preparación anticipada (máximo 3 días antes)
            eventos_anticipacion = [
                'Buen_Fin', 'Black_Friday', 'Cyber_Monday', 'Hot_Sale',
                'Navidad', 'Nochebuena', 'Semana_Santa'
            ]

            for evento_prep in eventos_anticipacion:
                if evento_prep in evento_nombre and abs(dias_diferencia) <= 3:
                    return True

            # Eventos estacionales NO deben aplicarse antes de tiempo
            eventos_estacionales = [
                'Inicio_Verano', 'Inicio_Invierno', 'Inicio_Primavera', 'Inicio_Otoño'
            ]

            for evento_estacional in eventos_estacionales:
                if evento_estacional in evento_nombre:
                    return False  # No aplicar eventos estacionales futuros

            return False  # Por defecto, no usar eventos futuros

        # Para eventos PASADOS, aplicar reglas MÁS ESTRICTAS
        if dias_diferencia > 0:

            # Eventos de un solo día - válidos solo el MISMO DÍA
            eventos_un_dia = [
                'Dia_Padre', 'Dia_Madres', 'San_Valentin', 'Dia_Mujer_8M',
                'Dia_Trabajo', 'Independencia_Mexico', 'Dia_Muertos', 'Halloween',
                'Navidad', 'Nochebuena', 'Año_Nuevo', 'Dia_Reyes'
            ]

            for evento_corto in eventos_un_dia:
                if evento_corto in evento_nombre and dias_diferencia == 0:
                    return True

            # Eventos de temporada - válidos más tiempo
            eventos_temporada = [
                'Hot_Sale', 'Buen_Fin', 'Black_Friday', 'Cyber_Monday',
                'Semana_Santa', 'Posadas', 'Temporada_Lluvias'
            ]

            for evento_largo in eventos_temporada:
                if evento_largo in evento_nombre and dias_diferencia <= 5:
                    return True

            # Eventos climáticos/estacionales - válidos hasta 10 días
            eventos_climaticos = [
                'Inicio_Primavera', 'Inicio_Verano', 'Inicio_Otoño', 'Inicio_Invierno',
                'Temporada_Huracanes', 'Fin_Temporada_Huracanes'
            ]

            for evento_clima in eventos_climaticos:
                if evento_clima in evento_nombre and dias_diferencia <= 10:
                    return True

        return False

    def _calculate_seasonal_factors(self, fecha: datetime, codigo_postal: str) -> Dict[str, Any]:
        """🌤️ Calcula factores basados en época del año cuando no hay eventos específicos"""

        mes = fecha.month
        dia = fecha.day

        # Determinar época y condiciones base
        if mes in [12, 1, 2]:  # Invierno
            epoca = "invierno"
            clima_base = "Invierno_Frio"
            factor_demanda_base = 1.0
            trafico_base = "Moderado"
            criticidad_base = "Normal"

        elif mes in [3, 4, 5]:  # Primavera
            epoca = "primavera"
            clima_base = "Primavera_Templado"
            factor_demanda_base = 1.1
            trafico_base = "Moderado"
            criticidad_base = "Normal"

        elif mes in [6, 7, 8]:  # Verano
            epoca = "verano"
            clima_base = "Verano_Lluvioso" if mes >= 6 else "Verano_Caluroso"
            factor_demanda_base = 1.2
            trafico_base = "Alto" if mes == 7 else "Moderado"  # Julio más tráfico por vacaciones
            criticidad_base = "Media"

        else:  # Otoño (9, 10, 11)
            epoca = "otoño"
            clima_base = "Otoño_Templado"
            factor_demanda_base = 1.0
            trafico_base = "Moderado"
            criticidad_base = "Normal"

        # Detectar efectos residuales de eventos conocidos
        evento_calculado = "Normal"

        # Efectos post-eventos específicos MÁS REALISTAS
        if mes == 6:
            if dia == 16:  # Solo 1 día después del Día del Padre
                evento_calculado = "Post_Dia_Padre"
                factor_demanda_base = 1.05  # Efecto mínimo
                criticidad_base = "Baja"
            elif 17 <= dia <= 19:  # 2-4 días después - casi normal
                evento_calculado = "Normal"
                factor_demanda_base = 1.0

        elif mes == 5:
            if dia == 11:  # Solo 1 día después del Día de las Madres
                evento_calculado = "Post_Dia_Madres"
                factor_demanda_base = 1.1
                criticidad_base = "Baja"
            elif 12 <= dia <= 15:  # Ya casi normal
                evento_calculado = "Normal"
                factor_demanda_base = 1.0

        elif mes == 2:
            if dia == 15:  # Solo el día después de San Valentín
                evento_calculado = "Post_San_Valentin"
                factor_demanda_base = 1.02
            elif dia >= 16:  # Ya normal
                evento_calculado = "Normal"
                factor_demanda_base = 1.0

        elif mes == 12:
            if 26 <= dia <= 31:  # Post Navidad
                evento_calculado = "Post_Navidad"
                factor_demanda_base = 0.8  # Baja demanda post navidad
                trafico_base = "Bajo"
                criticidad_base = "Baja"

        elif mes == 1:
            if 2 <= dia <= 10:  # Post Reyes
                evento_calculado = "Post_Reyes"
                factor_demanda_base = 0.9

        elif mes == 11:
            if 18 <= dia <= 25:  # Post Buen Fin
                evento_calculado = "Post_Buen_Fin"
                factor_demanda_base = 1.2  # Aún hay demanda residual

        # Ajustes por época específica sin eventos
        elif mes == 7 and 15 <= dia <= 31:  # Mitad de verano
            evento_calculado = "Temporada_Vacaciones_Verano"
            factor_demanda_base = 1.3
            trafico_base = "Alto"

        elif mes == 8 and 1 <= dia <= 31:  # Regreso a clases
            evento_calculado = "Regreso_Clases"
            factor_demanda_base = 1.4
            trafico_base = "Muy_Alto"
            criticidad_base = "Alta"

        logger.info(f"🌤️ Calculando factores estacionales para {epoca}: {evento_calculado}")

        # Obtener info del CP para zona de seguridad
        cp_info = self._get_cp_info_for_factors(codigo_postal)

        # Calcular impactos
        impacto_tiempo = self._calculate_time_impact_from_demand(factor_demanda_base, trafico_base)
        impacto_costo = self._calculate_cost_impact_from_demand(factor_demanda_base, evento_calculado)

        return {
            'evento_detectado': evento_calculado,
            'factor_demanda': factor_demanda_base,
            'condicion_clima': clima_base,
            'trafico_nivel': trafico_base,
            'criticidad_logistica': criticidad_base,
            'impacto_tiempo_extra_horas': impacto_tiempo,
            'impacto_costo_extra_pct': impacto_costo,
            'rango_cp_afectado': "00000-99999",
            'zona_seguridad': cp_info.get('zona_seguridad', 'Verde'),
            'es_temporada_alta': factor_demanda_base > 1.5,
            'es_temporada_critica': factor_demanda_base > 2.5,
            'fuente_datos': f'calculado_estacional_{epoca}',
            'observaciones_clima_regional': f'Factores calculados para {epoca} - {evento_calculado}'
        }

    def _calculate_time_impact_from_demand(self, factor_demanda: float, trafico_nivel: str) -> float:
        """⏱️ Calcula impacto en tiempo desde factor de demanda"""
        base_impact = max(0, (factor_demanda - 1.0) * 2.0)

        # Ajuste por tráfico
        trafico_multiplier = {
            'Bajo': 0.5,
            'Moderado': 1.0,
            'Alto': 1.5,
            'Muy_Alto': 2.0
        }.get(trafico_nivel, 1.0)

        final_impact = base_impact * trafico_multiplier
        return round(min(8.0, final_impact), 1)

    def _calculate_cost_impact_from_demand(self, factor_demanda: float, evento: str) -> float:
        """💰 Calcula impacto en costo desde factor de demanda"""
        base_impact = max(0, (factor_demanda - 1.0) * 15)

        # Eventos específicos que incrementan costos
        eventos_premium = ['Post_Navidad', 'Post_Buen_Fin', 'Regreso_Clases']
        if any(premium in evento for premium in eventos_premium):
            base_impact += 10.0

        return round(min(50.0, base_impact), 1)

    def _get_cp_info_for_factors(self, codigo_postal: str) -> Dict[str, Any]:
        """📍 Obtiene información del CP para factores externos"""
        try:
            cp_df = self.data_manager.get_data('codigos_postales')
            cp_int = int(codigo_postal)

            for row in cp_df.to_dicts():
                rango_cp = row.get('rango_cp', '')
                if '-' in rango_cp:
                    try:
                        inicio, fin = map(int, rango_cp.split('-'))
                        if inicio <= cp_int <= fin:
                            return row
                    except ValueError:
                        continue

            # Fallback
            return {'zona_seguridad': 'Verde', 'cobertura_liverpool': True}
        except:
            return {'zona_seguridad': 'Verde', 'cobertura_liverpool': True}

    def _process_factors(self, raw_data: Dict[str, Any], fecha: datetime, codigo_postal: str) -> Dict[str, Any]:
        """🔄 Procesa datos crudos del CSV a formato estándar - CORREGIDO"""

        # ✅ CORRECCIÓN: Limpiar factor_demanda que puede venir como string "1/08/2025"
        factor_demanda_raw = raw_data.get('factor_demanda', 1.0)

        if isinstance(factor_demanda_raw, str):
            if '/' in factor_demanda_raw:
                # Es una fecha mal formateada como "1/08/2025", extraer el factor real
                # Basándome en el patrón: "1/08/2025" debería ser 1.8
                try:
                    partes = factor_demanda_raw.split('/')
                    if len(partes) >= 2:
                        # Tomar primer número como entero y segundo como decimal
                        entero = int(partes[0])
                        decimal = int(partes[1][:2])  # Tomar solo primeros 2 dígitos del mes
                        factor_demanda = float(f"{entero}.{decimal:02d}")
                    else:
                        factor_demanda = 1.0
                except Exception as e:
                    logger.warning(f"⚠️ Error procesando factor_demanda: {factor_demanda_raw}, usando 1.0")
                    factor_demanda = 1.0
            else:
                # String normal, convertir a float
                try:
                    factor_demanda = float(factor_demanda_raw)
                except:
                    factor_demanda = 1.0
        else:
            # Ya es número
            try:
                factor_demanda = float(factor_demanda_raw)
            except:
                factor_demanda = 1.0

        # Obtener info del CP para zona de seguridad
        cp_info = self._get_cp_info_for_factors(codigo_postal)

        # Calcular impactos usando métodos existentes (mantener compatibilidad)
        impacto_tiempo = self._calculate_time_impact(factor_demanda, raw_data)
        impacto_costo = self._calculate_cost_impact(factor_demanda, raw_data)

        result = {
            'evento_detectado': raw_data.get('evento_detectado', 'Normal'),
            'factor_demanda': factor_demanda,
            'condicion_clima': raw_data.get('condicion_clima', 'Templado'),
            'trafico_nivel': raw_data.get('trafico_nivel', 'Moderado'),
            'criticidad_logistica': raw_data.get('criticidad_logistica', 'Normal'),
            'impacto_tiempo_extra_horas': impacto_tiempo,
            'impacto_costo_extra_pct': impacto_costo,
            'rango_cp_afectado': raw_data.get('rango_cp_afectado', '00000-99999'),
            'zona_seguridad': cp_info.get('zona_seguridad', 'Verde'),
            'es_temporada_alta': factor_demanda > 1.8,
            'es_temporada_critica': factor_demanda > 2.5,
            'fuente_datos': 'CSV_directo',
            'observaciones_clima_regional': raw_data.get('observaciones_clima_regional', '')
        }

        logger.info(
            f"🌤️ Factores procesados: demanda={factor_demanda:.2f}, tiempo_extra={impacto_tiempo}h, costo_extra={impacto_costo}%")
        return result

    # Mantener métodos existentes para compatibilidad
    def _calculate_time_impact(self, factor_demanda: float, factors: Dict[str, Any]) -> float:
        """⏱️ Calcula impacto en tiempo (método original)"""
        base_impact = max(0, (factor_demanda - 1.0) * 2.0)

        # Eventos críticos
        evento = factors.get('evento_detectado', '')
        if any(critical in evento for critical in ['Navidad', 'Nochebuena', 'Buen_Fin']):
            base_impact += 3.0

        return round(min(8.0, base_impact), 1)

    def _calculate_cost_impact(self, factor_demanda: float, factors: Dict[str, Any]) -> float:
        """💰 Calcula impacto en costo (método original)"""
        base_impact = max(0, (factor_demanda - 1.0) * 15)

        # Eventos premium
        evento = factors.get('evento_detectado', '')
        if any(premium in evento for premium in ['Navidad', 'Nochebuena', 'Buen_Fin']):
            base_impact += 20.0

        return round(min(50.0, base_impact), 1)

    def _generate_factors(self, fecha: datetime, cp: str) -> Dict[str, Any]:
        """🤖 Genera factores automáticos (mantener para compatibilidad)"""
        return self._calculate_seasonal_factors(fecha, cp)


class OptimizedFleetRepository:
    """🚚 Repositorio de flota externa optimizado"""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def get_best_carriers_for_cp(self, codigo_postal: str, peso_kg: float) -> List[Dict[str, Any]]:
        """🚛 Obtiene mejores carriers para CP y peso"""
        df = self.data_manager.get_data('flota_externa')
        cp_int = int(codigo_postal)

        # Filtrar carriers disponibles
        available = df.filter(
            (pl.col('activo') == True) &
            (pl.col('zona_cp_inicio') <= cp_int) &
            (pl.col('zona_cp_fin') >= cp_int) &
            (pl.col('peso_min_kg') <= peso_kg) &
            (pl.col('peso_max_kg') >= peso_kg)
        ).sort('costo_base_mxn')

        carriers = available.to_dicts()

        logger.info(f"🚚 Carriers disponibles para {codigo_postal} ({peso_kg}kg): {len(carriers)}")
        for carrier in carriers:
            logger.info(
                f"  📦 {carrier['carrier']}: ${carrier['costo_base_mxn']} base ({carrier['tiempo_entrega_dias_habiles']} días)")

        return carriers


# Repositorio unificado
class OptimizedRepositories:
    """🎯 Repositorios optimizados unificados"""

    def __init__(self, data_dir: Path):
        self.data_manager = DataManager(data_dir)
        self.product = OptimizedProductRepository(self.data_manager)
        self.store = OptimizedStoreRepository(self.data_manager)
        self.stock = OptimizedStockRepository(self.data_manager)
        self.external_factors = OptimizedExternalFactorsRepository(self.data_manager)
        self.fleet = OptimizedFleetRepository(self.data_manager)

        logger.info("🚀 Repositorios optimizados inicializados")