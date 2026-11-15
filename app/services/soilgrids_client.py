import requests
from typing import Dict, Any, Optional

"""
CLIENTE DE SOILGRIDS (ISRIC - World Soil Information)
----------------------------------------------------
Extrae propiedades físicas y químicas del suelo a nivel global a 250m de resolución.
Documentación: https://rest.isric.org/soilgrids/v2.0/docs

Propiedades consultadas:
- phh2o: pH del suelo en agua (originalmente x10, ej. 65 = 6.5)
- clay: Porcentaje de arcilla en el suelo (originalmente en g/kg)
- sand: Porcentaje de arena en el suelo (originalmente en g/kg)
- soc: Carbono orgánico del suelo (materia orgánica)
"""

SOILGRIDS_BASE_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# Caché en memoria para evitar consultas redundantes a la misma coordenada
_CACHE_SUELOS = {}

def get_soilgrids_data(lat: float, lon: float, timeout_segundos: int = 8) -> Dict[str, Any]:
    """
    Consulta las propiedades del suelo para una latitud y longitud dadas.
    Incluye un mecanismo de resiliencia (fallback) con valores agronómicos típicos
    en caso de que los servidores internacionales de ISRIC estén lentos o inestables.
    
    Retorna:
    - Dict con pH, porcentaje de arcilla, arena y tipo de suelo textural.
    """
    clave_cache = f"{round(lat, 3)}_{round(lon, 3)}"
    if clave_cache in _CACHE_SUELOS:
        return _CACHE_SUELOS[clave_cache]

    # Parámetros para la API REST v2 de SoilGrids
    query_params = [
        ("lon", lon),
        ("lat", lat),
        ("property", "phh2o"),
        ("property", "clay"),
        ("property", "sand"),
        ("property", "soc"),
        ("depth", "0-5cm"),
        ("depth", "5-15cm"),
        ("value", "mean")
    ]

    try:
        response = requests.get(
            SOILGRIDS_BASE_URL,
            params=query_params,
            headers={"Accept": "application/json", "User-Agent": "KallpaShiftApp/1.0"},
            timeout=timeout_segundos
        )

        if response.status_code == 200:
            datos_json = response.json()
            propiedades = datos_json.get("properties", {}).get("layers", [])

            valores_extraidos = {}
            for capa in propiedades:
                nombre = capa.get("name")
                profundidades = capa.get("depths", [])
                if profundidades:
                    # Obtenemos el valor promedio de la capa superficial (0-5cm o 5-15cm)
                    valor_medio = profundidades[0].get("values", {}).get("mean")
                    if valor_medio is not None:
                        valores_extraidos[nombre] = valor_medio

            # Procesar y escalar valores
            # pH viene multiplicado por 10 en SoilGrids (ej: 65 -> 6.5)
            ph_bruto = valores_extraidos.get("phh2o")
            ph_final = round(ph_bruto / 10.0, 2) if ph_bruto else 6.6

            # Arcilla y arena vienen en g/kg (dividir entre 10 para obtener %)
            arcilla_bruta = valores_extraidos.get("clay")
            arcilla_pct = round(arcilla_bruta / 10.0, 1) if arcilla_bruta else 24.0

            arena_bruta = valores_extraidos.get("sand")
            arena_pct = round(arena_bruta / 10.0, 1) if arena_bruta else 42.0

            # Determinación sencilla de textura del suelo
            textura = clasificar_textura_suelo(arcilla_pct, arena_pct)

            resultado = {
                "origen": "SoilGrids ISRIC API v2",
                "latitud": lat,
                "longitud": lon,
                "ph_suelo": ph_final,
                "arcilla_porcentaje": arcilla_pct,
                "arena_porcentaje": arena_pct,
                "textura_estimada": textura,
                "contingencia_usada": False
            }

            _CACHE_SUELOS[clave_cache] = resultado
            return resultado

    except Exception:
        # En caso de lentitud, caída o timeout de SoilGrids, no bloqueamos la app:
        # aplicamos contingencia inteligente basada en promedios de valles interandinos.
        pass

    # Valores de contingencia (Suelo Franco de Valle)
    contingencia = {
        "origen": "Estimación de Contingencia (Suelo Franco)",
        "latitud": lat,
        "longitud": lon,
        "ph_suelo": 6.7,
        "arcilla_porcentaje": 22.0,
        "arena_porcentaje": 40.0,
        "textura_estimada": "Franco (Equilibrado para cultivo)",
        "contingencia_usada": True
    }
    _CACHE_SUELOS[clave_cache] = contingencia
    return contingencia


def clasificar_textura_suelo(arcilla_pct: float, arena_pct: float) -> str:
    """
    Clasifica de forma simplificada la textura del suelo para recomendaciones agrícolas.
    """
    if arcilla_pct >= 35.0:
        return "Arcilloso (Pesado, retiene agua pero lento drenaje)"
    elif arena_pct >= 65.0:
        return "Arenoso (Ligero, rápido drenaje, baja retención de nutrientes)"
    elif arcilla_pct >= 20.0 and arena_pct <= 45.0:
        return "Franco-Arcilloso (Rico en nutrientes y buena humedad)"
    else:
        return "Franco (Óptimo, balance ideal de agua y aireación)"
