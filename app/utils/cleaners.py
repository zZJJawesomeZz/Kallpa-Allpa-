import pandas as pd
import numpy as np
from typing import Dict, Any, List

"""
MÓDULO DE LIMPIEZA Y PREPARACIÓN DE DATASETS (cleaners.py)
--------------------------------------------------------
Provee funciones modulares para:
1. Transformar respuestas crudas de APIs en DataFrames de pandas.
2. Limpiar valores centinelas de NASA (-999) y tratar valores nulos.
3. Crear características temporales y agronómicas para entrenar modelos de Machine Learning.
4. Generar resúmenes estadísticos para la interfaz web.
"""

def clean_nasa_to_dataframe(
    raw_data: Dict[str, Any],
    imputar_nulos: bool = True
) -> pd.DataFrame:
    """
    Convierte la respuesta cruda de NASA POWER en un DataFrame de pandas limpio
    y preparado con variables adicionales para entrenamiento de modelos de Machine Learning.
    
    Parámetros:
    - raw_data (dict): Respuesta cruda obtenida desde la NASA POWER API.
    - imputar_nulos (bool): Si es True, rellena valores faltantes mediante interpolación lineal.
    
    Retorna:
    - pd.DataFrame: DataFrame procesado con columnas de fecha, variables climáticas y temporales.
    """
    if not raw_data or "properties" not in raw_data or "parameter" not in raw_data["properties"]:
        return pd.DataFrame()

    parametros = raw_data["properties"]["parameter"]
    
    # Crear DataFrame inicial donde cada columna es una variable climática y el índice es la fecha YYYYMMDD
    df = pd.DataFrame(parametros)

    # Reemplazar valores centinela de NASA (-999, -999.0) por NaN de numpy
    df.replace([-999, -999.0], np.nan, inplace=True)

    # Convertir todas las columnas a tipo numérico (float)
    for columna in df.columns:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

    # Imputación opcional de nulos para algoritmos de Machine Learning
    if imputar_nulos:
        # Interpolación lineal hacia adelante y hacia atrás para series de tiempo
        df.interpolate(method="linear", limit_direction="both", inplace=True)
        # Si aún quedan valores nulos al principio o final, rellenar con la media
        df.fillna(df.mean(numeric_only=True), inplace=True)

    # Transformación del índice a formato fecha estándar (YYYY-MM-DD)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "fecha_codigo"}, inplace=True)
    
    # Parsear a datetime
    df["fecha"] = pd.to_datetime(df["fecha_codigo"], format="%Y%m%d", errors="coerce")
    
    # Crear características (features) útiles para el entrenamiento del modelo
    df["año"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month
    df["dia"] = df["fecha"].dt.day
    df["dia_del_año"] = df["fecha"].dt.dayofyear

    # Reordenar columnas para legibilidad
    columnas_ordenadas = ["fecha", "año", "mes", "dia", "dia_del_año"] + [
        c for c in df.columns if c not in ["fecha", "año", "mes", "dia", "dia_del_año", "fecha_codigo"]
    ]
    
    return df[columnas_ordenadas]


def clean_nasa_weather_data(raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convierte los datos limpios en una lista de diccionarios serializable a JSON.
    """
    df = clean_nasa_to_dataframe(raw_data, imputar_nulos=False)
    if df.empty:
        return []
    
    # Formatear la fecha como string para serialización JSON
    df_export = df.copy()
    df_export["fecha"] = df_export["fecha"].dt.strftime("%Y-%m-%d")
    return df_export.to_dict(orient="records")


def generar_resumen_estadistico(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Genera métricas descriptivas básicas de un DataFrame para mostrar en la interfaz web
    antes del entrenamiento del modelo (filas, columnas, nulos, promedios).
    """
    if df.empty:
        return {
            "total_filas": 0,
            "total_columnas": 0,
            "columnas": [],
            "resumen": {}
        }

    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Excluir columnas de calendario del resumen principal de variables
    columnas_a_analizar = [c for c in columnas_numericas if c not in ["año", "mes", "dia", "dia_del_año"]]
    if not columnas_a_analizar:
        columnas_a_analizar = columnas_numericas

    resumen_dict = {}
    for col in columnas_a_analizar:
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        resumen_dict[col] = {
            "minimo": round(float(min_val), 2) if pd.notna(min_val) else None,
            "maximo": round(float(max_val), 2) if pd.notna(max_val) else None,
            "promedio": round(float(mean_val), 2) if pd.notna(mean_val) else None,
            "desviacion": round(float(std_val), 2) if pd.notna(std_val) else None,
            "nulos": int(df[col].isna().sum())
        }

    return {
        "total_filas": int(len(df)),
        "total_columnas": int(len(df.columns)),
        "columnas": df.columns.tolist(),
        "resumen": resumen_dict
    }


def clean_gamc_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Plantilla base para procesar los datos de recursos de la Alcaldía de Cochabamba (GAMC).
    """
    if raw_data.get("status") == "error":
        return raw_data

    datos_procesados = {
        "origen": "GAMC Cochabamba",
        "total_registros": len(raw_data) if isinstance(raw_data, list) else 1,
        "datos": raw_data
    }
    
    return datos_procesados
