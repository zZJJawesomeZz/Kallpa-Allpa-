import os
from datetime import datetime
from fastapi import APIRouter, Query, Response, HTTPException
from typing import Optional

from app.services.nasa_client import get_nasa_power_data
from app.services.gamc_client import get_gamc_resource
from app.services.soilgrids_client import get_soilgrids_data
from app.utils.cleaners import (
    clean_nasa_weather_data,
    clean_nasa_to_dataframe,
    generar_resumen_estadistico,
    clean_gamc_data
)
from app.ml.predictor import (
    predecir_cultivos_optimos,
    recomendar_cultivos_satelitales,
    proyectar_cultivos_trimestral
)
from app.ml.train_model import entrenar_modelo_random_forest

# Creación del router principal de la API
api_router = APIRouter(prefix="/api/v1", tags=["Kallpa Shift API"])

@api_router.get("/health")
def health_check():
    """
    Ruta básica para comprobar que el servicio backend se encuentra activo.
    """
    return {
        "status": "online",
        "message": "Servicio Kallpa Shift Backend funcionando correctamente"
    }

@api_router.get("/clima")
def obtener_datos_climaticos(
    lat: float = Query(..., description="Latitud geográfica (ejemplo: -17.3895)"),
    lon: float = Query(..., description="Longitud geográfica (ejemplo: -66.1568)"),
    start_date: str = Query("20240101", description="Fecha de inicio en formato YYYYMMDD"),
    end_date: str = Query("20240105", description="Fecha de fin en formato YYYYMMDD"),
    parameters: str = Query("T2M,PRECTOTCORR,RH2M", description="Variables meteorológicas separadas por comas"),
    limpiar_datos: bool = Query(True, description="Indica si se deben limpiar los datos con cleaners.py")
):
    """
    Endpoint para conectar las coordenadas seleccionadas en el mapa con la NASA POWER API.
    Retorna los datos climáticos crudos o procesados y limpios.
    """
    datos_crudos = get_nasa_power_data(
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
        parameters=parameters
    )
    
    # Si hubo error al consultar la NASA POWER API, devolvemos el error directo
    if isinstance(datos_crudos, dict) and datos_crudos.get("status") == "error":
        return datos_crudos

    # Limpieza modular con las funciones preparadas para el equipo
    if limpiar_datos:
        datos_limpios = clean_nasa_weather_data(datos_crudos)
        return {
            "latitud": lat,
            "longitud": lon,
            "registros": datos_limpios
        }

    return datos_crudos

@api_router.get("/dataset/vista-previa")
def obtener_vista_previa_dataset(
    lat: float = Query(-17.3895, description="Latitud (ej. Cochabamba)"),
    lon: float = Query(-66.1568, description="Longitud"),
    start_date: str = Query("20230101", description="Fecha inicio YYYYMMDD"),
    end_date: str = Query("20231231", description="Fecha fin YYYYMMDD"),
    parameters: str = Query("T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN", description="Parámetros NASA"),
    imputar_nulos: bool = Query(True, description="Interpolar y rellenar nulos para entrenamiento ML")
):
    """
    Endpoint para la interfaz gráfica:
    Extrae los datos de NASA POWER, los procesa y devuelve:
    1. Resumen estadístico (promedios, mínimos, máximos, valores nulos).
    2. Las primeras filas tabulares para visualización.
    """
    datos_crudos = get_nasa_power_data(
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
        parameters=parameters
    )

    if isinstance(datos_crudos, dict) and datos_crudos.get("status") == "error":
        raise HTTPException(status_code=502, detail=datos_crudos["message"])

    # Limpieza y preparación para Machine Learning
    df = clean_nasa_to_dataframe(datos_crudos, imputar_nulos=imputar_nulos)
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No se obtuvieron registros para las coordenadas o fechas indicadas.")

    # Formatear la columna fecha para visualización JSON
    df_vista = df.copy()
    df_vista["fecha"] = df_vista["fecha"].dt.strftime("%Y-%m-%d")

    resumen = generar_resumen_estadistico(df)
    
    # Tomar las primeras 50 filas para la vista previa
    preview_filas = df_vista.head(50).to_dict(orient="records")

    return {
        "coordenadas": {"latitud": lat, "longitud": lon},
        "periodo": {"inicio": start_date, "fin": end_date},
        "resumen": resumen,
        "preview": preview_filas
    }

@api_router.get("/dataset/descargar-csv")
def descargar_dataset_csv(
    lat: float = Query(-17.3895, description="Latitud"),
    lon: float = Query(-66.1568, description="Longitud"),
    start_date: str = Query("20230101", description="Fecha inicio YYYYMMDD"),
    end_date: str = Query("20231231", description="Fecha fin YYYYMMDD"),
    parameters: str = Query("T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN", description="Parámetros NASA"),
    imputar_nulos: bool = Query(True, description="Interpolar y rellenar nulos para ML")
):
    """
    Genera y descarga directamente un archivo CSV limpio y listo para entrenar modelos de Machine Learning.
    """
    datos_crudos = get_nasa_power_data(
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
        parameters=parameters
    )

    if isinstance(datos_crudos, dict) and datos_crudos.get("status") == "error":
        raise HTTPException(status_code=502, detail=datos_crudos["message"])

    df = clean_nasa_to_dataframe(datos_crudos, imputar_nulos=imputar_nulos)
    if df.empty:
        raise HTTPException(status_code=404, detail="No se obtuvieron datos para generar el CSV.")

    # Convertir a CSV en memoria
    df_csv = df.copy()
    df_csv["fecha"] = df_csv["fecha"].dt.strftime("%Y-%m-%d")
    contenido_csv = df_csv.to_csv(index=False)

    nombre_archivo = f"dataset_agroclima_{lat}_{lon}_{start_date}_{end_date}.csv"

    return Response(
        content=contenido_csv,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={nombre_archivo}"}
    )

@api_router.post("/dataset/guardar-local")
def guardar_dataset_local(
    lat: float = Query(-17.3895),
    lon: float = Query(-66.1568),
    start_date: str = Query("20230101"),
    end_date: str = Query("20231231"),
    parameters: str = Query("T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN"),
    nombre_personalizado: Optional[str] = Query(None, description="Nombre opcional del archivo")
):
    """
    Guarda el dataset procesado directamente en el disco duro local (en la carpeta 'datasets/'),
    para que los scripts de entrenamiento (Jupyter, PyTorch, Scikit-learn) puedan leerlo directamente.
    """
    datos_crudos = get_nasa_power_data(
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
        parameters=parameters
    )

    if isinstance(datos_crudos, dict) and datos_crudos.get("status") == "error":
        raise HTTPException(status_code=502, detail=datos_crudos["message"])

    df = clean_nasa_to_dataframe(datos_crudos, imputar_nulos=True)
    if df.empty:
        raise HTTPException(status_code=404, detail="No hay datos disponibles para guardar.")

    # Asegurar que exista la carpeta datasets en la raíz del proyecto
    carpeta_destino = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))
    os.makedirs(carpeta_destino, exist_ok=True)

    if not nombre_personalizado:
        nombre_personalizado = f"agroclima_{lat}_{lon}_{start_date}_{end_date}.csv"
    elif not nombre_personalizado.endswith(".csv"):
        nombre_personalizado += ".csv"

    ruta_archivo = os.path.join(carpeta_destino, nombre_personalizado)
    df_csv = df.copy()
    df_csv["fecha"] = df_csv["fecha"].dt.strftime("%Y-%m-%d")
    df_csv.to_csv(ruta_archivo, index=False)

    return {
        "status": "guardado",
        "mensaje": f"Dataset guardado exitosamente en el servidor local.",
        "ruta_absoluta": ruta_archivo,
        "total_filas": len(df_csv),
        "columnas": df_csv.columns.tolist()
    }

@api_router.get("/gamc/recurso")
def obtener_recurso_gamc(
    endpoint: str = Query("datos-abiertos", description="Ruta relativa del recurso en recursosgamc.cochabamba.bo"),
    limpiar_datos: bool = Query(True, description="Indica si se deben filtrar los datos")
):
    """
    Endpoint para consultar datos locales de la Alcaldía de Cochabamba (GAMC).
    """
    datos_crudos = get_gamc_resource(endpoint=endpoint)
    
    if limpiar_datos:
        return clean_gamc_data(datos_crudos)
        
    return datos_crudos

@api_router.get("/suelo")
def obtener_propiedades_suelo(
    lat: float = Query(-17.3895, description="Latitud geográfica"),
    lon: float = Query(-66.1568, description="Longitud geográfica")
):
    """
    Endpoint para consultar las propiedades físicas y químicas del suelo (SoilGrids ISRIC v2):
    pH en agua, porcentaje de arcilla, arena y clasificación textural del terreno.
    """
    return get_soilgrids_data(lat=lat, lon=lon)

# ==============================================================================
# ENDPOINTS DE MACHINE LEARNING: RECOMENDACIÓN DE CULTIVOS CON RANDOM FOREST
# ==============================================================================

@api_router.get("/ml/recomendar-cultivos")
def endpoint_recomendar_cultivos(
    mes: int = Query(..., ge=1, le=12, description="Mes del año (1=Enero, ..., 12=Diciembre)"),
    temperatura_media: float = Query(16.5, description="Temperatura promedio del mes (°C)"),
    temperatura_minima: float = Query(8.0, description="Temperatura mínima esperada (°C)"),
    temperatura_maxima: float = Query(23.5, description="Temperatura máxima esperada (°C)"),
    precipitacion_mensual: float = Query(60.0, description="Precipitación acumulada del mes (mm)"),
    humedad_relativa: float = Query(65.0, description="Humedad relativa promedio (%)"),
    radiacion_solar: float = Query(18.5, description="Radiación solar diaria incidente (MJ/m²/día)"),
    ph_suelo: float = Query(6.6, ge=3.5, le=10.0, description="pH del suelo (ejemplo: 6.5 para suelo neutro)")
):
    """
    Predice y recomienda qué cultivos sembrar en determinado mes o época del año
    según las condiciones climáticas y edafológicas (suelo), utilizando Random Forest.
    """
    try:
        resultado = predecir_cultivos_optimos(
            mes=mes,
            temperatura_media=temperatura_media,
            temperatura_minima=temperatura_minima,
            temperatura_maxima=temperatura_maxima,
            precipitacion_mensual=precipitacion_mensual,
            humedad_relativa=humedad_relativa,
            radiacion_solar=radiacion_solar,
            ph_suelo=ph_suelo,
            top_k=3
        )
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción del modelo: {str(e)}")

@api_router.get("/ml/recomendar-por-satelite")
def endpoint_recomendar_satelite(
    lat: float = Query(-17.3895, description="Latitud geográfica (ej. Cochabamba)"),
    lon: float = Query(-66.1568, description="Longitud geográfica"),
    mes: Optional[int] = Query(None, ge=1, le=12, description="Mes del año (por defecto el mes actual)"),
    anio: Optional[int] = Query(None, ge=1990, le=2050, description="Año del análisis (por defecto el año actual)"),
    altitud_msnm: Optional[float] = Query(None, description="Altitud en msnm (opcional, calculada según relieve si se omite)")
):
    """
    Evaluación Agroclimática Integral:
    1. Consulta observaciones históricas o la Climatología regional multi-anual de NASA POWER para el período actual o solicitado.
    2. Consulta las propiedades de suelo en SoilGrids (ISRIC).
    3. Incorpora la altitud (msnm) y retorna los cultivos recomendados con porcentajes de idoneidad.
    """
    try:
        mes_evaluar = mes if mes is not None else datetime.now().month
        anio_evaluar = anio if anio is not None else datetime.now().year
        resultado = recomendar_cultivos_satelitales(
            lat=lat,
            lon=lon,
            mes=mes_evaluar,
            anio=anio_evaluar,
            altitud_msnm=altitud_msnm
        )
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error consultando satélite o prediciendo: {str(e)}")

@api_router.get("/ml/proyeccion-trimestral")
def endpoint_proyeccion_trimestral(
    lat: float = Query(-17.3895, description="Latitud geográfica"),
    lon: float = Query(-66.1568, description="Longitud geográfica"),
    mes_inicio: Optional[int] = Query(None, ge=1, le=12, description="Mes inicial de la proyección (por defecto el mes actual)"),
    anio_inicio: Optional[int] = Query(None, ge=1990, le=2050, description="Año de inicio de la proyección (por defecto el año actual)"),
    altitud_msnm: Optional[float] = Query(None, description="Altitud en msnm (opcional, calculada según relieve si se omite)")
):
    """
    Proyección Anticipada a 3 Meses en el Futuro:
    Calcula la evolución agroclimática esperada para los próximos 3 meses a partir de la fecha actual
    mediante climatología satelital de la NASA y evalúa la viabilidad agronómica trimestral considerando altitud.
    """
    try:
        mes_evaluar = mes_inicio if mes_inicio is not None else datetime.now().month
        anio_evaluar = anio_inicio if anio_inicio is not None else datetime.now().year
        resultado = proyectar_cultivos_trimestral(
            lat=lat,
            lon=lon,
            mes_inicio=mes_evaluar,
            anio_inicio=anio_evaluar,
            altitud_msnm=altitud_msnm
        )
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la proyección trimestral: {str(e)}")

@api_router.post("/ml/reentrenar")
def endpoint_reentrenar_modelo():
    """
    Reentrena el modelo Random Forest con nuevos datos agronómicos y actualiza el archivo en disco.
    """
    try:
        reporte = entrenar_modelo_random_forest()
        return reporte
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al reentrenar: {str(e)}")

