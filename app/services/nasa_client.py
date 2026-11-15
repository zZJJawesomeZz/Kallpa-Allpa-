import requests
from typing import Dict, Any, Optional
from app.config import settings

def get_nasa_power_data(
    lat: float,
    lon: float,
    start_date: str = "20240101",
    end_date: str = "20240105",
    parameters: Optional[str] = "T2M,PRECTOTCORR,RH2M"
) -> Dict[str, Any]:
    """
    Realiza una petición GET a la API de NASA POWER para consultar datos agroclimáticos.
    
    Parámetros:
    - lat (float): Latitud del punto geográfico (ej. -17.3895 para Cochabamba).
    - lon (float): Longitud del punto geográfico (ej. -66.1568 para Cochabamba).
    - start_date (str): Fecha de inicio en formato YYYYMMDD (por defecto '20240101').
    - end_date (str): Fecha de fin en formato YYYYMMDD (por defecto '20240105').
    - parameters (str): Variables meteorológicas separadas por comas.
                        Ejemplos:
                        - T2M: Temperatura a 2 metros
                        - PRECTOTCORR: Precipitación corregida
                        - RH2M: Humedad relativa a 2 metros
    
    Retorna:
    - Dict[str, Any]: Respuesta en formato JSON devuelta por la NASA POWER API.
    """
    # Parámetros requeridos por el servicio de agroclimatología de NASA POWER
    query_params = {
        "parameters": parameters,
        "community": "AG",         # AG = Agroclimatology (especial para agricultura)
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }

    try:
        # Petición GET con tiempo de espera configurable
        response = requests.get(
            settings.NASA_POWER_BASE_URL,
            params=query_params,
            timeout=settings.DEFAULT_TIMEOUT
        )
        
        # Verifica si hubo un error HTTP (códigos 4xx o 5xx)
        response.raise_for_status()
        
        return response.json()

    except requests.exceptions.RequestException as error:
        # En caso de error de red o timeout, se retorna un diccionario descriptivo
        return {
            "status": "error",
            "message": f"Error al comunicarse con NASA POWER API: {str(error)}"
        }

def get_nasa_climatology_data(
    lat: float,
    lon: float,
    parameters: Optional[str] = "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN"
) -> Dict[str, Any]:
    """
    Consulta los promedios climatológicos multi-anuales (30+ años de satélites NASA MERRA-2 y CERES)
    para modelar y proyectar las condiciones meteorológicas esperadas a futuro para cada mes.
    """
    query_params = {
        "parameters": parameters,
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "format": "JSON"
    }

    try:
        response = requests.get(
            settings.NASA_POWER_CLIMATOLOGY_URL,
            params=query_params,
            timeout=settings.DEFAULT_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as error:
        return {
            "status": "error",
            "message": f"Error al consultar Climatología de NASA POWER: {str(error)}"
        }
