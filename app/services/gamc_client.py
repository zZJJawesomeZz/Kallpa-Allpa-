import requests
from typing import Dict, Any, Optional
from app.config import settings

def get_gamc_resource(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Función base para realizar peticiones GET a los endpoints públicos del GAMC (recursosgamc.cochabamba.bo).
    
    Parámetros:
    - endpoint (str): Ruta relativa del recurso a consultar (ej. '/api/v1/clima' o '/datos-abiertos').
    - params (dict, opcional): Parámetros de consulta (query parameters) en caso de requerirse.
    
    Retorna:
    - Dict[str, Any]: Diccionario con la respuesta en formato JSON o mensaje de error estructurado.
    """
    # Construcción de la URL completa limpiando diagonales dobles
    endpoint_limpio = endpoint.lstrip("/")
    url_completa = f"{settings.GAMC_BASE_URL}/{endpoint_limpio}"

    # Encabezados comunes para peticiones API
    headers = {
        "User-Agent": "KallpaShiftApp/1.0 (NASA Space Apps Challenge)",
        "Accept": "application/json"
    }

    try:
        response = requests.get(
            url_completa,
            params=params,
            headers=headers,
            timeout=settings.DEFAULT_TIMEOUT
        )
        
        # Validar si el servidor respondió con un código de error HTTP
        response.raise_for_status()
        
        return response.json()

    except requests.exceptions.RequestException as error:
        # Si la API aún no está disponible o falla, devolvemos un estado controlado
        return {
            "status": "error",
            "message": f"Error al consultar el servicio GAMC en '{url_completa}': {str(error)}"
        }
