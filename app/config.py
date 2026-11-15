import os
from dotenv import load_dotenv

# Cargar variables de entorno desde un archivo .env si existe
load_dotenv()

class Settings:
    """
    Configuración general de la aplicación.
    Permite leer variables de entorno con valores por defecto amigables para desarrollo.
    """
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Kallpa Shift Backend")
    PROJECT_VERSION: str = os.getenv("PROJECT_VERSION", "1.0.0")
    
    # URL base del servicio NASA POWER API (Diario y Climatología proyectada)
    # Documentación: https://power.larc.nasa.gov/docs/services/api/
    NASA_POWER_BASE_URL: str = os.getenv(
        "NASA_POWER_BASE_URL", 
        "https://power.larc.nasa.gov/api/temporal/daily/point"
    )
    NASA_POWER_CLIMATOLOGY_URL: str = os.getenv(
        "NASA_POWER_CLIMATOLOGY_URL",
        "https://power.larc.nasa.gov/api/temporal/climatology/point"
    )
    
    # URL base de los servicios de la Alcaldía de Cochabamba (GAMC)
    GAMC_BASE_URL: str = os.getenv(
        "GAMC_BASE_URL", 
        "https://recursosgamc.cochabamba.bo"
    )
    
    # Configuración de red y timeouts (en segundos)
    DEFAULT_TIMEOUT: int = int(os.getenv("DEFAULT_TIMEOUT", "15"))

# Instancia única para reutilizar en toda la aplicación
settings = Settings()
