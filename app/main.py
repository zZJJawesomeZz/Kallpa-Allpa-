import os
import sys

# Asegurar que el directorio raíz del proyecto esté en el sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers.api import api_router

# Inicialización de la aplicación FastAPI con metadatos descriptivos
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="Backend y plataforma de datos para agricultura de precisión (NASA Space Apps Challenge)."
)

# Configuración de CORS (Cross-Origin Resource Sharing)
# Esto permite que aplicaciones web frontend puedan comunicarse con el servidor sin bloqueos del navegador.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Permite todos los orígenes en desarrollo
    allow_credentials=True,
    allow_methods=["*"],       # Permite todos los métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],       # Permite todas las cabeceras HTTP
)

# Montar carpeta de archivos estáticos (imágenes y recursos locales)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Inclusión de las rutas modulares de la API
app.include_router(api_router)

# Ruta absoluta al archivo HTML de la interfaz gráfica
HTML_PATH = os.path.join(os.path.dirname(__file__), "static", "index.html")

@app.get("/", response_class=HTMLResponse)
def interfaz_usuario():
    """
    Sirve la interfaz web visual para que el equipo pueda:
    1. Ingresar coordenadas, fechas y variables satelitales.
    2. Consultar y limpiar los datos de NASA POWER API.
    3. Descargar el dataset listo en CSV para entrenar modelos de Machine Learning.
    """
    if os.path.exists(HTML_PATH):
        with open(HTML_PATH, "r", encoding="utf-8") as archivo:
            return HTMLResponse(content=archivo.read())
    
    return HTMLResponse(
        content="<h2>Interfaz en construcción. Visite <a href='/docs'>/docs</a> para usar la API interactiva.</h2>",
        status_code=200
    )

@app.get("/api/info")
def info_api():
    """
    Ruta con información básica sobre los servicios activos y documentación.
    """
    return {
        "proyecto": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "documentacion_swagger": "/docs",
        "interfaz_web": "/"
    }

if __name__ == "__main__":
    import uvicorn
    # Permite iniciar el servidor directamente ejecutando: python app/main.py
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
