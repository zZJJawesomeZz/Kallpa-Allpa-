# Kallpa Allpa Cochabamba - Backend y Sistema de Recomendacion Agricola

Plataforma de agricultura de precision y recomendacion de cultivos para los Valles de Cochabamba (Bolivia), desarrollada para el NASA Space Apps Challenge e impulsada por modelos de Machine Learning (Random Forest) y datos satelitales.

El sistema procesa en tiempo real datos climaticos multi-anuales de NASA POWER, propiedades edafologicas y pH de SoilGrids (ISRIC), y pisos altitudinales de la Cordillera del Tunari y valles centrales para emitir recomendaciones optimas de siembra para agricultores.

---

## Caracteristicas Principales

1. **Interfaz Web Bilingue para Agricultores (Espanol / Quechua):**
   - Diseno directo y sin tecnicismos complejos para el productor rural.
   - Paso 1: Seleccion de la zona de cultivo (La Maica, Distrito 9 Sur, Distrito 13 Tunari o Valles Periurbanos) con mapas territoriales.
   - Paso 2: Eleccion de la epoca de siembra (mes calendario en quechua/castellano) y ventana de planificacion (mensual o trimestral).
   - Paso 3: Visualizacion de resultados con fotografias reales en primer plano de cada cultivo, medallas de compatibilidad porcentual, tiempos de cosecha en dias y consejos agronomicos en quechua y espanol.

2. **Modelo de Machine Learning con Altitud Integrada (Random Forest):**
   - Clasificador entrenado con 3.200 muestras agronomicas para los valles de Cochabamba.
   - Variables evaluadas (9 caracteristicas):
     - Mes del ano (1 a 12)
     - Temperatura media (grados Celsius)
     - Temperatura minima diaria (grados Celsius)
     - Temperatura maxima diaria (grados Celsius)
     - Precipitacion acumulada mensual (milimetros)
     - Humedad relativa promedio (%)
     - Radiacion solar incidente (MJ/m2/dia)
     - pH del suelo en agua (escala SoilGrids)
     - Altitud en metros sobre el nivel del mar (msnm)
   - Filtro agronomico de cota: Evita recomendar cultivos sensibles a heladas en laderas altas y prioriza variedades adaptadas (ej. Papa Waych'a y haba en laderas del Tunari a 2.850 msnm; tomate y maiz choclero en fondo de valle a 2.550 msnm).

3. **Conector Satelital NASA POWER API:**
   - Consulta automatizada de series diarias y Climatologia multi-anual basada en mas de 30 anos de observaciones satelitales continuas (misiones MERRA-2 y CERES).
   - Deteccion dinamica del periodo en curso para proveer siempre datos actualizados.

4. **Conector de Suelos SoilGrids ISRIC v2:**
   - Estimacion de propiedades fisicas y quimicas a 250 metros de resolucion (pH, porcentaje de arcilla, arena, materia organica y textura textural).

---

## Estructura del Proyecto

```text
kallpa-shift-backend/
│
├── app/
│   ├── __init__.py
│   ├── main.py                  # Servidor principal FastAPI y configuracion CORS
│   ├── config.py                # Variables de entorno y URLs de servicios externos
│   ├── static/
│   │   └── index.html           # Interfaz web bilingue para agricultores
│   ├── services/
│   │   ├── __init__.py
│   │   ├── nasa_client.py       # Conector API de NASA POWER (diario y climatologia)
│   │   ├── soilgrids_client.py  # Conector API de SoilGrids ISRIC v2 (suelos)
│   │   └── gamc_client.py       # Conector API de la Alcaldia de Cochabamba
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── crop_dataset.py      # Perfiles agronomicos y generador de datos sinteticos
│   │   ├── train_model.py       # Script de entrenamiento Random Forest con altitud
│   │   ├── predictor.py         # Motor de inferencia y proyeccion agroclimatica
│   │   └── modelo_cultivos_rf.joblib # Modelo serializado entrenado
│   ├── routers/
│   │   ├── __init__.py
│   │   └── api.py               # Endpoints REST de recomendacion y datasets
│   └── utils/
│       ├── __init__.py
│       └── cleaners.py          # Limpieza de nulos (-999) y preparacion tabular
│
├── datasets/                    # Carpeta para almacenar datasets descargados en CSV
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Esta guia de uso
```

---

## Requisitos del Sistema

- **Python 3.10 o superior** (verificar ejecutando `py --version` o `python --version`).
- Administrador de paquetes **pip**.
- Conexion a Internet para la primera consulta satelital a NASA POWER y SoilGrids.

---

## Instalacion Paso a Paso

Sigue estos comandos en tu terminal (PowerShell, CMD o Terminal de Linux/Mac):

### 1. Descomprimir el archivo
Extrae el contenido del archivo `.rar` y accede a la carpeta del proyecto:
```bash
cd kallpa-shift-backend
```

### 2. Crear y activar un entorno virtual (recomendado)

En Windows (PowerShell):
```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```
*(Nota: Si PowerShell restringe la ejecucion de scripts, ejecuta antes: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

En Windows (CMD):
```cmd
py -m venv venv
venv\Scripts\activate.bat
```

En Linux o macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar las dependencias
Con el entorno virtual activado, instala todas las librerias requeridas:
```bash
pip install -r requirements.txt
```

---

## Como Iniciar el Servidor

Para iniciar la aplicacion localmente, ejecuta cualquiera de los siguientes comandos:

**Opcion A (Directo con Python):**
```bash
py app/main.py
```

**Opcion B (Con Uvicorn y recarga automatica):**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Una vez que la consola indique `Uvicorn running on http://0.0.0.0:8000`, abre tu navegador:

- **Plataforma Web para Agricultores:** [http://localhost:8000/](http://localhost:8000/)
- **Documentacion interactiva Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Verificacion de estado (Health Check):** [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

## Instrucciones de Uso de la Interfaz Web

1. **Paso 1: Selecciona la zona de tu parcela:**
   - **La Maica (Distrito 9):** Fondo de valle plano (2.550 msnm), cuenca lechera.
   - **Distrito 9 Sur:** Valle central agricola (2.560 msnm), apto para maiz y hortalizas.
   - **Distrito 13 Tunari:** Laderas altas y pie de monte (2.850 msnm), clima fresco resistente a heladas.
   - **Valles Periurbanos:** Quillacollo, Sacaba, Colcapirhua (2.570 msnm), cinturones horticolas bajo riego.

2. **Paso 2: Elige la epoca y el horizonte de siembra:**
   - Selecciona el mes en que planeas sembrar (se indica el nombre del mes y su equivalente en el calendario agricola quechua).
   - Elige entre planificar para los proximos 3 meses (*Kimsa killapaq*) o un mes especifico (*Sapa killa*).

3. **Paso 3: Presiona el boton de consulta:**
   - Haz clic en `CONSULTAR / RIKUCHIY: Ver Cultivos Recomendados`.
   - La aplicacion se conectara a los satelites de la NASA y al servidor de suelos, procesara las variables con el modelo Random Forest y mostrara:
     - Resumen climatico simple: Temperatura, lluvia estimada, humedad, piso altitudinal y pH del suelo.
     - Fichas de cultivos idoneos con fotografia en primer plano, porcentaje de compatibilidad, consejos de siembra en espanol y quechua, dias de cosecha y demanda de riego.

---

## Endpoints Principales de la API REST

| Metodo | Endpoint | Descripcion |
| :--- | :--- | :--- |
| `GET` | `/` | Sirve la interfaz web para agricultores. |
| `GET` | `/api/v1/health` | Comprobacion de salud y estado del servidor. |
| `GET` | `/api/v1/ml/recomendar-por-satelite` | Recomendacion agroclimatica satelital para un mes y coordenadas. Parametros: `lat`, `lon`, `mes`, `anio`, `altitud_msnm`. |
| `GET` | `/api/v1/ml/proyeccion-trimestral` | Proyeccion anticipada a 3 meses con evaluacion de estabilidad agronomica y altitud. |
| `GET` | `/api/v1/clima` | Consulta directa al conector de NASA POWER diario/climatologia. |
| `GET` | `/api/v1/suelo` | Consulta de pH, arena, arcilla y textura en SoilGrids ISRIC. |
| `POST` | `/api/v1/ml/reentrenar` | Reentrena el modelo Random Forest con nuevos perfiles agronomicos. |
| `GET` | `/docs` | Documentacion interactiva Swagger con pruebas en vivo de todos los endpoints. |

---

## Creditos y Fuentes de Datos

- **NASA POWER Project:** Prediction Of Worldwide Energy Resources (Langley Research Center).
- **ISRIC - World Soil Information:** SoilGrids REST API v2.
- **GAMC:** Gobierno Autonomo Municipal de Cochabamba.
