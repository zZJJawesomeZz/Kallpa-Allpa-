# Guía de Instalación, Integración y Expansión de Variables
## Proyecto: Kallpa Shift (NASA Space Apps Challenge)

Este paquete contiene el backend en Python con **FastAPI** y el motor de **Machine Learning (Random Forest)** para la plataforma de agricultura de precisión, integrando datos de **NASA POWER API** (clima satelital) y **SoilGrids ISRIC v2** (propiedades de suelo y pH).

---

## 1. Requisitos Previos

Antes de comenzar, asegúrate de tener instalado en tu computadora:
- **Python 3.10 o superior** (puedes verificarlo en tu terminal ejecutando `python --version` o `py --version`).
- Administrador de paquetes **pip**.

---

## 2. Instalación Paso a Paso

Sigue estos sencillos pasos en tu terminal (PowerShell, Símbolo del Sistema o Terminal de Linux/Mac):

### Paso 2.1: Descomprimir el proyecto
Descomprime el archivo `.rar` en la carpeta donde desees trabajar y entra a la carpeta:
```bash
cd kallpa-shift-backend
```

### Paso 2.2: Crear y activar un entorno virtual
Recomendamos usar un entorno virtual para no alterar tus librerías globales de Python.

- **En Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
  *(Nota: Si PowerShell te muestra un error de políticas de ejecución, ejecuta antes: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

- **En Windows (Símbolo del sistema / CMD):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate.bat
  ```

- **En Linux o macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Paso 2.3: Instalar las dependencias
Con el entorno virtual activado, instala todas las librerías necesarias con un solo comando:
```bash
pip install -r requirements.txt
```
*(Esto instalará: FastAPI, Uvicorn, Requests, Pandas, Pydantic, Scikit-Learn y Joblib).*

---

## 3. Cómo Arrancar el Servidor

Para iniciar el servidor en tu máquina local, ejecuta:
```bash
python app/main.py
```
O también puedes iniciarlo con uvicorn y recarga en vivo:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Una vez que veas en la consola `Uvicorn running on http://0.0.0.0:8000`, el backend ya estará listo para recibir peticiones:

- 🌐 **Interfaz Web Integrada:** [http://localhost:8000/](http://localhost:8000/)
- 📖 **Documentación Swagger Interactiva (Endpoints y Pruebas):** [http://localhost:8000/docs](http://localhost:8000/docs)
- 🩺 **Verificación de Salud (Health Check):** [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

## 4. Cómo Conectar este Backend con tu Frontend / Interfaz

El backend ya tiene configurado **CORS abierto** (`allow_origins=["*"]`), por lo que puedes llamarlo desde cualquier aplicación frontend (React, Next.js, Vue, Angular, Leaflet o HTML/JavaScript nativo) sin bloqueos del navegador.

### Ejemplo de consumo desde JavaScript (Frontend):

#### A) Recomendación de Cultivos usando el Clima Satelital (NASA POWER):
```javascript
// Llamada por coordenadas y mes del año (ej. mes 10 = Octubre)
async function obtenerRecomendacionSatelital(lat, lon, mes) {
  const url = `http://localhost:8000/api/v1/ml/recomendar-por-satelite?lat=${lat}&lon=${lon}&mes=${mes}`;
  
  const respuesta = await fetch(url);
  const datos = await respuesta.json();
  
  console.log("Condiciones climáticas detectadas:", datos.condiciones_evaluadas);
  console.log("Top cultivos recomendados:", datos.recomendaciones);
  // datos.recomendaciones contiene:
  // [
  //   { cultivo: "Papa (Patata)", idoneidad_porcentaje: 82.5, mes_ideal: true, descripcion: "..." },
  //   { cultivo: "Maíz", idoneidad_porcentaje: 65.0, ... }
  // ]
  return datos;
}
```

#### B) Simulación de Cultivos con Parámetros Manuales:
```javascript
async function simularCultivo(mes, tempMedia, lluviaMes, humedad) {
  const url = `http://localhost:8000/api/v1/ml/recomendar-cultivos?mes=${mes}&temperatura_media=${tempMedia}&precipitacion_mensual=${lluviaMes}&humedad_relativa=${humedad}`;
  
  const respuesta = await fetch(url);
  const resultado = await respuesta.json();
  return resultado.recomendaciones;
}
```

---

## 5. Guía para Agregar una NUEVA Variable al Modelo (Machine Learning)

Si van a incorporar una nueva variable climática o agronómica (por ejemplo: `altitud_msnm`, `ph_suelo`, `velocidad_viento`, etc.), sigan estos 5 pasos ordenados:

### Paso 5.1: Definir la variable en los perfiles
Abran el archivo `app/ml/crop_dataset.py` y agreguen el rango óptimo de la nueva variable a cada cultivo en `PERFILES_CULTIVOS`.
*Ejemplo:*
```python
{
    "cultivo": "Papa (Patata)",
    "meses_optimos": [9, 10, 11, 12, 1],
    ...
    "altitud_msnm": (2200, 3800), # <-- NUEVA VARIABLE
}
```
Y en la función `generar_dataset_entrenamiento()`, añadan la generación de la columna:
```python
altitud = np.random.uniform(perfil["altitud_msnm"][0], perfil["altitud_msnm"][1])
registros.append({
    ...
    "altitud_msnm": round(float(altitud), 1)
})
```

### Paso 5.2: Añadir la variable a la lista de características
En `app/ml/train_model.py`, agreguen el nombre de la columna en la lista `FEATURES`:
```python
FEATURES = [
    "mes",
    "temperatura_media",
    "temperatura_minima",
    "temperatura_maxima",
    "precipitacion_mensual",
    "humedad_relativa",
    "radiacion_solar",
    "altitud_msnm"  # <-- NUEVA VARIABLE AQUÍ
]
```

### Paso 5.3: Actualizar el predictor
En `app/ml/predictor.py`, agreguen el parámetro a la función `predecir_cultivos_optimos()` y al diccionario `datos_entrada`:
```python
def predecir_cultivos_optimos(
    mes: int,
    temperatura_media: float,
    ...,
    altitud_msnm: float = 2500.0 # <-- NUEVO PARÁMETRO
):
    datos_entrada = pd.DataFrame([{
        "mes": int(mes),
        ...,
        "altitud_msnm": float(altitud_msnm)
    }])
```

### Paso 5.4: Reentrenar el modelo
Ejecuta en tu terminal:
```bash
python app/ml/train_model.py
```
El script generará el dataset actualizado, reentrenará el bosque de árboles `RandomForestClassifier` y sobreescribirá el archivo serializado `modelo_cultivos_rf.joblib` con las nuevas características.

### Paso 5.5: Exponerla en el endpoint de la API
En `app/routers/api.py`, añade el nuevo parámetro a los endpoints `/api/v1/ml/recomendar-cultivos` para que el frontend lo pueda enviar cuando haga peticiones HTTP.

---

## 6. Estructura de Carpetas

```text
kallpa-shift-backend/
│
├── app/
│   ├── main.py              # Punto de entrada FastAPI, CORS y servidor web
│   ├── config.py            # Configuración de URLs y timeouts
│   ├── ml/                  # Módulo de Machine Learning
│   │   ├── crop_dataset.py  # Perfiles agronómicos y generador de datos
│   │   ├── train_model.py   # Script de entrenamiento Random Forest
│   │   ├── predictor.py     # Lógica de inferencia y llamada satelital
│   │   └── modelo_cultivos_rf.joblib # Modelo entrenado serializado
│   ├── services/
│   │   ├── nasa_client.py   # Conector API NASA POWER
│   │   └── gamc_client.py   # Conector API GAMC Cochabamba
│   ├── routers/
│   │   └── api.py           # Endpoints REST para el mapa y frontend
│   ├── utils/
│   │   └── cleaners.py      # Limpieza de nulos (-999) y preparación de datos
│   └── static/
│       └── index.html       # Interfaz visual de prueba e interacción
│
├── datasets/                # Carpeta para guardar datasets CSV generados
├── requirements.txt         # Lista de dependencias del proyecto
├── README.md                # Documentación general
└── GUIA_INSTALACION_E_INTEGRACION.md # Esta guía para el equipo
```
