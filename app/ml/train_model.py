import os
import sys

# Asegurar que la raíz del proyecto esté en el sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from typing import Dict, Any

from app.ml.crop_dataset import generar_dataset_entrenamiento

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ruta donde se persistirá el modelo entrenado
MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_cultivos_rf.joblib")

# Nombres de las características predictoras (Clima + Época + Suelo SoilGrids)
FEATURES = [
    "mes",
    "temperatura_media",
    "temperatura_minima",
    "temperatura_maxima",
    "precipitacion_mensual",
    "humedad_relativa",
    "radiacion_solar",
    "ph_suelo",
    "altitud_msnm"
]

def entrenar_modelo_random_forest() -> Dict[str, Any]:
    """
    Entrena el modelo Random Forest para clasificar y recomendar cultivos según la época del año
    y variables agroclimáticas. Guarda el modelo entrenado en disco.
    """
    print("[INFO] Generando datos agronomicos de entrenamiento...")
    df = generar_dataset_entrenamiento(muestras_por_cultivo=400, semilla=42)

    X = df[FEATURES]
    y = df["cultivo_recomendado"]

    # Separación en conjunto de entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[INFO] Entrenando RandomForestClassifier con {len(X_train)} muestras...")
    
    # Configuración del clasificador Random Forest
    modelo = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(X_train, y_train)

    # Evaluación del rendimiento del modelo
    predicciones = modelo.predict(X_test)
    exactitud = accuracy_score(y_test, predicciones)

    # Cálculo de la importancia de cada variable climática en las decisiones del bosque
    importancias = dict(zip(FEATURES, [round(float(v), 4) for v in modelo.feature_importances_]))

    print(f"[OK] Entrenamiento completado. Exactitud (Accuracy): {exactitud:.4f}")

    # Empaquetar el modelo y sus metadatos
    paquete_modelo = {
        "modelo": modelo,
        "features": FEATURES,
        "clases": list(modelo.classes_),
        "exactitud_test": round(float(exactitud), 4),
        "importancia_variables": importancias
    }

    # Guardar en disco para inferencia rápida
    joblib.dump(paquete_modelo, MODEL_PATH)
    print(f"[OK] Modelo guardado exitosamente en: {MODEL_PATH}")

    return {
        "status": "completado",
        "exactitud": round(float(exactitud), 4),
        "total_muestras": len(df),
        "cultivos": list(modelo.classes_),
        "importancia_variables": importancias
    }

if __name__ == "__main__":
    entrenar_modelo_random_forest()
