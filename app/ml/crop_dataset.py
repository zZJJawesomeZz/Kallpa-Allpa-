import numpy as np
import pandas as pd
from typing import List, Dict, Any

"""
PERFILES AGROECOLÓGICOS Y EDAFOLÓGICOS DE CULTIVOS PARA COCHABAMBA
------------------------------------------------------------------
Define los rangos climáticos, ventanas estacionales y compatibilidad territorial
específica para el Valle de Cochabamba (Cercado, La Maica, Distrito 9, Distrito 13,
Valles Periurbanos) y el Trópico (Chapare).

Ajuste agronómico regional:
- En los Valles Centrales (La Maica / D9): Alfalfa, Maíz Choclero, Papa Waych'a, Haba,
  Cebolla, Zanahoria, Lechuga, Tomate, Avena Forrajera, Arveja.
- Se descarta la Quinua Real (exclusiva del Altiplano salino) y la Yuca (exclusiva del Trópico húmedo).
"""

PERFILES_CULTIVOS = [
    {
        "cultivo": "Alfalfa",
        "meses_optimos": [1, 2, 3, 9, 10, 11, 12],
        "temp_media": (15.0, 22.0),
        "temp_min": (8.0, 14.0),
        "temp_max": (22.0, 28.0),
        "lluvia_mensual": (45.0, 110.0),
        "humedad": (55.0, 75.0),
        "radiacion": (17.0, 23.0),
        "ph_suelo": (6.5, 7.8),
        "altitud_msnm": (2200, 2800),
        "ciclo_dias": "Perenne (Cortes cada 35-45 dias)",
        "demanda_hidrica": "Media-Alta (Riego periodico)",
        "zonas_compatibles": ["maica", "d9", "periurbanos", "valle_alto"],
        "descripcion": "Principal forraje de la cuenca lechera de La Maica. Excelente rendimiento en suelos francos neutros con riego."
    },
    {
        "cultivo": "Maíz Choclero",
        "meses_optimos": [10, 11, 12, 1, 2],
        "temp_media": (17.0, 25.0),
        "temp_min": (11.0, 17.0),
        "temp_max": (24.0, 31.0),
        "lluvia_mensual": (70.0, 150.0),
        "humedad": (60.0, 85.0),
        "radiacion": (18.0, 24.0),
        "ph_suelo": (5.8, 7.2),
        "altitud_msnm": (2200, 2750),
        "ciclo_dias": "120 - 150 dias",
        "demanda_hidrica": "Moderada a Alta",
        "zonas_compatibles": ["maica", "d9", "d13", "periurbanos", "valle_alto"],
        "descripcion": "Cultivo tradicional del Valle Central. Requiere temperaturas templadas y disponibilidad de humedad durante la floracion."
    },
    {
        "cultivo": "Papa Waych'a de Valle",
        "meses_optimos": [9, 10, 11, 12, 1],
        "temp_media": (12.0, 18.5),
        "temp_min": (4.0, 11.0),
        "temp_max": (18.0, 24.0),
        "lluvia_mensual": (50.0, 130.0),
        "humedad": (55.0, 80.0),
        "radiacion": (16.0, 22.0),
        "ph_suelo": (5.2, 6.6),
        "altitud_msnm": (2550, 3600),
        "ciclo_dias": "110 - 140 dias",
        "demanda_hidrica": "Moderada",
        "zonas_compatibles": ["maica", "d9", "d13", "periurbanos", "valle_alto"],
        "descripcion": "Variedad adaptada a valles y laderas. Prospera en suelos sueltos, bien drenados y con temperaturas moderadas."
    },
    {
        "cultivo": "Haba de Valle",
        "meses_optimos": [3, 4, 5, 6, 7],
        "temp_media": (10.0, 16.5),
        "temp_min": (2.0, 9.0),
        "temp_max": (17.0, 23.0),
        "lluvia_mensual": (15.0, 50.0),
        "humedad": (35.0, 60.0),
        "radiacion": (16.0, 21.0),
        "ph_suelo": (6.5, 7.9),
        "altitud_msnm": (2500, 3400),
        "ciclo_dias": "130 - 160 dias",
        "demanda_hidrica": "Baja a Moderada",
        "zonas_compatibles": ["maica", "d9", "d13", "periurbanos", "valle_alto"],
        "descripcion": "Leguminosa ideal para siembra de otono e invierno en valles. Tolera descensos termicos y mejora la fertilidad del suelo."
    },
    {
        "cultivo": "Cebolla Roja de Valle",
        "meses_optimos": [4, 5, 6, 7, 8],
        "temp_media": (13.0, 19.5),
        "temp_min": (5.0, 12.0),
        "temp_max": (20.0, 26.0),
        "lluvia_mensual": (10.0, 45.0),
        "humedad": (40.0, 65.0),
        "radiacion": (15.0, 21.0),
        "ph_suelo": (6.0, 7.2),
        "altitud_msnm": (2200, 2800),
        "ciclo_dias": "120 - 150 dias",
        "demanda_hidrica": "Baja (Riego por goteo/surco)",
        "zonas_compatibles": ["maica", "d9", "periurbanos", "valle_alto"],
        "descripcion": "Hortaliza de bulbo adaptada al ciclo seco de invierno en Cochabamba. Requiere buena insolacion y suelos francos."
    },
    {
        "cultivo": "Zanahoria",
        "meses_optimos": [2, 3, 4, 5, 8, 9],
        "temp_media": (13.0, 18.5),
        "temp_min": (6.0, 12.0),
        "temp_max": (19.0, 24.5),
        "lluvia_mensual": (25.0, 70.0),
        "humedad": (45.0, 70.0),
        "radiacion": (16.0, 21.0),
        "ph_suelo": (5.8, 7.0),
        "altitud_msnm": (2300, 2850),
        "ciclo_dias": "90 - 110 dias",
        "demanda_hidrica": "Moderada",
        "zonas_compatibles": ["maica", "d9", "periurbanos", "valle_alto"],
        "descripcion": "Hortaliza de raiz comun en zonas periurbanas y valles bajos. Demanda suelos profundos sin piedras y humedad regular."
    },
    {
        "cultivo": "Tomate de Valle",
        "meses_optimos": [8, 9, 10, 11, 12],
        "temp_media": (18.5, 25.5),
        "temp_min": (11.0, 17.0),
        "temp_max": (25.0, 31.0),
        "lluvia_mensual": (35.0, 95.0),
        "humedad": (50.0, 75.0),
        "radiacion": (18.0, 24.0),
        "ph_suelo": (5.8, 6.8),
        "altitud_msnm": (1800, 2650),
        "ciclo_dias": "100 - 130 dias",
        "demanda_hidrica": "Media a Alta",
        "zonas_compatibles": ["maica", "d9", "d13", "periurbanos"],
        "descripcion": "Cultivo de alto valor en valles. Sensible a heladas, requiere temperaturas diurnas templadas y buena ventilacion."
    },
    {
        "cultivo": "Lechuga y Hortalizas Periurbanas",
        "meses_optimos": [1, 2, 3, 4, 5, 8, 9, 10, 11, 12],
        "temp_media": (14.0, 20.0),
        "temp_min": (6.0, 13.0),
        "temp_max": (20.0, 26.0),
        "lluvia_mensual": (30.0, 80.0),
        "humedad": (50.0, 75.0),
        "radiacion": (16.0, 22.0),
        "ph_suelo": (6.0, 7.0),
        "altitud_msnm": (2400, 2800),
        "ciclo_dias": "60 - 80 dias",
        "demanda_hidrica": "Moderada continua",
        "zonas_compatibles": ["maica", "d9", "d13", "periurbanos"],
        "descripcion": "Produccion intensiva bajo riego en La Maica y cinturones horticolas. Ciclo corto con multiples cosechas anuales."
    },
    {
        "cultivo": "Avena Forrajera",
        "meses_optimos": [4, 5, 6, 7, 8],
        "temp_media": (9.0, 15.5),
        "temp_min": (1.0, 7.0),
        "temp_max": (16.0, 21.0),
        "lluvia_mensual": (10.0, 40.0),
        "humedad": (35.0, 55.0),
        "radiacion": (15.0, 20.0),
        "ph_suelo": (5.5, 7.2),
        "altitud_msnm": (2600, 3800),
        "ciclo_dias": "90 - 120 dias",
        "demanda_hidrica": "Baja",
        "zonas_compatibles": ["maica", "d9", "d13", "periurbanos", "valle_alto"],
        "descripcion": "Forraje de corte para la temporada seca y fria de invierno en La Maica. Excelente resistencia a bajas temperaturas."
    },
    {
        "cultivo": "Arveja",
        "meses_optimos": [3, 4, 5, 8, 9],
        "temp_media": (12.0, 18.0),
        "temp_min": (5.0, 11.0),
        "temp_max": (18.0, 24.0),
        "lluvia_mensual": (20.0, 60.0),
        "humedad": (40.0, 65.0),
        "radiacion": (16.0, 21.0),
        "ph_suelo": (6.0, 7.5),
        "altitud_msnm": (2400, 3200),
        "ciclo_dias": "90 - 120 dias",
        "demanda_hidrica": "Baja a Moderada",
        "zonas_compatibles": ["maica", "d9", "d13", "periurbanos", "valle_alto"],
        "descripcion": "Leguminosa de ciclo corto para valles. Muy valorada como cultivo de rotacion invernal para enriquecer el suelo."
    }
]

def generar_dataset_entrenamiento(muestras_por_cultivo: int = 400, semilla: int = 42) -> pd.DataFrame:
    """
    Genera el conjunto de datos de entrenamiento con perfiles agronómicos estrictos
    para la región de los Valles de Cochabamba.
    """
    np.random.seed(semilla)
    registros = []

    for perfil in PERFILES_CULTIVOS:
        cultivo = perfil["cultivo"]
        meses_optimos = perfil["meses_optimos"]

        for _ in range(muestras_por_cultivo):
            if np.random.rand() < 0.85:
                mes = np.random.choice(meses_optimos)
            else:
                mes = np.random.randint(1, 13)

            t_media = np.random.uniform(perfil["temp_media"][0], perfil["temp_media"][1])
            t_min = np.random.uniform(perfil["temp_min"][0], perfil["temp_min"][1])
            t_max = np.random.uniform(perfil["temp_max"][0], perfil["temp_max"][1])
            lluvia = np.random.uniform(perfil["lluvia_mensual"][0], perfil["lluvia_mensual"][1])
            humedad = np.random.uniform(perfil["humedad"][0], perfil["humedad"][1])
            radiacion = np.random.uniform(perfil["radiacion"][0], perfil["radiacion"][1])
            ph = np.random.uniform(perfil["ph_suelo"][0], perfil["ph_suelo"][1])
            altitud = np.random.uniform(perfil["altitud_msnm"][0], perfil["altitud_msnm"][1])

            t_media += np.random.normal(0, 0.4)
            t_min += np.random.normal(0, 0.3)
            t_max += np.random.normal(0, 0.5)
            lluvia = max(0.0, lluvia + np.random.normal(0, 3.0))
            humedad = np.clip(humedad + np.random.normal(0, 2.0), 15.0, 99.0)
            radiacion = max(5.0, radiacion + np.random.normal(0, 0.6))
            ph = np.clip(ph + np.random.normal(0, 0.12), 4.5, 9.0)
            altitud = np.clip(altitud + np.random.normal(0, 25.0), 1800.0, 4000.0)

            registros.append({
                "mes": int(mes),
                "temperatura_media": round(float(t_media), 2),
                "temperatura_minima": round(float(t_min), 2),
                "temperatura_maxima": round(float(t_max), 2),
                "precipitacion_mensual": round(float(lluvia), 2),
                "humedad_relativa": round(float(humedad), 2),
                "radiacion_solar": round(float(radiacion), 2),
                "ph_suelo": round(float(ph), 2),
                "altitud_msnm": round(float(altitud), 1),
                "cultivo_recomendado": cultivo
            })

    df = pd.DataFrame(registros)
    df = df.sample(frac=1.0, random_state=semilla).reset_index(drop=True)
    return df
