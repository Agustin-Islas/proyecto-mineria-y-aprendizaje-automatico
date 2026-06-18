"""
Clasificacion de Parkinson con biomarcadores de voz.

Modelo final elegido:
    ExtraTreesClassifier

Motivo:
    Fue el mejor modelo en validacion cruzada agrupada por id y obtuvo buen
    desempeno en test sin mezclar grabaciones del mismo sujeto.

Archivo esperado en Colab:
    pd_speech_features.csv
"""

# %%
import os
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


# %%
SEMILLA = 42
RUTA_CSV = "pd_speech_features.csv"

ID = "id"
TARGET = "class"

PROPORCION_TEST = 0.20
FOLDS = 5


# %%
def subir_csv_si_hace_falta(ruta_csv):
    if os.path.exists(ruta_csv):
        return ruta_csv

    try:
        # pyrefly: ignore [missing-import]
        from google.colab import files

        print("Selecciona pd_speech_features.csv desde tu computadora.")
        files.upload()
    except ImportError:
        print("No se encontro el CSV y no se esta ejecutando en Colab.")

    if os.path.exists(ruta_csv):
        return ruta_csv

    archivos_csv = [archivo for archivo in os.listdir(".") if archivo.endswith(".csv")]
    if len(archivos_csv) == 1:
        print(f"Usando CSV encontrado: {archivos_csv[0]}")
        return archivos_csv[0]

    raise FileNotFoundError(f"No se encontro {ruta_csv}.")


def cargar_datos(ruta_csv):
    # El CSV tiene una primera fila descriptiva y la segunda fila contiene los
    # nombres reales de las columnas.
    datos = pd.read_csv(subir_csv_si_hace_falta(ruta_csv), header=1)
    datos = datos.dropna(axis=1, how="all")

    datos[ID] = datos[ID].astype(int)
    datos[TARGET] = datos[TARGET].astype(int)

    for columna in datos.columns:
        if columna not in [ID, TARGET]:
            datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

    datos = datos.dropna().reset_index(drop=True)
    return datos


def separar_variables(datos):
    x = datos.drop(columns=[ID, TARGET])
    y = datos[TARGET]
    grupos = datos[ID]
    return x, y, grupos


def separar_train_test_agrupado(datos):
    x, y, grupos = separar_variables(datos)
    separador = GroupShuffleSplit(
        n_splits=1,
        test_size=PROPORCION_TEST,
        random_state=SEMILLA,
    )

    idx_train, idx_test = next(separador.split(x, y, grupos))

    train = datos.iloc[idx_train].reset_index(drop=True)
    test = datos.iloc[idx_test].reset_index(drop=True)

    assert set(train[ID]).isdisjoint(set(test[ID]))
    return train, test


def crear_modelo():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "modelo",
                ExtraTreesClassifier(
                    n_estimators=600,
                    max_depth=8,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=SEMILLA,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def calcular_metricas(y_real, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_real, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, zero_division=0),
        "recall": recall_score(y_real, y_pred, zero_division=0),
        "f1": f1_score(y_real, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_real, y_proba),
    }


def imprimir_metricas(nombre, metricas):
    print(f"\n{'='*50}")
    print(f" METRICAS: {nombre.upper()}")
    print(f"{'='*50}")
    for metrica, valor in metricas.items():
        nombre_metrica = metrica.replace('_', ' ').title()
        print(f"  {nombre_metrica:<20}: {valor:.4f}")


def imprimir_distribucion(nombre, datos):
    sujetos = datos[[ID, TARGET]].drop_duplicates()
    print(f"\n{'-'*50}")
    print(f" DISTRIBUCION: {nombre.upper()}")
    print(f"{'-'*50}")
    print(f"  Filas totales:   {len(datos)}")
    print(f"  Sujetos unicos:  {datos[ID].nunique()}")
    print("\n  Distribucion de clases (por fila):")
    for clase, cuenta in datos[TARGET].value_counts().sort_index().items():
        print(f"    Clase {clase}: {cuenta} muestras")
    print("\n  Distribucion de clases (por sujeto):")
    for clase, cuenta in sujetos[TARGET].value_counts().sort_index().items():
        print(f"    Clase {clase}: {cuenta} sujetos")


def validar_con_groupkfold(datos_train):
    x_train, y_train, grupos_train = separar_variables(datos_train)
    kfold = GroupKFold(n_splits=FOLDS)

    pred_oof = np.zeros(len(datos_train), dtype=int)
    proba_oof = np.zeros(len(datos_train), dtype=float)

    for fold, (idx_tr, idx_val) in enumerate(
        kfold.split(x_train, y_train, grupos_train),
        start=1,
    ):
        modelo = deepcopy(crear_modelo())
        modelo.fit(x_train.iloc[idx_tr], y_train.iloc[idx_tr])

        pred_oof[idx_val] = modelo.predict(x_train.iloc[idx_val])
        proba_oof[idx_val] = modelo.predict_proba(x_train.iloc[idx_val])[:, 1]

        print(f"Fold {fold} terminado")

    metricas = calcular_metricas(y_train, pred_oof, proba_oof)
    matriz = confusion_matrix(y_train, pred_oof)
    return metricas, matriz


def evaluar_en_test(train, test):
    x_train, y_train, _ = separar_variables(train)
    x_test, y_test, _ = separar_variables(test)

    modelo = crear_modelo()
    modelo.fit(x_train, y_train)

    pred = modelo.predict(x_test)
    proba = modelo.predict_proba(x_test)[:, 1]

    metricas = calcular_metricas(y_test, pred, proba)
    matriz = confusion_matrix(y_test, pred)
    reporte = classification_report(
        y_test,
        pred,
        target_names=["Control", "Parkinson"],
        zero_division=0,
    )

    return metricas, matriz, reporte, modelo, y_test, proba


def mostrar_importancias(modelo_entrenado, columnas):
    importancias = modelo_entrenado.named_steps["modelo"].feature_importances_
    ranking = pd.DataFrame(
        {
            "variable": columnas,
            "importancia": importancias,
        }
    ).sort_values("importancia", ascending=False)

    print("\nVariables mas importantes (Top 10):")
    print(ranking.head(10).to_string(index=False))
    return ranking


def graficar_matriz_confusion(matriz, titulo="Matriz de Confusion"):
    plt.figure(figsize=(6, 4))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Control", "Parkinson"],
                yticklabels=["Control", "Parkinson"])
    plt.title(titulo)
    plt.ylabel("Etiqueta Real")
    plt.xlabel("Prediccion")
    plt.tight_layout()
    plt.show()


def graficar_importancia_variables(ranking, top_n=20, titulo="Top Variables Mas Importantes"):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=ranking.head(top_n), x='importancia', y='variable', hue='variable', legend=False, palette='viridis')
    plt.title(titulo)
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.show()


def graficar_curva_roc(y_real, y_proba, titulo="Curva ROC"):
    plt.figure(figsize=(6, 6))
    from sklearn.metrics import RocCurveDisplay
    RocCurveDisplay.from_predictions(y_real, y_proba, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title(titulo)
    plt.tight_layout()
    plt.show()


# %%
print("\n" + "="*80)
print(" 1. CARGA Y LIMPIEZA DE DATOS ")
print("="*80)
print("-> Cargando el archivo CSV y realizando limpieza inicial de datos faltantes.")
print("-> Objetivo: Detectar Parkinson (clase 1) vs Control (clase 0) usando voz.")
datos = cargar_datos(RUTA_CSV)

print(f"\n{'='*50}")
print(" RESUMEN DEL DATASET")
print(f"{'='*50}")
print(f"  Filas totales:     {len(datos)}")
print(f"  Columnas totales:  {datos.shape[1]}")
print(f"  Sujetos unicos:    {datos[ID].nunique()}")
print("\n  Grabaciones por sujeto:")
for grabaciones, cuenta in datos.groupby(ID).size().value_counts().sort_index().items():
    print(f"    {cuenta} sujetos tienen {grabaciones} grabaciones")

imprimir_distribucion("Dataset completo", datos)


# %%
print("\n" + "="*80)
print(" 2. SEPARACIÓN DE DATOS (TRAIN / TEST) ")
print("="*80)
print("-> Se separan los datos en Entrenamiento (Train) y Prueba (Test).")
print("-> IMPORTANTE: Se usa 'GroupShuffleSplit' agrupando por 'id' de paciente.")
print("-> Esto evita 'fuga de datos', asegurando que un mismo paciente NO esté")
print("-> simultáneamente en Train y en Test, obligando al modelo a generalizar.")
train, test = separar_train_test_agrupado(datos)

imprimir_distribucion("Distribución en Entrenamiento (Train)", train)
imprimir_distribucion("Distribución en Prueba (Test)", test)


# %%
print("\n" + "="*80)
print(" 3. VALIDACIÓN CRUZADA EN ENTRENAMIENTO (5 FOLDS) ")
print("="*80)
print("-> Entrenando el Pipeline (Imputación + ExtraTreesClassifier).")
print("-> Se simula el entrenamiento dividiendo los datos en 5 pliegues (Folds).")
print("-> Sirve para ver qué tan estable es el modelo antes de la prueba final.")
metricas_cv, matriz_cv = validar_con_groupkfold(train)

imprimir_metricas("Promedio de Validación Cruzada (CV)", metricas_cv)

print("\n-> GRAFICANDO: Matriz de Confusión de Validación Cruzada.")
print("   (Muestra cuántos aciertos y errores hubo en los Folds)")
graficar_matriz_confusion(matriz_cv, titulo="Matriz de Confusion (CV Agrupada)")


# %%
print("\n" + "="*80)
print(" 4. EVALUACIÓN FINAL DEL MODELO EN DATOS DE PRUEBA (TEST) ")
print("="*80)
print("-> Entrenando el modelo final con TODOS los datos de Entrenamiento.")
print("-> Evaluando predicciones sobre el 20% de pacientes (Test) que el modelo nunca vio.")
metricas_test, matriz_test, reporte_test, modelo_final, y_test, proba_test = evaluar_en_test(train, test)

imprimir_metricas("Resultados del Test Final", metricas_test)
print("  * Recall: Capacidad clave médica de no detectar falsos negativos.")
print("  * Balanced Acc: Exactitud promedio compensando si hay más sanos que enfermos.")

print("\nReporte Detallado de Clasificación:")
print(reporte_test)

print("\n" + "="*80)
print(" 5. INTERPRETACIÓN VISUAL DE RESULTADOS ")
print("="*80)

print("\n-> GRAFICANDO: Matriz de Confusión (Test).")
print("   (Muestra los Verdaderos Positivos/Negativos y los Falsos Positivos/Negativos)")
graficar_matriz_confusion(matriz_test, titulo="Matriz de Confusion (Test Final)")

print("\n-> GRAFICANDO: Curva ROC.")
print("   (Muestra la calidad del modelo: más cerca de la esquina superior izquierda = mejor)")
graficar_curva_roc(y_test, proba_test, titulo="Curva ROC (Test Final)")

x_train, _, _ = separar_variables(train)
ranking = mostrar_importancias(modelo_final, x_train.columns)

print("\n-> GRAFICANDO: Importancia de Variables (Top Biomarcadores).")
print("   (Muestra qué características acústicas ayudaron más a detectar la enfermedad)")
graficar_importancia_variables(ranking)
