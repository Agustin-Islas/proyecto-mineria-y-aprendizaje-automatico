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
    print(f"\n{nombre}")
    for metrica, valor in metricas.items():
        print(f"{metrica}: {valor:.3f}")


def imprimir_distribucion(nombre, datos):
    sujetos = datos[[ID, TARGET]].drop_duplicates()
    print(f"\n{nombre}")
    print("Filas:", len(datos))
    print("Sujetos:", datos[ID].nunique())
    print("Distribucion por filas:")
    print(datos[TARGET].value_counts().sort_index())
    print("Distribucion por sujetos:")
    print(sujetos[TARGET].value_counts().sort_index())


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

    return metricas, matriz, reporte, modelo


def mostrar_importancias(modelo_entrenado, columnas):
    importancias = modelo_entrenado.named_steps["modelo"].feature_importances_
    ranking = pd.DataFrame(
        {
            "variable": columnas,
            "importancia": importancias,
        }
    ).sort_values("importancia", ascending=False)

    print("\nVariables mas importantes:")
    print(ranking.head(25).to_string(index=False))


# %%
datos = cargar_datos(RUTA_CSV)

print("Dataset cargado")
print("Filas:", len(datos))
print("Columnas:", datos.shape[1])
print("Sujetos:", datos[ID].nunique())
print("Grabaciones por sujeto:")
print(datos.groupby(ID).size().value_counts().sort_index())

imprimir_distribucion("Dataset completo", datos)


# %%
train, test = separar_train_test_agrupado(datos)

imprimir_distribucion("Train agrupado por id", train)
imprimir_distribucion("Test agrupado por id", test)


# %%
metricas_cv, matriz_cv = validar_con_groupkfold(train)

imprimir_metricas("Validacion cruzada agrupada por id", metricas_cv)

print("\nMatriz de confusion CV:")
print(matriz_cv)


# %%
metricas_test, matriz_test, reporte_test, modelo_final = evaluar_en_test(train, test)

imprimir_metricas("Test final agrupado por id", metricas_test)

print("\nMatriz de confusion test:")
print(matriz_test)

print("\nReporte de clasificacion test:")
print(reporte_test)

x_train, _, _ = separar_variables(train)
mostrar_importancias(modelo_final, x_train.columns)
