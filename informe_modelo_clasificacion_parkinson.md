# Informe del Modelo de Clasificación de Parkinson

Este informe detalla el funcionamiento paso a paso del script de Python para el proyecto, el cual utiliza aprendizaje automático para detectar la enfermedad de Parkinson basándose en biomarcadores de voz.

## 1. Carga y Limpieza de Datos (`cargar_datos`)

El dataset contiene múltiples grabaciones de voz por cada paciente (identificado con un `id`).

- **Carga Dinámica:** La función `subir_csv_si_hace_falta` está diseñada para facilitar su uso en Google Colab, permitiéndote subir el archivo CSV manualmente si no se encuentra en el entorno.
- **Limpieza Básica:** El dataset tiene una estructura particular donde los nombres de las columnas están en la segunda fila. El script los lee correctamente y elimina columnas vacías (`dropna`). Luego, asegura que tanto el `id` como la variable objetivo (`class`) sean números enteros y convierte el resto de métricas acústicas a formato numérico, filtrando finalmente cualquier fila que haya quedado con datos corruptos.

## 2. Separación de Datos (Train y Test Agrupado)

> [!CAUTION]
> **Riesgo de Fuga de Datos (Data Leakage):** Dado que un mismo sujeto tiene múltiples grabaciones, si mezclamos las filas aleatoriamente, el modelo podría "memorizar" la firma vocal de un paciente durante el entrenamiento y luego acertar en la prueba simplemente reconociendo al sujeto, sin realmente aprender a detectar la enfermedad.

Para solucionar esto, se utiliza la función `separar_train_test_agrupado` junto con `GroupShuffleSplit`. Esto garantiza que **todas las grabaciones de un mismo paciente** vayan íntegramente a Entrenamiento (Train) o íntegramente a Prueba (Test), pero nunca a ambos. Por defecto, el 20% de los sujetos se reserva como grupo de prueba final.

## 3. Creación del Pipeline y Modelo (`crear_modelo`)

El modelo no es un algoritmo simple, sino un **Pipeline** (tubería de procesamiento de Scikit-Learn) compuesto por 2 etapas:

1. **Imputación (`SimpleImputer`):** Como medida de seguridad, si a alguna fila le falta un valor (`NaN`) que sobrevivió a la limpieza, se rellena automáticamente con la mediana de esa característica.
2. **Clasificador (`ExtraTreesClassifier`):** Se eligió un ensamblado de múltiples árboles de decisión (*Extra Trees*), un algoritmo excelente para lidiar con conjuntos de datos que tienen cientas de columnas (alta dimensionalidad) como este. Se configuró estratégicamente con:
   - `n_estimators=600`: Construye 600 árboles independientes.
   - `max_depth=8`: Limita la profundidad de los árboles para evitar que memoricen el ruido de los datos (sobreajuste o *overfitting*).
   - `class_weight="balanced"`: En medicina es común tener más personas sanas que enfermas (o viceversa). Este parámetro le otorga mayor peso matemático a la clase minoritaria para que el modelo preste la misma atención a ambos casos.

## 4. Validación Cruzada Agrupada (`validar_con_groupkfold`)

Antes de someter al modelo al examen final con los datos de Test, necesitamos una forma de estimar qué tan bien va a funcionar **sin gastar nuestros datos de prueba**. Para eso usamos la Validación Cruzada de 5 pliegues (Folds).

### ¿Cómo funciona?

Se toman los datos de Entrenamiento y se dividen en 5 bloques. En cada iteración (Fold), el modelo se **re-entrena desde cero** usando 4 bloques y se prueba con el bloque restante. Al rotar quién es el grupo de prueba, todos los pacientes pasan eventualmente por ese rol:

```
Fold 1: [PRUEBA] [Entrena] [Entrena] [Entrena] [Entrena]
Fold 2: [Entrena] [PRUEBA] [Entrena] [Entrena] [Entrena]
Fold 3: [Entrena] [Entrena] [PRUEBA] [Entrena] [Entrena]
Fold 4: [Entrena] [Entrena] [Entrena] [PRUEBA] [Entrena]
Fold 5: [Entrena] [Entrena] [Entrena] [Entrena] [PRUEBA]
```

### ¿Por qué agrupada por ID de paciente?

Al igual que en la separación Train/Test (sección 2), aquí también se usa agrupamiento (`GroupKFold`). Esto asegura que **todas las grabaciones de un mismo paciente caigan juntas** dentro del mismo bloque. Si no hiciéramos esto, el modelo podría "reconocer la voz" de un paciente que ya vio en entrenamiento y acertar por memorización en vez de por detección real de la enfermedad.

### ¿Cómo se interpretan los resultados?

Las métricas finales que se reportan son el promedio acumulado de las predicciones de los 5 Folds. Esto da una estimación mucho más confiable que entrenar una sola vez:

- Si el modelo anda **bien en todos los Folds** → es estable y confiable, generaliza bien a pacientes nuevos.
- Si anda **bien en algunos Folds y mal en otros** → el rendimiento depende de qué pacientes le toquen, lo cual es una señal de que el modelo no es robusto.


## 5. Evaluación Final (`evaluar_en_test`)

Aquí es donde el modelo final se entrena con el 100% del conjunto de Entrenamiento y se somete al "examen final" usando el 20% de sujetos (Test) que aislamos al principio. La función `calcular_metricas` reporta varios KPIs fundamentales:

- **Accuracy (Exactitud):** El porcentaje de aciertos totales.
- **Balanced Accuracy:** El promedio de los aciertos tomando en cuenta ambas clases por igual (Sano vs Parkinson), una métrica mucho más realista si los datos están desbalanceados.
- **Precision (Precisión):** De todos los pacientes que el algoritmo diagnosticó como "Parkinson", ¿cuántos lo tenían realmente?
- **Recall (Exhaustividad):** De todos los pacientes que **realmente** tenían Parkinson, ¿qué porcentaje logró detectar el algoritmo? *(En medicina, esta es una de las métricas más críticas, ya que no queremos que se nos escapen personas enfermas).*
- **F1-Score:** Una media armónica que equilibra la Precisión y el Recall en un solo número.
- **ROC AUC:** La capacidad global del modelo para discernir entre un caso sano y uno enfermo observando sus niveles de certeza (probabilidades).

## 6. Interpretación de Resultados y Gráficos

Finalmente, el flujo de ejecución principal imprime de manera estructurada todas estas distribuciones y métricas, y ejecuta tres gráficos clave para hacer el informe más visual e interpretable:

- **Matriz de Confusión (`graficar_matriz_confusion`):** Es una cuadrícula que cruza la realidad vs. la predicción. Te permite visualizar gráficamente los Verdaderos Positivos, Verdaderos Negativos, Falsos Positivos (se le diagnosticó Parkinson a un paciente sano) y Falsos Negativos (se dio de alta a un paciente enfermo).
- **Curva ROC (`graficar_curva_roc`):** Una representación gráfica del balance entre la tasa de verdaderos positivos contra los falsos positivos en diferentes umbrales. Cuanto más cerca esté la curva del vértice superior izquierdo (área más grande bajo la curva), mejor será el diagnóstico.
- **Importancia de Variables (`graficar_importancia_variables`):** El modelo Extra Trees tiene la capacidad intrínseca de contarnos qué datos miró más para tomar sus decisiones. El gráfico de barras resalta el **Top 10 de biomarcadores** (características acústicas) que mayor impacto tuvieron para distinguir los temblores o deficiencias vocales asociados al Parkinson.