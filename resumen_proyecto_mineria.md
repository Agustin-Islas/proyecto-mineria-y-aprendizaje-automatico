# Resumen del Proyecto: Clasificación de Parkinson mediante Análisis Acústico

## 1. Descripción del Problema
El objetivo de este proyecto es construir un **clasificador automático** que permita distinguir entre personas sanas (Control) y personas con la enfermedad de Parkinson utilizando grabaciones de voz.
La base de datos (`pd_speech_features.csv`) cuenta con 755 características acústicas extraídas de las voces de 252 sujetos (con 3 grabaciones por persona). 
Un desafío crítico del proyecto es el **fuerte desbalance de clases**: hay aproximadamente 3 pacientes con Parkinson por cada persona sana.

---

## 2. Decisiones Metodológicas (Paso a Paso)

Para construir un modelo robusto y confiable, se tomaron las siguientes decisiones a lo largo del pipeline de ejecución:

### Paso 1: Carga y Limpieza de Datos
Se eliminan las columnas y filas con datos faltantes irreparables y se asegura que todas las características acústicas tengan el tipo de dato numérico correcto para ser procesadas de manera uniforme.

### Paso 2: Separación Train / Test (Evitando Fuga de Datos)
En lugar de dividir las grabaciones al azar, se utilizó **`GroupShuffleSplit`** agrupando por paciente. Esto garantiza que las 3 grabaciones de una persona vayan juntas a Entrenamiento (Train) o a Prueba (Test). De esta forma evitamos el *data leakage* (que el modelo "memorice" la voz del paciente en lugar de aprender a detectar verdaderamente la enfermedad).

### Paso Intermedio: Definición del Pipeline (Plantilla)
Se estructuró el modelo utilizando un **Pipeline** para encadenar procesos sin hacer trampa estadística:
1. **SimpleImputer**: Rellena datos faltantes con la mediana evaluada solo en el conjunto de entrenamiento.
2. **SelectKBest** (opcional): Herramienta preparada para seleccionar las mejores variables más adelante en los experimentos.
3. **ExtraTreesClassifier**: Se eligió este algoritmo (un ensamble de árboles aleatorios) porque es muy robusto al sobreajuste en datasets con muchas variables y pocas filas.

### Paso 3: Validación Cruzada (GroupKFold)
Para medir el rendimiento interno de manera honesta se usó validación cruzada con **5 folds agrupados** sobre el Pipeline base, obteniendo predicciones limpias (*Out-Of-Fold*).

### Paso 4: Evaluación en Test
Se entrenó el modelo base con el 80% de entrenamiento y se evaluó su capacidad de generalizar frente al 20% de prueba, sirviendo esto como punto de partida (*baseline*).

---

### Fase de Optimización y Experimentación
#### Agregación de Probabilidades por Paciente
A partir de este punto, todas las evaluaciones y experimentos (Pasos 5, 7, 8 y 9) se realizan promediando las probabilidades de las 3 grabaciones de cada paciente. Esto permite tomar decisiones a **nivel de sujeto** (lo cual es clínicamente relevante y más estable) en lugar de evaluar cada grabación por separado.

### Paso 5: Experimento — Pesos de Clase
Para afrontar el desbalance (3:1), se experimentó dándole mayor peso a la clase minoritaria (Sanos). Esto fuerza al modelo a penalizar más fuertemente el error de clasificar a una persona sana como enferma, buscando equilibrar el Balanced Accuracy.

### Paso 6: Importancia de Variables (Biomarcadores)
A partir del modelo entrenado, se analizó internamente cuáles de las 755 variables acústicas fueron las más utilizadas para tomar decisiones. Esto aporta explicabilidad clínica al modelo, demostrando en qué frecuencias o atributos vocales se apoya para detectar el Parkinson.

### Paso 7: Experimento — Selección de Variables (Feature Selection)
Usando `SelectKBest` con el criterio de **Información Mutua**, se probó entrenar el modelo recortando las variables a los Top 10, 20, 40, etc. Se descubrió que usar solo las **40 mejores variables** mejora radicalmente el rendimiento general, ya que elimina el "ruido" que introducían las otras cientos de variables poco útiles.

### Paso 8: Búsqueda del Umbral Óptimo
Dado que las decisiones médicas requieren ajustar la sensibilidad del modelo, se buscó el mejor umbral de decisión (iterando de 0.30 a 0.80) analizando las predicciones **agrupadas por sujeto**. Se encontró que un umbral más exigente mejora drásticamente la detección de la clase Control sin sacrificar excesivamente el diagnóstico de Parkinson.

### Paso 9: Evaluación Final por Sujeto
Se unificaron todas las optimizaciones descubiertas en el modelo definitivo:
- **Top 40 características**.
- **Umbral de probabilidad ajustado**.
- **Predicción a nivel paciente**: Se promedian las 3 grabaciones de cada individuo antes de dar el veredicto final.
La combinación de estas técnicas produjo un modelo capaz de diagnosticar la enfermedad de manera robusta, precisa y realista frente a pacientes desconocidos.

---

## 3. Flujo de Ejecución del Modelo

El siguiente diagrama resume visualmente el flujo de los datos y la sucesión de experimentos realizados en el código:

```text
         CSV (pd_speech_features.csv)
                      │
                      ▼
        ┌─────────────────────────────┐
        │  PASO 1: Carga y Limpieza   │
        └──────────────┬──────────────┘
                       │
                       ▼
        ┌─────────────────────────────┐
        │  PASO 2: Split Train / Test │
        │   (Agrupado por Paciente)   │
        └──────────────┬──────────────┘
                       │
                       ▼
        ┌─────────────────────────────┐
        │       Paso Intermedio:      │
        │    Definición del Pipeline  │
        └──────────────┬──────────────┘
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
       Train (80%)            Test (20%)
            │                     │
            ▼                     ▼
┌──────────────────────┐  ┌──────────────────────┐
│ PASO 3: Validación   │  │ PASO 4: Evaluación   │
│ Cruzada (GroupKFold) │  │ Base en Test         │
└──────────────────────┘  └──────────┬───────────┘
                                     │
                                     ▼
                      ┌────────────────────────────┐
                      │   Fase de Optimización:    │
                      │  (Promediando por Sujeto)  │
                      └──────────────┬─────────────┘
                                     │
                                     ▼
                      ┌────────────────────────────┐
                      │  PASO 5: Experimento       │
                      │  Pesos de Clase            │
                      └──────────────┬─────────────┘
                                     │
                                     ▼
                      ┌────────────────────────────┐
                      │  PASO 6: Importancia de    │
                      │  Variables Acústicas       │
                      └──────────────┬─────────────┘
                                     │
                                     ▼
                      ┌────────────────────────────┐
                      │  PASO 7: Experimento       │
                      │  Selección de Features     │
                      └──────────────┬─────────────┘
                                     │
                                     ▼
                      ┌────────────────────────────┐
                      │  PASO 8: Búsqueda de       │
                      │  Umbral Óptimo             │
                      └──────────────┬─────────────┘
                                     │
                                     ▼
                      ┌────────────────────────────┐
                      │  PASO 9: Evaluación Final  │
                      │  por Sujeto                │
                      └────────────────────────────┘
```
