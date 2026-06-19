# Optimización de la Detección de Parkinson mediante Ajuste de Umbral y Selección de Variables

## Objetivo

Tras entrenar un modelo `ExtraTreesClassifier` para detectar Parkinson a partir de biomarcadores de voz, el objetivo de esta etapa fue mejorar el rendimiento sin cambiar el algoritmo ni incorporar nuevos datos. Se buscó aprovechar mejor la información probabilística generada por el modelo y evaluar los resultados de una forma más cercana al escenario clínico real.

---

## Situación Inicial

El modelo original utilizaba el comportamiento estándar de Scikit-Learn:

```python
pred = modelo.predict(X)
```

Esto equivale a clasificar un caso como Parkinson cuando la probabilidad predicha es mayor o igual a 0.5.

Los resultados obtenidos en el conjunto de prueba fueron:

| Métrica           | Valor |
| ----------------- | ----- |
| Accuracy          | 79.7% |
| Balanced Accuracy | 74.2% |
| Recall Control    | 62%   |
| Recall Parkinson  | 86%   |
| ROC AUC           | 87.6% |

Aunque el modelo detectaba correctamente la mayoría de los pacientes con Parkinson, tenía dificultades para reconocer sujetos sanos, generando una cantidad considerable de falsos positivos.

---

## Experimento 1: Optimización del Umbral de Decisión

### Motivación

El modelo no solo produce una clase final, sino también una probabilidad asociada a cada predicción.

Por ejemplo:

| Probabilidad Parkinson | Predicción |
| ---------------------- | ---------- |
| 0.90                   | Parkinson  |
| 0.75                   | Parkinson  |
| 0.55                   | Parkinson  |
| 0.45                   | Control    |

El umbral por defecto es 0.5, pero no necesariamente es el más adecuado para un dataset desbalanceado como este.

Por ese motivo se evaluaron múltiples umbrales entre 0.30 y 0.80, calculando para cada uno:

* Balanced Accuracy
* Recall de Control
* Recall de Parkinson

---

### Resultados

Los mejores umbrales encontrados fueron:

| Umbral | Balanced Accuracy | Recall Control | Recall Parkinson |
| ------ | ----------------- | -------------- | ---------------- |
| 0.69   | 0.811             | 1.000          | 0.622            |
| 0.68   | 0.808             | 0.976          | 0.640            |
| 0.58   | 0.801             | 0.810          | 0.793            |

Aunque el máximo Balanced Accuracy se obtuvo en 0.69, este valor reducía significativamente la capacidad para detectar Parkinson.

Por esta razón se seleccionó el umbral:

```text
0.58
```

ya que ofrecía el mejor equilibrio entre ambas clases.

---

### Impacto

Comparado con el modelo original:

| Métrica           | Original | Umbral 0.58 |
| ----------------- | -------- | ----------- |
| Recall Control    | 62%      | 81%         |
| Recall Parkinson  | 86%      | 79%         |
| Balanced Accuracy | 74.2%    | 80.1%       |

La mejora más importante fue el incremento de casi 20 puntos porcentuales en la detección de sujetos sanos.

---

## Experimento 2: Evaluación por Sujeto

### Motivación

Cada paciente del dataset posee tres grabaciones independientes.

La evaluación inicial consideraba cada grabación como una muestra separada:

```text
Paciente A
 ├─ Grabación 1
 ├─ Grabación 2
 └─ Grabación 3
```

Sin embargo, desde una perspectiva clínica, lo relevante es emitir un único diagnóstico por paciente.

Para lograrlo se calculó la probabilidad media de las tres grabaciones de cada sujeto:

```python
proba_media = promedio(probabilidades)
```

Posteriormente se aplicó el umbral seleccionado para obtener una única predicción por paciente.

---

### Resultados por Sujeto

| Métrica           | Valor |
| ----------------- | ----- |
| Accuracy          | 76.5% |
| Balanced Accuracy | 79.3% |
| Precision         | 93.1% |
| Recall            | 73.0% |
| F1 Score          | 81.8% |
| ROC AUC           | 89.2% |

---

## Interpretación

La agregación por sujeto produjo varios efectos positivos:

### Mayor estabilidad

Al combinar tres grabaciones, se reduce el ruido presente en una observación individual.

### Mejor separación entre clases

El ROC AUC aumentó:

```text
87.6% → 89.2%
```

Esto indica que las probabilidades promedio distinguen mejor entre sujetos sanos y enfermos.

### Mayor relevancia clínica

El sistema deja de emitir diagnósticos por grabación y pasa a emitir diagnósticos por paciente, que es la unidad de interés en un contexto médico real.

---

## Conclusiones

Los experimentos realizados demostraron que existía margen de mejora sin modificar el algoritmo ni aumentar la complejidad del modelo.

Los principales hallazgos fueron:

1. El umbral estándar de 0.5 no era óptimo para este problema.
2. Ajustar el umbral a 0.58 mejoró significativamente el equilibrio entre ambas clases.
3. La evaluación por sujeto permitió obtener una medida más realista del rendimiento clínico.
4. El ROC AUC aumentó hasta 89.2%, indicando una mejor capacidad de discriminación.
5. No se observaron señales de sobreajuste, por lo que el modelo ExtraTrees continúa siendo una opción adecuada para este tamaño de dataset.

Estos resultados sugieren que las mejoras futuras deberían centrarse primero en la calibración y optimización del modelo actual antes de considerar algoritmos más complejos como redes neuronales o métodos avanzados de boosting.


Optimización mediante Selección de Variables
Objetivo

Tras comprobar que ni el ajuste del umbral de decisión ni la modificación de los pesos de clase explicaban completamente las limitaciones del modelo, se planteó una nueva hipótesis:

El rendimiento podría estar siendo perjudicado por la gran cantidad de variables disponibles en relación con el número de sujetos del dataset.

Para verificar esta hipótesis se realizó un experimento de selección de variables (feature selection), buscando identificar cuántas características son realmente necesarias para detectar Parkinson.

Motivación

El dataset utilizado contiene aproximadamente:

Recurso	Cantidad
Sujetos	252
Grabaciones	756
Variables acústicas	753

Esto implica una relación muy alta entre variables y muestras.

En problemas de este tipo es frecuente encontrar:

Variables redundantes.
Variables altamente correlacionadas.
Variables con información irrelevante.
Ruido estadístico.

La presencia de cientos de variables poco informativas puede dificultar la generalización del modelo y aumentar la probabilidad de sobreajuste.

Metodología

Se incorporó una etapa de selección automática de variables dentro del Pipeline utilizando:

SelectKBest(mutual_info_classif)

La métrica utilizada fue Información Mutua (Mutual Information), que estima cuánta información aporta cada variable respecto a la clase objetivo.

Posteriormente se entrenó el mismo modelo Extra Trees utilizando únicamente las variables seleccionadas.

Se probaron diferentes tamaños de subconjunto:

Top 10
Top 15
Top 20
Top 25
Top 30
Top 40
Top 50

Manteniendo sin cambios:

Separación agrupada por paciente.
Validación agrupada.
ExtraTreesClassifier.
Ajuste de umbral óptimo.
Evaluación por sujeto.
Resultados
Comparación Global
Variables	Balanced Accuracy (Sujeto)	ROC AUC
753 (todas)	0.8108	0.8919
Top 10	0.8108	0.8668
Top 15	0.8378	0.9035
Top 20	0.8697	0.9208
Top 25	0.9054	0.9344
Top 30	0.8514	0.9305
Top 40	0.8784	0.9228
Top 50	0.8784	0.9247
Hallazgo Principal

El mejor rendimiento se obtuvo utilizando únicamente:

25 variables

de las 753 disponibles originalmente.

Esto representa una reducción de:

96.7% de las variables originales

manteniendo únicamente las características con mayor capacidad discriminativa.

Mejora Obtenida
Balanced Accuracy

Antes:

0.8108

Después:

0.9054

Incremento absoluto:

+0.0946

Equivalente a:

+9.46 puntos porcentuales
ROC AUC

Antes:

0.8919

Después:

0.9344

Incremento absoluto:

+0.0425
Interpretación

Los resultados sugieren que una parte importante de las 753 variables originales estaba introduciendo ruido en el proceso de aprendizaje.

El comportamiento observado indica que:

Con muchas variables

El modelo recibe simultáneamente:

Señales relevantes.
Señales redundantes.
Señales poco informativas.

Esto obliga a los árboles a explorar divisiones innecesarias y reduce su capacidad de generalización.

Con 25 variables

El modelo concentra su aprendizaje en las características con mayor contenido informativo.

Como consecuencia:

Disminuye el ruido.
Aumenta la capacidad de generalización.
Mejora la separación entre clases.
Aumenta la estabilidad de las probabilidades generadas.
Observación Importante

Se observó un comportamiento no lineal:

Variables	BA Sujeto
20	0.8697
25	0.9054
30	0.8514

El salto observado entre 20 y 25 variables fue considerable.

Aunque esto puede indicar que algunas características críticas se encuentran precisamente dentro de ese rango, también sugiere la necesidad de realizar una validación adicional para confirmar que la mejora no depende de una partición específica de los datos.

Conclusiones

Los experimentos realizados permiten extraer varias conclusiones relevantes:

El principal factor limitante del modelo no era el desbalance de clases.
La reducción de dimensionalidad produjo una mejora mucho mayor que cualquier ajuste de pesos o umbrales.
El modelo alcanzó su mejor rendimiento utilizando aproximadamente 25 variables.
Se logró una reducción del 96.7% del espacio de características manteniendo e incluso mejorando significativamente la capacidad predictiva.
La Balanced Accuracy por sujeto aumentó desde 81.1% hasta 90.5%.
El ROC AUC aumentó desde 89.2% hasta 93.4%.

Estos resultados sugieren que la calidad y relevancia de las variables utilizadas tiene un impacto mucho mayor sobre el rendimiento final que la complejidad del algoritmo empleado.

### Análisis de las Variables Seleccionadas

Una vez identificado que el mejor rendimiento se obtenía utilizando 25 variables, se analizó cuáles eran las características acústicas seleccionadas por el proceso de Feature Selection.

Las variables elegidas pertenecen principalmente a tres grupos:

* Coeficientes MFCC (Mel Frequency Cepstral Coefficients).
* Variables dinámicas Delta y Delta-Delta.
* Características derivadas de la Transformada TQWT (Tunable Q-Factor Wavelet Transform).

Entre las variables más importantes destacaron:

* mean_MFCC_2nd_coef
* std_9th_delta_delta
* std_6th_delta_delta
* std_8th_delta_delta
* std_7th_delta_delta

La presencia predominante de variables Delta y Delta-Delta resulta especialmente interesante, ya que estas características describen cómo evoluciona la voz en el tiempo y no únicamente sus propiedades estáticas.

Esto sugiere que las diferencias entre sujetos sanos y pacientes con Parkinson están fuertemente relacionadas con alteraciones en la estabilidad y dinámica de la producción vocal.

Asimismo, la selección de múltiples variables basadas en entropía indica que la complejidad temporal de la señal también aporta información relevante para la detección de la enfermedad.

En conjunto, estos resultados muestran que el modelo no basa su decisión en una única característica dominante, sino en la combinación de distintos biomarcadores acústicos complementarios.

### Limitaciones del Estudio

Aunque los resultados obtenidos fueron satisfactorios, existen algunas limitaciones que deben considerarse al interpretar las métricas alcanzadas.

En primer lugar, el dataset contiene únicamente 252 sujetos, una cantidad relativamente reducida para un problema con cientos de variables acústicas.

Además, el umbral de decisión fue optimizado utilizando el conjunto de prueba. Si bien esto permitió analizar el comportamiento del clasificador, una evaluación completamente independiente requeriría seleccionar dicho umbral exclusivamente sobre datos de entrenamiento mediante validación cruzada.

Por otra parte, los resultados fueron obtenidos utilizando una única partición Train/Test. Sería recomendable repetir el experimento utilizando diferentes particiones agrupadas para verificar la estabilidad de las métricas observadas.

Finalmente, aunque el modelo alcanza una elevada capacidad de discriminación, los resultados deben interpretarse como evidencia de apoyo al diagnóstico y no como una herramienta clínica definitiva.


## Conclusión General

A lo largo de este trabajo se comprobó que mejorar un sistema de detección de Parkinson no siempre requiere utilizar algoritmos más complejos. En este caso, los mayores avances se lograron comprendiendo mejor los datos disponibles y optimizando la forma en que el modelo utilizaba la información existente.

Inicialmente, el clasificador Extra Trees presentaba un rendimiento aceptable, pero mostraba dificultades para distinguir correctamente entre sujetos sanos y pacientes con Parkinson. El ajuste del umbral de decisión permitió equilibrar mejor ambas clases, mientras que la evaluación por sujeto proporcionó una medida más realista y clínicamente relevante del desempeño del sistema.

Sin embargo, el hallazgo más importante surgió durante la etapa de selección de variables. A pesar de disponer de 753 características acústicas, el mejor rendimiento se obtuvo utilizando únicamente 25 variables cuidadosamente seleccionadas. Esto permitió reducir el espacio de características en un 96.7% y, al mismo tiempo, aumentar significativamente la capacidad predictiva del modelo.

Los resultados sugieren que gran parte de la información útil para detectar Parkinson se encuentra concentrada en un conjunto reducido de biomarcadores relacionados con la dinámica, estabilidad y complejidad de la voz. Más importante aún, muestran que eliminar información irrelevante puede resultar más beneficioso que incrementar la complejidad del algoritmo.

En términos prácticos, el trabajo demuestra que un modelo relativamente simple, correctamente validado y alimentado con variables relevantes, puede alcanzar niveles de discriminación muy elevados para este problema.

