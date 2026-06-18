# Clasificacion de Parkinson con biomarcadores de voz

## Objetivo

El objetivo del nuevo enfoque es clasificar si una persona pertenece al grupo
con Parkinson o al grupo sano usando biomarcadores extraidos de grabaciones de
voz.

La variable objetivo es:

```text
class
```

Donde:

```text
1 = Parkinson
0 = Control sano
```

## Por que este dataset es mas adecuado

El dataset anterior estaba orientado a predecir severidad motora
(`motor_UPDRS`) y mostro baja generalizacion para pacientes nuevos. En cambio,
este dataset esta planteado directamente como un problema de clasificacion:
distinguir pacientes con Parkinson de controles sanos.

Ademas, contiene una gran cantidad de caracteristicas acusticas:

- Jitter
- Shimmer
- MFCC
- Wavelet features
- TQWT features
- Parametros de intensidad
- Frecuencias formantes
- Medidas de complejidad de la senal

Esto permite aplicar tecnicas de aprendizaje automatico de forma mas natural.

## Riesgo principal: fuga de informacion

El dataset tiene 756 filas, pero corresponden a 252 sujetos. Cada sujeto aparece
3 veces.

Por eso no se debe dividir el dataset aleatoriamente por filas. Si se hiciera
eso, grabaciones de un mismo sujeto podrian quedar tanto en entrenamiento como
en test, inflando artificialmente los resultados.

La regla central sera:

```text
Todas las particiones deben agruparse por id.
```

Esto significa que las tres grabaciones de una misma persona deben quedar juntas
en entrenamiento, validacion o test.

## Estrategia de modelado

Se usara una estrategia simple y defendible:

1. Cargar el dataset.
2. Eliminar la primera fila descriptiva del CSV.
3. Separar:

```text
X = biomarcadores + gender
y = class
grupos = id
```

4. Reservar un test final agrupado por `id`.
5. Hacer validacion cruzada con `GroupKFold`.
6. Comparar modelos candidatos.
7. Elegir el modelo con mejor desempeno en validacion.
8. Evaluar una sola vez sobre el test final.

## Modelos evaluados

Se evaluaron modelos adecuados para muchos atributos y pocos sujetos:

- Logistic Regression regularizada
- SVM con kernel RBF
- Random Forest
- Extra Trees
- HistGradientBoosting

El mejor modelo en validacion cruzada agrupada por `id` fue:

```text
Extra Trees
```

Este modelo resulto adecuado porque maneja relaciones no lineales, no requiere
escalado de variables y es robusto frente a conjuntos con muchas caracteristicas.

## Metricas

Como la clase esta desbalanceada, no alcanza con accuracy.

Se reportaran:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Matriz de confusion

La metrica principal sera:

```text
Balanced accuracy
```

porque considera el rendimiento en ambas clases.

## Criterio de validez

Este dataset deberia permitir obtener resultados mas utiles que el dataset de
telemonitoreo para un proyecto de aprendizaje automatico.

La conclusion sera valida solo si el modelo mantiene buen desempeno con
particiones agrupadas por `id`. Si los resultados bajan mucho bajo esa
validacion, se debera reportar que el problema era mas dificil de lo que parecia
y que resultados altos con split aleatorio no son confiables.

## Enfoque defendible del proyecto

El proyecto no debe presentarse solo como:

```text
clasificacion de Parkinson con alta accuracy
```

Sino como:

```text
clasificacion de Parkinson mediante biomarcadores vocales, evaluada con un
protocolo que evita fuga de informacion entre grabaciones del mismo sujeto.
```

Ese enfoque tiene mas valor metodologico y cientifico.

## Resultados obtenidos

La validacion y el test se realizaron agrupando por `id`, por lo que las tres
grabaciones de cada sujeto quedaron siempre en el mismo subconjunto.

Distribucion del dataset:

| Conjunto | Filas | Sujetos | Controles | Parkinson |
|---|---:|---:|---:|---:|
| Completo | 756 | 252 | 64 | 188 |
| Train | 603 | 201 | 50 | 151 |
| Test | 153 | 51 | 14 | 37 |

Resultados de validacion cruzada:

| Modelo | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Extra Trees | 0.774 | 0.698 | 0.850 | 0.850 | 0.850 | 0.776 |
| Random Forest | 0.809 | 0.697 | 0.841 | 0.921 | 0.879 | 0.776 |
| SVM RBF | 0.730 | 0.680 | 0.849 | 0.779 | 0.812 | 0.730 |
| Logistic Regression | 0.701 | 0.668 | 0.847 | 0.735 | 0.787 | 0.757 |
| HistGradientBoosting | 0.799 | 0.664 | 0.823 | 0.934 | 0.875 | 0.798 |

Resultados finales en test para Extra Trees:

| Modelo | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Extra Trees | 0.797 | 0.742 | 0.857 | 0.865 | 0.861 | 0.876 |

Matriz de confusion en test:

| | Predicho control | Predicho Parkinson |
|---|---:|---:|
| Control real | 26 | 16 |
| Parkinson real | 15 | 96 |

El modelo tiene mejor desempeno para la clase Parkinson que para la clase
control. Esto es esperable porque el dataset esta desbalanceado: hay 188 sujetos
con Parkinson y 64 controles.

## Interpretacion

El resultado es aceptable y metodologicamente defendible. El modelo logra un
ROC-AUC de 0.876 y una balanced accuracy de 0.742 en sujetos no vistos.

La principal limitacion es que todavia confunde una proporcion relevante de
controles como Parkinson. Por eso no debe presentarse como una herramienta
diagnostica clinica, sino como un modelo experimental de clasificacion basado en
biomarcadores vocales.

## Sobre el uso de redes neuronales

Las redes neuronales no son el mejor candidato inicial para este dataset.

El motivo principal es la relacion entre cantidad de variables y cantidad de
sujetos:

```text
754 atributos
252 sujetos
```

Hay muchas caracteristicas y pocos sujetos independientes. En ese contexto, una
red neuronal puede memorizar patrones del conjunto de entrenamiento y aparentar
un buen desempeno si la validacion no esta bien agrupada por `id`.

Para usar redes neuronales de forma responsable haria falta:

- Validacion agrupada por `id`.
- Regularizacion fuerte.
- Dropout.
- Early stopping.
- Red pequena.
- Comparacion obligatoria contra Extra Trees y modelos lineales.

Por ahora, Extra Trees es una opcion mas estable y defendible para el proyecto.
