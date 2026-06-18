# Análisis de Resultados — Modelo de Clasificación de Parkinson

Este documento interpreta los resultados obtenidos al ejecutar el modelo `ExtraTreesClassifier` sobre el dataset de biomarcadores de voz para la detección de la enfermedad de Parkinson.

---

## 1. Composición del Dataset

| Aspecto | Valor |
|---|---|
| Filas totales | 756 |
| Columnas (features) | 753 (755 − `id` − `class`) |
| Sujetos únicos | 252 |
| Grabaciones por sujeto | 3 (uniforme) |

### Distribución de clases

| Clase | Sujetos | Filas | Proporción |
|---|---|---|---|
| 0 (Control / Sano) | 64 | 192 | 25.4% |
| 1 (Parkinson) | 188 | 564 | 74.6% |

> [!IMPORTANT]
> **El dataset está desbalanceado (~3:1).** Hay casi 3 veces más pacientes con Parkinson que sanos. Esto significa que un modelo "tonto" que siempre prediga "Parkinson" ya acertaría el 74.6% de las veces sin haber aprendido nada. Por eso es fundamental mirar el **Balanced Accuracy** y no solo el Accuracy crudo.

---

## 2. Separación Train / Test

| Conjunto | Sujetos | Filas | Control | Parkinson |
|---|---|---|---|---|
| Train | 201 (79.8%) | 603 | 50 sujetos (150 filas) | 151 sujetos (453 filas) |
| Test | 51 (20.2%) | 153 | 14 sujetos (42 filas) | 37 sujetos (111 filas) |

La proporción de clases se mantiene similar en ambos conjuntos (~25% Control, ~75% Parkinson), lo cual es correcto. Además, ningún paciente aparece en ambos conjuntos simultáneamente, eliminando el riesgo de fuga de datos.

---

## 3. Resultados de la Validación Cruzada (5 Folds)

| Métrica | Valor |
|---|---|
| Accuracy | 0.7745 (77.5%) |
| Balanced Accuracy | 0.6983 (69.8%) |
| Precision | 0.8499 (85.0%) |
| Recall | 0.8499 (85.0%) |
| F1-Score | 0.8499 (85.0%) |
| ROC AUC | 0.7758 (77.6%) |

### Interpretación

- El **Accuracy del 77.5%** parece razonable, pero hay que tener cuidado: como el 74.6% del dataset es Parkinson, un modelo que siempre diga "Parkinson" ya tendría ~74.6% de accuracy. Nuestro modelo supera eso por solo ~3 puntos porcentuales en accuracy crudo.
- El indicador más honesto es el **Balanced Accuracy del 69.8%**, que promedia los aciertos de ambas clases por igual. Esto nos dice que, en promedio, el modelo acierta correctamente el ~70% de los casos cuando le damos el mismo peso a detectar sanos y enfermos.
- La **Precision, Recall y F1 del 85%** están calculadas sobre la clase positiva (Parkinson). Son buenas, pero se benefician del hecho de que esa clase es mayoritaria.
- El **ROC AUC de 0.776** indica una capacidad de discriminación aceptable pero con margen de mejora. Un modelo perfecto tendría 1.0 y uno aleatorio 0.5.

### Matriz de Confusión (CV Agrupada)

|  | Predijo Control | Predijo Parkinson |
|---|---|---|
| **Era Control** | 82 ✅ | 68 ❌ |
| **Era Parkinson** | 68 ❌ | 385 ✅ |

- De 150 muestras Control: acertó 82, falló 68 → **Recall Control CV = 54.7%** (preocupante).
- De 453 muestras Parkinson: acertó 385, falló 68 → **Recall Parkinson CV = 85.0%**.
- El error se concentra fuertemente en la clase Control: casi la mitad de los sanos fueron clasificados como enfermos durante la validación cruzada.

> [!NOTE]
> La Validación Cruzada nos sirve como *estimación previa*. Los resultados del Test final (sección 4) son los que realmente importan, porque usan pacientes completamente aislados.

---

## 4. Resultados del Test Final

| Métrica | Valor CV | Valor Test | Diferencia |
|---|---|---|---|
| Accuracy | 0.7745 | **0.7974** | +0.023 ↑ |
| Balanced Accuracy | 0.6983 | **0.7420** | +0.044 ↑ |
| Precision | 0.8499 | **0.8571** | +0.007 ↑ |
| Recall | 0.8499 | **0.8649** | +0.015 ↑ |
| F1-Score | 0.8499 | **0.8610** | +0.011 ↑ |
| ROC AUC | 0.7758 | **0.8756** | +0.100 ↑ |

### Interpretación

**Buena noticia: el Test superó a la Validación Cruzada en todas las métricas.** Esto es una señal positiva que indica que el modelo **no está sobreajustado** (no memorizó los datos de entrenamiento). Si estuviera sobreajustado, esperaríamos ver métricas altas en CV pero una caída fuerte en Test.

- **Accuracy del 79.7%**: El modelo acierta ~80% de las predicciones generales.
- **Balanced Accuracy del 74.2%**: Compensando el desbalance, acierta ~74% en promedio entre ambas clases. Es una mejora significativa respecto al 69.8% de CV.
- **ROC AUC del 87.6%**: Es el salto más notable (+10 puntos). Indica que el modelo tiene una **buena capacidad de separación** entre las probabilidades que asigna a pacientes sanos vs. enfermos. Esto es muy positivo.

### Matriz de Confusión (Test Final)

|  | Predijo Control | Predijo Parkinson |
|---|---|---|
| **Era Control** | 26 ✅ | 16 ❌ |
| **Era Parkinson** | 15 ❌ | 96 ✅ |

- **26 Verdaderos Negativos**: Sanos correctamente identificados como sanos.
- **16 Falsos Positivos**: Sanos diagnosticados erróneamente como Parkinson.
- **15 Falsos Negativos**: Enfermos que el modelo dejó pasar como sanos.
- **96 Verdaderos Positivos**: Enfermos correctamente detectados.

Comparado con la CV, el Test mejoró notablemente en la clase Control (de 54.7% a 62% de recall), confirmando que entrenar con más datos ayudó.

### Curva ROC (Test Final) — AUC = 0.88

La curva ROC grafica la tasa de verdaderos positivos (eje Y) contra la tasa de falsos positivos (eje X) en distintos umbrales de decisión.

| Rango AUC | Interpretación |
|---|---|
| 0.50 | Aleatorio (inútil) |
| 0.60 – 0.70 | Pobre |
| 0.70 – 0.80 | Aceptable |
| **0.80 – 0.90** | **Bueno ← nuestro modelo (0.88)** |
| 0.90 – 1.00 | Excelente |

La curva sube rápidamente en la zona izquierda del gráfico, lo que significa que el modelo logra detectar una gran proporción de enfermos reales (alta sensibilidad) **antes de empezar a cometer muchos errores con los sanos**. La zona donde se aplana (~0.2 a ~0.4 en el eje X) indica los casos más ambiguos, probablemente pacientes con síntomas vocales leves que son difíciles de clasificar.

### Análisis por clase (Reporte de Clasificación)

| Clase | Precision | Recall | F1-Score | Muestras |
|---|---|---|---|---|
| Control (Sano) | 0.63 | 0.62 | 0.63 | 42 |
| Parkinson | 0.86 | 0.86 | 0.86 | 111 |

Aquí está el hallazgo más importante del análisis:

> [!WARNING]
> **El modelo detecta bien el Parkinson (86% recall), pero tiene dificultades para identificar correctamente a los pacientes sanos (62% recall).** Esto significa que aproximadamente **4 de cada 10 personas sanas son clasificadas erróneamente como enfermas de Parkinson** (falsos positivos).

**¿Qué significan estos números en la práctica?**

- **Recall del 86% en Parkinson**: De cada 100 pacientes que realmente tienen Parkinson, el modelo detecta correctamente a 86. Se le escapan 14 (falsos negativos). En un contexto de screening médico, esto es aceptable pero mejorable.
- **Recall del 62% en Control**: De cada 100 personas sanas, el modelo identifica correctamente a 62. Las otras 38 reciben un "falso diagnóstico" de Parkinson. En un contexto real, esto generaría derivaciones innecesarias a especialistas, aunque un segundo examen clínico lo descartaría.
- **Precision del 86% en Parkinson**: De cada 100 pacientes que el modelo diagnostica como "Parkinson", 86 realmente lo tienen. Los otros 14 son falsos positivos.
- **Precision del 63% en Control**: De cada 100 pacientes que el modelo clasifica como "sanos", solo 63 lo están realmente. Los otros 37 son enfermos que el modelo no detectó.

---

## 5. Variables Más Importantes (Top Biomarcadores)

| Posición | Variable | Importancia |
|---|---|---|
| 1 | `std_9th_delta_delta` | 0.0131 |
| 2 | `std_9th_delta` | 0.0090 |
| 3 | `std_6th_delta_delta` | 0.0089 |
| 4 | `std_8th_delta_delta` | 0.0084 |
| 5 | `std_7th_delta_delta` | 0.0081 |
| 6 | `tqwt_kurtosisValue_dec_36` | 0.0079 |
| 7 | `std_6th_delta` | 0.0078 |
| 8 | `mean_MFCC_2nd_coef` | 0.0078 |
| 9 | `tqwt_entropy_shannon_dec_16` | 0.0071 |
| 10 | `std_11th_delta_delta` | 0.0069 |

### Interpretación

Las variables se agrupan en dos familias principales de biomarcadores de voz:

**1. Coeficientes MFCC y sus derivadas (delta y delta-delta)**

Los **MFCC** (Mel-Frequency Cepstral Coefficients) son una representación compacta del espectro de frecuencias de la voz. Las variantes `delta` y `delta-delta` capturan cómo cambian esos coeficientes **a lo largo del tiempo** (velocidad y aceleración del cambio espectral). La columna `std` indica la desviación estándar, es decir, qué tan *irregular* es ese cambio.

- **¿Por qué son relevantes para Parkinson?** La enfermedad de Parkinson afecta el control motor de las cuerdas vocales, generando temblores, rigidez y lentitud en los movimientos articulatorios. Esto produce patrones de variación espectral (delta-delta) más erráticos o más planos de lo normal, que los MFCC capturan con mucha precisión.
- **Que la variable #1 sea `std_9th_delta_delta`** indica que la *irregularidad en la aceleración del cambio espectral del 9no coeficiente MFCC* es el rasgo más discriminante. En otras palabras: la inestabilidad fina de la voz es lo que más delata la enfermedad.

**2. Características TQWT (Tunable Q-factor Wavelet Transform)**

Las variables `tqwt_kurtosisValue_dec_36` y `tqwt_entropy_shannon_dec_16` provienen de una descomposición en wavelets que analiza la señal de voz en diferentes bandas de frecuencia.

- **Kurtosis**: Mide qué tan "puntiaguda" es la distribución de la señal en una banda específica. Valores altos indican picos abruptos (posibles temblores vocales).
- **Entropía de Shannon**: Mide el desorden o irregularidad de la señal en esa banda. Mayor entropía = señal más caótica, lo cual puede ser indicativo de una pérdida de control motor.

> [!NOTE]
> **Dato importante**: Ninguna variable individual tiene una importancia mayor al 1.3%. Esto es esperable con 753 features: el modelo distribuye su capacidad de decisión entre muchas variables complementarias. No hay un solo "marcador mágico" de Parkinson en la voz, sino una combinación de decenas de biomarcadores acústicos que, juntos, permiten una detección razonable.

---

## 6. Resumen y Conclusiones

| Aspecto | Evaluación |
|---|---|
| ¿El modelo generaliza bien? | ✅ Sí. Test superó a CV → no hay sobreajuste. |
| ¿Detecta bien Parkinson? | ✅ Sí. Recall del 86% en la clase enferma. |
| ¿Detecta bien a los sanos? | ⚠️ Regular. Recall del 62% en la clase Control. |
| ¿Es confiable la separación? | ✅ Sí. ROC AUC de 0.88 en Test (rango "Bueno"). |
| ¿Hay fuga de datos? | ✅ No. Se agrupó por ID en todo momento. |

### Fortalezas del modelo

- **No hay sobreajuste**: Las métricas de Test son iguales o mejores que las de CV.
- **Buena separación probabilística**: Un ROC AUC de 0.876 indica que el modelo asigna probabilidades más altas a los verdaderos enfermos y más bajas a los sanos, lo cual es valioso si se quiere ajustar un umbral de decisión.
- **Diseño experimental correcto**: La agrupación por paciente evita la fuga de datos, un error muy común en datasets con múltiples observaciones por sujeto.

### Debilidades y puntos de mejora

- **Rendimiento débil en la clase Control**: El modelo confunde a muchos sanos con enfermos. Esto se debe en parte al desbalance del dataset (3:1), donde el modelo tiene menos ejemplos de personas sanas para aprender. Posibles mejoras:
  - Técnicas de sobremuestreo (SMOTE) para la clase minoritaria.
  - Ajuste del umbral de decisión (actualmente 0.5) buscando un punto que optimice el Balanced Accuracy.
  - Selección de features más agresiva para reducir el ruido de las 753 columnas.
- **Dataset relativamente pequeño**: 252 sujetos es un tamaño modesto para un problema de clasificación médica. Los resultados deben interpretarse con cautela y no como evidencia clínica concluyente.
