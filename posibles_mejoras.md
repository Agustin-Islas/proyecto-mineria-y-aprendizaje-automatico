# Posibles Mejoras — ¿Conviene usar Redes Neuronales con las Top 10 Variables?

Análisis basado en los resultados obtenidos en [analisis_resultados.md](analisis_resultados.md).

---

## Respuesta corta

**No es la mejor estrategia.** Usar redes neuronales con solo las 10 mejores variables probablemente **empeoraría** los resultados en lugar de mejorarlos. A continuación se explica por qué, y qué alternativas sí tienen más chances de mejorar el rendimiento débil en la clase Control.

---

## ¿Por qué NO conviene una red neuronal aquí?

### 1. El dataset es demasiado pequeño para redes neuronales

Las redes neuronales necesitan **miles a decenas de miles de muestras** para aprender patrones sin memorizar los datos. Nosotros tenemos:

| Recurso | Cantidad |
|---|---|
| Sujetos totales | 252 |
| Filas totales | 756 |
| Sujetos Control (clase débil) | 64 |
| Filas Control | 192 |

Con solo **64 sujetos sanos** (y encima tenemos que separar Train/Test agrupando por ID), una red neuronal tendría muy pocos ejemplos de la clase que justamente queremos mejorar. El riesgo de **sobreajuste sería altísimo**.

> [!CAUTION]
> Una red neuronal con 252 sujetos es como usar un cañón para matar una mosca: mucha capacidad de aprendizaje, pero muy pocos datos para guiarla. El resultado más probable es que memorice el ruido en vez de aprender patrones reales.

### 2. Reducir a 10 variables descarta demasiada información

En el análisis de resultados vimos que **ninguna variable individual supera el 1.3% de importancia**. Esto significa que el modelo actual distribuye su decisión entre cientos de variables complementarias. Si reducimos a solo 10:

- Las top 10 variables acumulan apenas un ~**8.5%** de la importancia total.
- Estaríamos descartando el **91.5% restante** de la información que el modelo usa para decidir.
- Muchas variables individualmente débiles pueden ser muy útiles **en combinación** (el modelo Extra Trees ya captura estas interacciones).

### 3. El problema de la clase Control NO es culpa del algoritmo

El rendimiento débil en Control (62% recall) se debe principalmente a:

- **Desbalance 3:1**: El modelo tiene 3 veces más ejemplos de Parkinson que de Control, por lo que "aprende" mejor a reconocer la enfermedad.
- **Dificultad intrínseca del problema**: Algunos pacientes sanos pueden tener patrones vocales similares a los de Parkinson (fumadores, personas mayores con disfonía, etc.).

Cambiar el algoritmo de ExtraTrees a una red neuronal **no resuelve ninguno de estos dos problemas**. Son problemas de datos, no de modelo.

---

## ¿Qué alternativas SÍ podrían mejorar la clase Control?

### Opción 1: Ajustar el umbral de decisión (impacto alto, esfuerzo bajo)

Actualmente el modelo predice "Parkinson" si la probabilidad es ≥ 0.5. Pero dado el desbalance, podemos **subir ese umbral** (por ejemplo a 0.6 o 0.65) para que el modelo sea más exigente antes de diagnosticar Parkinson. Esto:
- ↑ Aumentaría el Recall de Control (menos falsos positivos).
- ↓ Podría reducir levemente el Recall de Parkinson.
- Es el cambio más simple y rápido de probar.

```python
# Ejemplo: buscar el umbral óptimo
from sklearn.metrics import balanced_accuracy_score
import numpy as np

umbrales = np.arange(0.3, 0.8, 0.01)
for umbral in umbrales:
    pred_ajustado = (proba_test >= umbral).astype(int)
    ba = balanced_accuracy_score(y_test, pred_ajustado)
    print(f"Umbral: {umbral:.2f} -> Balanced Accuracy: {ba:.4f}")
```

### Opción 2: Sobremuestreo SMOTE (impacto medio-alto, esfuerzo bajo)

SMOTE genera **muestras sintéticas** de la clase minoritaria (Control) interpolando entre vecinos cercanos. Esto equilibra las clases sin perder datos reales.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
x_train_bal, y_train_bal = smote.fit_resample(x_train, y_train)
# Ahora ambas clases tienen la misma cantidad de filas
```

> [!IMPORTANT]
> SMOTE debe aplicarse **solo dentro de cada Fold** de la validación cruzada, nunca antes de separar. Si se aplica antes, las muestras sintéticas podrían filtrar información entre Train y Test.

### Opción 3: Selección de features intermedia (impacto medio, esfuerzo medio)

En vez de pasar de 753 a 10 (muy agresivo), usar entre **30 y 100 variables** seleccionadas con métodos estadísticos:

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif

selector = SelectKBest(mutual_info_classif, k=50)
x_train_sel = selector.fit_transform(x_train, y_train)
```

Esto elimina el ruido de las columnas irrelevantes sin perder tanta información como al quedarse con solo 10.

### Opción 4: Probar Gradient Boosting (XGBoost / LightGBM)

Si se quiere probar otro algoritmo, los modelos de **Gradient Boosting** son una opción mucho mejor que redes neuronales para este escenario:

| Característica | Red Neuronal | Gradient Boosting |
|---|---|---|
| Datos mínimos necesarios | Miles+ | Cientos (suficiente) |
| Manejo de tablas | Regular | Excelente |
| Manejo de desbalance | Requiere técnicas extra | `scale_pos_weight` nativo |
| Interpretabilidad | Baja (caja negra) | Media (importancias) |
| Riesgo de sobreajuste aquí | Alto | Controlable |

```python
from xgboost import XGBClassifier

modelo = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    scale_pos_weight=3,  # compensa el desbalance 3:1
    random_state=42,
)
```

### Opción 5: Combinar estrategias (impacto más alto)

La combinación más prometedora sería:

1. Seleccionar las **top 50 features** (en vez de 10 o 753).
2. Aplicar **SMOTE** dentro de cada Fold.
3. Usar **ExtraTrees o XGBoost** (no red neuronal).
4. **Ajustar el umbral** de decisión post-entrenamiento.

---

## Resumen comparativo

| Estrategia | ¿Mejora Control? | Riesgo | Esfuerzo | Recomendada |
|---|---|---|---|---|
| Red neuronal + 10 features | ❌ Probablemente no | Alto (sobreajuste) | Alto | ❌ No |
| Ajuste de umbral | ✅ Sí | Bajo | Bajo | ✅ Sí |
| SMOTE | ✅ Sí | Medio | Bajo | ✅ Sí |
| Selección de 30-100 features | ✅ Posiblemente | Bajo | Medio | ✅ Sí |
| XGBoost / LightGBM | ✅ Posiblemente | Bajo | Medio | ✅ Sí |
| Combinar varias | ✅ Mayor probabilidad | Medio | Medio | ✅ Sí |

---

## Conclusión

El camino más efectivo para mejorar la detección de pacientes Control **no pasa por redes neuronales ni por reducir drásticamente las features**. Pasa por atacar directamente el desbalance de clases (SMOTE + ajuste de umbral) y, si se quiere experimentar con otro modelo, optar por Gradient Boosting en lugar de redes neuronales. Con 252 sujetos, los modelos basados en árboles siguen siendo la herramienta adecuada para este problema.
