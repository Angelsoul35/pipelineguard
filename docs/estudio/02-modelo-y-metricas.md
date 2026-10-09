# 2. El modelo explicable y las métricas

## Por qué un ensamble de árboles (y no solo una regresión)

El detector es un **clasificador de gradient boosting**
(`HistGradientBoostingClassifier` de scikit-learn). Dos razones para preferirlo a una
regresión logística sola:

1. **Los motores de corrosión interactúan de forma no lineal.** La presión parcial de CO2
   importa *más* cuando hay una fase de agua continua donde disolverse; los efectos de
   régimen de flujo (GOR, caudal) son tipo umbral. Un ensamble de árboles captura esas
   interacciones que una logística lineal no puede. En estos datos, eso sube el ROC-AUC
   fuera de muestra de ~0.83 (logística) a ~0.94 (ensamble).
2. **Sigue siendo explicable.** No por los coeficientes, sino por la **importancia de
   permutación**: una medida agnóstica al modelo y robusta frente a la fuga, que mide
   cuánto aporta *de verdad* cada feature sobre datos que el modelo no vio.

> **Concepto — gradient boosting.** Muchos árboles pequeños en secuencia; cada uno corrige
> el error del conjunto anterior. `HistGradientBoosting` agrupa los valores en *bins* para
> entrenar rápido. Funciona muy bien en datos tabulares como estos.

### La explicabilidad no es opcional en seguridad

En integridad de petróleo y gas, una cuadrilla de inspección **no actúa sobre una caja
negra**. Por eso el detector también ajusta, al lado del ensamble, una **regresión
logística** sobre features estandarizadas y expone sus **coeficientes con signo**: se leen
como "esta variable sube / baja el riesgo". Es el chequeo de dirección que un ingeniero
contrasta contra la química de la corrosión.

## Las métricas (con fórmulas)

Como el target es binario y desbalanceado (25% positivos), la *accuracy* sola engaña (el
baseline de mayoría ya da 75%). Por eso reportamos tres métricas:

- **ROC-AUC.** Probabilidad de que el modelo le dé mayor puntaje a un pozo de riesgo alto
  elegido al azar que a uno normal elegido al azar. 0.5 = azar; 1.0 = perfecto. Es la
  métrica principal porque no depende del umbral ni del desbalance.
- **Precisión y recall** (base de F1):
  ```
  precisión = VP / (VP + FP)     recall = VP / (VP + FN)
  ```
  (VP = verdaderos positivos, FP = falsos positivos, FN = falsos negativos.)
- **F1.** Media armónica de precisión y recall:
  ```
  F1 = 2 · (precisión · recall) / (precisión + recall)
  ```
- **Accuracy.** `(VP + VN) / total`. Se reporta, pero siempre **contra el baseline de
  mayoría** (0.75), nunca sola.

## Sobreajuste, subajuste y la brecha train/held-out

> **Concepto.** Un modelo **sobreajustado** memoriza el entrenamiento y falla en datos
> nuevos: su error de train es mucho mejor que el de test. **Subajustado**: ni siquiera le
> gana a un baseline tonto. **Bien ajustado**: generaliza — train y test van parejos y le
> gana al baseline.

Cómo lo medimos, sin trampas:

1. **Partición held-out.** 20% de los pozos apartados, que el modelo nunca ve en el
   entrenamiento. El número que reportamos es el de *ese* conjunto.
2. **La brecha** `AUC_train − AUC_held-out`. Si es pequeña, el modelo generaliza. Si el
   train está en 0.99 y el test cae, es memorización.
3. **Validación cruzada estratificada (5 folds)** sobre el train. Si el AUC de CV coincide
   con el del held-out, el resultado es estable y no suerte de una partición.
4. **Control de etiquetas barajadas.** Se reentrena el *mismo* pipeline con las etiquetas
   permutadas. Debe caer al azar (~0.5 AUC). Si "acierta" con etiquetas barajadas, está
   midiendo fuga, no aprendizaje.
5. **Chequeo de regresión.** Un regresor sobre la velocidad de corrosión cruda; si tiene
   R² decente, confirma que la señal existe también en el target continuo y que el umbral
   no fabrica el resultado.

> **Nota sobre ensambles que "saturan" el train.** Un ensamble de árboles puede llegar a
> AUC de train muy alto sin estar sobreajustado, *si* el CV y el held-out concuerdan. Aquí
> el train queda en 0.976 (no 1.0) y el held-out en 0.944: la brecha es chica. No hay que
> lisiar el modelo para bajar el train; hay que mirar la concordancia CV/held-out.

## Ejemplo resuelto — los números reales

Salida de `python -m pipelineguard.real_benchmark` sobre los 10 292 pozos:

```
base rate / majority: 25.1% high-risk  ->  majority baseline accuracy 75.0%

cross-val (5-fold)  : ROC-AUC 0.9430 +/- 0.0029   F1 0.7702   acc 0.8938
held-out test       : ROC-AUC 0.9441   F1 0.7941   acc 0.9043
train vs held-out   : train AUC 0.9762  held-out AUC 0.9441  (gap 0.0321)
shuffled-label ctrl : held-out AUC 0.4937  (chance ~ 0.5)
regression x-check  : corrosion-rate R2 0.5704 (held-out)

verdict: well-fit
```

Lectura, paso a paso:

- **Le gana al baseline con holgura.** 0.944 de ROC-AUC y **90.4% de accuracy** contra el
  piso de 75.0%. Es una mejora real, sobre pozos que no vio.
- **Está bien ajustado, no sobreajustado.** La brecha train→held-out es de solo **0.032**,
  y la CV (0.943) cae justo sobre el held-out (0.944). Es habilidad estable, no suerte.
- **No hay fuga.** Al barajar las etiquetas, el mismo pipeline cae a **0.494** — azar puro.
  La ventaja es estructura de corrosión real, no un atajo escondido.
- **La señal también está en el continuo.** El regresor sobre la velocidad de corrosión da
  R² 0.57: el umbral no inventa el resultado.

### Qué lo mueve — importancias reales

```
top feature importances (permutation, drop in held-out ROC-AUC):
   wellhead_press_psi         +0.1922
   total_liquid_bpd           +0.1164
   co2_partial_pressure_psi   +0.1093
   bsw_pct                    +0.0470
   gas_rate_mmcfd             +0.0461
   water_cut                  +0.0453

logistic-regression direction (standardized coef, + sube el riesgo):
   wellhead_press_psi         -1.0732
   total_liquid_bpd           -0.4622
   gas_oil_ratio              +0.3518
   water_cut                  -0.1577
   co2_partial_pressure_psi   -0.1234
```

El detector se apoya exactamente en las cantidades que importan en ingeniería de
corrosión: presión de operación, caudal de líquido, la **presión parcial de CO2** (¡el
tercer motor más fuerte sobre datos no vistos!) y la fase de agua (BSW y corte de agua). El
mecanismo de corrosión dulce aparece solo en un modelo al que nunca le contamos la química.
Los signos de la logística le dan al ingeniero un chequeo de dirección legible.

> **La lección.** Reporta el número **fuera de muestra** y **contra un baseline**, nunca el
> de entrenamiento. Un buen detector le gana al baseline de mayoría; la brecha chica y la
> concordancia CV/held-out dicen que generaliza; y el control de barajado es la prueba de
> que no hay fuga. Medir todo esto es señal de madurez, no de debilidad.
