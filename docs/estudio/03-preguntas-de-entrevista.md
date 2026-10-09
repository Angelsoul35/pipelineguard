# 3. Preguntas de entrevista

Preguntas típicas sobre PipelineGuard, con respuestas cortas que puedes sustentar.

### ¿Qué hace PipelineGuard, en una frase?

Marca qué pozos de petróleo y gas tienen riesgo alto de corrosión interna (el modo de
falla detrás de la mayoría de las fugas) a partir de las lecturas de proceso del cabezal,
con un modelo explicable y con base física.

### ¿Por qué detección (clasificación) y no regresión?

Porque la pregunta operativa es binaria: *¿inspecciono este pozo o no?* Clasificar da el
vocabulario que usa el equipo de confiabilidad (ROC-AUC, F1, precisión/recall) y es
robusto a la cola ruidosa de la velocidad de corrosión. Aun así reportamos un chequeo de
regresión (R² 0.57) para mostrar que la señal existe también en el valor continuo.

### ¿Cómo definiste el target y por qué así?

`high_risk = 1` si la velocidad de corrosión está en el peor cuartil de la flota (percentil
75, ≈ 0.2328). Es la decisión real de priorizar inspección. El umbral es una definición de
operación fija, aplicada igual a todos los pozos.

### ¿Cómo evitaste la fuga de datos?

La columna de corrosión se usa **solo** para la etiqueta; nunca como feature, y ninguna
feature es una transformación de ella (todas salen de los ocho sensores de proceso). La
prueba: al barajar las etiquetas, el modelo cae al azar (AUC 0.49). Si hubiera fuga,
seguiría "acertando".

### ¿Qué es la presión parcial de CO2 y por qué la calculas?

`pCO2 = (CO2_mol% / 100) × presión`. El CO2 se disuelve en el agua y forma ácido carbónico,
que corroe el acero. Lo que fija cuánto CO2 se disuelve no es el % molar solo, sino su
presión parcial (ley de Henry). Es el motor central de las correlaciones de De
Waard–Milliams. En el benchmark, es el tercer feature más importante sobre datos no vistos.

### ¿Qué es el corte de agua y el residual de balance de masa?

Corte de agua = `BWPD/(BOPD+BWPD)`: la fracción de líquido que es agua, el electrolito de
la corrosión. El residual de balance = `BSW/100 − corte_de_agua`: dos estimaciones
independientes de la fracción de agua que deberían coincidir; un residual grande indica que
el balance de agua no cierra (deriva de medidor o anomalía de separación).

### ¿Por qué gradient boosting y no una regresión logística?

Los motores de corrosión interactúan de forma no lineal (el CO2 corroe más con fase de
agua; los efectos de flujo son tipo umbral). El ensamble captura eso y sube el AUC de ~0.83
(logística) a ~0.94. Mantenemos la logística al lado para leer la **dirección** de cada
variable, porque en seguridad el modelo tiene que ser explicable.

### ¿Cómo sabes que no está sobreajustado?

La brecha entre el AUC de train (0.976) y el de held-out (0.944) es de solo 0.032, y la
validación cruzada de 5 folds (0.943) coincide con el held-out. Si memorizara, el test se
desplomaría. El ensamble no satura el train (0.976, no 1.0).

### ¿Qué significa que un ensamble tenga train alto?

No implica sobreajuste por sí solo. La prueba está en la concordancia CV/held-out. No hay
que degradar el modelo para bajar el train artificialmente; hay que mirar la generalización.

### ¿Cuál es el baseline y por qué importa?

El baseline de mayoría ("digo normal a todos") acierta 75% porque solo 25% son de riesgo
alto. Un número de accuracy sin ese baseline engaña. El detector llega a 90.4%: esa es la
mejora real.

### ¿Qué harías para llevarlo a producción?

Monitorear *drift* en las variables de proceso, recalibrar el umbral de probabilidad según
el costo de inspección vs. el de una fuga, validar contra inspecciones reales (ILI,
cupones de corrosión) y versionar datos y modelo. Y mantener la salida explicable: la
cuadrilla tiene que poder defender cada marca.
