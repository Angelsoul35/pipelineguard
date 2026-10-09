# 1. Corrosión y features de física

## Por qué se corroe un pozo por dentro

El modo de falla número uno en tuberías de petróleo y gas no es un golpe externo: es la
**corrosión interna**. El gas producido trae CO2. El CO2 por sí solo, como gas seco, casi
no ataca al acero. El problema empieza cuando hay **agua**: el CO2 se disuelve en el agua
producida y forma ácido carbónico,

```
CO2 + H2O  ->  H2CO3  (ácido carbónico)
```

y ese ácido ataca la pared de acero (la llamada *corrosión dulce* o *sweet corrosion*). Es
un proceso lento y electroquímico: la pared se adelgaza, aparece una picadura, y al final
una fuga o una pérdida de contención. Para cuando se ve la fuga, el daño ya ocurrió. Por
eso interesa **predecir el riesgo** a partir de las variables de proceso, antes de que
haya daño visible.

Tres cosas gobiernan la velocidad de corrosión:

1. **Cuánto CO2 se disuelve.** No depende del porcentaje molar de CO2 solo, sino de su
   **presión parcial** (ley de Henry): a más presión parcial, más CO2 disuelto, más ácido.
2. **Si hay fase de agua continua en la pared.** El agua es el electrolito. Sin agua
   mojando el acero, no hay corrosión acuosa. Esto es el **corte de agua** (water cut).
3. **El régimen de flujo y el esfuerzo cortante en la pared.** Mucho gas respecto al
   líquido cambia el patrón de flujo (hacia niebla/anular) y puede arrancar la capa
   protectora de carbonato de hierro (FeCO3) → *erosión-corrosión*.

Estas tres ideas son exactamente las que codifican las features del modelo.

## Los datos reales

Usamos `maisaian/oil-gas-leakage` de Hugging Face: **10 292 pozos**, cada uno con nueve
lecturas de proceso en estado estable. No es un generador sintético: son mediciones de
cabezal. Las columnas crudas son temperatura y presión de cabezal, caudales de gas
(MMCFD), petróleo (BOPD) y agua (BWPD), BSW (sedimento básico y agua, %), CO2 (% molar),
gravedad del gas y `CR-corrosion defect` (la velocidad de corrosión medida).

## El objetivo (target) y la fuga de datos

`CR-corrosion defect` es la **señal de integridad**: la velocidad de corrosión medida. En
vez de predecir su valor exacto (regresión, ruidosa en la cola alta), lo planteamos como
**detección**, que es la decisión operativa real: *¿marco este pozo para inspección o no?*

> **Definición del target.** Un pozo es `high_risk = 1` si su velocidad de corrosión está
> en el **peor cuartil de la flota**, es decir en el **percentil 75 o más** (≈ 0.2328).
> Es un umbral fijo, aplicado igual a todos los pozos. La tasa base es ~25%, así que el
> **baseline de mayoría** ("digo normal a todos") ya acierta 75%: ese es el piso a vencer.

> **Concepto — fuga de datos (leakage).** La columna de corrosión se usa **solo** para
> construir la etiqueta. Nunca es una feature, y ninguna feature es una transformación de
> ella. Todas las features salen de los ocho sensores de proceso. Así, la cantidad que
> *define* el target no puede colarse en las entradas que lo predicen. La prueba de que no
> hay fuga es el **control de etiquetas barajadas** (doc 2): si barajamos las etiquetas, el
> modelo cae al azar. Si hubiera fuga, "acertaría" igual.

## Las features con base física (con fórmulas)

En `features.py` convertimos los nueve números crudos en una representación centrada en la
química y la física del flujo. Cada feature es algo que un ingeniero de corrosión calcula a
mano.

### 1. Presión parcial de CO2 — `co2_partial_pressure_psi`

```
pCO2 = (CO2_mol% / 100) × presión_de_cabezal      [psi]
```

El verdadero motor de la corrosión dulce. No es el % molar de CO2 solo: es su presión
parcial la que fija cuánto CO2 se disuelve (Henry) y, por tanto, cuánto ácido carbónico se
forma. Es la variable central de las correlaciones clásicas de De Waard–Milliams para la
velocidad de corrosión por CO2. Al codificar el CO2 así, fundimos el % molar y la presión
en el único término con sentido físico.

### 2. Corte de agua — `water_cut`

```
water_cut = BWPD / (BOPD + BWPD)      [fracción, 0–1]
```

La fracción del líquido producido que es agua. El agua es el electrolito: sin una fase de
agua continua en la pared, la corrosión acuosa es mínima. Un corte de agua que sube es una
bandera roja clásica de integridad.

### 3. Líquido total — `total_liquid_bpd`

```
total_liquid = BOPD + BWPD      [bbl/día]
```

El caudal (throughput). Junto con el corte de agua, es simplemente una **reparametrización
física** de los caudales crudos de petróleo y agua: `BOPD = total_liquid × (1 − water_cut)`.
Pasamos de "cuánto petróleo y cuánta agua" a los dos ejes que gobiernan el flujo: *cuánto
fluido* y *qué tan mojado*.

### 4. Relación gas-petróleo (GOR) — `gas_oil_ratio`

```
GOR = gas_MMCFD × 1e6 / BOPD      [scf/bbl]
```

Un GOR alto empuja el régimen de flujo hacia niebla/anular y sube el esfuerzo cortante en
la pared, que puede arrancar la capa protectora de FeCO3 y acelerar la erosión-corrosión.

### 5. Residual de balance de agua — `water_balance_residual`

```
residual = BSW/100 − water_cut
```

Una **comprobación de balance de masa**. El BSW (medido) y el corte de agua (calculado de
los caudales) son dos estimaciones independientes de la *misma* fracción de agua, así que
deberían coincidir. Un residual grande significa que el balance de agua **no cierra**: una
deriva de medidor o una anomalía de separación — justo el tipo de upset de proceso que
precede a un evento de integridad. Por eso lo llevamos como feature/bandera propia.

> Dato real e interesante: en este dataset el BSW y el corte de agua de los caudales están
> casi **sin correlación** entre sí. Es decir, las columnas se midieron/generaron de forma
> independiente. Lejos de ser un problema, hace del residual una señal legítima y no
> trivial, y es un ejemplo de rigor: inspeccionar los datos antes de confiar en ellos.

### Features crudas que se quedan

No todo se transforma. Se mantienen como features los sensores que aportan señal
**ortogonal**: la **temperatura de cabezal** (cinética de la reacción), la **presión de
operación** (actúa por varios canales, no solo vía pCO2: régimen de flujo, condiciones in
situ), el **caudal de gas** y la **gravedad del gas** (composición de la fase gaseosa).

En total: **10 features** (5 crudas + 5 derivadas de física). Ninguna toca la lectura de
corrosión. Así se ve el diseño: no duplicamos una columna cruda y su función exacta a la
vez (el CO2 va solo como presión parcial; petróleo y agua van como corte + caudal), pero
sí conservamos la presión cruda porque lleva información que el término pCO2 no captura.
