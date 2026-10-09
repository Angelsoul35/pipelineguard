# Guía de estudio — PipelineGuard

Esta guía está en español y explica el proyecto para que lo estudies y lo sustentes en una
entrevista. El código y la documentación técnica están en inglés; aquí va el *porqué*: la
química de la corrosión, las features de física, el modelo explicable y cómo se demuestra
que no hay sobreajuste ni fuga de datos.

## Orden de lectura

1. [Corrosión y features de física](01-corrosion-y-features.md) — por qué un pozo se
   corroe por dentro, qué es la presión parcial de CO2 y el corte de agua, y cómo cada
   feature derivada codifica esa física (con fórmulas).
2. [El modelo explicable y las métricas](02-modelo-y-metricas.md) — por qué un ensamble
   de árboles, cómo se mide (ROC-AUC, F1), la detección de sobreajuste con la brecha
   train/held-out, el ejemplo resuelto con los números reales y la prueba de control.
3. [Preguntas de entrevista](03-preguntas-de-entrevista.md) — preguntas típicas con
   respuestas.

## Cómo correrlo mientras estudias

```bash
pip install -e ".[dev]"
python -m pipelineguard.cli demo        # entrena, puntúa y explica sobre datos reales
python -m pipelineguard.real_benchmark  # el benchmark completo contra el sobreajuste
pytest                                  # 30 pruebas
```

El concepto central: **detectar qué pozos tienen riesgo alto de corrosión interna** a
partir de las lecturas de proceso que el cabezal ya reporta, con un modelo **explicable**
(dice de qué variables depende y en qué dirección) y **con base física** (sus features son
las cantidades que un ingeniero de confiabilidad ya calcula a mano). En un dominio de
seguridad, un indicador explicable se usa; una caja negra se ignora.
