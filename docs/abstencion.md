# Abstención exploratoria en ShiftLab

Con el umbral fijado se rechazan seis errores en cada condición, junto con
7, 8 y 12 aciertos, respectivamente. Entre las imágenes aceptadas siguen
quedando 2, 6 y 23 errores. **El análisis es exploratorio:** la validación
original ya se utilizó durante el desarrollo de ShiftLab; subdividirla después
no aporta confirmación independiente.

La logística original (`StandardScaler` y
`LogisticRegression(C=1, solver=lbfgs, max_iter=2000)`) se ajustó solo con las
1078 imágenes limpias del train original. `max(predict_proba)` es una puntuación
para seleccionar imágenes, no una probabilidad calibrada de acertar.

Las 359 imágenes de validación se dividieron en selección (180) y medición
(179), mediante `StratifiedGroupKFold(n_splits=2, shuffle=True,
random_state=20261111)`. Se separaron los grupos de duplicados exactos por
SHA256; todas las versiones de cada imagen conservan su subconjunto.

El umbral conservado es **`t = 0.6735979689840444`**. Se eligió el mayor `t`
que aceptaba al menos `ceil(0.95 × 180) = 171` imágenes limpias de selección,
con la regla `score >= t`, incluidos todos los empates. La elección usó solo
esas puntuaciones, sin etiquetas ni errores. El **95 % se obtuvo en selección
limpia (171/180)**; no es una garantía de cobertura en medición ni con ruido.
El umbral quedó fijado antes de medir la segunda mitad y se mantuvo en las
tres condiciones.

Resultados sobre las mismas 179 imágenes de medición, usando las
transformaciones originales del proyecto:

| Condición | Aceptadas | Rechazadas | Errores rechazados | Aciertos rechazados | Errores aceptados | Cobertura | Tasa de error entre aceptadas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Limpio | 166 | 13 | 6 | 7 | 2 | 92,74 % | 1,20 % |
| Ruido σ=0,05 | 165 | 14 | 6 | 8 | 6 | 92,18 % | 3,64 % |
| Ruido σ=0,10 | 161 | 18 | 6 | 12 | 23 | 89,94 % | 14,29 % |

Los recuentos proceden del [JSON conservado](../artifacts/abstention-20261111/results.json).
Los aciertos rechazados se calculan como `rejected - errors_rejected`.
Los errores de clasificación obligatoria se conservan al sumar los rechazados
y los que siguen entre aceptadas: `8 = 6 + 2`, `12 = 6 + 6` y `29 = 6 + 23`.
Los errores rechazados son los que habría cometido el clasificador si hubiera
respondido. Con ruido σ=0,10, la tasa solo baja del 16,20 % al 14,29 %, a costa
de rechazar también 12 aciertos. Si no se acepta ninguna imagen, la tasa queda
sin definir (`null` en JSON).

La [gráfica existente de cobertura frente a tasa de error](../artifacts/abstention-20261111/cobertura-error.html)
marca el umbral fijado con estrellas y la clasificación obligatoria con
círculos. Las curvas son descriptivas y no se usan para escoger otro umbral.
El HTML autónomo se incluye junto al documento: para verlo de forma
interactiva, abrirlo en un navegador desde una copia local del repositorio.
El [informe generado](../artifacts/abstention-20261111/informe.txt) conserva
las seis filas de comparación completas.

Para reproducir desde la raíz, con el entorno original y una carpeta de
salida nueva:

```bash
.venv/bin/python -m experiments.abstention --prepared artifacts/prepared.json --output artifacts/abstention-reproduccion-01
```

El experimento utiliza train y validación originales; no evalúa el test final.
