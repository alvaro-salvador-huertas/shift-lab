# Ficha de ShiftLab

**Uso previsto.** Aprender a construir una evaluación reproducible de robustez y examinar predicciones individuales. Compara cinco candidatos y diez condiciones fijadas de antemano. Sirve para explicar particiones, aumentación ponderada y degradaciones controladas; no para seleccionar un producto de reconocimiento de escritura.

**Datos.** Imágenes reales de dígitos, 8×8 píxeles y diez clases, incluidas en `sklearn.datasets.load_digits`. Las intensidades se dividen por 16. Ruido, desenfoque y cambio de contraste son transformaciones sintéticas. Se mantiene la relación con la imagen original y el mismo tensor de evaluación para todos los candidatos.

**Procedencia y condiciones.** Alpaydin, E. y Kaynak, C. (1998), *Optical Recognition of Handwritten Digits*, UCI Machine Learning Repository, DOI [10.24432/C50P49](https://doi.org/10.24432/C50P49). UCI indica [CC BY 4.0](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits). Aquí se normalizan intensidades y se producen degradaciones sintéticas; se conserva esta atribución. `load_digits` contiene el test original de UCI; la separación train/validación/test de ShiftLab se hace dentro de ese subconjunto, no reproduce la división original por escritores. [Descripción del loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

**Modelos.** Baseline mayoritario, regresión logística y SVM RBF; las dos últimas se ajustan con train limpio o aumentado. Los escaladores reciben solo el entrenamiento correspondiente y sus pesos. La aumentación conserva original y tres derivados, cada uno con peso 1/4: cada original conserva peso total 1.

**Qué se mide.** Accuracy, macro-F1, matrices 10×10, métricas por clase y predicciones. Las diferencias limpia/degradada se calculan sobre las mismas imágenes. Una mejora, un empate o una curva no monótona se conserva tal como resulta.

**Límites.** El hash agrupa duplicados exactos, no escritores. No se dispone de identificadores de autor para demostrar generalización a personas nuevas. Estos resultados sobre imágenes pequeñas y degradaciones elegidas no acreditan robustez en producción, ante ataques ni en otras cámaras. No se presentan márgenes de SVM como probabilidades o confianza calibrada. No hay intervalos de confianza ni réplicas independientes de cada versión de una imagen.

**Operación.** CPU local, sin cuentas ni llamadas a servicios externos. La interfaz carga archivos y reconstruye imágenes; no entrena al cambiar controles. Los archivos contienen código y versiones identificables, índices, pesos, predicciones y hashes de entrada. La congelación se realiza después del desarrollo con validación y no equivale a una preregistración independiente.

**Desarrollo.** Implementación y revisión asistidas por IA. El proyecto proporciona material verificable para aprender; no acredita por sí mismo experiencia profesional del autor.

**Lectura metodológica posterior (4 de noviembre).** La unidad es una imagen de un dígito, no una persona. El ruido σ se expresa en intensidad normalizada [0,1], se suma después de dividir por 16 y antes de StandardScaler, y se recorta a [0,1]. Se modifica todo el cuadro, incluido el fondo negro: en cero los valores negativos se recortan y los positivos elevan el fondo, por lo que el ruido efectivo ya no tiene media cero. El contraste c=0,70 también eleva un fondo cero a 0,15. El desenfoque usa σ en píxeles, una unidad distinta. No hay máscara del trazo ni selección de características.

El contraste limpio/aumentado compara 1078 filas frente a 4312 derivadas de los mismos 1078 originales, con peso total 1078 en ambos casos. Cambian simultáneamente distribución, escalado, estimador y gamma efectivo de la SVM. Una sola partición y una sola realización base de ruido por imagen (compartida entre niveles) no permiten estimar variación entre particiones o semillas; 50 combinaciones no son 50 réplicas independientes. El salto destacado está verificado aritméticamente, pero no demuestra robustez general ni un efecto causal aislado de la aumentación por ruido. Véanse las [explicaciones de defensa](defensa-respuestas.md).

**Publicación.** Código propio bajo [MIT](../LICENSE); los recursos de terceros
mantienen sus condiciones y [avisos](TERCEROS.md). La instalación limpia con
los pins originales está comprobada; la CI ejecuta los tests sin reevaluar el
test final. Se distribuye una aplicación local y resultados archivados, sin
despliegue cloud ni afirmación de robustez general.
