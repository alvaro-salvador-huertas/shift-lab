# Protocolo experimental de ShiftLab

Escrito antes de ajustar modelos o consultar métricas de test. La congelación
posterior al desarrollo con validación es una medida de reproducibilidad, no
una preregistración independiente. No se harán búsquedas de hiperparámetros.

## Pregunta y datos

¿Cómo cambia el rendimiento de un clasificador cuando las imágenes se degradan,
y qué cambia al entrenarlo con aumentación? Usamos `load_digits`, 1797 imágenes
reales de dígitos manuscritos de 8×8, con intensidad original 0…16 dividida por 16.
La normalización es fija y no aprende estadísticas del dataset.

Este loader redistribuye el test original del conjunto de UCI, de Alpaydin y
Kaynak (1998), DOI `10.24432/C50P49`. UCI indica licencia **CC BY 4.0**: conservar
atribución e indicar que aquí normalizamos y degradamos las imágenes. No se
descargan datos durante el uso; vienen incluidos con scikit-learn. La nueva
partición es interna a estas 1797 imágenes; no se dispone de identificadores de
escritor utilizables y no se estima rendimiento sobre escritores nuevos.

## Partición y duplicados

Cada imagen original se representa mediante los 64 píxeles enteros `uint8` en
orden C y se agrupa por su SHA256, sin incluir la etiqueta. `StratifiedGroupKFold`
con cinco folds, `shuffle=True` y semilla **20261029** asigna el fold 0 a test,
el 1 a validación y los folds 2, 3 y 4 a train, aproximadamente 60/20/20.

`artifacts/prepared.json` conserva todos los índices, la asignación de folds,
grupos, tamaños, distribuciones de clases y reporte de duplicados. Los posibles
duplicados con etiquetas contradictorias se registran y se conservan, agrupados
juntos. Nunca se eliminan según su efecto en una puntuación. La separación por
hash evita duplicados exactos entre particiones, no dependencias más generales
entre estilos de escritura.

## Modelos fijos

Se comparan cinco candidatos: DummyClassifier `most_frequent`; StandardScaler
seguido de LogisticRegression con `C=1`, `solver=lbfgs`, `max_iter=2000`, limpio
y aumentado; y StandardScaler seguido de SVC con `kernel=rbf`, `C=1`,
`gamma=scale`, `probability=False`, limpio y aumentado.

El escalador se ajusta exclusivamente con el entrenamiento de su candidato.
Se pasan explícitamente `scaler__sample_weight` y `classifier__sample_weight`
al Pipeline. Las APIs instaladas admiten estos pesos. El registro conserva
inputs, pesos, relación original/derivado, estadísticas del escalador y avisos.
Los avisos también se emiten a stderr; no se ocultan avisos de convergencia.
No se transforman márgenes SVM en porcentajes ni se afirman probabilidades
calibradas. La aplicación utiliza etiquetas y predicciones, sin confianza.

## Aumentación y degradaciones

Por cada original de train se conservan cuatro versiones: original, ruido
gaussiano σ=0,05, desenfoque σ=0,45 y contraste c=0,70. Cada versión pesa **1/4**,
incluido el original; el peso total por original sigue siendo 1. Train limpio
conserva una versión con peso 1. Validación y test no aportan derivados a train.

La evaluación usa las diez condiciones, fijadas ahora:

| Condición | Niveles |
| --- | --- |
| Limpio | Original normalizado |
| Ruido gaussiano | σ 0,025; 0,05; 0,10, con clipping a [0,1] |
| Desenfoque gaussiano | σ 0,25; 0,45; 0,65 píxeles |
| Contraste a gris medio | c 0,90; 0,70; 0,50 mediante c·x+(1−c)·0,5 |

El desenfoque usa `scipy.ndimage.gaussian_filter`, `mode=reflect`, `truncate=4.0`,
sigma `(0,sigma,sigma)`: nunca mezcla imágenes a lo largo del eje del lote.

Streams separados: partición 20261029; aumentación 20261030; evaluación
20261031. El ruido base de cada imagen se crea con
`SeedSequence([stream_seed, original_row_id])`. Se reutiliza el mismo patrón
base entre intensidades y candidatos. El clipping puede hacer que el cambio
observado deje de ser lineal en los píxeles de borde. El orden de las imágenes
o seleccionar una sola imagen no cambia su ruido.

Antes de congelar se inspeccionará `artifacts/training-transformations.png`:
una imagen de cada clase, tomada exclusivamente de train, en las diez condiciones.
La cuadrícula utiliza el stream de evaluación sobre filas de train; no contiene
imágenes ni predicciones de test.

## Desarrollo, coste y congelación

Orden explícito de CLI: `prepare`, `cost`, `validation`, `freeze`, `final`.
El coste ajusta los cinco candidatos con todo train y predice solo validación
limpia. Se usa un margen conservador de ×20 para estimar el coste final.
Si rebasa 600 s se detiene y cualquier cambio de diseño se registra antes de
test; no se eligen configuraciones según su puntuación. Hilos numéricos: 1.

La validación ejecuta los cinco candidatos y diez escenarios, sin acceder a
predicciones de test. Tras revisar la cuadrícula y la interfaz con validación,
`freeze --visual-reviewed --review-note '…'` guarda configuración, identidades de
datos/particiones, versiones, hashes de todos los módulos del motor, requisitos
y este protocolo; también conserva sus bytes en `artifacts/frozen/source/`.

Solo después, `final` ejecuta cinco ajustes y cincuenta evaluaciones sobre test.
El presupuesto de la fase final es **600 s** y el comando se ejecutará además
con un timeout externo. Los archivos se abren en modo exclusivo: no se
sobrescriben resultados, congelaciones ni cuadrículas. Una corrección posterior
a test requiere carpeta nueva, conservar resultados antiguos y documentar la
reutilización; no sería una confirmación independiente.

## Métricas e interpretación

Accuracy y macro-F1 con etiquetas 0…9 y `zero_division=0`; se conservan matrices
10×10, métricas por clase y predicción por imagen. Macro-F1 promedia el F1 de
las diez clases, incluido 0 para clases ausentes en fixtures pequeños. Las
matrices usan filas reales y columnas predichas, siempre en orden 0…9.

Cada candidato recibe exactamente los mismos tensores de evaluación, cuyos
hashes se conservan. Las diferencias se emparejan por imagen original entre
limpio y cada condición: acierto→error, error→acierto, ambos aciertos, ambos
errores, cambio individual de acierto y diferencias globales de accuracy/F1.
Macro-F1 no es la media de cambios individuales; se recalcula del vector de
predicciones de cada condición. Las versiones de una imagen no son muestras
independientes. No se exige degradación monótona ni que la aumentación ayude.

Son degradaciones sintéticas de un dataset real pequeño. No representan toda
la variabilidad de cámaras, iluminación o escritura, ni robustez en producción.
Los resultados de validación se identifican como desarrollo y los de test como
evaluación final del protocolo congelado; no se mezclan en las gráficas.

## Referencias primarias

- [UCI y condiciones de uso](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).
- [load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
- [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- [StandardScaler y pesos](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
- [SVC y pesos](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
- [gaussian_filter y bordes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html).

Consultadas el 29 de octubre de 2026. Los requisitos registran las versiones
realmente instaladas; las páginas `stable` pueden cambiar.
