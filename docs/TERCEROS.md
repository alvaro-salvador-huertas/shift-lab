# Procedencia, licencias y avisos

Revisado el 4 de noviembre de 2026. El titular ha autorizado [MIT](../LICENSE)
para el código propio: Copyright (c) 2026 Álvaro Salvador Huertas.
El texto estándar procede de [Choose a License](https://choosealicense.com/licenses/mit/).
La licencia propia no reasigna autoría ni modifica las condiciones de los datos,
dependencias o recursos de terceros descritos a continuación.

## Datos e imágenes derivadas

Alpaydin, E. y Kaynak, C. (1998), *Optical Recognition of Handwritten Digits*,
UCI Machine Learning Repository, [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49).
[UCI declara CC BY 4.0](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits);
[texto y condiciones de CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Se mantiene esta atribución para los datos redistribuidos en NPZ, cuadrículas,
interfaz y capturas. Modificaciones de ShiftLab: división de intensidades por
16, partición interna y derivados sintéticos de ruido, desenfoque y contraste;
las etiquetas originales se conservan. No se implica respaldo de los autores.

Las características originales son recuentos de píxeles activos en bloques de
4×4 de bitmaps de 32×32, resumidos en 8×8, según UCI. Cada muestra representa
un dígito. [load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
redistribuye las 1797 muestras del test original; no proporciona IDs de escritor
para verificar aquí una separación por personas. No se mide identidad,
capacidad ni ningún atributo personal del escritor.

## Bibliotecas y gráficos

Los requisitos fijan las dependencias, que conservan sus propias licencias y
avisos en sus distribuciones. No se copia el entorno Python como parte de la
entrega. `requirements.txt` fija las versiones y los paquetes instalados
conservan sus avisos. El inventario detallado de revisión se guarda localmente
y se excluye de la publicación; no sustituye una auditoría jurídica exhaustiva.

Los HTML autónomos contienen plotly.js v4.1.1 e identifican Copyright
2012–2026, Plotly, Inc., licencia MIT. Se conservan sus cabeceras y los avisos
incluidos en los bundles originales, además del aviso distribuido por el
paquete Plotly en [plotly-LICENSE.txt](terceros/plotly-LICENSE.txt).
No se han regenerado ni modificado los HTML históricos. Las capturas documentan
la aplicación y contienen derivados del dataset con la atribución anterior.

La atribución del dataset y los avisos de los bundles se conservan en el
contenido distribuido. No se infiere la licencia del código a partir de la
licencia de los datos. `pyproject.toml` se conserva sin cambios porque integra
la identidad del experimento archivado; la licencia propia está en LICENSE.
