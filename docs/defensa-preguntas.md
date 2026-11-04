# Seis preguntas para defender ShiftLab

Contesta inspeccionando los resultados y el código. Las respuestas están separadas en [defensa-respuestas.md](defensa-respuestas.md).

1. Dos filas contienen la misma imagen, pero etiquetas distintas. ¿Qué hace la partición, qué problema evita y qué problema permanece? ¿Podemos afirmar que los escritores de test son nuevos?
2. El candidato aumentado usa cuatro versiones de cada imagen. ¿Por qué cada una pesa 1/4? ¿Dónde deben llegar esos pesos y qué comprobarías para demostrar que no solo se almacenan en el informe?
3. En ruido gaussiano, ¿por qué se comparte un patrón base entre niveles y se exige exactamente la misma entrada para todos los candidatos? ¿Qué representan acierto→error y error→acierto?
4. La aumentación mejora una condición y empeora otra. ¿Es un fallo de la implementación? ¿Podrías escoger después la transformación que da el mejor test y seguir llamándolo evaluación final independiente?
5. ¿Por qué macro-F1 puede cambiar de forma distinta a accuracy? ¿Se puede promediar una diferencia individual por imagen para obtener el delta de macro-F1? ¿Por qué un margen SVM de 2 no significa 200 % de confianza?
6. ¿Qué explica el salto 22,22 % → 98,89 % con ruido σ=0,05? ¿Qué cambian el escalador, el fondo, las cuatro versiones y `gamma=scale`? ¿Qué experimento faltaría para aislar esas causas?
