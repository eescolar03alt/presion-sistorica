# Estimador de presión sistólica

Prototipo educativo en Python, sin dependencias externas. Estima `ap_hi` (mmHg) usando edad, código de género, estatura, peso, categorías de colesterol y glucosa, tabaquismo, alcohol y actividad física. No sustituye un tensiómetro ni se debe usar para diagnosticar o decidir tratamientos.

## Iniciar

Con Python 3 instalado y `cardio_train.csv` en esta carpeta:

```bash
python3 sistema.py
```

Abre http://127.0.0.1:8000 en el navegador. El botón **Cargar registro aleatorio** rellena el formulario con una fila válida del dataset y permite comparar la estimación con su presión registrada. Puede seleccionar tanto datos de entrenamiento como de prueba: esta comparación individual es ilustrativa, no una evaluación independiente. Si editas un campo, se elimina la comparación con el registro original. La primera vez se entrena el modelo; detén el servidor con Ctrl+C. Para volver a entrenar con el CSV actual usa `python3 sistema.py --entrenar`; para entrenar sin iniciar el servidor, `python3 sistema.py --entrenar --solo-entrenar`. El archivo `modelo.json` contiene parámetros y métricas, no los registros individuales.

## Cómo funciona

El archivo utiliza `;` como separador y la edad está en días; el formulario recibe años. Se eliminan registros con valores faltantes o fuera de los rangos del código (incluida presión sistólica de 70 a 250 mmHg); estos filtros son decisiones prácticas, no criterios clínicos ni garantía de calidad. Se divide aleatoriamente 80 % para entrenamiento y 20 % para prueba, con semilla 42. Se ajusta una regresión lineal con regularización; la estandarización se calcula solo con entrenamiento. El error absoluto medio (MAE) compara predicciones con valores reales y se contrasta con predecir siempre la media de entrenamiento. Se excluyen identificador, presión diastólica y etiqueta cardiovascular para evitar depender de mediciones o resultados adicionales que podrían no estar disponibles al predecir.

La interfaz muestra nombres en lugar de códigos: género 1 = mujer y 2 = hombre; colesterol y glucosa 1 = normal, 2 = por encima de lo normal y 3 = muy por encima de lo normal. Conserva los valores originales al enviarlos al modelo. Esta interpretación sigue la [descripción del conjunto de datos](https://github.com/Arilli2004/Cardiovascular-Disease-Prediction/blob/main/docs/dataset.md); el CSV no contiene su propio diccionario, por lo que conviene confirmar la procedencia de esta copia. La partición aleatoria no comprueba rendimiento en otras poblaciones o centros, ni equidad, ni valida el sistema para atención médica. Las relaciones son estadísticas y no prueban causalidad. Las estimaciones individuales pueden desviarse mucho del valor medido.
