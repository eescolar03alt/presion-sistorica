# Estimador de presión sistólica

Prototipo educativo en Python, sin dependencias externas. Estima `ap_hi` (mmHg) usando edad, código de género, estatura, peso, categorías de colesterol y glucosa, tabaquismo, alcohol y actividad física. No sustituye un tensiómetro ni se debe usar para diagnosticar o decidir tratamientos.

## Requisitos

- Python 3.10 o superior (probado en 3.10, 3.12 y 3.13). No hay dependencias externas: no hace falta `pip`, ni un entorno virtual.
- `sistema.py`, `index.html` y `cardio_train.csv` en esta carpeta. `modelo.json` se genera solo si falta.

## Cómo ejecutarlo

1. Abre una terminal en esta carpeta:

   ```powershell
   cd "C:\ruta\a\presion_sist"
   ```

2. Inicia el servidor (en Linux o macOS usa `python3`):

   ```powershell
   python sistema.py
   ```

3. La terminal muestra la dirección; si es la primera vez, antes entrena el modelo (unos 5 segundos):

   ```
   Servidor en marcha: abre http://127.0.0.1:8000 en el navegador.
   Deja esta ventana abierta mientras lo usas; Ctrl+C para detener.
   ```

4. Abre <http://127.0.0.1:8000> en el navegador.

5. Para detenerlo, pulsa `Ctrl+C` en esa terminal.

La página solo funciona con el servidor en marcha: si abres `index.html` con doble clic, o si cierras la terminal, el formulario mostrará que no puede conectar. El servidor escucha únicamente en `127.0.0.1`, así que debe abrirse esa dirección y no el archivo.

### Opciones

| Comando | Qué hace |
| --- | --- |
| `python sistema.py` | Entrena solo si falta `modelo.json` y sirve la página en el puerto 8000 |
| `python sistema.py --puerto 8080` | Usa otro puerto (1024–65535); abre la dirección que imprima |
| `python sistema.py --entrenar` | Reentrena con el CSV actual antes de servir |
| `python sistema.py --entrenar --solo-entrenar` | Entrena y termina, sin iniciar el servidor |
| `python test_sistema.py` | Ejecuta las 9 comprobaciones (también `python -m unittest test_sistema`) |

### Problemas comunes

| Síntoma | Solución |
| --- | --- |
| «No se pudo conectar con el servidor» | El servidor no está en ejecución o abriste el archivo directamente. Vuelve a ejecutar `python sistema.py` y entra a <http://127.0.0.1:8000> |
| «El puerto 8000 está ocupado» | Quedó otra instancia abierta: ciérrala con `Ctrl+C` o inicia con `--puerto 8080` |
| `python` no se reconoce como comando | Instala Python desde <https://www.python.org/downloads/> marcando «Add Python to PATH», o usa `py sistema.py` en Windows |
| «No hay valores válidos en la columna…» | Falta `cardio_train.csv` en esta carpeta o el archivo está dañado |
| El navegador muestra una versión vieja | Recarga con `Ctrl+F5`; el servidor lee `index.html` en cada visita |

## La interfaz

El botón **Probar con un registro aleatorio** rellena el formulario con una fila depurada del dataset y permite comparar la estimación con su presión registrada. Puede seleccionar tanto datos de entrenamiento como de prueba: esta comparación individual es ilustrativa, no una evaluación independiente. Si editas un campo, se elimina la comparación con el registro original. El archivo `modelo.json` contiene parámetros y métricas, no los registros individuales.

## Cómo funciona

El archivo utiliza `;` como separador y la edad está en días; el formulario recibe años. La depuración ocurre al leer el CSV, que permanece intacto: se eliminan las filas duplicadas (idénticas en las 12 columnas, el identificador no cuenta; son 24 en esta copia) y cada valor faltante o fuera de rango se sustituye por la mediana de su columna. Los rangos son 18–100 años, estatura 120–220 cm, peso 30–250 kg, categorías 1–3, binarias 0–1 y presión sistólica de 70 a 250 mmHg; también se imputa una sistólica menor o igual que la diastólica cuando la diastólica es creíble. Son decisiones prácticas, no criterios clínicos: imputar la presión de una fila no la convierte en una medición, y por eso el registro aleatorio solo usa filas con presión medida. Se divide aleatoriamente 80 % para entrenamiento y 20 % para prueba, con semilla 42. Se ajusta una regresión lineal con regularización; la estandarización se calcula solo con entrenamiento. El error absoluto medio (MAE) compara predicciones con valores reales y se contrasta con predecir siempre la media de entrenamiento; también se reporta R² y el error de cada intervalo de 20 mmHg, porque el modelo acierta cerca de la media y falla en los extremos. Se excluyen identificador, presión diastólica y etiqueta cardiovascular para evitar depender de mediciones o resultados adicionales que podrían no estar disponibles al predecir.

La interfaz muestra nombres en lugar de códigos: género 1 = mujer y 2 = hombre; colesterol y glucosa 1 = normal, 2 = por encima de lo normal y 3 = muy por encima de lo normal. Conserva los valores originales al enviarlos al modelo. Esta interpretación sigue la [descripción del conjunto de datos](https://github.com/Arilli2004/Cardiovascular-Disease-Prediction/blob/main/docs/dataset.md); el CSV no contiene su propio diccionario, por lo que conviene confirmar la procedencia de esta copia. La partición aleatoria no comprueba rendimiento en otras poblaciones o centros, ni equidad, ni valida el sistema para atención médica. Las relaciones son estadísticas y no prueban causalidad. Las estimaciones individuales pueden desviarse mucho del valor medido, sobre todo fuera del rango de 110–150 mmHg.
