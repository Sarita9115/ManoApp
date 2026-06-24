# Gesture Recognition + Statistical Indicators

Sistema de dos partes para la actividad de reconocimiento de gestos en vivo.

## 1. Instalación

```bash
python3 -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Uso

### Parte 1 — Captura y logging en vivo

```bash
python gesture_logger.py
```

- Abre la webcam, detecta manos con MediaPipe, cuenta dedos extendidos (0-5),
  clasifica el gesto (`Fist`, `Open Palm`, `Thumbs Up`, `Unknown`) e identifica
  `Left`/`Right` (la etiqueta ya viene corregida respecto al efecto espejo de
  la cámara: si levantas tu mano derecha, se reporta `Right`).
- Detecta automáticamente la cara de quien está frente a la cámara y le asigna
  un `person_id` estable (`Person_1`, `Person_2`, ...). Si esa persona se aleja
  y vuelve a aparecer, el sistema la reconoce y reutiliza el mismo id en vez de
  crear uno nuevo (recuadro y etiqueta amarilla en el video).
- Cada detección se agrega en tiempo real a `gestures_log.csv` con las columnas:
  `timestamp, requestid, person_id, handedness, user_count, gesture_name`.
- Cambia la constante `STATION_ID` dentro de `gesture_logger.py` para identificar
  tu equipo/estación (ej. `"Station_B"`).
- Muestra un mini-dashboard en vivo sobre el video (persona actual, personas
  distintas vistas, gesto más común, % de Thumbs Up, conteo por mano).
  Presiona `q` para salir.

> Nota sobre identificación de persona: usa un método ligero (cara detectada
> con Haar cascade de OpenCV + comparación de similitud de la imagen de la
> cara, sin librerías pesadas como `dlib`/`face_recognition`). Funciona mejor
> con una persona a la vez frente a cámara, buena iluminación y mirando de
> frente. Si dos personas se ven muy parecidas o hay poca luz, puede fallar el
> re-reconocimiento; ajusta `FACE_SIMILARITY_THRESHOLD` en `gesture_logger.py`
> si ves que crea demasiadas identidades nuevas o que confunde a dos personas.

### Parte 2 — Análisis estadístico

```bash
python analyze_gestures.py
```

Imprime en consola:

1. `df.describe()` — resumen general.
2. **Tendencia central**: gesto más común y promedio/moda de dedos levantados.
3. **Dispersión y proporciones**: desviación estándar de dedos, % de
   `Thumbs Up` sobre el total, y tabla de distribución de gestos por mano.
4. **Comparación entre estaciones/equipos** (agrupado por `requestid`).
5. **Comparación entre personas** (agrupado por `person_id`): total de
   detecciones, promedio de dedos, gesto más común, y primera/última vez que
   esa persona fue vista (útil para confirmar que alguien que se fue y volvió
   conservó su identificador).

Si quieres comparar varios equipos, junta los CSV de cada estación en un solo
archivo (mismas columnas) antes de correr `analyze_gestures.py`.
