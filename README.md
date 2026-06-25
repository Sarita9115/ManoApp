# Hand recognition program [Gesture Recognition + Statistical Indicators]

Team: KaSaJoGio - Hand recognition program

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

> Nota sobre identificación de persona: usa reconocimiento facial real con
> los modelos oficiales de OpenCV Zoo — **YuNet** (detección, `cv2.FaceDetectorYN`)
> + **SFace** (embedding de 128 numeros por cara, `cv2.FaceRecognizerSF`),
> descargados en `models/`. Compara por similitud coseno contra las caras ya
> vistas. Si dos personas se ven muy parecidas o hay poca luz, ajusta
> `FACE_SIMILARITY_THRESHOLD` en `gesture_logger.py` (más alto = más estricto).

### Parte 3 — Identidad por secuencia de gestos (sin depender de la cara)

```bash
python identity_sequence.py live
```

Esto resuelve el reto bonus del enunciado ("Gesture-Based Identity
Recognition"), pero **sin requerir que se vea la cara**: identifica a la
persona únicamente por la secuencia de gestos que hace con la mano.

**Modo `live` (recomendado para el demo en clase)**: la cámara queda
corriendo de forma continua. Cada persona hace su secuencia de gestos (ej.
"1 dedo, 2 dedos, 3 dedos" o "Thumbs Up, Open Palm, 1 dedo") y, en cuanto
deja de detectarse mano por `SEQUENCE_GAP_SECONDS` (≈1.8s, "se acabó su
turno"), el sistema cierra esa secuencia y la compara contra todas las que
ya conoce:

- Si el **patrón** (gesto + mano, en orden) ya existe — de esta sesión o de
  una anterior, porque las firmas se guardan en `identity_signatures.json` —
  reconoce a esa persona, sin importar cuántas otras secuencias distintas
  hayan pasado en medio, ni a qué velocidad la repita (el ritmo solo se
  reporta como dato informativo, no es obligatorio que coincida).
- Si el patrón es nuevo, lo registra automáticamente como una persona nueva
  (`Person_1`, `Person_2`, ...) para reconocerla la próxima vez.
- Cada gesto sin nombre clasificado (ej. 1, 2, 3, 4 dedos) se identifica por
  su número de dedos, así que secuencias numéricas como "1, 2, 3" funcionan
  igual que secuencias de gestos con nombre.

**Modos manuales** (útiles para pruebas controladas o demostrar el registro
explícito con nombre):

```bash
python identity_sequence.py enroll Ana
python identity_sequence.py enroll Beto
python identity_sequence.py verify
```

- `enroll <nombre>`: captura una secuencia fija de `SEQUENCE_LENGTH` pasos (4
  por defecto) y la guarda con ese nombre en `identity_signatures.json`
  (mismo archivo que usa el modo `live`, así que ambos modos comparten
  galería de identidades).
- `verify`: captura una secuencia y la compara contra todas las firmas
  guardadas, exigiendo coincidencia exacta de patrón y reportando si el
  ritmo cae dentro de `TIMING_TOLERANCE` (0.18 por defecto).

Ajusta en `identity_sequence.py`: `SEQUENCE_GAP_SECONDS` (qué tan larga debe
ser la pausa para cerrar un turno en modo `live`), `MIN_SEQUENCE_STEPS`
(pasos mínimos para que cuente como secuencia real, evita registrar un solo
gesto suelto como persona nueva), y `TIMING_TOLERANCE` (margen de ritmo).

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

