"""
Parte 1: Detector de gestos en tiempo real + logging a CSV.

Usa la webcam para detectar manos con MediaPipe, cuenta dedos extendidos,
clasifica un gesto basico y guarda cada deteccion en un CSV en tiempo real.
Tambien dibuja un mini-dashboard en vivo (no solo numeros crudos) con los
indicadores que se piden en la actividad: tendencia central, proporcion
y comparacion por mano.

Ademas, detecta automaticamente la cara de la persona frente a la camara y le
asigna un identificador estable (Person_1, Person_2, ...). Si esa persona se
va y luego regresa, el sistema la vuelve a reconocer y reutiliza el mismo
identificador en vez de crear uno nuevo (conexion directa con el reto de
"Gesture-Based Identity Recognition" del enunciado, version simplificada
basada en similitud facial en lugar de una secuencia completa de gestos).

Controles:
    q -> salir
"""

import csv
import os
from collections import Counter
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

# ----------------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------------
STATION_ID = "Station_A"  # Identificador hardcodeado del equipo/estacion
CSV_PATH = "gestures_log.csv"
CSV_COLUMNS = [
    "timestamp",
    "requestid",
    "person_id",
    "handedness",
    "user_count",
    "gesture_name",
]

# Identificacion de persona por similitud facial.
FACE_EMBED_SIZE = (64, 64)
FACE_SIMILARITY_THRESHOLD = 0.80  # mas alto = mas estricto para considerar "misma persona"
FACE_SWITCH_CONFIRM_FRAMES = 5    # frames consecutivos necesarios para confirmar un cambio de persona

# Puntas de dedo (tip) y su articulacion intermedia (pip) segun MediaPipe Hands.
# El orden es: indice, medio, anular, menique. El pulgar se trata aparte porque
# su movimiento es lateral (x) y no vertical (y) como los demas dedos.
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
WRIST = 0
PINKY_MCP = 17


def ensure_csv_header(path: str, columns: list[str]) -> None:
    """Crea el CSV con encabezado si todavia no existe."""
    if not os.path.exists(path):
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(columns)


def _dist(a, b) -> float:
    """Distancia euclidiana 2D entre dos landmarks normalizados."""
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def count_extended_fingers(landmarks) -> tuple[int, bool]:
    """
    Cuenta cuantos dedos estan extendidos (0-5) usando distancias entre
    landmarks en vez de comparar solo el eje y. Esto es mucho mas robusto
    cuando la mano esta inclinada o rotada frente a la camara, que es la
    causa principal de conteos erroneos con la heuristica "tip.y < pip.y".

    Devuelve (cantidad_total, pulgar_extendido) porque el pulgar se necesita
    aparte para distinguir el gesto "Thumbs Up".
    """
    count = 0
    wrist = landmarks[WRIST]

    # Indice, medio, anular, menique: un dedo extendido aleja su punta (tip)
    # de la muñeca mas de lo que la aleja su articulacion media (pip),
    # sin importar si la mano esta derecha, inclinada o rotada en el plano
    # de la imagen.
    for tip_idx, pip_idx in zip(FINGER_TIPS, FINGER_PIPS):
        tip_dist = _dist(landmarks[tip_idx], wrist)
        pip_dist = _dist(landmarks[pip_idx], wrist)
        if tip_dist > pip_dist * 1.1:  # margen para evitar falsos positivos por ruido
            count += 1

    # Pulgar: en vez de depender del handedness (eje x cambia de signo segun
    # la mano y se rompe si la etiqueta viene invertida), usamos una prueba
    # independiente de la mano: el pulgar esta extendido si su punta se aleja
    # de la base del menique (landmark 17) mas que su articulacion IP.
    pinky_mcp = landmarks[PINKY_MCP]
    thumb_tip_dist = _dist(landmarks[THUMB_TIP], pinky_mcp)
    thumb_ip_dist = _dist(landmarks[THUMB_IP], pinky_mcp)
    thumb_extended = thumb_tip_dist > thumb_ip_dist * 1.1

    if thumb_extended:
        count += 1

    return count, thumb_extended


class PersonIdentifier:
    """
    Reconocimiento ligero de identidad basado en similitud facial, sin
    dependencias pesadas (dlib/face_recognition): usa el detector Haar de
    OpenCV para encontrar la cara y compara un "embedding" simple (imagen en
    escala de grises, ecualizada y normalizada) contra una galeria de
    personas ya vistas.

    Si la cara coincide con una persona conocida (similitud >= umbral),
    reutiliza su id. Si no coincide con nadie, registra una persona nueva.
    Un pequeno "debounce" evita que el id cambie por un solo frame ruidoso
    cuando dos personas se turnan frente a la camara.
    """

    def __init__(
        self,
        similarity_threshold: float = FACE_SIMILARITY_THRESHOLD,
        switch_confirm_frames: int = FACE_SWITCH_CONFIRM_FRAMES,
    ):
        self.similarity_threshold = similarity_threshold
        self.switch_confirm_frames = switch_confirm_frames

        self.known_ids: list[str] = []
        self.known_embeddings: list[np.ndarray] = []
        self.next_index = 1

        self.current_id: str | None = None
        self._pending_id: str | None = None
        self._pending_count = 0

    @staticmethod
    def _embed(face_gray: np.ndarray) -> np.ndarray:
        face = cv2.resize(face_gray, FACE_EMBED_SIZE)
        face = cv2.equalizeHist(face)
        vector = face.astype(np.float32).flatten()
        vector -= vector.mean()
        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            vector /= norm
        return vector

    def _register_new_person(self, embedding: np.ndarray) -> str:
        person_id = f"Person_{self.next_index}"
        self.next_index += 1
        self.known_ids.append(person_id)
        self.known_embeddings.append(embedding)
        return person_id

    def identify(self, face_gray: np.ndarray) -> str:
        """Devuelve el person_id (estable) para la cara recibida en este frame."""
        embedding = self._embed(face_gray)

        if not self.known_embeddings:
            candidate_id = self._register_new_person(embedding)
        else:
            similarities = [float(np.dot(embedding, known)) for known in self.known_embeddings]
            best_idx = int(np.argmax(similarities))
            if similarities[best_idx] >= self.similarity_threshold:
                candidate_id = self.known_ids[best_idx]
                # Promedio movil del embedding para tolerar cambios leves de
                # pose/iluminacion sin perder la identidad ya registrada.
                updated = 0.9 * self.known_embeddings[best_idx] + 0.1 * embedding
                norm = np.linalg.norm(updated)
                self.known_embeddings[best_idx] = updated / norm if norm > 1e-6 else updated
            else:
                candidate_id = self._register_new_person(embedding)

        return self._debounce(candidate_id)

    def _debounce(self, candidate_id: str) -> str:
        if self.current_id is None:
            self.current_id = candidate_id
            return self.current_id

        if candidate_id == self.current_id:
            self._pending_id = None
            self._pending_count = 0
            return self.current_id

        if candidate_id == self._pending_id:
            self._pending_count += 1
        else:
            self._pending_id = candidate_id
            self._pending_count = 1

        if self._pending_count >= self.switch_confirm_frames:
            self.current_id = candidate_id
            self._pending_id = None
            self._pending_count = 0

        return self.current_id


def classify_gesture(finger_count: int, thumb_extended: bool, landmarks) -> str:
    """
    Clasificacion simple basada en reglas (no ML) sobre el numero de dedos
    extendidos y la orientacion del pulgar.
    """
    if finger_count == 0:
        return "Fist"

    if finger_count == 5:
        return "Open Palm"

    # Thumbs Up: solo el pulgar extendido y, ademas, apuntando hacia arriba
    # (la punta del pulgar esta por encima de la muñeca en la imagen).
    if finger_count == 1 and thumb_extended:
        thumb_tip_y = landmarks[THUMB_TIP].y
        wrist_y = landmarks[WRIST].y
        if thumb_tip_y < wrist_y:
            return "Thumbs Up"

    return "Unknown"


def draw_dashboard(frame, session_log: list[dict], current_person_id: str | None) -> None:
    """
    Dibuja sobre el frame los indicadores en vivo de la sesion actual
    (no son datos crudos, son resumenes ya calculados):
      - Tendencia central: gesto mas comun.
      - Proporcion: % de "Thumbs Up" sobre el total.
      - Comparacion: conteo de detecciones por mano (Left vs Right).
      - Identidad: persona actual y cuantas personas distintas se han visto.
    """
    total = len(session_log)
    y0 = 30
    line_height = 28

    person_label = current_person_id or "Unknown"
    distinct_people = len({row["person_id"] for row in session_log}) if session_log else 0

    if total == 0:
        cv2.putText(frame, "Esperando detecciones...", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Persona actual: {person_label}", (10, y0 + line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return

    gestures = [row["gesture_name"] for row in session_log]
    hands = [row["handedness"] for row in session_log]

    most_common_gesture, mc_count = Counter(gestures).most_common(1)[0]
    thumbs_up_pct = 100.0 * gestures.count("Thumbs Up") / total
    hand_counts = Counter(hands)

    lines = [
        f"Total detecciones: {total}  (Estacion: {STATION_ID})",
        f"Persona actual: {person_label}  |  Personas distintas vistas: {distinct_people}",
        f"Gesto mas comun (tendencia central): {most_common_gesture} ({mc_count}/{total})",
        f"Proporcion 'Thumbs Up': {thumbs_up_pct:.1f}%",
        f"Comparacion por mano -> Left: {hand_counts.get('Left', 0)}  Right: {hand_counts.get('Right', 0)}",
    ]

    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, y0 + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main() -> None:
    ensure_csv_header(CSV_PATH, CSV_COLUMNS)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    person_identifier = PersonIdentifier()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam (indice 0).")

    session_log: list[dict] = []  # Solo para el dashboard en vivo de esta corrida
    current_person_id: str | None = None

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        max_num_hands=2,
    ) as hands, open(CSV_PATH, mode="a", newline="", encoding="utf-8") as csv_file:

        writer = csv.writer(csv_file)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Espejamos la imagen para que se sienta como un espejo natural
            # (el usuario se ve a si mismo como en un espejo de pared).
            frame = cv2.flip(frame, 1)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- Identificacion de persona (cara) ---------------------------
            # Se hace sobre el frame YA espejado, que es exactamente lo mismo
            # que ve la persona y lo mismo que procesamos para las manos, asi
            # que no hay inconsistencia de orientacion entre cara y manos.
            faces = face_detector.detectMultiScale(
                gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
            )
            if len(faces) > 0:
                # Si aparece mas de una cara, asumimos que la "activa" es la
                # mas grande (la mas cercana a la camara), ya que la actividad
                # es de una persona a la vez por estacion.
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                face_crop = gray_frame[fy:fy + fh, fx:fx + fw]
                current_person_id = person_identifier.identify(face_crop)
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 200, 255), 2)
                cv2.putText(frame, current_person_id, (fx, fy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            # Si no se detecta cara en este frame, conservamos el ultimo
            # current_person_id conocido (p. ej. la persona se inclino o giro
            # un poco) en vez de perder la identidad por un frame suelto.

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # MediaPipe documenta que su etiqueta de handedness ya
                    # esta pensada para una camara selfie (imagen espejada):
                    # asume que "Right" es la mano derecha tal como la ve la
                    # persona en el espejo. Como aqui SI espejamos el frame
                    # antes de procesarlo, la etiqueta cruda de MediaPipe ya
                    # coincide con la mano real de la persona, sin invertirla.
                    hand_label = handedness.classification[0].label  # "Left" / "Right"

                    landmarks = hand_landmarks.landmark
                    finger_count, thumb_extended = count_extended_fingers(landmarks)
                    gesture_name = classify_gesture(finger_count, thumb_extended, landmarks)

                    timestamp = datetime.now().isoformat()
                    row = {
                        "timestamp": timestamp,
                        "requestid": STATION_ID,
                        "person_id": current_person_id or "Unknown",
                        "handedness": hand_label,
                        "user_count": finger_count,
                        "gesture_name": gesture_name,
                    }

                    # Logging en tiempo real: se escribe y se vuelca a disco
                    # en cada frame con deteccion, no al final del programa.
                    writer.writerow([row[c] for c in CSV_COLUMNS])
                    csv_file.flush()

                    session_log.append(row)

                    # Texto junto a la mano detectada.
                    h, w, _ = frame.shape
                    x = int(landmarks[WRIST].x * w)
                    y = int(landmarks[WRIST].y * h)
                    label_text = f"{hand_label} | {finger_count} dedos | {gesture_name}"
                    cv2.putText(frame, label_text, (x - 60, y + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            draw_dashboard(frame, session_log, current_person_id)
            cv2.imshow("Gesture Logger - presiona 'q' para salir", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Sesion finalizada. {len(session_log)} detecciones guardadas en '{CSV_PATH}'.")


if __name__ == "__main__":
    main()
