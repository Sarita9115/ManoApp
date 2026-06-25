"""
Reto bonus: identificacion de persona por la SECUENCIA de gestos de la mano,
sin depender de que se vea la cara (requisito explicito del profesor: la cara
es opcional, la identidad se reconoce con la secuencia de gestos).

Cada persona registra una secuencia de referencia (ej. Thumbs Up con la mano
derecha, luego Fist con la izquierda, luego Open Palm...). Para cada paso se
guarda: el gesto, la mano usada, cuanto tiempo se sostuvo el gesto, y cuanto
tiempo paso desde que termino el paso anterior. Luego, cuando alguien repite
una secuencia, el sistema la compara contra todas las secuencias guardadas
usando tanto el patron (gesto+mano) como el ritmo (timing), y dice a quien
pertenece o "Unknown" si no hay una coincidencia suficientemente buena.

Uso:
    python identity_sequence.py live              -> modo continuo (recomendado para el demo)
    python identity_sequence.py enroll Ana         -> registro manual con nombre
    python identity_sequence.py verify             -> verifica una sola secuencia y termina

En el modo "live" no hace falta avisarle al programa quien va a pasar: la
camara queda corriendo, cada persona hace su secuencia (sin mostrar la cara
si no quiere) y, en cuanto deja de detectarse mano por un rato (se acabo su
turno), el sistema compara esa secuencia contra todas las que ya conoce.
Si el patron (gesto+mano, en orden) ya existe, reconoce a esa persona aunque
hayan pasado otras secuencias distintas en medio; si el patron es nuevo, lo
registra solo como una persona nueva (Person_N) y lo recuerda para la
siguiente vez (se guarda en disco en cada registro).

Controles durante la captura:
    q -> salir / cancelar
"""

import csv
import json
import os
import sys
import time
from datetime import datetime

import cv2
import mediapipe as mp

from gesture_logger import classify_gesture, count_extended_fingers

# ----------------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------------
SEQUENCE_LENGTH = 4        # numero de gestos por secuencia en los modos "enroll"/"verify"
MIN_HOLD_SECONDS = 0.5     # tiempo minimo sosteniendo un gesto para que cuente como paso
MIN_SEQUENCE_STEPS = 2     # pasos minimos para que el modo "live" considere que hubo una secuencia real
SEQUENCE_GAP_SECONDS = 1.8  # pausa sin mano detectada que marca "termino el turno de esta persona"
SIGNATURES_PATH = "identity_signatures.json"

# Que tanto puede variar el RITMO de la secuencia (proporciones normalizadas,
# no segundos crudos) para seguir considerandose la misma persona. Subelo si
# el profesor quiere mas tolerancia (gente nerviosa, ritmos distintos);
# bajalo si quiere mas precision/seguridad.
TIMING_TOLERANCE = 0.18

# Pausa minima sin mano (en segundos) para considerar que la siguiente mano
# que aparece pertenece a una persona DISTINTA (o al mismo turno nuevo).
# Evita que los gestos de Persona 2 se mezclen con la secuencia de Persona 1
# cuando la transicion entre ellas ocurre en menos de SEQUENCE_GAP_SECONDS.
HAND_REAPPEAR_THRESHOLD = 0.3

IDENTITY_CSV_PATH = "identity_log.csv"
IDENTITY_CSV_COLUMNS = ["timestamp", "person_id", "handedness", "user_count", "gesture_name"]


def ensure_csv_header(path: str, columns: list) -> None:
    """Crea el CSV con encabezado si todavia no existe."""
    if not os.path.exists(path):
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(columns)


def load_signatures() -> dict:
    if not os.path.exists(SIGNATURES_PATH):
        return {}
    with open(SIGNATURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_signatures(signatures: dict) -> None:
    with open(SIGNATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(signatures, f, indent=2, ensure_ascii=False)


def _step_label(gesture_name: str, finger_count: int) -> str:
    """
    Etiqueta de un paso de la secuencia. Si el gesto tiene un nombre
    clasificado (Fist, Open Palm, Thumbs Up) usamos ese nombre; si no,
    usamos directamente el numero de dedos levantados (por ejemplo "1", "2",
    "3"), porque secuencias como "1, 2, 3 dedos" son tan validas como
    "Thumbs Up, Open Palm, 1" para identificar a una persona.
    """
    return gesture_name if gesture_name != "Unknown" else str(finger_count)


def _empty_hand_state() -> dict:
    return {"gesture": None, "finger_count": None, "hold_start": None, "last_seen": None}


def _extract_hand_detections(results, mp_drawing, mp_hands, frame) -> list[dict]:
    """
    Devuelve todas las manos detectadas en el frame con su etiqueta de gesto.
    Cada deteccion contiene: hand, gesture y finger_count.
    """
    if not (results.multi_hand_landmarks and results.multi_handedness):
        return []

    detections: list[dict] = []
    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = hand_landmarks.landmark
        finger_count, thumb_extended = count_extended_fingers(landmarks)
        gesture_name = classify_gesture(finger_count, thumb_extended, landmarks)
        label = _step_label(gesture_name, finger_count)
        hand_label = handedness.classification[0].label
        detections.append({
            "gesture": label,
            "hand": hand_label,
            "finger_count": finger_count,
        })

    detections.sort(key=lambda item: item["hand"])
    return detections


def _finalize_hand_step(hand_label: str, state: dict, now: float, last_step_end: float | None) -> dict | None:
    gesture = state.get("gesture")
    hold_start = state.get("hold_start")
    finger_count = state.get("finger_count")

    state["gesture"] = None
    state["finger_count"] = None
    state["hold_start"] = None
    state["last_seen"] = None

    if gesture is None or hold_start is None:
        return None

    held = now - hold_start
    if held < MIN_HOLD_SECONDS:
        return None

    gap_before = (hold_start - last_step_end) if last_step_end else 0.0
    return {
        "gesture": gesture,
        "hand": hand_label,
        "finger_count": finger_count,
        "hold_seconds": round(held, 3),
        "gap_before_seconds": round(max(gap_before, 0.0), 3),
    }


def capture_sequence(sequence_length: int = SEQUENCE_LENGTH) -> list[dict]:
    """
    Abre la webcam y captura una secuencia de `sequence_length` pasos. Cada
    paso es un gesto (clasificado, o el numero de dedos si no tiene nombre)
    sostenido sin cambiar al menos MIN_HOLD_SECONDS; en cuanto deja de
    detectarse esa mano/gesto se cierra el paso y se mide cuanto duro
    (hold_seconds) y cuanto paso desde que cerro el paso anterior hasta que
    empezo este (gap_before_seconds).

    Devuelve una lista de dicts:
        [{"gesture": ..., "hand": ..., "hold_seconds": ..., "gap_before_seconds": ...}, ...]
    """
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam (indice 0).")

    steps: list[dict] = []
    active_states = {"Left": _empty_hand_state(), "Right": _empty_hand_state()}
    last_step_end = None

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        max_num_hands=2,
    ) as hands:
        while len(steps) < sequence_length:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            now = time.time()

            detections = _extract_hand_detections(results, mp_drawing, mp_hands, frame)
            seen_hands = set()
            sequence_done = False

            for detection in detections:
                hand_label = detection["hand"]
                seen_hands.add(hand_label)
                state = active_states.setdefault(hand_label, _empty_hand_state())

                if state["gesture"] is None:
                    state["gesture"] = detection["gesture"]
                    state["finger_count"] = detection["finger_count"]
                    state["hold_start"] = now
                    state["last_seen"] = now
                    continue

                if detection["gesture"] != state["gesture"]:
                    step = _finalize_hand_step(hand_label, state, now, last_step_end)
                    if step is not None:
                        steps.append(step)
                        last_step_end = now
                        if len(steps) >= sequence_length:
                            sequence_done = True
                            break

                    state["gesture"] = detection["gesture"]
                    state["finger_count"] = detection["finger_count"]
                    state["hold_start"] = now
                    state["last_seen"] = now

                state["last_seen"] = now

            if not sequence_done:
                for hand_label, state in active_states.items():
                    if hand_label in seen_hands:
                        continue

                    if state["gesture"] is None or state["last_seen"] is None:
                        continue

                    if (now - state["last_seen"]) < HAND_REAPPEAR_THRESHOLD:
                        continue

                    step = _finalize_hand_step(hand_label, state, now, last_step_end)
                    if step is not None:
                        steps.append(step)
                        last_step_end = now
                        if len(steps) >= sequence_length:
                            sequence_done = True
                            break

            _draw_capture_hud(frame, steps, sequence_length, active_states, now)
            cv2.imshow("Captura de secuencia de identidad - 'q' para cancelar", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if sequence_done:
                break

    cap.release()
    cv2.destroyAllWindows()
    return steps


def _draw_capture_hud(frame, steps, sequence_length, active_states, now) -> None:
    cv2.putText(frame, f"Paso {len(steps) + 1}/{sequence_length}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    y = 65
    for hand_label in ("Left", "Right"):
        state = active_states.get(hand_label)
        if state and state["gesture"] is not None and state["hold_start"] is not None:
            held = now - state["hold_start"]
            cv2.putText(frame, f"{hand_label} - {state['gesture']} ({held:.1f}s)", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y += 25

    for i, step in enumerate(steps):
        text = (f"{i + 1}. {step['hand']} {step['gesture']} "
                f"hold={step['hold_seconds']}s gap={step['gap_before_seconds']}s")
        cv2.putText(frame, text, (10, 100 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def _normalize(values: list[float]) -> list[float]:
    """Convierte una lista de duraciones en proporciones que suman 1."""
    total = sum(values)
    if total <= 1e-6:
        return [0.0 for _ in values]
    return [v / total for v in values]


def compare_sequences(candidate_steps: list[dict], reference_steps: list[dict]) -> tuple[bool, float | None]:
    """
    Compara una secuencia capturada contra una secuencia de referencia.

    1) Patron (gesto + mano, en orden): se exige coincidencia EXACTA. El
       gesto/mano es una señal discreta y confiable (no depende de que tan
       rapido o lento se mueva la persona), asi que funciona como filtro
       duro: si el patron no coincide, ni se compara el ritmo.

    2) Timing: en vez de comparar segundos crudos, normalizamos cada
       duracion contra el total de la secuencia (proporcion 0-1). Esto
       compara el RITMO relativo (ej. "el paso 2 dura el doble que el paso 1")
       en lugar de tiempos absolutos, porque una misma persona rara vez repite
       una secuencia exactamente igual de rapido, pero si tiende a mantener
       las proporciones entre gestos.

    Devuelve (coincide_patron, error_de_ritmo). Si el patron no coincide,
    error_de_ritmo es None.
    """
    if len(candidate_steps) != len(reference_steps):
        return False, None

    for cand, ref in zip(candidate_steps, reference_steps):
        if cand["gesture"] != ref["gesture"] or cand["hand"] != ref["hand"]:
            return False, None

    cand_holds = _normalize([s["hold_seconds"] for s in candidate_steps])
    ref_holds = _normalize([s["hold_seconds"] for s in reference_steps])
    cand_gaps = _normalize([s["gap_before_seconds"] for s in candidate_steps])
    ref_gaps = _normalize([s["gap_before_seconds"] for s in reference_steps])

    hold_error = sum(abs(c - r) for c, r in zip(cand_holds, ref_holds)) / len(cand_holds)
    gap_error = sum(abs(c - r) for c, r in zip(cand_gaps, ref_gaps)) / len(cand_gaps)
    timing_error = (hold_error + gap_error) / 2
    return True, timing_error


def identify_person(
    candidate_steps: list[dict], signatures: dict, tolerance: float = TIMING_TOLERANCE
) -> tuple[str | None, float | None]:
    """
    Busca, entre todas las firmas registradas, la que mejor coincide con la
    secuencia capturada. Solo se consideran candidatos cuyo patron de
    gesto/mano coincide exactamente; entre esos, se elige el de menor error
    de ritmo, siempre que ese error este dentro de la tolerancia.
    """
    best_person, best_error = None, None
    for person, data in signatures.items():
        matches_pattern, timing_error = compare_sequences(candidate_steps, data["steps"])
        if not matches_pattern:
            continue
        if timing_error <= tolerance and (best_error is None or timing_error < best_error):
            best_person, best_error = person, timing_error
    return best_person, best_error


def _print_sequence(steps: list[dict]) -> None:
    for i, step in enumerate(steps, start=1):
        print(f"  {i}. {step['hand']} - {step['gesture']} "
              f"(hold={step['hold_seconds']}s, gap_previo={step['gap_before_seconds']}s)")


def find_matching_person(candidate_steps: list[dict], signatures: dict) -> tuple[str | None, float | None]:
    """
    Busca una firma cuyo PATRON (gesto+mano, en orden) coincida exactamente
    con la secuencia capturada. El patron es la identidad: dos personas con
    secuencias unicas nunca deberian compartir patron, asi que esto es
    suficiente para reconocerlas sin importar el ritmo. El ritmo (timing) se
    devuelve solo como dato informativo, no como condicion para identificar.
    """
    for person, data in signatures.items():
        matches_pattern, timing_error = compare_sequences(candidate_steps, data["steps"])
        if matches_pattern:
            return person, timing_error
    return None, None


def _next_free_person_id(signatures: dict) -> str:
    index = 1
    while f"Person_{index}" in signatures:
        index += 1
    return f"Person_{index}"


def _close_turn(
    sequence: list[dict], signatures: dict
) -> tuple[str, str, dict]:
    """
    Evalua una secuencia completada: reconoce a la persona si su patron ya
    existe, o la registra como nueva. Devuelve (texto_resultado, person_id,
    signatures actualizado).
    """
    person, timing_error = find_matching_person(sequence, signatures)
    if person is None:
        person = _next_free_person_id(signatures)
        signatures[person] = {"steps": sequence}
        save_signatures(signatures)
        result_text = f"Persona NUEVA registrada: {person}"
    else:
        timing_note = f", ritmo={timing_error:.3f}" if timing_error is not None else ""
        result_text = f"Reconocido: {person}{timing_note}"
    print(f"[{time.strftime('%H:%M:%S')}] {result_text} "
          f"(secuencia: {[(s['hand'], s['gesture']) for s in sequence]})")
    return result_text, person, signatures


def run_live() -> None:
    """
    Modo continuo: la camara queda corriendo y el sistema va segmentando
    "turnos" automaticamente. Un turno termina cuando:

      a) Se detecta una mano nueva despues de un gap >= HAND_REAPPEAR_THRESHOLD
         (transicion rapida entre personas: se cierra el turno anterior de
         inmediato, sin esperar SEQUENCE_GAP_SECONDS). Esto evita que los
         gestos de Persona 2 se mezclen con la secuencia de Persona 1.

      b) No se detecta ninguna mano durante SEQUENCE_GAP_SECONDS seguidos
         (la persona se retiro y no llego nadie mas: cierre por timeout).

    Al cerrar un turno con al menos MIN_SEQUENCE_STEPS pasos:
      - Si su patron de gesto+mano ya esta registrado (de esta sesion o de
        una anterior, porque las firmas se guardan en disco), se reconoce a
        esa persona aunque hayan pasado otras secuencias distintas en medio.
      - Si el patron es nuevo, se registra automaticamente como Person_N.
    """
    ensure_csv_header(IDENTITY_CSV_PATH, IDENTITY_CSV_COLUMNS)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    signatures = load_signatures()
    print(f"Modo live: {len(signatures)} persona(s) ya registradas. Presiona 'q' para salir.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam (indice 0).")

    current_sequence: list[dict] = []
    current_turn_rows: list[list] = []   # filas CSV pendientes del turno actual
    active_states = {"Left": _empty_hand_state(), "Right": _empty_hand_state()}
    last_step_end = None
    no_hand_since = None
    last_result_text = ""

    def flush_turn(seq, rows, sigs):
        """Cierra el turno, resuelve el person_id y vuelca las filas al CSV."""
        result_text, person_id, sigs = _close_turn(seq, sigs)
        for row in rows:
            row[1] = person_id   # columna person_id
            writer.writerow(row)
        csv_file.flush()
        return result_text, sigs

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        max_num_hands=2,
    ) as hands, open(IDENTITY_CSV_PATH, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            now = time.time()

            detections = _extract_hand_detections(results, mp_drawing, mp_hands, frame)

            if detections:
                if no_hand_since is not None and current_sequence:
                    gap = now - no_hand_since
                    if gap >= HAND_REAPPEAR_THRESHOLD:
                        if len(current_sequence) >= MIN_SEQUENCE_STEPS:
                            last_result_text, signatures = flush_turn(current_sequence, current_turn_rows, signatures)
                        current_sequence = []
                        current_turn_rows = []
                        last_step_end = None

                no_hand_since = None
                seen_hands = set()

                for detection in detections:
                    hand_label = detection["hand"]
                    seen_hands.add(hand_label)
                    state = active_states.setdefault(hand_label, _empty_hand_state())

                    if state["gesture"] is None:
                        state["gesture"] = detection["gesture"]
                        state["finger_count"] = detection["finger_count"]
                        state["hold_start"] = now
                        state["last_seen"] = now
                        continue

                    if detection["gesture"] != state["gesture"]:
                        step = _finalize_hand_step(hand_label, state, now, last_step_end)
                        if step is not None:
                            current_sequence.append(step)
                            current_turn_rows.append([
                                datetime.now().isoformat(),
                                "Pending",
                                step["hand"],
                                step["finger_count"],
                                step["gesture"],
                            ])
                            last_step_end = now

                        state["gesture"] = detection["gesture"]
                        state["finger_count"] = detection["finger_count"]
                        state["hold_start"] = now
                        state["last_seen"] = now

                    state["last_seen"] = now

                for hand_label, state in active_states.items():
                    if hand_label in seen_hands:
                        continue

                    if state["gesture"] is None or state["last_seen"] is None:
                        continue

                    if (now - state["last_seen"]) < HAND_REAPPEAR_THRESHOLD:
                        continue

                    step = _finalize_hand_step(hand_label, state, now, last_step_end)
                    if step is not None:
                        current_sequence.append(step)
                        current_turn_rows.append([
                            datetime.now().isoformat(),
                            "Pending",
                            step["hand"],
                            step["finger_count"],
                            step["gesture"],
                        ])
                        last_step_end = now

            else:
                if no_hand_since is None:
                    no_hand_since = now

                for hand_label, state in active_states.items():
                    if state["gesture"] is None or state["last_seen"] is None:
                        continue

                    if (now - state["last_seen"]) < HAND_REAPPEAR_THRESHOLD:
                        continue

                    step = _finalize_hand_step(hand_label, state, now, last_step_end)
                    if step is not None:
                        current_sequence.append(step)
                        current_turn_rows.append([
                            datetime.now().isoformat(),
                            "Pending",
                            step["hand"],
                            step["finger_count"],
                            step["gesture"],
                        ])
                        last_step_end = now

                # Caso (b): nadie ha vuelto a aparecer -> cierre por timeout.
                if current_sequence and (now - no_hand_since) >= SEQUENCE_GAP_SECONDS:
                    if len(current_sequence) >= MIN_SEQUENCE_STEPS:
                        last_result_text, signatures = flush_turn(current_sequence, current_turn_rows, signatures)
                    current_sequence = []
                    current_turn_rows = []
                    last_step_end = None

            _draw_live_hud(frame, current_sequence, active_states, now, last_result_text)
            cv2.imshow("Identidad por secuencia (live) - 'q' para salir", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def _draw_live_hud(frame, current_sequence, active_states, now, last_result_text) -> None:
    cv2.putText(frame, f"Pasos del turno actual: {len(current_sequence)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    y = 60
    for hand_label in ("Left", "Right"):
        state = active_states.get(hand_label)
        if state and state["gesture"] is not None and state["hold_start"] is not None:
            held = now - state["hold_start"]
            cv2.putText(frame, f"{hand_label} - {state['gesture']} ({held:.1f}s)", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
            y += 24

    for i, step in enumerate(current_sequence):
        text = f"{i + 1}. {step['hand']} {step['gesture']}"
        cv2.putText(frame, text, (10, 95 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if last_result_text:
        cv2.putText(frame, last_result_text, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("enroll", "verify", "live"):
        print("Uso:")
        print("  python identity_sequence.py live               -> modo continuo (recomendado para el demo)")
        print("  python identity_sequence.py enroll <nombre>    -> registra una secuencia de referencia con nombre")
        print("  python identity_sequence.py verify              -> identifica una sola secuencia y termina")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "live":
        run_live()
        return

    signatures = load_signatures()

    if mode == "enroll":
        if len(sys.argv) < 3:
            print("Falta el nombre de la persona. Ejemplo: python identity_sequence.py enroll Ana")
            sys.exit(1)

        person_name = sys.argv[2]
        print(
            f"Registrando secuencia de referencia para '{person_name}'. "
            f"Realiza {SEQUENCE_LENGTH} gestos claros (Fist, Open Palm, Thumbs Up, etc.), "
            f"sosteniendo cada uno al menos {MIN_HOLD_SECONDS}s y haciendo una pausa entre cada uno."
        )
        steps = capture_sequence()
        if len(steps) < SEQUENCE_LENGTH:
            print("Secuencia incompleta (se cancelo antes de tiempo). No se guardo nada.")
            return

        signatures[person_name] = {"steps": steps}
        save_signatures(signatures)
        print(f"\nSecuencia guardada para '{person_name}':")
        _print_sequence(steps)

    elif mode == "verify":
        if not signatures:
            print("Todavia no hay personas registradas. Usa primero: python identity_sequence.py enroll <nombre>")
            return

        print(f"Realiza tu secuencia de {SEQUENCE_LENGTH} gestos para identificarte...")
        steps = capture_sequence()
        if len(steps) < SEQUENCE_LENGTH:
            print("Secuencia incompleta (se cancelo antes de tiempo). No se puede verificar.")
            return

        print("\nSecuencia capturada:")
        _print_sequence(steps)

        person, error = identify_person(steps, signatures)
        print()
        if person is not None:
            print(f"Identidad reconocida: {person}  (error de ritmo: {error:.3f}, tolerancia: {TIMING_TOLERANCE})")
        else:
            print("Identidad: Unknown (ninguna secuencia registrada coincide dentro de la tolerancia)")

        print("\nDetalle por persona registrada:")
        for name, data in signatures.items():
            matches_pattern, timing_error = compare_sequences(steps, data["steps"])
            if not matches_pattern:
                print(f"  - {name}: patron de gesto/mano NO coincide")
            else:
                veredicto = "dentro de tolerancia" if timing_error <= TIMING_TOLERANCE else "fuera de tolerancia"
                print(f"  - {name}: patron coincide, error de ritmo={timing_error:.3f} ({veredicto})")


if __name__ == "__main__":
    main()
