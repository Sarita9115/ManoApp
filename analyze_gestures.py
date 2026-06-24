"""
Parte 2: Analisis estadistico del CSV generado por gesture_logger.py.

Lee 'gestures_log.csv' con pandas y calcula los indicadores pedidos en la
actividad, conectandolos con los conceptos de clase:

  - Estadistica vs. indicador: un conteo crudo (ej. "12 Thumbs Up") es una
    estadistica; se vuelve indicador cuando se compara contra algo
    (el total, otra mano, otra estacion, otra ronda).
  - Tendencia central: gesto mas comun (moda) y promedio/moda de dedos.
  - Dispersion / proporciones: % de un gesto especifico sobre el total,
    y desviacion estandar de los dedos levantados.
  - Comparacion de grupos: agrupando por mano (handedness) y por estacion
    (requestid) para ver diferencias de comportamiento.
"""

import sys

import pandas as pd

CSV_PATH = "gestures_log.csv"
TARGET_GESTURE = "Thumbs Up"  # Gesto de referencia para la proporcion


def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"No se encontro '{path}'. Corre primero gesture_logger.py para generar datos.")
        sys.exit(1)

    if df.empty:
        print(f"'{path}' existe pero no tiene detecciones todavia.")
        sys.exit(1)

    return df


def print_general_summary(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("RESUMEN GENERAL (df.describe())")
    print("=" * 70)
    # include="all" para que tambien resuma columnas categoricas
    # (gesture_name, handedness, requestid), no solo user_count.
    print(df.describe(include="all"))
    print()


def print_central_tendency(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("TENDENCIA CENTRAL")
    print("=" * 70)

    gesture_mode = df["gesture_name"].mode().iloc[0]
    gesture_counts = df["gesture_name"].value_counts()

    finger_mean = df["user_count"].mean()
    finger_mode = df["user_count"].mode().iloc[0]

    print(f"Gesto mas comun: '{gesture_mode}' ({gesture_counts[gesture_mode]} de {len(df)} detecciones)")
    print(f"Promedio de dedos levantados: {finger_mean:.2f}")
    print(f"Moda de dedos levantados: {finger_mode}")
    print()


def print_dispersion_and_rates(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("DISPERSION Y PROPORCIONES")
    print("=" * 70)

    total = len(df)
    finger_std = df["user_count"].std()
    print(f"Desviacion estandar de dedos levantados: {finger_std:.2f} "
          f"(que tan dispersas estan las cuentas de dedos respecto al promedio)")

    target_count = (df["gesture_name"] == TARGET_GESTURE).sum()
    target_pct = 100.0 * target_count / total
    print(f"Proporcion de '{TARGET_GESTURE}': {target_count}/{total} = {target_pct:.1f}% del total")
    print()

    print("Comparacion por mano (handedness) -> distribucion de gestos:")
    hand_gesture_table = pd.crosstab(df["handedness"], df["gesture_name"], normalize="index") * 100
    print(hand_gesture_table.round(1).astype(str) + " %")
    print()

    print("Conteo absoluto por mano:")
    print(df["handedness"].value_counts())
    print()


def print_station_comparison(df: pd.DataFrame) -> None:
    """
    Comparacion entre grupos (requestid = estacion/equipo). Si solo hay una
    estacion en el CSV, esta tabla sirve igual para comparar entre rondas
    futuras una vez que se junten varios archivos/estaciones.
    """
    print("=" * 70)
    print("COMPARACION ENTRE ESTACIONES/EQUIPOS (requestid)")
    print("=" * 70)

    station_summary = df.groupby("requestid").agg(
        total_detecciones=("gesture_name", "count"),
        promedio_dedos=("user_count", "mean"),
        gesto_mas_comun=("gesture_name", lambda s: s.mode().iloc[0]),
    )
    station_summary["promedio_dedos"] = station_summary["promedio_dedos"].round(2)
    print(station_summary)
    print()


def print_person_comparison(df: pd.DataFrame) -> None:
    """
    Comparacion entre personas identificadas automaticamente (person_id).
    Permite ver, por ejemplo, si Person_1 reaparece varias veces (porque salio
    y volvio a la estacion) y como se compara su comportamiento con Person_2.
    """
    print("=" * 70)
    print("COMPARACION ENTRE PERSONAS (person_id)")
    print("=" * 70)

    if "person_id" not in df.columns:
        print("Este CSV no tiene columna 'person_id' (fue generado con una version anterior).")
        print()
        return

    person_summary = df.groupby("person_id").agg(
        total_detecciones=("gesture_name", "count"),
        promedio_dedos=("user_count", "mean"),
        gesto_mas_comun=("gesture_name", lambda s: s.mode().iloc[0]),
        primera_aparicion=("timestamp", "min"),
        ultima_aparicion=("timestamp", "max"),
    )
    person_summary["promedio_dedos"] = person_summary["promedio_dedos"].round(2)
    print(person_summary)
    print()


def main() -> None:
    df = load_data(CSV_PATH)

    print_general_summary(df)
    print_central_tendency(df)
    print_dispersion_and_rates(df)
    print_station_comparison(df)
    print_person_comparison(df)


if __name__ == "__main__":
    main()
