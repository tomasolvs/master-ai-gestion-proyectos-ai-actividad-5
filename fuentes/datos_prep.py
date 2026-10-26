"""Funciones de limpieza y validación del dataset."""
from pathlib import Path
import pandas as pd

TARGET = "diagnostico"
ID_COL = "id_registro"

def limpiar_dataset(input_path, output_path):
    """Lee CSV, normaliza nombres/texto, valida nulos y elimina duplicados exactos."""
    df = pd.read_csv(input_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f"No se encontró la variable objetivo {TARGET!r}")
    filas_iniciales = len(df)
    nulos_iniciales = int(df.isna().sum().sum())
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()
    df = df.drop_duplicates()
    # El dataset público no contiene nulos en las variables de este proyecto.
    # No se imputan valores artificialmente; se informa si aparecieran.
    nulos_finales = int(df.isna().sum().sum())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return {
        "filas_iniciales": filas_iniciales,
        "filas_finales": len(df),
        "duplicados_eliminados": filas_iniciales-len(df),
        "nulos_iniciales": nulos_iniciales,
        "nulos_finales": nulos_finales,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    source = root / "datos/datos_ini/breast_cancer_original.csv"
    target = root / "datos/datos_limp/breast_cancer_limpio.csv"
    print(limpiar_dataset(source, target))
