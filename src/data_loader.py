import zipfile
import pandas as pd

def load_raw_data(zip_path: str) -> pd.DataFrame:
    processed_files = [
        "heart+disease/processed.cleveland.data",
        "heart+disease/processed.hungarian.data",
        "heart+disease/processed.switzerland.data",
        "heart+disease/processed.va.data",
    ]

    cols = [
        "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
        "exang","oldpeak","slope","ca","thal","target"
    ]

    rows = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for pf in processed_files:
            raw = z.read(pf).decode('latin1').splitlines()
            for line in raw:
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < len(cols):
                    parts += ["?"] * (len(cols) - len(parts))
                elif len(parts) > len(cols):
                    parts = parts[:len(cols)]
                rows.append(parts)

    df_raw = pd.DataFrame(rows, columns=cols)
    df_raw.replace("?", pd.NA, inplace=True)
    for c in cols:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

    return df_raw
