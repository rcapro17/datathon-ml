import pandas as pd
from pathlib import Path

DATASET_PATH = Path("data/raw/PEDE_PASSOS_DATASET_FIAP.csv")

def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {DATASET_PATH}")

 
    df = pd.read_csv(DATASET_PATH, sep=";", encoding="utf-8")
    print(f"[OK] Dataset carregado: {DATASET_PATH} | shape={df.shape}")
    return df

def profile(df: pd.DataFrame, top_nulls: int = 25):
    print("\n--- COLUNAS ---")
    print(df.columns.tolist())

    print("\n--- TIPOS (TOP 30) ---")
    print(df.dtypes.head(30))

    print("\n--- NULOS (TOP) ---")
    nulls = df.isna().sum().sort_values(ascending=False)
    print(nulls.head(top_nulls))

    print("\n--- AMOSTRA ---")
    print(df.head(3))

if __name__ == "__main__":
    df = load_dataset()
    profile(df)

    # checar targets prováveis
    candidates = [c for c in df.columns if "DEFASAGEM" in c.upper()]
    print("\n--- CANDIDATAS A TARGET ---")
    print(candidates)

    print("\n--- DEFASAGEM_2021 value_counts ---")
    print(df["DEFASAGEM_2021"].value_counts(dropna=False))

    print("\n--- % NULOS DEFASAGEM_2021 ---")
    print(df["DEFASAGEM_2021"].isna().mean())

