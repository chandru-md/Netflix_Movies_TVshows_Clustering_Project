import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    df["director"] = df["director"].fillna("Unknown")
    df["cast"] = df["cast"].fillna("Unknown")
    df["country"] = df["country"].fillna("Unknown")
    df["rating"] = df["rating"].fillna("Not Rated")

    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df = df.dropna(subset=["date_added"])

    df["year_added"] = df["date_added"].dt.year

    return df