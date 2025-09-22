import pandas as pd
from sklearn.preprocessing import StandardScaler

def remove_outliers(data: pd.DataFrame, cols: list) -> pd.DataFrame:
    clean_data = data.copy()
    for col in cols:
        Q1 = clean_data[col].quantile(0.25)
        Q3 = clean_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        clean_data = clean_data[(clean_data[col] >= lower) & (clean_data[col] <= upper)]
    return clean_data

def preprocess_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    # dealing with missing values
    for col in ["ca", "thal", "slope"]:
        mode_val = df_raw[col].mode()[0]
        df_raw[col].fillna(mode_val, inplace=True)

    for col in df_raw.columns:
        if df_raw[col].isna().sum() > 0:
            median_val = df_raw[col].median()
            df_raw[col].fillna(median_val, inplace=True)

    # Removing outliers
    numeric_cols = ["age","trestbps","chol","thalach","oldpeak"]
    df_clean = remove_outliers(df_raw, numeric_cols)

    # prepating target
    df_clean["target"] = df_clean["target"].apply(lambda x: 1 if x > 0 else 0)

    # Scaling
    X = df_clean.drop("target", axis=1)
    y = df_clean["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_ready = pd.DataFrame(X_scaled, columns=X.columns)
    df_ready["target"] = y.reset_index(drop=True)

    return df_ready
