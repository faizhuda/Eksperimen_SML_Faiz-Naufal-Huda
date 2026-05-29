import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    print(f"[load_data] Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # customerID is not a feature
    df.drop(columns=["customerID"], inplace=True)

    # TotalCharges has 11 rows with whitespace instead of a number
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_val = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_val)

    print(f"[handle_missing] Missing values after fix: {df.isnull().sum().sum()}")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # "No phone service" and "No internet service" are semantically "No"
    service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in service_cols:
        df[col] = df[col].replace({"No phone service": "No", "No internet service": "No"})

    # Binary Yes/No → 1/0
    binary_cols = [
        "Partner", "Dependents", "PhoneService", "MultipleLines",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
    ]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Gender → 1/0
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    # Multi-category ordinal/nominal → integer codes via LabelEncoder
    le = LabelEncoder()
    for col in ["InternetService", "Contract", "PaymentMethod"]:
        df[col] = le.fit_transform(df[col].astype(str))

    # Target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    print(f"[encode_features] Encoding complete. dtype check:\n{df.dtypes.value_counts()}")
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f"[scale_features] Scaled columns: {numeric_cols}")
    return df


def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    df = load_data(input_path)
    df = handle_missing(df)
    df = encode_features(df)
    df = scale_features(df)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[preprocess] Saved {df.shape[0]} rows -> {output_path}")
    return df


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(script_dir, "..", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    out_path = os.path.join(script_dir, "telco_churn_preprocessed.csv")
    preprocess(raw_path, out_path)
