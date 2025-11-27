# train_dynamic_price_model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "src/data/electronics_training_dataset_1000.csv"
MODEL_DIR = "src/model"
MODEL_PATH = os.path.join(MODEL_DIR, "dynamic_price_model.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load and normalize
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

# Required columns (lowercase)
required_cols = ["category", "brand", "cost_price", "competitor_price", "stock", "current_price"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in dataset: {missing}")

# 2. Fill missing / clean
df = df.copy()
df["category"] = df["category"].fillna("unknown").astype(str)
df["brand"] = df["brand"].fillna("unknown").astype(str)
df["cost_price"] = pd.to_numeric(df["cost_price"], errors="coerce").fillna(0.0)
df["competitor_price"] = pd.to_numeric(df["competitor_price"], errors="coerce").fillna(df["competitor_price"].median())
df["stock"] = pd.to_numeric(df["stock"], errors="coerce").fillna(0).astype(int)
df["current_price"] = pd.to_numeric(df["current_price"], errors="coerce").fillna(df["current_price"].median())

# 3. Feature engineering
df["price_diff"] = df["competitor_price"] - df["cost_price"]
df["margin"] = df["current_price"] - df["cost_price"]
df["competitor_ratio"] = df["competitor_price"] / (df["cost_price"] + 1e-6)

# 4. Define columns used
categorical_cols = ["category", "brand"]
numeric_cols = ["cost_price", "competitor_price", "stock", "price_diff", "margin", "competitor_ratio"]

# 5. Encoder - compatible with older scikit-learn versions
ohe = OneHotEncoder(handle_unknown="ignore")  # don't rely on sparse arg

# Fit encoder on training categories
ohe.fit(df[categorical_cols])
# build encoded column names robustly
try:
    encoded_columns = ohe.get_feature_names_out(categorical_cols)
except Exception:
    encoded_columns = ohe.get_feature_names(categorical_cols)

# 6. Build X, y
# transform categorical to dense
encoded = ohe.transform(df[categorical_cols])
if hasattr(encoded, "toarray"):
    encoded = encoded.toarray()
encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=df.index)

X = pd.concat([df[numeric_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
y = df["current_price"]

# 7. Train/test split + model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# 8. Save artifacts
joblib.dump({
    "model": model,
    "encoder": ohe,
    "categorical_cols": categorical_cols,
    "numeric_cols": numeric_cols,
    "encoded_columns": list(encoded_columns)
}, MODEL_PATH)

print("🎉 Training complete.")
print(f"Train R²: {model.score(X_train, y_train):.4f}")
print(f"Test  R²: {model.score(X_test, y_test):.4f}")
print(f"Saved model package to: {MODEL_PATH}")
