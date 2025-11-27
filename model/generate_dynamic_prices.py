import pandas as pd
import joblib
import os
from datetime import datetime
import glob

USE_FILE_PICKER = False  # change to True if you want manual selection

# Optional File Picker
if USE_FILE_PICKER:
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename


# --------------------------------------------------------
# Paths
# --------------------------------------------------------
MODEL_PATH = "src/model/dynamic_price_model.joblib"
DATA_DIR = "src/data"
OUTPUT_DIR = "src/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------
# Step 1 — Select latest scraped file
# --------------------------------------------------------
if USE_FILE_PICKER:
    Tk().withdraw()
    print("\n📂 Select updated dataset file...")
    DATA_PATH = askopenfilename(
        title="Select Updated Dataset",
        filetypes=[("CSV Files", "*.csv")]
    )
else:
    # Auto-select latest updated file
    files = glob.glob(f"{DATA_DIR}/updated_*.csv")
    if not files:
        raise FileNotFoundError("❌ No updated_*.csv file found in src/data. Run pipeline first!")

    DATA_PATH = max(files, key=os.path.getmtime)

print(f"✔ Using dataset: {DATA_PATH}")


# --------------------------------------------------------
# Load dataset
# --------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()


# --------------------------------------------------------
# Load trained model
# --------------------------------------------------------
saved = joblib.load(MODEL_PATH)
model = saved["model"]
encoder = saved["encoder"]
encoded_cols = saved["encoded_columns"]


# --------------------------------------------------------
# Step 2 — Feature Engineering
# --------------------------------------------------------
df["price_diff"] = df["competitor_price"] - df["cost_price"]
df["margin"] = df["current_price"] - df["cost_price"]
df["competitor_ratio"] = df["competitor_price"] / (df["cost_price"] + 1e-6)

X = df[
    [
        "category",
        "brand",
        "cost_price",
        "competitor_price",
        "stock",
        "price_diff",
        "margin",
        "competitor_ratio",
    ]
]


# --------------------------------------------------------
# Step 3 — Encode category + brand safely
# --------------------------------------------------------
encoded = encoder.transform(X[["category", "brand"]]).toarray()

encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)


# --------------------------------------------------------
# Step 4 — Build final ML input
# --------------------------------------------------------
X_final = pd.concat([X.drop(["category", "brand"], axis=1), encoded_df], axis=1)


# --------------------------------------------------------
# Step 5 — Predict + Business rules
# --------------------------------------------------------
predicted = model.predict(X_final)
recommended = []

for i, pred_price in enumerate(predicted):
    cost = df.loc[i, "cost_price"]
    comp = df.loc[i, "competitor_price"]

    min_price = cost * 1.05
    min_comp = comp * 0.95
    max_comp = comp * 1.10

    final_price = pred_price

    if final_price < min_price:
        final_price = min_price
    if final_price > max_comp:
        final_price = max_comp
    if final_price < min_comp:
        final_price = min_comp

    recommended.append(round(final_price, 2))

df["recommended_price"] = recommended


# --------------------------------------------------------
# Step 6 — Save output
# --------------------------------------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_file = f"{OUTPUT_DIR}/pricing_result_{timestamp}.csv"
df.to_csv(output_file, index=False)

print(f"\n✅ Recommended pricing saved to: {output_file}")
