import pandas as pd
import joblib
import os
import asyncio
import aiohttp
from aiohttp import ClientTimeout, TCPConnector
from bs4 import BeautifulSoup
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# ============================================================
#                PATHS & DIRECTORIES
# ============================================================

DATA_DIR = "src/data"
MODEL_DIR = "src/model"
OUTPUT_DIR = "src/output"
LOG_DIR = "src/logs"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{LOG_DIR}/scraping_log.csv"
MODEL_PATH = f"{MODEL_DIR}/dynamic_price_model.joblib"


# ============================================================
#                CREATE LOG FILE IF NOT EXISTS
# ============================================================

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("timestamp,product_name,amazon_price,flipkart_price,croma_price,avg_price,fallback_used\n")


# ============================================================
#                LOGGING HELPER
# ============================================================

def log_scrape(product_name, amazon, flipkart, croma, avg, fallback):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},"
            f"{product_name},{amazon},{flipkart},{croma},{avg},{fallback}\n"
        )


# ============================================================
#                SCRAPER HELPERS (FAST)
# ============================================================

HEADERS = {"User-Agent": "Mozilla/5.0"}

def clean_price(text):
    text = text.replace(",", "")
    price = re.sub(r"[^0-9]", "", text)
    return int(price) if price else None


async def fetch_html(session, url):
    """Fast HTML fetcher with retry."""
    for _ in range(2):  # retry twice
        try:
            async with session.get(url, headers=HEADERS) as resp:
                return await resp.text()
        except:
            continue
    return ""


async def scrape_amazon(session, product):
    url = "https://www.amazon.in/s?k=" + product.replace(" ", "+")
    html = await fetch_html(session, url)
    soup = BeautifulSoup(html, "lxml")

    selectors = [
        ("span", {"class": "a-price-whole"}),
        ("span", {"class": "a-offscreen"})
    ]

    for tag, cl in selectors:
        block = soup.find(tag, cl)
        if block:
            return clean_price(block.text)
    return None


async def scrape_flipkart(session, product):
    url = "https://www.flipkart.com/search?q=" + product.replace(" ", "+")
    html = await fetch_html(session, url)
    soup = BeautifulSoup(html, "lxml")

    selectors = [("div", {"class": "_30jeq3"})]

    for tag, cl in selectors:
        block = soup.find(tag, cl)
        if block:
            return clean_price(block.text)
    return None


async def scrape_croma(session, product):
    url = "https://www.croma.com/search/?text=" + product.replace(" ", "%20")
    html = await fetch_html(session, url)
    soup = BeautifulSoup(html, "lxml")

    block = soup.find("span", {"class": "amount"})
    if block:
        return clean_price(block.text)

    return None


# ============================================================
#         FETCH COMPETITOR PRICE USING SCRAPERS
# ============================================================

async def fetch_competitor_price(session, product_name, cost_price=None):

    amazon_task = asyncio.create_task(scrape_amazon(session, product_name))
    flipkart_task = asyncio.create_task(scrape_flipkart(session, product_name))
    croma_task = asyncio.create_task(scrape_croma(session, product_name))

    amazon_price, flipkart_price, croma_price = await asyncio.gather(
        amazon_task, flipkart_task, croma_task
    )

    prices = [p for p in [amazon_price, flipkart_price, croma_price] if p is not None]
    fallback_used = False

    if prices:
        avg_price = round(sum(prices) / len(prices), 2)
    else:
        fallback_used = True
        avg_price = None if cost_price is None else round(cost_price * 1.20, 2)

    log_scrape(product_name, amazon_price, flipkart_price, croma_price, avg_price, fallback_used)

    return avg_price


# ============================================================
#       UPDATE DATASET WITH COMPETITOR SCRAPED PRICES
# ============================================================

async def update_dataset_with_scraping(input_file):

    df = pd.read_csv(input_file)
    df.columns = df.columns.str.lower()

    if "product_name" not in df.columns:
        raise ValueError("❌ 'product_name' column missing!")

    if "competitor_price" not in df.columns:
        df["competitor_price"] = None

    connector = TCPConnector(limit=20)  # fast concurrent scraping
    timeout = ClientTimeout(total=8)

    print("🔄 Scraping started...")

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            fetch_competitor_price(session, str(row["product_name"]), float(row.get("cost_price", 0)))
            for _, row in df.iterrows()
        ]

        df["competitor_price"] = await asyncio.gather(*tasks)

    print("✅ Scraping completed! Updating dataset...")

    updated_file = f"{DATA_DIR}/updated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(updated_file, index=False)

    print("📁 Dataset updated:", updated_file)

    return updated_file


# ============================================================
#       CLEAN AND NORMALIZE NUMERIC COLUMNS
# ============================================================

def clean_numeric(df):
    numeric_cols = ["competitor_price", "cost_price", "current_price"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    df["competitor_price"].fillna(df["cost_price"] * 1.20, inplace=True)
    return df


# ============================================================
#           TRAIN MODEL AFTER CLEANING
# ============================================================

def train_model(cleaned_file):

    df = pd.read_csv(cleaned_file)
    df.columns = df.columns.str.lower()

    df = clean_numeric(df)

    df["price_diff"] = df["competitor_price"] - df["cost_price"]
    df["margin"] = df["current_price"] - df["cost_price"]
    df["competitor_ratio"] = df["competitor_price"] / (df["cost_price"] + 1e-6)

    X = df[["category", "brand", "cost_price", "competitor_price",
            "stock", "price_diff", "margin", "competitor_ratio"]]

    y = df["current_price"]

    encoder = OneHotEncoder(handle_unknown="ignore")
    encoded = encoder.fit_transform(X[["category", "brand"]]).toarray()
    encoded_cols = encoder.get_feature_names_out(["category", "brand"])
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)

    X_final = pd.concat([X.drop(["category", "brand"], axis=1), encoded_df], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump({
        "model": model,
        "encoder": encoder,
        "encoded_columns": encoded_cols.tolist()
    }, MODEL_PATH)

    print("🤖 Model training complete!")
    return MODEL_PATH


def generate_recommended_output(updated_file):
    df = pd.read_csv(updated_file)
    df.columns = df.columns.str.lower()

    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    encoder = model_data["encoder"]

    # Derived features (must match training features)
    df["price_diff"] = df["competitor_price"] - df["cost_price"]
    df["margin"] = df["current_price"] - df["cost_price"]
    df["competitor_ratio"] = df["competitor_price"] / (df["cost_price"] + 1e-6)

    X = df[["category", "brand", "cost_price", "competitor_price", "stock",
            "price_diff", "margin", "competitor_ratio"]]

    encoded = encoder.transform(X[["category", "brand"]]).toarray()
    encoded_cols = encoder.get_feature_names_out(["category", "brand"])
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)

    X_final = pd.concat([X.drop(["category", "brand"], axis=1), encoded_df], axis=1)

    df["recommended_price"] = model.predict(X_final)

    output_df = df[["product_name", "cost_price", "recommended_price"]]
    output_file = f"{OUTPUT_DIR}/recommended_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_df.to_csv(output_file, index=False)

    print("📄 Generated recommended price file:", output_file)
    return output_file


# ============================================================
#           RUN FULL PIPELINE
# ============================================================

from tkinter import Tk
from tkinter.filedialog import askopenfilename

def run_pipeline():

    # GUI SELECTOR
    Tk().withdraw()  # hide tkinter root window
    input_file = askopenfilename(
        title="Select Dataset CSV File",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )

    if not input_file:
        print("❌ No file selected. Exiting.")
        return

    print("\n📂 Selected file:", input_file)

    print("\n🔄 Updating dataset with live competitor prices...")
    updated_file = asyncio.run(update_dataset_with_scraping(input_file))

    print("\n🔧 Cleaning & Training Model...")
    model_path = train_model(updated_file)

    print("\n📊 Generating Recommended Pricing Output...")
    output_file = generate_recommended_output(updated_file)

    print(f"\n🎉 Pipeline Completed Successfully!")
    print(f"📁 Updated file saved at: {updated_file}")
    print(f"🧠 Model saved at: {model_path}")



# ============================================================

if __name__ == "__main__":
    run_pipeline()
