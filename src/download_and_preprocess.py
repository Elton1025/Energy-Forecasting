import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path

UCI_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_ZIP_PATH = RAW_DATA_DIR / "household_power_consumption.zip"
RAW_TXT_PATH = RAW_DATA_DIR / "household_power_consumption.txt"
OUTPUT_CSV_PATH = PROCESSED_DATA_DIR / "household_power_hourly.csv"

def download_and_extract():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "notebooks").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "src" / "models").mkdir(parents=True, exist_ok=True)

    if not RAW_TXT_PATH.exists():
        if not RAW_ZIP_PATH.exists():
            print(f"[1/4] Downloading dataset from {UCI_URL}...")
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(UCI_URL, headers=headers)
            with urllib.request.urlopen(req) as response, open(RAW_ZIP_PATH, "wb") as out_file:
                total_size = response.headers.get("Content-Length")
                total_size = int(total_size) if total_size else None
                downloaded = 0
                block_size = 1024 * 1024  # 1MB
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if total_size:
                        percent = downloaded / total_size * 100
                        print(f"  Downloaded: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="\r")
                    else:
                        print(f"  Downloaded: {downloaded / (1024*1024):.1f} MB", end="\r")
            print(f"\nDownload complete: {RAW_ZIP_PATH}")
        
        print("[2/4] Extracting zip file...")
        with zipfile.ZipFile(RAW_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        print(f"Extracted to: {RAW_TXT_PATH}")
    else:
        print(f"Raw file already exists at {RAW_TXT_PATH}. Skipping download.")

def preprocess_and_resample():
    print("[3/4] Reading raw minute-level data (2+ million rows)...")
    # Columns: Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3
    df = pd.read_csv(
        RAW_TXT_PATH,
        sep=";",
        na_values=["?", ""],
        low_memory=False
    )
    print(f"  Raw rows read: {len(df):,}")

    # Combine Date and Time
    print("  Parsing datetime and handling missing values...")
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df.drop(columns=["Date", "Time"], inplace=True)
    df.set_index("datetime", inplace=True)

    numeric_cols = [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Linear interpolation for small missing bursts at the minute level
    df[numeric_cols] = df[numeric_cols].interpolate(method="time").ffill().bfill()

    print("[4/4] Resampling from 1-minute to 1-hour intervals...")
    # Resample to hourly mean
    df_hourly = df.resample("1h").mean()

    # In case of any remaining gaps in hourly timestamps
    df_hourly = df_hourly.interpolate(method="time").ffill().bfill()

    # Engineer cyclical temporal features
    print("  Adding temporal features (hour, day of week, month, cyclical sin/cos)...")
    df_hourly["hour"] = df_hourly.index.hour
    df_hourly["dayofweek"] = df_hourly.index.dayofweek
    df_hourly["month"] = df_hourly.index.month
    df_hourly["is_weekend"] = (df_hourly["dayofweek"] >= 5).astype(int)

    # Cyclical sin / cos encodings
    df_hourly["hour_sin"] = np.sin(2 * np.pi * df_hourly["hour"] / 24.0)
    df_hourly["hour_cos"] = np.cos(2 * np.pi * df_hourly["hour"] / 24.0)
    df_hourly["day_sin"] = np.sin(2 * np.pi * df_hourly["dayofweek"] / 7.0)
    df_hourly["day_cos"] = np.cos(2 * np.pi * df_hourly["dayofweek"] / 7.0)
    df_hourly["month_sin"] = np.sin(2 * np.pi * (df_hourly["month"] - 1) / 12.0)
    df_hourly["month_cos"] = np.cos(2 * np.pi * (df_hourly["month"] - 1) / 12.0)

    # Save to CSV
    df_hourly.to_csv(OUTPUT_CSV_PATH)
    file_size_mb = OUTPUT_CSV_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSuccessfully generated: {OUTPUT_CSV_PATH}")
    print(f"Hourly dataset shape: {df_hourly.shape}")
    print(f"File size: {file_size_mb:.2f} MB")
    print("\n--- Summary Statistics (Global_active_power) ---")
    print(df_hourly["Global_active_power"].describe())
    print("\nFirst 3 rows:")
    print(df_hourly.head(3))

if __name__ == "__main__":
    download_and_extract()
    preprocess_and_resample()
