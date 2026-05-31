"""
automate_NelsonFw.py
======================
Script otomatisasi preprocessing untuk dataset Store Sales - Time Series Forecasting.
Menjalankan seluruh pipeline preprocessing dan menyimpan data siap latih.
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Konfigurasi
# ─────────────────────────────────────────────
DATA_PATH = './store-sales-time-series-forecasting/'
OUTPUT_PATH = './store_sales_preprocessing/'
DATE_FILTER_START = '2016-01-01'

FEATURE_COLS = [
    'store_nbr', 'family_encoded', 'onpromotion', 'city_encoded',
    'state_encoded', 'type_encoded', 'cluster', 'oil_price', 'is_holiday',
    'year', 'month', 'day', 'dayofweek', 'weekofyear', 'quarter',
    'is_weekend', 'is_month_start', 'is_month_end',
    'lag_7', 'lag_14', 'rolling_mean_7'
]
TARGET_COL = 'sales_log'


# ─────────────────────────────────────────────
# Fungsi-fungsi preprocessing
# ─────────────────────────────────────────────

def load_data(data_path: str) -> tuple:
    """Memuat semua file dataset dari path yang diberikan."""
    logger.info("Memuat dataset...")
    train_df = pd.read_csv(data_path + 'train.csv', parse_dates=['date'])
    stores_df = pd.read_csv(data_path + 'stores.csv')
    oil_df = pd.read_csv(data_path + 'oil.csv', parse_dates=['date'])
    holidays_df = pd.read_csv(data_path + 'holidays_events.csv', parse_dates=['date'])
    logger.info(f"Train: {train_df.shape}, Stores: {stores_df.shape}, "
                f"Oil: {oil_df.shape}, Holidays: {holidays_df.shape}")
    return train_df, stores_df, oil_df, holidays_df


def filter_data(train_df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """Memfilter data berdasarkan tanggal mulai."""
    filtered = train_df[train_df['date'] >= start_date].copy()
    logger.info(f"Data setelah filter ({start_date}): {filtered.shape}")
    return filtered


def merge_datasets(train_df: pd.DataFrame,
                   stores_df: pd.DataFrame,
                   oil_df: pd.DataFrame,
                   holidays_df: pd.DataFrame) -> pd.DataFrame:
    """Menggabungkan semua dataset menjadi satu DataFrame."""
    logger.info("Menggabungkan dataset...")

    df = train_df.merge(stores_df, on='store_nbr', how='left')

    # Interpolasi harga minyak harian
    oil_daily = (
        oil_df.set_index('date')['dcoilwtico']
        .resample('D')
        .interpolate(method='linear')
        .reset_index()
    )
    oil_daily.columns = ['date', 'oil_price']
    df = df.merge(oil_daily, on='date', how='left')

    # Flag hari libur nasional
    national_holidays = holidays_df[
        (holidays_df['locale'] == 'National') &
        (holidays_df['transferred'] == False)
    ]['date'].unique()
    df['is_holiday'] = df['date'].isin(national_holidays).astype(int)

    logger.info(f"Shape setelah merge: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Membuat fitur tambahan dari tanggal dan lag penjualan."""
    logger.info("Feature engineering...")

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    df = df.sort_values(['store_nbr', 'family', 'date'])
    df['lag_7'] = df.groupby(['store_nbr', 'family'])['sales'].shift(7)
    df['lag_14'] = df.groupby(['store_nbr', 'family'])['sales'].shift(14)
    df['rolling_mean_7'] = df.groupby(['store_nbr', 'family'])['sales'].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Menangani missing values."""
    logger.info("Menangani missing values...")
    before = df.isnull().sum().sum()

    df['oil_price'] = df['oil_price'].fillna(df['oil_price'].median())
    df['lag_7'] = df['lag_7'].fillna(0)
    df['lag_14'] = df['lag_14'].fillna(0)
    df['rolling_mean_7'] = df['rolling_mean_7'].fillna(0)

    after = df.isnull().sum().sum()
    logger.info(f"Missing values: {before} -> {after}")
    return df


def encode_categorical(df: pd.DataFrame) -> tuple:
    """Encoding variabel kategorikal menggunakan LabelEncoder."""
    logger.info("Encoding variabel kategorikal...")

    encoders = {}
    for col in ['family', 'city', 'state', 'type']:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


def handle_outliers_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Menangani outlier dan transformasi log pada target."""
    logger.info("Handling outlier dan log transform...")

    Q1 = df['sales'].quantile(0.01)
    Q3 = df['sales'].quantile(0.99)
    df['sales_clipped'] = df['sales'].clip(lower=Q1, upper=Q3)
    df['sales_log'] = np.log1p(df['sales_clipped'])

    logger.info(f"Sales log range: {df['sales_log'].min():.3f} - {df['sales_log'].max():.3f}")
    return df


def normalize_features(df: pd.DataFrame,
                        feature_cols: list) -> tuple:
    """Normalisasi fitur numerik dengan StandardScaler."""
    logger.info("Normalisasi fitur...")

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler


def save_artifacts(df: pd.DataFrame,
                   scaler: StandardScaler,
                   encoders: dict,
                   feature_cols: list,
                   target_col: str,
                   output_path: str):
    """Menyimpan dataset hasil preprocessing dan artefak pendukung."""
    os.makedirs(output_path, exist_ok=True)

    # Simpan dataset
    output_file = os.path.join(output_path, 'train_preprocessed.csv')
    df[feature_cols + [target_col]].to_csv(output_file, index=False)
    logger.info(f"Dataset tersimpan: {output_file}")

    # Simpan scaler
    with open(os.path.join(output_path, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Simpan encoders
    with open(os.path.join(output_path, 'encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)

    # Simpan metadata
    metadata = {
        'feature_cols': feature_cols,
        'target_col': target_col,
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'date_filter_start': DATE_FILTER_START,
    }
    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Semua artefak tersimpan di: {output_path}")


# ─────────────────────────────────────────────
# Pipeline utama
# ─────────────────────────────────────────────

def run_preprocessing_pipeline(data_path: str = DATA_PATH,
                                output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    """
    Menjalankan seluruh pipeline preprocessing dari load hingga simpan.
    
    Returns:
        pd.DataFrame: Data yang sudah diproses dan siap dilatih.
    """
    logger.info("=" * 50)
    logger.info("MULAI PIPELINE PREPROCESSING")
    logger.info("=" * 50)

    # Load
    train_df, stores_df, oil_df, holidays_df = load_data(data_path)

    # Filter
    train_df = filter_data(train_df, DATE_FILTER_START)

    # Merge
    df = merge_datasets(train_df, stores_df, oil_df, holidays_df)

    # Feature engineering
    df = engineer_features(df)

    # Missing values
    df = handle_missing_values(df)

    # Encoding
    df, encoders = encode_categorical(df)

    # Outlier + transform
    df = handle_outliers_and_transform(df)

    # Normalisasi
    df, scaler = normalize_features(df, FEATURE_COLS)

    # Simpan
    save_artifacts(df, scaler, encoders, FEATURE_COLS, TARGET_COL, output_path)

    logger.info("=" * 50)
    logger.info(f"PREPROCESSING SELESAI - Shape: {df[FEATURE_COLS + [TARGET_COL]].shape}")
    logger.info("=" * 50)

    return df[FEATURE_COLS + [TARGET_COL]]


if __name__ == '__main__':
    result = run_preprocessing_pipeline()
    print(result.head())
