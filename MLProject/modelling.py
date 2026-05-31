"""
modelling.py (MLProject version)
=================================
Versi modelling.py yang mendukung argparse untuk dijalankan via MLProject.
Menyimpan model, artefak, dan metrics ke MLflow Tracking.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Argparse
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Train StoreSales DNN Model')
parser.add_argument('--epochs',       type=int,   default=30)
parser.add_argument('--batch_size',   type=int,   default=2048)
parser.add_argument('--learning_rate',type=float, default=0.001)
parser.add_argument('--dropout_rate', type=float, default=0.3)
parser.add_argument('--units_layer1', type=int,   default=256)
parser.add_argument('--units_layer2', type=int,   default=128)
parser.add_argument('--units_layer3', type=int,   default=64)
parser.add_argument('--test_size',    type=float, default=0.2)
args = parser.parse_args()

# ─────────────────────────────────────────────
# Konfigurasi MLflow
# ─────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000/')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

DATA_PATH = './store_sales_preprocessing/train_preprocessed.csv'
ARTIFACTS_DIR = './ci_artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

FEATURE_COLS = [
    'store_nbr', 'family_encoded', 'onpromotion', 'city_encoded',
    'state_encoded', 'type_encoded', 'cluster', 'oil_price', 'is_holiday',
    'year', 'month', 'day', 'dayofweek', 'weekofyear', 'quarter',
    'is_weekend', 'is_month_start', 'is_month_end',
    'lag_7', 'lag_14', 'rolling_mean_7'
]
TARGET_COL = 'sales_log'


def load_data(path):
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    return train_test_split(X, y, test_size=args.test_size, random_state=42)


def build_model():
    model = keras.Sequential([
        layers.Input(shape=(len(FEATURE_COLS),)),
        layers.Dense(args.units_layer1, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(args.dropout_rate),
        layers.Dense(args.units_layer2, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(args.dropout_rate),
        layers.Dense(args.units_layer3, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model


def plot_loss_curve(history, path):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def main():
    print(f"TensorFlow: {tf.__version__}")
    print(f"MLflow tracking: {MLFLOW_TRACKING_URI}")

    X_train, X_test, y_train, y_test = load_data(DATA_PATH)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    mlflow.tensorflow.autolog(log_models=True, log_input_examples=True)

    active_run = mlflow.active_run()
    if active_run is None:
        active_run = mlflow.start_run(run_name="CI-DNN-StoreSales")

    with active_run:
        model = build_model()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3
            )
        ]

        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        y_pred = model.predict(X_test).flatten()
        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)

        # Plot artefak
        loss_path = os.path.join(ARTIFACTS_DIR, 'training_loss_curve.png')
        plot_loss_curve(history, loss_path)
        mlflow.log_artifact(loss_path)

        # Simpan model ke disk juga (untuk Docker build)
        model.save(os.path.join(ARTIFACTS_DIR, 'saved_model.keras'))
        mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, 'saved_model.keras'))

        print(f"\nMAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}")

        # Tulis run_id ke file
        run_id = mlflow.active_run().info.run_id
        with open('latest_run_id.txt', 'w') as f:
            f.write(run_id)
        print(f"Run ID: {run_id}")


if __name__ == '__main__':
    main()
