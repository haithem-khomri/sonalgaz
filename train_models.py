"""Train and persist forecasting models for the Sonelgaz consumption prototype.

Pipeline
--------
1. Read ``sonelgaz_consumption_data.csv`` (from ``generate_data.py``).
2. **Chronological** 80/20 split (no shuffle — respects time order).
3. **MinMaxScaler** on features ``X`` and target ``y`` (same scalers used at
   inference time in ``app.py``).
4. Train four regressors; all predict **next timestep consumption** from inputs
   (tabular for RF/XGB, 24-step sequences for LSTM/Transformer).
5. Save everything under ``models/``:
   - RF & XGBoost: ``pickle``
   - Deep models: Keras **JSON architecture** + **``.weights.h5``** (portable
     across many TensorFlow installs; avoids fragile single-file ``.h5`` loads)

Important nuance vs the desktop UI labels
------------------------------------------
The GUI maps Daily→RF, Weekly→XGB, Monthly→LSTM, Quarterly→Transformer. Those
labels are **which model to invoke**, not a different statistical horizon in the
CSV — every model was trained on the same target column.

See ``SYSTEM_DESIGN.md`` and ``models/README.md`` for artifact layout.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Sequential
from xgboost import XGBRegressor

RANDOM_SEED = 42
DATA_PATH = Path("sonelgaz_consumption_data.csv")
MODELS_DIR = Path("models")
FEATURES = ["Hour", "DayOfWeek", "Month", "Season", "Temperature", "Current", "IsHoliday"]
TARGET = "Consumption"
SEQ_LENGTH = 24
EPOCHS = 10
BATCH_SIZE = 32

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def prepare_data(df: pd.DataFrame):
    """Split temporally + fit ``MinMaxScaler`` on train only, transform test."""
    X = df[FEATURES]
    y = df[TARGET]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y


def print_regression_metrics(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print standard regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")


def train_random_forest(X_train, y_train, X_test, y_test, scaler_y):
    print("Training Random Forest...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED)
    model.fit(X_train, y_train.ravel())

    preds_scaled = model.predict(X_test)
    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1))
    y_true = scaler_y.inverse_transform(y_test)
    print_regression_metrics("RF", y_true, preds)
    return model


def train_xgboost(X_train, y_train, X_test, y_test, scaler_y):
    print("Training XGBoost...")
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train.ravel())

    preds_scaled = model.predict(X_test)
    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1))
    y_true = scaler_y.inverse_transform(y_test)
    print_regression_metrics("XGB", y_true, preds)
    return model


def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int):
    """Slide a window of ``seq_length`` rows; target is row *after* the window."""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i : i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)


def train_lstm(X_train, y_train, X_test, y_test, scaler_y):
    print("Training LSTM...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQ_LENGTH)

    model = Sequential(
        [
            Input(shape=(SEQ_LENGTH, X_train.shape[1])),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train_seq, y_train_seq, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    preds_scaled = model.predict(X_test_seq, verbose=0)
    preds = scaler_y.inverse_transform(preds_scaled)
    y_true = scaler_y.inverse_transform(y_test_seq)
    print_regression_metrics("LSTM", y_true, preds)
    return model


def transformer_encoder(inputs, head_size: int, num_heads: int, ff_dim: int, dropout: float = 0.0):
    """Single transformer encoder block."""
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res


def train_transformer(X_train, y_train, X_test, y_test, scaler_y):
    print("Training Transformer...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQ_LENGTH)

    inputs = Input(shape=(SEQ_LENGTH, X_train.shape[1]))
    x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = GlobalAveragePooling1D(data_format="channels_last")(x)
    outputs = Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train_seq, y_train_seq, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    preds_scaled = model.predict(X_test_seq, verbose=0)
    preds = scaler_y.inverse_transform(preds_scaled)
    y_true = scaler_y.inverse_transform(y_test_seq)
    print_regression_metrics("Transformer", y_true, preds)
    return model


def save_artifacts(rf_model, xgb_model, lstm_model, transformer_model, scaler_X, scaler_y) -> None:
    """Persist model artifacts to disk."""
    MODELS_DIR.mkdir(exist_ok=True)

    with open(MODELS_DIR / "rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open(MODELS_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    with open(MODELS_DIR / "scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open(MODELS_DIR / "scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)

    # Save architectures + weights to reduce keras deserialization incompatibilities.
    lstm_model.save_weights(MODELS_DIR / "lstm_weights.weights.h5")
    with open(MODELS_DIR / "lstm_architecture.json", "w", encoding="utf-8") as f:
        f.write(lstm_model.to_json())

    transformer_model.save_weights(MODELS_DIR / "transformer_weights.weights.h5")
    with open(MODELS_DIR / "transformer_architecture.json", "w", encoding="utf-8") as f:
        f.write(transformer_model.to_json())

    print(f"All models and scalers saved in '{MODELS_DIR}/'.")


def main() -> None:
    """Train all supported models and save artifacts."""
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = prepare_data(df)

    rf_model = train_random_forest(X_train, y_train, X_test, y_test, scaler_y)
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test, scaler_y)
    lstm_model = train_lstm(X_train, y_train, X_test, y_test, scaler_y)
    transformer_model = train_transformer(X_train, y_train, X_test, y_test, scaler_y)

    save_artifacts(rf_model, xgb_model, lstm_model, transformer_model, scaler_X, scaler_y)


if __name__ == "__main__":
    main()
