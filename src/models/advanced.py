"""
Fábrica de modelos e utilitários avançados (RF, XGBoost e LSTM).
- Modelos clássicos (linear, rf, xgb) retornam objetos sklearn-compatíveis.
- LSTM usa TensorFlow/Keras e requer o extra 'deep' instalado.

Também inclui funções para preparar dados em janelas temporais (sequências) para LSTM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# XGBoost (opcional, mas incluído em requirements)
try:
    from xgboost import XGBRegressor, XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False

# TensorFlow (opcional via extra 'deep')
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    _HAS_TF = True
except Exception:
    _HAS_TF = False


NUMERIC_EXCLUDE = {"date", "ticker", "target_reg_5d", "target_cls_5d"}

def feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in NUMERIC_EXCLUDE]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


# --------------------------------------------------------------------------------------
# FÁBRICA DE MODELOS (REG/CLS) — linear, rf, xgb
# --------------------------------------------------------------------------------------
def get_regressor(algo: str, random_state: int = 42):
    algo = algo.lower()
    if algo == "linear":
        return LinearRegression()
    if algo == "rf":
        return RandomForestRegressor(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=random_state
        )
    if algo == "xgb":
        if not _HAS_XGB:
            raise ImportError("xgboost não está instalado. Instale com: pip install xgboost")
        return XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=random_state,
            tree_method="hist", objective="reg:squarederror"
        )
    if algo == "lstm":
        raise ValueError("Para LSTM use as funções específicas (build_lstm_reg/build_lstm_cls).")
    raise ValueError(f"Algo desconhecido: {algo}")


def get_classifier(algo: str, random_state: int = 42):
    algo = algo.lower()
    if algo == "linear":
        return LogisticRegression(max_iter=500)
    if algo == "rf":
        return RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, class_weight="balanced",
            random_state=random_state
        )
    if algo == "xgb":
        if not _HAS_XGB:
            raise ImportError("xgboost não está instalado. Instale com: pip install xgboost")
        return XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=random_state,
            tree_method="hist", objective="binary:logistic", eval_metric="logloss"
        )
    if algo == "lstm":
        raise ValueError("Para LSTM use as funções específicas (build_lstm_reg/build_lstm_cls).")
    raise ValueError(f"Algo desconhecido: {algo}")


# --------------------------------------------------------------------------------------
# PREPARAÇÃO DE SEQUÊNCIAS PARA LSTM
# --------------------------------------------------------------------------------------
@dataclass
class SequenceData:
    X: np.ndarray
    y: np.ndarray
    index: pd.DataFrame


def prepare_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_len: int = 20,
) -> SequenceData:
    """
    Constrói janelas temporais por ticker para treinar LSTM.
    Retorna X com shape (samples, seq_len, n_features) e y com shape (samples,).
    """
    assert "date" in df.columns and "ticker" in df.columns
    g = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    X_list, y_list, idx_rows = [], [], []
    for tkr, sub in g.groupby("ticker"):
        sub = sub.dropna(subset=feature_cols + [target_col]).sort_values("date")
        vals = sub[feature_cols].values
        yv = sub[target_col].values
        dates = sub["date"].values

        if len(sub) <= seq_len:
            continue

        for i in range(seq_len, len(sub)):
            X_list.append(vals[i - seq_len : i, :])
            y_list.append(yv[i])
            idx_rows.append({"ticker": tkr, "date": pd.to_datetime(dates[i])})

    if not X_list:
        return SequenceData(np.empty((0, seq_len, len(feature_cols))), np.empty((0,)), pd.DataFrame(columns=["ticker","date"]))

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    idx = pd.DataFrame(idx_rows)
    return SequenceData(X=X, y=y, index=idx)


def prepare_sequences_infer(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 20,
) -> SequenceData:
    """
    Constrói UMA janela por ticker (últimos `seq_len` passos) para inferência.
    """
    assert "date" in df.columns and "ticker" in df.columns
    g = df.sort_values(["ticker", "date"])

    X_list, idx_rows = [], []
    for tkr, sub in g.groupby("ticker"):
        sub = sub.dropna(subset=feature_cols).sort_values("date")
        if len(sub) < seq_len:
            continue
        window = sub[feature_cols].values[-seq_len:, :]
        X_list.append(window)
        idx_rows.append({"ticker": tkr, "date": pd.to_datetime(sub["date"].iloc[-1])})

    if not X_list:
        return SequenceData(np.empty((0, seq_len, len(feature_cols))), np.empty((0,)), pd.DataFrame(columns=["ticker","date"]))

    X = np.stack(X_list, axis=0)
    idx = pd.DataFrame(idx_rows)
    return SequenceData(X=X, y=np.array([]), index=idx)


# --------------------------------------------------------------------------------------
# MODELOS LSTM
# --------------------------------------------------------------------------------------
def build_lstm_reg(input_shape: Tuple[int, int]) -> "tf.keras.Model":
    if not _HAS_TF:
        raise ImportError("TensorFlow não está instalado. Use: pip install '.[deep]'")
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_lstm_cls(input_shape: Tuple[int, int]) -> "tf.keras.Model":
    if not _HAS_TF:
        raise ImportError("TensorFlow não está instalado. Use: pip install '.[deep]'")
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model


def fit_lstm(model: "tf.keras.Model", X: np.ndarray, y: np.ndarray, epochs: int = 20, batch_size: int = 64):
    if not _HAS_TF:
        raise ImportError("TensorFlow não está instalado. Use: pip install '.[deep]'")
    cb = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    n = len(X)
    split = int(n * 0.8)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[cb],
    )
    return hist
