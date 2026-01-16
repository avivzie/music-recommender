import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from src.config import AUDIO_COLS, ARTIFACTS_DIR

SCALER_PATH = ARTIFACTS_DIR / "audio_scaler.joblib"
AUDIO_MATRIX_PATH = ARTIFACTS_DIR / "audio_matrix.npy"

def build_audio_matrix(df: pd.DataFrame, fit: bool = True) -> np.ndarray:
    X = df[AUDIO_COLS].astype(float)
    X = X.fillna(X.median()).to_numpy()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if fit:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        dump(scaler, SCALER_PATH)
    else:
        scaler = load(SCALER_PATH)
        Xs = scaler.transform(X)

    np.save(AUDIO_MATRIX_PATH, Xs)
    return Xs
