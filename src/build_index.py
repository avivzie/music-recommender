import numpy as np
import pandas as pd
from joblib import dump
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

from src.config import CLEAN_CSV, ARTIFACTS_DIR
from src.features import build_audio_matrix

def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN_CSV)

    # ---- TF-IDF ----
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
    tfidf_matrix = tfidf.fit_transform(df["lyrics_clean"].astype(str).tolist())
    dump(tfidf, ARTIFACTS_DIR / "tfidf_vectorizer.joblib")
    sparse.save_npz(ARTIFACTS_DIR / "tfidf_matrix.npz", tfidf_matrix)

    knn_tfidf = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=200)
    knn_tfidf.fit(tfidf_matrix)
    dump(knn_tfidf, ARTIFACTS_DIR / "knn_tfidf.joblib")

    # ---- SBERT ----
    model = SentenceTransformer("all-MiniLM-L6-v2")  # good speed/quality
    texts = df["lyrics_clean"].astype(str).tolist()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(ARTIFACTS_DIR / "sbert_embeddings.npy", embeddings)

    knn_sbert = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=200)
    knn_sbert.fit(embeddings)
    dump(knn_sbert, ARTIFACTS_DIR / "knn_sbert.joblib")

    # ---- Audio ----
    build_audio_matrix(df, fit=True)

    print("Artifacts built successfully in /artifacts")

if __name__ == "__main__":
    main()
