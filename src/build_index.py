import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
from joblib import dump
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec

from src.config import (
    CLEAN_CSV, ARTIFACTS_DIR,
    COL_PLAYLIST_ID, COL_TRACK_ID
)
from src.features import build_audio_matrix

def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN_CSV)

    texts = df["lyrics_clean"].astype(str).tolist()
    tokenized = [t.split() for t in texts]

    # ---- TF-IDF ----
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
    tfidf_matrix = tfidf.fit_transform(texts)
    dump(tfidf, ARTIFACTS_DIR / "tfidf_vectorizer.joblib")
    sparse.save_npz(ARTIFACTS_DIR / "tfidf_matrix.npz", tfidf_matrix)

    knn_tfidf = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=200)
    knn_tfidf.fit(tfidf_matrix)
    dump(knn_tfidf, ARTIFACTS_DIR / "knn_tfidf.joblib")

    # ---- Bag of Words ----
    bow = CountVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
    bow_matrix = bow.fit_transform(texts)
    dump(bow, ARTIFACTS_DIR / "bow_vectorizer.joblib")
    sparse.save_npz(ARTIFACTS_DIR / "bow_matrix.npz", bow_matrix)

    knn_bow = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=200)
    knn_bow.fit(bow_matrix)
    dump(knn_bow, ARTIFACTS_DIR / "knn_bow.joblib")

    # ---- Word2Vec (avg word embeddings) ----
    w2v = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=2, workers=4, sg=1)
    w2v_path = ARTIFACTS_DIR / "w2v.model"
    w2v.save(str(w2v_path))

    def _doc_embedding(tokens):
        vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
        if not vecs:
            return np.zeros(w2v.vector_size, dtype=np.float32)
        return np.mean(vecs, axis=0)

    w2v_embeddings = np.vstack([_doc_embedding(toks) for toks in tokenized])
    np.save(ARTIFACTS_DIR / "w2v_embeddings.npy", w2v_embeddings)

    knn_w2v = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=200)
    knn_w2v.fit(w2v_embeddings)
    dump(knn_w2v, ARTIFACTS_DIR / "knn_w2v.joblib")

    # ---- Jaccard token sets ----
    token_sets = [set(toks) for toks in tokenized]
    dump(token_sets, ARTIFACTS_DIR / "token_sets.joblib")

    # ---- SBERT ----
    model = SentenceTransformer("all-MiniLM-L6-v2")  # good speed/quality
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

    knn_sbert_euclid = NearestNeighbors(metric="euclidean", algorithm="brute", n_neighbors=200)
    knn_sbert_euclid.fit(embeddings)
    dump(knn_sbert_euclid, ARTIFACTS_DIR / "knn_sbert_euclidean.joblib")

    # ---- Audio + Clustering + Outliers ----
    audio_matrix = build_audio_matrix(df, fit=True)

    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(audio_matrix)
    np.save(ARTIFACTS_DIR / "audio_clusters.npy", cluster_labels)
    np.save(ARTIFACTS_DIR / "audio_cluster_centers.npy", kmeans.cluster_centers_)

    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    iso.fit(audio_matrix)
    outlier_scores = iso.decision_function(audio_matrix)
    outlier_flags = iso.predict(audio_matrix)  # -1 outlier, 1 inlier
    np.save(ARTIFACTS_DIR / "outlier_scores.npy", outlier_scores)
    np.save(ARTIFACTS_DIR / "outlier_flags.npy", outlier_flags)

    # ---- Association Rules (playlist co-occurrence) ----
    assoc_rules = {}
    if COL_PLAYLIST_ID in df.columns:
        tmp = df[[COL_PLAYLIST_ID, COL_TRACK_ID]].dropna()
        playlist_groups = tmp.groupby(COL_PLAYLIST_ID)[COL_TRACK_ID].apply(list)
        cooc = defaultdict(Counter)
        track_playlist_counts = Counter()

        for tracks in playlist_groups:
            uniq = list(set(tracks))
            for t in uniq:
                track_playlist_counts[t] += 1
            for a, b in combinations(uniq, 2):
                cooc[a][b] += 1
                cooc[b][a] += 1

        top_n = 50
        for t, counter in cooc.items():
            total = track_playlist_counts[t]
            if total == 0:
                continue
            pairs = [(other, cnt / total) for other, cnt in counter.items()]
            pairs.sort(key=lambda x: x[1], reverse=True)
            assoc_rules[t] = pairs[:top_n]

    dump(assoc_rules, ARTIFACTS_DIR / "assoc_rules.joblib")

    print("Artifacts built successfully in /artifacts")

if __name__ == "__main__":
    main()
