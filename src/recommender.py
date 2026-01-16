from dataclasses import dataclass
import numpy as np
import pandas as pd
from joblib import load
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    CLEAN_CSV, ARTIFACTS_DIR,
    COL_TRACK_ID, COL_TRACK_NAME, COL_ARTIST, COL_GENRE, COL_SUBGENRE
)

@dataclass
class RecConfig:
    model_type: str = "SBERT"      # "SBERT" or "TFIDF"
    alpha: float = 0.8
    genre_bonus: float = 0.02
    max_per_artist: int = 2
    topN_candidates: int = 200
    K: int = 10
    use_subgenre: bool = False

class Recommender:
    def __init__(self):
        self.df = pd.read_csv(CLEAN_CSV)

        self.tfidf = load(ARTIFACTS_DIR / "tfidf_vectorizer.joblib")
        self.tfidf_matrix = sparse.load_npz(ARTIFACTS_DIR / "tfidf_matrix.npz")
        self.knn_tfidf = load(ARTIFACTS_DIR / "knn_tfidf.joblib")

        self.sbert_embeddings = np.load(ARTIFACTS_DIR / "sbert_embeddings.npy")
        self.knn_sbert = load(ARTIFACTS_DIR / "knn_sbert.joblib")

        self.audio_matrix = np.load(ARTIFACTS_DIR / "audio_matrix.npy")

        # simple lookup key: "track_name — artist"
        self.key_list = (
            self.df[COL_TRACK_NAME].astype(str).str.lower().str.strip()
            + " — "
            + self.df[COL_ARTIST].astype(str).str.lower().str.strip()
        ).tolist()

    def _seed_indices(self, seeds_title_artist: list[str]) -> list[int]:
        idxs = []
        for s in seeds_title_artist:
            k = s.strip().lower()
            if k in self.key_list:
                idxs.append(self.key_list.index(k))
        return idxs

    def recommend(self, selected_genre: str, seeds_title_artist: list[str], cfg: RecConfig) -> pd.DataFrame:
        seed_idxs = self._seed_indices(seeds_title_artist)
        if len(seed_idxs) == 0:
            raise ValueError("No valid seeds found. Use format: 'track_name — artist' exactly as in dataset.")

        # profiles
        audio_profile = self.audio_matrix[seed_idxs].mean(axis=0, keepdims=True)

        if cfg.model_type.upper() == "TFIDF":
            seed_vecs = self.tfidf_matrix[seed_idxs]
            text_profile = seed_vecs.mean(axis=0)
            dists, neigh = self.knn_tfidf.kneighbors(text_profile, n_neighbors=cfg.topN_candidates)
            cand_idxs = neigh[0].tolist()
            text_sims = 1.0 - dists[0]
        else:
            seed_vecs = self.sbert_embeddings[seed_idxs]
            text_profile = seed_vecs.mean(axis=0, keepdims=True)
            dists, neigh = self.knn_sbert.kneighbors(text_profile, n_neighbors=cfg.topN_candidates)
            cand_idxs = neigh[0].tolist()
            text_sims = 1.0 - dists[0]

        seed_set = set(seed_idxs)
        filtered = [(i, float(ts)) for i, ts in zip(cand_idxs, text_sims) if i not in seed_set]

        genre_col = COL_SUBGENRE if cfg.use_subgenre else COL_GENRE
        selected_genre_norm = str(selected_genre).strip().lower()

        rows = []
        for idx, text_sim in filtered:
            audio_sim = float(cosine_similarity(audio_profile, self.audio_matrix[idx:idx+1])[0][0])
            g = str(self.df.iloc[idx][genre_col]).strip().lower()
            bonus = cfg.genre_bonus if g == selected_genre_norm else 0.0
            score = cfg.alpha * text_sim + (1.0 - cfg.alpha) * audio_sim + bonus
            rows.append((idx, score, text_sim, audio_sim, bonus))

        rows.sort(key=lambda x: x[1], reverse=True)

        # diversity rule
        result = []
        artist_count: dict[str, int] = {}
        for idx, score, text_sim, audio_sim, bonus in rows:
            artist = str(self.df.iloc[idx][COL_ARTIST])
            if artist_count.get(artist, 0) >= cfg.max_per_artist:
                continue
            artist_count[artist] = artist_count.get(artist, 0) + 1
            result.append((idx, score, text_sim, audio_sim, bonus))
            if len(result) >= cfg.K:
                break

        # fallback fill
        if len(result) < cfg.K:
            for idx, score, text_sim, audio_sim, bonus in rows:
                if any(r[0] == idx for r in result):
                    continue
                result.append((idx, score, text_sim, audio_sim, bonus))
                if len(result) >= cfg.K:
                    break

        out = []
        for idx, score, text_sim, audio_sim, bonus in result:
            r = self.df.iloc[idx]
            out.append({
                "track_name": r[COL_TRACK_NAME],
                "artist": r[COL_ARTIST],
                "genre": r[genre_col],
                "score": score,
                "text_sim": text_sim,
                "audio_sim": audio_sim,
                "genre_bonus": bonus,
                "track_id": r.get(COL_TRACK_ID, "")
            })
        return pd.DataFrame(out)
