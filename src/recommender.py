from dataclasses import dataclass
import re
import numpy as np
import pandas as pd
from joblib import load
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec

from src.config import (
    CLEAN_CSV, ARTIFACTS_DIR,
    COL_TRACK_ID, COL_TRACK_NAME, COL_ARTIST, COL_GENRE, COL_SUBGENRE
)
from src.preprocess import clean_lyrics

@dataclass
class RecConfig:
    model_type: str = "SBERT_COSINE"      # "SBERT_COSINE", "TFIDF_COSINE", "BOW_COSINE", "W2V_COSINE", "JACCARD", "SBERT_EUCLIDEAN"
    alpha: float = 0.8
    genre_bonus: float = 0.02
    cluster_bonus: float = 0.02
    assoc_bonus: float = 0.05
    max_per_artist: int = 2
    topN_candidates: int = 200
    K: int = 10
    use_subgenre: bool = False
    use_clusters: bool = True
    use_assoc: bool = False
    filter_outliers: bool = False

class Recommender:
    def __init__(self):
        self.df = pd.read_csv(CLEAN_CSV)

        self.tfidf = load(ARTIFACTS_DIR / "tfidf_vectorizer.joblib")
        self.tfidf_matrix = sparse.load_npz(ARTIFACTS_DIR / "tfidf_matrix.npz")
        self.knn_tfidf = load(ARTIFACTS_DIR / "knn_tfidf.joblib")

        self.bow = load(ARTIFACTS_DIR / "bow_vectorizer.joblib")
        self.bow_matrix = sparse.load_npz(ARTIFACTS_DIR / "bow_matrix.npz")
        self.knn_bow = load(ARTIFACTS_DIR / "knn_bow.joblib")

        self.w2v_embeddings = np.load(ARTIFACTS_DIR / "w2v_embeddings.npy")
        self.knn_w2v = load(ARTIFACTS_DIR / "knn_w2v.joblib")

        self.token_sets = load(ARTIFACTS_DIR / "token_sets.joblib")
        self.token_set_sizes = np.array([len(s) for s in self.token_sets], dtype=np.int32)

        self.sbert_embeddings = np.load(ARTIFACTS_DIR / "sbert_embeddings.npy")
        self.knn_sbert = load(ARTIFACTS_DIR / "knn_sbert.joblib")
        self.knn_sbert_euclid = load(ARTIFACTS_DIR / "knn_sbert_euclidean.joblib")
        self._sbert_model = None

        self.audio_matrix = np.load(ARTIFACTS_DIR / "audio_matrix.npy")
        self.audio_clusters = np.load(ARTIFACTS_DIR / "audio_clusters.npy")
        self.outlier_scores = np.load(ARTIFACTS_DIR / "outlier_scores.npy")
        self.outlier_flags = np.load(ARTIFACTS_DIR / "outlier_flags.npy")

        self.assoc_rules = load(ARTIFACTS_DIR / "assoc_rules.joblib")

        self.track_id_list = self.df[COL_TRACK_ID].astype(str).tolist()
        self._w2v_model = None

        # simple lookup key: "track_name — artist"
        self.key_list = (
            self.df[COL_TRACK_NAME].astype(str).str.lower().str.strip()
            + " — "
            + self.df[COL_ARTIST].astype(str).str.lower().str.strip()
        ).tolist()

        self.canonical_keys = [
            self._canonical_key(
                str(self.df.iloc[i][COL_TRACK_NAME]),
                str(self.df.iloc[i][COL_ARTIST])
            )
            for i in range(len(self.df))
        ]

    def _canonical_key(self, track_name: str, artist: str) -> str:
        t = track_name.lower().strip()
        # remove brackets/parentheses and trailing dash segments (often versions/remasters)
        t = re.sub(r"\(.*?\)|\[.*?\]", " ", t)
        t = re.sub(r"\s*-\s*.*$", " ", t)
        # remove version-like tokens
        t = re.sub(r"\b(remaster(ed)?|mix|version|edit|mono|stereo|live|feat|featuring)\b", " ", t)
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()

        a = artist.lower().strip()
        a = re.sub(r"\b(feat|featuring)\b.*$", " ", a).strip()
        a = re.sub(r"[^a-z0-9\s]", " ", a)
        a = re.sub(r"\s+", " ", a).strip()
        return f"{t}|{a}"

    def _ensure_sbert(self):
        if self._sbert_model is None:
            self._sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

    def _ensure_w2v(self):
        if self._w2v_model is None:
            self._w2v_model = Word2Vec.load(str(ARTIFACTS_DIR / "w2v.model"))

    def _seed_indices(self, seeds_title_artist: list[str]) -> list[int]:
        idxs = []
        for s in seeds_title_artist:
            k = s.strip().lower()
            if k in self.key_list:
                idxs.append(self.key_list.index(k))
        return idxs

    def _text_candidates(self, seed_idxs: list[int], cfg: RecConfig):
        model = cfg.model_type.upper()
        if model in {"TFIDF", "TFIDF_COSINE"}:
            seed_vecs = self.tfidf_matrix[seed_idxs]
            text_profile = np.asarray(seed_vecs.mean(axis=0))
            dists, neigh = self.knn_tfidf.kneighbors(text_profile, n_neighbors=cfg.topN_candidates)
            text_sims = 1.0 - dists[0]
        elif model in {"BOW", "BOW_COSINE"}:
            seed_vecs = self.bow_matrix[seed_idxs]
            text_profile = np.asarray(seed_vecs.mean(axis=0))
            dists, neigh = self.knn_bow.kneighbors(text_profile, n_neighbors=cfg.topN_candidates)
            text_sims = 1.0 - dists[0]
        elif model in {"W2V", "W2V_COSINE"}:
            seed_vecs = self.w2v_embeddings[seed_idxs]
            text_profile = seed_vecs.mean(axis=0, keepdims=True)
            dists, neigh = self.knn_w2v.kneighbors(text_profile, n_neighbors=cfg.topN_candidates)
            text_sims = 1.0 - dists[0]
        elif model in {"SBERT", "SBERT_COSINE"}:
            seed_vecs = self.sbert_embeddings[seed_idxs]
            text_profile = seed_vecs.mean(axis=0, keepdims=True)
            dists, neigh = self.knn_sbert.kneighbors(text_profile, n_neighbors=cfg.topN_candidates)
            text_sims = 1.0 - dists[0]
        elif model == "SBERT_EUCLIDEAN":
            seed_vecs = self.sbert_embeddings[seed_idxs]
            text_profile = seed_vecs.mean(axis=0, keepdims=True)
            dists, neigh = self.knn_sbert_euclid.kneighbors(text_profile, n_neighbors=cfg.topN_candidates)
            text_sims = 1.0 / (1.0 + dists[0])
        elif model == "JACCARD":
            seed_set = set()
            for i in seed_idxs:
                seed_set.update(self.token_sets[i])
            seed_size = len(seed_set)
            sims = np.zeros(len(self.token_sets), dtype=np.float32)
            for i, tset in enumerate(self.token_sets):
                inter = len(seed_set & tset)
                union = seed_size + self.token_set_sizes[i] - inter
                sims[i] = (inter / union) if union else 0.0
            topn = min(cfg.topN_candidates, len(sims))
            cand_idxs = np.argpartition(-sims, topn - 1)[:topn].tolist()
            text_sims = sims[cand_idxs]
            return cand_idxs, text_sims
        else:
            raise ValueError(f"Unknown model_type: {cfg.model_type}")

        cand_idxs = neigh[0].tolist()
        return cand_idxs, text_sims

    def _text_candidates_from_query(self, query_text: str, cfg: RecConfig):
        model = cfg.model_type.upper()
        q = clean_lyrics(query_text)
        if model in {"TFIDF", "TFIDF_COSINE"}:
            q_vec = self.tfidf.transform([q])
            dists, neigh = self.knn_tfidf.kneighbors(q_vec, n_neighbors=cfg.topN_candidates)
            text_sims = 1.0 - dists[0]
        elif model in {"BOW", "BOW_COSINE"}:
            q_vec = self.bow.transform([q])
            dists, neigh = self.knn_bow.kneighbors(q_vec, n_neighbors=cfg.topN_candidates)
            text_sims = 1.0 - dists[0]
        elif model in {"W2V", "W2V_COSINE"}:
            self._ensure_w2v()
            tokens = q.split()
            vecs = [self._w2v_model.wv[t] for t in tokens if t in self._w2v_model.wv]
            if vecs:
                q_vec = np.mean(vecs, axis=0, keepdims=True)
            else:
                q_vec = np.zeros((1, self._w2v_model.vector_size), dtype=np.float32)
            dists, neigh = self.knn_w2v.kneighbors(q_vec, n_neighbors=cfg.topN_candidates)
            text_sims = 1.0 - dists[0]
        elif model in {"SBERT", "SBERT_COSINE"}:
            self._ensure_sbert()
            q_vec = self._sbert_model.encode([q], normalize_embeddings=True)
            dists, neigh = self.knn_sbert.kneighbors(q_vec, n_neighbors=cfg.topN_candidates)
            text_sims = 1.0 - dists[0]
        elif model == "SBERT_EUCLIDEAN":
            self._ensure_sbert()
            q_vec = self._sbert_model.encode([q], normalize_embeddings=True)
            dists, neigh = self.knn_sbert_euclid.kneighbors(q_vec, n_neighbors=cfg.topN_candidates)
            text_sims = 1.0 / (1.0 + dists[0])
        elif model == "JACCARD":
            q_set = set(q.split())
            q_size = len(q_set)
            sims = np.zeros(len(self.token_sets), dtype=np.float32)
            for i, tset in enumerate(self.token_sets):
                inter = len(q_set & tset)
                union = q_size + self.token_set_sizes[i] - inter
                sims[i] = (inter / union) if union else 0.0
            topn = min(cfg.topN_candidates, len(sims))
            cand_idxs = np.argpartition(-sims, topn - 1)[:topn].tolist()
            text_sims = sims[cand_idxs]
            return cand_idxs, text_sims
        else:
            raise ValueError(f"Unknown model_type: {cfg.model_type}")

        cand_idxs = neigh[0].tolist()
        return cand_idxs, text_sims

    def recommend(self, selected_genre: str, seeds_title_artist: list[str], cfg: RecConfig) -> pd.DataFrame:
        seed_idxs = self._seed_indices(seeds_title_artist)
        if len(seed_idxs) == 0:
            raise ValueError("No valid seeds found. Use format: 'track_name — artist' exactly as in dataset.")

        # profiles
        audio_profile = self.audio_matrix[seed_idxs].mean(axis=0, keepdims=True)

        cand_idxs, text_sims = self._text_candidates(seed_idxs, cfg)

        seed_set = set(seed_idxs)
        seed_keys = set(self.key_list[i] for i in seed_idxs)
        seed_canon = set(self.canonical_keys[i] for i in seed_idxs)
        seed_track_ids = set()
        for i in seed_idxs:
            raw_tid = self.track_id_list[i]
            if pd.isna(raw_tid):
                continue
            seed_track_ids.add(str(raw_tid).strip())

        filtered = []
        for i, ts in zip(cand_idxs, text_sims):
            if i in seed_set:
                continue
            cand_key = self.key_list[i]
            if cand_key in seed_keys:
                continue
            cand_canon = self.canonical_keys[i]
            if cand_canon in seed_canon:
                continue
            raw_tid = self.track_id_list[i]
            cand_tid = "" if pd.isna(raw_tid) else str(raw_tid).strip()
            if cand_tid and cand_tid in seed_track_ids:
                continue
            filtered.append((i, float(ts)))

        genre_col = COL_SUBGENRE if cfg.use_subgenre else COL_GENRE
        selected_genre_norm = str(selected_genre).strip().lower()

        rows = []
        seed_clusters = self.audio_clusters[seed_idxs] if cfg.use_clusters else None
        majority_cluster = None
        if cfg.use_clusters and len(seed_clusters) > 0:
            vals, counts = np.unique(seed_clusters, return_counts=True)
            majority_cluster = vals[np.argmax(counts)]

        assoc_scores = {}
        if cfg.use_assoc:
            for idx in seed_idxs:
                tid = str(self.track_id_list[idx])
                for other_id, conf in self.assoc_rules.get(tid, []):
                    assoc_scores[other_id] = max(assoc_scores.get(other_id, 0.0), conf)

        for idx, text_sim in filtered:
            if cfg.filter_outliers and self.outlier_flags[idx] == -1:
                continue
            audio_sim = float(cosine_similarity(audio_profile, self.audio_matrix[idx:idx+1])[0][0])
            g = str(self.df.iloc[idx][genre_col]).strip().lower()
            bonus = cfg.genre_bonus if g == selected_genre_norm else 0.0
            cluster_bonus = 0.0
            if cfg.use_clusters and majority_cluster is not None and self.audio_clusters[idx] == majority_cluster:
                cluster_bonus = cfg.cluster_bonus

            assoc_bonus = 0.0
            if cfg.use_assoc:
                cand_id = str(self.track_id_list[idx])
                assoc_bonus = assoc_scores.get(cand_id, 0.0) * cfg.assoc_bonus

            score = cfg.alpha * text_sim + (1.0 - cfg.alpha) * audio_sim + bonus + cluster_bonus + assoc_bonus
            rows.append((idx, score, text_sim, audio_sim, bonus, cluster_bonus, assoc_bonus))

        rows.sort(key=lambda x: x[1], reverse=True)

        # diversity rule + de-dup
        result = []
        artist_count: dict[str, int] = {}
        seen_tracks: set[str] = set()
        for idx, score, text_sim, audio_sim, bonus, cluster_bonus, assoc_bonus in rows:
            raw_tid = self.track_id_list[idx]
            track_id = "" if pd.isna(raw_tid) else str(raw_tid).strip()
            track_name = str(self.df.iloc[idx][COL_TRACK_NAME]).strip().lower()
            track_artist = str(self.df.iloc[idx][COL_ARTIST]).strip().lower()
            track_key = self.canonical_keys[idx] or track_id or f"{track_name}|{track_artist}"
            if track_key in seen_tracks:
                continue
            artist = str(self.df.iloc[idx][COL_ARTIST])
            if artist_count.get(artist, 0) >= cfg.max_per_artist:
                continue
            artist_count[artist] = artist_count.get(artist, 0) + 1
            result.append((idx, score, text_sim, audio_sim, bonus, cluster_bonus, assoc_bonus))
            seen_tracks.add(track_key)
            if len(result) >= cfg.K:
                break

        # fallback fill
        if len(result) < cfg.K:
            for idx, score, text_sim, audio_sim, bonus, cluster_bonus, assoc_bonus in rows:
                if any(r[0] == idx for r in result):
                    continue
                raw_tid = self.track_id_list[idx]
                track_id = "" if pd.isna(raw_tid) else str(raw_tid).strip()
                track_name = str(self.df.iloc[idx][COL_TRACK_NAME]).strip().lower()
                track_artist = str(self.df.iloc[idx][COL_ARTIST]).strip().lower()
                track_key = self.canonical_keys[idx] or track_id or f"{track_name}|{track_artist}"
                if track_key in seen_tracks:
                    continue
                result.append((idx, score, text_sim, audio_sim, bonus, cluster_bonus, assoc_bonus))
                seen_tracks.add(track_key)
                if len(result) >= cfg.K:
                    break

        out = []
        for rank, (idx, score, text_sim, audio_sim, bonus, cluster_bonus, assoc_bonus) in enumerate(result, start=1):
            r = self.df.iloc[idx]
            out.append({
                "rank": rank,
                "track_name": r[COL_TRACK_NAME],
                "artist": r[COL_ARTIST],
                "track_artist": r[COL_ARTIST],
                "genre": r[genre_col],
                "playlist_genre": r.get(COL_GENRE, ""),
                "playlist_subgenre": r.get(COL_SUBGENRE, ""),
                "score": score,
                "text_sim": text_sim,
                "audio_sim": audio_sim,
                "genre_bonus": bonus,
                "cluster_bonus": cluster_bonus,
                "assoc_bonus": assoc_bonus,
                "cluster": int(self.audio_clusters[idx]),
                "outlier_flag": int(self.outlier_flags[idx]),
                "outlier_score": float(self.outlier_scores[idx]),
                "track_id": r.get(COL_TRACK_ID, ""),
                "row_idx": idx
            })
        return pd.DataFrame(out)

    def recommend_from_text(self, query_text: str, selected_genre: str, cfg: RecConfig) -> pd.DataFrame:
        if not query_text or not str(query_text).strip():
            raise ValueError("Query text is empty.")

        # For text-only queries, keep scoring text-focused
        if cfg.alpha < 1.0:
            cfg = RecConfig(**{**cfg.__dict__, "alpha": 1.0, "use_clusters": False, "use_assoc": False})

        cand_idxs, text_sims = self._text_candidates_from_query(query_text, cfg)
        q_clean = clean_lyrics(query_text)
        q_set = set(q_clean.split())
        q_size = len(q_set)

        genre_col = COL_SUBGENRE if cfg.use_subgenre else COL_GENRE
        selected_genre_norm = str(selected_genre).strip().lower()

        rows = []
        for idx, text_sim in zip(cand_idxs, text_sims):
            if cfg.filter_outliers and self.outlier_flags[idx] == -1:
                continue
            g = str(self.df.iloc[idx][genre_col]).strip().lower()
            bonus = cfg.genre_bonus if g == selected_genre_norm else 0.0
            jacc_bonus = 0.0
            if q_size > 0 and cfg.model_type.upper() != "JACCARD":
                inter = len(q_set & self.token_sets[idx])
                union = q_size + self.token_set_sizes[idx] - inter
                if union:
                    jacc_bonus = 0.1 * (inter / union)
            score = cfg.alpha * float(text_sim) + bonus + jacc_bonus
            rows.append((idx, score, float(text_sim), bonus))

        rows.sort(key=lambda x: x[1], reverse=True)

        # de-dup and artist diversity
        result = []
        artist_count: dict[str, int] = {}
        seen_tracks: set[str] = set()
        for idx, score, text_sim, bonus in rows:
            raw_tid = self.track_id_list[idx]
            track_id = "" if pd.isna(raw_tid) else str(raw_tid).strip()
            track_key = self.canonical_keys[idx] or track_id
            if track_key in seen_tracks:
                continue
            artist = str(self.df.iloc[idx][COL_ARTIST])
            if artist_count.get(artist, 0) >= cfg.max_per_artist:
                continue
            artist_count[artist] = artist_count.get(artist, 0) + 1
            result.append((idx, score, text_sim, bonus))
            seen_tracks.add(track_key)
            if len(result) >= cfg.K:
                break

        out = []
        for rank, (idx, score, text_sim, bonus) in enumerate(result, start=1):
            r = self.df.iloc[idx]
            out.append({
                "rank": rank,
                "track_name": r[COL_TRACK_NAME],
                "artist": r[COL_ARTIST],
                "track_artist": r[COL_ARTIST],
                "genre": r[genre_col],
                "playlist_genre": r.get(COL_GENRE, ""),
                "playlist_subgenre": r.get(COL_SUBGENRE, ""),
                "score": score,
                "text_sim": text_sim,
                "audio_sim": 0.0,
                "genre_bonus": bonus,
                "cluster_bonus": 0.0,
                "assoc_bonus": 0.0,
                "cluster": int(self.audio_clusters[idx]),
                "outlier_flag": int(self.outlier_flags[idx]),
                "outlier_score": float(self.outlier_scores[idx]),
                "track_id": r.get(COL_TRACK_ID, ""),
                "row_idx": idx
            })
        return pd.DataFrame(out)
