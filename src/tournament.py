"""Tournament utilities for comparing models outside the UI layer."""
from dataclasses import asdict
import time
import numpy as np
import pandas as pd

from src.recommender import RecConfig

def normalize_minmax(values: list[float]) -> list[float]:
    """Normalize a list to [0,1] for fair comparison across models."""
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [0.5 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]

def playlist_relevance(df: pd.DataFrame, seed_idxs: list[int], rec_idxs: list[int]) -> dict:
    """Proxy relevance: overlap of playlist_id between seeds and recs."""
    if not rec_idxs or "playlist_id" not in df.columns:
        return {"precision": 0.0, "hit_rate": 0.0}
    seed_playlists = set(df.loc[seed_idxs, "playlist_id"].dropna().astype(str).tolist())
    if not seed_playlists:
        return {"precision": 0.0, "hit_rate": 0.0}
    rec_playlists = df.loc[rec_idxs, "playlist_id"].dropna().astype(str)
    hits = rec_playlists.isin(seed_playlists).sum()
    precision = hits / len(rec_idxs)
    hit_rate = 1.0 if hits > 0 else 0.0
    return {"precision": float(precision), "hit_rate": float(hit_rate)}

def novelty(df: pd.DataFrame, rec_idxs: list[int]) -> float:
    """Higher novelty favors less‑popular tracks."""
    if not rec_idxs or "track_popularity" not in df.columns:
        return 0.0
    pop = pd.to_numeric(df.loc[rec_idxs, "track_popularity"], errors="coerce").fillna(0.0)
    return float((1.0 - (pop / 100.0)).mean())

def build_tournament_candidates(k: int) -> list[RecConfig]:
    """Default model configs used in the tournament."""
    base = dict(
        genre_bonus=0.02,
        cluster_bonus=0.02,
        assoc_bonus=0.05,
        max_per_artist=2,
        K=k,
        use_subgenre=False,
        use_clusters=True,
        use_assoc=True,
        auto_alpha=True,
        out_of_genre_ratio=0.30,
        adaptive_genre_bonus=True,
        genre_penalty=0.03,
        filter_outliers=True,
    )
    return [
        RecConfig(model_type="SBERT_COSINE", alpha=0.8, **base),
        RecConfig(model_type="TFIDF_COSINE", alpha=0.9, **base),
        RecConfig(model_type="W2V_COSINE", alpha=0.85, **base),
        RecConfig(model_type="BOW_COSINE", alpha=0.9, **base),
        RecConfig(model_type="JACCARD", alpha=0.7, **base),
        RecConfig(model_type="SBERT_EUCLIDEAN", alpha=0.8, **base),
    ]

def run_seed_tournament(rec, selected_genre: str, seeds: list[str], k: int):
    """Run all models on seed tracks and return a ranked scoreboard."""
    candidates = build_tournament_candidates(k)
    results = []
    seed_idxs = rec._seed_indices(seeds) if seeds else []

    for cfg in candidates:
        t0 = time.time()
        df_out = rec.recommend(selected_genre=selected_genre, seeds_title_artist=seeds, cfg=cfg)
        latency = time.time() - t0

        rec_idxs = df_out["row_idx"].tolist() if "row_idx" in df_out.columns else []
        mean_score = float(df_out["score"].mean()) if "score" in df_out.columns and not df_out.empty else 0.0
        diversity = df_out["track_artist"].nunique() / len(df_out) if "track_artist" in df_out.columns and len(df_out) else 0.0
        nov = novelty(rec.df, rec_idxs)
        rel = playlist_relevance(rec.df, seed_idxs, rec_idxs)

        if "score" in df_out.columns and not df_out.empty:
            norm_scores = normalize_minmax(df_out["score"].tolist())
            norm_mean = float(np.mean(norm_scores)) if norm_scores else 0.0
        else:
            norm_mean = 0.0

        composite = 0.45 * norm_mean + 0.35 * rel["precision"] + 0.10 * diversity + 0.10 * nov

        results.append({
            "model": cfg.model_type,
            "cfg": asdict(cfg),
            "df_out": df_out,
            "composite": composite,
            "norm_mean_score": norm_mean,
            "playlist_precision": rel["precision"],
            "playlist_hit_rate": rel["hit_rate"],
            "diversity": diversity,
            "novelty": nov,
            "mean_score_raw": mean_score,
            "latency_s": latency,
        })

    results_sorted = sorted(results, key=lambda x: x["composite"], reverse=True)
    best = results_sorted[0] if results_sorted else None
    scoreboard = pd.DataFrame([
        {
            "model": r["model"],
            "composite": r["composite"],
            "norm_mean_score": r["norm_mean_score"],
            "playlist_precision": r["playlist_precision"],
            "playlist_hit_rate": r["playlist_hit_rate"],
            "diversity": r["diversity"],
            "novelty": r["novelty"],
            "latency_s": r["latency_s"],
        }
        for r in results_sorted
    ])
    return {"results": results_sorted, "best": best, "scoreboard": scoreboard}

def choose_text_model(query_text: str, lyrics_mode: bool) -> str:
    """Pick a text model based on prompt length."""
    words = [w for w in str(query_text).split() if w.strip()]
    use_tfidf = lyrics_mode or len(words) >= 20
    return "TFIDF_COSINE" if use_tfidf else "SBERT_COSINE"

def run_text_prompt(rec, selected_genre: str, query_text: str, k: int, lyrics_mode: bool):
    """Text‑prompt path (no tournament)."""
    model_type = choose_text_model(query_text, lyrics_mode)
    cfg = RecConfig(
        model_type=model_type,
        alpha=1.0,
        genre_bonus=0.02,
        cluster_bonus=0.0,
        assoc_bonus=0.0,
        max_per_artist=2,
        K=k,
        use_subgenre=False,
        use_clusters=False,
        use_assoc=False,
        filter_outliers=True,
    )
    t0 = time.time()
    df_out = rec.recommend_from_text(
        query_text=query_text,
        selected_genre=selected_genre,
        cfg=cfg,
    )
    latency = time.time() - t0
    return {"df_out": df_out, "model": model_type, "latency_s": latency}
