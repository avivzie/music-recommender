"""Streamlit UI for the recommender (user flow + DS lab insights)."""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import time
from sklearn.manifold import TSNE
from src.recommender import Recommender, RecConfig
from src.evaluation import (
    EvalConfig,
)
from src.preprocess import clean_lyrics

st.set_page_config(page_title="Music Recommender Demo", layout="wide")
st.title("🎧 Music Recommender (Lyrics + Audio)")

@st.cache_resource
def load_model():
    return Recommender()

rec = load_model()

# ---------- UI helpers ----------
@st.cache_data
def build_ui_catalog(df: pd.DataFrame):
    df2 = df.copy()
    df2["track_option"] = df2["track_name"].fillna("") + " — " + df2["track_artist"].fillna("")
    genres = sorted(df2["playlist_genre"].dropna().unique().tolist())

    # demo presets = top popular per genre (if exists)
    demos = {}
    demo_limit = 4
    for g in ["pop", "rock", "rap"]:
        if g in genres:
            demos[g] = (
                df2[df2["playlist_genre"] == g]
                .sort_values("track_popularity", ascending=False)["track_option"]
                .dropna().unique().tolist()[:demo_limit]
            )
        else:
            demos[g] = []
    return df2, genres, demos

df_ui, GENRES, DEMOS = build_ui_catalog(rec.df)

def render_text_audio_contrib(row):
    text_sim = float(row.get("text_sim", 0.0) or 0.0)
    audio_sim = float(row.get("audio_sim", 0.0) or 0.0)
    text_sim = max(text_sim, 0.0)
    audio_sim = max(audio_sim, 0.0)
    total = text_sim + audio_sim
    if total <= 0:
        st.caption("Similarity mix: text 0% · audio 0%")
        return
    text_pct = text_sim / total
    audio_pct = audio_sim / total
    st.caption(f"Similarity mix: text {text_pct:.0%} · audio {audio_pct:.0%}")
    st.progress(text_pct, text="Text similarity")
    st.progress(audio_pct, text="Audio similarity")

def render_feedback_controls(track_key: str):
    if "rec_feedback" not in st.session_state:
        st.session_state["rec_feedback"] = {}
    current = st.session_state["rec_feedback"].get(track_key, "😐")
    choice = st.radio(
        "Your reaction",
        ["👍", "😐", "👎"],
        index=["👍", "😐", "👎"].index(current),
        horizontal=True,
        key=f"feedback_{track_key}",
        label_visibility="collapsed"
    )
    st.session_state["rec_feedback"][track_key] = choice

def render_reco_list(recos: list[dict]):
    if not recos:
        return
    st.subheader("Top recommendations")
    for row in recos:
        # Keep each card consistent across reruns so feedback doesn't reset the list.
        rank = int(row.get("rank", 0))
        st.markdown(f"**{rank}. {row.get('track_name', '')}** — {row.get('track_artist', '')}")
        st.caption(f"{row.get('playlist_genre', '')} · {row.get('playlist_subgenre', '')}")
        render_text_audio_contrib(row)
        feedback_key = str(row.get("track_id", "") or f"{row.get('track_name', '')}_{row.get('track_artist', '')}").strip()
        render_feedback_controls(feedback_key)
        track_id = str(row.get("track_id", "")).strip()
        if track_id:
            embed_html = (
                '<iframe style="border-radius:12px" '
                f'src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator" '
                'width="100%" height="152" frameBorder="0" allowfullscreen="" '
                'allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" '
                'loading="lazy"></iframe>'
            )
            components.html(embed_html, height=170)
        st.divider()

tab_user, tab_rec = st.tabs(["Recommend", "DS Lab"])

# =========================================================
# User tab (simple)
# =========================================================
with tab_user:
    st.subheader("Get recommendations")
    # User-facing flow: pick seeds or describe a vibe, then show top tracks.

    input_mode = st.radio(
        "Input type",
        ["Seed tracks", "Text prompt"],
        horizontal=True
    )

    c1, c2, c3, _ = st.columns([1, 1, 1, 6])
    if c1.button("Load demo: Pop", key="demo_user_pop"):
        st.session_state["selected_genre_user"] = "pop"
        st.session_state["seed_tracks_user"] = DEMOS.get("pop", [])
        st.rerun()
    if c2.button("Load demo: Rock", key="demo_user_rock"):
        st.session_state["selected_genre_user"] = "rock"
        st.session_state["seed_tracks_user"] = DEMOS.get("rock", [])
        st.rerun()
    if c3.button("Load demo: Rap", key="demo_user_rap"):
        st.session_state["selected_genre_user"] = "rap"
        st.session_state["seed_tracks_user"] = DEMOS.get("rap", [])
        st.rerun()

    default_genre_user = "pop" if "pop" in GENRES else (GENRES[0] if GENRES else "")
    if "selected_genre_user" in st.session_state and st.session_state["selected_genre_user"] not in GENRES:
        st.session_state["selected_genre_user"] = default_genre_user
    selected_genre_user = st.selectbox(
        "Selected genre",
        options=GENRES,
        index=GENRES.index(default_genre_user) if default_genre_user in GENRES else 0,
        key="selected_genre_user"
    )

    df_pool_user = df_ui[df_ui["playlist_genre"] == selected_genre_user].copy()
    track_options_user = sorted(df_pool_user["track_option"].dropna().unique().tolist())

    if "seed_tracks_user" not in st.session_state:
        st.session_state["seed_tracks_user"] = []
    current_seeds_user = st.session_state.get("seed_tracks_user", [])
    filtered_seeds_user = [x for x in current_seeds_user if x in track_options_user]
    if filtered_seeds_user != current_seeds_user:
        st.session_state["seed_tracks_user"] = filtered_seeds_user

    seeds_user = []
    query_text = ""
    if input_mode == "Seed tracks":
        st.caption("Pick 3–5 songs you already like.")
        seeds_user = st.multiselect(
            "Seed tracks (choose 3–5)",
            options=track_options_user,
            key="seed_tracks_user",
            help="Search by track or artist. Format: track_name — track_artist"
        )
    else:
        st.caption("Describe the mood or theme in free text.")
        query_text = st.text_area(
            "Describe the vibe you want",
            placeholder="I feel nostalgic and want upbeat classic rock with uplifting lyrics..."
        )
        lyrics_mode = st.checkbox("Lyrics match (TF‑IDF)", value=False)

    K_user = st.slider("How many recommendations?", 5, 15, 10, 1)

    if input_mode == "Seed tracks" and seeds_user:
        st.subheader("Your selected songs")
        track_id_map = (
            df_pool_user.dropna(subset=["track_id"])
            .drop_duplicates(subset=["track_option"])
            .set_index("track_option")["track_id"]
            .astype(str)
            .to_dict()
        )
        cols = st.columns(2)
        for i, seed in enumerate(seeds_user):
            track_id = track_id_map.get(seed, "")
            if not track_id:
                continue
            embed_html = (
                '<iframe style="border-radius:12px" '
                f'src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator" '
                'width="100%" height="152" frameBorder="0" allowfullscreen="" '
                'allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" '
                'loading="lazy"></iframe>'
            )
            with cols[i % 2]:
                components.html(embed_html, height=170)

    def _normalize_minmax(values: list[float]) -> list[float]:
        if not values:
            return []
        vmin, vmax = min(values), max(values)
        if vmax == vmin:
            return [0.5 for _ in values]
        return [(v - vmin) / (vmax - vmin) for v in values]

    def _playlist_relevance(rec_df: pd.DataFrame, seed_idxs: list[int], rec_idxs: list[int]) -> dict:
        if not rec_idxs or "playlist_id" not in rec_df.columns:
            return {"precision": 0.0, "hit_rate": 0.0}
        seed_playlists = set(
            rec_df.loc[seed_idxs, "playlist_id"].dropna().astype(str).tolist()
        )
        if not seed_playlists:
            return {"precision": 0.0, "hit_rate": 0.0}
        rec_playlists = rec_df.loc[rec_idxs, "playlist_id"].dropna().astype(str)
        hits = rec_playlists.isin(seed_playlists).sum()
        precision = hits / len(rec_idxs)
        hit_rate = 1.0 if hits > 0 else 0.0
        return {"precision": float(precision), "hit_rate": float(hit_rate)}

    def _novelty(rec_df: pd.DataFrame, rec_idxs: list[int]) -> float:
        if not rec_idxs or "track_popularity" not in rec_df.columns:
            return 0.0
        pop = pd.to_numeric(rec_df.loc[rec_idxs, "track_popularity"], errors="coerce").fillna(0.0)
        return float((1.0 - (pop / 100.0)).mean())

    if input_mode == "Seed tracks":
        run_disabled_user = len(seeds_user) < 3 or len(seeds_user) > 5
    else:
        run_disabled_user = len(str(query_text).strip()) == 0
    if st.button("Recommend", type="primary", disabled=run_disabled_user, key="recommend_user"):
        if input_mode == "Text prompt":
            start_ts = time.time()
            words = [w for w in str(query_text).split() if w.strip()]
            use_tfidf = lyrics_mode or len(words) >= 20
            text_model = "TFIDF_COSINE" if use_tfidf else "SBERT_COSINE"
            cfg_text = RecConfig(
                model_type=text_model,
                alpha=1.0,
                genre_bonus=0.02,
                cluster_bonus=0.0,
                assoc_bonus=0.0,
                max_per_artist=2,
                K=K_user,
                use_subgenre=False,
                use_clusters=False,
                use_assoc=False,
                filter_outliers=False
            )
            try:
                # Short prompts work better with SBERT; long lyrics favor TF‑IDF.
                df_out = rec.recommend_from_text(
                    query_text=query_text,
                    selected_genre=selected_genre_user,
                    cfg=cfg_text
                )
                st.success(f"Text prompt mode: {text_model}")
                if df_out.empty:
                    st.warning("No recommendations were generated.")
                else:
                    ordered = df_out.sort_values("rank", ascending=True) if "rank" in df_out.columns else df_out
                    st.session_state["user_recos"] = ordered.to_dict(orient="records")
                    st.session_state["last_run"] = {
                        "mode": "text_prompt",
                        "model": text_model,
                        "selected_genre": selected_genre_user,
                        "query_text": query_text,
                        "rec_idxs": ordered["row_idx"].tolist() if "row_idx" in ordered.columns else [],
                        "latency_s": time.time() - start_ts,
                    }
            except Exception as e:
                st.error(str(e))
        else:
            candidates = [
                RecConfig(model_type="SBERT_COSINE", alpha=0.8, genre_bonus=0.02, cluster_bonus=0.02, assoc_bonus=0.05,
                          max_per_artist=2, K=K_user, use_subgenre=False, use_clusters=True, use_assoc=False,
                          auto_alpha=True, out_of_genre_ratio=0.30, adaptive_genre_bonus=True, genre_penalty=0.03),
                RecConfig(model_type="TFIDF_COSINE", alpha=0.9, genre_bonus=0.02, cluster_bonus=0.02, assoc_bonus=0.05,
                          max_per_artist=2, K=K_user, use_subgenre=False, use_clusters=True, use_assoc=False,
                          auto_alpha=True, out_of_genre_ratio=0.30, adaptive_genre_bonus=True, genre_penalty=0.03),
                RecConfig(model_type="W2V_COSINE", alpha=0.85, genre_bonus=0.02, cluster_bonus=0.02, assoc_bonus=0.05,
                          max_per_artist=2, K=K_user, use_subgenre=False, use_clusters=True, use_assoc=False,
                          auto_alpha=True, out_of_genre_ratio=0.30, adaptive_genre_bonus=True, genre_penalty=0.03),
                RecConfig(model_type="BOW_COSINE", alpha=0.9, genre_bonus=0.02, cluster_bonus=0.02, assoc_bonus=0.05,
                          max_per_artist=2, K=K_user, use_subgenre=False, use_clusters=True, use_assoc=False,
                          auto_alpha=True, out_of_genre_ratio=0.30, adaptive_genre_bonus=True, genre_penalty=0.03),
                RecConfig(model_type="JACCARD", alpha=0.7, genre_bonus=0.02, cluster_bonus=0.02, assoc_bonus=0.05,
                          max_per_artist=2, K=K_user, use_subgenre=False, use_clusters=True, use_assoc=False,
                          auto_alpha=True, out_of_genre_ratio=0.30, adaptive_genre_bonus=True, genre_penalty=0.03),
                RecConfig(model_type="SBERT_EUCLIDEAN", alpha=0.8, genre_bonus=0.02, cluster_bonus=0.02, assoc_bonus=0.05,
                          max_per_artist=2, K=K_user, use_subgenre=False, use_clusters=True, use_assoc=False,
                          auto_alpha=True, out_of_genre_ratio=0.30, adaptive_genre_bonus=True, genre_penalty=0.03),
            ]

            progress = st.progress(0, text="Running model tournament...")
            results = []

            try:
                # Tournament: run all models, then pick the best composite score.
                seed_idxs_user = rec._seed_indices(seeds_user) if seeds_user else []
                for i, cfg in enumerate(candidates, start=1):
                    t0 = time.time()
                    df_out = rec.recommend(
                        selected_genre=selected_genre_user,
                        seeds_title_artist=seeds_user,
                        cfg=cfg
                    )
                    latency = time.time() - t0
                    rec_idxs = df_out["row_idx"].tolist() if "row_idx" in df_out.columns else []
                    mean_score = float(df_out["score"].mean()) if "score" in df_out.columns and not df_out.empty else 0.0
                    diversity = df_out["track_artist"].nunique() / len(df_out) if "track_artist" in df_out.columns and len(df_out) else 0.0
                    novelty = _novelty(rec.df, rec_idxs)
                    playlist_rel = _playlist_relevance(rec.df, seed_idxs_user, rec_idxs)

                    # Normalize within-model scores (min-max) for fair comparison
                    if "score" in df_out.columns and not df_out.empty:
                        norm_scores = _normalize_minmax(df_out["score"].tolist())
                        norm_mean = float(np.mean(norm_scores)) if norm_scores else 0.0
                    else:
                        norm_mean = 0.0

                    # Composite: normalized sim + playlist proxy + diversity + novelty
                    composite = (
                        0.45 * norm_mean
                        + 0.35 * playlist_rel["precision"]
                        + 0.10 * diversity
                        + 0.10 * novelty
                    )

                    results.append({
                        "model": cfg.model_type,
                        "df_out": df_out,
                        "composite": composite,
                        "norm_mean_score": norm_mean,
                        "playlist_precision": playlist_rel["precision"],
                        "playlist_hit_rate": playlist_rel["hit_rate"],
                        "diversity": diversity,
                        "novelty": novelty,
                        "mean_score_raw": mean_score,
                        "latency_s": latency,
                    })
                    progress.progress(i / len(candidates), text=f"Evaluating {cfg.model_type}...")

                if not results:
                    st.warning("No recommendations were generated.")
                else:
                    results_sorted = sorted(results, key=lambda x: x["composite"], reverse=True)
                    best = results_sorted[0]
                    st.success(f"Best model: {best['model']}")
                    if seeds_user:
                        st.caption("Auto-alpha adjusts text/audio weighting based on seed similarity.")

                    scoreboard = pd.DataFrame([
                        {
                            "model": r["model"],
                            "composite": r["composite"],
                            "norm_mean_score": r["norm_mean_score"],
                            "playlist_precision": r["playlist_precision"],
                            "playlist_hit_rate": r["playlist_hit_rate"],
                            "diversity": r["diversity"],
                            "novelty": r["novelty"],
                        }
                        for r in results_sorted
                    ])
                    st.session_state["tournament_scoreboard"] = scoreboard
                    st.session_state["tournament_best"] = best
                    best_out = best["df_out"]
                    st.session_state["last_run"] = {
                        "mode": "seed_tracks",
                        "model": best["model"],
                        "selected_genre": selected_genre_user,
                        "seeds": seeds_user,
                        "rec_idxs": best_out["row_idx"].tolist() if best_out is not None and "row_idx" in best_out.columns else [],
                        "latency_s": best.get("latency_s", 0.0),
                        "model_latencies": {r["model"]: r.get("latency_s", 0.0) for r in results_sorted},
                    }
                    if best_out is None or best_out.empty:
                        st.warning("No recommendations were generated.")
                    else:
                        ordered = best_out.sort_values("rank", ascending=True) if "rank" in best_out.columns else best_out
                        st.session_state["user_recos"] = ordered.to_dict(orient="records")
            except Exception as e:
                st.error(str(e))

    if "user_recos" in st.session_state and st.session_state["user_recos"]:
        render_reco_list(st.session_state["user_recos"])

# =========================================================
# Recommend tab (DS)
# =========================================================
with tab_rec:
    st.subheader("DS Lab")

    st.divider()
    st.subheader("Model tournament (background process)")
    with st.expander("What you’re seeing here"):
        st.markdown(
            "- **Tournament scoreboard**: models compete on the same seeds; the best composite score wins.\n"
            "- **Run Insights**: shows what happened in the last user run (latency, success rate, embedding map).\n"
            "- **Embedding map**: PCA/t‑SNE view of seed vs recommendation positions in SBERT space.\n"
            "- **Debug NLP**: shows cleaned text so you can verify preprocessing."
        )
    if "tournament_scoreboard" in st.session_state:
        scoreboard = st.session_state["tournament_scoreboard"]
        st.dataframe(scoreboard, width="stretch", hide_index=True)
        with st.expander("How the winner is chosen"):
            st.markdown(
                "- **Composite score** blends normalized similarity, playlist proxy, diversity, and novelty.\n"
                "- Models with higher composite scores win and are shown to the user.\n"
                "- Latency is shown below for transparency."
            )
    else:
        st.info("Run a recommendation in the Recommend tab to see model competition.")

    st.divider()
    st.subheader("Run Insights (latest Recommend run)")
    last_run = st.session_state.get("last_run")
    if not last_run:
        st.info("Run a recommendation in the Recommend tab to populate insights here.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mode", last_run.get("mode", "-"))
        c2.metric("Model", last_run.get("model", "-"))
        c3.metric("Latency (s)", f"{last_run.get('latency_s', 0.0):.2f}")

        feedback = st.session_state.get("rec_feedback", {})
        likes = sum(1 for v in feedback.values() if v == "👍")
        dislikes = sum(1 for v in feedback.values() if v == "👎")
        neutrals = sum(1 for v in feedback.values() if v == "😐")
        rated = likes + dislikes
        success_rate = (likes / rated) if rated else 0.0
        st.metric("Success Rate", f"{success_rate:.0%}")
        st.caption(f"👍 {likes} | 😐 {neutrals} | 👎 {dislikes}")

        rec_idxs = last_run.get("rec_idxs", [])
        if rec_idxs:
            st.subheader("Embedding map (SBERT)")
            method = st.selectbox("Projection method", ["PCA", "t-SNE"], index=0)
            points = rec.sbert_embeddings[rec_idxs]
            pca_df = pd.DataFrame(points, columns=[f"d{i}" for i in range(points.shape[1])])
            pca_df["label"] = "recommendation"
            if last_run.get("mode") == "seed_tracks":
                seed_idxs = rec._seed_indices(last_run.get("seeds", []))
                if seed_idxs:
                    seed_pts = rec.sbert_embeddings[seed_idxs]
                    seed_df = pd.DataFrame(seed_pts, columns=pca_df.columns[:-1])
                    seed_df["label"] = "seed"
                    pca_df = pd.concat([pca_df, seed_df], ignore_index=True)

            # downsample for t-SNE
            if method == "t-SNE" and len(pca_df) > 200:
                pca_df = pca_df.sample(200, random_state=42).reset_index(drop=True)

            X = pca_df.drop(columns=["label"]).to_numpy()
            if method == "PCA":
                X = X - X.mean(axis=0, keepdims=True)
                U, S, Vt = np.linalg.svd(X, full_matrices=False)
                coords = U[:, :2] * S[:2]
            else:
                perplexity = max(2, min(20, (len(X) - 1) // 3))
                coords = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(X)

            pca_df["x"] = coords[:, 0]
            pca_df["y"] = coords[:, 1]
            fig = px.scatter(pca_df, x="x", y="y", color="label", title=f"{method} projection")
            st.plotly_chart(fig, width="stretch")

        st.subheader("Latency + model comparison (latest tournament)")
        if last_run.get("mode") == "seed_tracks" and "model_latencies" in last_run:
            lat_df = pd.DataFrame(
                [{"model": k, "latency_s": v} for k, v in last_run["model_latencies"].items()]
            ).sort_values("latency_s")
            st.dataframe(lat_df, width="stretch", hide_index=True)
            fig_lat = px.bar(lat_df, x="model", y="latency_s", title="Latency per model")
            st.plotly_chart(fig_lat, width="stretch")
        else:
            st.caption("Latency table is available for seed-track tournament runs.")

        st.subheader("Debug NLP (cleaned text)")
        if last_run.get("mode") == "text_prompt":
            cleaned = clean_lyrics(last_run.get("query_text", ""))
            st.code(cleaned, language="text")
        else:
            seeds = last_run.get("seeds", [])
            if seeds:
                df_lookup = rec.df.copy()
                df_lookup["track_option"] = df_lookup["track_name"].fillna("") + " — " + df_lookup["track_artist"].fillna("")
                samples = df_lookup[df_lookup["track_option"].isin(seeds)]
                if "lyrics_clean" in samples.columns:
                    st.dataframe(samples[["track_option", "lyrics_clean"]], width="stretch", hide_index=True)
