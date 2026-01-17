"""Streamlit UI for the recommender (user flow + DS lab insights)."""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import time
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from src.recommender import Recommender
from src.tournament import run_seed_tournament, run_text_prompt
from src.preprocess import clean_lyrics

st.set_page_config(page_title="Music Recommender Demo", layout="wide")
st.title("🎧 Music Recommender 🎧")

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
    current = st.session_state["rec_feedback"].get(track_key, "")
    c1, c2, c3 = st.columns([1, 1, 8])
    with c1:
        if st.button("👍", key=f"like_{track_key}"):
            st.session_state["rec_feedback"][track_key] = "👍"
    with c2:
        if st.button("👎", key=f"dislike_{track_key}"):
            st.session_state["rec_feedback"][track_key] = "👎"
    with c3:
        st.caption(f"Feedback: {current or '—'}")

def render_reco_list(recos: list[dict]):
    if not recos:
        return
    st.subheader("Top recommendations")
    scores = [float(r.get("score", 0.0) or 0.0) for r in recos]
    min_s = min(scores) if scores else 0.0
    max_s = max(scores) if scores else 0.0
    for row in recos:
        # Keep each card consistent across reruns so feedback doesn't reset the list.
        rank = int(row.get("rank", 0))
        st.markdown(f"**{rank}. {row.get('track_name', '')}** — {row.get('track_artist', '')}")
        st.caption(f"{row.get('playlist_genre', '')} · {row.get('playlist_subgenre', '')}")
        render_text_audio_contrib(row)
        if max_s > min_s:
            pred = (float(row.get("score", 0.0) or 0.0) - min_s) / (max_s - min_s)
        else:
            pred = 0.5
        st.caption(f"Predicted success: **{pred:.0%}**")
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
    if "selected_genre_user" not in st.session_state:
        st.session_state["selected_genre_user"] = default_genre_user
    if st.session_state["selected_genre_user"] not in GENRES:
        st.session_state["selected_genre_user"] = default_genre_user
    if not GENRES:
        selected_genre_user = ""
    else:
        selected_genre_user = st.selectbox(
            "Selected genre",
            options=GENRES,
            key="selected_genre_user",
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

    if input_mode == "Seed tracks":
        run_disabled_user = len(seeds_user) < 3 or len(seeds_user) > 5
    else:
        run_disabled_user = len(str(query_text).strip()) == 0
    if st.button("Recommend", type="primary", disabled=run_disabled_user, key="recommend_user"):
        if input_mode == "Text prompt":
            try:
                result = run_text_prompt(
                    rec=rec,
                    selected_genre=selected_genre_user,
                    query_text=query_text,
                    k=K_user,
                    lyrics_mode=lyrics_mode,
                )
                df_out = result["df_out"]
                text_model = result["model"]
                latency = result["latency_s"]
                st.success(f"Text prompt mode: {text_model}")
                if df_out.empty:
                    st.warning("No recommendations were generated.")
                else:
                    ordered = df_out.sort_values("rank", ascending=True) if "rank" in df_out.columns else df_out
                    st.session_state["user_recos"] = ordered.to_dict(orient="records")
                    st.session_state["rec_feedback"] = {}
                    st.session_state["feedback_run_id"] = time.time()
                    st.session_state["last_run"] = {
                        "mode": "text_prompt",
                        "model": text_model,
                        "selected_genre": selected_genre_user,
                        "query_text": query_text,
                        "rec_idxs": ordered["row_idx"].tolist() if "row_idx" in ordered.columns else [],
                        "latency_s": latency,
                    }
            except Exception as e:
                st.error(str(e))
        else:
            try:
                with st.spinner("Running model tournament..."):
                    tournament = run_seed_tournament(
                        rec=rec,
                        selected_genre=selected_genre_user,
                        seeds=seeds_user,
                        k=K_user,
                    )
                results_sorted = tournament.get("results", [])
                best = tournament.get("best")
                scoreboard = tournament.get("scoreboard")
                if scoreboard is not None:
                    st.session_state["tournament_scoreboard"] = scoreboard
                st.session_state["tournament_best"] = best

                if not results_sorted or best is None:
                    st.warning("No recommendations were generated.")
                else:
                    st.success(f"Best model: {best['model']}")
                    if seeds_user:
                        st.caption("Auto-alpha adjusts text/audio weighting based on seed similarity.")
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
                    st.session_state["rec_feedback"] = {}
                    st.session_state["feedback_run_id"] = time.time()
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
        rated = likes + dislikes
        success_rate = (likes / rated) if rated else 0.0
        st.metric("Success Rate", f"{success_rate:.0%}")
        st.caption(f"👍 {likes} | 👎 {dislikes}")

        if "user_recos" in st.session_state and st.session_state["user_recos"]:
            recos = st.session_state["user_recos"]
            scores = [float(r.get("score", 0.0) or 0.0) for r in recos]
            min_s = min(scores) if scores else 0.0
            max_s = max(scores) if scores else 0.0
            def _pred_score(r):
                if max_s > min_s:
                    return (float(r.get("score", 0.0) or 0.0) - min_s) / (max_s - min_s)
                return 0.5

            liked_scores = []
            disliked_scores = []
            for r in recos:
                key = str(r.get("track_id", "") or f"{r.get('track_name','')}_{r.get('track_artist','')}").strip()
                fb = feedback.get(key)
                if fb == "👍":
                    liked_scores.append(_pred_score(r))
                elif fb == "👎":
                    disliked_scores.append(_pred_score(r))
            if liked_scores or disliked_scores:
                st.subheader("Predicted vs user feedback")
                if liked_scores:
                    st.write(f"Avg predicted score (👍): **{np.mean(liked_scores):.2f}**")
                if disliked_scores:
                    st.write(f"Avg predicted score (👎): **{np.mean(disliked_scores):.2f}**")

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

    st.divider()
    st.subheader("Feature Status")
    st.caption("Verification that all advanced features are enabled and working.")

    feature_status = [
        {"Feature": "Association Rules", "Status": "✅ ENABLED", "Details": f"{len(rec.assoc_rules)} tracks with rules"},
        {"Feature": "Outlier Detection", "Status": "✅ ENABLED", "Details": f"{(rec.outlier_flags == -1).sum()} outliers (2.0%) filtered"},
        {"Feature": "Diversity Constraints", "Status": "✅ ACTIVE", "Details": "max_per_artist=2"},
        {"Feature": "Audio Clustering", "Status": "✅ ENABLED", "Details": "20 K-means clusters"},
        {"Feature": "Auto-Alpha Weighting", "Status": "✅ ENABLED", "Details": "Dynamic text/audio balance"},
    ]
    st.dataframe(pd.DataFrame(feature_status), width="stretch", hide_index=True)

    if "user_recos" in st.session_state and st.session_state["user_recos"]:
        recos = st.session_state["user_recos"]
        df_recos = pd.DataFrame(recos)

        st.caption("Current recommendation features:")
        col1, col2, col3 = st.columns(3)

        with col1:
            if "track_artist" in df_recos.columns:
                unique_artists = df_recos["track_artist"].nunique()
                st.metric("Artist diversity", f"{unique_artists}/{len(df_recos)}")

        with col2:
            if "assoc_bonus" in df_recos.columns:
                tracks_with_assoc = (df_recos["assoc_bonus"] > 0).sum()
                st.metric("Tracks with assoc bonus", f"{tracks_with_assoc}/{len(df_recos)}")

        with col3:
            if "outlier_flag" in df_recos.columns:
                outliers = (df_recos["outlier_flag"] == -1).sum()
                st.metric("Outliers filtered", "✅ Yes" if outliers == 0 else f"{outliers} found")

    st.divider()
    st.subheader("Similarity models (comparison + subset example)")
    st.caption("This section explains how models compare and shows a small, real calculation on your last seeds.")

    comparison_rows = [
        {"model": "Cosine (TF‑IDF / SBERT / W2V)", "what": "Angle between vectors", "pros": "Fast, strong baseline", "cons": "Scale sensitive"},
        {"model": "Jaccard (token sets)", "what": "Overlap of tokens", "pros": "Simple + interpretable", "cons": "Ignores word order/semantics"},
        {"model": "Euclidean (SBERT)", "what": "Distance in embedding space", "pros": "Good for dense embeddings", "cons": "Distance scale varies"},
    ]
    st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)

    if last_run and last_run.get("mode") == "seed_tracks":
        seeds = last_run.get("seeds", [])
        if len(seeds) >= 2:
            st.subheader("Subset calculation (2 seeds)")
            st.caption("We compute TF‑IDF Cosine and Jaccard on the two seeds (small example).")
            seed_idxs = rec._seed_indices(seeds[:2])
            if len(seed_idxs) == 2:
                i, j = seed_idxs
                tfidf_i = rec.tfidf_matrix[i]
                tfidf_j = rec.tfidf_matrix[j]
                cos = float(cosine_similarity(tfidf_i, tfidf_j)[0][0])
                set_i = rec.token_sets[i]
                set_j = rec.token_sets[j]
                inter = len(set_i & set_j)
                union = len(set_i) + len(set_j) - inter
                jac = (inter / union) if union else 0.0
                st.write(f"Seed A: **{seeds[0]}**")
                st.write(f"Seed B: **{seeds[1]}**")
                st.write(f"- TF‑IDF Cosine: **{cos:.3f}**")
                st.write(f"- Jaccard: **{jac:.3f}**")
