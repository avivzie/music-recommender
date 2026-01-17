import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from src.recommender import Recommender, RecConfig
from src.evaluation import (
    generate_queries_df,
    evaluate_configs,
    evaluate_baselines,
    build_ablation_configs,
    EvalConfig,
)

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
    total = text_sim + audio_sim
    if total <= 0:
        st.caption("Similarity mix: text 0% · audio 0%")
        return
    text_pct = text_sim / total
    audio_pct = audio_sim / total
    st.caption(f"Similarity mix: text {text_pct:.0%} · audio {audio_pct:.0%}")
    st.progress(text_pct, text="Text similarity")
    st.progress(audio_pct, text="Audio similarity")

tab_user, tab_rec = st.tabs(["Recommend", "DS Lab"])

# =========================================================
# User tab (simple)
# =========================================================
with tab_user:
    st.subheader("Get recommendations")

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
                df_out = rec.recommend_from_text(
                    query_text=query_text,
                    selected_genre=selected_genre_user,
                    cfg=cfg_text
                )
                st.success(f"Text prompt mode: {text_model}")
                if df_out.empty:
                    st.warning("No recommendations were generated.")
                else:
                    st.subheader("Top recommendations")
                    ordered = df_out.sort_values("rank", ascending=True) if "rank" in df_out.columns else df_out
                    for _, row in ordered.iterrows():
                        rank = int(row["rank"]) if "rank" in row else 0
                        st.markdown(f"**{rank}. {row['track_name']}** — {row['track_artist']}")
                        st.caption(f"{row.get('playlist_genre', '')} · {row.get('playlist_subgenre', '')}")
                        render_text_audio_contrib(row)
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
            except Exception as e:
                st.error(str(e))
        else:
            candidates = [
                RecConfig(model_type="SBERT_COSINE", alpha=0.8, genre_bonus=0.02, cluster_bonus=0.02, assoc_bonus=0.05,
                          max_per_artist=2, K=K_user, use_subgenre=False, use_clusters=True, use_assoc=False, auto_alpha=True),
                RecConfig(model_type="TFIDF_COSINE", alpha=0.9, genre_bonus=0.02, cluster_bonus=0.02, assoc_bonus=0.05,
                          max_per_artist=2, K=K_user, use_subgenre=False, use_clusters=True, use_assoc=False, auto_alpha=True),
                RecConfig(model_type="W2V_COSINE", alpha=0.85, genre_bonus=0.02, cluster_bonus=0.02, assoc_bonus=0.05,
                          max_per_artist=2, K=K_user, use_subgenre=False, use_clusters=True, use_assoc=False, auto_alpha=True),
                RecConfig(model_type="BOW_COSINE", alpha=0.9, genre_bonus=0.02, cluster_bonus=0.02, assoc_bonus=0.05,
                          max_per_artist=2, K=K_user, use_subgenre=False, use_clusters=True, use_assoc=False, auto_alpha=True),
                RecConfig(model_type="JACCARD", alpha=0.7, genre_bonus=0.02, cluster_bonus=0.02, assoc_bonus=0.05,
                          max_per_artist=2, K=K_user, use_subgenre=False, use_clusters=True, use_assoc=False, auto_alpha=True),
                RecConfig(model_type="SBERT_EUCLIDEAN", alpha=0.8, genre_bonus=0.02, cluster_bonus=0.02, assoc_bonus=0.05,
                          max_per_artist=2, K=K_user, use_subgenre=False, use_clusters=True, use_assoc=False, auto_alpha=True),
            ]

            progress = st.progress(0, text="Running model tournament...")
            results = []

            try:
                seed_idxs_user = rec._seed_indices(seeds_user) if seeds_user else []
                for i, cfg in enumerate(candidates, start=1):
                    df_out = rec.recommend(
                        selected_genre=selected_genre_user,
                        seeds_title_artist=seeds_user,
                        cfg=cfg
                    )
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
                    if best_out is None or best_out.empty:
                        st.warning("No recommendations were generated.")
                    else:
                        st.subheader("Top recommendations")
                        ordered = best_out.sort_values("rank", ascending=True) if "rank" in best_out.columns else best_out
                        for _, row in ordered.iterrows():
                            rank = int(row["rank"]) if "rank" in row else 0
                            st.markdown(f"**{rank}. {row['track_name']}** — {row['track_artist']}")
                            st.caption(f"{row.get('playlist_genre', '')} · {row.get('playlist_subgenre', '')}")
                            render_text_audio_contrib(row)
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
            except Exception as e:
                st.error(str(e))

# =========================================================
# Recommend tab (DS)
# =========================================================
with tab_rec:
    st.subheader("DS Lab")

    with st.expander("Model settings", expanded=True):
        model_type = st.selectbox(
            "Text model",
            ["SBERT_COSINE", "TFIDF_COSINE", "BOW_COSINE", "W2V_COSINE", "JACCARD", "SBERT_EUCLIDEAN"]
        )
        alpha = st.slider(
            "alpha (text vs audio)",
            0.0, 1.0, 0.8, 0.05,
            help="0 = only audio similarity, 1 = only text (lyrics) similarity."
        )
        genre_bonus = st.slider(
            "genre_bonus",
            0.0, 0.10, 0.02, 0.01,
            help="Extra boost if track genre matches the selected genre/subgenre."
        )
        cluster_bonus = st.slider(
            "cluster_bonus",
            0.0, 0.10, 0.02, 0.01,
            help="Bonus for tracks in the dominant audio cluster of the seed songs."
        )
        assoc_bonus = st.slider(
            "association_bonus",
            0.0, 0.20, 0.05, 0.01,
            help="Boost for tracks that often co-occur with seeds in playlists."
        )
        max_per_artist = st.selectbox("Max tracks per artist in Top-K", [1, 2, 3], index=1)
        use_subgenre = st.checkbox("Use subgenre (playlist_subgenre)", value=False)
        use_clusters = st.checkbox(
            "Use audio clusters bonus",
            value=True,
            help="Encourages recommendations from the main audio cluster of the seeds."
        )
        use_assoc = st.checkbox(
            "Use association rules bonus",
            value=False,
            help="Uses playlist co-occurrence rules to boost likely pairs."
        )
        filter_outliers = st.checkbox(
            "Filter outliers",
            value=False,
            help="Remove tracks flagged as audio outliers."
        )
        K = st.selectbox("K", [5, 10, 15, 20], index=1)

    st.subheader("1) Choose genre/subgenre + 3–5 seed tracks")

    # Demo seeds buttons (place before widgets that bind to session_state keys)
    st.caption("Tip: Load demo seeds to test quickly, then refine.")
    c1, c2, c3, _ = st.columns([1, 1, 1, 6])
    if c1.button("Load demo: Pop", key="demo_ds_pop"):
        st.session_state["selected_genre"] = "pop"
        st.session_state["seed_tracks"] = DEMOS.get("pop", [])
        st.rerun()
    if c2.button("Load demo: Rock", key="demo_ds_rock"):
        st.session_state["selected_genre"] = "rock"
        st.session_state["seed_tracks"] = DEMOS.get("rock", [])
        st.rerun()
    if c3.button("Load demo: Rap", key="demo_ds_rap"):
        st.session_state["selected_genre"] = "rap"
        st.session_state["seed_tracks"] = DEMOS.get("rap", [])
        st.rerun()

    # Genre dropdown (instead of free text)
    default_genre = "pop" if "pop" in GENRES else (GENRES[0] if GENRES else "")
    if "selected_genre" in st.session_state and st.session_state["selected_genre"] not in GENRES:
        st.session_state["selected_genre"] = default_genre
    selected_genre = st.selectbox(
        "Selected genre (from dataset)",
        options=GENRES,
        index=GENRES.index(default_genre) if default_genre in GENRES else 0,
        key="selected_genre"
    )

    selected_subgenre = None
    df_pool = df_ui[df_ui["playlist_genre"] == selected_genre].copy()

    # If using subgenre - show dependent dropdown and filter pool
    if use_subgenre:
        subgenres = sorted(
            df_pool["playlist_subgenre"].dropna().unique().tolist()
        )
        if subgenres:
            selected_subgenre = st.selectbox("Selected subgenre", options=subgenres)
            df_pool = df_pool[df_pool["playlist_subgenre"] == selected_subgenre].copy()
        else:
            st.info("No subgenres found for this genre. Using genre only.")
            selected_subgenre = None

    # Track options for seeds (autocomplete)
    track_options = sorted(df_pool["track_option"].dropna().unique().tolist())
    if "seed_tracks" not in st.session_state:
        st.session_state["seed_tracks"] = []
    current_seeds = st.session_state.get("seed_tracks", [])
    filtered_seeds = [x for x in current_seeds if x in track_options]
    if filtered_seeds != current_seeds:
        st.session_state["seed_tracks"] = filtered_seeds

    seeds = st.multiselect(
        "Seed tracks (choose 3–5)",
        options=track_options,
        key="seed_tracks",
        help="Search by track or artist. Format: track_name — track_artist"
    )

    # Validate count (soft enforcement; avoid Streamlit max_selections warning)
    if len(seeds) not in (3, 4, 5):
        st.warning("Please select **3–5** seed tracks to get recommendations.")

    # Run
    run_disabled = len(seeds) < 3 or len(seeds) > 5
    if st.button("Recommend", type="primary", disabled=run_disabled):
        cfg = RecConfig(
            model_type=model_type,
            alpha=alpha,
            genre_bonus=genre_bonus,
            cluster_bonus=cluster_bonus,
            assoc_bonus=assoc_bonus,
            max_per_artist=max_per_artist,
            K=K,
            use_subgenre=use_subgenre,
            use_clusters=use_clusters,
            use_assoc=use_assoc,
            filter_outliers=filter_outliers
        )
        try:
            # IMPORTANT: keep your API as-is
            # selected_genre is passed as string; when use_subgenre=True we pass subgenre if selected
            genre_value = selected_subgenre if (use_subgenre and selected_subgenre) else selected_genre

            df_out = rec.recommend(
                selected_genre=genre_value,
                seeds_title_artist=seeds,
                cfg=cfg
            )

            st.success("Done")

            # Nice display (keep original df, just reorder if possible)
            preferred_cols = [
                "rank", "track_name", "track_artist", "playlist_genre", "playlist_subgenre",
                "final_score", "text_sim", "audio_sim"
            ]
            cols_to_show = [c for c in preferred_cols if c in df_out.columns]
            st.dataframe(df_out[cols_to_show] if cols_to_show else df_out, use_container_width=True, hide_index=True)

            # Tiny UX: show artist diversity if possible
            if "track_artist" in df_out.columns and len(df_out) > 0:
                st.caption(f"Artist diversity: **{df_out['track_artist'].nunique()}/{len(df_out)}** unique artists")

            if show_explain:
                st.subheader("Explainability: Text vs Audio")
                ordered = df_out.sort_values("rank", ascending=True) if "rank" in df_out.columns else df_out
                for _, row in ordered.iterrows():
                    rank = int(row["rank"]) if "rank" in row else 0
                    st.markdown(f"**{rank}. {row['track_name']}** — {row['track_artist']}")
                    render_text_audio_contrib(row)
                st.caption("These bars show the relative similarity contribution from lyrics vs audio features.")

            with st.expander("Why these recommendations?"):
                st.write(
                    "Ranking combines **lyrics similarity** and **audio-feature similarity** with a soft genre boost "
                    "and an artist diversity constraint."
                )
                st.write(f"- Text model: **{model_type}**")
                st.write(f"- alpha (text vs audio): **{alpha}**")
                st.write(f"- genre bonus: **{genre_bonus}**")
                st.write(f"- cluster bonus: **{cluster_bonus}**")
                st.write(f"- association bonus: **{assoc_bonus}**")
                st.write(f"- max tracks per artist: **{max_per_artist}**")
                st.write(f"- K: **{K}**")

        except Exception as e:
            st.error(str(e))

    show_explain = st.checkbox("Show explainability for DS recommendations", value=False)
    st.divider()
    st.subheader("Quick evaluation")

    if "tournament_scoreboard" in st.session_state:
        st.subheader("Model tournament scoreboard")
        st.dataframe(st.session_state["tournament_scoreboard"], use_container_width=True, hide_index=True)
        with st.expander("Metric explanations"):
            st.markdown(
                "- **Composite**: weighted score from normalized similarity + playlist proxy + diversity + novelty.\n"
                "- **Normalized similarity**: per-model min-max normalization of recommendation scores.\n"
                "- **Playlist precision@K**: fraction of top-K songs that co-occur with seed playlists.\n"
                "- **Hit rate@K**: whether any top-K song co-occurs with seed playlists.\n"
                "- **Diversity**: fraction of unique artists in the top-K.\n"
                "- **Novelty**: higher when recommendations are less popular."
            )
        if "tournament_best" in st.session_state:
            best = st.session_state["tournament_best"]
            with st.expander("Why this model won"):
                st.write(
                    "The winner maximizes the **composite score**, which blends normalized similarity, playlist co-occurrence "
                    "signals, and diversity/novelty. This avoids bias toward models with larger raw score ranges."
                )
                st.write(f"- Normalized similarity: **{best['norm_mean_score']:.3f}**")
                st.write(f"- Playlist precision@K: **{best['playlist_precision']:.3f}**")
                st.write(f"- Hit rate@K: **{best['playlist_hit_rate']:.3f}**")
                st.write(f"- Diversity: **{best['diversity']:.3f}**")
                st.write(f"- Novelty: **{best['novelty']:.3f}**")

    eval_cfg = EvalConfig(use_subgenre=use_subgenre, n_queries=8, seeds_per_query=4, K=K)
    ground_truth = st.selectbox("Ground truth", ["playlist", "genre", "subgenre", "album", "artist"], index=0)
    n_queries = st.slider("Number of queries", 4, 15, eval_cfg.n_queries)
    seeds_per_query = st.slider("Seeds per query", 2, 6, eval_cfg.seeds_per_query)
    run_baselines = st.checkbox("Run baselines (random/popular/genre/artist)", value=True)
    run_ablations = st.checkbox("Run ablation study", value=False)

    models = st.multiselect(
        "Models to compare",
        ["SBERT_COSINE", "TFIDF_COSINE", "BOW_COSINE", "W2V_COSINE", "JACCARD", "SBERT_EUCLIDEAN"],
        default=["SBERT_COSINE", "TFIDF_COSINE", "W2V_COSINE"]
    )

    if st.button("Run evaluation"):
        if not models:
            st.error("Select at least one model to evaluate.")
            st.stop()

        with st.spinner("Running evaluation..."):
            queries_df = generate_queries_df(
                rec.df,
                use_subgenre=use_subgenre,
                n_queries=n_queries,
                seeds_per_query=seeds_per_query,
                ground_truth=ground_truth
            )

            cfgs = [
                RecConfig(
                    model_type=m,
                    alpha=alpha,
                    genre_bonus=genre_bonus,
                    cluster_bonus=cluster_bonus,
                    assoc_bonus=assoc_bonus,
                    max_per_artist=max_per_artist,
                    K=K,
                    use_subgenre=use_subgenre,
                    use_clusters=use_clusters,
                    use_assoc=use_assoc,
                    filter_outliers=filter_outliers
                )
                for m in models
            ]

            summary, per_query = evaluate_configs(
                rec,
                queries_df,
                cfgs,
                use_subgenre,
                ground_truth=queries_df.iloc[0]["ground_truth"] if not queries_df.empty else "genre"
            )

            st.session_state["ds_summary"] = summary
            st.session_state["ds_per_query"] = per_query

            if run_baselines:
                baselines = ["random", "popular", "same_genre", "same_artist"]
                base_df = evaluate_baselines(
                    rec,
                    queries_df,
                    baselines,
                    use_subgenre,
                    queries_df.iloc[0]["ground_truth"] if not queries_df.empty else "genre",
                    K,
                    42
                )
                st.session_state["ds_baselines"] = base_df

            if run_ablations:
                base_cfg = RecConfig(
                    model_type="SBERT_COSINE",
                    alpha=alpha,
                    genre_bonus=genre_bonus,
                    cluster_bonus=cluster_bonus,
                    assoc_bonus=assoc_bonus,
                    K=K,
                    use_subgenre=use_subgenre,
                    use_clusters=use_clusters,
                    use_assoc=use_assoc,
                    filter_outliers=filter_outliers
                )
                ablations = build_ablation_configs(base_cfg)
                summary_ab, per_query_ab = evaluate_configs(
                    rec,
                    queries_df,
                    ablations,
                    use_subgenre,
                    ground_truth=queries_df.iloc[0]["ground_truth"] if not queries_df.empty else "genre"
                )
                st.session_state["ds_ablations_summary"] = summary_ab
                st.session_state["ds_ablations"] = per_query_ab

        if summary.empty:
            st.warning("No valid queries were generated.")
        else:
            st.success("Evaluation complete")
            with st.expander("Metric explanations"):
                st.markdown(
                    "- **Precision@K**: how many of the top-K are relevant.\n"
                    "- **Recall@K**: how many of the relevant items were retrieved.\n"
                    "- **nDCG@K**: rewards putting relevant items higher in the ranking.\n"
                    "- **Hit Rate@K**: did we get at least one relevant item.\n"
                    "- **Artist/Genre Diversity**: fraction of unique artists/genres in top-K.\n"
                    "- **Novelty**: higher when recommended tracks are less popular.\n"
                    "- **Coverage**: how many unique items are ever recommended."
                )
            st.dataframe(summary, use_container_width=True, hide_index=True)

            st.subheader("Per-query results")
            st.dataframe(per_query.drop(columns=["rec_idxs"], errors="ignore"), use_container_width=True, hide_index=True)

    if "ds_summary" in st.session_state:
        st.divider()
        st.subheader("Dashboard")
        with st.expander("How to read this dashboard"):
            st.markdown(
                "- **Precision@K**: share of top‑K recommendations that are relevant.\n"
                "- **nDCG@K**: ranking quality (higher means relevant items appear higher).\n"
                "- **Diversity**: variety of artists in top‑K (higher means less repetition).\n"
                "- **Novelty**: favors less‑popular tracks (higher = more discovery)."
            )
        summary = st.session_state["ds_summary"]
        best_row = summary.sort_values("ndcg@K", ascending=False).iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision@K", f"{best_row['precision@K']:.3f}")
        c2.metric("nDCG@K", f"{best_row['ndcg@K']:.3f}")
        c3.metric("Diversity", f"{best_row['artist_diversity']:.3f}")
        c4.metric("Novelty", f"{best_row['novelty']:.3f}")

        st.subheader("Model comparison")
        chart_df = summary[["model_type", "ndcg@K"]].set_index("model_type")
        st.bar_chart(chart_df)
        st.caption("Bar chart shows average nDCG@K per model (higher is better).")

        st.subheader("Parameter sweep (alpha)")
        st.caption("Shows how the text-vs-audio balance affects quality for a single model.")
        sweep_model = st.selectbox("Model for alpha sweep", summary["model_type"].tolist(), index=0)
        if st.button("Run alpha sweep"):
            alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
            sweep_rows = []
            for a in alphas:
                cfg = RecConfig(
                    model_type=sweep_model,
                    alpha=a,
                    genre_bonus=genre_bonus,
                    cluster_bonus=cluster_bonus,
                    assoc_bonus=assoc_bonus,
                    K=K,
                    use_subgenre=use_subgenre,
                    use_clusters=use_clusters,
                    use_assoc=use_assoc,
                    filter_outliers=filter_outliers
                )
                queries_df = generate_queries_df(
                    rec.df,
                    use_subgenre=use_subgenre,
                    n_queries=n_queries,
                    seeds_per_query=seeds_per_query,
                    ground_truth=ground_truth
                )
                summ, _ = evaluate_configs(
                    rec,
                    queries_df,
                    [cfg],
                    use_subgenre,
                    ground_truth=queries_df.iloc[0]["ground_truth"] if not queries_df.empty else "genre"
                )
                if not summ.empty:
                    sweep_rows.append({"alpha": a, "ndcg@K": float(summ.iloc[0]["ndcg@K"])})
            if sweep_rows:
                sweep_df = pd.DataFrame(sweep_rows).set_index("alpha")
                st.line_chart(sweep_df)
                st.caption("Line chart shows how nDCG@K changes with alpha (higher is better).")

        st.subheader("Top failures")
        per_query = st.session_state.get("ds_per_query", pd.DataFrame())
        if not per_query.empty:
            failures = per_query.sort_values("ndcg@K").head(10)
            st.dataframe(
                failures[["query_id", "model_type", "ndcg@K", "precision@K", "recall@K", "hit_rate@K", "n_relevant"]],
                use_container_width=True,
                hide_index=True
            )
            st.caption("Lowest nDCG@K cases to inspect where the model fails and why.")

        if "ds_baselines" in st.session_state:
            st.subheader("Baselines")
            base_df = st.session_state["ds_baselines"]
            base_summary = (
                base_df.groupby("model_type")[["precision@K", "recall@K", "ndcg@K", "hit_rate@K"]]
                .mean()
                .reset_index()
            )
            st.dataframe(base_summary, use_container_width=True, hide_index=True)
            st.caption("Baselines help validate the model adds value over naive approaches.")

        if "ds_ablations_summary" in st.session_state:
            st.subheader("Ablation study")
            st.dataframe(st.session_state["ds_ablations_summary"], use_container_width=True, hide_index=True)
            st.caption("Ablations show which components contribute most to performance.")
