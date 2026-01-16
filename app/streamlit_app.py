import streamlit as st
import pandas as pd
from src.recommender import Recommender, RecConfig
from src.evaluation import generate_queries_df, evaluate_configs, EvalConfig, demo_subset

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

# ---------- Sidebar ----------
st.sidebar.header("Settings")
model_type = st.sidebar.selectbox(
    "Text model",
    ["SBERT_COSINE", "TFIDF_COSINE", "BOW_COSINE", "W2V_COSINE", "JACCARD", "SBERT_EUCLIDEAN"]
)
alpha = st.sidebar.slider(
    "alpha (text vs audio)",
    0.0, 1.0, 0.8, 0.05,
    help="0 = only audio similarity, 1 = only text (lyrics) similarity."
)
genre_bonus = st.sidebar.slider(
    "genre_bonus",
    0.0, 0.10, 0.02, 0.01,
    help="Extra boost if track genre matches the selected genre/subgenre."
)
cluster_bonus = st.sidebar.slider(
    "cluster_bonus",
    0.0, 0.10, 0.02, 0.01,
    help="Bonus for tracks in the dominant audio cluster of the seed songs."
)
assoc_bonus = st.sidebar.slider(
    "association_bonus",
    0.0, 0.20, 0.05, 0.01,
    help="Boost for tracks that often co-occur with seeds in playlists."
)
max_per_artist = st.sidebar.selectbox("Max tracks per artist in Top-K", [1, 2, 3], index=1)
use_subgenre = st.sidebar.checkbox("Use subgenre (playlist_subgenre)", value=False)
use_clusters = st.sidebar.checkbox(
    "Use audio clusters bonus",
    value=True,
    help="Encourages recommendations from the main audio cluster of the seeds."
)
use_assoc = st.sidebar.checkbox(
    "Use association rules bonus",
    value=False,
    help="Uses playlist co-occurrence rules to boost likely pairs."
)
filter_outliers = st.sidebar.checkbox(
    "Filter outliers",
    value=False,
    help="Remove tracks flagged as audio outliers."
)
K = st.sidebar.selectbox("K", [5, 10, 15, 20], index=1)

tab_user, tab_rec, tab_eval = st.tabs(["Recommend", "DS Lab", "Evaluation"])

# =========================================================
# User tab (simple)
# =========================================================
with tab_user:
    st.subheader("Pick 3–5 songs you like")

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

    seeds_user = st.multiselect(
        "Seed tracks (choose 3–5)",
        options=track_options_user,
        key="seed_tracks_user",
        help="Search by track or artist. Format: track_name — track_artist"
    )

    run_disabled_user = len(seeds_user) < 3 or len(seeds_user) > 5
    if st.button("Recommend", type="primary", disabled=run_disabled_user, key="recommend_user"):
        cfg_user = RecConfig(
            model_type="SBERT_COSINE",
            alpha=0.8,
            genre_bonus=0.02,
            cluster_bonus=0.02,
            assoc_bonus=0.05,
            max_per_artist=2,
            K=K,
            use_subgenre=False,
            use_clusters=True,
            use_assoc=False,
            filter_outliers=False
        )
        try:
            df_out = rec.recommend(
                selected_genre=selected_genre_user,
                seeds_title_artist=seeds_user,
                cfg=cfg_user
            )
            st.success("Here are your recommendations")

            cols = st.columns(3)
            for i, row in df_out.iterrows():
                col = cols[i % 3]
                with col:
                    st.markdown(f"**{row['track_name']}**")
                    st.write(row["track_artist"])
                    st.caption(f"{row.get('playlist_genre', '')} · {row.get('playlist_subgenre', '')}")
        except Exception as e:
            st.error(str(e))

# =========================================================
# Recommend tab (DS)
# =========================================================
with tab_rec:
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

# =========================================================
# Evaluation tab
# =========================================================
with tab_eval:
    st.subheader("Quick evaluation")

    eval_cfg = EvalConfig(use_subgenre=use_subgenre, n_queries=8, seeds_per_query=4, K=K)
    ground_truth = st.selectbox("Ground truth", ["playlist", "genre", "subgenre", "album", "artist"], index=0)
    n_queries = st.slider("Number of queries", 4, 15, eval_cfg.n_queries)
    seeds_per_query = st.slider("Seeds per query", 2, 6, eval_cfg.seeds_per_query)

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

            st.subheader("Subset demo (first query)")
            first = queries_df.iloc[0]
            seeds = [s.strip() for s in first["seeds"].split("|") if s.strip()]

            demo = demo_subset(
                rec,
                first["selected_genre"],
                seeds,
                RecConfig(model_type=models[0], K=K, use_subgenre=use_subgenre),
                n_show=min(5, K)
            )
            st.dataframe(demo, use_container_width=True, hide_index=True)
