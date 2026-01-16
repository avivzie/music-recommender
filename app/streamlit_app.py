import streamlit as st
from src.recommender import Recommender, RecConfig

st.set_page_config(page_title="Music Recommender Demo", layout="wide")
st.title("🎧 Music Recommender (Lyrics + Audio)")

@st.cache_resource
def load_model():
    return Recommender()

rec = load_model()

st.sidebar.header("Settings")
model_type = st.sidebar.selectbox("Text model", ["SBERT", "TFIDF"])
alpha = st.sidebar.slider("alpha (text vs audio)", 0.0, 1.0, 0.8, 0.05)
genre_bonus = st.sidebar.slider("genre_bonus", 0.0, 0.10, 0.02, 0.01)
max_per_artist = st.sidebar.selectbox("Max tracks per artist in Top-K", [1, 2, 3], index=1)
use_subgenre = st.sidebar.checkbox("Use subgenre (playlist_subgenre)", value=False)
K = st.sidebar.selectbox("K", [5, 10, 15, 20], index=1)

st.subheader("1) Choose genre/subgenre + 3–5 seed tracks")
selected_genre = st.text_input("Selected genre/subgenre (must match dataset values)", value="pop")

seeds_text = st.text_area(
    "Seeds (one per line) in format: track_name — artist",
    value=""
)

if st.button("Recommend"):
    seeds = [s.strip() for s in seeds_text.splitlines() if s.strip()]
    cfg = RecConfig(
        model_type=model_type,
        alpha=alpha,
        genre_bonus=genre_bonus,
        max_per_artist=max_per_artist,
        K=K,
        use_subgenre=use_subgenre
    )
    try:
        df_out = rec.recommend(selected_genre=selected_genre, seeds_title_artist=seeds, cfg=cfg)
        st.success("Done")
        st.dataframe(df_out, use_container_width=True)
    except Exception as e:
        st.error(str(e))

st.caption("Tip: Seeds must exist exactly as 'track_name — track_artist' in the dataset.")
