# Music Recommender (Lyrics + Audio) — TF-IDF vs SBERT

Content-based music recommender that suggests Top-K tracks from 3–5 seed songs using:
- Lyrics similarity (TF-IDF baseline vs SBERT embeddings)
- Audio-feature similarity (tempo/energy/danceability/etc.)
- Soft genre bias + artist diversity constraint
- Streamlit demo UI

## Quickstart

Run:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

    # Put dataset under data/spotify_songs.csv
    python -m src.preprocess
    python -m src.build_index

    streamlit run app/streamlit_app.py

## Project Structure
- src/preprocess.py – dataset cleaning
- src/build_index.py – TF-IDF/SBERT vectors + KNN indexes + audio scaling
- src/recommender.py – recommendation logic + rules
- app/streamlit_app.py – Streamlit demo UI
