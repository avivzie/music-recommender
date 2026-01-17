# Music Recommender (Lyrics + Audio)

Content-based music recommender that suggests Top-K tracks from either:
- 3–5 seed songs (similar tracks)
- a free-text prompt (text-to-song)

Core signals:
- Lyrics similarity (TF-IDF, BoW, W2V, SBERT)
- Audio-feature similarity (tempo/energy/danceability/etc.)
- Soft genre bias + diversity constraint
- Optional association rules + audio clustering bonuses

UI highlights:
- User tab with Spotify embeds + feedback (👍/😐/👎)
- Model tournament in the background (picks best model per run)
- DS Lab with tournament scoreboard, embedding map (PCA/t-SNE), latency, and debug NLP

## Quickstart

Run:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

    # Put dataset under data/spotify_songs.csv
    python3 -m src.preprocess
    python3 -m src.build_index

    PYTHONPATH=. streamlit run app/streamlit_app.py

## Project Structure
- src/preprocess.py – dataset cleaning
- src/build_index.py – text/audio vectors + KNN indexes + clustering/outliers + association rules
- src/recommender.py – recommendation logic, de-dup, and text-prompt flow
- src/evaluation.py – metrics and model comparison utilities
- app/streamlit_app.py – Streamlit demo UI
