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
- User tab with Spotify embeds + feedback (👍/👎)
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
- src/tournament.py – model tournament system for automatic best-model selection
- src/test_features.py – feature verification and testing script
- app/streamlit_app.py – Streamlit demo UI

## Features & Verification

### Similarity Methods (7 total)
1. **Cosine (TF-IDF)** - Fast sparse text matching
2. **Cosine (SBERT)** - Semantic neural embeddings
3. **Cosine (Word2Vec)** - Average word vectors
4. **Cosine (BoW)** - Simple count vectors
5. **Jaccard** - Token set overlap
6. **Euclidean (SBERT)** - Dense embedding distance
7. **Cosine (Audio)** - Audio feature similarity

### Advanced Features

#### 1. Association Rules ✅
- **What**: Tracks that frequently appear together in playlists get bonus scores
- **Data**: 15,327 tracks with up to 50 associated tracks each
- **Status**: ENABLED in tournament (use_assoc=True)
- **Impact**: Boosts playlist-coherent recommendations

#### 2. Outlier Detection ✅
- **What**: Isolation Forest identifies anomalous audio features
- **Data**: 308 outliers (2.0% of dataset) detected
- **Status**: ENABLED in tournament (filter_outliers=True)
- **Impact**: Removes tracks with unusual audio signatures

#### 3. Diversity Constraints ✅
- **What**: Maximum 2 tracks per artist
- **Status**: ACTIVE (max_per_artist=2)
- **Impact**: Prevents artist repetition, increases variety to 97.5%+

#### 4. Audio Clustering ✅
- **What**: K-means clustering (20 clusters) groups similar audio profiles
- **Status**: ENABLED (use_clusters=True)
- **Impact**: +0.02 bonus for tracks in same cluster as seeds

#### 5. Auto-Alpha Weighting ✅
- **What**: Automatically balances text vs audio similarity based on seed coherence
- **Status**: ENABLED (auto_alpha=True)
- **Impact**: Adapts weighting from 0.2-0.9 dynamically

### Tournament System

All recommendations run a **model tournament** comparing 6 methods:
- SBERT_COSINE, TFIDF_COSINE, W2V_COSINE, BOW_COSINE, JACCARD, SBERT_EUCLIDEAN

**Selection criteria**: Composite score = 0.45×normalized_score + 0.35×playlist_precision + 0.10×diversity + 0.10×novelty

The best model is automatically selected for each query.

## Testing & Evaluation

### Run Feature Tests
```bash
python3 -m src.test_features
```
This verifies:
- Diversity constraints reduce repetition
- Association rules boost playlist-coherent tracks
- Outlier filtering removes anomalous tracks

### Run Full Evaluation
```bash
# Generate test queries
python3 -m src.evaluation --make_queries --n_queries 10 --ground_truth playlist

# Evaluate models with ablation study
python3 -m src.evaluation --evaluate --models "SBERT_COSINE,TFIDF_COSINE,W2V_COSINE" --K 10 --run_ablations
```

Results saved to:
- `data/evaluation_results.csv` - Per-query results
- `data/evaluation_ablations_summary.csv` - Feature impact analysis

### Evaluation Results (Sample)

| Model | NDCG@10 | Precision@10 | Artist Diversity | Novelty |
|-------|---------|--------------|------------------|---------|
| SBERT_COSINE | 0.0256 | 0.0250 | 97.5% | 58.9% |
| TFIDF_COSINE | 0.0087 | 0.0125 | 97.5% | 61.1% |
| W2V_COSINE | 0.0083 | 0.0125 | 98.8% | 55.0% |

### Ablation Study (Feature Impact)

| Configuration | NDCG@10 | Impact |
|---------------|---------|--------|
| Base (all features) | 0.0256 | Baseline |
| No text (audio only) | 0.0205 | -20% (text is critical) |
| No audio (text only) | 0.0198 | -23% (audio improves quality) |
| No diversity | 0.0256 | No quality loss, but repetition |
| No clusters | 0.0256 | Minimal impact on NDCG |

**Key Finding**: Text+audio fusion provides the best results. Removing either signal degrades quality by 20%+.
