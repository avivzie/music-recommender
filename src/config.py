from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_CSV = DATA_DIR / "spotify_songs.csv"
CLEAN_CSV = DATA_DIR / "spotify_songs_en_clean.csv"

COL_TRACK_ID = "track_id"
COL_TRACK_NAME = "track_name"
COL_ARTIST = "track_artist"
COL_GENRE = "playlist_genre"
COL_SUBGENRE = "playlist_subgenre"
COL_LYRICS = "lyrics"
COL_LANG = "language"

AUDIO_COLS = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "key", "mode"
]
