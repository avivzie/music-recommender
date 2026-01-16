from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_CSV = DATA_DIR / "spotify_songs.csv"
CLEAN_CSV = DATA_DIR / "spotify_songs_en_clean.csv"

COL_TRACK_ID = "track_id"
COL_TRACK_NAME = "track_name"
COL_ARTIST = "track_artist"
COL_ALBUM_ID = "track_album_id"
COL_ALBUM_NAME = "track_album_name"
COL_ALBUM_RELEASE_DATE = "track_album_release_date"
COL_GENRE = "playlist_genre"
COL_SUBGENRE = "playlist_subgenre"
COL_PLAYLIST_ID = "playlist_id"
COL_PLAYLIST_NAME = "playlist_name"
COL_LYRICS = "lyrics"
COL_LANG = "language"
COL_TRACK_POPULARITY = "track_popularity"

AUDIO_COLS = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "key", "mode"
]
