"""Text cleaning + basic dataset filtering."""
import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.config import RAW_CSV, CLEAN_CSV, COL_LANG, COL_LYRICS

TAG_RE = re.compile(
    r"\[(chorus|verse|bridge|intro|outro|hook|pre-chorus|refrain).*?\]",
    re.IGNORECASE
)

def clean_lyrics(text: str) -> str:
    # Basic normalization: strip tags/links, lowercase, keep letters/apostrophes.
    text = str(text)
    text = TAG_RE.sub(" ", text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)

def main():
    df = pd.read_csv(RAW_CSV)

    # Keep English only and remove rows without lyrics.
    df = df[df[COL_LANG].astype(str).str.lower().eq("en")]
    df = df.dropna(subset=[COL_LYRICS])

    df["lyrics_clean"] = df[COL_LYRICS].map(clean_lyrics)

    # Remove too-short lyrics to reduce noisy text.
    df = df[df["lyrics_clean"].str.len() >= 60].copy()

    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"Saved clean dataset: {CLEAN_CSV} | rows={len(df)}")

if __name__ == "__main__":
    main()
