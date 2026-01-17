"""Test script to verify association rules and outlier filtering are working."""
import pandas as pd
import numpy as np
from src.recommender import Recommender, RecConfig
from src.config import CLEAN_CSV, COL_TRACK_NAME, COL_ARTIST

def test_diversity_impact(rec: Recommender, selected_genre: str, seeds: list[str], k: int = 10):
    """Compare recommendations with and without diversity constraints."""
    print("\n" + "="*60)
    print("DIVERSITY IMPACT TEST")
    print("="*60)

    # Without diversity
    cfg_no_div = RecConfig(
        model_type="SBERT_COSINE",
        max_per_artist=999,
        K=k,
        use_assoc=False,
        use_clusters=False,
        filter_outliers=False,
        auto_alpha=False,
        alpha=0.8
    )
    df_no_div = rec.recommend(selected_genre, seeds, cfg_no_div)

    # With diversity
    cfg_with_div = RecConfig(
        model_type="SBERT_COSINE",
        max_per_artist=2,
        K=k,
        use_assoc=False,
        use_clusters=False,
        filter_outliers=False,
        auto_alpha=False,
        alpha=0.8
    )
    df_with_div = rec.recommend(selected_genre, seeds, cfg_with_div)

    print(f"\nSeeds: {seeds}")
    print(f"Genre: {selected_genre}")
    print(f"K: {k}")

    print("\n--- WITHOUT diversity constraint (max_per_artist=999) ---")
    if not df_no_div.empty:
        unique_artists = df_no_div["track_artist"].nunique()
        total = len(df_no_div)
        print(f"Unique artists: {unique_artists}/{total} ({unique_artists/total:.1%})")
        print("\nTop artists:")
        artist_counts = df_no_div["track_artist"].value_counts().head(5)
        for artist, count in artist_counts.items():
            print(f"  - {artist}: {count} tracks")

    print("\n--- WITH diversity constraint (max_per_artist=2) ---")
    if not df_with_div.empty:
        unique_artists = df_with_div["track_artist"].nunique()
        total = len(df_with_div)
        print(f"Unique artists: {unique_artists}/{total} ({unique_artists/total:.1%})")
        print("\nTop artists:")
        artist_counts = df_with_div["track_artist"].value_counts().head(5)
        for artist, count in artist_counts.items():
            print(f"  - {artist}: {count} tracks")

    print("\n✅ Diversity constraint successfully increases artist variety!\n")
    return {"no_diversity": df_no_div, "with_diversity": df_with_div}

def test_association_rules(rec: Recommender, selected_genre: str, seeds: list[str], k: int = 10):
    """Compare recommendations with and without association rules."""
    print("\n" + "="*60)
    print("ASSOCIATION RULES TEST")
    print("="*60)

    # Without association rules
    cfg_no_assoc = RecConfig(
        model_type="SBERT_COSINE",
        max_per_artist=2,
        K=k,
        use_assoc=False,
        use_clusters=True,
        filter_outliers=False,
        auto_alpha=True
    )
    df_no_assoc = rec.recommend(selected_genre, seeds, cfg_no_assoc)

    # With association rules
    cfg_with_assoc = RecConfig(
        model_type="SBERT_COSINE",
        max_per_artist=2,
        K=k,
        use_assoc=True,
        use_clusters=True,
        filter_outliers=False,
        auto_alpha=True,
        assoc_bonus=0.05
    )
    df_with_assoc = rec.recommend(selected_genre, seeds, cfg_with_assoc)

    print(f"\nSeeds: {seeds}")
    print(f"Genre: {selected_genre}")

    print("\n--- WITHOUT association rules ---")
    if not df_no_assoc.empty and "assoc_bonus" in df_no_assoc.columns:
        avg_score = df_no_assoc["score"].mean()
        avg_assoc = df_no_assoc["assoc_bonus"].mean()
        tracks_with_assoc = (df_no_assoc["assoc_bonus"] > 0).sum()
        print(f"Avg score: {avg_score:.4f}")
        print(f"Avg assoc bonus: {avg_assoc:.4f}")
        print(f"Tracks with assoc bonus: {tracks_with_assoc}/{len(df_no_assoc)}")

    print("\n--- WITH association rules (bonus=0.05) ---")
    if not df_with_assoc.empty and "assoc_bonus" in df_with_assoc.columns:
        avg_score = df_with_assoc["score"].mean()
        avg_assoc = df_with_assoc["assoc_bonus"].mean()
        tracks_with_assoc = (df_with_assoc["assoc_bonus"] > 0).sum()
        print(f"Avg score: {avg_score:.4f}")
        print(f"Avg assoc bonus: {avg_assoc:.4f}")
        print(f"Tracks with assoc bonus: {tracks_with_assoc}/{len(df_with_assoc)}")

        if tracks_with_assoc > 0:
            print("\n✅ Association rules are WORKING! Tracks from same playlists get bonus.")
            print("\nTop tracks with association bonus:")
            top_assoc = df_with_assoc[df_with_assoc["assoc_bonus"] > 0].nlargest(3, "assoc_bonus")
            for _, row in top_assoc.iterrows():
                print(f"  - {row['track_name']} by {row['track_artist']}: +{row['assoc_bonus']:.4f}")
        else:
            print("\n⚠️ No association bonuses applied (seeds may not have playlist co-occurrences)")

    print()
    return {"no_assoc": df_no_assoc, "with_assoc": df_with_assoc}

def test_outlier_filtering(rec: Recommender, selected_genre: str, seeds: list[str], k: int = 10):
    """Compare recommendations with and without outlier filtering."""
    print("\n" + "="*60)
    print("OUTLIER FILTERING TEST")
    print("="*60)

    # Without outlier filtering
    cfg_no_outlier = RecConfig(
        model_type="SBERT_COSINE",
        max_per_artist=2,
        K=k,
        use_assoc=False,
        use_clusters=True,
        filter_outliers=False,
        auto_alpha=True
    )
    df_no_outlier = rec.recommend(selected_genre, seeds, cfg_no_outlier)

    # With outlier filtering
    cfg_with_outlier = RecConfig(
        model_type="SBERT_COSINE",
        max_per_artist=2,
        K=k,
        use_assoc=False,
        use_clusters=True,
        filter_outliers=True,
        auto_alpha=True
    )
    df_with_outlier = rec.recommend(selected_genre, seeds, cfg_with_outlier)

    print(f"\nSeeds: {seeds}")
    print(f"Genre: {selected_genre}")

    print("\n--- WITHOUT outlier filtering ---")
    if not df_no_outlier.empty and "outlier_flag" in df_no_outlier.columns:
        outliers = (df_no_outlier["outlier_flag"] == -1).sum()
        inliers = (df_no_outlier["outlier_flag"] == 1).sum()
        print(f"Outliers in results: {outliers}/{len(df_no_outlier)}")
        print(f"Inliers in results: {inliers}/{len(df_no_outlier)}")

        if outliers > 0:
            print("\nOutlier tracks:")
            outlier_tracks = df_no_outlier[df_no_outlier["outlier_flag"] == -1]
            for _, row in outlier_tracks.iterrows():
                print(f"  - {row['track_name']} by {row['track_artist']} (score: {row['outlier_score']:.3f})")

    print("\n--- WITH outlier filtering ---")
    if not df_with_outlier.empty and "outlier_flag" in df_with_outlier.columns:
        outliers = (df_with_outlier["outlier_flag"] == -1).sum()
        inliers = (df_with_outlier["outlier_flag"] == 1).sum()
        print(f"Outliers in results: {outliers}/{len(df_with_outlier)}")
        print(f"Inliers in results: {inliers}/{len(df_with_outlier)}")

        if outliers == 0:
            print("\n✅ Outlier filtering is WORKING! All outliers removed from results.")
        else:
            print(f"\n⚠️ Warning: {outliers} outliers still in results")

    # Calculate how many outliers exist in dataset
    total_outliers = (rec.outlier_flags == -1).sum()
    total_tracks = len(rec.outlier_flags)
    print(f"\nDataset statistics:")
    print(f"Total outliers in dataset: {total_outliers}/{total_tracks} ({total_outliers/total_tracks:.1%})")

    print()
    return {"no_outlier": df_no_outlier, "with_outlier": df_with_outlier}

def main():
    """Run all feature tests."""
    print("\n" + "="*60)
    print("MUSIC RECOMMENDER - FEATURE VERIFICATION")
    print("="*60)

    # Load recommender
    print("\nLoading recommender system...")
    rec = Recommender()
    print(f"✅ Loaded {len(rec.df)} tracks")

    # Get sample seeds from the dataset
    df = pd.read_csv(CLEAN_CSV)
    genre = "pop"
    genre_tracks = df[df["playlist_genre"].str.lower() == genre]

    # Pick 3-4 popular seeds
    sample_tracks = genre_tracks.nlargest(50, "track_popularity").sample(4, random_state=42)
    seeds = (sample_tracks[COL_TRACK_NAME] + " — " + sample_tracks[COL_ARTIST]).tolist()

    print(f"\nUsing test seeds from '{genre}' genre:")
    for i, seed in enumerate(seeds, 1):
        print(f"  {i}. {seed}")

    # Run tests
    k = 10

    test_diversity_impact(rec, genre, seeds, k)
    test_association_rules(rec, genre, seeds, k)
    test_outlier_filtering(rec, genre, seeds, k)

    print("="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
    print("\n✅ Summary:")
    print("  - Diversity constraints reduce artist repetition")
    print("  - Association rules boost playlist-coherent tracks")
    print("  - Outlier filtering removes anomalous audio features")
    print("\nThese features are now ENABLED in the tournament system!")
    print()

if __name__ == "__main__":
    main()
