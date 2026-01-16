import argparse
import pandas as pd
from src.config import CLEAN_CSV, COL_GENRE, COL_SUBGENRE, COL_TRACK_NAME, COL_ARTIST

def make_queries(use_subgenre: bool = False, n_queries: int = 8, seeds_per_query: int = 4):
    df = pd.read_csv(CLEAN_CSV)
    genre_col = COL_SUBGENRE if use_subgenre else COL_GENRE

    genres = df[genre_col].value_counts().head(n_queries).index.tolist()
    queries = []
    for g in genres:
        subset = df[df[genre_col] == g].sample(min(50, (df[genre_col] == g).sum()), random_state=42)
        seeds = subset.sample(seeds_per_query, random_state=7)
        seed_list = (seeds[COL_TRACK_NAME] + " — " + seeds[COL_ARTIST]).tolist()
        queries.append({"query_id": f"Q_{g}", "selected_genre": g, "seeds": " | ".join(seed_list)})

    out = pd.DataFrame(queries)
    out.to_csv("evaluation_queries.csv", index=False)
    print("Wrote evaluation_queries.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_queries", action="store_true")
    ap.add_argument("--use_subgenre", action="store_true")
    args = ap.parse_args()

    if args.make_queries:
        make_queries(use_subgenre=args.use_subgenre)

if __name__ == "__main__":
    main()
