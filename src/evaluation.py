import argparse
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd

from src.config import (
    CLEAN_CSV,
    COL_GENRE,
    COL_SUBGENRE,
    COL_TRACK_NAME,
    COL_ARTIST,
    COL_PLAYLIST_ID,
    COL_ALBUM_ID,
    COL_TRACK_POPULARITY,
)
from src.recommender import Recommender, RecConfig

@dataclass
class EvalConfig:
    use_subgenre: bool = False
    n_queries: int = 8
    seeds_per_query: int = 4
    K: int = 10
    random_state: int = 42
    ground_truth: str = "playlist"  # playlist | genre | subgenre | album | artist

def _seed_list(df: pd.DataFrame, idxs: Iterable[int]) -> list[str]:
    seeds = df.loc[list(idxs), COL_TRACK_NAME] + " — " + df.loc[list(idxs), COL_ARTIST]
    return seeds.tolist()

def _majority_label(series: pd.Series) -> str:
    if series.empty:
        return ""
    return str(series.value_counts().index[0])

def _eligible_groups(df: pd.DataFrame, group_col: str, min_size: int) -> list:
    vc = df[group_col].dropna().value_counts()
    return vc[vc >= min_size].index.tolist()

def split_groups(df: pd.DataFrame, group_col: str, test_frac: float, val_frac: float, random_state: int):
    groups = _eligible_groups(df, group_col, min_size=2)
    rng = np.random.default_rng(random_state)
    rng.shuffle(groups)
    n_total = len(groups)
    n_test = int(n_total * test_frac)
    n_val = int(n_total * val_frac)
    test = groups[:n_test]
    val = groups[n_test:n_test + n_val]
    train = groups[n_test + n_val:]
    return {"train": train, "val": val, "test": test}

def _build_group_queries(
    df: pd.DataFrame,
    group_col: str,
    use_subgenre: bool,
    n_queries: int,
    seeds_per_query: int,
    random_state: int,
    group_ids: list | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    if group_ids is None:
        groups = _eligible_groups(df, group_col, seeds_per_query + 1)
    else:
        groups = [g for g in group_ids if g in set(_eligible_groups(df, group_col, seeds_per_query + 1))]
    if not groups:
        return pd.DataFrame()
    sampled = rng.choice(groups, size=min(n_queries, len(groups)), replace=False)

    rows = []
    for g in sampled:
        subset = df[df[group_col] == g]
        seeds = subset.sample(seeds_per_query, random_state=random_state + 7).index
        selected_genre = _majority_label(subset[COL_SUBGENRE if use_subgenre else COL_GENRE])
        rows.append({
            "query_id": f"Q_{group_col}_{g}",
            "group_value": g,
            "selected_genre": selected_genre,
            "seeds": " | ".join(_seed_list(df, seeds)),
            "ground_truth": group_col
        })
    return pd.DataFrame(rows)

def generate_queries_df(
    df: pd.DataFrame,
    use_subgenre: bool = False,
    n_queries: int = 8,
    seeds_per_query: int = 4,
    random_state: int = 42,
    ground_truth: str = "genre",
    group_ids: list | None = None,
) -> pd.DataFrame:
    if ground_truth == "playlist":
        return _build_group_queries(df, COL_PLAYLIST_ID, use_subgenre, n_queries, seeds_per_query, random_state, group_ids)
    if ground_truth == "album":
        return _build_group_queries(df, COL_ALBUM_ID, use_subgenre, n_queries, seeds_per_query, random_state, group_ids)
    if ground_truth == "artist":
        return _build_group_queries(df, COL_ARTIST, use_subgenre, n_queries, seeds_per_query, random_state, group_ids)
    if ground_truth == "subgenre":
        use_subgenre = True
    genre_col = COL_SUBGENRE if use_subgenre else COL_GENRE
    genres = df[genre_col].value_counts().head(n_queries).index.tolist()
    queries = []
    for g in genres:
        subset = df[df[genre_col] == g].sample(
            min(50, (df[genre_col] == g).sum()),
            random_state=random_state
        )
        seeds = subset.sample(seeds_per_query, random_state=random_state + 7).index
        queries.append({
            "query_id": f"Q_{g}",
            "selected_genre": g,
            "group_value": g,
            "seeds": " | ".join(_seed_list(df, seeds)),
            "ground_truth": genre_col
        })
    return pd.DataFrame(queries)

def make_queries(use_subgenre: bool = False, n_queries: int = 8, seeds_per_query: int = 4, ground_truth: str = "genre"):
    df = pd.read_csv(CLEAN_CSV)
    out = generate_queries_df(
        df,
        use_subgenre=use_subgenre,
        n_queries=n_queries,
        seeds_per_query=seeds_per_query,
        ground_truth=ground_truth
    )
    out.to_csv("evaluation_queries.csv", index=False)
    print("Wrote evaluation_queries.csv")

def _precision_at_k(recs: list[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for r in recs[:k] if r in relevant)
    return hits / k

def _recall_at_k(recs: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in recs[:k] if r in relevant)
    return hits / len(relevant)

def _hit_rate_at_k(recs: list[int], relevant: set[int], k: int) -> float:
    return 1.0 if any(r in relevant for r in recs[:k]) else 0.0

def _ndcg_at_k(recs: list[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    dcg = 0.0
    for i, r in enumerate(recs[:k]):
        if r in relevant:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return (dcg / idcg) if idcg > 0 else 0.0

def _parse_seeds(seed_str: str) -> list[str]:
    return [s.strip() for s in seed_str.split("|") if s.strip()]

def _relevant_set(df: pd.DataFrame, ground_truth: str, group_value, seed_idxs: list[int]) -> set[int]:
    if ground_truth == COL_PLAYLIST_ID:
        mask = df[COL_PLAYLIST_ID] == group_value
    elif ground_truth == COL_ALBUM_ID:
        mask = df[COL_ALBUM_ID] == group_value
    elif ground_truth == COL_ARTIST:
        mask = df[COL_ARTIST] == group_value
    elif ground_truth in (COL_GENRE, COL_SUBGENRE):
        g_norm = str(group_value).strip().lower()
        mask = df[ground_truth].astype(str).str.strip().str.lower().eq(g_norm)
    else:
        return set()
    relevant = set(df[mask].index.tolist())
    return relevant - set(seed_idxs)

def _diversity_metrics(df: pd.DataFrame, rec_idxs: list[int], genre_col: str) -> dict:
    if not rec_idxs:
        return {"artist_diversity": 0.0, "genre_diversity": 0.0}
    artists = df.loc[rec_idxs, COL_ARTIST].astype(str)
    genres = df.loc[rec_idxs, genre_col].astype(str)
    return {
        "artist_diversity": artists.nunique() / len(rec_idxs),
        "genre_diversity": genres.nunique() / len(rec_idxs),
    }

def _novelty(df: pd.DataFrame, rec_idxs: list[int]) -> float:
    if not rec_idxs or COL_TRACK_POPULARITY not in df.columns:
        return 0.0
    pop = pd.to_numeric(df.loc[rec_idxs, COL_TRACK_POPULARITY], errors="coerce").fillna(0.0)
    return float((1.0 - (pop / 100.0)).mean())

def evaluate_queries(
    rec: Recommender,
    queries_df: pd.DataFrame,
    cfg: RecConfig,
    use_subgenre: bool,
    ground_truth: str
) -> pd.DataFrame:
    genre_col = COL_SUBGENRE if use_subgenre else COL_GENRE
    rows = []
    for _, row in queries_df.iterrows():
        seeds = _parse_seeds(row["seeds"])
        selected_genre = row.get("selected_genre", "")
        seed_idxs = rec._seed_indices(seeds)
        if len(seed_idxs) == 0:
            continue
        relevant = _relevant_set(rec.df, ground_truth, row["group_value"], seed_idxs)
        out = rec.recommend(selected_genre=selected_genre, seeds_title_artist=seeds, cfg=cfg)
        rec_idxs = out["row_idx"].tolist()

        div = _diversity_metrics(rec.df, rec_idxs, genre_col)
        rows.append({
            "query_id": row["query_id"],
            "group_value": row["group_value"],
            "ground_truth": ground_truth,
            "model_type": cfg.model_type,
            "K": cfg.K,
            "precision@K": _precision_at_k(rec_idxs, relevant, cfg.K),
            "recall@K": _recall_at_k(rec_idxs, relevant, cfg.K),
            "ndcg@K": _ndcg_at_k(rec_idxs, relevant, cfg.K),
            "hit_rate@K": _hit_rate_at_k(rec_idxs, relevant, cfg.K),
            "n_relevant": len(relevant),
            "artist_diversity": div["artist_diversity"],
            "genre_diversity": div["genre_diversity"],
            "novelty": _novelty(rec.df, rec_idxs),
            "rec_idxs": rec_idxs,
        })
    return pd.DataFrame(rows)

def evaluate_configs(rec: Recommender, queries_df: pd.DataFrame, cfgs, use_subgenre: bool, ground_truth: str):
    per_query = []
    for item in cfgs:
        if isinstance(item, tuple):
            name, cfg = item
        else:
            cfg = item
            name = cfg.model_type
        dfq = evaluate_queries(rec, queries_df, cfg, use_subgenre, ground_truth)
        if not dfq.empty:
            dfq["model_type"] = name
        per_query.append(dfq)
    per_query_df = pd.concat(per_query, ignore_index=True) if per_query else pd.DataFrame()

    if per_query_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        per_query_df.groupby("model_type")[
            ["precision@K", "recall@K", "ndcg@K", "hit_rate@K", "artist_diversity", "genre_diversity", "novelty"]
        ]
        .mean()
        .reset_index()
        .sort_values(by="ndcg@K", ascending=False)
    )

    # coverage per model (unique recommended items across queries)
    coverage_rows = []
    total_items = len(rec.df)
    for model, dfm in per_query_df.groupby("model_type"):
        all_recs = set()
        for rec_idxs in dfm["rec_idxs"].tolist():
            all_recs.update(rec_idxs)
        coverage = len(all_recs)
        coverage_rows.append({
            "model_type": model,
            "coverage": coverage,
            "coverage_ratio": coverage / total_items if total_items else 0.0
        })
    coverage_df = pd.DataFrame(coverage_rows)
    summary = summary.merge(coverage_df, on="model_type", how="left")

    return summary, per_query_df

def demo_subset(rec: Recommender, selected_genre: str, seeds: list[str], cfg: RecConfig, n_show: int = 5) -> pd.DataFrame:
    out = rec.recommend(selected_genre=selected_genre, seeds_title_artist=seeds, cfg=cfg)
    cols = ["track_name", "artist", "genre", "text_sim", "audio_sim", "score"]
    return out[cols].head(n_show)

def _baseline_recommendations(
    df: pd.DataFrame,
    seed_idxs: list[int],
    selected_genre: str,
    baseline: str,
    K: int,
    rng: np.random.Generator
) -> list[int]:
    seed_set = set(seed_idxs)
    if baseline == "random":
        candidates = [i for i in range(len(df)) if i not in seed_set]
        rng.shuffle(candidates)
        return candidates[:K]

    if baseline == "popular":
        pop = pd.to_numeric(df[COL_TRACK_POPULARITY], errors="coerce").fillna(0.0)
        order = pop.sort_values(ascending=False).index.tolist()
        return [i for i in order if i not in seed_set][:K]

    if baseline == "same_genre":
        g = str(selected_genre).strip().lower()
        mask = df[COL_GENRE].astype(str).str.strip().str.lower().eq(g)
        subset = df[mask]
        pop = pd.to_numeric(subset[COL_TRACK_POPULARITY], errors="coerce").fillna(0.0)
        order = pop.sort_values(ascending=False).index.tolist()
        return [i for i in order if i not in seed_set][:K]

    if baseline == "same_artist":
        artists = df.loc[seed_idxs, COL_ARTIST].astype(str).unique().tolist()
        subset = df[df[COL_ARTIST].isin(artists)]
        pop = pd.to_numeric(subset[COL_TRACK_POPULARITY], errors="coerce").fillna(0.0)
        order = pop.sort_values(ascending=False).index.tolist()
        return [i for i in order if i not in seed_set][:K]

    return []

def evaluate_baselines(
    rec: Recommender,
    queries_df: pd.DataFrame,
    baselines: list[str],
    use_subgenre: bool,
    ground_truth: str,
    K: int,
    random_state: int
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    genre_col = COL_SUBGENRE if use_subgenre else COL_GENRE

    for baseline in baselines:
        for _, row in queries_df.iterrows():
            seeds = _parse_seeds(row["seeds"])
            selected_genre = row.get("selected_genre", "")
            seed_idxs = rec._seed_indices(seeds)
            if len(seed_idxs) == 0:
                continue
            relevant = _relevant_set(rec.df, ground_truth, row["group_value"], seed_idxs)
            rec_idxs = _baseline_recommendations(rec.df, seed_idxs, selected_genre, baseline, K, rng)
            div = _diversity_metrics(rec.df, rec_idxs, genre_col)
            rows.append({
                "query_id": row["query_id"],
                "group_value": row["group_value"],
                "ground_truth": ground_truth,
                "model_type": f"baseline_{baseline}",
                "K": K,
                "precision@K": _precision_at_k(rec_idxs, relevant, K),
                "recall@K": _recall_at_k(rec_idxs, relevant, K),
                "ndcg@K": _ndcg_at_k(rec_idxs, relevant, K),
                "hit_rate@K": _hit_rate_at_k(rec_idxs, relevant, K),
                "n_relevant": len(relevant),
                "artist_diversity": div["artist_diversity"],
                "genre_diversity": div["genre_diversity"],
                "novelty": _novelty(rec.df, rec_idxs),
            })
    return pd.DataFrame(rows)

def build_ablation_configs(base: RecConfig) -> list[tuple[str, RecConfig]]:
    return [
        ("base", base),
        ("no_text", RecConfig(**{**base.__dict__, "alpha": 0.0})),
        ("no_audio", RecConfig(**{**base.__dict__, "alpha": 1.0})),
        ("no_genre_bonus", RecConfig(**{**base.__dict__, "genre_bonus": 0.0})),
        ("no_diversity", RecConfig(**{**base.__dict__, "max_per_artist": 999})),
        ("no_clusters", RecConfig(**{**base.__dict__, "use_clusters": False})),
        ("no_assoc", RecConfig(**{**base.__dict__, "use_assoc": False})),
    ]

def export_outliers(rec: Recommender, top_n: int = 50):
    df = rec.df.copy()
    df["outlier_score"] = rec.outlier_scores
    df["outlier_flag"] = rec.outlier_flags
    out = df.sort_values("outlier_score").head(top_n)
    out.to_csv("outliers_top.csv", index=False)
    print("Wrote outliers_top.csv")

def export_clusters(rec: Recommender):
    df = rec.df.copy()
    df["audio_cluster"] = rec.audio_clusters
    df.to_csv("audio_clusters.csv", index=False)
    print("Wrote audio_clusters.csv")

def export_assoc_rules(rec: Recommender, top_n: int = 200):
    rows = []
    for src_id, pairs in rec.assoc_rules.items():
        for tgt_id, conf in pairs:
            rows.append({"track_id": src_id, "co_track_id": tgt_id, "confidence": conf})
    if not rows:
        print("No association rules found.")
        return
    df = pd.DataFrame(rows).sort_values("confidence", ascending=False).head(top_n)
    df.to_csv("association_rules_top.csv", index=False)
    print("Wrote association_rules_top.csv")

def export_failure_cases(per_query: pd.DataFrame, top_n: int = 20):
    if per_query.empty:
        return
    cols = ["query_id", "model_type", "ndcg@K", "precision@K", "recall@K", "hit_rate@K", "n_relevant"]
    out = per_query.sort_values("ndcg@K").head(top_n)
    out[cols].to_csv("evaluation_failures.csv", index=False)
    print("Wrote evaluation_failures.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_queries", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--demo_subset", action="store_true")
    ap.add_argument("--use_subgenre", action="store_true")
    ap.add_argument("--n_queries", type=int, default=8)
    ap.add_argument("--seeds_per_query", type=int, default=4)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--models", type=str, default="SBERT_COSINE,TFIDF_COSINE,W2V_COSINE,BOW_COSINE,JACCARD,SBERT_EUCLIDEAN")
    ap.add_argument("--ground_truth", type=str, default="playlist")
    ap.add_argument("--run_baselines", action="store_true")
    ap.add_argument("--run_ablations", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--genre_bonus", type=float, default=0.02)
    ap.add_argument("--cluster_bonus", type=float, default=0.02)
    ap.add_argument("--assoc_bonus", type=float, default=0.05)
    ap.add_argument("--split_groups", action="store_true")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--export_outliers", action="store_true")
    ap.add_argument("--export_clusters", action="store_true")
    ap.add_argument("--export_assoc", action="store_true")
    ap.add_argument("--export_failures", action="store_true")
    args = ap.parse_args()

    if args.make_queries:
        make_queries(use_subgenre=args.use_subgenre, n_queries=args.n_queries, seeds_per_query=args.seeds_per_query, ground_truth=args.ground_truth)
        return

    rec = Recommender()
    df = rec.df
    group_ids = None
    if args.split_groups and args.ground_truth in {"playlist", "album", "artist"}:
        group_col = {
            "playlist": COL_PLAYLIST_ID,
            "album": COL_ALBUM_ID,
            "artist": COL_ARTIST
        }[args.ground_truth]
        splits = split_groups(df, group_col, test_frac=args.test_frac, val_frac=args.val_frac, random_state=42)
        group_ids = splits["test"]

    queries_df = generate_queries_df(
        df,
        use_subgenre=args.use_subgenre,
        n_queries=args.n_queries,
        seeds_per_query=args.seeds_per_query,
        ground_truth=args.ground_truth,
        group_ids=group_ids
    )

    if queries_df.empty:
        print("No queries generated for this ground truth.")
        return

    if args.demo_subset:
        first = queries_df.iloc[0]
        seeds = _parse_seeds(first["seeds"])
        cfg = RecConfig(model_type="SBERT_COSINE", K=args.K, use_subgenre=args.use_subgenre)
        demo = demo_subset(rec, first["selected_genre"], seeds, cfg, n_show=min(5, args.K))
        print(demo.to_string(index=False))
        return

    if args.export_outliers:
        export_outliers(rec)
    if args.export_clusters:
        export_clusters(rec)
    if args.export_assoc:
        export_assoc_rules(rec)

    if args.evaluate:
        model_list = [m.strip() for m in args.models.split(",") if m.strip()]
        cfgs = [
            RecConfig(
                model_type=m,
                alpha=args.alpha,
                genre_bonus=args.genre_bonus,
                cluster_bonus=args.cluster_bonus,
                assoc_bonus=args.assoc_bonus,
                K=args.K,
                use_subgenre=args.use_subgenre
            )
            for m in model_list
        ]
        summary, per_query = evaluate_configs(rec, queries_df, cfgs, args.use_subgenre, ground_truth=queries_df.iloc[0]["ground_truth"])
        print(summary.to_string(index=False))
        per_query.drop(columns=["rec_idxs"], errors="ignore").to_csv("evaluation_results.csv", index=False)
        print("Wrote evaluation_results.csv")
        if args.export_failures:
            export_failure_cases(per_query)

        if args.run_baselines:
            baselines = ["random", "popular", "same_genre", "same_artist"]
            base_df = evaluate_baselines(rec, queries_df, baselines, args.use_subgenre, queries_df.iloc[0]["ground_truth"], args.K, 42)
            base_df.to_csv("evaluation_baselines.csv", index=False)
            print("Wrote evaluation_baselines.csv")

        if args.run_ablations:
            base_cfg = RecConfig(
                model_type="SBERT_COSINE",
                alpha=args.alpha,
                genre_bonus=args.genre_bonus,
                cluster_bonus=args.cluster_bonus,
                assoc_bonus=args.assoc_bonus,
                K=args.K,
                use_subgenre=args.use_subgenre
            )
            ablations = build_ablation_configs(base_cfg)
            summary_ab, per_query_ab = evaluate_configs(rec, queries_df, ablations, args.use_subgenre, ground_truth=queries_df.iloc[0]["ground_truth"])
            summary_ab.to_csv("evaluation_ablations_summary.csv", index=False)
            per_query_ab.drop(columns=["rec_idxs"], errors="ignore").to_csv("evaluation_ablations.csv", index=False)
            print("Wrote evaluation_ablations.csv")

if __name__ == "__main__":
    main()
