"""
Checkpoint 6: sanity evaluation.

Two parts:
  1. Intrinsic (no humans needed): for same-domain recs, do the top-10
     neighbours share a genre with the query more often than random items?
  2. Human survey: from survey/responses.csv, how often do raters pick the
     model's top candidate out of {top, middle, bottom}? Chance is 1/3.

Run:
    python src/evaluate.py
"""

import os
import random

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from recommender import VibeRecommender, ROOT

RESPONSES_PATH = os.path.join(ROOT, "survey", "responses.csv")


def genre_precision(rec, domain, n_queries=300, k=10, seed=0):
    """Share of top-k same-domain recs that share a genre with the query,
    vs. the same measure for k random catalog items."""
    rng = random.Random(seed)
    df = rec.songs if domain == "song" else rec.movies
    genres_of = (lambda r: {r.genre}) if domain == "song" else (lambda r: set(r.genres))
    id_col = "track_id" if domain == "song" else "wikidata_id"
    # "drama film" alone is on a huge share of movies -- it makes any
    # baseline look good, so leave it out of the overlap check
    ignore = {"drama film"}

    model_hits, random_hits = [], []
    for i in rng.sample(range(len(df)), min(n_queries, len(df))):
        row = df.iloc[i]
        q_genres = genres_of(row) - ignore
        if not q_genres:
            continue
        item_id = int(row[id_col]) if domain == "song" else row[id_col]
        recs = rec.recommend({"type": domain, "id": item_id}, target=domain, k=k)["results"]
        rec_genres = [({r["genre"]} if domain == "song" else set(r["genres"])) - ignore for r in recs]
        model_hits.append(np.mean([bool(q_genres & g) for g in rec_genres]))
        rand_rows = df.iloc[rng.sample(range(len(df)), k)]
        random_hits.append(np.mean([bool(q_genres & (genres_of(r) - ignore)) for r in rand_rows.itertuples()]))
    return float(np.mean(model_hits)), float(np.mean(random_hits))


def survey_report(path=RESPONSES_PATH):
    if not os.path.exists(path):
        print("No survey responses yet -- run `streamlit run survey/app.py` to collect some.")
        return
    df = pd.read_csv(path)
    answered = df[df["choice_rank"] != "none"]
    print(f"{len(df)} responses from {df['rater'].nunique()} raters "
          f"({len(df) - len(answered)} skipped as 'none fit').")
    if answered.empty:
        return

    def line(sub, label):
        hits = int((sub["choice_rank"] == "top").sum())
        n = len(sub)
        p = binomtest(hits, n, 1 / 3, alternative="greater").pvalue
        dist = sub["choice_rank"].value_counts(normalize=True).reindex(["top", "middle", "bottom"]).fillna(0)
        print(f"  {label:<12} n={n:<4} picked model's top: {hits / n:6.1%}  (chance 33.3%, p={p:.4f})   "
              f"top/mid/bottom = {dist['top']:.0%}/{dist['middle']:.0%}/{dist['bottom']:.0%}")

    line(answered, "overall")
    for direction, sub in answered.groupby("direction"):
        line(sub, direction)


if __name__ == "__main__":
    rec = VibeRecommender()
    print("== Intrinsic: genre overlap of same-domain top-10 ==")
    for domain in ("song", "movie"):
        model, baseline = genre_precision(rec, domain)
        print(f"  {domain:<6} model {model:.1%}   random {baseline:.1%}   lift x{model / baseline:.2f}")

    print("\n== Human survey ==")
    survey_report()
