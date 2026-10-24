"""
Checkpoint 2: turn structured features into descriptive "vibe" text.

Songs:  librosa features (+ estimated key mode) + iTunes genre -> mood scores
Movies: zero-shot similarity of the Wikipedia overview to each mood's
        anchor sentence + Wikidata genres -> mood scores

Both domains then go through the same template, built only from the
shared mood vocabulary in moods.py. That is deliberate: if song text
talked about BPM and movie text talked about plots, the embedding model
would mostly learn "this is a song" vs "this is a movie". Writing both in
one vocabulary makes the vectors comparable across domains.

Catalog statistics (feature means, per-mood centring, energy thresholds)
are saved to models/vibe_stats.json so out-of-catalog songs, movies and
free-text queries can be described at request time exactly the same way.

Run:
    python src/vibe_descriptions.py
"""

import json
import os

import numpy as np
import pandas as pd

from moods import MOODS, MOOD_ANCHORS, AUDIO_FEATURES, AUDIO_PROFILES, genre_prior
from embedder import embed

SONGS_PATH = "data/processed/songs.parquet"
MOVIES_PATH = "data/processed/movies.parquet"
SONGS_VIBES_PATH = "data/processed/songs_vibes.parquet"
MOVIES_VIBES_PATH = "data/processed/movies_vibes.parquet"
STATS_PATH = "models/vibe_stats.json"

# How hard genre labels pull on the mood scores. Movies get a stronger
# prior because Wikipedia intros for popular films are mostly cast, crew
# and box-office facts, which makes zero-shot mood scoring noisy. Tuned by
# checking e.g. how often horror films get eerie/dark/tense: 88% at 1.5,
# 94% at 5; past ~5 films collapse onto their genre's three moods.
SONG_PRIOR_WEIGHT = 1.5
TEXT_PRIOR_WEIGHT = 5.0
N_MOODS = 4          # moods per description

ENERGY_SENTENCES = {
    "high": "High-energy, intense and fast-paced.",
    "mid": "Steady, mid-tempo and balanced.",
    "low": "Slow, quiet and low-key.",
}

# Krumhansl-Kessler key profiles, for a rough major/minor estimate from chroma
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_mode(chroma_mean):
    """+1 if the chroma profile best matches a major key, -1 for minor."""
    chroma = np.asarray(chroma_mean, dtype=float)
    if chroma.std() < 1e-9:  # flat profile (near-silent clip) -- no key to find
        return 1.0
    best_major = max(np.corrcoef(chroma, np.roll(_MAJOR_PROFILE, k))[0, 1] for k in range(12))
    best_minor = max(np.corrcoef(chroma, np.roll(_MINOR_PROFILE, k))[0, 1] for k in range(12))
    return 1.0 if best_major >= best_minor else -1.0


def _song_feature_matrix(songs):
    """Raw (unscaled) feature matrix in AUDIO_FEATURES order."""
    major = np.array([estimate_mode(c) for c in songs["chroma_mean"]])
    return np.column_stack([songs["energy"], songs["tempo"], songs["brightness"], songs["zcr"], major])


def _song_raw_scores(raw_features, stats):
    z = (raw_features - np.array(stats["feature_mean"])) / np.array(stats["feature_std"])
    z[:, -1] = raw_features[:, -1]  # mode is already +-1, keep it as-is
    profiles = np.array([AUDIO_PROFILES[m] for m in MOODS])  # (M, F)
    return z @ profiles.T


_anchor_vecs = None


def _text_raw_scores(texts):
    global _anchor_vecs
    if _anchor_vecs is None:
        _anchor_vecs = embed([MOOD_ANCHORS[m] for m in MOODS])
    return embed(texts) @ _anchor_vecs.T


def _finalise(raw, centre, scale, priors, prior_weight):
    """Standardise each mood against the catalog (so no mood wins everywhere
    just because its anchor or audio profile is generically strong), z-score
    within each item, then add the genre prior."""
    x = (raw - np.array(centre)) / np.array(scale)
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-9)
    return x + prior_weight * np.asarray(priors)


def _energy_score(scores):
    idx = {m: i for i, m in enumerate(MOODS)}
    up = scores[:, idx["energetic"]] + scores[:, idx["chaotic"]] + scores[:, idx["aggressive"]]
    down = scores[:, idx["calm"]] + scores[:, idx["intimate"]] + scores[:, idx["dreamy"]]
    return up - down


def _describe(scores, energy_thresholds):
    """mood score matrix -> (top mood lists, energy levels, vibe texts)"""
    lo, hi = energy_thresholds
    energy = _energy_score(scores)
    all_moods, levels, texts = [], [], []
    for row, e in zip(scores, energy):
        top = [MOODS[i] for i in np.argsort(-row)[:N_MOODS]]
        level = "high" if e >= hi else "low" if e <= lo else "mid"
        article = "an" if top[3][0] in "aeiou" else "a"
        text = (f"{top[0].capitalize()}, {top[1]} and {top[2]}, with {article} {top[3]} undertone. "
                f"{ENERGY_SENTENCES[level]}")
        all_moods.append(top)
        levels.append(level)
        texts.append(text)
    return all_moods, levels, texts


# ---------- single-item helpers, used at request time ----------

def load_stats(path=STATS_PATH):
    with open(path) as f:
        return json.load(f)


def song_mood_scores(songs, stats):
    """(n, 24) mood scores for songs: DataFrame with energy/tempo/brightness/
    zcr/chroma_mean/genre."""
    raw = _song_raw_scores(_song_feature_matrix(songs), stats)
    priors = [genre_prior([g] if isinstance(g, str) else []) for g in songs["genre"]]
    return _finalise(raw, stats["song_centre"], stats["song_scale"], priors, SONG_PRIOR_WEIGHT)


def text_mood_scores(texts, genre_lists, stats, standardise=True):
    """(n, 24) mood scores for movie overviews or free-text queries.

    Pass standardise=False for short user queries: the catalog statistics
    come from long Wikipedia intros and skew a one-line query's moods
    ("rainy late-night drive" came out "triumphant" instead of "melancholic")."""
    raw = _text_raw_scores(texts)
    priors = [genre_prior(list(g)) for g in genre_lists]
    centre, scale = (stats["text_centre"], stats["text_scale"]) if standardise else (0.0, 1.0)
    return _finalise(raw, centre, scale, priors, TEXT_PRIOR_WEIGHT)


def describe_songs(songs, stats):
    return _describe(song_mood_scores(songs, stats), stats["song_energy_thresholds"])


def describe_texts(texts, genre_lists, stats, standardise=True):
    """Movie overviews, or free-text queries (with empty genre lists)."""
    return _describe(text_mood_scores(texts, genre_lists, stats, standardise), stats["text_energy_thresholds"])


# ---------- catalog build ----------

def _thresholds(scores):
    e = _energy_score(scores)
    return [float(np.percentile(e, 33)), float(np.percentile(e, 67))]


def build():
    songs = pd.read_parquet(SONGS_PATH)
    movies = pd.read_parquet(MOVIES_PATH)
    print(f"Describing {len(songs)} songs and {len(movies)} movies...")

    # catalog stats: songs
    feats = _song_feature_matrix(songs)
    stats = {
        "feature_names": AUDIO_FEATURES,
        "feature_mean": feats.mean(axis=0).tolist(),
        "feature_std": (feats.std(axis=0) + 1e-9).tolist(),
    }
    song_raw = _song_raw_scores(feats, stats)
    stats["song_centre"] = song_raw.mean(axis=0).tolist()
    stats["song_scale"] = (song_raw.std(axis=0) + 1e-9).tolist()
    song_priors = [genre_prior([g] if isinstance(g, str) else []) for g in songs["genre"]]
    stats["song_energy_thresholds"] = _thresholds(
        _finalise(song_raw, stats["song_centre"], stats["song_scale"], song_priors, SONG_PRIOR_WEIGHT))

    # catalog stats: movies
    movie_raw = _text_raw_scores(movies["overview"].tolist())
    stats["text_centre"] = movie_raw.mean(axis=0).tolist()
    stats["text_scale"] = (movie_raw.std(axis=0) + 1e-9).tolist()
    movie_priors = [genre_prior(list(g)) for g in movies["genres"]]
    stats["text_energy_thresholds"] = _thresholds(
        _finalise(movie_raw, stats["text_centre"], stats["text_scale"], movie_priors, TEXT_PRIOR_WEIGHT))

    os.makedirs("models", exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    songs["mode"] = np.where(feats[:, -1] > 0, "major", "minor")
    songs["moods"], songs["energy_level"], songs["vibe_text"] = describe_songs(songs, stats)
    movies["moods"], movies["energy_level"], movies["vibe_text"] = describe_texts(
        movies["overview"].tolist(), movies["genres"], stats)

    songs.to_parquet(SONGS_VIBES_PATH, index=False)
    movies.to_parquet(MOVIES_VIBES_PATH, index=False)
    print(f"Saved {SONGS_VIBES_PATH}, {MOVIES_VIBES_PATH} and {STATS_PATH}")

    for name, df, label in [("songs", songs, "track_name"), ("movies", movies, "title")]:
        print(f"\nSample {name}:")
        for _, r in df.sample(min(5, len(df)), random_state=1).iterrows():
            print(f"  {r[label]}: {r['vibe_text']}")


if __name__ == "__main__":
    build()
