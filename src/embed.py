"""
Checkpoints 3 + 4: embed both catalogs into the shared vector space and
build genre centroid embeddings.

Outputs (models/):
  songs.parquet, movies.parquet      catalog metadata + moods + vibe text
  song_vibe.npy, movie_vibe.npy      shared-space vibe embeddings (cross-domain)
  movie_content.npy                  overview embeddings (movie -> movie only)
  song_audio.npy                     standardised audio vectors (song -> song only)
  genres.parquet                     genre names, sizes and characteristic moods
  genre_vibe.npy                     distinctive genre directions (genre <-> genre)
  genre_mean.npy                     plain genre centroids (genre -> song / movie)
  song_clap.npy, movie_clap.npy,     the song <-> film matching space, see cross_space.py
  song_mood.npy, movie_mood.npy,
  song_title_mood.npy, cross_space.npz

Run (after vibe_descriptions.py, soundtrack_benchmark.py and clap_cache.py):
    python src/embed.py
"""

import json
import os

import numpy as np
import pandas as pd

import cross_space
from embedder import embed
from vibe_descriptions import (SONGS_VIBES_PATH, MOVIES_VIBES_PATH, STATS_PATH, load_stats, estimate_mode,
                               _describe, _thresholds)

MODELS_DIR = "models"

MIN_SONGS_PER_GENRE = 5
MIN_MOVIES_PER_GENRE = 10


def _normalise(x):
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def audio_vectors(df, stats):
    """Standardised [energy, tempo, brightness, zcr, mode] + chroma shape.
    Used as a secondary signal for song -> song so two songs with the same
    mood words but very different sound don't tie."""
    mode = np.array([estimate_mode(c) for c in df["chroma_mean"]])
    raw = np.column_stack([df["energy"], df["tempo"], df["brightness"], df["zcr"], mode])
    z = (raw - np.array(stats["feature_mean"])) / np.array(stats["feature_std"])
    z[:, -1] = mode
    chroma = np.stack(df["chroma_mean"].to_numpy())
    chroma = (chroma - chroma.mean(axis=1, keepdims=True)) / (chroma.std(axis=1, keepdims=True) + 1e-9)
    return _normalise(np.hstack([z, 0.3 * chroma]))


def _genre_centroids(labels, vecs, moods, domain, min_count):
    """labels: one list of genre names per item; moods: one mood list per item.

    Two vectors per genre. A plain mean of a few hundred vectors drifts
    towards the catalog mean, so every genre ends up ~0.98 similar to every
    other -- subtracting the domain mean keeps only what's distinctive, which
    is what genre <-> genre comparison needs. But that difference vector is
    the wrong thing to search songs/movies with (it rewards items that are
    merely far from the film average: film noir -> AC/DC), so the plain mean
    is kept for genre -> item queries."""
    domain_mean = vecs.mean(axis=0)
    base_rate = pd.Series([m for ms in moods for m in ms]).value_counts(normalize=True)
    by_genre = {}
    for i, gs in enumerate(labels):
        for g in set(gs):
            by_genre.setdefault(g, []).append(i)
    rows, distinct, means = [], [], []
    for g, idx in sorted(by_genre.items()):
        if len(idx) < min_count:
            continue
        # characteristic moods: most over-represented vs. the whole domain
        freq = pd.Series([m for i in idx for m in moods[i]]).value_counts(normalize=True)
        top = (freq / base_rate.reindex(freq.index)).sort_values(ascending=False).index[:4].tolist()
        rows.append({"genre_id": f"{domain}:{g}", "name": g, "domain": domain,
                     "count": len(idx), "moods": top})
        mean = vecs[idx].mean(axis=0)
        distinct.append(mean - domain_mean)
        means.append(mean)
    return rows, distinct, means


def build():
    songs = pd.read_parquet(SONGS_VIBES_PATH)
    movies = pd.read_parquet(MOVIES_VIBES_PATH)
    stats = load_stats()

    # the same track can show up under several genre charts, or as a remaster
    songs = songs.drop_duplicates("track_id").reset_index(drop=True)
    movies = movies.drop_duplicates("wikidata_id").reset_index(drop=True)
    print(f"Embedding {len(songs)} songs and {len(movies)} movies...")

    # song <-> film matching space (CLAP + learned map + titles). Its mood
    # scores replace the hand-written audio rules for what song cards say,
    # so the moods shown are the ones the matching actually used.
    song_scores = cross_space.build(songs, movies, stats)
    stats["song_energy_thresholds"] = _thresholds(song_scores)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    songs["moods"], songs["energy_level"], songs["vibe_text"] = _describe(song_scores, stats["song_energy_thresholds"])

    song_vibe = embed(songs["vibe_text"].tolist())
    movie_vibe = embed(movies["vibe_text"].tolist())
    movie_content = embed(movies["overview"].tolist())
    song_audio = audio_vectors(songs, stats)

    song_rows, song_distinct, song_means = _genre_centroids(
        [[g] if isinstance(g, str) else [] for g in songs["genre"]], song_vibe,
        [list(m) for m in songs["moods"]], "music", MIN_SONGS_PER_GENRE)
    movie_rows, movie_distinct, movie_means = _genre_centroids(
        [list(g) for g in movies["genres"]], movie_vibe,
        [list(m) for m in movies["moods"]], "film", MIN_MOVIES_PER_GENRE)
    genres = pd.DataFrame(song_rows + movie_rows)
    genre_vibe = _normalise(np.array(song_distinct + movie_distinct))
    genre_mean = _normalise(np.array(song_means + movie_means))

    os.makedirs(MODELS_DIR, exist_ok=True)
    songs.to_parquet(f"{MODELS_DIR}/songs.parquet", index=False)
    movies.to_parquet(f"{MODELS_DIR}/movies.parquet", index=False)
    genres.to_parquet(f"{MODELS_DIR}/genres.parquet", index=False)
    np.save(f"{MODELS_DIR}/song_vibe.npy", song_vibe)
    np.save(f"{MODELS_DIR}/movie_vibe.npy", movie_vibe)
    np.save(f"{MODELS_DIR}/movie_content.npy", movie_content)
    np.save(f"{MODELS_DIR}/song_audio.npy", song_audio)
    np.save(f"{MODELS_DIR}/genre_vibe.npy", genre_vibe)
    np.save(f"{MODELS_DIR}/genre_mean.npy", genre_mean)

    print(f"Saved to {MODELS_DIR}/: {len(songs)} songs, {len(movies)} movies, "
          f"{len(song_rows)} music genres, {len(movie_rows)} film genres.")


if __name__ == "__main__":
    build()
