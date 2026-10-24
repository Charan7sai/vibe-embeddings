"""
Cached CLAP audio embeddings, keyed by iTunes track id. Embedding a
preview takes ~0.5s on CPU, so each track is only ever embedded once.

    python src/clap_cache.py     # embed every catalog + soundtrack preview
"""

import os

import numpy as np
import pandas as pd

import clap_audio

CACHE_PATH = "data/processed/clap_audio_cache.npz"
PREVIEW_DIR = "data/raw/previews"


def load():
    if not os.path.exists(CACHE_PATH):
        return {}
    data = np.load(CACHE_PATH)
    return dict(zip(data["ids"].tolist(), data["vecs"]))


def save(cache):
    ids = np.array(list(cache.keys()), dtype=np.int64)
    vecs = np.stack([cache[i] for i in ids]) if len(ids) else np.zeros((0, 512), np.float32)
    np.savez(CACHE_PATH, ids=ids, vecs=vecs)


def embeddings(track_ids, batch_size=16, verbose=True):
    """(n, 512) CLAP vectors for these tracks, computing any not yet cached
    from data/raw/previews/<id>.wav. Rows are NaN where the preview is missing."""
    cache = load()
    todo = [int(t) for t in dict.fromkeys(track_ids) if int(t) not in cache
            and os.path.exists(f"{PREVIEW_DIR}/{int(t)}.wav")]
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        vecs = clap_audio.embed_audio([f"{PREVIEW_DIR}/{t}.wav" for t in batch])
        cache.update(zip(batch, vecs))
        if verbose and (i // batch_size) % 10 == 0:
            print(f"  CLAP: {i + len(batch)}/{len(todo)} previews embedded")
        if (i // batch_size) % 25 == 0:
            save(cache)
    if todo:
        save(cache)
    out = np.full((len(track_ids), 512), np.nan, dtype=np.float32)
    for k, t in enumerate(track_ids):
        if int(t) in cache:
            out[k] = cache[int(t)]
    return out


if __name__ == "__main__":
    songs = pd.read_parquet("data/processed/songs.parquet")
    pairs = pd.read_parquet("data/processed/soundtrack_pairs.parquet")
    pairs = pairs[pairs["wikidata_id"].notna()]
    ids = songs["track_id"].astype(int).tolist() + pairs["track_id"].astype(int).tolist()
    vecs = embeddings(ids)
    print(f"Done: {int((~np.isnan(vecs[:, 0])).sum())}/{len(ids)} tracks have CLAP embeddings.")
