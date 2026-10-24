"""
Ground truth for cross-domain matching: films and songs from their own
soundtracks.

There's no labelled dataset of "this song feels like this film", but a
film's soundtrack was chosen or written to fit it, so a good song -> film
model should rank a soundtrack song's own film near the top, and vice
versa. This script collects those pairs for catalog films from iTunes,
with the same audio features as the song catalog.

Run (resumable):
    python src/soundtrack_benchmark.py --films 450     # search iTunes
    python src/soundtrack_benchmark.py --features      # audio features, in parallel
Writes data/processed/soundtrack_pairs.parquet
"""

import argparse
import os
import re
import time

import pandas as pd
import requests

import audio_features

PAIRS_PATH = "data/processed/soundtrack_pairs.parquet"
MOVIES_PATH = "models/movies.parquet"
SEARCH_URL = "https://itunes.apple.com/search"
PER_FILM = 4

# albums that borrow the film's name but aren't its soundtrack
NOT_THE_SOUNDTRACK = re.compile(r"lo-?fi|remix|cover|tribute|karaoke|piano|lullab|"
                                r"8-bit|instrumental version|music box|guitar|chill|relax", re.I)
SUFFIXES = ("", " the album", " soundtrack", " the soundtrack", " original soundtrack",
            " original motion picture soundtrack", " music from the motion picture",
            " original motion picture score", " the score", " score")


def _norm(s):
    return re.sub(r"[^a-z0-9 ]+", "", s.lower().replace("&", "and")).strip()


def is_soundtrack_of(collection, title):
    """'Sinners (Original Motion Picture Soundtrack)' is Sinners' soundtrack;
    'Barbie: The Pearl Princess (Music from the Motion Picture)' isn't Barbie's."""
    if not collection or NOT_THE_SOUNDTRACK.search(collection):
        return False
    base = re.split(r" \(| - (single|ep)\b", collection, flags=re.I)[0]
    base, t = _norm(base), _norm(title)
    if not any(base == t + s for s in SUFFIXES):
        return False
    # a bare-title album only counts if it says it's from the film
    if base == t and not re.search(r"soundtrack|motion picture|score|from the film|the album", collection, re.I):
        return False
    return True


def find_soundtrack(title, limit=PER_FILM):
    resp = requests.get(SEARCH_URL, timeout=30, params={
        "term": f"{title} soundtrack", "media": "music", "entity": "song", "limit": 50, "country": "us"})
    if resp.status_code != 200:
        raise RuntimeError(f"iTunes search returned {resp.status_code}")
    tracks, seen = [], set()
    for r in resp.json().get("results", []):
        if not r.get("previewUrl") or not is_soundtrack_of(r.get("collectionName", ""), title):
            continue
        key = _norm(r.get("trackName", ""))
        if key in seen:
            continue
        seen.add(key)
        tracks.append({"track_id": r["trackId"], "track_name": r["trackName"], "artist_name": r["artistName"],
                       "genre": r.get("primaryGenreName"), "preview_url": r["previewUrl"],
                       "release_date": r.get("releaseDate"), "collection": r.get("collectionName")})
        if len(tracks) >= limit:
            break
    return tracks


def collect(n_films, seed=0):
    movies = pd.read_parquet(MOVIES_PATH)
    # recent films are far more likely to have their soundtrack on iTunes
    pool = movies[movies["year"].astype(int) >= 1990].sample(frac=1.0, random_state=seed)
    existing = pd.read_parquet(PAIRS_PATH) if os.path.exists(PAIRS_PATH) else pd.DataFrame()
    searched = set(existing["searched_film"]) if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []

    todo = [m for m in pool.itertuples() if m.wikidata_id not in searched][: max(0, n_films - len(searched))]
    print(f"{len(searched)} films already searched, {len(todo)} to go.")
    # phase 1: search only -- the rate limit, not audio, sets the pace here
    for k, m in enumerate(todo, 1):
        try:
            tracks = find_soundtrack(m.title)
        except Exception as e:
            print(f"  search failed for {m.title}: {e}; pausing")
            time.sleep(30)
            continue
        for t in tracks:
            rows.append({**t, "wikidata_id": m.wikidata_id, "film_title": m.title, "searched_film": m.wikidata_id})
        if not tracks:  # remember we looked, so a resume doesn't search again
            rows.append({"searched_film": m.wikidata_id, "wikidata_id": None})
        print(f"  [{k}/{len(todo)}] {m.title} ({m.year}): {len(tracks)} soundtrack songs")
        if k % 20 == 0:
            pd.DataFrame(rows).to_parquet(PAIRS_PATH, index=False)
        time.sleep(3.2)  # iTunes search allows ~20 requests a minute
    pd.DataFrame(rows).to_parquet(PAIRS_PATH, index=False)


def _features(track):
    return audio_features.get_features_for_track(track)


def extract(workers=6):
    """phase 2: audio features for every pair that doesn't have them yet,
    in parallel -- each preview is an independent download + analysis."""
    from multiprocessing import Pool

    df = pd.read_parquet(PAIRS_PATH)
    if "tempo" not in df:
        df["tempo"] = None
    need = df[df["wikidata_id"].notna() & df["tempo"].isna()]
    print(f"Extracting audio features for {len(need)} soundtrack songs with {workers} workers...")
    # track_id comes back as a float here (the "nothing found" rows are NaN),
    # and it names the preview file on disk -- keep it an int
    tracks = [{"track_id": int(r["track_id"]), "track_name": r["track_name"], "preview_url": r["preview_url"]}
              for r in need.to_dict("records")]
    with Pool(workers) as pool:
        results = pool.map(_features, tracks, chunksize=4)
    feats = {r["track_id"]: r for r in results if r}
    for col in ("tempo", "energy", "brightness", "zcr", "chroma_mean"):
        if col not in df:
            df[col] = None
        df[col] = df[col].astype(object)
    for i, r in need.iterrows():
        f = feats.get(r["track_id"])
        if f:
            for col in ("tempo", "energy", "brightness", "zcr", "chroma_mean"):
                df.at[i, col] = f[col]
    df.to_parquet(PAIRS_PATH, index=False)
    done = df[df["wikidata_id"].notna() & df["tempo"].notna()]
    print(f"Done: {len(done)} soundtrack songs with features, for {done['wikidata_id'].nunique()} films.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--films", type=int, default=450, help="How many catalog films to search")
    parser.add_argument("--features", action="store_true", help="Run the parallel audio feature step")
    args = parser.parse_args()
    if args.features:
        extract()
    else:
        collect(args.films)
