"""
Wikipedia intros for songs, where the song has an article.

Audio can't tell that "Heartless" is a breakup song -- it hears an upbeat
808 beat. The song's Wikipedia intro says what it's about, and it can be
read for mood with exactly the same text pipeline as the films, which puts
it straight into the film-mood space.

    python src/song_wiki.py      # resumable; writes data/processed/song_wiki.parquet
"""

import os
import re
import time

import pandas as pd

from wikidata_client import WIKIPEDIA_API_URL, _get

WIKI_PATH = "data/processed/song_wiki.parquet"


def _norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", str(s).lower()).split()


def _clean_title(title):
    # "Blinding Lights (Remix)", "honeybee (feat. X)" -> the bare song title
    return re.sub(r"\s*[\(\[].*?[\)\]]", "", str(title)).strip()


def find_intro(title, artist):
    """Intro paragraph of the song's article, or "" if none is clearly it."""
    bare = _clean_title(title)
    for query in (f'"{bare}" {artist} song', f"{bare} {artist}"):
        page, intro = _search(query, bare, artist)
        if page:
            return page, intro
    return "", ""


def _search(query, bare, artist):
    resp = _get(WIKIPEDIA_API_URL, params={
        "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
        "generator": "search", "gsrsearch": query, "gsrlimit": 5,
        "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": 5,
    })
    if resp.status_code != 200:
        return "", ""
    pages = sorted(resp.json().get("query", {}).get("pages", []), key=lambda p: p.get("index", 99))
    artist_words = [w for w in _norm(re.split(r",|&| feat\.? | x ", artist)[0]) if len(w) > 2]
    title_words = _norm(bare)
    for p in pages:
        extract = p.get("extract", "").strip()
        opening = extract[:400].lower()
        page_title = _norm(p.get("title", ""))
        # it has to be about this song: title in the page name, the artist and
        # "song" / "single" in the opening lines
        if (title_words and all(w in page_title for w in title_words)
                and any(w in opening for w in artist_words)
                and re.search(r"\b(song|single|track)\b", opening)):
            return p["title"], extract.split("\n")[0]
    return "", ""


def _lookup(r):
    try:
        page, intro = find_intro(r.track_name, r.artist_name)
    except Exception as e:
        return None, f"{r.track_name}: {e}"
    return {"track_id": int(r.track_id), "wiki_title": page, "intro": intro}, None


def fetch(songs, workers=4):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    existing = pd.read_parquet(WIKI_PATH) if os.path.exists(WIKI_PATH) else pd.DataFrame(columns=["track_id"])
    done = set(existing["track_id"].astype(int))
    rows = existing.to_dict("records")
    todo = songs[~songs["track_id"].astype(int).isin(done)]
    print(f"{len(done)} songs already looked up, {len(todo)} to go.", flush=True)
    # a few lookups in parallel -- modest enough for Wikipedia's API etiquette
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_lookup, r) for r in todo.itertuples()]
        for k, f in enumerate(as_completed(futures), 1):
            row, err = f.result()
            if err:
                print(f"  lookup failed for {err}", flush=True)
                continue
            rows.append(row)
            if k % 100 == 0:
                found = sum(1 for x in rows if x.get("wiki_title"))
                print(f"  {k}/{len(todo)} looked up, {found} intros found so far", flush=True)
                pd.DataFrame(rows).to_parquet(WIKI_PATH, index=False)
    df = pd.DataFrame(rows)
    df.to_parquet(WIKI_PATH, index=False)
    found = (df["intro"].str.len() > 40).sum()
    print(f"Done: {found} of {len(df)} songs have a Wikipedia intro.")


if __name__ == "__main__":
    songs = pd.read_parquet("data/processed/songs.parquet")
    pairs = pd.read_parquet("data/processed/soundtrack_pairs.parquet")
    pairs = pairs[pairs["wikidata_id"].notna()][["track_id", "track_name", "artist_name"]]
    fetch(pd.concat([songs[["track_id", "track_name", "artist_name"]], pairs]).drop_duplicates("track_id"))
