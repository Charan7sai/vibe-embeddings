"""
iTunes client -- pulls song metadata and preview URLs.
No account or API key needed; replaces Spotify after it deprecated
audio-features access for new apps.
"""

import requests
import time

SEARCH_URL = "https://itunes.apple.com/search"
RSS_URL_TEMPLATE = "https://itunes.apple.com/us/rss/topsongs/limit={limit}/genre={genre_id}/json"

# iTunes genre IDs, used to pull a genre-balanced catalog via the top-charts RSS feed
GENRE_IDS = {
    "pop": 14,
    "hip_hop_rap": 18,
    "alternative": 20,
    "rock": 21,
    "country": 6,
    "electronic": 7,
    "rnb_soul": 15,
    "jazz": 11,
    "classical": 5,
    "latin": 12,
    "singer_songwriter": 10,
    "metal": 1153,
    "indie": 1146,
}


def search_tracks(query, limit=25):
    """Free-text song search -- used when a user looks up a specific track."""
    params = {"term": query, "media": "music", "entity": "song", "limit": limit}
    resp = requests.get(SEARCH_URL, params=params)
    resp.raise_for_status()
    results = resp.json()["results"]
    return [_clean_track(r) for r in results if r.get("previewUrl")]


def get_top_songs_by_genre(genre_key, limit=100):
    """Pull a genre's current top chart. The RSS feed only gives track IDs,
    so we look those up separately to get full metadata + preview URLs."""
    genre_id = GENRE_IDS[genre_key]
    url = RSS_URL_TEMPLATE.format(limit=limit, genre_id=genre_id)
    resp = requests.get(url)
    resp.raise_for_status()
    entries = resp.json()["feed"]["entry"]
    track_ids = [e["id"]["attributes"]["im:id"] for e in entries]
    return _lookup_tracks(track_ids, genre_key)


def _lookup_tracks(track_ids, genre_key=None):
    """Batch lookup -- iTunes caps this at 200 IDs per request."""
    tracks = []
    for i in range(0, len(track_ids), 200):
        batch = track_ids[i:i + 200]
        resp = requests.get("https://itunes.apple.com/lookup", params={"id": ",".join(batch)})
        resp.raise_for_status()
        for r in resp.json()["results"]:
            if r.get("previewUrl"):
                cleaned = _clean_track(r)
                if genre_key:
                    cleaned["seed_genre"] = genre_key
                tracks.append(cleaned)
        time.sleep(0.2)
    return tracks


def build_seed_catalog(per_genre=50):
    """Pull a genre-balanced starter catalog across all known genres."""
    all_tracks = []
    for genre_key in GENRE_IDS:
        print(f"Pulling top {per_genre} for genre: {genre_key}")
        all_tracks.extend(get_top_songs_by_genre(genre_key, limit=per_genre))
        time.sleep(0.3)
    return all_tracks


def _clean_track(raw):
    return {
        "track_id": raw.get("trackId"),
        "track_name": raw.get("trackName"),
        "artist_name": raw.get("artistName"),
        "genre": raw.get("primaryGenreName"),
        "preview_url": raw.get("previewUrl"),
        "release_date": raw.get("releaseDate"),
    }


if __name__ == "__main__":
    results = search_tracks("Blinding Lights", limit=1)
    if results:
        t = results[0]
        print(f"Connected OK. Found: {t['track_name']} by {t['artist_name']}")
        print(f"Preview URL: {t['preview_url']}")
    else:
        print("No results -- check your internet connection.")