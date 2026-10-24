"""
iTunes client -- pulls song metadata and preview URLs.
No account or API key needed; replaces Spotify after it deprecated
audio-features access for new apps.
"""

import requests
import time

SEARCH_URL = "https://itunes.apple.com/search"
RSS_URL_TEMPLATE = "https://itunes.apple.com/{country}/rss/topsongs/limit={limit}/genre={genre_id}/json"
# Apple Music's current most-played chart per storefront -- what people are
# actually listening to right now, unlike the genre charts above, which for
# rock / jazz / R&B are dominated by decades-old catalogue hits
MOST_PLAYED_URL = "https://rss.applemarketingtools.com/api/v2/{country}/music/most-played/100/songs.json"
# English-speaking storefronts only: the catalog is English-language
CHART_COUNTRIES = ["us", "gb", "ca", "au", "ie", "nz"]
# genre charts are pulled from several storefronts too -- each caps at ~100
# tracks, and the UK / Canadian / Australian charts differ enough to add range
GENRE_CHART_COUNTRIES = ["us", "gb", "ca", "au"]

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
    "singer_songwriter": 10,
    "metal": 1153,
    "indie": 1146,
    # second wave: each chart caps at ~100 tracks, so more genres is the
    # only way to grow the catalog -- and it widens the vibe range too
    "blues": 2,
    "soundtrack": 16,
    "dance": 17,
    "reggae": 24,
}


def search_tracks(query, limit=25):
    """Free-text song search -- used when a user looks up a specific track."""
    params = {"term": query, "media": "music", "entity": "song", "limit": limit}
    resp = requests.get(SEARCH_URL, params=params)
    resp.raise_for_status()
    results = resp.json()["results"]
    return [_clean_track(r) for r in results if r.get("previewUrl")]


def get_top_songs_by_genre(genre_key, limit=100, country="us"):
    """Pull a genre's current top chart. The RSS feed only gives track IDs,
    so we look those up separately to get full metadata + preview URLs."""
    genre_id = GENRE_IDS[genre_key]
    url = RSS_URL_TEMPLATE.format(country=country, limit=limit, genre_id=genre_id)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    entries = resp.json()["feed"].get("entry", [])
    track_ids = [e["id"]["attributes"]["im:id"] for e in entries]
    return _lookup_tracks(track_ids, genre_key, country=country)


def get_most_played(country="us", limit=100):
    """Current most-played songs in one Apple Music storefront. The Apple
    Music feed is slow and sometimes times out, so retry, then fall back to
    the older iTunes top-songs feed (which doesn't cover every country)."""
    track_ids = []
    for attempt in range(3):
        try:
            resp = requests.get(MOST_PLAYED_URL.format(country=country), timeout=45)
            resp.raise_for_status()
            track_ids = [r["id"] for r in resp.json()["feed"]["results"][:limit]]
            break
        except (requests.RequestException, ValueError, KeyError):
            time.sleep(3 * (attempt + 1))
    if not track_ids:
        resp = requests.get(f"https://itunes.apple.com/{country}/rss/topsongs/limit={limit}/json", timeout=30)
        resp.raise_for_status()
        track_ids = [e["id"]["attributes"]["im:id"] for e in resp.json()["feed"]["entry"]]
    # look up in the same storefront -- local hits often aren't in the US store
    return _lookup_tracks(track_ids, f"chart_{country}", country=country)


def _lookup_tracks(track_ids, genre_key=None, country="us"):
    """Batch lookup -- iTunes caps this at 200 IDs per request."""
    tracks = []
    for i in range(0, len(track_ids), 200):
        batch = track_ids[i:i + 200]
        resp = requests.get("https://itunes.apple.com/lookup",
                            params={"id": ",".join(batch), "country": country})
        resp.raise_for_status()
        for r in resp.json()["results"]:
            if r.get("previewUrl"):
                cleaned = _clean_track(r)
                if genre_key:
                    cleaned["seed_genre"] = genre_key
                tracks.append(cleaned)
        time.sleep(0.2)
    return tracks


def get_artwork(track_ids):
    """{track_id: 600px album art URL}, for backfilling older catalog rows."""
    art = {}
    for i in range(0, len(track_ids), 200):
        batch = [str(t) for t in track_ids[i:i + 200]]
        resp = requests.get("https://itunes.apple.com/lookup", params={"id": ",".join(batch)}, timeout=30)
        resp.raise_for_status()
        for r in resp.json()["results"]:
            if r.get("trackId") and r.get("artworkUrl100"):
                art[r["trackId"]] = _artwork(r["artworkUrl100"])
        time.sleep(0.2)
    return art


def _artwork(url100):
    # Apple serves any size from the same path; 600px is crisp on a big card
    return (url100 or "").replace("100x100bb", "600x600bb")


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
        "artwork_url": _artwork(raw.get("artworkUrl100")),
    }


if __name__ == "__main__":
    results = search_tracks("Blinding Lights", limit=1)
    if results:
        t = results[0]
        print(f"Connected OK. Found: {t['track_name']} by {t['artist_name']}")
        print(f"Preview URL: {t['preview_url']}")
    else:
        print("No results -- check your internet connection.")