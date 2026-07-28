"""
Wikidata client -- pulls movie metadata via public SPARQL queries.
Replaces TMDB, whose terms explicitly prohibit ML/AI use. Wikidata is
CC0-licensed, so no such restriction applies. No API key needed.
"""

import requests
import time
from urllib.parse import quote

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

# Wikidata asks for a descriptive User-Agent or requests can get throttled
HEADERS = {"User-Agent": "vibe-embeddings/0.1 (personal ML portfolio project)"}


def run_sparql(query, timeout=60):
    resp = requests.get(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        headers=HEADERS,
        timeout=timeout
    )
    resp.raise_for_status()
    return resp.json()["results"]["bindings"]


def _build_movies_query(limit, offset):
    # Q11424 = "film". No ORDER BY on purpose -- sorting forces Wikidata
    # to gather the entire matching set before returning anything, which
    # timed out during testing. Unsorted queries stream results instead,
    # so results here aren't ranked by popularity.
    return f"""
    SELECT ?film ?filmLabel ?genreLabel ?date WHERE {{
      ?film wdt:P31 wd:Q11424;
            wdt:P136 ?genre;
            wdt:P577 ?date.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}
    OFFSET {offset}
    """


def get_movies_batch(limit=200, offset=0):
    """Pull one page of movies, grouping multiple genre rows per film
    into a single record."""
    rows = run_sparql(_build_movies_query(limit, offset))

    movies = {}
    for row in rows:
        film_id = row["film"]["value"]
        title = row["filmLabel"]["value"]
        genre = row["genreLabel"]["value"]
        year = row["date"]["value"][:4]

        if film_id not in movies:
            movies[film_id] = {"wikidata_id": film_id, "title": title, "year": year, "genres": []}
        if genre not in movies[film_id]["genres"]:
            movies[film_id]["genres"].append(genre)

    return list(movies.values())


def build_seed_catalog(target_count=3000, batch_size=200):
    """Page through results until target_count unique movies are collected."""
    all_movies = {}
    offset = 0
    while len(all_movies) < target_count:
        print(f"Fetching movies, offset={offset}...")
        batch = get_movies_batch(limit=batch_size, offset=offset)
        if not batch:
            break
        for m in batch:
            all_movies[m["wikidata_id"]] = m
        offset += batch_size
        time.sleep(0.5)
    return list(all_movies.values())


def get_wikipedia_summary(title):
    """Fetch the intro summary of a film's Wikipedia article -- our stand-in
    for plot text, since Wikidata itself doesn't store it. Returns "" if no
    matching article is found (Wikidata and Wikipedia titles occasionally differ)."""
    url = WIKIPEDIA_SUMMARY_URL.format(quote(title.replace(" ", "_")))
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        return ""
    return resp.json().get("extract", "")


if __name__ == "__main__":
    movies = get_movies_batch(limit=5, offset=0)
    print(f"Connected OK. Found {len(movies)} movies.")
    for m in movies:
        print(f"  {m['title']} ({m['year']}) -- genres: {', '.join(m['genres'])}")

    if movies:
        summary = get_wikipedia_summary(movies[0]["title"])
        print(f"\nSample summary for {movies[0]['title']}:")
        print(f"  {summary[:200]}...")