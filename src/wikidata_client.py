"""
Wikidata client -- pulls movie metadata via public SPARQL queries.
Replaces TMDB, whose terms explicitly prohibit ML/AI use. Wikidata is
CC0-licensed, so no such restriction applies. No API key needed.
"""

import json
import requests
import time
from urllib.parse import quote, unquote

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# Wikimedia asks for a descriptive User-Agent with a contact point, or
# requests get throttled much more aggressively
HEADERS = {"User-Agent": "vibe-embeddings/0.1 (personal ML portfolio project; "
                         "https://github.com/Charan7sai/vibe-embeddings)"}


def _get(url, params=None, timeout=30, retries=5):
    """GET with backoff on 429 / 5xx. Wikipedia rate-limits bursts, and
    treating a 429 as "no article" silently drops good films."""
    for attempt in range(retries):
        resp = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = float(resp.headers.get("Retry-After", 2 ** attempt * 5))
            print(f"  ({resp.status_code} from {url.split('/')[2]}, waiting {wait:.0f}s)")
            time.sleep(wait)
            continue
        return resp
    return resp


def run_sparql(query, timeout=60):
    resp = _get(SPARQL_ENDPOINT, params={"query": query, "format": "json"}, timeout=timeout)
    resp.raise_for_status()
    # strict=False: a few film labels contain raw control characters, which
    # the default parser rejects -- failing the whole page every time
    return json.loads(resp.text, strict=False)["results"]["bindings"]


def get_popular_film_ids(limit=4000, min_sitelinks=20):
    """Film QIDs ranked by number of Wikipedia language editions -- a
    decent popularity proxy. Paging through *all* films with OFFSET got
    slower the deeper it went until Wikidata timed out, and returned an
    arbitrary slice of mostly obscure films. This one query takes seconds."""
    query = f"""
    SELECT ?film ?links WHERE {{
      ?film wdt:P31 wd:Q11424; wikibase:sitelinks ?links.
      FILTER(?links >= {min_sitelinks})
    }} ORDER BY DESC(?links) LIMIT {limit}
    """
    return [row["film"]["value"].rsplit("/", 1)[-1] for row in run_sparql(query, timeout=90)]


def get_films_with_years(min_sitelinks=8, english_only=True):
    """Every film with at least min_sitelinks Wikipedia editions, with its
    first release year: [(qid, sitelinks, year)]. ~20k English-language
    films (P364 = Q1860) in one query, ~35k without the language filter.

    Ranking *all* films by sitelinks favours classics -- a 1972 film has had
    50 years to get translated -- so the catalog builder ranks within each
    release year instead."""
    language = "?film wdt:P364 wd:Q1860." if english_only else ""
    query = f"""
    SELECT ?film ?links (MIN(?d) AS ?first) WHERE {{
      ?film wdt:P31 wd:Q11424; wikibase:sitelinks ?links.
      FILTER(?links >= {min_sitelinks})
      {language}
      ?film wdt:P577 ?d.
    }} GROUP BY ?film ?links
    """
    out = []
    for row in run_sparql(query, timeout=120):
        year = row["first"]["value"][:4]
        if year.isdigit():
            out.append((row["film"]["value"].rsplit("/", 1)[-1], int(row["links"]["value"]), int(year)))
    return out


def get_film_details(qids):
    """Title, first release year, genres and English Wikipedia article for
    a batch of film QIDs (keep batches ~50). Aggregated server-side because
    popular films have dozens of country-specific release dates, and the
    genre x date cross product made responses huge enough to get truncated.

    The ?article sitelink gives the film's exact English Wikipedia page --
    looking pages up by bare title instead landed on the wrong article
    surprisingly often ("Brazil" -> the country, "Godzilla" -> the monster)."""
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
    SELECT ?film ?title ?article (MIN(?date) AS ?first)
           (GROUP_CONCAT(DISTINCT ?genreName; separator="|") AS ?genres) WHERE {{
      VALUES ?film {{ {values} }}
      ?film rdfs:label ?title. FILTER(LANG(?title) = "en")
      ?film wdt:P577 ?date; wdt:P136 ?genre.
      ?genre rdfs:label ?genreName. FILTER(LANG(?genreName) = "en")
      ?article schema:about ?film; schema:isPartOf <https://en.wikipedia.org/>.
    }} GROUP BY ?film ?title ?article
    """
    movies = {}
    for row in run_sparql(query):
        film_id = row["film"]["value"]
        movies[film_id] = {
            "wikidata_id": film_id,
            "title": row["title"]["value"],
            "year": row["first"]["value"][:4],
            "wiki_title": unquote(row["article"]["value"].rsplit("/wiki/", 1)[-1]),
            # a few genre items are mislabelled with a Wikipedia category name
            # ("Category:Animated films based on manhua"); those aren't genres
            "genres": [g for g in row["genres"]["value"].split("|") if not g.startswith("Category:")],
        }
    # keep popularity order
    return [movies[f"http://www.wikidata.org/entity/{q}"] for q in qids
            if f"http://www.wikidata.org/entity/{q}" in movies]


def get_wikipedia_summary(title):
    """Fetch the intro summary of a film's Wikipedia article -- our stand-in
    for plot text, since Wikidata itself doesn't store it. Returns "" if no
    matching article is found, or if the title resolved to a disambiguation
    page instead of the actual film (common for short/generic titles)."""
    url = WIKIPEDIA_SUMMARY_URL.format(quote(title.replace(" ", "_")))
    resp = _get(url, timeout=15)
    if resp.status_code != 200:
        return ""
    data = resp.json()
    if data.get("type") == "disambiguation":
        return ""
    return data.get("extract", "")


def get_wikipedia_summaries(titles):
    """Batch version for catalog builds: intro paragraph and lead image
    (the poster, for films) for up to 20 article titles per request via the
    MediaWiki query API, instead of one REST call per film.

    Returns {title: {"overview": str, "poster_url": str}}, "" where missing.
    pilicense=any matters: posters are non-free, and the page-image API
    silently skips non-free images by default."""
    out = {}
    for i in range(0, len(titles), 20):
        chunk = titles[i:i + 20]
        resp = _get(WIKIPEDIA_API_URL, params={
            "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
            "prop": "extracts|pageprops|pageimages", "exintro": 1, "explaintext": 1, "exlimit": 20,
            "ppprop": "disambiguation", "piprop": "thumbnail", "pithumbsize": 500,
            "pilicense": "any", "pilimit": 20,
            "titles": "|".join(t.replace("_", " ") for t in chunk),
        })
        resp.raise_for_status()
        query = resp.json().get("query", {})
        # follow the API's title normalisation + redirects back to our input titles
        alias = {}
        for step in query.get("normalized", []) + query.get("redirects", []):
            alias[step["from"]] = step["to"]
        pages = {p["title"]: p for p in query.get("pages", [])}
        for t in chunk:
            name = t.replace("_", " ")
            while name in alias:
                name = alias[name]
            page = pages.get(name, {})
            if "disambiguation" in page.get("pageprops", {}):
                out[t] = {"overview": "", "poster_url": ""}
                continue
            # first paragraph only, to match the REST summary's length
            paragraphs = [p for p in page.get("extract", "").split("\n") if p.strip()]
            poster = page.get("thumbnail", {}).get("source", "").split("?")[0]  # drop utm_ params
            out[t] = {"overview": paragraphs[0].strip() if paragraphs else "", "poster_url": poster}
        time.sleep(0.5)
    return out


if __name__ == "__main__":
    movies = get_film_details(get_popular_film_ids(limit=5))
    print(f"Connected OK. Found {len(movies)} movies.")
    for m in movies:
        print(f"  {m['title']} ({m['year']}) -- genres: {', '.join(m['genres'])}")

    if movies:
        page = get_wikipedia_summaries([movies[0]["wiki_title"]])[movies[0]["wiki_title"]]
        print(f"\nSample summary for {movies[0]['title']}:")
        print(f"  {page['overview'][:200]}...")
        print(f"  poster: {page['poster_url']}")
