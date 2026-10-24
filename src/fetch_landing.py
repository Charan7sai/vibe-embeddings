"""
Builds web/data/landing.json: the iconic characters and singers shown on
the landing page, with image URLs and the credit / license line each one
needs.

Images are the lead images of the Wikipedia articles below. Singer photos
are freely licensed (CC BY / CC BY-SA / public domain); most character
images are non-free film stills that Wikipedia uses under fair use, so they
are linked from Wikimedia rather than copied into this repo, and every
image is credited on the page.

Run:
    python src/fetch_landing.py
"""

import json
import os
import re
import time

import requests

from wikidata_client import HEADERS, WIKIPEDIA_API_URL

OUT_PATH = "web/data/landing.json"

# (Wikipedia article, label shown on the page, note, what clicking starts from)
CHARACTERS = [
    ("Peter Parker (Marvel Cinematic Universe)", "Spider-Man", "Tom Holland", "Spider-Man: No Way Home"),
    ("Joker (The Dark Knight)", "The Joker", "Heath Ledger", "The Dark Knight"),
    ("Darth Vader", "Darth Vader", "Star Wars", "The Empire Strikes Back"),
    ("Tony Stark (Marvel Cinematic Universe)", "Tony Stark", "Robert Downey Jr.", "Iron Man"),
    ("Jack Sparrow", "Jack Sparrow", "Johnny Depp", "Pirates of the Caribbean: The Curse of the Black Pearl"),
    ("Neo (The Matrix)", "Neo", "Keanu Reeves", "The Matrix"),
    ("John Wick (character)", "John Wick", "Keanu Reeves", "John Wick"),
    ("Indiana Jones (character)", "Indiana Jones", "Harrison Ford", "Raiders of the Lost Ark"),
]
SINGERS = [
    ("Kanye West", "Kanye West", "Hip-hop", "Heartless Kanye West"),
    ("The Weeknd", "The Weeknd", "R&B", "Blinding Lights The Weeknd"),
    ("Billie Eilish", "Billie Eilish", "Alt pop", "Birds of a Feather Billie Eilish"),
    ("Freddie Mercury", "Freddie Mercury", "Queen", "Bohemian Rhapsody Queen"),
    ("Beyoncé", "Beyoncé", "R&B", "Halo Beyonce"),
    ("Kendrick Lamar", "Kendrick Lamar", "Hip-hop", "Not Like Us Kendrick Lamar"),
    ("Michael Jackson", "Michael Jackson", "Pop", "Billie Jean Michael Jackson"),
    ("Sabrina Carpenter", "Sabrina Carpenter", "Pop", "Espresso Sabrina Carpenter"),
]


def _strip_html(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def _file_info(name):
    """License, author and description page for a file, which lives on
    Commons when free and on English Wikipedia when it's fair use."""
    for host in ("commons.wikimedia.org", "en.wikipedia.org"):
        resp = requests.get(f"https://{host}/w/api.php", headers=HEADERS, timeout=30, params={
            "action": "query", "format": "json", "formatversion": 2,
            "titles": f"File:{name}", "prop": "imageinfo", "iiprop": "extmetadata|url",
        })
        resp.raise_for_status()
        page = resp.json()["query"]["pages"][0]
        if "imageinfo" in page:
            info = page["imageinfo"][0]
            md = info["extmetadata"]
            return {
                "license": _strip_html(md.get("LicenseShortName", {}).get("value", "")),
                "author": _strip_html(md.get("Artist", {}).get("value", ""))[:120],
                "source_page": info["descriptionurl"],
                "free": host == "commons.wikimedia.org",
            }
    return None


def fetch(entries, seed_type):
    titles = [t for t, _, _, _ in entries]
    resp = requests.get(WIKIPEDIA_API_URL, headers=HEADERS, timeout=30, params={
        "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
        "prop": "pageimages", "piprop": "thumbnail|name", "pithumbsize": 800,
        "pilicense": "any", "titles": "|".join(titles),
    })
    resp.raise_for_status()
    query = resp.json()["query"]
    alias = {s["from"]: s["to"] for s in query.get("normalized", []) + query.get("redirects", [])}
    pages = {p["title"]: p for p in query["pages"]}

    out = []
    for title, label, note, seed in entries:
        name = title
        while name in alias:
            name = alias[name]
        page = pages.get(name, {})
        if "thumbnail" not in page:
            print(f"  no image for {title}, skipping")
            continue
        info = _file_info(page["pageimage"])
        if info is None:
            print(f"  no file info for {title}, skipping")
            continue
        out.append({"name": label, "note": note, "image": page["thumbnail"]["source"].split("?")[0],
                    "article": f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}",
                    "seed": {"type": seed_type, "query": seed}, **info})
        print(f"  {label}: {info['license']}")
        time.sleep(0.3)
    return out


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    print("Characters:")
    characters = fetch(CHARACTERS, "movie")
    print("Singers:")
    singers = fetch(SINGERS, "song")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"characters": characters, "singers": singers}, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(characters)} characters and {len(singers)} singers to {OUT_PATH}")
