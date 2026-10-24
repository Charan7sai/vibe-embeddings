"""
Checkpoint 1: build the real seed catalogs.

Pulls a recent, English-language song catalog (Apple Music most-played
charts from English-speaking countries + iTunes genre charts, with
librosa audio features and album art) and a recency-weighted catalog of
English-language movies (Wikidata + Wikipedia summaries and posters),
saving both as parquet files under data/processed/.

Resumable: if interrupted, re-running skips anything already saved.
Run independently:
    python src/build_catalog.py --songs
    python src/build_catalog.py --movies
Or both:
    python src/build_catalog.py --songs --movies
"""

import argparse
import os
import re
import time
import unicodedata
import pandas as pd

import itunes_client
import audio_features
import wikidata_client

SONGS_PATH = "data/processed/songs.parquet"
MOVIES_PATH = "data/processed/movies.parquet"

SAVE_EVERY = 25  # checkpoint frequency -- don't lose too much on a crash


def _load_existing(path):
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


# Charts now carry a lot of non-music: white/brown noise, sleep and nature
# sounds, binaural beats. They'd pollute every "calm" or "dreamy" result.
NON_MUSIC_GENRES = {"meditation", "relaxation", "nature", "healing", "sound effects",
                    "children's music", "children’s music", "fitness & workout", "spoken word"}
NON_MUSIC_TITLE = re.compile(r"\b(white|pink|brown|green|fan|bird) noises?\b|\bfor (deep )?sleep(ing)?\b|"
                             r"\bsounds to (sleep|relax)\b|\bsleep ?therapy\b|\bpurring\b|"
                             r"\bsleep (sounds|music|therapy)\b|\bbinaural\b|\bsounds? effects?\b|"
                             r"\b(nature|ocean|rain|forest|wolf|bird) sounds?\b|\bbirds chirping\b|"
                             r"\bloopable\b|\bspa music\b|\bmeditation\b|\b\d+ ?hz\b", re.I)


def _is_music(track):
    text = f"{track.get('track_name') or ''} {track.get('artist_name') or ''}"
    genre = (track.get("genre") or "").lower()
    # "Sleep" on a New Age chart is a sleep aid; on a country chart it's a song
    sleep_aid = genre == "new age" and "sleep" in (track.get("track_name") or "").lower()
    return genre not in NON_MUSIC_GENRES and not NON_MUSIC_TITLE.search(text) and not sleep_aid


# The catalog is English-language. iTunes has no language field, so this
# combines three signals: English-speaking storefronts (see itunes_client),
# genre labels that mark non-English music, and the title itself.
NON_ENGLISH_GENRES = re.compile(
    r"\b(?:latin|latino|latina|mexican|urbano|reggaeton|salsa|tropical|flamenco|tango|bolero|"
    r"k-pop|j-pop|c-pop|cantopop|mandopop|korean|japanese|anime|enka|kayokyoku|"
    r"bollywood|tamil|telugu|punjabi|indian|sertanejo|brazil|brasileir|mpb|samba|pagode|forr|"
    r"baile funk|bossa nova|turkish|arab|khaleeji|levant|dangdut|indo pop|pilipino|french|"
    r"vari[eé]t[eé]|schlager|german|m[uú]sica\b|espa[nñ]ol|alternativa|rock y|banda|cl[aá]sica|"
    r"maskandi|gqom|amapiano|israeli|islamic|barnmusik)", re.I)
# Strong Spanish / Portuguese / French / German title markers. Deliberately
# leaves out words that also occur in English titles ("Die With A Smile",
# "La La Land", "Te Amo" is fine to lose, "Mi" / "Tu" are too ambiguous).
NON_ENGLISH_TITLE = re.compile(
    r"\b(amor|coraz[oó]n|quiero|baila|bailar|noche|vida|eres|nada|siempre|nunca|contigo|"
    r"mujer|beb[eé]|mami|chica|bonita|fiesta|loca|porque|tambi[eé]n|cu[aá]ndo|d[oó]nde|qu[eé]|"
    r"c[oó]mo|est[aá]s|tengo|n[aã]o|voc[eê]|meu|minha|saudade|ich|und|nicht|liebe|je|moi|toi|"
    r"oui|mon|ma vie|avec|pour|dans|sin ti|con|los|las|del|una|para)\b", re.I)
# accented letters common in Spanish / Portuguese / French / Italian / Turkish
# titles. Matched case-sensitively: with re.I, Turkish dotless/dotted i fold
# onto a plain "i" and flag every English title.
NON_ENGLISH_LETTERS = re.compile("[ñÑ¿¡ãõÃÕğĞışŞİçÇàèìòùÀÈÌÒÙâêôûÂÊÔÛéÉ]")


def _is_latin_script(text):
    return all(not ch.isalpha() or "LATIN" in unicodedata.name(ch, "") for ch in text)


def _is_english(track):
    # featured artists aren't the language of the song ("feat. Lana Del Rey")
    title = re.sub(r"[(\[](feat|ft|with)\.? [^)\]]*[)\]]", "", track.get("track_name") or "", flags=re.I)
    artist = track.get("artist_name") or ""
    genre = track.get("genre") or ""
    return (_is_latin_script(title + artist + genre)
            and not NON_ENGLISH_GENRES.search(genre)
            and not NON_ENGLISH_TITLE.search(title)
            and not NON_ENGLISH_LETTERS.search(title))


def _release_year(track):
    try:
        return int(str(track.get("release_date") or "")[:4])
    except ValueError:
        return None


def _song_sources(per_genre):
    """(label, fetch) pairs: current most-played charts from English-speaking
    countries first, then each genre chart from several storefronts for range."""
    for country in itunes_client.CHART_COUNTRIES:
        yield f"most played: {country}", lambda c=country: itunes_client.get_most_played(c)
    for genre_key in itunes_client.GENRE_IDS:
        for country in itunes_client.GENRE_CHART_COUNTRIES:
            yield (f"genre: {genre_key} ({country})",
                   lambda g=genre_key, c=country: itunes_client.get_top_songs_by_genre(g, per_genre, c))


def build_song_catalog(per_genre=150, min_year=2015):
    """min_year: genre charts for rock / jazz / R&B are mostly decades-old
    catalogue hits, so tracks released before this are skipped (and dropped
    from an existing catalog on resume)."""
    existing = _load_existing(SONGS_PATH)
    if not existing.empty:
        years = pd.to_datetime(existing["release_date"], errors="coerce").dt.year
        too_old = years < min_year
        if too_old.any():
            print(f"Dropped {int(too_old.sum())} saved songs released before {min_year}.")
        existing = existing[~too_old]
        is_music = existing.apply(lambda r: _is_music(r.to_dict()), axis=1)
        if (~is_music).any():
            print(f"Dropped {int((~is_music).sum())} saved non-music tracks (noise, sleep sounds...).")
        existing = existing[is_music]
        # language comes mostly from *where* a track charted, so drop rows
        # from charts that are no longer sources (non-English countries,
        # K-pop / J-pop / Latin / Brazilian / world genre charts)
        sources = {f"chart_{c}" for c in itunes_client.CHART_COUNTRIES} | set(itunes_client.GENRE_IDS)
        from_source = existing["seed_genre"].isin(sources)
        if (~from_source).any():
            print(f"Dropped {int((~from_source).sum())} saved songs from non-English charts.")
        existing = existing[from_source]
        is_english = existing.apply(lambda r: _is_english(r.to_dict()), axis=1)
        if (~is_english).any():
            print(f"Dropped {int((~is_english).sum())} saved non-English tracks.")
        existing = existing[is_english]
        # rows saved before artwork was collected
        if "artwork_url" not in existing or existing["artwork_url"].isna().any():
            missing = existing["track_id"] if "artwork_url" not in existing                 else existing.loc[existing["artwork_url"].isna(), "track_id"]
            print(f"Backfilling album art for {len(missing)} saved songs...")
            art = itunes_client.get_artwork(missing.tolist())
            existing = existing.copy()
            existing["artwork_url"] = existing.get("artwork_url", pd.Series(index=existing.index, dtype=object))
            existing["artwork_url"] = existing["artwork_url"].fillna(existing["track_id"].map(art))
            existing = existing[existing["artwork_url"].notna()]
    seen_ids = set(existing["track_id"]) if not existing.empty else set()
    print(f"Resuming with {len(seen_ids)} songs already saved." if seen_ids
          else "Starting fresh song catalog.")

    rows = existing.to_dict("records") if not existing.empty else []
    since_last_save = 0

    for label, fetch in _song_sources(per_genre):
        print(f"\n=== {label} ===")
        try:
            tracks = fetch()
        except Exception as e:
            print(f"  Failed to pull {label}: {e}")
            continue

        for i, track in enumerate(tracks):
            if track["track_id"] in seen_ids:
                continue  # already processed in a prior run
            year = _release_year(track)
            if year is None or year < min_year or not _is_music(track) or not _is_english(track):
                continue

            print(f"  [{i+1}/{len(tracks)}] {track['track_name']} - {track['artist_name']} ({year})")
            enriched = audio_features.get_features_for_track(track)
            if enriched is None:
                continue  # download/decode failed, skip and move on

            rows.append(enriched)
            seen_ids.add(track["track_id"])
            since_last_save += 1

            if since_last_save >= SAVE_EVERY:
                pd.DataFrame(rows).to_parquet(SONGS_PATH, index=False)
                print(f"  -- checkpoint saved ({len(rows)} songs total) --")
                since_last_save = 0

    pd.DataFrame(rows).to_parquet(SONGS_PATH, index=False)
    print(f"\nDone. {len(rows)} songs saved to {SONGS_PATH}")


# English-language films per release year. Ranking all films by sitelinks favours classics
# (only 7% of the first catalog was from 2020+), so popularity is ranked
# *within* each year and recent years get the biggest share.
RECENT_FROM = 2015      # 2015 -> now: RECENT_PER_YEAR films each
RECENT_PER_YEAR = 150
MODERN_FROM = 2000      # 2000 -> 2014: MODERN_PER_YEAR films each
MODERN_PER_YEAR = 50
CLASSICS = 450          # everything older, as one pool


def _year_bucket(year):
    if year >= RECENT_FROM:
        return year, RECENT_PER_YEAR
    if year >= MODERN_FROM:
        return year, MODERN_PER_YEAR
    return "classic", CLASSICS


def build_movie_catalog(batch_size=50):
    existing = _load_existing(MOVIES_PATH)
    seen_ids = set(existing["wikidata_id"]) if not existing.empty else set()
    print(f"Resuming with {len(seen_ids)} movies already saved." if seen_ids
          else "Starting fresh movie catalog.")

    rows = existing.to_dict("records") if not existing.empty else []
    filled = {}
    for m in rows:
        key, _ = _year_bucket(int(m["year"]))
        filled[key] = filled.get(key, 0) + 1
    since_last_save = 0

    # top films per bucket by sitelinks, over-fetched: some lack genres,
    # an English article or a usable intro
    by_bucket = {}
    films = wikidata_client.get_films_with_years()
    for qid, links, year in films:
        if year > time.localtime().tm_year:
            continue  # announced, not released
        key, quota = _year_bucket(year)
        by_bucket.setdefault(key, (quota, []))[1].append((links, qid))
    quotas = {key: quota for key, (quota, _) in by_bucket.items()}
    todo = []
    for key, (quota, cands) in by_bucket.items():
        cands.sort(reverse=True)
        todo += [(key, qid) for _, qid in cands[:int(quota * 1.6)]
                 if f"http://www.wikidata.org/entity/{qid}" not in seen_ids]
    print(f"{len(films)} films found; {len(todo)} candidates across {len(quotas)} year buckets.")

    for start in range(0, len(todo), batch_size):
        chunk = [(k, q) for k, q in todo[start:start + batch_size] if filled.get(k, 0) < quotas[k]]
        if not chunk:
            continue
        bucket_of = {f"http://www.wikidata.org/entity/{q}": k for k, q in chunk}
        try:
            batch = wikidata_client.get_film_details([q for _, q in chunk])
            summaries = wikidata_client.get_wikipedia_summaries([m["wiki_title"] for m in batch])
        except Exception as e:
            print(f"  Batch at {start} failed ({e}), skipping it.")
            continue

        for movie in batch:
            key = bucket_of[movie["wikidata_id"]]
            page = summaries.get(movie["wiki_title"], {})
            if len(page.get("overview", "")) < 40 or not page.get("poster_url"):
                continue  # too thin to describe a vibe from, or nothing to show
            if filled.get(key, 0) >= quotas[key]:
                continue  # bucket full

            movie["overview"] = page["overview"]
            movie["poster_url"] = page["poster_url"]
            rows.append(movie)
            seen_ids.add(movie["wikidata_id"])
            filled[key] = filled.get(key, 0) + 1
            since_last_save += 1
            print(f"  [{len(rows)}] {movie['title']} ({movie['year']})")

            if since_last_save >= SAVE_EVERY:
                pd.DataFrame(rows).to_parquet(MOVIES_PATH, index=False)
                print(f"  -- checkpoint saved ({len(rows)} movies total) --")
                since_last_save = 0

        time.sleep(0.5)  # be polite to Wikidata

    pd.DataFrame(rows).to_parquet(MOVIES_PATH, index=False)
    print(f"\nDone. {len(rows)} movies saved to {MOVIES_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--songs", action="store_true", help="Build the song catalog")
    parser.add_argument("--movies", action="store_true", help="Build the movie catalog")
    parser.add_argument("--per-genre", type=int, default=150, help="Songs per genre")
    parser.add_argument("--min-song-year", type=int, default=2015, help="Skip songs released earlier")
    args = parser.parse_args()

    os.makedirs("data/processed", exist_ok=True)

    if not args.songs and not args.movies:
        print("Specify --songs, --movies, or both.")
    if args.songs:
        build_song_catalog(per_genre=args.per_genre, min_year=args.min_song_year)
    if args.movies:
        build_movie_catalog()
