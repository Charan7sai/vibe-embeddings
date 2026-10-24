"""
Checkpoint 5: the unified recommendation engine.

Any of song / movie / genre / free text in, any of song / movie / genre out.

    from recommender import VibeRecommender
    rec = VibeRecommender()
    rec.recommend({"type": "song", "query": "Blinding Lights"}, target="movie")
    rec.recommend({"type": "movie", "query": "Interstellar"}, target="song")
    rec.recommend({"type": "genre", "id": "film:horror film"}, target="song")
    rec.recommend({"type": "text", "query": "rainy late-night drive"}, target="song")

Cross-domain results come purely from the shared vibe space. Same-domain
results blend in a domain-specific signal (audio features for songs,
plot embeddings for movies) so they aren't just "same four mood words".

Songs and movies not in the catalog are handled live: songs via iTunes
search + librosa on the preview clip, movies via the Wikipedia summary.
"""

import difflib
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cross_space import CrossSpace, zscore_rows
from embedder import embed
from vibe_descriptions import load_stats, text_mood_scores, describe_texts, _describe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get("VIBE_MODELS_DIR", os.path.join(ROOT, "models"))

SONG_CLAP_WEIGHT = 0.6      # song -> song: 0.6 sound (CLAP) + 0.4 mood sentence
MOVIE_CONTENT_WEIGHT = 0.4  # movie -> movie: 0.6 vibe + 0.4 plot


@dataclass
class Query:
    """A resolved query item, from the catalog or built live."""
    domain: str                      # song | movie | genre | text
    label: str
    vibe: np.ndarray
    moods: list
    vibe_text: str
    item_id: object = None           # catalog id, excluded from its own results
    extra: dict = field(default_factory=dict)  # audio / content vectors
    card: dict = None                # how to display the query item itself


class VibeRecommender:
    def __init__(self, models_dir=MODELS_DIR):
        self.songs = pd.read_parquet(f"{models_dir}/songs.parquet")
        self.movies = pd.read_parquet(f"{models_dir}/movies.parquet")
        self.genres = pd.read_parquet(f"{models_dir}/genres.parquet")
        self.song_vibe = np.load(f"{models_dir}/song_vibe.npy")
        self.movie_vibe = np.load(f"{models_dir}/movie_vibe.npy")
        self.movie_content = np.load(f"{models_dir}/movie_content.npy")
        self.song_audio = np.load(f"{models_dir}/song_audio.npy")
        self.genre_vibe = np.load(f"{models_dir}/genre_vibe.npy")  # distinctive direction
        self.genre_mean = np.load(f"{models_dir}/genre_mean.npy")  # plain centroid
        self.stats = load_stats(f"{models_dir}/vibe_stats.json")
        self.cross = CrossSpace(models_dir, self.stats)
        self._song_pos = {tid: i for i, tid in enumerate(self.songs["track_id"])}
        self._movie_pos = {mid: i for i, mid in enumerate(self.movies["wikidata_id"])}
        self._genre_pos = {gid: i for i, gid in enumerate(self.genres["genre_id"])}
        self._resolved = {}

        # which catalog rows belong to each genre, so clicking a genre shows
        # that genre's actual songs / films rather than only vibe neighbours
        self._genre_members = {}
        for r in self.genres.itertuples():
            if r.domain == "music":
                mask = (self.songs["genre"] == r.name).to_numpy()
            else:
                mask = self.movies["genres"].apply(lambda gs, n=r.name: n in list(gs)).to_numpy()
            self._genre_members[r.genre_id] = np.flatnonzero(mask)
        self._genre_covers = {gid: self._covers(gid) for gid in self.genres["genre_id"]}

    def _covers(self, genre_id, n=4):
        """Images of the genre's most typical members, for its tile."""
        i = self._genre_pos[genre_id]
        members = self._genre_members[genre_id]
        is_music = self.genres.iloc[i]["domain"] == "music"
        vecs, df, col = (self.song_vibe, self.songs, "artwork_url") if is_music \
            else (self.movie_vibe, self.movies, "poster_url")
        order = members[np.argsort(-(vecs[members] @ self.genre_mean[i]))]
        return [str(u) for u in df.iloc[order][col].dropna().drop_duplicates().head(n)]

    # ---------- catalog search ----------

    def search(self, text, domain, limit=10):
        """Case-insensitive title search over the catalog, for autocomplete."""
        df, col = self._frame(domain)
        text = text.strip().lower()
        if not text:
            return []
        titles = df[col].str.lower()
        hits = df[titles.str.contains(text, regex=False)]
        if len(hits) < limit:
            close = difflib.get_close_matches(text, titles.tolist(), n=limit, cutoff=0.6)
            hits = pd.concat([hits, df[titles.isin(close)]]).drop_duplicates(col)
        return [self._card(domain, i) for i in hits.index[:limit]]

    def _frame(self, domain):
        return {
            "song": (self.songs, "track_name"),
            "movie": (self.movies, "title"),
            "genre": (self.genres, "name"),
        }[domain]

    # ---------- query resolution ----------

    def resolve(self, q):
        """q: {"type": song|movie|genre|text, "id": ... or "query": ...}

        Cached, since a UI typically asks for songs, movies and genres for the
        same query back to back, and live lookups download + analyse audio."""
        key = (q["type"], q.get("id"), q.get("query"))
        if key not in self._resolved:
            if len(self._resolved) > 256:
                self._resolved.clear()
            self._resolved[key] = self._resolve(q)
        return self._resolved[key]

    def _resolve(self, q):
        kind = q["type"]
        if kind == "text":
            return self._from_text(q["query"])
        if kind == "genre":
            return self._from_genre(q.get("id") or q["query"])
        if q.get("id") is not None:
            return self._from_catalog(kind, q["id"])

        hits = self.search(q["query"], kind, limit=1)
        if hits and hits[0]["title"].lower() == q["query"].strip().lower():
            return self._from_catalog(kind, hits[0]["id"])
        live = self._live_song(q["query"]) if kind == "song" else self._live_movie(q["query"])
        if live is not None:
            return live
        # live lookup failed -- fall back to a catalog title containing the
        # query, but never to a merely similar-looking one (Interstellar != Intervista)
        if hits and q["query"].strip().lower() in hits[0]["title"].lower():
            return self._from_catalog(kind, hits[0]["id"])
        raise LookupError(f"Couldn't find a {kind} matching '{q['query']}'")

    def _from_catalog(self, kind, item_id):
        if kind == "song":
            i = self._song_pos[int(item_id)]
            r = self.songs.iloc[i]
            return Query("song", f"{r.track_name} - {r.artist_name}", self.song_vibe[i], list(r.moods),
                         r.vibe_text, int(r.track_id), {"cross": self.cross.song_repr(i)}, self._card("song", i))
        i = self._movie_pos[item_id]
        r = self.movies.iloc[i]
        return Query("movie", f"{r.title} ({r.year})", self.movie_vibe[i], list(r.moods), r.vibe_text,
                     r.wikidata_id, {"content": self.movie_content[i], "cross": self.cross.movie_repr(i)},
                     self._card("movie", i))

    def _from_genre(self, genre_id):
        if genre_id not in self._genre_pos:  # allow bare names, e.g. "horror film"
            matches = self.genres[self.genres["name"].str.lower() == str(genre_id).lower()]
            if matches.empty:
                raise LookupError(f"Unknown genre '{genre_id}'")
            genre_id = matches.iloc[0]["genre_id"]
        i = self._genre_pos[genre_id]
        r = self.genres.iloc[i]
        # search items with the plain centroid, other genres with the distinctive direction;
        # the other domain is matched from the members' average position in the matching space
        members = self._genre_members[r.genre_id]
        repr_of = self.cross.song_repr if r.domain == "music" else self.cross.movie_repr
        cross = self.cross.mean_repr([repr_of(int(m)) for m in members])
        return Query("genre", f"{r['name']} ({r.domain})", self.genre_mean[i], list(r.moods),
                     "", r.genre_id, {"distinct": self.genre_vibe[i], "cross": cross}, self._card("genre", i))

    def _from_text(self, text):
        scores = text_mood_scores([text], [[]], self.stats, standardise=False)
        moods, _, vibe_texts = _describe(scores, self.stats["text_energy_thresholds"])
        # typed moods -> songs is text-to-audio search, CLAP's home turf
        return Query("text", text, embed(vibe_texts)[0], moods[0], vibe_texts[0],
                     extra={"cross": self.cross.text_query(text, scores[0])})

    def _live_song(self, query):
        import itunes_client
        import audio_features
        import clap_audio

        try:
            results = itunes_client.search_tracks(query, limit=1)
        except Exception:
            return None
        if not results:
            return None
        track = audio_features.get_features_for_track(results[0])
        if track is None:
            return None
        if track["track_id"] in self._song_pos:
            return self._from_catalog("song", track["track_id"])
        import song_wiki

        wav = os.path.join(audio_features.TEMP_DIR, f"{track['track_id']}.wav")
        try:  # what the song is about, if it has a Wikipedia article
            _, intro = song_wiki.find_intro(track["track_name"], track["artist_name"])
        except Exception:
            intro = ""
        cross, scores = self.cross.live_song(clap_audio.embed_audio([wav])[0], track["track_name"],
                                             track.get("genre"), intro)
        moods, _, vibe_texts = _describe(scores[None], self.stats["song_energy_thresholds"])
        card = {"type": "song", "id": int(track["track_id"]), "title": track["track_name"],
                "subtitle": track["artist_name"], "genre": track.get("genre") or "", "moods": moods[0],
                "vibe_text": vibe_texts[0], "preview_url": track.get("preview_url") or "",
                "artwork_url": track.get("artwork_url") or "", "tempo": round(float(track["tempo"])),
                "year": str(track.get("release_date") or "")[:4], "live": True}
        return Query("song", f"{track['track_name']} - {track['artist_name']}", embed(vibe_texts)[0],
                     moods[0], vibe_texts[0], int(track["track_id"]), {"cross": cross}, card)

    def _live_movie(self, title):
        import wikidata_client

        page = {}
        # bare titles are often disambiguation pages ("Interstellar")
        for candidate in (f"{title} (film)", title):
            try:
                page = wikidata_client.get_wikipedia_summaries([candidate])[candidate]
            except Exception:
                continue
            # and bare titles can be the book / ship / country instead
            if len(page["overview"]) >= 40 and "film" in page["overview"][:250].lower():
                break
            page = {}
        if not page:
            return None
        import clap_audio
        from cross_space import film_caption

        scores = text_mood_scores([page["overview"]], [[]], self.stats)
        moods, _, vibe_texts = _describe(scores, self.stats["text_energy_thresholds"])
        cross = {"clap": clap_audio.embed_text([film_caption(moods[0], [])])[0],
                 "mood": self.cross.film_mood_from_scores(scores[0])}
        card = {"type": "movie", "id": None, "title": title, "subtitle": "", "genres": [],
                "moods": moods[0], "vibe_text": vibe_texts[0], "overview": page["overview"],
                "poster_url": page["poster_url"], "live": True}
        return Query("movie", title, embed(vibe_texts)[0], moods[0], vibe_texts[0],
                     extra={"content": embed([page["overview"]])[0], "cross": cross}, card=card)

    # ---------- recommendation ----------

    def recommend(self, q, target="movie", k=10):
        query = q if isinstance(q, Query) else self.resolve(q)
        genre_domain = None
        if query.domain == "genre":
            genre_domain = "song" if query.card["subtitle"] == "music" else "movie"

        candidates = None  # all rows, unless browsing a genre's own members
        cross_domain = target in ("song", "movie") and "cross" in query.extra and (
            query.domain == "text" and target == "song"
            or query.domain in ("song", "movie") and query.domain != target
            or query.domain == "genre" and target != genre_domain)

        if target == genre_domain:
            # browsing a genre in its own domain: its members, most typical first
            candidates = self._genre_members[query.item_id]
            vecs = self.song_vibe if target == "song" else self.movie_vibe
            scores = vecs @ query.vibe
        elif cross_domain:
            # song <-> film: CLAP + learned map + title, see cross_space.py
            film_side = query.domain in ("movie", "text") or genre_domain == "movie"
            scores = self.cross.score(query.extra["cross"], target, title_from_query_mood=film_side)
        elif target == "song" and query.domain == "song":
            # how alike two tracks sound (CLAP) and feel (mood sentence)
            scores = (SONG_CLAP_WEIGHT * zscore_rows(self.cross.song_clap @ query.extra["cross"]["clap"])
                      + (1 - SONG_CLAP_WEIGHT) * zscore_rows(self.song_vibe @ query.vibe))
        elif target == "movie" and query.domain == "movie" and "content" in query.extra:
            scores = (1 - MOVIE_CONTENT_WEIGHT) * self.movie_vibe @ query.vibe \
                     + MOVIE_CONTENT_WEIGHT * self.movie_content @ query.extra["content"]
        else:
            vec = query.extra["distinct"] if target == "genre" and "distinct" in query.extra else query.vibe
            scores = {"song": self.song_vibe, "movie": self.movie_vibe, "genre": self.genre_vibe}[target] @ vec

        pool = np.arange(len(scores)) if candidates is None else candidates
        ranked = pool[np.argsort(-scores[pool])]
        # "match" = share of the other candidates this one beats, easier to
        # read than raw scores that mix several normalised signals
        order = np.argsort(np.argsort(scores[pool]))
        match = dict(zip(pool.tolist(), (100 * order / max(len(pool) - 1, 1)).round().astype(int).tolist()))

        results, seen = [], set()
        for i in ranked:
            card = self._card(target, int(i))
            if card["id"] == query.item_id:
                continue
            dedupe_key = card["title"].lower().split(" (")[0] + "|" + card["subtitle"].lower()
            if dedupe_key in seen:  # remasters, feat. versions, etc.
                continue
            seen.add(dedupe_key)
            card["score"] = round(float(scores[i]), 4)
            card["match"] = match[int(i)]
            card["shared_moods"] = [m for m in card["moods"] if m in query.moods]
            results.append(card)
            if len(results) == k:
                break

        return {
            "query": {"type": query.domain, "label": query.label, "moods": [str(m) for m in query.moods],
                      "vibe_text": query.vibe_text, "card": query.card},
            "target": target,
            "genre_members": target == genre_domain,
            "results": results,
        }

    def _card(self, domain, i):
        # plain Python types throughout -- parquet hands back numpy arrays and
        # scalars, which the API can't serialise
        if domain == "song":
            r = self.songs.iloc[i]
            return {"type": "song", "id": int(r.track_id), "title": str(r.track_name),
                    "subtitle": str(r.artist_name), "genre": str(r.genre or ""),
                    "moods": [str(m) for m in r.moods], "vibe_text": str(r.vibe_text),
                    "preview_url": str(r.preview_url or ""), "artwork_url": str(r.get("artwork_url") or ""),
                    "tempo": round(float(r.tempo)), "mode": str(r["mode"]),
                    "year": str(r.release_date or "")[:4]}
        if domain == "movie":
            r = self.movies.iloc[i]
            return {"type": "movie", "id": str(r.wikidata_id), "title": str(r.title), "subtitle": str(r.year),
                    "genres": [str(g) for g in r.genres], "moods": [str(m) for m in r.moods],
                    "vibe_text": str(r.vibe_text), "overview": str(r.overview),
                    "poster_url": str(r.get("poster_url") or ""),
                    "wiki_url": "https://en.wikipedia.org/wiki/" + str(r.wiki_title)}
        r = self.genres.iloc[i]
        return {"type": "genre", "id": str(r.genre_id), "title": str(r["name"]), "subtitle": str(r.domain),
                "count": int(r["count"]), "moods": [str(m) for m in r.moods],
                "covers": self._genre_covers.get(r.genre_id, []) if hasattr(self, "_genre_covers") else []}


if __name__ == "__main__":
    import sys

    rec = VibeRecommender()
    text = " ".join(sys.argv[1:]) or "rainy late-night drive through the city"
    for target in ("song", "movie", "genre"):
        out = rec.recommend({"type": "text", "query": text}, target=target, k=5)
        print(f"\n'{text}' -> {target}s   [{out['query']['vibe_text']}]")
        for r in out["results"]:
            print(f"  {r['score']:.3f}  {r['title']} -- {r['subtitle']}")
