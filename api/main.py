"""
Checkpoint 7: the web app -- REST API under /api, and the site itself
(web/) served from the same process, so one container is the whole app.

Run:
    uvicorn api.main:app --reload
Site at http://localhost:8000, API docs at http://localhost:8000/docs
"""

import os
import sys
import threading
from typing import Literal, Optional, Union

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, model_validator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(ROOT, "web")
sys.path.insert(0, os.path.join(ROOT, "src"))
from recommender import VibeRecommender  # noqa: E402
from embedder import get_model  # noqa: E402

app = FastAPI(
    title="vibe-embeddings",
    description="Cross-domain music <-> movie recommendations by vibe, not genre.",
    version="1.1.0",
)
api = APIRouter(prefix="/api")
rec = VibeRecommender()
# load the embedding model now rather than on the first mood / live query,
# which otherwise makes some visitor wait several seconds
get_model()


def _warm_audio():
    try:
        import audio_features
        import clap_audio
        clap_audio.embed_text(["warm-up"])  # loads CLAP (~600 MB) before anyone needs it
        audio_features.warm_up()
    except Exception as e:  # never block startup on this
        print(f"audio warm-up skipped: {e}")


# loading CLAP and librosa's first compile take ~30s; do them off the request path
threading.Thread(target=_warm_audio, daemon=True).start()

Domain = Literal["song", "movie", "genre"]


class RecommendRequest(BaseModel):
    type: Literal["song", "movie", "genre", "text"]
    id: Optional[Union[int, str]] = None
    query: Optional[str] = None
    target: Domain = "movie"
    k: int = 10

    @model_validator(mode="after")
    def _need_id_or_query(self):
        if self.id is None and not self.query:
            raise ValueError("Provide either 'id' (a catalog item) or 'query' (a title or vibe text).")
        if self.type == "text" and not self.query:
            raise ValueError("type 'text' needs a 'query'.")
        self.k = max(1, min(self.k, 60))
        return self


@api.get("/health")
def health():
    return {"status": "ok", "songs": len(rec.songs), "movies": len(rec.movies), "genres": len(rec.genres)}


@api.get("/search")
def search(q: str, type: Domain = "song", limit: int = Query(10, ge=1, le=50)):
    return rec.search(q, type, limit=limit)


@api.get("/genres")
def genres(domain: Optional[Literal["music", "film"]] = None):
    df = rec.genres if domain is None else rec.genres[rec.genres["domain"] == domain]
    order = df.sort_values("count", ascending=False).index
    return [rec._card("genre", int(i)) for i in order]


@api.get("/featured")
def featured(k: int = Query(12, ge=1, le=40)):
    """What the explore page shows before anyone searches: the newest films
    in the catalog and what's currently most played in the US."""
    newest = rec.movies.assign(y=rec.movies["year"].astype(int)).sort_values("y", ascending=False, kind="stable")
    charting = rec.songs[rec.songs["seed_genre"] == "chart_us"]
    if len(charting) < k:
        charting = rec.songs.sort_values("release_date", ascending=False)
    return {
        "movies": [rec._card("movie", int(i)) for i in newest.index[:k]],
        "songs": [rec._card("song", int(i)) for i in charting.index[:k]],
    }


_pairings_cache = {}


@api.get("/pairings")
def pairings(n: int = Query(6, ge=2, le=12)):
    """Real pairs for the landing page, from this week's trending songs and
    the newest films. These are the first thing a visitor sees, so of the
    top 40 trending songs and 30 newest films, the pairs the model is most
    confident about are shown, half in each direction, with no partner
    repeated. Computed once per process."""
    if n not in _pairings_cache:
        songs = rec.songs[rec.songs["seed_genre"] == "chart_us"].head(40)
        films = rec.movies.assign(y=rec.movies["year"].astype(int)).sort_values("y", ascending=False, kind="stable")
        candidates = {"song": [], "movie": []}
        for q, target in ([({"type": "song", "id": int(s.track_id)}, "movie") for s in songs.itertuples()]
                          + [({"type": "movie", "id": m.wikidata_id}, "song") for m in films.head(30).itertuples()]):
            res = rec.recommend(q, target=target, k=1)
            if res["results"]:
                top = res["results"][0]
                candidates[q["type"]].append((top["score"], {"from": res["query"]["card"], "to": top,
                                                             "shared": top["shared_moods"]}))
        out, used = [], set()
        for kind in ("song", "movie"):
            for _, pair in sorted(candidates[kind], key=lambda c: -c[0]):
                if pair["to"]["id"] in used:
                    continue
                used.add(pair["to"]["id"])
                out.append(pair)
                if sum(p["from"]["type"] == kind for p in out) == n // 2:
                    break
        # interleave the two directions on the board
        songs_first = [p for p in out if p["from"]["type"] == "song"]
        films_first = [p for p in out if p["from"]["type"] == "movie"]
        _pairings_cache[n] = [p for pair in zip(songs_first, films_first) for p in pair]
    return _pairings_cache[n]


@api.post("/recommend")
def recommend(req: RecommendRequest):
    q = {"type": req.type, "id": req.id, "query": req.query}
    try:
        return rec.recommend(q, target=req.target, k=req.k)
    except (LookupError, KeyError) as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'\""))


app.include_router(api)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # browsers ask for /favicon.ico regardless of the <link> tag
    return FileResponse(os.path.join(WEB_DIR, "favicon.svg"), media_type="image/svg+xml")


# the site; mounted last so /api and /docs take precedence
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
