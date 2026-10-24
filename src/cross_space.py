"""
The song <-> film matching space (built by embed.py, used by recommender.py).

Measured on soundtrack pairs (src/cross_eval.py), the first version --
comparing 4-mood sentences -- was no better than random across domains.
This replaces it with signals that each beat random and add up:

  1. CLAP direct: the song's audio vs. a caption for the film, e.g.
     "dark, tense, mysterious music from the soundtrack of a horror film".
  2. A learned map: ridge regression from CLAP audio into the film-mood
     space, trained on (soundtrack song, its film) pairs.
  3. The song's title, read for mood by the sentence model.
  4. For songs with a Wikipedia article, what the article says the song is
     about, read for mood exactly the way film summaries are -- the only
     signal that knows "Heartless" is a breakup song (src/song_wiki.py).

Each item gets a CLAP vector and a mood vector in the film-mood space;
songs also get title and article mood vectors. Scores are z-normalised per
signal over the candidates, summed, and corrected for hubness.
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from moods import genre_prior
from vibe_descriptions import text_mood_scores, _text_raw_scores

MODELS_DIR = "models"
PAIRS_PATH = "data/processed/soundtrack_pairs.parquet"
WIKI_PATH = "data/processed/song_wiki.parquet"
TITLE_WEIGHT = 0.5
SONG_TEXT_WEIGHT = 2.0    # song article vs. film moods; on soundtrack songs with an article, 0 -> 2.0 took song->film from 40% to 22% mean percentile (gains flatten after 2)
SONG_PRIOR_WEIGHT = 1.5   # genre labels still nudge the moods shown on song cards
# Soundtrack ground truth is mostly orchestral score, so a film -> song model
# tuned on it alone fills every film's list with piano and strings. People
# asking "what songs feel like Barbie" want songs, so sung tracks get a
# nudge (CLAP tells sung from instrumental reliably: classical scores -1.6,
# R&B +0.7 on this scale). A strong instrumental fit can still rank.
VOCAL_WEIGHT = 0.75
VOCAL_PROMPTS = ("a song with a person singing vocals and lyrics", "instrumental music with no vocals")
# Hubness: in a shared space some items sit close to everything. Before this
# correction one comedy (Blockers) was in the top 5 films for 684 of 2,189
# songs. Subtracting each item's average score over the whole other side
# means it has to fit *this* query unusually well. At 0.75, sung soundtrack
# songs rank their own film as well as before (19.4% vs 19.7% mean
# percentile) while 1,683 films reach someone's top 5 instead of 958.
HUB_WEIGHT = 0.75


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def film_caption(moods, genres):
    genre = ", ".join(list(genres)[:2]) or "film"
    return f"{', '.join(list(moods)[:3])} music from the soundtrack of a {genre}"


def zscore_rows(s):
    """z-normalise a score vector over the candidates, so signals with
    different spreads can be summed."""
    return (s - s.mean()) / (s.std() + 1e-9)


def _row_z(m):
    return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)


def song_intros(track_ids):
    """{track_id: Wikipedia intro} for songs that have an article."""
    if not os.path.exists(WIKI_PATH):
        return {}
    wiki = pd.read_parquet(WIKI_PATH)
    wiki = wiki[wiki["intro"].fillna("").str.len() > 40]
    lookup = dict(zip(wiki["track_id"].astype(int), wiki["intro"]))
    return {int(t): lookup[int(t)] for t in track_ids if int(t) in lookup}


def intro_moods(intros, stats, film_mean, film_std):
    """Article intros -> unit mood vectors in the film-mood space (same
    scoring as film summaries, minus the film genre prior)."""
    scores = text_mood_scores(intros, [[] for _ in intros], stats)
    return unit((scores - film_mean) / film_std), (scores - film_mean) / film_std


# ---------- build ----------

def build(songs, movies, stats):
    """Compute and save the matching space. Returns (n_songs, 24) mood
    scores for songs, used for the moods and vibe text shown on cards."""
    import clap_audio
    import clap_cache

    # films: mood profile (z-scored over the catalog) + CLAP caption
    film_scores = text_mood_scores(movies["overview"].tolist(), movies["genres"], stats)
    film_mean, film_std = film_scores.mean(axis=0), film_scores.std(axis=0) + 1e-9
    film_mood = (film_scores - film_mean) / film_std
    movie_clap = clap_audio.embed_text([film_caption(m, g) for m, g in zip(movies["moods"], movies["genres"])])

    # songs: CLAP audio
    song_clap = clap_cache.embeddings(songs["track_id"].astype(int).tolist())
    if np.isnan(song_clap).any():
        raise RuntimeError("Some catalog songs have no CLAP embedding; run src/clap_cache.py first.")

    # learned map CLAP audio -> film moods, from soundtrack pairs
    pairs = pd.read_parquet(PAIRS_PATH)
    pairs = pairs[pairs["wikidata_id"].isin(set(movies["wikidata_id"]))].drop_duplicates("track_id")
    pair_clap = clap_cache.embeddings(pairs["track_id"].astype(int).tolist(), verbose=False)
    ok = ~np.isnan(pair_clap[:, 0])
    film_pos = {w: i for i, w in enumerate(movies["wikidata_id"])}
    targets = film_mood[pairs["wikidata_id"].map(film_pos).to_numpy()[ok]]
    ridge = Ridge(alpha=1.0).fit(pair_clap[ok], targets)
    learned = ridge.predict(song_clap)
    learned_mean, learned_std = learned.mean(axis=0), learned.std(axis=0) + 1e-9
    song_learned = (learned - learned_mean) / learned_std

    # song titles, read for mood
    title_raw = _text_raw_scores(songs["track_name"].astype(str).tolist())
    title_mean, title_std = title_raw.mean(axis=0), title_raw.std(axis=0) + 1e-9
    song_title = (title_raw - title_mean) / title_std

    # what the song's Wikipedia article says it's about, where there is one
    intros = song_intros(songs["track_id"])
    song_text = np.zeros_like(song_learned, dtype=np.float32)
    song_text_z = np.zeros_like(song_learned)
    if intros:
        rows = [k for k, t in enumerate(songs["track_id"]) if int(t) in intros]
        u, z = intro_moods([intros[int(songs.iloc[k]["track_id"])] for k in rows], stats, film_mean, film_std)
        song_text[rows], song_text_z[rows] = u, z

    sung, instrumental = clap_audio.embed_text(list(VOCAL_PROMPTS))
    vocal = song_clap @ sung - song_clap @ instrumental
    song_vocal = (vocal - vocal.mean()) / (vocal.std() + 1e-9)

    os.makedirs(MODELS_DIR, exist_ok=True)
    np.save(f"{MODELS_DIR}/song_clap.npy", song_clap.astype(np.float32))
    np.save(f"{MODELS_DIR}/movie_clap.npy", movie_clap)
    np.save(f"{MODELS_DIR}/song_mood.npy", unit(song_learned))
    np.save(f"{MODELS_DIR}/song_title_mood.npy", unit(song_title))
    np.save(f"{MODELS_DIR}/song_text_mood.npy", song_text)
    np.save(f"{MODELS_DIR}/song_vocal.npy", song_vocal.astype(np.float32))
    np.save(f"{MODELS_DIR}/movie_mood.npy", unit(film_mood))
    film_hub, song_hub = _hubs(song_clap, unit(song_learned), unit(song_title), song_text,
                               movie_clap, unit(film_mood))
    np.savez(f"{MODELS_DIR}/cross_space.npz", ridge_coef=ridge.coef_, ridge_intercept=ridge.intercept_,
             film_mean=film_mean, film_std=film_std, learned_mean=learned_mean, learned_std=learned_std,
             title_mean=title_mean, title_std=title_std, film_hub=film_hub, song_hub=song_hub)
    print(f"  matching space: learned map trained on {int(ok.sum())} soundtrack songs from "
          f"{pairs[ok]['wikidata_id'].nunique()} films; {len(intros)} songs have a Wikipedia intro")

    priors = np.array([genre_prior([g] if isinstance(g, str) else []) for g in songs["genre"]])
    return display_scores(song_learned, song_title, song_text_z, priors)


def _hubs(song_clap, song_mood, song_title, song_text, movie_clap, movie_mood):
    """Each film's average score over every catalog song (song -> film), and
    each song's average score over every film (film -> song), using the same
    signals as score() minus the vocal nudge."""
    has_text = np.linalg.norm(song_text, axis=1) > 0
    text_sf = _row_z(song_text[has_text] @ movie_mood.T) if has_text.any() else np.zeros((1, len(movie_mood)))
    film_hub = (_row_z(song_clap @ movie_clap.T) + _row_z(song_mood @ movie_mood.T)
                + TITLE_WEIGHT * _row_z(song_title @ movie_mood.T)).mean(axis=0) \
        + SONG_TEXT_WEIGHT * text_sf.mean(axis=0) * has_text.mean()
    song_hub = (_row_z(movie_clap @ song_clap.T) + _row_z(movie_mood @ song_mood.T)
                + TITLE_WEIGHT * _row_z(movie_mood @ song_title.T)
                + SONG_TEXT_WEIGHT * _row_z(movie_mood @ song_text.T)).mean(axis=0)
    return film_hub.astype(np.float32), song_hub.astype(np.float32)


def display_scores(song_learned, song_title, song_text_z, priors):
    """Mood scores for the chips and vibe text on song cards: the same
    signals the matching uses, plus the genre prior. Rows without an
    article have song_text_z = 0."""
    s = song_learned + TITLE_WEIGHT * song_title + SONG_TEXT_WEIGHT * song_text_z
    s = (s - s.mean(axis=-1, keepdims=True)) / (s.std(axis=-1, keepdims=True) + 1e-9)
    return s + SONG_PRIOR_WEIGHT * priors


# ---------- runtime ----------

class CrossSpace:
    def __init__(self, models_dir, stats=None):
        self.song_clap = np.load(f"{models_dir}/song_clap.npy")
        self.movie_clap = np.load(f"{models_dir}/movie_clap.npy")
        self.song_mood = np.load(f"{models_dir}/song_mood.npy")
        self.song_title = np.load(f"{models_dir}/song_title_mood.npy")
        self.song_text = np.load(f"{models_dir}/song_text_mood.npy")
        self.song_vocal = np.load(f"{models_dir}/song_vocal.npy")
        self.movie_mood = np.load(f"{models_dir}/movie_mood.npy")
        p = np.load(f"{models_dir}/cross_space.npz")
        self.p = {k: p[k] for k in p.files}
        self.stats = stats

    # representations for things that aren't in the catalog
    def live_song(self, clap_vec, title, genre, intro=""):
        learned = clap_vec @ self.p["ridge_coef"].T + self.p["ridge_intercept"]
        learned = (learned - self.p["learned_mean"]) / self.p["learned_std"]
        title_raw = _text_raw_scores([str(title)])[0]
        title_z = (title_raw - self.p["title_mean"]) / self.p["title_std"]
        text_u, text_z = np.zeros(len(learned), np.float32), np.zeros(len(learned))
        if intro and self.stats is not None:
            u, z = intro_moods([intro], self.stats, self.p["film_mean"], self.p["film_std"])
            text_u, text_z = u[0], z[0]
        prior = np.array(genre_prior([genre] if genre else []))
        scores = display_scores(learned[None], title_z[None], text_z[None], prior[None])[0]
        rep = {"clap": unit(clap_vec), "mood": unit(learned), "title": unit(title_z)}
        if intro:
            rep["text"] = text_u
        return rep, scores

    def film_mood_from_scores(self, scores):
        return unit((scores - self.p["film_mean"]) / self.p["film_std"])

    def text_query(self, text, mood_scores):
        import clap_audio
        return {"clap": clap_audio.embed_text([text])[0], "mood": unit(mood_scores)}

    def song_repr(self, i):
        rep = {"clap": self.song_clap[i], "mood": self.song_mood[i], "title": self.song_title[i]}
        if np.linalg.norm(self.song_text[i]) > 0:
            rep["text"] = self.song_text[i]
        return rep

    def movie_repr(self, i):
        return {"clap": self.movie_clap[i], "mood": self.movie_mood[i]}

    def mean_repr(self, reprs):
        keys = set.intersection(*(set(r) for r in reprs))
        return {k: unit(np.mean([r[k] for r in reprs], axis=0)) for k in keys}

    # scoring
    def score(self, q, target, title_from_query_mood, weights=None):
        """Cross-domain scores for every candidate in `target` ("song" or
        "movie"). q: {"clap", "mood", optional "title" / "text"}.
        `weights` overrides the module constants (used for tuning)."""
        w = {"title": TITLE_WEIGHT, "text": SONG_TEXT_WEIGHT, "vocal": VOCAL_WEIGHT, "hub": HUB_WEIGHT,
             **(weights or {})}
        if target == "song":
            total = zscore_rows(self.song_clap @ q["clap"]) + zscore_rows(self.song_mood @ q["mood"])
            if title_from_query_mood:
                total += w["title"] * zscore_rows(self.song_title @ q["mood"])
                total += w["text"] * zscore_rows(self.song_text @ q["mood"])
                total += w["vocal"] * self.song_vocal
            hub = self.p["song_hub"]
        else:
            total = zscore_rows(self.movie_clap @ q["clap"]) + zscore_rows(self.movie_mood @ q["mood"])
            if "title" in q:
                total += w["title"] * zscore_rows(self.movie_mood @ q["title"])
            if "text" in q:
                total += w["text"] * zscore_rows(self.movie_mood @ q["text"])
            hub = self.p["film_hub"]
        return total - w["hub"] * hub
