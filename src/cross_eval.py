"""
How well does song <-> film matching work? Measured on soundtrack pairs.

For every soundtrack song we know the right film (the one it's from). A
good model ranks that film near the top of all ~3,000 films; random
guessing puts it in the middle (50th percentile). The reverse direction
ranks a film's own soundtrack songs among the whole song catalog.

Tracks whose title contains the film's name ("Oppenheimer", "Dune Part
Two Suite") are dropped, so no method can win by matching names.

Run (after soundtrack_benchmark.py):
    python src/cross_eval.py
"""

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from embedder import embed
from moods import MOODS, MOOD_ANCHORS
from vibe_descriptions import (load_stats, song_mood_scores, text_mood_scores, describe_songs,
                               _song_feature_matrix, _text_raw_scores)

PAIRS_PATH = "data/processed/soundtrack_pairs.parquet"


def _norm(s):
    return re.sub(r"[^a-z0-9 ]+", "", str(s).lower()).strip()


def zscore(x, ref):
    """Standardise each mood against a reference set (the domain's catalog),
    so "dark for a song" lines up with "dark for a film"."""
    return (x - ref.mean(axis=0)) / (ref.std(axis=0) + 1e-9)


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def audio_matrix(df, stats):
    """z-scored [energy, tempo, brightness, zcr, mode] + chroma shape."""
    raw = _song_feature_matrix(df)
    z = (raw - np.array(stats["feature_mean"])) / np.array(stats["feature_std"])
    z[:, -1] = raw[:, -1]
    chroma = np.stack(df["chroma_mean"].to_numpy())
    chroma = (chroma - chroma.mean(axis=1, keepdims=True)) / (chroma.std(axis=1, keepdims=True) + 1e-9)
    return np.hstack([z, chroma])


def title_moods(titles):
    raw = _text_raw_scores([str(t) for t in titles])
    return raw


def song_to_film_report(name, sims, true_idx):
    """sims: (n_songs, n_films). Rank of each song's own film."""
    true = sims[np.arange(len(true_idx)), true_idx]
    ranks = 1 + (sims > true[:, None]).sum(axis=1)
    pct = ranks / sims.shape[1]
    return {"method": name, "median rank": int(np.median(ranks)), "mean percentile": f"{pct.mean():.1%}",
            "top 100": f"{(ranks <= 100).mean():.1%}", "top 300": f"{(ranks <= 300).mean():.1%}"}


def film_to_song_report(name, film_vecs, pool_vecs, pair_rows, film_idx_of_pair):
    """Rank each film's own soundtrack songs among the pool (catalog songs +
    all soundtrack songs). pair_rows: positions of soundtrack songs in pool."""
    pcts = []
    for film in np.unique(film_idx_of_pair):
        sims = pool_vecs @ film_vecs[film]
        own = pair_rows[film_idx_of_pair == film]
        for r in own:
            pcts.append((sims > sims[r]).sum() / len(sims))
    pcts = np.array(pcts)
    return {"method": name, "mean percentile": f"{pcts.mean():.1%}", "top 5%": f"{(pcts <= 0.05).mean():.1%}"}


def main():
    stats = load_stats()
    movies = pd.read_parquet("models/movies.parquet")
    songs = pd.read_parquet("models/songs.parquet")
    pairs = pd.read_parquet(PAIRS_PATH)
    pairs = pairs[pairs["wikidata_id"].notna() & pairs["tempo"].notna()].copy()
    pairs = pairs[pairs["wikidata_id"].isin(set(movies["wikidata_id"]))]
    named = pairs.apply(lambda r: _norm(r["film_title"]) in _norm(r["track_name"]), axis=1)
    pairs = pairs[~named].drop_duplicates("track_id").reset_index(drop=True)
    for col in ("tempo", "energy", "brightness", "zcr"):
        pairs[col] = pairs[col].astype(float)
    film_pos = {w: i for i, w in enumerate(movies["wikidata_id"])}
    true_idx = pairs["wikidata_id"].map(film_pos).to_numpy()
    print(f"{len(pairs)} soundtrack songs from {pairs['wikidata_id'].nunique()} films "
          f"({int(named.sum())} dropped for naming their film). Random baseline: 50% percentile, "
          f"top 100 = {100 / len(movies):.1%}.\n")

    # --- film side, shared by all methods
    film_moods = text_mood_scores(movies["overview"].tolist(), movies["genres"], stats)
    film_std = zscore(film_moods, film_moods)
    movie_vibe = np.load("models/movie_vibe.npy")

    # --- song side pieces
    cat_audio_moods = song_mood_scores(songs, stats)
    pair_audio_moods = song_mood_scores(pairs, stats)
    audio_std = zscore(pair_audio_moods, cat_audio_moods)
    cat_audio_std = zscore(cat_audio_moods, cat_audio_moods)

    cat_title_raw = title_moods(songs["track_name"])
    pair_title_raw = title_moods(pairs["track_name"])
    title_std = zscore(pair_title_raw, cat_title_raw)
    cat_title_std = zscore(cat_title_raw, cat_title_raw)

    # learned audio -> film-mood map, evaluated on films it never saw
    X = audio_matrix(pairs, stats)
    X_cat = audio_matrix(songs, stats)
    Y = film_std[true_idx]
    learned = np.zeros_like(Y)
    for train, test in GroupKFold(n_splits=5).split(X, Y, groups=pairs["wikidata_id"]):
        learned[test] = Ridge(alpha=10.0).fit(X[train], Y[train]).predict(X[test])
    learned_cat = Ridge(alpha=10.0).fit(X, Y).predict(X_cat)

    # --- song -> film
    rows = []
    _, _, vibe_texts = describe_songs(pairs, stats)
    rows.append(song_to_film_report("current (4-mood sentence embedding)", embed(vibe_texts) @ movie_vibe.T, true_idx))
    F = unit(film_std)
    candidates = {
        "full mood profiles (audio)": audio_std,
        "song title only": title_std,
        "learned audio map": learned,
        "audio + title": audio_std + title_std,
        "audio + title + learned": audio_std + title_std + learned,
        "title + learned": title_std + learned,
    }
    for name, vec in candidates.items():
        rows.append(song_to_film_report(name, unit(vec) @ F.T, true_idx))

    # --- CLAP: a model that has actually heard music described in words
    import clap_audio
    import clap_cache
    clap_pairs = clap_cache.embeddings(pairs["track_id"].astype(int).tolist())
    clap_cat = clap_cache.embeddings(songs["track_id"].astype(int).tolist())
    keep = ~np.isnan(clap_pairs[:, 0])
    if not keep.all():
        print(f"(CLAP missing for {int((~keep).sum())} soundtrack songs; they're excluded from CLAP rows)")
    mood_text = clap_audio.embed_text([f"{m} music" for m in MOODS])
    clap_moods_pairs = zscore(clap_pairs[keep] @ mood_text.T, clap_cat @ mood_text.T)
    clap_moods_cat = zscore(clap_cat @ mood_text.T, clap_cat @ mood_text.T)
    genre_of = lambda gs: ", ".join(list(gs)[:2]) or "film"
    top_moods = lambda ms: ", ".join(list(ms)[:3])
    captions = {
        "film mood sentence": movies["vibe_text"].tolist(),
        "soundtrack of a <genre>": [f"the soundtrack of a {genre_of(g)}" for g in movies["genres"]],
        "<moods> soundtrack of a <genre>": [f"{top_moods(m)} music from the soundtrack of a {genre_of(g)}"
                                            for m, g in zip(movies["moods"], movies["genres"])],
    }
    film_clap = {k: clap_audio.embed_text(v) for k, v in captions.items()}
    tk = true_idx[keep]
    for k, Ft in film_clap.items():
        rows.append(song_to_film_report(f"CLAP direct, caption: {k}", clap_pairs[keep] @ Ft.T, tk))
    rows.append(song_to_film_report("CLAP zero-shot moods", unit(clap_moods_pairs) @ F.T, tk))
    Xc = clap_pairs[keep]
    learned_clap = np.zeros((keep.sum(), F.shape[1]))
    groups = pairs["wikidata_id"].to_numpy()[keep]
    for train, test in GroupKFold(n_splits=5).split(Xc, groups=groups):
        learned_clap[test] = Ridge(alpha=1.0).fit(Xc[train], film_std[tk][train]).predict(Xc[test])
    rows.append(song_to_film_report("CLAP learned map", unit(learned_clap) @ F.T, tk))
    rows.append(song_to_film_report("CLAP zero-shot moods + title",
                                    unit(clap_moods_pairs + title_std[keep]) @ F.T, tk))
    best_caption = film_clap["<moods> soundtrack of a <genre>"]
    direct = clap_pairs[keep] @ best_caption.T
    moods_sim = unit(clap_moods_pairs + title_std[keep]) @ F.T
    zs = lambda s: (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-9)
    rows.append(song_to_film_report("CLAP direct + zero-shot moods + title", zs(direct) + zs(moods_sim), tk))
    learned_sim = unit(learned_clap) @ F.T
    title_sim = unit(title_std[keep]) @ F.T
    rows.append(song_to_film_report("CLAP learned + direct", zs(learned_sim) + zs(direct), tk))
    rows.append(song_to_film_report("CLAP learned + direct + title", zs(learned_sim) + zs(direct) + zs(title_sim), tk))
    rows.append(song_to_film_report("CLAP learned + direct + title (0.5)",
                                    zs(learned_sim) + zs(direct) + 0.5 * zs(title_sim), tk))
    print("SONG -> FILM  (rank of the song's own film among %d)" % len(movies))
    print(pd.DataFrame(rows).to_string(index=False))

    # --- film -> song
    pool_n = len(songs)
    pair_rows = np.arange(pool_n, pool_n + len(pairs))
    rows = []
    song_vibe = np.load("models/song_vibe.npy")
    pool_vibe = np.vstack([song_vibe, embed(vibe_texts)])
    rows.append(film_to_song_report("current (4-mood sentence embedding)", movie_vibe, pool_vibe, pair_rows, true_idx))
    pools = {
        "full mood profiles (audio)": np.vstack([cat_audio_std, audio_std]),
        "audio + title": np.vstack([cat_audio_std + cat_title_std, audio_std + title_std]),
        "audio + title + learned": np.vstack([cat_audio_std + cat_title_std + learned_cat, audio_std + title_std + learned]),
    }
    for name, pool in pools.items():
        rows.append(film_to_song_report(name, F, unit(pool), pair_rows, true_idx))

    # CLAP rows: pool = catalog songs + soundtrack songs that have embeddings
    pair_rows_k = np.arange(pool_n, pool_n + int(keep.sum()))
    rows.append(film_to_song_report("CLAP direct (<moods> soundtrack of a <genre>)", best_caption,
                                    np.vstack([clap_cat, clap_pairs[keep]]), pair_rows_k, tk))
    rows.append(film_to_song_report("CLAP zero-shot moods + title", F,
                                    unit(np.vstack([clap_moods_cat + cat_title_std, clap_moods_pairs + title_std[keep]])),
                                    pair_rows_k, tk))
    # learned map: held-out predictions for soundtrack songs, a full fit for the catalog
    learned_clap_cat = Ridge(alpha=1.0).fit(Xc, film_std[tk]).predict(clap_cat)
    pool_learned = unit(np.vstack([learned_clap_cat, learned_clap]))
    pool_clap = np.vstack([clap_cat, clap_pairs[keep]])
    pool_title = unit(np.vstack([cat_title_std, title_std[keep]]))

    def film_to_song_combo(name, w_learned, w_title):
        pcts = []
        for film in np.unique(tk):
            parts = [pool_clap @ best_caption[film]]
            if w_learned:
                parts.append(w_learned * (pool_learned @ F[film]))
            if w_title:
                parts.append(w_title * (pool_title @ F[film]))
            sims = sum((p - p.mean()) / (p.std() + 1e-9) for p in parts)
            for r in pair_rows_k[tk == film]:
                pcts.append((sims > sims[r]).sum() / len(sims))
        pcts = np.array(pcts)
        return {"method": name, "mean percentile": f"{pcts.mean():.1%}", "top 5%": f"{(pcts <= 0.05).mean():.1%}"}

    rows.append(film_to_song_combo("CLAP direct + learned", 1.0, 0))
    rows.append(film_to_song_combo("CLAP direct + learned + title (0.5)", 1.0, 0.5))
    rows.append(film_to_song_combo("CLAP direct + title (0.5)", 0, 0.5))
    print("\nFILM -> SONG  (percentile of the film's own soundtrack songs among %d songs; random = 50%%)"
          % (pool_n + len(pairs)))
    print(pd.DataFrame(rows).to_string(index=False))


def production(text_weights=(0.0, 0.5, 1.0, 1.5)):
    """The app's actual scoring (cross_space.CrossSpace.score), with the
    learned map re-fit per fold so each test film's soundtrack is unseen.
    Reported overall, for sung songs only, and for songs with a Wikipedia
    intro -- and used to choose SONG_TEXT_WEIGHT."""
    import copy
    import clap_audio
    import clap_cache
    import cross_space as cx

    stats = load_stats()
    movies = pd.read_parquet("models/movies.parquet")
    cs = cx.CrossSpace("models", stats)
    pairs = pd.read_parquet(PAIRS_PATH)
    pairs = pairs[pairs["wikidata_id"].isin(set(movies["wikidata_id"])) & pairs["tempo"].notna()]
    named = pairs.apply(lambda r: _norm(r["film_title"]) in _norm(r["track_name"]), axis=1)
    pairs = pairs[~named].drop_duplicates("track_id").reset_index(drop=True)
    true = pairs["wikidata_id"].map({w: i for i, w in enumerate(movies["wikidata_id"])}).to_numpy()

    clap = clap_cache.embeddings(pairs["track_id"].astype(int).tolist(), verbose=False)
    film_scores = text_mood_scores(movies["overview"].tolist(), movies["genres"], stats)
    film_mean, film_std = film_scores.mean(axis=0), film_scores.std(axis=0) + 1e-9
    film_mood = (film_scores - film_mean) / film_std

    # held-out learned-map predictions for every soundtrack song
    learned = np.zeros((len(pairs), film_mood.shape[1]))
    for train, test in GroupKFold(n_splits=5).split(clap, groups=pairs["wikidata_id"]):
        learned[test] = Ridge(alpha=1.0).fit(clap[train], film_mood[true[train]]).predict(clap[test])
    learned = (learned - cs.p["learned_mean"]) / cs.p["learned_std"]
    title_z = (_text_raw_scores(pairs["track_name"].astype(str).tolist()) - cs.p["title_mean"]) / cs.p["title_std"]
    intros = cx.song_intros(pairs["track_id"])
    text_u = np.zeros_like(title_z, dtype=np.float32)
    rows_with = [k for k, t in enumerate(pairs["track_id"]) if int(t) in intros]
    if rows_with:
        text_u[rows_with] = cx.intro_moods([intros[int(pairs.iloc[k]["track_id"])] for k in rows_with],
                                           stats, cs.p["film_mean"], cs.p["film_std"])[0]
    sung, inst = clap_audio.embed_text(list(cx.VOCAL_PROMPTS))
    cat_v = cs.song_clap @ sung - cs.song_clap @ inst
    vocal = ((clap @ sung - clap @ inst) - cat_v.mean()) / cat_v.std()

    # film -> song: pool = catalog songs + these soundtrack songs
    pool = copy.copy(cs)
    pool.song_clap = np.vstack([cs.song_clap, clap])
    pool.song_mood = np.vstack([cs.song_mood, cx.unit(learned)])
    pool.song_title = np.vstack([cs.song_title, cx.unit(title_z)])
    pool.song_text = np.vstack([cs.song_text, text_u])
    pool.song_vocal = np.concatenate([cs.song_vocal, vocal])
    pool.p = dict(cs.p)
    _, pool.p["song_hub"] = cx._hubs(pool.song_clap, pool.song_mood, pool.song_title, pool.song_text,
                                     cs.movie_clap, cs.movie_mood)
    n_cat = len(cs.song_clap)

    subsets = {"all": np.ones(len(pairs), bool), "sung": vocal > 0,
               "with Wikipedia intro": np.isin(np.arange(len(pairs)), rows_with)}
    print(f"\nPRODUCTION SCORING ({len(pairs)} soundtrack songs, {pairs['wikidata_id'].nunique()} films; "
          f"{int(subsets['sung'].sum())} sung, {len(rows_with)} with an intro)")
    out = []
    for tw in text_weights:
        weights = {"text": tw}
        s2f = np.zeros(len(pairs))
        for k in range(len(pairs)):
            rep = {"clap": clap[k], "mood": cx.unit(learned[k]), "title": cx.unit(title_z[k])}
            if k in rows_with:
                rep["text"] = text_u[k]
            s = cs.score(rep, "movie", False, weights)
            s2f[k] = (s > s[true[k]]).sum() / len(s)
        f2s = np.full(len(pairs), np.nan)
        for film in np.unique(true):
            s = pool.score(cs.movie_repr(film), "song", True, weights)
            for k in np.flatnonzero(true == film):
                f2s[k] = (s > s[n_cat + k]).sum() / len(s)
        for name, m in subsets.items():
            if m.any():
                out.append({"article weight": tw, "subset": name, "n": int(m.sum()),
                            "song->film pct": f"{s2f[m].mean():.1%}", "song->film top 100": f"{(s2f[m] * len(movies) < 100).mean():.1%}",
                            "film->song pct": f"{f2s[m].mean():.1%}", "film->song top 5%": f"{(f2s[m] <= 0.05).mean():.1%}"})
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    import sys
    if "--production" in sys.argv:
        production()
    else:
        main()
