# vibe-embeddings

A cross-domain recommendation engine that matches music and movies by semantic "vibe" rather than genre labels — give it a song, get a movie back (and vice versa), plus same-domain recommendations for either.

## Why this exists

Genre tags are a weak signal for "does this feel the same." Two tracks can both be tagged "pop" and feel nothing alike. This project instead embeds songs and movies into a shared vector space based on descriptive "vibe" text, so recommendations are driven by mood and tone rather than a fixed taxonomy.

## Status

🚧 Work in progress — currently on **Checkpoint 1** (building the seed catalogs). See [Roadmap](#roadmap) below.

## Data sources

| Domain | Source | Why |
|---|---|---|
| Song metadata + preview audio | [iTunes Search API](https://itunes.apple.com) | Free, no auth. Spotify's audio-features endpoint was deprecated in Nov 2024 and Developer Mode now requires a Premium subscription — see [notes](#a-note-on-data-source-changes) below. |
| Audio features (tempo, energy, brightness, etc.) | Computed locally via [librosa](https://librosa.org) | Since Spotify no longer provides these, they're extracted directly from 30s preview clips. |
| Movie metadata | [Wikidata](https://www.wikidata.org) (SPARQL) | Free, no auth, CC0-licensed. TMDB's terms explicitly prohibit using their API/content for ML or AI training, so Wikidata is used instead. |
| Movie plot/overview text | Wikipedia summary API | Wikidata doesn't store plot text, so the intro paragraph of the film's Wikipedia article is used as a stand-in. |

### A note on data source changes
This project's data sources changed twice during setup, both documented for transparency:
- **Spotify → iTunes + librosa**: Spotify deprecated the `audio-features`/`audio-analysis` endpoints for all new developer apps in November 2024, and as of February 2026 requires the developer account itself to have an active Premium subscription just for remaining endpoints. Rather than depend on an increasingly restricted third-party API, audio features are now computed directly from real audio — arguably a stronger technical demonstration than calling a pre-built endpoint.
- **TMDB → Wikidata**: TMDB's API Terms of Use explicitly list ML/AI training as a prohibited use case. Wikidata is CC0-licensed with no such restriction.

## Project structure
```
vibe-embeddings/
├── data/
│   ├── raw/          # downloaded audio previews (gitignored)
│   └── processed/    # cleaned catalogs, embeddings (songs.parquet, movies.parquet)
├── src/
│   ├── itunes_client.py     # song search + genre chart pulls
│   ├── audio_features.py    # audio feature extraction via librosa/ffmpeg
│   ├── wikidata_client.py   # movie metadata via SPARQL
│   └── build_catalog.py     # Checkpoint 1: builds the full seed catalogs
├── notebooks/         # exploration, sanity checks
├── models/            # saved embeddings/index (later checkpoints)
├── requirements.txt
└── README.md
```

## Setup

**Requirements:** Python 3.10+, [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) on your system PATH (used to decode audio previews).

```bash
git clone https://github.com/Charan7sai/vibe-embeddings.git
cd vibe-embeddings
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

No API keys or accounts are required — every data source used here is fully public.

## Roadmap

- [x] **Checkpoint 0** — data source clients working end-to-end (iTunes, librosa audio features, Wikidata)
- [ ] **Checkpoint 1** — build genre-diverse seed catalogs (~2,000 songs, ~3,000 movies)
- [ ] **Checkpoint 2** — generate "vibe descriptions" (structured features → descriptive text) for both domains
- [ ] **Checkpoint 3** — embed everything into a shared vector space via sentence-transformers
- [ ] **Checkpoint 4** — genre-level recommendations via genre centroid embeddings
- [ ] **Checkpoint 5** — unified recommendation engine (song/movie/genre → song/movie/genre)
- [ ] **Checkpoint 6** — sanity evaluation against a self-collected human-labeled dataset ([survey tool](https://github.com/Charan7sai/vibe-embeddings/tree/main/survey))
- [ ] **Checkpoint 7** — deployment (FastAPI + Streamlit, Dockerized)
- [ ] **Checkpoint 8** — write-up and documentation

## License

MIT