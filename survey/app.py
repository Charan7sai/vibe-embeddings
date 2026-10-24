"""
Checkpoint 6: human-labelled survey for sanity-checking cross-domain recs.

Each round shows one song (or movie) and three candidates from the other
domain: the model's top pick, a middle-ranked one and a bottom-ranked one,
in random order. The rater picks the best vibe match. If the model is
capturing something real, raters should pick its top choice well above
the 1-in-3 chance rate -- src/evaluate.py measures exactly that.

Run:
    streamlit run survey/app.py
Responses are appended to survey/responses.csv.
"""

import csv
import os
import random
import sys
import uuid
from datetime import datetime, timezone

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from recommender import VibeRecommender  # noqa: E402

RESPONSES_PATH = os.path.join(ROOT, "survey", "responses.csv")
FIELDS = ["timestamp", "rater", "direction", "query_id", "query_label",
          "option_ids", "option_ranks", "choice_rank"]


@st.cache_resource
def get_recommender():
    return VibeRecommender()


def new_round(rec):
    """Pick a random query and its top / middle / bottom candidates."""
    direction = random.choice(["song->movie", "movie->song"])
    src, dst = direction.split("->")
    df = rec.songs if src == "song" else rec.movies
    row = df.sample(1).iloc[0]
    item_id = int(row.track_id) if src == "song" else row.wikidata_id

    n = len(rec.movies if dst == "movie" else rec.songs)
    ranked = rec.recommend({"type": src, "id": item_id}, target=dst, k=n)["results"]
    picks = {"top": ranked[0], "middle": ranked[len(ranked) // 2], "bottom": ranked[-1]}
    options = list(picks.items())
    random.shuffle(options)
    return {"direction": direction, "src": src, "query": rec._card(src, df.index.get_loc(row.name)),
            "options": options}


def save_response(rnd, rater, choice_rank):
    is_new = not os.path.exists(RESPONSES_PATH)
    with open(RESPONSES_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rater": rater,
            "direction": rnd["direction"],
            "query_id": rnd["query"]["id"],
            "query_label": f"{rnd['query']['title']} - {rnd['query']['subtitle']}",
            "option_ids": "|".join(str(c["id"]) for _, c in rnd["options"]),
            "option_ranks": "|".join(r for r, _ in rnd["options"]),
            "choice_rank": choice_rank,
        })


def show_item(card, compact=False):
    st.markdown(f"**{card['title']}** — {card['subtitle']}")
    if card["type"] == "song":
        st.caption(card.get("genre") or "")
        if card.get("preview_url"):
            st.audio(card["preview_url"])
    else:
        overview = card.get("overview", "")
        st.caption(overview[:280] + ("…" if len(overview) > 280 else "") if compact else overview)


st.set_page_config(page_title="Vibe Survey", page_icon="🎧")
st.title("Which one has the same vibe?")
st.write("Listen to / read the item on top, then pick the option below whose "
         "*mood and feel* matches it best. Ignore genre — go with your gut.")

rec = get_recommender()
if "rater" not in st.session_state:
    st.session_state.rater = uuid.uuid4().hex[:8]
if "round" not in st.session_state:
    st.session_state.round = new_round(rec)
    st.session_state.count = 0

rater = st.text_input("Your name or initials (optional)", value="")
rnd = st.session_state.round

st.divider()
show_item(rnd["query"])
st.divider()

cols = st.columns(3)
for col, (rank, card) in zip(cols, rnd["options"]):
    with col:
        show_item(card, compact=True)
        if st.button("This one", key=f"pick-{card['id']}", use_container_width=True):
            save_response(rnd, rater or st.session_state.rater, rank)
            st.session_state.count += 1
            st.session_state.round = new_round(rec)
            st.rerun()

st.divider()
left, right = st.columns([1, 1])
if left.button("None fit / skip"):
    save_response(rnd, rater or st.session_state.rater, "none")
    st.session_state.round = new_round(rec)
    st.rerun()
right.caption(f"You've answered {st.session_state.count} this session. Thank you!")
