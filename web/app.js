"use strict";

const API = "/api";
const MOODS = ["energetic", "calm", "melancholic", "uplifting", "dark", "tense", "romantic", "dreamy",
  "aggressive", "playful", "nostalgic", "epic", "eerie", "warm", "gritty", "rebellious", "intimate",
  "mysterious", "triumphant", "bittersweet", "chaotic", "sensual", "whimsical", "brooding"];

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const ICON_PLAY = '<svg class="i-play" viewBox="0 0 14 14" aria-hidden="true"><path d="M3 1.5v11l9-5.5z" fill="currentColor"/></svg>';
const ICON_PAUSE = '<svg class="i-pause" viewBox="0 0 14 14" aria-hidden="true"><path d="M3 1.5h3v11H3zM8 1.5h3v11H8z" fill="currentColor"/></svg>';

const CATALOG = {};  // song and film counts, from /api/health

const state = {
  mode: "song",
  query: null,       // the request that produced the current results
  results: {},       // target -> API response
  tab: null,
  suggestions: [],
  active: -1,        // highlighted suggestion
  genres: null,
  requestId: 0,
};

/* Wikidata film genres arrive lowercase ("action thriller") while iTunes
   music genres are Title Case ("Arena Rock"); show them all the same way.
   Only the first letter of each word part changes, so "children's" stays
   "Children's" and "LGBTQ" / "R&B" keep their capitals. */
const SMALL_WORDS = new Set(["of", "the", "and", "a", "an", "in", "on", "for", "to", "with", "or"]);
function formatGenre(name) {
  let first = true;
  return String(name ?? "").split(/([\s\-\/]+)/).map((part) => {
    if (!part || /^[\s\-\/]+$/.test(part)) return part;
    const keepSmall = !first && SMALL_WORDS.has(part.toLowerCase());
    first = false;
    return keepSmall ? part.toLowerCase() : part.charAt(0).toUpperCase() + part.slice(1);
  }).join("");
}

function prettifyGenres(value) {
  if (Array.isArray(value)) { value.forEach(prettifyGenres); return value; }
  if (value && typeof value === "object") {
    if (value.type === "genre" && value.title) value.title = formatGenre(value.title);
    if (value.type === "movie" && Array.isArray(value.genres)) value.genres = value.genres.map(formatGenre);
    if (value.type === "song" && value.genre) value.genre = formatGenre(value.genre);
    Object.values(value).forEach((v) => { if (v && typeof v === "object") prettifyGenres(v); });
  }
  return value;
}

async function getJSON(url, options) {
  const resp = await fetch(url, options);
  let body = null;
  try { body = prettifyGenres(await resp.json()); } catch (e) { /* non-JSON error page */ }
  if (!resp.ok) {
    const detail = body && (typeof body.detail === "string" ? body.detail : null);
    throw new Error(detail || `The server returned an error (${resp.status}).`);
  }
  return body;
}

const recommend = (q, target, k) => getJSON(`${API}/recommend`, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ ...q, target, k }),
});

/* ---------------- theme ---------------- */

function currentTheme() {
  const set = document.documentElement.getAttribute("data-theme");
  if (set) return set;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function updateThemeButton() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  $("#theme-toggle").setAttribute("aria-label", `Switch to ${next} mode`);
}

$("#theme-toggle").addEventListener("click", () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("theme", next); } catch (e) { /* storage blocked */ }
  updateThemeButton();
});
updateThemeButton();

/* ---------------- routing ---------------- */

function parseHash() {
  const hash = location.hash || "#/";
  if (!hash.startsWith("#/explore")) return { view: "landing", anchor: hash.startsWith("#/") ? null : hash.slice(1) };
  const params = new URLSearchParams(hash.split("?")[1] || "");
  return { view: "explore", params };
}

function queryToHash(q) {
  const p = new URLSearchParams({ type: q.type });
  if (q.id != null) p.set("id", q.id);
  if (q.query) p.set("q", q.query);
  return `#/explore?${p}`;
}

function go(q) {
  const target = queryToHash(q);
  if (location.hash === target) route();
  else location.hash = target;
}

function route() {
  const r = parseHash();
  $("#landing").hidden = r.view !== "landing";
  $("#explore").hidden = r.view !== "explore";
  $$(".site-nav a").forEach((a) => a.classList.toggle("active", a.dataset.nav === "explore" && r.view === "explore"));
  closeSheet();
  if (!player.paused) player.pause();  // a new page shouldn't keep the last preview playing

  if (r.view === "landing") {
    if (r.anchor) document.getElementById(r.anchor)?.scrollIntoView();
    else window.scrollTo(0, 0);
    return;
  }

  const type = r.params.get("type");
  if (type) {
    const q = { type };
    if (r.params.get("id")) q.id = type === "song" ? Number(r.params.get("id")) : r.params.get("id");
    if (r.params.get("q")) q.query = r.params.get("q");
    if (!sameQuery(q, state.query)) runQuery(q);
  } else {
    state.query = null;
    $("#results").hidden = true;
    $("#query-banner").hidden = true;
    setStatus("");
    showFeatured();
  }
  window.scrollTo(0, 0);
}

const sameQuery = (a, b) => !!a && !!b && a.type === b.type && String(a.id ?? "") === String(b.id ?? "")
  && (a.query ?? "") === (b.query ?? "");

window.addEventListener("hashchange", route);

/* ---------------- landing ---------------- */

async function loadLanding() {
  $("#mood-vocab").innerHTML = MOODS.map((m) => `<span class="chip">${m}</span>`).join("");

  getJSON(`${API}/health`).then((h) => {
    Object.assign(CATALOG, h);
    $("#intro-stats").textContent = `${h.songs.toLocaleString()} English songs · `
      + `${h.movies.toLocaleString()} English films · matched on 24 moods`;
  }).catch(() => { /* keep the static numbers */ });

  loadPairings();
  getJSON(`${API}/featured?k=6`).then((f) => {
    $("#intro-art").innerHTML = f.movies.filter((m) => m.poster_url).slice(0, 6)
      .map((m) => `<img src="${esc(m.poster_url)}" alt="" referrerpolicy="no-referrer" title="${esc(m.title)}">`).join("");
  }).catch(() => {});

  let data;
  try { data = await getJSON("data/landing.json"); } catch (e) { return; }
  const tile = (p) => `
    <button class="cast-tile" type="button" data-seed-type="${esc(p.seed.type)}" data-seed-query="${esc(p.seed.query)}"
      aria-label="${esc(p.name)}: start from ${esc(p.seed.type === "movie" ? p.seed.query : "their song")}">
      <figure style="margin:0;height:100%">
        <img src="${esc(p.image)}" alt="${esc(p.name)}" loading="lazy" referrerpolicy="no-referrer">
        <figcaption><span class="cast-name">${esc(p.name)}</span><span class="cast-note">${esc(p.note)}</span></figcaption>
      </figure>
    </button>`;
  $("#cast-characters").innerHTML = data.characters.map(tile).join("");
  $("#cast-singers").innerHTML = data.singers.map(tile).join("");
  $$(".cast-tile").forEach((b) => b.addEventListener("click", () => {
    setMode(b.dataset.seedType);
    $("#search-input").value = "";
    go({ type: b.dataset.seedType, query: b.dataset.seedQuery });
  }));

  const credit = (p) => {
    const who = p.author ? `${esc(p.author)}, ` : (p.free ? "" : "rights holder, ");
    return `<li>${esc(p.name)}: ${who}<a href="${esc(p.source_page)}" target="_blank" rel="noopener">${esc(p.license || "see source")}</a></li>`;
  };
  $("#credits-list").innerHTML = [...data.characters, ...data.singers].map(credit).join("");
}

function pairSide(c) {
  const img = c.type === "song"
    ? `<img class="img-song" src="${esc(thumb(c.artwork_url, 120))}" alt="" loading="lazy" referrerpolicy="no-referrer">`
    : `<img class="img-movie" src="${esc(c.poster_url)}" alt="" loading="lazy" referrerpolicy="no-referrer">`;
  const sub = c.type === "song" ? c.subtitle : `Film, ${c.subtitle}`;
  return `${img}<div><div class="pair-title">${esc(c.title)}</div><div class="pair-sub">${esc(sub)}</div></div>`;
}

async function loadPairings() {
  const board = $("#board");
  board.innerHTML = '<li class="board-loading loading-dots">Working out pairings</li>';
  let pairs;
  try { pairs = await getJSON(`${API}/pairings?n=6`); } catch (e) { board.innerHTML = ""; return; }
  board.innerHTML = pairs.map((p) => `
    <li class="pair" tabindex="0" role="link" data-type="${esc(p.from.type)}" data-id="${esc(p.from.id)}"
      aria-label="${esc(p.from.title)} paired with ${esc(p.to.title)}">
      ${pairSide(p.from)}
      <span class="pair-arrow" aria-hidden="true">&rarr;</span>
      ${pairSide(p.to)}
      <div class="pair-moods">${p.shared.length ? `Both <b>${p.shared.map(esc).join(", ")}</b>` : "Closest overall mood"}</div>
    </li>`).join("");
  $$(".pair", board).forEach((li) => {
    const open = () => go({ type: li.dataset.type, id: li.dataset.type === "song" ? Number(li.dataset.id) : li.dataset.id });
    li.addEventListener("click", open);
    li.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
  });
  renderAnatomy(pairs.find((p) => p.from.type === "song") || pairs[0]);
}

function renderAnatomy(p) {
  if (!p) return;
  const song = p.from.type === "song" ? p.from : p.to;
  const film = p.from.type === "song" ? p.to : p.from;
  const firstSentence = (film.overview || "").split(/(?<=\.)\s/)[0];
  const moodChips = (c) => chipsHtml(c.moods || [], p.shared);
  $("#anatomy").innerHTML = `
    <div class="anat-side">
      <div class="anat-head">
        <img class="img-song" src="${esc(thumb(song.artwork_url, 240))}" alt="" referrerpolicy="no-referrer">
        <div><p class="anat-kind">Song</p><h3>${esc(song.title)}</h3><p class="anat-kind">${esc(song.subtitle)}</p></div>
      </div>
      <p class="anat-step">What the 30-second preview measured</p>
      <dl class="anat-specs">
        <dt>Tempo</dt><dd>${esc(song.tempo)} BPM</dd>
        <dt>Key</dt><dd>${esc(song.mode)}</dd>
        <dt>Genre</dt><dd>${esc(song.genre)}</dd>
      </dl>
      <p class="anat-step">In mood words</p>
      <p class="anat-words">${esc(song.vibe_text)}</p>
      <div class="chips" style="margin-top:10px">${moodChips(song)}</div>
    </div>
    <div class="anat-mid">
      <p>Shared moods</p>
      <div class="chips">${p.shared.length ? chipsHtml(p.shared, p.shared) : '<span class="chip">none exactly, closest overall</span>'}</div>
      <div class="anat-score">#1</div>
      <p>best match for this ${p.from.type === "song" ? "song" : "film"} out of
        ${p.from.type === "song" ? (CATALOG.movies ? CATALOG.movies.toLocaleString() + " films" : "every film")
                                 : (CATALOG.songs ? CATALOG.songs.toLocaleString() + " songs" : "every song")}</p>
    </div>
    <div class="anat-side">
      <div class="anat-head">
        <img class="img-movie" src="${esc(film.poster_url)}" alt="" referrerpolicy="no-referrer">
        <div><p class="anat-kind">Film</p><h3>${esc(film.title)}</h3><p class="anat-kind">${esc(film.subtitle)}</p></div>
      </div>
      <p class="anat-step">What Wikipedia says</p>
      <p class="anat-quote">${esc(firstSentence)}</p>
      <p class="anat-step">Genres</p>
      <p class="anat-words" style="font-style:normal">${esc((film.genres || []).slice(0, 4).join(", "))}</p>
      <p class="anat-step">In mood words</p>
      <p class="anat-words">${esc(film.vibe_text)}</p>
      <div class="chips" style="margin-top:10px">${moodChips(film)}</div>
    </div>`;
}

const QUICK_PLACEHOLDERS = {
  song: "Blinding Lights, Espresso, Not Like Us...",
  movie: "Oppenheimer, Sinners, Past Lives...",
  genre: "Horror Film, Country, Film Noir...",
  text: "a rainy late-night drive through the city",
};
$$("input[name='quick-mode']").forEach((r) => r.addEventListener("change", () => {
  $("#quick-input").placeholder = QUICK_PLACEHOLDERS[r.value];
  $("#quick-input").focus();
}));
$("#quick-input").placeholder = QUICK_PLACEHOLDERS.song;

$("#quick-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const mode = $("input[name='quick-mode']:checked").value;
  const text = $("#quick-input").value.trim();
  if (!text) { $("#quick-input").focus(); return; }
  setMode(mode);
  $("#search-input").value = text;
  if (mode === "genre") {
    const hits = await getJSON(`${API}/search?q=${encodeURIComponent(text)}&type=genre&limit=1`).catch(() => []);
    go(hits.length ? { type: "genre", id: hits[0].id } : { type: "text", query: text });
    return;
  }
  go({ type: mode, query: text });
});

/* ---------------- explore: mode + search ---------------- */

const PLACEHOLDERS = {
  song: "Search a song, like Espresso or Blinding Lights",
  movie: "Search a film, like Oppenheimer or Past Lives",
  genre: "Search a genre, like Horror Film or Country",
  text: "Describe a mood, like rainy late-night drive",
};

function setMode(mode) {
  state.mode = mode;
  $$(".mode-switch button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
  $("#search-input").placeholder = PLACEHOLDERS[mode];
  hideSuggestions();
}

$$(".mode-switch button").forEach((b) => b.addEventListener("click", () => {
  setMode(b.dataset.mode);
  $("#search-input").value = "";
  $("#search-input").focus();
  if (!state.query) showFeatured();
}));

let searchTimer = null;
$("#search-input").addEventListener("input", () => {
  clearTimeout(searchTimer);
  const text = $("#search-input").value.trim();
  if (state.mode === "text" || text.length < 2) { hideSuggestions(); return; }
  searchTimer = setTimeout(() => loadSuggestions(text), 180);
});

async function loadSuggestions(text) {
  const mode = state.mode;
  let hits = [];
  try {
    hits = await getJSON(`${API}/search?q=${encodeURIComponent(text)}&type=${mode}&limit=8`);
  } catch (e) { hits = []; }
  if (mode !== state.mode || $("#search-input").value.trim() !== text) return;  // stale

  state.suggestions = hits.map((h) => ({ kind: "item", card: h }));
  if (mode === "song" || mode === "movie") {
    state.suggestions.push({ kind: "live", text });
  }
  state.active = -1;
  renderSuggestions();
}

function renderSuggestions() {
  const list = $("#suggestions");
  if (!state.suggestions.length) { hideSuggestions(); return; }
  list.innerHTML = state.suggestions.map((s, i) => {
    if (s.kind === "live") {
      const where = state.mode === "song" ? "iTunes" : "Wikipedia";
      return `<li role="option" class="s-live" data-i="${i}" aria-selected="${i === state.active}">Look up "${esc(s.text)}" on ${where}</li>`;
    }
    const c = s.card;
    const img = c.type === "song" ? thumb(c.artwork_url, 120)
      : c.type === "movie" ? c.poster_url : (c.covers || [])[0];
    const imgHtml = img ? `<img src="${esc(img)}" alt="" class="${c.type === "movie" ? "movie-thumb" : ""}" loading="lazy" referrerpolicy="no-referrer">`
      : '<span class="thumb-blank"></span>';
    const sub = c.type === "genre" ? `${c.subtitle} genre, ${c.count} titles` : c.subtitle;
    return `<li role="option" data-i="${i}" aria-selected="${i === state.active}">${imgHtml}
      <div><div class="s-title">${esc(c.title)}</div><div class="s-sub">${esc(sub)}</div></div></li>`;
  }).join("");
  list.hidden = false;
  $$("li", list).forEach((li) => li.addEventListener("mousedown", (e) => {
    e.preventDefault();  // keep focus in the input until we've handled it
    pickSuggestion(Number(li.dataset.i));
  }));
}

function hideSuggestions() {
  $("#suggestions").hidden = true;
  state.suggestions = [];
  state.active = -1;
}

function pickSuggestion(i) {
  const s = state.suggestions[i];
  if (!s) return;
  hideSuggestions();
  if (s.kind === "live") {
    go({ type: state.mode, query: s.text });
  } else {
    $("#search-input").value = s.card.title;
    go({ type: s.card.type, id: s.card.id });
  }
}

$("#search-input").addEventListener("keydown", (e) => {
  const n = state.suggestions.length;
  if (e.key === "ArrowDown" && n) { e.preventDefault(); state.active = (state.active + 1) % n; renderSuggestions(); }
  else if (e.key === "ArrowUp" && n) { e.preventDefault(); state.active = (state.active - 1 + n) % n; renderSuggestions(); }
  else if (e.key === "Escape") hideSuggestions();
});
$("#search-input").addEventListener("blur", () => setTimeout(hideSuggestions, 120));

$("#search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("#search-input").value.trim();
  if (state.active >= 0) { pickSuggestion(state.active); return; }
  if (!text) { $("#search-input").focus(); return; }
  hideSuggestions();
  if (state.mode === "genre") {
    // first catalog match, if any
    getJSON(`${API}/search?q=${encodeURIComponent(text)}&type=genre&limit=1`).then((hits) => {
      if (hits.length) go({ type: "genre", id: hits[0].id });
      else setStatus(`No genre matches "${text}".`, true);
    });
    return;
  }
  go({ type: state.mode, query: text });
});

function setStatus(html, isError = false) {
  const el = $("#status");
  el.innerHTML = html ? `<p class="${isError ? "status-error" : ""}">${html}</p>` : "";
}

/* ---------------- explore: running a query ---------------- */

const DEFAULT_TAB = { song: "movie", movie: "song", text: "movie" };
const SIZES = { song: 24, movie: 24, genre: 12 };

async function runQuery(q) {
  const id = ++state.requestId;
  state.query = q;
  setMode(q.type);
  $("#featured").innerHTML = "";
  $("#results").hidden = true;
  $("#query-banner").hidden = true;
  const live = q.id == null && q.type !== "text" && q.type !== "genre";
  setStatus(live && q.type === "song"
    ? '<span class="loading-dots">Finding the song and listening to its preview</span>'
    : '<span class="loading-dots">Matching</span>');

  try {
    // the query is resolved (and cached) on the first call, which matters for
    // live song lookups that download and analyse audio; then the other two
    const first = q.type === "genre" ? null : DEFAULT_TAB[q.type];
    const results = {};
    if (first) results[first] = await recommend(q, first, SIZES[first]);
    const rest = ["song", "movie", "genre"].filter((t) => !(t in results));
    const responses = await Promise.all(rest.map((t) => recommend(q, t, SIZES[t])));
    rest.forEach((t, i) => { results[t] = responses[i]; });
    if (id !== state.requestId) return;  // a newer query started

    state.results = results;
    const any = results.song || results.movie;
    renderBanner(any.query);
    let tab = first;
    if (q.type === "genre") {
      tab = results.song.genre_members ? "song" : "movie";
    }
    setStatus("");
    $("#results").hidden = false;
    showTab(tab);
  } catch (e) {
    if (id !== state.requestId) return;
    setStatus(esc(e.message), true);
  }
}

function renderBanner(q) {
  const c = q.card;
  const banner = $("#query-banner");
  const moods = chipsHtml(q.moods || [], []);
  let img = "", kicker = "", title = esc(q.label), sub = "";
  if (!c) {
    kicker = "Your mood";
    title = `"${esc(q.label)}"`;
  } else if (c.type === "song") {
    kicker = c.live ? "Song, looked up on iTunes" : "Song";
    img = c.artwork_url ? `<img src="${esc(thumb(c.artwork_url, 300))}" alt="" referrerpolicy="no-referrer">` : "";
    title = esc(c.title);
    sub = esc([c.subtitle, c.genre, c.year].filter(Boolean).join(" · "));
  } else if (c.type === "movie") {
    kicker = c.live ? "Film, looked up on Wikipedia" : "Film";
    img = c.poster_url ? `<img class="poster" src="${esc(c.poster_url)}" alt="" referrerpolicy="no-referrer">` : "";
    title = esc(c.title);
    sub = esc([c.subtitle, (c.genres || []).slice(0, 3).join(", ")].filter(Boolean).join(" · "));
  } else {
    kicker = c.subtitle === "music" ? "Music genre" : "Film genre";
    img = c.covers && c.covers[0] ? `<img class="${c.subtitle === "film" ? "poster" : ""}" src="${esc(c.covers[0])}" alt="" referrerpolicy="no-referrer">` : "";
    title = esc(c.title);
    sub = `${c.count} titles in the catalog`;
  }
  banner.innerHTML = `<div class="banner">${img}<div>
      <p class="banner-kicker">${kicker}</p><h2>${title}</h2>
      ${sub ? `<p class="banner-sub">${sub}</p>` : ""}
      ${q.vibe_text ? `<p class="banner-vibe">${esc(q.vibe_text)}</p>` : ""}
      ${moods ? `<div class="chips">${moods}</div>` : ""}
    </div></div>`;
  banner.hidden = false;
}

function showTab(tab) {
  state.tab = tab;
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === tab)));
  $$("[data-count]").forEach((s) => { s.textContent = state.results[s.dataset.count]?.results.length ?? ""; });

  const res = state.results[tab];
  const q = res.query;
  const label = { song: "songs", movie: "films", genre: "genres" }[tab];
  let caption;
  if (res.genre_members) caption = `The ${label} filed under ${q.card.title}, most typical first.`;
  else if (q.type === "genre") caption = `The ${label} whose mood is closest to ${q.card.title}.`;
  else caption = `The ${label} with the closest mood, best match first. Moods shared with your pick are filled in.`;
  $("#panel-caption").textContent = caption;

  const panel = $("#panel");
  if (!res.results.length) {
    panel.innerHTML = '<p class="empty">Nothing to show here.</p>';
    return;
  }
  const scores = res.results.map((r) => r.score);
  const hi = Math.max(...scores), lo = Math.min(...scores);
  panel.innerHTML = res.results.map((r, i) => {
    const strength = hi > lo ? 0.3 + 0.7 * (r.score - lo) / (hi - lo) : 1;
    return cardHtml(r, i + 1, strength, q.moods || []);
  }).join("");
  bindCards(panel);
}

$$(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

/* ---------------- cards ---------------- */

function thumb(url, size) {
  return (url || "").replace(/\d+x\d+bb/, `${size}x${size}bb`);
}

function chipsHtml(moods, shared) {
  return moods.map((m) => `<span class="chip${shared.includes(m) ? " shared" : ""}">${esc(m)}</span>`).join("");
}

function cardHtml(c, rank, strength, queryMoods) {
  const shared = c.shared_moods || [];
  const rankHtml = "";
  const matchHtml = rank ? `<div class="match-row"><span class="rank">${String(rank).padStart(2, "0")}</span>`
    + `<div class="match" aria-hidden="true"><span style="width:${Math.round(strength * 100)}%"></span></div></div>` : "";

  if (c.type === "movie") {
    const genres = (c.genres || []).slice(0, 2).join(", ");
    return `<article class="card movie-card" tabindex="0" data-type="movie" data-id="${esc(c.id)}">
      ${rankHtml}
      <h3>${esc(c.title)}</h3>
      <p class="meta">${esc(c.subtitle)}${genres ? " · " + esc(genres) : ""}</p>
      <div class="chips">${chipsHtml((c.moods || []).slice(0, 3), shared)}</div>
      ${matchHtml}
      <div class="pop" role="dialog" aria-label="${esc(c.title)}">
        <button class="pop-close" type="button" aria-label="Close">&#x2715;</button>
        <h3>${esc(c.title)}</h3>
        <p class="pop-sub">${esc(c.subtitle)}</p>
        <div class="pop-row">
          <img class="pop-poster" data-src="${esc(c.poster_url)}" alt="Poster for ${esc(c.title)}" referrerpolicy="no-referrer">
          <div>
            <p class="plot">${esc(c.overview)}</p>
            <p class="pop-meta">${esc((c.genres || []).slice(0, 4).join(", "))}</p>
          </div>
        </div>
        <div class="chips">${chipsHtml(c.moods || [], shared)}</div>
        <div class="pop-actions">
          <button class="btn btn-primary" type="button" data-act="more">Match from this film</button>
          ${c.wiki_url ? `<a class="btn btn-ghost" href="${esc(c.wiki_url)}" target="_blank" rel="noopener">Wikipedia</a>` : ""}
        </div>
      </div>
    </article>`;
  }

  if (c.type === "song") {
    const meta2 = [c.genre, c.year, c.tempo ? `${c.tempo} BPM` : ""].filter(Boolean).join(" · ");
    const play = c.preview_url ? `<button class="play" type="button" data-preview="${esc(c.preview_url)}" aria-label="Play preview of ${esc(c.title)}">${ICON_PLAY}${ICON_PAUSE}</button>` : "";
    return `<article class="card song-card" tabindex="0" data-type="song" data-id="${esc(c.id)}">
      ${rankHtml}
      <div class="song-head">
        <img src="${esc(thumb(c.artwork_url, 120))}" alt="" loading="lazy" referrerpolicy="no-referrer">
        <div><h3>${esc(c.title)}</h3><p class="meta">${esc(c.subtitle)}</p></div>
        ${play}
      </div>
      <p class="meta">${esc(meta2)}</p>
      <div class="chips">${chipsHtml((c.moods || []).slice(0, 3), shared)}</div>
      ${matchHtml}
      <div class="pop" role="dialog" aria-label="${esc(c.title)}">
        <button class="pop-close" type="button" aria-label="Close">&#x2715;</button>
        <h3>${esc(c.title)}</h3>
        <p class="pop-sub">${esc(c.subtitle)}</p>
        <div class="pop-row">
          <img class="pop-art" data-src="${esc(thumb(c.artwork_url, 600))}" alt="Cover art for ${esc(c.title)}" referrerpolicy="no-referrer">
          <div>
            <p class="pop-meta" style="margin-top:0">${esc([c.genre, c.year].filter(Boolean).join(" · "))}</p>
            <p class="pop-meta">${esc([c.tempo ? `${c.tempo} BPM` : "", c.mode ? `${c.mode} key` : ""].filter(Boolean).join(" · "))}</p>
            ${c.vibe_text ? `<p class="plot" style="margin-top:10px">${esc(c.vibe_text)}</p>` : ""}
          </div>
        </div>
        <div class="chips">${chipsHtml(c.moods || [], shared)}</div>
        <div class="pop-actions">
          ${play ? `<button class="btn btn-ghost play-wide" type="button" data-preview="${esc(c.preview_url)}">Play preview</button>` : ""}
          <button class="btn btn-primary" type="button" data-act="more">Match from this song</button>
        </div>
      </div>
    </article>`;
  }

  // genre
  const covers = (c.covers || []).slice(0, 4);
  return `<article class="card genre-card" tabindex="0" data-type="genre" data-id="${esc(c.id)}">
    <div class="covers" style="grid-template-columns:repeat(${Math.max(covers.length, 1)},1fr)">${covers.map((u) => `<img src="${esc(u)}" alt="" loading="lazy" referrerpolicy="no-referrer">`).join("")}</div>
    <div class="genre-body">
      ${rankHtml}
      <h3>${esc(c.title)}</h3>
      <p class="meta">${c.subtitle === "music" ? "Music" : "Film"} · ${c.count} titles</p>
      <div class="chips">${chipsHtml((c.moods || []).slice(0, 3), c.shared_moods || [])}</div>
    </div>
  </article>`;
}

const canHover = matchMedia("(hover: hover)").matches;

function bindCards(root) {
  $$(".card", root).forEach((card) => {
    const pop = $(".pop", card);
    const reveal = () => {
      // load the big image only once someone actually looks at it
      $$("img[data-src]", card).forEach((img) => {
        if (img.dataset.src) { img.src = img.dataset.src; img.removeAttribute("data-src"); }
      });
      // open towards the side with room
      if (pop) {
        const rect = card.getBoundingClientRect();
        card.classList.toggle("flip", rect.left + 410 > document.documentElement.clientWidth);
      }
    };
    card.addEventListener("mouseenter", reveal);
    if (canHover) {
      card.addEventListener("mouseleave", () => {
        // a clicked button keeps focus; let go of it so this card really
        // closes instead of staying open next to the one being hovered
        if (card.contains(document.activeElement)) document.activeElement.blur();
        stopPreviewIn(card);
      });
    }
    card.addEventListener("focusin", reveal);
    card.addEventListener("focusout", (e) => {
      // keyboard users tabbing out of a card close it too
      if (!card.contains(e.relatedTarget) && !card.matches(":hover")) stopPreviewIn(card);
    });

    card.addEventListener("click", (e) => {
      const preview = e.target.closest("[data-preview]");
      if (preview) { e.stopPropagation(); togglePreview(preview.dataset.preview); return; }
      if (e.target.closest(".pop-close")) { e.stopPropagation(); closeSheet(); return; }
      if (e.target.closest("a")) return;
      // inside the big card, only its buttons do anything
      if (e.target.closest(".pop") && !e.target.closest("[data-act]")) return;
      const type = card.dataset.type;
      const id = type === "song" ? Number(card.dataset.id) : card.dataset.id;
      if (type === "genre" || e.target.closest("[data-act='more']") || canHover) {
        go({ type, id });
        return;
      }
      // touch: first tap opens the big card
      reveal();
      openSheet(card);
    });
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && e.target === card) {
        const type = card.dataset.type;
        go({ type, id: type === "song" ? Number(card.dataset.id) : card.dataset.id });
      }
    });
  });
  syncPlayButtons();
  preloadImages(root);
}

// fetch the big posters / covers while the browser is idle, so they're
// already there when someone hovers instead of popping in half-loaded
function preloadImages(root) {
  const pending = $$("img[data-src]", root);
  const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 300));
  const next = () => {
    const batch = pending.splice(0, 4);
    batch.forEach((img) => {
      if (img.dataset.src) { img.src = img.dataset.src; img.removeAttribute("data-src"); }
    });
    if (pending.length) idle(next);
  };
  idle(next);
}

function openSheet(card) {
  closeSheet();
  card.classList.add("open");
  $("#sheet-backdrop").hidden = false;
}
function closeSheet() {
  $$(".card.open").forEach((c) => {
    c.classList.remove("open");
    stopPreviewIn(c);
  });
  $("#sheet-backdrop").hidden = true;
}

// a preview started from a big card stops when that card closes
function stopPreviewIn(card) {
  const playingHere = $$("[data-preview]", card).some((b) => b.dataset.preview === player.src);
  if (playingHere && !player.paused) player.pause();
}
$("#sheet-backdrop").addEventListener("click", closeSheet);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSheet(); });

/* ---------------- audio ---------------- */

const player = $("#player");
function togglePreview(url) {
  if (player.src === url && !player.paused) {
    player.pause();
  } else {
    if (player.src !== url) player.src = url;
    player.play().catch(() => setStatus("Your browser blocked audio playback.", true));
  }
}
function syncPlayButtons() {
  $$("[data-preview]").forEach((b) => {
    const on = player.src === b.dataset.preview && !player.paused;
    b.classList.toggle("playing", on && b.classList.contains("play"));
    if (b.classList.contains("play-wide")) b.textContent = on ? "Pause preview" : "Play preview";
  });
}
["play", "pause", "ended"].forEach((ev) => player.addEventListener(ev, syncPlayButtons));

/* ---------------- featured (explore with no query) ---------------- */

async function showFeatured() {
  const el = $("#featured");
  if (state.mode === "genre") return showGenreBrowser();
  if (el.dataset.kind === "featured" && el.innerHTML) return;
  el.dataset.kind = "featured";
  el.innerHTML = '<p class="empty loading-dots">Loading</p>';
  try {
    const f = await getJSON(`${API}/featured?k=12`);
    if (state.query) return;
    el.innerHTML = `
      <section class="feature">
        <h2>Newest films in the catalog</h2>
        <p>${canHover ? "Hover a film for its poster and story, or click it" : "Tap a film for its poster and story, then match from it"} to find songs with the same mood.</p>
        <div class="grid">${f.movies.map((c) => cardHtml(c, 0, 1, [])).join("")}</div>
      </section>
      <section class="feature">
        <h2>Trending plays</h2>
        <p>The most played songs on Apple Music this week. ${canHover ? "Click a song" : "Tap a song, then match from it,"} to find films with the same mood.</p>
        <div class="grid">${f.songs.map((c) => cardHtml(c, 0, 1, [])).join("")}</div>
      </section>`;
    bindCards(el);
  } catch (e) {
    el.innerHTML = `<p class="empty status-error">${esc(e.message)}</p>`;
  }
}

async function showGenreBrowser() {
  const el = $("#featured");
  el.dataset.kind = "genres";
  el.innerHTML = '<p class="empty loading-dots">Loading</p>';
  try {
    state.genres = state.genres || await getJSON(`${API}/genres`);
    if (state.query || state.mode !== "genre") return;
    const film = state.genres.filter((g) => g.subtitle === "film");
    const music = state.genres.filter((g) => g.subtitle === "music");
    el.innerHTML = `
      <section class="feature">
        <h2>Film genres</h2>
        <p>Click a genre to see its films, and the songs that share its mood.</p>
        <div class="grid">${film.map((c) => cardHtml(c, 0, 1, [])).join("")}</div>
      </section>
      <section class="feature">
        <h2>Music genres</h2>
        <p>Click a genre to see its songs, and the films that share its mood.</p>
        <div class="grid">${music.map((c) => cardHtml(c, 0, 1, [])).join("")}</div>
      </section>`;
    bindCards(el);
  } catch (e) {
    el.innerHTML = `<p class="empty status-error">${esc(e.message)}</p>`;
  }
}

/* ---------------- start ---------------- */

setMode("song");
loadLanding();
route();
