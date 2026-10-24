"""
The shared mood vocabulary that both domains are described in.

Songs and movies have completely different raw signals (audio features
vs. plot text), so the trick that makes cross-domain matching work is
to translate both into the *same* small set of mood words first. Each
mood has:
  - an anchor sentence, used for zero-shot scoring of free text
    (movie overviews, user queries) against the mood
  - an audio profile, used to score songs from their librosa features
And genre labels from either domain add a prior towards certain moods.
"""

import re

# mood -> anchor sentence for zero-shot text scoring
MOOD_ANCHORS = {
    "energetic":   "an energetic, fast-moving, adrenaline-filled experience",
    "calm":        "a calm, peaceful, slow and relaxing mood",
    "melancholic": "a sad, melancholic story about loss and sorrow",
    "uplifting":   "an uplifting, hopeful, feel-good story that leaves you happy",
    "dark":        "a dark, bleak and disturbing atmosphere",
    "tense":       "a tense, suspenseful, nerve-wracking situation full of danger",
    "romantic":    "a romantic love story between two people falling in love",
    "dreamy":      "a dreamy, hazy, surreal and floating atmosphere",
    "aggressive":  "an aggressive, violent, furious confrontation",
    "playful":     "a playful, funny, lighthearted and silly romp",
    "nostalgic":   "a nostalgic look back at youth, memories and the past",
    "epic":        "an epic, grand-scale saga with sweeping stakes",
    "eerie":       "an eerie, creepy, haunting and supernatural presence",
    "warm":        "a warm, cozy, tender and comforting feeling of family and friendship",
    "gritty":      "a gritty, raw, street-level story of crime and survival",
    "rebellious":  "a rebellious story of defiance against authority and the system",
    "intimate":    "an intimate, quiet, personal and introspective portrait",
    "mysterious":  "a mysterious puzzle full of secrets and unanswered questions",
    "triumphant":  "a triumphant victory against all odds",
    "bittersweet": "a bittersweet mix of joy and sadness, love and regret",
    "chaotic":     "a chaotic, frantic, out-of-control whirlwind",
    "sensual":     "a sensual, seductive and passionate desire",
    "whimsical":   "a whimsical, magical, fairy-tale fantasy world",
    "brooding":    "a brooding, moody, dark and introspective loner",
}

MOODS = list(MOOD_ANCHORS)

# Audio feature order for the profiles below. All features are z-scored
# against the song catalog first; "major" is +1 for major key, -1 minor.
AUDIO_FEATURES = ["energy", "tempo", "brightness", "zcr", "major"]

# mood -> weights over AUDIO_FEATURES
AUDIO_PROFILES = {
    "energetic":   (1.0, 0.8, 0.3, 0.2, 0.2),
    "calm":        (-1.0, -0.7, -0.2, -0.3, 0.0),
    "melancholic": (-0.5, -0.4, -0.3, 0.0, -1.0),
    "uplifting":   (0.5, 0.3, 0.5, 0.0, 1.0),
    "dark":        (0.0, 0.0, -1.0, 0.0, -0.8),
    "tense":       (0.4, 0.3, 0.0, 0.3, -0.6),
    "romantic":    (-0.3, -0.3, 0.0, -0.3, 0.4),
    "dreamy":      (-0.6, -0.3, 0.1, -0.6, 0.0),
    "aggressive":  (1.0, 0.5, 0.5, 1.0, -0.4),
    "playful":     (0.4, 0.5, 0.6, 0.0, 0.7),
    "nostalgic":   (-0.2, -0.2, -0.3, -0.2, 0.2),
    "epic":        (0.7, 0.0, 0.0, -0.2, 0.0),
    "eerie":       (-0.4, -0.3, -0.4, 0.2, -0.7),
    "warm":        (-0.1, -0.2, -0.6, -0.4, 0.5),
    "gritty":      (0.4, 0.0, 0.2, 0.8, -0.3),
    "rebellious":  (0.7, 0.4, 0.4, 0.6, -0.2),
    "intimate":    (-0.8, -0.5, -0.3, -0.4, 0.0),
    "mysterious":  (-0.3, -0.2, -0.3, 0.0, -0.5),
    "triumphant":  (0.7, 0.2, 0.4, 0.0, 0.8),
    "bittersweet": (-0.3, -0.2, 0.0, 0.0, -0.3),
    "chaotic":     (0.8, 0.7, 0.5, 0.9, -0.3),
    "sensual":     (-0.2, -0.4, -0.3, -0.4, 0.0),
    "whimsical":   (0.0, 0.2, 0.6, -0.1, 0.6),
    "brooding":    (-0.4, -0.4, -0.6, 0.0, -0.8),
}

# genre keyword (regex, matched on word boundaries) -> moods it nudges
# towards. Covers both iTunes genre names and Wikidata film genre labels.
GENRE_PRIORS = {
    # music
    r"pop|k-pop":                   ["playful", "uplifting", "romantic"],
    r"hip[- ]hop|rap":              ["rebellious", "gritty", "energetic"],
    r"alternative":                 ["bittersweet", "brooding", "rebellious"],
    r"rock":                        ["rebellious", "energetic", "gritty"],
    r"country":                     ["nostalgic", "warm", "bittersweet"],
    r"electronic|dance|house":      ["energetic", "dreamy", "sensual"],
    r"r&b|soul":                    ["sensual", "romantic", "warm", "intimate"],
    r"jazz":                        ["warm", "intimate", "nostalgic", "sensual"],
    r"classical":                   ["epic", "calm", "melancholic", "dreamy"],
    r"latin|reggaeton|salsa":       ["sensual", "playful", "energetic"],
    r"singer/songwriter":           ["intimate", "bittersweet", "nostalgic"],
    r"metal":                       ["aggressive", "dark", "chaotic"],
    r"indie":                       ["dreamy", "bittersweet", "nostalgic"],
    r"soundtrack":                  ["epic", "dreamy"],
    r"christian|gospel":            ["uplifting", "triumphant"],
    r"reggae":                      ["warm", "calm", "playful"],
    r"blues":                       ["melancholic", "gritty", "warm"],
    r"new age|ambient|easy listening": ["calm", "dreamy", "intimate"],
    r"world|afrobeats?|african":    ["warm", "playful", "energetic"],
    r"j-pop|anime":                 ["playful", "energetic", "whimsical"],
    r"brazilian|mpb|samba|bossa nova|sertanejo|funk": ["sensual", "warm", "playful"],
    # film
    r"horror|slasher":              ["eerie", "dark", "tense"],
    r"thriller|suspense":           ["tense", "mysterious", "dark"],
    r"crime|gangster|noir|heist":   ["gritty", "dark", "brooding"],
    r"comedy":                      ["playful", "whimsical", "uplifting"],
    r"romance|romantic":            ["romantic", "warm", "bittersweet"],
    r"drama":                       ["bittersweet", "melancholic", "intimate"],
    r"action":                      ["energetic", "chaotic", "triumphant"],
    r"adventure|swashbuckler":      ["epic", "triumphant", "playful"],
    r"fantasy|fairy tale":          ["whimsical", "dreamy", "epic"],
    r"science fiction|sci-fi|cyberpunk": ["mysterious", "epic", "eerie"],
    r"war":                         ["epic", "dark", "chaotic"],
    r"western":                     ["gritty", "nostalgic", "epic"],
    r"musical|music-themed":        ["uplifting", "playful", "romantic"],
    r"animated|animation|children's|family": ["whimsical", "playful", "warm"],
    r"mystery|detective":           ["mysterious", "tense"],
    r"documentary":                 ["intimate"],
    r"historical|period":           ["nostalgic", "epic"],
    r"teen|coming-of-age":          ["nostalgic", "bittersweet", "rebellious"],
    r"psychological":               ["tense", "brooding", "eerie"],
    r"erotic":                      ["sensual"],
    r"superhero":                   ["triumphant", "energetic", "epic"],
    r"disaster":                    ["chaotic", "tense"],
    r"biographical":                ["intimate", "triumphant"],
    r"sports?":                     ["triumphant", "energetic"],
    r"satire|satirical|black comedy": ["playful", "dark"],
}

_GENRE_PATTERNS = [(re.compile(rf"\b(?:{k})\b", re.I), v) for k, v in GENRE_PRIORS.items()]


def genre_prior(genres):
    """Mood prior from a list of genre labels, normalised to sum to 1 so a
    film tagged with 14 genres doesn't get 14x the push of a film with one."""
    counts = dict.fromkeys(MOODS, 0.0)
    for g in genres:
        for pattern, moods in _GENRE_PATTERNS:
            if pattern.search(g):
                for m in moods:
                    counts[m] += 1.0
    total = sum(counts.values())
    if total:
        counts = {m: c / total for m, c in counts.items()}
    return [counts[m] for m in MOODS]
