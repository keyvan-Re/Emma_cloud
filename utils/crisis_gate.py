"""
crisis_gate.py

Deterministic, rule-based safety gate for EMMA.

This MUST run before any memory retrieval, query classification, or LLM
prompt construction -- not after. For HIGH and IMMINENT levels, the response
returned here is fixed text, never LLM-generated: that removes any risk of
the model improvising, minimizing, or engaging with method-level detail in
a crisis moment.

This is an interim safeguard while your trained severity classifier is
built. It is deliberately tuned to over-trigger (false positives are the
acceptable failure direction here, false negatives are not).

KNOWN LIMITATIONS -- read before relying on this in production:
- English-only. It will not fire on crisis language in Persian or any other
  language. If EMMA has non-English-speaking users, treat closing this gap
  as equally urgent as the ML classifier, not a later nice-to-have.
- Exact-substring matching. It will miss rephrasings, typos, and indirect
  expressions of risk ("I don't think I'll be around much longer"), and it
  will over-fire on some benign contexts (e.g. quoted lyrics, hypotheticals).
  That tradeoff is intentional and correct for a stopgap.
- It does NOT attempt to detect requests for specific means/methods. A
  reliable means-detector needs clinical input and careful validation, not
  a hand-typed keyword list -- that belongs in the classifier project, not
  here.
"""

from enum import Enum
import re
import json
import os
import datetime


class CrisisLevel(str, Enum):
    NONE = "none"
    CONCERN = "concern"
    HIGH = "high"
    IMMINENT = "imminent"


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s']", " ", text)  # strip punctuation, keep contractions
    text = re.sub(r"\s+", " ", text)
    # Collapse common contraction variants onto one form so a single pattern
    # entry catches "i am", "i'm", and "im" alike, instead of needing every
    # spelling variant hand-listed. This is the fix for the "i am tired of
    # life" miss -- it previously only matched "im"/"i'm", not "i am".
    text = re.sub(r"\bi am\b", "im", text)
    text = re.sub(r"\bi'm\b", "im", text)
    text = re.sub(r"\bdo not\b", "dont", text)
    text = re.sub(r"\bdon't\b", "dont", text)
    text = re.sub(r"\bi've\b", "ive", text)
    text = re.sub(r"\bi have\b", "ive", text)
    return text


# NOTE: both contracted and uncontracted forms are listed because
# normalization does not expand/strip apostrophes consistently across
# copy-paste sources -- keep both rather than relying on one regex trick.
IMMINENT_PATTERNS = [
    "i want to hurt myself right now",
    "im going to kill myself right now",
    "i have the means right here",
    "ive made up my mind",
    "im doing it tonight",
    "want to end it all tonight",
    "going to end it all tonight",
]

HIGH_RISK_PATTERNS = [
    "i want to kill myself",
    "i want to end my life",
    "i want to hurt myself",
    "dont want to be alive anymore",
    "dont want to live anymore",
    "i want to die",
    "want to end it all",
    "better off dead",
    "no reason to keep living",
    "no point in living",
]

CONCERN_PATTERNS = [
    "i wish i didnt exist",
    "life isnt worth it",
    "life isn't worth it",
    "cant take it anymore",
    "can't take it anymore",
    "its all over",
    "it's all over",
    "whats the point of going on",
    "what's the point of going on",
    "im tired of life",
    "tired of life",
    "dont want to be here anymore",
]


def detect_crisis_level(text: str) -> CrisisLevel:
    normalized = _normalize(text)

    if any(p in normalized for p in IMMINENT_PATTERNS):
        return CrisisLevel.IMMINENT
    if any(p in normalized for p in HIGH_RISK_PATTERNS):
        return CrisisLevel.HIGH
    if any(p in normalized for p in CONCERN_PATTERNS):
        return CrisisLevel.CONCERN
    return CrisisLevel.NONE


# ---------------------------------------------------------------------------
# Response construction. Fixed text only for HIGH/IMMINENT -- never routed
# through the LLM, so there is no personalization/memory path to bypass.
# ---------------------------------------------------------------------------

GENERAL_RESOURCES = (
    "If you are in immediate danger, please contact your local emergency number now.\n\n"
    "You can find a crisis line for your country here: https://findahelpline.com\n"
    "(US: call or text 988. Canada: call or text 988. UK: call 111.)"
)

# Optional: map a user's registered 'residence' string to a specific hotline
# line here once you've validated numbers for your actual user base. Until
# then, GENERAL_RESOURCES with the findahelpline.com directory is the safer
# default -- it avoids the risk of surfacing a wrong or outdated local number.
COUNTRY_HOTLINES: dict[str, str] = {
    # "iran": "...",
    # "canada": "Call or text 988",
}


def _resource_block(user_profile: dict | None) -> str:
    residence = (user_profile or {}).get("residence", "").strip().lower()
    for key, line in COUNTRY_HOTLINES.items():
        if key in residence:
            return f"{line}\n\nYou can also find other options here: https://findahelpline.com"
    return GENERAL_RESOURCES


def build_crisis_response(level: CrisisLevel, user_profile: dict | None = None) -> str:
    resources = _resource_block(user_profile)

    if level == CrisisLevel.IMMINENT:
        return (
            "I'm really concerned about what you just shared, and I want to make sure "
            "you're safe right now. I'm not able to help with this myself -- please "
            "reach out to a crisis line or emergency services immediately.\n\n"
            f"{resources}\n\n"
            "Please don't wait -- contact one of these now, or someone you trust who "
            "can be with you."
        )
    if level == CrisisLevel.HIGH:
        return (
            "Thank you for telling me this. It sounds like you're going through "
            "something very painful, and I'm taking it seriously. I'm not able to "
            "provide the kind of support you need for this on my own -- please reach "
            "out to a crisis line so you can talk to someone trained to help right now.\n\n"
            f"{resources}"
        )
    return ""  # CONCERN and NONE don't use scripted text -- see app.py integration


def log_crisis_event(log_dir: str, user_name: str | None, level: CrisisLevel) -> None:
    """
    Appends a minimal record for human follow-up review. Deliberately does
    NOT log the raw message text by default -- crisis content is sensitive
    and you likely don't want it sitting in a plain log file. If you decide
    you need the raw text for clinical follow-up, store it encrypted and
    access-controlled, not in this log.
    """
    os.makedirs(log_dir, exist_ok=True)
    entry = {
        "user_name": user_name,
        "level": level.value,
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
    }
    with open(os.path.join(log_dir, "crisis_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
