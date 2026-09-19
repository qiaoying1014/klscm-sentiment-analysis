from __future__ import annotations

import hashlib
import re
import unicodedata

import emoji
from ftfy import fix_text

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"(?<!\w)@[\w.]+", re.UNICODE)
HASHTAG_RE = re.compile(r"(?<!\w)#([^\s#]+)", re.UNICODE)
LANGUAGE_HASHTAG_RE = re.compile(r"#\S*", re.UNICODE)
EVENT_METADATA_RE = re.compile(r"\bKLSCM(?:19|20)?\d{2}\b|\bKLSCM\b", re.I | re.UNICODE)
SPACE_RE = re.compile(r"\s+")


def repair_and_normalize(value: object) -> str:
    text = fix_text("" if value is None else str(value))
    text = unicodedata.normalize("NFKC", text)
    return SPACE_RE.sub(" ", text).strip()


def extract_hashtags(text: str) -> list[str]:
    return [m.group(1).rstrip(".,!?:;)]}").casefold() for m in HASHTAG_RE.finditer(text)]


def extract_emojis(text: str) -> tuple[list[str], list[str]]:
    items = [item["emoji"] for item in emoji.emoji_list(text)]
    aliases = [emoji.demojize(item).strip(":") for item in items]
    return items, aliases


def linguistic_text(text: str) -> str:
    """Return natural-language content with social metadata removed."""
    value = URL_RE.sub(" ", text)
    value = MENTION_RE.sub(" ", value)
    value = LANGUAGE_HASHTAG_RE.sub(" ", value)
    value = EVENT_METADATA_RE.sub(" ", value)
    value = emoji.replace_emoji(value, replace=" ")
    value = SPACE_RE.sub(" ", value).strip()
    return value


def semantic_text(text: str) -> str:
    """Keep sentiment-bearing emoji aliases/hashtags but suppress URL and mention noise."""
    value = URL_RE.sub(" ", text)
    value = MENTION_RE.sub(" ", value)
    value = emoji.demojize(value, delimiters=(" ", " "))
    value = HASHTAG_RE.sub(lambda m: " hashtag_" + m.group(1).casefold() + " ", value)
    return SPACE_RE.sub(" ", value).strip()


def normalized_hash(text: str) -> str:
    canonical = re.sub(r"[^\w]+", " ", text.casefold(), flags=re.UNICODE).strip()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def stable_id(*parts: object) -> str:
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

