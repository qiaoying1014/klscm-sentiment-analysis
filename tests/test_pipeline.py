from dataclasses import replace

import pandas as pd

from marathon_absa.chunking import SentenceChunker
from marathon_absa.config import SETTINGS
from marathon_absa.text import extract_emojis, extract_hashtags, linguistic_text, normalized_hash


def test_hashtags_and_emoji_are_preserved_as_features():
    text = "Heat was brutal 🥵 but volunteers were great 👏 #KLSCM2024"
    assert extract_hashtags(text) == ["klscm2024"]
    icons, aliases = extract_emojis(text)
    assert icons == ["🥵", "👏"]
    assert "hot_face" in aliases


def test_linguistic_text_removes_hashtags_and_other_social_noise():
    value = linguistic_text("@runner Great race! https://x.test #KLSCM_2024 🏃")
    assert "runner" not in value
    assert "http" not in value
    assert "KLSCM" not in value
    assert value == "Great race!"


def test_hashtag_only_text_has_no_linguistic_content():
    assert linguistic_text("#KLSCM2024 #RunKualaLumpur 🏃") == ""


def test_normalized_hash_ignores_case_and_punctuation():
    assert normalized_hash("Great race!") == normalized_hash("great race")


def test_chunker_preserves_all_sentences():
    settings = replace(SETTINGS, chunk_max_tokens=12, chunk_target_tokens=10)
    chunker = SentenceChunker(settings)
    text = "The route was beautiful. The weather was hot. Volunteers were excellent."
    chunks = chunker.chunk(text)
    reconstructed = " ".join(item[0] for item in chunks)
    assert reconstructed == text
    assert all(item[3] <= 12 for item in chunks)



def test_linguistic_text_removes_chained_and_malformed_hashtag_tokens():
    value = linguistic_text("Great race word#TagOne #TagTwo#TagThree #")
    assert value == "Great race word"
    assert "#" not in value

def test_linguistic_text_removes_klscm_event_identifier():
    value = linguistic_text("KLSCM2019 Kids Dash done! Great weather this morning")
    assert value == "Kids Dash done! Great weather this morning"