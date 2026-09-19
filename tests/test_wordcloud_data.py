from pathlib import Path

import pandas as pd

from marathon_absa.wordcloud_data import (
    EVIDENCE_STOPWORDS,
    EVENT_STOPWORDS,
    evidence_expression_table,
    evidence_frequency_table,
    filter_document_ids,
    filter_evidence_mentions,
    frequency_table,
    load_wordcloud_sources,
    normalize_and_tokenize,
    tokenize_documents,
    tokenize_evidence_mentions,
)


def test_wordcloud_population_is_exact_frozen_substantive_population():
    documents, mentions = load_wordcloud_sources()
    assert len(documents) == documents.document_id.nunique() == 7704
    assert set(documents.taxonomy_status) == {"substantive_absa_ready"}
    contextual = set(pd.read_csv("data/processed/topic_discovery_v1/final_taxonomy_v1/final_contextual_or_excluded_topic_corpus_v1.csv", dtype={"document_id": str}).document_id)
    outliers = set(pd.read_csv("data/processed/topic_discovery_v1/final_taxonomy_v1/final_outlier_corpus_v1.csv", dtype={"document_id": str}).document_id)
    assert set(documents.document_id).isdisjoint(contextual | outliers)
    assert set(mentions.document_id).issubset(set(documents.document_id))


def test_tokenization_removes_urls_and_event_noise_but_keeps_meaningful_hashtags():
    tokens, hashtags, _ = normalize_and_tokenize(
        "https://example.com test@example.com #KLSCM2019 #MentalKoyak training route crowd finish pain volunteer weather"
    )
    assert "https" not in tokens and "example" not in tokens
    assert "klscm2019" not in tokens and "klscm2019" not in hashtags
    assert "mentalkoyak" in tokens and "mentalkoyak" in hashtags
    assert {"training", "route", "crowd", "finish", "pain", "volunteer", "weather"}.issubset(tokens)
    assert {"klscm", "marathon", "running", "runner", "run"}.issubset(EVENT_STOPWORDS)


def test_aspect_and_sentiment_filters_use_the_same_frozen_mention_row():
    documents, mentions = load_wordcloud_sources()
    ids = filter_document_ids(documents, mentions, aspect="route_course", sentiment="negative")
    exact = set(mentions.loc[mentions.aspect.eq("route_course") & mentions.sentiment.eq("negative"), "document_id"])
    assert ids == exact


def test_evidence_sentiment_filters_use_only_exact_frozen_evidence_rows():
    _, mentions = load_wordcloud_sources()
    evidence = tokenize_evidence_mentions(mentions)
    negative = filter_evidence_mentions(evidence, sentiment="negative")
    positive = filter_evidence_mentions(evidence, sentiment="positive")
    assert set(negative.sentiment) == {"negative"}
    assert set(positive.sentiment) == {"positive"}
    assert negative.evidence_text.tolist() == evidence.loc[evidence.sentiment.eq("negative"), "evidence_text"].tolist()
    exact = filter_evidence_mentions(evidence, aspect="physical_experience", sentiment="negative")
    assert set(zip(exact.aspect, exact.sentiment)) == {("physical_experience", "negative")}


def test_evidence_tokenization_preserves_negation_and_multilingual_colloquial_text():
    tokens, _, _ = normalize_and_tokenize("not enough training tak tidak x bukan 没破4 不好 Mental koyak Serammmm berjaya", source="evidence")
    assert {"not", "tak", "tidak", "x", "bukan", "没", "不好", "mental", "koyak", "serammmm", "berjaya"}.issubset(tokens)
    assert {"not", "no", "tak", "tidak", "x", "bukan", "没", "不"}.isdisjoint(EVIDENCE_STOPWORDS)


def test_evidence_frequencies_and_expressions_are_span_bounded():
    evidence = pd.DataFrame([
        {"document_id": "a", "tokens": ["not", "enough", "training"], "evidence_text": "Not enough training", "sentiment": "negative", "aspect": "training_preparation_pacing"},
        {"document_id": "a", "tokens": ["not", "ready"], "evidence_text": "not ready", "sentiment": "negative", "aspect": "training_preparation_pacing"},
        {"document_id": "b", "tokens": ["not", "enough", "training"], "evidence_text": " not  enough training! ", "sentiment": "negative", "aspect": "training_preparation_pacing"},
    ])
    frequencies = evidence_frequency_table(evidence)
    training = frequencies.loc[frequencies.token.eq("training")].iloc[0]
    assert training.document_count == 2 and training.evidence_count == 2 and training["count"] == 2
    assert "friends" not in set(frequencies.token)
    expressions = evidence_expression_table(evidence)
    row = expressions.loc[expressions.Expression.str.contains("enough training", case=False)].iloc[0]
    assert row["Evidence Occurrences"] == 2 and row["Unique Documents"] == 2


def test_document_frequency_counts_each_document_once():
    tokenized = pd.DataFrame({"document_id": ["a", "b"], "tokens": [["finish", "finish"], ["finish"]]})
    table = frequency_table(tokenized, {"a", "b"})
    row = table.loc[table.token.eq("finish")].iloc[0]
    assert row.document_count == 2 and row["count"] == 3


def test_wordcloud_module_contains_no_online_or_model_calls():
    source = Path("marathon_absa/wordcloud_data.py").read_text(encoding="utf-8").lower()
    for forbidden in ["openai", "huggingface", "transformers", "bertopic", "requests.", "urlopen"]:
        assert forbidden not in source
