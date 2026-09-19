from __future__ import annotations

import html
import re
from collections import Counter
from pathlib import Path

import jieba
import pandas as pd

POPULATION_PATH = Path("data/processed/absa_v1/production/absa_v1_production_v1/absa_v1_production_population_manifest_v1.csv")
MENTIONS_PATH = Path("data/processed/absa_v1/production/absa_v1_production_v1/absa_v1_production_mentions_v1.csv")
CONTEXTUAL_PATH = Path("data/processed/topic_discovery_v1/final_taxonomy_v1/final_contextual_or_excluded_topic_corpus_v1.csv")
OUTLIER_PATH = Path("data/processed/topic_discovery_v1/final_taxonomy_v1/final_outlier_corpus_v1.csv")
EXPECTED_DOCUMENTS = 7704
TOKENIZATION_CONFIG_VERSION = "wordcloud_v1_config_2"

GENERIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from", "had", "has", "have",
    "he", "her", "his", "i", "if", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "so",
    "about", "after", "all", "any", "can", "did", "do", "does", "into", "just", "more", "most", "no", "not",
    "out", "over", "some", "than", "that", "the", "their", "them", "then", "there", "they", "this", "to", "under",
    "up", "very", "was", "we", "were", "when", "with", "you", "your",
    "ada", "adalah", "aku", "akan", "anda", "atau", "dalam", "dan", "dari", "dengan", "dia", "di", "ini", "itu",
    "kami", "kita", "ke", "kerana", "lagi", "mereka", "ni", "pada", "pun", "saya", "sudah", "tak", "tidak", "untuk",
    "yang", "aja", "apa", "banget", "bisa", "buat", "dah", "gak", "jadi", "juga", "karena", "kamu", "nggak", "sama",
    "sangat", "sih", "udah", "ya", "了", "的", "是", "在", "和", "我", "你", "他", "她", "也", "有", "就", "都",
}
EVENT_STOPWORDS = {
    "klscm", "scklm", "standardcharteredmarathon", "standard", "chartered", "marathon", "kualalumpur", "kuala",
    "lumpur", "running", "runner", "runners", "run", "runs", "2019", "2023", "2024", "2025",
}
NEGATION_TOKENS = {"not", "no", "never", "tak", "tidak", "x", "bukan", "没", "不"}
CAPTION_CONTEXT_STOPWORDS = GENERIC_STOPWORDS
EVIDENCE_STOPWORDS = GENERIC_STOPWORDS - NEGATION_TOKENS
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
TOKEN_RE = re.compile(r"#?[A-Za-zÀ-ÖØ-öø-ÿ0-9_']+|[\u3400-\u9fff]+", re.UNICODE)
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]", re.UNICODE)


def load_wordcloud_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    from .cloud_bundle import load_cloud_bundle
    bundle = load_cloud_bundle("wordcloud")
    if bundle is not None:
        return bundle["documents"], bundle["mentions"]
    documents = pd.read_csv(POPULATION_PATH, dtype={"document_id": str})
    mentions = pd.read_csv(MENTIONS_PATH, dtype={"document_id": str})
    required = {"document_id", "original_caption", "event_year", "primary_language", "final_consolidated_topic_label", "taxonomy_status"}
    if required - set(documents.columns):
        raise ValueError(f"Word-cloud population fields missing: {sorted(required - set(documents.columns))}")
    if len(documents) != EXPECTED_DOCUMENTS or documents.document_id.nunique() != EXPECTED_DOCUMENTS:
        raise ValueError("Word-cloud population must contain exactly 7,704 unique documents")
    if set(documents.taxonomy_status) != {"substantive_absa_ready"}:
        raise ValueError("Word-cloud population contains a non-substantive document")
    ids = set(documents.document_id)
    contextual = set(pd.read_csv(CONTEXTUAL_PATH, usecols=["document_id"], dtype={"document_id": str}).document_id)
    outliers = set(pd.read_csv(OUTLIER_PATH, usecols=["document_id"], dtype={"document_id": str}).document_id)
    if ids & contextual or ids & outliers:
        raise ValueError("Excluded/contextual/outlier document entered the word-cloud population")
    if not set(mentions.document_id).issubset(ids):
        raise ValueError("Frozen mention references an unknown word-cloud document")
    return documents, mentions


def normalize_and_tokenize(text: str, source: str = "caption_context") -> tuple[list[str], list[str], list[str]]:
    if source not in {"caption_context", "evidence"}:
        raise ValueError(f"Unknown word-cloud text source: {source}")
    mode = source
    stopwords = CAPTION_CONTEXT_STOPWORDS if mode == "caption_context" else EVIDENCE_STOPWORDS
    cleaned = "" if pd.isna(text) else html.unescape(str(text or ""))
    hashtags = [value.lower().lstrip("#") for value in re.findall(r"#[\w\u3400-\u9fff]+", cleaned, re.UNICODE)]
    emojis = EMOJI_RE.findall(cleaned)
    cleaned = EMAIL_RE.sub(" ", URL_RE.sub(" ", re.sub(r"<[^>]+>", " ", cleaned)))
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(cleaned):
        raw = raw.lower().lstrip("#").strip("_' ")
        if not raw:
            continue
        parts = list(jieba.cut(raw, cut_all=False)) if re.fullmatch(r"[\u3400-\u9fff]+", raw) else [raw]
        for token in parts:
            token = token.strip().lower()
            if (not token or token in stopwords or token in EVENT_STOPWORDS
                    or re.fullmatch(r"(?:klscm|scklm)\d*", token)):
                continue
            is_chinese = bool(re.fullmatch(r"[\u3400-\u9fff]+", token))
            if token.isdigit() or (not is_chinese and len(token) < 2 and not (mode == "evidence" and token in NEGATION_TOKENS)):
                continue
            tokens.append(token)
    meaningful_hashtags = [tag for tag in hashtags if tag not in GENERIC_STOPWORDS and tag not in EVENT_STOPWORDS
                           and not re.fullmatch(r"(?:klscm|scklm)\d*", tag)]
    return tokens, meaningful_hashtags, emojis


def tokenize_documents(documents: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in documents.itertuples(index=False):
        tokens, hashtags, emojis = normalize_and_tokenize(row.original_caption)
        rows.append({"document_id": row.document_id, "tokens": tokens, "hashtags": hashtags, "emojis": emojis})
    return pd.DataFrame(rows)


def tokenize_evidence_mentions(mentions: pd.DataFrame) -> pd.DataFrame:
    required = {"mention_id", "document_id", "aspect", "sentiment", "evidence_text", "event_year", "primary_language", "final_topic_name"}
    if required - set(mentions.columns):
        raise ValueError(f"Frozen evidence fields missing: {sorted(required - set(mentions.columns))}")
    rows = []
    for row in mentions.itertuples(index=False):
        tokens, hashtags, emojis = normalize_and_tokenize(row.evidence_text, source="evidence")
        rows.append({
            "mention_id": row.mention_id, "document_id": str(row.document_id), "aspect": row.aspect,
            "sentiment": row.sentiment, "evidence_text": "" if pd.isna(row.evidence_text) else str(row.evidence_text), "event_year": int(row.event_year),
            "primary_language": row.primary_language, "final_topic_name": row.final_topic_name,
            "tokens": tokens, "hashtags": hashtags, "emojis": emojis,
        })
    return pd.DataFrame(rows)


def filter_document_ids(
    documents: pd.DataFrame,
    mentions: pd.DataFrame,
    year: str | int = "All",
    aspect: str = "All",
    sentiment: str = "All",
    topic: str = "All",
    language: str = "All",
) -> set[str]:
    selected = documents
    if year != "All": selected = selected[selected.event_year.eq(int(year))]
    if topic != "All": selected = selected[selected.final_consolidated_topic_label.eq(topic)]
    if language != "All": selected = selected[selected.primary_language.eq(language)]
    ids = set(selected.document_id)
    if aspect != "All" or sentiment != "All":
        matched = mentions
        if aspect != "All": matched = matched[matched.aspect.eq(aspect)]
        if sentiment != "All": matched = matched[matched.sentiment.eq(sentiment)]
        ids &= set(matched.document_id)
    return ids


def frequency_table(tokenized: pd.DataFrame, document_ids: set[str], minimum_document_frequency: int = 2) -> pd.DataFrame:
    raw: Counter[str] = Counter()
    document: Counter[str] = Counter()
    selected = tokenized[tokenized.document_id.isin(document_ids)]
    for tokens in selected.tokens:
        raw.update(tokens)
        document.update(set(tokens))
    rows = [{"token": token, "count": raw[token], "document_count": count}
            for token, count in document.items() if count >= minimum_document_frequency]
    return pd.DataFrame(rows, columns=["token", "count", "document_count"]).sort_values(
        ["document_count", "count", "token"], ascending=[False, False, True], ignore_index=True)


def filter_evidence_mentions(
    evidence: pd.DataFrame,
    year: str | int = "All",
    aspect: str = "All",
    sentiment: str = "All",
    topic: str = "All",
    language: str = "All",
) -> pd.DataFrame:
    selected = evidence
    if year != "All": selected = selected[selected.event_year.eq(int(year))]
    if aspect != "All": selected = selected[selected.aspect.eq(aspect)]
    if sentiment != "All": selected = selected[selected.sentiment.eq(sentiment)]
    if topic != "All": selected = selected[selected.final_topic_name.eq(topic)]
    if language != "All": selected = selected[selected.primary_language.eq(language)]
    return selected.copy()


def evidence_frequency_table(evidence: pd.DataFrame, minimum_document_frequency: int = 2) -> pd.DataFrame:
    raw: Counter[str] = Counter()
    evidence_count: Counter[str] = Counter()
    document_sets: dict[str, set[str]] = {}
    for row in evidence.itertuples(index=False):
        raw.update(row.tokens)
        for token in set(row.tokens):
            evidence_count[token] += 1
            document_sets.setdefault(token, set()).add(str(row.document_id))
    rows = [{"token": token, "count": raw[token], "evidence_count": evidence_count[token],
             "document_count": len(documents)} for token, documents in document_sets.items()
            if len(documents) >= minimum_document_frequency]
    return pd.DataFrame(rows, columns=["token", "count", "evidence_count", "document_count"]).sort_values(
        ["document_count", "evidence_count", "count", "token"], ascending=[False, False, False, True], ignore_index=True)


def normalize_evidence_expression(value: str) -> str:
    source = "" if pd.isna(value) else str(value or "")
    return re.sub(r"\s+", " ", source).strip().strip(".,;:!?¡¿…—–-()[]{}\"'“”‘’").casefold()


def evidence_expression_table(evidence: pd.DataFrame) -> pd.DataFrame:
    if evidence.empty:
        return pd.DataFrame(columns=["Expression", "Sentiment", "Aspect", "Evidence Occurrences", "Unique Documents"])
    working = evidence[["document_id", "evidence_text", "sentiment", "aspect"]].copy()
    working["normalized"] = working.evidence_text.map(normalize_evidence_expression)
    working = working[working.normalized.ne("")]
    rows = []
    for (normalized, sentiment, aspect), group in working.groupby(["normalized", "sentiment", "aspect"], sort=False):
        rows.append({"Expression": group.evidence_text.iloc[0].strip(), "Sentiment": sentiment.title(), "Aspect": aspect,
                     "Evidence Occurrences": len(group), "Unique Documents": group.document_id.nunique()})
    return pd.DataFrame(rows).sort_values(
        ["Unique Documents", "Evidence Occurrences", "Expression"], ascending=[False, False, True], ignore_index=True)


def top_items(tokenized: pd.DataFrame, document_ids: set[str], column: str, limit: int = 12) -> list[tuple[str, int]]:
    values: Counter[str] = Counter()
    for items in tokenized.loc[tokenized.document_id.isin(document_ids), column]:
        values.update(set(items))
    return values.most_common(limit)
