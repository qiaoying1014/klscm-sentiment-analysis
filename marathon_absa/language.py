from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd
from lingua import Language, LanguageDetectorBuilder

from .config import Settings

LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+", re.UNICODE)

# OpenLID uses ISO 639-3 codes. Build readable names from Lingua and override
# model-specific codes and names that differ from Lingua's terminology.
LABELS = {
    item.iso_code_639_3.name.lower(): item.name.replace("_", " ").title()
    for item in Language.all()
}
LABELS.update({
    "eng": "English", "zsm": "Malay", "msa": "Malay", "ind": "Indonesian",
    "cmn": "Chinese", "yue": "Chinese", "tam": "Tamil", "fil": "Tagalog",
    "deu": "German", "vie": "Vietnamese", "fra": "French", "jpn": "Japanese",
    "por": "Portuguese", "spa": "Spanish", "tha": "Thai", "kor": "Korean",
    "zxx": "No linguistic content",
})

CORE_LANGUAGES = {
    "English", "Malay", "Indonesian", "Chinese", "Tamil", "Spanish", "French",
    "German", "Portuguese", "Japanese", "Korean", "Thai", "Vietnamese", "Tagalog",
}

MALAY_SHORTFORMS = {
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "sbb": "sebab",
    "dh": "sudah", "dah": "sudah", "nk": "nak", "sy": "saya",
    "saya": "saya", "mcm": "macam", "blh": "boleh", "jgn": "jangan",
    "org": "orang", "korang": "kamu semua", "je": "sahaja", "jer": "sahaja",
    "tak": "tidak", "tk": "tidak", "xde": "tidak ada", "xnak": "tidak nak",
    "ni": "ini", "tu": "itu", "kat": "dekat", "mmg": "memang",
    "tgh": "tengah", "kwn": "kawan", "skrg": "sekarang", "nnti": "nanti",
    "dkt": "dekat", "smua": "semua", "lg": "lagi", "la": "lah",
}
TOKEN_RE = re.compile(r"(?<!\w)([A-Za-z]+)(?!\w)", re.UNICODE)


def malay_shortform_hits(text: str) -> int:
    tokens = [match.group(1).casefold() for match in TOKEN_RE.finditer(text)]
    return sum(token in MALAY_SHORTFORMS for token in tokens)


def normalize_malay_shortforms(text: str) -> str:
    """Expand colloquial Malay only for language identification."""
    if malay_shortform_hits(text) < 2:
        return text
    return TOKEN_RE.sub(lambda match: MALAY_SHORTFORMS.get(match.group(1).casefold(), match.group(0)), text)


def _label(raw):
    value = raw.removeprefix("__label__")
    parts = value.split("_")
    code = parts[0]
    script = parts[1] if len(parts) > 1 else ""
    return LABELS.get(code, "Unsupported/uncertain"), f"{code}_{script}".rstrip("_")


class HybridLanguageService:
    def __init__(self, settings: Settings):
        self.settings = settings
        languages = [getattr(Language, name) for name in settings.languages if hasattr(Language, name)]
        self.lingua = LanguageDetectorBuilder.from_languages(*languages).with_preloaded_language_models().build()
        self.accepted_languages = {language.name.replace("_", " ").title() for language in languages} | CORE_LANGUAGES
        self.iso_by_name = {
            language.name.replace("_", " ").title(): language.iso_code_639_3.name.lower()
            for language in languages
        }
        self.openlid = None
        if settings.openlid_path.exists():
            import fasttext
            fasttext.FastText.eprint = lambda *_: None
            self.openlid = fasttext.load_model(str(settings.openlid_path))

    def _lingua(self, text):
        values = self.lingua.compute_language_confidence_values(text)
        return (values[0].language.name.replace("_", " ").title(), float(values[0].value)) if values else ("undetermined", 0.0)

    def _predict_many(self, texts, k=10):
        if not texts:
            return []
        cleaned = [item.replace("\n", " ") for item in texts]
        # fasttext-wheel's multilinePredict repeats the highest probability for
        # every label with this model. The low-level predictor returns the
        # correct probability attached to each label and also avoids its NumPy
        # 2 incompatibility in the single-string wrapper.
        if hasattr(self.openlid, "f"):
            output = []
            for text in cleaned:
                predictions = self.openlid.f.predict(text + "\n", k, 0.0, "")
                output.append([(*_label(label), float(score)) for score, label in predictions])
            return output
        output = []
        for text in cleaned:
            labels, scores = self.openlid.predict(text, k=k)
            output.append([(*_label(label), float(score)) for label, score in zip(labels, scores)])
        return output

    def _spans(self, text):
        sentences = [item.strip() for item in SENTENCE_RE.split(text) if len(LETTER_RE.findall(item)) >= self.settings.min_language_chars]
        spans = sentences[:]
        words = text.split()
        step = max(1, self.settings.language_window_words - self.settings.language_window_overlap)
        if len(words) > self.settings.language_window_words:
            spans.extend(
                window for i in range(0, len(words), step)
                if len(LETTER_RE.findall(window := " ".join(words[i:i + self.settings.language_window_words]))) >= self.settings.min_language_chars
            )
        return spans or [text]

    def _base(self, status, reason):
        return {
            "primary_language": status, "language_iso": "und", "detected_language": status,
            "detected_languages": [], "language_confidence": 0.0, "second_language": "",
            "second_language_confidence": 0.0, "language_margin": 0.0,
            "language_status": status, "is_mixed_language": False,
            "possible_code_switching": False, "language_method": "rule",
            "language_review_reason": reason, "language_adjudication_status": "not_required",
        }

    def _accepted_candidate(self, candidate, lingua_name, chars, component=False):
        name, _, score = candidate
        if name in {"No linguistic content", "Unsupported/uncertain"} or name not in self.accepted_languages:
            return False
        if name in CORE_LANGUAGES:
            threshold = 0.20 if component else 0.10
            return score >= threshold or name == lingua_name
        # Additional major languages need stronger, corroborated evidence.
        threshold = 0.65 if component else 0.60
        return chars >= 20 and score >= threshold and name == lingua_name

    def _compose(self, text, lingua_name, lingua_score, top, span_predictions, malay_hint=False):
        chars_total = len(LETTER_RE.findall(text))
        accepted_top = [item for item in top if self._accepted_candidate(item, lingua_name, chars_total)]
        if malay_hint:
            primary, iso, confidence = "Malay", "zsm_Latn", max(lingua_score, 0.85)
            second, second_score = "", 0.0
        elif accepted_top:
            primary, iso, confidence = accepted_top[0]
            second, _, second_score = accepted_top[1] if len(accepted_top) > 1 else ("", "", 0.0)
        elif lingua_name in self.accepted_languages and lingua_score >= 0.70:
            primary, iso, confidence = lingua_name, self.iso_by_name.get(lingua_name, "und"), lingua_score
            second, second_score = "", 0.0
        else:
            primary, iso, confidence = "undetermined", "und", max([item[2] for item in top], default=0.0)
            second, second_score = "", 0.0

        coverage = defaultdict(int)
        weighted = defaultdict(float)
        for span, predictions in zip(self._spans(text), span_predictions):
            chars = len(LETTER_RE.findall(span))
            selected = next((item for item in predictions if self._accepted_candidate(item, lingua_name, chars, component=True)), None)
            if not selected:
                continue
            name, span_iso, score = selected
            if malay_hint and name == "Indonesian":
                name, span_iso = "Malay", "zsm_Latn"
            coverage[(name, span_iso)] += chars
            weighted[(name, span_iso)] += score * chars

        accepted_chars = sum(coverage.values())
        components = sorted((
            {
                "language": name, "iso_script": span_iso,
                "coverage": round(chars / accepted_chars, 4) if accepted_chars else 0.0,
                "confidence": round(weighted[(name, span_iso)] / chars, 4), "characters": chars,
            }
            for (name, span_iso), chars in coverage.items()
        ), key=lambda item: item["coverage"], reverse=True)
        if primary == "undetermined" and components:
            primary = components[0]["language"]
            iso = components[0]["iso_script"]
            confidence = components[0]["confidence"]

        substantial = [item for item in components if item["coverage"] >= self.settings.mixed_min_coverage and item["characters"] >= self.settings.mixed_min_chars and item["confidence"] >= 0.35]
        mixed = len(substantial) > 1
        margin = confidence - second_score
        disagreement = lingua_name != primary
        malay_indonesian = {lingua_name, primary} == {"Malay", "Indonesian"}
        low = confidence < self.settings.language_confidence_threshold or margin < self.settings.language_margin_threshold

        status, adjudication, reason = "ok", "not_required", ""
        if primary == "undetermined":
            status, adjudication, reason = "undetermined", "pending", "no_credible_supported_language"
        elif malay_indonesian:
            primary, iso = "Malay/Indonesian uncertain", "ms-id"
            status, adjudication, reason = "review_required", "pending", "malay_indonesian_disagreement"
        elif mixed:
            status, adjudication, reason = "mixed", "pending", "multiple_substantial_language_spans"
        elif low:
            status, adjudication, reason = "low_confidence", "pending", "openlid_low_confidence"
        elif disagreement:
            status, adjudication, reason = "review_required", "pending", "openlid_lingua_disagreement"

        return {
            "primary_language": primary, "language_iso": iso, "detected_language": primary,
            "detected_languages": components, "language_confidence": confidence,
            "second_language": second, "second_language_confidence": second_score,
            "language_margin": margin, "language_status": status,
            "is_mixed_language": mixed, "possible_code_switching": mixed,
            "language_method": "openlid_masked" if mixed else "openlid_constrained",
            "language_review_reason": reason, "language_adjudication_status": adjudication,
        }

    def detect(self, normalized, linguistic):
        if not normalized.strip():
            return self._base("no_text", "empty_normalized_text")
        malay_hint = malay_shortform_hits(linguistic or "") >= 2
        detection_text = normalize_malay_shortforms(linguistic or "")
        if len(LETTER_RE.findall(detection_text)) < self.settings.min_language_chars:
            return self._base("insufficient_text", "fewer_than_minimum_linguistic_characters")
        lingua_name, lingua_score = self._lingua(detection_text)
        if self.openlid is None:
            result = self._base(lingua_name, "openlid_model_missing")
            result.update({"primary_language": lingua_name, "detected_language": lingua_name, "language_confidence": lingua_score, "language_status": "review_required", "language_method": "lingua_fallback", "language_adjudication_status": "pending"})
            return result
        spans = self._spans(detection_text)
        return self._compose(detection_text, lingua_name, lingua_score, self._predict_many([detection_text], 10)[0], self._predict_many(spans, 10), malay_hint)

    def apply(self, documents: pd.DataFrame) -> pd.DataFrame:
        rows = [None] * len(documents)
        local = []
        review_languages = CORE_LANGUAGES - {"English", "Chinese", "Tamil"}
        for pos, row in enumerate(documents.itertuples(index=False)):
            normalized = str(row.normalized_text)
            malay_hint = malay_shortform_hits(str(row.linguistic_text)) >= 2
            detection_text = normalize_malay_shortforms(str(row.linguistic_text))
            if not normalized.strip():
                rows[pos] = self._base("no_text", "empty_normalized_text")
                continue
            if len(LETTER_RE.findall(detection_text)) < self.settings.min_language_chars:
                rows[pos] = self._base("insufficient_text", "fewer_than_minimum_linguistic_characters")
                continue
            lingua_name, lingua_score = self._lingua(detection_text)
            if self.openlid is None:
                result = self._base(lingua_name, "openlid_model_missing")
                result.update({"primary_language": lingua_name, "detected_language": lingua_name, "language_confidence": lingua_score, "language_status": "review_required", "language_method": "lingua_fallback", "language_adjudication_status": "pending"})
                rows[pos] = result
                continue
            multi_sentence = len([item for item in SENTENCE_RE.split(detection_text) if item.strip()]) > 1
            needs_openlid = lingua_score < self.settings.language_confidence_threshold or lingua_name in review_languages or multi_sentence
            if needs_openlid:
                local.append((pos, detection_text, lingua_name, lingua_score, multi_sentence, malay_hint))
            else:
                iso = self.iso_by_name.get(lingua_name, "und")
                rows[pos] = {
                    "primary_language": lingua_name, "language_iso": iso,
                    "detected_language": lingua_name,
                    "detected_languages": [{"language": lingua_name, "iso_script": iso, "coverage": 1.0, "confidence": round(lingua_score, 4), "characters": len(LETTER_RE.findall(detection_text))}],
                    "language_confidence": lingua_score, "second_language": "",
                    "second_language_confidence": 0.0, "language_margin": lingua_score,
                    "language_status": "ok", "is_mixed_language": False,
                    "possible_code_switching": False, "language_method": "lingua_high_confidence",
                    "language_review_reason": "", "language_adjudication_status": "not_required",
                }
        if local:
            texts = [item[1] for item in local]
            top_batches = self._predict_many(texts, 10)
            all_spans = []
            span_ranges = {}
            for idx, (_, text, _, _, _, _) in enumerate(local):
                start = len(all_spans)
                all_spans.extend(self._spans(text))
                span_ranges[idx] = (start, len(all_spans))
            span_batches = self._predict_many(all_spans, 10)
            for idx, ((pos, text, lingua_name, lingua_score, _, malay_hint), top) in enumerate(zip(local, top_batches)):
                predictions = span_batches[slice(*span_ranges[idx])]
                rows[pos] = self._compose(text, lingua_name, lingua_score, top, predictions, malay_hint)
        detected = pd.DataFrame(rows)
        documents = documents.drop(columns=list(documents.columns.intersection(detected.columns)))
        return pd.concat([documents.reset_index(drop=True), detected], axis=1)


LinguaService = HybridLanguageService