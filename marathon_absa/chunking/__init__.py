from __future__ import annotations

import math
import re
import pandas as pd
import tiktoken
from ..config import Settings
from ..text import stable_id

SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+", re.UNICODE)


class SentenceChunker:
    def __init__(self, settings: Settings):
        self.settings = settings
        try: self.encoding = tiktoken.encoding_for_model(settings.chat_model)
        except Exception:
            try: self.encoding = tiktoken.get_encoding("o200k_base")
            except Exception: self.encoding = None

    def tokens(self, text: str) -> int:
        return len(self.encoding.encode(text)) if self.encoding is not None else max(1, math.ceil(len(text.encode("utf-8")) / 3.5))

    def split_sentences(self, text: str) -> list[str]:
        return [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]

    def chunk(self, text: str) -> list[tuple[str, int, int, int]]:
        sentences = self.split_sentences(text)
        if not sentences: return []
        chunks, current, current_tokens, start = [], [], 0, 0
        for position, sentence in enumerate(sentences):
            count = self.tokens(sentence)
            if current and current_tokens + count > self.settings.chunk_max_tokens:
                joined = " ".join(current); chunks.append((joined, start, position - 1, self.tokens(joined)))
                current, current_tokens, start = [], 0, position
            if count > self.settings.chunk_max_tokens:
                if self.encoding is not None:
                    ids = self.encoding.encode(sentence); pieces = [self.encoding.decode(ids[o:o+self.settings.chunk_max_tokens]) for o in range(0, len(ids), self.settings.chunk_max_tokens)]
                else: pieces = [sentence[o:o+self.settings.chunk_max_tokens] for o in range(0, len(sentence), self.settings.chunk_max_tokens)]
                for piece in pieces: chunks.append((piece, position, position, self.tokens(piece)))
                start = position + 1
            else: current.append(sentence); current_tokens += count
        if current:
            joined = " ".join(current); chunks.append((joined, start, len(sentences) - 1, self.tokens(joined)))
        return chunks

    def build_units(self, documents: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for doc in documents[documents.processing_status == "ready"].itertuples(index=False):
            pieces = self.chunk(doc.normalized_text) if doc.source == "blog" or self.tokens(doc.normalized_text) > self.settings.chunk_max_tokens else [(doc.normalized_text, 0, 0, self.tokens(doc.normalized_text))]
            for ordinal, (text, start, end, count) in enumerate(pieces):
                rows.append({
                    "unit_id": stable_id(doc.document_id, ordinal), "document_id": doc.document_id, "chunk_index": ordinal,
                    "sentence_start": start, "sentence_end": end, "token_count": count, "text": text, "source": doc.source,
                    "event_year": doc.event_year or "Unknown", "detected_language": doc.detected_language,
                    "primary_language": doc.primary_language, "language_iso": doc.language_iso,
                    "detected_languages": doc.detected_languages, "is_mixed_language": doc.is_mixed_language,
                    "language_method": doc.language_method, "language_status": doc.language_status,
                    "possible_code_switching": doc.possible_code_switching, "url": doc.url,
                })
        return pd.DataFrame(rows)

