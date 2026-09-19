from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    instagram_path: Path = ROOT / "Instagram" / "instagram_cleanded.csv"
    blog_path: Path = ROOT / "Online Review Blog" / "raw-data.json"
    output_dir: Path = ROOT / "data" / "processed"
    cache_dir: Path = ROOT / "data" / "cache"
    model_dir: Path = ROOT / "models" / "openlid-v3"
    openlid_repo: str = "HPLT/OpenLID-v3"
    openlid_revision: str = "6b9560483e17e42f48d86cebf22b4b58dffeaa70"
    openlid_filename: str = "openlid-v3.bin"
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.6-luna")
    stronger_model: str = os.getenv("OPENAI_STRONGER_MODEL", "gpt-5.6-terra")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    min_language_chars: int = 8
    language_confidence_threshold: float = 0.80
    language_margin_threshold: float = 0.15
    mixed_min_coverage: float = 0.20
    mixed_min_chars: int = 10
    language_window_words: int = 16
    language_window_overlap: int = 5
    chunk_target_tokens: int = 400
    chunk_max_tokens: int = 500
    random_seed: int = 42
    languages: tuple[str, ...] = field(default=(
        "ENGLISH", "MALAY", "INDONESIAN", "CHINESE", "TAMIL",
        "SPANISH", "FRENCH", "GERMAN", "PORTUGUESE", "JAPANESE",
        "KOREAN", "THAI", "VIETNAMESE", "TAGALOG", "ARABIC", "HINDI",
        "DUTCH", "ITALIAN", "RUSSIAN", "TURKISH", "POLISH", "UKRAINIAN",
        "DANISH", "SWEDISH", "FINNISH", "BOKMAL", "NYNORSK", "CZECH",
        "GREEK", "HEBREW", "BENGALI", "SWAHILI", "PERSIAN", "URDU",
    ))

    @property
    def openlid_path(self) -> Path:
        return self.model_dir / self.openlid_filename

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()

