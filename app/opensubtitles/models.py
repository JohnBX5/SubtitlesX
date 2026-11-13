"""OpenSubtitles domain models (plain dataclasses, JSON round-trippable)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class TitleQuery:
    imdb_id: Optional[str] = None
    query: Optional[str] = None
    year: Optional[int] = None
    media_type: Optional[str] = None  # movie | episode | all
    season_number: Optional[int] = None
    episode_number: Optional[int] = None

    def cache_parts(self) -> Dict[str, Any]:
        return {
            "imdb": normalize_imdb_id(self.imdb_id),
            "q": (self.query or "").strip().lower() or None,
            "y": self.year,
            "t": self.media_type if self.media_type in {"movie", "episode"} else None,
            "s": self.season_number,
            "e": self.episode_number,
        }


@dataclass(slots=True)
class MovieInfo:
    imdb_id: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None


@dataclass(slots=True)
class SubtitleFile:
    file_id: int
    file_name: str = ""
    cd_number: int = 1


@dataclass(slots=True)
class SubtitleCandidate:
    id: str
    subtitle_id: str
    language: str
    download_count: int = 0
    new_download_count: int = 0
    hearing_impaired: bool = False
    hd: bool = False
    fps: Optional[float] = None
    votes: int = 0
    ratings: float = 0.0
    from_trusted: bool = False
    foreign_parts_only: bool = False
    ai_translated: bool = False
    machine_translated: bool = False
    release: Optional[str] = None
    uploader_rank: Optional[str] = None
    files: List[SubtitleFile] = field(default_factory=list)
    ranking_score: float = 0.0
    ranking_reason: str = ""

    @property
    def primary_file(self) -> Optional[SubtitleFile]:
        return self.files[0] if self.files else None

    @property
    def is_single_file(self) -> bool:
        return len(self.files) == 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubtitleCandidate":
        files = [SubtitleFile(**f) for f in data.get("files", [])]
        payload = {k: v for k, v in data.items() if k != "files" and k in cls.__dataclass_fields__}
        return cls(files=files, **payload)


@dataclass(slots=True)
class LanguageInfo:
    language_code: str
    language_name: str
    available_files: int


def normalize_imdb_id(imdb_id: Optional[str]) -> Optional[int]:
    """``tt0111161`` / ``0111161`` / ``111161`` -> ``111161``; invalid -> None."""
    if not imdb_id:
        return None
    cleaned = str(imdb_id).strip().lower()
    if cleaned.startswith("tt"):
        cleaned = cleaned[2:]
    return int(cleaned) if cleaned.isdigit() and int(cleaned) > 0 else None


def format_imdb_id(value: Any) -> Optional[str]:
    try:
        num = int(str(value).lower().lstrip("t"))
    except (TypeError, ValueError):
        return None
    return f"tt{num:07d}" if num > 0 else None

