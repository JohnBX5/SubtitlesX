"""Format-independent subtitle model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass(slots=True)
class Cue:
    start: float  # seconds
    end: float  # seconds
    text: str  # normalized text; may contain <i>/<b>/<u> and a leading {\anN} tag

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class SubtitleDocument:
    cues: List[Cue]
    source_format: str = "srt"
    encoding: str = "utf-8"
    removed_ads: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def cue_count(self) -> int:
        return len(self.cues)

    def with_cues(self, cues: Sequence[Cue]) -> "SubtitleDocument":
        return SubtitleDocument(
            cues=list(cues),
            source_format=self.source_format,
            encoding=self.encoding,
            removed_ads=self.removed_ads,
            meta=dict(self.meta),
        )


@dataclass(slots=True)
class RenderedSubtitle:
    content: str
    format: str  # "srt" | "vtt"
    cue_count: int

    @property
    def content_bytes(self) -> bytes:
        return self.content.encode("utf-8")

    @property
    def media_type(self) -> str:
        return "text/vtt; charset=utf-8" if self.format == "vtt" else "text/plain; charset=utf-8"

