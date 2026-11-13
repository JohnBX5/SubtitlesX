"""Deterministic metadata ranking of subtitle candidates.

Metadata only predicts quality; it cannot guarantee timing for a given rip.
That is why the sync engine re-ranks the top candidates acoustically.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Optional, Set, Tuple

from app.opensubtitles.models import SubtitleCandidate

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {"the", "and", "movie", "sub", "subs", "srt", "www", "com", "org", "mp4", "mkv", "m3u8", "index"}
# Release-type tokens that strongly predict timing compatibility when they match.
_SOURCE_TOKENS = {
    "bluray", "bdrip", "brrip", "web", "webrip", "webdl", "dl", "hdtv", "dvdrip", "hdrip",
    "remux", "amzn", "nf", "dsnp", "hmax", "atvp", "hulu", "1080p", "720p", "2160p", "480p",
    "x264", "x265", "hevc", "h264", "yify", "yts", "rarbg", "extended", "uncut", "directors",
}


def _tokens(text: Optional[str]) -> Set[str]:
    if not text:
        return set()
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in _STOP}


def score_candidate(
    cand: SubtitleCandidate,
    *,
    release_hint: Optional[str] = None,
    prefer_hearing_impaired: bool = False,
) -> Tuple[float, List[str]]:
    score = 0.0
    why: List[str] = []

    def add(points: float, label: str) -> None:
        nonlocal score
        if points:
            score += points
            why.append(f"{label}({points:+.0f})")

    if cand.from_trusted:
        add(100.0, "trusted")
    if cand.uploader_rank and cand.uploader_rank.lower() in {"administrator", "platinum member", "gold member", "trusted"}:
        add(15.0, "uploader_rank")
    if cand.ratings > 0:
        # Bayesian-ish: ratings only count fully with enough votes.
        weight = min(1.0, cand.votes / 5.0) if cand.votes else 0.3
        add(cand.ratings * 10.0 * weight, "rating")
    if cand.votes > 0:
        add(min(cand.votes * 2.0, 40.0), "votes")
    if cand.download_count > 0:
        add(min(math.log10(cand.download_count + 1) * 15.0, 75.0), "downloads")
    if cand.new_download_count > 0:
        add(min(math.log10(cand.new_download_count + 1) * 5.0, 15.0), "recent_downloads")
    if cand.hd:
        add(15.0, "hd")
    if cand.machine_translated:
        add(-150.0, "machine_translated")
    if cand.ai_translated:
        add(-120.0, "ai_translated")
    if cand.foreign_parts_only:
        add(-200.0, "foreign_parts_only")
    if len(cand.files) > 1:
        add(-60.0, "multi_cd")  # multi-CD releases need manual joining; offsets differ per CD
    if cand.hearing_impaired:
        add(25.0 if prefer_hearing_impaired else -10.0, "hearing_impaired")

    if release_hint and cand.release:
        hint, rel = _tokens(release_hint), _tokens(cand.release)
        common = hint & rel
        if common:
            source_matches = len(common & _SOURCE_TOKENS)
            add(min(len(common) * 6.0 + source_matches * 10.0, 45.0), "release_match")
    return round(score, 2), why


def rank_candidates(
    candidates: Iterable[SubtitleCandidate],
    *,
    release_hint: Optional[str] = None,
    prefer_hearing_impaired: bool = False,
) -> List[SubtitleCandidate]:
    ranked: List[SubtitleCandidate] = []
    for cand in candidates:
        if not cand.files:
            continue
        cand.ranking_score, reasons = score_candidate(
            cand, release_hint=release_hint, prefer_hearing_impaired=prefer_hearing_impaired
        )
        cand.ranking_reason = ", ".join(reasons)
        ranked.append(cand)
    ranked.sort(key=lambda c: (-c.ranking_score, -c.download_count, -c.ratings, c.files[0].file_id))
    return ranked


def select_top(
    candidates: Iterable[SubtitleCandidate],
    *,
    limit: int = 3,
    release_hint: Optional[str] = None,
    prefer_hearing_impaired: bool = False,
) -> List[SubtitleCandidate]:
    out: List[SubtitleCandidate] = []
    seen: Set[int] = set()
    for cand in rank_candidates(candidates, release_hint=release_hint, prefer_hearing_impaired=prefer_hearing_impaired):
        fid = cand.files[0].file_id
        if fid in seen:
            continue
        seen.add(fid)
        out.append(cand)
        if len(out) >= limit:
            break
    return out

