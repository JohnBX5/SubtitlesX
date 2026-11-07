"""Deterministic subtitle candidate ranking service.

Evaluates and ranks subtitle candidates returned by OpenSubtitles to select
the most suitable candidate for playback in Cinema Hub.

NOTE: Subtitle ranking is based on metadata, community ratings, download volume,
and uploader reputation. High rating or download volume does NOT guarantee perfect
synchronization with every specific video rip or release.
"""

import math
import re
from typing import List, Optional, Tuple

from app.schemas.subtitle import SubtitleCandidate


def calculate_candidate_score(
    candidate: SubtitleCandidate,
    query_hint: Optional[str] = None,
    prefer_hearing_impaired: bool = False,
) -> Tuple[float, List[str]]:
    """Calculate a deterministic suitability score for a subtitle candidate.

    Scoring System:
    - Trusted source (`from_trusted`): +100.0 pts
    - Community rating (`ratings`, 0-10): `ratings * 10.0` (up to +100.0 pts)
    - Rating confidence (`votes`): `min(votes * 2.0, 50.0)` pts
    - Download volume (`download_count`): Log-scaled `min(log10(downloads + 1) * 15.0, 75.0)` pts
    - HD release timing (`hd`): +15.0 pts
    - Machine translation penalty (`machine_translated`): -150.0 pts
    - AI translation penalty (`ai_translated`): -150.0 pts
    - Foreign parts only penalty (`foreign_parts_only`): -50.0 pts
    - Hearing impaired alignment:
        If prefer_hearing_impaired is False and candidate is hearing_impaired: -10.0 pts
        If prefer_hearing_impaired is True and candidate is hearing_impaired: +20.0 pts
    - Release name match bonus: up to +30.0 pts if candidate release matches query keywords.

    Returns:
        Tuple of (total_score, reason_tokens)
    """
    score = 0.0
    reasons: List[str] = []

    # 1. Trusted source
    if candidate.from_trusted:
        score += 100.0
        reasons.append("trusted_source(+100)")

    # 2. Ratings & Votes
    if candidate.ratings > 0:
        rating_pts = candidate.ratings * 10.0
        score += rating_pts
        reasons.append(f"rating_{candidate.ratings:.1f}(+{rating_pts:.1f})")

    if candidate.votes > 0:
        vote_pts = min(candidate.votes * 2.0, 50.0)
        score += vote_pts
        reasons.append(f"votes_{candidate.votes}(+{vote_pts:.1f})")

    # 3. Download count (logarithmic scale)
    if candidate.download_count > 0:
        dl_pts = min(math.log10(candidate.download_count + 1) * 15.0, 75.0)
        score += dl_pts
        reasons.append(f"downloads_{candidate.download_count}(+{dl_pts:.1f})")

    # 4. HD flag
    if candidate.hd:
        score += 15.0
        reasons.append("hd(+15)")

    # 5. Machine / AI translation penalties
    if candidate.machine_translated:
        score -= 150.0
        reasons.append("machine_translated(-150)")
    if candidate.ai_translated:
        score -= 150.0
        reasons.append("ai_translated(-150)")

    # 6. Foreign parts only
    if candidate.foreign_parts_only:
        score -= 50.0
        reasons.append("foreign_parts_only(-50)")

    # 7. Hearing impaired handling
    if candidate.hearing_impaired:
        if prefer_hearing_impaired:
            score += 20.0
            reasons.append("hearing_impaired_preferred(+20)")
        else:
            score -= 10.0
            reasons.append("hearing_impaired(-10)")

    # 8. Release name matching
    if query_hint and candidate.release:
        query_words = set(re.findall(r"[A-Za-z0-9]+", query_hint.lower()))
        release_words = set(re.findall(r"[A-Za-z0-9]+", candidate.release.lower()))
        common = query_words.intersection(release_words)
        # Exclude trivial stop words
        common = {w for w in common if len(w) > 2 and w not in {"the", "and", "movie", "sub"}}
        if common:
            match_pts = min(len(common) * 10.0, 30.0)
            score += match_pts
            reasons.append(f"release_match_{len(common)}(+{match_pts:.1f})")

    return score, reasons


def rank_subtitle_candidates(
    candidates: List[SubtitleCandidate],
    query_hint: Optional[str] = None,
    prefer_hearing_impaired: bool = False,
) -> List[SubtitleCandidate]:
    """Score and sort candidate subtitles deterministically.

    Order:
    1. ranking_score DESC
    2. download_count DESC
    3. ratings DESC
    4. file_id ASC (predictable tie-breaker)
    """
    if not candidates:
        return []

    scored_candidates: List[SubtitleCandidate] = []
    for cand in candidates:
        score, reasons = calculate_candidate_score(
            cand,
            query_hint=query_hint,
            prefer_hearing_impaired=prefer_hearing_impaired,
        )
        cand.ranking_score = round(score, 2)
        cand.ranking_reason = ", ".join(reasons)
        scored_candidates.append(cand)

    # Sort with deterministic tie-breaking
    def sort_key(item: SubtitleCandidate):
        primary_file_id = item.files[0].file_id if item.files else 0
        return (
            -item.ranking_score,
            -item.download_count,
            -item.ratings,
            primary_file_id,
        )

    scored_candidates.sort(key=sort_key)
    return scored_candidates


def select_best_candidate(
    candidates: List[SubtitleCandidate],
    query_hint: Optional[str] = None,
    prefer_hearing_impaired: bool = False,
) -> Optional[SubtitleCandidate]:
    """Select the single best subtitle candidate according to the ranking policy."""
    ranked = rank_subtitle_candidates(
        candidates,
        query_hint=query_hint,
        prefer_hearing_impaired=prefer_hearing_impaired,
    )
    return ranked[0] if ranked else None


def select_top_candidates(
    candidates: List[SubtitleCandidate],
    query_hint: Optional[str] = None,
    prefer_hearing_impaired: bool = False,
    max_candidates: int = 3,
) -> List[SubtitleCandidate]:
    """Select up to `max_candidates` top-ranked distinct subtitle candidates with downloadable files.

    Used by the acoustic synchronization engine to test multiple subtitle releases
    against the actual video audio timeline and pick the best acoustic match.
    """
    ranked = rank_subtitle_candidates(
        candidates,
        query_hint=query_hint,
        prefer_hearing_impaired=prefer_hearing_impaired,
    )
    selected: List[SubtitleCandidate] = []
    seen_file_ids = set()

    for cand in ranked:
        if not cand.files:
            continue
        fid = cand.files[0].file_id
        if fid in seen_file_ids:
            continue
        seen_file_ids.add(fid)
        selected.append(cand)
        if len(selected) >= max_candidates:
            break

    return selected


