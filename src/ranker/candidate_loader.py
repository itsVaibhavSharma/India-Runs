"""Candidate Loader — Streams and parses candidate JSONL files.

Supports both plain .jsonl and gzipped .jsonl.gz formats.
Produces Candidate objects that the signal extractors can consume directly.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes (match the candidate_schema.json spec exactly)
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    name: str
    proficiency: str          # beginner / intermediate / advanced / expert
    endorsements: int
    duration_months: int


@dataclass
class CareerEntry:
    company: str
    title: str
    start_date: str
    end_date: Optional[str]
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str


@dataclass
class EducationEntry:
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str]
    tier: str                 # tier_1 / tier_2 / tier_3 / tier_4 / unknown


@dataclass
class RedrobSignals:
    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: Dict[str, float]
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_range_inr_lpa: Dict[str, float]
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


@dataclass
class Candidate:
    candidate_id: str
    profile: Dict[str, Any]
    career_history: List[CareerEntry]
    education: List[EducationEntry]
    skills: List[Skill]
    certifications: List[Dict[str, Any]]
    languages: List[Dict[str, Any]]
    redrob_signals: RedrobSignals
    _text_for_embedding: str = field(default="", repr=False)

    # ------------------------------------------------------------------
    # Convenience property accessors (read from profile dict)
    # ------------------------------------------------------------------

    @property
    def years_of_experience(self) -> float:
        return float(self.profile.get("years_of_experience", 0.0) or 0.0)

    @property
    def current_title(self) -> str:
        return self.profile.get("current_title", "") or ""

    @property
    def current_company(self) -> str:
        return self.profile.get("current_company", "") or ""

    @property
    def current_industry(self) -> str:
        return self.profile.get("current_industry", "") or ""

    @property
    def location(self) -> str:
        return self.profile.get("location", "") or ""

    @property
    def country(self) -> str:
        return self.profile.get("country", "") or ""

    @property
    def summary(self) -> str:
        return self.profile.get("summary", "") or ""

    @property
    def headline(self) -> str:
        return self.profile.get("headline", "") or ""

    @property
    def anonymized_name(self) -> str:
        return self.profile.get("anonymized_name", "") or ""


# ---------------------------------------------------------------------------
# CandidateLoader
# ---------------------------------------------------------------------------

class CandidateLoader:
    """Streams and parses candidates from a JSONL (or gzipped JSONL) file."""

    def __init__(self, candidates_path: str):
        self.candidates_path = Path(candidates_path)

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_skill(self, data: Dict[str, Any]) -> Skill:
        return Skill(
            name=data.get("name", "") or "",
            proficiency=data.get("proficiency", "beginner") or "beginner",
            endorsements=int(data.get("endorsements", 0) or 0),
            duration_months=int(data.get("duration_months", 0) or 0),
        )

    def _parse_career(self, data: Dict[str, Any]) -> CareerEntry:
        return CareerEntry(
            company=data.get("company", "") or "",
            title=data.get("title", "") or "",
            start_date=data.get("start_date", "") or "",
            end_date=data.get("end_date"),
            duration_months=int(data.get("duration_months", 0) or 0),
            is_current=bool(data.get("is_current", False)),
            industry=data.get("industry", "") or "",
            company_size=data.get("company_size", "") or "",
            description=data.get("description", "") or "",
        )

    def _parse_education(self, data: Dict[str, Any]) -> EducationEntry:
        return EducationEntry(
            institution=data.get("institution", "") or "",
            degree=data.get("degree", "") or "",
            field_of_study=data.get("field_of_study", "") or "",
            start_year=int(data.get("start_year", 0) or 0),
            end_year=int(data.get("end_year", 0) or 0),
            grade=data.get("grade"),
            tier=data.get("tier", "unknown") or "unknown",
        )

    def _parse_signals(self, data: Dict[str, Any]) -> RedrobSignals:
        sal = data.get("expected_salary_range_inr_lpa", {}) or {}
        return RedrobSignals(
            profile_completeness_score=float(data.get("profile_completeness_score", 0.0) or 0.0),
            signup_date=data.get("signup_date", "") or "",
            last_active_date=data.get("last_active_date", "") or "",
            open_to_work_flag=bool(data.get("open_to_work_flag", False)),
            profile_views_received_30d=int(data.get("profile_views_received_30d", 0) or 0),
            applications_submitted_30d=int(data.get("applications_submitted_30d", 0) or 0),
            recruiter_response_rate=float(data.get("recruiter_response_rate", 0.0) or 0.0),
            avg_response_time_hours=float(data.get("avg_response_time_hours", 0.0) or 0.0),
            skill_assessment_scores=data.get("skill_assessment_scores", {}) or {},
            connection_count=int(data.get("connection_count", 0) or 0),
            endorsements_received=int(data.get("endorsements_received", 0) or 0),
            notice_period_days=int(data.get("notice_period_days", 0) or 0),
            expected_salary_range_inr_lpa={"min": float(sal.get("min", 0) or 0), "max": float(sal.get("max", 0) or 0)},
            preferred_work_mode=data.get("preferred_work_mode", "flexible") or "flexible",
            willing_to_relocate=bool(data.get("willing_to_relocate", False)),
            github_activity_score=float(data.get("github_activity_score", -1.0) if data.get("github_activity_score") is not None else -1.0),
            search_appearance_30d=int(data.get("search_appearance_30d", 0) or 0),
            saved_by_recruiters_30d=int(data.get("saved_by_recruiters_30d", 0) or 0),
            interview_completion_rate=float(data.get("interview_completion_rate", 0.0) or 0.0),
            offer_acceptance_rate=float(data.get("offer_acceptance_rate", -1.0) if data.get("offer_acceptance_rate") is not None else -1.0),
            verified_email=bool(data.get("verified_email", False)),
            verified_phone=bool(data.get("verified_phone", False)),
            linkedin_connected=bool(data.get("linkedin_connected", False)),
        )

    def _build_embedding_text(self, candidate: Candidate) -> str:
        """Builds the text used for semantic embedding from structured profile fields."""
        parts: List[str] = []

        # Profile narrative
        if candidate.summary:
            parts.append(candidate.summary)
        if candidate.headline:
            parts.append(candidate.headline)

        # Career descriptions (rich signal for technical depth)
        for career in candidate.career_history:
            if career.description:
                parts.append(career.description)
            # Also include job title + company for context
            if career.title and career.company:
                parts.append(f"{career.title} at {career.company}")

        # Skill names (helps semantic matching to JD)
        skill_names = [s.name for s in candidate.skills if s.name]
        if skill_names:
            parts.append("Skills: " + ", ".join(skill_names))

        return " ".join(parts)

    def _parse_candidate(self, data: Dict[str, Any]) -> Candidate:
        candidate = Candidate(
            candidate_id=data.get("candidate_id", "") or "",
            profile=data.get("profile", {}) or {},
            career_history=[self._parse_career(c) for c in data.get("career_history", [])],
            education=[self._parse_education(e) for e in data.get("education", [])],
            skills=[self._parse_skill(s) for s in data.get("skills", [])],
            certifications=data.get("certifications", []) or [],
            languages=data.get("languages", []) or [],
            redrob_signals=self._parse_signals(data.get("redrob_signals", {}) or {}),
        )
        candidate._text_for_embedding = self._build_embedding_text(candidate)
        return candidate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _open_file(self):
        """Returns a file-like object for reading the candidates file."""
        if self.candidates_path.suffix == ".gz":
            return gzip.open(self.candidates_path, "rt", encoding="utf-8")
        return open(self.candidates_path, "r", encoding="utf-8")

    def load_candidates(self) -> Iterator[Candidate]:
        """Stream candidates one at a time (memory-efficient)."""
        with self._open_file() as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield self._parse_candidate(data)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    logger.warning("Skipping line %d: %s", line_num, exc)

    def load_all_candidates(self) -> List[Candidate]:
        """Load all candidates into memory. Use only when RAM allows."""
        return list(self.load_candidates())

    def count_candidates(self) -> int:
        """Count lines without parsing."""
        count = 0
        with self._open_file() as f:
            for line in f:
                if line.strip():
                    count += 1
        return count