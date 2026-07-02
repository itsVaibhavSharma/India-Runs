"""Candidate Loader - Streams and parses candidate JSONL with signal extraction."""

import json
import gzip
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterator, Any
from pathlib import Path
from datetime import datetime
import numpy as np


@dataclass
class Skill:
    name: str
    proficiency: str
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
    tier: str


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

    @property
    def years_of_experience(self) -> float:
        return self.profile.get('years_of_experience', 0.0)

    @property
    def current_title(self) -> str:
        return self.profile.get('current_title', '')

    @property
    def current_company(self) -> str:
        return self.profile.get('current_company', '')

    @property
    def current_industry(self) -> str:
        return self.profile.get('current_industry', '')

    @property
    def location(self) -> str:
        return self.profile.get('location', '')

    @property
    def country(self) -> str:
        return self.profile.get('country', '')

    @property
    def summary(self) -> str:
        return self.profile.get('summary', '')

    @property
    def headline(self) -> str:
        return self.profile.get('headline', '')


class CandidateLoader:
    PROFICIENCY_WEIGHTS = {
        'beginner': 0.2,
        'intermediate': 0.5,
        'advanced': 0.8,
        'expert': 1.0
    }

    SERVICE_INDUSTRIES = {
        'it services', 'consulting', 'outsourcing', 'system integrator',
        'managed services', 'professional services', 'staffing'
    }

    CONSULTING_FIRMS = {
        'tcs', 'infosys', 'wipro', 'cognizant', 'capgemini', 'accenture',
        'hcl', 'tech mahindra', 'lti', 'mindtree', 'mphasis', 'hexaware',
        'l&t infotech', 'zensar', 'ntt data', 'cgi', 'virtusa', 'synechron'
    }

    def __init__(self, candidates_path: str):
        self.candidates_path = Path(candidates_path)

    def _parse_skill(self, skill_data: Dict[str, Any]) -> Skill:
        return Skill(
            name=skill_data.get('name', ''),
            proficiency=skill_data.get('proficiency', 'beginner'),
            endorsements=skill_data.get('endorsements', 0),
            duration_months=skill_data.get('duration_months', 0)
        )

    def _parse_career(self, career_data: Dict[str, Any]) -> CareerEntry:
        return CareerEntry(
            company=career_data.get('company', ''),
            title=career_data.get('title', ''),
            start_date=career_data.get('start_date', ''),
            end_date=career_data.get('end_date'),
            duration_months=career_data.get('duration_months', 0),
            is_current=career_data.get('is_current', False),
            industry=career_data.get('industry', ''),
            company_size=career_data.get('company_size', ''),
            description=career_data.get('description', '')
        )

    def _parse_education(self, edu_data: Dict[str, Any]) -> EducationEntry:
        return EducationEntry(
            institution=edu_data.get('institution', ''),
            degree=edu_data.get('degree', ''),
            field_of_study=edu_data.get('field_of_study', ''),
            start_year=edu_data.get('start_year', 0),
            end_year=edu_data.get('end_year', 0),
            grade=edu_data.get('grade'),
            tier=edu_data.get('tier', 'unknown')
        )

    def _parse_signals(self, signals_data: Dict[str, Any]) -> RedrobSignals:
        return RedrobSignals(
            profile_completeness_score=signals_data.get('profile_completeness_score', 0.0),
            signup_date=signals_data.get('signup_date', ''),
            last_active_date=signals_data.get('last_active_date', ''),
            open_to_work_flag=signals_data.get('open_to_work_flag', False),
            profile_views_received_30d=signals_data.get('profile_views_received_30d', 0),
            applications_submitted_30d=signals_data.get('applications_submitted_30d', 0),
            recruiter_response_rate=signals_data.get('recruiter_response_rate', 0.0),
            avg_response_time_hours=signals_data.get('avg_response_time_hours', 0.0),
            skill_assessment_scores=signals_data.get('skill_assessment_scores', {}),
            connection_count=signals_data.get('connection_count', 0),
            endorsements_received=signals_data.get('endorsements_received', 0),
            notice_period_days=signals_data.get('notice_period_days', 0),
            expected_salary_range_inr_lpa=signals_data.get('expected_salary_range_inr_lpa', {'min': 0, 'max': 0}),
            preferred_work_mode=signals_data.get('preferred_work_mode', 'flexible'),
            willing_to_relocate=signals_data.get('willing_to_relocate', False),
            github_activity_score=signals_data.get('github_activity_score', -1.0),
            search_appearance_30d=signals_data.get('search_appearance_30d', 0),
            saved_by_recruiters_30d=signals_data.get('saved_by_recruiters_30d', 0),
            interview_completion_rate=signals_data.get('interview_completion_rate', 0.0),
            offer_acceptance_rate=signals_data.get('offer_acceptance_rate', -1.0),
            verified_email=signals_data.get('verified_email', False),
            verified_phone=signals_data.get('verified_phone', False),
            linkedin_connected=signals_data.get('linkedin_connected', False)
        )

    def _build_embedding_text(self, candidate: Candidate) -> str:
        parts = []
        if candidate.summary:
            parts.append(candidate.summary)
        if candidate.headline:
            parts.append(candidate.headline)
        for career in candidate.career_history:
            if career.description:
                parts.append(career.description)
        return ' '.join(parts)

    def load_candidates(self) -> Iterator[Candidate]:
        if self.candidates_path.suffix == '.gz':
            opener = gzip.open
            mode = 'rt'
        else:
            opener = open
            mode = 'r'

        with opener(self.candidates_path, mode, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    candidate = self._parse_candidate(data)
                    yield candidate
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue

    def _parse_candidate(self, data: Dict[str, Any]) -> Candidate:
        candidate = Candidate(
            candidate_id=data.get('candidate_id', ''),
            profile=data.get('profile', {}),
            career_history=[self._parse_career(c) for c in data.get('career_history', [])],
            education=[self._parse_education(e) for e in data.get('education', [])],
            skills=[self._parse_skill(s) for s in data.get('skills', [])],
            certifications=data.get('certifications', []),
            languages=data.get('languages', []),
            redrob_signals=self._parse_signals(data.get('redrob_signals', {}))
        )
        candidate._text_for_embedding = self._build_embedding_text(candidate)
        return candidate

    def load_all_candidates(self) -> List[Candidate]:
        return list(self.load_candidates())

    def count_candidates(self) -> int:
        count = 0
        if self.candidates_path.suffix == '.gz':
            opener = gzip.open
            mode = 'rt'
        else:
            opener = open
            mode = 'r'

        with opener(self.candidates_path, mode, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def is_service_company(self, company: str) -> bool:
        company_lower = company.lower()
        return any(firm in company_lower for firm in self.CONSULTING_FIRMS)

    def is_service_industry(self, industry: str) -> bool:
        industry_lower = industry.lower()
        return any(svc in industry_lower for svc in self.SERVICE_INDUSTRIES)

    def get_skill_trust_score(self, skill: Skill) -> float:
        endorsement_factor = min(1.0, skill.endorsements / 10.0)
        duration_factor = min(1.0, skill.duration_months / 24.0)
        proficiency_weight = self.PROFICIENCY_WEIGHTS.get(skill.proficiency, 0.2)
        return endorsement_factor * duration_factor * proficiency_weight