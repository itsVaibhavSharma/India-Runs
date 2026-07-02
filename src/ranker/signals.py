"""Signal Extractors — Computes 7 independent scoring signals per candidate.

Signals:
  1. title_career   — Title match, production evidence, product vs services company
  2. skill_depth    — Trust-weighted skill score (endorsements × duration × proficiency)
  3. experience     — Years-of-experience fit for the [5-9] target band
  4. education      — Institution tier + field relevance + degree level
  5. location       — Preferred city match and willingness-to-relocate
  6. behavioral     — Availability gate + engagement signals (response rate, GitHub…)
  7. honeypot_penalty — Detects impossible / fabricated profiles

All scores are in [0, 1].  honeypot_penalty is subtracted / multiplied in fusion.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SignalScores:
    title_career: float = 0.0
    skill_depth: float = 0.0
    experience: float = 0.0
    education: float = 0.0
    location: float = 0.0
    behavioral: float = 0.0
    honeypot_penalty: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "title_career": self.title_career,
            "skill_depth": self.skill_depth,
            "experience": self.experience,
            "education": self.education,
            "location": self.location,
            "behavioral": self.behavioral,
            "honeypot_penalty": self.honeypot_penalty,
        }

    def to_list(self) -> List[float]:
        return [
            self.title_career,
            self.skill_depth,
            self.experience,
            self.education,
            self.location,
            self.behavioral,
            -self.honeypot_penalty,   # negative — used as penalty in fusion
        ]


# ---------------------------------------------------------------------------
# Constants (JD-derived, Senior AI Engineer @ Redrob)
# ---------------------------------------------------------------------------

REQUIRED_SKILLS: Dict[str, float] = {
    # Core retrieval / ranking tech — highest weight
    "embeddings": 3.0,
    "vector search": 3.0,
    "rag": 3.0,
    "retrieval": 3.0,
    "ranking": 3.0,
    "sentence-transformers": 3.0,
    "sentence transformers": 3.0,
    "bge": 2.5,
    "e5": 2.5,
    "faiss": 3.0,
    "pinecone": 2.5,
    "weaviate": 2.5,
    "qdrant": 2.5,
    "milvus": 2.5,
    "opensearch": 2.0,
    "elasticsearch": 2.0,
    "hybrid search": 3.0,
    "semantic search": 3.0,
    "dense retrieval": 2.5,
    "sparse retrieval": 2.5,
    "bm25": 2.5,
    # Evaluation
    "ndcg": 2.5,
    "mrr": 2.5,
    "map": 2.0,
    "a/b testing": 2.0,
    "learning to rank": 3.0,
    "ltr": 2.5,
    # Foundational
    "python": 2.5,
    "production": 2.0,
    "deployed": 2.0,
    "shipped": 2.0,
}

NICE_TO_HAVE_SKILLS: Dict[str, float] = {
    "lora": 1.5, "qlora": 1.5, "peft": 1.5, "fine-tuning": 1.5,
    "fine-tuning llms": 1.5, "finetuning": 1.5,
    "xgboost": 1.5, "lightgbm": 1.5, "catboost": 1.5,
    "lambdamart": 1.5, "listnet": 1.5, "ranklib": 1.5,
    "hr tech": 1.2, "recruiting": 1.2, "marketplace": 1.2,
    "distributed systems": 1.2, "inference optimization": 1.2,
    "mlops": 1.3, "model serving": 1.3, "triton": 1.2,
    "kubeflow": 1.2, "airflow": 1.1, "spark": 1.1,
    "kubernetes": 1.1, "docker": 1.0,
}

RANKING_KEYWORDS = {
    "ndcg", "mrr", "map", "precision@", "recall@", "hit rate",
    "offline evaluation", "online evaluation", "a/b test", "ab test",
    "ranking quality", "retrieval quality", "search quality",
    "learning to rank", "ltr", "lambdamart", "ranklib", "xgboost ranking",
    "click-through", "ctr", "mean reciprocal", "average precision",
    "interleaving", "multi-armed bandit",
}

PRODUCTION_KEYWORDS = {
    "production", "deployed", "shipped", "real users", "at scale", "serving",
    "inference", "latency", "throughput", "api", "endpoint", "model serving",
    "mlops", "ci/cd", "pipeline", "monitoring", "millions",
}

RETRIEVAL_KEYWORDS = {
    "embeddings", "vector", "retrieval", "search", "ranking",
    "recommendation", "recommender", "matching", "similarity", "rag",
    "information retrieval", "ir system",
}

TARGET_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "senior ai engineer", "senior ml engineer", "applied scientist",
    "ranking engineer", "search engineer", "recommendation engineer",
    "nlp engineer", "data scientist", "research engineer",
    "applied ml engineer", "applied ai engineer", "senior data scientist",
    "staff engineer", "principal engineer", "software engineer, ml",
    "software engineer, ai",
}

DISQUALIFIER_TITLES = {
    "engineering manager", "vp engineering", "vp of engineering",
    "head of engineering", "director of engineering", "cto",
    "principal architect", "solution architect", "enterprise architect",
}

SERVICE_FIRMS = {
    "tcs", "infosys", "wipro", "cognizant", "capgemini", "accenture",
    "hcl", "tech mahindra", "lti", "mindtree", "mphasis", "hexaware",
    "l&t infotech", "zensar", "ntt data", "cgi", "virtusa", "synechron",
    "ltimindtree", "persistent systems", "kpit", "cyient", "niit tech",
    "mastech", "happiest minds",
}

SERVICE_INDUSTRIES = {
    "it services", "consulting", "outsourcing", "system integrator",
    "managed services", "professional services", "staffing",
    "bpo", "kpo", "ites",
}

TIER1_INSTITUTIONS = {
    "iit", "iisc", "iit bombay", "iit delhi", "iit madras", "iit kanpur",
    "iit kharagpur", "iit roorkee", "iit guwahati", "iit hyderabad",
    "bits pilani", "bits goa", "bits hyderabad", "nit trichy", "nit surathkal",
    "nit warangal", "nit calicut", "nit rourkela", "nit kurukshetra",
    "iiit hyderabad", "iiit bangalore", "iiit delhi", "iiit allahabad",
    "anna university", "dtu", "nsut", "jadavpur university",
    "pec chandigarh", "thapar", "vit vellore", "srm",
}

TIER2_INSTITUTIONS = {
    "pec", "nit", "nit ", "iiit", "nitk", "svnit",
    "manipal", "amrita", "symbiosis", "christ", "mu",
    "lovely professional university", "lpu", "chandigarh university",
    "shiv nadar", "bennett", "flame", "jain university",
}

PREFERRED_LOCATIONS_HIGH = {"pune", "noida"}  # score 1.0
PREFERRED_LOCATIONS_MED = {
    "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "bengaluru",
    "bangalore", "ncr", "national capital region", "navi mumbai",
    "thane", "ghaziabad", "faridabad", "noidagurgaon",
}  # score 0.85
OTHER_INDIA = {"chennai", "kolkata", "ahmedabad", "jaipur", "bhopal",
               "nagpur", "kochi", "chandigarh", "lucknow", "indore"}  # score 0.5

PROFICIENCY_WEIGHTS = {
    "beginner": 0.2,
    "intermediate": 0.5,
    "advanced": 0.8,
    "expert": 1.0,
}

# Known company founding years for honeypot detection
KNOWN_FOUNDING_YEARS = {
    "openai": 2015, "anthropic": 2021, "deepmind": 2010,
    "hugging face": 2016, "huggingface": 2016, "cohere": 2019,
    "adept": 2022, "character.ai": 2021, "inflection": 2022,
    "mistral": 2023, "databricks": 2013, "redrob": 2021,
    "perplexity": 2022, "together ai": 2022, "groq": 2016,
}

CURRENT_YEAR = 2026


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return text.lower().strip() if text else ""


def _is_service_company(company: str) -> bool:
    cl = _normalise(company)
    return any(firm in cl for firm in SERVICE_FIRMS)


def _is_service_industry(industry: str) -> bool:
    il = _normalise(industry)
    return any(svc in il for svc in SERVICE_INDUSTRIES)


def _days_since(date_str: str) -> Optional[int]:
    """Return days since date_str (ISO format). None if unparseable."""
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str)
        return max(0, (datetime.now() - dt).days)
    except (ValueError, TypeError):
        return None


def skill_trust_score(skill_name: str, proficiency: str,
                      endorsements: int, duration_months: int) -> float:
    """Trust-weighted skill score used both in SignalExtractor and Candidate."""
    endorsement_factor = min(1.0, endorsements / 10.0)
    duration_factor = min(1.0, duration_months / 24.0)
    prof_weight = PROFICIENCY_WEIGHTS.get(proficiency, 0.2)
    return endorsement_factor * duration_factor * prof_weight


# ---------------------------------------------------------------------------
# SignalExtractor
# ---------------------------------------------------------------------------

class SignalExtractor:
    """Extracts all 7 signals from a Candidate object."""

    def __init__(self, jd_requirements=None):
        self.jd_requirements = jd_requirements
        # Merge JD-specific requirements into our defaults
        if jd_requirements:
            self._merge_from_jd(jd_requirements)

    def _merge_from_jd(self, req) -> None:
        if getattr(req, "must_have_skills", None):
            for s in req.must_have_skills:
                if s.lower() not in REQUIRED_SKILLS:
                    REQUIRED_SKILLS[s.lower()] = 2.0
        if getattr(req, "nice_to_have_skills", None):
            for s in req.nice_to_have_skills:
                if s.lower() not in NICE_TO_HAVE_SKILLS:
                    NICE_TO_HAVE_SKILLS[s.lower()] = 1.2

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def extract_all_signals(self, candidate) -> SignalScores:
        scores = SignalScores()
        scores.title_career = self.extract_title_career_score(candidate)
        scores.skill_depth = self.extract_skill_depth_score(candidate)
        scores.experience = self.extract_experience_score(candidate)
        scores.education = self.extract_education_score(candidate)
        scores.location = self.extract_location_score(candidate)
        scores.behavioral = self.extract_behavioral_score(candidate)
        scores.honeypot_penalty = self.extract_honeypot_penalty(candidate)
        return scores

    # ------------------------------------------------------------------
    # Signal 1: Title / Career
    # ------------------------------------------------------------------

    def extract_title_career_score(self, candidate) -> float:
        score = 0.0
        current_title = _normalise(candidate.current_title)
        current_company = _normalise(candidate.current_company)
        current_industry = _normalise(candidate.current_industry)

        # ---- Current title match (max 0.30) ----
        title_score = 0.0
        for target in TARGET_TITLES:
            if target in current_title:
                title_score = 1.0
                break
            words = target.split()
            if any(w in current_title for w in words if len(w) > 3):
                title_score = max(title_score, 0.55)
        # Disqualifier titles
        for dtitle in DISQUALIFIER_TITLES:
            if dtitle in current_title:
                title_score = min(title_score, 0.2)
                break
        score += title_score * 0.30

        # ---- Career history analysis (max 0.50) ----
        has_production = False
        has_retrieval = False
        has_ranking_eval = False
        has_product_experience = False
        career_score = 0.0

        for career in candidate.career_history:
            desc = _normalise(career.description)
            company = _normalise(career.company)
            title = _normalise(career.title)
            industry = _normalise(career.industry)
            dur_months = getattr(career, "duration_months", 0) or 0

            # Production deployment evidence
            prod_hits = sum(1 for kw in PRODUCTION_KEYWORDS if kw in desc)
            if prod_hits >= 1:
                has_production = True
                career_score += min(0.15, prod_hits * 0.04)

            # Retrieval / ranking / recommendation experience
            ret_hits = sum(1 for kw in RETRIEVAL_KEYWORDS if kw in desc)
            if ret_hits >= 1:
                has_retrieval = True
                career_score += min(0.15, ret_hits * 0.04)

            # Explicit ranking evaluation keywords
            if any(kw in desc for kw in RANKING_KEYWORDS):
                has_ranking_eval = True
                career_score += 0.12

            # Product company (not services)
            is_svc = _is_service_company(company) or _is_service_industry(industry)
            if not is_svc:
                has_product_experience = True
                # Tenure-weighted bonus
                career_score += min(0.08, dur_months / 1000.0)

            # Title in career matches target roles
            if any(t in title for t in TARGET_TITLES):
                career_score += 0.06

        career_score = min(0.50, career_score)
        score += career_score

        # ---- Company type adjustment (max ±0.20) ----
        is_currently_services = (
            _is_service_company(current_company) or
            _is_service_industry(current_industry)
        )
        if is_currently_services:
            score -= 0.20
        elif has_product_experience:
            score += 0.12

        # ---- Must-have penalties ----
        if not has_production:
            score *= 0.65
        if not has_retrieval:
            score *= 0.75

        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Signal 2: Skill Depth
    # ------------------------------------------------------------------

    def extract_skill_depth_score(self, candidate) -> float:
        if not candidate.skills:
            return 0.0

        raw_score = 0.0
        required_hits = 0
        nice_hits = 0

        for skill in candidate.skills:
            name = _normalise(skill.name)
            trust = skill_trust_score(
                name, skill.proficiency,
                skill.endorsements, skill.duration_months
            )

            # Check required skills (highest multipliers)
            matched_req = False
            for req_name, multiplier in REQUIRED_SKILLS.items():
                if req_name in name or name in req_name:
                    raw_score += trust * multiplier
                    required_hits += 1
                    matched_req = True
                    break

            if not matched_req:
                # Check nice-to-have
                matched_nice = False
                for nice_name, multiplier in NICE_TO_HAVE_SKILLS.items():
                    if nice_name in name or name in nice_name:
                        raw_score += trust * multiplier
                        nice_hits += 1
                        matched_nice = True
                        break
                if not matched_nice:
                    # Generic skill — small contribution
                    raw_score += trust * 0.3

        # Normalise: expected max is roughly covering all required skills at trust=1
        # Using sum of required multipliers as denominator
        req_max = sum(REQUIRED_SKILLS.values())
        nice_max = sum(NICE_TO_HAVE_SKILLS.values()) * 0.5   # only 50% coverage expected
        normaliser = req_max + nice_max

        normalised = raw_score / normaliser if normaliser > 0 else 0.0

        # Coverage bonus: fraction of required skills actually present
        req_coverage = required_hits / max(1, len(REQUIRED_SKILLS))
        normalised += req_coverage * 0.15

        # Skill assessment scores bonus
        assessments = {}
        try:
            assessments = candidate.redrob_signals.skill_assessment_scores or {}
        except AttributeError:
            pass
        if assessments:
            avg_assessment = sum(assessments.values()) / len(assessments)
            normalised += (avg_assessment / 100.0) * 0.05

        return min(1.0, normalised)

    # ------------------------------------------------------------------
    # Signal 3: Experience
    # ------------------------------------------------------------------

    def extract_experience_score(self, candidate) -> float:
        years = candidate.years_of_experience or 0.0

        # JD says 5-9 years; we accept 4-10 at lower penalty
        if 5.0 <= years <= 9.0:
            base = 1.0
        elif 4.0 <= years < 5.0:
            base = 0.85
        elif 9.0 < years <= 11.0:
            # Slightly over — still good, slight decay
            base = max(0.6, 1.0 - (years - 9.0) * 0.08)
        elif years < 4.0:
            # Too junior — linear decay from 0.5 at 3 yrs to 0 at 0 yrs
            base = max(0.1, years / 5.0)
        else:
            # > 11 years — significant overqualification
            base = max(0.35, 1.0 - (years - 9.0) * 0.06)

        # Bonus: check for applied ML career longevity
        applied_ml_months = 0
        for career in candidate.career_history:
            desc = _normalise(career.description)
            if any(kw in desc for kw in RETRIEVAL_KEYWORDS | PRODUCTION_KEYWORDS):
                applied_ml_months += (getattr(career, "duration_months", 0) or 0)

        applied_ml_years = applied_ml_months / 12.0
        if applied_ml_years >= 4.0:
            base = min(1.0, base + 0.08)
        elif applied_ml_years >= 2.0:
            base = min(1.0, base + 0.04)

        return max(0.0, min(1.0, base))

    # ------------------------------------------------------------------
    # Signal 4: Education
    # ------------------------------------------------------------------

    def extract_education_score(self, candidate) -> float:
        if not candidate.education:
            return 0.15   # no education listed — small default

        best = 0.0
        for edu in candidate.education:
            inst = _normalise(edu.institution)
            field = _normalise(edu.field_of_study)
            tier = _normalise(getattr(edu, "tier", "unknown"))
            degree = _normalise(edu.degree)
            end_year = getattr(edu, "end_year", 0) or 0

            s = 0.0

            # Tier scoring
            if tier == "tier_1" or any(t in inst for t in TIER1_INSTITUTIONS):
                s += 0.55
            elif tier == "tier_2" or any(t in inst for t in TIER2_INSTITUTIONS):
                s += 0.35
            elif tier == "tier_3":
                s += 0.20
            else:
                s += 0.10

            # Field relevance
            if any(f in field for f in [
                "computer", "machine learning", "artificial intelligence",
                "ai", "data", "statistics", "math", "information technology",
                "electronics", "software"
            ]):
                s += 0.20

            # Degree level
            if any(x in degree for x in ["ph.d", "phd", "doctor"]):
                s += 0.12
            elif any(x in degree for x in ["master", "m.tech", "m.e.", "m.sc", "ms"]):
                s += 0.08
            elif any(x in degree for x in ["b.tech", "b.e.", "b.sc", "bachelor"]):
                s += 0.04

            # Recency (graduated recently = potentially more modern ML knowledge)
            if end_year >= 2020:
                s += 0.03

            best = max(best, s)

        return min(1.0, best)

    # ------------------------------------------------------------------
    # Signal 5: Location
    # ------------------------------------------------------------------

    def extract_location_score(self, candidate) -> float:
        location = _normalise(candidate.location)
        country = _normalise(candidate.country)
        signals = candidate.redrob_signals

        # Check preferred locations
        for loc in PREFERRED_LOCATIONS_HIGH:
            if loc in location:
                return 1.0

        for loc in PREFERRED_LOCATIONS_MED:
            if loc in location:
                return 0.85

        for loc in OTHER_INDIA:
            if loc in location:
                # Willing-to-relocate bonus
                if signals.willing_to_relocate:
                    return 0.65
                return 0.45

        # India but unspecified city
        if country == "india":
            if signals.willing_to_relocate:
                return 0.60
            return 0.35

        # Outside India — possible for remote/hybrid
        if signals.willing_to_relocate:
            return 0.25
        return 0.10

    # ------------------------------------------------------------------
    # Signal 6: Behavioral (Availability Gate)
    # ------------------------------------------------------------------

    def extract_behavioral_score(self, candidate) -> float:
        signals = candidate.redrob_signals

        # ---- HARD GATE: must pass all three to get any score ----
        if not signals.open_to_work_flag:
            return 0.0

        days_inactive = _days_since(signals.last_active_date)
        if days_inactive is None or days_inactive > 60:
            return 0.0

        if signals.recruiter_response_rate < 0.10:
            return 0.0

        # ---- Passed gate — compute engagement score ----
        score = 0.0

        # Response rate (40% weight)
        score += signals.recruiter_response_rate * 0.40

        # Recency bonus — more recent = better
        if days_inactive <= 7:
            score += 0.12
        elif days_inactive <= 14:
            score += 0.09
        elif days_inactive <= 30:
            score += 0.06
        else:
            score += 0.03

        # Profile views (recruiter interest, 12% weight)
        score += min(1.0, signals.profile_views_received_30d / 50.0) * 0.12

        # GitHub activity — key proxy for "writes code" (12% weight)
        # Note: -1 means no GitHub profile linked (neutral — not a penalty)
        gh = signals.github_activity_score
        if gh > 0:
            score += min(1.0, gh / 50.0) * 0.12
        # gh == -1: no GitHub linked — neutral, no bonus no penalty

        # Interview completion rate (10% weight)
        score += signals.interview_completion_rate * 0.10

        # Saved by recruiters (8% weight)
        score += min(1.0, signals.saved_by_recruiters_30d / 10.0) * 0.08

        # Verified identity (small bonus)
        if signals.verified_email and signals.verified_phone:
            score += 0.03

        # Profile completeness
        completeness = getattr(signals, "profile_completeness_score", 0) or 0
        score += (completeness / 100.0) * 0.03

        return min(1.0, score)

    # ------------------------------------------------------------------
    # Signal 7: Honeypot Penalty
    # ------------------------------------------------------------------

    def extract_honeypot_penalty(self, candidate) -> float:
        penalty = 0.0
        signals = candidate.redrob_signals
        profile = candidate.profile
        years_exp = candidate.years_of_experience or 0.0

        # --- Rule 1: Tenure > total experience (impossible timeline) ---
        for career in candidate.career_history:
            dur_yrs = (getattr(career, "duration_months", 0) or 0) / 12.0
            if dur_yrs > years_exp + 1.5:
                penalty += 0.35
                break

        # --- Rule 2: Skill inflation (too many expert skills, tiny duration) ---
        expert_skills = [
            s for s in candidate.skills if s.proficiency == "expert"
        ]
        if len(expert_skills) > 8:
            short_expert = sum(
                1 for s in expert_skills
                if (s.duration_months or 0) < 12
            )
            if short_expert > 4:
                penalty += 0.40
        # Also flag zero-endorsement experts en masse
        zero_end_expert = sum(
            1 for s in expert_skills if s.endorsements == 0
        )
        if zero_end_expert > 5:
            penalty += 0.25

        # --- Rule 3: Company age vs tenure ---
        for career in candidate.career_history:
            company_lower = _normalise(career.company)
            dur_yrs = (getattr(career, "duration_months", 0) or 0) / 12.0
            for known_co, founded in KNOWN_FOUNDING_YEARS.items():
                if known_co in company_lower:
                    max_possible = CURRENT_YEAR - founded
                    if dur_yrs > max_possible + 0.5:
                        penalty += 0.50
                    break

        # --- Rule 4: Salary / experience mismatch ---
        sal_max = 0.0
        try:
            sal_max = signals.expected_salary_range_inr_lpa.get("max", 0) or 0
        except AttributeError:
            pass
        if years_exp < 3 and sal_max > 40:
            penalty += 0.30
        if years_exp < 5 and sal_max > 60:
            penalty += 0.35

        # --- Rule 5: Education timeline anomalies ---
        for edu in candidate.education:
            end_yr = getattr(edu, "end_year", 0) or 0
            start_yr = getattr(edu, "start_year", 0) or 0
            if end_yr > CURRENT_YEAR:
                penalty += 0.30
            # Degree takes > 8 years (impossible for UG / most PG)
            if start_yr > 0 and end_yr > 0 and (end_yr - start_yr) > 8:
                penalty += 0.20

        # --- Rule 6: Services background with AI keyword stuffing ---
        if _is_service_company(profile.get("current_company", "")):
            ai_skills = sum(
                1 for s in candidate.skills
                if any(kw in _normalise(s.name) for kw in [
                    "ai", "ml", "llm", "rag", "embedding", "vector",
                    "deep learning", "transformer"
                ])
            )
            if ai_skills > 6 and years_exp < 5:
                penalty += 0.30

        # --- Rule 7: Job hopper (> 4 companies in 6 years) ---
        recent_jobs = [
            c for c in candidate.career_history
            if (getattr(c, "duration_months", 0) or 0) > 0
        ]
        if len(recent_jobs) > 4:
            total_months = sum(
                (getattr(c, "duration_months", 0) or 0)
                for c in recent_jobs
            )
            avg_tenure = total_months / len(recent_jobs)
            if avg_tenure < 18:   # < 18 months average tenure
                penalty += 0.20

        return min(1.0, penalty)