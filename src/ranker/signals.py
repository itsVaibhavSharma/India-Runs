"""Signal Extractors - Compute 7 independent signal scores for each candidate."""

from dataclasses import dataclass
from typing import List, Dict, Set, Optional
import re
from datetime import datetime
import math


@dataclass
class SignalScores:
    title_career: float = 0.0
    skill_depth: float = 0.0
    experience: float = 0.0
    education: float = 0.0
    location: float = 0.0
    behavioral: float = 0.0
    honeypot_penalty: float = 0.0

    def to_list(self) -> List[float]:
        return [
            self.title_career,
            self.skill_depth,
            self.experience,
            self.education,
            self.location,
            self.behavioral,
            -self.honeypot_penalty  # Negative for penalty
        ]


class SignalExtractor:
    REQUIRED_SKILLS = {
        'embeddings', 'vector search', 'rag', 'retrieval', 'ranking',
        'sentence-transformers', 'bge', 'e5', 'faiss', 'pinecone', 'weaviate',
        'qdrant', 'milvus', 'opensearch', 'elasticsearch', 'hybrid search',
        'semantic search', 'dense retrieval', 'sparse retrieval', 'bm25',
        'learning to rank', 'ltr', 'ndcg', 'mrr', 'map', 'a/b testing',
        'python', 'production', 'deployed', 'shipped', 'scale', 'real users'
    }

    NICE_TO_HAVE_SKILLS = {
        'lora', 'qlora', 'peft', 'fine-tuning', 'xgboost', 'lightgbm',
        'catboost', 'lambdamart', 'listnet', 'ranklib', 'hr tech',
        'recruiting', 'marketplace', 'distributed systems', 'inference optimization'
    }

    RANKING_KEYWORDS = {
        'ndcg', 'mrr', 'map', 'precision@', 'recall@', 'hit rate',
        'offline evaluation', 'online evaluation', 'a/b test', 'ab test',
        'ranking quality', 'retrieval quality', 'search quality',
        'learning to rank', 'ltr', 'lambdamart', 'ranklib', 'xgboost ranking'
    }

    TARGET_TITLES = {
        'ai engineer', 'ml engineer', 'machine learning engineer',
        'senior ai engineer', 'senior ml engineer', 'applied scientist',
        'ranking engineer', 'search engineer', 'recommendation engineer',
        'nlp engineer', 'data scientist', 'research engineer'
    }

    PRODUCTION_KEYWORDS = {
        'production', 'deployed', 'shipped', 'real users', 'scale',
        'serving', 'inference', 'latency', 'throughput', 'api', 'endpoint',
        'model serving', 'mlops', 'ci/cd', 'pipeline', 'monitoring'
    }

    SERVICE_FIRMS = {
        'tcs', 'infosys', 'wipro', 'cognizant', 'capgemini', 'accenture',
        'hcl', 'tech mahindra', 'lti', 'mindtree', 'mphasis', 'hexaware',
        'l&t infotech', 'zensar', 'ntt data', 'cgi', 'virtusa', 'synechron'
    }

    TIER1_INSTITUTIONS = {
        'iit', 'iisc', 'iit bombay', 'iit delhi', 'iit madras', 'iit kanpur',
        'iit kharagpur', 'iit roorkee', 'iit guwahati', 'iit hyderabad',
        'bits pilani', 'bits goa', 'bits hyderabad', 'nit trichy', 'nit surathkal',
        'nit warangal', 'nit calicut', 'nit rourkela', 'nit kurukshetra',
        'iiit hyderabad', 'iiit bangalore', 'iiit delhi', 'anna university',
        'dtu', 'nsut', 'iiit allahabad', 'jadavpur university'
    }

    PREFERRED_LOCATIONS = {
        'pune', 'noida', 'hyderabad', 'mumbai', 'delhi', 'gurgaon',
        'bangalore', 'bengaluru', 'chennai', 'ncr', 'national capital region',
        'navi mumbai', 'thane', 'ghaziabad', 'faridabad', 'ghaziabad'
    }

    RELOCATION_FRIENDLY = {
        'hyderabad', 'mumbai', 'delhi', 'gurgaon', 'bangalore', 'bengaluru',
        'chennai', 'pune', 'noida', 'ncr', 'national capital region'
    }

    def __init__(self, jd_requirements=None):
        self.jd_requirements = jd_requirements
        if jd_requirements:
            self._update_from_jd(jd_requirements)

    def _update_from_jd(self, req):
        if req.must_have_skills:
            self.REQUIRED_SKILLS.update(set(req.must_have_skills))
        if req.nice_to_have_skills:
            self.NICE_TO_HAVE_SKILLS.update(set(req.nice_to_have_skills))
        if req.required_titles:
            self.TARGET_TITLES.update(set(req.required_titles))
        if req.preferred_locations:
            self.PREFERRED_LOCATIONS.update(set(req.preferred_locations))

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

    def extract_title_career_score(self, candidate) -> float:
        score = 0.0

        current_title = candidate.current_title.lower()
        current_company = candidate.current_company.lower()
        current_industry = candidate.current_industry.lower()

        # Current title match
        title_match = 0.0
        for target in self.TARGET_TITLES:
            if target in current_title:
                title_match = 1.0
                break
            # Partial match
            target_words = target.split()
            if any(word in current_title for word in target_words if len(word) > 3):
                title_match = max(title_match, 0.6)

        score += title_match * 0.35

        # Career history evidence
        career_evidence = 0.0
        has_production = False
        has_ranking = False
        has_product_experience = False

        for career in candidate.career_history:
            desc = career.description.lower()
            company = career.company.lower()
            title = career.title.lower()

            # Production deployment evidence
            if any(kw in desc for kw in self.PRODUCTION_KEYWORDS):
                has_production = True
                career_evidence += 0.15

            # Ranking/retrieval experience
            if any(kw in desc for kw in self.RANKING_KEYWORDS):
                has_ranking = True
                career_evidence += 0.2

            # Product company experience
            if not self.is_service_company(company) and not self.is_service_industry(career.industry):
                has_product_experience = True
                career_evidence += 0.1

            # Title progression
            if any(target in title for target in self.TARGET_TITLES):
                career_evidence += 0.1

        career_evidence = min(1.0, career_evidence)
        score += career_evidence * 0.4

        # Company type bonus/penalty
        if self.is_service_company(current_company) or self.is_service_industry(current_industry):
            score -= 0.2  # Penalty for services
        elif has_product_experience:
            score += 0.15  # Bonus for product experience

        # Must have production + ranking for top score
        if not has_production:
            score *= 0.7
        if not has_ranking:
            score *= 0.8

        return max(0.0, min(1.0, score))

    def extract_skill_depth_score(self, candidate) -> float:
        if not candidate.skills:
            return 0.0

        total_score = 0.0
        required_hits = 0
        nice_hits = 0

        for skill in candidate.skills:
            skill_name = skill.name.lower()
            trust = candidate.get_skill_trust_score(skill)

            if skill_name in self.REQUIRED_SKILLS:
                total_score += trust * 3.0  # 3x multiplier for required
                required_hits += 1
            elif skill_name in self.NICE_TO_HAVE_SKILLS:
                total_score += trust * 1.5
                nice_hits += 1
            else:
                total_score += trust * 0.5

        # Normalize by expected max
        max_possible = len(self.REQUIRED_SKILLS) * 3.0 + len(self.NICE_TO_HAVE_SKILLS) * 1.5
        normalized = total_score / max_possible if max_possible > 0 else 0.0

        # Bonus for covering required skills
        required_coverage = required_hits / len(self.REQUIRED_SKILLS) if self.REQUIRED_SKILLS else 0
        normalized += required_coverage * 0.2

        return min(1.0, normalized)

    def extract_experience_score(self, candidate) -> float:
        years = candidate.years_of_experience
        min_exp, max_exp = 4, 10

        if min_exp <= years <= max_exp:
            return 1.0
        elif years < min_exp:
            return max(0.3, years / min_exp)
        else:
            # Decay for overqualified
            excess = years - max_exp
            return max(0.4, 1.0 - excess * 0.05)

    def extract_education_score(self, candidate) -> float:
        if not candidate.education:
            return 0.2

        max_score = 0.0
        for edu in candidate.education:
            inst = edu.institution.lower()
            field = edu.field_of_study.lower()
            tier = edu.tier.lower()

            score = 0.0

            # Tier scoring
            if tier == 'tier_1':
                score += 0.6
            elif tier == 'tier_2':
                score += 0.4
            elif tier == 'tier_3':
                score += 0.25
            else:
                score += 0.1

            # Institution name check
            if any(t1 in inst for t1 in self.TIER1_INSTITUTIONS):
                score += 0.2

            # Field relevance
            if any(f in field for f in ['computer', 'machine learning', 'ai', 'data', 'statistics', 'math']):
                score += 0.15

            # Degree level
            if 'ph.d' in edu.degree.lower() or 'phd' in edu.degree.lower():
                score += 0.1
            elif 'master' in edu.degree.lower() or 'm.tech' in edu.degree.lower() or 'm.e.' in edu.degree.lower():
                score += 0.05

            max_score = max(max_score, score)

        return min(1.0, max_score)

    def extract_location_score(self, candidate) -> float:
        location = candidate.location.lower()
        country = candidate.country.lower()
        signals = candidate.redrob_signals

        # Direct location match
        for pref in self.PREFERRED_LOCATIONS:
            if pref in location:
                return 1.0

        # Country level - India preferred
        if country == 'india':
            # Check if willing to relocate to preferred
            if signals.willing_to_relocate:
                return 0.6
            return 0.4

        # Other countries
        if signals.willing_to_relocate:
            return 0.3

        return 0.1

    def extract_behavioral_score(self, candidate) -> float:
        signals = candidate.redrob_signals

        # Availability gate - hard filter
        if not signals.open_to_work_flag:
            return 0.0

        # Recency check
        try:
            last_active = datetime.fromisoformat(signals.last_active_date)
            days_inactive = (datetime.now() - last_active).days
            if days_inactive > 60:
                return 0.0
        except (ValueError, TypeError):
            return 0.0

        # Response rate threshold
        if signals.recruiter_response_rate < 0.1:
            return 0.0

        # If passed gate, compute score
        score = 0.0
        score += signals.recruiter_response_rate * 0.35
        score += min(1.0, signals.profile_views_received_30d / 50.0) * 0.2
        score += min(1.0, max(0, signals.github_activity_score) / 50.0) * 0.2
        score += signals.interview_completion_rate * 0.15
        score += min(1.0, signals.saved_by_recruiters_30d / 10.0) * 0.1

        return min(1.0, score)

    def extract_honeypot_penalty(self, candidate) -> float:
        penalty = 0.0
        signals = candidate.redrob_signals
        profile = candidate.profile

        # 1. Timeline impossibility: tenure > experience
        for career in candidate.career_history:
            if career.duration_months / 12 > profile.years_of_experience + 1:
                penalty += 0.3
                break

        # 2. Skill inflation: too many expert skills with short duration
        expert_skills = [s for s in candidate.skills if s.proficiency == 'expert']
        if len(expert_skills) > 8:
            short_expert = sum(1 for s in expert_skills if s.duration_months < 12)
            if short_expert > 5:
                penalty += 0.4

        # 3. Company age vs tenure (known founding years)
        known_founding = {
            'openai': 2015, 'anthropic': 2021, 'deepmind': 2010,
            'hugging face': 2016, 'cohere': 2019, 'adept': 2022,
            'character.ai': 2021, 'inflection': 2022, 'mistral': 2023
        }
        for career in candidate.career_history:
            company_lower = career.company.lower()
            for known, year in known_founding.items():
                if known in company_lower:
                    max_possible = 2026 - year
                    if career.duration_months / 12 > max_possible + 1:
                        penalty += 0.5
                    break

        # 4. Salary/experience mismatch
        exp = profile.years_of_experience
        sal_max = signals.expected_salary_range_inr_lpa.get('max', 0)
        if exp < 3 and sal_max > 40:
            penalty += 0.3
        if exp < 5 and sal_max > 60:
            penalty += 0.4

        # 5. Education timeline issues
        for edu in candidate.education:
            if edu.end_year > 2026:
                penalty += 0.3
            if edu.start_year < 1980 and profile.years_of_experience < 30:
                penalty += 0.2

        # 6. Services background with AI keyword stuffing
        if self.is_service_company(profile.current_company):
            ai_skills = sum(1 for s in candidate.skills
                          if any(kw in s.name.lower() for kw in ['ai', 'ml', 'llm', 'rag', 'embedding', 'vector']))
            if ai_skills > 5 and profile.years_of_experience < 5:
                penalty += 0.3

        return min(1.0, penalty)

    def is_service_company(self, company: str) -> bool:
        company_lower = company.lower()
        return any(firm in company_lower for firm in self.SERVICE_FIRMS)

    def is_service_industry(self, industry: str) -> bool:
        industry_lower = industry.lower()
        return any(svc in industry_lower for svc in {'it services', 'consulting', 'outsourcing'})