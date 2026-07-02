"""Reasoning Generator - Creates specific, factual, JD-connected reasoning for each candidate."""

from typing import Dict, List, Any
from datetime import datetime
import re


class ReasoningGenerator:
    TECH_KEYWORDS = {
        'embeddings', 'vector search', 'rag', 'retrieval', 'ranking', 'faiss', 'pinecone',
        'weaviate', 'qdrant', 'milvus', 'opensearch', 'elasticsearch', 'sentence-transformers',
        'bge', 'e5', 'hybrid search', 'semantic search', 'bm25', 'learning to rank',
        'ndcg', 'mrr', 'map', 'xgboost', 'lightgbm', 'lambdamart', 'lora', 'qlora', 'peft',
        'fine-tuning', 'python', 'production', 'deployed', 'shipped', 'scale', 'mlops'
    }

    SERVICE_INDUSTRIES = {
        'it services', 'consulting', 'outsourcing', 'system integrator'
    }

    SERVICE_FIRMS = {
        'tcs', 'infosys', 'wipro', 'cognizant', 'capgemini', 'accenture',
        'hcl', 'tech mahindra', 'lti', 'mindtree', 'mphasis', 'hexaware'
    }

    def __init__(self, jd_requirements=None):
        self.jd_requirements = jd_requirements

    def generate(self, candidate, signals: Dict[str, float], rank: int, score: float) -> str:
        parts = []

        # 1. Title/Company/Experience
        p = candidate.profile
        company_type = self._get_company_type(p.current_company, p.current_industry)
        parts.append(f"{p.current_title} at {p.current_company} ({company_type}) with {p.years_of_experience:.1f} yrs")

        # 2. Technical evidence from career
        tech_evidence = self._extract_tech_evidence(candidate)
        if tech_evidence:
            parts.append(tech_evidence)

        # 3. Key skill matches
        skill_evidence = self._extract_skill_evidence(candidate)
        if skill_evidence:
            parts.append(skill_evidence)

        # 4. Behavioral signals
        behavioral = self._extract_behavioral_evidence(candidate)
        if behavioral:
            parts.append(behavioral)

        # 5. Concerns/gaps (honest)
        concerns = self._extract_concerns(candidate)
        if concerns:
            parts.append(f"Concern: {concerns}")

        reasoning = ". ".join(parts) + "."

        # Ensure length is reasonable (1-2 sentences as per spec)
        if len(reasoning) > 300:
            reasoning = self._trim_reasoning(reasoning)

        return reasoning

    def _get_company_type(self, company: str, industry: str) -> str:
        company_lower = company.lower()
        industry_lower = industry.lower()

        if any(firm in company_lower for firm in self.SERVICE_FIRMS):
            return 'services'
        if any(svc in industry_lower for svc in self.SERVICE_INDUSTRIES):
            return 'services'
        return 'product'

    def _extract_tech_evidence(self, candidate) -> str:
        evidence = []
        for career in candidate.career_history:
            desc = career.description.lower()
            found = []
            for kw in self.TECH_KEYWORDS:
                if kw in desc:
                    found.append(kw)
            if found:
                # Deduplicate and limit
                unique = list(dict.fromkeys(found))[:4]
                evidence.append(f"built {', '.join(unique)}")
                break  # Only most recent relevant role

        if evidence:
            return "; ".join(evidence)
        return ""

    def _extract_skill_evidence(self, candidate) -> str:
        matched = []
        for skill in candidate.skills:
            skill_lower = skill.name.lower()
            if skill_lower in self.TECH_KEYWORDS and skill.endorsements > 0:
                trust = "high" if skill.endorsements > 10 and skill.duration_months > 12 else "moderate"
                matched.append(f"{skill.name} ({trust} trust)")

        if matched:
            return f"skills: {', '.join(matched[:5])}"
        return ""

    def _extract_behavioral_evidence(self, candidate) -> str:
        s = candidate.redrob_signals
        parts = []

        if s.last_active_date:
            try:
                last = datetime.fromisoformat(s.last_active_date)
                days = (datetime.now() - last).days
                if days < 7:
                    parts.append("very recent activity")
                elif days < 30:
                    parts.append(f"active {days}d ago")
                elif days < 60:
                    parts.append(f"last active {days}d ago")
            except:
                pass

        if s.recruiter_response_rate > 0:
            rate = s.recruiter_response_rate
            if rate > 0.5:
                parts.append(f"high response rate ({rate:.0%})")
            elif rate > 0.2:
                parts.append(f"moderate response rate ({rate:.0%})")

        if s.github_activity_score > 20:
            parts.append(f"GitHub activity ({s.github_activity_score:.0f})")
        elif s.github_activity_score > 0:
            parts.append("some GitHub activity")

        if s.saved_by_recruiters_30d > 5:
            parts.append(f"saved by {s.saved_by_recruiters_30d} recruiters")

        return "; ".join(parts) if parts else ""

    def _extract_concerns(self, candidate) -> str:
        concerns = []
        s = candidate.redrob_signals
        p = candidate.profile

        if s.notice_period_days > 60:
            concerns.append(f"{s.notice_period_days}-day notice")

        if self._get_company_type(p.current_company, p.current_industry) == 'services':
            concerns.append("services background")

        python_skill = next((sk for sk in candidate.skills if 'python' in sk.name.lower()), None)
        if not python_skill or python_skill.proficiency not in ('advanced', 'expert'):
            concerns.append("Python not advanced")

        if p.years_of_experience < 4:
            concerns.append("junior for role")

        if s.recruiter_response_rate < 0.15:
            concerns.append("low recruiter response")

        if s.github_activity_score == -1:
            concerns.append("no GitHub linked")

        # Check for required tech in career
        has_required_tech = False
        for career in candidate.career_history:
            desc = career.description.lower()
            if any(kw in desc for kw in ['embedding', 'vector', 'faiss', 'pinecone', 'retrieval', 'ranking', 'rag']):
                has_required_tech = True
                break
        if not has_required_tech:
            concerns.append("no embedding/retrieval production evidence")

        return "; ".join(concerns) if concerns else ""

    def _trim_reasoning(self, reasoning: str) -> str:
        sentences = re.split(r'\.\s+', reasoning)
        if len(sentences) > 2:
            return ". ".join(sentences[:2]) + "."
        return reasoning


def generate_reasoning(candidate, signals, rank, score, jd_requirements=None) -> str:
    generator = ReasoningGenerator(jd_requirements)
    return generator.generate(candidate, signals, rank, score)