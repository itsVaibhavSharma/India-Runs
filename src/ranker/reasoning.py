"""Reasoning Generator - Creates specific, factual reasoning for each ranked candidate."""

from typing import Dict, List, Any
from datetime import datetime


SERVICE_INDUSTRIES = {
    'it services', 'consulting', 'outsourcing', 'system integrator',
    'managed services', 'professional services', 'staffing'
}

CONSULTING_FIRMS = {
    'tcs', 'infosys', 'wipro', 'cognizant', 'capgemini', 'accenture',
    'hcl', 'tech mahindra', 'lti', 'mindtree', 'mphasis', 'hexaware',
    'l&t infotech', 'zensar', 'ntt data', 'cgi', 'virtusa', 'synechron'
}

REQUIRED_TECH_KEYWORDS = {
    'embeddings', 'vector search', 'rag', 'retrieval', 'ranking',
    'sentence-transformers', 'bge', 'e5', 'faiss', 'pinecone', 'weaviate',
    'qdrant', 'milvus', 'opensearch', 'elasticsearch', 'hybrid search',
    'semantic search', 'dense retrieval', 'sparse retrieval', 'bm25',
    'learning to rank', 'ltr', 'ndcg', 'mrr', 'map', 'a/b testing',
    'python', 'production', 'deployed', 'shipped', 'scale', 'real users'
}

NICE_TECH_KEYWORDS = {
    'lora', 'qlora', 'peft', 'fine-tuning', 'xgboost', 'lightgbm',
    'catboost', 'lambdamart', 'listnet', 'ranklib', 'hr tech',
    'recruiting', 'marketplace', 'distributed systems', 'inference optimization'
}


class ReasoningGenerator:
    def __init__(self):
        pass

    def generate(self, candidate, signals: Dict[str, float], rank: int, score: float) -> str:
        parts = []

        # 1. Title/Company/Experience fact
        p = candidate.profile
        company_type = "services" if self._is_service_company(p.get('current_company', '')) else "product"
        parts.append(
            f"{p.get('current_title', 'N/A')} at {p.get('current_company', 'N/A')} "
            f"({company_type}) with {p.get('years_of_experience', 0):.1f} yrs"
        )

        # 2. Technical evidence from career descriptions
        tech_evidence = self._extract_tech_evidence(candidate)
        if tech_evidence:
            parts.append(tech_evidence)

        # 3. Behavioral signals
        behavioral = self._format_behavioral(candidate.redrob_signals)
        if behavioral:
            parts.append(behavioral)

        # 4. Honest concerns
        concerns = self._format_concerns(candidate, signals)
        if concerns:
            parts.append(f"Concern: {concerns}")

        reasoning = ". ".join(parts) + "."
        return reasoning

    def _extract_tech_evidence(self, candidate) -> str:
        evidence = []

        # Check career descriptions for required tech
        for career in candidate.career_history:
            desc = career.description.lower()
            found = []

            for kw in REQUIRED_TECH_KEYWORDS:
                if kw in desc:
                    found.append(kw)

            for kw in NICE_TECH_KEYWORDS:
                if kw in desc:
                    found.append(kw)

            if found:
                # Deduplicate and take top 4
                unique = list(dict.fromkeys(found))[:4]
                evidence.append(f"built {', '.join(unique)} at {career.company}")

        # Check skills for required tech
        skill_evidence = []
        for skill in candidate.skills:
            name = skill.name.lower()
            if name in REQUIRED_TECH_KEYWORDS and skill.endorsements > 0:
                skill_evidence.append(f"{skill.name} ({skill.endorsements} endorsements, {skill.duration_months}mo)")
            elif name in NICE_TECH_KEYWORDS and skill.endorsements > 0:
                skill_evidence.append(f"{skill.name} ({skill.endorsements} endorsements)")

        if skill_evidence:
            evidence.append(f"skills: {', '.join(skill_evidence[:3])}")

        # GitHub activity
        if candidate.redrob_signals.github_activity_score > 0:
            evidence.append(f"GitHub activity {candidate.redrob_signals.github_activity_score:.0f}")

        # Skill assessments
        assessments = candidate.redrob_signals.skill_assessment_scores
        if assessments:
            top_assess = sorted(assessments.items(), key=lambda x: x[1], reverse=True)[:2]
            evidence.append(f"assessments: {', '.join(f'{k} {v:.0f}' for k, v in top_assess)}")

        return "; ".join(evidence[:3]) if evidence else ""

    def _format_behavioral(self, signals) -> str:
        parts = []

        # Recency
        try:
            last_active = datetime.fromisoformat(signals.last_active_date)
            days_ago = (datetime.now() - last_active).days
            if days_ago <= 7:
                parts.append(f"active {days_ago}d ago")
            elif days_ago <= 30:
                parts.append(f"active {days_ago}d ago")
            else:
                parts.append(f"last active {days_ago}d ago")
        except (ValueError, TypeError):
            pass

        # Response rate
        if signals.recruiter_response_rate > 0:
            parts.append(f"response rate {signals.recruiter_response_rate:.0%}")

        # Recruiter interest
        if signals.saved_by_recruiters_30d > 0:
            parts.append(f"saved by {signals.saved_by_recruiters_30d} recruiters (30d)")

        # Interview completion
        if signals.interview_completion_rate > 0:
            parts.append(f"interview completion {signals.interview_completion_rate:.0%}")

        # Open to work
        if signals.open_to_work_flag:
            parts.append("open to work")

        return "; ".join(parts) if parts else ""

    def _format_concerns(self, candidate, signals: Dict[str, float]) -> str:
        concerns = []

        # Notice period
        if candidate.redrob_signals.notice_period_days > 60:
            concerns.append(f"{candidate.redrob_signals.notice_period_days}-day notice")

        # Services background
        if self._is_service_company(candidate.profile.get('current_company', '')):
            concerns.append("services background")

        # Location mismatch
        loc = candidate.profile.get('location', '').lower()
        preferred = ['pune', 'noida', 'hyderabad', 'mumbai', 'delhi', 'gurgaon', 'bangalore', 'chennai']
        if not any(p in loc for p in preferred) and not candidate.redrob_signals.willing_to_relocate:
            concerns.append("location mismatch")

        # Python proficiency
        python_skill = next((s for s in candidate.skills if s.name.lower() == 'python'), None)
        if not python_skill or python_skill.proficiency not in ('advanced', 'expert'):
            concerns.append("Python not advanced")

        # Honeypot penalty
        if signals.get('honeypot_penalty', 0) > 0.3:
            concerns.append("profile anomalies detected")

        # Low behavioral score
        if signals.get('behavioral', 1) < 0.3:
            concerns.append("low engagement signals")

        # Inactive
        try:
            last_active = datetime.fromisoformat(candidate.redrob_signals.last_active_date)
            if (datetime.now() - last_active).days > 60:
                concerns.append("inactive >60d")
        except (ValueError, TypeError):
            pass

        return "; ".join(concerns) if concerns else ""

    def _is_service_company(self, company: str) -> bool:
        company_lower = company.lower()
        return any(firm in company_lower for firm in CONSULTING_FIRMS)

    def _is_service_industry(self, industry: str) -> bool:
        industry_lower = industry.lower()
        return any(svc in industry_lower for svc in SERVICE_INDUSTRIES)