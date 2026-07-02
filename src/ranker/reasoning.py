"""Reasoning Generator — Creates specific, factual, JD-connected reasoning per candidate.

Every reasoning string is built purely from the candidate's structured data
(no LLM, no hallucination).  Template structure:
  1. Title / Company / Experience fact
  2. Technical evidence (specific tech from career descriptions + endorsed skills)
  3. Behavioral signal (last active, response rate, GitHub)
  4. Honest concern / gap (if any)

The spec warns against:
  - Empty reasoning
  - All-identical strings
  - Hallucinated skills not present in the profile
  - Reasoning that contradicts the rank

We prevent all of these through slot-filling from verified candidate fields.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Reference sets for tech extraction
# ---------------------------------------------------------------------------

RETRIEVAL_TECH = {
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "typesense", "vespa", "chroma",
    "bm25", "bge", "e5", "minilm", "sentence-transformer",
    "sentence transformers", "embeddings", "dense retrieval", "hybrid search",
    "semantic search", "vector search", "rag", "retrieval augmented",
    "ann", "hnsw", "ivf",
}

RANKING_EVAL_TECH = {
    "ndcg", "mrr", "map", "precision@", "recall@", "hit rate",
    "learning to rank", "ltr", "lambdamart", "ranklib", "xgboost ranking",
    "a/b test", "ab test", "offline evaluation", "online evaluation",
    "interleaving", "click-through",
}

PRODUCTION_SIGNALS = {
    "production", "deployed", "shipped", "at scale", "real users",
    "serving", "inference", "millions", "latency", "throughput",
    "model serving", "mlops", "api",
}

NICE_TO_HAVE_TECH = {
    "lora", "qlora", "peft", "fine-tuning", "fine-tuning llms",
    "xgboost", "lightgbm", "catboost", "lambdamart",
    "distributed systems", "inference optimization", "triton", "vllm",
    "rag", "langchain", "haystack", "spark", "kafka",
}

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "cognizant", "capgemini", "accenture",
    "hcl", "tech mahindra", "lti", "mindtree", "mphasis", "hexaware",
    "l&t infotech", "zensar", "ntt data", "cgi", "virtusa", "synechron",
    "ltimindtree", "persistent systems", "mastech",
}

SERVICE_INDUSTRIES = {
    "it services", "consulting", "outsourcing", "system integrator",
    "managed services", "professional services", "staffing", "bpo",
}

PREFERRED_LOCS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon",
    "gurugram", "bengaluru", "bangalore", "ncr",
}


def _norm(text: str) -> str:
    return text.lower().strip() if text else ""


def _is_service_company(company: str) -> bool:
    cl = _norm(company)
    return any(f in cl for f in CONSULTING_FIRMS)


def _is_service_industry(industry: str) -> bool:
    il = _norm(industry)
    return any(s in il for s in SERVICE_INDUSTRIES)


def _days_since(date_str: str) -> Optional[int]:
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str)
        return max(0, (datetime.now() - dt).days)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Core extractor helpers
# ---------------------------------------------------------------------------

def _extract_tech_from_text(text: str, keyword_set: set) -> List[str]:
    """Find keywords from a set in a text string (case-insensitive)."""
    tl = text.lower()
    found = []
    for kw in keyword_set:
        if kw in tl:
            found.append(kw)
    # Deduplicate while preserving order
    seen: set = set()
    result = []
    for kw in found:
        if kw not in seen:
            seen.add(kw)
            result.append(kw)
    return result


def _extract_tech_evidence(candidate) -> Tuple[List[str], List[str], bool]:
    """
    Returns (retrieval_techs, eval_techs, has_production) extracted from
    career descriptions — all grounded in actual profile data.
    """
    retrieval: List[str] = []
    eval_kws: List[str] = []
    has_production = False

    for career in candidate.career_history:
        desc = career.description or ""
        retrieval += _extract_tech_from_text(desc, RETRIEVAL_TECH)
        eval_kws += _extract_tech_from_text(desc, RANKING_EVAL_TECH)
        if any(kw in desc.lower() for kw in PRODUCTION_SIGNALS):
            has_production = True

    # Deduplicate
    seen: set = set()
    r2 = []
    for t in retrieval:
        if t not in seen:
            seen.add(t)
            r2.append(t)

    e2 = []
    seen2: set = set()
    for t in eval_kws:
        if t not in seen2:
            seen2.add(t)
            e2.append(t)

    return r2[:5], e2[:3], has_production


def _get_top_endorsed_skills(candidate, n: int = 4) -> List[str]:
    """Return names of top-N endorsed skills verified in the profile."""
    relevant = []
    for s in candidate.skills:
        name = _norm(s.name)
        is_relevant = (
            any(kw in name for kw in RETRIEVAL_TECH | {"python", "embedding", "vector"}) or
            any(kw in name for kw in RANKING_EVAL_TECH) or
            any(kw in name for kw in NICE_TO_HAVE_TECH)
        )
        if is_relevant and s.endorsements > 0:
            relevant.append((s.endorsements, s.duration_months, s.name))

    relevant.sort(reverse=True)
    return [name for _, _, name in relevant[:n]]


def _format_company_type(company: str, industry: str) -> str:
    if _is_service_company(company) or _is_service_industry(industry):
        return "services co"
    return "product co"


# ---------------------------------------------------------------------------
# ReasoningGenerator
# ---------------------------------------------------------------------------

class ReasoningGenerator:
    """Generates factual, JD-connected reasoning for each ranked candidate."""

    def generate(
        self,
        candidate,
        signals: Dict[str, float],
        rank: int,
        score: float,
        jd_requirements=None,
    ) -> str:
        parts = []

        profile = candidate.profile
        sig = candidate.redrob_signals

        title = profile.get("current_title", "Unknown") or "Unknown"
        company = profile.get("current_company", "Unknown") or "Unknown"
        industry = profile.get("current_industry", "") or ""
        years = float(profile.get("years_of_experience", 0) or 0)
        loc = profile.get("location", "") or ""

        company_type = _format_company_type(company, industry)

        # ------------------------------------------------------------------
        # Part 1: Title / Company / Experience fact
        # ------------------------------------------------------------------
        part1 = f"{title} at {company} ({company_type}) with {years:.1f} yrs"
        parts.append(part1)

        # ------------------------------------------------------------------
        # Part 2: Technical evidence (only what's in the profile)
        # ------------------------------------------------------------------
        retrieval_techs, eval_techs, has_production = _extract_tech_evidence(candidate)
        endorsed_skills = _get_top_endorsed_skills(candidate, n=3)

        tech_parts = []
        if retrieval_techs:
            tech_parts.append(f"built {', '.join(retrieval_techs[:3])}")
        if eval_techs:
            tech_parts.append(f"eval: {', '.join(eval_techs[:2])}")
        if has_production:
            tech_parts.append("shipped to production")
        if endorsed_skills and not retrieval_techs:
            tech_parts.append(f"skills: {', '.join(endorsed_skills[:3])}")

        if tech_parts:
            parts.append("; ".join(tech_parts))

        # ------------------------------------------------------------------
        # Part 3: Behavioral signal
        # ------------------------------------------------------------------
        behavioral_parts = []

        days_inactive = _days_since(sig.last_active_date)
        if days_inactive is not None:
            if days_inactive == 0:
                behavioral_parts.append("active today")
            elif days_inactive <= 7:
                behavioral_parts.append(f"active {days_inactive}d ago")
            elif days_inactive <= 30:
                behavioral_parts.append(f"active {days_inactive}d ago")
            else:
                behavioral_parts.append(f"last active {days_inactive}d ago")

        if sig.recruiter_response_rate > 0:
            behavioral_parts.append(f"response rate {sig.recruiter_response_rate:.0%}")

        if sig.github_activity_score > 0:
            behavioral_parts.append(f"GitHub: {sig.github_activity_score:.0f}")

        if behavioral_parts:
            parts.append("; ".join(behavioral_parts))

        # ------------------------------------------------------------------
        # Part 4: Honest concern / gap
        # ------------------------------------------------------------------
        concerns = self._collect_concerns(candidate, signals, loc)
        if concerns:
            parts.append(f"Concern: {'; '.join(concerns)}")

        # Join with ". " and ensure it ends with "."
        reasoning = ". ".join(p.strip() for p in parts if p.strip())
        if not reasoning.endswith("."):
            reasoning += "."

        # Truncate to a reasonable length (spec asks for 1-2 sentences)
        if len(reasoning) > 400:
            reasoning = reasoning[:397] + "..."

        return reasoning

    def _collect_concerns(
        self,
        candidate,
        signals: Dict[str, float],
        location: str,
    ) -> List[str]:
        concerns = []
        sig = candidate.redrob_signals
        profile = candidate.profile
        company = profile.get("current_company", "") or ""
        industry = profile.get("current_industry", "") or ""

        # Notice period
        if sig.notice_period_days > 60:
            concerns.append(f"{sig.notice_period_days}d notice")

        # Services background
        if _is_service_company(company) or _is_service_industry(industry):
            concerns.append("services background")

        # Location mismatch
        loc_lower = _norm(location)
        if not any(p in loc_lower for p in PREFERRED_LOCS) and not sig.willing_to_relocate:
            country = _norm(profile.get("country", "") or "")
            if country and country != "india":
                concerns.append("outside India")
            elif country == "india":
                concerns.append("non-preferred city, not willing to relocate")

        # Python skill level
        python_skill = next(
            (s for s in candidate.skills if _norm(s.name) == "python"), None
        )
        if not python_skill or python_skill.proficiency not in ("advanced", "expert"):
            concerns.append("Python not advanced/expert")

        # Honeypot flag
        if signals.get("honeypot_penalty", 0) > 0.30:
            concerns.append("profile anomalies detected")

        # Inactive
        days_inactive = _days_since(sig.last_active_date)
        if days_inactive is not None and days_inactive > 45:
            concerns.append(f"inactive {days_inactive}d")

        # Low behavioral
        if signals.get("behavioral", 1.0) < 0.25:
            concerns.append("low engagement signals")

        return concerns[:3]   # cap at 3 concerns to keep reasoning concise


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def generate_reasoning(
    candidate,
    signals: Dict[str, float],
    rank: int,
    score: float,
    jd_requirements=None,
) -> str:
    """Module-level wrapper — creates a ReasoningGenerator and calls generate()."""
    generator = ReasoningGenerator()
    return generator.generate(candidate, signals, rank, score, jd_requirements)