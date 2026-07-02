"""JD Parser — Extracts structured requirements from the job description.

Parses job_description.docx (or .md / .txt) and returns a JDRequirements
dataclass used by signal extractors and the fusion model trainer.

For the Senior AI Engineer role at Redrob, many requirements are hard-coded
here because the JD is fixed for this challenge — but the parser also reads
the actual document to pick up any numeric ranges or location preferences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# JD Requirements dataclass
# ---------------------------------------------------------------------------

@dataclass
class JDRequirements:
    must_have_skills: List[str] = field(default_factory=list)
    nice_to_have_skills: List[str] = field(default_factory=list)
    disqualifier_skills: List[str] = field(default_factory=list)
    required_experience_range: tuple = (4, 10)      # inclusive, flexible
    optimal_experience_range: tuple = (5, 9)         # ideal band
    required_titles: List[str] = field(default_factory=list)
    disqualifier_titles: List[str] = field(default_factory=list)
    preferred_locations: List[str] = field(default_factory=list)
    product_company_keywords: List[str] = field(default_factory=list)
    consulting_firms: List[str] = field(default_factory=list)
    required_tech_keywords: List[str] = field(default_factory=list)
    ranking_evaluation_keywords: List[str] = field(default_factory=list)
    disqualifier_patterns: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hard-coded knowledge from the Redrob JD (Senior AI Engineer — Founding Team)
# ---------------------------------------------------------------------------

_MUST_HAVE_SKILLS = [
    # Retrieval & embeddings
    "embeddings", "embedding", "vector search", "retrieval", "rag",
    "semantic search", "hybrid search", "dense retrieval", "bm25",
    "sentence-transformers", "sentence transformers", "bge", "e5",
    # Vector DBs
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch",
    # Ranking & evaluation
    "ranking", "ndcg", "mrr", "map", "a/b testing", "learning to rank",
    "ltr", "offline evaluation", "online evaluation",
    # Core engineering
    "python", "production", "deployed", "shipped",
]

_NICE_TO_HAVE_SKILLS = [
    "lora", "qlora", "peft", "fine-tuning", "fine-tuning llms", "finetuning",
    "xgboost", "lightgbm", "catboost", "lambdamart", "listnet", "ranklib",
    "hr tech", "recruiting", "marketplace",
    "distributed systems", "large-scale inference", "inference optimization",
    "open-source", "mlops", "model serving",
]

_DISQUALIFIER_SKILLS = [
    # Red flags: LLM-framework-only, recent bandwagon
    "langchain", "llamaindex", "llama_index", "haystack", "langgraph",
    "autogen", "crewai", "dspy", "guidance",
]

_REQUIRED_TITLES = [
    "ai engineer", "ml engineer", "machine learning engineer",
    "senior ai engineer", "senior ml engineer",
    "applied scientist", "applied ai", "applied ml",
    "ranking engineer", "search engineer", "recommendation engineer",
    "nlp engineer", "data scientist", "research engineer",
    "software engineer, ml", "software engineer, ai",
]

_DISQUALIFIER_TITLES = [
    "engineering manager", "vp engineering", "vp of engineering",
    "director of engineering", "head of engineering",
    "principal architect", "solution architect", "enterprise architect",
    "cto", "coo",
]

_PREFERRED_LOCATIONS = [
    "pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon",
    "gurugram", "bangalore", "bengaluru", "ncr", "national capital region",
    "navi mumbai", "thane",
]

_CONSULTING_FIRMS = [
    "tcs", "infosys", "wipro", "cognizant", "capgemini", "accenture",
    "hcl", "tech mahindra", "lti", "mindtree", "mphasis", "hexaware",
    "l&t infotech", "zensar", "ntt data", "cgi", "virtusa", "synechron",
    "ltimindtree", "persistent systems", "mastech",
]

_PRODUCT_KEYWORDS = [
    "product", "saas", "platform", "startup", "ai startup", "ai company",
    "fintech", "edtech", "healthtech", "ecommerce", "marketplace",
    "social", "gaming", "media", "streaming", "rideshare", "delivery",
    "logistics tech", "series a", "series b",
]

_RANKING_EVAL_KEYWORDS = [
    "ndcg", "mrr", "map", "precision@", "recall@", "hit rate",
    "offline evaluation", "online evaluation", "a/b test", "ab test",
    "ranking quality", "retrieval quality", "search quality",
    "learning to rank", "ltr", "lambdamart", "interleaving",
    "mean reciprocal rank", "average precision",
]

_REQUIRED_TECH = [
    "embeddings", "vector", "faiss", "pinecone", "weaviate", "qdrant",
    "milvus", "opensearch", "elasticsearch", "sentence-transformers",
    "bge", "e5", "semantic search", "hybrid search",
    "ndcg", "mrr", "ltr", "learning to rank",
]


# ---------------------------------------------------------------------------
# JDParser
# ---------------------------------------------------------------------------

class JDParser:
    """Parses a job description file and returns JDRequirements."""

    def __init__(self, jd_path: Optional[str] = None):
        self.jd_text = ""
        if jd_path:
            self.load_jd(jd_path)

    def load_jd(self, jd_path: str) -> str:
        path = Path(jd_path)
        if not path.exists():
            raise FileNotFoundError(f"JD file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".docx":
            try:
                import docx
                doc = docx.Document(path)
                self.jd_text = "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                raise ImportError(
                    "python-docx is required to read .docx files. "
                    "Install it with: pip install python-docx"
                )
        elif suffix in (".md", ".txt"):
            self.jd_text = path.read_text(encoding="utf-8")
        else:
            raise ValueError(f"Unsupported JD format: {suffix}")

        return self.jd_text

    def parse(self) -> JDRequirements:
        text_lower = self.jd_text.lower()

        # ---- Skills ----
        must_have = [s for s in _MUST_HAVE_SKILLS if s in text_lower]
        # If JD text available but empty (fallback), use full list
        if not must_have:
            must_have = list(_MUST_HAVE_SKILLS)

        nice_to_have = [s for s in _NICE_TO_HAVE_SKILLS if s in text_lower]
        if not nice_to_have:
            nice_to_have = list(_NICE_TO_HAVE_SKILLS)

        disqualifier = [s for s in _DISQUALIFIER_SKILLS if s in text_lower]
        if not disqualifier:
            disqualifier = list(_DISQUALIFIER_SKILLS)

        # ---- Experience range (from text) ----
        exp_range = (4, 10)
        optimal = (5, 9)
        match = re.search(r"(\d+)\s*[–\-]\s*(\d+)\s*years?", text_lower)
        if match:
            lo, hi = int(match.group(1)), int(match.group(2))
            optimal = (lo, hi)
            exp_range = (max(1, lo - 1), hi + 1)

        # ---- Titles ----
        req_titles = [t for t in _REQUIRED_TITLES if t in text_lower]
        if not req_titles:
            req_titles = list(_REQUIRED_TITLES)

        disq_titles = [t for t in _DISQUALIFIER_TITLES if t in text_lower]
        if not disq_titles:
            disq_titles = list(_DISQUALIFIER_TITLES)

        # ---- Locations ----
        pref_loc = [l for l in _PREFERRED_LOCATIONS if l in text_lower]
        if not pref_loc:
            pref_loc = list(_PREFERRED_LOCATIONS)

        return JDRequirements(
            must_have_skills=must_have,
            nice_to_have_skills=nice_to_have,
            disqualifier_skills=disqualifier,
            required_experience_range=exp_range,
            optimal_experience_range=optimal,
            required_titles=req_titles,
            disqualifier_titles=disq_titles,
            preferred_locations=pref_loc,
            product_company_keywords=list(_PRODUCT_KEYWORDS),
            consulting_firms=list(_CONSULTING_FIRMS),
            required_tech_keywords=_REQUIRED_TECH,
            ranking_evaluation_keywords=_RANKING_EVAL_KEYWORDS,
            disqualifier_patterns=[
                r"pure\s+research",
                r"no\s+production",
                r"langchain\s+only",
            ],
        )

    def generate_synthetic_pairs(
        self,
        requirements: JDRequirements,
        n_pairs: int = 300,
    ) -> List[tuple]:
        """Generate (positive_desc, negative_desc) text pairs for fusion training."""
        import random
        random.seed(42)

        product_companies = [
            "Swiggy", "Zomato", "Flipkart", "Ola", "PhonePe", "Razorpay",
            "CRED", "Meesho", "Unacademy", "Dream11", "Nykaa", "PolicyBazaar",
            "Zerodha", "Groww", "Slice", "Jar", "Fi", "Jupiter", "Khatabook",
            "ShareChat", "Rapido", "Porter", "Dunzo", "Licious", "BharatPe",
            "Udaan", "BigBasket", "Urban Company",
        ]
        consulting_companies = [
            "TCS", "Infosys", "Wipro", "Cognizant", "Capgemini",
            "Accenture", "HCL", "Tech Mahindra", "LTI", "Mindtree", "Mphasis",
        ]
        req_techs = requirements.must_have_skills[:8] or ["FAISS", "embeddings"]
        nice_techs = requirements.nice_to_have_skills[:5] or ["XGBoost", "LoRA"]
        disq_techs = requirements.disqualifier_skills[:3] or ["LangChain"]

        positive_templates = [
            "{title} at {company} ({yr} yrs); built embedding-based retrieval using "
            "{tech1} + {tech2}; deployed to production at scale; "
            "evaluates with NDCG/MRR; strong Python; active on platform.",
            "Applied Scientist at {company} with {yr} yrs; shipped vector search "
            "pipeline with {tech1}; learning-to-rank with XGBoost; "
            "A/B testing in production; product company.",
            "Senior ML Engineer at {company} ({yr} yrs); {tech1} + {tech2} "
            "in production; ranking systems for real users; "
            "evaluation frameworks: NDCG, MAP; open to work.",
            "ML Engineer at {company}; {yr} years; semantic search with "
            "{tech1}; recommendation engine deployed; "
            "GitHub activity high; Pune/Noida location.",
            "{title} ({yr} yrs); built hybrid search (BM25 + dense) with "
            "{tech1}; RAG pipeline in production; fine-tuning with LoRA; "
            "A/B tested ranking improvements.",
        ]

        negative_templates = [
            "Operations Manager at {company} ({yr} yrs); uses {disq} for "
            "report automation; no production ML; services background.",
            "Research Scientist at {company} ({yr} yrs); pure academic "
            "research, no deployment; {disq} demos; no GitHub activity.",
            "Tech Lead at {company}; {yr} yrs; management role, no code "
            "written in 18 months; {disq} chatbot project.",
            "Consultant at {company} ({yr} yrs); services background; "
            "AI keywords but no production; {disq} RAG demo.",
            "Junior Dev at {company}; {yr} yrs; only {disq} side projects; "
            "no ranking eval experience; not open to work.",
            "Backend Engineer at {company} ({yr} yrs); Kafka + Spark only; "
            "no embedding or retrieval experience; services company.",
            "HR Manager at {company}; {yr} yrs; uses ChatGPT for job posts; "
            "claims AI experience; no technical depth.",
        ]

        pairs = []
        for _ in range(n_pairs // 2):
            # Positive
            co = random.choice(product_companies)
            yr = random.randint(4, 9)
            t1 = random.choice(req_techs) if req_techs else "FAISS"
            t2 = random.choice(req_techs) if req_techs else "embeddings"
            title = random.choice([
                "Senior ML Engineer", "AI Engineer", "Applied Scientist",
                "ML Engineer", "Senior AI Engineer", "Ranking Engineer",
            ])
            pos = random.choice(positive_templates).format(
                company=co, yr=yr, tech1=t1, tech2=t2, title=title
            )

            # Negative
            neg_co = random.choice(consulting_companies + product_companies)
            neg_yr = random.randint(1, 15)
            disq = random.choice(disq_techs) if disq_techs else "LangChain"
            neg = random.choice(negative_templates).format(
                company=neg_co, yr=neg_yr, disq=disq
            )

            pairs.append((pos, neg))

        return pairs


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def parse_jd(jd_path: str) -> JDRequirements:
    """Parse a JD file and return structured requirements."""
    parser = JDParser(jd_path)
    return parser.parse()