"""JD Parser - Extracts key requirements from job description."""

import re
from dataclasses import dataclass
from typing import List, Set, Dict, Optional
from pathlib import Path


@dataclass
class JDRequirements:
    must_have_skills: List[str]
    nice_to_have_skills: List[str]
    disqualifier_skills: List[str]
    required_experience_range: tuple
    required_titles: List[str]
    disqualifier_titles: List[str]
    preferred_locations: List[str]
    product_company_keywords: List[str]
    consulting_firms: List[str]
    required_tech_keywords: List[str]
    ranking_evaluation_keywords: List[str]


class JDParser:
    CONSULTING_FIRMS = {
        'tcs', 'infosys', 'wipro', 'cognizant', 'capgemini', 'accenture',
        'hcl', 'tech mahindra', 'lti', 'mindtree', 'mphasis', 'hexaware',
        'l&t infotech', 'zensar', 'ntt data', 'cgi', 'virtusa', 'synechron'
    }

    PRODUCT_INDUSTRY_KEYWORDS = {
        'product', 'saas', 'platform', 'startup', 'ai', 'ml', 'machine learning',
        'fintech', 'edtech', 'healthtech', 'ecommerce', 'marketplace', 'social',
        'gaming', 'media', 'streaming', 'rideshare', 'delivery', 'logistics tech'
    }

    MUST_HAVE_SKILLS = [
        'embeddings', 'vector search', 'rag', 'retrieval', 'ranking',
        'sentence-transformers', 'bge', 'e5', 'faiss', 'pinecone', 'weaviate',
        'qdrant', 'milvus', 'opensearch', 'elasticsearch', 'hybrid search',
        'semantic search', 'dense retrieval', 'sparse retrieval', 'bm25',
        'learning to rank', 'ltr', 'ndcg', 'mrr', 'map', 'a/b testing',
        'python', 'production', 'deployed', 'shipped', 'scale', 'real users'
    ]

    NICE_TO_HAVE_SKILLS = [
        'lora', 'qlora', 'peft', 'fine-tuning', 'xgboost', 'lightgbm',
        'catboost', 'lambdamart', 'listnet', 'ranklib', 'hr tech',
        'recruiting', 'marketplace', 'distributed systems', 'inference optimization'
    ]

    DISQUALIFIER_SKILLS = [
        'langchain', 'llama_index', 'haystack', 'langgraph', 'autogen',
        'crewai', 'dspy', 'guidance', 'instructor'
    ]

    REQUIRED_TITLES = [
        'ai engineer', 'ml engineer', 'machine learning engineer',
        'senior ai engineer', 'senior ml engineer', 'applied scientist',
        'ranking engineer', 'search engineer', 'recommendation engineer',
        'nlp engineer', 'data scientist', 'research engineer'
    ]

    DISQUALIFIER_TITLES = [
        'architect', 'tech lead', 'engineering manager', 'vp engineering',
        'director', 'head of', 'principal architect', 'staff engineer'
    ]

    PREFERRED_LOCATIONS = [
        'pune', 'noida', 'hyderabad', 'mumbai', 'delhi', 'gurgaon',
        'bangalore', 'bengaluru', 'chennai', 'ncr', 'national capital region'
    ]

    RANKING_EVAL_KEYWORDS = [
        'ndcg', 'mrr', 'map', 'precision@', 'recall@', 'hit rate',
        'offline evaluation', 'online evaluation', 'a/b test', 'ab test',
        'ranking quality', 'retrieval quality', 'search quality'
    ]

    def __init__(self, jd_path: Optional[str] = None):
        self.jd_text = ""
        if jd_path:
            self.load_jd(jd_path)

    def load_jd(self, jd_path: str) -> str:
        path = Path(jd_path)
        if path.suffix.lower() == '.docx':
            import docx
            doc = docx.Document(path)
            self.jd_text = '\n'.join([p.text for p in doc.paragraphs])
        elif path.suffix.lower() in ['.md', '.txt']:
            self.jd_text = path.read_text(encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        return self.jd_text

    def parse(self) -> JDRequirements:
        text_lower = self.jd_text.lower()

        must_have = [s for s in self.MUST_HAVE_SKILLS if s in text_lower]
        nice_to_have = [s for s in self.NICE_TO_HAVE_SKILLS if s in text_lower]
        disqualifier = [s for s in self.DISQUALIFIER_SKILLS if s in text_lower]

        exp_match = re.search(r'(\d+)[\-\s]*(\d+)\s*years?', text_lower)
        if exp_match:
            exp_range = (int(exp_match.group(1)), int(exp_match.group(2)))
        else:
            exp_range = (4, 10)

        req_titles = [t for t in self.REQUIRED_TITLES if t in text_lower]
        disq_titles = [t for t in self.DISQUALIFIER_TITLES if t in text_lower]

        pref_loc = [l for l in self.PREFERRED_LOCATIONS if l in text_lower]

        return JDRequirements(
            must_have_skills=must_have,
            nice_to_have_skills=nice_to_have,
            disqualifier_skills=disqualifier,
            required_experience_range=exp_range,
            required_titles=req_titles,
            disqualifier_titles=disq_titles,
            preferred_locations=pref_loc,
            product_company_keywords=list(self.PRODUCT_INDUSTRY_KEYWORDS),
            consulting_firms=list(self.CONSULTING_FIRMS),
            required_tech_keywords=must_have,
            ranking_evaluation_keywords=self.RANKING_EVAL_KEYWORDS
        )

    def generate_synthetic_pairs(self, requirements: JDRequirements, n_pairs: int = 200) -> List[tuple]:
        import random
        pairs = []

        positive_templates = [
            "Senior ML Engineer at {company} (product) with {years} years; built embedding-based retrieval using {tech}; strong Python; ships to production; evaluates with NDCG/MRR",
            "AI Engineer at {company} with {years} yrs; shipped vector search with {tech} at scale; ranking eval experience; product company",
            "Applied Scientist at {company} ({years} yrs); production RAG with {tech}; ranking infrastructure; strong coding",
            "ML Engineer at {company}; {years} years; embedding models + {tech}; learning-to-rank with XGBoost; deployed to users"
        ]

        negative_templates = [
            "Marketing Manager at {company} with {years} yrs; skills: {disq_tech}; no production ML",
            "Research Scientist at {company} ({years} yrs); pure research, no deployment; {disq_tech}",
            "Tech Lead at {company}; {years} yrs; management only, no coding recently; {disq_tech}",
            "Consultant at {company} ({years} yrs); services background; keyword-stuffed {disq_tech}",
            "HR Manager at {company}; {years} yrs; AI keywords but non-technical role; {disq_tech}"
        ]

        product_companies = ['Swiggy', 'Zomato', 'Flipkart', 'Ola', 'PhonePe', 'Razorpay', 'Cred', 'Meesho', 'Unacademy', 'Byju', 'Dream11', 'Nykaa', 'PolicyBazaar', 'Zerodha', 'Groww', 'Slice', 'Jar', 'Fi', 'Jupiter', 'Khatabook']
        consulting_companies = ['TCS', 'Infosys', 'Wipro', 'Cognizant', 'Capgemini', 'Accenture', 'HCL', 'Tech Mahindra', 'LTI', 'Mindtree', 'Mphasis']

        all_tech = requirements.must_have_skills + requirements.nice_to_have_skills
        disq_tech = requirements.disqualifier_skills

        for _ in range(n_pairs // 2):
            template = random.choice(positive_templates)
            company = random.choice(product_companies)
            years = random.randint(4, 9)
            tech = random.choice(all_tech) if all_tech else 'FAISS'
            pos = template.format(company=company, years=years, tech=tech)

            neg_template = random.choice(negative_templates)
            neg_company = random.choice(consulting_companies + product_companies)
            neg_years = random.randint(2, 15)
            disq = random.choice(disq_tech) if disq_tech else 'LangChain'
            neg = neg_template.format(company=neg_company, years=neg_years, disq_tech=disq)

            pairs.append((pos, neg))

        return pairs


def parse_jd(jd_path: str) -> JDRequirements:
    parser = JDParser(jd_path)
    return parser.parse()