"""Redrob Ranker Package — Intelligent Candidate Discovery & Ranking.

Team: ThreeTwoOne
Leader: Vaibhav Sharma
Members: Vaibhav Sharma, Shreya Khantal
"""

from .jd_parser import JDParser, JDRequirements, parse_jd
from .candidate_loader import (
    CandidateLoader, Candidate, Skill, CareerEntry,
    EducationEntry, RedrobSignals,
)
from .signals import SignalExtractor, SignalScores, skill_trust_score
from .embeddings import EmbeddingManager, load_embeddings_from_cache
from .fusion import SignalFusion
from .reasoning import ReasoningGenerator, generate_reasoning
from .__main__ import Ranker, run_ranking

__version__ = "1.0.0"
__author__ = "ThreeTwoOne"

__all__ = [
    # JD
    "JDParser", "JDRequirements", "parse_jd",
    # Candidates
    "CandidateLoader", "Candidate", "Skill", "CareerEntry",
    "EducationEntry", "RedrobSignals",
    # Signals
    "SignalExtractor", "SignalScores", "skill_trust_score",
    # Embeddings
    "EmbeddingManager", "load_embeddings_from_cache",
    # Fusion
    "SignalFusion",
    # Reasoning
    "ReasoningGenerator", "generate_reasoning",
    # Pipeline
    "Ranker", "run_ranking",
]