"""Redrob Ranker Package - Intelligent Candidate Discovery & Ranking."""

from . import jd_parser, candidate_loader, signals, embeddings, fusion, reasoning, __main__

# Public API
from .jd_parser import JDParser, JDRequirements, parse_jd
from .candidate_loader import CandidateLoader, Candidate, Skill, CareerEntry, EducationEntry, RedrobSignals
from .signals import SignalExtractor, SignalScores
from .embeddings import EmbeddingManager
from .fusion import SignalFusion
from .reasoning import generate_reasoning
from .__main__ import run_ranking

__version__ = "1.0.0"
__author__ = "ThreeTwoOne"
__all__ = [
    'JDParser', 'JDRequirements', 'parse_jd',
    'CandidateLoader', 'Candidate', 'Skill', 'CareerEntry', 'EducationEntry', 'RedrobSignals',
    'SignalExtractor', 'SignalScores',
    'EmbeddingManager',
    'SignalFusion',
    'generate_reasoning',
    'run_ranking',
]