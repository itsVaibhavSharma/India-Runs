"""Main Ranker - Orchestrates the two-stage ranking pipeline."""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm

from .jd_parser import JDParser, parse_jd
from .candidate_loader import CandidateLoader, Candidate
from .signals import SignalExtractor
from .embeddings import EmbeddingManager
from .fusion import SignalFusion
from .reasoning import generate_reasoning

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Ranker:
    STAGE1_TOP_K = 500
    FINAL_TOP_K = 100

    def __init__(self, jd_path: str, candidates_path: str, cache_dir: str = 'models'):
        self.jd_path = jd_path
        self.candidates_path = candidates_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.jd_requirements = None
        self.jd_embedding = None
        self.candidates = []
        self.candidate_embeddings = None
        self.candidate_ids = None

    def run(self, output_path: str, force_recompute: bool = False):
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("Starting Redrob Candidate Ranking Pipeline")
        logger.info("=" * 60)

        # Stage 0: Parse JD
        logger.info("Stage 0: Parsing job description...")
        self.jd_requirements = parse_jd(self.jd_path)
        logger.info(f"  Extracted {len(self.jd_requirements.must_have_skills)} must-have skills")
        logger.info(f"  Experience range: {self.jd_requirements.required_experience_range}")

        # Stage 1: Load candidates and extract signals
        logger.info("Stage 1: Loading candidates and extracting signals...")
        loader = CandidateLoader(self.candidates_path)
        total_candidates = loader.count_candidates()
        logger.info(f"  Total candidates: {total_candidates}")

        signal_extractor = SignalExtractor(self.jd_requirements)
        embedding_manager = EmbeddingManager(self.cache_dir / 'embeddings')

        # Load or compute candidate embeddings
        logger.info("  Computing/loading candidate embeddings...")
        self.candidates = loader.load_all_candidates()
        self.candidate_embeddings, self.candidate_ids = embedding_manager.get_candidate_embeddings(
            self.candidates, force_recompute=force_recompute
        )

        # Compute JD embedding
        logger.info("  Computing JD embedding...")
        jd_text = self._build_jd_text()
        self.jd_embedding = embedding_manager.get_jd_embedding(jd_text)

        # Compute signal scores for all candidates
        logger.info("  Extracting signals for all candidates...")
        all_signals = []
        valid_candidates = []
        candidate_indices = []

        for idx, candidate in enumerate(tqdm(self.candidates, desc="Signal extraction")):
            signals = signal_extractor.extract_all_signals(candidate)
            # Apply behavioral gate early
            if signals.behavioral == 0.0 and signals.honeypot_penalty < 0.5:
                continue  # Skip unavailable candidates unless they have low honeypot penalty

            signal_dict = {
                'title_career': signals.title_career,
                'skill_depth': signals.skill_depth,
                'experience': signals.experience,
                'education': signals.education,
                'location': signals.location,
                'behavioral': signals.behavioral,
                'honeypot_penalty': signals.honeypot_penalty
            }
            all_signals.append(signal_dict)
            valid_candidates.append(candidate)
            candidate_indices.append(idx)

        logger.info(f"  Candidates passing behavioral gate: {len(valid_candidates)}")

        # Stage 2: Semantic similarity filtering
        logger.info("Stage 2: Semantic similarity filtering...")
        similarities = embedding_manager.compute_similarities(
            self.jd_embedding, self.candidate_embeddings
        )

        # Get top K by similarity from valid candidates
        valid_similarities = similarities[candidate_indices]
        top_k = min(self.STAGE1_TOP_K, len(valid_similarities))
        top_indices = np.argpartition(valid_similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(valid_similarities[top_indices])[::-1]]

        stage2_candidates = [valid_candidates[i] for i in top_indices]
        stage2_signals = [all_signals[i] for i in top_indices]
        stage2_similarities = valid_similarities[top_indices]

        logger.info(f"  Advanced to reranking: {len(stage2_candidates)}")

        # Stage 3: Signal fusion and reranking
        logger.info("Stage 3: Signal fusion and reranking...")
        fusion = SignalFusion(self.cache_dir / 'fusion')

        if not fusion.load():
            logger.info("  Training fusion model...")
            fusion.train(self.jd_requirements)

        final_scores = fusion.predict_batch(stage2_signals)

        # Combine with similarity (small weight)
        combined_scores = final_scores * 0.9 + stage2_similarities * 0.1

        # Stage 4: Honeypot penalty enforcement
        logger.info("Stage 4: Applying honeypot penalties...")
        for i, signals in enumerate(stage2_signals):
            if signals['honeypot_penalty'] > 0.3:
                combined_scores[i] *= (1.0 - signals['honeypot_penalty'])
            if signals['honeypot_penalty'] > 0.5:
                combined_scores[i] *= 0.1  # Heavy penalty for strong honeypots

        # Stage 5: Sort and generate output
        logger.info("Stage 5: Sorting and generating output...")
        ranked_indices = np.argsort(combined_scores)[::-1]
        final_candidates = [stage2_candidates[i] for i in ranked_indices[:self.FINAL_TOP_K]]
        final_scores = combined_scores[ranked_indices[:self.FINAL_TOP_K]]
        final_signals = [stage2_signals[i] for i in ranked_indices[:self.FINAL_TOP_K]]

        # Enforce monotonic scores
        final_scores = np.maximum.accumulate(final_scores[::-1])[::-1]

        # Generate reasoning
        logger.info("  Generating reasoning for top 100...")
        output_rows = []
        for rank, (candidate, score, signals) in enumerate(zip(final_candidates, final_scores, final_signals), 1):
            reasoning = generate_reasoning(candidate, signals, rank, float(score), self.jd_requirements)
            output_rows.append({
                'candidate_id': candidate.candidate_id,
                'rank': rank,
                'score': round(float(score), 6),
                'reasoning': reasoning
            })

        # Write CSV
        self._write_submission(output_rows, output_path)

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Ranking complete in {elapsed:.1f}s")
        logger.info(f"Output written to: {output_path}")
        logger.info("=" * 60)

    def _build_jd_text(self) -> str:
        req = self.jd_requirements
        parts = [
            "Senior AI Engineer Founding Team",
            "Production embeddings retrieval ranking vector search",
            "Python FAISS Pinecone Weaviate Qdrant Milvus",
            "Sentence-transformers BGE E5 semantic search",
            "Learning to rank NDCG MRR MAP A/B testing",
            "Product company experience shipped to real users",
            "5-9 years experience",
            "Pune Noida Hyderabad Mumbai Delhi NCR"
        ]
        if req.must_have_skills:
            parts.append(" ".join(req.must_have_skills))
        if req.nice_to_have_skills:
            parts.append(" ".join(req.nice_to_have_skills))
        return " ".join(parts)

    def _write_submission(self, rows: List[Dict], output_path: str):
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['candidate_id', 'rank', 'score', 'reasoning'])
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Redrob Candidate Ranker')
    parser.add_argument('--candidates', required=True, help='Path to candidates.jsonl or .gz')
    parser.add_argument('--jd', default='../Dataset/job_description.docx', help='Path to job description')
    parser.add_argument('--out', required=True, help='Output CSV path')
    parser.add_argument('--force-recompute', action='store_true', help='Force recompute embeddings')
    parser.add_argument('--cache-dir', default='models', help='Cache directory')

    args = parser.parse_args()

    ranker = Ranker(args.jd, args.candidates, args.cache_dir)
    ranker.run(args.out, force_recompute=args.force_recompute)


if __name__ == '__main__':
    main()