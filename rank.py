#!/usr/bin/env python3
"""
Redrob Candidate Ranker — Main Entry Point

Usage:
    python rank.py --candidates path/to/candidates.jsonl --out submission.csv
    python rank.py --candidates path/to/candidates.jsonl --jd path/to/jd.docx --out submission.csv

Pre-computation (run once before ranking):
    python precompute.py --candidates path/to/candidates.jsonl --jd path/to/jd.docx
"""

# ── Suppress TF/GPU BEFORE any ML import ─────────────────────────────────────
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ranker import run_ranking

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Redrob Candidate Ranker')
    parser.add_argument('--candidates', required=True, help='Path to candidates.jsonl or .gz')
    parser.add_argument('--jd', default='../Dataset/job_description.docx', help='Path to job description')
    parser.add_argument('--out', default='submission.csv', help='Output CSV path')
    parser.add_argument('--cache-dir', default='models', help='Cache directory for embeddings/models')
    parser.add_argument('--force-recompute', action='store_true', help='Force recompute embeddings')
    parser.add_argument('--top-k', type=int, default=100, help='Number of candidates to rank (default 100)')
    parser.add_argument('--stage1-k', type=int, default=500, help='Stage 1 candidates to advance (default 500)')

    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    jd_path = Path(args.jd)
    out_path = Path(args.out)

    if not candidates_path.exists():
        logger.error(f"Candidates file not found: {candidates_path}")
        sys.exit(1)

    if not jd_path.exists():
        logger.error(f"JD file not found: {jd_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Redrob Intelligent Candidate Discovery & Ranking")
    logger.info("Team: ThreeTwoOne | Leader: Vaibhav Sharma")
    logger.info("=" * 60)

    start_time = time.time()

    try:
        run_ranking(
            candidates_path=str(candidates_path),
            jd_path=str(jd_path),
            output_path=str(out_path),
            cache_dir=args.cache_dir,
            force_recompute=args.force_recompute,
            final_top_k=args.top_k,
            stage1_top_k=args.stage1_k
        )

        elapsed = time.time() - start_time
        logger.info(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        # Quick validation
        validate_output(out_path)

    except Exception as e:
        logger.error(f"Ranking failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def validate_output(csv_path: Path):
    """Quick validation of output format."""
    import csv

    logger.info(f"Validating output: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if header != ['candidate_id', 'rank', 'score', 'reasoning']:
        logger.warning(f"Header mismatch: {header}")
        return

    if len(rows) != 100:
        logger.warning(f"Expected 100 rows, got {len(rows)}")
        return

    ranks = [int(r[1]) for r in rows]
    if set(ranks) != set(range(1, 101)):
        logger.warning(f"Rank issues: missing/duplicate ranks")
        return

    scores = [float(r[2]) for r in rows]
    for i in range(len(scores) - 1):
        if scores[i] < scores[i + 1] - 1e-9:
            logger.warning(f"Score not monotonic at rank {i+1}: {scores[i]} < {scores[i+1]}")
            return

    logger.info("✅ Output validation passed!")


if __name__ == '__main__':
    main()