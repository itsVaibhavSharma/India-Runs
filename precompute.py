#!/usr/bin/env python3
"""
Pre-computation script for embeddings and fusion model.
Run once before ranking to cache embeddings and train the fusion model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ranker import (
    CandidateLoader, EmbeddingManager, SignalFusion, parse_jd
)


def precompute_embeddings(candidates_path: str, cache_dir: str = 'models'):
    """Pre-compute embeddings for all candidates."""
    print("=" * 60)
    print("Pre-computing candidate embeddings...")
    print("=" * 60)

    loader = CandidateLoader(candidates_path)
    total = loader.count_candidates()
    print(f"Total candidates: {total}")

    candidates = loader.load_all_candidates()
    print(f"Loaded {len(candidates)} candidates")

    mgr = EmbeddingManager(Path(cache_dir) / 'embeddings')
    embeddings, ids = mgr.get_candidate_embeddings(candidates, force_recompute=True)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Cached to: {mgr.CACHE_DIR}")
    print("Done!")
    return embeddings, ids


def train_fusion_model(jd_path: str, cache_dir: str = 'models'):
    """Train the signal fusion model on synthetic JD-derived pairs."""
    print("=" * 60)
    print("Training fusion model...")
    print("=" * 60)

    req = parse_jd(jd_path)
    fusion = SignalFusion(Path(cache_dir) / 'fusion')
    fusion.train(req)

    print("Fusion model trained and cached!")
    return fusion


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Pre-compute embeddings and train models')
    parser.add_argument('--candidates', default='../Dataset/candidates.jsonl',
                       help='Path to candidates.jsonl or .gz')
    parser.add_argument('--jd', default='../Dataset/job_description.docx',
                       help='Path to job description')
    parser.add_argument('--cache-dir', default='models', help='Cache directory')
    parser.add_argument('--embeddings-only', action='store_true', help='Only compute embeddings')
    parser.add_argument('--fusion-only', action='store_true', help='Only train fusion model')

    args = parser.parse_args()

    if not args.fusion_only:
        precompute_embeddings(args.candidates, args.cache_dir)

    if not args.embeddings_only:
        train_fusion_model(args.jd, args.cache_dir)

    print("\n✅ All pre-computation complete!")


if __name__ == '__main__':
    main()