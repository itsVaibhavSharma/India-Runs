#!/usr/bin/env python3
"""Pre-computation script — run once before ranking.

Tasks:
  1. Download & save the embedding model locally (models/embedding_model/)
  2. Pre-compute candidate embeddings (models/embeddings/)
  3. Train & save the signal fusion model (models/fusion/)

After running this script, python rank.py will work fully offline.

Usage:
  # Full precompute (all tasks):
  python precompute.py --candidates ../Dataset/candidates.jsonl --jd ../Dataset/job_description.docx

  # Embeddings only (if you already trained fusion):
  python precompute.py --candidates ../Dataset/candidates.jsonl --embeddings-only

  # Fusion only (e.g. to retrain after tuning signal weights):
  python precompute.py --jd ../Dataset/job_description.docx --fusion-only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ranker.candidate_loader import CandidateLoader
from ranker.embeddings import EmbeddingManager
from ranker.fusion import SignalFusion
from ranker.jd_parser import parse_jd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def download_model(cache_dir: str = "models") -> None:
    """Download and save sentence-transformer model locally for offline use."""
    model_dir = Path(cache_dir) / "embedding_model"
    if model_dir.exists() and any(model_dir.iterdir()):
        logger.info("Embedding model already cached at %s — skipping download.", model_dir)
        return

    logger.info("Downloading all-MiniLM-L6-v2 from HuggingFace…")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir))
    logger.info("Model saved to %s", model_dir)


def precompute_embeddings(
    candidates_path: str,
    cache_dir: str = "models",
    force_recompute: bool = False,
) -> None:
    """Compute and cache embeddings for all candidates."""
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Pre-computing candidate embeddings")
    logger.info("Input: %s", candidates_path)
    logger.info("=" * 60)

    loader = CandidateLoader(candidates_path)

    logger.info("Loading candidates…")
    candidates = loader.load_all_candidates()
    logger.info("  Loaded %d candidates.", len(candidates))

    emb_manager = EmbeddingManager(
        cache_dir=str(Path(cache_dir) / "embeddings"),
        model_dir=str(Path(cache_dir) / "embedding_model"),
    )
    embeddings, ids = emb_manager.get_candidate_embeddings(
        candidates, force_recompute=force_recompute
    )

    elapsed = time.time() - t0
    logger.info(
        "Embeddings complete: %d candidates | shape %s | %.1fs",
        len(ids), embeddings.shape, elapsed,
    )


def train_fusion_model(
    jd_path: str,
    cache_dir: str = "models",
) -> None:
    """Train and save the signal fusion model."""
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Training signal fusion model")
    logger.info("JD: %s", jd_path)
    logger.info("=" * 60)

    req = parse_jd(jd_path)
    logger.info(
        "JD parsed: %d must-have | %d nice-to-have",
        len(req.must_have_skills), len(req.nice_to_have_skills),
    )

    fusion = SignalFusion(model_dir=str(Path(cache_dir) / "fusion"))
    fusion.train(req)

    elapsed = time.time() - t0
    logger.info("Fusion model trained and saved in %.1fs", elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute embeddings and train models for offline ranking"
    )
    parser.add_argument(
        "--candidates",
        default="../Dataset/candidates.jsonl",
        help="Path to candidates.jsonl or .jsonl.gz",
    )
    parser.add_argument(
        "--jd",
        default="../Dataset/job_description.docx",
        help="Path to job description (.docx / .md / .txt)",
    )
    parser.add_argument(
        "--cache-dir",
        default="models",
        help="Base directory for all cached artifacts",
    )
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Only compute candidate embeddings (skip fusion training)",
    )
    parser.add_argument(
        "--fusion-only",
        action="store_true",
        help="Only train the fusion model (skip embedding computation)",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force re-embedding even if cache exists",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip model download (model already at models/embedding_model/)",
    )

    args = parser.parse_args()

    total_start = time.time()

    # Step 0: Download model (if needed)
    if not args.skip_download and not args.fusion_only:
        download_model(args.cache_dir)

    # Step 1: Candidate embeddings
    if not args.fusion_only:
        candidates_path = Path(args.candidates)
        if not candidates_path.exists():
            logger.error("Candidates file not found: %s", candidates_path)
            sys.exit(1)
        precompute_embeddings(
            str(candidates_path),
            args.cache_dir,
            force_recompute=args.force_recompute,
        )

    # Step 2: Fusion model
    if not args.embeddings_only:
        jd_path = Path(args.jd)
        if not jd_path.exists():
            logger.error("JD file not found: %s", jd_path)
            sys.exit(1)
        train_fusion_model(str(jd_path), args.cache_dir)

    total_elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info("All pre-computation complete in %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
    logger.info("You can now run: python rank.py --candidates <path> --out submission.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()