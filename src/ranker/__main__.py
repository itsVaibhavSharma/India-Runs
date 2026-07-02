"""Main Ranker Pipeline — Two-stage candidate ranking for the Redrob challenge.

Stage 0: Parse JD → structured requirements
Stage 1: Load all 100K candidates, extract 7 signals (NO hard behavioral gate)
Stage 2: Top-500 by composite (50% semantic + 15% title + 12% skill + 10% exp + 8% behavioral)
Stage 3: Signal fusion (LogisticRegression + IsotonicRegression) + calibration
Stage 4: Honeypot penalty enforcement
Stage 5: Sort by score desc + candidate_id asc, generate reasoning, write submission.csv

All computation is CPU-only.  No network calls are made during ranking.
Pre-computed artifacts (candidate embeddings, fusion model) are loaded
from the local models/ directory.
"""

from __future__ import annotations

# ── Suppress TensorFlow / GPU imports BEFORE any ML library is loaded ────────
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# ─────────────────────────────────────────────────────────────────────────────

import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .jd_parser import JDRequirements, parse_jd
from .candidate_loader import Candidate, CandidateLoader
from .signals import SignalExtractor, SignalScores
from .embeddings import EmbeddingManager
from .fusion import SignalFusion
from .reasoning import generate_reasoning

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STAGE1_TOP_K = 500      # candidates that advance to stage 2
_FINAL_TOP_K = 100       # candidates in the submission
_SIM_WEIGHT = 0.10       # weight for semantic similarity in final score
_FUSION_WEIGHT = 0.90    # weight for fusion score in final score


# ---------------------------------------------------------------------------
# JD text builder (used to compute JD embedding)
# ---------------------------------------------------------------------------

def _build_jd_query_text(req: JDRequirements) -> str:
    """Build a rich query string from JD requirements for semantic search."""
    parts = [
        "Senior AI Engineer Founding Team production ranking retrieval",
        "embeddings vector search semantic search hybrid search",
        "sentence-transformers FAISS Pinecone Weaviate Qdrant Milvus",
        "NDCG MRR MAP A/B testing learning to rank evaluation",
        "Python production deployment real users scale",
        "5-9 years experience product company startup",
        "Pune Noida Hyderabad Mumbai Delhi NCR",
    ]
    if req.must_have_skills:
        parts.append(" ".join(req.must_have_skills[:10]))
    if req.nice_to_have_skills:
        parts.append(" ".join(req.nice_to_have_skills[:5]))
    return " ".join(parts)


def _days_since(date_str: Optional[str]) -> int:
    """Return number of days since *date_str* (ISO-8601). Returns 9999 on failure."""
    if not date_str:
        return 9999
    try:
        import datetime
        d = datetime.date.fromisoformat(str(date_str)[:10])
        return max(0, (datetime.date.today() - d).days)
    except Exception:
        return 9999


# ---------------------------------------------------------------------------
# Ranker class
# ---------------------------------------------------------------------------

class Ranker:
    """Orchestrates the two-stage ranking pipeline."""

    def __init__(
        self,
        jd_path: str,
        candidates_path: str,
        cache_dir: str = "models",
    ):
        self.jd_path = str(jd_path)
        self.candidates_path = str(candidates_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.jd_requirements: Optional[JDRequirements] = None

    # ------------------------------------------------------------------
    # Main pipeline entry point
    # ------------------------------------------------------------------

    def run(
        self,
        output_path: str,
        force_recompute: bool = False,
        final_top_k: int = _FINAL_TOP_K,
        stage1_top_k: int = _STAGE1_TOP_K,
    ) -> None:
        wall_start = time.time()
        logger.info("=" * 65)
        logger.info("  Redrob Intelligent Candidate Ranking Pipeline")
        logger.info("  Team: ThreeTwoOne | Vaibhav Sharma & Shreya Khantal")
        logger.info("=" * 65)

        # ----------------------------------------------------------------
        # Stage 0: Parse JD
        # ----------------------------------------------------------------
        logger.info("[Stage 0] Parsing job description: %s", self.jd_path)
        self.jd_requirements = parse_jd(self.jd_path)
        logger.info(
            "  must-have skills: %d | nice-to-have: %d | exp: %s",
            len(self.jd_requirements.must_have_skills),
            len(self.jd_requirements.nice_to_have_skills),
            self.jd_requirements.optimal_experience_range,
        )

        # ----------------------------------------------------------------
        # Stage 1: Load candidates & extract signals
        # ----------------------------------------------------------------
        logger.info("[Stage 1] Loading candidates from: %s", self.candidates_path)
        loader = CandidateLoader(self.candidates_path)

        # Load or compute candidate embeddings
        emb_manager = EmbeddingManager(
            cache_dir=str(self.cache_dir / "embeddings"),
            model_dir=str(self.cache_dir / "embedding_model"),
        )

        # Load all candidates (needed for embedding alignment)
        all_candidates: List[Candidate] = []
        logger.info("  Reading all candidates (this may take ~30s for 100K)…")
        all_candidates = loader.load_all_candidates()
        logger.info("  Loaded %d candidates.", len(all_candidates))

        # Get embeddings (from cache or compute)
        logger.info("  Fetching candidate embeddings…")
        candidate_embeddings, candidate_ids = emb_manager.get_candidate_embeddings(
            all_candidates, force_recompute=force_recompute
        )
        # Build id→index map for alignment
        id_to_idx: Dict[str, int] = {cid: i for i, cid in enumerate(candidate_ids)}

        # ----------------------------------------------------------------
        # Stage 1b: Signal extraction (ALL candidates — no hard gate)
        # ----------------------------------------------------------------
        logger.info("[Stage 1b] Extracting signals for all candidates…")
        signal_extractor = SignalExtractor(self.jd_requirements)

        all_signals_list: List[Dict[str, float]] = []
        all_scores_list:  List[SignalScores] = []
        valid_candidates: List[Candidate] = []
        valid_emb_indices: List[int] = []

        n_hard_excluded = 0
        for candidate in tqdm(all_candidates, desc="Signal extraction", unit="cand"):
            scores: SignalScores = signal_extractor.extract_all_signals(candidate)

            # Hard-exclude only extreme honeypots (fabricated profiles)
            if scores.honeypot_penalty >= 0.95:
                n_hard_excluded += 1
                continue

            # Hard-exclude completely ghost candidates:
            # open_to_work=False AND inactive >180d AND response_rate=0
            sig = candidate.redrob_signals
            days_inactive = _days_since(sig.last_active_date) or 9999
            is_ghost = (
                not sig.open_to_work_flag
                and days_inactive > 180
                and sig.recruiter_response_rate == 0.0
            )
            if is_ghost:
                n_hard_excluded += 1
                continue

            idx = id_to_idx.get(candidate.candidate_id, -1)
            if idx == -1:
                continue

            valid_candidates.append(candidate)
            all_signals_list.append(scores.to_dict())
            all_scores_list.append(scores)
            valid_emb_indices.append(idx)

        logger.info(
            "  Signals extracted: %d / %d  (hard-excluded: %d honeypots/ghosts)",
            len(valid_candidates), len(all_candidates), n_hard_excluded,
        )

        # ----------------------------------------------------------------
        # Stage 2: Semantic similarity + signal composite → top-K
        # ----------------------------------------------------------------
        logger.info("[Stage 2] Semantic similarity filtering (top %d)…", stage1_top_k)
        jd_text = _build_jd_query_text(self.jd_requirements)
        jd_embedding = emb_manager.get_jd_embedding(jd_text)

        valid_emb_matrix = candidate_embeddings[valid_emb_indices]
        similarities = emb_manager.compute_similarities(jd_embedding, valid_emb_matrix)

        # Stage-1 composite score:
        #   50% semantic similarity (JD query match)
        #   15% title/career signal  (most discriminative for role fit)
        #   12% skill depth          (endorsement-weighted tech skills)
        #   10% experience           (years-of-experience fit)
        #    8% behavioral           (availability/engagement — softer now)
        #    5% honeypot penalty     (negative)
        title_arr    = np.array([s["title_career"]     for s in all_signals_list], dtype=np.float32)
        skill_arr    = np.array([s["skill_depth"]       for s in all_signals_list], dtype=np.float32)
        exp_arr      = np.array([s["experience"]        for s in all_signals_list], dtype=np.float32)
        behav_arr    = np.array([s["behavioral"]        for s in all_signals_list], dtype=np.float32)
        hp_arr       = np.array([s["honeypot_penalty"]  for s in all_signals_list], dtype=np.float32)

        quick_score = (
            similarities * 0.50 +
            title_arr    * 0.15 +
            skill_arr    * 0.12 +
            exp_arr      * 0.10 +
            behav_arr    * 0.08 -
            hp_arr       * 0.05
        )

        # Select top-K by quick composite
        top_k   = min(stage1_top_k, len(quick_score))
        top_idx = np.argpartition(quick_score, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(quick_score[top_idx])[::-1]]

        stage2_candidates = [valid_candidates[i] for i in top_idx]
        stage2_signals    = [all_signals_list[i]  for i in top_idx]
        stage2_sims       = similarities[top_idx]

        logger.info("  Advanced to stage-2 reranking: %d", len(stage2_candidates))


        # ----------------------------------------------------------------
        # Stage 3: Signal fusion + calibration
        # ----------------------------------------------------------------
        logger.info("[Stage 3] Signal fusion and score calibration…")
        fusion = SignalFusion(model_dir=str(self.cache_dir / "fusion"))

        if not fusion.load():
            logger.info("  Fusion model not cached — training from JD-derived pairs…")
            fusion.train(self.jd_requirements)

        fusion_scores = fusion.predict_batch(stage2_signals)

        # Combine: fusion is primary, semantic similarity is a tiebreaker
        combined_scores = (
            fusion_scores * _FUSION_WEIGHT +
            stage2_sims   * _SIM_WEIGHT
        )

        # ----------------------------------------------------------------
        # Stage 4: Honeypot penalty enforcement
        # ----------------------------------------------------------------
        logger.info("[Stage 4] Applying honeypot penalties…")
        for i, sig in enumerate(stage2_signals):
            hp = sig.get("honeypot_penalty", 0.0)
            if hp > 0.50:
                combined_scores[i] *= 0.05   # severe penalty
            elif hp > 0.30:
                combined_scores[i] *= (1.0 - hp * 0.80)

        # ----------------------------------------------------------------
        # Stage 5: Sort, tie-break, generate output
        # ----------------------------------------------------------------
        logger.info("[Stage 5] Sorting and generating submission…")
        # Primary sort: combined_score descending
        # Tie-break: candidate_id ascending (as per spec)
        sort_keys = list(zip(
            [-s for s in combined_scores],                  # descending score
            [c.candidate_id for c in stage2_candidates],   # ascending id
        ))
        sorted_order = sorted(range(len(stage2_candidates)), key=lambda i: sort_keys[i])

        # Take top-K
        final_order = sorted_order[:final_top_k]
        final_candidates = [stage2_candidates[i] for i in final_order]
        final_scores     = np.array([combined_scores[i] for i in final_order], dtype=np.float64)
        final_signals    = [stage2_signals[i] for i in final_order]

        # Enforce monotonically non-increasing scores (should already be so, but guard)
        for i in range(1, len(final_scores)):
            if final_scores[i] > final_scores[i - 1]:
                final_scores[i] = final_scores[i - 1]

        # Calibrate to [0, 1]
        s_max = final_scores[0] if final_scores[0] > 0 else 1.0
        s_min = final_scores[-1]
        if s_max > s_min:
            final_scores = (final_scores - s_min) / (s_max - s_min)
        else:
            final_scores = np.linspace(1.0, 0.01, len(final_scores))

        # Re-enforce monotonicity after calibration
        for i in range(1, len(final_scores)):
            if final_scores[i] > final_scores[i - 1]:
                final_scores[i] = final_scores[i - 1]

        # Generate reasoning & build rows
        logger.info("  Generating reasoning for %d candidates…", len(final_candidates))
        rows = []
        for rank, (cand, score, sig) in enumerate(
            zip(final_candidates, final_scores, final_signals), start=1
        ):
            reasoning = generate_reasoning(
                cand, sig, rank, float(score), self.jd_requirements
            )
            rows.append({
                "candidate_id": cand.candidate_id,
                "rank": rank,
                "score": round(float(score), 6),
                "reasoning": reasoning,
            })

        # Pad to exactly final_top_k rows if we have fewer
        # With 100K candidates this should rarely trigger.
        if len(rows) < final_top_k:
            logger.warning(
                "Only %d candidates in submission — padding to %d with best remaining",
                len(rows), final_top_k,
            )
            seen_ids = {r["candidate_id"] for r in rows}

            # Reuse already-extracted signals from valid_candidates / all_signals_list
            # (avoids redundant signal extraction)
            remaining_pairs = [
                (cand, sig)
                for cand, sig in zip(valid_candidates, all_signals_list)
                if cand.candidate_id not in seen_ids
            ]

            # Score remaining by signal composite + tie-break by id
            scored_remaining = sorted(
                [
                    (
                        sig["title_career"] * 0.30 +
                        sig["skill_depth"]   * 0.25 +
                        sig["experience"]    * 0.20 +
                        sig["education"]     * 0.10 +
                        sig["location"]      * 0.05 +
                        sig["behavioral"]    * 0.10 -
                        sig["honeypot_penalty"] * 0.15,
                        cand.candidate_id,
                        cand,
                        sig,
                    )
                    for cand, sig in remaining_pairs
                ],
                key=lambda x: (-x[0], x[1]),
            )

            pad_rank = len(rows) + 1
            last_nonzero = next((r["score"] for r in reversed(rows) if r["score"] > 0), 0.010)
            base_score = max(0.001, last_nonzero)
            n_needed   = min(len(scored_remaining), final_top_k - len(rows))

            if n_needed > 0:
                step_size = (base_score - 0.000001) / max(n_needed, 1)
                for idx, (_, _, cand, sig_dict) in enumerate(scored_remaining[:n_needed]):
                    pad_score = round(base_score - (idx + 1) * step_size, 6)
                    pad_score = max(0.000001, pad_score)
                    reasoning = generate_reasoning(cand, sig_dict, pad_rank, pad_score, self.jd_requirements)
                    rows.append({
                        "candidate_id": cand.candidate_id,
                        "rank": pad_rank,
                        "score": pad_score,
                        "reasoning": reasoning,
                    })
                    seen_ids.add(cand.candidate_id)
                    pad_rank += 1

        # ── Sort by score desc + candidate_id asc (tie-break) ──────────────
        # Must happen BEFORE the monotonicity sweep so that padded candidates
        # with positive scores are correctly interleaved above zero-scored
        # primary candidates (which happens when only a few pass the gate).
        rows = sorted(rows, key=lambda r: (-r["score"], r["candidate_id"]))

        # ── Monotonicity sweep — enforce non-increasing scores ───────────────
        for i in range(1, len(rows)):
            if rows[i]["score"] > rows[i - 1]["score"]:
                rows[i]["score"] = rows[i - 1]["score"]

        # ── Re-assign ranks 1..N sequentially ───────────────────────────────
        for i, row in enumerate(rows, start=1):
            row["rank"] = i

        # Cap to exactly final_top_k rows and write CSV
        rows = rows[:final_top_k]
        _write_submission(rows, output_path)


        elapsed = time.time() - wall_start
        logger.info("=" * 65)
        logger.info("  Ranking complete in %.1fs (%.1f min)", elapsed, elapsed / 60)
        logger.info("  Output: %s", output_path)
        logger.info("=" * 65)


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _write_submission(rows: List[Dict], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), output_path)


# ---------------------------------------------------------------------------
# Public API for programmatic use
# ---------------------------------------------------------------------------

def run_ranking(
    candidates_path: str,
    jd_path: str,
    output_path: str,
    cache_dir: str = "models",
    force_recompute: bool = False,
    final_top_k: int = _FINAL_TOP_K,
    stage1_top_k: int = _STAGE1_TOP_K,
) -> None:
    """Programmatic entry point — mirrors rank.py CLI."""
    ranker = Ranker(jd_path, candidates_path, cache_dir)
    ranker.run(output_path, force_recompute, final_top_k, stage1_top_k)


# ---------------------------------------------------------------------------
# CLI entry point (also called by python -m ranker)
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Redrob Intelligent Candidate Discovery & Ranking"
    )
    parser.add_argument("--candidates", required=True, help="candidates.jsonl or .jsonl.gz")
    parser.add_argument("--jd", default="../Dataset/job_description.docx",
                        help="Path to job description (.docx / .md / .txt)")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--cache-dir", default="models", help="Cache directory")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Force re-embedding (ignore cache)")
    parser.add_argument("--top-k", type=int, default=_FINAL_TOP_K,
                        help="Number of candidates in final output (default 100)")
    parser.add_argument("--stage1-k", type=int, default=_STAGE1_TOP_K,
                        help="Candidates advanced to reranking stage (default 500)")

    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    jd_path = Path(args.jd)

    if not candidates_path.exists():
        logger.error("Candidates file not found: %s", candidates_path)
        sys.exit(1)
    if not jd_path.exists():
        logger.error("JD file not found: %s", jd_path)
        sys.exit(1)

    run_ranking(
        candidates_path=str(candidates_path),
        jd_path=str(jd_path),
        output_path=args.out,
        cache_dir=args.cache_dir,
        force_recompute=args.force_recompute,
        final_top_k=args.top_k,
        stage1_top_k=args.stage1_k,
    )


if __name__ == "__main__":
    main()