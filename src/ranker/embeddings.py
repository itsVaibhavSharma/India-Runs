"""Embeddings module — Handles sentence-transformer embeddings with full offline support.

The embedding model (all-MiniLM-L6-v2) is stored locally in models/embedding_model/
so that NO network download happens during ranking.  Run precompute.py once
to download and cache the model + candidate embeddings.

Workflow:
  1. precompute.py downloads the model to models/embedding_model/ (one-time)
  2. precompute.py computes candidate embeddings → models/embeddings/candidate_embeddings.npy
  3. At ranking time, embeddings are loaded from disk — zero network access.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file's location
# ---------------------------------------------------------------------------

_MODULE_DIR = Path(__file__).parent                       # src/ranker/
_REPO_ROOT = _MODULE_DIR.parent.parent                    # Implementation/
_DEFAULT_MODEL_DIR = _REPO_ROOT / "models" / "embedding_model"
_DEFAULT_CACHE_DIR = _REPO_ROOT / "models" / "embeddings"

# Model identifiers
_HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384


# ---------------------------------------------------------------------------
# EmbeddingManager
# ---------------------------------------------------------------------------

class EmbeddingManager:
    """Manages sentence-transformer embeddings for candidates and JD.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory where candidate_embeddings.npy and candidate_ids.pkl are stored.
    model_dir : str or Path, optional
        Directory containing the locally saved SentenceTransformer model.
        Falls back to downloading from HuggingFace if not present (requires network).
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._candidate_embeddings: Optional[np.ndarray] = None
        self._candidate_ids: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Internal: lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the SentenceTransformer model — offline-first."""
        if self._model is not None:
            return

        import os
        # Suppress TensorFlow loading (we use PyTorch only)
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

        from sentence_transformers import SentenceTransformer

        if self.model_dir.exists() and any(self.model_dir.iterdir()):
            logger.info("Loading embedding model from local path: %s", self.model_dir)
            self._model = SentenceTransformer(str(self.model_dir))
        else:
            # Fallback: download from HuggingFace (only during precompute)
            logger.warning(
                "Local model not found at %s. Downloading %s from HuggingFace "
                "(requires network — run precompute.py to avoid this at ranking time).",
                self.model_dir, _HF_MODEL_NAME,
            )
            self._model = SentenceTransformer(_HF_MODEL_NAME)
            # Save locally for future offline use
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self._model.save(str(self.model_dir))
            logger.info("Model saved to %s for offline use.", self.model_dir)

        logger.info("Embedding model loaded (dim=%d).", _EMBEDDING_DIM)

    # ------------------------------------------------------------------
    # JD embedding
    # ------------------------------------------------------------------

    def get_jd_embedding(self, jd_text: str) -> np.ndarray:
        """Return normalised embedding for the JD text (computed at runtime, fast)."""
        self._load_model()
        emb = self._model.encode(
            [jd_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb[0].astype(np.float32)

    # ------------------------------------------------------------------
    # Candidate embeddings (cached on disk)
    # ------------------------------------------------------------------

    def get_candidate_embeddings(
        self,
        candidates: List,
        force_recompute: bool = False,
    ) -> Tuple[np.ndarray, List[str]]:
        """Return (embeddings, ids) arrays.

        If cached files exist and ``force_recompute`` is False, loads from disk.
        Otherwise computes from candidates and caches.
        """
        emb_file  = self.cache_dir / "candidate_embeddings.npy"
        ids_file  = self.cache_dir / "candidate_ids.pkl"

        if not force_recompute and emb_file.exists() and ids_file.exists():
            logger.info("Loading cached candidate embeddings from %s", self.cache_dir)
            embeddings = np.load(emb_file)
            with open(ids_file, "rb") as fh:
                ids = pickle.load(fh)
            self._candidate_embeddings = embeddings
            self._candidate_ids = ids
            logger.info("Loaded %d cached embeddings (shape %s).", len(ids), embeddings.shape)
            return embeddings, ids

        logger.info("Computing embeddings for %d candidates...", len(candidates))
        self._load_model()

        texts, ids = [], []
        for c in candidates:
            text = getattr(c, "_text_for_embedding", "") or ""
            if not text:
                parts = []
                if c.summary:
                    parts.append(c.summary)
                if c.headline:
                    parts.append(c.headline)
                for career in c.career_history:
                    if career.description:
                        parts.append(career.description)
                text = " ".join(parts)
            texts.append(text[:4096])   # cap length for memory safety
            ids.append(c.candidate_id)

        # Batch encode
        batch_size = 256
        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embs = self._model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=64,
            )
            all_embeddings.append(batch_embs.astype(np.float32))
            logger.info(
                "  Encoded %d / %d candidates...",
                min(i + batch_size, len(texts)), len(texts),
            )

        embeddings = np.vstack(all_embeddings).astype(np.float32)

        # Persist to disk
        np.save(emb_file, embeddings)
        with open(ids_file, "wb") as fh:
            pickle.dump(ids, fh)

        self._candidate_embeddings = embeddings
        self._candidate_ids = ids
        logger.info(
            "Computed and cached %d embeddings → %s", len(ids), self.cache_dir
        )
        return embeddings, ids

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    def compute_similarities(
        self,
        jd_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Cosine similarity (embeddings assumed to be L2-normalised)."""
        return candidate_embeddings @ jd_embedding   # shape (N,)

    def get_top_k_similar(
        self,
        jd_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidate_ids: List[str],
        k: int = 500,
    ) -> Tuple[List[str], np.ndarray]:
        """Return (ids, similarities) for top-k most similar candidates."""
        sims = self.compute_similarities(jd_embedding, candidate_embeddings)
        top_k = min(k, len(sims))
        top_idx = np.argpartition(sims, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return [candidate_ids[i] for i in top_idx], sims[top_idx]


# ---------------------------------------------------------------------------
# Convenience loader (used in precompute.py / rank.py)
# ---------------------------------------------------------------------------

def load_embeddings_from_cache(
    cache_dir: str,
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Load pre-computed candidate embeddings from disk. Returns (None, None) if missing."""
    p = Path(cache_dir)
    emb_file = p / "candidate_embeddings.npy"
    ids_file = p / "candidate_ids.pkl"

    if emb_file.exists() and ids_file.exists():
        embeddings = np.load(emb_file)
        with open(ids_file, "rb") as fh:
            ids = pickle.load(fh)
        return embeddings, ids
    return None, None