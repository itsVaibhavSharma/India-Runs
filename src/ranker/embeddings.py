"""Embeddings module - Handles candidate and JD embeddings with caching."""

import os
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import pickle
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    MODEL_NAME = 'all-MiniLM-L6-v2'
    EMBEDDING_DIM = 384
    CACHE_DIR = Path('models/embeddings')

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir:
            self.CACHE_DIR = Path(cache_dir)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.model = None
        self._candidate_embeddings = None
        self._candidate_ids = None
        self._candidate_norms = None

    def load_model(self):
        if self.model is None:
            logger.info(f"Loading embedding model: {self.MODEL_NAME}")
            self.model = SentenceTransformer(self.MODEL_NAME)
            logger.info("Model loaded successfully")

    def get_jd_embedding(self, jd_text: str) -> np.ndarray:
        self.load_model()
        embedding = self.model.encode([jd_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return embedding.astype(np.float32)

    def get_candidate_embeddings(self, candidates: List, force_recompute: bool = False) -> Tuple[np.ndarray, List[str]]:
        cache_file = self.CACHE_DIR / 'candidate_embeddings.npy'
        ids_file = self.CACHE_DIR / 'candidate_ids.pkl'
        norms_file = self.CACHE_DIR / 'candidate_norms.npy'

        if not force_recompute and cache_file.exists() and ids_file.exists() and norms_file.exists():
            logger.info("Loading cached candidate embeddings...")
            self._candidate_embeddings = np.load(cache_file)
            with open(ids_file, 'rb') as f:
                self._candidate_ids = pickle.load(f)
            self._candidate_norms = np.load(norms_file)
            return self._candidate_embeddings, self._candidate_ids

        logger.info("Computing candidate embeddings...")
        self.load_model()

        texts = []
        ids = []
        for c in candidates:
            text = c._text_for_embedding if hasattr(c, '_text_for_embedding') else ''
            if not text:
                text = f"{c.summary} {c.headline}"
                for career in c.career_history:
                    if career.description:
                        text += f" {career.description}"
            texts.append(text)
            ids.append(c.candidate_id)

        batch_size = 256
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
            )
            embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Cache
        np.save(cache_file, embeddings)
        with open(ids_file, 'wb') as f:
            pickle.dump(ids, f)
        np.save(norms_file, norms)

        self._candidate_embeddings = embeddings
        self._candidate_ids = ids
        self._candidate_norms = norms

        logger.info(f"Computed and cached embeddings for {len(ids)} candidates")
        return embeddings, ids

    def compute_similarities(self, jd_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
        # Cosine similarity (embeddings are already normalized)
        return np.dot(candidate_embeddings, jd_embedding)

    def get_top_k_by_similarity(self, jd_embedding: np.ndarray, candidate_embeddings: np.ndarray,
                                 candidate_ids: List[str], k: int = 500) -> Tuple[np.ndarray, List[str]]:
        similarities = self.compute_similarities(jd_embedding, candidate_embeddings)
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
        return similarities[top_k_indices], [candidate_ids[i] for i in top_k_indices]


def load_embeddings_from_cache(cache_dir: str) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[np.ndarray]]:
    cache_path = Path(cache_dir)
    emb_file = cache_path / 'candidate_embeddings.npy'
    ids_file = cache_path / 'candidate_ids.pkl'
    norms_file = cache_path / 'candidate_norms.npy'

    if emb_file.exists() and ids_file.exists() and norms_file.exists():
        embeddings = np.load(emb_file)
        with open(ids_file, 'rb') as f:
            ids = pickle.load(f)
        norms = np.load(norms_file)
        return embeddings, ids, norms
    return None, None, None