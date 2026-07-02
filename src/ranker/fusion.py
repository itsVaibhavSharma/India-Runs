"""Signal Fusion — Combines 7 signal scores into a calibrated final ranking score.

Uses Logistic Regression trained on synthetic JD-derived preference pairs.
Calibration via Isotonic Regression ensures score ordering matches quality.

Pipeline:
  train(jd_requirements) → fits LR + isotonic calibrator → saves to disk
  load()                 → restores from disk
  predict_proba(signals) → single-candidate score in [0, 1]
  predict_batch(signals) → batch scores (fast)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Signal names (order must match _extract_features)
SIGNAL_NAMES = [
    "title_career",
    "skill_depth",
    "experience",
    "education",
    "location",
    "behavioral",
    "honeypot_penalty",
]


# ---------------------------------------------------------------------------
# SignalFusion
# ---------------------------------------------------------------------------

class SignalFusion:
    """Logistic Regression signal fusion with Isotonic calibration."""

    def __init__(self, model_dir: Optional[str] = None):
        if model_dir:
            self.model_dir = Path(model_dir)
        else:
            # Default: relative to the package root
            _here = Path(__file__).parent.parent.parent   # Implementation/
            self.model_dir = _here / "models" / "fusion"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[LogisticRegression] = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.weights: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, signals: Dict[str, float]) -> np.ndarray:
        """11-dim feature vector: 7 base signals + 4 interaction terms."""
        base = np.array(
            [signals.get(name, 0.0) for name in SIGNAL_NAMES],
            dtype=np.float64,
        )

        # Interaction terms capture that strong candidates must be strong
        # across multiple dimensions simultaneously
        interactions = np.array([
            base[0] * base[1],   # title_career × skill_depth
            base[0] * base[5],   # title_career × behavioral
            base[1] * base[5],   # skill_depth × behavioral
            base[2] * base[0],   # experience × title_career
        ], dtype=np.float64)

        return np.concatenate([base, interactions])

    # ------------------------------------------------------------------
    # Synthetic training data
    # ------------------------------------------------------------------

    def _generate_training_data(
        self, jd_requirements=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic preference pairs from JD requirements.

        Positive = strong candidate matching JD requirements
        Negative = candidate failing on one or more critical signals
        """
        rng = np.random.default_rng(42)
        X, y = [], []

        # ---- Positive templates (high-quality matches) ----
        positive_templates = [
            # Classic: product AI/ML engineer, all signals strong
            dict(title_career=0.90, skill_depth=0.88, experience=0.95,
                 education=0.72, location=0.90, behavioral=0.85, honeypot_penalty=0.0),
            # Good with slightly lower education (self-taught / Tier-2)
            dict(title_career=0.85, skill_depth=0.82, experience=0.88,
                 education=0.50, location=0.75, behavioral=0.80, honeypot_penalty=0.0),
            # Applied scientist with ranking eval background
            dict(title_career=0.80, skill_depth=0.92, experience=0.80,
                 education=0.90, location=0.65, behavioral=0.78, honeypot_penalty=0.0),
            # Senior ranking/search engineer at product startup
            dict(title_career=0.88, skill_depth=0.80, experience=0.85,
                 education=0.65, location=1.00, behavioral=0.82, honeypot_penalty=0.0),
            # Strong behavioral + skills, relocated from non-preferred city
            dict(title_career=0.82, skill_depth=0.85, experience=0.90,
                 education=0.60, location=0.60, behavioral=0.90, honeypot_penalty=0.0),
            # High skill depth, good location, mid experience
            dict(title_career=0.78, skill_depth=0.90, experience=0.78,
                 education=0.70, location=0.85, behavioral=0.75, honeypot_penalty=0.02),
        ]

        # ---- Negative templates (poor matches) ----
        negative_templates = [
            # Services company, low title match
            dict(title_career=0.25, skill_depth=0.40, experience=0.65,
                 education=0.40, location=0.50, behavioral=0.30, honeypot_penalty=0.30),
            # Pure researcher, no production
            dict(title_career=0.20, skill_depth=0.65, experience=0.55,
                 education=0.90, location=0.35, behavioral=0.45, honeypot_penalty=0.40),
            # Management/architect role (no code)
            dict(title_career=0.15, skill_depth=0.30, experience=0.85,
                 education=0.75, location=0.65, behavioral=0.55, honeypot_penalty=0.20),
            # Consulting background, keyword stuffed
            dict(title_career=0.22, skill_depth=0.38, experience=0.72,
                 education=0.52, location=0.45, behavioral=0.38, honeypot_penalty=0.25),
            # Honeypot / fabricated profile
            dict(title_career=0.45, skill_depth=0.55, experience=0.30,
                 education=0.30, location=0.25, behavioral=0.22, honeypot_penalty=0.80),
            # Inactive candidate (behavioral gate should have excluded, but be safe)
            dict(title_career=0.55, skill_depth=0.62, experience=0.75,
                 education=0.60, location=0.55, behavioral=0.05, honeypot_penalty=0.10),
            # Recent LLM-framework-only, no depth
            dict(title_career=0.35, skill_depth=0.42, experience=0.40,
                 education=0.52, location=0.60, behavioral=0.42, honeypot_penalty=0.22),
            # Non-India, low engagement
            dict(title_career=0.50, skill_depth=0.55, experience=0.70,
                 education=0.55, location=0.10, behavioral=0.25, honeypot_penalty=0.15),
            # Data engineer without ML/retrieval
            dict(title_career=0.20, skill_depth=0.25, experience=0.70,
                 education=0.50, location=0.60, behavioral=0.45, honeypot_penalty=0.05),
        ]

        n_positive_per = 40
        n_negative_per = 25

        for template in positive_templates:
            for _ in range(n_positive_per):
                noise = rng.normal(0, 0.04, 7)
                vals = np.clip(
                    [template[n] for n in SIGNAL_NAMES] + noise, 0.0, 1.0
                )
                feat = self._extract_features(
                    {n: v for n, v in zip(SIGNAL_NAMES, vals)}
                )
                X.append(feat)
                y.append(1)

        for template in negative_templates:
            for _ in range(n_negative_per):
                noise = rng.normal(0, 0.04, 7)
                vals = np.clip(
                    [template[n] for n in SIGNAL_NAMES] + noise, 0.0, 1.0
                )
                feat = self._extract_features(
                    {n: v for n, v in zip(SIGNAL_NAMES, vals)}
                )
                X.append(feat)
                y.append(0)

        return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, jd_requirements=None) -> None:
        logger.info("Training signal fusion model on synthetic JD-derived data…")
        X, y = self._generate_training_data(jd_requirements)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = LogisticRegression(
            C=2.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )
        self.model.fit(X_scaled, y)

        # Isotonic calibration
        probs = self.model.predict_proba(X_scaled)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(probs, y)

        # Store weights for logging / interpretability
        self.weights = self.model.coef_[0]
        feature_names = SIGNAL_NAMES + ["t_c*s_d", "t_c*beh", "s_d*beh", "exp*t_c"]
        weight_log = {n: round(float(w), 4) for n, w in zip(feature_names, self.weights)}
        logger.info("Fusion model trained. Weights: %s", weight_log)

        self.save()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, signals: Dict[str, float]) -> float:
        """Score a single candidate in [0, 1]."""
        feat = self._extract_features(signals).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)
        prob = self.model.predict_proba(feat_scaled)[0, 1]
        calibrated = float(self.calibrator.transform([prob])[0])
        return float(np.clip(calibrated, 0.0, 1.0))

    def predict_batch(self, signals_list: List[Dict[str, float]]) -> np.ndarray:
        """Score a batch of candidates. Returns float32 array of length N."""
        if not signals_list:
            return np.array([], dtype=np.float32)
        features = np.array(
            [self._extract_features(s) for s in signals_list], dtype=np.float64
        )
        feat_scaled = self.scaler.transform(features)
        probs = self.model.predict_proba(feat_scaled)[:, 1]
        calibrated = self.calibrator.transform(probs)
        return np.clip(calibrated, 0.0, 1.0).astype(np.float32)

    def get_score_breakdown(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Return per-feature contribution for interpretability."""
        if self.weights is None or self.scaler is None:
            return {n: signals.get(n, 0.0) for n in SIGNAL_NAMES}
        feat = self._extract_features(signals)
        feat_scaled = self.scaler.transform(feat.reshape(1, -1))[0]
        contribs = feat_scaled * self.weights
        names = SIGNAL_NAMES + ["t_c*s_d", "t_c*beh", "s_d*beh", "exp*t_c"]
        return {n: float(v) for n, v in zip(names, contribs)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save model, scaler, calibrator to disk."""
        files = {
            "fusion_model.pkl": self.model,
            "scaler.pkl": self.scaler,
            "calibrator.pkl": self.calibrator,
        }
        for fname, obj in files.items():
            with open(self.model_dir / fname, "wb") as f:
                pickle.dump(obj, f)
        if self.weights is not None:
            np.save(self.model_dir / "weights.npy", self.weights)
        logger.info("Fusion model artifacts saved to %s", self.model_dir)

    def load(self) -> bool:
        """Load saved model from disk. Returns True if successful."""
        required = ["fusion_model.pkl", "scaler.pkl", "calibrator.pkl"]
        if not all((self.model_dir / f).exists() for f in required):
            return False
        try:
            with open(self.model_dir / "fusion_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open(self.model_dir / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            with open(self.model_dir / "calibrator.pkl", "rb") as f:
                self.calibrator = pickle.load(f)
            weights_path = self.model_dir / "weights.npy"
            if weights_path.exists():
                self.weights = np.load(weights_path)
            logger.info("Fusion model loaded from %s", self.model_dir)
            return True
        except Exception as exc:
            logger.warning("Failed to load fusion model: %s — will retrain.", exc)
            return False