"""Signal Fusion - Combines 7 signal scores into final ranking score."""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalFusion:
    MODEL_DIR = Path('models/fusion')
    SIGNAL_NAMES = [
        'title_career', 'skill_depth', 'experience', 'education',
        'location', 'behavioral', 'honeypot_penalty'
    ]

    def __init__(self, model_dir: Optional[str] = None):
        if model_dir:
            self.MODEL_DIR = Path(model_dir)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.calibrator = None
        self.scaler = None
        self.weights = None

    def _extract_features(self, signals: Dict[str, float]) -> np.ndarray:
        base_features = np.array([signals.get(name, 0.0) for name in self.SIGNAL_NAMES])

        # Interaction features
        interactions = [
            base_features[0] * base_features[1],  # title_career * skill_depth
            base_features[0] * base_features[5],  # title_career * behavioral
            base_features[1] * base_features[5],  # skill_depth * behavioral
            base_features[2] * base_features[0],  # experience * title_career
        ]

        return np.concatenate([base_features, interactions])

    def _generate_synthetic_training_data(self, jd_requirements) -> Tuple[np.ndarray, np.ndarray]:
        X = []
        y = []

        # Positive examples (high-quality candidates matching JD)
        positive_templates = [
            # Strong product AI/ML engineer
            {'title_career': 0.9, 'skill_depth': 0.85, 'experience': 0.9, 'education': 0.7,
             'location': 0.8, 'behavioral': 0.8, 'honeypot_penalty': 0.0},
            # Startup ML engineer with shipping experience
            {'title_career': 0.85, 'skill_depth': 0.8, 'experience': 0.75, 'education': 0.6,
             'location': 0.7, 'behavioral': 0.75, 'honeypot_penalty': 0.0},
            # Applied scientist with ranking background
            {'title_career': 0.8, 'skill_depth': 0.9, 'experience': 0.8, 'education': 0.85,
             'location': 0.6, 'behavioral': 0.7, 'honeypot_penalty': 0.0},
            # Senior engineer with vector search production experience
            {'title_career': 0.88, 'skill_depth': 0.82, 'experience': 0.85, 'education': 0.65,
             'location': 0.9, 'behavioral': 0.8, 'honeypot_penalty': 0.0},
        ]

        # Negative examples (candidates that don't match)
        negative_templates = [
            # Services company with keyword stuffing
            {'title_career': 0.3, 'skill_depth': 0.4, 'experience': 0.6, 'education': 0.4,
             'location': 0.5, 'behavioral': 0.3, 'honeypot_penalty': 0.3},
            # Pure researcher, no production
            {'title_career': 0.2, 'skill_depth': 0.6, 'experience': 0.5, 'education': 0.9,
             'location': 0.3, 'behavioral': 0.4, 'honeypot_penalty': 0.4},
            # Manager who doesn't code
            {'title_career': 0.15, 'skill_depth': 0.3, 'experience': 0.8, 'education': 0.7,
             'location': 0.6, 'behavioral': 0.5, 'honeypot_penalty': 0.2},
            # Consulting background
            {'title_career': 0.25, 'skill_depth': 0.35, 'experience': 0.7, 'education': 0.5,
             'location': 0.4, 'behavioral': 0.35, 'honeypot_penalty': 0.25},
            # Keyword stuffer / honeypot
            {'title_career': 0.4, 'skill_depth': 0.5, 'experience': 0.3, 'education': 0.3,
             'location': 0.2, 'behavioral': 0.2, 'honeypot_penalty': 0.7},
            # Inactive candidate
            {'title_career': 0.5, 'skill_depth': 0.6, 'experience': 0.7, 'education': 0.6,
             'location': 0.5, 'behavioral': 0.05, 'honeypot_penalty': 0.1},
            # Recent LLM framework only, no depth
            {'title_career': 0.35, 'skill_depth': 0.45, 'experience': 0.4, 'education': 0.5,
             'location': 0.6, 'behavioral': 0.4, 'honeypot_penalty': 0.2},
        ]

        # Add variations
        for template in positive_templates:
            for _ in range(25):
                noise = np.random.normal(0, 0.05, 7)
                features = np.clip(np.array([template[n] for n in self.SIGNAL_NAMES]) + noise, 0, 1)
                X.append(self._extract_features({n: features[i] for i, n in enumerate(self.SIGNAL_NAMES)}))
                y.append(1)

        for template in negative_templates:
            for _ in range(25):
                noise = np.random.normal(0, 0.05, 7)
                features = np.clip(np.array([template[n] for n in self.SIGNAL_NAMES]) + noise, 0, 1)
                X.append(self._extract_features({n: features[i] for i, n in enumerate(self.SIGNAL_NAMES)}))
                y.append(0)

        return np.array(X), np.array(y)

    def train(self, jd_requirements=None):
        logger.info("Training signal fusion model on synthetic JD-derived data...")
        X, y = self._generate_synthetic_training_data(jd_requirements)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced', random_state=42
        )
        self.model.fit(X_scaled, y)

        # Get probabilities for calibration
        probs = self.model.predict_proba(X_scaled)[:, 1]

        # Fit isotonic calibration
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(probs, y)

        # Store weights for interpretability
        self.weights = self.model.coef_[0]

        self.save()
        logger.info(f"Model trained. Feature weights: {dict(zip([*self.SIGNAL_NAMES, 't_c*s_d', 't_c*b', 's_d*b', 'exp*t_c'], self.weights))}")

    def predict_proba(self, signals: Dict[str, float]) -> float:
        if self.model is None:
            self.load()
            if self.model is None:
                self.train()
                self.load()

        features = self._extract_features(signals).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        prob = self.model.predict_proba(features_scaled)[0, 1]
        calibrated = self.calibrator.transform([prob])[0]
        return float(np.clip(calibrated, 0.0, 1.0))

    def predict_batch(self, signals_list: List[Dict[str, float]]) -> np.ndarray:
        if self.model is None:
            self.load()
            if self.model is None:
                self.train()
                self.load()

        features = np.array([self._extract_features(s) for s in signals_list])
        features_scaled = self.scaler.transform(features)
        probs = self.model.predict_proba(features_scaled)[:, 1]
        calibrated = self.calibrator.transform(probs)
        return np.clip(calibrated, 0.0, 1.0)

    def get_score_breakdown(self, signals: Dict[str, float]) -> Dict[str, float]:
        if self.weights is None:
            return {name: signals.get(name, 0.0) for name in self.SIGNAL_NAMES}

        base = np.array([signals.get(name, 0.0) for name in self.SIGNAL_NAMES])
        interactions = np.array([
            base[0] * base[1], base[0] * base[5],
            base[1] * base[5], base[2] * base[0]
        ])

        all_features = np.concatenate([base, interactions])
        scaled = self.scaler.transform(all_features.reshape(1, -1))[0]

        contributions = scaled * self.weights
        return {
            'title_career': contributions[0],
            'skill_depth': contributions[1],
            'experience': contributions[2],
            'education': contributions[3],
            'location': contributions[4],
            'behavioral': contributions[5],
            'honeypot_penalty': contributions[6],
            'interactions': float(sum(contributions[7:]))
        }

    def save(self):
        model_file = self.MODEL_DIR / 'fusion_model.pkl'
        scaler_file = self.MODEL_DIR / 'scaler.pkl'
        calibrator_file = self.MODEL_DIR / 'calibrator.pkl'
        weights_file = self.MODEL_DIR / 'weights.npy'

        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(calibrator_file, 'wb') as f:
            pickle.dump(self.calibrator, f)
        np.save(weights_file, self.weights)
        logger.info(f"Model saved to {self.MODEL_DIR}")

    def load(self):
        model_file = self.MODEL_DIR / 'fusion_model.pkl'
        scaler_file = self.MODEL_DIR / 'scaler.pkl'
        calibrator_file = self.MODEL_DIR / 'calibrator.pkl'
        weights_file = self.MODEL_DIR / 'weights.npy'

        if all(f.exists() for f in [model_file, scaler_file, calibrator_file, weights_file]):
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(calibrator_file, 'rb') as f:
                self.calibrator = pickle.load(f)
            self.weights = np.load(weights_file)
            logger.info("Fusion model loaded from cache")
            return True
        return False