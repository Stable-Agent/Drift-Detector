"""Non-LLM failure detector for coding-agent trajectories.

Loads the bundled L1 logistic regression model (10 process + structural
features) and scores trajectories. AUC 0.76 on n=185 SWE-bench
trajectories; saturates at turn ~50 with real signal from turn 10.
"""

from __future__ import annotations

import pickle
from importlib.resources import files
from pathlib import Path
from typing import Any

from .features import FEATURES, extract_features

__all__ = ["Detector", "Assessment"]


_DEFAULT_THRESHOLD = 0.5


class Assessment:
    """Result of scoring a trajectory.

    Attributes:
        score: float in [0, 1]. Probability the trajectory will fail.
        threshold: float. The decision threshold used.
        triggered: bool. True iff score > threshold.
        features: dict[str, Any]. The raw extracted feature values.
        top_contributors: list[tuple[str, float]]. Features sorted by
            absolute standardized contribution to the score.
    """

    __slots__ = ("score", "threshold", "triggered", "features", "top_contributors")

    def __init__(self, score: float, threshold: float, features: dict,
                 top_contributors: list[tuple[str, float]]):
        self.score = score
        self.threshold = threshold
        self.triggered = score > threshold
        self.features = features
        self.top_contributors = top_contributors

    def __repr__(self) -> str:
        top = ", ".join(f"{k}={v:+.2f}" for k, v in self.top_contributors[:3])
        return (f"Assessment(score={self.score:.3f}, threshold={self.threshold}, "
                f"triggered={self.triggered}, top=[{top}])")

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "threshold": self.threshold,
            "triggered": self.triggered,
            "features": self.features,
            "top_contributors": [{"feature": f, "contribution": c}
                                 for f, c in self.top_contributors],
        }


class Detector:
    """Non-LLM coding-agent drift detector.

    Loads a pre-trained L1 logistic regression and scores trajectories
    via process + structural feature extraction.

    >>> from drift_detector import Detector
    >>> det = Detector()
    >>> result = det.assess(messages)
    >>> if result.triggered:
    ...     print(f"Drift! score={result.score:.2f}")
    """

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD,
                 model_path: str | Path | None = None):
        if model_path is not None:
            with open(model_path, "rb") as f:
                self._bundle = pickle.load(f)
        else:
            data_path = files("drift_detector").joinpath("data/non_llm_detector_v0.pkl")
            with data_path.open("rb") as f:
                self._bundle = pickle.load(f)
        self._pipeline = self._bundle["pipeline"]
        self._features = self._bundle["features"]
        assert self._features == FEATURES, (
            f"model feature list {self._features} doesn't match package FEATURES")
        self.threshold = threshold

    def score(self, messages: list[dict], final_patch: str | None = None) -> float:
        """Return P(failure) in [0, 1] for the trajectory."""
        return self.assess(messages, final_patch).score

    def assess(self, messages: list[dict], final_patch: str | None = None) -> Assessment:
        """Score the trajectory and return detailed Assessment."""
        import numpy as np
        feats = extract_features(messages, final_patch=final_patch)
        x = np.array([[feats[k] for k in FEATURES]], dtype=float)
        score = float(self._pipeline.predict_proba(x)[0, 1])

        # Per-feature standardized contribution = scaled_feature * coef
        scaler = self._pipeline.named_steps["scale"]
        lr = self._pipeline.named_steps["lr"]
        scaled = scaler.transform(x)[0]
        coefs = lr.coef_[0]
        contributions = list(zip(FEATURES, (scaled * coefs).tolist()))
        contributions.sort(key=lambda kv: -abs(kv[1]))

        return Assessment(score=score, threshold=self.threshold,
                          features=feats, top_contributors=contributions)
