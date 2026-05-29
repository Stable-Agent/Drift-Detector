"""drift-detector — non-LLM failure detector for coding agents.

Predict whether an in-flight coding-agent trajectory is heading toward
failure, using 10 structural features (no LLM calls). AUC 0.76 on n=185
SWE-bench trajectories; saturates at turn ~50.

>>> from drift_detector import Detector
>>> det = Detector()
>>> assessment = det.assess(messages)
>>> if assessment.triggered:
...     print(f"Drift detected: score={assessment.score:.2f}")
"""

from .detector import Assessment, Detector
from .features import FEATURES, extract_features

__version__ = "0.1.0"

__all__ = ["Detector", "Assessment", "FEATURES", "extract_features",
           "NeighborScore"]


def __getattr__(name):
    """Lazy-import NeighborScore so users without sentence-transformers
    can still use the structural detector."""
    if name == "NeighborScore":
        from .neighbor import NeighborScore
        return NeighborScore
    raise AttributeError(f"module 'drift_detector' has no attribute {name!r}")
