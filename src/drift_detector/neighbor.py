"""Nearest-neighbor failure-rate predictor.

Uses sentence-transformers to embed the trajectory and retrieve the K
most-similar reference trajectories (bundled in data/nn_reference_v0.npz).
The neighbor-failure-rate is returned as a probability in [0, 1].

Held-out mean AUC (5-fold instance-level CV, n=746): 0.746 ± 0.033
Combined with the structural detector: 0.792 ± 0.036.

Pooled operating points (struct AND nn):
  struct>0.5 AND nn>0.7 → 100% precision (31/31), 9% recall
  struct>0.5 AND nn>0.6 →  96% precision (66/69), 19% recall
  struct>0.5 AND nn>0.5 →  91% precision (108/119), 31% recall

The reference embeddings are precomputed; at inference we only need to
embed the current trajectory once (~50ms) and do one matrix multiply.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Optional

import numpy as np

__all__ = ["NeighborScore", "neighbor_unavailable_reason"]


_MIN_TEXT_CHARS = 200
_DEFAULT_K = 50


def _msg_text(m: dict) -> str:
    if "blocks" in m:
        parts = []
        for b in m.get("blocks") or []:
            if not isinstance(b, dict):
                continue
            bt = b.get("block_type") or b.get("type")
            if bt == "text":
                parts.append(b.get("text") or "")
            elif bt in ("tool_use", "tool"):
                nm = b.get("name") or b.get("tool_name") or ""
                inp = b.get("input") or b.get("arguments") or ""
                parts.append(f"[{nm} {json.dumps(inp)[:300] if not isinstance(inp, str) else inp[:300]}]")
            elif bt in ("tool_result", "tool_output"):
                tr = b.get("content") or b.get("output") or ""
                if isinstance(tr, list):
                    tr = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in tr)
                parts.append(str(tr)[:1000])
        return "\n".join(parts)
    c = m.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return json.dumps(c)
    return ""


def _trajectory_text(messages: list[dict], max_chars: int = 8000) -> str:
    lines = []
    for m in messages:
        text = _msg_text(m)
        if not text:
            continue
        lines.append(f"{m.get('role', '?')}: {text[:1500]}")
    full = "\n".join(lines)
    return full[-max_chars:] if len(full) > max_chars else full


def neighbor_unavailable_reason() -> Optional[str]:
    """Return a string reason if NN scoring can't run, else None."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return "sentence-transformers not installed (pip install sentence-transformers)"
    try:
        files("drift_detector").joinpath("data/nn_reference_v0.npz")
    except Exception as e:
        return f"reference bundle missing: {e}"
    return None


class NeighborScore:
    """K-NN failure-rate predictor against a bundled reference set.

    >>> from drift_detector import NeighborScore
    >>> ns = NeighborScore()
    >>> p_fail = ns.score(messages)  # in [0, 1] or None on failure
    """

    def __init__(self, K: int = _DEFAULT_K,
                 model_name: Optional[str] = None,
                 reference_path: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        if reference_path is not None:
            data = np.load(reference_path, allow_pickle=True)
        else:
            with files("drift_detector").joinpath("data/nn_reference_v0.npz").open("rb") as f:
                data = np.load(f, allow_pickle=True)
                # load into memory so we can close the resource
                data = {k: data[k] for k in data.files}
        self._embeddings = data["embeddings"]
        self._labels = data["labels"]
        bundled_model = str(data.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
        if isinstance(data.get("model_name"), np.ndarray):
            bundled_model = str(data["model_name"].item())
        self.model_name = model_name or bundled_model
        self.K = min(K, len(self._labels))
        self._embedder = SentenceTransformer(self.model_name)
        self._n = len(self._labels)

    def score(self, messages: list[dict]) -> Optional[float]:
        """Return P(failure) in [0, 1] from K nearest neighbors, or None."""
        if not messages:
            return None
        text = _trajectory_text(messages)
        if len(text) < _MIN_TEXT_CHARS:
            return None
        try:
            q = self._embedder.encode([text], normalize_embeddings=True)
        except Exception:
            return None
        sims = q[0] @ self._embeddings.T
        top_k = np.argsort(-sims)[: self.K]
        return float(self._labels[top_k].mean())
