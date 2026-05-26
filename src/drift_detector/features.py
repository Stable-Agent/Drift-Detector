"""Feature extraction for the non-LLM coding-drift detector.

Pulls 10 deterministic structural signals from a coding-agent trajectory.
Pure regex + counting; no LLM calls.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["FEATURES", "extract_features"]


FEATURES = [
    "n_custom_scripts",
    "n_custom_test_runs",
    "n_repo_test_runs",
    "n_reverts",
    "n_msgs",
    "n_intermediate_diffs",
    "inflight_diff_lines_max",
    "inflight_diff_files_max",
    "diff_size_growth",
    "final_patch_lines",
]


# Custom-script creation patterns (`cat > foo.py`, heredocs, tee, touch).
_RE_CREATE_PY = re.compile(
    r"(?:cat\b[^>]*>+\s*|(?:tee|touch)\s+)"
    r"((?:/testbed/)?[A-Za-z][A-Za-z0-9_.-]*\.py)\b"
)
_RE_RUN_PY = re.compile(r"\bpython3?\s+((?:/testbed/)?[A-Za-z][\w./-]*\.py)\b")
_RE_RUN_REPO_TEST = re.compile(
    r"(?:tests/runtests\.py|\bruntests\.py\b|python3?\s+-m\s+pytest"
    r"|python3?\s+manage\.py\s+test)"
)
_RE_REVERT = re.compile(r"\bgit\s+(?:checkout|restore|reset\s+--hard|stash)\b")
_RE_DIFF_BLOCK = re.compile(r"(diff --git .+?)(?=(?:</output>|\Z))", re.DOTALL)


def _msg_text(m: dict) -> str:
    c = m.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return json.dumps(c)
    return ""


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _diff_stats(diff_text: str) -> dict:
    files = re.findall(r"^diff --git a/(\S+) b/", diff_text, re.MULTILINE)
    added = len(re.findall(r"^\+(?!\+\+)", diff_text, re.MULTILINE))
    removed = len(re.findall(r"^-(?!--)", diff_text, re.MULTILINE))
    return {"n_files": len(set(files)), "n_lines": added + removed}


def _intermediate_diffs(messages: list[dict]) -> list[dict]:
    """Extract `git diff` outputs that appeared in tool results."""
    rows = []
    for m in messages:
        if m.get("role") != "user":
            continue
        text = _msg_text(m)
        if "diff --git" not in text:
            continue
        for match in _RE_DIFF_BLOCK.finditer(text):
            d = match.group(1).strip()
            if d.startswith("diff --git") and len(d) > 50:
                rows.append(_diff_stats(d))
    return rows


def extract_features(messages: list[dict], final_patch: str | None = None) -> dict[str, Any]:
    """Compute the 10 v1 features over a (possibly partial) trajectory.

    `final_patch` is the agent's submitted diff if available. At inference
    time on a live, in-progress trajectory it's typically empty (zero).
    """
    all_text = "\n".join(_msg_text(m) for m in messages)

    custom_basenames = {_basename(p) for p in _RE_CREATE_PY.findall(all_text)}
    custom_test_runs = sum(
        1 for p in _RE_RUN_PY.findall(all_text) if _basename(p) in custom_basenames
    )

    int_rows = _intermediate_diffs(messages)
    lines_max = max((r["n_lines"] for r in int_rows), default=0)
    files_max = max((r["n_files"] for r in int_rows), default=0)
    growth = (int_rows[-1]["n_lines"] - int_rows[0]["n_lines"]) if len(int_rows) >= 2 else 0

    return {
        "n_msgs": len(messages),
        "n_custom_scripts": len(custom_basenames),
        "n_custom_test_runs": custom_test_runs,
        "n_repo_test_runs": len(_RE_RUN_REPO_TEST.findall(all_text)),
        "n_reverts": len(_RE_REVERT.findall(all_text)),
        "n_intermediate_diffs": len(int_rows),
        "inflight_diff_lines_max": lines_max,
        "inflight_diff_files_max": files_max,
        "diff_size_growth": growth,
        "final_patch_lines": (final_patch or "").count("\n"),
    }
