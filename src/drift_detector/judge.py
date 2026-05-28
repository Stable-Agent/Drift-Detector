"""Optional LLM judge for two-tier drift detection.

Wraps a call to the `claude` CLI (must be on PATH) to ask a stronger
model (default Opus 4.7) whether a trajectory looks like it's heading
toward failure. Returns a continuous probability in [0, 1].

Used by the hook as the second tier when the structural detector's
score is borderline — combining both gives ~100% precision at 11.1%
recall on the validation set vs structural-alone's 100%/1.4% at the
near-zero-FP operating point.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Optional

__all__ = ["OpusJudge", "judge_unavailable_reason"]


_NUM_RE = re.compile(r"\b(0?\.\d+|1\.0+|0\.0+|0|1)\b")

_PROMPT_TEMPLATE = """You are evaluating a coding agent's in-flight trajectory.

The agent was given a bug-fix task in a Python repo. The trajectory below contains its reasoning, tool calls, and tool results so far.

Your job: estimate the probability that this agent will FAIL to produce a patch that passes the hidden test suite. Output a SINGLE number between 0.0 and 1.0 — nothing else. 0.0 means certain success; 1.0 means certain failure; 0.5 means equally likely either way.

Most coding agents on hard bugs make recoverable mistakes; default toward 0.5-0.7 unless you see strong evidence either way.

=== TRAJECTORY (most recent {n_msgs} messages, last {n_chars} chars) ===

{trajectory_text}

=== END TRAJECTORY ===

Output ONLY a single number in [0.0, 1.0]. No prose, no explanation."""


def _msg_text_for_judge(m: dict) -> str:
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
                parts.append(f"[{nm} {json.dumps(inp)[:500] if not isinstance(inp, str) else inp[:500]}]")
            elif bt in ("tool_result", "tool_output"):
                tr = b.get("content") or b.get("output") or ""
                if isinstance(tr, list):
                    tr = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in tr)
                parts.append(str(tr)[:2000])
        return "\n".join(parts)
    c = m.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return json.dumps(c)
    return ""


def _format_trajectory(messages: list[dict], max_chars: int) -> str:
    lines = []
    for i, m in enumerate(messages):
        role = m.get("role", "?")
        text = _msg_text_for_judge(m)
        if not text:
            continue
        lines.append(f"[{i}] {role}: {text[:3000]}")
    full = "\n".join(lines)
    return full[-max_chars:] if len(full) > max_chars else full


def judge_unavailable_reason() -> Optional[str]:
    """Return a string reason if the judge can't run, else None."""
    if not shutil.which("claude"):
        return "claude CLI not on PATH"
    return None


class OpusJudge:
    """Lightweight wrapper around `claude -p --model <opus>`.

    Defaults to claude-opus-4-7. Falls back to None on call failure so
    the hook can gracefully skip the judge tier (and refuse to block,
    since the AND-gate requires both signals).
    """

    def __init__(self, model: str = "claude-opus-4-7",
                 max_chars: int = 30000,
                 timeout: int = 60):
        self.model = model
        self.max_chars = max_chars
        self.timeout = timeout
        self._claude_bin = shutil.which("claude")

    def score(self, messages: list[dict]) -> Optional[float]:
        """Return P(failure) in [0, 1] or None if the call/parse failed."""
        if not self._claude_bin:
            return None
        if not messages:
            return None
        traj_text = _format_trajectory(messages, self.max_chars)
        prompt = _PROMPT_TEMPLATE.format(
            n_msgs=len(messages), n_chars=len(traj_text),
            trajectory_text=traj_text)
        try:
            result = subprocess.run(
                [self._claude_bin, "-p", "--model", self.model, prompt],
                capture_output=True, text=True, timeout=self.timeout,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        if result.returncode != 0:
            return None
        m = _NUM_RE.search(result.stdout or "")
        if not m:
            return None
        try:
            v = float(m.group(1))
        except ValueError:
            return None
        if not (0.0 <= v <= 1.0):
            return None
        return v
