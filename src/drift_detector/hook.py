#!/usr/bin/env python
"""Claude Code hook: detect coding-agent drift mid-session and block.

Wire-up in `~/.claude/settings.json`:

  {
    "hooks": {
      "PostToolUse": [{"matcher": ".*",
                       "hooks": [{"type": "command",
                                  "command": "drift-hook"}]}],
      "Stop":        [{"matcher": ".*",
                       "hooks": [{"type": "command",
                                  "command": "drift-hook"}]}]
    }
  }

Behavior:
  - Reads Claude Code's transcript from the hook's stdin JSON.
  - Throttles PostToolUse so a full detector pass runs at most once
    per DRIFT_THROTTLE_SECONDS (default 30). Stop ignores throttle.
  - If the detector's score crosses DRIFT_THRESHOLD (default 0.5),
    returns a `decision: block` reason describing the drift, prompting
    the user/agent to decide whether to continue.
  - Never blocks the harness on errors — emits empty JSON instead so
    Claude Code proceeds normally.

Two-tier decision logic:
  struct < STRUCT_LOW  → pass through
  struct ≥ EXTREME     → block immediately (no LLM call)
  STRUCT_LOW ≤ struct < EXTREME → consult LLM judge (Opus by default);
                                  block iff struct ≥ STRUCT_HIGH AND
                                  judge ≥ JUDGE_THRESHOLD

This matches the validated operating point on n=66 trajectories:
  struct>0.7 AND Opus>0.3 → 100% precision, 11.1% recall
(vs structural-alone at struct>0.8: 100% precision, only 1.4% recall.)

Env config:
  DRIFT_STRUCT_LOW         — default 0.5 (below this: pass through)
  DRIFT_STRUCT_HIGH        — default 0.7 (struct part of AND-gate)
  DRIFT_STRUCT_EXTREME     — default 0.8 (above this: block w/o judge)
  DRIFT_JUDGE_THRESHOLD    — default 0.3 (judge part of AND-gate)
  DRIFT_JUDGE_MODEL        — default claude-opus-4-7
  DRIFT_USE_JUDGE          — set to "0" to disable second-tier judge
  DRIFT_THROTTLE_SECONDS   — default 30
  DRIFT_DISABLED           — set to "1" to no-op
  DRIFT_DEBUG              — set to "1" for stderr diagnostics
  DRIFT_STATE_DIR          — default /tmp (throttle state files)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

__all__ = ["main"]


def _log(msg: str) -> None:
    if os.environ.get("DRIFT_DEBUG") == "1":
        sys.stderr.write(f"[drift-hook] {msg}\n")
        sys.stderr.flush()


def _emit(payload: dict | None = None) -> None:
    json.dump(payload or {}, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _content_to_text(content) -> str:
    """Flatten Claude Code's message.content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for b in content:
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "text":
            parts.append(b.get("text") or "")
        elif bt == "tool_use":
            name = b.get("name", "?")
            inp = b.get("input", {})
            parts.append(f"[tool:{name} {json.dumps(inp)[:300]}]")
        elif bt == "tool_result":
            tr = b.get("content")
            if isinstance(tr, list):
                tr = "".join(x.get("text", "") for x in tr if isinstance(x, dict))
            parts.append(f"[tool_result: {tr[:500] if isinstance(tr, str) else ''}]")
    return "\n".join(parts)


def _load_transcript(path: Path) -> list[dict]:
    msgs: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("type") not in ("user", "assistant"):
            continue
        m = rec.get("message") or {}
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _content_to_text(m.get("content", ""))
        if not text:
            continue
        msgs.append({"role": role, "content": text})
    return msgs


def _throttle_ok(session_id: str, event: str, throttle_seconds: int) -> bool:
    """Return True iff enough time has passed since the last check.
    Stop events ignore throttle."""
    if event == "Stop":
        return True
    state_dir = Path(os.environ.get("DRIFT_STATE_DIR", "/tmp"))
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / f"drift_detector_{session_id}.json"
    now = time.time()
    if state_file.exists():
        try:
            last = json.loads(state_file.read_text()).get("last_check_at", 0)
            if (now - last) < throttle_seconds:
                _log(f"throttled (last check {now - last:.0f}s ago < {throttle_seconds}s)")
                return False
        except Exception:
            pass
    state_file.write_text(json.dumps({"last_check_at": now}))
    return True


def _block_reason(assessment, judge_score: float | None) -> str:
    top = ", ".join(f"{f}={c:+.2f}" for f, c in assessment.top_contributors[:3])
    if judge_score is not None:
        confirm = (
            f" LLM judge confirmed (score {judge_score:.2f}). Joint signal "
            f"gives ~100% precision at this operating point on the n=66 "
            f"validation set."
        )
    else:
        confirm = " High structural confidence (no LLM-judge needed)."
    return (
        f"[drift-detector] Drift detected — structural score {assessment.score:.2f}. "
        f"Top features: {top}.{confirm} "
        f"Review the agent's recent direction and decide: continue as-is, give "
        f"corrective guidance, or stop. To dismiss this check for the rest of "
        f"the session, set DRIFT_DISABLED=1."
    )


def main() -> int:
    if os.environ.get("DRIFT_DISABLED") == "1":
        _emit()
        return 0

    raw = sys.stdin.read()
    if not raw.strip():
        _emit()
        return 0
    try:
        hook_input = json.loads(raw)
    except Exception as e:
        _log(f"hook input parse failed: {e}")
        _emit()
        return 0

    event = hook_input.get("hook_event_name", "")
    session_id = hook_input.get("session_id", "default")
    transcript_path_str = hook_input.get("transcript_path", "")
    if not transcript_path_str:
        _log("no transcript_path in hook input")
        _emit()
        return 0
    transcript_path = Path(transcript_path_str)
    if not transcript_path.exists():
        _log(f"transcript path does not exist: {transcript_path}")
        _emit()
        return 0

    throttle = int(os.environ.get("DRIFT_THROTTLE_SECONDS", "30"))
    if not _throttle_ok(session_id, event, throttle):
        _emit()
        return 0

    try:
        messages = _load_transcript(transcript_path)
    except Exception as e:
        _log(f"transcript load failed: {e}")
        _emit()
        return 0

    if len(messages) < 5:
        _log(f"only {len(messages)} messages — too short to score")
        _emit()
        return 0

    # Two-tier thresholds:
    #   struct >= STRUCT_HIGH       -> block immediately (rare, extreme cases)
    #   STRUCT_LOW <= struct < HIGH -> consult LLM judge; block iff judge confirms
    #   struct < STRUCT_LOW         -> pass through
    struct_low = float(os.environ.get("DRIFT_STRUCT_LOW", "0.5"))
    struct_high = float(os.environ.get("DRIFT_STRUCT_HIGH", "0.7"))
    judge_thr = float(os.environ.get("DRIFT_JUDGE_THRESHOLD", "0.3"))
    struct_extreme = float(os.environ.get("DRIFT_STRUCT_EXTREME", "0.8"))
    use_judge = os.environ.get("DRIFT_USE_JUDGE", "1") == "1"

    try:
        from .detector import Detector
        det = Detector(threshold=struct_high)
        assessment = det.assess(messages)
    except Exception as e:
        _log(f"detector failed: {e}")
        _emit()
        return 0

    _log(f"event={event} n_msgs={len(messages)} struct={assessment.score:.3f}")

    # Pass through if structural is below the low threshold.
    if assessment.score < struct_low:
        _emit()
        return 0

    # Extreme-confidence path: block without judge.
    if assessment.score >= struct_extreme:
        _log(f"struct {assessment.score:.2f} >= extreme {struct_extreme} — blocking without judge")
        _emit({"decision": "block",
               "reason": _block_reason(assessment, judge_score=None)})
        return 0

    # Borderline: optionally consult judge. AND-gate: block iff
    # struct >= STRUCT_HIGH and judge >= JUDGE_THRESHOLD.
    if not use_judge:
        _log("DRIFT_USE_JUDGE=0; refusing to block on structural-only borderline")
        _emit()
        return 0

    try:
        from .judge import OpusJudge, judge_unavailable_reason
    except Exception as e:
        _log(f"judge import failed: {e}; pass through")
        _emit()
        return 0
    unavail = judge_unavailable_reason()
    if unavail:
        _log(f"judge unavailable ({unavail}); pass through")
        _emit()
        return 0

    judge_model = os.environ.get("DRIFT_JUDGE_MODEL", "claude-opus-4-7")
    judge = OpusJudge(model=judge_model)
    judge_score = judge.score(messages)
    _log(f"judge ({judge_model}) → {judge_score}")
    if judge_score is None:
        _log("judge call failed; pass through (no block on structural alone)")
        _emit()
        return 0

    if assessment.score >= struct_high and judge_score >= judge_thr:
        _emit({"decision": "block",
               "reason": _block_reason(assessment, judge_score=judge_score)})
        return 0

    _log(f"borderline cleared: struct={assessment.score:.2f}, judge={judge_score:.2f}")
    _emit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
