# drift-detector

Non-LLM failure detector for coding agents (Claude Code, SWE-agent, OpenHands, …),
shipping with a Claude Code hook that surfaces drift to the human for review.

Predicts whether an in-flight coding-agent trajectory is heading toward
failure using 10 structural features (no LLM calls).

**Performance numbers** (retrain on a representative random sample, 80/20
stratified split, n=746 SWE-bench trajectories from livesweagent + sonar
at opus-4.5, sonnet-4.5, gemini-3):

| operating point | precision | recall |
|---|---|---|
| structural only @ 0.7 | 78% | 15% |
| structural only @ 0.8 | 100% | 1.4% |
| **two-tier: struct≥0.7 AND Opus-judge≥0.3** | **100%** | **11%** |
| structural alone, held-out AUC | 0.65 | — |
| LLM judge alone (Opus 4.7), held-out AUC | 0.74 | — |

The default Claude Code hook uses the two-tier mode: cheap structural
detector always runs; if it crosses the borderline range, a single Opus
4.7 call confirms before blocking. The AND-gate gives 8× the recall of
structural-alone at the same near-zero false-positive target (validated
on n=66 trajectories; 95% CI ±10pp).

An earlier version of this package claimed AUC 0.76 — that was measured
on a hand-curated training sample whose resolved/failed classes had
systematically different feature distributions, and it does not
generalize. Numbers above are honest held-out validation.

## Install

```bash
pip install -e .
```

## SDK

```python
from drift_detector import Detector

det = Detector(threshold=0.5)  # default

# Score in-flight messages (list of {"role": ..., "content": ...} dicts)
assessment = det.assess(messages)

if assessment.triggered:
    print(f"Drift! score={assessment.score:.2f}")
    for feat, contribution in assessment.top_contributors[:3]:
        print(f"  {feat}: {contribution:+.2f}")
```

`assessment.to_dict()` returns a serializable summary.

## Claude Code hook

Install once (`pip install -e .` registers the `drift-hook` entry point), then
add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": [{"matcher": ".*", "hooks": [
      {"type": "command", "command": "drift-hook"}
    ]}],
    "Stop": [{"matcher": ".*", "hooks": [
      {"type": "command", "command": "drift-hook"}
    ]}]
  }
}
```

See `examples/settings.json`.

**Two-tier behavior:**

- `struct < DRIFT_STRUCT_LOW` (0.5): pass through, no block.
- `struct ≥ DRIFT_STRUCT_EXTREME` (0.8): block immediately, no LLM call.
- `DRIFT_STRUCT_LOW ≤ struct < EXTREME`: call the Opus judge. Block iff
  `struct ≥ DRIFT_STRUCT_HIGH` (0.7) AND `judge ≥ DRIFT_JUDGE_THRESHOLD`
  (0.3). If the judge call fails (rate limit, missing claude CLI, etc.),
  the hook passes through — it never blocks on structural-only borderline.

The block payload includes the top contributing features and whether the
judge confirmed. Claude Code surfaces it in the chat, pausing the agent
and letting you decide: continue as-is, give corrective guidance, or stop.

**Throttling:** `PostToolUse` runs at most once per
`DRIFT_THROTTLE_SECONDS` (default 30). `Stop` ignores throttle so
end-of-turn checks always run.

### Environment

| Variable | Default | Effect |
|---|---|---|
| `DRIFT_STRUCT_LOW` | `0.5` | Below this: pass through |
| `DRIFT_STRUCT_HIGH` | `0.7` | Structural side of the AND-gate |
| `DRIFT_STRUCT_EXTREME` | `0.8` | Above this: block without judge |
| `DRIFT_JUDGE_THRESHOLD` | `0.3` | Judge side of the AND-gate |
| `DRIFT_JUDGE_MODEL` | `claude-opus-4-7` | Model for the second tier |
| `DRIFT_USE_JUDGE` | `1` | Set to `0` to disable second-tier judge entirely |
| `DRIFT_THROTTLE_SECONDS` | `30` | Min seconds between PostToolUse checks |
| `DRIFT_DISABLED` | unset | Set to `1` to no-op the hook for a session |
| `DRIFT_DEBUG` | unset | Set to `1` for stderr diagnostics |
| `DRIFT_STATE_DIR` | `/tmp` | Per-session throttle state files |

## What the model is

L1-regularized logistic regression on 10 features:

- **Process:** `n_msgs`, `n_custom_scripts`, `n_custom_test_runs`, `n_repo_test_runs`, `n_reverts`
- **In-flight diff:** `n_intermediate_diffs`, `inflight_diff_lines_max`, `inflight_diff_files_max`, `diff_size_growth`, `final_patch_lines`

Trained on 746 SWE-bench trajectories (random sample, stratified across
livesweagent + sonar at opus-4.5, sonnet-4.5, gemini-3); 80/20 held-out
AUC 0.65. Dominant features after L1 selection: `n_msgs` (+0.73),
`diff_size_growth` (+0.55), `n_repo_test_runs` (−0.53; agents that run
the actual test suite succeed), `n_custom_scripts` (+0.30).

The Opus 4.7 judge prompt asks for a calibrated 0–1 probability of
failure given the truncated trajectory text. Held-out AUC 0.74 on the
n=66 LLM-judge pilot.

## License

MIT.
