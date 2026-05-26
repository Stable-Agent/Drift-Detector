# drift-detector

Non-LLM failure detector for coding agents (Claude Code, SWE-agent, OpenHands, …),
shipping with a Claude Code hook that surfaces drift to the human for review.

Predicts whether an in-flight coding-agent trajectory is heading toward
failure using 10 structural features (no LLM calls).

**Honest performance numbers** (2026-05-26 retrain on a representative
random sample, 80/20 stratified split, n=746 SWE-bench trajectories from
livesweagent + sonar at opus-4.5, sonnet-4.5, gemini-3):

| threshold | recall | precision | FP rate |
|---|---|---|---|
| 0.5 | 46% | 58% | 29% |
| 0.7 | 10% | 78% | 2.5% |
| 0.8 | 1.4% | 100% | 0% |

Held-out AUC: 0.65 (per-source 0.61–0.74). An earlier version of this
package claimed AUC 0.76 — that was measured on a hand-curated training
sample whose resolved/failed classes had systematically different feature
distributions, and it does not generalize. The current model is honest
but signal-limited; the structural features alone are not strong enough
to reach near-zero false-positives at meaningful recall. AND-gating with
additional signal types (in progress) is the path to better operating
points.

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

**Behavior:** when the detector's score exceeds the threshold, the hook
returns a `decision: block` payload with a reason naming the top contributing
features. Claude Code surfaces this in the chat, pausing the agent and
letting you decide: continue as-is, give corrective guidance, or stop.

**Throttling:** `PostToolUse` runs at most once per
`DRIFT_THROTTLE_SECONDS` (default 30). `Stop` ignores throttle so
end-of-turn checks always run.

### Environment

| Variable | Default | Effect |
|---|---|---|
| `DRIFT_THRESHOLD` | `0.5` | Score above this fires the block |
| `DRIFT_THROTTLE_SECONDS` | `30` | Min seconds between PostToolUse checks |
| `DRIFT_DISABLED` | unset | Set to `1` to no-op the hook for a session |
| `DRIFT_DEBUG` | unset | Set to `1` for stderr diagnostics |
| `DRIFT_STATE_DIR` | `/tmp` | Per-session throttle state files |

## What the model is

L1-regularized logistic regression on 10 features:

- **Process:** `n_msgs`, `n_custom_scripts`, `n_custom_test_runs`, `n_repo_test_runs`, `n_reverts`
- **In-flight diff:** `n_intermediate_diffs`, `inflight_diff_lines_max`, `inflight_diff_files_max`, `diff_size_growth`, `final_patch_lines`

Trained on 185 SWE-bench trajectories from livesweagent and sonar
(opus-4.5, sonnet-4.5, gemini-3); 5-fold CV AUC 0.76. Dominant features
after L1 selection: `n_msgs`, `n_custom_scripts`, `n_custom_test_runs`.

## License

MIT.
