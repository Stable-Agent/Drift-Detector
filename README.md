# drift-detector

> **⚠️ KNOWN BROKEN — DO NOT USE FOR PRODUCTION (as of 2026-05-26)**
>
> The bundled model (`non_llm_detector_v0.pkl`) was trained on a
> hand-curated sample whose resolved/failed classes had systematically
> different feature distributions (median n_msgs 96 vs 108; mean
> n_custom_scripts 2.5 vs 5.1). On a held-out random sample of 400
> resolved trajectories from the same 4 sources, the detector fires at
> **88% false-positive rate** at threshold 0.5 — it essentially predicts
> "failure" on nearly every trajectory.
>
> The previously-claimed "CV AUC 0.76" was real on the biased training
> sample but does not generalize. **A corrected model is in progress.**
> Until then, the SDK and hook will emit a `DeprecationWarning` on use.

Non-LLM failure detector for coding agents (Claude Code, SWE-agent, OpenHands, …),
shipping with a Claude Code hook that surfaces drift to the human for review.

Intended behavior: predict whether an in-flight coding-agent trajectory
is heading toward failure using 10 structural features (no LLM calls).
Signal is real from turn ~10 and saturates around turn ~50 on the
training sample — **but per the warning above, the current model does
not generalize to a representative sample.**

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
