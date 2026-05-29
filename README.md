# drift-detector

Non-LLM failure detector for coding agents (Claude Code, SWE-agent, OpenHands, …),
shipping with a Claude Code hook that surfaces drift to the human for review.

Predicts whether an in-flight coding-agent trajectory is heading toward
failure using 10 structural features (no LLM calls).

**Performance numbers** (5-fold instance-level CV, n=746 SWE-bench
trajectories from livesweagent + sonar at opus-4.5, sonnet-4.5, gemini-3):

| signal | mean held-out AUC |
|---|---|
| structural detector alone | 0.70 ± 0.04 |
| nearest-neighbor K=50 alone | 0.75 ± 0.03 |
| Opus 4.7 LLM judge (n=66) | 0.74 |
| **avg(structural, NN K=50)** | **0.79 ± 0.04** |

**Operating points (pooled across 5 folds, n=746 predictions):**

| config | precision | recall |
|---|---|---|
| structural only @ 0.8 | 100% | 1.4% |
| **struct≥0.5 AND NN≥0.7** | **100%** (31/31) | **9.0%** |
| struct≥0.5 AND NN≥0.6 | 96% (66/69) | 19.1% |
| struct≥0.5 AND NN≥0.5 | 91% (108/119) | 31.2% |
| struct≥0.7 AND NN≥0.5 | 100% (32/32) | 9.2% |

The default Claude Code hook uses a two-tier AND-gate: cheap structural
detector always runs; in the borderline range, a K-NN retrieval over
bundled reference embeddings (~50 ms, no LLM call) confirms before
blocking. At the recommended operating point this gives 6× the recall of
structural-alone at the same near-zero false-positive target.

An earlier version of this package claimed AUC 0.76 — that was measured
on a hand-curated training sample whose resolved/failed classes had
systematically different feature distributions, and it does not
generalize. Numbers above are honest held-out CV.

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
- `struct ≥ DRIFT_STRUCT_EXTREME` (0.85): block immediately.
- borderline range: consult the K-NN tier. Block iff `struct ≥ STRUCT_HIGH`
  AND `nn ≥ NN_THRESHOLD`. If the NN call fails for any reason, the hook
  passes through — it never blocks on structural-only borderline.

Optionally enable a third Opus judge tier with `DRIFT_USE_JUDGE=1`;
the hook will block if either second tier confirms.

The block payload includes the top contributing features and which tier
confirmed. Claude Code surfaces it in the chat, pausing the agent and
letting you decide: continue as-is, give corrective guidance, or stop.

**Throttling:** `PostToolUse` runs at most once per
`DRIFT_THROTTLE_SECONDS` (default 30). `Stop` ignores throttle so
end-of-turn checks always run.

### Environment

| Variable | Default | Effect |
|---|---|---|
| `DRIFT_STRUCT_LOW` | `0.5` | Below this: pass through |
| `DRIFT_STRUCT_HIGH` | `0.5` | Structural side of the AND-gate |
| `DRIFT_STRUCT_EXTREME` | `0.85` | Above this: block immediately |
| `DRIFT_NN_THRESHOLD` | `0.7` | NN side of the AND-gate |
| `DRIFT_NN_K` | `50` | Number of neighbors to average |
| `DRIFT_USE_NN` | `1` | Set to `0` to disable NN tier |
| `DRIFT_USE_JUDGE` | `0` | Set to `1` to enable Opus judge tier |
| `DRIFT_JUDGE_THRESHOLD` | `0.3` | Judge side of the AND-gate (when enabled) |
| `DRIFT_JUDGE_MODEL` | `claude-opus-4-7` | Model for the judge tier |
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

The K-NN tier uses `sentence-transformers/all-MiniLM-L6-v2` (90 MB on
first download) to embed the trajectory, then retrieves the K=50 most
similar trajectories from a bundled 746-trajectory reference set
(~1 MB). The returned score is the failure rate among those K
neighbors. 5-fold instance-level CV AUC: 0.75 ± 0.03.

The optional Opus 4.7 judge prompt asks for a calibrated 0–1
probability of failure given the truncated trajectory text. Held-out
AUC 0.74 on the n=66 LLM-judge pilot. Comparable signal to K-NN but
requires a `claude` CLI call (slower, subject to rate limits).

## License

MIT.
