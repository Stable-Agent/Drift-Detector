"""Smoke tests for the SDK."""

import warnings

warnings.filterwarnings("ignore")

from drift_detector import Detector, FEATURES, extract_features


def _msg(role, content):
    return {"role": role, "content": content}


def _failing_trajectory_template(n_turns=50):
    """Build a synthetic trajectory that should score HIGH for failure:
    many custom scripts, many turns, no repo tests."""
    msgs = [_msg("user", "Fix bug X in repo")]
    for i in range(n_turns):
        msgs.append(_msg("assistant", f"Let me try approach {i}"))
        msgs.append(_msg("user", f"[tool_result: cat > test_v{i}.py <<EOF\nprint('x')\nEOF\npython test_v{i}.py\n]"))
    return msgs


def _passing_trajectory_template(n_turns=20):
    """Short trajectory with repo-test runs and no custom scripts."""
    msgs = [_msg("user", "Fix bug Y")]
    for i in range(n_turns):
        msgs.append(_msg("assistant", "Running tests"))
        msgs.append(_msg("user", "[tool_result: python -m pytest tests/]"))
    return msgs


def test_features_shape():
    msgs = _failing_trajectory_template(n_turns=20)
    f = extract_features(msgs)
    assert set(f.keys()) == set(FEATURES)
    assert f["n_msgs"] == len(msgs)
    assert f["n_custom_scripts"] == 20  # one new script per turn
    assert f["n_custom_test_runs"] == 20  # ran each one


def test_features_passing_template():
    msgs = _passing_trajectory_template()
    f = extract_features(msgs)
    assert f["n_custom_scripts"] == 0
    assert f["n_repo_test_runs"] == 20  # pytest each turn
    assert f["n_msgs"] == len(msgs)


def test_detector_scores_failing_higher_than_passing():
    det = Detector()
    fail_score = det.score(_failing_trajectory_template(n_turns=50))
    pass_score = det.score(_passing_trajectory_template(n_turns=20))
    assert fail_score > pass_score, (
        f"failing template ({fail_score:.3f}) should score above passing ({pass_score:.3f})")


def test_assessment_structure():
    det = Detector()
    a = det.assess(_failing_trajectory_template(n_turns=30))
    assert 0.0 <= a.score <= 1.0
    assert a.threshold == 0.5
    assert isinstance(a.triggered, bool)
    assert set(a.features.keys()) == set(FEATURES)
    assert len(a.top_contributors) == len(FEATURES)
    # Top contributor magnitudes are sorted
    abs_contribs = [abs(c) for _, c in a.top_contributors]
    assert abs_contribs == sorted(abs_contribs, reverse=True)


def test_detector_custom_threshold():
    det = Detector(threshold=0.99)
    a = det.assess(_failing_trajectory_template(n_turns=30))
    assert a.threshold == 0.99
    # At 0.99 threshold most things won't trigger
