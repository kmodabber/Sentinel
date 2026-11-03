"""TC-03: split has no leakage (SEN-FR-04)."""
import numpy as np
import pandas as pd

from sentinel_data.split import assign_window, split

GAP_S = 1


def fixture_frame(seed=0):
    """Three windows (Monday benign, Tuesday benign, Tuesday SSH-Patator) of
    random flows over about 3 hours, shuffled so row order isn't time order."""
    rng = np.random.default_rng(seed)
    frames = []
    for day, label, n, start in [
        ("Monday", "BENIGN", 1000, 0),
        ("Tuesday", "BENIGN", 800, 0),
        ("Tuesday", "SSH-Patator", 300, 5_000_000),
    ]:
        frames.append(pd.DataFrame({
            "day": day,
            "label": label,
            "bidirectional_first_seen_ms": start + rng.integers(0, 3 * 3600 * 1000, n),
            "bidirectional_duration_ms": rng.integers(0, 120_000, n),
            "bidirectional_packets": rng.integers(1, 50, n),
        }))
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed)


def test_no_flow_in_two_parts():
    """Every flow ends up in exactly one of train, val or test."""
    out = split(fixture_frame(), GAP_S)
    assert out.index.is_unique
    assert set(out["split"]) == {"train", "val", "test"}


def test_no_flow_crosses_a_boundary():
    """In every window, train ends (plus gap) before val starts, and val
    ends (plus gap) before test starts."""
    out = split(fixture_frame(), GAP_S)
    gap_ms = GAP_S * 1000
    out = out.assign(end=out["bidirectional_first_seen_ms"] + out["bidirectional_duration_ms"])
    for _, g in out.groupby(["day", "label"]):
        by = g.groupby("split")
        assert by["end"].max()["train"] + gap_ms <= by["bidirectional_first_seen_ms"].min()["val"]
        assert by["end"].max()["val"] + gap_ms <= by["bidirectional_first_seen_ms"].min()["test"]


def test_roughly_60_20_20():
    """Each window is close to 60/20/20."""
    out = split(fixture_frame(), GAP_S)
    for _, g in out.groupby(["day", "label"]):
        share = g["split"].value_counts(normalize=True)
        # long flows near a boundary get dropped, so allow some slack
        assert 0.5 < share["train"] < 0.65
        assert 0.12 < share["val"] < 0.22
        assert 0.18 < share["test"] < 0.25


def test_same_input_same_split():
    """Shuffling the rows doesn't change the result."""
    df = fixture_frame()
    # coarse timestamps so lots of flows share a start time, like real floods
    df["bidirectional_first_seen_ms"] = df["bidirectional_first_seen_ms"] // 600_000 * 600_000
    a = split(df, GAP_S)
    b = split(df.sample(frac=1, random_state=99), GAP_S)
    pd.testing.assert_frame_equal(a, b)


def test_ties_at_boundary_stay_together():
    """Flows that start at the same moment never get split across parts."""
    starts = np.array([0, 0, 0, 0, 0, 0, 1_000_000, 1_000_000, 1_000_000, 1_000_000])
    parts = assign_window(starts, starts, gap_ms=0)
    for t in np.unique(starts):
        assert len(set(parts[starts == t]) - {"gap"}) <= 1


def test_long_flow_at_boundary_is_dropped():
    """A flow that starts in train but is still running when val starts is dropped."""
    starts = np.arange(10) * 1000
    ends = starts.copy()
    ends[5] = 50_000
    parts = assign_window(starts, ends, gap_ms=0)
    assert parts[5] == "gap"
