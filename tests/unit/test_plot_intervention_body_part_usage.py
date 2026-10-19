import numpy as np

from scripts.data.plot_intervention_body_part_usage import (
    action_magnitude,
    activation_event_counts,
    body_part_counts,
    filter_short_runs,
    segment_presence_counts,
)


def test_body_part_counts_allow_overlapping_frames():
    deltas = np.zeros((5, 23))
    deltas[0, [0, 7]] = 0.2
    deltas[1, 3] = 0.2
    deltas[2, [4, 15]] = 0.2
    deltas[3, :] = 0.002
    deltas[4, :] = 0.004

    thresholds = {
        "base": 0.003,
        "torso": 0.01,
        "joints": 0.02,
        "grippers": 0.75,
    }
    assert body_part_counts(deltas, thresholds) == {
        "base": 2,
        "torso": 2,
        "left_arm": 1,
        "right_arm": 1,
    }


def test_action_magnitude_does_not_subtract_state_from_base_action():
    action = np.zeros((1, 23))
    state = np.zeros((1, 46))
    action[0, :3] = [0.004, -0.002, 0.0]
    state[0, :3] = [10.0, 10.0, 10.0]
    action[0, 3] = 0.2
    state[0, 3] = 0.1

    magnitude = action_magnitude(action, state, state_start=0)

    np.testing.assert_allclose(magnitude[0, :3], [0.004, 0.002, 0.0])
    np.testing.assert_allclose(magnitude[0, 3], 0.1)


def test_grippers_use_their_dedicated_threshold():
    deltas = np.zeros((3, 23))
    deltas[0, 14] = 0.6
    deltas[1, 14] = 0.8
    deltas[2, 22] = 0.8
    thresholds = {
        "base": 0.003,
        "torso": 0.01,
        "joints": 0.02,
        "grippers": 0.75,
    }

    counts = body_part_counts(deltas, thresholds)

    assert counts["left_arm"] == 1
    assert counts["right_arm"] == 1


def test_filter_short_runs_keeps_only_qualifying_consecutive_frames():
    mask = np.asarray([False, True, True, False, True, True, True, False])

    filtered = filter_short_runs(mask, min_frames=3)

    np.testing.assert_array_equal(
        filtered,
        [False, False, False, False, True, True, True, False],
    )


def test_segment_presence_counts_each_body_once_per_intervention_segment():
    intervention = np.asarray([True, True, True, False, True, True])
    activity = {
        "base": np.asarray([True, True, False, False, True, True]),
        "torso": np.asarray([False, True, True, False, False, False]),
        "left_arm": np.asarray([False, False, False, False, True, False]),
        "right_arm": np.zeros(6, dtype=bool),
    }

    counts, num_segments = segment_presence_counts(activity, intervention)

    assert num_segments == 2
    assert counts == {"base": 2, "torso": 1, "left_arm": 1, "right_arm": 0}


def test_activation_event_counts_multiple_runs_within_one_segment():
    activity = {
        "base": np.asarray([True, True, False, True, True]),
        "torso": np.asarray([False, True, True, False, False]),
        "left_arm": np.zeros(5, dtype=bool),
        "right_arm": np.zeros(5, dtype=bool),
    }

    counts = activation_event_counts(activity)

    assert counts == {"base": 2, "torso": 1, "left_arm": 0, "right_arm": 0}
