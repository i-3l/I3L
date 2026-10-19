import numpy as np

from iiil.gello.utils.joint_utils import expand_goal_to_full_dof


def test_expand_goal_matches_dof_returns_copy():
    goal = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    current = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    expanded = expand_goal_to_full_dof(goal, current, 3)

    assert np.allclose(expanded, goal)
    assert expanded is not goal  # ensure copy


def test_expand_goal_pads_when_shorter():
    goal = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    current = np.arange(10, dtype=np.float32) * 0.1  # length 10

    expanded = expand_goal_to_full_dof(goal, current, 10)

    assert len(expanded) == 10
    assert np.allclose(expanded[: len(goal)], goal)
    assert np.allclose(expanded[len(goal):], current[len(goal):])


def test_expand_goal_raises_on_overflow():
    goal = np.zeros(5, dtype=np.float32)
    current = np.zeros(4, dtype=np.float32)

    try:
        expand_goal_to_full_dof(goal, current, 4)
    except ValueError as exc:
        assert "Goal has 5 joints" in str(exc)
    else:
        raise AssertionError("Expected ValueError for oversized goal")


def test_expand_goal_raises_on_current_length_mismatch():
    goal = np.zeros(3, dtype=np.float32)
    current = np.zeros(2, dtype=np.float32)

    try:
        expand_goal_to_full_dof(goal, current, 3)
    except ValueError as exc:
        assert "Current positions length" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched current length")
