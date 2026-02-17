import numpy as np

from autotune import count_valid_parameter_combinations


def test_phase1_combination_count_depends_on_minutes_and_entry_only():
    minutes_values = np.arange(5, 16, 5)  # 5, 10, 15
    entry_values = np.array([0.55, 0.6])

    phase1_count = count_valid_parameter_combinations(
        minutes_values,
        entry_values,
        [0.5],
        ["off"],
    )

    assert phase1_count == len(minutes_values) * len(entry_values)


def test_phase1_count_scales_with_second_entry_and_modes():
    minutes_values = np.arange(5, 16, 5)
    entry_values = np.array([0.55, 0.6])

    baseline_count = count_valid_parameter_combinations(
        minutes_values,
        entry_values,
        [0.5],
        ["off"],
    )
    expanded_count = count_valid_parameter_combinations(
        minutes_values,
        entry_values,
        [0.5, 0.55, 0.6],
        ["additive", "sole"],
    )

    assert expanded_count > baseline_count
