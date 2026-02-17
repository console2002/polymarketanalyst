import numpy as np

from autotune import count_valid_parameter_combinations


def test_phase1_combination_count_uses_single_hold_value():
    minutes_values = np.arange(5, 16, 5)  # 5, 10, 15
    entry_values = np.array([0.55, 0.6])
    phase1_hold_values = np.array([0.6])

    phase1_count = count_valid_parameter_combinations(
        minutes_values,
        entry_values,
        phase1_hold_values,
        [0.5],
        ["off"],
    )

    assert phase1_count == len(minutes_values) * len(entry_values)


def test_phase1_count_is_lower_than_hold_sweep_count():
    minutes_values = np.arange(5, 16, 5)
    entry_values = np.array([0.55, 0.6])

    fixed_hold_count = count_valid_parameter_combinations(
        minutes_values,
        entry_values,
        [0.6],
        [0.5],
        ["off"],
    )
    hold_sweep_count = count_valid_parameter_combinations(
        minutes_values,
        entry_values,
        [0.6, 0.65, 0.7],
        [0.5],
        ["off"],
    )

    assert hold_sweep_count > fixed_hold_count
