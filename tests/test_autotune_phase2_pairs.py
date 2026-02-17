import pandas as pd

from autotune import (
    count_valid_parameter_combinations_for_pairs,
    run_coarse_autotune_for_pairs,
)


def test_count_valid_parameter_combinations_for_pairs_counts_cartesian_product():
    pair_values = [(5, 0.6), (7, 0.7)]

    count = count_valid_parameter_combinations_for_pairs(
        pair_values,
        [0.5, 0.6],
        ["additive", "sole"],
    )

    assert count == len(pair_values) * 2 * 2


def test_run_coarse_autotune_for_pairs_never_scores_non_shortlisted_pairs():
    seen_pairs = set()

    def calculate_metrics(_df, _time_col, minutes, entry, _hold, _mode, _second):
        seen_pairs.add((round(float(minutes), 6), round(float(entry), 2)))
        return {
            "strike_rate": 0.7,
            "win_rate_needed": 0.6,
            "total_count": 250,
            "expectancy": 1.2,
            "expected_pnl": 300,
        }

    shortlisted_pairs = [(5.0, 0.6), (9.0, 0.65)]
    results = run_coarse_autotune_for_pairs(
        pd.DataFrame(),
        "timestamp",
        calculate_metrics,
        minutes_entry_pairs=shortlisted_pairs,
        hold_until_close_threshold=1.0,
        second_entry_threshold_range=[0.5],
        modes=["additive"],
    )

    output_pairs = {
        (round(float(row["minutes_after_open"]), 6), round(float(row["entry_threshold"]), 2))
        for row in results
    }
    expected_pairs = {(5.0, 0.6), (9.0, 0.65)}

    assert seen_pairs == expected_pairs
    assert output_pairs == expected_pairs


def test_run_coarse_autotune_for_pairs_progress_tracks_true_candidate_count():
    progress_events = []

    def progress_callback(current_step, total_steps, _message):
        progress_events.append((current_step, total_steps))

    def calculate_metrics(_df, _time_col, _minutes, _entry, _hold, _mode, _second):
        return {
            "strike_rate": 0.7,
            "win_rate_needed": 0.6,
            "total_count": 250,
            "expectancy": 1.2,
            "expected_pnl": 300,
        }

    run_coarse_autotune_for_pairs(
        pd.DataFrame(),
        "timestamp",
        calculate_metrics,
        minutes_entry_pairs=[(5.0, 0.65)],
        hold_until_close_threshold=1.0,
        second_entry_threshold_range=[0.5, 0.55],
        modes=["additive", "sole"],
        progress_callback=progress_callback,
    )

    # valid evaluations = 1 pair * 2 second-entry thresholds * 2 modes = 4
    assert progress_events
    assert progress_events[-1] == (4, 4)
