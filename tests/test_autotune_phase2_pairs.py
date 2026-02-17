import pandas as pd

from autotune import (
    count_valid_parameter_combinations_for_pairs,
    run_coarse_autotune_for_pairs,
)


def test_count_valid_parameter_combinations_for_pairs_counts_only_valid_holds():
    pair_values = [(5, 0.6), (7, 0.7)]
    hold_values = [0.55, 0.6, 0.7]

    count = count_valid_parameter_combinations_for_pairs(
        pair_values,
        hold_values,
        [0.5, 0.6],
        ["additive", "sole"],
    )

    # (5,0.6) => 2 valid holds (0.6, 0.7), (7,0.7) => 1 valid hold (0.7)
    # total = (2 + 1) * 2 second-entry thresholds * 2 modes = 12
    assert count == 12


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
        hold_until_close_threshold_range=[0.6, 0.7],
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
