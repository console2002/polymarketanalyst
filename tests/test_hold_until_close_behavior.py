import pandas as pd

from dashboard_processing import _resolve_market_winner, calculate_market_trade_records
from second_entry_processing import calculate_market_trade_records_with_second_entry


def _build_market_df(up_prices, down_prices=None):
    if down_prices is None:
        down_prices = [1 - p for p in up_prices]
    timestamps = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 00:05:00",
            "2024-01-01 00:10:00",
            "2024-01-01 00:15:00",
        ]
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "UpPrice": up_prices,
            "DownPrice": down_prices,
            "TargetTime": ["2024-01-01 00:00:00"] * len(up_prices),
        }
    )


def test_primary_processing_holds_to_market_close_when_entry_is_at_or_above_hold_threshold():
    df = _build_market_df([0.50, 0.70, 0.62, 0.40])

    records = calculate_market_trade_records(
        df,
        time_column="timestamp",
        minutes_after_open=0,
        entry_threshold=0.60,
        hold_until_close_threshold=0.65,
        time_format="%Y-%m-%d %H:%M:%S",
    )

    record = records[0]
    assert record["entry_price"] == 0.70
    assert record["exit_reason"] == "held_to_close"
    assert record["exit_time"] == pd.Timestamp("2024-01-01 00:15:00")


def test_primary_processing_always_holds_to_market_close_after_entry():
    df = _build_market_df([0.50, 0.62, 0.66, 0.40])

    records = calculate_market_trade_records(
        df,
        time_column="timestamp",
        minutes_after_open=0,
        entry_threshold=0.60,
        hold_until_close_threshold=0.65,
        time_format="%Y-%m-%d %H:%M:%S",
    )

    record = records[0]
    assert record["entry_price"] == 0.62
    assert record["exit_reason"] == "held_to_close"
    assert record["exit_time"] == pd.Timestamp("2024-01-01 00:15:00")


def test_second_entry_processing_holds_to_market_close_when_entry_is_at_or_above_hold_threshold():
    df = _build_market_df([0.50, 0.70, 0.62, 0.40])

    records = calculate_market_trade_records_with_second_entry(
        df,
        time_column="timestamp",
        minutes_after_open=0,
        entry_threshold=0.60,
        hold_until_close_threshold=0.65,
        time_format="%Y-%m-%d %H:%M:%S",
        second_entry_threshold=0.75,
        second_entry_mode="off",
        use_cache=False,
    )

    record = records[0]
    assert record["entry_price"] == 0.70
    assert record["exit_reason"] == "held_to_close"
    assert record["exit_time"] == pd.Timestamp("2024-01-01 00:15:00")


def test_second_entry_does_not_trigger_after_market_reaches_99_percent():
    df = _build_market_df([0.50, 0.77, 0.99, 0.40], down_prices=[0.50, 0.23, 0.01, 0.60])

    records = calculate_market_trade_records_with_second_entry(
        df,
        time_column="timestamp",
        minutes_after_open=0,
        entry_threshold=0.70,
        hold_until_close_threshold=0.65,
        time_format="%Y-%m-%d %H:%M:%S",
        second_entry_threshold=0.40,
        second_entry_mode="additive",
        use_cache=False,
    )

    record = records[0]
    assert record["trigger_price"] == 0.77
    assert record["second_entry_time"] is None
    assert record["position_multiplier"] == 1
    assert record["exit_time"] == pd.Timestamp("2024-01-01 00:10:00")
    assert record["market_close_time"] == pd.Timestamp("2024-01-01 00:10:00")
    assert record["outcome"] == "Win"
    assert record["exit_price"] == 1.0


def test_primary_processing_uses_volume_as_tiebreaker_when_close_prices_are_tied():
    df = _build_market_df([0.50, 0.80, 0.20, 0.50], down_prices=[0.50, 0.20, 0.80, 0.50])
    df["UpVol"] = [10, 12, 15, 22]
    df["DownVol"] = [10, 9, 7, 0]

    records = calculate_market_trade_records(
        df,
        time_column="timestamp",
        minutes_after_open=0,
        entry_threshold=0.75,
        hold_until_close_threshold=0.65,
        time_format="%Y-%m-%d %H:%M:%S",
    )

    record = records[0]
    assert record["close_up"] == record["close_down"] == 0.50
    assert record["winning_side"] == "Up"
    assert record["winning_side_method"] == "volume"
    assert record["outcome"] == "Win"


def test_second_entry_processing_uses_volume_as_tiebreaker_when_close_prices_are_tied():
    df = _build_market_df([0.50, 0.80, 0.20, 0.50], down_prices=[0.50, 0.20, 0.80, 0.50])
    df["UpVol"] = [3, 8, 11, 18]
    df["DownVol"] = [4, 6, 5, 0]

    records = calculate_market_trade_records_with_second_entry(
        df,
        time_column="timestamp",
        minutes_after_open=0,
        entry_threshold=0.75,
        hold_until_close_threshold=0.65,
        time_format="%Y-%m-%d %H:%M:%S",
        second_entry_threshold=0.45,
        second_entry_mode="off",
        use_cache=False,
    )

    record = records[0]
    assert record["winning_side"] == "Up"
    assert record["winning_side_method"] == "volume"
    assert record["outcome"] == "Win"


def test_resolve_market_winner_uses_pre_collapse_snapshot_when_both_close_prices_are_near_zero():
    market_group = pd.DataFrame(
        {
            "UpPrice": [0.90, 0.97, 0.01],
            "DownPrice": [0.10, 0.03, 0.01],
        }
    )

    winner, method = _resolve_market_winner(market_group, close_up=0.01, close_down=0.01)

    assert winner == "Up"
    assert method == "pre_collapse_price"
