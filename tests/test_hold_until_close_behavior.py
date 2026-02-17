import pandas as pd

from dashboard_processing import calculate_market_trade_records
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
        second_entry_threshold=0.55,
        second_entry_mode="off",
        use_cache=False,
    )

    record = records[0]
    assert record["entry_price"] == 0.70
    assert record["exit_reason"] == "held_to_close"
    assert record["exit_time"] == pd.Timestamp("2024-01-01 00:15:00")
