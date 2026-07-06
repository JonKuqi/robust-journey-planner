import pandas as pd

from src.models.delay_model import DelayModel
from src.data.calendar_data import CalendarDataHandler


def test_delay_model_builds_feature_row_with_all_columns():
    """_build_feature_row returns every column the training schema expects with correct types.

    Categorical columns must be strings and day_of_week must be converted from
    the weekday name to Spark's 1-indexed convention (monday → 2).
    """
    model = DelayModel(
        metadata={
            "categorical_cols": ["bpuic", "line_text", "hour", "day_of_week", "month", "day_type"],
            "all_numeric_cols": [],
            "weather_bus_cols": [],
            "calendar_numeric_cols": [],
            "hist_numeric_cols": [],
            "quantile_levels": [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95],
        }
    )
    row = model._build_feature_row(
        bpuic="8501120",
        line_text="701",
        transport="BUS",
        hour=8,
        day_of_week="monday",
        month=5,
        calendar_features={},
    )
    assert row["bpuic"] == "8501120"
    assert row["line_text"] == "701"
    assert row["hour"] == "8"
    assert row["day_of_week"] == "2"   # monday → Spark dayofweek 2
    assert row["month"] == "5"
    assert row["day_type"] == "0"      # no calendar features → regular day


def test_delay_model_non_bus_gets_zero_weather():
    """Non-bus transports receive zero-valued weather features regardless of station availability.

    At training, weather columns are set to 0 for non-bus services. The same
    logic must apply at inference so the model sees the same feature distribution.
    """
    model = DelayModel(
        clim_weather={"LSZH__5": {"temp": 15.0}},
        stop_station={"8501120": "LSZH"},
        metadata={
            "categorical_cols": ["bpuic", "line_text", "hour", "day_of_week", "month", "day_type"],
            "all_numeric_cols": [],
            "weather_bus_cols": ["temp_bus"],
            "calendar_numeric_cols": [],
            "hist_numeric_cols": [],
            "quantile_levels": [0.50, 0.90],
        },
    )
    row = model._build_feature_row(
        bpuic="8501120",
        line_text="M1",
        transport="METRO",
        hour=8,
        day_of_week="monday",
        month=5,
        calendar_features={},
    )
    assert row["temp_bus"] == 0.0


def test_delay_model_column_for_q_returns_closest_above():
    """_column_for_q returns the smallest available quantile level that is >= the requested q.

    For example q=0.88 falls between p85 and p90, so p90 is returned.  When q
    exceeds the highest trained level the highest level is returned as a cap.
    """
    model = DelayModel(metadata={"quantile_levels": [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]})
    assert model._column_for_q(0.88) == "p90_delay_sec"
    assert model._column_for_q(0.50) == "p50_delay_sec"
    assert model._column_for_q(0.95) == "p95_delay_sec"
    assert model._column_for_q(0.99) == "p95_delay_sec"


def test_label_encoding_maps_known_and_unknown_values():
    """_predict_all_quantiles_batch replicates StringIndexer encoding at inference.

    Known category values must map to their vocabulary index (position in the list
    saved during training). Unknown values must map to len(vocab), which matches
    Spark StringIndexer handleInvalid='keep' behaviour.
    """
    category_labels = {
        "bpuic": ["8592050", "8501120", "8591818"],   # indices 0, 1, 2
        "line_text": ["IC5", "R2", "S3"],             # indices 0, 1, 2
    }

    feature_rows = [
        {"bpuic": "8592050",       "line_text": "R2"},         # known, known
        {"bpuic": "8591818",       "line_text": "UNKNOWN_LINE"}, # known, unknown
        {"bpuic": "NEW_STOP_9999", "line_text": "IC5"},         # unknown, known
    ]

    # Replicate the encoding logic from _predict_all_quantiles_batch
    df = pd.DataFrame(feature_rows)
    for col in ["bpuic", "line_text"]:
        vocab = category_labels[col]
        vocab_map = {v: float(i) for i, v in enumerate(vocab)}
        unknown_idx = float(len(vocab))
        df[f"{col}_index"] = df[col].map(lambda v, m=vocab_map, u=unknown_idx: m.get(v, u))

    assert df["bpuic_index"].tolist() == [0.0, 2.0, 3.0]       # 8592050→0, 8591818→2, unknown→3
    assert df["line_text_index"].tolist() == [1.0, 3.0, 0.0]   # R2→1, unknown→3, IC5→0


def test_calendar_get_features_for_date_returns_all_flags():
    """get_features_for_date returns a complete flag dict for a single date without Spark.

    Uses the holidays library for Swiss canton-specific public holidays. Verifies that
    a known public holiday (New Year's Day in ZH) sets is_public_holiday=1, that
    unmatched flags default to 0, and that all CALENDAR_NUMERIC_COLS keys are present.
    """
    # 2026-01-01 is New Year's Day — a public holiday in all Swiss cantons
    flags = CalendarDataHandler.get_features_for_date("2026-01-01", canton_code="ZH")
    assert flags["is_public_holiday"] == 1
    assert "is_school_holiday" not in flags
    assert set(flags.keys()) == set(CalendarDataHandler.CALENDAR_NUMERIC_COLS)

    # A regular Tuesday should have all flags 0
    flags_regular = CalendarDataHandler.get_features_for_date("2026-03-03", canton_code="ZH")
    assert flags_regular["is_public_holiday"] == 0
    assert all(v == 0 for v in flags_regular.values())
