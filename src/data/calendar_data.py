from __future__ import annotations

"""Per-canton Swiss public-holiday features for the delay model."""

from datetime import date, timedelta

from src.config.settings import ProjectSettings, get_settings
from src.config.spark_session import get_spark_session


# All 26 Swiss canton codes, matching both iceberg.geo.shapes `region` and the `holidays` library
_CH_CANTONS = (
    "AG", "AI", "AR", "BE", "BL", "BS", "FR", "GE", "GL", "GR",
    "JU", "LU", "NE", "NW", "OW", "SG", "SH", "SO", "SZ", "TG",
    "TI", "UR", "VD", "VS", "ZG", "ZH",
)


class CalendarDataHandler:
    """Build daily public-holiday features, per Swiss canton."""

    CALENDAR_NUMERIC_COLS = [
        "is_public_holiday",
        "is_day_before_public_holiday",
        "is_day_after_public_holiday",
        "is_bridge_day",
    ]

    def __init__(self, spark=None, settings: ProjectSettings | None = None):
        self.settings = settings or get_settings()
        self.spark = spark

    def get_spark(self):
        if self.spark is None:
            self.spark = get_spark_session(self.settings, app_name="calendar-data")
        return self.spark

    def build_calendar_features(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        """Build a (canton_code, operating_day) holiday-flag Spark DataFrame.

        Uses the `holidays` library for per-canton Swiss public holidays across
        all 26 cantons. Returns only rows where at least one flag is non-zero;
        missing (canton, day) pairs are treated as 0 via a left-join + fillna in
        the feature pipeline. The result (~2–4k rows) is small enough to broadcast.
        """
        import holidays as hol_lib

        start_date = start_date or self.settings.model_start_date
        end_date = end_date or self.settings.model_end_date
        start_d = date.fromisoformat(start_date)
        end_d = date.fromisoformat(end_date)
        years = list(range(start_d.year, end_d.year + 1))

        rows: list[tuple] = []
        for canton in _CH_CANTONS:
            ph_dates = set(hol_lib.Switzerland(subdiv=canton, years=years).keys())
            current = start_d
            while current <= end_d:
                yesterday = current - timedelta(days=1)
                tomorrow = current + timedelta(days=1)
                is_ph = int(current in ph_dates)
                is_before = int(tomorrow in ph_dates)
                is_after = int(yesterday in ph_dates)
                # Bridge: Friday sandwiched between a Thursday holiday and the weekend,
                # or Monday sandwiched between the weekend and a Tuesday holiday.
                # Python weekday(): 0=Mon, 4=Fri
                is_bridge = int(
                    (yesterday in ph_dates and current.weekday() == 4)
                    or (tomorrow in ph_dates and current.weekday() == 0)
                )
                if any([is_ph, is_before, is_after, is_bridge]):
                    rows.append((canton, current, is_ph, is_before, is_after, is_bridge))
                current += timedelta(days=1)

        spark = self.get_spark()
        schema = (
            "canton_code STRING, operating_day DATE, "
            "is_public_holiday INT, is_day_before_public_holiday INT, "
            "is_day_after_public_holiday INT, is_bridge_day INT"
        )
        features_df = spark.createDataFrame(rows or [], schema=schema)
        return features_df, list(self.CALENDAR_NUMERIC_COLS)

    def build_stop_canton_mapping(self, bpuic_df=None):
        """Assign each bpuic to its Swiss canton via a Sedona spatial join.

        Covers all Swiss stops from the timetable (not just training stops) so
        inference on any stop has a canton lookup regardless of training coverage.
        Stops that fall outside all canton polygons are dropped; callers should
        apply a ZH fallback for missing entries.
        """
        from pyspark.sql import functions as F

        spark = self.get_spark()
        self._register_sedona(spark)

        max_pub_date = (
            spark.table(self.settings.timetable_stops_table)
            .select(F.max("pub_date"))
            .first()[0]
        )

        stops_with_coords = (
            spark.table(self.settings.timetable_stops_table)
            .filter(F.col("pub_date") == F.lit(max_pub_date))
            .withColumn("bpuic_raw", F.split(F.col("stop_id"), ":").getItem(0).cast("int"))
            .filter(F.col("bpuic_raw").cast("string").startswith("85"))
            .groupBy("bpuic_raw")
            .agg(
                F.avg("stop_lat").alias("stop_lat"),
                F.avg("stop_lon").alias("stop_lon"),
            )
            .withColumn("bpuic", F.col("bpuic_raw").cast("string"))
            .select("bpuic", "stop_lat", "stop_lon")
            .filter(F.col("stop_lat").isNotNull() & F.col("stop_lon").isNotNull())
        )

        canton_shapes = (
            spark.table(self.settings.geo_shapes_table)
            .filter(F.col("country") == "CH")
            .filter(F.col("level") == "canton")
            .select(F.col("region").alias("canton_code"), "wkb_geometry")
        )

        return (
            stops_with_coords.crossJoin(F.broadcast(canton_shapes))
            .filter(
                F.expr("ST_Contains(ST_GeomFromWKB(wkb_geometry), ST_Point(stop_lon, stop_lat))")
            )
            .select("bpuic", "canton_code")
            .distinct()
        )

    @classmethod
    def get_features_for_date(cls, date_str: str, canton_code: str | None = None) -> dict:
        """Return calendar flag dict for a single date without Spark.

        Args:
            date_str: ISO date string (e.g. '2025-12-25').
            canton_code: 2-letter Swiss canton code (e.g. 'VD'). Defaults to 'ZH'
                when None or unrecognised — Zurich has the most conservative holiday
                set, so this is the safest fallback for an unknown stop location.
        """
        import holidays as hol_lib

        target = date.fromisoformat(date_str)
        canton = canton_code if canton_code in _CH_CANTONS else "ZH"
        ph_dates = set(
            hol_lib.Switzerland(subdiv=canton, years=[target.year - 1, target.year, target.year + 1]).keys()
        )

        flags = {col: 0 for col in cls.CALENDAR_NUMERIC_COLS}
        yesterday = target - timedelta(days=1)
        tomorrow = target + timedelta(days=1)

        if target in ph_dates:
            flags["is_public_holiday"] = 1
        if tomorrow in ph_dates:
            flags["is_day_before_public_holiday"] = 1
        if yesterday in ph_dates:
            flags["is_day_after_public_holiday"] = 1
        if yesterday in ph_dates and target.weekday() == 4:
            flags["is_bridge_day"] = 1
        if tomorrow in ph_dates and target.weekday() == 0:
            flags["is_bridge_day"] = 1

        return flags

    @staticmethod
    def _register_sedona(spark) -> None:
        """Register Sedona SQL functions if available."""
        try:
            from sedona.spark import SedonaContext
            SedonaContext.create(spark)
        except Exception:
            pass
