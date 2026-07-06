from __future__ import annotations

"""Delay-model data preparation from Assignment 2 Part IV.

This module keeps the modeling scope, cleaning, weather/calendar joins, and
feature table construction used by Part IV, without the exploratory plots and
notebook-only cells.
"""

from typing import Iterable

from src.config.settings import ProjectSettings, get_settings
from src.config.spark_session import get_spark_session
from .calendar_data import CalendarDataHandler
from .weather_data import WeatherDataHandler


class DelayDataHandler:
    """Build Spark DataFrames for arrival-delay modeling."""

    def __init__(self, spark=None, settings: ProjectSettings | None = None):
        """Initialize the delay data handler.

        Args:
            spark: Optional existing Spark session.
            settings: Shared project settings.
        """
        self.settings = settings or get_settings()
        self.spark = spark

    def get_spark(self):
        """Return the configured Spark session, creating it lazily if needed."""
        if self.spark is None:
            self.spark = get_spark_session(self.settings, app_name="delay-data-handler")
        return self.spark

    @staticmethod
    def clean_status_col(col):
        """Normalize the Istdaten status column used in Assignment 2 Part IV."""
        from pyspark.sql import functions as F

        return F.when(F.trim(col) == "", "PROGNOSE").otherwise(F.upper(F.trim(col)))

    def build_scope_stops(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        region_uuids: Iterable[str] | None = None,
        operator_ids: Iterable[str] | None = None,
        force_rebuild_region_stops: bool = False,
    ):
        """Return regional stops, or fallback to the old operator-derived scope."""
        from pyspark.sql import functions as F

        spark = self.get_spark()
        start_date = start_date or self.settings.model_start_date
        end_date = end_date or self.settings.model_end_date
        region_uuids = tuple(region_uuids if region_uuids is not None else self.settings.region_uuids)

        if region_uuids:
            return self._build_spark_region_scope_stops(region_uuids)

        operator_ids = tuple(operator_ids if operator_ids is not None else self.settings.operator_ids)

        df = (
            spark.table(self.settings.istdaten_table)
            .filter(F.col("operating_day").between(start_date, end_date))
            .filter(F.col("bpuic").isNotNull())
        )

        if operator_ids:
            df = df.filter(F.col("operator_id").isin(*operator_ids))

        return df.select(F.col("bpuic").cast("string").alias("bpuic")).distinct()

    def _build_spark_region_scope_stops(self, region_uuids: Iterable[str]):
        """Build regional stop ids in Spark without touching Trino CSA tables."""
        from pyspark.sql import functions as F

        spark = self.get_spark()
        self._register_sedona_sql(spark)

        max_pub_date = spark.table(self.settings.timetable_stops_table).select(F.max("pub_date")).first()[0]

        stops_df = (
            spark.table(self.settings.timetable_stops_table)
            .filter(F.col("pub_date") == F.lit(max_pub_date))
            .withColumn("bpuic", F.split(F.col("stop_id"), ":").getItem(0).cast("int"))
            .filter(F.col("bpuic").cast("string").startswith("85"))
            .groupBy("bpuic")
            .agg(
                F.first("stop_name", ignorenulls=True).alias("stop_name"),
                F.avg("stop_lat").alias("stop_lat"),
                F.avg("stop_lon").alias("stop_lon"),
            )
            .filter(F.col("stop_lat").isNotNull() & F.col("stop_lon").isNotNull())
        )

        regions_df = (
            spark.table(getattr(self.settings, "geo_shapes_table", "iceberg.geo.shapes"))
            .filter(F.col("uuid").cast("string").isin([str(region) for region in region_uuids]))
            .select("wkb_geometry")
        )

        return (
            stops_df.crossJoin(F.broadcast(regions_df))
            .filter(F.expr("ST_Contains(ST_GeomFromWKB(wkb_geometry), ST_Point(stop_lon, stop_lat))"))
            .select(F.col("bpuic").cast("string").alias("bpuic"))
            .distinct()
        )

    @staticmethod
    def _register_sedona_sql(spark) -> None:
        """Register Sedona SQL functions when the Python package is available."""
        try:
            from sedona.spark import SedonaContext

            SedonaContext.create(spark)
        except Exception:
            pass

    def build_raw_delay_data(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        region_uuids: Iterable[str] | None = None,
        operator_ids: Iterable[str] | None = None,
        restrict_to_scope_stops: bool = True,
        force_rebuild_region_stops: bool = False,
    ):
        """Build the raw Part IV delay modeling scope."""
        from pyspark.sql import functions as F

        spark = self.get_spark()
        start_date = start_date or self.settings.model_start_date
        end_date = end_date or self.settings.model_end_date
        region_uuids = tuple(region_uuids if region_uuids is not None else self.settings.region_uuids)
        operator_ids = tuple(
            operator_ids if operator_ids is not None else (() if region_uuids else self.settings.operator_ids)
        )

        raw = (
            spark.table(self.settings.istdaten_table)
            .filter(F.col("operating_day").between(start_date, end_date))
            .filter((F.col("failed") == False) | F.col("failed").isNull())
            .filter(F.col("arr_time").isNotNull())
            .filter(F.col("arr_actual").isNotNull())
            .filter(F.col("arr_status").isNotNull())
            .withColumn("clean_arr_status", self.clean_status_col(F.col("arr_status")))
            .filter(F.col("clean_arr_status") != "UNBEKANNT")
            .withColumn("bpuic", F.col("bpuic").cast("string"))
        )

        if operator_ids:
            raw = raw.filter(F.col("operator_id").isin(*operator_ids))

        if restrict_to_scope_stops:
            scope_stops = self.build_scope_stops(
                start_date=start_date,
                end_date=end_date,
                region_uuids=region_uuids,
                operator_ids=operator_ids,
                force_rebuild_region_stops=force_rebuild_region_stops,
            )
            raw = raw.join(scope_stops, on="bpuic", how="inner")

        return (
            raw.withColumn("arr_time_ts", F.col("arr_time").cast("timestamp"))
            .withColumn("arr_actual_ts", F.col("arr_actual").cast("timestamp"))
            .filter(F.col("arr_time_ts").isNotNull() & F.col("arr_actual_ts").isNotNull())
            .withColumn("delay_sec", F.col("arr_actual_ts").cast("long") - F.col("arr_time_ts").cast("long"))
            .withColumn("delay_min", F.col("delay_sec") / F.lit(60.0))
            .withColumn("year", F.year("operating_day"))
            .withColumn("month", F.month("operating_day"))
            .withColumn("day_of_week", F.dayofweek("operating_day"))
            .withColumn("hour", F.hour("arr_time_ts"))
            .select(
                "operating_day",
                "arr_time_ts",
                "arr_actual_ts",
                "year",
                "month",
                "day_of_week",
                "hour",
                "bpuic",
                "stop_name",
                "trip_id",
                "operator_id",
                "operator_abrv",
                "product_id",
                "transport",
                "line_id",
                "line_text",
                "clean_arr_status",
                "delay_sec",
                "delay_min",
            )
        )

    def clean_modeling_data(self, df, max_delay_min: float = 60.0):
        """Standardize transport/product fields and create `label_delay_min`."""
        from pyspark.sql import functions as F

        return (
            df.withColumn("clean_transport", F.trim(F.upper(F.col("transport"))))
            .withColumn("clean_product_id", F.trim(F.upper(F.col("product_id"))))
            .withColumn(
                "clean_transport",
                F.when(F.col("clean_transport") == "BUS", "B").otherwise(F.col("clean_transport")),
            )
            .withColumn(
                "clean_product_id",
                F.when(F.col("clean_transport").isin("CAR", "EV", "KB", "B", "EXB", "BN", "BP", "RUB"), "BUS")
                .when(F.col("clean_transport") == "T", "TRAM")
                .when(F.col("clean_transport") == "M", "METRO")
                .when(F.col("clean_transport").isin("BAT", "FAE"), "SCHIFF")
                .when(F.col("clean_transport").isin("GB", "PB", "SL", "ASC", "FUN", "CC"), "ZAHNRADBAHN")
                .when(
                    F.col("clean_transport").isin(
                        "ZUG",
                        "TGV",
                        "EST",
                        "IC",
                        "ICE",
                        "RJX",
                        "EC",
                        "IR",
                        "IRE",
                        "ATZ",
                        "ARZ",
                        "NJ",
                        "EN",
                        "RB",
                        "TER",
                        "RE",
                        "R",
                        "PE",
                        "S",
                        "SN",
                        "EXT",
                    ),
                    "ZUG",
                )
                .when((F.col("clean_transport") == "") & (F.col("clean_product_id") == "ZUG"), "ZUG")
                .otherwise(F.col("clean_product_id")),
            )
            .filter(F.col("trip_id").isNotNull() & (F.trim(F.col("trip_id")) != ""))
            .filter(F.col("operator_id").isNotNull() & (F.trim(F.col("operator_id")) != ""))
            .filter(F.col("clean_transport").isNotNull() & (F.col("clean_transport") != ""))
            .filter(F.col("clean_product_id").isNotNull() & (F.col("clean_product_id") != ""))
            .withColumn("raw_delay_min", F.col("delay_min"))
            .withColumn(
                "label_delay_min",
                F.least(F.greatest(F.col("delay_min"), F.lit(0.0)), F.lit(float(max_delay_min))),
            )
        )

    def build_training_data(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        region_uuids: Iterable[str] | None = None,
        operator_ids: Iterable[str] | None = None,
        restrict_to_scope_stops: bool = True,
        max_delay_min: float = 60.0,
        force_rebuild_region_stops: bool = False,
    ):
        """Build cleaned Part IV delay rows before external feature joins."""
        raw_df = self.build_raw_delay_data(
            start_date=start_date,
            end_date=end_date,
            region_uuids=region_uuids,
            operator_ids=operator_ids,
            restrict_to_scope_stops=restrict_to_scope_stops,
            force_rebuild_region_stops=force_rebuild_region_stops,
        )
        return self.clean_modeling_data(raw_df, max_delay_min=max_delay_min)

    def build_feature_input_data(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        region_uuids: Iterable[str] | None = None,
        operator_ids: Iterable[str] | None = None,
        include_weather: bool = True,
        include_calendar: bool = True,
        force_rebuild_region_stops: bool = False,
        sample_fraction: float | None = None,
        sample_seed: int = 42,
    ):
        """Build the final Part IV Spark ML feature input table."""
        from pyspark.sql import functions as F

        clean_df = self.build_training_data(
            start_date=start_date,
            end_date=end_date,
            region_uuids=region_uuids,
            operator_ids=operator_ids,
            force_rebuild_region_stops=force_rebuild_region_stops,
        )

        if sample_fraction is not None and 0 < sample_fraction < 1.0:
            # Sample before joins: calendar and weather are small broadcast lookups,
            # so joining on 10% of rows is 10× cheaper and semantically equivalent.
            fractions = {
                row["operator_id"]: sample_fraction
                for row in clean_df.select("operator_id").distinct().collect()
            }
            clean_df = clean_df.sampleBy("operator_id", fractions=fractions, seed=sample_seed)

        numeric_cols: list[str] = []

        feature_df = clean_df
        if include_weather:
            feature_df, weather_bus_cols = WeatherDataHandler(spark=self.get_spark(), settings=self.settings).add_weather_features(
                feature_df
            )
            numeric_cols.extend(weather_bus_cols)

        if include_calendar:
            cal_handler = CalendarDataHandler(spark=self.get_spark(), settings=self.settings)

            # Assign each stop to its Swiss canton (needed for per-canton holiday lookup)
            stop_canton_df = cal_handler.build_stop_canton_mapping()
            feature_df = feature_df.join(F.broadcast(stop_canton_df), on="bpuic", how="left")

            # Build (canton_code, operating_day) → holiday-flag table, then join
            calendar_df, calendar_numeric_cols = cal_handler.build_calendar_features(
                start_date=start_date, end_date=end_date
            )
            feature_df = (
                feature_df.withColumn("operating_day", F.to_date("operating_day"))
                .join(
                    F.broadcast(calendar_df),
                    on=["canton_code", "operating_day"],
                    how="left",
                )
                .fillna(0.0, subset=calendar_numeric_cols)
            )
            numeric_cols.extend(calendar_numeric_cols)
            feature_df = feature_df.withColumn(
                "day_type",
                F.when(F.col("is_public_holiday") >= 1, F.lit(1))
                 .when(
                     (F.col("is_bridge_day") >= 1) |
                     (F.col("is_day_before_public_holiday") >= 1) |
                     (F.col("is_day_after_public_holiday") >= 1),
                     F.lit(2),
                 )
                 .otherwise(F.lit(0)),
            )
        else:
            feature_df = feature_df.withColumn("canton_code", F.lit(None).cast("string"))
            feature_df = feature_df.withColumn("day_type", F.lit(0))

        categorical_cols = ["bpuic", "line_text", "hour", "day_of_week", "month", "day_type"]
        selected_cols = ["operating_day", *categorical_cols, *numeric_cols, "label_delay_min"]

        feature_df = (
            feature_df.filter(F.col("label_delay_min").isNotNull())
            .filter(F.col("hour").isNotNull())
            .filter(F.col("day_of_week").isNotNull())
            .filter(F.col("month").isNotNull())
        )

        return feature_df.select(*selected_cols), categorical_cols, numeric_cols

    def write_training_data(self, path: str, mode: str = "overwrite", **kwargs) -> str:
        """Build and write cleaned delay-training data to Parquet."""
        df, _, _ = self.build_feature_input_data(**kwargs)
        df.write.mode(mode).parquet(path)
        return path
