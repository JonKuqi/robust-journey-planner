from __future__ import annotations

"""Weather feature preparation for the delay model."""

from src.config.settings import ProjectSettings, get_settings
from src.config.spark_session import get_spark_session


class WeatherDataHandler:
    """Load weather observations and join them to delay records."""

    # Retained after bus-only Pearson r analysis (training_analysis.ipynb, 2026-05-19):
    # - temp:     r=+0.041  seasonal temperature signal
    # - rh:       r=-0.045  strongest signal; humidity as seasonal proxy
    # - dewpt:    r=+0.020  moderate, partially independent from temp
    # - uv_index: r=+0.026  sunshine/seasonal signal independent from temp
    #
    # Removed (9 features) — near-zero or artifact on bus-only rows:
    # - feels_like/wc/heat_index: r≈+0.040, redundant with temp
    # - pressure:  r=-0.008, was r=+0.077 all-rows — pure artifact (non-bus=0, bus≈1013)
    # - wspd:      r=+0.013, marginal
    # - wdir:      r=+0.005, near-zero
    # - precip_hrly/is_rainy: r≈+0.003, rain has no measurable effect on Swiss bus delays
    # - precip_hrly_missing:  r=+0.006, near-zero
    WEATHER_NUMERIC_COLS = [
        "temp",
        "rh",
        "dewpt",
        "uv_index",
    ]
    WEATHER_BUS_COLS = [f"{col}_bus" for col in WEATHER_NUMERIC_COLS]

    SELECTED_WEATHER_COLS = [
        "location_id",
        "valid_time_gmt",
        "temp",
        "rh",
        "dewpt",
        "uv_index",
    ]

    def __init__(self, spark=None, settings: ProjectSettings | None = None):
        """Initialize the weather feature handler."""
        self.settings = settings or get_settings()
        self.spark = spark

    def get_spark(self):
        """Return a Spark session."""
        if self.spark is None:
            self.spark = get_spark_session(self.settings, app_name="weather-data")
        return self.spark

    def weather_history_path(self) -> str:
        """Return the configured weather history path."""
        if self.settings.weather_history_path:
            return self.settings.weather_history_path
        if not self.settings.hadoop_fs:
            raise RuntimeError("HADOOP_FS or COM490_WEATHER_HISTORY_PATH is required for weather history.")
        return f"{self.settings.hadoop_fs}/data/com-490/silver/weather/history"

    def weather_stations_path(self) -> str:
        """Return the configured weather stations path."""
        if self.settings.weather_stations_path:
            return self.settings.weather_stations_path
        if not self.settings.hadoop_fs:
            raise RuntimeError("HADOOP_FS or COM490_WEATHER_STATIONS_PATH is required for weather stations.")
        return f"{self.settings.hadoop_fs}/data/com-490/silver/weather/stations"

    def load_weather_history(self):
        """Load raw weather history from the course silver data."""
        spark = self.get_spark()
        return spark.read.format("iceberg").load(self.weather_history_path())

    def load_stations(self):
        """Load raw weather station metadata."""
        spark = self.get_spark()
        return spark.read.parquet(self.weather_stations_path())

    def clean_stations(self, stations_df=None):
        """Parse station metadata into station id and coordinates."""
        from pyspark.sql import functions as F

        stations_df = stations_df or self.load_stations()
        return (
            stations_df.filter(F.col("ws_name") != "Name,City,Canton,ID,Active,lat,lon")
            .withColumn("parts", F.split(F.col("ws_name"), ","))
            .select(
                F.col("parts")[0].alias("station_name"),
                F.col("parts")[1].alias("station_city"),
                F.col("parts")[2].alias("station_canton"),
                F.col("parts")[3].alias("station_id"),
                F.col("parts")[4].cast("boolean").alias("station_active"),
                F.col("parts")[5].cast("double").alias("station_lat"),
                F.col("parts")[6].cast("double").alias("station_lon"),
            )
            .filter(F.col("station_id").isNotNull())
            .filter(F.col("station_lat").isNotNull())
            .filter(F.col("station_lon").isNotNull())
        )

    def clean_weather_history(self, weather_history_df=None):
        """Clean selected hourly weather features."""
        from pyspark.sql import functions as F

        weather_history_df = weather_history_df or self.load_weather_history()
        return (
            weather_history_df.select(*self.SELECTED_WEATHER_COLS)
            .filter(F.col("location_id").isNotNull() & F.col("valid_time_gmt").isNotNull())
            .withColumn(
                "weather_hour",
                F.date_trunc("hour", F.from_utc_timestamp("valid_time_gmt", "Europe/Zurich")),
            )
            .select(
                F.col("location_id").alias("station_id"),
                "weather_hour",
                *self.WEATHER_NUMERIC_COLS,
            )
            .dropDuplicates(["station_id", "weather_hour"])
        )

    def build_stop_station_mapping(self, bpuic_df, stations_clean_df=None):
        """Map each modeled stop to its nearest weather station."""
        from pyspark.sql import Window
        from pyspark.sql import functions as F

        spark = self.get_spark()
        stations_clean_df = stations_clean_df or self.clean_stations()
        max_pub_date = spark.table(self.settings.timetable_stops_table).select(F.max("pub_date")).first()[0]

        stop_coords = (
            bpuic_df.select(F.col("bpuic").cast("string").alias("bpuic"))
            .distinct()
            .join(
                spark.table(self.settings.timetable_stops_table)
                .filter(F.col("pub_date") == max_pub_date)
                .withColumn("bpuic", F.split(F.col("stop_id"), ":").getItem(0).cast("string"))
                .groupBy("bpuic")
                .agg(
                    F.first("stop_name", ignorenulls=True).alias("stop_name"),
                    F.avg("stop_lat").alias("stop_lat"),
                    F.avg("stop_lon").alias("stop_lon"),
                ),
                on="bpuic",
                how="left",
            )
        )

        return (
            stop_coords.filter(F.col("stop_lat").isNotNull() & F.col("stop_lon").isNotNull())
            .crossJoin(F.broadcast(stations_clean_df))
            .withColumn(
                "distance_km",
                F.expr(
                    """
                    6371 * 2 * asin(LEAST(sqrt(
                        pow(sin(radians(station_lat - stop_lat) / 2), 2) +
                        cos(radians(stop_lat)) * cos(radians(station_lat)) *
                        pow(sin(radians(station_lon - stop_lon) / 2), 2)
                    ), 1))
                    """
                ),
            )
            .withColumn("rn", F.row_number().over(Window.partitionBy("bpuic").orderBy(F.col("distance_km"))))
            .filter(F.col("rn") == 1)
            .select(
                "bpuic",
                F.col("station_id").alias("nearest_station_id"),
                F.col("station_city").alias("nearest_station_city"),
                "distance_km",
            )
        )

    def add_weather_features(self, delay_df, weather_clean_df=None, stop_station_df=None):
        """Join nearest-station weather and create bus-specific weather columns."""
        from pyspark.sql import functions as F

        weather_clean_df = weather_clean_df or self.clean_weather_history()
        stop_station_df = stop_station_df or self.build_stop_station_mapping(delay_df.select("bpuic"))

        mean_row = weather_clean_df.agg(
            *[F.mean(F.col(col)).alias(col) for col in self.WEATHER_NUMERIC_COLS]
        ).first()
        fill_values = {
            col: float(mean_row[col]) if mean_row[col] is not None else 0.0
            for col in self.WEATHER_NUMERIC_COLS
        }

        joined = (
            delay_df.alias("p")
            .join(
                stop_station_df.select(
                    F.col("bpuic").alias("ss_bpuic"),
                    "nearest_station_id",
                    "nearest_station_city",
                    "distance_km",
                ).alias("s"),
                F.col("p.bpuic") == F.col("s.ss_bpuic"),
                "left",
            )
            .withColumn("arr_hour", F.date_trunc("hour", F.col("p.arr_time_ts")))
            .join(
                weather_clean_df.alias("w"),
                (F.col("nearest_station_id") == F.col("w.station_id"))
                & (F.col("arr_hour") == F.col("w.weather_hour")),
                "left",
            )
            .drop("ss_bpuic", "station_id", "weather_hour")
            .fillna(fill_values)
        )

        bus_exprs = {
            f"{col}_bus": F.when(F.col("clean_product_id") == "BUS", F.col(col).cast("double")).otherwise(F.lit(0.0))
            for col in self.WEATHER_NUMERIC_COLS
        }
        return joined.withColumns(bus_exprs), list(self.WEATHER_BUS_COLS)
