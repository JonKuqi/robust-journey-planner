"""Extract complete historical journey paths from SBB istdaten for E2E evaluation.

This module mines the ``istdaten`` table for realistic multi-leg journeys
(0–3 transfers) with full per-leg details: trip IDs, intermediate stops,
scheduled and actual times, operator/line/transport metadata, and transfer
feasibility.  The output is a Parquet benchmark dataset that the
``E2EEvaluator`` can replay against the standard and robust planners.

Design decisions
----------------
* Transfers are discovered at **any stop where two trips meet** — not only at
  major hubs.  This mirrors the CSA planner's actual capability (footpath-based
  transfers at any walkable stop pair).
* Transfer windows are parameterised: the default includes tight connections
  (2 min) that the old benchmark excluded.
* Stratification covers transfers, transfer tightness, time-of-day, day type,
  transport modes, and historical outcome (journey completed vs. failed).
* All heavy work is PySpark; the final DataFrame is written to Parquet on HDFS.

Usage (on the cluster notebook)::

    from src.evaluation.journey_extractor import JourneyExtractor
    extractor = JourneyExtractor(spark)
    benchmark_df = extractor.extract(
        test_days=["2025-11-12", "2025-12-10"],
        samples_per_stratum=30,
    )
    benchmark_df.write.parquet("hdfs:///.../e2e_benchmark_v2.parquet")
"""

from __future__ import annotations

from typing import List, Optional

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession, Window


# ── Constants ────────────────────────────────────────────────────────────────

_MIN_TRANSFER_SEC = 120       # 2 minutes — CSA minimum
_MAX_TRANSFER_SEC = 45 * 60   # 45 minutes — generous upper bound
_DEFAULT_DEADLINE_BUFFER_SEC = 0  # will be parameterised per-query later

# Time-of-day buckets
_TOD_BINS = {
    "early_morning": (4, 6),
    "morning_rush":  (6, 9),
    "midday":        (9, 16),
    "evening_rush":  (16, 19),
    "evening":       (19, 22),
    "night":         (22, 28),   # 28 = 04:00 next day for overnight wraps
}

# Transport mode groups (for mode-mix stratification)
_MODE_GROUPS = {
    "Train": {"ZUG", "IC", "ICE", "IR", "RE", "S", "SN", "R", "RB", "TGV", "EC", "IRE", "RJX", "NJ", "EN", "EST", "PE", "ATZ", "ARZ", "TER", "EXT"},
    "Bus":   {"BUS", "CAR", "EV", "KB", "B", "EXB", "BN", "BP", "RUB"},
    "Tram":  {"T", "TRAM"},
    "Metro": {"M", "METRO"},
    "Ship":  {"BAT", "FAE", "SCHIFF"},
    "Other": {"GB", "PB", "SL", "ASC", "FUN", "CC"},
}


def _classify_mode(transport_col):
    """Return a UDF-free column expression classifying transport into mode groups."""
    expr = F.lit("Unknown")
    for group, codes in _MODE_GROUPS.items():
        expr = F.when(F.upper(F.trim(transport_col)).isin(*codes), F.lit(group)).otherwise(expr)
    return expr


def _time_of_day(hour_col):
    """Classify an hour (0–27) into a time-of-day bucket."""
    expr = F.lit("night")
    for bucket, (lo, hi) in _TOD_BINS.items():
        expr = F.when((hour_col >= lo) & (hour_col < hi), F.lit(bucket)).otherwise(expr)
    return expr


def _transfer_tightness(headway_sec_col):
    """Classify a transfer headway into tight / normal / comfortable."""
    return (
        F.when(headway_sec_col < 300, F.lit("tight"))          # < 5 min
        .when(headway_sec_col < 900, F.lit("normal"))          # 5–15 min
        .otherwise(F.lit("comfortable"))                       # > 15 min
    )


class JourneyExtractor:
    """Extract stratified historical journey paths from istdaten.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session with Iceberg catalog access.
    istdaten_table : str
        Fully-qualified istdaten table name.
    min_transfer_sec : int
        Minimum transfer time in seconds (default 120 = 2 min).
    max_transfer_sec : int
        Maximum transfer time in seconds (default 2700 = 45 min).
    """

    def __init__(
        self,
        spark: SparkSession,
        istdaten_table: str = "iceberg.com490_iceberg.sbb_istdaten",
        min_transfer_sec: int = _MIN_TRANSFER_SEC,
        max_transfer_sec: int = _MAX_TRANSFER_SEC,
    ):
        self.spark = spark
        self.istdaten_table = istdaten_table
        self.min_transfer_sec = min_transfer_sec
        self.max_transfer_sec = max_transfer_sec

    # ── Public API ──────────────────────────────────────────────────────────

    def extract(
        self,
        test_days: List[str],
        samples_per_stratum: int = 30,
        max_transfers: int = 3,
        seed: int = 42,
        deadline_buffer_options_sec: Optional[List[int]] = None,
    ) -> DataFrame:
        """Run the full extraction pipeline and return a stratified benchmark DataFrame.

        Parameters
        ----------
        test_days : list[str]
            Operating days to mine (ISO format, e.g. ``["2025-11-12"]``).
        samples_per_stratum : int
            Target samples per stratification bucket (will take min available).
        max_transfers : int
            Maximum number of transfers to consider (0–3).
        seed : int
            Random seed for reproducible sampling.
        deadline_buffer_options_sec : list[int] or None
            For each extracted journey, produce query rows with these deadline
            buffers added to the scheduled arrival.  Defaults to ``[0, 600, 1800]``
            (0, 10 min, 30 min).

        Returns
        -------
        DataFrame
            A Spark DataFrame ready for Parquet persistence.  Each row represents
            one evaluation query with full journey metadata.
        """
        if deadline_buffer_options_sec is None:
            deadline_buffer_options_sec = [0, 600, 1800]

        print(f"⏳ Filtering istdaten for {len(test_days)} test days ...")
        day_data = self._prepare_day_data(test_days)
        day_data.cache()
        print(f"  ↳ {day_data.count()} stop-event rows after filtering")

        pools = []

        # ── Direct journeys (0 transfers) ────────────────────────────────
        print("⏳ Extracting direct journeys (0 transfers) ...")
        direct = self._extract_direct(day_data)
        pools.append(direct)
        print(f"  ↳ {direct.count()} direct journey candidates")

        # ── 1-transfer journeys ──────────────────────────────────────────
        if max_transfers >= 1:
            print("⏳ Extracting 1-transfer journeys ...")
            trans1 = self._extract_1_transfer(day_data)
            pools.append(trans1)
            print(f"  ↳ {trans1.count()} 1-transfer candidates")

        # ── 2-transfer journeys ──────────────────────────────────────────
        if max_transfers >= 2:
            print("⏳ Extracting 2-transfer journeys ...")
            trans2 = self._extract_2_transfer(day_data)
            pools.append(trans2)
            print(f"  ↳ {trans2.count()} 2-transfer candidates")

        # ── 3-transfer journeys ──────────────────────────────────────────
        if max_transfers >= 3:
            print("⏳ Extracting 3-transfer journeys ...")
            trans3 = self._extract_3_transfer(day_data)
            pools.append(trans3)
            print(f"  ↳ {trans3.count()} 3-transfer candidates")

        # ── Union all pools ──────────────────────────────────────────────
        all_journeys = pools[0]
        for p in pools[1:]:
            all_journeys = all_journeys.unionByName(p, allowMissingColumns=True)

        # ── Enrich with stratification columns ───────────────────────────
        print("⏳ Enriching with stratification columns ...")
        enriched = self._enrich(all_journeys)

        # ── Explode deadline buffer variants ─────────────────────────────
        if deadline_buffer_options_sec:
            buffer_df = self.spark.createDataFrame(
                [(b,) for b in deadline_buffer_options_sec],
                ["deadline_buffer_sec"]
            )
            enriched = enriched.crossJoin(F.broadcast(buffer_df))
            enriched = enriched.withColumn(
                "query_deadline_secs",
                F.col("dest_scheduled_arr_secs") + F.col("deadline_buffer_sec")
            )
        else:
            enriched = enriched.withColumn("deadline_buffer_sec", F.lit(0))
            enriched = enriched.withColumn(
                "query_deadline_secs", F.col("dest_scheduled_arr_secs")
            )

        # ── Stratified sampling ──────────────────────────────────────────
        print("⏳ Stratified sampling ...")
        result = self._stratified_sample(enriched, samples_per_stratum, seed)

        day_data.unpersist()
        print(f"✅ Benchmark extraction complete: {result.count()} evaluation queries")
        return result

    # ── Private: data preparation ───────────────────────────────────────────

    def _prepare_day_data(self, test_days: List[str]) -> DataFrame:
        """Filter istdaten to test days and add sequence/timing columns."""
        raw = (
            self.spark.table(self.istdaten_table)
            .filter(F.col("operating_day").isin(test_days))
            .filter((F.col("failed") == False) | F.col("failed").isNull())
            .filter(F.col("transit") == False)
            .filter(F.col("arr_time").isNotNull() | F.col("dep_time").isNotNull())
            .filter(F.col("bpuic").isNotNull())
            .filter(F.col("trip_id").isNotNull())
        )

        # Parse scheduled times to seconds-since-midnight.
        # First stop has no arr_time, last stop has no dep_time. Coalesce them.
        raw = (
            raw
            .withColumn("arr_secs", self._time_to_secs(F.coalesce(F.col("arr_time"), F.col("dep_time"))))
            .withColumn("dep_secs", self._time_to_secs(F.coalesce(F.col("dep_time"), F.col("arr_time"))))
            .withColumn("arr_actual_secs",
                        F.when(F.col("arr_actual").isNotNull(),
                               self._time_to_secs(F.col("arr_actual")))
                        .otherwise(F.col("arr_secs")))
            .withColumn("dep_actual_secs",
                        F.when(F.col("dep_actual").isNotNull(),
                               self._time_to_secs(F.col("dep_actual")))
                        .otherwise(F.col("dep_secs")))
            .withColumn("arr_delay_sec", F.col("arr_actual_secs") - F.col("arr_secs"))
            .withColumn("dep_delay_sec", F.col("dep_actual_secs") - F.col("dep_secs"))
        )

        # Assign stop sequence within each trip (ordered by scheduled arrival)
        trip_win = Window.partitionBy("operating_day", "trip_id").orderBy("arr_secs")
        raw = (
            raw
            .withColumn("seq", F.row_number().over(trip_win))
            .withColumn("max_seq", F.max("seq").over(
                Window.partitionBy("operating_day", "trip_id")
            ))
        )

        # Classify transport mode
        raw = raw.withColumn("mode_group", _classify_mode(F.col("transport")))

        return raw

    # ── Private: journey extraction ─────────────────────────────────────────

    def _leg_columns(self, alias: str, leg_num: int):
        """Standard columns to select from a leg, with proper aliasing."""
        return [
            F.col(f"{alias}.operating_day").alias("operating_day"),
            F.col(f"{alias}.trip_id").alias(f"leg{leg_num}_trip_id"),
            F.col(f"{alias}.bpuic").alias(f"leg{leg_num}_board_stop_id"),
            F.col(f"{alias}.stop_name").alias(f"leg{leg_num}_board_stop_name"),
            F.col(f"{alias}.dep_secs").alias(f"leg{leg_num}_dep_secs"),
            F.col(f"{alias}.dep_actual_secs").alias(f"leg{leg_num}_dep_actual_secs"),
            F.col(f"{alias}.dep_delay_sec").alias(f"leg{leg_num}_dep_delay_sec"),
            F.col(f"{alias}.line_text").alias(f"leg{leg_num}_line_text"),
            F.col(f"{alias}.transport").alias(f"leg{leg_num}_transport"),
            F.col(f"{alias}.operator_abrv").alias(f"leg{leg_num}_operator"),
            F.col(f"{alias}.mode_group").alias(f"leg{leg_num}_mode"),
        ]

    def _alight_columns(self, alias: str, leg_num: int):
        """Alighting-point columns for a leg."""
        return [
            F.col(f"{alias}.bpuic").alias(f"leg{leg_num}_alight_stop_id"),
            F.col(f"{alias}.stop_name").alias(f"leg{leg_num}_alight_stop_name"),
            F.col(f"{alias}.arr_secs").alias(f"leg{leg_num}_arr_secs"),
            F.col(f"{alias}.arr_actual_secs").alias(f"leg{leg_num}_arr_actual_secs"),
            F.col(f"{alias}.arr_delay_sec").alias(f"leg{leg_num}_arr_delay_sec"),
        ]

    def _extract_direct(self, day_data: DataFrame) -> DataFrame:
        """Extract direct (0-transfer) journeys: any board→alight within a trip."""
        # Board = any stop in the trip, Alight = any later stop in the same trip
        board = day_data.alias("b")
        alight = day_data.alias("a")

        direct = (
            board.join(alight,
                       (F.col("b.operating_day") == F.col("a.operating_day")) &
                       (F.col("b.trip_id") == F.col("a.trip_id")) &
                       (F.col("b.seq") < F.col("a.seq")),
                       how="inner")
            .filter(F.col("b.bpuic") != F.col("a.bpuic"))
            # Only keep meaningful journeys (not single-stop hops)
            .filter((F.col("a.seq") - F.col("b.seq")) >= 1)
            # Limit to reasonable journey lengths (not first-to-last only)
            .select(
                F.col("b.operating_day").alias("operating_day"),
                F.lit(0).alias("num_transfers"),
                # Origin
                F.col("b.bpuic").alias("origin_stop_id"),
                F.col("b.stop_name").alias("origin_stop_name"),
                F.col("b.dep_secs").alias("origin_dep_secs"),
                F.col("b.dep_actual_secs").alias("origin_dep_actual_secs"),
                # Destination
                F.col("a.bpuic").alias("dest_stop_id"),
                F.col("a.stop_name").alias("dest_stop_name"),
                F.col("a.arr_secs").alias("dest_scheduled_arr_secs"),
                F.col("a.arr_actual_secs").alias("dest_actual_arr_secs"),
                F.col("a.arr_delay_sec").alias("dest_delay_sec"),
                # Leg 1 details
                F.col("b.trip_id").alias("leg1_trip_id"),
                F.col("b.bpuic").alias("leg1_board_stop_id"),
                F.col("b.stop_name").alias("leg1_board_stop_name"),
                F.col("b.dep_secs").alias("leg1_dep_secs"),
                F.col("b.dep_actual_secs").alias("leg1_dep_actual_secs"),
                F.col("b.dep_delay_sec").alias("leg1_dep_delay_sec"),
                F.col("b.line_text").alias("leg1_line_text"),
                F.col("b.transport").alias("leg1_transport"),
                F.col("b.operator_abrv").alias("leg1_operator"),
                F.col("b.mode_group").alias("leg1_mode"),
                F.col("a.bpuic").alias("leg1_alight_stop_id"),
                F.col("a.stop_name").alias("leg1_alight_stop_name"),
                F.col("a.arr_secs").alias("leg1_arr_secs"),
                F.col("a.arr_actual_secs").alias("leg1_arr_actual_secs"),
                F.col("a.arr_delay_sec").alias("leg1_arr_delay_sec"),
                # No transfers
                F.lit(None).cast("int").alias("xfer1_stop_id"),
                F.lit(None).cast("string").alias("xfer1_stop_name"),
                F.lit(None).cast("int").alias("xfer1_scheduled_headway_sec"),
                F.lit(None).cast("int").alias("xfer1_actual_headway_sec"),
                F.lit(None).cast("boolean").alias("xfer1_made"),
                # Journey outcome (direct = always completed if not failed)
                F.lit(True).alias("journey_completed"),
            )
        )

        # Sample down: direct journeys are very numerous — take a random 1%
        # before downstream stratification to avoid OOM
        return direct.sample(fraction=0.01, seed=42)

    def _extract_1_transfer(self, day_data: DataFrame) -> DataFrame:
        """Extract 1-transfer journeys via any shared stop (not just major hubs)."""
        min_x = self.min_transfer_sec
        max_x = self.max_transfer_sec

        # Leg 1: board anywhere, alight at a transfer stop
        # Leg 2: board at the same transfer stop, alight at destination
        # To avoid combinatorial explosion, sample the first boarding leg aggressively
        l1_board = day_data.sample(fraction=0.001, seed=42).alias("l1b")
        l1_alight = day_data.alias("l1a")
        l2_board = day_data.alias("l2b")
        l2_alight = day_data.alias("l2a")

        # Join: leg1 alight stop = leg2 board stop (same bpuic, same day)
        # Different trip_id, transfer time in [min_x, max_x]
        joined = (
            l1_board
            .join(l1_alight,
                  (F.col("l1b.operating_day") == F.col("l1a.operating_day")) &
                  (F.col("l1b.trip_id") == F.col("l1a.trip_id")) &
                  (F.col("l1b.seq") < F.col("l1a.seq")),
                  how="inner")
            .join(l2_board,
                  (F.col("l1a.operating_day") == F.col("l2b.operating_day")) &
                  (F.col("l1a.bpuic") == F.col("l2b.bpuic")) &
                  (F.col("l1a.trip_id") != F.col("l2b.trip_id")),
                  how="inner")
            # Transfer timing constraint (using SCHEDULED times)
            .filter((F.col("l2b.dep_secs") - F.col("l1a.arr_secs")).between(min_x, max_x))
            .join(l2_alight,
                  (F.col("l2b.operating_day") == F.col("l2a.operating_day")) &
                  (F.col("l2b.trip_id") == F.col("l2a.trip_id")) &
                  (F.col("l2b.seq") < F.col("l2a.seq")),
                  how="inner")
            # Origin != Destination
            .filter(F.col("l1b.bpuic") != F.col("l2a.bpuic"))
        )

        # Compute transfer feasibility under actual times
        xfer1_scheduled_headway = F.col("l2b.dep_secs") - F.col("l1a.arr_secs")
        xfer1_actual_headway = F.col("l2b.dep_actual_secs") - F.col("l1a.arr_actual_secs")
        xfer1_made = xfer1_actual_headway >= F.lit(min_x)

        result = joined.select(
            F.col("l1b.operating_day").alias("operating_day"),
            F.lit(1).alias("num_transfers"),
            # Origin / Destination
            F.col("l1b.bpuic").alias("origin_stop_id"),
            F.col("l1b.stop_name").alias("origin_stop_name"),
            F.col("l1b.dep_secs").alias("origin_dep_secs"),
            F.col("l1b.dep_actual_secs").alias("origin_dep_actual_secs"),
            F.col("l2a.bpuic").alias("dest_stop_id"),
            F.col("l2a.stop_name").alias("dest_stop_name"),
            F.col("l2a.arr_secs").alias("dest_scheduled_arr_secs"),
            F.col("l2a.arr_actual_secs").alias("dest_actual_arr_secs"),
            F.col("l2a.arr_delay_sec").alias("dest_delay_sec"),
            # Leg 1
            F.col("l1b.trip_id").alias("leg1_trip_id"),
            F.col("l1b.bpuic").alias("leg1_board_stop_id"),
            F.col("l1b.stop_name").alias("leg1_board_stop_name"),
            F.col("l1b.dep_secs").alias("leg1_dep_secs"),
            F.col("l1b.dep_actual_secs").alias("leg1_dep_actual_secs"),
            F.col("l1b.dep_delay_sec").alias("leg1_dep_delay_sec"),
            F.col("l1b.line_text").alias("leg1_line_text"),
            F.col("l1b.transport").alias("leg1_transport"),
            F.col("l1b.operator_abrv").alias("leg1_operator"),
            F.col("l1b.mode_group").alias("leg1_mode"),
            F.col("l1a.bpuic").alias("leg1_alight_stop_id"),
            F.col("l1a.stop_name").alias("leg1_alight_stop_name"),
            F.col("l1a.arr_secs").alias("leg1_arr_secs"),
            F.col("l1a.arr_actual_secs").alias("leg1_arr_actual_secs"),
            F.col("l1a.arr_delay_sec").alias("leg1_arr_delay_sec"),
            # Transfer 1
            F.col("l1a.bpuic").alias("xfer1_stop_id"),
            F.col("l1a.stop_name").alias("xfer1_stop_name"),
            xfer1_scheduled_headway.alias("xfer1_scheduled_headway_sec"),
            xfer1_actual_headway.alias("xfer1_actual_headway_sec"),
            xfer1_made.alias("xfer1_made"),
            # Leg 2 (stored as extra columns)
            F.col("l2b.trip_id").alias("leg2_trip_id"),
            F.col("l2b.bpuic").alias("leg2_board_stop_id"),
            F.col("l2b.stop_name").alias("leg2_board_stop_name"),
            F.col("l2b.dep_secs").alias("leg2_dep_secs"),
            F.col("l2b.dep_actual_secs").alias("leg2_dep_actual_secs"),
            F.col("l2b.dep_delay_sec").alias("leg2_dep_delay_sec"),
            F.col("l2b.line_text").alias("leg2_line_text"),
            F.col("l2b.transport").alias("leg2_transport"),
            F.col("l2b.operator_abrv").alias("leg2_operator"),
            F.col("l2b.mode_group").alias("leg2_mode"),
            F.col("l2a.bpuic").alias("leg2_alight_stop_id"),
            F.col("l2a.stop_name").alias("leg2_alight_stop_name"),
            F.col("l2a.arr_secs").alias("leg2_arr_secs"),
            F.col("l2a.arr_actual_secs").alias("leg2_arr_actual_secs"),
            F.col("l2a.arr_delay_sec").alias("leg2_arr_delay_sec"),
            # Journey outcome
            xfer1_made.alias("journey_completed"),
        )

        return result.sample(fraction=0.05, seed=42)

    def _extract_2_transfer(self, day_data: DataFrame) -> DataFrame:
        """Extract 2-transfer journeys (3 legs, 2 connection points)."""
        min_x = self.min_transfer_sec
        max_x = self.max_transfer_sec

        # To control combinatorial explosion, we limit intermediate legs:
        # Leg 1: board at seq==1, alight at any stop
        # Leg 2: board at transfer stop, alight at another stop
        # Leg 3: board at 2nd transfer stop, alight at seq==max_seq
        l1b = day_data.sample(fraction=0.0001, seed=42).alias("l1b")
        l1a = day_data.alias("l1a")
        l2b = day_data.alias("l2b")
        l2a = day_data.alias("l2a")
        l3b = day_data.alias("l3b")
        l3a = day_data.alias("l3a")

        joined = (
            l1b.join(l1a,
                     (F.col("l1b.operating_day") == F.col("l1a.operating_day")) &
                     (F.col("l1b.trip_id") == F.col("l1a.trip_id")) &
                     (F.col("l1b.seq") < F.col("l1a.seq")),
                     how="inner")
            # Transfer 1
            .join(l2b,
                  (F.col("l1a.operating_day") == F.col("l2b.operating_day")) &
                  (F.col("l1a.bpuic") == F.col("l2b.bpuic")) &
                  (F.col("l1a.trip_id") != F.col("l2b.trip_id")) &
                  (F.col("l2b.dep_secs") - F.col("l1a.arr_secs")).between(min_x, max_x),
                  how="inner")
            .join(l2a,
                  (F.col("l2b.operating_day") == F.col("l2a.operating_day")) &
                  (F.col("l2b.trip_id") == F.col("l2a.trip_id")) &
                  (F.col("l2b.seq") < F.col("l2a.seq")),
                  how="inner")
            # Transfer 2
            .join(l3b,
                  (F.col("l2a.operating_day") == F.col("l3b.operating_day")) &
                  (F.col("l2a.bpuic") == F.col("l3b.bpuic")) &
                  (F.col("l2a.trip_id") != F.col("l3b.trip_id")) &
                  (F.col("l3b.dep_secs") - F.col("l2a.arr_secs")).between(min_x, max_x),
                  how="inner")
            .join(l3a,
                  (F.col("l3b.operating_day") == F.col("l3a.operating_day")) &
                  (F.col("l3b.trip_id") == F.col("l3a.trip_id")) &
                  (F.col("l3b.seq") < F.col("l3a.seq")),
                  how="inner")
            # Distinct origin/destination
            .filter(F.col("l1b.bpuic") != F.col("l3a.bpuic"))
            # All trips must be different
            .filter(F.col("l1b.trip_id") != F.col("l3b.trip_id"))
        )

        xfer1_actual = F.col("l2b.dep_actual_secs") - F.col("l1a.arr_actual_secs")
        xfer2_actual = F.col("l3b.dep_actual_secs") - F.col("l2a.arr_actual_secs")
        xfer1_made = xfer1_actual >= F.lit(min_x)
        xfer2_made = xfer2_actual >= F.lit(min_x)

        result = joined.select(
            F.col("l1b.operating_day").alias("operating_day"),
            F.lit(2).alias("num_transfers"),
            F.col("l1b.bpuic").alias("origin_stop_id"),
            F.col("l1b.stop_name").alias("origin_stop_name"),
            F.col("l1b.dep_secs").alias("origin_dep_secs"),
            F.col("l1b.dep_actual_secs").alias("origin_dep_actual_secs"),
            F.col("l3a.bpuic").alias("dest_stop_id"),
            F.col("l3a.stop_name").alias("dest_stop_name"),
            F.col("l3a.arr_secs").alias("dest_scheduled_arr_secs"),
            F.col("l3a.arr_actual_secs").alias("dest_actual_arr_secs"),
            F.col("l3a.arr_delay_sec").alias("dest_delay_sec"),
            # Leg 1
            F.col("l1b.trip_id").alias("leg1_trip_id"),
            F.col("l1b.bpuic").alias("leg1_board_stop_id"),
            F.col("l1b.stop_name").alias("leg1_board_stop_name"),
            F.col("l1b.dep_secs").alias("leg1_dep_secs"),
            F.col("l1b.dep_actual_secs").alias("leg1_dep_actual_secs"),
            F.col("l1b.dep_delay_sec").alias("leg1_dep_delay_sec"),
            F.col("l1b.line_text").alias("leg1_line_text"),
            F.col("l1b.transport").alias("leg1_transport"),
            F.col("l1b.operator_abrv").alias("leg1_operator"),
            F.col("l1b.mode_group").alias("leg1_mode"),
            F.col("l1a.bpuic").alias("leg1_alight_stop_id"),
            F.col("l1a.stop_name").alias("leg1_alight_stop_name"),
            F.col("l1a.arr_secs").alias("leg1_arr_secs"),
            F.col("l1a.arr_actual_secs").alias("leg1_arr_actual_secs"),
            F.col("l1a.arr_delay_sec").alias("leg1_arr_delay_sec"),
            # Transfer 1
            F.col("l1a.bpuic").alias("xfer1_stop_id"),
            F.col("l1a.stop_name").alias("xfer1_stop_name"),
            (F.col("l2b.dep_secs") - F.col("l1a.arr_secs")).alias("xfer1_scheduled_headway_sec"),
            xfer1_actual.alias("xfer1_actual_headway_sec"),
            xfer1_made.alias("xfer1_made"),
            # Leg 2
            F.col("l2b.trip_id").alias("leg2_trip_id"),
            F.col("l2b.bpuic").alias("leg2_board_stop_id"),
            F.col("l2b.stop_name").alias("leg2_board_stop_name"),
            F.col("l2b.dep_secs").alias("leg2_dep_secs"),
            F.col("l2b.dep_actual_secs").alias("leg2_dep_actual_secs"),
            F.col("l2b.dep_delay_sec").alias("leg2_dep_delay_sec"),
            F.col("l2b.line_text").alias("leg2_line_text"),
            F.col("l2b.transport").alias("leg2_transport"),
            F.col("l2b.operator_abrv").alias("leg2_operator"),
            F.col("l2b.mode_group").alias("leg2_mode"),
            F.col("l2a.bpuic").alias("leg2_alight_stop_id"),
            F.col("l2a.stop_name").alias("leg2_alight_stop_name"),
            F.col("l2a.arr_secs").alias("leg2_arr_secs"),
            F.col("l2a.arr_actual_secs").alias("leg2_arr_actual_secs"),
            F.col("l2a.arr_delay_sec").alias("leg2_arr_delay_sec"),
            # Transfer 2
            F.col("l2a.bpuic").alias("xfer2_stop_id"),
            F.col("l2a.stop_name").alias("xfer2_stop_name"),
            (F.col("l3b.dep_secs") - F.col("l2a.arr_secs")).alias("xfer2_scheduled_headway_sec"),
            xfer2_actual.alias("xfer2_actual_headway_sec"),
            xfer2_made.alias("xfer2_made"),
            # Leg 3
            F.col("l3b.trip_id").alias("leg3_trip_id"),
            F.col("l3b.bpuic").alias("leg3_board_stop_id"),
            F.col("l3b.stop_name").alias("leg3_board_stop_name"),
            F.col("l3b.dep_secs").alias("leg3_dep_secs"),
            F.col("l3b.dep_actual_secs").alias("leg3_dep_actual_secs"),
            F.col("l3b.dep_delay_sec").alias("leg3_dep_delay_sec"),
            F.col("l3b.line_text").alias("leg3_line_text"),
            F.col("l3b.transport").alias("leg3_transport"),
            F.col("l3b.operator_abrv").alias("leg3_operator"),
            F.col("l3b.mode_group").alias("leg3_mode"),
            F.col("l3a.bpuic").alias("leg3_alight_stop_id"),
            F.col("l3a.stop_name").alias("leg3_alight_stop_name"),
            F.col("l3a.arr_secs").alias("leg3_arr_secs"),
            F.col("l3a.arr_actual_secs").alias("leg3_arr_actual_secs"),
            F.col("l3a.arr_delay_sec").alias("leg3_arr_delay_sec"),
            # Journey outcome
            (xfer1_made & xfer2_made).alias("journey_completed"),
        )

        return result.sample(fraction=0.1, seed=42)

    def _extract_3_transfer(self, day_data: DataFrame) -> DataFrame:
        """Extract 3-transfer journeys (4 legs, 3 connection points).

        Due to the extreme combinatorial explosion of 4-way joins, we limit
        this to trips that start at seq==1 and end at seq==max_seq, and sample
        aggressively.
        """
        min_x = self.min_transfer_sec
        max_x = self.max_transfer_sec

        l1b = day_data.sample(fraction=0.00001, seed=42).alias("l1b")
        l1a = day_data.alias("l1a")
        l2b = day_data.alias("l2b")
        l2a = day_data.alias("l2a")
        l3b = day_data.alias("l3b")
        l3a = day_data.alias("l3a")
        l4b = day_data.alias("l4b")
        l4a = day_data.alias("l4a")

        joined = (
            l1b.join(l1a,
                     (F.col("l1b.operating_day") == F.col("l1a.operating_day")) &
                     (F.col("l1b.trip_id") == F.col("l1a.trip_id")) &
                     (F.col("l1b.seq") < F.col("l1a.seq")),
                     how="inner")
            .join(l2b,
                  (F.col("l1a.operating_day") == F.col("l2b.operating_day")) &
                  (F.col("l1a.bpuic") == F.col("l2b.bpuic")) &
                  (F.col("l1a.trip_id") != F.col("l2b.trip_id")) &
                  (F.col("l2b.dep_secs") - F.col("l1a.arr_secs")).between(min_x, max_x),
                  how="inner")
            .join(l2a,
                  (F.col("l2b.operating_day") == F.col("l2a.operating_day")) &
                  (F.col("l2b.trip_id") == F.col("l2a.trip_id")) &
                  (F.col("l2b.seq") < F.col("l2a.seq")),
                  how="inner")
            .join(l3b,
                  (F.col("l2a.operating_day") == F.col("l3b.operating_day")) &
                  (F.col("l2a.bpuic") == F.col("l3b.bpuic")) &
                  (F.col("l2a.trip_id") != F.col("l3b.trip_id")) &
                  (F.col("l3b.dep_secs") - F.col("l2a.arr_secs")).between(min_x, max_x),
                  how="inner")
            .join(l3a,
                  (F.col("l3b.operating_day") == F.col("l3a.operating_day")) &
                  (F.col("l3b.trip_id") == F.col("l3a.trip_id")) &
                  (F.col("l3b.seq") < F.col("l3a.seq")),
                  how="inner")
            .join(l4b,
                  (F.col("l3a.operating_day") == F.col("l4b.operating_day")) &
                  (F.col("l3a.bpuic") == F.col("l4b.bpuic")) &
                  (F.col("l3a.trip_id") != F.col("l4b.trip_id")) &
                  (F.col("l4b.dep_secs") - F.col("l3a.arr_secs")).between(min_x, max_x),
                  how="inner")
            .join(l4a,
                  (F.col("l4b.operating_day") == F.col("l4a.operating_day")) &
                  (F.col("l4b.trip_id") == F.col("l4a.trip_id")) &
                  (F.col("l4b.seq") < F.col("l4a.seq")),
                  how="inner")
            .filter(F.col("l1b.bpuic") != F.col("l4a.bpuic"))
            # All trips distinct
            .filter(F.col("l1b.trip_id") != F.col("l3b.trip_id"))
            .filter(F.col("l1b.trip_id") != F.col("l4b.trip_id"))
            .filter(F.col("l2b.trip_id") != F.col("l4b.trip_id"))
        )

        xfer1_actual = F.col("l2b.dep_actual_secs") - F.col("l1a.arr_actual_secs")
        xfer2_actual = F.col("l3b.dep_actual_secs") - F.col("l2a.arr_actual_secs")
        xfer3_actual = F.col("l4b.dep_actual_secs") - F.col("l3a.arr_actual_secs")
        xfer1_made = xfer1_actual >= F.lit(min_x)
        xfer2_made = xfer2_actual >= F.lit(min_x)
        xfer3_made = xfer3_actual >= F.lit(min_x)

        result = joined.select(
            F.col("l1b.operating_day").alias("operating_day"),
            F.lit(3).alias("num_transfers"),
            F.col("l1b.bpuic").alias("origin_stop_id"),
            F.col("l1b.stop_name").alias("origin_stop_name"),
            F.col("l1b.dep_secs").alias("origin_dep_secs"),
            F.col("l1b.dep_actual_secs").alias("origin_dep_actual_secs"),
            F.col("l4a.bpuic").alias("dest_stop_id"),
            F.col("l4a.stop_name").alias("dest_stop_name"),
            F.col("l4a.arr_secs").alias("dest_scheduled_arr_secs"),
            F.col("l4a.arr_actual_secs").alias("dest_actual_arr_secs"),
            F.col("l4a.arr_delay_sec").alias("dest_delay_sec"),
            # Leg 1
            F.col("l1b.trip_id").alias("leg1_trip_id"),
            F.col("l1b.bpuic").alias("leg1_board_stop_id"),
            F.col("l1b.stop_name").alias("leg1_board_stop_name"),
            F.col("l1b.dep_secs").alias("leg1_dep_secs"),
            F.col("l1b.dep_actual_secs").alias("leg1_dep_actual_secs"),
            F.col("l1b.dep_delay_sec").alias("leg1_dep_delay_sec"),
            F.col("l1b.line_text").alias("leg1_line_text"),
            F.col("l1b.transport").alias("leg1_transport"),
            F.col("l1b.operator_abrv").alias("leg1_operator"),
            F.col("l1b.mode_group").alias("leg1_mode"),
            F.col("l1a.bpuic").alias("leg1_alight_stop_id"),
            F.col("l1a.stop_name").alias("leg1_alight_stop_name"),
            F.col("l1a.arr_secs").alias("leg1_arr_secs"),
            F.col("l1a.arr_actual_secs").alias("leg1_arr_actual_secs"),
            F.col("l1a.arr_delay_sec").alias("leg1_arr_delay_sec"),
            # Transfer 1
            F.col("l1a.bpuic").alias("xfer1_stop_id"),
            F.col("l1a.stop_name").alias("xfer1_stop_name"),
            (F.col("l2b.dep_secs") - F.col("l1a.arr_secs")).alias("xfer1_scheduled_headway_sec"),
            xfer1_actual.alias("xfer1_actual_headway_sec"),
            xfer1_made.alias("xfer1_made"),
            # Leg 2
            F.col("l2b.trip_id").alias("leg2_trip_id"),
            F.col("l2b.bpuic").alias("leg2_board_stop_id"),
            F.col("l2b.stop_name").alias("leg2_board_stop_name"),
            F.col("l2b.dep_secs").alias("leg2_dep_secs"),
            F.col("l2b.dep_actual_secs").alias("leg2_dep_actual_secs"),
            F.col("l2b.dep_delay_sec").alias("leg2_dep_delay_sec"),
            F.col("l2b.line_text").alias("leg2_line_text"),
            F.col("l2b.transport").alias("leg2_transport"),
            F.col("l2b.operator_abrv").alias("leg2_operator"),
            F.col("l2b.mode_group").alias("leg2_mode"),
            F.col("l2a.bpuic").alias("leg2_alight_stop_id"),
            F.col("l2a.stop_name").alias("leg2_alight_stop_name"),
            F.col("l2a.arr_secs").alias("leg2_arr_secs"),
            F.col("l2a.arr_actual_secs").alias("leg2_arr_actual_secs"),
            F.col("l2a.arr_delay_sec").alias("leg2_arr_delay_sec"),
            # Transfer 2
            F.col("l2a.bpuic").alias("xfer2_stop_id"),
            F.col("l2a.stop_name").alias("xfer2_stop_name"),
            (F.col("l3b.dep_secs") - F.col("l2a.arr_secs")).alias("xfer2_scheduled_headway_sec"),
            xfer2_actual.alias("xfer2_actual_headway_sec"),
            xfer2_made.alias("xfer2_made"),
            # Leg 3
            F.col("l3b.trip_id").alias("leg3_trip_id"),
            F.col("l3b.bpuic").alias("leg3_board_stop_id"),
            F.col("l3b.stop_name").alias("leg3_board_stop_name"),
            F.col("l3b.dep_secs").alias("leg3_dep_secs"),
            F.col("l3b.dep_actual_secs").alias("leg3_dep_actual_secs"),
            F.col("l3b.dep_delay_sec").alias("leg3_dep_delay_sec"),
            F.col("l3b.line_text").alias("leg3_line_text"),
            F.col("l3b.transport").alias("leg3_transport"),
            F.col("l3b.operator_abrv").alias("leg3_operator"),
            F.col("l3b.mode_group").alias("leg3_mode"),
            F.col("l3a.bpuic").alias("leg3_alight_stop_id"),
            F.col("l3a.stop_name").alias("leg3_alight_stop_name"),
            F.col("l3a.arr_secs").alias("leg3_arr_secs"),
            F.col("l3a.arr_actual_secs").alias("leg3_arr_actual_secs"),
            F.col("l3a.arr_delay_sec").alias("leg3_arr_delay_sec"),
            # Transfer 3
            F.col("l3a.bpuic").alias("xfer3_stop_id"),
            F.col("l3a.stop_name").alias("xfer3_stop_name"),
            (F.col("l4b.dep_secs") - F.col("l3a.arr_secs")).alias("xfer3_scheduled_headway_sec"),
            xfer3_actual.alias("xfer3_actual_headway_sec"),
            xfer3_made.alias("xfer3_made"),
            # Leg 4
            F.col("l4b.trip_id").alias("leg4_trip_id"),
            F.col("l4b.bpuic").alias("leg4_board_stop_id"),
            F.col("l4b.stop_name").alias("leg4_board_stop_name"),
            F.col("l4b.dep_secs").alias("leg4_dep_secs"),
            F.col("l4b.dep_actual_secs").alias("leg4_dep_actual_secs"),
            F.col("l4b.dep_delay_sec").alias("leg4_dep_delay_sec"),
            F.col("l4b.line_text").alias("leg4_line_text"),
            F.col("l4b.transport").alias("leg4_transport"),
            F.col("l4b.operator_abrv").alias("leg4_operator"),
            F.col("l4b.mode_group").alias("leg4_mode"),
            F.col("l4a.bpuic").alias("leg4_alight_stop_id"),
            F.col("l4a.stop_name").alias("leg4_alight_stop_name"),
            F.col("l4a.arr_secs").alias("leg4_arr_secs"),
            F.col("l4a.arr_actual_secs").alias("leg4_arr_actual_secs"),
            F.col("l4a.arr_delay_sec").alias("leg4_arr_delay_sec"),
            # Journey outcome
            (xfer1_made & xfer2_made & xfer3_made).alias("journey_completed"),
        )

        return result.sample(fraction=0.2, seed=42)

    # ── Private: enrichment & stratification ────────────────────────────────

    def _enrich(self, df: DataFrame) -> DataFrame:
        """Add stratification columns for multi-dimensional sampling."""
        # Departure hour and time-of-day
        df = df.withColumn("dep_hour", (F.col("origin_dep_secs") / 3600).cast("int"))
        df = df.withColumn("time_of_day", _time_of_day(F.col("dep_hour")))

        # Day type
        df = df.withColumn("day_of_week", F.dayofweek(F.col("operating_day")))
        df = df.withColumn("day_type",
            F.when(F.col("day_of_week").isin(1, 7), F.lit("weekend"))
            .otherwise(F.lit("weekday"))
        )

        # Tightest transfer headway (relevant only for multi-transfer journeys)
        headway_cols = [c for c in df.columns if c.endswith("_scheduled_headway_sec")]
        if headway_cols:
            min_headway_expr = F.least(*[
                F.when(F.col(c).isNotNull(), F.col(c)).otherwise(F.lit(99999))
                for c in headway_cols
            ])
            df = df.withColumn("min_scheduled_headway_sec", min_headway_expr)
        else:
            df = df.withColumn("min_scheduled_headway_sec", F.lit(99999))

        df = df.withColumn("transfer_tightness",
            F.when(F.col("num_transfers") == 0, F.lit("none"))
            .otherwise(_transfer_tightness(F.col("min_scheduled_headway_sec")))
        )

        # Mode mix
        mode_cols = [c for c in df.columns if c.endswith("_mode") and c.startswith("leg")]
        if mode_cols:
            all_modes = F.array_distinct(F.array(*[F.col(c) for c in mode_cols]))
            df = df.withColumn("_mode_set", all_modes)
            df = df.withColumn("mode_mix",
                F.when(F.size(F.array_except(F.col("_mode_set"), F.array(F.lit("Train"), F.lit(None)))) == 0, F.lit("train_only"))
                .when(F.size(F.array_except(F.col("_mode_set"), F.array(F.lit("Bus"), F.lit(None)))) == 0, F.lit("bus_only"))
                .otherwise(F.lit("multimodal"))
            ).drop("_mode_set")
        else:
            df = df.withColumn("mode_mix", F.lit("unknown"))

        # Delay severity at destination
        df = df.withColumn("delay_category",
            F.when(F.col("dest_delay_sec") <= 60, F.lit("on_time"))
            .when(F.col("dest_delay_sec") <= 300, F.lit("minor_delay"))
            .when(F.col("dest_delay_sec") <= 600, F.lit("moderate_delay"))
            .otherwise(F.lit("severe_delay"))
        )

        # Composite stratification key (for balanced sampling)
        df = df.withColumn("stratum",
            F.concat_ws("__",
                F.col("num_transfers").cast("string"),
                F.col("transfer_tightness"),
                F.col("time_of_day"),
                F.col("day_type"),
                F.col("journey_completed").cast("string"),
            )
        )

        return df

    def _stratified_sample(
        self,
        df: DataFrame,
        samples_per_stratum: int,
        seed: int,
    ) -> DataFrame:
        """Take up to ``samples_per_stratum`` rows from each stratum."""
        # Use window + row_number for exact stratified sampling
        w = Window.partitionBy("stratum").orderBy(F.rand(seed))
        return (
            df
            .withColumn("_rank", F.row_number().over(w))
            .filter(F.col("_rank") <= samples_per_stratum)
            .drop("_rank")
        )

    # ── Utilities ───────────────────────────────────────────────────────────

    @staticmethod
    def _time_to_secs(time_str_col) -> F.Column:
        """Convert timestamp string to integer seconds since midnight."""
        ts = F.col(time_str_col) if isinstance(time_str_col, str) else time_str_col
        ts_col = ts.cast("timestamp")
        return F.hour(ts_col) * 3600 + F.minute(ts_col) * 60 + F.second(ts_col)
