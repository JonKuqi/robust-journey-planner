from __future__ import annotations

"""Data preparation and validation utilities for CSA table pipelines.

This module defines `CSADataHandler`, which builds transit connection tables
used by a Connection Scan Algorithm (CSA) journey planner.
"""

from trino.auth import JWTAuthentication
from urllib.parse import urlparse
from trino.dbapi import connect
from contextlib import closing
from typing import Iterable, Optional
import pandas as pd
import os
import time


class CSADataHandler:
    """
    Builds and validates CSA-ready tables inside a user schema.

    Main flow:
        raw GTFS-like tables
            -> stop_times_seq
            -> stop_times_trips_seq
            -> full_table_seq
            -> connections

    Optionally, this class can later also rebuild prerequisite tables
    such as stops and stop_to_stop.
    """

    def __init__(
        self,
        schema: str,
        shared_schema: str,
        table_names: Optional[dict[str, str]] = None,
    ):
        """Initialize data handler and resolve source/target table names.

        Args:
            schema: User schema where derived CSA tables are created.
            shared_schema: Shared schema containing source transit tables.
            table_names: Optional per-table overrides for default names.
        """
        self.schema = schema
        self.shared_schema = shared_schema
        
         # Creates connection
        self.conn = self._create_connection()
        
        self.max_pub_date = self.get_max_pub_date()
        

        default_tables = {
            # source tables
            "src_stop_times": f"{shared_schema}.sbb_stop_times",
            "src_trips": f"{shared_schema}.sbb_trips",
            "src_calendar": f"{shared_schema}.sbb_calendar",
            "src_stops": f"{shared_schema}.sbb_stops",
            "src_geo": f"{shared_schema}.geo",

            # derived / working tables
            "stops": f"{schema}.stops",
            "stop_to_stop": f"{schema}.stop_to_stop",
            "stop_times_seq": f"{schema}.stop_times_seq",
            "stop_times_trips_seq": f"{schema}.stop_times_trips_seq",
            "full_table_seq": f"{schema}.full_table_seq",
            "connections": f"{schema}.connections",
        }

        if table_names:
            default_tables.update(table_names)

        self.tables = default_tables
        

    # Public API

    def build_all(self, regions=None, rebuild_prerequisites: bool = False, force: bool = False) -> None:
        import time

        if rebuild_prerequisites:
            t0 = time.perf_counter()
            self.build_stops(regions=regions, force=force)
            print(f"build_stops: {time.perf_counter() - t0:.2f}s")

            t0 = time.perf_counter()
            self.build_footpaths(force=force)
            print(f"build_footpaths: {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        self.build_stop_times_seq(force=force)
        print(f"build_stop_times_seq: {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        self.build_stop_times_trips_seq(force=force)
        print(f"build_stop_times_trips_seq: {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        self.build_full_table_seq(force=force)
        print(f"build_full_table_seq: {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        self.build_connections(force=force)
        print(f"build_connections: {time.perf_counter() - t0:.2f}s")


    def validate_all(self, sample_limit: int = 10) -> dict[str, pd.DataFrame]:
        """Run all validation checks and return their result DataFrames.

        Args:
            sample_limit: Max number of sample rows returned per check.

        Returns:
            Mapping of check names to validation query results.
        """
        return {
            "connection_count": self.validate_connection_count(),
            "broken_sequences": self.validate_no_broken_sequences(sample_limit=sample_limit),
            "negative_travel_times": self.validate_no_negative_travel_times(sample_limit=sample_limit),
            "null_times": self.validate_no_null_times(sample_limit=sample_limit),
            "duplicate_edges": self.validate_no_duplicate_edges(sample_limit=sample_limit),
        }


    def build_and_validate(self, rebuild_prerequisites: bool = False, force: bool = False, sample_limit: int = 10):
        """Build CSA tables and immediately run the validation suite.

        Args:
            rebuild_prerequisites: Whether to rebuild prerequisite base tables.
            force: Whether to rebuild target tables even if they already exist.
            sample_limit: Max number of sample rows returned per check.

        Returns:
            Validation output from `validate_all`.
        """
        self.build_all(rebuild_prerequisites=rebuild_prerequisites, force=force)
        return self.validate_all(sample_limit=sample_limit)
            

        
        
    # Connection
    def _create_connection(self):
        """Create and return an authenticated Trino database connection."""
        trinoAuth = JWTAuthentication(os.environ.get('EPFL_COM490_TOKEN'))
        trinoUrl = urlparse(os.environ.get('TRINO_URL'))

        print(f"Connecting to {trinoUrl.hostname}:{trinoUrl.port}...")

        conn = connect(
            host=trinoUrl.hostname,
            port=trinoUrl.port,
            auth=trinoAuth,
            http_scheme=trinoUrl.scheme,
            verify=True
        )

        print("Connected to Trino")
        return conn
    


    # Helpers
    
    def _table_exists(self, full_table_name: str) -> bool:
        """Check whether a fully qualified table exists in Trino metadata.

        Args:
            full_table_name: Table name as catalog.schema.table.

        Returns:
            True if the table exists, else False.

        Raises:
            ValueError: If the table name is not fully qualified.
        """
        parts = full_table_name.split(".")
        if len(parts) != 3:
            raise ValueError(
                f"Expected fully qualified table name 'catalog.schema.table', got: {full_table_name}"
            )

        catalog, schema, table = parts

        query = f"""
        SELECT COUNT(*) AS cnt
        FROM {catalog}.information_schema.tables
        WHERE table_schema = '{schema}'
        AND table_name = '{table}'
        """

        df = self._query_df(query)
        return df.iloc[0]["cnt"] > 0


    def _maybe_create(self, name: str, create_query: str, force: bool):
        """Create a table if missing, or always recreate when forced.

        Args:
            name: Logical table key in `self.tables`.
            create_query: SQL CREATE TABLE AS query.
            force: If True, run creation regardless of existing table.
        """
        table_name = self._table(name)

        if force:
            print(f"Creating {name}...")
            try:
                self._execute(f"DROP TABLE IF EXISTS {table_name}")
            except Exception:
                pass
            self._execute(create_query)
            return

        if self._table_exists(table_name):
            print(f"Skipping {name} (already exists)")
            return

        print(f"Creating {name}...")
        self._execute(create_query)




    def reset_table_names(self, overrides: dict[str, str]) -> None:
        """
        Override one or more table names after initialization.
        """
        self.tables.update(overrides)

    def _table(self, name: str) -> str:
        """Resolve a logical table key to its fully qualified table name."""
        if name not in self.tables:
            raise KeyError(f"Unknown table name: {name}")
        return self.tables[name]

    def _execute(self, query: str) -> None:
        """Execute a SQL statement that does not need a DataFrame result."""
        with closing(self.conn.cursor()) as cur:
            cur.execute(query)

    def _query_df(self, query: str) -> pd.DataFrame:
        """Execute a SQL query and return the result as a DataFrame."""
        return pd.read_sql(query, self.conn)






    # Core build steps

    def build_stops(self, regions = None, force: bool = False) -> None:
        """Build the stop table, optionally filtered to selected regions."""
        if regions:
            region_values = ", ".join(f"'{region}'" for region in regions)

            create_query = f"""
            CREATE TABLE {self._table('stops')} AS
            WITH target_regions AS (
                SELECT wkb_geometry
                FROM {self._table('src_geo')}
                WHERE CAST(uuid AS VARCHAR) IN ({region_values})
            ),
            merged_stops AS (
                SELECT
                    CAST(split_part(stop_id, ':', 1) AS INTEGER) AS stop_id,
                    MAX(stop_name) AS stop_name,
                    AVG(stop_lat) AS stop_lat,
                    AVG(stop_lon) AS stop_lon
                FROM {self._table('src_stops')}
                WHERE pub_date = DATE '{self.max_pub_date}'
                AND split_part(stop_id, ':', 1) LIKE '85%'
                GROUP BY CAST(split_part(stop_id, ':', 1) AS INTEGER)
            )
            SELECT DISTINCT
                s.stop_id,
                s.stop_name,
                s.stop_lat,
                s.stop_lon
            FROM merged_stops s
            JOIN target_regions r
            ON ST_Contains(ST_GeomFromBinary(r.wkb_geometry), ST_Point(s.stop_lon, s.stop_lat))
            """
        else:
            create_query = f"""
            CREATE TABLE {self._table('stops')} AS
            WITH merged_stops AS (
                SELECT
                    CAST(split_part(stop_id, ':', 1) AS INTEGER) AS stop_id,
                    MAX(stop_name) AS stop_name,
                    AVG(stop_lat) AS stop_lat,
                    AVG(stop_lon) AS stop_lon
                FROM {self._table('src_stops')}
                WHERE pub_date = DATE '{self.max_pub_date}'
                AND split_part(stop_id, ':', 1) LIKE '85%'
                GROUP BY CAST(split_part(stop_id, ':', 1) AS INTEGER)
            )
            SELECT DISTINCT
                stop_id,
                stop_name,
                stop_lat,
                stop_lon
            FROM merged_stops
            """

        self._maybe_create("stops", create_query, force)

    def build_footpaths(self, force: bool = False) -> None:
        """Build the stop_to_stop table for walking transfers."""
        create_query = f"""
        CREATE TABLE {self._table('stop_to_stop')} AS
        WITH calculated_distances AS (
            SELECT
                a.stop_id AS a_stop_id,
                b.stop_id AS b_stop_id,
                great_circle_distance(a.stop_lat, a.stop_lon, b.stop_lat, b.stop_lon) * 1000 AS distance
            FROM {self._table('stops')} a
            CROSS JOIN {self._table('stops')} b
            WHERE a.stop_id <> b.stop_id
        )
        SELECT
            a_stop_id,
            b_stop_id,
            distance,
            2.0 + (distance / 50.0) AS walk_time_min
        FROM calculated_distances
        WHERE distance <= 500
        """

        self._maybe_create("stop_to_stop", create_query, force)

    def build_stop_times_seq(self, force: bool = False) -> None:
        """Build the sequenced stop-times table filtered to known stops."""
        create_query = f"""
        CREATE TABLE {self._table('stop_times_seq')} AS
        WITH normalized_stop_times AS (
            SELECT
                trip_id,
                CAST(split_part(stop_id, ':', 1) AS INTEGER) AS stop_id,
                stop_sequence,
                arrival_time,
                departure_time,
                pub_date
            FROM {self._table('src_stop_times')}
            WHERE pub_date = DATE '{self.max_pub_date}'
        )
        SELECT
            st.trip_id,
            st.stop_id,
            st.stop_sequence,
            st.arrival_time,
            st.departure_time,
            st.pub_date
        FROM normalized_stop_times st
        JOIN {self._table('stops')} s
            ON st.stop_id = s.stop_id
        """
        self._maybe_create("stop_times_seq", create_query, force)


    def build_stop_times_trips_seq(self, force: bool = False) -> None:
        """Join sequenced stop-times with trip service identifiers."""
        create_query = f"""
        CREATE TABLE {self._table('stop_times_trips_seq')} AS
        SELECT
            st.trip_id,
            st.stop_id,
            st.stop_sequence,
            st.arrival_time,
            st.departure_time,
            st.pub_date,
            tr.service_id
        FROM {self._table('stop_times_seq')} st
        JOIN (
            SELECT trip_id, service_id
            FROM {self._table('src_trips')}
            WHERE pub_date = DATE '{self.max_pub_date}'
        ) tr
            ON st.trip_id = tr.trip_id
        """

        self._maybe_create("stop_times_trips_seq", create_query, force)


    def build_full_table_seq(self, force: bool = False) -> None:
        """Join trip stop-times with service-day calendar flags."""
        create_query = f"""
        CREATE TABLE {self._table('full_table_seq')} AS
        SELECT
            st.trip_id,
            st.stop_id,
            st.stop_sequence,
            st.arrival_time,
            st.departure_time,
            st.service_id,
            cal.monday,
            cal.tuesday,
            cal.wednesday,
            cal.thursday,
            cal.friday,
            cal.saturday,
            cal.sunday
        FROM {self._table('stop_times_trips_seq')} st
        JOIN (
            SELECT
                service_id,
                monday,
                tuesday,
                wednesday,
                thursday,
                friday,
                saturday,
                sunday
            FROM {self._table('src_calendar')}
            WHERE pub_date = DATE '{self.max_pub_date}'
        ) cal
            ON st.service_id = cal.service_id
        """

        self._maybe_create("full_table_seq", create_query, force)


    def build_connections(self, force: bool = False) -> None:
        """Build directed stop-to-stop trip segments for CSA scanning.

        Also:
        Drop suspicious adjacent edges that appear to be shifted by exactly +24h.
        This targets the anomaly we observed in the timetable data without blocking
        normal GTFS times above 24:00.
        
        """
        create_query = f"""
        CREATE TABLE {self._table('connections')} AS
        SELECT
            a.trip_id,
            a.stop_id        AS dep_stop_id,
            b.stop_id        AS arr_stop_id,
            a.stop_sequence  AS dep_stop_sequence,
            b.stop_sequence  AS arr_stop_sequence,
            a.departure_time AS dep_time,
            b.arrival_time   AS arr_time,
    
            CAST(split_part(a.departure_time, ':', 1) AS INTEGER) * 3600 +
            CAST(split_part(a.departure_time, ':', 2) AS INTEGER) * 60 +
            CAST(split_part(a.departure_time, ':', 3) AS INTEGER) AS dep_secs,
    
            CAST(split_part(b.arrival_time, ':', 1) AS INTEGER) * 3600 +
            CAST(split_part(b.arrival_time, ':', 2) AS INTEGER) * 60 +
            CAST(split_part(b.arrival_time, ':', 3) AS INTEGER) AS arr_secs,
    
            a.service_id,
            a.monday,
            a.tuesday,
            a.wednesday,
            a.thursday,
            a.friday,
            a.saturday,
            a.sunday
        FROM {self.schema}.stop_times a
        JOIN {self.schema}.stop_times b
            ON a.trip_id = b.trip_id
        AND b.stop_sequence = a.stop_sequence + 1
        WHERE a.departure_time IS NOT NULL
        AND b.arrival_time IS NOT NULL
        AND a.stop_id <> b.stop_id
        AND (
                CAST(split_part(b.arrival_time, ':', 1) AS INTEGER) * 3600 +
                CAST(split_part(b.arrival_time, ':', 2) AS INTEGER) * 60 +
                CAST(split_part(b.arrival_time, ':', 3) AS INTEGER)
            ) >= (
                CAST(split_part(a.departure_time, ':', 1) AS INTEGER) * 3600 +
                CAST(split_part(a.departure_time, ':', 2) AS INTEGER) * 60 +
                CAST(split_part(a.departure_time, ':', 3) AS INTEGER)
            )
        AND NOT (
                (
                    (
                        CAST(split_part(b.arrival_time, ':', 1) AS INTEGER) * 3600 +
                        CAST(split_part(b.arrival_time, ':', 2) AS INTEGER) * 60 +
                        CAST(split_part(b.arrival_time, ':', 3) AS INTEGER)
                    ) - (
                        CAST(split_part(a.departure_time, ':', 1) AS INTEGER) * 3600 +
                        CAST(split_part(a.departure_time, ':', 2) AS INTEGER) * 60 +
                        CAST(split_part(a.departure_time, ':', 3) AS INTEGER)
                    )
                ) >= 24 * 3600
            AND (
                    (
                        CAST(split_part(b.arrival_time, ':', 1) AS INTEGER) * 3600 +
                        CAST(split_part(b.arrival_time, ':', 2) AS INTEGER) * 60 +
                        CAST(split_part(b.arrival_time, ':', 3) AS INTEGER)
                    ) - (
                        CAST(split_part(a.departure_time, ':', 1) AS INTEGER) * 3600 +
                        CAST(split_part(a.departure_time, ':', 2) AS INTEGER) * 60 +
                        CAST(split_part(a.departure_time, ':', 3) AS INTEGER)
                    ) - 24 * 3600
                ) BETWEEN 0 AND 6 * 3600
            )
        """
        self._maybe_create("connections", create_query, force)
        
    
    
    #   Getting Data
    
    def fetch_stops(self) -> pd.DataFrame:
        """Fetch the stop list used by the planner."""
        query = f"""
        SELECT stop_id
        FROM {self._table('stops')}
        """
        return self._query_df(query)


    def fetch_footpaths(self) -> pd.DataFrame:
        """Fetch walk-transfer edges between nearby stops."""
        query = f"""
        SELECT *
        FROM {self._table('stop_to_stop')}
        """
        return self._query_df(query)


    def fetch_connections(self) -> pd.DataFrame:
        """Fetch CSA connections sorted by departure time."""
        query = f"""
        SELECT
            dep_stop_id,
            arr_stop_id,
            dep_secs,
            arr_secs,
            trip_id,
            monday,
            tuesday,
            wednesday,
            thursday,
            friday,
            saturday,
            sunday
        FROM {self._table('connections')}
        ORDER BY dep_secs, trip_id, dep_stop_sequence
        """
        return self._query_df(query)
    
    
    
    
        
  
    # Validation
    def validate_connection_count(self) -> pd.DataFrame:
        """Return the total number of connection rows."""
        query = f"""
        SELECT COUNT(*) AS n_connections
        FROM {self._table('connections')}
        """
        return self._query_df(query)

    def validate_no_broken_sequences(self, sample_limit: int = 10) -> pd.DataFrame:
        """Return samples where adjacent stop sequences are inconsistent."""
        query = f"""
        SELECT *
        FROM {self._table('connections')}
        WHERE arr_stop_sequence <> dep_stop_sequence + 1
        LIMIT {sample_limit}
        """
        return self._query_df(query)

    def validate_no_negative_travel_times(self, sample_limit: int = 10) -> pd.DataFrame:
        """Return samples where arrival seconds are before departure seconds."""
        query = f"""
        SELECT *
        FROM {self._table('connections')}
        WHERE arr_secs < dep_secs
        LIMIT {sample_limit}
        """
        return self._query_df(query)

    def validate_no_null_times(self, sample_limit: int = 10) -> pd.DataFrame:
        """Return samples with missing time fields used by CSA."""
        query = f"""
        SELECT *
        FROM {self._table('connections')}
        WHERE dep_time IS NULL
           OR arr_time IS NULL
           OR dep_secs IS NULL
           OR arr_secs IS NULL
        LIMIT {sample_limit}
        """
        return self._query_df(query)

    def validate_no_duplicate_edges(self, sample_limit: int = 10) -> pd.DataFrame:
        """Return duplicated trip edges based on stop-time identity fields."""
        query = f"""
        SELECT
            trip_id,
            dep_stop_id,
            arr_stop_id,
            dep_stop_sequence,
            arr_stop_sequence,
            dep_time,
            arr_time,
            COUNT(*) AS n_duplicates
        FROM {self._table('connections')}
        GROUP BY
            trip_id,
            dep_stop_id,
            arr_stop_id,
            dep_stop_sequence,
            arr_stop_sequence,
            dep_time,
            arr_time
        HAVING COUNT(*) > 1
        LIMIT {sample_limit}
        """
        return self._query_df(query)

    def preview_connections(self, limit: int = 10) -> pd.DataFrame:
        """Preview a limited, time-ordered slice of connection rows."""
        query = f"""
        SELECT *
        FROM {self._table('connections')}
        ORDER BY dep_secs, trip_id, dep_stop_sequence
        LIMIT {limit}
        """
        return self._query_df(query)
    
    
    def get_max_pub_date(self):
        """Return the most recent publication date from source stop_times."""
        query = f"""
        SELECT MAX(pub_date) AS max_pub_date
        FROM {self.shared_schema}.sbb_stop_times
        """
        df = pd.read_sql(query, self.conn)
        return df.iloc[0]["max_pub_date"]















if __name__ == "__main__":
    import os
    import json
    import time
    import base64 as b64
    import re
    from urllib.parse import urlparse
    from trino.dbapi import connect
    from trino.auth import JWTAuthentication

    def get_username():
        payload = os.environ.get("EPFL_COM490_TOKEN").split(".")[1]
        payload = payload + "=" * (4 - len(payload) % 4)
        obj = json.loads(b64.urlsafe_b64decode(payload))
        if time.time() > int(obj.get("exp")) - 3600:
            raise Exception(
                "Your credentials have expired. Restart your JupyterHub server."
            )
        return obj.get("sub")

    group_name = "J1"   # change if needed

    username = get_username()
    userns = f"iceberg.{username}_iceberg"
    sharedns = "iceberg.com490_iceberg"

    trino_auth = JWTAuthentication(os.environ.get("EPFL_COM490_TOKEN"))
    trino_url = urlparse(os.environ.get("TRINO_URL"))

    print(f"Connecting to Data Query Engine URL: {trino_url.scheme}://{trino_url.hostname}:{trino_url.port}/")

    conn = connect(
        host=trino_url.hostname,
        port=trino_url.port,
        auth=trino_auth,
        http_scheme=trino_url.scheme,
        verify=True
    )

    print("Connected to Trino")
    #  compute this BEFORE builder init
    max_pub_date = pd.read_sql(f"""
        SELECT MAX(pub_date) AS max_pub_date
        FROM {sharedns}.sbb_stop_times
    """, conn).iloc[0]["max_pub_date"]

    print(f"Using max_pub_date: {max_pub_date}")

    builder = CSADataHandler(
        schema=userns,
        shared_schema=sharedns,
    )

    print("Builder initialized")

    builder.build_all(rebuild_prerequisites=False)
    print("Build complete")

    print(builder.preview_connections(10))

    results = builder.validate_all()
    for name, df in results.items():
        print(f"\n=== {name} ===")
        print(df)
