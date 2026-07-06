from __future__ import annotations

"""CSA table preparation for the scheduled journey planner.

`CSADataHandler` keeps the timetable preparation in Trino, as in Assignment 1,
but centralizes credentials and table names through `src.config`.
"""

from contextlib import closing
import time
from typing import Iterable, Optional

from src.config.settings import ProjectSettings, get_settings
from src.config.trino_connection import create_trino_connection

_BAR = "=" * 44
_THIN_BAR = "-" * 44

"""
NOTE: You can remove line and operatorm not needed for trips

"""


class CSADataHandler:
    """Build the Trino tables consumed by the Connection Scan Algorithm."""

    def __init__(
        self,
        schema: str | None = None,
        shared_schema: str | None = None,
        table_names: Optional[dict[str, str]] = None,
        settings: ProjectSettings | None = None,
        conn=None,
        max_pub_date: str | None = None,
    ):
        """Initialize the data handler.

        Args:
            schema: User schema where derived CSA tables are created.
            shared_schema: Shared schema containing source transit tables.
            table_names: Optional overrides for default source/target names.
            settings: Shared project settings.
            conn: Optional existing Trino connection.
            max_pub_date: Optional timetable publication date override.
        """
        self.settings = settings or get_settings()
        self.schema = schema or self.settings.user_schema
        self.shared_schema = shared_schema or self.settings.shared_schema

        if not self.schema:
            raise ValueError(
                "No user schema configured. Set COM490_USER_SCHEMA or EPFL_COM490_TOKEN."
            )

        self.conn = conn or create_trino_connection(self.settings)

        default_tables = {
            "src_stop_times": f"{self.shared_schema}.sbb_stop_times",
            "src_trips": f"{self.shared_schema}.sbb_trips",
            "src_calendar": f"{self.shared_schema}.sbb_calendar",
            "src_stops": f"{self.shared_schema}.sbb_stops",
            "src_routes": f"{self.shared_schema}.sbb_routes",
            "src_geo": f"{self.shared_schema}.geo",
            "stops": f"{self.schema}.stops",
            "stop_to_stop": f"{self.schema}.stop_to_stop",
            "stop_times_seq": f"{self.schema}.stop_times_seq",
            "stop_times_trips_seq": f"{self.schema}.stop_times_trips_seq",
            "full_table_seq": f"{self.schema}.full_table_seq",
            "connections": f"{self.schema}.connections",
        }
        if table_names:
            default_tables.update(table_names)

        self.tables = default_tables
        self.max_pub_date = self._format_date_literal(max_pub_date or self.get_max_pub_date())

    def build_all(
        self,
        regions: Iterable[str] | None = None,
        rebuild_prerequisites: bool = False,
        force: bool = False,
    ) -> None:
        """Build all CSA tables needed by `JourneyPlanner`.

        Args:
            regions: Optional geo UUIDs used to filter stops.
            rebuild_prerequisites: Whether to rebuild stops and footpaths.
            force: Whether to overwrite existing target tables.
        """
        regions = tuple(regions or ())

        print("\n" + _BAR)
        print("  BUILD CSA TABLES")
        print(_BAR)
        build_start = time.perf_counter()

        if rebuild_prerequisites:
            self._timed("build_stops", self.build_stops, regions=regions, force=force)
            self._timed("build_footpaths", self.build_footpaths, force=force)

        self._timed("build_stop_times_seq", self.build_stop_times_seq, force=force)
        self._timed("build_stop_times_trips_seq", self.build_stop_times_trips_seq, force=force)
        self._timed("build_full_table_seq", self.build_full_table_seq, force=force)
        self._timed("build_connections", self.build_connections, force=force)

        print(_THIN_BAR)
        print(f"  {'total build_all':<28}  {time.perf_counter() - build_start:>6.2f}s")
        print(_BAR + "\n")

    def reset_table_names(self, overrides: dict[str, str]) -> None:
        """Override one or more table names."""
        self.tables.update(overrides)

    def build_stops(self, regions: Iterable[str] | None = None, force: bool = False) -> None:
        """Build merged BPUIC stop coordinates, optionally restricted to regions."""
        regions = tuple(regions or ())
        if regions:
            region_values = ", ".join(f"'{region}'" for region in regions)
            region_join = f"""
            JOIN (
                SELECT wkb_geometry
                FROM {self._table('src_geo')}
                WHERE CAST(uuid AS VARCHAR) IN ({region_values})
            ) r
              ON ST_Contains(ST_GeomFromBinary(r.wkb_geometry), ST_Point(s.stop_lon, s.stop_lat))
            """
        else:
            region_join = ""

        create_query = f"""
        CREATE TABLE {self._table('stops')} AS
        WITH merged_stops AS (
            SELECT
                TRY_CAST(split_part(stop_id, ':', 1) AS INTEGER) AS stop_id,
                MAX(stop_name) AS stop_name,
                AVG(stop_lat) AS stop_lat,
                AVG(stop_lon) AS stop_lon
            FROM {self._table('src_stops')}
            WHERE pub_date = DATE '{self.max_pub_date}'
              AND split_part(stop_id, ':', 1) LIKE '85%'
            GROUP BY TRY_CAST(split_part(stop_id, ':', 1) AS INTEGER)
        )
        SELECT DISTINCT
            s.stop_id,
            s.stop_name,
            s.stop_lat,
            s.stop_lon
        FROM merged_stops s
        {region_join}
        WHERE s.stop_id IS NOT NULL
        """
        self._maybe_create("stops", create_query, force)

    def build_footpaths(self, force: bool = False) -> None:
        """Build per-region walking edges by filtering the precomputed OSM footpaths table.

        The global footpaths_osm table is created once for all of Switzerland by
        src/util/create_footpath_data.py (auto-triggered on first use). This method
        filters it to stops that belong to the current region.
        """
        from src.data.footpaths_data import ensure_footpaths_table
        footpaths_osm = ensure_footpaths_table(self.conn, self.settings)

        create_query = f"""
        CREATE TABLE {self._table('stop_to_stop')} AS
        SELECT f.a_stop_id, f.b_stop_id, f.distance, f.walk_time_min
        FROM {footpaths_osm} f
        WHERE f.a_stop_id IN (SELECT stop_id FROM {self._table('stops')})
          AND f.b_stop_id IN (SELECT stop_id FROM {self._table('stops')})
        """
        self._maybe_create("stop_to_stop", create_query, force)

    def build_stop_times_seq(self, force: bool = False) -> None:
        """Build stop-times filtered to the selected stop set."""
        create_query = f"""
        CREATE TABLE {self._table('stop_times_seq')} AS
        WITH normalized_stop_times AS (
            SELECT
                trip_id,
                TRY_CAST(split_part(stop_id, ':', 1) AS INTEGER) AS stop_id,
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
        WHERE st.stop_id IS NOT NULL
        """
        self._maybe_create("stop_times_seq", create_query, force)

    def build_stop_times_trips_seq(self, force: bool = False) -> None:
        """Join stop-times with trip service and route metadata."""
        create_query = f"""
        CREATE TABLE {self._table('stop_times_trips_seq')} AS
        SELECT
            st.trip_id,
            st.stop_id,
            st.stop_sequence,
            st.arrival_time,
            st.departure_time,
            st.pub_date,
            tr.service_id,
            tr.route_id,
            tr.trip_headsign,
            tr.trip_short_name
        FROM {self._table('stop_times_seq')} st
        JOIN (
            SELECT trip_id, service_id, route_id, trip_headsign, trip_short_name
            FROM {self._table('src_trips')}
            WHERE pub_date = DATE '{self.max_pub_date}'
        ) tr
          ON st.trip_id = tr.trip_id
        """
        self._maybe_create("stop_times_trips_seq", create_query, force)

    def build_full_table_seq(self, force: bool = False) -> None:
        """Join stop-times with calendar flags and route labels."""
        create_query = f"""
        CREATE TABLE {self._table('full_table_seq')} AS
        SELECT
            st.trip_id,
            st.stop_id,
            st.stop_sequence,
            st.arrival_time,
            st.departure_time,
            st.service_id,
            st.route_id,
            COALESCE(NULLIF(CAST(r.route_short_name AS VARCHAR), ''), CAST(st.trip_short_name AS VARCHAR), CAST(st.route_id AS VARCHAR)) AS line_text,
            CASE
                WHEN r.agency_id IS NULL THEN NULL
                WHEN strpos(CAST(r.agency_id AS VARCHAR), ':') > 0 THEN CAST(r.agency_id AS VARCHAR)
                ELSE concat('85:', CAST(r.agency_id AS VARCHAR))
            END AS operator_id,
            COALESCE(CAST(r.route_desc AS VARCHAR), CAST(r.route_type AS VARCHAR)) AS transport,
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
        LEFT JOIN (
            SELECT route_id, route_short_name, agency_id, route_desc, route_type
            FROM {self._table('src_routes')}
            WHERE pub_date = DATE '{self.max_pub_date}'
        ) r
          ON st.route_id = r.route_id
        """
        self._maybe_create("full_table_seq", create_query, force)

    def build_connections(self, force: bool = False) -> None:
        """Build directed adjacent trip segments for CSA scanning.

        Important: full_table_seq is already filtered to the selected stop set.
        Using stop_sequence + 1 breaks trips when intermediate stops fall outside
        the region. LEAD() over the filtered sequence finds the true next stop
        regardless of sequence numbering gaps.
        """
        dep_expr = """
            CAST(split_part(departure_time, ':', 1) AS INTEGER) * 3600 +
            CAST(split_part(departure_time, ':', 2) AS INTEGER) * 60 +
            CAST(split_part(departure_time, ':', 3) AS INTEGER)
        """
        arr_expr_next = """
            CAST(split_part(next_arr_time, ':', 1) AS INTEGER) * 3600 +
            CAST(split_part(next_arr_time, ':', 2) AS INTEGER) * 60 +
            CAST(split_part(next_arr_time, ':', 3) AS INTEGER)
        """
        create_query = f"""
        CREATE TABLE {self._table('connections')} AS
        WITH with_next AS (
            SELECT
                trip_id,
                line_text,
                operator_id,
                transport,
                service_id,
                stop_id,
                stop_sequence,
                arrival_time,
                departure_time,
                monday,
                tuesday,
                wednesday,
                thursday,
                friday,
                saturday,
                sunday,
                LEAD(stop_sequence) OVER (
                    PARTITION BY trip_id ORDER BY stop_sequence
                ) AS next_seq,
                LEAD(stop_id) OVER (
                    PARTITION BY trip_id ORDER BY stop_sequence
                ) AS next_stop_id,
                LEAD(arrival_time) OVER (
                    PARTITION BY trip_id ORDER BY stop_sequence
                ) AS next_arr_time
            FROM {self._table('full_table_seq')}
        )
        SELECT
            trip_id,
            line_text,
            operator_id,
            transport,
            stop_id          AS dep_stop_id,
            next_stop_id     AS arr_stop_id,
            stop_sequence    AS dep_stop_sequence,
            next_seq         AS arr_stop_sequence,
            departure_time   AS dep_time,
            next_arr_time    AS arr_time,
            {dep_expr}       AS dep_secs,
            {arr_expr_next}  AS arr_secs,
            service_id,
            monday,
            tuesday,
            wednesday,
            thursday,
            friday,
            saturday,
            sunday
        FROM with_next
        WHERE next_stop_id IS NOT NULL
        AND departure_time IS NOT NULL
        AND next_arr_time IS NOT NULL
        AND stop_id <> next_stop_id
        AND ({arr_expr_next}) >= ({dep_expr})
        AND NOT (
            (({arr_expr_next}) - ({dep_expr})) >= 24 * 3600
            AND (({arr_expr_next}) - ({dep_expr}) - 24 * 3600) BETWEEN 0 AND 6 * 3600
        )
        """
        self._maybe_create("connections", create_query, force)

    def fetch_stops(self):
        """Fetch stop metadata used by the planner."""
        t = time.perf_counter()
        df = self._query_df(f"SELECT stop_id, stop_name, stop_lat, stop_lon FROM {self._table('stops')}")
        print(f"  {'fetch_stops':<28}  {time.perf_counter() - t:>6.2f}s  ({len(df)} rows)")
        return df

    def fetch_footpaths(self):
        """Fetch walking-transfer edges."""
        t = time.perf_counter()
        df = self._query_df(f"SELECT * FROM {self._table('stop_to_stop')}")
        print(f"  {'fetch_footpaths':<28}  {time.perf_counter() - t:>6.2f}s  ({len(df)} rows)")
        return df

    def fetch_connections(self):
        """Fetch CSA connections sorted by departure time."""
        query = f"""
        SELECT
            dep_stop_id,
            arr_stop_id,
            dep_secs,
            arr_secs,
            trip_id,
            line_text,
            operator_id,
            transport,
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
        t = time.perf_counter()
        df = self._query_df(query)
        print(f"  {'fetch_connections':<28}  {time.perf_counter() - t:>6.2f}s  ({len(df)} rows)")
        return df

    def preview_connections(self, limit: int = 10):
        """Preview a limited slice of connection rows."""
        query = f"""
        SELECT *
        FROM {self._table('connections')}
        ORDER BY dep_secs, trip_id, dep_stop_sequence
        LIMIT {limit}
        """
        return self._query_df(query)

    def get_max_pub_date(self):
        """Return the latest timetable publication date available in Trino."""
        df = self._query_df(
            f"""
            SELECT MAX(pub_date) AS max_pub_date
            FROM {self._table('src_stop_times')}
            """
        )
        return df.iloc[0]["max_pub_date"]

    def _table(self, name: str) -> str:
        if name not in self.tables:
            raise KeyError(f"Unknown table name: {name}")
        return self.tables[name]

    def _execute(self, query: str) -> None:
        with closing(self.conn.cursor()) as cur:
            cur.execute(query)

    def _query_df(self, query: str):
        import pandas as pd

        return pd.read_sql(query, self.conn)

    def _table_exists(self, full_table_name: str) -> bool:
        parts = full_table_name.split(".")
        if len(parts) != 3:
            raise ValueError(f"Expected catalog.schema.table, got {full_table_name}")

        catalog, schema, table = parts
        query = f"""
        SELECT COUNT(*) AS cnt
        FROM {catalog}.information_schema.tables
        WHERE table_schema = '{schema}'
          AND table_name = '{table}'
        """
        return int(self._query_df(query).iloc[0]["cnt"]) > 0

    def _maybe_create(self, name: str, create_query: str, force: bool) -> None:
        table_name = self._table(name)
        if force:
            self._drop_existing_relation(table_name)
            print(f"  creating  {name}...")
            self._execute(create_query)
            return

        if self._table_exists(table_name):
            print(f"  skipping  {name} (already exists)")
            return

        print(f"  creating  {name}...")
        self._execute(create_query)

    def _drop_existing_relation(self, table_name: str) -> None:
        """Drop an existing Trino table/view/materialized view with this name."""
        try:
            self._execute(f"DROP TABLE IF EXISTS {table_name}")
            return
        except Exception as exc:
            message = str(exc).lower()
            if "materialized view" in message:
                self._execute(f"DROP MATERIALIZED VIEW IF EXISTS {table_name}")
                return
            if "view with that name exists" in message or "not a table" in message:
                self._execute(f"DROP VIEW IF EXISTS {table_name}")
                return
            raise

    def _timed(self, label: str, func, **kwargs) -> None:
        start = time.perf_counter()
        func(**kwargs)
        print(f"  {label:<28}  {time.perf_counter() - start:>6.2f}s")

    @staticmethod
    def _format_date_literal(value) -> str:
        """Normalize date-like values for Trino DATE literals."""
        if hasattr(value, "date"):
            value = value.date()
        return str(value)[:10]
