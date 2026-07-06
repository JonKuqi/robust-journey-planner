"""Manages the OSM-based footpaths Trino table.

The table `{user_schema}.footpaths_osm` is computed once for all of Switzerland
by src/util/create_footpath_data.py and then filtered per region inside
CSADataHandler.build_footpaths().
"""
from __future__ import annotations

import pandas as pd


def ensure_footpaths_table(conn, settings) -> str:
    """Return the footpaths_osm table name, building it if it doesn't exist yet.

    This is the only entry point CSADataHandler needs to call. It is a no-op
    on every subsequent run after the first.
    """
    table = f"{settings.user_schema}.footpaths_osm"
    if not _table_exists(conn, table):
        print(f"Table {table!r} not found — building OSM footpaths (runs once)…")
        _run_create(settings)
    return table


def _table_exists(conn, full_table_name: str) -> bool:
    catalog, schema, table = full_table_name.split(".")
    df = pd.read_sql(
        f"""
        SELECT COUNT(*) AS cnt
        FROM {catalog}.information_schema.tables
        WHERE table_schema = '{schema}'
          AND table_name   = '{table}'
        """,
        conn,
    )
    return int(df.iloc[0]["cnt"]) > 0


def _run_create(settings) -> None:
    from src.util.create_footpath_data import main as _create
    _create(
        max_walk_m = settings.max_walk_m,
        walk_speed = settings.walking_speed_m_per_min,
        walk_base  = settings.walking_transfer_base_sec / 60,
    )
