from __future__ import annotations

"""Download daily SBB Ist Daten CSV files from opentransportdata.swiss.

Each day's file contains only trips that operated on that specific day.
No API key is needed — the public dataset HTML page is scraped with
BeautifulSoup to discover per-day download URLs (the CKAN REST API
returns 403 without auth, but the HTML page is open).

Column names in the source CSV are German; this module renames them to
match the cluster table schema (iceberg.sbb.istdaten).
"""

import csv
import io
import re
from contextlib import closing
from datetime import date, timedelta
from typing import Optional

import requests

from src.config.settings import ProjectSettings, get_settings
from src.config.trino_connection import create_trino_connection

COLUMN_MAP: dict[str, str] = {
    "BETRIEBSTAG":         "operating_day",
    "FAHRT_BEZEICHNER":    "trip_id",
    "BETREIBER_ID":        "operator_id",
    "BETREIBER_ABK":       "operator_abrv",
    "BETREIBER_NAME":      "operator_name",
    "PRODUKT_ID":          "product_id",
    "LINIEN_ID":           "line_id",
    "LINIEN_TEXT":         "line_text",
    "UMLAUF_ID":           "circuit_id",
    "VERKEHRSMITTEL_TEXT": "transport",
    "ZUSATZFAHRT_TF":      "unplanned",
    "FAELLT_AUS_TF":       "failed",
    "BPUIC":               "bpuic",
    "HALTESTELLEN_NAME":   "stop_name",
    "ANKUNFTSZEIT":        "arr_time",
    "AN_PROGNOSE":         "arr_actual",
    "AN_PROGNOSE_STATUS":  "arr_status",
    "ABFAHRTSZEIT":        "dep_time",
    "AB_PROGNOSE":         "dep_actual",
    "AB_PROGNOSE_STATUS":  "dep_status",
    "DURCHFAHRT_TF":       "transit",
    # SLOID has no equivalent in the cluster table — dropped
}


class IstDatenFetcher:
    """Fetch daily Ist Daten CSV files and prepare rows for Trino insertion."""

    def __init__(
        self,
        settings: ProjectSettings | None = None,
        conn=None,
        timeout_sec: int = 60,
    ):
        self.settings = settings or get_settings()
        self.conn = conn or create_trino_connection(self.settings)
        self.timeout_sec = timeout_sec
        self._resource_index: Optional[dict[str, str]] = None

    def fetch_day(self, day: date | str) -> list[dict]:
        """Download one day's Ist Daten and return rows with English column names.

        Args:
            day: Date as a date object or ISO string (YYYY-MM-DD).

        Returns:
            List of dicts with columns matching the cluster iceberg.sbb.istdaten schema.
        """
        date_str = day if isinstance(day, str) else day.isoformat()
        url = self._resolve_url(date_str)
        return self._download_and_parse(url)

    def fetch_range(self, start: date | str, end: date | str) -> list[dict]:
        """Download and combine multiple days.

        Args:
            start: First date (inclusive).
            end:   Last date (inclusive).

        Returns:
            Combined list of row dicts for the date range.
        """
        start_d = date.fromisoformat(start) if isinstance(start, str) else start
        end_d   = date.fromisoformat(end)   if isinstance(end,   str) else end

        rows: list[dict] = []
        current = start_d
        while current <= end_d:
            try:
                rows.extend(self.fetch_day(current))
            except Exception as exc:
                print(f"[IstDatenFetcher] skipping {current}: {exc}")
            current += timedelta(days=1)

        if not rows:
            raise RuntimeError(f"No data fetched for {start_d} – {end_d}.")
        return rows

    def copy_from_shared(self, force: bool = False) -> None:
        """Copy the shared cluster istdaten table into the user's personal schema.

        Creates {user_schema}.istdaten as a full copy of the frozen shared table.
        After this, new rows from the internet can be appended daily.
        Pass force=True to drop and recreate.

        Note: settings.istdaten_table is the Spark path (iceberg.sbb.istdaten).
        The Trino path follows the shared schema convention: {shared_schema}.sbb_istdaten.
        """
        target = f"{self.settings.user_schema}.istdaten"
        source = f"{self.settings.shared_schema}.sbb_istdaten"

        if force:
            self._drop_if_exists(target)

        if self._table_exists(target):
            print(f"  skipping copy — {target} already exists (use force=True to recreate)")
            return

        print(f"  copying {source} -> {target} (from 2025-01-01 onwards) ...")
        self._execute(f"CREATE TABLE {target} AS SELECT * FROM {source} WHERE operating_day >= DATE '2025-01-01'")
        print("  done.")

    def copy_from_shared_spark(self, spark, force: bool = False) -> None:
        """Copy the shared istdaten table into the user's local spark_catalog warehouse.

        Writes to spark_catalog (user's writable HDFS warehouse) so the table is
        visible to all subsequent Spark jobs. Trino is not involved here.
        """
        target = self.settings.spark_user_istdaten_table  # e.g. "default.istdaten"
        source = self.settings.istdaten_table              # e.g. "iceberg.sbb.istdaten"

        if spark.catalog.tableExists(target):
            if not force:
                print(f"  skipping Spark copy — {target} already exists (use force=True to recreate)")
                return
            spark.sql(f"DROP TABLE {target}")

        print(f"  copying {source} -> {target} via Spark (from 2025-01-01 onwards) ...")
        (
            spark.table(source)
            .where("operating_day >= '2025-01-01'")
            .write
            .format("iceberg")
            .saveAsTable(target)
        )
        print("  done.")

    def append_to_table(self, rows: list[dict], table: str) -> None:
        """Insert rows into a Trino table.

        Not yet implemented — placeholder for Part 2 of the Model Updator build.
        Will INSERT fetched rows into the user's personal istdaten table.
        """
        raise NotImplementedError("Trino istdaten table append not yet implemented.")

    def _table_exists(self, full_table_name: str) -> bool:
        parts = full_table_name.split(".")
        if len(parts) != 3:
            raise ValueError(f"Expected catalog.schema.table, got {full_table_name!r}")
        catalog, schema, table = parts
        with closing(self.conn.cursor()) as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM {catalog}.information_schema.tables"
                f" WHERE table_schema = '{schema}' AND table_name = '{table}'"
            )
            return cur.fetchone()[0] > 0

    def _execute(self, query: str) -> None:
        with closing(self.conn.cursor()) as cur:
            cur.execute(query)

    def _drop_if_exists(self, table: str) -> None:
        try:
            self._execute(f"DROP TABLE IF EXISTS {table}")
        except Exception as exc:
            msg = str(exc).lower()
            if "materialized view" in msg:
                self._execute(f"DROP MATERIALIZED VIEW IF EXISTS {table}")
            elif "view with that name exists" in msg or "not a table" in msg:
                self._execute(f"DROP VIEW IF EXISTS {table}")
            else:
                raise

    def _resolve_url(self, date_str: str) -> str:
        if self._resource_index is None:
            self._resource_index = self._build_resource_index()

        url = self._resource_index.get(date_str)
        if url is None:
            self._resource_index = self._build_resource_index()
            url = self._resource_index.get(date_str)

        if url is None:
            available = sorted(self._resource_index)[-5:]
            raise ValueError(
                f"No Ist Daten resource found for {date_str}. "
                f"Most recent available: {available}"
            )
        return url

    def _build_resource_index(self) -> dict[str, str]:
        """Scrape the public dataset HTML page and build a date → download URL mapping."""
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise RuntimeError("beautifulsoup4 is required: pip install beautifulsoup4") from exc

        resp = requests.get(self.settings.istdaten_source_url, timeout=self.timeout_sec)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        pattern = re.compile(r"/download/(\d{4}-\d{2}-\d{2})_istdaten\.csv", re.IGNORECASE)

        index: dict[str, str] = {}
        for tag in soup.find_all("a", href=True):
            href: str = tag["href"]
            m = pattern.search(href)
            if m:
                date_str = m.group(1)
                url = href if href.startswith("http") else f"https://data.opentransportdata.swiss{href}"
                index[date_str] = url

        if not index:
            raise RuntimeError(
                "No download links found on the dataset page. "
                "The page structure may have changed."
            )
        return index

    def _download_and_parse(self, url: str) -> list[dict]:
        """Download the semicolon-delimited CSV and return renamed rows as dicts."""
        resp = requests.get(url, timeout=self.timeout_sec)
        resp.raise_for_status()

        text = resp.content.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text), delimiter=";")

        rows = []
        for raw in reader:
            row = {COLUMN_MAP[k]: v for k, v in raw.items() if k in COLUMN_MAP}
            rows.append(row)
        return rows
