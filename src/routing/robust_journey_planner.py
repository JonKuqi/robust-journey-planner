from __future__ import annotations

"""Scheduled journey planning with the profile Connection Scan Algorithm (pCSA).

`plan()` is a thin wrapper around `plan_candidates`, which runs a
single bounded backward *profile* CSA scan with confidence baked in.

Design notes
------------
Everything that does not depend on the *query* is precomputed in `prepare()`:

  * connections are stored as column arrays (struct-of-arrays) per weekday, so
    the hot loop indexes flat lists of ints instead of unpacking tuples and
    touching string trip ids;
  * footpaths are sorted by distance ascending, so the walk loops can `break`
    as soon as they pass `max_walk_m`;
  * trip metadata is flattened into a list indexed by `trip_idx`, so
    reconstruction is an O(1) lookup instead of `str(trip_id)` -> dict;
  * `(a, b) -> walk_seconds` is materialized for cheap reconstruction.

The hot path therefore only does: array reads, two `bisect` calls per
connection (profile transfer eval) and the Pareto inserts.

Profile fronts are kept sorted ascending by departure time, which (for a
non-dominated set under "later departure better / earlier arrival better")
makes arrival monotonically ascending too. That lets us answer
"earliest arrival reachable if ready at time t" with a single `bisect_left`
instead of the old O(n) linear scan.

Confidence is baked into the scan: each profile entry carries a `log_conf`
field (accumulated log-probability of making all downstream connections). The
gate `lc >= log(confidence_q)` is applied during the scan, so only routes
passing the confidence threshold reach the source stop profile.
Post-scan annotation is documentation-only, not filtering.
"""

import json
import math
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from cachetools import TTLCache, cached

_BAR = "=" * 44
_THIN_BAR = "-" * 44

from src.config.settings import ProjectSettings, get_settings
from src.data.csa_data_handler import CSADataHandler
from src.models.model_artifacts import ModelArtifacts
from src.util.confidence import (
    DEFAULT_QUANTILE_LEVELS,
    normalize_confidence_q,
    route_confidence_details_from_steps,
)


DAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_SPARK_DOW = {
    "sunday": 1,
    "monday": 2,
    "tuesday": 3,
    "wednesday": 4,
    "thursday": 5,
    "friday": 6,
    "saturday": 7,
}
# Python weekday() (0=Mon…6=Sun) → Spark dayofweek (1=Sun, 2=Mon…7=Sat)
_WEEKDAY_TO_SPARK_DOW = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 1}
_DEFAULT_DELAY_CATEGORICAL_COLS = ("bpuic", "line_text", "hour", "day_of_week", "month", "day_type")

# exit-leg kinds stored in the trip label / profile payloads
_KIND_NONE = 0
_KIND_WALK = 1   # alight and walk to the target
_KIND_XFER = 2   # alight and transfer to another connection

LOG_ZERO = -1e18


class RobustJourneyPlanner:
    """Profile-CSA scheduled journey planner with full step reconstruction."""

    # ------------------------------------------------------------------
    # Setup and prepare-time state
    # ------------------------------------------------------------------

    def __init__(
        self,
        schema: str | None = None,
        settings: ProjectSettings | None = None,
        data_handler: CSADataHandler | None = None,
    ):
        """Initialize a planner for a user schema."""
        self.settings = settings or get_settings()
        self.schema = schema or self.settings.user_schema
        if not self.schema:
            raise ValueError("No user schema configured. Set COM490_USER_SCHEMA or EPFL_COM490_TOKEN.")

        self.shared_schema = self.settings.shared_schema
        self.data_handler = data_handler

        self.connections_by_day = None
        self.dep_secs_by_day = None
        self.footpaths = None
        self.footpath_dist = {}
        self.footpath_secs = {}
        self.stops = None
        self.stop_metadata = {}
        self.trip_id_to_idx = {}
        self.trip_idx_to_id = []
        self.trip_metadata_by_id = {}
        self.trip_meta_by_idx = []
        self.n_trips = 0
        self.conn = {}
        self.delay_lookup = {}
        self.delay_categorical_cols = _DEFAULT_DELAY_CATEGORICAL_COLS
        self.delay_quantile_levels = DEFAULT_QUANTILE_LEVELS
        self.delay_fallback_cols = (
            ("bpuic", "line_text", "hour"),
            ("bpuic", "line_text"),
        )
        self.delay_fallback_lookups = []
        self.delay_model = None
        self.day_delay_lookup: dict = {}  # 3-field baked lookup; used by annotation

        self.prepared = False
        self.min_transfer_secs = 120

    # ------------------------------------------------------------------
    # Loading the delays in memory
    # ------------------------------------------------------------------

    def configure_delay_lookup(
        self,
        precomputed_delays: dict,
        metadata: dict | None = None,
    ) -> "RobustJourneyPlanner":
        """Attach precomputed delay quantiles loaded from model artifacts."""
        metadata = metadata or {}
        self.delay_lookup = precomputed_delays or {}
        self.delay_categorical_cols = tuple(
            metadata.get("categorical_cols") or _DEFAULT_DELAY_CATEGORICAL_COLS
        )
        self.delay_quantile_levels = tuple(
            float(q) for q in (
                metadata.get("quantile_levels")
                or metadata.get("quantiles")
                or DEFAULT_QUANTILE_LEVELS
            )
        )
        self.delay_fallback_lookups = self._build_delay_fallback_lookups()
        self._route_cache.clear()
        return self


    def prepare(
        self,
        regions: Iterable[str] | None = None,
        travel_date: str | None = None,
        rebuild: bool = False,
        rebuild_prerequisites: bool = True,
        load_delay_lookup: bool = False,
        require_delay_lookup: bool = True,
    ) -> "RobustJourneyPlanner":
        """Build/load CSA data and materialize in-memory routing structures.

        When travel_date (ISO string, e.g. "2026-02-05") is given together with
        load_delay_lookup=True, per-connection delay quantiles are baked into
        conn[day]["quantiles"] at prepare() time so the hot scan loop does zero
        dict lookups and zero string ops per connection.
        """
        t_total = time.perf_counter()
        regions = tuple(regions if regions is not None else self.settings.region_uuids)

        if load_delay_lookup:
            self._load_delay_lookup_from_artifacts(required=require_delay_lookup, region_uuids=regions)
            self._load_delay_model()


        self.data_handler = self.data_handler or CSADataHandler(
            schema=self.schema,
            shared_schema=self.shared_schema,
            settings=self.settings,
        )
        self.data_handler.build_all(
            regions=regions,
            rebuild_prerequisites=rebuild_prerequisites,
            force=rebuild,
        )

        print(_BAR)
        print("  FETCH DATA")
        print(_BAR)
        stops_df = self.data_handler.fetch_stops()
        footpaths_df = self.data_handler.fetch_footpaths()
        connections_df = self.data_handler.fetch_connections()
        print(_BAR + "\n")

        print(_BAR)
        print("  MATERIALIZE IN-MEMORY")
        print(_BAR)

        t = time.perf_counter()
        self.stops = set(int(stop_id) for stop_id in stops_df["stop_id"].tolist())
        self.stop_metadata = {
            int(row.stop_id): {
                "stop_name": getattr(row, "stop_name", None),
                "stop_lat": getattr(row, "stop_lat", None),
                "stop_lon": getattr(row, "stop_lon", None),
            }
            for row in stops_df.itertuples(index=False)
        }
        print(f"  {'stops dict':<28}  {time.perf_counter() - t:>6.2f}s  ({len(self.stops)} stops)")

        t = time.perf_counter()
        self.footpaths = defaultdict(list)
        self.footpath_dist = {}
        self.footpath_secs = {}
        for row in footpaths_df.itertuples(index=False):
            a = int(row.a_stop_id)
            b = int(row.b_stop_id)
            secs = int(round(float(row.walk_time_min) * 60))
            dist = float(row.distance)
            self.footpaths[a].append((b, secs, dist))
            self.footpath_dist[(a, b)] = dist
            self.footpath_secs[(a, b)] = secs
        for a in self.footpaths:
            self.footpaths[a].sort(key=lambda e: e[2])
        print(f"  {'footpaths dict':<28}  {time.perf_counter() - t:>6.2f}s  ({len(footpaths_df)} edges)")

        t = time.perf_counter()
        self.connections_by_day = {day: [] for day in DAY_NAMES}
        self.dep_secs_by_day = {day: [] for day in DAY_NAMES}
        self.trip_id_to_idx = {}
        self.trip_idx_to_id = []
        self.trip_metadata_by_id = {}

        for row in connections_df.itertuples(index=False):
            trip_id = str(row.trip_id)
            if trip_id not in self.trip_id_to_idx:
                self.trip_id_to_idx[trip_id] = len(self.trip_idx_to_id)
                self.trip_idx_to_id.append(trip_id)

            self.trip_metadata_by_id.setdefault(
                trip_id,
                {
                    "trip_id": trip_id,
                    "line_text": self._row_value(row, "line_text"),
                    "operator_id": self._row_value(row, "operator_id"),
                    "transport": self._row_value(row, "transport"),
                },
            )

            dep_secs = int(row.dep_secs)
            arr_secs = int(row.arr_secs)
            connection = (
                int(row.dep_stop_id),
                int(row.arr_stop_id),
                dep_secs,
                arr_secs,
                self.trip_id_to_idx[trip_id],
                arr_secs + 86400 * (dep_secs > arr_secs),
                trip_id,
            )

            for day in DAY_NAMES:
                if bool(self._row_value(row, day)):
                    self.connections_by_day[day].append(connection)
        print(f"  {'connections loop':<28}  {time.perf_counter() - t:>6.2f}s  ({len(connections_df)} rows)")

        t = time.perf_counter()
        self.n_trips = len(self.trip_idx_to_id)
        self.conn = {}
        for day in DAY_NAMES:
            conns = self.connections_by_day[day]
            conns.sort(key=lambda c: (c[2], c[4], c[0], c[1]))
            self.conn[day] = {
                "dep_stop": [c[0] for c in conns],
                "arr_stop": [c[1] for c in conns],
                "dep_secs": [c[2] for c in conns],
                "arr_adj":  [c[5] for c in conns],
                "trip_idx": [c[4] for c in conns],
            }
            self.dep_secs_by_day[day] = self.conn[day]["dep_secs"]
        print(f"  {'sort + columns':<28}  {time.perf_counter() - t:>6.2f}s  ({self.n_trips} trips)")

        t = time.perf_counter()
        self.trip_meta_by_idx = [
            self.trip_metadata_by_id[self.trip_idx_to_id[k]] for k in range(self.n_trips)
        ]
        print(f"  {'trip meta by idx':<28}  {time.perf_counter() - t:>6.2f}s")

        print(_THIN_BAR)
        print(f"  {'TOTAL prepare()':<28}  {time.perf_counter() - t_total:>6.2f}s")
        print(_BAR + "\n")

        if travel_date is not None and load_delay_lookup:
            self._bake_quantiles_for_date(travel_date)

        self.min_transfer_secs = int(self.settings.walking_transfer_base_sec)
        self.prepared = True
        return self


    def _load_delay_lookup_from_artifacts(
        self,
        required: bool = True,
        region_uuids: tuple | None = None,
    ) -> bool:
        """Load precomputed delay quantiles into memory from per-district artifact files.

        Metadata (model schema) is always read from the global artifact namespace.
        Precomputed delays are loaded per district: one JSON file per UUID in
        region_uuids, merged in memory.  Any UUID whose file doesn't exist yet is
        silently skipped — XGBoost fallback covers it.

        When region_uuids is None or empty the global combined file is used instead,
        which is the safe fallback for the no-region case.
        """
        print(_BAR)
        print("  LOAD DELAY LOOKUP")
        print(_BAR)
        t = time.perf_counter()

        global_artifacts = ModelArtifacts(settings=self.settings).global_delay_artifacts()
        metadata_path = Path(global_artifacts.model_metadata_path())

        if not metadata_path.exists():
            print(f"  Missing model metadata: {metadata_path}")
            print(_BAR + "\n")
            if required:
                return False
            self.configure_delay_lookup(precomputed_delays={}, metadata={})
            return True

        try:
            with metadata_path.open("r", encoding="utf-8") as fh:
                metadata = json.load(fh)
        except Exception as exc:
            print(f"  Could not read model metadata: {exc}")
            print(_BAR + "\n")
            if required:
                return False
            self.configure_delay_lookup(precomputed_delays={}, metadata={})
            return True

        merged: dict = {}
        if region_uuids:
            found = 0
            for uuid in region_uuids:
                district_path = Path(
                    ModelArtifacts(settings=self.settings)
                    .for_region([uuid])
                    .precomputed_delays_json_path()
                )
                if district_path.exists():
                    try:
                        with district_path.open("r", encoding="utf-8") as fh:
                            merged.update(json.load(fh))
                        found += 1
                    except Exception as exc:
                        print(f"  Warning: could not read {district_path}: {exc}")
            print(f"  Loaded {found}/{len(region_uuids)} district files")
        else:
            global_path = Path(global_artifacts.precomputed_delays_json_path())
            if global_path.exists():
                try:
                    with global_path.open("r", encoding="utf-8") as fh:
                        merged = json.load(fh)
                except Exception as exc:
                    print(f"  Warning: could not read global delay file: {exc}")

        self.configure_delay_lookup(precomputed_delays=merged, metadata=metadata)
        print(
            f"  {'delay_lookup.load':<28}  {time.perf_counter() - t:>6.2f}s  "
            f"({len(self.delay_lookup)} keys)"
        )
        print(_BAR + "\n")
        return True


    def _load_delay_model(self) -> None:
        """Load the XGBoost delay model for live fallback inference at prepare() time.

        region_artifacts=None so the model's own precomputed lookup stays empty —
        it will always call _predict_xgb() live.  This is intentional: the model
        here is a fallback engine, not a second lookup layer.
        """
        from src.models.delay_model import DelayModel
        try:
            global_artifacts = ModelArtifacts(settings=self.settings).global_delay_artifacts()
            self.delay_model = DelayModel.load(artifacts=global_artifacts, region_artifacts=None)
            print(f"  delay model loaded  ({len(self.delay_model.xgb_models)} quantile boosters)")
        except Exception as exc:
            print(f"  Could not load XGBoost delay model: {exc}")
            self.delay_model = None

    _route_cache = TTLCache(maxsize=512, ttl=300)

    # ------------------------------------------------------------------
    # Public route planning entry point
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Bounded profile CSA candidate search
    # ------------------------------------------------------------------

    @cached(
        _route_cache,
        key=lambda self,
                start_stop_id,
                end_stop_id,
                travel_date,
                arrival_deadline,
                max_routes=5,
                max_walk_m=None,
                search_window_minutes=180,
                confidence_q=None: (
            id(self),
            start_stop_id,
            end_stop_id,
            str(travel_date),
            arrival_deadline,
            max_routes,
            max_walk_m if max_walk_m is not None else self.settings.max_walk_m,
            search_window_minutes,
            normalize_confidence_q(confidence_q),
            id(self.delay_lookup),
        )
    )
    def plan(
        self,
        start_stop_id: int,
        end_stop_id: int,
        travel_date: str | date,
        arrival_deadline: str,
        max_routes: int = 5,
        max_walk_m: float | None = None,
        search_window_minutes: int = 180,
        confidence_q: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return scheduled route dictionaries sorted by latest departure first.

        Runs a single bounded backward *profile* CSA scan. Confidence is baked
        in: each profile entry carries accumulated log-confidence and the gate
        `lc >= log_q` is applied during insertion. Post-scan, routes are
        additionally filtered by annotation confidence, which recomputes
        route_confidence from reconstructed step headways and is authoritative.

        Cost: O(c log p) for the scan (c = connections in the window,
        p = profile size per stop) plus O(R * L) for reconstructing R routes of
        length L. No per-step re-scanning.
        """
        self._require_prepared()
        t_total = time.perf_counter()
        INF = float("inf")

        day = self._day_from_travel_date(travel_date)
        deadline_secs = self._deadline_to_relative_secs(travel_date, arrival_deadline)
        max_walk_m = self.settings.max_walk_m if max_walk_m is None else max_walk_m
        earliest_dep = max(0, deadline_secs - search_window_minutes * 60)
        min_transfer = self.min_transfer_secs
        footpaths = self.footpaths
        prof_insert = self._prof_insert

        confidence_q = normalize_confidence_q(confidence_q) if confidence_q is not None else 0.75
        log_q = math.log(confidence_q) if confidence_q > 0.0 else LOG_ZERO
        _p_leq = self._p_leq_headway
        levels = self.delay_quantile_levels

        cols = self.conn[day]
        dep_stop_a  = cols["dep_stop"]
        arr_stop_a  = cols["arr_stop"]
        dep_secs_a  = cols["dep_secs"]
        arr_adj_a   = cols["arr_adj"]
        trip_idx_a  = cols["trip_idx"]
        quantiles_a = cols.get("quantiles")   # None when no travel_date was given to prepare()
        dep_secs_list = self.dep_secs_by_day[day]

        start_idx = bisect_left(dep_secs_list, earliest_dep)
        end_idx = bisect_right(dep_secs_list, deadline_secs) - 1
        if end_idx < start_idx:
            return []

        # D[stop] = walk seconds from stop to destination
        D = {end_stop_id: 0}
        for nb, wsecs, dist in footpaths.get(end_stop_id, ()):
            if dist > max_walk_m:
                break
            if wsecs < D.get(nb, INF):
                D[nb] = wsecs

        # Per-stop Pareto profiles:
        #   pdep[stop] = sorted departure times
        #   pent[stop] = (arrival, board, exit, kind, log_conf)
        pdep: dict[int, list[int]] = {}
        pent: dict[int, list[tuple]] = {}

        # Trip labels: best arrival, exit connection, exit kind, accumulated log-confidence
        T_arr      = [INF] * self.n_trips
        T_exit     = [-1]  * self.n_trips
        T_kind     = bytearray(self.n_trips)
        T_log_conf = [0.0] * self.n_trips  # log(1.0) = 0.0

        n_scanned = 0
        i = end_idx
        while i >= start_idx:
            ti     = trip_idx_a[i]
            a_stop = arr_stop_a[i]
            a_adj  = arr_adj_a[i]
            n_scanned += 1
            # Read quantiles once per connection; shared for tau1 and tau3.
            # None when prepare() was called without travel_date → optimistic (lc=0.0).
            q_i = quantiles_a[i] if quantiles_a is not None else None

            # tau1: alight and walk to destination
            dwalk = D.get(a_stop, INF)
            if dwalk < INF:
                hw1 = deadline_secs - a_adj - dwalk
                if hw1 >= 0:
                    if q_i is not None:
                        p1  = _p_leq(q_i, levels, hw1)
                        lc1 = math.log(p1) if p1 > 0.0 else LOG_ZERO
                    else:
                        lc1 = 0.0
                else:
                    lc1 = LOG_ZERO
                tau1 = (a_adj + dwalk) if lc1 >= log_q else INF
            else:
                tau1 = INF
                lc1  = LOG_ZERO

            # tau2: stay seated on this trip
            tau2 = T_arr[ti]
            lc2  = T_log_conf[ti]

            # tau3: transfer at arrival stop
            ds = pdep.get(a_stop)
            if ds is None:
                tau3 = INF
                lc3  = LOG_ZERO
            else:
                k = bisect_left(ds, a_adj + min_transfer)
                if k < len(ds):
                    hw3 = ds[k] - a_adj - min_transfer
                    if q_i is not None:
                        p3       = _p_leq(q_i, levels, hw3)
                        lc3_ride = math.log(p3) if p3 > 0.0 else LOG_ZERO
                    else:
                        lc3_ride = 0.0
                    lc3  = lc3_ride + pent[a_stop][k][4]
                    tau3 = pent[a_stop][k][0] if lc3 >= log_q else INF
                else:
                    tau3 = INF
                    lc3  = LOG_ZERO

            # Prefer staying seated to avoid unnecessary transfers;
            # prefer walking to transferring to end the journey sooner
            if tau2 <= tau1 and tau2 <= tau3:
                tau_c = tau2; ex = T_exit[ti]; kind = T_kind[ti]; lc = lc2
            elif tau1 <= tau3:
                tau_c = tau1; ex = i; kind = _KIND_WALK; lc = lc1
            else:
                tau_c = tau3; ex = i; kind = _KIND_XFER; lc = lc3

            # Update trip label only if this path is better AND passes the confidence gate
            if tau_c < T_arr[ti] and lc >= log_q:
                T_arr[ti]      = tau_c
                T_exit[ti]     = ex
                T_kind[ti]     = kind
                T_log_conf[ti] = lc

            if tau_c == INF or tau_c > deadline_secs:
                i -= 1
                continue

            d_stop = dep_stop_a[i]
            d_secs = dep_secs_a[i]
            prof_insert(pdep, pent, d_stop, d_secs, tau_c, i, ex, kind, lc)
            check_early_stop = d_stop == start_stop_id
            for nb, wsecs, dist in footpaths.get(d_stop, ()):
                if dist > max_walk_m:
                    break
                wdep = d_secs - wsecs
                if wdep < earliest_dep:
                    continue
                prof_insert(pdep, pent, nb, wdep, tau_c, i, ex, kind, lc)
                if nb == start_stop_id:
                    check_early_stop = True

            # Early stopping
            if check_early_stop:
                src_dep = pdep.get(start_stop_id)
                if src_dep and len(src_dep) >= max_routes+2 and d_secs < src_dep[0]:
                    break
                

            i -= 1


        # Build the routes and sort them
        routes: list[dict[str, Any]] = []
        src_dep = pdep.get(start_stop_id)
        if src_dep is not None:
            src_ent = pent[start_stop_id]
            order = sorted(range(len(src_dep)), key=lambda j: (-src_dep[j], src_ent[j][0]))
            seen = set()
            for j in order:
                dep_t = src_dep[j]
                arr_t, board, ex, kind, _lc = src_ent[j]
                if dep_t < earliest_dep or arr_t > deadline_secs:
                    continue

                steps = self._reconstruct_route(
                    board, ex, kind,
                    start_stop_id, end_stop_id,
                    dep_stop_a, arr_stop_a, dep_secs_a, arr_adj_a, trip_idx_a,
                    pdep, pent,
                    D, min_transfer,
                )
                if not steps:
                    continue

                total_walk_m = sum(
                    self._footpath_dist_m(s["from_stop"], s["to_stop"])
                    for s in steps if s["type"] == "walk"
                )
                if total_walk_m > max_walk_m:
                    continue
                n_rides = sum(1 for s in steps if s["type"] == "ride")
                departure_secs = steps[0]["departure_secs"]
                arrival_secs   = steps[-1]["arrival_secs"]

                route = {
                    "start_stop":   int(start_stop_id),
                    "end_stop":     int(end_stop_id),
                    "travel_date":  str(travel_date) if travel_date is not None else None,
                    "day":          day,
                    "day_of_week":  day,
                    "departure_secs": departure_secs,
                    "arrival_secs":   arrival_secs,
                    "duration_sec":   max(0, arrival_secs - departure_secs),
                    "n_transfers":    max(0, n_rides - 1),
                    "total_walk_m":   total_walk_m,
                    "steps":          steps,
                    "arrival_deadline_secs": deadline_secs,
                }

                sig = self._route_signature(route)
                if sig in seen:
                    continue
                seen.add(sig)

                self._format_times(route)
                route["arrival_deadline_time"] = self._secs_to_time(deadline_secs)
                route = self._annotate_route_confidence(
                    route,
                    travel_date=travel_date,
                    day=day,
                    deadline_secs=deadline_secs,
                    confidence_q=confidence_q,
                )

                # The scan gates on a per-connection probability estimate;
                # annotation recomputes from reconstructed headways and is more
                # accurate. Drop routes that fail the annotation threshold.
                if route.get("route_confidence", 1.0) < confidence_q:
                    continue

                routes.append(route)
                if len(routes) >= max_routes:
                    break

        elapsed = time.perf_counter() - t_total
        print(
            f"  plan                total={elapsed*1000:.1f}ms  "
            f"scanned={n_scanned}  routes={len(routes)}"
        )
        routes.sort(key=lambda r: (-r.get("departure_secs", 0), r.get("n_transfers", 0), r.get("total_walk_m", 0)))
        return routes[:max_routes]

    # ------------------------------------------------------------------
    # Confidence helper for the hot scan loop
    # ------------------------------------------------------------------

    @staticmethod
    def _p_leq_headway(quantiles: list[float], levels: tuple, headway: float) -> float:
        """Linearly interpolate P(delay ≤ headway) from stored quantile points."""
        for idx, q in enumerate(quantiles):
            if headway <= q:
                if idx == 0:
                    return float(levels[0]) * (headway / q) if q > 0 else 0.0
                lo_q, hi_q = quantiles[idx - 1], q
                lo_p, hi_p = float(levels[idx - 1]), float(levels[idx])
                t = (headway - lo_q) / (hi_q - lo_q) if hi_q > lo_q else 1.0
                return lo_p + t * (hi_p - lo_p)
        return 1.0

    # ------------------------------------------------------------------
    # Route confidence annotation
    # ------------------------------------------------------------------

    def _annotate_route_confidence(
        self,
        route: dict[str, Any],
        travel_date: str | date,
        day: str,
        deadline_secs: int,
        confidence_q: float | None,
    ) -> dict[str, Any]:
        """Annotate a route using the precomputed delay-quantile lookup only."""
        q = normalize_confidence_q(confidence_q)
        steps = route.get("steps", [])
        levels = self.delay_quantile_levels
        quantiles_by_step_id: dict[int, list[float] | None] = {}

        for step in steps:
            if step.get("type") != "ride":
                step["predicted_delay_sec"] = 0.0
                continue

            quantiles = self._delay_quantiles_for_step(step, travel_date, day)
            if quantiles is None:
                quantiles_by_step_id[id(step)] = None
                step["predicted_delay_sec"] = 0.0
                continue

            clean = self._clean_delay_quantiles(quantiles, levels)
            quantiles_by_step_id[id(step)] = clean
            for level, value in zip(levels, clean):
                step[f"p{int(round(level * 100)):02d}_delay_sec"] = value
            step["predicted_delay_sec"] = self._delay_at_confidence(clean, levels, q)

        confidence, probabilities = route_confidence_details_from_steps(
            steps=steps,
            deadline_secs=deadline_secs,
            quantiles_for_step=lambda step: quantiles_by_step_id.get(id(step)),
            levels=levels,
        )

        missed_connection = None
        for row in probabilities:
            if row["headway_sec"] < 0:
                missed_connection = {
                    "step_index": row["step_index"],
                    "headway_sec": row["headway_sec"],
                }
                break

        robust_cursor = None
        for step in steps:
            scheduled_departure = step.get("departure_secs", robust_cursor or 0)
            if step.get("type") == "ride":
                delay = float(step.get("predicted_delay_sec", 0.0) or 0.0)
                robust_cursor = float(step.get("arrival_secs", scheduled_departure)) + delay
            else:
                duration = float(step.get("duration_sec", 0.0) or 0.0)
                start = max(float(scheduled_departure), float(robust_cursor or scheduled_departure))
                robust_cursor = start + duration
            step["robust_arrival_secs"] = robust_cursor
            step["robust_arrival_time"] = self._secs_to_time(robust_cursor)

        route["confidence_q"] = q
        route["arrival_deadline_secs"] = deadline_secs
        route["robust_arrival_secs"] = robust_cursor if robust_cursor is not None else route.get("arrival_secs")
        route["robust_arrival_time"] = self._secs_to_time(route["robust_arrival_secs"])
        route["route_confidence"] = confidence
        route["per_step_probabilities"] = probabilities
        route["missed_connection"] = missed_connection
        route["passes_confidence"] = missed_connection is None and confidence >= q
        return route

    # ------------------------------------------------------------------
    # Baked-quantile helpers
    # ------------------------------------------------------------------

    def _bake_quantiles_for_date(self, travel_date: str) -> None:
        """Fill conn[day]["quantiles"] with per-connection delay quantile lists.

        1. Generate (or reuse) the day-delay JSON file via DayDelayGenerator.
        2. One O(n) pass over conn[day]: key lookup → quantiles_a[i].
        3. Batch XGBoost inference for any misses (combos absent from the file).
        4. Store as conn[day]["quantiles"]; hot loop reads quantiles_a[i] directly.

        After this method every entry in quantiles_a is a list[float] (no Nones)
        when a delay model is available.  When delay_model is None, misses stay
        as None and the hot loop treats them as optimistic (lc = 0.0).
        """
        from src.models.day_delay_generator import DayDelayGenerator
        from src.data.calendar_data import CalendarDataHandler
        from datetime import date as _date

        day = self._day_from_travel_date(travel_date)

        print(_BAR)
        print(f"  BAKE QUANTILES — {travel_date} ({day})")
        print(_BAR)
        t0 = time.perf_counter()

        gen      = DayDelayGenerator(delay_model=self.delay_model, settings=self.settings)
        day_path = Path(gen._day_delay_path(travel_date))

        if not day_path.exists():
            if self.delay_model is None:
                print("  No delay model — cannot generate day-delay file, skipping bake")
                return
            print(f"  Generating day-delay file for {travel_date}...")
            gen.generate(travel_date, self.conn[day], self.trip_meta_by_idx)

        day_lookup = DayDelayGenerator.load_for_date(travel_date, self.settings)
        if not day_lookup:
            print(f"  Day-delay lookup empty for {travel_date}, skipping bake")
            return
        print(f"  Day-delay lookup: {len(day_lookup)} entries")

        arr_stop_a = self.conn[day]["arr_stop"]
        arr_adj_a  = self.conn[day]["arr_adj"]
        trip_idx_a = self.conn[day]["trip_idx"]
        n_conns    = len(arr_stop_a)

        quantiles_a: list = [None] * n_conns
        # (bpuic, line_text_upper, hour, transport) → list of connection indices
        missing: dict[tuple, list[int]] = {}

        for idx in range(n_conns):
            ti        = trip_idx_a[idx]
            meta      = self.trip_meta_by_idx[ti]
            bpuic     = str(arr_stop_a[idx])
            line_text = str(meta.get("line_text") or "UNKNOWN").upper()
            hour      = str(int(arr_adj_a[idx] // 3600 % 24))
            key       = f"{bpuic}__{line_text}__{hour}"
            q_list    = day_lookup.get(key)
            quantiles_a[idx] = q_list
            if q_list is None:
                combo = (bpuic, line_text, hour, str(meta.get("transport") or ""))
                if combo not in missing:
                    missing[combo] = []
                missing[combo].append(idx)

        n_miss = sum(len(v) for v in missing.values())
        print(
            f"  {n_conns} connections: {n_conns - n_miss} hits, "
            f"{n_miss} misses ({len(missing)} unique combos)"
        )

        if missing and self.delay_model is not None:
            d     = _date.fromisoformat(travel_date)
            dow   = _WEEKDAY_TO_SPARK_DOW[d.weekday()]
            month = d.month
            cal   = CalendarDataHandler.get_features_for_date(travel_date, canton_code="ZH")

            unique_combos = list(missing.keys())
            feature_rows  = [
                self.delay_model._build_feature_row(
                    bpuic=bpuic, line_text=lt, transport=transport,
                    hour=int(hour), day_of_week=dow, month=month,
                    calendar_features=cal,
                )
                for bpuic, lt, hour, transport in unique_combos
            ]

            t_xgb    = time.perf_counter()
            all_preds = self.delay_model._predict_xgb(feature_rows)
            q_levels  = sorted(all_preds.keys())
            print(
                f"  XGBoost fallback: {len(feature_rows)} combos in "
                f"{time.perf_counter() - t_xgb:.1f}s"
            )

            for j, combo in enumerate(unique_combos):
                vals   = sorted(all_preds[q][j] for q in q_levels)
                q_list = [round(v, 2) for v in vals]
                for idx in missing[combo]:
                    quantiles_a[idx] = q_list

        elif missing:
            print(f"  No delay model — {len(missing)} unique combos remain None (optimistic)")

        self.conn[day]["quantiles"] = quantiles_a
        # Keep the day-delay lookup so _delay_quantiles_for_step can use the
        # 3-field key for post-scan confidence annotation without a second lookup layer.
        self.day_delay_lookup = day_lookup

        # Lightweight distribution summary — tells us what the model sees for this day
        covered = [q for q in quantiles_a if q is not None]
        n_cov = len(covered)
        if n_cov:
            levels_t = self.delay_quantile_levels  # e.g. (0.5, 0.6, ..., 0.95)
            p50_idx  = next((i for i, l in enumerate(levels_t) if abs(l - 0.50) < 0.01), 0)
            p90_idx  = next((i for i, l in enumerate(levels_t) if abs(l - 0.90) < 0.01), min(5, len(levels_t) - 1))
            p50s = [q[p50_idx] for q in covered if len(q) > p50_idx]
            p90s = [q[p90_idx] for q in covered if len(q) > p90_idx]
            n    = len(p50s) or 1
            bkt  = [
                sum(1 for v in p50s if v < 30),
                sum(1 for v in p50s if 30 <= v < 60),
                sum(1 for v in p50s if 60 <= v < 120),
                sum(1 for v in p50s if 120 <= v < 300),
                sum(1 for v in p50s if v >= 300),
            ]
            med_p50  = sorted(p50s)[len(p50s) // 2]
            mean_p90 = sum(p90s) / len(p90s) if p90s else 0.0
            print(
                f"  Coverage: {n_cov}/{n_conns} ({100 * n_cov / n_conns:.1f}%)"
            )
            print(
                f"  p50 buckets:  <30s {100*bkt[0]/n:.0f}%  "
                f"30-60s {100*bkt[1]/n:.0f}%  "
                f"1-2min {100*bkt[2]/n:.0f}%  "
                f"2-5min {100*bkt[3]/n:.0f}%  "
                f"5min+  {100*bkt[4]/n:.0f}%"
            )
            print(f"  Median p50={med_p50:.0f}s  Mean p90={mean_p90:.0f}s")

        print(f"  Bake complete in {time.perf_counter() - t0:.1f}s")
        print(_BAR + "\n")

    # ------------------------------------------------------------------
    # Delay quantile lookup helpers
    # ------------------------------------------------------------------

    def _delay_quantiles_for_step(
        self,
        step: dict,
        travel_date: str | date,
        day: str,
    ) -> list[float] | None:
        """Return saved delay quantiles for one ride step, or None on lookup miss.

        Tries the baked 3-field day-delay lookup first (always correctly keyed,
        covers every connection in the timetable), then falls back to the 6-field
        per-district precomputed lookup for cases where baking was not done.
        """
        arrival_secs = step.get("arrival_secs")
        hour = int(float(arrival_secs) // 3600 % 24) if isinstance(arrival_secs, (int, float)) else 0

        if self.day_delay_lookup:
            bpuic3 = str(step.get("to_stop") if step.get("to_stop") is not None else "UNKNOWN")
            lt3    = str(step.get("line_text") or "UNKNOWN").upper()
            key3   = f"{bpuic3}__{lt3}__{hour}"
            result = self.day_delay_lookup.get(key3)
            if result is not None:
                return result

        if not self.delay_lookup:
            return None

        values = {
            "bpuic": str(step.get("to_stop") if step.get("to_stop") is not None else "UNKNOWN"),
            "line_text": str(step.get("line_text") or "UNKNOWN").upper(),
            "hour": str(hour),
            "day_of_week": str(_SPARK_DOW.get(day, 0)),
            "month": str(self._month_from_travel_date(travel_date)),
            "day_type": "0",
        }
        key = "__".join(values.get(col, "UNKNOWN") for col in self.delay_categorical_cols)
        exact = self.delay_lookup.get(key)
        if exact is not None:
            return exact

        if "day_type" in self.delay_categorical_cols:
            parts = [values.get(col, "UNKNOWN") for col in self.delay_categorical_cols]
            day_type_idx = self.delay_categorical_cols.index("day_type")
            for day_type in ("1", "2"):
                parts[day_type_idx] = day_type
                quantiles = self.delay_lookup.get("__".join(parts))
                if quantiles is not None:
                    return quantiles

        for cols, lookup in zip(self.delay_fallback_cols, self.delay_fallback_lookups):
            fallback_key = tuple(values.get(col, "UNKNOWN") for col in cols)
            quantiles = lookup.get(fallback_key)
            if quantiles is not None:
                return quantiles
        return None

    def _build_delay_fallback_lookups(self) -> list[dict[tuple[str, ...], tuple[float, ...]]]:
        """Build coarse lookup indexes from the same precomputed artifact."""
        if not self.delay_lookup:
            return []

        levels = self.delay_quantile_levels
        n_levels = len(levels)
        cat_cols = tuple(self.delay_categorical_cols)
        col_pos = {col: idx for idx, col in enumerate(cat_cols)}
        aggregators: list[dict[tuple[str, ...], list[Any]]] = [
            {} for _ in self.delay_fallback_cols
        ]

        for raw_key, raw_quantiles in self.delay_lookup.items():
            parts = str(raw_key).split("__")
            if len(parts) < len(cat_cols):
                continue
            quantiles = self._clean_delay_quantiles(raw_quantiles, levels)

            for cols, agg in zip(self.delay_fallback_cols, aggregators):
                try:
                    key = tuple(parts[col_pos[col]] for col in cols)
                except KeyError:
                    continue

                bucket = agg.get(key)
                if bucket is None:
                    agg[key] = [1, quantiles[:n_levels]]
                    continue

                bucket[0] += 1
                sums = bucket[1]
                for idx, value in enumerate(quantiles[:n_levels]):
                    sums[idx] += value

        lookups: list[dict[tuple[str, ...], tuple[float, ...]]] = []
        for agg in aggregators:
            lookup: dict[tuple[str, ...], tuple[float, ...]] = {}
            for key, (count, sums) in agg.items():
                if count:
                    lookup[key] = tuple(value / count for value in sums)
            lookups.append(lookup)
        return lookups

    @staticmethod
    def _clean_delay_quantiles(quantiles: Iterable[Any], levels: Iterable[float]) -> list[float]:
        """Convert saved quantiles to non-negative, monotone seconds."""
        n = len(tuple(levels))
        clean: list[float] = []
        for raw in quantiles:
            try:
                value = max(0.0, float(raw))
            except (TypeError, ValueError):
                value = 0.0
            if value != value:
                value = 0.0
            clean.append(value)
            if len(clean) >= n:
                break
        if len(clean) < n:
            clean.extend([0.0] * (n - len(clean)))
        return sorted(clean)

    @staticmethod
    def _delay_at_confidence(quantiles: list[float], levels: Iterable[float], confidence_q: float) -> float:
        """Pick the stored delay quantile at the smallest available level >= q."""
        levels_tuple = tuple(levels)
        for idx, level in enumerate(levels_tuple):
            if confidence_q <= float(level):
                return quantiles[idx] if idx < len(quantiles) else 0.0
        return quantiles[-1] if quantiles else 0.0

    @staticmethod
    def _month_from_travel_date(travel_date: str | date) -> int:
        if isinstance(travel_date, date):
            return int(travel_date.month)
        text = str(travel_date).strip().lower()
        if text in DAY_NAMES:
            return 0
        try:
            return int(datetime.fromisoformat(text).month)
        except ValueError:
            return 0

    # ------------------------------------------------------------------
    # Pareto route reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_route(
        self,
        board: int,
        ex: int,
        kind: int,
        start_stop_id: int,
        end_stop_id: int,
        dep_stop_a: list[int],
        arr_stop_a: list[int],
        dep_secs_a: list[int],
        arr_adj_a: list[int],
        trip_idx_a: list[int],
        pdep: dict[int, list[int]],
        pent: dict[int, list[tuple]],
        D: dict[int, int],
        min_transfer: int,
    ) -> list[dict] | None:
        """Reconstruct a complete route from Pareto profile pointers."""
        steps = []
        cur = start_stop_id
        for _ in range(256):
            bstop = dep_stop_a[board]
            bsecs = dep_secs_a[board]

            if cur != bstop:
                wsecs = self._walk_secs(cur, bstop)
                if wsecs is None:
                    return None
                steps.append({
                    "type": "walk",
                    "from_stop": int(cur), "to_stop": int(bstop),
                    "departure_secs": int(bsecs - wsecs), "arrival_secs": int(bsecs),
                    "duration_sec": int(wsecs),
                })
                cur = bstop

            ti   = trip_idx_a[board]
            meta = self.trip_meta_by_idx[ti]

            a_stop = arr_stop_a[ex]
            a_secs = arr_adj_a[ex]

            steps.append({
                "type": "ride",
                "trip_id": meta["trip_id"],
                "line_text": meta.get("line_text"),
                "operator_id": meta.get("operator_id"),
                "transport": meta.get("transport"),
                "from_stop": int(bstop),
                "to_stop": int(a_stop),
                "departure_secs": int(bsecs),
                "arrival_secs": int(a_secs),
                "duration_sec": int(max(0, a_secs - bsecs)),
            })
            cur = a_stop

            if kind == _KIND_WALK:
                if cur != end_stop_id:
                    wsecs = D.get(cur)
                    if wsecs is None:
                        return None
                    steps.append({
                        "type": "walk",
                        "from_stop": int(cur), "to_stop": int(end_stop_id),
                        "departure_secs": int(a_secs), "arrival_secs": int(a_secs + wsecs),
                        "duration_sec": int(wsecs),
                    })
                    cur = end_stop_id
                return steps

            t_ready = a_secs + min_transfer
            ds = pdep.get(cur)
            if ds is None:
                return None
            k = bisect_left(ds, t_ready)
            if k >= len(ds):
                return None
            _, board, ex, kind, _lc = pent[cur][k]
        return None

    # ------------------------------------------------------------------
    # Profile dominance insert
    # ------------------------------------------------------------------

    @staticmethod
    def _prof_insert(
        pdep: dict[int, list[int]],
        pent: dict[int, list[tuple]],
        stop: int,
        dep: int,
        arr: int | float,
        board: int,
        ex: int,
        kind: int,
        lc: float = 0.0,
    ) -> bool:
        ds = pdep.get(stop)

        if ds is None:
            pdep[stop] = [dep]
            pent[stop] = [(arr, board, ex, kind, lc)]
            return True

        es = pent[stop]
        pl = bisect_left(ds, dep)

        if pl < len(ds) and es[pl][0] <= arr:
            return False

        pr = bisect_right(ds, dep)
        lo = pr

        while lo > 0 and es[lo - 1][0] >= arr:
            lo -= 1

        ds[lo:pr] = [dep]
        es[lo:pr] = [(arr, board, ex, kind, lc)]
        return True

    # ------------------------------------------------------------------
    # Formatting, date, and small utility helpers
    # ------------------------------------------------------------------

    def _format_times(self, route: dict) -> dict:
        """Add human-readable time strings — called only on surviving routes."""
        route["departure_time"] = self._secs_to_time(route["departure_secs"])
        route["arrival_time"] = self._secs_to_time(route["arrival_secs"])
        for step in route["steps"]:
            step["departure_time"] = self._secs_to_time(step["departure_secs"])
            step["arrival_time"] = self._secs_to_time(step["arrival_secs"])
        return route

    def _footpath_dist_m(self, from_stop: int, to_stop: int) -> float:
        """Walk distance in metres (direction-agnostic; footpaths are symmetric)."""
        if from_stop == to_stop:
            return 0.0
        dist = self.footpath_dist.get((from_stop, to_stop))
        if dist is None:
            dist = self.footpath_dist.get((to_stop, from_stop))
        return float(dist) if dist is not None else 0.0

    def _walk_secs(self, from_stop: int, to_stop: int) -> int | None:
        """Walk time in seconds (direction-agnostic)."""
        if from_stop == to_stop:
            return 0
        secs = self.footpath_secs.get((from_stop, to_stop))
        if secs is None:
            secs = self.footpath_secs.get((to_stop, from_stop))
        return secs

    def _require_prepared(self) -> None:
        if not self.prepared:
            raise RuntimeError("RobustJourneyPlanner.prepare() must be called before routing.")

    @staticmethod
    def _row_value(row, name, default=None):
        return getattr(row, name, default)

    @staticmethod
    @lru_cache(maxsize=256)
    def _time_to_secs(time_str: str) -> int:
        parts = str(time_str).split(":")
        if len(parts) == 2:
            h, m = parts
            s = 0
        elif len(parts) == 3:
            h, m, s = parts
        else:
            raise ValueError(f"Invalid time format: {time_str}")
        return int(h) * 3600 + int(m) * 60 + int(s)

    @staticmethod
    @lru_cache(maxsize=2048)
    def _secs_to_time(seconds: int | float) -> str:
        seconds = int(round(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _normalize_day(day: str) -> str:
        day = str(day).strip().lower()
        if day not in DAY_NAMES:
            raise ValueError(f"Invalid day: {day}")
        return day

    @staticmethod
    @lru_cache(maxsize=256)
    def _day_from_travel_date(travel_date: str | date) -> str:
        if isinstance(travel_date, date):
            return DAY_NAMES[travel_date.weekday()]
        text = str(travel_date).strip().lower()
        if text in DAY_NAMES:
            return text
        parsed = datetime.fromisoformat(text).date()
        return DAY_NAMES[parsed.weekday()]

    def _deadline_to_relative_secs(self, travel_date: str | date, arrival_deadline: str) -> int:
        deadline = str(arrival_deadline).strip()
        if "T" in deadline or ("-" in deadline and ":" in deadline):
            travel_day = self._parse_date(travel_date)
            deadline_dt = datetime.fromisoformat(deadline)
            return int((deadline_dt.date() - travel_day).days * 86400) + self._time_to_secs(
                deadline_dt.time().isoformat()
            )
        return self._time_to_secs(deadline)

    @staticmethod
    def _parse_date(value: str | date) -> date:
        if isinstance(value, date):
            return value
        text = str(value).strip().lower()
        if text in DAY_NAMES:
            return datetime.today().date()
        return datetime.fromisoformat(text).date()

    @staticmethod
    def _route_signature(route: dict[str, Any]) -> tuple:
        return tuple(
            (
                step.get("type"),
                step.get("from_stop"),
                step.get("to_stop"),
                step.get("trip_id"),
                step.get("departure_secs"),
                step.get("arrival_secs"),
            )
            for step in route.get("steps", [])
        )
