from __future__ import annotations

"""Scheduled journey planning with the profile Connection Scan Algorithm (pCSA).

`plan_candidates()` now uses the *profile* CSA scan (the logic that used to live
in `plan_profile`) but is extended with journey pointers so it can reconstruct
real `steps` (rides + walks, transfers, walk distances) — i.e. it returns the
exact same dict shape the old backward-CSA `plan_candidates` produced.

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
"""

import time
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Iterable
from functools import lru_cache
from cachetools import TTLCache, cached

_BAR = "=" * 44
_THIN_BAR = "-" * 44

from src.config.settings import ProjectSettings, get_settings
from src.data.csa_data_handler import CSADataHandler


DAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

# exit-leg kinds stored in the trip label / profile payloads
_KIND_NONE = 0
_KIND_WALK = 1   # alight and walk to the target
_KIND_XFER = 2   # alight and transfer to another connection


class JourneyPlanner:
    """Profile-CSA scheduled journey planner with full step reconstruction."""

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
        self.delay_model = None
        self.confidence_evaluator = None

        self.prepared = False
        self.min_transfer_secs = 120

    def configure_robustness(self, delay_model=None, confidence_evaluator=None) -> "JourneyPlanner":
        """Attach robust-routing helpers loaded by RobustJourneyPlanner.prepare()."""
        self.delay_model = delay_model
        self.confidence_evaluator = confidence_evaluator
        self._route_cache.clear()
        return self


    def prepare(
        self,
        regions: Iterable[str] | None = None,
        rebuild: bool = False,
        rebuild_prerequisites: bool = True,
    ) -> "JourneyPlanner":
        """Build/load CSA data and materialize in-memory routing structures."""
        t_total = time.perf_counter()
        regions = tuple(regions if regions is not None else self.settings.region_uuids)

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

        self.min_transfer_secs = int(self.settings.walking_transfer_base_sec)
        self.prepared = True
        return self

    _route_cache = TTLCache(maxsize=512, ttl=300)
    
    
    
    @cached(
        _route_cache,
        key=lambda self,
                start_stop_id,
                end_stop_id,
                travel_date,
                arrival_deadline,
                max_routes=5,
                max_walk_m=None,
                max_probe_window_minutes=400,
                candidate_lookback_minutes=20,
                confidence_q=None: (
            id(self),
            start_stop_id,
            end_stop_id,
            str(travel_date),
            arrival_deadline,
            max_routes,
            max_walk_m if max_walk_m is not None else self.settings.max_walk_m,
            max_probe_window_minutes,
            candidate_lookback_minutes,
            confidence_q,
            id(self.delay_model) if confidence_q is not None else None,
            id(self.confidence_evaluator) if confidence_q is not None else None,
        )
    )
    def plan_candidates(
        self,
        start_stop_id: int,
        end_stop_id: int,
        travel_date: str | date,
        arrival_deadline: str,
        max_routes: int = 5,
        max_walk_m: float | None = None,
        max_probe_window_minutes: int = 400,
        candidate_lookback_minutes: int = 20,
        confidence_q: float | None = None,
    ) -> list[dict[str, Any]]:
        self._require_prepared()

        day = self._day_from_travel_date(travel_date)
        deadline_secs = self._deadline_to_relative_secs(travel_date, arrival_deadline)
        max_walk_m = self.settings.max_walk_m if max_walk_m is None else max_walk_m

        probe_earliest_dep = max(0, deadline_secs - max_probe_window_minutes * 60)

        seed_route = None
        if confidence_q is not None and self.delay_model is not None and self.confidence_evaluator is not None:
            seed_route = self._quick_backward_robust_seed(
                start_stop_id=start_stop_id,
                end_stop_id=end_stop_id,
                travel_date=travel_date,
                day=day,
                deadline_secs=deadline_secs,
                max_walk_m=max_walk_m,
                earliest_dep=probe_earliest_dep,
                confidence_q=confidence_q,
            )
            found_dep = seed_route["departure_secs"] if seed_route is not None else None
        else:
            found_dep = self._quick_backward_scan(
                start_stop_id=start_stop_id,
                end_stop_id=end_stop_id,
                day=day,
                deadline_secs=deadline_secs,
                max_walk_m=max_walk_m,
                earliest_dep=probe_earliest_dep,
            )

        if found_dep is None:
            return []

        # Search only near the latest feasible departure.
        window_start = max(0, found_dep - candidate_lookback_minutes * 60)
        window_minutes = max(1, (deadline_secs - window_start + 59) // 60)

        routes = self.plan_candidates_in(
            start_stop_id=start_stop_id,
            end_stop_id=end_stop_id,
            travel_date=travel_date,
            arrival_deadline=arrival_deadline,
            max_routes=max_routes,
            max_walk_m=max_walk_m,
            search_window_minutes=window_minutes,
        )

        if seed_route is not None:
            seed_sig = self._route_signature(seed_route)
            if all(self._route_signature(route) != seed_sig for route in routes):
                if len(routes) >= max_routes:
                    routes = routes[:max(0, max_routes - 1)]
                routes.append(seed_route)
            routes.sort(key=lambda r: -r.get("departure_secs", 0))
            return routes

        return routes
    
    
    def _quick_backward_scan(
        self,
        start_stop_id: int,
        end_stop_id: int,
        day: str,
        deadline_secs: int,
        max_walk_m: float,
        earliest_dep: int = 0,
    ) -> int | None:
        """Fast latest-departure feasibility probe.

        Returns the latest departure from start_stop_id that can reach end_stop_id
        by deadline_secs using at least one vehicle leg.

        No pure walking start->end route is considered.
        """
        INF = float("inf")

        footpaths = self.footpaths
        min_xfer = self.min_transfer_secs

        cols = self.conn[day]
        dep_stop_a = cols["dep_stop"]
        arr_stop_a = cols["arr_stop"]
        dep_secs_a = cols["dep_secs"]
        arr_adj_a = cols["arr_adj"]
        trip_idx_a = cols["trip_idx"]
        dep_secs_list = self.dep_secs_by_day[day]

        end_idx = bisect_right(dep_secs_list, deadline_secs) - 1
        start_idx = bisect_left(dep_secs_list, earliest_dep)

        if end_idx < start_idx:
            return None

        # D[stop] = final walking seconds from an alighting stop to destination.
        # This is only used AFTER a vehicle leg, so it does not create pure walking.
        D = {end_stop_id: 0}

        for nb, wsecs, dist in footpaths.get(end_stop_id, ()):
            if dist is not None and dist > max_walk_m:
                break  # safe because prepare() sorts footpaths by distance
            if wsecs < D.get(nb, INF):
                D[nb] = wsecs

        # departures[stop] = latest time we can be ready at this stop and still
        # reach the target through at least one future vehicle leg.
        #
        # IMPORTANT:
        # Do NOT seed departures[end_stop_id] = deadline_secs.
        # That would allow pure walking chains to look like a valid transit journey.
        departures: dict[int, float] = defaultdict(lambda: -1.0)

        trip_reachable = bytearray(self.n_trips)

        for i in range(end_idx, start_idx - 1, -1):
            if departures[start_stop_id] >= 0.0:
                break

            ti = trip_idx_a[i]
            a_stop = arr_stop_a[i]
            a_adj = arr_adj_a[i]
            d_stop = dep_stop_a[i]
            d_secs = dep_secs_a[i]

            # Case 1: ride this connection, then finish by walking to destination.
            # No extra min_transfer here. The final walking time itself is in D.
            final_walk = D.get(a_stop, INF)
            can_finish_by_final_walk = (
                final_walk < INF
                and a_adj + final_walk <= deadline_secs
            )

            # Case 2: ride this connection, then transfer to a later reachable vehicle.
            # This DOES require min transfer time.
            can_transfer_to_later_leg = (
                departures[a_stop] >= 0.0
                and a_adj + min_xfer <= departures[a_stop]
            )

            if can_finish_by_final_walk or can_transfer_to_later_leg:
                trip_reachable[ti] = 1

            if trip_reachable[ti] and d_secs > departures[d_stop]:
                departures[d_stop] = float(d_secs)

                # Initial/transfer walking INTO the boarding stop.
                # This includes your 2-min walking base because wsecs comes from stop_to_stop.
                for nb, wsecs, dist in footpaths.get(d_stop, ()):
                    if dist is not None and dist > max_walk_m:
                        break
                    wdep = d_secs - wsecs
                    if wdep >= earliest_dep and wdep > departures[nb]:
                        departures[nb] = float(wdep)

        dep = departures[start_stop_id]
        return int(dep) if dep >= 0.0 else None

    def _quick_backward_robust_seed(
        self,
        start_stop_id: int,
        end_stop_id: int,
        travel_date: str | date,
        day: str,
        deadline_secs: int,
        max_walk_m: float,
        earliest_dep: int,
        confidence_q: float,
    ) -> dict | None:
        """Find one late-departing candidate that already passes the confidence check.

        This is intentionally a seed probe, not the full robust Pareto scan. It keeps
        the scheduled quick-scan state compact, evaluates complete start-stop routes
        as soon as they become reachable, and continues backward until no later
        passenger departure can beat the best passing seed found so far.
        """
        INF = float("inf")

        footpaths = self.footpaths
        min_xfer = self.min_transfer_secs

        cols = self.conn[day]
        dep_stop_a = cols["dep_stop"]
        arr_stop_a = cols["arr_stop"]
        dep_secs_a = cols["dep_secs"]
        arr_adj_a = cols["arr_adj"]
        trip_idx_a = cols["trip_idx"]
        dep_secs_list = self.dep_secs_by_day[day]

        end_idx = bisect_right(dep_secs_list, deadline_secs) - 1
        start_idx = bisect_left(dep_secs_list, earliest_dep)
        if end_idx < start_idx:
            return None

        D = {end_stop_id: 0}
        for nb, wsecs, dist in footpaths.get(end_stop_id, ()):
            if dist is not None and dist > max_walk_m:
                break
            if wsecs < D.get(nb, INF):
                D[nb] = wsecs

        departures: dict[int, float] = defaultdict(lambda: -1.0)
        stop_state: dict[int, tuple[int, int, int, int]] = {}

        trip_reachable = bytearray(self.n_trips)
        trip_exit = [-1] * self.n_trips
        trip_kind = bytearray(self.n_trips)

        best_seed = None
        best_departure = -1

        for i in range(end_idx, start_idx - 1, -1):
            d_secs = dep_secs_a[i]
            if best_seed is not None and d_secs < best_departure:
                break

            ti = trip_idx_a[i]
            a_stop = arr_stop_a[i]
            a_adj = arr_adj_a[i]
            d_stop = dep_stop_a[i]

            final_walk = D.get(a_stop, INF)
            can_finish_by_final_walk = (
                final_walk < INF
                and a_adj + final_walk <= deadline_secs
            )
            can_transfer_to_later_leg = (
                departures[a_stop] >= 0.0
                and a_adj + min_xfer <= departures[a_stop]
            )

            if not trip_reachable[ti]:
                if can_finish_by_final_walk:
                    trip_reachable[ti] = 1
                    trip_exit[ti] = i
                    trip_kind[ti] = _KIND_WALK
                elif can_transfer_to_later_leg:
                    trip_reachable[ti] = 1
                    trip_exit[ti] = i
                    trip_kind[ti] = _KIND_XFER

            if not trip_reachable[ti]:
                continue

            ex = trip_exit[ti]
            kind = int(trip_kind[ti])
            if ex < 0:
                continue

            if d_stop == start_stop_id and d_secs >= earliest_dep:
                seed = self._evaluate_quick_seed_route(
                    board=i,
                    ex=ex,
                    kind=kind,
                    start_stop_id=start_stop_id,
                    end_stop_id=end_stop_id,
                    travel_date=travel_date,
                    day=day,
                    deadline_secs=deadline_secs,
                    max_walk_m=max_walk_m,
                    dep_stop_a=dep_stop_a,
                    arr_stop_a=arr_stop_a,
                    dep_secs_a=dep_secs_a,
                    arr_adj_a=arr_adj_a,
                    trip_idx_a=trip_idx_a,
                    D=D,
                    stop_state=stop_state,
                    min_transfer=min_xfer,
                    confidence_q=confidence_q,
                )
                if seed is not None and seed["departure_secs"] > best_departure:
                    best_seed = seed
                    best_departure = seed["departure_secs"]

            state = (int(d_secs), i, ex, kind)
            if d_secs > departures[d_stop]:
                departures[d_stop] = float(d_secs)
                stop_state[d_stop] = state

            for nb, wsecs, dist in footpaths.get(d_stop, ()):
                if dist is not None and dist > max_walk_m:
                    break
                wdep = d_secs - wsecs
                if wdep < earliest_dep:
                    continue
                if nb == start_stop_id:
                    seed = self._evaluate_quick_seed_route(
                        board=i,
                        ex=ex,
                        kind=kind,
                        start_stop_id=start_stop_id,
                        end_stop_id=end_stop_id,
                        travel_date=travel_date,
                        day=day,
                        deadline_secs=deadline_secs,
                        max_walk_m=max_walk_m,
                        dep_stop_a=dep_stop_a,
                        arr_stop_a=arr_stop_a,
                        dep_secs_a=dep_secs_a,
                        arr_adj_a=arr_adj_a,
                        trip_idx_a=trip_idx_a,
                        D=D,
                        stop_state=stop_state,
                        min_transfer=min_xfer,
                        confidence_q=confidence_q,
                    )
                    if seed is not None and seed["departure_secs"] > best_departure:
                        best_seed = seed
                        best_departure = seed["departure_secs"]
                if wdep > departures[nb]:
                    departures[nb] = float(wdep)
                    stop_state[nb] = (int(wdep), i, ex, kind)

        return best_seed

    def _evaluate_quick_seed_route(
        self,
        board: int,
        ex: int,
        kind: int,
        start_stop_id: int,
        end_stop_id: int,
        travel_date: str | date,
        day: str,
        deadline_secs: int,
        max_walk_m: float,
        dep_stop_a: list[int],
        arr_stop_a: list[int],
        dep_secs_a: list[int],
        arr_adj_a: list[int],
        trip_idx_a: list[int],
        D: dict[int, int],
        stop_state: dict[int, tuple[int, int, int, int]],
        min_transfer: int,
        confidence_q: float,
    ) -> dict | None:
        steps = self._reconstruct_quick_route(
            board=board,
            ex=ex,
            kind=kind,
            start_stop_id=start_stop_id,
            end_stop_id=end_stop_id,
            dep_stop_a=dep_stop_a,
            arr_stop_a=arr_stop_a,
            dep_secs_a=dep_secs_a,
            arr_adj_a=arr_adj_a,
            trip_idx_a=trip_idx_a,
            D=D,
            stop_state=stop_state,
            min_transfer=min_transfer,
        )
        route = self._route_from_steps(
            steps=steps,
            start_stop_id=start_stop_id,
            end_stop_id=end_stop_id,
            travel_date=travel_date,
            day=day,
            deadline_secs=deadline_secs,
            max_walk_m=max_walk_m,
        )
        if route is None:
            return None

        delayed_route = self.delay_model.add_delays_to_route(route, q=confidence_q)
        evaluated_route = self.confidence_evaluator.route_passes(
            delayed_route,
            arrival_deadline=deadline_secs,
            confidence_q=confidence_q,
        )
        return evaluated_route if evaluated_route.get("passes_confidence") else None

    def _reconstruct_quick_route(
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
        D: dict[int, int],
        stop_state: dict[int, tuple[int, int, int, int]],
        min_transfer: int,
    ) -> list[dict] | None:
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

            ti = trip_idx_a[board]
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
                return steps

            t_ready = a_secs + min_transfer
            state = stop_state.get(cur)
            if state is None or state[0] < t_ready:
                return None
            _, board, ex, kind = state
        return None
            
        
    
    
    
    
        
    def plan_candidates_in(
        self,
        start_stop_id: int,
        end_stop_id: int,
        travel_date: str | date,
        arrival_deadline: str,
        max_routes: int = 5,
        max_walk_m: float | None = None,
        search_window_minutes: int = 180,
    ) -> list[dict[str, Any]]:
        """Return scheduled route dictionaries sorted by latest departure first.

        Runs a single bounded backward *profile* CSA scan. One scan produces the
        whole Pareto front of (departure, arrival) pairs at `start_stop_id`; we
        then reconstruct the top `max_routes` of them into full step dictionaries
        — identical in shape to the old `plan_candidates` output.

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

        cols = self.conn[day]
        dep_stop_a = cols["dep_stop"]
        arr_stop_a = cols["arr_stop"]
        dep_secs_a = cols["dep_secs"]
        arr_adj_a = cols["arr_adj"]
        trip_idx_a = cols["trip_idx"]
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

        # Per-stop Pareto profiles: pdep[stop] = departures, pent[stop] = (arrival, board, exit, kind)
        pdep: dict[int, list[int]] = {}
        pent: dict[int, list[tuple]] = {}


        # Trip labels: best arrival, exit connection, and exit kind while seated
        T_arr = [INF] * self.n_trips
        T_exit = [-1] * self.n_trips
        T_kind = bytearray(self.n_trips)
        


        n_scanned = 0
        for i in range(end_idx, start_idx - 1, -1):
            ti = trip_idx_a[i]
            a_stop = arr_stop_a[i]
            a_adj = arr_adj_a[i]
            n_scanned += 1

            # Three ways to finish the journey from this connection:
            # 1. Alight and walk to target
            # 2. Stay seated on the trip
            # 3. Transfer at arrival stop
            dwalk = D.get(a_stop, INF)
            tau1 = a_adj + dwalk if dwalk < INF else INF
            tau2 = T_arr[ti]
            ds = pdep.get(a_stop)
            if ds is None:
                tau3 = INF
            else:
                k = bisect_left(ds, a_adj + min_transfer)
                tau3 = pent[a_stop][k][0] if k < len(ds) else INF

            # Prefer staying seated (tau2) to avoid splitting trips;
            # prefer walking (tau1) to transferring (tau3) to end early
            if tau2 <= tau1 and tau2 <= tau3:
                tau_c = tau2; ex = T_exit[ti]; kind = T_kind[ti]
            elif tau1 <= tau3:
                tau_c = tau1; ex = i; kind = _KIND_WALK
            else:
                tau_c = tau3; ex = i; kind = _KIND_XFER

            if tau_c < T_arr[ti]:
                T_arr[ti] = tau_c
                T_exit[ti] = ex
                T_kind[ti] = kind

            if tau_c == INF or tau_c > deadline_secs:
                continue

            # Board at departure stop; nearby stops can walk in too
            d_stop = dep_stop_a[i]
            d_secs = dep_secs_a[i]
            prof_insert(pdep, pent, d_stop, d_secs, tau_c, i, ex, kind)
            for nb, wsecs, dist in footpaths.get(d_stop, ()):
                if dist > max_walk_m:
                    break
                wdep = d_secs - wsecs
                if wdep < earliest_dep:
                    continue
                prof_insert(pdep, pent, nb, wdep, tau_c, i, ex, kind)








        routes: list[dict[str, Any]] = []
        src_dep = pdep.get(start_stop_id)
        if src_dep is not None:
            src_ent = pent[start_stop_id]
            order = sorted(range(len(src_dep)), key=lambda j: (-src_dep[j], src_ent[j][0]))
            seen = set()
            for j in order:
                dep_t = src_dep[j]
                arr_t, board, ex, kind = src_ent[j]
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
                arrival_secs = steps[-1]["arrival_secs"]

                route = {
                    "start_stop": int(start_stop_id),
                    "end_stop": int(end_stop_id),
                    "travel_date": str(travel_date) if travel_date is not None else None,
                    "day": day,
                    "day_of_week": day,
                    "departure_secs": departure_secs,
                    "arrival_secs": arrival_secs,
                    "duration_sec": max(0, arrival_secs - departure_secs),
                    "n_transfers": max(0, n_rides - 1),
                    "total_walk_m": total_walk_m,
                    "steps": steps,
                    "arrival_deadline_secs": deadline_secs,
                }

                sig = self._route_signature(route)
                if sig in seen:
                    continue
                seen.add(sig)

                self._format_times(route)
                route["arrival_deadline_time"] = self._secs_to_time(deadline_secs)
                routes.append(route)
                if len(routes) >= max_routes:
                    break

        elapsed = time.perf_counter() - t_total
        print(
            f"  plan_candidates     total={elapsed*1000:.1f}ms  "
            f"scanned={n_scanned}  routes={len(routes)}"
        )
        return routes














    def _route_from_steps(
        self,
        steps: list[dict] | None,
        start_stop_id: int,
        end_stop_id: int,
        travel_date: str | date,
        day: str,
        deadline_secs: int,
        max_walk_m: float,
    ) -> dict[str, Any] | None:
        """Build the public route dict from reconstructed steps."""
        if not steps:
            return None

        total_walk_m = sum(
            self._footpath_dist_m(s["from_stop"], s["to_stop"])
            for s in steps if s["type"] == "walk"
        )
        if total_walk_m > max_walk_m:
            return None

        n_rides = sum(1 for s in steps if s["type"] == "ride")
        departure_secs = steps[0]["departure_secs"]
        arrival_secs = steps[-1]["arrival_secs"]

        route = {
            "start_stop": int(start_stop_id),
            "end_stop": int(end_stop_id),
            "travel_date": str(travel_date) if travel_date is not None else None,
            "day": day,
            "day_of_week": day,
            "departure_secs": departure_secs,
            "arrival_secs": arrival_secs,
            "duration_sec": max(0, arrival_secs - departure_secs),
            "n_transfers": max(0, n_rides - 1),
            "total_walk_m": total_walk_m,
            "steps": steps,
            "arrival_deadline_secs": deadline_secs,
        }
        self._format_times(route)
        route["arrival_deadline_time"] = self._secs_to_time(deadline_secs)
        return route

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
        """Reconstruct a complete route from Pareto profile pointers.

        Starting from a (board, exit, kind) triple, follows the chain of
        connections and transfer profiles to build walk + ride steps.
        """
        steps = []
        cur = start_stop_id
        for _ in range(256):
            bstop = dep_stop_a[board]
            bsecs = dep_secs_a[board]

            # Walk into boarding stop (initial or transfer)
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

            # Ride from board to exit connection
            ti = trip_idx_a[board]
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
                # Final walk to destination
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

            # Transfer: find next boarding connection from profile
            t_ready = a_secs + min_transfer
            ds = pdep.get(cur)
            if ds is None:
                return None
            k = bisect_left(ds, t_ready)
            if k >= len(ds):
                return None
            _, board, ex, kind = pent[cur][k]
        return None
    
    
    
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
    ) -> bool:
        ds = pdep.get(stop)

        if ds is None:
            pdep[stop] = [dep]
            pent[stop] = [(arr, board, ex, kind)]
            return True

        es = pent[stop]
        pl = bisect_left(ds, dep)

        # Dominated by same/later departure with no worse arrival.
        if pl < len(ds) and es[pl][0] <= arr:
            return False

        pr = bisect_right(ds, dep)
        lo = pr

        while lo > 0 and es[lo - 1][0] >= arr:
            lo -= 1

        ds[lo:pr] = [dep]
        es[lo:pr] = [(arr, board, ex, kind)]
        return True
        
    
    
    
    
    
    

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
            raise RuntimeError("JourneyPlanner.prepare() must be called before routing.")

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
