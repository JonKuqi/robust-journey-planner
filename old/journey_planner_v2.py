from __future__ import annotations

"""Scheduled journey planning with the Connection Scan Algorithm."""

import time
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Iterable, Optional
from functools import lru_cache
from cachetools import TTLCache, cached

_BAR = "=" * 44
_THIN_BAR = "-" * 44

from src.config.settings import ProjectSettings, get_settings
from src.data.csa_data_handler import CSADataHandler


DAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


class JourneyPlanner:
    """CSA-based scheduled journey planner."""

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
        self.stops = None
        self.stop_metadata = {}
        self.trip_id_to_idx = {}
        self.trip_idx_to_id = []
        self.trip_metadata_by_id = {}
        self.n_trips = 0
        self.prepared = False
        self.min_transfer_secs = 120 



    def prepare(
        self,
        regions: Iterable[str] | None = None,
        rebuild: bool = False,
        rebuild_prerequisites: bool = True,
    ) -> "JourneyPlanner":
        """Build/load CSA data and materialize in-memory routing structures."""
        t_total = time.perf_counter()
        regions = tuple(regions if regions is not None else self.settings.region_uuids)

        # ── build Trino tables (prints its own section) ──────────────────────
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

        # ── fetch from Trino (each fetch prints its own line) ────────────────
        print(_BAR)
        print("  FETCH DATA")
        print(_BAR)
        stops_df = self.data_handler.fetch_stops()
        footpaths_df = self.data_handler.fetch_footpaths()
        connections_df = self.data_handler.fetch_connections()
        print(_BAR + "\n")

        # ── materialize in-memory structures ─────────────────────────────────
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
        for row in footpaths_df.itertuples(index=False):
            self.footpaths[int(row.a_stop_id)].append(
                (
                    int(row.b_stop_id),
                    int(round(float(row.walk_time_min) * 60)),
                    float(row.distance),
                )
            )
        self.footpath_dist = {
            (int(row.a_stop_id), int(row.b_stop_id)): float(row.distance)
            for row in footpaths_df.itertuples(index=False)
        }
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
        for day in DAY_NAMES:
            self.connections_by_day[day].sort(key=lambda c: (c[2], c[4], c[0], c[1]))
            self.dep_secs_by_day[day] = [c[2] for c in self.connections_by_day[day]]
        self.n_trips = len(self.trip_idx_to_id)
        print(f"  {'sort connections':<28}  {time.perf_counter() - t:>6.2f}s  ({self.n_trips} trips)")

        print(_THIN_BAR)
        print(f"  {'TOTAL prepare()':<28}  {time.perf_counter() - t_total:>6.2f}s")
        print(_BAR + "\n")
        
        
        self.min_transfer_secs = int(self.settings.walking_transfer_base_sec)

        self.prepared = True
        return self
    
        
        
    def route(
        self,
        start_id: int,
        end_id: int,
        deadline_secs: int,
        departure_cutoff: int,
        connections: list,
        dep_secs_list: list,
        max_walk_m: float,
        earliest_dep: int = 0,
        end_idx: int = 0,
        start_idx: int = 0,
        trip_reachable: bytearray | None = None,
    ) -> tuple[list, int] | None:
        """Return (route_tuples, departure_secs) for the latest departure from start_id
        that reaches end_id by deadline_secs and departs at most at departure_cutoff.

        deadline_secs is the arrival deadline (constant across iterations).
        departure_cutoff limits which departures from start_id are accepted and
        decreases each iteration to enumerate distinct earlier routes.
        """
        departures: dict[int, float] = defaultdict(lambda: -1)
        pred: dict = {}
        

        departures[end_id] = float(deadline_secs)
        self._apply_backward_footpaths(
            departures, pred, end_id, deadline_secs, max_walk_m, start_id, departure_cutoff
        )

        # Scan connections backward by dep_time (descending). This guarantees that
        # departures[arr_stop] is fully resolved before we relay through it — sorting
        # by arr_time would break correctness whenever an overtaking connection exists.
        for i in range(end_idx, start_idx - 1, -1):
            dep_stop, arr_stop, dep_secs, arr_secs, trip_idx, arr_secs_adj, trip_id = connections[i]

            # Stopping criterion: scanning is decreasing by dep_secs, so once
            # start_id has a valid departure no later connection can improve it.
            if departures[start_id] >= 0:
                break

            if arr_secs_adj + self.min_transfer_secs <= departures[arr_stop]:
                trip_reachable[trip_idx] = 1

            if trip_reachable[trip_idx] and dep_secs > departures[dep_stop]:
                if dep_stop == start_id and dep_secs > departure_cutoff:
                    continue
                departures[dep_stop] = float(dep_secs)
                pred[dep_stop] = ("trip", dep_stop, arr_stop, dep_secs, arr_secs_adj, trip_id)
                self._apply_backward_footpaths(
                    departures, pred, dep_stop, dep_secs, max_walk_m, start_id, departure_cutoff
                )

        if departures[start_id] < 0:
            return None

        departure_secs = int(departures[start_id])
        segments = self._reconstruct_backward_route(pred, start_id, end_id)
        if segments is None:
            return None

        return self._segments_to_required_tuples(segments), departure_secs


    _route_cache = TTLCache(maxsize=512, ttl=300)  # 5 min TTL

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
                search_step_minutes=5:
            (
                id(self),
                start_stop_id,
                end_stop_id,
                str(travel_date),
                arrival_deadline,
                max_routes,
                max_walk_m if max_walk_m is not None else self.settings.max_walk_m,
                search_window_minutes,
                search_step_minutes,
            )
    )
    def plan_candidates(
        self,
        start_stop_id: int,
        end_stop_id: int,
        travel_date: str | date,
        arrival_deadline: str,
        max_routes: int = 5,
        max_walk_m: int | None = None,
        search_window_minutes: int = 180,
        search_step_minutes: int = 5,
    ) -> list[dict[str, Any]]:
        """Return scheduled route dictionaries sorted by latest departure first.

        Uses iterative backward CSA: each call finds the latest-departing route
        by deadline, then sets the next deadline to one second before that
        departure to discover earlier alternatives.  O(k * c) total where k =
        max_routes and c = number of connections, versus O(36 * c) for the
        old approach that re-ran forward CSA at every 5-minute step.
        """
        t_total = time.perf_counter()
        day = self._day_from_travel_date(travel_date)
        deadline_secs = self._deadline_to_relative_secs(travel_date, arrival_deadline)
        max_walk_m = self.settings.max_walk_m if max_walk_m is None else max_walk_m
        earliest_dep = max(0, deadline_secs - search_window_minutes * 60)
        connections = self.connections_by_day[day]
        dep_secs_list = self.dep_secs_by_day[day]

        # Hoist out of route() — constant across all iterations since
        # deadline_secs and earliest_dep never change between calls.
        end_idx = bisect_right(dep_secs_list, deadline_secs) - 1
        start_idx = bisect_left(dep_secs_list, earliest_dep)
        if end_idx < 0:
            return []

        # Warm-started across iterations — trips reachable relative to the
        # fixed deadline remain reachable in later iterations, so we never reset.
        trip_reachable = bytearray(self.n_trips)

        seen = set()
        routes = []
        current_deadline = deadline_secs
        departure_cutoff = deadline_secs
        n_backward_calls = 0
        t_backward_total = 0.0

        while len(routes) < max_routes:
            
            trip_reachable = bytearray(self.n_trips)
            
            end_idx = bisect_right(dep_secs_list, current_deadline) - 1
            if end_idx < start_idx:
                break
            
            t_bwd = time.perf_counter()
            result = self.route(
                start_stop_id, end_stop_id, current_deadline, departure_cutoff,
                connections, dep_secs_list, max_walk_m,
                earliest_dep=earliest_dep,
                end_idx=end_idx,
                start_idx=start_idx,
                trip_reachable=trip_reachable,
            )
            t_backward_total += time.perf_counter() - t_bwd
            n_backward_calls += 1

            if result is None:
                break

            tuples, departure_secs = result
            if departure_secs < earliest_dep:
                break

            # No time strings yet — just ints, fast to build and cheap to discard
            route = self.route_tuples_to_dict(
                tuples,
                start_stop_id=start_stop_id,
                end_stop_id=end_stop_id,
                requested_departure_secs=departure_secs,
                travel_date=travel_date,
                day=day,
            )
            if route is None or route["arrival_secs"] > current_deadline:
                departure_cutoff = departure_secs - 1
                continue

            if route["total_walk_m"] > max_walk_m:
                departure_cutoff = departure_secs - 1
                continue

            route["arrival_deadline_secs"] = deadline_secs

            signature = self._route_signature(route)
            if signature not in seen:
                seen.add(signature)
                # Only format times now — route has survived every filter
                self._format_times(route)
                route["arrival_deadline_time"] = self._secs_to_time(deadline_secs)
                routes.append(route)
                # Key fix: force next candidate to arrive earlier
                current_deadline = route["arrival_secs"] - 1

            departure_cutoff = departure_secs - 1

        elapsed = time.perf_counter() - t_total
        avg_bwd = (t_backward_total / n_backward_calls * 1000) if n_backward_calls else 0
        print(
            f"  plan_candidates     total={elapsed*1000:.1f}ms  "
            f"backward_calls={n_backward_calls}  avg_backward={avg_bwd:.1f}ms  "
            f"routes={len(routes)}"
        )
        return routes
        

    def _format_times(self, route: dict) -> dict:
        """Add human-readable time strings — called only on routes that survive all filters."""
        route["departure_time"] = self._secs_to_time(route["departure_secs"])
        route["arrival_time"] = self._secs_to_time(route["arrival_secs"])
        for step in route["steps"]:
            step["departure_time"] = self._secs_to_time(step["departure_secs"])
            step["arrival_time"] = self._secs_to_time(step["arrival_secs"])
        return route


    def route_tuples_to_dict(
        self,
        route_tuples,
        start_stop_id: int,
        end_stop_id: int,
        requested_departure_secs: int,
        travel_date: str | date | None = None,
        day: str | None = None,
    ) -> dict[str, Any] | None:
        """Convert Assignment 1 tuple output into plain route dictionaries."""
        if not route_tuples:
            return None

        steps = []
        cursor_time = requested_departure_secs
        i = 0
        while i < len(route_tuples):
            ts, from_stop, trip_id, to_stop = route_tuples[i]
            ts = int(ts)

            if trip_id is None:
                dep = int(cursor_time)
                steps.append(
                    {
                        "type": "walk",
                        "from_stop": int(from_stop),
                        "to_stop": int(to_stop),
                        "departure_secs": dep,
                        "arrival_secs": ts,
                        "duration_sec": max(0, ts - dep),
                    }
                )
                cursor_time = ts
                i += 1
                continue

            if from_stop is None:
                i += 1
                continue

            board_time = ts
            board_stop = int(from_stop)
            alight_time = None
            alight_stop = None
            j = i + 1
            while j < len(route_tuples):
                next_ts, next_from_stop, next_trip_id, next_to_stop = route_tuples[j]
                if next_trip_id == trip_id and next_from_stop is None:
                    alight_time = int(next_ts)
                    alight_stop = int(next_to_stop)
                    break
                j += 1

            if alight_time is None or alight_stop is None:
                return None

            metadata = self.trip_metadata_by_id.get(str(trip_id), {})
            steps.append(
                {
                    "type": "ride",
                    "trip_id": str(trip_id),
                    "line_text": metadata.get("line_text"),
                    "operator_id": metadata.get("operator_id"),
                    "transport": metadata.get("transport"),
                    "from_stop": board_stop,
                    "to_stop": alight_stop,
                    "departure_secs": board_time,
                    "arrival_secs": alight_time,
                    "duration_sec": max(0, alight_time - board_time),
                }
            )
            cursor_time = alight_time
            i = j + 1

        if not steps:
            return None

        departure_secs = steps[0]["departure_secs"]
        arrival_secs = steps[-1]["arrival_secs"]

        total_walk_m = sum(
            self._footpath_dist_m(s["from_stop"], s["to_stop"])
            for s in steps
            if s["type"] == "walk"
        )

        return {
            "start_stop": int(start_stop_id),
            "end_stop": int(end_stop_id),
            "travel_date": str(travel_date) if travel_date is not None else None,
            "day": day,
            "day_of_week": day,
            "departure_secs": departure_secs,
            "arrival_secs": arrival_secs,
            "duration_sec": max(0, arrival_secs - departure_secs),
            "n_transfers": max(0, sum(1 for s in steps if s["type"] == "ride") - 1),
            "total_walk_m": total_walk_m,
            "steps": steps,
        }

    def _segments_to_required_tuples(self, segments):
        route = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            if seg[0] == "walk":
                _, from_stop, to_stop, arr_time = seg
                route.append((arr_time, from_stop, None, to_stop))
                i += 1
                continue

            _, dep_stop, arr_stop, dep_secs, arr_secs, trip_id = seg
            board_stop = dep_stop
            board_time = dep_secs
            final_arr_stop = arr_stop
            final_arr_time = arr_secs
            i += 1

            while i < len(segments):
                nxt = segments[i]
                if nxt[0] == "trip" and nxt[5] == trip_id and nxt[1] == final_arr_stop:
                    _, _, next_arr_stop, _, next_arr_secs, _ = nxt
                    final_arr_stop = next_arr_stop
                    final_arr_time = next_arr_secs
                    i += 1
                else:
                    break

            route.append((board_time, board_stop, trip_id, None))
            route.append((final_arr_time, None, trip_id, final_arr_stop))

        return route

 

    def _footpath_dist_m(self, from_stop: int, to_stop: int) -> float:
        dist = self.footpath_dist.get((from_stop, to_stop))
        return float(dist) if dist is not None else 0.0
    
    

    def _apply_backward_footpaths(self, departures, pred, from_stop, base_dep_time, max_walk_m,
                                   start_id=None, departure_cutoff=None):
        """Propagate latest departure backward: neighbors that walk TO from_stop in time."""
        for neighbor, walk_secs, distance_m in self.footpaths.get(from_stop, ()):
            if distance_m is not None and distance_m > max_walk_m:
                continue
            candidate = base_dep_time - walk_secs
            if neighbor == start_id and departure_cutoff is not None and candidate > departure_cutoff:
                continue
            if candidate > departures[neighbor]:
                departures[neighbor] = float(candidate)
                pred[neighbor] = ("walk", neighbor, from_stop, base_dep_time)
                
                

    def _reconstruct_backward_route(self, pred, start_id, end_id):
        """Follow pred forward from start_id to end_id, returning ordered segments."""
        segments = []
        cur = start_id
        visited = set()
        while cur != end_id:
            if cur in visited or cur not in pred:
                return None
            visited.add(cur)
            info = pred[cur]
            if info[0] == "trip":
                _, dep_stop, arr_stop, dep_secs, arr_secs, trip_id = info
                segments.append(("trip", dep_stop, arr_stop, dep_secs, arr_secs, trip_id))
                cur = arr_stop
            elif info[0] == "walk":
                _, from_stop, to_stop, arr_secs = info
                segments.append(("walk", from_stop, to_stop, arr_secs))
                cur = to_stop
            else:
                return None
        return segments

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
    @staticmethod
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
            return int((deadline_dt.date() - travel_day).days * 86400) + self._time_to_secs(deadline_dt.time().isoformat())
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


