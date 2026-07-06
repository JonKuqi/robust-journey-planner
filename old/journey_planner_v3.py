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
    
        
        


    @staticmethod
    def _row_value(row, name, default=None):
        return getattr(row, name, default)
    
    def _require_prepared(self) -> None:
        if not self.prepared:
            raise RuntimeError("JourneyPlanner.prepare() must be called before routing.")
        
        
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
            return int((deadline_dt.date() - travel_day).days * 86400) + self._time_to_secs(deadline_dt.time().isoformat())
        return self._time_to_secs(deadline)


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
                search_window_minutes=180:
            (
                id(self),
                int(start_stop_id),
                int(end_stop_id),
                str(travel_date),
                str(arrival_deadline),
                int(max_routes),
                int(max_walk_m if max_walk_m is not None else self.settings.max_walk_m),
                int(search_window_minutes),
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
    ) -> list[dict[str, Any]]:
        """
        Fast profile-based replacement for the old plan_candidates().

        Single backward scan.
        Stores pure-dict predecessor labels.
        Returns same route dict shape as old plan_candidates().
        """
        self._require_prepared()

        t_total = time.perf_counter()
        INF = 10**18

        start_stop_id = int(start_stop_id)
        end_stop_id = int(end_stop_id)

        day = self._day_from_travel_date(travel_date)
        deadline_secs = self._deadline_to_relative_secs(travel_date, arrival_deadline)
        max_walk_m = self.settings.max_walk_m if max_walk_m is None else max_walk_m
        max_walk_m = float(max_walk_m)

        earliest_dep = max(0, deadline_secs - search_window_minutes * 60)

        connections = self.connections_by_day[day]
        dep_secs_list = self.dep_secs_by_day[day]

        start_idx = bisect_left(dep_secs_list, earliest_dep)
        end_idx = bisect_right(dep_secs_list, deadline_secs) - 1

        if end_idx < start_idx:
            return []

        self._ensure_profile_footpaths_to()

        # Profiles:
        # S[stop] = list of labels sorted by dep ascending.
        #
        # label shape:
        # {
        #   "dep": int,
        #   "arr": int,
        #   "kind": "ride" | "walk",
        #   "from_stop": int,
        #   "to_stop": int,
        #   "step_dep": int,
        #   "step_arr": int,
        #   "trip_id": str | None,
        #   "next": label | None,
        # }
        S = defaultdict(list)

        # T[trip_idx] = best known continuation if already seated on this trip.
        #
        # {
        #   "arr": int,
        #   "alight_stop": int,
        #   "alight_time": int,
        #   "next": label | None,
        # }
        T = [None] * self.n_trips

        # Direct final walking labels into target.
        final_walk_to_end = {}

        final_walk_to_end[end_stop_id] = (0, 0.0)

        for neighbor, walk_secs, dist in self.footpaths_to.get(end_stop_id, ()):
            if dist is not None and dist <= max_walk_m:
                final_walk_to_end[int(neighbor)] = (int(walk_secs), float(dist))

        n_scanned = 0
        n_inserted = 0

        for i in range(end_idx, start_idx - 1, -1):
            dep_stop, arr_stop, dep_secs, arr_secs, trip_idx, arr_secs_adj, trip_id = connections[i]

            dep_stop = int(dep_stop)
            arr_stop = int(arr_stop)
            dep_secs = int(dep_secs)
            arr_secs_adj = int(arr_secs_adj)
            trip_idx = int(trip_idx)
            trip_id = str(trip_id)

            n_scanned += 1

            if dep_secs < earliest_dep:
                continue

            best = None

            # Option 1: ride to arr_stop, then walk directly to destination.
            fw = final_walk_to_end.get(arr_stop)
            if fw is not None:
                walk_secs, _dist = fw
                final_arr = arr_secs_adj + int(walk_secs)

                if final_arr <= deadline_secs:
                    if arr_stop == end_stop_id:
                        next_label = None
                    else:
                        next_label = {
                            "dep": arr_secs_adj,
                            "arr": final_arr,
                            "kind": "walk",
                            "from_stop": arr_stop,
                            "to_stop": end_stop_id,
                            "step_dep": arr_secs_adj,
                            "step_arr": final_arr,
                            "trip_id": None,
                            "next": None,
                        }

                    best = {
                        "arr": final_arr,
                        "alight_stop": arr_stop,
                        "alight_time": arr_secs_adj,
                        "next": next_label,
                        "priority": 1,
                    }

            # Option 2: stay seated on same trip.
            trip_state = T[trip_idx]
            if trip_state is not None and trip_state["arr"] <= deadline_secs:
                cand = {
                    "arr": int(trip_state["arr"]),
                    "alight_stop": int(trip_state["alight_stop"]),
                    "alight_time": int(trip_state["alight_time"]),
                    "next": trip_state["next"],
                    "priority": 0,  # prefer staying seated on ties
                }
                if best is None or (cand["arr"], cand["priority"]) < (best["arr"], best["priority"]):
                    best = cand

            # Option 3: ride to arr_stop, then transfer.
            ready_time = arr_secs_adj + self.min_transfer_secs
            transfer_label = self._profile_eval_label(S[arr_stop], ready_time)

            if transfer_label is not None and transfer_label["arr"] <= deadline_secs:
                cand = {
                    "arr": int(transfer_label["arr"]),
                    "alight_stop": arr_stop,
                    "alight_time": arr_secs_adj,
                    "next": transfer_label,
                    "priority": 2,
                }
                if best is None or (cand["arr"], cand["priority"]) < (best["arr"], best["priority"]):
                    best = cand

            if best is None:
                continue

            # Update trip continuation for earlier connections of same trip.
            old_trip_state = T[trip_idx]
            if (
                old_trip_state is None
                or best["arr"] < old_trip_state["arr"]
                or (
                    best["arr"] == old_trip_state["arr"]
                    and best["alight_time"] > old_trip_state["alight_time"]
                )
            ):
                T[trip_idx] = {
                    "arr": int(best["arr"]),
                    "alight_stop": int(best["alight_stop"]),
                    "alight_time": int(best["alight_time"]),
                    "next": best["next"],
                }

            ride_label = {
                "dep": dep_secs,
                "arr": int(best["arr"]),
                "kind": "ride",
                "from_stop": dep_stop,
                "to_stop": int(best["alight_stop"]),
                "step_dep": dep_secs,
                "step_arr": int(best["alight_time"]),
                "trip_id": trip_id,
                "next": best["next"],
            }

            if self._profile_insert_label(S[dep_stop], ride_label):
                n_inserted += 1

            # Walk into this departure stop from nearby stops.
            for neighbor, walk_secs, dist in self.footpaths_to.get(dep_stop, ()):
                if dist is not None and dist > max_walk_m:
                    continue

                neighbor = int(neighbor)
                walk_secs = int(walk_secs)
                walk_dep = dep_secs - walk_secs

                if walk_dep < earliest_dep:
                    continue

                walk_label = {
                    "dep": walk_dep,
                    "arr": int(best["arr"]),
                    "kind": "walk",
                    "from_stop": neighbor,
                    "to_stop": dep_stop,
                    "step_dep": walk_dep,
                    "step_arr": dep_secs,
                    "trip_id": None,
                    "next": ride_label,
                }

                if self._profile_insert_label(S[neighbor], walk_label):
                    n_inserted += 1

        source_profile = sorted(S[start_stop_id], key=lambda x: (-x["dep"], x["arr"]))

        routes = []
        seen = set()

        for label in source_profile:
            if len(routes) >= max_routes:
                break

            if label["dep"] < earliest_dep:
                continue

            if label["arr"] > deadline_secs:
                continue

            steps = self._profile_reconstruct_steps(label)

            if not steps:
                continue

            departure_secs = int(steps[0]["departure_secs"])
            arrival_secs = int(steps[-1]["arrival_secs"])

            total_walk_m = sum(
                self._profile_footpath_dist_m(s["from_stop"], s["to_stop"])
                for s in steps
                if s["type"] == "walk"
            )

            if total_walk_m > max_walk_m:
                continue

            route = {
                "start_stop": start_stop_id,
                "end_stop": end_stop_id,
                "travel_date": str(travel_date) if travel_date is not None else None,
                "day": day,
                "day_of_week": day,
                "departure_secs": departure_secs,
                "arrival_secs": arrival_secs,
                "duration_sec": max(0, arrival_secs - departure_secs),
                "n_transfers": max(0, sum(1 for s in steps if s["type"] == "ride") - 1),
                "total_walk_m": total_walk_m,
                "steps": steps,
                "arrival_deadline_secs": deadline_secs,
            }

            signature = self._route_signature(route)

            if signature in seen:
                continue

            seen.add(signature)

            self._format_times(route)
            route["arrival_deadline_time"] = self._secs_to_time(deadline_secs)

            routes.append(route)

        elapsed = time.perf_counter() - t_total

        print(
            f"  plan_candidates     total={elapsed*1000:.1f}ms  "
            f"scanned={n_scanned}  inserted={n_inserted}  routes={len(routes)}"
        )

        return routes


    def _ensure_profile_footpaths_to(self) -> None:
        """
        Build reverse footpath index lazily.

        self.footpaths:
            from_stop -> [(to_stop, walk_secs, dist)]

        self.footpaths_to:
            to_stop -> [(from_stop, walk_secs, dist)]
        """
        if hasattr(self, "footpaths_to") and self.footpaths_to is not None:
            return

        footpaths_to = defaultdict(list)

        for from_stop, edges in self.footpaths.items():
            from_stop = int(from_stop)

            for to_stop, walk_secs, dist in edges:
                footpaths_to[int(to_stop)].append(
                    (
                        from_stop,
                        int(walk_secs),
                        float(dist) if dist is not None else None,
                    )
                )

        self.footpaths_to = footpaths_to


    def _profile_eval_label(self, profile: list[dict[str, Any]], ready_time: int) -> dict[str, Any] | None:
        """
        Return best label with dep >= ready_time.

        Since profile is Pareto-clean and sorted by dep ascending, the first feasible
        label is enough in the normal case.
        """
        ready_time = int(ready_time)

        for label in profile:
            if label["dep"] >= ready_time:
                return label

        return None


    def _profile_insert_label(self, profile: list[dict[str, Any]], label: dict[str, Any]) -> bool:
        """
        Insert pure-dict label into Pareto profile.

        Dominance:
        - later departure is better
        - earlier arrival is better
        """
        dep = int(label["dep"])
        arr = int(label["arr"])

        # Existing dominates new.
        for old in profile:
            if old["dep"] >= dep and old["arr"] <= arr:
                return False

        # New dominates existing.
        kept = []
        for old in profile:
            if dep >= old["dep"] and arr <= old["arr"]:
                continue
            kept.append(old)

        kept.append(label)
        kept.sort(key=lambda x: x["dep"])

        profile[:] = kept
        return True


    def _profile_reconstruct_steps(self, label: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Follow pure-dict labels and build old plan_candidates-compatible steps.
        """
        steps = []
        seen = set()

        while label is not None:
            state = (
                label.get("kind"),
                label.get("from_stop"),
                label.get("to_stop"),
                label.get("step_dep"),
                label.get("step_arr"),
                label.get("trip_id"),
            )

            if state in seen:
                return []

            seen.add(state)

            kind = label["kind"]

            if kind == "walk":
                dep = int(label["step_dep"])
                arr = int(label["step_arr"])

                if arr < dep:
                    return []

                steps.append(
                    {
                        "type": "walk",
                        "from_stop": int(label["from_stop"]),
                        "to_stop": int(label["to_stop"]),
                        "departure_secs": dep,
                        "arrival_secs": arr,
                        "duration_sec": max(0, arr - dep),
                    }
                )

            elif kind == "ride":
                dep = int(label["step_dep"])
                arr = int(label["step_arr"])
                trip_id = str(label["trip_id"])

                if arr < dep:
                    return []

                metadata = self.trip_metadata_by_id.get(trip_id, {})

                steps.append(
                    {
                        "type": "ride",
                        "trip_id": trip_id,
                        "line_text": metadata.get("line_text"),
                        "operator_id": metadata.get("operator_id"),
                        "transport": metadata.get("transport"),
                        "from_stop": int(label["from_stop"]),
                        "to_stop": int(label["to_stop"]),
                        "departure_secs": dep,
                        "arrival_secs": arr,
                        "duration_sec": max(0, arr - dep),
                    }
                )

            else:
                return []

            label = label.get("next")

        # Merge consecutive rides on same trip, just in case.
        return self._profile_merge_same_trip_rides(steps)


    def _profile_merge_same_trip_rides(self, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Defensive cleanup:
        If reconstruction creates consecutive ride steps on the same trip, merge them.
        """
        if not steps:
            return steps

        merged = []

        for step in steps:
            if (
                merged
                and step["type"] == "ride"
                and merged[-1]["type"] == "ride"
                and step.get("trip_id") == merged[-1].get("trip_id")
                and step["from_stop"] == merged[-1]["to_stop"]
            ):
                merged[-1]["to_stop"] = step["to_stop"]
                merged[-1]["arrival_secs"] = step["arrival_secs"]
                merged[-1]["duration_sec"] = max(
                    0,
                    merged[-1]["arrival_secs"] - merged[-1]["departure_secs"],
                )
            else:
                merged.append(step)

        return merged


    def _profile_footpath_dist_m(self, from_stop: int, to_stop: int) -> float:
        """
        Distance lookup with reverse fallback.
        """
        direct = self.footpath_dist.get((int(from_stop), int(to_stop)))

        if direct is not None:
            return float(direct)

        reverse = self.footpath_dist.get((int(to_stop), int(from_stop)))

        if reverse is not None:
            return float(reverse)

        return 0.0
    
    
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
    def _parse_date(value: str | date) -> date:
        if isinstance(value, date):
            return value
        text = str(value).strip().lower()
        if text in DAY_NAMES:
            return datetime.today().date()
        return datetime.fromisoformat(text).date()
    
    def _format_times(self, route: dict) -> dict:
        """Add human-readable time strings to route and steps."""
        route["departure_time"] = self._secs_to_time(route["departure_secs"])
        route["arrival_time"] = self._secs_to_time(route["arrival_secs"])

        for step in route["steps"]:
            step["departure_time"] = self._secs_to_time(step["departure_secs"])
            step["arrival_time"] = self._secs_to_time(step["arrival_secs"])

        return route


    @staticmethod
    @lru_cache(maxsize=2048)
    def _secs_to_time(seconds: int | float) -> str:
        seconds = int(round(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"


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