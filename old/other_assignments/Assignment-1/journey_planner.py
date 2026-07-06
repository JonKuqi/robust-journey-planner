"""Journey planning based on the Connection Scan Algorithm (CSA).

This module prepares timetable and walking-transfer data, then computes
earliest-arrival journeys using a forward CSA scan with predecessor tracking.
"""

from collections import defaultdict, deque
from math import inf
from bisect import bisect_left

from csa_data_handler import CSADataHandler



class JourneyPlanner:
    
    def __init__(self, schema: str):
        """Initialize a journey planner for a database schema.

        Args:
            schema: Source schema used to build and query CSA input tables.
        """
        self.schema = schema
        self.shared_schema = "iceberg.com490_iceberg"
        self.connections_by_day = None
        self.footpaths = None
        self.stops = None
        self.prepared = False
        self.data_handler = None
        self.dep_secs_by_day = None
        self.trip_id_to_idx = None
        self.trip_idx_to_id = None
        self.stop_id_to_idx = None
        self.stop_idx_to_id = None
        self.n_stops = 0
        self.n_trips = 0
        

    def prepare(self, regions=None, rebuild: bool = False):
        """Load and index all data required for CSA routing.

        Builds prerequisite tables through `CSADataHandler` and materializes
        in-memory structures for stops, footpaths, and day-specific connections.

        Args:
            regions: Reserved for future regional filtering.
            rebuild: If True, forces rebuilding source data artifacts.
        """
        self.data_handler = CSADataHandler(
            schema=self.schema,
            shared_schema=self.shared_schema
        )

        self.data_handler.build_all(
            rebuild_prerequisites=True,
            force=rebuild,
            regions=regions
        )
        stops_df = self.data_handler.fetch_stops()
        footpaths_df = self.data_handler.fetch_footpaths()
        
        
        connections_df = self.data_handler.fetch_connections()

        self.stops = set(stops_df["stop_id"].tolist())

        self.footpaths = defaultdict(list)
        for row in footpaths_df.itertuples(index=False):
            self.footpaths[int(row.a_stop_id)].append(
                (
                    int(row.b_stop_id),
                    int(round(float(row.walk_time_min) * 60)),
                    float(row.distance),
                )
            )

        self.connections_by_day = {
            "monday": [],
            "tuesday": [],
            "wednesday": [],
            "thursday": [],
            "friday": [],
            "saturday": [],
            "sunday": [],
        }
        self.dep_secs_by_day = {
            "monday": [],
            "tuesday": [],
            "wednesday": [],
            "thursday": [],
            "friday": [],
            "saturday": [],
            "sunday": [],
        }
        self.trip_id_to_idx = {}
        self.trip_idx_to_id = []

        for row in connections_df.itertuples(index=False):
            trip_id = row.trip_id

            if trip_id not in self.trip_id_to_idx:
                self.trip_id_to_idx[trip_id] = len(self.trip_idx_to_id)
                self.trip_idx_to_id.append(trip_id)

            trip_idx = self.trip_id_to_idx[trip_id]

            c = (
                int(row.dep_stop_id),
                int(row.arr_stop_id),
                int(row.dep_secs),
                int(row.arr_secs),
                trip_idx,
            )

            if row.monday:
                self.connections_by_day["monday"].append(c)
                self.dep_secs_by_day["monday"].append(int(row.dep_secs))
            if row.tuesday:
                self.connections_by_day["tuesday"].append(c)
                self.dep_secs_by_day["tuesday"].append(int(row.dep_secs))
            if row.wednesday:
                self.connections_by_day["wednesday"].append(c)
                self.dep_secs_by_day["wednesday"].append(int(row.dep_secs))
            if row.thursday:
                self.connections_by_day["thursday"].append(c)
                self.dep_secs_by_day["thursday"].append(int(row.dep_secs))
            if row.friday:
                self.connections_by_day["friday"].append(c)
                self.dep_secs_by_day["friday"].append(int(row.dep_secs))
            if row.saturday:
                self.connections_by_day["saturday"].append(c)
                self.dep_secs_by_day["saturday"].append(int(row.dep_secs))
            if row.sunday:
                self.connections_by_day["sunday"].append(c)
                self.dep_secs_by_day["sunday"].append(int(row.dep_secs))
        
        self.n_trips = len(self.trip_idx_to_id)
        self.prepared = True
        


    def route(self, start_id, end_id=None, departs="12:30", day="monday", max_walk_m=100):
        """Compute an earliest-arrival journey using a CSA scan.

        Args:
            start_id: Origin stop identifier.
            end_id: Optional destination stop identifier. If omitted, returns
                earliest arrival times for all reachable stops.
            departs: Departure time as HH:MM or HH:MM:SS.
            day: Service day key (e.g., "monday").
            max_walk_m: Maximum walking edge distance in meters.

        Returns:
            If `end_id` is None, a mapping of reachable stop_id to arrival time
            in seconds. Otherwise, a route encoded in assignment tuple format.

        Raises:
            RuntimeError: If planner data has not been prepared.
            ValueError: If day or stop identifiers are invalid.
        """

        if day not in self.connections_by_day:
            raise ValueError(f"Invalid day: {day}")

        if start_id not in self.stops:
            raise ValueError(f"Unknown start stop: {start_id}")

        if end_id is not None and end_id not in self.stops:
            raise ValueError(f"Unknown end stop: {end_id}")

        departure_secs = self._time_to_secs(departs)
        connections = self.connections_by_day[day]
        start_idx = bisect_left(self.dep_secs_by_day[day], departure_secs)

        arrivals = self._initialize_arrivals(start_id, departure_secs)
        pred = {}
        trip_reachable = [False] * self.n_trips

        self._apply_footpaths_with_pred(arrivals, pred, start_id, departure_secs, max_walk_m)

        for i in range(start_idx, len(connections)):
            dep_stop, arr_stop, dep_secs, arr_secs, trip_idx = connections[i]
            arr_secs += 86400 * (dep_secs>arr_secs) # If arrival is in the next day
            if end_id is not None and arrivals[end_id] <= dep_secs:
                break

            if trip_reachable[trip_idx] or arrivals[dep_stop] <= dep_secs:
                trip_reachable[trip_idx] = True

                if arr_secs < arrivals[arr_stop]:
                    arrivals[arr_stop] = arr_secs
                    pred[arr_stop] = (
                        "trip",
                        dep_stop,
                        dep_secs,
                        arr_secs,
                        self.trip_idx_to_id[trip_idx],
                    )
                    self._apply_footpaths_with_pred(arrivals, pred, arr_stop, arr_secs, max_walk_m)

        if end_id is None:
            return {stop: t for stop, t in arrivals.items() if t < inf}
            
        if arrivals[end_id] > 86400 : # The end location has not been reached within the day, so we'll also look at the next day.
            # Note that the departure time has always been in the same day so far, so all connections before midnight have been already checked.
            start_idx = bisect_left(self.dep_secs_by_day[self._next(day)], 0)
            connections = self.connections_by_day[self._next(day)]
            for i in range(start_idx, len(connections)):
                dep_stop, arr_stop, dep_secs, arr_secs, trip_idx = connections[i]
                dep_secs += 86400
                arr_secs += 86400
                if end_id is not None and arrivals[end_id] <= dep_secs:
                    break
    
                if trip_reachable[trip_idx] or arrivals[dep_stop] <= dep_secs:
                    trip_reachable[trip_idx] = True
                    if arr_secs  < arrivals[arr_stop]:
                        arrivals[arr_stop] = arr_secs
                        pred[arr_stop] = (
                            "trip",
                            dep_stop,
                            dep_secs,
                            arr_secs,
                            self.trip_idx_to_id[trip_idx],
                        )
                        self._apply_footpaths_with_pred(arrivals, pred, arr_stop, arr_secs, max_walk_m)
                    
        if arrivals[end_id] == inf:
            return []

        segments = self._reconstruct_route(pred, start_id, end_id)
        if segments is None:
            return []

        return self._segments_to_required_tuples(segments)






    def _reconstruct_route(self, pred, start_id, end_id):
        """Reconstruct route segments by backtracking predecessor links.

        Args:
            pred: Predecessor map built during scanning.
            start_id: Origin stop identifier.
            end_id: Destination stop identifier.

        Returns:
            A chronological segment list using:
            ("walk", from_stop, to_stop, arr_time) and
            ("trip", dep_stop, arr_stop, dep_secs, arr_secs, trip_id),
            or None when no valid predecessor chain exists.
        """
        segments = []
        cur = end_id

        while cur != start_id:
            if cur not in pred:
                return None

            info = pred[cur]

            if info[0] == "walk":
                _, prev_stop, arr_time = info
                segments.append(("walk", prev_stop, cur, arr_time))
                cur = prev_stop

            elif info[0] == "trip":
                _, dep_stop, dep_secs, arr_secs, trip_id = info
                segments.append(("trip", dep_stop, cur, dep_secs, arr_secs, trip_id))
                cur = dep_stop

            else:
                raise ValueError(f"Unknown predecessor type: {info[0]}")

        segments.reverse()
        return segments


    def _segments_to_required_tuples(self, segments):
        """Convert internal segments to assignment output tuple format.

        Output tuple semantics:
            (ts, stop, trip_id, None): board a trip at `stop`.
            (ts, None, trip_id, stop): alight from a trip at `stop`.
            (ts, from_stop, None, stop): complete a walking transfer.

        Consecutive segments on the same trip are merged into one board/alight
        pair to keep the output compact.
        """
        if not segments:
            return []

        route = []
        i = 0

        while i < len(segments):
            seg = segments[i]

            if seg[0] == "walk":
                _, from_stop, to_stop, arr_time = seg
                route.append((arr_time, from_stop, None, to_stop))
                i += 1
                continue

            # trip segment(s)
            _, dep_stop, arr_stop, dep_secs, arr_secs, trip_id = seg
            board_stop = dep_stop
            board_time = dep_secs
            final_arr_stop = arr_stop
            final_arr_time = arr_secs

            i += 1

            # merge consecutive segments on the same trip
            while i < len(segments):
                nxt = segments[i]
                if (
                    nxt[0] == "trip"
                    and nxt[5] == trip_id
                    and nxt[1] == final_arr_stop
                ):
                    _, _, next_arr_stop, _, next_arr_secs, _ = nxt
                    final_arr_stop = next_arr_stop
                    final_arr_time = next_arr_secs
                    i += 1
                else:
                    break

            route.append((board_time, board_stop, trip_id, None))
            route.append((final_arr_time, None, trip_id, final_arr_stop))

        return route


    def _apply_footpaths_with_pred(self, arrivals, pred, from_stop, base_time, max_walk_m):
        """
        Relax direct walking-transfer edges from one stop.

        Assumptions / design choice:
        - `self.footpaths` is treated as a table of DIRECT transfer edges between
        nearby stops, not as a transitively closed CSA footpath relation.
        - Therefore, this function only relaxes one-hop walking edges that were
        explicitly precomputed in `stop_to_stop`.
        - We intentionally do NOT perform BFS / multi-hop walking expansion here.
        Chaining multiple walking edges during routing would allow indirect walks
        through intermediate stops without explicitly tracking the total walking
        distance over the full journey.
        - This is a deliberate simplification chosen to stay consistent with the
        current data model and the assignment assumptions: walking transfers are
        allowed between nearby stops, and each direct edge already includes the
        transfer-time formula (2 min base + 1 min per 50 m).
        - `max_walk_m` is applied as a per-edge filter on direct walking transfers.

        Effect:
        - If walking from `from_stop` to `to_stop` improves the earliest known
        arrival time, we update:
            arrivals[to_stop]
            pred[to_stop] = ("walk", from_stop, new_time)

        Note:
        - This version does NOT model cumulative walking-budget constraints across
        the entire trip. Supporting that correctly would require extending the
        routing state, rather than doing repeated queue expansion here.
        """
        for to_stop, walk_secs, distance_m in self.footpaths.get(from_stop, ()):
            if walk_secs is None:
                continue
            if distance_m is not None and distance_m > max_walk_m:
                continue

            new_time = base_time + walk_secs

            if new_time < arrivals[to_stop]:
                arrivals[to_stop] = new_time
                pred[to_stop] = ("walk", from_stop, new_time)


      

    def _apply_footpaths_with_pred_old(self, arrivals, pred, from_stop, base_time, max_walk_m):
        """
        Old multi-hop walking version, kept only as reference.

        It used a queue/BFS style expansion over walking edges, so it could chain
        several nearby-stop transfers in a row. We do not use it anymore because our
        current footpath table is meant to represent direct transfers only, and this
        older approach could create indirect multi-hop walking paths without properly
        controlling total walking distance.

        The current routing logic only applies direct one-hop walking transfers from
        the precomputed footpath table.

        Kept only for comparison or debugging. Not used by the active algorithm.
        
        --------------------
        Propagate walking transfers from a stop and update predecessors.

        Performs a queue-based relaxation over footpath edges reachable from
        `from_stop`, bounded by `max_walk_m`, and records predecessor entries
        of the form:
            pred[to_stop] = ("walk", previous_stop, arrival_time)
        """
        q = deque([(from_stop, base_time)])

        while q:
            current_stop, current_time = q.popleft()

            if current_stop not in self.footpaths:
                continue

            for to_stop, walk_secs, distance_m in self.footpaths[current_stop]:
                if walk_secs is None:
                    continue
                if distance_m is not None and distance_m > max_walk_m:
                    continue

                new_time = current_time + walk_secs

                if new_time < arrivals[to_stop]:
                    arrivals[to_stop] = new_time
                    pred[to_stop] = ("walk", current_stop, new_time)
                    q.append((to_stop, new_time))

    
            
            
        
    def _initialize_arrivals(self, start_id: int, departure_secs: int) -> dict[int, float]:
        """Create arrival-time labels initialized to infinity.

        Args:
            start_id: Origin stop identifier.
            departure_secs: Departure timestamp in seconds.

        Returns:
            Dictionary mapping stop_id to earliest known arrival time.
        """
        arrivals = {stop_id: inf for stop_id in self.stops}
        arrivals[start_id] = departure_secs
        return arrivals

            
            
        
        
    def _time_to_secs(self, time_str: str) -> int:
        """Convert an HH:MM or HH:MM:SS time string to seconds.

        Args:
            time_str: Time string in 24-hour format.

        Returns:
            Seconds elapsed since 00:00:00.

        Raises:
            ValueError: If the provided time format is invalid.
        """
        parts = time_str.split(":")
        if len(parts) == 2:
            h, m = parts
            s = 0
        elif len(parts) == 3:
            h, m, s = parts
        else:
            raise ValueError(f"Invalid time format: {time_str}")

        return int(h) * 3600 + int(m) * 60 + int(s)



    
    def _next(self, day):
            
            """
            Returns the next day
            """
            
            days = ['sunday', 'saturday', 'friday', 'thursday', 'wednesday', 'tuesday', 'monday']
            return days[days.index(day)-1]



















