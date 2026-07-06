from collections import defaultdict

from src.routing.journey_planner_1 import JourneyPlanner


def test_walk_budget_enforced_cumulatively():
    """Two walks each under max_walk_m but combined over it should be rejected."""
    planner = JourneyPlanner.__new__(JourneyPlanner)
    planner.prepared = True
    planner.stops = {1, 2, 3, 4}
    planner.footpaths = defaultdict(list)
    planner.footpaths[1].append((2, 60, 300.0))   # walk stop 1→2: 300 m
    planner.footpaths[3].append((4, 60, 300.0))   # walk stop 3→4: 300 m

    dep = 8 * 3600
    planner.connections_by_day = {"monday": [(2, 3, dep + 60, dep + 120, 0)]}
    planner.dep_secs_by_day = {"monday": [dep + 60]}
    planner.n_trips = 1
    planner.trip_idx_to_id = ["T1"]
    planner.trip_metadata_by_id = {"T1": {}}

    class _Settings:
        max_walk_m = 500

    planner.settings = _Settings()

    # Total walk = 300 + 300 = 600 m > 500 m budget → rejected
    assert planner.route(1, 4, departs="08:00", day="monday", max_walk_m=500) == []
    # Same route passes when budget is relaxed to 600 m
    assert planner.route(1, 4, departs="08:00", day="monday", max_walk_m=600) != []


def test_route_tuple_conversion_to_plain_dict():
    planner = JourneyPlanner.__new__(JourneyPlanner)
    planner.trip_metadata_by_id = {"T1": {"line_text": "M1", "operator_id": "85:151", "transport": "METRO"}}

    route = planner.route_tuples_to_dict(
        [
            (8 * 3600, 8501120, "T1", None),
            (8 * 3600 + 20 * 60, None, "T1", 8501200),
        ],
        start_stop_id=8501120,
        end_stop_id=8501200,
        requested_departure_secs=8 * 3600,
        travel_date="2026-05-11",
        day="monday",
    )

    assert route["departure_time"] == "08:00:00"
    assert route["arrival_time"] == "08:20:00"
    assert route["steps"][0]["line_text"] == "M1"

