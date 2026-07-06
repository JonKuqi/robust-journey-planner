from old.confidence_evaluator import ConfidenceEvaluator


# Quantile profiles used across tests.
# "slow" service: covers the full [p50, p95] quantile range
_SLOW = {
    "p50_delay_sec": 120.0,
    "p60_delay_sec": 180.0,
    "p70_delay_sec": 240.0,
    "p80_delay_sec": 360.0,
    "p85_delay_sec": 420.0,
    "p90_delay_sec": 480.0,
    "p95_delay_sec": 600.0,
}
# "fast" service: tight distribution, rarely late
_FAST = {
    "p50_delay_sec": 30.0,
    "p60_delay_sec": 50.0,
    "p70_delay_sec": 70.0,
    "p80_delay_sec": 90.0,
    "p85_delay_sec": 135.0,
    "p90_delay_sec": 180.0,
    "p95_delay_sec": 300.0,
}


def test_confidence_evaluator_rejects_low_probability_route():
    """Route fails when CDF product < Q, even if individual legs look schedulable.

    Ride A arrives at 08:30 and Ride B departs at 08:35 (5-min headway = 300 sec).
    With a slow-service quantile profile (p80 = 6 min), P(delay ≤ 300 sec) ≈ 0.725.
    Ride B arrives at 09:00, deadline is 09:10 (10-min headway = 600 sec = p95).
    P(last leg) = 0.95.  route_confidence ≈ 0.725 × 0.95 = 0.689 < 0.90.
    """
    route = {
        "departure_secs": 8 * 3600,
        "arrival_secs": 9 * 3600,
        "day_of_week": "monday",
        "steps": [
            {
                "type": "ride",
                "departure_secs": 8 * 3600,
                "arrival_secs": 8 * 3600 + 30 * 60,
                "predicted_delay_sec": 6 * 60,
                **_SLOW,
            },
            {
                "type": "ride",
                "departure_secs": 8 * 3600 + 35 * 60,
                "arrival_secs": 9 * 3600,
                "predicted_delay_sec": 0,
                **_SLOW,
            },
        ],
    }

    evaluated = ConfidenceEvaluator().route_passes(route, "09:10", 0.90)

    assert evaluated["passes_confidence"] is False
    assert evaluated["route_confidence"] < 0.90
    # Tight-but-positive headway is not a structurally impossible connection.
    assert evaluated["missed_connection"] is None


def test_confidence_evaluator_accepts_safe_route():
    """Route passes when CDF product ≥ Q.

    Ride A arrives at 08:30, Ride B departs at 08:35 (headway = 5 min = 300 sec = p95 of
    the fast-service profile).  P(Ride A delay ≤ 300 sec) = 0.95.
    Ride B arrives at 08:50, deadline is 09:00 (headway = 10 min = 600 sec, above p95).
    P(Ride B delay ≤ 600 sec) extrapolates to 1.0.
    route_confidence = 0.95 × 1.0 = 0.95 ≥ 0.90.
    The robust display arrival (08:50 + 5 min predicted delay) is 08:55.
    """
    route = {
        "departure_secs": 8 * 3600,
        "arrival_secs": 8 * 3600 + 50 * 60,
        "day_of_week": "monday",
        "steps": [
            {
                "type": "ride",
                "departure_secs": 8 * 3600,
                "arrival_secs": 8 * 3600 + 30 * 60,
                "predicted_delay_sec": 4 * 60,
                **_FAST,
            },
            {
                "type": "ride",
                "departure_secs": 8 * 3600 + 35 * 60,
                "arrival_secs": 8 * 3600 + 50 * 60,
                "predicted_delay_sec": 5 * 60,
                **_FAST,
            },
        ],
    }

    evaluated = ConfidenceEvaluator().route_passes(route, "09:00", 0.90)

    assert evaluated["passes_confidence"] is True
    assert evaluated["route_confidence"] >= 0.90
    # Display: last ride 08:50 + 5-min predicted delay = 08:55.
    assert evaluated["robust_arrival_time"] == "08:55:00"


def test_confidence_evaluator_rejects_impossible_schedule():
    """Route fails with missed_connection when scheduled headway is negative.

    Ride A arrives at 08:35 but Ride B is scheduled to depart at 08:30.
    The headway is -5 min regardless of delays, so the connection is impossible.
    """
    route = {
        "departure_secs": 8 * 3600,
        "arrival_secs": 9 * 3600,
        "day_of_week": "monday",
        "steps": [
            {
                "type": "ride",
                "departure_secs": 8 * 3600,
                "arrival_secs": 8 * 3600 + 35 * 60,
                "predicted_delay_sec": 0,
                **_FAST,
            },
            {
                "type": "ride",
                "departure_secs": 8 * 3600 + 30 * 60,  # departs before Ride A arrives
                "arrival_secs": 9 * 3600,
                "predicted_delay_sec": 0,
                **_FAST,
            },
        ],
    }

    evaluated = ConfidenceEvaluator().route_passes(route, "09:10", 0.90)

    assert evaluated["passes_confidence"] is False
    assert evaluated["missed_connection"] is not None
    assert evaluated["missed_connection"]["step_index"] == 1
    assert evaluated["missed_connection"]["headway_sec"] < 0


def test_confidence_evaluator_single_leg_route():
    """Single-leg route passes when P(arrive by deadline) ≥ Q.

    One ride: departs 08:00, arrives 08:50. Deadline 09:00 → headway = 10 min = 600 sec.
    Using fast-service profile: P(delay ≤ 600 sec) extrapolates to 1.0.
    route_confidence = 1.0 ≥ 0.90.
    """
    route = {
        "departure_secs": 8 * 3600,
        "arrival_secs": 8 * 3600 + 50 * 60,
        "day_of_week": "monday",
        "steps": [
            {
                "type": "ride",
                "departure_secs": 8 * 3600,
                "arrival_secs": 8 * 3600 + 50 * 60,
                "predicted_delay_sec": 2 * 60,
                **_FAST,
            },
        ],
    }

    evaluated = ConfidenceEvaluator().route_passes(route, "09:00", 0.90)

    assert evaluated["passes_confidence"] is True
    assert len(evaluated["per_step_probabilities"]) == 1
    assert evaluated["per_step_probabilities"][0]["headway_sec"] == 600.0


def test_confidence_evaluator_no_quantile_data_neutral():
    """Falls back to 0.5 per step when no quantile data is available on steps."""
    route = {
        "departure_secs": 8 * 3600,
        "arrival_secs": 8 * 3600 + 50 * 60,
        "day_of_week": "monday",
        "steps": [
            {
                "type": "ride",
                "departure_secs": 8 * 3600,
                "arrival_secs": 8 * 3600 + 50 * 60,
                "predicted_delay_sec": 0,
                # no p50/p80/p90/p95 fields → neutral 0.5 fallback
            },
        ],
    }

    evaluated = ConfidenceEvaluator().route_passes(route, "09:00", 0.40)

    # With 0.5 fallback and q=0.40: route_confidence=0.5 ≥ 0.40 → passes.
    assert evaluated["passes_confidence"] is True
    assert abs(evaluated["route_confidence"] - 0.5) < 1e-9


def test_route_sorting_tiebreakers():
    """Verify routing tie-breakers: departure (desc), transfers (asc), walk (asc)."""
    # 1. Latest departure (Primary)
    route_fastest = {"departure_secs": 30600, "n_transfers": 5, "total_walk_m": 1000}
    
    # 2. Earlier departure, but fewest transfers (Secondary)
    route_few_transfers = {"departure_secs": 28800, "n_transfers": 0, "total_walk_m": 500}
    
    # 3. Same departure/transfers as #4, but less walk (Tertiary)
    route_less_walk = {"departure_secs": 28800, "n_transfers": 1, "total_walk_m": 100}
    
    # 4. Same departure/transfers as #3, but more walk (Last)
    route_more_walk = {"departure_secs": 28800, "n_transfers": 1, "total_walk_m": 300}

    unsorted = [route_more_walk, route_fastest, route_less_walk, route_few_transfers]
    sorted_routes = ConfidenceEvaluator.sort_routes(unsorted)

    # Asserts the specific order [1, 2, 3, 4]
    assert sorted_routes == [route_fastest, route_few_transfers, route_less_walk, route_more_walk]



