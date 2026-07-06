from __future__ import annotations

"""Evaluate whether a route satisfies the user's confidence requirement.

Route confidence is computed as:

    route_confidence = ∏ P(delay_k ≤ headway_k)
                       k

where headway_k = next_departure – cumulative_walk – scheduled_arrival for leg k,
and P(delay ≤ headway) is estimated by linear CDF interpolation over the
p50/p60/p70/p80/p85/p90/p95 quantile columns written by ``DelayModel.add_delays_to_route``.

A route is accepted when route_confidence ≥ confidence_q (and no connection has a
negative scheduled headway).  Step-level robust-arrival display fields are derived from
predicted_delay_sec and remain available regardless of the pass/fail outcome.
"""

from copy import deepcopy


class ConfidenceEvaluator:
    """Check route confidence using per-connection CDF probabilities."""

    def route_passes(
        self,
        route: dict,
        arrival_deadline: str | int | float,
        confidence_q: float,
    ) -> dict:
        """Return a route copy annotated with computed confidence and pass/fail status.

        Fields added to the returned dict
        ----------------------------------
        route_confidence : float
            P(route completes on time) = product of per-connection P(delay ≤ headway).
        per_step_probabilities : list[dict]
            Per-ride-step breakdown: step_index, headway_sec, probability.
        passes_confidence : bool
            True when no impossible scheduled connection and route_confidence ≥ confidence_q.
        missed_connection : dict | None
            Set only when a connection has a *negative scheduled headway* (structurally
            impossible regardless of delays).  Not set for tight-but-possible connections.
        robust_arrival_secs / robust_arrival_time : float / str
            Scenario-based robust arrival (schedule + predicted_delay_sec) for display.
        """
        out = deepcopy(route)
        deadline_secs = self._deadline_secs(arrival_deadline)
        steps = out.get("steps", [])

        # ── Pass 1: propagate robust display timing using predicted_delay_sec ──────────
        robust_cursor = None
        for step in steps:
            scheduled_departure = step.get("departure_secs", robust_cursor or 0)
            if step.get("type") == "ride":
                delay = float(step.get("predicted_delay_sec", 0.0) or 0.0)
                robust_cursor = float(step.get("arrival_secs", scheduled_departure)) + delay
            else:
                duration = float(step.get("duration_sec", 0.0) or 0.0)
                start = max(
                    float(scheduled_departure),
                    float(robust_cursor or scheduled_departure),
                )
                robust_cursor = start + duration
            step["robust_arrival_secs"] = robust_cursor
            step["robust_arrival_time"] = self._secs_to_time(robust_cursor)

        robust_arrival_secs = robust_cursor if robust_cursor is not None else out.get("arrival_secs")

        # ── Pass 2: compute per-connection P(delay ≤ headway) and their product ───────
        route_confidence = 1.0
        per_step_probabilities = []
        missed_connection = None

        for idx, step in enumerate(steps):
            if step.get("type") != "ride":
                continue

            ride_arrival = float(step.get("arrival_secs", 0))

            # Walk forward to find the next ride step (accumulating walk durations).
            acc_walk = 0.0
            j = idx + 1
            while j < len(steps) and steps[j].get("type") == "walk":
                acc_walk += float(steps[j].get("duration_sec", 0.0) or 0.0)
                j += 1

            if j < len(steps):
                next_dep = float(steps[j].get("departure_secs", 0))
            else:
                next_dep = float(deadline_secs)

            # headway: time available for delay before missing the connection / deadline.
            headway = next_dep - acc_walk - ride_arrival

            # Flag structurally impossible connections (negative scheduled headway).
            if headway < 0 and missed_connection is None:
                missed_connection = {
                    "step_index": j if j < len(steps) else idx,
                    "headway_sec": headway,
                }

            quantile_data = {
                "p50_delay_sec": step.get("p50_delay_sec"),
                "p60_delay_sec": step.get("p60_delay_sec"),
                "p70_delay_sec": step.get("p70_delay_sec"),
                "p80_delay_sec": step.get("p80_delay_sec"),
                "p85_delay_sec": step.get("p85_delay_sec"),
                "p90_delay_sec": step.get("p90_delay_sec"),
                "p95_delay_sec": step.get("p95_delay_sec"),
            }
            prob = self._cdf_from_quantiles(headway, quantile_data)
            per_step_probabilities.append(
                {"step_index": idx, "headway_sec": headway, "probability": prob}
            )
            route_confidence *= prob

        out["confidence_q"] = confidence_q
        out["arrival_deadline_secs"] = deadline_secs
        out["robust_arrival_secs"] = robust_arrival_secs
        out["robust_arrival_time"] = self._secs_to_time(robust_arrival_secs)
        out["route_confidence"] = route_confidence
        out["per_step_probabilities"] = per_step_probabilities
        out["missed_connection"] = missed_connection
        out["passes_confidence"] = missed_connection is None and route_confidence >= confidence_q
        return out

    @staticmethod
    def _cdf_from_quantiles(headway_sec: float, quantile_data: dict) -> float:
        """Estimate P(delay ≤ headway_sec) by linear interpolation over known quantile points.

        Known anchor points (delay_seconds, cumulative_probability):
          (0, 0.0)  – conservative lower bound: delay < 0 is impossible (clipped at 0).
          (p50, 0.50), (p80, 0.80), (p90, 0.90), (p95, 0.95) – from quantile table.

        For headway below p50: linearly interpolate from (0, 0.0) to (p50, 0.50).
        For headway above p95: linearly extrapolate the p90→p95 slope, capped at 1.0.
        Between two known points: linear interpolation.

        The lower anchor (0, 0.0) is conservative — in practice some fraction of vehicles
        arrive on time (delay = 0), so the true CDF at 0 is > 0.  Using 0 means we
        under-estimate the probability for very tight connections, which errs on the safe side.
        """
        if headway_sec < 0:
            return 0.0

        points = []
        for col, prob in (
            ("p50_delay_sec", 0.50),
            ("p60_delay_sec", 0.60),
            ("p70_delay_sec", 0.70),
            ("p80_delay_sec", 0.80),
            ("p85_delay_sec", 0.85),
            ("p90_delay_sec", 0.90),
            ("p95_delay_sec", 0.95),
        ):
            v = quantile_data.get(col)
            if v is not None:
                try:
                    fv = float(v)
                    if fv == fv:  # guard against NaN
                        points.append((max(0.0, fv), prob))
                except (TypeError, ValueError):
                    pass

        if not points:
            return 0.5  # no quantile data available: neutral fallback

        # Deduplicate: if multiple quantiles map to the same delay value, keep the highest prob.
        seen: dict[float, float] = {}
        for d, p in points:
            if d not in seen or p > seen[d]:
                seen[d] = p
        points = sorted(seen.items())
        delays = [p[0] for p in points]
        probs = [p[1] for p in points]

        # Below the lowest known quantile: linear from (0, 0.0) to (delays[0], probs[0]).
        if headway_sec <= delays[0]:
            if delays[0] == 0.0:
                return probs[0]
            return (headway_sec / delays[0]) * probs[0]

        # Above the highest known quantile: extrapolate the last slope, cap at 1.0.
        if headway_sec >= delays[-1]:
            if len(delays) >= 2 and delays[-1] > delays[-2]:
                slope = (probs[-1] - probs[-2]) / (delays[-1] - delays[-2])
                return min(1.0, probs[-1] + slope * (headway_sec - delays[-1]))
            return 1.0

        # Between two known points: linear interpolation.
        for i in range(len(delays) - 1):
            if delays[i] <= headway_sec < delays[i + 1]:
                span = delays[i + 1] - delays[i]
                if span == 0.0:
                    return probs[i + 1]
                t = (headway_sec - delays[i]) / span
                return probs[i] + t * (probs[i + 1] - probs[i])

        return probs[-1]

    @staticmethod
    def sort_routes(routes: list[dict]) -> list[dict]:
        """Sort robust routes: latest departure first, then min transfers, then min walk."""
        return sorted(
            routes,
            key=lambda r: (
                -r.get("departure_secs", 0), # Primary (Descending)
                r.get("n_transfers", 0),     # Secondary (Ascending)
                r.get("total_walk_m", 0),    # Tertiary (Ascending)
            ),
        )

    @staticmethod
    def _deadline_secs(value: str | int | float) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value)
        if "T" in text:
            text = text.split("T", 1)[1]
        if " " in text:
            text = text.split(" ", 1)[1]
        parts = text.split(":")
        if len(parts) == 2:
            h, m = parts
            s = 0
        elif len(parts) == 3:
            h, m, s = parts
        else:
            raise ValueError(f"Invalid deadline time: {value}")
        return int(h) * 3600 + int(m) * 60 + int(s)

    @staticmethod
    def _secs_to_time(value) -> str:
        seconds = int(round(value or 0))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
