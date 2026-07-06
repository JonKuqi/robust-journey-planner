from __future__ import annotations

"""Fast confidence calculations from precomputed delay quantiles."""

from collections.abc import Callable, Sequence


DEFAULT_CONFIDENCE_Q = 0.75
DEFAULT_QUANTILE_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)


def normalize_confidence_q(value: float | int | None, default: float = DEFAULT_CONFIDENCE_Q) -> float:
    """Normalize confidence input to [0, 1], accepting either 0.9 or 90."""
    if value is None:
        return default
    q = float(value)
    if q > 1.0:
        q /= 100.0
    if q < 0.0:
        return 0.0
    if q > 1.0:
        return 1.0
    return q


def cdf_from_quantiles(
    headway_sec: float,
    quantiles: Sequence[float] | None,
    levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
    fallback_probability: float = 0.5,
) -> float:
    """Estimate P(delay <= headway_sec) by interpolating stored delay quantiles."""
    if headway_sec < 0:
        return 0.0
    if not quantiles:
        return fallback_probability

    raw_points: list[tuple[float, float]] = []
    for raw_delay, raw_prob in zip(quantiles, levels):
        try:
            delay = max(0.0, float(raw_delay))
            prob = float(raw_prob)
        except (TypeError, ValueError):
            continue
        if delay != delay or prob != prob:
            continue
        raw_points.append((delay, prob))

    if not raw_points:
        return fallback_probability

    sorted_delays = sorted(delay for delay, _ in raw_points)
    sorted_probs = sorted(prob for _, prob in raw_points)

    points: dict[float, float] = {}
    for delay, prob in zip(sorted_delays, sorted_probs):
        if delay not in points or prob > points[delay]:
            points[delay] = prob

    ordered = sorted(points.items())
    delays = [p[0] for p in ordered]
    probs = [p[1] for p in ordered]

    if headway_sec <= delays[0]:
        if delays[0] == 0.0:
            return probs[0]
        return max(0.0, min(1.0, (headway_sec / delays[0]) * probs[0]))

    if headway_sec >= delays[-1]:
        if len(delays) >= 2 and delays[-1] > delays[-2]:
            slope = (probs[-1] - probs[-2]) / (delays[-1] - delays[-2])
            return max(0.0, min(1.0, probs[-1] + slope * (headway_sec - delays[-1])))
        return 1.0

    for i in range(len(delays) - 1):
        left_delay = delays[i]
        right_delay = delays[i + 1]
        if left_delay <= headway_sec < right_delay:
            span = right_delay - left_delay
            if span == 0.0:
                return probs[i + 1]
            t = (headway_sec - left_delay) / span
            return max(0.0, min(1.0, probs[i] + t * (probs[i + 1] - probs[i])))

    return max(0.0, min(1.0, probs[-1]))


def route_confidence_from_steps(
    steps: list[dict],
    deadline_secs: int | float,
    quantiles_for_step: Callable[[dict], Sequence[float] | None],
    levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
    fallback_probability: float = 0.5,
) -> float:
    """Compute product confidence for making all transfers and the final deadline."""
    confidence, _ = route_confidence_details_from_steps(
        steps=steps,
        deadline_secs=deadline_secs,
        quantiles_for_step=quantiles_for_step,
        levels=levels,
        fallback_probability=fallback_probability,
    )
    return confidence


def route_confidence_details_from_steps(
    steps: list[dict],
    deadline_secs: int | float,
    quantiles_for_step: Callable[[dict], Sequence[float] | None],
    levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
    fallback_probability: float = 0.5,
) -> tuple[float, list[dict]]:
    """Return route confidence plus one compact probability row per ride step."""
    confidence = 1.0
    probabilities: list[dict] = []
    n = len(steps)

    for idx, step in enumerate(steps):
        if step.get("type") != "ride":
            continue

        ride_arrival = float(step.get("arrival_secs", 0.0) or 0.0)
        acc_walk = 0.0
        j = idx + 1
        while j < n and steps[j].get("type") == "walk":
            acc_walk += float(steps[j].get("duration_sec", 0.0) or 0.0)
            j += 1

        if j < n:
            next_limit = float(steps[j].get("departure_secs", 0.0) or 0.0)
        else:
            next_limit = float(deadline_secs)

        headway = next_limit - acc_walk - ride_arrival
        prob = cdf_from_quantiles(
            headway,
            quantiles_for_step(step),
            levels=levels,
            fallback_probability=fallback_probability,
        )
        probabilities.append(
            {"step_index": idx, "headway_sec": headway, "probability": prob}
        )
        confidence *= prob
        if confidence <= 0.0:
            return 0.0, probabilities

    return max(0.0, min(1.0, confidence)), probabilities
