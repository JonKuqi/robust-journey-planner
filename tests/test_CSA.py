"""
CSA benchmark helpers.

Import into tests_results/csa_speed.ipynb to measure data-build and routing
performance.  Every function returns a dict of raw timings so you can do your
own analysis on top of the printed output.

Quick start in a notebook
--------------------------
    from tests.test_CSA import bench_prepare, bench_plan_candidates, bench_route

    iterations = 6

    # time the full prepare() — the per-phase breakdown is printed by the
    # instrumented code inside csa_data_handler and journey_planner
    bench_prepare(planner, rebuild=True, rebuild_prerequisites=True)

    # time plan_candidates averaged over N iterations
    bench_plan_candidates(planner, iterations=iterations,
                          start_stop_id=8501120, end_stop_id=8501117,
                          travel_date="2026-05-11", arrival_deadline="09:00")

    # time the forward CSA route() call
    bench_route(planner, iterations=iterations,
                start_id=8501120, end_id=8501117,
                departs="08:30", day="monday")
"""

from __future__ import annotations

import time
from statistics import mean, stdev
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Internal formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

_W = 44


def _bar() -> str:
    return "=" * _W


def _thin() -> str:
    return "-" * _W


def _row(label: str, ms: float, suffix: str = "") -> str:
    s = f"  {label:<22}  {ms:>8.2f} ms"
    if suffix:
        s += f"   {suffix}"
    return s


def _header(title: str) -> None:
    print(f"\n{_bar()}")
    print(f"  {title}")
    print(_bar())


def _footer(label: str, ms: float) -> None:
    print(_thin())
    print(_row(label, ms))
    print(_bar() + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# bench_prepare
# ─────────────────────────────────────────────────────────────────────────────

def bench_prepare(
    planner,
    regions=None,
    rebuild: bool = False,
    rebuild_prerequisites: bool = True,
) -> dict[str, float]:
    """
    Time the full planner.prepare() call.

    The instrumented code inside csa_data_handler and journey_planner prints
    the per-phase breakdown automatically.  This function wraps the whole call
    and returns ``{"total_sec": float}``.

    Parameters
    ----------
    planner:
        A JourneyPlanner instance (not yet prepared, or re-prepare with
        rebuild=True to force fresh tables).
    regions:
        Optional region UUID list passed through to prepare().
    rebuild:
        Pass True to drop and recreate CSA tables.
    rebuild_prerequisites:
        Pass True to also rebuild stops / footpaths.
    """
    _header("bench_prepare")
    t = time.perf_counter()
    planner.prepare(
        regions=regions,
        rebuild=rebuild,
        rebuild_prerequisites=rebuild_prerequisites,
    )
    total = time.perf_counter() - t
    _footer("bench_prepare total", total * 1000)
    return {"total_sec": total}


# ─────────────────────────────────────────────────────────────────────────────
# bench_plan_candidates
# ─────────────────────────────────────────────────────────────────────────────

def bench_plan_candidates(
    planner,
    iterations: int = 6,
    start_stop_id: int = 8501120,
    end_stop_id: int = 8501117,
    travel_date: str = "2026-05-11",
    arrival_deadline: str = "09:00",
    max_routes: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run plan_candidates ``iterations`` times and report timing statistics.

    Each run prints a single timing line (from the instrumented
    journey_planner code).  After all runs this function prints the
    aggregate: avg / min / max / stdev.

    Returns a dict with keys:
        times_ms   list[float]  — per-run wall times in ms
        avg_ms     float
        min_ms     float
        max_ms     float
        stdev_ms   float  (0.0 when iterations < 2)
        n_routes   list[int]   — routes returned each run
    """
    _header(f"bench_plan_candidates  (n={iterations})")

    times: list[float] = []
    n_routes: list[int] = []

    for i in range(iterations):
        print(f"  run {i + 1:>2}/{iterations}  ", end="", flush=True)
        t = time.perf_counter()
        result = planner.plan_candidates(
            start_stop_id=start_stop_id,
            end_stop_id=end_stop_id,
            travel_date=travel_date,
            arrival_deadline=arrival_deadline,
            max_routes=max_routes,
            **kwargs,
        )
        elapsed_ms = (time.perf_counter() - t) * 1000
        times.append(elapsed_ms)
        n_routes.append(len(result))

    avg = mean(times)
    mn = min(times)
    mx = max(times)
    sd = stdev(times) if len(times) >= 2 else 0.0

    print(_thin())
    print(_row("avg", avg))
    print(_row("min", mn))
    print(_row("max", mx))
    print(_row("stdev", sd))
    _footer("bench_plan_candidates done", avg)

    return {
        "times_ms": times,
        "avg_ms": avg,
        "min_ms": mn,
        "max_ms": mx,
        "stdev_ms": sd,
        "n_routes": n_routes,
    }


def bench_plan_profile(
    planner,
    iterations: int = 6,
    start_stop_id: int = 8501120,
    end_stop_id: int = 8501117,
    travel_date: str = "2026-05-11",
    arrival_deadline: str = "09:00",
    max_routes: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run plan_profile ``iterations`` times and report timing statistics.

    Each run prints a single timing line (from the instrumented
    journey_planner code).  After all runs this function prints the
    aggregate: avg / min / max / stdev.

    Returns a dict with keys:
        times_ms   list[float]  — per-run wall times in ms
        avg_ms     float
        min_ms     float
        max_ms     float
        stdev_ms   float  (0.0 when iterations < 2)
        n_routes   list[int]   — routes returned each run
    """
    _header(f"bench_plan_profile  (n={iterations})")

    times: list[float] = []
    n_routes: list[int] = []

    for i in range(iterations):
        print(f"  run {i + 1:>2}/{iterations}  ", end="", flush=True)
        t = time.perf_counter()
        result = planner.plan_profile(
            start_stop_id=start_stop_id,
            end_stop_id=end_stop_id,
            travel_date=travel_date,
            arrival_deadline=arrival_deadline,
            max_routes=max_routes,
            **kwargs,
        )
        elapsed_ms = (time.perf_counter() - t) * 1000
        times.append(elapsed_ms)
        n_routes.append(len(result))

    avg = mean(times)
    mn = min(times)
    mx = max(times)
    sd = stdev(times) if len(times) >= 2 else 0.0

    print(_thin())
    print(_row("avg", avg))
    print(_row("min", mn))
    print(_row("max", mx))
    print(_row("stdev", sd))
    _footer("bench_plan_profile done", avg)

    return {
        "times_ms": times,
        "avg_ms": avg,
        "min_ms": mn,
        "max_ms": mx,
        "stdev_ms": sd,
        "n_routes": n_routes,
    }
# ─────────────────────────────────────────────────────────────────────────────
# bench_route  (forward CSA)
# ─────────────────────────────────────────────────────────────────────────────

def bench_route(
    planner,
    iterations: int = 6,
    start_id: int = 8501120,
    end_id: int | None = 8501117,
    departs: str = "08:30",
    day: str = "monday",
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run route() ``iterations`` times and report timing statistics.

    Each run prints a single timing line from the instrumented route()
    method.  The aggregate stats are printed afterwards.

    Returns the same shape dict as bench_plan_candidates.
    """
    mode = "one-to-all" if end_id is None else "point-to-point"
    _header(f"bench_route [{mode}]  (n={iterations})")

    times: list[float] = []

    for i in range(iterations):
        print(f"  run {i + 1:>2}/{iterations}  ", end="", flush=True)
        t = time.perf_counter()
        planner.route(
            start_id=start_id,
            end_id=end_id,
            departs=departs,
            day=day,
            **kwargs,
        )
        elapsed_ms = (time.perf_counter() - t) * 1000
        times.append(elapsed_ms)

    avg = mean(times)
    mn = min(times)
    mx = max(times)
    sd = stdev(times) if len(times) >= 2 else 0.0

    print(_thin())
    print(_row("avg", avg))
    print(_row("min", mn))
    print(_row("max", mx))
    print(_row("stdev", sd))
    _footer("bench_route done", avg)

    return {
        "times_ms": times,
        "avg_ms": avg,
        "min_ms": mn,
        "max_ms": mx,
        "stdev_ms": sd,
    }


# ─────────────────────────────────────────────────────────────────────────────
# bench_route_backward  (single backward CSA pass)
# ─────────────────────────────────────────────────────────────────────────────

def bench_route_backward(
    planner,
    iterations: int = 6,
    start_id: int = 8501120,
    end_id: int = 8501117,
    deadline: str = "09:00",
    day: str = "monday",
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Benchmark the raw route_backward() call directly (no candidate loop).

    Useful for isolating how long a single backward CSA scan takes.
    """
    _header(f"bench_route_backward  (n={iterations})")

    deadline_secs = planner._time_to_secs(deadline)
    max_walk_m = planner.settings.max_walk_m

    times: list[float] = []

    for i in range(iterations):
        t = time.perf_counter()
        planner.route(
            start_id, end_id, deadline_secs, deadline_secs, day, max_walk_m, **kwargs
        )
        elapsed_ms = (time.perf_counter() - t) * 1000
        times.append(elapsed_ms)
        print(f"  run {i + 1:>2}/{iterations}   {elapsed_ms:>7.2f} ms")

    avg = mean(times)
    mn = min(times)
    mx = max(times)
    sd = stdev(times) if len(times) >= 2 else 0.0

    print(_thin())
    print(_row("avg", avg))
    print(_row("min", mn))
    print(_row("max", mx))
    print(_row("stdev", sd))
    _footer("bench_route_backward done", avg)

    return {
        "times_ms": times,
        "avg_ms": avg,
        "min_ms": mn,
        "max_ms": mx,
        "stdev_ms": sd,
    }
