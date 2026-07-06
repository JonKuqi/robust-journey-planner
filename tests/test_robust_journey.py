"""
RobustJourneyPlanner benchmark helpers.

Import into tests_results/whole_system_speed.ipynb.

Quick start
-----------
    from src.routing.robust_journey_planner import RobustJourneyPlanner
    from tests.test_robust_journey import bench_robust_prepare, bench_robust_plan

    iterations = 6

    robust_planner = RobustJourneyPlanner()
    bench_robust_prepare(
        robust_planner,
        regions=REGION_UUIDS,
        force_rebuild=False,
        train_delay_model=False,
    )
    bench_robust_plan(robust_planner, iterations=iterations,
                      start_stop_id=8501120, end_stop_id=8501117,
                      travel_date="2026-05-11", arrival_deadline="09:00",
                      confidence_q=0.90)
"""

from __future__ import annotations

import contextlib
import io
import time
from statistics import mean, stdev
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers (shared style with test_CSA)
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


def _summarize_routes(routes: list[dict]) -> str:
    if not routes:
        return "routes=0"

    best = routes[0]
    dep = best.get("departure_time", best.get("departure_secs", "?"))
    arr = best.get("arrival_time", best.get("arrival_secs", "?"))
    robust_arr = best.get("robust_arrival_time", best.get("robust_arrival_secs"))
    conf = best.get("route_confidence")

    parts = [f"routes={len(routes)}", f"best={dep}->{arr}"]
    if robust_arr is not None:
        parts.append(f"robust_arr={robust_arr}")
    if isinstance(conf, (int, float)):
        parts.append(f"conf={conf:.3f}")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# bench_robust_prepare
# ─────────────────────────────────────────────────────────────────────────────

def bench_robust_prepare(
    robust_planner,
    regions=None,
    force_rebuild: bool = False,
    rebuild_csa_prerequisites: bool = True,
    train_delay_model: bool = False,
    force_retrain_delay_model: bool = False,
    travel_date: str | None = None,
    load_delay_lookup: bool = True,
) -> dict[str, float]:
    """
    Time the full RobustJourneyPlanner.prepare() call.

    The instrumented code inside robust_journey_planner and journey_planner
    prints the per-phase breakdown (CSA build, fetch, materialize, model load)
    automatically.  This function wraps the total and returns
    ``{"total_sec": float}``.

    Parameters
    ----------
    robust_planner:
        A RobustJourneyPlanner instance (unprepared, or re-prepare with
        force_rebuild=True).
    regions:
        Optional region UUID iterable passed through to prepare().
    force_rebuild:
        Drop and recreate Trino tables.
    rebuild_csa_prerequisites:
        Also rebuild stops / footpaths tables.
    travel_date:
        Forwarded to prepare(); triggers quantile baking when set.
    load_delay_lookup:
        Whether to load delay artifacts and bake per-connection quantiles.
        Defaults to True — set False only for pure CSA-speed benchmarks.
    """
    _header("bench_robust_prepare")
    t = time.perf_counter()
    robust_planner.prepare(
        regions=regions,
        rebuild=force_rebuild,
        rebuild_prerequisites=rebuild_csa_prerequisites,
        travel_date=travel_date,
        load_delay_lookup=load_delay_lookup,
    )
    total = time.perf_counter() - t
    _footer("bench_robust_prepare total", total * 1000)
    return {"total_sec": total}


# ─────────────────────────────────────────────────────────────────────────────
# bench_robust_plan
# ─────────────────────────────────────────────────────────────────────────────

def bench_robust_plan(
    robust_planner,
    iterations: int = 6,
    start_stop_id: int = 8501120,
    end_stop_id: int = 8501117,
    travel_date: str = "2026-05-11",
    arrival_deadline: str = "09:00",
    confidence_q: float = 0.90,
    max_routes: int = 5,
    verbose: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run RobustJourneyPlanner.plan() ``iterations`` times and report timing stats.

    After all runs prints aggregate: avg / min / max / stdev.
    Set verbose=False to suppress per-run output (only the summary is printed).

    Returns a dict with keys:
        times_ms        list[float]  — per-run wall times in ms
        avg_ms          float
        min_ms          float
        max_ms          float
        stdev_ms        float  (0.0 when iterations < 2)
        n_routes        list[int]   — routes returned each run
        route_summaries list[str]   — compact best-route summaries
    """
    _header(f"bench_robust_plan  (n={iterations})")

    times: list[float] = []
    n_routes: list[int] = []
    route_summaries: list[str] = []

    for i in range(iterations):
        if verbose:
            print(f"  run {i + 1:>2}/{iterations}  ", end="", flush=True)
        t = time.perf_counter()
        _sink = io.StringIO()
        _ctx = contextlib.redirect_stdout(_sink) if not verbose else contextlib.nullcontext()
        with _ctx:
            result = robust_planner.plan(
                start_stop_id=start_stop_id,
                end_stop_id=end_stop_id,
                travel_date=travel_date,
                arrival_deadline=arrival_deadline,
                confidence_q=confidence_q,
                max_routes=max_routes,
                **kwargs,
            )
        elapsed_ms = (time.perf_counter() - t) * 1000
        summary = _summarize_routes(result)
        if verbose:
            print(f"{elapsed_ms:>8.2f} ms   {summary}")
        times.append(elapsed_ms)
        n_routes.append(len(result))
        route_summaries.append(summary)

    avg = mean(times)
    mn = min(times)
    mx = max(times)
    sd = stdev(times) if len(times) >= 2 else 0.0

    print(_thin())
    print(_row("avg", avg))
    print(_row("min", mn))
    print(_row("max", mx))
    print(_row("stdev", sd))
    _footer("bench_robust_plan done", avg)

    return {
        "times_ms": times,
        "avg_ms": avg,
        "min_ms": mn,
        "max_ms": mx,
        "stdev_ms": sd,
        "n_routes": n_routes,
        "route_summaries": route_summaries,
    }
